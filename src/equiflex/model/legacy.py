"""Archived Flex model versions (V1-V4) and their non-SH layer modules.

Kept for reference only. The active model is model.models.FlexModel (the folded
V4 chemistry trunk + V5 SH flow trunk). Shared modules are imported from
model.layers; the legacy-only layer variants are defined below.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cuequivariance as cue
import cuequivariance_torch as cuet
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Tuple
from torch_scatter import scatter
import math

from .layers import (
    _cuet_fp32,
    FCLayer, SwiGLU, AdaLN, ConditionedTransition, MultiHeadAttention,
    SelfOuterProductModule, TimeEmbedder, InitialEmbedder, AtomTransformer,
    TriangleUpdatingModule,
)
from .models import FlexConfig


class PoolingLayer(nn.Module):
    def __init__(self,
                 mode: str = 'mean'):
        super().__init__()
        assert mode in ['mean', 'max', 'sum'], "Invalid pooling mode"
        self.mode = mode

    def forward(self, x: torch.Tensor, uid_vec: torch.Tensor, dim_size: int):
        pooled = scatter(x, uid_vec, dim=0, dim_size=dim_size, reduce=self.mode)
        return pooled


class UnpoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, uid_vec: torch.Tensor):
        unpooled = x[uid_vec.long()]
        return unpooled


class PairwiseFeatureEmbedder(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 out_dim: int):
        super().__init__()
        self.disto_embedder = FCLayer(in_dim=41, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.adj_embedder = FCLayer(in_dim=26, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.mask_embedder = FCLayer(in_dim=1, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.final_proj = nn.Linear(out_dim, out_dim, bias=False)
        self._initialize_weights()

    def forward(self, distogram: torch.Tensor, adj_mat: torch.Tensor, block_mask: Optional[torch.Tensor] = None):
        '''
        Args:
            distogram: [B, N, N, n_bins] - pairwise distances
            adj_mat: [B, N, N, 26] - bond features
            block_mask: [B, N, N, 1] - True where atoms belong to same entity (optional)
        '''
        distogram = self._rbf_bin_distogram(distogram)
        disto_emb = self.disto_embedder(distogram) * block_mask
        adj_emb = self.adj_embedder(adj_mat) * block_mask
        mask_emb = self.mask_embedder(block_mask)
        output = disto_emb + adj_emb + mask_emb
        output = self.final_proj(output)

        return output

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.final_proj.weight)
        
    def _rbf_bin_distogram(self, distogram: torch.Tensor):
        # Soft binning with rbf
        # Distogram: [B, N, N, 1]
        # Centers: [20]
        # Width: 0.5
        # RBF features: [B, N, N, 20]
        centers = torch.arange(0, 20.5, 0.5).to(distogram.device)
        width = 0.5
        rbf_features = torch.exp(-width * (distogram - centers) ** 2)
        return rbf_features


class PairwiseFeatureEmbedderWithConstraints(nn.Module):
    '''
    For V3 model we incorporate constraint matrix into pairwise features
    '''
    def __init__(self, 
                 hidden_dim: int, 
                 out_dim: int):
        super().__init__()
        self.disto_embedder = FCLayer(in_dim=41, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.adj_embedder = FCLayer(in_dim=26, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.constraint_embedder = FCLayer(in_dim=41, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.mask_embedder = FCLayer(in_dim=1, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.final_proj = nn.Linear(out_dim, out_dim, bias=False)
        self._initialize_weights()

    def forward(self, distogram: torch.Tensor, adj_mat: torch.Tensor, constraint_mat: torch.Tensor, block_mask: Optional[torch.Tensor] = None):
        '''
        Args:
            distogram: [B, N, N, n_bins] - pairwise distances
            adj_mat: [B, N, N, 26] - bond features
            constraint_mat: [B, N, N, 41] - constraint features
            block_mask: [B, N, N, 1] - True where atoms belong to same entity (optional)
        '''
        disto_emb = self._rbf_bin_distogram(distogram)
        disto_emb = self.disto_embedder(disto_emb) * block_mask
        adj_emb = self.adj_embedder(adj_mat) * block_mask
        constraint_emb = self._rbf_bin_distogram(constraint_mat)
        constraint_emb = self.constraint_embedder(constraint_emb) * block_mask
        mask_emb = self.mask_embedder(block_mask)
        output = disto_emb + adj_emb + constraint_emb + mask_emb
        output = self.final_proj(output)

        return output

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.final_proj.weight)

    def _rbf_bin_distogram(self, distogram: torch.Tensor):
        # Soft binning with rbf
        # Distogram: [B, N, N, 1]
        # Centers: [20]
        # Width: 0.5
        # RBF features: [B, N, N, 20]
        centers = torch.arange(0, 20.5, 0.5).to(distogram.device)
        width = 0.5
        rbf_features = torch.exp(-width * (distogram - centers) ** 2)
        return rbf_features


class EVAModule(nn.Module):
    '''
    A module that collects relative vector features from each atom's coordinates
    With self-attention and einsum aggregate the features into vectors of num_heads
    '''

    def __init__(self,
                 scalar_dim: int,
                 num_heads: int,
                 attenuate_by_dist: bool = True,
                 soft_cap: float = 50.0,
                 ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.num_heads = num_heads
        self.head_dim = scalar_dim // num_heads
        self.attenuate_by_dist = attenuate_by_dist
        self.soft_cap = soft_cap

        self.w_q = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.w_k = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.norm_1 = nn.LayerNorm(scalar_dim)

        self._initialize_weights()
                
    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        # Relative coordinates
        r_ij = r.unsqueeze(2) - r.unsqueeze(1)  # [B, N, N, 3]

        # Project and reshape
        B, N, _ = x.shape
        x = self.norm_1(x)
        q = self.w_q(x)
        k = self.w_k(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)

        # Compute attention
        attn = torch.einsum('bihd, bjhd -> bijh', q, k)  # [B, N, N, num_heads]
        del q, k  # Free after einsum
        attn = attn / math.sqrt(float(self.head_dim))

        if bias is not None:
            attn = attn + bias

        if self.attenuate_by_dist:
            dist = torch.norm(r_ij, dim=-1, keepdim=True)  # [B, N, N, 1]
            dist_mask = dist < 1e-4 # [B, N, N, 1], for numerical stability
            r_ij = r_ij / (dist + 1e-6) # Normalize r_ij by distance
            attn = attn - dist
            # Soft cap attention scores before masking and softmax
            attn = self.soft_cap * torch.tanh(attn / self.soft_cap)
            valid_attn_mask = ~dist_mask
            if atom_mask is not None:
                key_mask = atom_mask.bool().unsqueeze(1).unsqueeze(-1)
                valid_attn_mask = valid_attn_mask & key_mask
                attn = attn.masked_fill(~key_mask, -1e9)
            attn = attn.masked_fill(dist_mask, -1e9)
            attn = attn.softmax(dim=2)
            attn = attn * valid_attn_mask.to(attn.dtype)
            attn = attn / attn.sum(dim=2, keepdim=True).clamp(min=1e-8)
            del dist, dist_mask

        else:
            # Soft cap attention scores before softmax
            attn = self.soft_cap * torch.tanh(attn / self.soft_cap)
            valid_attn_mask = None
            if atom_mask is not None:
                key_mask = atom_mask.bool().unsqueeze(1).unsqueeze(-1)
                attn = attn.masked_fill(~key_mask, -1e9)
                valid_attn_mask = key_mask
            attn = attn.softmax(dim=2)
            if valid_attn_mask is not None:
                attn = attn * valid_attn_mask.to(attn.dtype)
                attn = attn / attn.sum(dim=2, keepdim=True).clamp(min=1e-8)

        # Collect attention stats before deletion if needed
        attn_stats = None
        if return_attn:
            # Shannon entropy: H = -sum(p * log(p)), averaged over all distributions
            # attn shape: [B, N, N, num_heads], softmax over dim=2
            entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=2).mean()
            attn_stats = {
                'mean': attn.mean().detach(),
                'max': attn.max().detach(),
                'entropy': entropy.detach(),
            }

        v_out = torch.einsum('bijh, bijc -> bihc', attn, r_ij)  # [B, N, num_heads, 3]
        del attn, r_ij  # Free pairwise tensors
        if atom_mask is not None:
            v_out = v_out * atom_mask.to(v_out.dtype).unsqueeze(-1).unsqueeze(-1)

        if return_attn:
            return v_out, attn_stats
        return v_out
    
    def _initialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GeometricAttentionModule(nn.Module):
    """
    Geometric attention combining scalar and vector features with geometric scores.
    Scalar attention scores and geometric (vector dot-product) scores are fused
    before softmax, and vectors are gated by scalar information.
    """

    def __init__(self,
                 ns: int,
                 nv: int,
                 num_heads: int,
                 bias_dim: int,
                 attn_dropout: float,
                 use_gated_attention: bool = True,
                 soft_cap: float = 50.0,
                 use_xsa: bool = True):
        super().__init__()

        assert ns % num_heads == 0, f'ns ({ns}) must be divisible by num_heads ({num_heads})'
        assert nv % num_heads == 0, f'nv ({nv}) must be divisible by num_heads ({num_heads})'

        self.ns = ns
        self.nv = nv
        self.num_heads = num_heads
        self.head_dim_s = ns // num_heads
        self.head_dim_v = nv // num_heads
        self.attn_dropout = attn_dropout
        self.use_gated_attention = use_gated_attention
        self.soft_cap = soft_cap
        self.use_xsa = use_xsa

        # Scalar Q, K, V projections
        # Headwise gating: one scalar gate per head (not per dimension)
        # Output: ns + num_heads extra values for gates
        if use_gated_attention:
            self.scalar_q = nn.Linear(ns, ns + num_heads, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            self.scalar_q = nn.Linear(ns, ns, bias=False)
        
        self.scalar_k = nn.Linear(ns, ns, bias=False)
        self.scalar_v = nn.Linear(ns, ns, bias=False)
        self.scalar_out = nn.Linear(ns, ns, bias=False)

        # Vector Q, K, V projections
        ir_v = cue.Irreps('O3', f'{nv}x1o')
        layout = cue.mul_ir # ir_mul often bugs equivariance
        self.vector_q = cuet.Linear(ir_v, ir_v, layout=layout)
        self.vector_k = cuet.Linear(ir_v, ir_v, layout=layout)
        self.vector_v = cuet.Linear(ir_v, ir_v, layout=layout)
        self.vector_out = cuet.Linear(ir_v, ir_v, layout=layout)
        
        # Bias & gating
        self.bias_norm = nn.LayerNorm(bias_dim)
        self.bias_proj = nn.Linear(bias_dim, num_heads, bias=False)
        self.vector_gate = nn.Sequential(
            nn.Linear(ns, nv, bias=False),
            nn.Sigmoid()
        )
        
        # Normalization
        self.scalar_norm = nn.LayerNorm(ns, elementwise_affine=True)

        # FFN for scalars
        self.scalar_ffn = SwiGLU(ns)
        self.scalar_ffn_norm = nn.LayerNorm(ns, elementwise_affine=True)

        self._initialize_weights()

    def forward(self,
                scalar_feats: torch.Tensor,  # [B, N, ns]
                vector_feats: torch.Tensor,  # [B, N, nv*3]
                bias: Optional[torch.Tensor] = None,  # [B, N, N, bias_dim] or [B, H, N, N]
                atom_mask: Optional[torch.Tensor] = None,
                return_attn: bool = False,
                ):
        B, N, _ = scalar_feats.shape
        mask_f = atom_mask.to(scalar_feats.dtype).unsqueeze(-1) if atom_mask is not None else None
        if mask_f is not None:
            scalar_feats = scalar_feats * mask_f
            vector_feats = vector_feats * mask_f

        # Residuals
        scalar_residual = scalar_feats
        vector_residual = vector_feats

        # Pre-norm to scalar
        scalar_feats = self.scalar_norm(scalar_feats)
        if mask_f is not None:
            scalar_feats = scalar_feats * mask_f
        
        # No vector normalization: vectors carry magnitude information (flow strength)
        # Division-based normalization amplifies noise when vectors are near zero
        # (which represents valid physical states like equilibrium/no flow)
        vector_3d = vector_feats.reshape(B, N, self.nv, 3)  # [B, N, nv*3] -> [B, N, nv, 3]
        
        # Scalar attention logic
        if self.use_gated_attention:
            Q_s_out = self.scalar_q(scalar_feats)  # [B, N, ns + num_heads]
            Q_s, Q_s_gate = torch.split(Q_s_out, [self.ns, self.num_heads], dim=-1)
        else:
            Q_s = self.scalar_q(scalar_feats)  # [B, N, ns]
        
        K_s = self.scalar_k(scalar_feats)  # [B, N, ns]
        V_s = self.scalar_v(scalar_feats)  # [B, N, ns]

        Q_s = Q_s.reshape(B, N, self.num_heads, self.head_dim_s).permute(0, 2, 1, 3)
        K_s = K_s.reshape(B, N, self.num_heads, self.head_dim_s).permute(0, 2, 1, 3)
        V_s = V_s.reshape(B, N, self.num_heads, self.head_dim_s).permute(0, 2, 1, 3)

        attn_score_s = torch.matmul(Q_s, K_s.transpose(-2, -1)) / (self.head_dim_s ** 0.5)
        del Q_s, K_s

        # Vector attention logic
        vector_flat = vector_3d.reshape(B * N, self.nv * 3)
        Q_v = _cuet_fp32(self.vector_q, vector_flat).reshape(B, N, self.nv, 3)
        K_v = _cuet_fp32(self.vector_k, vector_flat).reshape(B, N, self.nv, 3)
        V_v = _cuet_fp32(self.vector_v, vector_flat).reshape(B, N, self.nv, 3)
        del vector_flat
        
        Q_v = Q_v.view(B, N, self.num_heads, self.head_dim_v, 3).permute(0, 2, 1, 3, 4)
        K_v = K_v.view(B, N, self.num_heads, self.head_dim_v, 3).permute(0, 2, 1, 3, 4)

        attn_score_v = torch.einsum('bhidx, bhjdx -> bhij', Q_v, K_v)
        del Q_v, K_v
        attn_score_v = attn_score_v / ((3 * self.head_dim_v) ** 0.5)

        # Combine scores
        attn_score = attn_score_s + attn_score_v
        del attn_score_s, attn_score_v

        # Project bias: [B, N, N, bias_dim] -> [B, N, N, H] -> [B, H, N, N]
        if bias is not None:
            if bias.shape[-1] != self.num_heads:
                bias = self.bias_norm(bias)
                bias = self.bias_proj(bias)  # [B, N, N, H]
            bias = bias.permute(0, 3, 1, 2)  # [B, H, N, N]
            attn_score = attn_score + bias

        # Soft cap attention scores to prevent softmax saturation
        attn_score = self.soft_cap * torch.tanh(attn_score / self.soft_cap)
        if atom_mask is not None:
            key_mask = atom_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_score = attn_score.masked_fill(~key_mask, -1e9)

        # Softmax
        attn_weights = F.softmax(attn_score, dim=-1)  # [B, H, N, N]
        del attn_score

        # Collect attention stats before dropout if needed
        attn_stats = None
        if return_attn:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean()
            attn_stats = {
                'mean': attn_weights.mean().detach(),
                'max': attn_weights.max().detach(),
                'entropy': entropy.detach(),
            }

        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # Scalar aggregation
        scalar_out = torch.matmul(attn_weights, V_s)

        # XSA: project out self-value component from scalar attention output
        if self.use_xsa:
            vn = F.normalize(V_s, dim=-1)
            scalar_out = scalar_out - (scalar_out * vn).sum(dim=-1, keepdim=True) * vn
            del vn

        del V_s
        
        # Headwise gating: one scalar gate per head, broadcast over head_dim
        if self.use_gated_attention:
            gate = Q_s_gate.unsqueeze(-1).permute(0, 2, 1, 3)
            scalar_out = scalar_out * self.sigmoid(gate)
            del Q_s_gate, gate

        scalar_out = scalar_out.permute(0, 2, 1, 3).reshape(B, N, self.ns)
        scalar_out = self.scalar_out(scalar_out)
        
        # Vector aggregation
        V_v_heads = V_v.view(B, N, self.num_heads, self.head_dim_v, 3).permute(0, 2, 1, 3, 4)
        V_v_flat = V_v_heads.flatten(start_dim=3)
        del V_v_heads
        vector_out = torch.matmul(attn_weights, V_v_flat)
        del V_v_flat, attn_weights
        vector_out = vector_out.view(B, self.num_heads, N, self.head_dim_v, 3).permute(0, 2, 1, 3, 4)
        vector_out = vector_out.reshape(B, N, self.nv, 3)
        vector_out_flat = vector_out.reshape(B * N, self.nv * 3)
        vector_out = _cuet_fp32(self.vector_out, vector_out_flat).reshape(B, N, self.nv, 3)
        
        # Scalar-to-vector gating
        vec_gate = self.vector_gate(scalar_out)  # [B, N, nv]
        vector_out = vector_out * vec_gate.unsqueeze(-1)
        del vec_gate
        vector_out = vector_out.reshape(B, N, self.nv * 3)

        # Residual connections
        scalar_out = scalar_out + scalar_residual
        del scalar_residual
        vector_out = vector_out + vector_residual
        del vector_residual

        # SwiGLU FFN for scalars
        scalar_ffn = self.scalar_ffn_norm(scalar_out)
        scalar_out = scalar_out + self.scalar_ffn(scalar_ffn)
        del scalar_ffn

        if mask_f is not None:
            scalar_out = scalar_out * mask_f
            vector_out = vector_out * mask_f

        if return_attn:
            return scalar_out, vector_out, attn_stats
        return scalar_out, vector_out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, cuet.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Extra stability
        for m in [self.scalar_out, self.vector_out]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)


class TokenwiseTransformerModule(nn.Module):
    '''
    Main module that processes the tokenwise features, similar to AF3 Pairformer.
    Maintains persistent pair representation z across layers (EvoFormer-style).
    '''

    def __init__(self,
                 ns: int,
                 nv: int,
                 num_geo_heads: int,
                 attn_dropout: float,
                 use_gated_attention: bool = True,
                 ):
        super().__init__()
        self.ns = ns
        self.nv = nv
        self.outer_product = SelfOuterProductModule(in_dim = ns, hidden_dim = ns // 4)
        self.geometric_attn = GeometricAttentionModule(ns = ns, nv = nv, num_heads = num_geo_heads, attn_dropout = attn_dropout, use_gated_attention = use_gated_attention)
        self.triangle_updates = TriangleUpdatingModule(ns = ns)
        self.w_fusion = nn.Linear(ns * 2, ns, bias = False)
        self.sigmoid = nn.Sigmoid()
        self.use_gated_attention = use_gated_attention

    def forward(self,
                s_atom: torch.Tensor,  # [B, N_tokens, ns]
                s_emb: torch.Tensor,   # [B, N_tokens, ns]
                v: torch.Tensor,       # [B, N_tokens, nv*3]
                z: torch.Tensor,       # [B, N_tokens, N_tokens, ns] - persistent pair features
                ):
        s = self.w_fusion(torch.cat([s_atom, s_emb], dim=-1))  # [B, N_tokens, ns]

        # Update z with new information from current layer
        z = z + self.outer_product(s_emb)

        # Triangle update refines geometric consistency
        z = self.triangle_updates(z)

        s, v = self.geometric_attn(s, v, bias=z)

        s_atom = s_atom + s
        s_emb = s_emb + s

        return s_atom, s_emb, v, z


class FlowMatchingModule(nn.Module):
    '''
    Flow matching module for projecting atomwise features to Euclidean space.
    Estimates velocity field v(x_t, t) for the flow ODE: dx/dt = v(x_t, t)

    Output: scalar magnitude * vector direction = velocity field
    '''

    def __init__(self,
                 ns: int,
                 nv: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ns = ns
        self.nv = nv

        ir_v_in = cue.Irreps('O3', f'{nv}x1o')
        ir_v_half = cue.Irreps('O3', f'{nv // 2}x1o')
        ir_v_out = cue.Irreps('O3', '1x1o')
        layout = cue.mul_ir
        self.vector_proj_1 = cuet.Linear(ir_v_in, ir_v_half, layout=layout)
        self.vector_proj_2 = cuet.Linear(ir_v_half, ir_v_out, layout=layout)

        # Scalar projection: ns -> 1 (magnitude)
        self.scalar_projector = nn.Sequential(
            nn.LayerNorm(ns),
            nn.Linear(ns, ns // 4, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ns // 4, 1, bias=False)
        )

        self._initialize_weights()

    def forward(self,
                scalar_feats: torch.Tensor,  # [B, N, ns] or [B*N, ns]
                vector_feats: torch.Tensor,  # [B, N, nv*3] or [B*N, nv*3]
                ) -> torch.Tensor:
        '''
        Returns:
            velocity: [B, N, 3] or [B*N, 3] - predicted velocity field
        '''
        # Handle both batched [B, N, F] and flattened [B*N, F] inputs
        input_3d = scalar_feats.dim() == 3
        if input_3d:
            B, N, _ = scalar_feats.shape
            scalar_feats = scalar_feats.reshape(B * N, -1)
            vector_feats = vector_feats.reshape(B * N, -1)

        # Project to single vector with equivariant normalization between layers
        vector_feats = _cuet_fp32(self.vector_proj_1, vector_feats)  # [B*N, nv_half*3]
        vector_feats = _cuet_fp32(self.vector_proj_2, vector_feats)  # [B*N, 3]

        scalar_feats = self.scalar_projector(scalar_feats)   # [B*N, 1]

        # Velocity = magnitude * direction
        velocity = scalar_feats * vector_feats  # [B*N, 3]

        if input_3d:
            velocity = velocity.reshape(B, N, 3)

        return velocity

    def _initialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, cuet.Linear):
                nn.init.xavier_uniform_(m.weight)


class FlexModelV1(nn.Module):

    def __init__(self, config: FlexConfig):
        super().__init__()
        self.config = config

        self.time_embedder = TimeEmbedder(
            hidden_dim=config.ns,
            output_dim=config.ns
        )

        self.initial_embedder = InitialEmbedder(
            atom_dim=config.atom_feat_dim,
            use_embeddings=config.use_embeddings,
            esm_dim=config.esm_dim,
            unimol_dim=config.unimol_dim,
            hidden_dim=config.ns,
            dropout=config.initial_dropout,
        )

        # AdaLN for timestep conditioning at multiple stages
        self.time_adaln_initial = AdaLN(in_dim=config.ns)  # After initial embedding

        # Pairwise feature embedder - creates initial p from distogram/adjacency
        self.pair_embedder = PairwiseFeatureEmbedder(
            hidden_dim=config.pair_hidden_dim,
            out_dim=config.pair_hidden_dim
        )

        self.atom_layers = nn.ModuleDict({
            f'atom_layer_{i}': AtomTransformer(
                single_hidden_dim=config.ns,
                pair_hidden_dim=config.pair_hidden_dim,
                num_heads=config.num_atom_heads,
                attn_dropout=config.attn_dropout,
                use_gated_attention=config.use_gated_attention,
                use_outer_product=config.use_outer_product,
                use_qk_norm=config.use_qk_norm,
            ) for i in range(config.num_atom_layers)
        })
        
        self.adaln_atom_layers = nn.ModuleDict({
            f'adaln_atom_{i}': AdaLN(in_dim=config.ns) for i in range(config.num_atom_layers)
        })

        # EVA module: extracts nv vectors from coordinates AFTER atom layers
        self.eva_module = EVAModule(
            scalar_dim=config.ns,
            num_heads=config.nv,
            attenuate_by_dist=config.attenuate_by_dist,
            soft_cap=config.soft_cap,
        )

        self.pool = PoolingLayer(mode='mean')
        self.unpool = UnpoolingLayer()

        self.token_layers = nn.ModuleDict({
            f'token_layer_{i}': TokenwiseTransformerModule(
                ns=config.ns,
                nv=config.nv,
                num_geo_heads=config.num_geo_heads,
                attn_dropout=config.attn_dropout,
                use_gated_attention=config.use_gated_attention
            ) for i in range(config.num_token_layers)
        })
        self.adaln_emb_tokens = nn.ModuleDict({
            f'adaln_emb_tokens_{i}': AdaLN(in_dim=config.ns) for i in range(config.num_token_layers)
        })
        self.adaln_atom_tokens = nn.ModuleDict({
            f'adaln_atom_tokens_{i}': AdaLN(in_dim=config.ns) for i in range(config.num_token_layers)
        })

        # Fusion layers for atom features after unpooling
        self.atom_fusion = nn.Linear(config.ns * 2, config.ns, bias=False)

        # Flow matching head (predicts velocity field)
        self.flow_head = FlowMatchingModule(
            ns=config.ns,
            nv=config.nv,
            dropout=config.output_dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.atom_fusion.weight)

    def forward(
        self,
        # Receptor inputs
        rec_atom_feats: torch.Tensor,      # [B, N_rec, 124]
        rec_atom_coords: torch.Tensor,     # [B, N_rec, 3] - noisy coordinates x_t
        rec_atom_ref_coords: torch.Tensor, # [B, N_rec, 3] - reference coordinates
        rec_bond_feats: torch.Tensor,      # [B, N_rec, N_rec, 26]
        rec_uid_vec: torch.Tensor,         # [B, N_rec] - residue index for each receptor atom
        rec_atom_mask: torch.Tensor,       # [B, N_rec] - mask for valid receptor atoms
        esm_embeddings: torch.Tensor,      # [B, N_rec_tokens, 1536]

        # Ligand inputs
        lig_atom_feats: torch.Tensor,      # [B, N_lig, 124]
        lig_atom_coords: torch.Tensor,     # [B, N_lig, 3] - noisy coordinates x_t
        lig_atom_ref_coords: torch.Tensor, # [B, N_lig, 3] - reference coordinates
        lig_bond_feats: torch.Tensor,      # [B, N_lig, N_lig, 26]
        lig_uid_vec: torch.Tensor,         # [B, N_lig] - ligand index for each ligand atom
        lig_atom_mask: torch.Tensor,       # [B, N_lig] - mask for valid ligand atoms
        unimol_embeddings: torch.Tensor,   # [B, N_lig, 1536]

        # Timestep
        t: torch.Tensor,  # [B] or [B, 1] - diffusion timestep
    ) -> Dict[str, torch.Tensor]:

        B, N_rec, _ = rec_atom_feats.shape
        N_lig = lig_atom_feats.shape[1]
        N_atoms = N_rec + N_lig
        N_rec_tokens = esm_embeddings.shape[1]
        N_lig_tokens = N_lig  # Ligand atoms = ligand tokens
        N_tokens = N_rec_tokens + N_lig_tokens
        device = rec_atom_feats.device
        dtype = rec_atom_feats.dtype

        # Scale coordinates down for numerical stability
        scale = self.config.coord_scale
        rec_atom_coords = rec_atom_coords / scale
        lig_atom_coords = lig_atom_coords / scale
        rec_atom_ref_coords = rec_atom_ref_coords / scale
        lig_atom_ref_coords = lig_atom_ref_coords / scale

        # Concatenate atom components
        atom_feats = torch.cat([rec_atom_feats, lig_atom_feats], dim=1)  # [B, N_atoms, 124]
        atom_coords = torch.cat([rec_atom_coords, lig_atom_coords], dim=1)  # [B, N_atoms, 3]
        atom_ref_coords = torch.cat([rec_atom_ref_coords, lig_atom_ref_coords], dim=1)  # [B, N_atoms, 3]

        # Build combined bond features (block diagonal structure)
        bond_feats = torch.zeros(B, N_atoms, N_atoms, rec_bond_feats.shape[-1], device=device, dtype=dtype)
        bond_feats[:, :N_rec, :N_rec, :] = rec_bond_feats
        bond_feats[:, N_rec:, N_rec:, :] = lig_bond_feats

        # Build block-diagonal mask from separate receptor and ligand masks
        # The comparison is per-batch, so global offsets don't affect the result
        # Receptor: atoms with same residue uid can attend to each other
        rec_block = rec_uid_vec.unsqueeze(-1) == rec_uid_vec.unsqueeze(-2)  # [B, N_rec, N_rec]
        # Ligand: atoms with same molecule uid can attend to each other
        lig_block = lig_uid_vec.unsqueeze(-1) == lig_uid_vec.unsqueeze(-2)  # [B, N_lig, N_lig]
        
        # Combine into block-diagonal structure (no cross-attention between rec and lig at block level)
        block_mask = torch.zeros(B, N_atoms, N_atoms, device=device, dtype=torch.bool)
        block_mask[:, :N_rec, :N_rec] = rec_block
        block_mask[:, N_rec:, N_rec:] = lig_block
        block_mask = block_mask.unsqueeze(-1).float()  # [B, N_atoms, N_atoms, 1]
        
        del rec_block, lig_block
        
        t_emb = self.time_embedder(t)  # [B, ns]

        # Initial embedding + timestep injection (stage 1)
        s_atom, esm_emb, unimol_emb = self.initial_embedder(atom_feats, esm_embeddings, unimol_embeddings)
        s_atom = self.time_adaln_initial(s_atom, t_emb.unsqueeze(1))  # [B, N_atoms, ns]
        
        del atom_feats, esm_embeddings, unimol_embeddings
        
        # Pairwise features
        ref_disto = torch.cdist(atom_ref_coords, atom_ref_coords)  # [B, N_atoms, N_atoms]
        ref_disto = ref_disto.unsqueeze(-1)  # [B, N_atoms, N_atoms, 1]


        p = self.pair_embedder(
            distogram=ref_disto,
            adj_mat=bond_feats,
            block_mask=block_mask
        )  # [B, N_atoms, N_atoms, pair_hidden_dim]
        
        del ref_disto, bond_feats, block_mask  # Free after pair embedding

        # Atom transformer forward
        for index, layer in enumerate(self.atom_layers.values()):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                s_atom, p = checkpoint(layer, s_atom, p, use_reentrant=False)
            else:
                s_atom, p = layer(q=s_atom, p=p)
            s_atom = self.adaln_atom_layers[f'adaln_atom_{index}'](s_atom, t_emb.unsqueeze(1))  # [B, N_atoms, ns]
        del p  # Free pairwise tensor after atom layers

        # EVA module for vector extraction
        v_atom = self.eva_module(
            x=s_atom,
            r=atom_coords
        )  # [B, N_atoms, nv, 3]
        v_atom = v_atom.reshape(B, N_atoms, -1)  # [B, N_atoms, nv*3]

        # Split receptor - ligand, scalar - vector features
        s_rec = s_atom[:, :N_rec, :]  # [B, N_rec, ns]
        s_lig = s_atom[:, N_rec:, :]  # [B, N_lig, ns]
        v_rec = v_atom[:, :N_rec, :]  # [B, N_rec, nv*3]
        v_lig = v_atom[:, N_rec:, :]  # [B, N_lig, nv*3]

        # Receptor: scatter atoms to residue tokens
        # Only scatter valid atoms (mask out padding with uid_vec=-1)
        # rec_uid_vec already has batch offsets from dataloader
        rec_mask_flat = rec_atom_mask.reshape(B * N_rec)
        rec_uid_flat = rec_uid_vec.reshape(B * N_rec)

        # Filter to valid atoms only for scatter
        valid_rec_idx = rec_mask_flat.nonzero(as_tuple=True)[0]
        s_rec_valid = s_rec.reshape(B * N_rec, -1)[valid_rec_idx]
        v_rec_valid = v_rec.reshape(B * N_rec, -1)[valid_rec_idx]
        rec_uid_valid = rec_uid_flat[valid_rec_idx]
        
        s_rec_tokens = self.pool(
            s_rec_valid,
            rec_uid_valid,
            dim_size=B * N_rec_tokens
        ).reshape(B, N_rec_tokens, -1)  # [B, N_rec_tokens, ns]

        v_rec_tokens = self.pool(
            v_rec_valid,
            rec_uid_valid,
            dim_size=B * N_rec_tokens
        ).reshape(B, N_rec_tokens, -1)  # [B, N_rec_tokens, nv*3]

        # Ligand: atoms are tokens directly (1:1 mapping)
        s_lig_tokens = s_lig  # [B, N_lig, ns]
        v_lig_tokens = v_lig  # [B, N_lig, nv*3]

        # Concatenate to form full token sequence
        s_token_from_atoms = torch.cat([s_rec_tokens, s_lig_tokens], dim=1)  # [B, N_tokens, ns]
        v_token = torch.cat([v_rec_tokens, v_lig_tokens], dim=1)  # [B, N_tokens, nv*3]

        # Token embeddings from ESM (residues) / UniMol (ligand atoms)
        s_token_emb = torch.cat([esm_emb, unimol_emb], dim=1)  # [B, N_tokens, ns]

        # Token-level Processing
        s_atom_tok = s_token_from_atoms
        s_emb_tok = s_token_emb

        # Initialize persistent pair representation z from token features
        z = s_atom_tok.unsqueeze(1) + s_atom_tok.unsqueeze(2)  # [B, N_tokens, N_tokens, ns]

        for index, layer in enumerate(self.token_layers.values()):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                s_atom_tok, s_emb_tok, v_token, z = checkpoint(
                    layer, s_atom_tok, s_emb_tok, v_token, z, use_reentrant=False
                )
            else:
                s_atom_tok, s_emb_tok, v_token, z = layer(
                    s_atom=s_atom_tok,
                    s_emb=s_emb_tok,
                    v=v_token,
                    z=z
                )
            s_atom_tok = self.adaln_emb_tokens[f'adaln_emb_tokens_{index}'](s_atom_tok, t_emb[:, :N_tokens, :] if t_emb.dim() == 3 else t_emb.unsqueeze(1))
            s_emb_tok = self.adaln_atom_tokens[f'adaln_atom_tokens_{index}'](s_emb_tok, t_emb[:, :N_tokens, :] if t_emb.dim() == 3 else t_emb.unsqueeze(1))

        del z  # Free pair features after token layers
            
        # Timestep injection after token layers (stage 3)
        # Unpool to Atom Level
        s_token_final = s_atom_tok + s_emb_tok  # [B, N_tokens, ns]

        # Split back to receptor/ligand tokens
        s_rec_tok_final = s_token_final[:, :N_rec_tokens, :]  # [B, N_rec_tokens, ns]
        s_lig_tok_final = s_token_final[:, N_rec_tokens:, :]  # [B, N_lig, ns]
        v_rec_tok_final = v_token[:, :N_rec_tokens, :]  # [B, N_rec_tokens, nv*3]
        v_lig_tok_final = v_token[:, N_rec_tokens:, :]  # [B, N_lig, nv*3]

        # Receptor: unpool residue tokens to atoms
        # rec_uid_vec has global indices from dataloader
        s_rec_unpooled = self.unpool(
            s_rec_tok_final.reshape(B * N_rec_tokens, -1),
            rec_uid_vec.reshape(B * N_rec)
        ).reshape(B, N_rec, -1)  # [B, N_rec, ns]

        v_rec_unpooled = self.unpool(
            v_rec_tok_final.reshape(B * N_rec_tokens, -1),
            rec_uid_vec.reshape(B * N_rec)
        ).reshape(B, N_rec, -1)  # [B, N_rec, nv*3]

        # Ligand: tokens are atoms (1:1 mapping)
        s_lig_unpooled = s_lig_tok_final  # [B, N_lig, ns]
        v_lig_unpooled = v_lig_tok_final  # [B, N_lig, nv*3]

        del s_lig_tok_final, v_lig_tok_final, s_rec_tok_final, v_rec_tok_final

        # Residual connection from atom features
        s_rec_unpooled = s_rec_unpooled + s_rec
        v_rec_unpooled = v_rec_unpooled + v_rec
        s_lig_unpooled = s_lig_unpooled + s_lig
        v_lig_unpooled = v_lig_unpooled + v_lig

        # Concatenate back to atom level
        s_unpooled = torch.cat([s_rec_unpooled, s_lig_unpooled], dim=1)  # [B, N_atoms, ns]
        v_unpooled = torch.cat([v_rec_unpooled, v_lig_unpooled], dim=1)  # [B, N_atoms, nv*3]

        # Fuse with original atom features
        s_final = self.atom_fusion(torch.cat([s_atom, s_unpooled], dim=-1))

        # Flow Matching Head
        velocity = self.flow_head(s_final, v_unpooled)  # [B, N_atoms, 3]

        # Scale velocity vectors back to original coordinate space
        velocity = velocity * scale

        # Split velocities back to receptor and ligand
        rec_velocity = velocity[:, :N_rec, :]  # [B, N_rec, 3]
        lig_velocity = velocity[:, N_rec:, :]  # [B, N_lig, 3]

        return {
            'velocity': velocity,
            'rec_velocity': rec_velocity,
            'lig_velocity': lig_velocity,
        }


class FlexModelV2(nn.Module):
    '''
    Simpler model without aggregating atoms to tokens:
    We focus on placing atoms to the right place in the receptor.
    '''
    
    def __init__(self, config: FlexConfig):
        super().__init__()
        self.config = config
        self._instantiate_layers()
        self._initialize_weights()
        
    def forward(self,
                rec_atom_feats: torch.Tensor,
                rec_atom_coords: torch.Tensor,
                rec_atom_ref_coords: torch.Tensor,
                rec_bond_feats: torch.Tensor,
                rec_uid_vec: torch.Tensor,
                rec_atom_mask: torch.Tensor,
                rec_backbone_mask: torch.Tensor,  # [B, N_rec] - 1 for backbone, 0 for sidechain
                esm_embeddings: torch.Tensor,
                lig_atom_feats: torch.Tensor,
                lig_atom_coords: torch.Tensor,
                lig_atom_ref_coords: torch.Tensor,
                lig_bond_feats: torch.Tensor,
                lig_uid_vec: torch.Tensor,
                lig_atom_mask: torch.Tensor,
                unimol_embeddings: torch.Tensor,
                t: torch.Tensor,
                return_attn: bool = False,
                ):
        device = rec_atom_feats.device
        dtype = rec_atom_feats.dtype
        B, N_rec, _ = rec_atom_feats.shape
        N_lig = lig_atom_feats.shape[1]
        N_atoms = N_rec + N_lig
        
        # Scale coordinates down for numerical stability
        scale = self.config.coord_scale
        rec_atom_coords = rec_atom_coords / scale
        lig_atom_coords = lig_atom_coords / scale

        # Concatenate atom components
        atom_feats = torch.cat([rec_atom_feats, lig_atom_feats], dim=1)  # [B, N_atoms, 124]
        atom_coords = torch.cat([rec_atom_coords, lig_atom_coords], dim=1)  # [B, N_atoms, 3]
        atom_ref_coords = torch.cat([rec_atom_ref_coords, lig_atom_ref_coords], dim=1)  # [B, N_atoms, 3]
        
        # Build combined bond features (block diagonal structure)
        bond_feats = torch.zeros(B, N_atoms, N_atoms, rec_bond_feats.shape[-1], device=device, dtype=dtype)
        bond_feats[:, :N_rec, :N_rec, :] = rec_bond_feats
        bond_feats[:, N_rec:, N_rec:, :] = lig_bond_feats

        # Build block-diagonal mask from separate receptor and ligand masks
        rec_block = rec_uid_vec.unsqueeze(-1) == rec_uid_vec.unsqueeze(-2)  # [B, N_rec, N_rec]
        lig_block = lig_uid_vec.unsqueeze(-1) == lig_uid_vec.unsqueeze(-2)  # [B, N_lig, N_lig]
        
        # Combine into block-diagonal structure (no cross-attention between rec and lig at block level)
        block_mask = torch.zeros(B, N_atoms, N_atoms, device=device, dtype=torch.bool)
        block_mask[:, :N_rec, :N_rec] = rec_block
        block_mask[:, N_rec:, N_rec:] = lig_block
        block_mask = block_mask.unsqueeze(-1).float()  # [B, N_atoms, N_atoms, 1]
        
        del rec_block, lig_block
        
        # Time embedding
        t_emb = self.time_embedder(t)

        # Initial embedding
        q, esm_emb, unimol_emb = self.initial_embedder(atom_feats, esm_embeddings, unimol_embeddings)
        if self.config.use_embeddings:
            q_esm = torch.gather(esm_emb, dim=1, index=rec_uid_vec.unsqueeze(-1).expand(-1, -1, esm_emb.shape[-1]))
            q_rec = q[:,:N_rec,:]
            q_lig = q[:,N_rec:,:]
            Q_rec = self.rec_fusion(torch.cat([q_esm, q_rec], dim=-1))
            Q_lig = self.lig_fusion(torch.cat([unimol_emb, q_lig], dim=-1))
            q = torch.cat([Q_rec, Q_lig], dim=1)
            del q_esm, q_rec, q_lig, Q_rec, Q_lig
        else:
            del esm_emb, unimol_emb
        q = self.adaln_initial(q, t_emb.unsqueeze(1))

        # Pairwise feature embedding
        ref_disto = torch.cdist(atom_ref_coords, atom_ref_coords) # [B, N_atoms, N_atoms]
        ref_disto = ref_disto.unsqueeze(-1) # [B, N_atoms, N_atoms, 1]

        p = self.pair_embedder(
            distogram=ref_disto,
            adj_mat=bond_feats,
            block_mask=block_mask
        )  # [B, N_atoms, N_atoms, pair_hidden_dim]
        del ref_disto, bond_feats, block_mask

        # Atom transformer modules
        atom_attn_entropies = []
        for index, layer in enumerate(self.atom_layers.values()):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                q, p = checkpoint(layer, q, p, use_reentrant=False)
            else:
                if return_attn:
                    q, p, atom_stats = layer(q=q, p=p, return_attn=True)
                    atom_attn_entropies.append(atom_stats['entropy'])
                else:
                    q, p = layer(q=q, p=p)
            q = self.adaln_atom_layers[f'adaln_atom_{index}'](q, t_emb.unsqueeze(1))

        # EVA module for vector extraction
        attn_stats = {} if return_attn else None
        if return_attn and atom_attn_entropies:
            attn_stats['atom_entropy'] = torch.stack(atom_attn_entropies).mean()

        if return_attn:
            v, eva_attn_stats = self.eva_module(x=q, r=atom_coords, return_attn=True)
            attn_stats['eva_entropy'] = eva_attn_stats['entropy']
        else:
            v = self.eva_module(x=q, r=atom_coords)
        v = v.reshape(B, N_atoms, -1)  # [B, N_atoms, nv*3]

        geo_attn_entropies = []
        for index, layer in enumerate(self.geo_attention_layers.values()):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                q, v = checkpoint(layer, q, v, p, use_reentrant=False)
            else:
                if return_attn:
                    q, v, geo_stats = layer(scalar_feats=q, vector_feats=v, bias=p, return_attn=True)
                    geo_attn_entropies.append(geo_stats['entropy'])
                else:
                    q, v = layer(scalar_feats=q, vector_feats=v, bias=p)
            q = self.adaln_geo_layers[f'adaln_geo_{index}'](q, t_emb.unsqueeze(1))

        if return_attn and geo_attn_entropies:
            attn_stats['geo_entropy'] = torch.stack(geo_attn_entropies).mean()

        flow = self.flow_head(q, v)

        # Scale flow vectors back to original coordinate space
        flow = flow * scale

        rec_flow = flow[:, :N_rec, :]
        lig_flow = flow[:, N_rec:, :]

        result = {
            'rec_flow': rec_flow,
            'lig_flow': lig_flow,
        }
        if return_attn:
            result['attn_stats'] = attn_stats
        return result
        
    def _instantiate_layers(self):
        self.time_embedder = TimeEmbedder(
            hidden_dim=self.config.ns,
            output_dim=self.config.ns
        )
        self.initial_embedder = InitialEmbedder(
            atom_dim=self.config.atom_feat_dim,
            use_embeddings=self.config.use_embeddings,
            esm_dim=self.config.esm_dim,
            unimol_dim=self.config.unimol_dim,
            hidden_dim=self.config.ns,
            dropout=self.config.initial_dropout,
        )
        self.rec_fusion = nn.Linear(self.config.ns * 2, self.config.ns, bias=False)
        self.lig_fusion = nn.Linear(self.config.ns * 2, self.config.ns, bias=False)
        
        self.adaln_initial = AdaLN(in_dim=self.config.ns)
        
        self.pair_embedder = PairwiseFeatureEmbedder(
            hidden_dim=self.config.pair_hidden_dim,
            out_dim=self.config.pair_hidden_dim
        )
        self.atom_layers = nn.ModuleDict(
            {f'atom_layer_{i}': AtomTransformer(
                single_hidden_dim=self.config.ns,
                pair_hidden_dim=self.config.pair_hidden_dim,
                num_heads=self.config.num_atom_heads,
                attn_dropout=self.config.attn_dropout,
                use_gated_attention=self.config.use_gated_attention,
                use_outer_product=self.config.use_outer_product,
                use_qk_norm=self.config.use_qk_norm,
            ) for i in range(self.config.num_atom_layers)}
        )
        self.adaln_atom_layers = nn.ModuleDict(
            {f'adaln_atom_{i}': AdaLN(in_dim=self.config.ns) for i in range(self.config.num_atom_layers)}
        )
        self.eva_module = EVAModule(
            scalar_dim=self.config.ns,
            num_heads=self.config.nv,
            attenuate_by_dist=self.config.attenuate_by_dist,
            soft_cap=self.config.soft_cap,
        )
        self.geo_attention_layers = nn.ModuleDict(
            {f'geo_attention_{i}': GeometricAttentionModule(
                ns=self.config.ns,
                nv=self.config.nv,
                num_heads=self.config.num_geo_heads,
                bias_dim=self.config.pair_hidden_dim,
                attn_dropout=self.config.attn_dropout,
                use_gated_attention=self.config.use_gated_attention,
                soft_cap=self.config.soft_cap,
            ) for i in range(self.config.num_geo_layers)}
        )
        self.adaln_geo_layers = nn.ModuleDict(
            {f'adaln_geo_{i}': AdaLN(in_dim=self.config.ns) for i in range(self.config.num_geo_layers)}
        )
        # Separate flow heads for backbone, sidechain, and ligand
        # Each learns different flow patterns
        self.flow_head = FlowMatchingModule(
            ns=self.config.ns,
            nv=self.config.nv,
            dropout=self.config.output_dropout
        )
        
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.rec_fusion.weight)
        nn.init.xavier_uniform_(self.lig_fusion.weight)


class FlexModelV3(nn.Module):
    '''
    Similar to FlexModelV2, but with constraint matrix input.
    Added constraint matrix input to the pair embedding module.
    Optimization for use_embeddings = False case
    '''
    
    def __init__(self, config: FlexConfig):
        super().__init__()
        self.config = config
        self._instantiate_layers()
        self._initialize_weights()
        
    def forward(self,
                rec_atom_feats: torch.Tensor,
                rec_atom_coords: torch.Tensor,
                rec_atom_ref_coords: torch.Tensor,
                rec_constraint_matrix: torch.Tensor,
                rec_bond_feats: torch.Tensor,
                rec_uid_vec: torch.Tensor,
                rec_atom_mask: torch.Tensor,
                rec_backbone_mask: torch.Tensor,  # [B, N_rec] - 1 for backbone, 0 for sidechain
                esm_embeddings: torch.Tensor,
                lig_atom_feats: torch.Tensor,
                lig_atom_coords: torch.Tensor,
                lig_atom_ref_coords: torch.Tensor,
                lig_constraint_matrix: torch.Tensor,
                lig_bond_feats: torch.Tensor,
                lig_uid_vec: torch.Tensor,
                lig_atom_mask: torch.Tensor,
                unimol_embeddings: torch.Tensor,
                t: torch.Tensor,
                return_attn: bool = False,
                ):
        device = rec_atom_feats.device
        dtype = rec_atom_feats.dtype
        B, N_rec, _ = rec_atom_feats.shape
        N_lig = lig_atom_feats.shape[1]
        N_atoms = N_rec + N_lig
        
        # Scale coordinates down for numerical stability
        scale = self.config.coord_scale
        rec_atom_coords = rec_atom_coords / scale
        lig_atom_coords = lig_atom_coords / scale

        # Concatenate atom components
        atom_feats = torch.cat([rec_atom_feats, lig_atom_feats], dim=1)  # [B, N_atoms, 124]
        atom_coords = torch.cat([rec_atom_coords, lig_atom_coords], dim=1)  # [B, N_atoms, 3]
        atom_ref_coords = torch.cat([rec_atom_ref_coords, lig_atom_ref_coords], dim=1)  # [B, N_atoms, 3]
        
        # Build combined bond features (block diagonal structure)
        bond_feats = torch.zeros(B, N_atoms, N_atoms, rec_bond_feats.shape[-1], device=device, dtype=dtype)
        bond_feats[:, :N_rec, :N_rec, :] = rec_bond_feats
        bond_feats[:, N_rec:, N_rec:, :] = lig_bond_feats
        
        # Build combined constraint matrix (block diagonal structure)
        constraint_matrix = torch.zeros(B, N_atoms, N_atoms, rec_constraint_matrix.shape[-1], device=device, dtype=dtype)
        constraint_matrix[:, :N_rec, :N_rec, :] = rec_constraint_matrix
        constraint_matrix[:, N_rec:, N_rec:, :] = lig_constraint_matrix

        # Build block-diagonal mask from separate receptor and ligand masks
        rec_block = rec_uid_vec.unsqueeze(-1) == rec_uid_vec.unsqueeze(-2)  # [B, N_rec, N_rec]
        lig_block = lig_uid_vec.unsqueeze(-1) == lig_uid_vec.unsqueeze(-2)  # [B, N_lig, N_lig]
        
        # Combine into block-diagonal structure (no cross-attention between rec and lig at block level)
        block_mask = torch.zeros(B, N_atoms, N_atoms, device=device, dtype=torch.bool)
        block_mask[:, :N_rec, :N_rec] = rec_block
        block_mask[:, N_rec:, N_rec:] = lig_block
        block_mask = block_mask.unsqueeze(-1).float()  # [B, N_atoms, N_atoms, 1]
        
        del rec_block, lig_block
        
        # Time embedding
        t_emb = self.time_embedder(t)

        # Initial embedding
        q, esm_emb, unimol_emb = self.initial_embedder(atom_feats, esm_embeddings, unimol_embeddings)
        if self.config.use_embeddings:
            q_esm = torch.gather(esm_emb, dim=1, index=rec_uid_vec.unsqueeze(-1).expand(-1, -1, esm_emb.shape[-1]))
            q_rec = q[:,:N_rec,:]
            q_lig = q[:,N_rec:,:]
            Q_rec = self.rec_fusion(torch.cat([q_esm, q_rec], dim=-1))
            Q_lig = self.lig_fusion(torch.cat([unimol_emb, q_lig], dim=-1))
            q = torch.cat([Q_rec, Q_lig], dim=1)
            del q_esm, q_rec, q_lig, Q_rec, Q_lig
        else:
            del esm_emb, unimol_emb
        q = self.adaln_initial(q, t_emb.unsqueeze(1))

        # Pairwise feature embedding
        ref_disto = torch.cdist(atom_ref_coords, atom_ref_coords) # [B, N_atoms, N_atoms]
        ref_disto = ref_disto.unsqueeze(-1) # [B, N_atoms, N_atoms, 1]

        p = self.pair_embedder(
            distogram=ref_disto,
            adj_mat=bond_feats,
            constraint_mat=constraint_matrix,
            block_mask=block_mask
        )  # [B, N_atoms, N_atoms, pair_hidden_dim]
        del ref_disto, bond_feats, constraint_matrix, block_mask

        # Atom transformer modules
        atom_attn_entropies = []
        for index, layer in enumerate(self.atom_layers.values()):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                q, p = checkpoint(layer, q, p, use_reentrant=False)
            else:
                if return_attn:
                    q, p, atom_stats = layer(q=q, p=p, return_attn=True)
                    atom_attn_entropies.append(atom_stats['entropy'])
                else:
                    q, p = layer(q=q, p=p)
            q = self.adaln_atom_layers[f'adaln_atom_{index}'](q, t_emb.unsqueeze(1))

        # EVA module for vector extraction
        attn_stats = {} if return_attn else None
        if return_attn and atom_attn_entropies:
            attn_stats['atom_entropy'] = torch.stack(atom_attn_entropies).mean()

        if return_attn:
            v, eva_attn_stats = self.eva_module(x=q, r=atom_coords, return_attn=True)
            attn_stats['eva_entropy'] = eva_attn_stats['entropy']
        else:
            v = self.eva_module(x=q, r=atom_coords)
        eva_output = v # Copy for gramian loss term
        v = v.reshape(B, N_atoms, -1)  # [B, N_atoms, nv*3]

        geo_attn_entropies = []
        for index, layer in enumerate(self.geo_attention_layers.values()):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                q, v = checkpoint(layer, q, v, p, use_reentrant=False)
            else:
                if return_attn:
                    q, v, geo_stats = layer(scalar_feats=q, vector_feats=v, bias=p, return_attn=True)
                    geo_attn_entropies.append(geo_stats['entropy'])
                else:
                    q, v = layer(scalar_feats=q, vector_feats=v, bias=p)
            q = self.adaln_geo_layers[f'adaln_geo_{index}'](q, t_emb.unsqueeze(1))

        if return_attn and geo_attn_entropies:
            attn_stats['geo_entropy'] = torch.stack(geo_attn_entropies).mean()

        flow = self.flow_head(q, v)

        # Scale flow vectors back to original coordinate space
        flow = flow * scale

        rec_flow = flow[:, :N_rec, :]
        lig_flow = flow[:, N_rec:, :]

        result = {
            'rec_flow': rec_flow,
            'lig_flow': lig_flow,
            'eva_output': eva_output,
        }
        if return_attn:
            result['attn_stats'] = attn_stats
        return result
        
    def _instantiate_layers(self):
        self.time_embedder = TimeEmbedder(
            hidden_dim=self.config.ns,
            output_dim=self.config.ns
        )
        self.initial_embedder = InitialEmbedder(
            atom_dim=self.config.atom_feat_dim,
            use_embeddings=self.config.use_embeddings,
            esm_dim=self.config.esm_dim,
            unimol_dim=self.config.unimol_dim,
            hidden_dim=self.config.ns,
            dropout=self.config.initial_dropout,
        )
        if self.config.use_embeddings:
            self.rec_fusion = nn.Linear(self.config.ns * 2, self.config.ns, bias=False)
            self.lig_fusion = nn.Linear(self.config.ns * 2, self.config.ns, bias=False)
        
        self.adaln_initial = AdaLN(in_dim=self.config.ns)
        
        self.pair_embedder = PairwiseFeatureEmbedderWithConstraints(
            hidden_dim=self.config.pair_hidden_dim,
            out_dim=self.config.pair_hidden_dim
        )
        self.atom_layers = nn.ModuleDict(
            {f'atom_layer_{i}': AtomTransformer(
                single_hidden_dim=self.config.ns,
                pair_hidden_dim=self.config.pair_hidden_dim,
                num_heads=self.config.num_atom_heads,
                attn_dropout=self.config.attn_dropout,
                use_gated_attention=self.config.use_gated_attention,
                use_outer_product=self.config.use_outer_product,
                use_qk_norm=self.config.use_qk_norm,
            ) for i in range(self.config.num_atom_layers)}
        )
        self.adaln_atom_layers = nn.ModuleDict(
            {f'adaln_atom_{i}': AdaLN(in_dim=self.config.ns) for i in range(self.config.num_atom_layers)}
        )
        self.eva_module = EVAModule(
            scalar_dim=self.config.ns,
            num_heads=self.config.nv,
            attenuate_by_dist=self.config.attenuate_by_dist,
            soft_cap=self.config.soft_cap,
        )
        self.geo_attention_layers = nn.ModuleDict(
            {f'geo_attention_{i}': GeometricAttentionModule(
                ns=self.config.ns,
                nv=self.config.nv,
                num_heads=self.config.num_geo_heads,
                bias_dim=self.config.pair_hidden_dim,
                attn_dropout=self.config.attn_dropout,
                use_gated_attention=self.config.use_gated_attention,
                soft_cap=self.config.soft_cap,
            ) for i in range(self.config.num_geo_layers)}
        )
        self.adaln_geo_layers = nn.ModuleDict(
            {f'adaln_geo_{i}': AdaLN(in_dim=self.config.ns) for i in range(self.config.num_geo_layers)}
        )
        # Separate flow heads for backbone, sidechain, and ligand
        # Each learns different flow patterns
        self.flow_head = FlowMatchingModule(
            ns=self.config.ns,
            nv=self.config.nv,
            dropout=self.config.output_dropout
        )
        
    def _initialize_weights(self):
        if self.config.use_embeddings:
            nn.init.xavier_uniform_(self.rec_fusion.weight)
            nn.init.xavier_uniform_(self.lig_fusion.weight)
        else:
            print('WARNING: Model not using ESM/UniMol embeddings')
            print('If not running ablation study please check config files')


class FlexModelV4(nn.Module):
    '''
    Two-trunk architecture separating system-level chemistry from
    per-timestep flow matching.

    Chemistry trunk (B=1): AtomTransformer + TriangleUpdate (interleaved), no timestep.
    Flow matching trunk (B timesteps): EVA + GeometricAttention + FlowMatchingModule.
    '''

    def __init__(self, config: FlexConfig):
        super().__init__()
        self.config = config
        self._instantiate_layers()
        self._initialize_weights()

    def forward_chemistry(
        self,
        rec_atom_feats: torch.Tensor,       # [S, N_rec, atom_feat_dim]
        rec_atom_ref_coords: torch.Tensor,   # [S, N_rec, 3]
        rec_constraint_matrix: torch.Tensor, # [S, N_rec, N_rec, C]
        rec_bond_feats: torch.Tensor,        # [S, N_rec, N_rec, 26]
        rec_uid_vec: torch.Tensor,           # [S, N_rec]
        esm_embeddings: torch.Tensor,        # [S, N_rec_tokens, esm_dim]
        lig_atom_feats: torch.Tensor,        # [S, N_lig, atom_feat_dim]
        lig_atom_ref_coords: torch.Tensor,   # [S, N_lig, 3]
        lig_constraint_matrix: torch.Tensor, # [S, N_lig, N_lig, C]
        lig_bond_feats: torch.Tensor,        # [S, N_lig, N_lig, 26]
        lig_uid_vec: torch.Tensor,           # [S, N_lig]
        unimol_embeddings: torch.Tensor,     # [S, N_lig, unimol_dim]
        rec_atom_mask: Optional[torch.Tensor] = None,
        lig_atom_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Process system-level features once per distinct padded system. No
        timestep conditioning.

        Returns:
            q_chem: [S, N_atoms, ns]
            p_chem: [S, N_atoms, N_atoms, pair_hidden_dim]
        '''
        device = rec_atom_feats.device
        dtype = rec_atom_feats.dtype
        S, N_rec, _ = rec_atom_feats.shape
        N_lig = lig_atom_feats.shape[1]
        N_atoms = N_rec + N_lig

        if rec_atom_mask is None:
            rec_atom_mask = torch.ones(S, N_rec, device=device, dtype=torch.bool)
        else:
            rec_atom_mask = rec_atom_mask.bool()
        if lig_atom_mask is None:
            lig_atom_mask = torch.ones(S, N_lig, device=device, dtype=torch.bool)
        else:
            lig_atom_mask = lig_atom_mask.bool()
        atom_mask = torch.cat([rec_atom_mask, lig_atom_mask], dim=1)
        atom_mask_f = atom_mask.to(dtype).unsqueeze(-1)
        pair_mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
        pair_mask_f = pair_mask.to(dtype).unsqueeze(-1)

        # Concatenate rec+lig
        atom_feats = torch.cat([rec_atom_feats, lig_atom_feats], dim=1)
        atom_ref_coords = torch.cat([rec_atom_ref_coords, lig_atom_ref_coords], dim=1)

        # Block-diagonal bond features
        bond_feats = torch.zeros(S, N_atoms, N_atoms, rec_bond_feats.shape[-1], device=device, dtype=dtype)
        bond_feats[:, :N_rec, :N_rec, :] = rec_bond_feats
        bond_feats[:, N_rec:, N_rec:, :] = lig_bond_feats

        # Block-diagonal constraint matrix
        constraint_matrix = torch.zeros(S, N_atoms, N_atoms, rec_constraint_matrix.shape[-1], device=device, dtype=dtype)
        constraint_matrix[:, :N_rec, :N_rec, :] = rec_constraint_matrix
        constraint_matrix[:, N_rec:, N_rec:, :] = lig_constraint_matrix

        # Block-diagonal mask, with padded atoms excluded.
        rec_block = rec_uid_vec.unsqueeze(-1) == rec_uid_vec.unsqueeze(-2)
        rec_block = rec_block & rec_atom_mask.unsqueeze(-1) & rec_atom_mask.unsqueeze(-2)
        lig_block = lig_uid_vec.unsqueeze(-1) == lig_uid_vec.unsqueeze(-2)
        lig_block = lig_block & lig_atom_mask.unsqueeze(-1) & lig_atom_mask.unsqueeze(-2)
        block_mask = torch.zeros(S, N_atoms, N_atoms, device=device, dtype=torch.bool)
        block_mask[:, :N_rec, :N_rec] = rec_block
        block_mask[:, N_rec:, N_rec:] = lig_block
        block_mask = block_mask.unsqueeze(-1).float()
        del rec_block, lig_block

        # Initial embedding (no timestep)
        q, esm_emb, unimol_emb = self.initial_embedder(atom_feats, esm_embeddings, unimol_embeddings)
        if self.config.use_embeddings:
            gather_uid = rec_uid_vec.long().clamp(min=0)
            q_esm = torch.gather(esm_emb, dim=1, index=gather_uid.unsqueeze(-1).expand(-1, -1, esm_emb.shape[-1]))
            q_rec = q[:, :N_rec, :]
            q_lig = q[:, N_rec:, :]
            Q_rec = self.rec_fusion(torch.cat([q_esm, q_rec], dim=-1))
            Q_lig = self.lig_fusion(torch.cat([unimol_emb, q_lig], dim=-1))
            q = torch.cat([Q_rec, Q_lig], dim=1)
            del q_esm, q_rec, q_lig, Q_rec, Q_lig
        else:
            del esm_emb, unimol_emb
        q = q * atom_mask_f

        # Pairwise features
        ref_disto = torch.cdist(atom_ref_coords, atom_ref_coords).unsqueeze(-1)
        p = self.pair_embedder(
            distogram=ref_disto,
            adj_mat=bond_feats,
            constraint_mat=constraint_matrix,
            block_mask=block_mask,
        )
        p = p * pair_mask_f
        # Block-diagonal pair mask for triangle updates: [S, N, N], 1 on valid
        # intra-entity pairs only. Keeps triangle propagation intra-molecular
        # (cross-entity pairs carry no geometry here) and excludes padding.
        tri_mask = block_mask.squeeze(-1).contiguous()
        del ref_disto, bond_feats, constraint_matrix, block_mask

        # Interleaved AtomTransformer + TriangleUpdate
        for index, (atom_layer, tri_layer) in enumerate(
            zip(self.atom_layers.values(), self.triangle_layers.values())
        ):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                q, p = checkpoint(
                    atom_layer, q, p,
                    use_reentrant=False,
                    atom_mask=atom_mask,
                    pair_mask=pair_mask,
                )
                p = p * pair_mask_f
                p = checkpoint(tri_layer, p, tri_mask, use_reentrant=False)
            else:
                q, p = atom_layer(q=q, p=p, atom_mask=atom_mask, pair_mask=pair_mask)
                p = p * pair_mask_f
                p = tri_layer(p, mask=tri_mask)
            p = p * pair_mask_f
            q = q * atom_mask_f

        del tri_mask
        return q, p

    def forward_flow(
        self,
        q_chem: torch.Tensor,     # [S, N_atoms, ns] or [B, N_atoms, ns]
        p_chem: torch.Tensor,     # [S, N_atoms, N_atoms, pair_hidden_dim] or [B, ...]
        atom_coords: torch.Tensor, # [B, N_atoms, 3] — noisy coordinates x_t
        t: torch.Tensor,          # [B]
        N_rec: int,
        atom_eva_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        system_index_per_timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        '''
        Per-timestep flow matching given cached chemistry features.
        '''
        B = atom_coords.shape[0]
        N_atoms = atom_coords.shape[1]
        scale = self.config.coord_scale
        atom_coords = atom_coords / scale

        if system_index_per_timestep is not None:
            sys_idx = system_index_per_timestep.to(q_chem.device).long()
            q = q_chem.index_select(0, sys_idx)
            p = p_chem.index_select(0, sys_idx)
        elif q_chem.shape[0] == 1 and B > 1:
            q = q_chem.expand(B, -1, -1)
            p = p_chem
        elif q_chem.shape[0] == B:
            q = q_chem
            p = p_chem
        else:
            raise ValueError(
                f"Chemistry batch ({q_chem.shape[0]}) cannot feed flow batch ({B}); "
                "provide system_index_per_timestep for padded multi-system batches."
            )

        mask_f = atom_eva_mask.to(q.dtype).unsqueeze(-1) if atom_eva_mask is not None else None

        # Timestep embedding and injection
        t_emb = self.time_embedder(t)
        q = self.adaln_flow_init(q, t_emb.unsqueeze(1))
        if mask_f is not None:
            q = q * mask_f

        # EVA: extract vector features from noisy coordinates
        attn_stats = {} if return_attn else None
        if return_attn:
            v, eva_attn_stats = self.eva_module(
                x=q, r=atom_coords, atom_mask=atom_eva_mask, return_attn=True,
            )
            attn_stats['eva_entropy'] = eva_attn_stats['entropy']
        else:
            v = self.eva_module(x=q, r=atom_coords, atom_mask=atom_eva_mask)
        v = v.reshape(B, N_atoms, -1)

        # Geometric attention layers with AdaLN(t). Pair tensor p is static
        # chemistry context (AF3-faithful: diffusion path has no triangle updates).
        geo_attn_entropies = []
        for index, layer in enumerate(self.geo_attention_layers.values()):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                q, v = checkpoint(layer, q, v, p, atom_eva_mask, use_reentrant=False)
            else:
                if return_attn:
                    q, v, geo_stats = layer(
                        scalar_feats=q, vector_feats=v, bias=p,
                        atom_mask=atom_eva_mask, return_attn=True,
                    )
                    geo_attn_entropies.append(geo_stats['entropy'])
                else:
                    q, v = layer(scalar_feats=q, vector_feats=v, bias=p, atom_mask=atom_eva_mask)
            q = self.adaln_geo_layers[f'adaln_geo_{index}'](q, t_emb.unsqueeze(1))
            if mask_f is not None:
                q = q * mask_f
                v = v * mask_f

        if return_attn and geo_attn_entropies:
            attn_stats['geo_entropy'] = torch.stack(geo_attn_entropies).mean()

        del p  # Free pair tensor before flow_head

        # Flow head
        flow = self.flow_head(q, v)
        flow = flow * scale
        if atom_eva_mask is not None:
            flow = flow * atom_eva_mask.to(flow.dtype).unsqueeze(-1)

        rec_flow = flow[:, :N_rec, :]
        lig_flow = flow[:, N_rec:, :]

        result = {
            'rec_flow': rec_flow,
            'lig_flow': lig_flow,
            'eva_output': None,
        }
        if return_attn:
            result['attn_stats'] = attn_stats
        return result

    def forward(
        self,
        rec_atom_feats: torch.Tensor,
        rec_atom_coords: torch.Tensor,
        rec_atom_ref_coords: torch.Tensor,
        rec_constraint_matrix: torch.Tensor,
        rec_bond_feats: torch.Tensor,
        rec_uid_vec: torch.Tensor,
        rec_atom_mask: torch.Tensor,
        rec_backbone_mask: torch.Tensor,
        esm_embeddings: torch.Tensor,
        lig_atom_feats: torch.Tensor,
        lig_atom_coords: torch.Tensor,
        lig_atom_ref_coords: torch.Tensor,
        lig_constraint_matrix: torch.Tensor,
        lig_bond_feats: torch.Tensor,
        lig_uid_vec: torch.Tensor,
        lig_atom_mask: torch.Tensor,
        unimol_embeddings: torch.Tensor,
        t: torch.Tensor,
        return_attn: bool = False,
        q_chem: Optional[torch.Tensor] = None,
        p_chem: Optional[torch.Tensor] = None,
        rec_missing_mask: Optional[torch.Tensor] = None,
        system_index_per_timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        '''
        V3-compatible entry point. If q_chem/p_chem are provided, skips chemistry trunk.
        '''
        N_rec = rec_atom_feats.shape[1]

        if q_chem is None or p_chem is None:
            q_chem, p_chem = self.forward_chemistry(
                rec_atom_feats=rec_atom_feats[0:1],
                rec_atom_ref_coords=rec_atom_ref_coords[0:1],
                rec_constraint_matrix=rec_constraint_matrix[0:1],
                rec_bond_feats=rec_bond_feats[0:1],
                rec_uid_vec=rec_uid_vec[0:1],
                esm_embeddings=esm_embeddings[0:1],
                lig_atom_feats=lig_atom_feats[0:1],
                lig_atom_ref_coords=lig_atom_ref_coords[0:1],
                lig_constraint_matrix=lig_constraint_matrix[0:1],
                lig_bond_feats=lig_bond_feats[0:1],
                lig_uid_vec=lig_uid_vec[0:1],
                unimol_embeddings=unimol_embeddings[0:1],
                rec_atom_mask=rec_atom_mask[0:1],
                lig_atom_mask=lig_atom_mask[0:1],
            )

        atom_coords = torch.cat([rec_atom_coords, lig_atom_coords], dim=1)
        if rec_missing_mask is not None:
            rec_eva_mask = rec_atom_mask.bool() & (~rec_missing_mask.bool())
        else:
            rec_eva_mask = rec_atom_mask.bool()
        atom_eva_mask = torch.cat([rec_eva_mask, lig_atom_mask.bool()], dim=1)

        return self.forward_flow(
            q_chem=q_chem,
            p_chem=p_chem,
            atom_coords=atom_coords,
            t=t,
            N_rec=N_rec,
            atom_eva_mask=atom_eva_mask,
            return_attn=return_attn,
            system_index_per_timestep=system_index_per_timestep,
        )

    def _instantiate_layers(self):
        # --- Chemistry Trunk ---
        self.initial_embedder = InitialEmbedder(
            atom_dim=self.config.atom_feat_dim,
            use_embeddings=self.config.use_embeddings,
            esm_dim=self.config.esm_dim,
            unimol_dim=self.config.unimol_dim,
            hidden_dim=self.config.ns,
            dropout=self.config.initial_dropout,
        )
        if self.config.use_embeddings:
            self.rec_fusion = nn.Linear(self.config.ns * 2, self.config.ns, bias=False)
            self.lig_fusion = nn.Linear(self.config.ns * 2, self.config.ns, bias=False)

        self.pair_embedder = PairwiseFeatureEmbedderWithConstraints(
            hidden_dim=self.config.pair_hidden_dim,
            out_dim=self.config.pair_hidden_dim,
        )

        self.atom_layers = nn.ModuleDict({
            f'atom_layer_{i}': AtomTransformer(
                single_hidden_dim=self.config.ns,
                pair_hidden_dim=self.config.pair_hidden_dim,
                num_heads=self.config.num_atom_heads,
                attn_dropout=self.config.attn_dropout,
                use_gated_attention=self.config.use_gated_attention,
                use_outer_product=self.config.use_outer_product,
                use_qk_norm=self.config.use_qk_norm,
            ) for i in range(self.config.num_atom_layers)
        })

        self.triangle_layers = nn.ModuleDict({
            f'triangle_layer_{i}': TriangleUpdatingModule(
                ns=self.config.pair_hidden_dim,
            ) for i in range(self.config.num_atom_layers)
        })

        # --- Flow Matching Trunk ---
        self.time_embedder = TimeEmbedder(
            hidden_dim=self.config.ns,
            output_dim=self.config.ns,
        )

        self.adaln_flow_init = AdaLN(in_dim=self.config.ns)

        self.eva_module = EVAModule(
            scalar_dim=self.config.ns,
            num_heads=self.config.nv,
            attenuate_by_dist=self.config.attenuate_by_dist,
            soft_cap=self.config.soft_cap,
        )

        self.geo_attention_layers = nn.ModuleDict({
            f'geo_attention_{i}': GeometricAttentionModule(
                ns=self.config.ns,
                nv=self.config.nv,
                num_heads=self.config.num_geo_heads,
                bias_dim=self.config.pair_hidden_dim,
                attn_dropout=self.config.attn_dropout,
                use_gated_attention=self.config.use_gated_attention,
                soft_cap=self.config.soft_cap,
            ) for i in range(self.config.num_geo_layers)
        })

        self.adaln_geo_layers = nn.ModuleDict({
            f'adaln_geo_{i}': AdaLN(in_dim=self.config.ns)
            for i in range(self.config.num_geo_layers)
        })

        self.flow_head = FlowMatchingModule(
            ns=self.config.ns,
            nv=self.config.nv,
            dropout=self.config.output_dropout,
        )

    def _initialize_weights(self):
        if self.config.use_embeddings:
            nn.init.xavier_uniform_(self.rec_fusion.weight)
            nn.init.xavier_uniform_(self.lig_fusion.weight)
