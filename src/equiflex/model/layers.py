import torch
import torch.nn as nn
import torch.nn.functional as F
import cuequivariance as cue
import cuequivariance_torch as cuet
from typing import Optional

import math
from torch_scatter import scatter


def _cuet_fp32(op, *args):
    """Run cuet op in fp32 with autocast disabled, restore original dtype on output.

    Under torch bf16-mixed autocast, activations are bf16 but cuet's internal
    weights stay fp32 — producing an unsupported dispatch combo like
    `fused_tensor_product_fwd_bf16_bf16_fp32_bf16_fp32`. We force the op to
    the fp32 dispatch path (always built), then cast the result back to the
    ambient activation dtype so downstream einsum/concat dtype-match.
    """
    ambient = next((a.dtype for a in args if torch.is_tensor(a)), torch.float32)
    with torch.amp.autocast('cuda', enabled=False):
        fp_args = tuple(a.float() if torch.is_tensor(a) else a for a in args)
        out = op(*fp_args)
    if torch.is_tensor(out) and out.dtype != ambient:
        out = out.to(ambient)
    return out


### Base Layers ###


class FCLayer(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 bias: bool = True, 
                 dropout: float = 0.0,
                 initial_affine: bool = True):
        super().__init__()
        self.layers = [nn.LayerNorm(in_dim, elementwise_affine=initial_affine),
                       nn.Linear(in_dim, hidden_dim, bias = bias),
                       nn.SiLU(),
                       nn.Dropout(dropout)]
        for i in range(num_layers - 2):
            self.layers.extend([nn.LayerNorm(hidden_dim),
                                nn.Linear(hidden_dim, hidden_dim, bias = bias),
                                nn.SiLU(),
                                nn.Dropout(dropout)])
        self.layers.extend([nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, out_dim, bias = bias)])
        self.layers = nn.Sequential(*self.layers)
        self._initialize_weights()
        
    def forward(self, x):
        return self.layers(x)
    
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


class SwiGLU(nn.Module):
    def __init__(self,
                 in_dim: int,
                 dim: int = None):
        super().__init__()
        if dim is None:
            dim = in_dim * 4
        self.w1 = nn.Linear(in_dim, dim, bias = False)
        self.w2 = nn.Linear(in_dim, dim, bias = False)
        self.w3 = nn.Linear(dim, in_dim, bias = False)
        self.silu = nn.SiLU()
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class AdaLN(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.proj = nn.Linear(in_dim, 2 * in_dim) 
        self._initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.proj(c).chunk(2, dim=-1)
        
        return self.ln(x) * (1 + scale) + shift

    def _initialize_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)


class ConditionedTransition(nn.Module):
    '''
    Conditioned transition module from AF3
    '''
    def __init__(self,
                 in_dim: int):
        super().__init__()
        self.adaln = AdaLN(in_dim)
        self.swish = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Linear(in_dim, in_dim * 2, bias=False)
        self.w2 = nn.Linear(in_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, in_dim, bias=True)
        self._initialize_weights()

        
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x = self.adaln(x, c)
        g, d = self.w1(x).chunk(2, dim=-1)
        x_ = self.swish(g) * d
        x = self.sigmoid(self.w3(c)) * self.w2(x_)
        
        return x
    
    def _initialize_weights(self):
        # w3 init bias to -2.0
        nn.init.constant_(self.w3.bias, -2.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class MultiHeadAttention(nn.Module):
    '''
    Following Gated Attention from QWEN 3
    Used pytorch SDPA for attention, with automatic backend selection
    + Added Exclusive Self Attention
    '''
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 attn_dropout: float,
                 use_qk_norm: bool = True,
                 use_gated_attention: bool = True,
                 use_xsa: bool = True):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.attn_dropout = attn_dropout
        self.use_gated_attention = use_gated_attention
        self.use_xsa = use_xsa
        
        # Headwise gating: one scalar gate per head (not per dimension)
        self.q_proj_dim = hidden_dim + num_heads if use_gated_attention else hidden_dim

        self.w_q = nn.Linear(in_dim, self.q_proj_dim, bias=False)
        self.w_k = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, in_dim, bias=False)
        
        if use_qk_norm:
            self.use_qk_norm = True
            self.norm_q = nn.LayerNorm(hidden_dim)
            self.norm_k = nn.LayerNorm(hidden_dim)
        else:
            self.use_qk_norm = False
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.swiglu_ffn = SwiGLU(hidden_dim, hidden_dim)
        
        self._initialize_weights()

        
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                return_attn: bool = False):
        B, N_q, _ = q.shape
        B, N_k, _ = k.shape
        residual = q

        q = self.norm1(q)

        q_out = self.w_q(q)

        # Headwise gating: one scalar gate per head
        gate_score = None
        if self.use_gated_attention:
            # Split to query (hidden_dim) and gate (num_heads)
            q, gate_score = torch.split(q_out, [self.hidden_dim, self.num_heads], dim=-1)
            # gate_score: [B, N_q, num_heads] -> [B, N_q, num_heads, 1] for broadcasting
            gate_score = gate_score.unsqueeze(-1)
        else:
            q = q_out

        k = self.w_k(k)
        v = self.w_v(v)
        
        if self.use_qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        q = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        attn_mask = None
        if bias is not None:
            # bias: [B, N_q, N_k, 1] or [B, N_q, N_k, H] -> [B, H, N_q, N_k]
            if bias.dim() == 4:
                if bias.shape[-1] == 1:
                    # [B, N_q, N_k, 1] -> [B, 1, N_q, N_k] -> broadcast to [B, H, N_q, N_k]
                    attn_mask = bias.squeeze(-1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                else:
                    # [B, N_q, N_k, H] -> [B, H, N_q, N_k]
                    attn_mask = bias.permute(0, 3, 1, 2)
            else:
                attn_mask = bias

        if mask is not None:
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(~mask, -1e9)
            else:
                attn_mask = mask

        # Save v reference for XSA (before potential consumption by SDPA)
        v_for_xsa = v if self.use_xsa else None

        # Compute attention weights manually if we need to return stats
        attn_stats = None
        if return_attn:
            # Manual attention computation to get weights
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            # Shannon entropy: H = -sum(p * log(p)), averaged over all distributions
            # Add small epsilon to avoid log(0)
            entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean()
            attn_stats = {
                'mean': attn_weights.mean().detach(),
                'max': attn_weights.max().detach(),
                'entropy': entropy.detach(),
            }
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout if self.training else 0.0, training=self.training)
            output = torch.matmul(attn_weights, v)
            del attn_weights
        else:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask, self.attn_dropout if self.training else 0.0)

        # XSA: project out self-value component from attention output
        if self.use_xsa:
            # output: [B, H, N, head_dim], v_for_xsa: [B, H, N, head_dim]
            vn = F.normalize(v_for_xsa, dim=-1)
            output = output - (output * vn).sum(dim=-1, keepdim=True) * vn
            del vn, v_for_xsa

        # Transpose to [B, N_q, num_heads, head_dim]
        output = output.transpose(1, 2)

        if self.use_gated_attention and gate_score is not None:
            output = output * torch.sigmoid(gate_score)

        output = output.reshape(B, N_q, self.hidden_dim)
        output = self.w_o(output) + residual
        residual = output

        output = self.norm2(output)
        output = self.swiglu_ffn(output)
        output = output + residual

        if return_attn:
            return output, attn_stats
        return output
        
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


class SelfOuterProductModule(nn.Module):
    '''
    An outer product module that computes the outer product of the input features
    Used to generate rich pair features from single features

    '''

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int):
        super().__init__()

        self.w_left = nn.Linear(in_dim, hidden_dim, bias = False)
        self.w_right = nn.Linear(in_dim, hidden_dim, bias = False)
        self.w_out = nn.Linear(hidden_dim, in_dim, bias = False)

        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        left = self.w_left(x)
        right = self.w_right(x)
        x_pair = left.unsqueeze(2) * right.unsqueeze(1) # [B, N, 1, C] * [B, 1, N, C] -> [B, N, N, C]
        x_pair = self.w_out(x_pair)
        return x_pair

    def _initialize_weights(self):
        for m in [self.w_left, self.w_right, self.w_out]:
            nn.init.xavier_uniform_(m.weight)


class TimeEmbedder(nn.Module):
    '''
    Sinusoidal time embedder for conditioning on timestep t
    '''
    def __init__(self,
                 hidden_dim: int,
                 output_dim: int,
                 max_period: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_period = max_period
        
        self.time_proj = FCLayer(in_dim = hidden_dim, hidden_dim = hidden_dim, out_dim = output_dim, num_layers = 2, bias = False, initial_affine = False)
        
    def forward(self, timesteps: torch.Tensor):

        # Handle 1-dim input
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)

        timesteps = timesteps * self.max_period

        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(self.max_period)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = timesteps.device) * -emb)
        emb = timesteps * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = -1)
        emb = self.time_proj(emb)
        return emb


class PairwiseFeatureEmbedderWithConstraints(nn.Module):
    '''
    V5 variant: uses nn.Linear for mask_embedder instead of FCLayer.
    FCLayer wraps LayerNorm(1) which collapses scalar input to 0,
    killing gradient through the mask embedding path.
    '''
    def __init__(self,
                 hidden_dim: int,
                 out_dim: int):
        super().__init__()
        self.disto_embedder = FCLayer(in_dim=41, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.adj_embedder = FCLayer(in_dim=26, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.constraint_embedder = FCLayer(in_dim=41, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2, bias=False, initial_affine=False)
        self.mask_embedder = nn.Linear(1, out_dim, bias=True)
        self.final_proj = nn.Linear(out_dim, out_dim, bias=False)
        self._initialize_weights()

    def forward(self, distogram: torch.Tensor, adj_mat: torch.Tensor, constraint_mat: torch.Tensor, block_mask: Optional[torch.Tensor] = None):
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
        nn.init.xavier_uniform_(self.mask_embedder.weight)
        nn.init.zeros_(self.mask_embedder.bias)

    def _rbf_bin_distogram(self, distogram: torch.Tensor):
        centers = torch.arange(0, 20.5, 0.5).to(distogram.device)
        width = 0.5
        rbf_features = torch.exp(-width * (distogram - centers) ** 2)
        return rbf_features


class InitialEmbedder(nn.Module):
    '''
    Simple fully-connected embedding scheme for the initial input features
    '''
    def __init__(self,
                 atom_dim: int,
                 use_embeddings: bool,
                 esm_dim: int,
                 unimol_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.use_embeddings = use_embeddings
        self.atom_embedder = FCLayer(in_dim=atom_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=2, bias=False, dropout=dropout, initial_affine=False)
        if use_embeddings:
            self.esm_embedder = FCLayer(in_dim=esm_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=2, bias=False, dropout=dropout, initial_affine=False)
            self.unimol_embedder = FCLayer(in_dim=unimol_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=2, bias=False, dropout=dropout, initial_affine=False)
        else:
            self.esm_embedder = None
            self.unimol_embedder = None
    
    def forward(self, atoms: torch.Tensor, esm: torch.Tensor, unimol: torch.Tensor):
        atom_emb = self.atom_embedder(atoms)

        if self.use_embeddings:
            assert esm is not None and unimol is not None, "ESM and unimol embeddings must be provided if use_embeddings is True"
            esm_emb = self.esm_embedder(esm)
            unimol_emb = self.unimol_embedder(unimol)
            return atom_emb, esm_emb, unimol_emb
        else:
            del esm, unimol
            return atom_emb, None, None


class AtomTransformer(nn.Module):
    '''
    Atom-level transformer that processes single features q and pairwise features p.
    Both q and p are updated through the layer for stacking.
    '''
    def __init__(self,
                 single_hidden_dim: int,
                 pair_hidden_dim: int,
                 num_heads: int,
                 attn_dropout: float,
                 use_gated_attention: bool,
                 use_outer_product: bool,
                 use_qk_norm: bool = True,
                 ):
        super().__init__()
        self.use_outer_product = use_outer_product
        self.pair_hidden_dim = pair_hidden_dim

        self.q_norm = nn.LayerNorm(single_hidden_dim)
        self.w_cond = nn.Linear(single_hidden_dim, single_hidden_dim, bias = False)  # For GLU
        self.sigmoid = nn.Sigmoid()

        # Pair projection removed - p is now passed in from outside
        self.w_pair_out = nn.Linear(pair_hidden_dim, pair_hidden_dim, bias = False)  # Projects mixed pair features

        if use_outer_product:
            self.outer_product = SelfOuterProductModule(in_dim = single_hidden_dim, hidden_dim = single_hidden_dim // 4)

        # Project outer product output to pair_hidden_dim
        self.outer_to_pair = nn.Linear(single_hidden_dim, pair_hidden_dim, bias = False)
        self.q_pair_norm = nn.LayerNorm(pair_hidden_dim)  # Normalize q_pair before adding to p

        # For attention bias: project pair features to scalar
        self.p_norm = nn.LayerNorm(pair_hidden_dim)
        self.pair_to_bias = nn.Linear(pair_hidden_dim, 1, bias = False)

        self.mixing_linear_1 = nn.Linear(pair_hidden_dim, pair_hidden_dim, bias = False)
        self.mixing_linear_2 = nn.Linear(pair_hidden_dim, pair_hidden_dim, bias = False)

        self.attn_core = MultiHeadAttention(in_dim = single_hidden_dim,
                                            hidden_dim = single_hidden_dim,
                                            num_heads = num_heads,
                                            attn_dropout = attn_dropout,
                                            use_qk_norm = use_qk_norm,
                                            use_gated_attention = use_gated_attention)
        self._initialize_weights()

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        return_attn: bool = False,
        atom_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ):
        '''
        Args:
            q: [B, N, single_hidden_dim] - single features
            p: [B, N, N, pair_hidden_dim] - pairwise features (created externally)
            return_attn: if True, return attention statistics
            atom_mask: optional [B, N] valid atom mask for padded batches
            pair_mask: optional [B, N, N] valid pair mask for padded batches
        Returns:
            q: [B, N, single_hidden_dim] - updated single features
            p: [B, N, N, pair_hidden_dim] - updated pairwise features
            attn_stats: (optional) dict with 'mean' and 'max' attention values
        '''
        mask_f = atom_mask.to(q.dtype).unsqueeze(-1) if atom_mask is not None else None
        pair_mask_f = pair_mask.to(p.dtype).unsqueeze(-1) if pair_mask is not None else None
        if mask_f is not None:
            q = q * mask_f
        if pair_mask_f is not None:
            p = p * pair_mask_f

        q_residual = q
        p_residual = p

        q = self.q_norm(q)  # Pre-norm first
        p = self.p_norm(p)
        if mask_f is not None:
            q = q * mask_f
        if pair_mask_f is not None:
            p = p * pair_mask_f
        g = self.sigmoid(self.w_cond(q))  # [B, N, F]

        # Single and pair repr. mixing: Make [B, N, N, F] out of [B, N, F]
        p = self.mix_single_to_pair(q, p, self.use_outer_product)
        if pair_mask_f is not None:
            p = p * pair_mask_f
        p = self.mixing_linear_1(p)
        if pair_mask_f is not None:
            p = p * pair_mask_f

        # Project p to attention bias
        attn_bias = self.pair_to_bias(p)  # [B, N, N, 1]
        # Attention connectivity follows pair_mask when given (so a block-diagonal
        # pair_mask yields block-diagonal attention); else fall back to padding.
        attn_mask = None
        if pair_mask is not None:
            attn_mask = pair_mask.bool().unsqueeze(1)                      # [B, 1, N, N]
        elif atom_mask is not None:
            valid = atom_mask.bool()
            attn_mask = valid.unsqueeze(1).unsqueeze(2) & valid.unsqueeze(1).unsqueeze(3)

        attn_stats = None
        if return_attn:
            q, attn_stats = self.attn_core(q, q, q, mask=attn_mask, bias=attn_bias, return_attn=True)
        else:
            q = self.attn_core(q, q, q, mask=attn_mask, bias=attn_bias)
        del attn_bias  # Free after attention
        q = q * g  # GLU
        del g  # Free gate
        q = q + q_residual
        if mask_f is not None:
            q = q * mask_f
        del q_residual

        # Mix again
        p = self.mix_single_to_pair(q, p, self.use_outer_product)
        if pair_mask_f is not None:
            p = p * pair_mask_f
        p = self.mixing_linear_2(p)
        p = self.w_pair_out(p)
        p = p + p_residual
        if pair_mask_f is not None:
            p = p * pair_mask_f
        del p_residual

        if return_attn:
            return q, p, attn_stats
        return q, p

    def mix_single_to_pair(self, q: torch.Tensor, p: torch.Tensor, use_outer_product: bool):
        if use_outer_product:
            q_pair = self.outer_product(q)  # [B, N, N, single_hidden_dim]
            q_pair = self.outer_to_pair(q_pair)  # [B, N, N, pair_hidden_dim]
        else:
            q_pair = q.unsqueeze(1) * q.unsqueeze(2)  # [B, N, N, single_hidden_dim]
            q_pair = self.outer_to_pair(q_pair)  # [B, N, N, pair_hidden_dim]

        # Normalize q_pair to match scale of normalized p
        q_pair = self.q_pair_norm(q_pair)
        p = p + q_pair

        del q_pair  # Memory extensive tensor
        return p
        
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


class TriangleUpdatingModule(nn.Module):
    '''
    Employ CuEquivariance Trimul module
    '''
    
    def __init__(self,
                 ns: int
                 ):
        super().__init__()
        
        # For outgoing updates
        self.norm_in_out = nn.LayerNorm(ns)
        self.p_in_out = nn.Linear(ns, 2*ns, bias=False) # Following CuEquivariance specs
        self.g_in_out = nn.Linear(ns, 2*ns, bias=False)
        self.norm_out_out = nn.LayerNorm(ns)
        self.p_out_out = nn.Linear(ns, ns, bias=False)
        self.g_out_out = nn.Linear(ns, ns, bias=False)
        
        # For incoming updates
        self.norm_in_inc = nn.LayerNorm(ns)
        self.p_in_inc = nn.Linear(ns, 2*ns, bias=False)
        self.g_in_inc = nn.Linear(ns, 2*ns, bias=False)
        self.norm_out_inc = nn.LayerNorm(ns)
        self.p_out_inc = nn.Linear(ns, ns, bias=False)
        self.g_out_inc = nn.Linear(ns, ns, bias=False)
        
        self._initialize_weights()
        
    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # mask: [B, N, N] (1 = valid pair, 0 = masked). Gates the input
        # projections so masked k (padding and cross-entity) do not enter the
        # triangle sum — keeps propagation intra-molecular and padding-free.
        # Outgoing updates + residual
        z = z + self._triangle_update(
            z, direction='outgoing', mask=mask,
            norm_in=self.norm_in_out, p_in=self.p_in_out, g_in=self.g_in_out,
            norm_out=self.norm_out_out, p_out=self.p_out_out, g_out=self.g_out_out
        )

        # Incoming updates + residual
        z = z + self._triangle_update(
            z, direction='incoming', mask=mask,
            norm_in=self.norm_in_inc, p_in=self.p_in_inc, g_in=self.g_in_inc,
            norm_out=self.norm_out_inc, p_out=self.p_out_inc, g_out=self.g_out_inc
        )

        return z

    def _triangle_update(self, x, direction, norm_in, p_in, g_in, norm_out, p_out, g_out, mask=None):
        return cuet.triangle_multiplicative_update(
            x,
            direction=direction,
            mask=mask,
            norm_in_weight=norm_in.weight,
            norm_in_bias=norm_in.bias,
            p_in_weight=p_in.weight,
            p_in_bias=p_in.bias,
            g_in_weight=g_in.weight,
            g_in_bias=g_in.bias,
            norm_out_weight=norm_out.weight,
            norm_out_bias=norm_out.bias,
            p_out_weight=p_out.weight,
            p_out_bias=p_out.bias,
            g_out_weight=g_out.weight,
            g_out_bias=g_out.bias,
        )
        
    def _initialize_weights(self):
        for m in [self.g_in_out, self.g_in_inc, self.g_out_out, self.g_out_inc]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in [self.p_in_out, self.p_in_inc, self.p_out_out, self.p_out_inc]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in [self.norm_in_out, self.norm_in_inc, self.norm_out_out, self.norm_out_inc]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)


class EVAModule(nn.Module):
    '''
    EVA variant emitting combined (l=1 + l=2) irrep features per node.

    Aggregates raw r_ij (l=1, magnitude-bearing — matches base EVA) AND
    Y_2(r̂_ij) (l=2, unit-norm) via the same scalar attention weights.
    Output is a flat tensor [B, N, nv*(3+5)] in cuet mul_ir layout for
    irreps `nv x 1o + nv x 2e` — first nv*3 dims are the l=1 block,
    last nv*5 dims are the l=2 block.
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

        # Fused SH kernel: l=2 only (l=1 comes from raw r_ij to preserve magnitude)
        self.sh_l2 = cuet.SphericalHarmonics(ls=[2], normalize=True)

        self._initialize_weights()

    def forward(self,
                x: torch.Tensor,
                r: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                atom_mask: Optional[torch.Tensor] = None,
                return_attn: bool = False,
                ):
        B, N, _ = x.shape
        r_ij = r.unsqueeze(2) - r.unsqueeze(1)  # [B, N, N, 3]

        x = self.norm_1(x)
        q = self.w_q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.w_k(x).reshape(B, N, self.num_heads, self.head_dim)

        attn = torch.einsum('bihd, bjhd -> bijh', q, k) / math.sqrt(float(self.head_dim))
        del q, k
        if bias is not None:
            attn = attn + bias

        if self.attenuate_by_dist:
            dist = torch.norm(r_ij, dim=-1, keepdim=True)
            dist_mask = dist < 1e-4
            r_l1 = r_ij / (dist + 1e-6)
            attn = attn - dist
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
        else:
            r_l1 = r_ij
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

        attn_stats = None
        if return_attn:
            entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=2).mean()
            attn_stats = {
                'mean': attn.mean().detach(),
                'max': attn.max().detach(),
                'entropy': entropy.detach(),
            }

        # l=1 aggregation matches base EVA: distance attenuation uses unit directions.
        v_l1 = torch.einsum('bijh, bijc -> bihc', attn, r_l1)  # [B, N, H, 3]

        # l=2 SH aggregation (unit-norm Y_2 of r_hat_ij)
        y_l2 = _cuet_fp32(self.sh_l2, r_ij.reshape(-1, 3)).reshape(B, N, N, 5)
        v_l2 = torch.einsum('bijh, bijc -> bihc', attn, y_l2)  # [B, N, H, 5]
        del attn, r_ij, r_l1, y_l2

        if atom_mask is not None:
            mask4 = atom_mask.to(v_l1.dtype).unsqueeze(-1).unsqueeze(-1)
            v_l1 = v_l1 * mask4
            v_l2 = v_l2 * mask4

        # Pack as flat irrep tensor: [nv x 1o block] || [nv x 2e block]
        out = torch.cat([
            v_l1.reshape(B, N, self.num_heads * 3),
            v_l2.reshape(B, N, self.num_heads * 5),
        ], dim=-1)
        if return_attn:
            return out, attn_stats
        return out

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
    '''
    Geometric attention variant carrying combined (l=1 + l=2) vector irreps.

    Vector channels evolve as `nv x 1o + nv x 2e` through cuet.Linear Q/K/V
    projections and a FullyConnectedTensorProduct head that mixes irreps via
    all valid CG paths producing 1o or 2e outputs (notably 2e ⊗ 1o → 1o).
    '''

    def __init__(self,
                 ns: int,
                 nv: int,
                 num_heads: int,
                 bias_dim: int,
                 attn_dropout: float,
                 use_gated_attention: bool = True,
                 soft_cap: float = 50.0,
                 use_xsa: bool = True,
                 dist_bins: int = 0,
                 ):
        super().__init__()
        assert ns % num_heads == 0, f'ns ({ns}) % num_heads ({num_heads}) != 0'
        assert nv % num_heads == 0, f'nv ({nv}) % num_heads ({num_heads}) != 0'

        self.ns = ns
        self.nv = nv
        self.num_heads = num_heads
        self.head_dim_s = ns // num_heads
        self.head_dim_v = nv // num_heads
        self.attn_dropout = attn_dropout
        self.use_gated_attention = use_gated_attention
        self.soft_cap = soft_cap
        self.use_xsa = use_xsa

        self.l1_dim = nv * 3
        self.l2_dim = nv * 5
        self.v_dim = self.l1_dim + self.l2_dim  # = nv * 8

        # Scalar Q, K, V
        if use_gated_attention:
            self.scalar_q = nn.Linear(ns, ns + num_heads, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            self.scalar_q = nn.Linear(ns, ns, bias=False)
        self.scalar_k = nn.Linear(ns, ns, bias=False)
        self.scalar_v = nn.Linear(ns, ns, bias=False)
        self.scalar_out = nn.Linear(ns, ns, bias=False)

        # Vector projections over combined irrep
        ir_node = cue.Irreps('O3', f'{nv}x1o + {nv}x2e')
        layout = cue.mul_ir
        self.vector_q = cuet.Linear(ir_node, ir_node, layout=layout)
        self.vector_k = cuet.Linear(ir_node, ir_node, layout=layout)
        self.vector_v = cuet.Linear(ir_node, ir_node, layout=layout)
        self.vector_out = cuet.Linear(ir_node, ir_node, layout=layout)

        # Per-layer TP head: mix irreps via CG paths (key expressivity gain)
        self.vector_tp_a = cuet.Linear(ir_node, ir_node, layout=layout)
        self.vector_tp_b = cuet.Linear(ir_node, ir_node, layout=layout)
        self.vector_tp = cuet.FullyConnectedTensorProduct(
            ir_node, ir_node, ir_node, layout=layout,
        )

        # Bias / gates
        self.bias_norm = nn.LayerNorm(bias_dim)
        self.bias_proj = nn.Linear(bias_dim, num_heads, bias=False)
        # Live per-head distance bias (#1): maps RBF(||x_i - x_j||) -> per-head
        # additive logit. Zero-init so it starts as a no-op. dist_bins=0 disables.
        self.dist_bins = dist_bins
        self.dist_bias = nn.Linear(dist_bins, num_heads, bias=False) if dist_bins > 0 else None
        self.vector_gate_l1 = nn.Sequential(nn.Linear(ns, nv, bias=False), nn.Sigmoid())
        self.vector_gate_l2 = nn.Sequential(nn.Linear(ns, nv, bias=False), nn.Sigmoid())

        # Norms / FFN
        self.scalar_norm = nn.LayerNorm(ns, elementwise_affine=True)
        self.scalar_ffn = SwiGLU(ns)
        self.scalar_ffn_norm = nn.LayerNorm(ns, elementwise_affine=True)

        self._initialize_weights()
        if self.dist_bias is not None:
            nn.init.zeros_(self.dist_bias.weight)   # start as no-op

    def forward(self,
                scalar_feats: torch.Tensor,  # [B, N, ns]
                vector_feats: torch.Tensor,  # [B, N, nv*8] — combined irrep
                bias: Optional[torch.Tensor] = None,
                atom_mask: Optional[torch.Tensor] = None,
                return_attn: bool = False,
                dist_rbf: Optional[torch.Tensor] = None,  # [B, N, N, dist_bins] live distance features
                ):
        B, N, _ = scalar_feats.shape
        mask_f = atom_mask.to(scalar_feats.dtype).unsqueeze(-1) if atom_mask is not None else None
        if mask_f is not None:
            scalar_feats = scalar_feats * mask_f
            vector_feats = vector_feats * mask_f

        scalar_residual = scalar_feats
        vector_residual = vector_feats

        scalar_feats = self.scalar_norm(scalar_feats)
        if mask_f is not None:
            scalar_feats = scalar_feats * mask_f

        # Scalar Q/K/V
        if self.use_gated_attention:
            Q_s_out = self.scalar_q(scalar_feats)
            Q_s, Q_s_gate = torch.split(Q_s_out, [self.ns, self.num_heads], dim=-1)
        else:
            Q_s = self.scalar_q(scalar_feats)
        K_s = self.scalar_k(scalar_feats)
        V_s = self.scalar_v(scalar_feats)

        Q_s = Q_s.reshape(B, N, self.num_heads, self.head_dim_s).permute(0, 2, 1, 3)
        K_s = K_s.reshape(B, N, self.num_heads, self.head_dim_s).permute(0, 2, 1, 3)
        V_s = V_s.reshape(B, N, self.num_heads, self.head_dim_s).permute(0, 2, 1, 3)

        attn_score_s = torch.matmul(Q_s, K_s.transpose(-2, -1)) / (self.head_dim_s ** 0.5)
        del Q_s, K_s

        # Vector Q/K/V over combined irrep (cuet.Linear preserves irrep structure)
        BN = B * N
        vec_flat = vector_feats.reshape(BN, self.v_dim)
        Q_v = _cuet_fp32(self.vector_q, vec_flat)
        K_v = _cuet_fp32(self.vector_k, vec_flat)
        V_v = _cuet_fp32(self.vector_v, vec_flat)
        del vec_flat

        # Split l=1 and l=2 blocks, then per-head
        Q_v_l1 = Q_v[:, :self.l1_dim].reshape(B, N, self.num_heads, self.head_dim_v, 3).permute(0, 2, 1, 3, 4)
        Q_v_l2 = Q_v[:, self.l1_dim:].reshape(B, N, self.num_heads, self.head_dim_v, 5).permute(0, 2, 1, 3, 4)
        K_v_l1 = K_v[:, :self.l1_dim].reshape(B, N, self.num_heads, self.head_dim_v, 3).permute(0, 2, 1, 3, 4)
        K_v_l2 = K_v[:, self.l1_dim:].reshape(B, N, self.num_heads, self.head_dim_v, 5).permute(0, 2, 1, 3, 4)
        del Q_v, K_v

        # Attention contributions: l=1·l=1 -> 0e AND l=2·l=2 -> 0e.
        score_l1 = torch.einsum('bhidx, bhjdx -> bhij', Q_v_l1, K_v_l1)
        score_l2 = torch.einsum('bhidx, bhjdx -> bhij', Q_v_l2, K_v_l2)
        del Q_v_l1, Q_v_l2, K_v_l1, K_v_l2
        attn_score_v = (score_l1 + score_l2) / math.sqrt(float((3 + 5) * self.head_dim_v))
        del score_l1, score_l2

        attn_score = attn_score_s + attn_score_v
        del attn_score_s, attn_score_v

        if bias is not None:
            if bias.shape[-1] != self.num_heads:
                bias = self.bias_norm(bias)
                bias = self.bias_proj(bias)
            bias = bias.permute(0, 3, 1, 2)
            attn_score = attn_score + bias

        # #1: live per-head distance bias — proximity-routing on the current pose.
        if self.dist_bias is not None and dist_rbf is not None:
            attn_score = attn_score + self.dist_bias(dist_rbf).permute(0, 3, 1, 2)

        attn_score = self.soft_cap * torch.tanh(attn_score / self.soft_cap)
        if atom_mask is not None:
            key_mask = atom_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_score = attn_score.masked_fill(~key_mask, -1e9)

        attn_weights = F.softmax(attn_score, dim=-1)
        del attn_score

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
        if self.use_xsa:
            vn = F.normalize(V_s, dim=-1)
            scalar_out = scalar_out - (scalar_out * vn).sum(dim=-1, keepdim=True) * vn
            del vn
        del V_s

        if self.use_gated_attention:
            gate = Q_s_gate.unsqueeze(-1).permute(0, 2, 1, 3)
            scalar_out = scalar_out * self.sigmoid(gate)
            del Q_s_gate, gate

        scalar_out = scalar_out.permute(0, 2, 1, 3).reshape(B, N, self.ns)
        scalar_out = self.scalar_out(scalar_out)

        # Vector aggregation per-head, per-irrep block
        V_v_l1 = V_v[:, :self.l1_dim].reshape(B, N, self.num_heads, self.head_dim_v * 3).permute(0, 2, 1, 3)
        V_v_l2 = V_v[:, self.l1_dim:].reshape(B, N, self.num_heads, self.head_dim_v * 5).permute(0, 2, 1, 3)
        del V_v
        vec_l1 = torch.matmul(attn_weights, V_v_l1)
        vec_l2 = torch.matmul(attn_weights, V_v_l2)
        del V_v_l1, V_v_l2, attn_weights
        vec_l1 = vec_l1.permute(0, 2, 1, 3).reshape(B, N, self.l1_dim)
        vec_l2 = vec_l2.permute(0, 2, 1, 3).reshape(B, N, self.l2_dim)
        vector_out = torch.cat([vec_l1, vec_l2], dim=-1)
        del vec_l1, vec_l2

        # Output projection
        vector_out = _cuet_fp32(self.vector_out, vector_out.reshape(BN, self.v_dim))

        # Scalar-to-vector gating (separate per irrep block)
        g_l1 = self.vector_gate_l1(scalar_out)  # [B, N, nv]
        g_l2 = self.vector_gate_l2(scalar_out)
        v_l1_blk = vector_out[:, :self.l1_dim].reshape(B, N, self.nv, 3) * g_l1.unsqueeze(-1)
        v_l2_blk = vector_out[:, self.l1_dim:].reshape(B, N, self.nv, 5) * g_l2.unsqueeze(-1)
        vector_out = torch.cat([
            v_l1_blk.reshape(B, N, self.l1_dim),
            v_l2_blk.reshape(B, N, self.l2_dim),
        ], dim=-1)
        del v_l1_blk, v_l2_blk, g_l1, g_l2

        # CG-path mixing via cuet TP — covers 2e⊗1o→1o, 1o⊗1o→2e, 2e⊗2e→2e, etc.
        vec_a = _cuet_fp32(self.vector_tp_a, vector_out.reshape(BN, self.v_dim))
        vec_b = _cuet_fp32(self.vector_tp_b, vector_out.reshape(BN, self.v_dim))
        vec_tp = _cuet_fp32(self.vector_tp, vec_a, vec_b).reshape(B, N, self.v_dim)
        vector_out = vector_out + vec_tp
        del vec_a, vec_b, vec_tp

        # Residuals
        scalar_out = scalar_out + scalar_residual
        vector_out = vector_out + vector_residual
        del scalar_residual, vector_residual

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
        # Smaller init on output paths for stability (matches V4)
        for m in [self.scalar_out, self.vector_out]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=0.01)


class FlowMatchingModule(nn.Module):
    '''
    Flow matching head consuming combined (l=1 + l=2) vector features.
    Slices the l=1 block and projects to a single l=1 output (velocity),
    scaled by a scalar magnitude. The l=2 portion is unused at the head
    but contributes through layers via TP mixing in upstream GeoAttn.
    '''

    def __init__(self,
                 ns: int,
                 nv: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ns = ns
        self.nv = nv
        self.l1_dim = nv * 3

        ir_v_in = cue.Irreps('O3', f'{nv}x1o')
        ir_v_half = cue.Irreps('O3', f'{nv // 2}x1o')
        ir_v_out = cue.Irreps('O3', '1x1o')
        layout = cue.mul_ir
        self.vector_proj_1 = cuet.Linear(ir_v_in, ir_v_half, layout=layout)
        self.vector_proj_2 = cuet.Linear(ir_v_half, ir_v_out, layout=layout)

        self.scalar_projector = nn.Sequential(
            nn.LayerNorm(ns),
            nn.Linear(ns, ns // 4, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ns // 4, 1, bias=False),
        )

        self._initialize_weights()

    def forward(self,
                scalar_feats: torch.Tensor,
                vector_feats: torch.Tensor,
                ) -> torch.Tensor:
        input_3d = scalar_feats.dim() == 3
        if input_3d:
            B, N, _ = scalar_feats.shape
            scalar_feats = scalar_feats.reshape(B * N, -1)
            vector_feats = vector_feats.reshape(B * N, -1)

        v_l1 = vector_feats[..., :self.l1_dim].contiguous()
        v_l1 = _cuet_fp32(self.vector_proj_1, v_l1)
        v_l1 = _cuet_fp32(self.vector_proj_2, v_l1)
        s = self.scalar_projector(scalar_feats)
        velocity = s * v_l1

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


class PocketInteraction(nn.Module):
    """Compute-once, bidirectional live rec<->lig pocket conditioning.

    From live coords x_t (per timestep, before the geo stack), each atom gets a
    distance-gated summary of its *cross-entity* partners:
      - invariant scalars (partner feature context, soft-min distance, soft shell
        count) -> added to scalar features q;
      - an equivariant unit-direction to the soft-nearest partner -> added to the
        l=1 block of the vector features v.
    Both directions (rec<-lig and lig<-rec) give mutual cross-talk. Output
    projections are zero-initialised so the module starts as a no-op.

    Equivariance: scalar terms are functions of distances / invariant features;
    the direction term is an l=1 vector scaled by invariant weights -> l=1.
    """

    def __init__(self, ns: int, nv: int, tau: float = 2.0, cutoff: float = 8.0):
        super().__init__()
        self.ns = ns
        self.nv = nv
        self.l1_dim = nv * 3
        self.tau = tau
        self.cutoff = cutoff
        self.rec_scal = nn.Linear(ns + 2, ns, bias=False)
        self.lig_scal = nn.Linear(ns + 2, ns, bias=False)
        self.rec_dir = nn.Linear(ns + 2, nv, bias=False)
        self.lig_dir = nn.Linear(ns + 2, nv, bias=False)
        for m in [self.rec_scal, self.lig_scal, self.rec_dir, self.lig_dir]:
            nn.init.zeros_(m.weight)

    def _interact(self, x_a, q_a, x_b, q_b, mask_b):
        """a attends to cross-entity partners b. Returns (scal [B,Na,ns+2],
        dir [B,Na,3] unit-direction to soft-nearest b)."""
        d = torch.cdist(x_a, x_b)                                   # [B,Na,Nb] (Å)
        valid = (d <= self.cutoff)
        if mask_b is not None:
            valid = valid & mask_b.bool().unsqueeze(1)
        neg = (d / self.tau).masked_fill(~valid, 1e4)              # finite -> no NaN
        w = torch.softmax(-neg, dim=-1)
        has = valid.any(dim=-1, keepdim=True)
        w = w * has                                                 # zero empty rows
        ctx = torch.einsum('ban, bnf -> baf', w, q_b)
        d_masked = d.masked_fill(~valid, 1e4)
        softmin = -self.tau * torch.logsumexp(-d_masked / self.tau, dim=-1)
        softmin = torch.where(has.squeeze(-1), softmin, d.new_full(softmin.shape, self.cutoff))
        count = (torch.sigmoid(self.cutoff - d) * valid).sum(dim=-1)
        scal = torch.cat([ctx, softmin.unsqueeze(-1), count.unsqueeze(-1)], dim=-1)
        diff = x_b.unsqueeze(1) - x_a.unsqueeze(2)                  # [B,Na,Nb,3]
        unit = diff / diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = torch.einsum('ban, banc -> bac', w, unit)      # [B,Na,3]
        return scal, direction

    def forward(self, q, v, x, n_rec, atom_mask=None):
        """q [B,N,ns], v [B,N,nv*8] (flat l1||l2), x [B,N,3] in Å. Returns q, v."""
        x_rec, x_lig = x[:, :n_rec], x[:, n_rec:]
        q_rec, q_lig = q[:, :n_rec], q[:, n_rec:]
        m_rec = m_lig = None
        if atom_mask is not None:
            m_rec, m_lig = atom_mask[:, :n_rec], atom_mask[:, n_rec:]

        scal_r, dir_r = self._interact(x_rec, q_rec, x_lig, q_lig, m_lig)  # rec <- lig
        scal_l, dir_l = self._interact(x_lig, q_lig, x_rec, q_rec, m_rec)  # lig <- rec

        q = torch.cat([q_rec + self.rec_scal(scal_r),
                       q_lig + self.lig_scal(scal_l)], dim=1)

        B = v.shape[0]
        vdir_r = (self.rec_dir(scal_r).unsqueeze(-1) * dir_r.unsqueeze(-2)).reshape(B, -1, self.l1_dim)
        vdir_l = (self.lig_dir(scal_l).unsqueeze(-1) * dir_l.unsqueeze(-2)).reshape(B, -1, self.l1_dim)
        vdir = torch.cat([vdir_r, vdir_l], dim=1)
        v = v.clone()
        v[:, :, :self.l1_dim] = v[:, :, :self.l1_dim] + vdir
        return q, v
