import torch
import torch.nn as nn
import torch.nn.functional as F
import cuequivariance as cue
import cuequivariance_torch as cuet
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from torch_scatter import scatter

from .layers import (
    AdaLN,
    TimeEmbedder,
    InitialEmbedder,
    PairwiseFeatureEmbedderWithConstraints,
    AtomTransformer,
    TriangleUpdatingModule,
    GeometricAttentionModule,
    FlowMatchingModule,
    PocketInteraction,
    PairTransition,
    CrossPairInit,
    _cuet_fp32,
)


@dataclass
class FlexConfig:

    # Input dimensions
    atom_feat_dim: int = 299  # 175 context (protein-name/element/neighbors/gasteiger/morgan/pharma/stereo) + 124 chemistry
    bond_feat_dim: int = 26
    esm_dim: int = 1536
    unimol_dim: int = 1536

    # Hidden dimensions
    ns: int = 256
    nv: int = 8
    pair_hidden_dim: int = 256

    # Layer counts
    num_atom_layers: int = 6
    num_geo_layers: int = 6

    # Attention settings
    num_atom_heads: int = 8
    num_geo_heads: int = 4
    attn_dropout: float = 0.1
    use_gated_attention: bool = True
    use_qk_norm: bool = True  # QK normalization for attention stability

    # Other settings
    initial_dropout: float = 0.0  # Dropout in initial embedding layers
    output_dropout: float = 0.1
    use_embeddings: bool = True  # Whether to use ESM/UniMol embeddings

    # Time embedding
    time_embed_dim: int = 256
    time_max_period: int = 10000

    # Gradient checkpointing
    use_gradient_checkpointing: bool = False
    gradient_checkpoint_every: int = 2  # Checkpoint every N layers (1 = all, 2 = every other)

    # Coordinate scaling
    coord_scale: float = 16.0  # Scale factor for coordinates (divide input, multiply output)
    geo_dist_bins: int = 32    # #1: RBF bins for the live distance bias (0 disables)
    geo_dist_max: float = 20.0  # max distance (Å) for the distance-bias RBF
    # Prior x0 noise scale (Å). Sidechain noise is N(CB_center, sigma^2); ligand
    # noise is N(backbone_COM, sigma^2). Ligand sigma ~ ligand radius of gyration
    # (~5 Å) so the prior cloud matches the molecule's spatial extent instead of
    # a tight 1-Å blob it must explode outward from.
    sidechain_prior_sigma: float = 2.0
    ligand_prior_sigma: float = 5.0
    # Attention soft capping
    soft_cap: float = 50.0  # Soft cap for attention scores (prevents saturation)

    # Attention stats logging
    return_attn_stats: bool = False

    # Chemistry trunk widths. The v10 architecture itself is not optional: the
    # SwiGLU FFN, per-head pair bias, pair transition, rec<->lig cross attention,
    # entity-scoped triangle, outer-product-mean mixing, per-layer r_ij message,
    # vector norm, vector->scalar readout and the l=2 flow head are always on.
    chem_ffn_mult: int = 4
    chem_pair_transition_mult: int = 2
    chem_opm_dim: int = 32         # outer-product-mean single->pair mixing width (c)
    tp_method: str = 'naive'       # cuet FCTP method; 'naive' is ~30x faster than 'fused'

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.ns % self.num_atom_heads == 0, \
            f"ns ({self.ns}) must be divisible by num_atom_heads ({self.num_atom_heads})"
        assert self.ns % self.num_geo_heads == 0, \
            f"ns ({self.ns}) must be divisible by num_geo_heads ({self.num_geo_heads})"
        assert self.nv >= 1, f"nv must be at least 1, got {self.nv}"
        assert self.nv % self.num_geo_heads == 0, \
            f"nv ({self.nv}) must be divisible by num_geo_heads ({self.num_geo_heads})"
        # forward_flow passes precomputed per-head biases; the geo layer tells them
        # apart from raw pair features by the last dim.
        assert self.pair_hidden_dim != self.num_geo_heads, \
            "pair_hidden_dim must differ from num_geo_heads"
        assert self.geo_dist_bins > 0, \
            "the per-layer r_ij message needs geo_dist_bins > 0"
        assert self.chem_opm_dim > 0, \
            f"chem_opm_dim must be positive, got {self.chem_opm_dim}"
        assert self.tp_method in ('fused', 'naive'), f"tp_method must be fused|naive, got {self.tp_method!r}"

        if self.time_embed_dim != self.ns:
            self.time_embed_dim = self.ns


class FlexModel(nn.Module):
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

        ref_disto = torch.cdist(atom_ref_coords, atom_ref_coords).unsqueeze(-1)
        p = self.pair_embedder(
            distogram=ref_disto,
            adj_mat=bond_feats,
            constraint_mat=constraint_matrix,
            block_mask=block_mask,
        )
        # block_mask is per RESIDUE for the receptor (rec_uid_vec = residue index)
        # and per ligand for the ligand; entities are {receptor, each ligand}.
        entity_id = torch.cat([torch.zeros_like(rec_uid_vec), 1 + lig_uid_vec], dim=1)
        same_entity = entity_id.unsqueeze(-1) == entity_id.unsqueeze(-2)        # [S, N, N]

        # The distogram (pair embedder, above) stays block-masked: receptor ref
        # coords are per-residue templates. Attention, single->pair mixing and the
        # per-layer re-mask use the padding-only pair_mask; inter-residue and
        # cross-entity pairs are seeded from the post-fusion single features.
        p = p * block_mask
        pair_class = torch.where(block_mask.squeeze(-1).bool(), 0, torch.where(same_entity, 1, 2))
        cross_mask = pair_mask_f * (1.0 - block_mask)
        p = p + self.cross_pair_init(q, pair_class) * cross_mask
        del pair_class, cross_mask
        block_mask_f = pair_mask_f
        block_mask_2d = pair_mask.to(p.dtype).contiguous()
        # Triangle multiplication is scoped to one entity: intra-receptor and
        # intra-ligand pairs only, padding excluded.
        tri_mask_2d = (pair_mask & same_entity).to(p.dtype).contiguous()
        del ref_disto, bond_feats, constraint_matrix, block_mask, same_entity, entity_id

        # Interleaved AtomTransformer + TriangleUpdate (+ PairTransition)
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
                    pair_mask=block_mask_2d,
                )
                p = p * block_mask_f
                p = checkpoint(tri_layer, p, tri_mask_2d, use_reentrant=False)
            else:
                q, p = atom_layer(q=q, p=p, atom_mask=atom_mask, pair_mask=block_mask_2d)
                p = p * block_mask_f
                p = tri_layer(p, mask=tri_mask_2d)
            p = p * block_mask_f
            pt_layer = self.pair_transition_layers[f'pair_transition_{index}']
            if use_ckpt:
                p = checkpoint(pt_layer, p, block_mask_f, use_reentrant=False)
            else:
                p = pt_layer(p, block_mask_f)
            q = q * atom_mask_f

        del block_mask_f, block_mask_2d, tri_mask_2d
        return q, p

    def forward_flow(
        self,
        q_chem: torch.Tensor,
        p_chem: torch.Tensor,
        atom_coords: torch.Tensor,
        t: torch.Tensor,
        N_rec: int,
        atom_eva_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        system_index_per_timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = atom_coords.shape[0]
        scale = self.config.coord_scale
        atom_coords = atom_coords / scale

        # Per-layer pair bias [S, N, N, H] computed on the distinct systems, then
        # gathered per timestep; the [B, N, N, pair] copy is never materialised.
        geo_layers = list(self.geo_attention_layers.values())
        if self.config.use_gradient_checkpointing and self.training:
            # Recompute LN(p_chem) [S, N, N, pair] in backward instead of keeping one per layer.
            biases = [checkpoint(lambda x, l=layer: l.bias_proj(l.bias_norm(x)), p_chem, use_reentrant=False)
                      for layer in geo_layers]
        else:
            biases = [layer.bias_proj(layer.bias_norm(p_chem)) for layer in geo_layers]

        if system_index_per_timestep is not None:
            sys_idx = system_index_per_timestep.to(q_chem.device).long()
            q = q_chem.index_select(0, sys_idx)
            biases = [b.index_select(0, sys_idx) for b in biases]
        elif q_chem.shape[0] == 1 and B > 1:
            q = q_chem.expand(B, -1, -1)
        elif q_chem.shape[0] == B:
            q = q_chem
        else:
            raise ValueError(
                f"Chemistry batch ({q_chem.shape[0]}) cannot feed flow batch ({B}); "
                "provide system_index_per_timestep for padded multi-system batches."
            )

        mask_f = atom_eva_mask.to(q.dtype).unsqueeze(-1) if atom_eva_mask is not None else None

        t_emb = self.time_embedder(t)
        q = self.adaln_flow_init(q, t_emb.unsqueeze(1))
        if mask_f is not None:
            q = q * mask_f

        # The vector stream starts empty: it is seeded by the per-layer r_ij
        # message inside each geometric attention layer.
        attn_stats = {} if return_attn else None
        v = q.new_zeros(B, q.shape[1], self.config.nv * 8)

        # Live rec<->lig pocket cross-conditioning (compute once, in Å, before
        # the geo stack). atom_coords was divided by scale above; multiply back.
        q, v = self.pocket_interaction(q, v, atom_coords * scale, N_rec, atom_eva_mask)

        # #1: live per-head distance bias. Compute RBF(||x_i - x_j||) ONCE (x_t is
        # fixed within this forward) and reuse across all geo layers — proximity
        # routing the similarity-only logits lack.
        dist_rbf = self._distance_rbf(atom_coords * scale) if self.config.geo_dist_bins > 0 else None
        geom = self._pair_geometry(atom_coords * scale, atom_eva_mask)

        geo_attn_entropies = []
        for index, (layer, bias) in enumerate(zip(geo_layers, biases)):
            use_ckpt = (self.config.use_gradient_checkpointing and self.training
                        and index % self.config.gradient_checkpoint_every == 0)
            if use_ckpt:
                q, v = checkpoint(layer, q, v, bias, atom_eva_mask, False, dist_rbf, geom, use_reentrant=False)
            else:
                if return_attn:
                    q, v, geo_stats = layer(
                        scalar_feats=q, vector_feats=v, bias=bias,
                        atom_mask=atom_eva_mask, return_attn=True, dist_rbf=dist_rbf, geom=geom,
                    )
                    geo_attn_entropies.append(geo_stats['entropy'])
                else:
                    q, v = layer(scalar_feats=q, vector_feats=v, bias=bias,
                                 atom_mask=atom_eva_mask, dist_rbf=dist_rbf, geom=geom)
            q = self.adaln_geo_layers[f'adaln_geo_{index}'](q, t_emb.unsqueeze(1))
            if mask_f is not None:
                q = q * mask_f
                v = v * mask_f

        if return_attn and geo_attn_entropies:
            attn_stats['geo_entropy'] = torch.stack(geo_attn_entropies).mean()

        del biases, geom

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

    def _distance_rbf(self, x_ang: torch.Tensor) -> torch.Tensor:
        """RBF expansion of pairwise distances (Å) for the #1 distance bias.
        Returns [B, N, N, geo_dist_bins]."""
        d = torch.cdist(x_ang, x_ang)
        centers = torch.linspace(0.0, self.config.geo_dist_max,
                                 self.config.geo_dist_bins, device=x_ang.device, dtype=x_ang.dtype)
        width = self.config.geo_dist_max / self.config.geo_dist_bins
        return torch.exp(-((d.unsqueeze(-1) - centers) / width) ** 2)

    def _pair_geometry(self, x_ang: torch.Tensor, atom_mask: torch.Tensor | None):
        """unit_ij = r_hat(x_i - x_j) [B, N, N, 3] and Y2(unit_ij) [B, N, N, 5];
        zero on the diagonal, coincident pairs and padded atoms."""
        r = x_ang.unsqueeze(2) - x_ang.unsqueeze(1)
        dist = r.norm(dim=-1, keepdim=True)
        invalid = dist < 1e-6
        if atom_mask is not None:
            valid = atom_mask.bool()
            invalid = invalid | ~(valid.unsqueeze(1) & valid.unsqueeze(2)).unsqueeze(-1)
        # Safe direction on invalid pairs keeps the SH/norm backward finite.
        r = torch.where(invalid, torch.ones_like(r), r)
        unit = r / r.norm(dim=-1, keepdim=True)
        B, N = x_ang.shape[:2]
        y2 = _cuet_fp32(self.rij_sh, unit.reshape(-1, 3)).reshape(B, N, N, 5)
        return unit.masked_fill(invalid, 0.0), y2.masked_fill(invalid, 0.0)

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
                use_qk_norm=self.config.use_qk_norm,
                chem_ffn_mult=self.config.chem_ffn_mult,
                opm_dim=self.config.chem_opm_dim,
            ) for i in range(self.config.num_atom_layers)
        })

        self.triangle_layers = nn.ModuleDict({
            f'triangle_layer_{i}': TriangleUpdatingModule(
                ns=self.config.pair_hidden_dim,
            ) for i in range(self.config.num_atom_layers)
        })
        self.pair_transition_layers = nn.ModuleDict({
            f'pair_transition_{i}': PairTransition(
                self.config.pair_hidden_dim, self.config.chem_pair_transition_mult,
            ) for i in range(self.config.num_atom_layers)
        })
        self.cross_pair_init = CrossPairInit(self.config.ns, self.config.pair_hidden_dim)

        # --- Flow Matching Trunk ---
        self.time_embedder = TimeEmbedder(
            hidden_dim=self.config.ns,
            output_dim=self.config.ns,
        )

        self.adaln_flow_init = AdaLN(in_dim=self.config.ns)

        self.geo_attention_layers = nn.ModuleDict({
            f'geo_attention_{i}': GeometricAttentionModule(
                ns=self.config.ns,
                nv=self.config.nv,
                num_heads=self.config.num_geo_heads,
                bias_dim=self.config.pair_hidden_dim,
                attn_dropout=self.config.attn_dropout,
                use_gated_attention=self.config.use_gated_attention,
                soft_cap=self.config.soft_cap,
                dist_bins=self.config.geo_dist_bins,
                tp_method=self.config.tp_method,
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
            tp_method=self.config.tp_method,
        )
        self.pocket_interaction = PocketInteraction(
            ns=self.config.ns,
            nv=self.config.nv,
        )
        self.rij_sh = cuet.SphericalHarmonics(ls=[2], normalize=True)

    def _initialize_weights(self):
        if self.config.use_embeddings:
            nn.init.xavier_uniform_(self.rec_fusion.weight)
            nn.init.xavier_uniform_(self.lig_fusion.weight)
