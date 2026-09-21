"""Pocket--ligand interface Pairformer for structure-aware affinity prediction.

This module is intentionally *Pairformer-inspired* rather than a verbatim copy
of AlphaFold 3.  A full square ``(N_protein + N_ligand)^2`` representation with
triangle updates is prohibitively expensive for the batch sizes used by
KDBNet.  Instead, it maintains the rectangular, task-relevant interface tensor

    z[pocket residue, ligand atom, pair channel]

and alternates:

* pair-biased protein -> ligand and ligand -> protein attention;
* row/column axial attention over the interface tensor;
* multiplicative row/column pair updates; and
* single/pair transition MLPs.

The representation changes whenever the protein--ligand relative geometry
changes, making it suitable for crystal/noisy/docked/AF-derived structures.
Geometric inputs combine multi-atom distances with directions expressed in a
residue-local N--CA--C frame.  The branch is therefore invariant to a joint
rigid rotation or translation of the complex while remaining sensitive to
small ligand-pose and pocket-side-chain changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InterfaceBatch:
    """Padded pocket/ligand tensors built from two batched PyG-like objects."""

    protein_single: torch.Tensor       # [B, P, C_p_raw]
    protein_seq: torch.Tensor          # [B, P]
    protein_esm: torch.Tensor          # [B, P, C_esm]
    protein_pos: torch.Tensor          # [B, P, 3] C-alpha
    protein_backbone_pos: torch.Tensor # [B, P, 4, 3] N,CA,C,O
    protein_backbone_mask: torch.Tensor# [B, P, 4]
    protein_local_frame: torch.Tensor  # [B, P, 3, 3], basis vectors as rows
    protein_frame_mask: torch.Tensor   # [B, P]
    protein_cb: torch.Tensor           # [B, P, 3], real CB or pseudo-CB
    protein_cb_mask: torch.Tensor      # [B, P]
    protein_sidechain_center: torch.Tensor  # [B, P, 3]
    protein_sidechain_mask: torch.Tensor    # [B, P]
    protein_atom_pos: torch.Tensor     # [B, P, A, 3]
    protein_atom_mask: torch.Tensor    # [B, P, A]
    protein_full_atom_mask: torch.Tensor # [B, P]
    protein_mask: torch.Tensor         # [B, P]
    ligand_single: torch.Tensor        # [B, L, C_l_raw]
    ligand_pos: torch.Tensor           # [B, L, 3]
    ligand_mask: torch.Tensor          # [B, L]


class GaussianRBF(nn.Module):
    """Fixed Gaussian radial basis encoding."""

    def __init__(self, num_rbf: int = 32, d_min: float = 0.0, d_max: float = 20.0):
        super().__init__()
        if num_rbf < 2:
            raise ValueError("num_rbf must be >= 2")
        centers = torch.linspace(float(d_min), float(d_max), int(num_rbf))
        width = (float(d_max) - float(d_min)) / float(num_rbf)
        self.register_buffer("centers", centers, persistent=False)
        self.width = max(width, 1e-6)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        delta = (distance.unsqueeze(-1) - self.centers) / self.width
        return torch.exp(-(delta ** 2))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    numerator = (x * weight).sum(dim=dim)
    denominator = weight.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class FeedForwardTransition(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class PairBiasedCrossAttention(nn.Module):
    """Cross-attention with one scalar pair bias per attention head."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        pair_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        pair: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        # query [B,Q,C], context [B,K,Cc], pair [B,Q,K,Cz]
        batch_size, query_len, query_dim = query.shape
        context_len = context.size(1)

        q = self.to_q(self.query_norm(query)).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.to_k(self.context_norm(context)).view(
            batch_size, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.to_v(self.context_norm(context)).view(
            batch_size, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        logits = logits + self.pair_bias(pair).permute(0, 3, 1, 2)

        valid = query_mask[:, None, :, None] & context_mask[:, None, None, :]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        # Fully padded query rows would otherwise become uniform after softmax.
        attention = torch.where(valid, attention, torch.zeros_like(attention))
        attention = self.dropout(attention)

        update = torch.einsum("bhqk,bhkd->bhqd", attention, v)
        update = update.transpose(1, 2).contiguous().view(batch_size, query_len, query_dim)
        update = self.out(update)
        update = update * query_mask.unsqueeze(-1).to(update.dtype)
        return query + self.dropout(update)


class AxialPairAttention(nn.Module):
    """Attention over ligand atoms (rows) and pocket residues (columns)."""

    def __init__(self, pair_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.row_norm = nn.LayerNorm(pair_dim)
        self.col_norm = nn.LayerNorm(pair_dim)
        self.row_attention = nn.MultiheadAttention(
            pair_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.col_attention = nn.MultiheadAttention(
            pair_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_protein, n_ligand, pair_dim = pair.shape
        pair_mask = protein_mask[:, :, None] & ligand_mask[:, None, :]

        # Row attention: for every protein residue, communicate across ligand atoms.
        row = self.row_norm(pair).reshape(batch_size * n_protein, n_ligand, pair_dim)
        row_padding = (~ligand_mask[:, None, :]).expand(
            batch_size, n_protein, n_ligand
        ).reshape(batch_size * n_protein, n_ligand)
        row_update, _ = self.row_attention(
            row, row, row, key_padding_mask=row_padding, need_weights=False
        )
        row_update = row_update.reshape(batch_size, n_protein, n_ligand, pair_dim)
        row_update = row_update * pair_mask.unsqueeze(-1).to(row_update.dtype)
        pair = pair + self.dropout(row_update)

        # Column attention: for every ligand atom, communicate across pocket residues.
        col = self.col_norm(pair).transpose(1, 2).reshape(
            batch_size * n_ligand, n_protein, pair_dim
        )
        col_padding = (~protein_mask[:, None, :]).expand(
            batch_size, n_ligand, n_protein
        ).reshape(batch_size * n_ligand, n_protein)
        col_update, _ = self.col_attention(
            col, col, col, key_padding_mask=col_padding, need_weights=False
        )
        col_update = col_update.reshape(batch_size, n_ligand, n_protein, pair_dim).transpose(1, 2)
        col_update = col_update * pair_mask.unsqueeze(-1).to(col_update.dtype)
        return pair + self.dropout(col_update)


class RectangularMultiplicativeUpdate(nn.Module):
    """Efficient rectangular analogue of triangle multiplicative updates.

    Row summaries collect all ligand contacts around one pocket residue; column
    summaries collect all pocket contacts around one ligand atom.  Their gated
    product updates each residue--atom pair.  Complexity is O(P*L*C), not
    O((P+L)^3*C).
    """

    def __init__(self, pair_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.row_value = nn.Linear(pair_dim, pair_dim)
        self.row_gate = nn.Linear(pair_dim, pair_dim)
        self.col_value = nn.Linear(pair_dim, pair_dim)
        self.col_gate = nn.Linear(pair_dim, pair_dim)
        self.out_gate = nn.Linear(pair_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = protein_mask[:, :, None] & ligand_mask[:, None, :]
        x = self.norm(pair)
        x = x * pair_mask.unsqueeze(-1).to(x.dtype)

        row_term = torch.sigmoid(self.row_gate(x)) * self.row_value(x)
        col_term = torch.sigmoid(self.col_gate(x)) * self.col_value(x)
        row_summary = _masked_mean(row_term, pair_mask, dim=2)  # [B,P,C]
        col_summary = _masked_mean(col_term, pair_mask, dim=1)  # [B,L,C]

        update = row_summary[:, :, None, :] * col_summary[:, None, :, :]
        update = torch.sigmoid(self.out_gate(x)) * self.out(update)
        update = update * pair_mask.unsqueeze(-1).to(update.dtype)
        return pair + self.dropout(update)


class InterfacePairformerBlock(nn.Module):
    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        dropout: float,
        transition_expansion: int = 4,
    ):
        super().__init__()
        self.protein_to_ligand = PairBiasedCrossAttention(
            query_dim=single_dim,
            context_dim=single_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ligand_to_protein = PairBiasedCrossAttention(
            query_dim=single_dim,
            context_dim=single_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.protein_transition = FeedForwardTransition(
            single_dim, expansion=transition_expansion, dropout=dropout
        )
        self.ligand_transition = FeedForwardTransition(
            single_dim, expansion=transition_expansion, dropout=dropout
        )

        self.pair_norm = nn.LayerNorm(pair_dim)
        self.protein_to_pair = nn.Linear(single_dim, pair_dim)
        self.ligand_to_pair = nn.Linear(single_dim, pair_dim)
        self.outer_left = nn.Linear(single_dim, pair_dim)
        self.outer_right = nn.Linear(single_dim, pair_dim)
        self.pair_gate = nn.Linear(pair_dim, pair_dim)
        self.pair_from_single = nn.Linear(pair_dim, pair_dim)
        self.pair_dropout = nn.Dropout(dropout)

        self.multiplicative_update = RectangularMultiplicativeUpdate(pair_dim, dropout)
        self.axial_attention = AxialPairAttention(pair_dim, num_heads, dropout)
        self.pair_transition = FeedForwardTransition(
            pair_dim, expansion=transition_expansion, dropout=dropout
        )

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        pair: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_mask = protein_mask[:, :, None] & ligand_mask[:, None, :]

        protein = self.protein_to_ligand(
            protein, ligand, pair, protein_mask, ligand_mask
        )
        ligand = self.ligand_to_protein(
            ligand, protein, pair.transpose(1, 2), ligand_mask, protein_mask
        )
        protein = self.protein_transition(protein)
        ligand = self.ligand_transition(ligand)
        protein = protein * protein_mask.unsqueeze(-1).to(protein.dtype)
        ligand = ligand * ligand_mask.unsqueeze(-1).to(ligand.dtype)

        pair_normalized = self.pair_norm(pair)
        additive = (
            self.protein_to_pair(protein)[:, :, None, :]
            + self.ligand_to_pair(ligand)[:, None, :, :]
        )
        outer = (
            self.outer_left(protein)[:, :, None, :]
            * self.outer_right(ligand)[:, None, :, :]
        )
        pair_update = self.pair_from_single(additive + outer)
        pair_update = torch.sigmoid(self.pair_gate(pair_normalized)) * pair_update
        pair = pair + self.pair_dropout(pair_update)
        pair = pair * pair_mask.unsqueeze(-1).to(pair.dtype)

        pair = self.multiplicative_update(pair, protein_mask, ligand_mask)
        pair = self.axial_attention(pair, protein_mask, ligand_mask)
        pair = self.pair_transition(pair)
        pair = pair * pair_mask.unsqueeze(-1).to(pair.dtype)
        return protein, ligand, pair


class PocketLigandInterfacePairformer(nn.Module):
    """Structure-aware local branch operating on pocket residues x ligand atoms.

    Parameters are deliberately moderate because LP-PDBbind is much smaller
    than AlphaFold-scale training corpora.  The module consumes the existing
    KDBNet protein/drug PyG batches directly, so no new serialized data format
    is required.
    """

    def __init__(
        self,
        protein_esm_dim: int = 1280,
        protein_scalar_dim: int = 6,
        ligand_scalar_dim: int = 66,
        single_dim: int = 128,
        pair_dim: int = 64,
        output_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        num_rbf: int = 32,
        fine_num_rbf: int = 32,
        max_pocket_residues: int = 64,
        max_ligand_atoms: int = 64,
        max_residue_atoms: int = 14,
        pocket_radius: float = 10.0,
        min_pocket_residues: int = 16,
        distance_max: float = 20.0,
        fine_distance_max: float = 8.0,
        dropout: float = 0.15,
    ):
        super().__init__()
        if single_dim % num_heads != 0:
            raise ValueError("single_dim must be divisible by num_heads")
        if pair_dim % num_heads != 0:
            raise ValueError("pair_dim must be divisible by num_heads")
        if max_pocket_residues <= 0 or max_ligand_atoms <= 0:
            raise ValueError("max_pocket_residues and max_ligand_atoms must be positive")
        if min_pocket_residues <= 0:
            raise ValueError("min_pocket_residues must be positive")

        self.protein_esm_dim = int(protein_esm_dim)
        self.protein_scalar_dim = int(protein_scalar_dim)
        self.ligand_scalar_dim = int(ligand_scalar_dim)
        self.single_dim = int(single_dim)
        self.pair_dim = int(pair_dim)
        self.output_dim = int(output_dim)
        self.max_pocket_residues = int(max_pocket_residues)
        self.max_ligand_atoms = int(max_ligand_atoms)
        self.max_residue_atoms = int(max_residue_atoms)
        if self.max_residue_atoms < 5:
            raise ValueError("max_residue_atoms must be at least 5")
        self.pocket_radius = float(pocket_radius)
        self.fine_distance_max = float(fine_distance_max)
        self.min_pocket_residues = min(int(min_pocket_residues), self.max_pocket_residues)

        self.amino_acid_embedding = nn.Embedding(21, single_dim)
        self.protein_scalar_projection = nn.Linear(protein_scalar_dim, single_dim)
        self.protein_esm_projection = nn.Linear(protein_esm_dim, single_dim)
        self.ligand_scalar_projection = nn.Linear(ligand_scalar_dim, single_dim)
        self.protein_type_embedding = nn.Parameter(torch.zeros(single_dim))
        self.ligand_type_embedding = nn.Parameter(torch.zeros(single_dim))
        self.single_dropout = nn.Dropout(dropout)

        # Two distance resolutions are used.  The broad basis preserves long-
        # range pocket context; the fine basis (0--8 A by default) gives the
        # model enough resolution to respond to sub-angstrom interface changes.
        self.rbf = GaussianRBF(num_rbf=num_rbf, d_min=0.0, d_max=distance_max)
        self.fine_rbf = GaussianRBF(
            num_rbf=fine_num_rbf, d_min=0.0, d_max=fine_distance_max
        )
        self.ca_distance_projection = nn.Linear(num_rbf, pair_dim)
        self.ca_fine_projection = nn.Linear(fine_num_rbf, pair_dim)
        self.min_atom_distance_projection = nn.Linear(fine_num_rbf, pair_dim)
        self.n_distance_projection = nn.Linear(fine_num_rbf, pair_dim)
        self.c_distance_projection = nn.Linear(fine_num_rbf, pair_dim)
        self.o_distance_projection = nn.Linear(fine_num_rbf, pair_dim)
        self.cb_distance_projection = nn.Linear(fine_num_rbf, pair_dim)
        self.sidechain_distance_projection = nn.Linear(fine_num_rbf, pair_dim)
        # local unit vector (3), clipped local displacement (3), side-chain
        # alignment (1), and three geometry-availability flags.
        self.direction_projection = nn.Sequential(
            nn.Linear(10, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )
        self.geometry_norm = nn.LayerNorm(pair_dim)
        self.pair_type_embedding = nn.Parameter(torch.zeros(pair_dim))
        self.pair_from_single_left = nn.Linear(single_dim, pair_dim)
        self.pair_from_single_right = nn.Linear(single_dim, pair_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.blocks = nn.ModuleList([
            InterfacePairformerBlock(
                single_dim=single_dim,
                pair_dim=pair_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(int(num_blocks))
        ])

        self.pool_score = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(single_dim * 2 + pair_dim),
            nn.Linear(single_dim * 2 + pair_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

        nn.init.normal_(self.protein_type_embedding, std=0.02)
        nn.init.normal_(self.ligand_type_embedding, std=0.02)
        nn.init.normal_(self.pair_type_embedding, std=0.02)

    @staticmethod
    def _batch_size(protein_batch: torch.Tensor, ligand_batch: torch.Tensor) -> int:
        protein_max = int(protein_batch.max().item()) if protein_batch.numel() else -1
        ligand_max = int(ligand_batch.max().item()) if ligand_batch.numel() else -1
        return max(protein_max, ligand_max) + 1

    def _build_interface_batch(self, protein_graph, ligand_graph) -> InterfaceBatch:
        required_protein = ("x", "batch", "seq", "node_s", "seq_emb")
        required_ligand = ("x", "batch", "node_s")
        for attr in required_protein:
            if not hasattr(protein_graph, attr):
                raise ValueError("Protein graph missing required attribute '{}'".format(attr))
        for attr in required_ligand:
            if not hasattr(ligand_graph, attr):
                raise ValueError("Ligand graph missing required attribute '{}'".format(attr))

        device = protein_graph.x.device
        dtype = protein_graph.x.dtype
        batch_size = self._batch_size(protein_graph.batch, ligand_graph.batch)
        if batch_size <= 0:
            raise ValueError("Empty protein/ligand batch")

        p_scalar = torch.zeros(
            batch_size, self.max_pocket_residues, self.protein_scalar_dim,
            device=device, dtype=protein_graph.node_s.dtype,
        )
        p_seq = torch.zeros(
            batch_size, self.max_pocket_residues, device=device, dtype=torch.long
        )
        p_esm = torch.zeros(
            batch_size, self.max_pocket_residues, self.protein_esm_dim,
            device=device, dtype=protein_graph.seq_emb.dtype,
        )
        p_pos = torch.zeros(
            batch_size, self.max_pocket_residues, 3, device=device, dtype=dtype
        )
        p_backbone = torch.zeros(
            batch_size, self.max_pocket_residues, 4, 3, device=device, dtype=dtype
        )
        p_backbone_mask = torch.zeros(
            batch_size, self.max_pocket_residues, 4, device=device, dtype=torch.bool
        )
        p_frame = torch.zeros(
            batch_size, self.max_pocket_residues, 3, 3, device=device, dtype=dtype
        )
        p_frame_mask = torch.zeros(
            batch_size, self.max_pocket_residues, device=device, dtype=torch.bool
        )
        p_cb = torch.zeros(
            batch_size, self.max_pocket_residues, 3, device=device, dtype=dtype
        )
        p_cb_mask = torch.zeros(
            batch_size, self.max_pocket_residues, device=device, dtype=torch.bool
        )
        p_side = torch.zeros_like(p_cb)
        p_side_mask = torch.zeros_like(p_cb_mask)
        p_atom = torch.zeros(
            batch_size, self.max_pocket_residues, self.max_residue_atoms, 3,
            device=device, dtype=dtype,
        )
        p_atom_mask = torch.zeros(
            batch_size, self.max_pocket_residues, self.max_residue_atoms,
            device=device, dtype=torch.bool,
        )
        p_full_atom_mask = torch.zeros(
            batch_size, self.max_pocket_residues, device=device, dtype=torch.bool
        )
        p_mask = torch.zeros(
            batch_size, self.max_pocket_residues, device=device, dtype=torch.bool
        )
        l_scalar = torch.zeros(
            batch_size, self.max_ligand_atoms, self.ligand_scalar_dim,
            device=device, dtype=ligand_graph.node_s.dtype,
        )
        l_pos = torch.zeros(
            batch_size, self.max_ligand_atoms, 3,
            device=device, dtype=ligand_graph.x.dtype,
        )
        l_mask = torch.zeros(
            batch_size, self.max_ligand_atoms, device=device, dtype=torch.bool
        )

        for batch_index in range(batch_size):
            protein_index = torch.nonzero(
                protein_graph.batch == batch_index, as_tuple=False
            ).flatten()
            ligand_index = torch.nonzero(
                ligand_graph.batch == batch_index, as_tuple=False
            ).flatten()
            if protein_index.numel() == 0 or ligand_index.numel() == 0:
                continue

            if ligand_index.numel() > self.max_ligand_atoms:
                ligand_index = ligand_index[: self.max_ligand_atoms]
            ligand_positions = ligand_graph.x[ligand_index]
            protein_positions = protein_graph.x[protein_index]

            residue_atom_pos = getattr(protein_graph, "residue_atom_pos", None)
            residue_atom_mask = getattr(protein_graph, "residue_atom_mask", None)
            if residue_atom_pos is not None and residue_atom_mask is not None:
                local_atom_pos = residue_atom_pos[protein_index]
                local_atom_mask = residue_atom_mask[protein_index].to(torch.bool)
                atom_distance = torch.linalg.norm(
                    local_atom_pos[:, :, None, :].float()
                    - ligand_positions[None, None, :, :].float(),
                    dim=-1,
                )
                atom_distance = atom_distance.masked_fill(
                    ~local_atom_mask[:, :, None], float("inf")
                )
                nearest = atom_distance.amin(dim=(1, 2))
                ca_nearest = torch.cdist(
                    protein_positions.float(), ligand_positions.float()
                ).amin(dim=1)
                nearest = torch.where(torch.isfinite(nearest), nearest, ca_nearest)
            else:
                nearest = torch.cdist(
                    protein_positions.float(), ligand_positions.float()
                ).amin(dim=1)
            nearest_order = torch.argsort(nearest)

            fixed_mask = getattr(protein_graph, "interface_pocket_mask", None)
            if fixed_mask is not None:
                local_fixed_mask = fixed_mask[protein_index].to(torch.bool)
                fixed_local = torch.nonzero(local_fixed_mask, as_tuple=False).flatten()
            else:
                fixed_local = torch.empty(0, device=device, dtype=torch.long)

            if fixed_local.numel() > 0:
                selected_local = fixed_local[torch.argsort(nearest[fixed_local])]
                if selected_local.numel() < self.min_pocket_residues:
                    selected_set = torch.zeros(
                        protein_positions.size(0), device=device, dtype=torch.bool
                    )
                    selected_set[selected_local] = True
                    extras = nearest_order[~selected_set[nearest_order]]
                    need = self.min_pocket_residues - selected_local.numel()
                    selected_local = torch.cat([selected_local, extras[:need]], dim=0)
            else:
                within_radius = torch.nonzero(
                    nearest <= self.pocket_radius, as_tuple=False
                ).flatten()
                if within_radius.numel() < self.min_pocket_residues:
                    selected_local = nearest_order[: self.min_pocket_residues]
                else:
                    selected_local = within_radius[torch.argsort(nearest[within_radius])]
            selected_local = selected_local[: self.max_pocket_residues]
            selected_protein = protein_index[selected_local]

            n_p = selected_protein.numel()
            n_l = ligand_index.numel()
            p_scalar[batch_index, :n_p] = protein_graph.node_s[selected_protein]
            p_seq[batch_index, :n_p] = protein_graph.seq[selected_protein]
            p_esm[batch_index, :n_p] = protein_graph.seq_emb[selected_protein]
            p_pos[batch_index, :n_p] = protein_graph.x[selected_protein]

            backbone = getattr(protein_graph, "backbone_pos", None)
            backbone_mask = getattr(protein_graph, "backbone_atom_mask", None)
            if backbone is not None:
                p_backbone[batch_index, :n_p] = backbone[selected_protein]
            else:
                p_backbone[batch_index, :n_p, 1] = protein_graph.x[selected_protein]
            if backbone_mask is not None:
                p_backbone_mask[batch_index, :n_p] = backbone_mask[selected_protein]
            else:
                p_backbone_mask[batch_index, :n_p, 1] = True

            frame = getattr(protein_graph, "local_frame", None)
            frame_mask = getattr(protein_graph, "local_frame_mask", None)
            if frame is not None:
                p_frame[batch_index, :n_p] = frame[selected_protein]
            if frame_mask is not None:
                p_frame_mask[batch_index, :n_p] = frame_mask[selected_protein]

            cb = getattr(protein_graph, "residue_cb", None)
            cb_mask = getattr(protein_graph, "residue_cb_mask", None)
            if cb is None:
                cb = getattr(protein_graph, "pseudo_cb", protein_graph.x)
            p_cb[batch_index, :n_p] = cb[selected_protein]
            if cb_mask is not None:
                p_cb_mask[batch_index, :n_p] = cb_mask[selected_protein]
            else:
                p_cb_mask[batch_index, :n_p] = True

            side = getattr(protein_graph, "sidechain_center", cb)
            side_mask = getattr(protein_graph, "sidechain_center_mask", cb_mask)
            p_side[batch_index, :n_p] = side[selected_protein]
            if side_mask is not None:
                p_side_mask[batch_index, :n_p] = side_mask[selected_protein]
            else:
                p_side_mask[batch_index, :n_p] = True

            if residue_atom_pos is not None and residue_atom_mask is not None:
                available_atoms = min(
                    self.max_residue_atoms, residue_atom_pos.size(1)
                )
                p_atom[batch_index, :n_p, :available_atoms] = (
                    residue_atom_pos[selected_protein, :available_atoms]
                )
                p_atom_mask[batch_index, :n_p, :available_atoms] = (
                    residue_atom_mask[selected_protein, :available_atoms]
                )
            else:
                p_atom[batch_index, :n_p, 0] = protein_graph.x[selected_protein]
                p_atom_mask[batch_index, :n_p, 0] = True
            full_atom = getattr(protein_graph, "has_full_atom_geometry", None)
            if full_atom is not None:
                p_full_atom_mask[batch_index, :n_p] = full_atom[selected_protein]

            p_mask[batch_index, :n_p] = True
            l_scalar[batch_index, :n_l] = ligand_graph.node_s[ligand_index]
            l_pos[batch_index, :n_l] = ligand_graph.x[ligand_index]
            l_mask[batch_index, :n_l] = True

        return InterfaceBatch(
            protein_single=p_scalar,
            protein_seq=p_seq,
            protein_esm=p_esm,
            protein_pos=p_pos,
            protein_backbone_pos=p_backbone,
            protein_backbone_mask=p_backbone_mask,
            protein_local_frame=p_frame,
            protein_frame_mask=p_frame_mask,
            protein_cb=p_cb,
            protein_cb_mask=p_cb_mask,
            protein_sidechain_center=p_side,
            protein_sidechain_mask=p_side_mask,
            protein_atom_pos=p_atom,
            protein_atom_mask=p_atom_mask,
            protein_full_atom_mask=p_full_atom_mask,
            protein_mask=p_mask,
            ligand_single=l_scalar,
            ligand_pos=l_pos,
            ligand_mask=l_mask,
        )

    @staticmethod
    def _distance_to_anchor(
        anchor: torch.Tensor,
        ligand: torch.Tensor,
        anchor_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distance = torch.linalg.norm(
            anchor[:, :, None, :].float() - ligand[:, None, :, :].float(), dim=-1
        ).to(anchor.dtype)
        valid = anchor_mask[:, :, None] & ligand_mask[:, None, :]
        return distance, valid

    def _distance_feature(
        self,
        distance: torch.Tensor,
        valid: torch.Tensor,
        projection: nn.Linear,
        fine: bool = True,
    ) -> torch.Tensor:
        basis = self.fine_rbf(distance) if fine else self.rbf(distance)
        encoded = projection(basis)
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)

    def _geometry_embedding(self, interface: InterfaceBatch) -> torch.Tensor:
        pair_mask = interface.protein_mask[:, :, None] & interface.ligand_mask[:, None, :]
        ca_distance, ca_valid = self._distance_to_anchor(
            interface.protein_pos,
            interface.ligand_pos,
            interface.protein_mask,
            interface.ligand_mask,
        )
        n_distance, n_valid = self._distance_to_anchor(
            interface.protein_backbone_pos[:, :, 0],
            interface.ligand_pos,
            interface.protein_backbone_mask[:, :, 0],
            interface.ligand_mask,
        )
        c_distance, c_valid = self._distance_to_anchor(
            interface.protein_backbone_pos[:, :, 2],
            interface.ligand_pos,
            interface.protein_backbone_mask[:, :, 2],
            interface.ligand_mask,
        )
        o_distance, o_valid = self._distance_to_anchor(
            interface.protein_backbone_pos[:, :, 3],
            interface.ligand_pos,
            interface.protein_backbone_mask[:, :, 3],
            interface.ligand_mask,
        )
        cb_distance, cb_valid = self._distance_to_anchor(
            interface.protein_cb,
            interface.ligand_pos,
            interface.protein_cb_mask,
            interface.ligand_mask,
        )
        side_distance, side_valid = self._distance_to_anchor(
            interface.protein_sidechain_center,
            interface.ligand_pos,
            interface.protein_sidechain_mask,
            interface.ligand_mask,
        )

        # Exact minimum distance from each ligand atom to all available heavy
        # atoms in a residue.  This is the most direct signal for clashes and
        # subtle side-chain/contact changes.
        atom_distance = torch.linalg.norm(
            interface.protein_atom_pos[:, :, :, None, :].float()
            - interface.ligand_pos[:, None, None, :, :].float(),
            dim=-1,
        ).to(interface.protein_pos.dtype)
        atom_valid = (
            interface.protein_atom_mask[:, :, :, None]
            & interface.ligand_mask[:, None, None, :]
        )
        atom_distance = atom_distance.masked_fill(~atom_valid, float("inf"))
        min_atom_distance = atom_distance.amin(dim=2)
        min_atom_valid = (
            interface.protein_atom_mask.any(dim=2)[:, :, None]
            & interface.ligand_mask[:, None, :]
        )
        min_atom_distance = torch.where(
            min_atom_valid, min_atom_distance, ca_distance
        )

        geometry = (
            self._distance_feature(ca_distance, ca_valid, self.ca_distance_projection, fine=False)
            + self._distance_feature(ca_distance, ca_valid, self.ca_fine_projection)
            + self._distance_feature(
                min_atom_distance, min_atom_valid, self.min_atom_distance_projection
            )
            + self._distance_feature(n_distance, n_valid, self.n_distance_projection)
            + self._distance_feature(c_distance, c_valid, self.c_distance_projection)
            + self._distance_feature(o_distance, o_valid, self.o_distance_projection)
            + self._distance_feature(cb_distance, cb_valid, self.cb_distance_projection)
            + self._distance_feature(
                side_distance, side_valid, self.sidechain_distance_projection
            )
        )

        displacement = (
            interface.ligand_pos[:, None, :, :]
            - interface.protein_pos[:, :, None, :]
        )
        # Frames store basis vectors as rows; projection gives coordinates in
        # the residue-local N--CA--C frame and is invariant to joint rotation.
        local_displacement = torch.einsum(
            "bpqc,bplc->bplq", interface.protein_local_frame, displacement
        )
        local_unit = local_displacement / ca_distance.clamp_min(1e-6).unsqueeze(-1)
        local_scaled = torch.clamp(
            local_displacement / max(self.fine_distance_max, 1e-6), -2.0, 2.0
        )

        side_vector = interface.protein_sidechain_center - interface.protein_pos
        side_unit = side_vector / torch.linalg.norm(
            side_vector, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        ligand_unit = displacement / ca_distance.clamp_min(1e-6).unsqueeze(-1)
        side_alignment = (
            side_unit[:, :, None, :] * ligand_unit
        ).sum(dim=-1, keepdim=True)

        frame_flag = interface.protein_frame_mask[:, :, None, None].expand(
            -1, -1, interface.ligand_pos.size(1), -1
        )
        full_atom_flag = interface.protein_full_atom_mask[:, :, None, None].expand_as(
            frame_flag
        )
        side_flag = interface.protein_sidechain_mask[:, :, None, None].expand_as(
            frame_flag
        )
        direction_features = torch.cat(
            [
                local_unit,
                local_scaled,
                side_alignment,
                frame_flag.to(local_unit.dtype),
                full_atom_flag.to(local_unit.dtype),
                side_flag.to(local_unit.dtype),
            ],
            dim=-1,
        )
        direction_valid = (
            interface.protein_frame_mask[:, :, None]
            & interface.ligand_mask[:, None, :]
        )
        direction_embedding = self.direction_projection(direction_features)
        direction_embedding = direction_embedding * direction_valid.unsqueeze(-1).to(
            direction_embedding.dtype
        )
        geometry = geometry + direction_embedding
        geometry = self.geometry_norm(geometry)
        return geometry * pair_mask.unsqueeze(-1).to(geometry.dtype)

    def forward(self, protein_graph, ligand_graph) -> torch.Tensor:
        interface = self._build_interface_batch(protein_graph, ligand_graph)
        protein = (
            self.amino_acid_embedding(interface.protein_seq)
            + self.protein_scalar_projection(interface.protein_single)
            + self.protein_esm_projection(interface.protein_esm)
            + self.protein_type_embedding.view(1, 1, -1)
        )
        ligand = (
            self.ligand_scalar_projection(interface.ligand_single)
            + self.ligand_type_embedding.view(1, 1, -1)
        )
        protein = self.single_dropout(protein)
        ligand = self.single_dropout(ligand)
        protein = protein * interface.protein_mask.unsqueeze(-1).to(protein.dtype)
        ligand = ligand * interface.ligand_mask.unsqueeze(-1).to(ligand.dtype)

        pair = (
            self._geometry_embedding(interface)
            + self.pair_from_single_left(protein)[:, :, None, :]
            + self.pair_from_single_right(ligand)[:, None, :, :]
            + self.pair_type_embedding.view(1, 1, 1, -1)
        )
        pair_mask = interface.protein_mask[:, :, None] & interface.ligand_mask[:, None, :]
        pair = self.pair_norm(pair)
        pair = pair * pair_mask.unsqueeze(-1).to(pair.dtype)

        for block in self.blocks:
            protein, ligand, pair = block(
                protein,
                ligand,
                pair,
                interface.protein_mask,
                interface.ligand_mask,
            )

        pair_logits = self.pool_score(pair).squeeze(-1)
        pair_logits = pair_logits.masked_fill(
            ~pair_mask, torch.finfo(pair_logits.dtype).min
        )
        pair_weight = torch.softmax(pair_logits.flatten(1), dim=-1).view_as(pair_logits)
        pair_weight = torch.where(pair_mask, pair_weight, torch.zeros_like(pair_weight))
        pair_pooled = (pair * pair_weight.unsqueeze(-1)).sum(dim=(1, 2))
        protein_pooled = _masked_mean(protein, interface.protein_mask, dim=1)
        ligand_pooled = _masked_mean(ligand, interface.ligand_mask, dim=1)

        return self.output(torch.cat([protein_pooled, ligand_pooled, pair_pooled], dim=-1))
