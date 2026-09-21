"""V27 dual-view pair-token model with optional privileged 3D conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from dta.optional_structure_staged_v22.model import (
    DISTANCE_BINS,
    DISTANCE_BOUNDARIES,
    RBF_BINS,
    AttentionPool,
    BidirectionalCrossBlock,
    CrossAttention,
    FeedForward,
    Ligand2DEncoder,
    PairBiasedSelfAttention,
    PairMLPBlock,
    Protein1DEncoder,
    _masked_softmax,
    _rbf,
)


@dataclass
class PairTokenEncoding:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    latent: torch.Tensor
    contact_logits: torch.Tensor


@dataclass
class GeometryTokenEncoding:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    latent: torch.Tensor
    contact_logits: torch.Tensor
    distance_logits: torch.Tensor
    quality: torch.Tensor


class SequenceLigandPairEncoder(nn.Module):
    """Builds sparse residue-atom tokens without accessing coordinates."""

    def __init__(self, width, pair_width, pair_factor, heads, layers, top_pairs, dropout):
        super().__init__()
        self.top_pairs = int(top_pairs)
        self.protein_pair = nn.Linear(int(width), int(pair_width), bias=False)
        self.ligand_pair = nn.Linear(int(width), int(pair_width), bias=False)
        self.outer_left = nn.Linear(int(width), int(pair_factor), bias=False)
        self.outer_right = nn.Linear(int(width), int(pair_factor), bias=False)
        self.outer_projection = nn.Linear(int(pair_factor), int(pair_width), bias=False)
        self.transition = nn.Sequential(
            nn.LayerNorm(int(pair_width)),
            nn.Linear(int(pair_width), 2 * int(pair_width)),
            nn.SiLU(),
            nn.Linear(2 * int(pair_width), int(pair_width)),
        )
        self.contact_head = nn.Linear(int(pair_width), 1)
        layer = nn.TransformerEncoderLayer(
            d_model=int(pair_width),
            nhead=int(heads),
            dim_feedforward=4 * int(pair_width),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.token_pool = AttentionPool(pair_width)

    @staticmethod
    def _gather(values, indices):
        return torch.gather(values, 1, indices[..., None].expand(-1, -1, values.shape[-1]))

    def forward(self, protein, ligand, protein_mask, ligand_mask):
        pair_mask = protein_mask[:, :, None] & ligand_mask[:, None, :]
        outer = self.outer_left(protein)[:, :, None, :] * self.outer_right(ligand)[:, None, :, :]
        pair = (
            self.protein_pair(protein)[:, :, None, :]
            + self.ligand_pair(ligand)[:, None, :, :]
            + self.outer_projection(outer)
        )
        pair = torch.tanh(pair + self.transition(pair))
        pair = pair * pair_mask[..., None].to(pair.dtype)
        contact_logits = self.contact_head(pair).squeeze(-1)
        batch_size, protein_count, ligand_count, pair_width = pair.shape
        flat_pair = pair.reshape(batch_size, protein_count * ligand_count, pair_width)
        flat_mask = pair_mask.reshape(batch_size, protein_count * ligand_count)
        flat_score = contact_logits.reshape(batch_size, protein_count * ligand_count)
        soft_weights = _masked_softmax(flat_score, flat_mask, dimension=1)
        soft_context = torch.sum(flat_pair * soft_weights[..., None], dim=1)
        token_count = min(self.top_pairs, flat_pair.shape[1])
        selected_index = torch.topk(
            flat_score.masked_fill(~flat_mask, -1.0e4),
            k=token_count,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        selected_mask = torch.gather(flat_mask, 1, selected_index)
        absent = ~selected_mask.any(dim=1)
        if absent.any():
            selected_mask = selected_mask.clone()
            selected_mask[absent, 0] = True
        tokens = self._gather(flat_pair, selected_index)
        tokens = self.token_encoder(tokens, src_key_padding_mask=~selected_mask)
        latent = 0.5 * (self.token_pool(tokens, selected_mask) + soft_context)
        return PairTokenEncoding(tokens, selected_mask, latent, contact_logits)


class MaskedGeometryPairEncoder(nn.Module):
    """Produces pocket-atom pair tokens while supporting masked cross-distance inputs."""

    def __init__(
        self,
        width,
        pair_width,
        heads,
        self_layers,
        pair_layers,
        token_layers,
        top_pairs,
        latent_width,
        dropout,
    ):
        super().__init__()
        self.top_pairs = int(top_pairs)
        self.pocket = nn.Sequential(
            nn.LayerNorm(640 + 15),
            nn.Linear(640 + 15, int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(width)),
        )
        self.pocket_aa = nn.Embedding(21, int(width))
        self.ligand_categorical = nn.Sequential(
            nn.LayerNorm(19),
            nn.Linear(19, int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(width)),
        )
        self.ligand_continuous = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(width)),
        )
        centers = torch.linspace(0.0, 24.0, RBF_BINS)
        self.register_buffer("distance_centers", centers, persistent=True)
        self.distance_width = float(centers[1] - centers[0])
        self.pocket_blocks = nn.ModuleList(
            PairBiasedSelfAttention(width, heads, RBF_BINS, dropout)
            for _ in range(int(self_layers))
        )
        self.ligand_blocks = nn.ModuleList(
            PairBiasedSelfAttention(width, heads, RBF_BINS, dropout)
            for _ in range(int(self_layers))
        )
        self.pocket_pair = nn.Linear(int(width), int(pair_width))
        self.ligand_pair = nn.Linear(int(width), int(pair_width))
        self.distance_pair = nn.Sequential(
            nn.Linear(RBF_BINS, int(pair_width)),
            nn.SiLU(),
            nn.Linear(int(pair_width), int(pair_width)),
        )
        self.visibility_embedding = nn.Embedding(2, int(pair_width))
        self.pair_blocks = nn.ModuleList(
            PairMLPBlock(pair_width, dropout) for _ in range(int(pair_layers))
        )
        self.contact_head = nn.Linear(int(pair_width), 1)
        self.distance_head = nn.Linear(int(pair_width), DISTANCE_BINS)
        layer = nn.TransformerEncoderLayer(
            d_model=int(pair_width),
            nhead=int(heads),
            dim_feedforward=4 * int(pair_width),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(layer, num_layers=int(token_layers))
        self.token_pool = AttentionPool(pair_width)
        self.output = nn.Sequential(
            nn.LayerNorm(int(pair_width)),
            nn.Linear(int(pair_width), int(latent_width)),
            nn.SiLU(),
            nn.Linear(int(latent_width), int(latent_width)),
        )

    @staticmethod
    def _gather(values, indices):
        return torch.gather(values, 1, indices[..., None].expand(-1, -1, values.shape[-1]))

    def forward(self, batch):
        pocket_mask = batch["pocket_mask"].bool()
        ligand_mask = batch["structure_ligand_mask"].bool()
        pocket = self.pocket(torch.cat((batch["pocket_esm"], batch["pocket_scalar"]), dim=-1))
        pocket = pocket + self.pocket_aa(batch["pocket_aa"].long().clamp(0, 20))
        ligand = self.ligand_categorical(batch["structure_ligand_categorical"] / 8.0)
        ligand = ligand + self.ligand_continuous(batch["structure_ligand_continuous"])
        pocket_distance = torch.cdist(batch["pocket_coords"], batch["pocket_coords"])
        ligand_distance = torch.cdist(
            batch["structure_ligand_coords"], batch["structure_ligand_coords"]
        )
        pocket_rbf = _rbf(pocket_distance, self.distance_centers, self.distance_width)
        ligand_rbf = _rbf(ligand_distance, self.distance_centers, self.distance_width)
        for block in self.pocket_blocks:
            pocket = block(pocket, pocket_rbf, pocket_mask)
        for block in self.ligand_blocks:
            ligand = block(ligand, ligand_rbf, ligand_mask)

        pair_mask = pocket_mask[:, :, None] & ligand_mask[:, None, :]
        visible = batch.get("geometry_visible", pair_mask).bool() & pair_mask
        cross_distance = batch["residue_ligand_distance"]
        cross_rbf = _rbf(cross_distance, self.distance_centers, self.distance_width)
        cross_rbf = cross_rbf * visible[..., None].to(cross_rbf.dtype)
        pair = (
            self.pocket_pair(pocket)[:, :, None, :]
            + self.ligand_pair(ligand)[:, None, :, :]
            + self.distance_pair(cross_rbf)
            + self.visibility_embedding(visible.long())
        )
        pair = torch.tanh(pair) * pair_mask[..., None].to(pair.dtype)
        for block in self.pair_blocks:
            pair = block(pair, pair_mask)
        contact_logits = self.contact_head(pair).squeeze(-1)
        distance_logits = self.distance_head(pair)

        batch_size, pocket_count, ligand_count, pair_width = pair.shape
        flat_pair = pair.reshape(batch_size, pocket_count * ligand_count, pair_width)
        flat_mask = pair_mask.reshape(batch_size, pocket_count * ligand_count)
        flat_score = contact_logits.reshape(batch_size, pocket_count * ligand_count)
        token_count = min(self.top_pairs, flat_pair.shape[1])
        selected_index = torch.topk(
            flat_score.masked_fill(~flat_mask, -1.0e4),
            k=token_count,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        selected_mask = torch.gather(flat_mask, 1, selected_index)
        absent = ~selected_mask.any(dim=1)
        if absent.any():
            selected_mask = selected_mask.clone()
            selected_mask[absent, 0] = True
        tokens = self._gather(flat_pair, selected_index)
        tokens = self.token_encoder(tokens, src_key_padding_mask=~selected_mask)
        context = self.token_pool(tokens, selected_mask)
        latent = self.output(context)
        valid_distance = cross_distance.masked_fill(~pair_mask, 24.0)
        minimum = valid_distance.flatten(1).min(dim=1).values.clamp(0.0, 24.0) / 24.0
        contact_fraction = ((cross_distance <= 4.5) & pair_mask).sum(dim=(1, 2)).to(pair.dtype)
        contact_fraction = contact_fraction / pair_mask.sum(dim=(1, 2)).clamp_min(1).to(pair.dtype)
        coverage = torch.log1p(pair_mask.sum(dim=(1, 2)).to(pair.dtype)) / 12.0
        quality = torch.stack((minimum, contact_fraction, coverage), dim=-1)
        return GeometryTokenEncoding(
            tokens, selected_mask, latent, contact_logits, distance_logits, quality
        )


@dataclass
class OptionalStructurePrivilegedOutput:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    base_latent: torch.Tensor
    full_latent: torch.Tensor
    sequence_pair_latent: torch.Tensor
    full_pair_latent: torch.Tensor
    sequence_contact_logits: torch.Tensor
    geometry_quality_prediction: torch.Tensor
    geometry_quality_target: torch.Tensor
    structure_latent: torch.Tensor
    structure_gate: torch.Tensor
    structure_delta: torch.Tensor
    effective_has_3d: torch.Tensor
    contact_logits: Optional[torch.Tensor]
    distance_logits: Optional[torch.Tensor]


class OptionalStructurePrivilegedPairDTA(nn.Module):
    """One checkpoint with a coordinate-free base and a shared optional-3D pair path."""

    def __init__(
        self,
        protein_input_width=1280,
        width=384,
        heads=8,
        protein_layers=2,
        ligand_layers=4,
        cross_layers=3,
        latent_width=768,
        pair_width=192,
        pair_factor=16,
        sequence_pair_layers=2,
        sequence_top_pairs=128,
        structure_self_layers=2,
        structure_pair_layers=2,
        structure_token_layers=2,
        structure_top_pairs=128,
        maximum_protein_tokens=512,
        dataset_count=3,
        adapter_width=192,
        dropout=0.10,
    ):
        super().__init__()
        self.dataset_count = int(dataset_count)
        self.protein_encoder = Protein1DEncoder(
            protein_input_width, width, heads, protein_layers, maximum_protein_tokens, dropout
        )
        self.ligand_encoder = Ligand2DEncoder(
            width, heads, ligand_layers, continuous_features=4, bond_features=7, dropout=dropout
        )
        self.cross_blocks = nn.ModuleList(
            BidirectionalCrossBlock(width, heads, dropout) for _ in range(int(cross_layers))
        )
        self.protein_pool = AttentionPool(width)
        self.ligand_pool = AttentionPool(width)
        self.base_latent = nn.Sequential(
            nn.LayerNorm(4 * int(width)),
            nn.Linear(4 * int(width), 2 * int(width)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * int(width), int(latent_width)),
            nn.LayerNorm(int(latent_width)),
            nn.SiLU(),
        )
        self.sequence_pair_encoder = SequenceLigandPairEncoder(
            width, pair_width, pair_factor, heads, sequence_pair_layers, sequence_top_pairs, dropout
        )
        self.base_pair_adapter = nn.Sequential(
            nn.LayerNorm(int(pair_width)),
            nn.Linear(int(pair_width), int(adapter_width)),
            nn.SiLU(),
            nn.Linear(int(adapter_width), int(latent_width)),
        )
        nn.init.zeros_(self.base_pair_adapter[-1].weight)
        nn.init.zeros_(self.base_pair_adapter[-1].bias)
        self.geometry_quality_head = nn.Sequential(
            nn.LayerNorm(int(pair_width)), nn.Linear(int(pair_width), 3), nn.Sigmoid()
        )
        self.task_adapters = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(int(latent_width)),
                nn.Linear(int(latent_width), int(adapter_width)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(adapter_width), int(latent_width)),
            )
            for _ in range(self.dataset_count)
        )
        for adapter in self.task_adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        self.base_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(int(latent_width), int(width)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(width), 1),
            )
            for _ in range(self.dataset_count)
        )
        self.relative_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(int(latent_width)),
                nn.Linear(int(latent_width), int(adapter_width)),
                nn.SiLU(),
                nn.Linear(int(adapter_width), 1),
            )
            for _ in range(self.dataset_count)
        )
        self.structure_encoder = MaskedGeometryPairEncoder(
            width,
            pair_width,
            heads,
            structure_self_layers,
            structure_pair_layers,
            structure_token_layers,
            structure_top_pairs,
            latent_width,
            dropout,
        )
        self.pair_cross = CrossAttention(pair_width, heads, dropout)
        self.pair_cross_pool = AttentionPool(pair_width)
        geometry_width = int(pair_width) + int(latent_width) + 3
        self.geometry_adapter = nn.Sequential(
            nn.LayerNorm(geometry_width),
            nn.Linear(geometry_width, int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(latent_width)),
        )
        nn.init.zeros_(self.geometry_adapter[-1].weight)
        nn.init.zeros_(self.geometry_adapter[-1].bias)
        gate_width = 3 * int(latent_width) + 3
        self.structure_gate = nn.Sequential(
            nn.LayerNorm(gate_width),
            nn.Linear(gate_width, int(width)),
            nn.SiLU(),
            nn.Linear(int(width), 1),
        )
        nn.init.zeros_(self.structure_gate[-1].weight)
        nn.init.zeros_(self.structure_gate[-1].bias)

    def _dataset_head(self, latent, dataset_index, heads):
        if dataset_index.min() < 0 or dataset_index.max() >= self.dataset_count:
            raise ValueError("dataset index is outside configured heads")
        output = latent.new_zeros(latent.shape[0])
        for index, head in enumerate(heads):
            selected = dataset_index == index
            if selected.any():
                value = head(latent[selected]).squeeze(-1).to(dtype=output.dtype)
                output = output.masked_scatter(selected, value)
        return output

    def _dataset_adapter(self, latent, dataset_index):
        output = latent
        for index, adapter in enumerate(self.task_adapters):
            selected = dataset_index == index
            if selected.any():
                indices = torch.nonzero(selected, as_tuple=False).squeeze(-1)
                adapted = latent[selected] + adapter(latent[selected]).to(dtype=latent.dtype)
                output = output.index_copy(0, indices, adapted)
        return output

    @staticmethod
    def _required_structure_batch(batch, indices):
        keys = (
            "pocket_esm",
            "pocket_scalar",
            "pocket_aa",
            "pocket_coords",
            "pocket_mask",
            "structure_ligand_categorical",
            "structure_ligand_continuous",
            "structure_ligand_coords",
            "structure_ligand_mask",
            "residue_ligand_distance",
        )
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError("3D samples require fields: %s" % ", ".join(missing))
        result = {key: batch[key].index_select(0, indices) for key in keys}
        if "geometry_visible" in batch:
            result["geometry_visible"] = batch["geometry_visible"].index_select(0, indices)
        return result

    def relative_difference(self, left, right, dataset_index):
        difference = left - right
        reverse = -difference
        forward_value = self._dataset_head(difference, dataset_index, self.relative_heads)
        reverse_value = self._dataset_head(reverse, dataset_index, self.relative_heads)
        return 0.5 * (forward_value - reverse_value)

    def forward(self, batch: Dict[str, torch.Tensor], force_no_3d=False):
        protein_mask = batch["protein_mask"].bool()
        ligand_mask = batch["ligand_mask"].bool()
        dataset_index = batch["dataset_index"].long()
        protein = self.protein_encoder(batch["protein_embeddings"], protein_mask)
        ligand = self.ligand_encoder(
            batch["ligand_categorical"],
            batch["ligand_continuous"],
            batch["bond_features"],
            ligand_mask,
        )
        for block in self.cross_blocks:
            protein, ligand = block(protein, ligand, protein_mask, ligand_mask)
        protein_pool = self.protein_pool(protein, protein_mask)
        ligand_pool = self.ligand_pool(ligand, ligand_mask)
        global_latent = self.base_latent(
            torch.cat(
                (protein_pool, ligand_pool, protein_pool * ligand_pool, torch.abs(protein_pool - ligand_pool)),
                dim=-1,
            )
        )
        sequence_pair = self.sequence_pair_encoder(protein, ligand, protein_mask, ligand_mask)
        base_latent = global_latent + self.base_pair_adapter(sequence_pair.latent).to(global_latent.dtype)
        base_latent = self._dataset_adapter(base_latent, dataset_index)
        base_prediction = self._dataset_head(base_latent, dataset_index, self.base_heads)
        quality_prediction = self.geometry_quality_head(sequence_pair.latent).to(base_latent.dtype)

        requested = batch.get("has_3d")
        if requested is None:
            requested = torch.zeros(base_prediction.shape[0], dtype=torch.bool, device=base_prediction.device)
        effective = requested.bool() & (not bool(force_no_3d))
        full_latent = base_latent
        full_pair_latent = sequence_pair.latent
        structure_latent = base_latent.new_zeros(base_latent.shape)
        quality_target = quality_prediction.new_zeros(quality_prediction.shape)
        gate = base_prediction.new_zeros(base_prediction.shape)
        contact_logits = None
        distance_logits = None
        if effective.any():
            indices = torch.nonzero(effective, as_tuple=False).squeeze(-1)
            encoded = self.structure_encoder(self._required_structure_batch(batch, indices))
            sequence_tokens = sequence_pair.tokens.index_select(0, indices)
            sequence_mask = sequence_pair.token_mask.index_select(0, indices)
            fused_tokens = self.pair_cross(
                sequence_tokens,
                encoded.tokens,
                sequence_mask,
                encoded.token_mask,
            )
            fused_pair = self.pair_cross_pool(fused_tokens, sequence_mask)
            full_pair_latent = full_pair_latent.index_copy(
                0, indices, fused_pair.to(full_pair_latent.dtype)
            )
            structure_latent = structure_latent.index_copy(
                0, indices, encoded.latent.to(structure_latent.dtype)
            )
            quality_target = quality_target.index_copy(
                0, indices, encoded.quality.to(quality_target.dtype)
            )
            geometry_context = torch.cat((fused_pair, encoded.latent, encoded.quality), dim=-1)
            geometry_update = self.geometry_adapter(geometry_context).to(base_latent.dtype)
            selected_base = base_latent.index_select(0, indices)
            gate_context = torch.cat(
                (
                    selected_base,
                    encoded.latent.to(selected_base.dtype),
                    encoded.latent.to(selected_base.dtype) - selected_base,
                    encoded.quality.to(selected_base.dtype),
                ),
                dim=-1,
            )
            selected_gate = 2.0 * torch.sigmoid(self.structure_gate(gate_context).squeeze(-1).float())
            selected_gate = selected_gate.to(base_latent.dtype)
            selected_full = selected_base + selected_gate[:, None] * geometry_update
            full_latent = full_latent.index_copy(0, indices, selected_full)
            gate = gate.index_copy(0, indices, selected_gate.to(gate.dtype))
            contact_shape = batch["residue_ligand_distance"].shape
            contact_logits = base_latent.new_zeros(contact_shape).index_copy(
                0, indices, encoded.contact_logits.to(base_latent.dtype)
            )
            distance_logits = base_latent.new_zeros(contact_shape + (DISTANCE_BINS,)).index_copy(
                0, indices, encoded.distance_logits.to(base_latent.dtype)
            )
        prediction = self._dataset_head(full_latent, dataset_index, self.base_heads)
        delta = (prediction - base_prediction) * effective.to(prediction.dtype)
        return OptionalStructurePrivilegedOutput(
            prediction=prediction,
            base_prediction=base_prediction,
            base_latent=base_latent,
            full_latent=full_latent,
            sequence_pair_latent=sequence_pair.latent,
            full_pair_latent=full_pair_latent,
            sequence_contact_logits=sequence_pair.contact_logits,
            geometry_quality_prediction=quality_prediction,
            geometry_quality_target=quality_target,
            structure_latent=structure_latent,
            structure_gate=gate,
            structure_delta=delta,
            effective_has_3d=effective,
            contact_logits=contact_logits,
            distance_logits=distance_logits,
        )


__all__ = [
    "DISTANCE_BINS",
    "DISTANCE_BOUNDARIES",
    "MaskedGeometryPairEncoder",
    "OptionalStructurePrivilegedOutput",
    "OptionalStructurePrivilegedPairDTA",
    "SequenceLigandPairEncoder",
]
