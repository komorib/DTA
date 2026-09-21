"""Coordinate-free 1D+2D trunk with a hard-routed invariant 3D expert."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


DISTANCE_BINS = 16
DISTANCE_BOUNDARIES = tuple(float(value) for value in torch.linspace(2.5, 17.5, DISTANCE_BINS - 1))
RBF_BINS = 32


def _masked_mean(values, mask, dimension):
    weights = mask.to(values.dtype)
    denominator = weights.sum(dim=dimension, keepdim=True).clamp_min(1.0)
    return (values * weights[..., None]).sum(dim=dimension) / denominator.squeeze(dimension)[..., None]


def _rbf(distances, centers, width):
    return torch.exp(-((distances[..., None] - centers) / width) ** 2)


def _masked_softmax(logits, mask, dimension=-1):
    masked = logits.masked_fill(~mask, -1.0e4)
    weights = torch.softmax(masked, dim=dimension)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dimension, keepdim=True).clamp_min(1.0e-8)


class FeedForward(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.layers = nn.Sequential(
            nn.Linear(int(width), 4 * int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * int(width), int(width)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, values):
        return values + self.layers(self.norm(values))


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, width, heads, pair_features, dropout):
        super().__init__()
        if int(width) % int(heads):
            raise ValueError("attention width must be divisible by heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.norm = nn.LayerNorm(self.width)
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.pair_bias = nn.Linear(int(pair_features), self.heads, bias=False)
        self.output = nn.Linear(self.width, self.width)
        self.dropout = nn.Dropout(float(dropout))
        self.feed_forward = FeedForward(self.width, dropout)

    def forward(self, values, pair_features, mask):
        batch, count, _ = values.shape
        normalized = self.norm(values)
        qkv = self.qkv(normalized).reshape(batch, count, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        logits = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(self.head_width)
        logits = logits + self.pair_bias(pair_features).permute(0, 3, 1, 2)
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        weights = _masked_softmax(logits, pair_mask, dimension=-1)
        attended = torch.einsum("bhij,bjhd->bihd", weights, value).reshape(batch, count, self.width)
        values = values + self.dropout(self.output(attended))
        values = values * mask[..., None].to(values.dtype)
        return self.feed_forward(values) * mask[..., None].to(values.dtype)


class CrossAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        if int(width) % int(heads):
            raise ValueError("cross-attention width must be divisible by heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.query_norm = nn.LayerNorm(self.width)
        self.context_norm = nn.LayerNorm(self.width)
        self.query = nn.Linear(self.width, self.width)
        self.key = nn.Linear(self.width, self.width)
        self.value = nn.Linear(self.width, self.width)
        self.output = nn.Linear(self.width, self.width)
        self.dropout = nn.Dropout(float(dropout))
        self.feed_forward = FeedForward(self.width, dropout)

    def forward(self, query_values, context_values, query_mask, context_mask):
        batch, query_count, _ = query_values.shape
        context_count = context_values.shape[1]
        query = self.query(self.query_norm(query_values)).reshape(
            batch, query_count, self.heads, self.head_width
        )
        context = self.context_norm(context_values)
        key = self.key(context).reshape(batch, context_count, self.heads, self.head_width)
        value = self.value(context).reshape(batch, context_count, self.heads, self.head_width)
        logits = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(self.head_width)
        mask = query_mask[:, None, :, None] & context_mask[:, None, None, :]
        weights = _masked_softmax(logits, mask, dimension=-1)
        attended = torch.einsum("bhij,bjhd->bihd", weights, value).reshape(
            batch, query_count, self.width
        )
        query_values = query_values + self.dropout(self.output(attended))
        query_values = query_values * query_mask[..., None].to(query_values.dtype)
        return self.feed_forward(query_values) * query_mask[..., None].to(query_values.dtype)


class BidirectionalCrossBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.protein_to_ligand = CrossAttention(width, heads, dropout)
        self.ligand_to_protein = CrossAttention(width, heads, dropout)

    def forward(self, protein, ligand, protein_mask, ligand_mask):
        updated_protein = self.protein_to_ligand(protein, ligand, protein_mask, ligand_mask)
        updated_ligand = self.ligand_to_protein(ligand, updated_protein, ligand_mask, protein_mask)
        return updated_protein, updated_ligand


class AttentionPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(int(width)), nn.Linear(int(width), 1))

    def forward(self, values, mask):
        logits = self.score(values).squeeze(-1)
        weights = _masked_softmax(logits, mask, dimension=-1)
        return torch.sum(values * weights[..., None], dim=1)


class Ligand2DEncoder(nn.Module):
    """Atom/bond encoder that never reads conformer coordinates."""

    CATEGORY_SIZES = (128, 9, 17, 10, 2, 10, 2, 9)

    def __init__(self, width, heads, layers, continuous_features=4, bond_features=7, dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, int(width)) for size in self.CATEGORY_SIZES
        )
        self.continuous = nn.Sequential(
            nn.LayerNorm(int(continuous_features)),
            nn.Linear(int(continuous_features), int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(width)),
        )
        self.norm = nn.LayerNorm(int(width))
        self.blocks = nn.ModuleList(
            PairBiasedSelfAttention(width, heads, bond_features, dropout)
            for _ in range(int(layers))
        )

    def forward(self, categorical, continuous, bond_features, mask):
        states = self.continuous(continuous)
        for index, embedding in enumerate(self.embeddings):
            states = states + embedding(categorical[..., index].long().clamp(0, embedding.num_embeddings - 1))
        states = self.norm(states) * mask[..., None].to(states.dtype)
        for block in self.blocks:
            states = block(states, bond_features, mask)
        return states


class Protein1DEncoder(nn.Module):
    def __init__(self, input_width, width, heads, layers, maximum_tokens, dropout):
        super().__init__()
        self.maximum_tokens = int(maximum_tokens)
        self.adapter = nn.Sequential(
            nn.LayerNorm(int(input_width)),
            nn.Linear(int(input_width), int(width)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(width)),
        )
        self.position = nn.Embedding(self.maximum_tokens, int(width))
        layer = nn.TransformerEncoderLayer(
            d_model=int(width),
            nhead=int(heads),
            dim_feedforward=4 * int(width),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.norm = nn.LayerNorm(int(width))

    def forward(self, embeddings, mask):
        if embeddings.shape[1] > self.maximum_tokens:
            raise ValueError("protein token count exceeds configured maximum")
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        states = self.adapter(embeddings) + self.position(positions)[None]
        states = self.encoder(states, src_key_padding_mask=~mask)
        return self.norm(states) * mask[..., None].to(states.dtype)


class PairMLPBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = FeedForward(width, dropout)

    def forward(self, pair, mask):
        return self.layers(pair) * mask[..., None].to(pair.dtype)


@dataclass
class StructureEncoding:
    latent: torch.Tensor
    contact_logits: torch.Tensor
    distance_logits: torch.Tensor
    quality: torch.Tensor


class Optional3DEncoder(nn.Module):
    """SE(3)-invariant pocket/ligand encoder; coordinates enter only through distances."""

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
            nn.LayerNorm(19), nn.Linear(19, int(width)), nn.SiLU(), nn.Linear(int(width), int(width))
        )
        self.ligand_continuous = nn.Sequential(
            nn.LayerNorm(2), nn.Linear(2, int(width)), nn.SiLU(), nn.Linear(int(width), int(width))
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
            nn.Linear(RBF_BINS, int(pair_width)), nn.SiLU(), nn.Linear(int(pair_width), int(pair_width))
        )
        self.pair_blocks = nn.ModuleList(
            PairMLPBlock(pair_width, dropout) for _ in range(int(pair_layers))
        )
        self.contact_head = nn.Linear(int(pair_width), 1)
        self.distance_head = nn.Linear(int(pair_width), DISTANCE_BINS)
        token_layer = nn.TransformerEncoderLayer(
            d_model=int(pair_width),
            nhead=int(heads),
            dim_feedforward=4 * int(pair_width),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=int(token_layers))
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
        pocket_mask = batch["pocket_mask"]
        ligand_mask = batch["structure_ligand_mask"]
        pocket = self.pocket(torch.cat((batch["pocket_esm"], batch["pocket_scalar"]), dim=-1))
        pocket = pocket + self.pocket_aa(batch["pocket_aa"].long().clamp(0, 20))
        ligand = self.ligand_categorical(batch["structure_ligand_categorical"] / 8.0)
        ligand = ligand + self.ligand_continuous(batch["structure_ligand_continuous"])
        pocket_distance = torch.cdist(batch["pocket_coords"], batch["pocket_coords"])
        ligand_distance = torch.cdist(batch["structure_ligand_coords"], batch["structure_ligand_coords"])
        pocket_rbf = _rbf(pocket_distance, self.distance_centers, self.distance_width)
        ligand_rbf = _rbf(ligand_distance, self.distance_centers, self.distance_width)
        for block in self.pocket_blocks:
            pocket = block(pocket, pocket_rbf, pocket_mask)
        for block in self.ligand_blocks:
            ligand = block(ligand, ligand_rbf, ligand_mask)

        pair_mask = pocket_mask[:, :, None] & ligand_mask[:, None, :]
        cross_distance = batch["residue_ligand_distance"]
        cross_rbf = _rbf(cross_distance, self.distance_centers, self.distance_width)
        pair = (
            self.pocket_pair(pocket)[:, :, None, :]
            + self.ligand_pair(ligand)[:, None, :, :]
            + self.distance_pair(cross_rbf)
        )
        pair = torch.tanh(pair) * pair_mask[..., None].to(pair.dtype)
        for block in self.pair_blocks:
            pair = block(pair, pair_mask)
        contact_logits = self.contact_head(pair).squeeze(-1)
        distance_logits = self.distance_head(pair)

        batch_size, pocket_count, ligand_count, pair_width = pair.shape
        flat_pair = pair.reshape(batch_size, pocket_count * ligand_count, pair_width)
        flat_distance = torch.where(
            pair_mask,
            cross_distance,
            torch.full_like(cross_distance, float("inf")),
        ).reshape(batch_size, pocket_count * ligand_count)
        token_count = min(self.top_pairs, flat_pair.shape[1])
        selected_distance, selected_index = torch.topk(
            flat_distance, k=token_count, dim=1, largest=False, sorted=True
        )
        selected_mask = torch.isfinite(selected_distance)
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
        return StructureEncoding(latent, contact_logits, distance_logits, quality)


@dataclass
class OptionalStructureOutput:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    base_latent: torch.Tensor
    structure_latent: torch.Tensor
    structure_gate: torch.Tensor
    structure_delta: torch.Tensor
    effective_has_3d: torch.Tensor
    contact_logits: Optional[torch.Tensor]
    distance_logits: Optional[torch.Tensor]


class OptionalStructureDTA(nn.Module):
    """One checkpoint for sequence/2D-only and sequence/2D/3D affinity inference."""

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
        structure_self_layers=2,
        structure_pair_layers=2,
        structure_token_layers=2,
        structure_top_pairs=128,
        maximum_protein_tokens=512,
        dataset_count=3,
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
        self.base_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(int(latent_width), int(width)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(width), 1),
            )
            for _ in range(self.dataset_count)
        )
        self.structure_encoder = Optional3DEncoder(
            width=width,
            pair_width=pair_width,
            heads=heads,
            self_layers=structure_self_layers,
            pair_layers=structure_pair_layers,
            token_layers=structure_token_layers,
            top_pairs=structure_top_pairs,
            latent_width=latent_width,
            dropout=dropout,
        )
        fusion_width = 3 * int(latent_width) + 3
        self.structure_gate = nn.Sequential(
            nn.LayerNorm(fusion_width),
            nn.Linear(fusion_width, int(width)),
            nn.SiLU(),
            nn.Linear(int(width), 1),
        )
        self.delta_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(fusion_width),
                nn.Linear(fusion_width, int(width)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(width), 1),
            )
            for _ in range(self.dataset_count)
        )
        for head in self.delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _dataset_head(self, latent, dataset_index, heads):
        if dataset_index.min() < 0 or dataset_index.max() >= self.dataset_count:
            raise ValueError("dataset index is outside configured heads")
        output = latent.new_zeros(latent.shape[0])
        for index, head in enumerate(heads):
            selected = dataset_index == index
            if selected.any():
                prediction = head(latent[selected]).squeeze(-1).to(dtype=output.dtype)
                output = output.masked_scatter(selected, prediction)
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
        return {key: batch[key].index_select(0, indices) for key in keys}

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
        base_latent = self.base_latent(
            torch.cat(
                (protein_pool, ligand_pool, protein_pool * ligand_pool, torch.abs(protein_pool - ligand_pool)),
                dim=-1,
            )
        )
        base_prediction = self._dataset_head(base_latent, dataset_index, self.base_heads)

        requested = batch.get("has_3d")
        if requested is None:
            requested = torch.zeros(base_prediction.shape[0], dtype=torch.bool, device=base_prediction.device)
        effective = requested.bool() & (not bool(force_no_3d))
        structure_latent = base_latent.new_zeros(base_latent.shape)
        gate = base_prediction.new_zeros(base_prediction.shape)
        delta = base_prediction.new_zeros(base_prediction.shape)
        contact_logits = None
        distance_logits = None
        if effective.any():
            indices = torch.nonzero(effective, as_tuple=False).squeeze(-1)
            structure_batch = self._required_structure_batch(batch, indices)
            encoded = self.structure_encoder(structure_batch)
            structure_latent = structure_latent.index_copy(
                0, indices, encoded.latent.to(dtype=structure_latent.dtype)
            )
            quality = base_latent.new_zeros((base_latent.shape[0], 3)).index_copy(
                0, indices, encoded.quality.to(dtype=base_latent.dtype)
            )
            fusion = torch.cat(
                (base_latent, structure_latent, structure_latent - base_latent, quality), dim=-1
            )
            raw_gate = torch.sigmoid(self.structure_gate(fusion).squeeze(-1))
            raw_delta = self._dataset_head(fusion, dataset_index, self.delta_heads)
            gate = raw_gate * effective.to(raw_gate.dtype)
            delta = raw_delta * effective.to(raw_delta.dtype)
            contact_shape = batch["residue_ligand_distance"].shape
            contact_logits = base_latent.new_zeros(contact_shape).index_copy(
                0, indices, encoded.contact_logits.to(dtype=base_latent.dtype)
            )
            distance_logits = base_latent.new_zeros(contact_shape + (DISTANCE_BINS,)).index_copy(
                0, indices, encoded.distance_logits.to(dtype=base_latent.dtype)
            )
        prediction = base_prediction + gate * delta
        return OptionalStructureOutput(
            prediction=prediction,
            base_prediction=base_prediction,
            base_latent=base_latent,
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
    "OptionalStructureDTA",
    "OptionalStructureOutput",
]
