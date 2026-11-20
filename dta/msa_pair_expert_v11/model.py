"""AF3-inspired pocket MSA-to-pair expert for protein-ligand affinity."""

from dataclasses import dataclass
import math

import torch
from torch import nn


def _rbf(distances, centers, width):
    return torch.exp(-torch.square((distances[..., None] - centers) / width))


def _masked_mean(value, mask, dimension):
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum(dim=dimension) / weight.sum(dim=dimension).clamp_min(1.0)


class AttentionUpdate(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        if int(width) % int(heads):
            raise ValueError("attention width must be divisible by heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.norm = nn.LayerNorm(self.width)
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.output = nn.Linear(self.width, self.width)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, states, mask, pair_mask=None, pair_bias=None):
        batch, count, _ = states.shape
        qkv = self.qkv(self.norm(states)).reshape(
            batch, count, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_width)
        if pair_bias is not None:
            scores = scores + pair_bias
        allowed = mask[:, None, None, :]
        if pair_mask is not None:
            allowed = allowed & pair_mask[:, None, :, :]
        scores = scores.masked_fill(~allowed, -1e4)
        attention = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        attention = self.dropout(attention)
        update = torch.matmul(attention, value).transpose(1, 2).reshape(batch, count, self.width)
        states = states + self.dropout(self.output(update))
        return states * mask[..., None].to(states.dtype)


class MSAAxialBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.row_attention = AttentionUpdate(width, heads, dropout)
        self.column_attention = AttentionUpdate(width, heads, dropout)
        self.transition_norm = nn.LayerNorm(int(width))
        self.transition = nn.Sequential(
            nn.Linear(int(width), 4 * int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * int(width), int(width)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, states, row_mask, same_chain, row_pair_bias):
        batch, depth, pocket, width = states.shape
        row_states = states.reshape(batch * depth, pocket, width)
        flat_mask = row_mask.reshape(batch * depth, pocket)
        flat_pair_mask = same_chain[:, None].expand(-1, depth, -1, -1).reshape(
            batch * depth, pocket, pocket
        )
        flat_pair_bias = row_pair_bias[:, None].expand(-1, depth, -1, -1, -1).reshape(
            batch * depth, row_pair_bias.shape[1], pocket, pocket
        )
        row_states = self.row_attention(
            row_states,
            flat_mask,
            pair_mask=flat_pair_mask,
            pair_bias=flat_pair_bias,
        )
        states = row_states.reshape(batch, depth, pocket, width)
        column_states = states.permute(0, 2, 1, 3).reshape(batch * pocket, depth, width)
        column_mask = row_mask.permute(0, 2, 1).reshape(batch * pocket, depth)
        column_states = self.column_attention(column_states, column_mask)
        states = column_states.reshape(batch, pocket, depth, width).permute(0, 2, 1, 3)
        states = states + self.dropout(self.transition(self.transition_norm(states)))
        return states * row_mask[..., None].to(states.dtype)


@dataclass(frozen=True)
class EvolutionaryFeatures:
    single: torch.Tensor
    pair: torch.Tensor
    pair_summary: torch.Tensor
    residue_contact_logits: torch.Tensor


class PocketMSAToPair(nn.Module):
    def __init__(
        self,
        width=128,
        layers=4,
        heads=4,
        pair_factor=8,
        pair_width=64,
        dropout=0.10,
    ):
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.pair_factor = int(pair_factor)
        self.pair_width = int(pair_width)
        self.token_embedding = nn.Embedding(23, self.width, padding_idx=22)
        self.insertion_embedding = nn.Sequential(nn.Linear(1, self.width), nn.SiLU())
        self.row_feature_embedding = nn.Sequential(
            nn.Linear(2, self.width), nn.SiLU(), nn.Linear(self.width, self.width)
        )
        self.query_row_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.width))
        self.relative_position_bias = nn.Embedding(66, self.heads)
        self.blocks = nn.ModuleList(
            MSAAxialBlock(self.width, self.heads, dropout) for _ in range(int(layers))
        )
        self.final_norm = nn.LayerNorm(self.width)
        self.outer_left = nn.Linear(self.width, self.pair_factor)
        self.outer_right = nn.Linear(self.width, self.pair_factor)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(self.pair_factor * self.pair_factor),
            nn.Linear(self.pair_factor * self.pair_factor, 2 * self.pair_width),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.pair_width, self.pair_width),
        )
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(self.pair_width),
            nn.Linear(self.pair_width, 2 * self.pair_width),
            nn.SiLU(),
            nn.Linear(2 * self.pair_width, self.pair_width),
        )
        self.residue_contact = nn.Linear(self.pair_width, 1)
        nn.init.normal_(self.query_row_embedding, std=0.02)

    def forward(self, batch):
        tokens = batch["pocket_msa_tokens"].clamp(0, 22)
        row_mask = batch["pocket_msa_row_mask"]
        states = self.token_embedding(tokens)
        states = states + self.insertion_embedding(batch["pocket_msa_insertions"][..., None])
        states = states + self.row_feature_embedding(batch["pocket_msa_row_features"])
        states[:, :1] = states[:, :1] + self.query_row_embedding
        states = states * row_mask[..., None].to(states.dtype)

        pocket_mask = batch["pocket_mask"]
        chain = batch["pocket_chain"]
        same_chain = (
            (chain[:, :, None] == chain[:, None, :])
            & (chain[:, :, None] >= 0)
            & pocket_mask[:, :, None]
            & pocket_mask[:, None, :]
        )
        position = batch["pocket_position"]
        separation = torch.abs(position[:, :, None] - position[:, None, :]).clamp(0, 64)
        separation = torch.where(same_chain, separation, torch.full_like(separation, 65))
        row_pair_bias = self.relative_position_bias(separation).permute(0, 3, 1, 2)
        for block in self.blocks:
            states = block(states, row_mask, same_chain, row_pair_bias)
        states = self.final_norm(states)
        single = _masked_mean(states, row_mask, dimension=1)

        left = self.outer_left(states) * row_mask[..., None].to(states.dtype)
        right = self.outer_right(states) * row_mask[..., None].to(states.dtype)
        outer = torch.einsum("bdpa,bdqf->bpqaf", left, right)
        counts = torch.einsum(
            "bdp,bdq->bpq", row_mask.to(states.dtype), row_mask.to(states.dtype)
        ).clamp_min(1.0)
        outer = outer / counts[..., None, None]
        pair = self.pair_projection(outer.flatten(start_dim=-2))
        pair = pair * same_chain[..., None].to(pair.dtype)
        pair = pair + self.pair_transition(pair)
        pair = pair * same_chain[..., None].to(pair.dtype)
        pair_summary = _masked_mean(pair, same_chain, dimension=2)
        residue_contact_logits = self.residue_contact(pair).squeeze(-1)
        return EvolutionaryFeatures(
            single=single,
            pair=pair,
            pair_summary=pair_summary,
            residue_contact_logits=residue_contact_logits,
        )


class GeometryAttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attention = AttentionUpdate(width, heads, dropout)
        self.pair_scale = nn.Parameter(torch.ones(int(heads)))
        self.norm = nn.LayerNorm(int(width))
        self.feedforward = nn.Sequential(
            nn.Linear(int(width), 4 * int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * int(width), int(width)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, states, pair_bias, mask):
        states = self.attention(
            states,
            mask,
            pair_bias=pair_bias * self.pair_scale[None, :, None, None],
        )
        states = states + self.dropout(self.feedforward(self.norm(states)))
        return states * mask[..., None].to(states.dtype)


@dataclass(frozen=True)
class MSAPairOutput:
    prediction: torch.Tensor
    latent: torch.Tensor
    contact_logits: torch.Tensor
    residue_contact_logits: torch.Tensor
    affinity_class_logits: torch.Tensor
    msa_gate: torch.Tensor


class MSAPairExpertV11(nn.Module):
    def __init__(
        self,
        model_width=384,
        layers=6,
        heads=8,
        dropout=0.10,
        latent_width=1024,
        distance_bins=32,
        msa_width=128,
        msa_layers=4,
        msa_heads=4,
        pair_factor=8,
        pair_width=64,
    ):
        super().__init__()
        self.model_width = int(model_width)
        self.heads = int(heads)
        self.distance_bins = int(distance_bins)
        self.pair_width = int(pair_width)
        if self.model_width % self.heads:
            raise ValueError("model width must be divisible by heads")
        self.msa_pair_encoder = PocketMSAToPair(
            width=msa_width,
            layers=msa_layers,
            heads=msa_heads,
            pair_factor=pair_factor,
            pair_width=pair_width,
            dropout=dropout,
        )
        self.profile_adapter = nn.Sequential(
            nn.LayerNorm(50),
            nn.Linear(50, self.model_width),
            nn.SiLU(),
            nn.Linear(self.model_width, self.model_width),
        )
        self.msa_single_adapter = nn.Sequential(
            nn.LayerNorm(int(msa_width)), nn.Linear(int(msa_width), self.model_width), nn.SiLU()
        )
        self.pair_summary_adapter = nn.Sequential(
            nn.LayerNorm(self.pair_width), nn.Linear(self.pair_width, self.model_width), nn.SiLU()
        )
        self.esm_adapter = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, 2 * self.model_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.model_width, self.model_width),
        )
        self.scalar_adapter = nn.Sequential(
            nn.LayerNorm(15), nn.Linear(15, self.model_width), nn.SiLU()
        )
        self.aa_embedding = nn.Embedding(21, self.model_width)
        self.msa_gate = nn.Sequential(nn.Linear(3, self.model_width), nn.Sigmoid())
        self.residue_norm = nn.LayerNorm(self.model_width)
        self.atomic_number_embedding = nn.Embedding(128, self.model_width)
        self.ligand_adapter = nn.Sequential(
            nn.LayerNorm(20),
            nn.Linear(20, 2 * self.model_width),
            nn.SiLU(),
            nn.Linear(2 * self.model_width, self.model_width),
        )
        self.ligand_norm = nn.LayerNorm(self.model_width)
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.model_width))
        self.token_type_embedding = nn.Embedding(3, self.model_width)
        self.pair_type_bias = nn.Embedding(9, self.heads)
        self.distance_bias = nn.Sequential(
            nn.Linear(self.distance_bins, 2 * self.heads),
            nn.SiLU(),
            nn.Linear(2 * self.heads, self.heads),
        )
        self.bond_bias = nn.Sequential(
            nn.Linear(7, 2 * self.heads), nn.SiLU(), nn.Linear(2 * self.heads, self.heads)
        )
        self.sequence_bias = nn.Embedding(34, self.heads)
        self.evolutionary_pair_bias = nn.Linear(self.pair_width, self.heads, bias=False)
        self.evolutionary_summary_bias = nn.Linear(self.pair_width, self.heads, bias=False)
        self.evolutionary_cross_gate = nn.Sequential(
            nn.Linear(self.distance_bins, self.heads), nn.Sigmoid()
        )
        centers = torch.linspace(0.0, 24.0, self.distance_bins)
        self.register_buffer("distance_centers", centers, persistent=True)
        self.distance_width = float(centers[1] - centers[0])
        self.blocks = nn.ModuleList(
            GeometryAttentionBlock(self.model_width, self.heads, dropout)
            for _ in range(int(layers))
        )
        self.final_norm = nn.LayerNorm(self.model_width)
        self.contact_residue = nn.Linear(self.model_width, 128)
        self.contact_ligand = nn.Linear(self.model_width, 128)
        self.contact_distance = nn.Sequential(
            nn.Linear(self.distance_bins, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.latent_head = nn.Sequential(
            nn.Linear(3 * self.model_width, 2 * self.model_width),
            nn.LayerNorm(2 * self.model_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.model_width, int(latent_width)),
            nn.LayerNorm(int(latent_width)),
            nn.SiLU(),
        )
        self.affinity_head = nn.Sequential(
            nn.Linear(int(latent_width), self.model_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_width, 1),
        )
        self.affinity_class_head = nn.Sequential(
            nn.Linear(int(latent_width), self.model_width // 2),
            nn.SiLU(),
            nn.Linear(self.model_width // 2, 1),
        )
        nn.init.normal_(self.global_token, std=0.02)

    def _tokens(self, batch, evolutionary):
        profile = batch["pocket_msa"]
        confidence = torch.stack((profile[..., 43], profile[..., 44], profile[..., 49]), dim=-1)
        gate = self.msa_gate(confidence)
        residue = self.esm_adapter(batch["pocket_esm"])
        residue = residue + gate * self.profile_adapter(profile)
        residue = residue + self.msa_single_adapter(evolutionary.single)
        residue = residue + self.pair_summary_adapter(evolutionary.pair_summary)
        residue = residue + self.scalar_adapter(batch["pocket_scalar"])
        residue = residue + self.aa_embedding(batch["pocket_aa"].clamp(0, 20))
        residue = self.residue_norm(residue)
        ligand_categorical = batch["ligand_categorical"]
        atomic_number = ligand_categorical[..., 0].long().clamp(0, 127)
        ligand_numeric = torch.cat(
            (ligand_categorical[..., 1:] / 8.0, batch["ligand_continuous"]), dim=-1
        )
        ligand = self.atomic_number_embedding(atomic_number) + self.ligand_adapter(ligand_numeric)
        ligand = self.ligand_norm(ligand)
        batch_size = residue.shape[0]
        global_token = self.global_token.expand(batch_size, -1, -1)
        states = torch.cat((global_token, residue, ligand), dim=1)
        pocket_count, ligand_count = residue.shape[1], ligand.shape[1]
        token_types = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.long, device=states.device),
                torch.ones((batch_size, pocket_count), dtype=torch.long, device=states.device),
                torch.full(
                    (batch_size, ligand_count), 2, dtype=torch.long, device=states.device
                ),
            ),
            dim=1,
        )
        states = states + self.token_type_embedding(token_types)
        mask = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.bool, device=states.device),
                batch["pocket_mask"],
                batch["ligand_mask"],
            ),
            dim=1,
        )
        return states * mask[..., None].to(states.dtype), mask, token_types, gate

    def _pair_bias(self, batch, token_types, evolutionary):
        batch_size = token_types.shape[0]
        pocket_count = batch["pocket_coords"].shape[1]
        ligand_count = batch["ligand_coords"].shape[1]
        token_count = 1 + pocket_count + ligand_count
        pair_type = token_types[:, :, None] * 3 + token_types[:, None, :]
        bias = self.pair_type_bias(pair_type).permute(0, 3, 1, 2)
        distances = torch.zeros(
            (batch_size, token_count, token_count),
            dtype=batch["pocket_coords"].dtype,
            device=batch["pocket_coords"].device,
        )
        distances[:, 1 : 1 + pocket_count, 1 : 1 + pocket_count] = torch.cdist(
            batch["pocket_coords"], batch["pocket_coords"]
        )
        distances[:, 1 + pocket_count :, 1 + pocket_count :] = torch.cdist(
            batch["ligand_coords"], batch["ligand_coords"]
        )
        cross = batch["residue_ligand_distance"]
        distances[:, 1 : 1 + pocket_count, 1 + pocket_count :] = cross
        distances[:, 1 + pocket_count :, 1 : 1 + pocket_count] = cross.transpose(1, 2)
        distance_rbf = _rbf(distances, self.distance_centers, self.distance_width)
        bias = bias + self.distance_bias(distance_rbf).permute(0, 3, 1, 2)
        ligand_bond_bias = self.bond_bias(batch["bond_features"]).permute(0, 3, 1, 2)
        bias[:, :, 1 + pocket_count :, 1 + pocket_count :] += ligand_bond_bias
        chain = batch["pocket_chain"]
        position = batch["pocket_position"]
        same_chain = (chain[:, :, None] == chain[:, None, :]) & (chain[:, :, None] >= 0)
        separation = torch.abs(position[:, :, None] - position[:, None, :]).clamp(0, 32)
        separation = torch.where(same_chain, separation, torch.full_like(separation, 33))
        bias[:, :, 1 : 1 + pocket_count, 1 : 1 + pocket_count] += self.sequence_bias(
            separation
        ).permute(0, 3, 1, 2)
        bias[:, :, 1 : 1 + pocket_count, 1 : 1 + pocket_count] += self.evolutionary_pair_bias(
            evolutionary.pair
        ).permute(0, 3, 1, 2)
        cross_rbf = _rbf(cross, self.distance_centers, self.distance_width)
        cross_gate = self.evolutionary_cross_gate(cross_rbf)
        summary = self.evolutionary_summary_bias(evolutionary.pair_summary)
        evolutionary_cross = (summary[:, :, None, :] * cross_gate).permute(0, 3, 1, 2)
        bias[:, :, 1 : 1 + pocket_count, 1 + pocket_count :] += evolutionary_cross
        bias[:, :, 1 + pocket_count :, 1 : 1 + pocket_count] += evolutionary_cross.transpose(2, 3)
        return bias

    def forward(self, batch):
        evolutionary = self.msa_pair_encoder(batch)
        states, mask, token_types, gate = self._tokens(batch, evolutionary)
        pair_bias = self._pair_bias(batch, token_types, evolutionary)
        for block in self.blocks:
            states = block(states, pair_bias, mask)
        states = self.final_norm(states)
        pocket_count = batch["pocket_mask"].shape[1]
        global_state = states[:, 0]
        pocket_state = states[:, 1 : 1 + pocket_count]
        ligand_state = states[:, 1 + pocket_count :]
        pocket_pool = _masked_mean(pocket_state, batch["pocket_mask"], dimension=1)
        ligand_pool = _masked_mean(ligand_state, batch["ligand_mask"], dimension=1)
        latent = self.latent_head(torch.cat((global_state, pocket_pool, ligand_pool), dim=-1))
        prediction = self.affinity_head(latent).squeeze(-1)
        affinity_class_logits = self.affinity_class_head(latent).squeeze(-1)
        contact_logits = torch.einsum(
            "bpd,bld->bpl",
            self.contact_residue(pocket_state),
            self.contact_ligand(ligand_state),
        ) / math.sqrt(128.0)
        contact_rbf = _rbf(
            batch["residue_ligand_distance"], self.distance_centers, self.distance_width
        )
        contact_logits = contact_logits + self.contact_distance(contact_rbf).squeeze(-1)
        return MSAPairOutput(
            prediction=prediction,
            latent=latent,
            contact_logits=contact_logits,
            residue_contact_logits=evolutionary.residue_contact_logits,
            affinity_class_logits=affinity_class_logits,
            msa_gate=gate,
        )


__all__ = ["MSAPairExpertV11", "MSAPairOutput", "PocketMSAToPair"]
