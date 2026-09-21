"""Geometry-biased joint Transformer for MSA-aware protein-ligand affinity."""

from dataclasses import dataclass
import math

import torch
from torch import nn


def _rbf(distances, centers, width):
    return torch.exp(-torch.square((distances[..., None] - centers) / width))


def _masked_mean(value, mask):
    weight = mask.to(value.dtype)[..., None]
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class GeometryAttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.norm1 = nn.LayerNorm(self.width)
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.output = nn.Linear(self.width, self.width)
        self.dropout = nn.Dropout(dropout)
        self.pair_scale = nn.Parameter(torch.ones(self.heads))
        self.norm2 = nn.LayerNorm(self.width)
        self.feedforward = nn.Sequential(
            nn.Linear(self.width, 4 * self.width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.width, self.width),
        )

    def forward(self, states, pair_bias, mask):
        batch, token_count, _ = states.shape
        normalized = self.norm1(states)
        qkv = self.qkv(normalized).reshape(
            batch, token_count, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_width)
        scores = scores + pair_bias * self.pair_scale[None, :, None, None]
        scores = scores.masked_fill(~mask[:, None, None, :], -1e4)
        attention = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        attention = self.dropout(attention)
        update = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, token_count, self.width
        )
        states = states + self.dropout(self.output(update))
        states = states + self.dropout(self.feedforward(self.norm2(states)))
        return states * mask[..., None].to(states.dtype)


@dataclass(frozen=True)
class MSAPocketOutput:
    prediction: torch.Tensor
    latent: torch.Tensor
    contact_logits: torch.Tensor
    msa_gate: torch.Tensor


class MSAPocketExpertV9(nn.Module):
    def __init__(
        self,
        model_width=384,
        layers=6,
        heads=8,
        dropout=0.10,
        latent_width=1024,
        distance_bins=32,
    ):
        super().__init__()
        self.model_width = int(model_width)
        self.heads = int(heads)
        self.distance_bins = int(distance_bins)
        if self.model_width % self.heads:
            raise ValueError("model width must be divisible by heads")
        self.msa_adapter = nn.Sequential(
            nn.LayerNorm(50),
            nn.Linear(50, self.model_width),
            nn.SiLU(),
            nn.Linear(self.model_width, self.model_width),
        )
        self.esm_adapter = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, 2 * self.model_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.model_width, self.model_width),
        )
        self.scalar_adapter = nn.Sequential(
            nn.LayerNorm(15),
            nn.Linear(15, self.model_width),
            nn.SiLU(),
        )
        self.aa_embedding = nn.Embedding(21, self.model_width)
        self.msa_gate = nn.Sequential(
            nn.Linear(3, self.model_width),
            nn.Sigmoid(),
        )
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
            nn.Linear(7, 2 * self.heads),
            nn.SiLU(),
            nn.Linear(2 * self.heads, self.heads),
        )
        self.sequence_bias = nn.Embedding(34, self.heads)
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
        nn.init.normal_(self.global_token, std=0.02)

    def _tokens(self, batch):
        msa = batch["pocket_msa"]
        confidence = torch.stack((msa[..., 43], msa[..., 44], msa[..., 49]), dim=-1)
        gate = self.msa_gate(confidence)
        residue = self.esm_adapter(batch["pocket_esm"])
        residue = residue + gate * self.msa_adapter(msa)
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
                torch.full((batch_size, ligand_count), 2, dtype=torch.long, device=states.device),
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
        return states * mask[..., None], mask, token_types, gate

    def _pair_bias(self, batch, token_types):
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
        residue_distance = torch.cdist(batch["pocket_coords"], batch["pocket_coords"])
        ligand_distance = torch.cdist(batch["ligand_coords"], batch["ligand_coords"])
        distances[:, 1 : 1 + pocket_count, 1 : 1 + pocket_count] = residue_distance
        distances[:, 1 + pocket_count :, 1 + pocket_count :] = ligand_distance
        cross = batch["residue_ligand_distance"]
        distances[:, 1 : 1 + pocket_count, 1 + pocket_count :] = cross
        distances[:, 1 + pocket_count :, 1 : 1 + pocket_count] = cross.transpose(1, 2)
        distance_rbf = _rbf(distances, self.distance_centers, self.distance_width)
        bias = bias + self.distance_bias(distance_rbf).permute(0, 3, 1, 2)
        ligand_bond_bias = self.bond_bias(batch["bond_features"]).permute(0, 3, 1, 2)
        bias[:, :, 1 + pocket_count :, 1 + pocket_count :] += ligand_bond_bias
        chain = batch["pocket_chain"]
        position = batch["pocket_position"]
        same_chain = chain[:, :, None] == chain[:, None, :]
        separation = torch.abs(position[:, :, None] - position[:, None, :]).clamp(0, 32)
        separation = torch.where(same_chain, separation, torch.full_like(separation, 33))
        sequence_bias = self.sequence_bias(separation).permute(0, 3, 1, 2)
        bias[:, :, 1 : 1 + pocket_count, 1 : 1 + pocket_count] += sequence_bias
        return bias

    def forward(self, batch):
        states, mask, token_types, gate = self._tokens(batch)
        pair_bias = self._pair_bias(batch, token_types)
        for block in self.blocks:
            states = block(states, pair_bias, mask)
        states = self.final_norm(states)
        pocket_count = batch["pocket_mask"].shape[1]
        global_state = states[:, 0]
        pocket_state = states[:, 1 : 1 + pocket_count]
        ligand_state = states[:, 1 + pocket_count :]
        pocket_pool = _masked_mean(pocket_state, batch["pocket_mask"])
        ligand_pool = _masked_mean(ligand_state, batch["ligand_mask"])
        latent = self.latent_head(torch.cat((global_state, pocket_pool, ligand_pool), dim=-1))
        prediction = self.affinity_head(latent).squeeze(-1)
        contact_logits = torch.einsum(
            "bpd,bld->bpl",
            self.contact_residue(pocket_state),
            self.contact_ligand(ligand_state),
        ) / math.sqrt(128.0)
        contact_rbf = _rbf(
            batch["residue_ligand_distance"], self.distance_centers, self.distance_width
        )
        contact_logits = contact_logits + self.contact_distance(contact_rbf).squeeze(-1)
        return MSAPocketOutput(
            prediction=prediction,
            latent=latent,
            contact_logits=contact_logits,
            msa_gate=gate,
        )


__all__ = ["MSAPocketExpertV9", "MSAPocketOutput"]
