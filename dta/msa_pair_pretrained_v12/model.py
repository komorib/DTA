"""MSA encoder with masked-token and residue-distogram pretraining heads."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from dta.msa_pair_expert_v11.model import (
    MSAAxialBlock,
    MSAPairExpertV11,
    _masked_mean,
    _rbf,
)


MSA_MASK_INDEX = 23
MSA_VOCABULARY_SIZE = 24
DISTOGRAM_BOUNDARIES = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0)
DISTOGRAM_BINS = len(DISTOGRAM_BOUNDARIES) + 1


def residue_distogram_targets(batch):
    distances = torch.cdist(batch["pocket_coords"], batch["pocket_coords"])
    boundaries = torch.as_tensor(
        DISTOGRAM_BOUNDARIES, dtype=distances.dtype, device=distances.device
    )
    targets = torch.bucketize(distances, boundaries)
    chain = batch["pocket_chain"]
    position = batch["pocket_position"]
    pocket_mask = batch["pocket_mask"]
    same_chain = (chain[:, :, None] == chain[:, None, :]) & (chain[:, :, None] >= 0)
    separated = torch.abs(position[:, :, None] - position[:, None, :]) >= 3
    upper = torch.triu(torch.ones_like(same_chain, dtype=torch.bool), diagonal=1)
    mask = (
        same_chain
        & separated
        & upper
        & pocket_mask[:, :, None]
        & pocket_mask[:, None, :]
    )
    return targets, mask, distances


@dataclass(frozen=True)
class EvolutionaryFeaturesV12:
    single: torch.Tensor
    pair: torch.Tensor
    pair_summary: torch.Tensor
    residue_contact_logits: torch.Tensor
    distogram_logits: torch.Tensor
    row_states: torch.Tensor


class PocketMSAToPairV12(nn.Module):
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
        self.token_embedding = nn.Embedding(
            MSA_VOCABULARY_SIZE, self.width, padding_idx=22
        )
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
        self.distogram_head = nn.Linear(self.pair_width, DISTOGRAM_BINS)
        nn.init.normal_(self.query_row_embedding, std=0.02)

    def forward(self, batch):
        tokens = batch["pocket_msa_tokens"].clamp(0, MSA_MASK_INDEX)
        row_mask = batch["pocket_msa_row_mask"]
        states = self.token_embedding(tokens)
        states = states + self.insertion_embedding(
            batch["pocket_msa_insertions"][..., None]
        )
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
        return EvolutionaryFeaturesV12(
            single=single,
            pair=pair,
            pair_summary=pair_summary,
            residue_contact_logits=self.residue_contact(pair).squeeze(-1),
            distogram_logits=self.distogram_head(pair),
            row_states=states,
        )


@dataclass(frozen=True)
class MSAPairOutputV12:
    prediction: torch.Tensor
    latent: torch.Tensor
    contact_logits: torch.Tensor
    residue_contact_logits: torch.Tensor
    distogram_logits: torch.Tensor
    affinity_class_logits: torch.Tensor
    msa_gate: torch.Tensor


class MSAPairExpertV12(MSAPairExpertV11):
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
        super().__init__(
            model_width=model_width,
            layers=layers,
            heads=heads,
            dropout=dropout,
            latent_width=latent_width,
            distance_bins=distance_bins,
            msa_width=msa_width,
            msa_layers=msa_layers,
            msa_heads=msa_heads,
            pair_factor=pair_factor,
            pair_width=pair_width,
        )
        self.msa_pair_encoder = PocketMSAToPairV12(
            width=msa_width,
            layers=msa_layers,
            heads=msa_heads,
            pair_factor=pair_factor,
            pair_width=pair_width,
            dropout=dropout,
        )

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
        latent = self.latent_head(
            torch.cat((global_state, pocket_pool, ligand_pool), dim=-1)
        )
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
        return MSAPairOutputV12(
            prediction=prediction,
            latent=latent,
            contact_logits=contact_logits,
            residue_contact_logits=evolutionary.residue_contact_logits,
            distogram_logits=evolutionary.distogram_logits,
            affinity_class_logits=affinity_class_logits,
            msa_gate=gate,
        )


__all__ = [
    "DISTOGRAM_BINS",
    "DISTOGRAM_BOUNDARIES",
    "MSA_MASK_INDEX",
    "MSAPairExpertV12",
    "MSAPairOutputV12",
    "PocketMSAToPairV12",
    "residue_distogram_targets",
]
