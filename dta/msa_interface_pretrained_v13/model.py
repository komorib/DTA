"""V13 MSA encoder with masked residue-ligand interface pretraining heads."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

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
INTERFACE_DISTANCE_BOUNDARIES = (3.0, 4.5, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0)
INTERFACE_DISTANCE_BINS = len(INTERFACE_DISTANCE_BOUNDARIES) + 1
INTERFACE_RBF_BINS = 16


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


def interface_distance_targets(batch):
    distances = batch["residue_ligand_distance"]
    boundaries = torch.as_tensor(
        INTERFACE_DISTANCE_BOUNDARIES,
        dtype=distances.dtype,
        device=distances.device,
    )
    targets = torch.bucketize(distances, boundaries)
    mask = (
        batch["contact_mask"]
        & batch["pocket_mask"][:, :, None]
        & batch["ligand_mask"][:, None, :]
    )
    return targets, mask, distances


@dataclass(frozen=True)
class EvolutionaryFeaturesV13:
    single: torch.Tensor
    pair: torch.Tensor
    pair_summary: torch.Tensor
    interface_pair: torch.Tensor
    interface_summary: torch.Tensor
    ligand_summary: torch.Tensor
    interface_distance_logits: torch.Tensor
    interface_contact_logits: torch.Tensor
    residue_contact_logits: torch.Tensor
    match_logits: torch.Tensor
    distogram_logits: torch.Tensor
    row_states: torch.Tensor



class InterfacePairBlockV13(nn.Module):
    def __init__(self, width, pair_width, distance_features, dropout):
        super().__init__()
        self.residue_norm = nn.LayerNorm(int(width))
        self.ligand_norm = nn.LayerNorm(int(width))
        self.residue_pair = nn.Linear(int(width), int(pair_width), bias=False)
        self.ligand_pair = nn.Linear(int(width), int(pair_width), bias=False)
        self.distance_pair = nn.Sequential(
            nn.Linear(int(distance_features), int(pair_width)),
            nn.SiLU(),
            nn.Linear(int(pair_width), int(pair_width)),
        )
        self.transition = nn.Sequential(
            nn.LayerNorm(int(pair_width)),
            nn.Linear(int(pair_width), 2 * int(pair_width)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * int(pair_width), int(pair_width)),
        )
        self.residue_update = nn.Linear(int(pair_width), int(width))
        self.ligand_update = nn.Linear(int(pair_width), int(width))
        self.residue_gate = nn.Sequential(nn.Linear(int(width), int(width)), nn.Sigmoid())
        self.ligand_gate = nn.Sequential(nn.Linear(int(width), int(width)), nn.Sigmoid())

    def forward(self, residue, ligand, distance_features, pair_mask):
        pair_mask_float = pair_mask[..., None].to(residue.dtype)
        pair = (
            self.residue_pair(self.residue_norm(residue))[:, :, None, :]
            + self.ligand_pair(self.ligand_norm(ligand))[:, None, :, :]
            + self.distance_pair(distance_features)
        )
        pair = pair + self.transition(pair)
        pair = pair * pair_mask_float
        residue_message = _masked_mean(pair, pair_mask, dimension=2)
        ligand_message = _masked_mean(pair, pair_mask, dimension=1)
        residue = residue + self.residue_gate(residue) * self.residue_update(residue_message)
        ligand = ligand + self.ligand_gate(ligand) * self.ligand_update(ligand_message)
        return residue, ligand, pair

class PocketMSAInterfaceV13(nn.Module):
    def __init__(
        self,
        width=128,
        layers=4,
        heads=4,
        pair_factor=8,
        pair_width=64,
        dropout=0.10,
        interface_blocks=2,
    ):
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.pair_factor = int(pair_factor)
        self.pair_width = int(pair_width)
        self.interface_block_count = int(interface_blocks)
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
        self.atomic_number_embedding = nn.Embedding(128, self.width)
        self.ligand_adapter = nn.Sequential(
            nn.LayerNorm(20),
            nn.Linear(20, 2 * self.width),
            nn.SiLU(),
            nn.Linear(2 * self.width, self.width),
        )
        self.ligand_norm = nn.LayerNorm(self.width)
        self.bond_adapter = nn.Sequential(
            nn.Linear(7, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )
        interface_centers = torch.linspace(0.0, 24.0, INTERFACE_RBF_BINS)
        self.register_buffer(
            "interface_distance_centers", interface_centers, persistent=True
        )
        self.interface_distance_width = float(
            interface_centers[1] - interface_centers[0]
        )
        self.interface_layers = nn.ModuleList(
            InterfacePairBlockV13(
                self.width,
                self.pair_width,
                INTERFACE_RBF_BINS + 1,
                dropout,
            )
            for _ in range(self.interface_block_count)
        )
        self.interface_distance_head = nn.Linear(
            self.pair_width, INTERFACE_DISTANCE_BINS
        )
        self.interface_contact_head = nn.Linear(self.pair_width, 1)
        self.match_residue = nn.Sequential(
            nn.LayerNorm(self.width), nn.Linear(self.width, self.pair_width)
        )
        self.match_ligand = nn.Sequential(
            nn.LayerNorm(self.width), nn.Linear(self.width, self.pair_width)
        )
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
        ligand_categorical = batch["ligand_categorical"]
        atomic_number = ligand_categorical[..., 0].long().clamp(0, 127)
        ligand_numeric = torch.cat(
            (ligand_categorical[..., 1:] / 8.0, batch["ligand_continuous"]), dim=-1
        )
        ligand = self.atomic_number_embedding(atomic_number) + self.ligand_adapter(
            ligand_numeric
        )
        ligand = self.ligand_norm(ligand)
        bond_mask = batch["bond_features"].abs().sum(dim=-1) > 0
        bond_state = self.bond_adapter(batch["bond_features"])
        ligand = ligand + _masked_mean(bond_state, bond_mask, dimension=2)

        interface_mask = (
            batch["contact_mask"]
            & pocket_mask[:, :, None]
            & batch["ligand_mask"][:, None, :]
        )
        visible = batch.get("interface_distance_visible", interface_mask)
        if visible.shape != interface_mask.shape:
            raise ValueError("interface distance visibility has the wrong shape")
        visible = visible & interface_mask
        distance_rbf = _rbf(
            batch["residue_ligand_distance"],
            self.interface_distance_centers,
            self.interface_distance_width,
        )
        distance_features = torch.cat(
            (
                distance_rbf * visible[..., None].to(distance_rbf.dtype),
                visible[..., None].to(distance_rbf.dtype),
            ),
            dim=-1,
        )
        residue_match = F.normalize(
            self.match_residue(_masked_mean(single, pocket_mask, dimension=1)), dim=-1
        )
        ligand_match = F.normalize(
            self.match_ligand(
                _masked_mean(ligand, batch["ligand_mask"], dimension=1)
            ),
            dim=-1,
        )
        match_logits = torch.matmul(residue_match, ligand_match.transpose(0, 1)) / 0.07
        interface_residue = single
        interface_pair = None
        for block in self.interface_layers:
            interface_residue, ligand, interface_pair = block(
                interface_residue, ligand, distance_features, interface_mask
            )
        interface_summary = _masked_mean(
            interface_pair, interface_mask, dimension=2
        )
        ligand_summary = _masked_mean(interface_pair, interface_mask, dimension=1)
        return EvolutionaryFeaturesV13(
            single=single,
            interface_pair=interface_pair,
            interface_summary=interface_summary,
            ligand_summary=ligand_summary,
            interface_distance_logits=self.interface_distance_head(interface_pair),
            interface_contact_logits=self.interface_contact_head(interface_pair).squeeze(-1),
            match_logits=match_logits,
            pair=pair,
            pair_summary=pair_summary,
            residue_contact_logits=self.residue_contact(pair).squeeze(-1),
            distogram_logits=self.distogram_head(pair),
            row_states=states,
        )


@dataclass(frozen=True)
class MSAInterfaceOutputV13:
    prediction: torch.Tensor
    latent: torch.Tensor
    contact_logits: torch.Tensor
    residue_contact_logits: torch.Tensor
    interface_distance_logits: torch.Tensor
    interface_contact_logits: torch.Tensor
    match_logits: torch.Tensor
    distogram_logits: torch.Tensor
    affinity_class_logits: torch.Tensor
    msa_gate: torch.Tensor


class MSAInterfaceExpertV13(MSAPairExpertV11):
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
        interface_blocks=2,
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
        self.msa_pair_encoder = PocketMSAInterfaceV13(
            width=msa_width,
            layers=msa_layers,
            heads=msa_heads,
            pair_factor=pair_factor,
            pair_width=pair_width,
            dropout=dropout,
            interface_blocks=interface_blocks,
        )
        self.interface_residue_adapter = nn.Sequential(
            nn.LayerNorm(int(pair_width)), nn.Linear(int(pair_width), int(model_width))
        )
        self.interface_ligand_adapter = nn.Sequential(
            nn.LayerNorm(int(pair_width)), nn.Linear(int(pair_width), int(model_width))
        )
        self.interface_pair_bias = nn.Linear(int(pair_width), int(heads), bias=False)
        nn.init.zeros_(self.interface_residue_adapter[-1].weight)
        nn.init.zeros_(self.interface_residue_adapter[-1].bias)
        nn.init.zeros_(self.interface_ligand_adapter[-1].weight)
        nn.init.zeros_(self.interface_ligand_adapter[-1].bias)
        nn.init.zeros_(self.interface_pair_bias.weight)

    def forward(self, batch):
        evolutionary = self.msa_pair_encoder(batch)
        states, mask, token_types, gate = self._tokens(batch, evolutionary)
        pocket_count = batch["pocket_mask"].shape[1]
        states = torch.cat(
            (
                states[:, :1],
                states[:, 1 : 1 + pocket_count]
                + self.interface_residue_adapter(evolutionary.interface_summary),
                states[:, 1 + pocket_count :]
                + self.interface_ligand_adapter(evolutionary.ligand_summary),
            ),
            dim=1,
        )
        pair_bias = self._pair_bias(batch, token_types, evolutionary).clone()
        interface_bias = self.interface_pair_bias(
            evolutionary.interface_pair
        ).permute(0, 3, 1, 2)
        pair_bias[:, :, 1 : 1 + pocket_count, 1 + pocket_count :] += interface_bias
        pair_bias[:, :, 1 + pocket_count :, 1 : 1 + pocket_count] += (
            interface_bias.transpose(2, 3)
        )
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
        contact_logits = (
            contact_logits
            + self.contact_distance(contact_rbf).squeeze(-1)
            + evolutionary.interface_contact_logits
        )
        return MSAInterfaceOutputV13(
            prediction=prediction,
            latent=latent,
            contact_logits=contact_logits,
            residue_contact_logits=evolutionary.residue_contact_logits,
            distogram_logits=evolutionary.distogram_logits,
            interface_distance_logits=evolutionary.interface_distance_logits,
            interface_contact_logits=evolutionary.interface_contact_logits,
            match_logits=evolutionary.match_logits,
            affinity_class_logits=affinity_class_logits,
            msa_gate=gate,
        )


__all__ = [
    "DISTOGRAM_BINS",
    "DISTOGRAM_BOUNDARIES",
    "INTERFACE_DISTANCE_BINS",
    "INTERFACE_DISTANCE_BOUNDARIES",
    "MSA_MASK_INDEX",
    "MSAInterfaceExpertV13",
    "MSAInterfaceOutputV13",
    "PocketMSAInterfaceV13",
    "interface_distance_targets",
    "residue_distogram_targets",
]
