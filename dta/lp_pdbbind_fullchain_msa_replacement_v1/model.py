"""Use complete-chain MSA Transformer features, never pocket-aligned MSA rows.

The frozen upstream encoder has already seen contiguous 512-column windows of
the complete alignment. Only after that encoding are the resulting residue
features gathered to the structural pocket for ligand interaction.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dta.lp_pdbbind_e2e_msa_structure_v3.model import E2EMSAInterfacePairFusionV3
from dta.msa_interface_pair_pooling_v18.model import (
    ContactAwarePairTokenPoolV18,
    EvolutionaryFeaturesV13,
    PocketMSAInterfaceV13,
)
from dta.msa_pair_expert_v11.model import _masked_mean, _rbf

from . import VERSION


OLD_MSA_KEYS = frozenset((
    "pocket_msa", "pocket_msa_tokens", "pocket_msa_insertions",
    "pocket_msa_row_features", "pocket_msa_row_mask",
))


class FullChainEvolutionaryEncoderV1(PocketMSAInterfaceV13):
    """Retain transferable pair/interface heads; replace the pocket MSA trunk."""

    def __init__(self, *args, fullchain_width=768,
                 input_key="fullchain_msa_pocket", **kwargs):
        super().__init__(*args, **kwargs)
        self.fullchain_width = int(fullchain_width)
        self.input_key = str(input_key)
        if self.fullchain_width <= 0:
            raise ValueError("fullchain_width must be positive")
        self.fullchain_input = nn.Sequential(
            nn.LayerNorm(self.fullchain_width),
            nn.Linear(self.fullchain_width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )
        # The old token/row embeddings and pocket-column axial attention must
        # not survive as an alternate input path or as unused trainable weights.
        for name in (
            "token_embedding", "insertion_embedding", "row_feature_embedding",
            "query_row_embedding", "relative_position_bias", "blocks", "final_norm",
        ):
            delattr(self, name)

    def forward(self, batch, interface_message=None, relation_inputs=None):
        features = batch.get(self.input_key)
        pocket_mask = batch["pocket_mask"]
        if features is None or features.shape != (*pocket_mask.shape, self.fullchain_width):
            raise ValueError("full-chain residue features are missing or misaligned")
        if features.device != pocket_mask.device or not torch.isfinite(features).all():
            raise ValueError("full-chain residue features have invalid device or values")
        single = self.fullchain_input(features.float())
        single = single * pocket_mask[..., None].to(single.dtype)
        states = single[:, None]
        row_mask = pocket_mask[:, None]

        chain = batch["pocket_chain"]
        same_chain = (
            (chain[:, :, None] == chain[:, None, :])
            & (chain[:, :, None] >= 0)
            & pocket_mask[:, :, None]
            & pocket_mask[:, None, :]
        )
        left = self.outer_left(states) * row_mask[..., None].to(states.dtype)
        right = self.outer_right(states) * row_mask[..., None].to(states.dtype)
        outer = torch.einsum("bdpa,bdqf->bpqaf", left, right)
        counts = torch.einsum(
            "bdp,bdq->bpq", row_mask.to(states.dtype), row_mask.to(states.dtype)
        ).clamp_min(1.0)
        pair = self.pair_projection((outer / counts[..., None, None]).flatten(start_dim=-2))
        pair = pair * same_chain[..., None].to(pair.dtype)
        pair = (pair + self.pair_transition(pair)) * same_chain[..., None].to(pair.dtype)
        pair_summary = _masked_mean(pair, same_chain, dimension=2)

        ligand_categorical = batch["ligand_categorical"]
        atomic_number = ligand_categorical[..., 0].long().clamp(0, 127)
        ligand_numeric = torch.cat(
            (ligand_categorical[..., 1:] / 8.0, batch["ligand_continuous"]), dim=-1
        )
        ligand = self.ligand_norm(
            self.atomic_number_embedding(atomic_number) + self.ligand_adapter(ligand_numeric)
        )
        bond_mask = batch["bond_features"].abs().sum(dim=-1) > 0
        ligand = ligand + _masked_mean(
            self.bond_adapter(batch["bond_features"]), bond_mask, dimension=2
        )
        interface_mask = (
            batch["contact_mask"] & pocket_mask[:, :, None]
            & batch["ligand_mask"][:, None, :]
        )
        visible = batch.get("interface_distance_visible", interface_mask)
        if visible.shape != interface_mask.shape:
            raise ValueError("interface distance visibility has the wrong shape")
        visible = visible & interface_mask
        distance_rbf = _rbf(
            batch["residue_ligand_distance"],
            self.interface_distance_centers, self.interface_distance_width,
        )
        distance_features = torch.cat(
            (
                distance_rbf * visible[..., None].to(distance_rbf.dtype),
                visible[..., None].to(distance_rbf.dtype),
            ), dim=-1,
        )
        residue_match = F.normalize(
            self.match_residue(_masked_mean(single, pocket_mask, dimension=1)), dim=-1
        )
        ligand_match = F.normalize(
            self.match_ligand(_masked_mean(ligand, batch["ligand_mask"], dimension=1)),
            dim=-1,
        )
        match_logits = torch.matmul(residue_match, ligand_match.transpose(0, 1)) / 0.07
        interface_residue = single
        interface_pair = None
        for block in self.interface_layers:
            if interface_message is None:
                interface_residue, ligand, interface_pair = block(
                    interface_residue, ligand, distance_features, interface_mask
                )
            else:
                interface_residue, ligand, interface_pair = block(
                    interface_residue, ligand, distance_features, interface_mask,
                    interface_message=interface_message, relation_inputs=relation_inputs,
                )
        interface_summary = _masked_mean(interface_pair, interface_mask, dimension=2)
        ligand_summary = _masked_mean(interface_pair, interface_mask, dimension=1)
        return EvolutionaryFeaturesV13(
            single=single,
            pair=pair,
            pair_summary=pair_summary,
            interface_pair=interface_pair,
            interface_summary=interface_summary,
            ligand_summary=ligand_summary,
            interface_distance_logits=self.interface_distance_head(interface_pair),
            interface_contact_logits=self.interface_contact_head(interface_pair).squeeze(-1),
            residue_contact_logits=self.residue_contact(pair).squeeze(-1),
            distogram_logits=self.distogram_head(pair),
            match_logits=match_logits,
            row_states=states,
        )


class SequencePairTokenPoolV1(ContactAwarePairTokenPoolV18):
    """Use query amino-acid identity, not a pocket-MSA query row."""

    def forward(self, evolutionary, batch):
        # The complex cache numbers canonical amino acids 1..20 (0=unknown),
        # while the transferred V18 residue embedding expects 0..19
        # (20=unknown). Their amino-acid ordering is otherwise identical.
        amino_acids = batch["pocket_aa"]
        if torch.any((amino_acids < 0) | (amino_acids > 20)):
            raise ValueError("pocket amino-acid identifier is outside 0..20")
        msa_tokens = torch.where(
            amino_acids > 0, amino_acids - 1, torch.full_like(amino_acids, 20)
        )
        proxy = dict(batch)
        proxy["pocket_msa_tokens"] = msa_tokens[:, None, :]
        return super().forward(evolutionary, proxy)


class FullChainMSAReplacementV1(E2EMSAInterfacePairFusionV3):
    """V3 geometry/ligand trunk with every pocket-MSA input path removed."""

    version = VERSION

    def __init__(self, *args, fullchain_width=768, **kwargs):
        super().__init__(*args, **kwargs)
        msa_width = int(kwargs.get("msa_width", 128))
        pair_width = int(kwargs.get("pair_width", 64))
        dropout = float(kwargs.get("dropout", 0.10))
        self.msa_pair_encoder = FullChainEvolutionaryEncoderV1(
            width=msa_width,
            layers=int(kwargs.get("msa_layers", 4)),
            heads=int(kwargs.get("msa_heads", 4)),
            pair_factor=int(kwargs.get("pair_factor", 8)),
            pair_width=pair_width,
            dropout=dropout,
            interface_blocks=int(kwargs.get("interface_blocks", 2)),
            fullchain_width=fullchain_width,
        )
        self.pair_token_pool = SequencePairTokenPoolV1(
            pair_width=pair_width,
            hidden_width=int(kwargs.get("pair_token_width", 256)),
            heads=int(kwargs.get("pair_token_heads", 8)),
            layers=int(kwargs.get("pair_token_layers", 2)),
            token_count=int(kwargs.get("pair_token_count", 96)),
            dropout=dropout,
        )
        del self.profile_adapter
        del self.msa_gate

    def _tokens(self, batch, evolutionary):
        # Corresponds to V11's token path, except no 50-dim pocket-MSA profile
        # or profile-derived gate is read at any point.
        residue = self.esm_adapter(batch["pocket_esm"])
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
        ligand = self.ligand_norm(
            self.atomic_number_embedding(atomic_number) + self.ligand_adapter(ligand_numeric)
        )
        ligand = self._augment_ligand_tokens(ligand, batch)
        batch_size = residue.shape[0]
        states = torch.cat((self.global_token.expand(batch_size, -1, -1), residue, ligand), dim=1)
        pocket_count, ligand_count = residue.shape[1], ligand.shape[1]
        token_types = torch.cat((
            torch.zeros((batch_size, 1), dtype=torch.long, device=states.device),
            torch.ones((batch_size, pocket_count), dtype=torch.long, device=states.device),
            torch.full((batch_size, ligand_count), 2, dtype=torch.long, device=states.device),
        ), dim=1)
        states = states + self.token_type_embedding(token_types)
        mask = torch.cat((
            torch.ones((batch_size, 1), dtype=torch.bool, device=states.device),
            batch["pocket_mask"], batch["ligand_mask"],
        ), dim=1)
        gate = torch.ones_like(residue)
        return states * mask[..., None].to(states.dtype), mask, token_types, gate

    def _augment_ligand_tokens(self, ligand, batch):
        """Identity by default; extensions run before token concatenation/context attention."""
        return ligand

    def forward(self, batch):
        forbidden = OLD_MSA_KEYS.intersection(batch)
        if forbidden:
            raise ValueError("old pocket-MSA inputs are forbidden: %s" % sorted(forbidden))
        return super().forward(batch)


def initialize_from_stage1_state(model_state, model_kwargs):
    """Transfer compatible v3 weights; never restore the removed MSA modules."""

    model = FullChainMSAReplacementV1(**model_kwargs)
    expected = model.state_dict()
    transfer = {
        key: value for key, value in model_state.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    loaded = model.load_state_dict(transfer, strict=False)
    expected_new = {key for key in expected if key.startswith("msa_pair_encoder.fullchain_input.")}
    if set(loaded.missing_keys) != expected_new or loaded.unexpected_keys:
        raise ValueError("full-chain replacement has unexpected missing pretrained weights")
    final = model.fusion_head.network[-1]
    if torch.count_nonzero(final.weight).item() or torch.count_nonzero(final.bias).item():
        raise ValueError("Stage 1 residual fusion head is not zero")
    model.freeze_fusion_head()
    return model, {
        "transferred_key_count": len(transfer),
        "new_keys": sorted(expected_new),
        "removed_old_keys": sorted(set(model_state) - set(transfer)),
    }


__all__ = [
    "FullChainEvolutionaryEncoderV1", "FullChainMSAReplacementV1",
    "SequencePairTokenPoolV1", "initialize_from_stage1_state",
]
