"""V18 pair-token backbone with an in-graph, bounded spatial residual."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from dta.msa_interface_pair_pooling_v18.model import (
    MSAInterfaceOutputV13,
    MSAInterfacePairPoolExpertV18,
)

from . import VERSION


SPATIAL_SHELL_BOUNDARIES = (4.5, 6.0, 8.0, 12.0, 24.0)
SPATIAL_FEATURES_PER_SHELL = 5
SPATIAL_FEATURE_WIDTH = (len(SPATIAL_SHELL_BOUNDARIES) + 1) * SPATIAL_FEATURES_PER_SHELL


@dataclass(frozen=True)
class E2EMSAInterfaceOutputV3(MSAInterfaceOutputV13):
    """V18-compatible task output plus the vector used by the fusion head."""

    fusion_features: torch.Tensor


class ResidualFusionHeadV3(nn.Module):
    """Small bounded residual head; zero initialization preserves its anchor."""

    def __init__(self, input_width, hidden_width=256, dropout=0.10, max_delta=1.0):
        super().__init__()
        if int(input_width) <= 0 or int(hidden_width) <= 0:
            raise ValueError("fusion dimensions must be positive")
        if float(max_delta) <= 0.0:
            raise ValueError("max_delta must be positive")
        self.max_delta = float(max_delta)
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_width)),
            nn.Linear(int(input_width), int(hidden_width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_width), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.max_delta * torch.tanh(self.network(features).squeeze(-1))


def _masked_mean_std(values, mask):
    weights = mask.to(values.dtype)
    count = weights.sum(dim=(1, 2)).clamp_min(1.0)
    mean = (values * weights).sum(dim=(1, 2)) / count
    variance = ((values - mean[:, None, None]).square() * weights).sum(
        dim=(1, 2)
    ) / count
    return mean, torch.sqrt(variance.clamp_min(0.0) + 1e-8)


def spatial_summary_v3(output, batch):
    """Differentiable, multiscale contact/distance summary from the same pass."""

    pair_mask = (
        batch["contact_mask"]
        & batch["pocket_mask"][:, :, None]
        & batch["ligand_mask"][:, None, :]
    )
    distances = batch["residue_ligand_distance"].float()
    probabilities = F.softmax(output.interface_distance_logits.float(), dim=-1)
    centers = torch.linspace(
        0.0,
        1.0,
        probabilities.shape[-1],
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    expected_distance = (probabilities * centers).sum(dim=-1)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1)
    entropy = entropy / torch.log(
        probabilities.new_tensor(float(probabilities.shape[-1]))
    ).clamp_min(1e-8)
    contact = torch.sigmoid(output.interface_contact_logits.float())
    denominator = pair_mask.to(distances.dtype).sum(dim=(1, 2)).clamp_min(1.0)

    boundaries = distances.new_tensor(SPATIAL_SHELL_BOUNDARIES)
    shell_ids = torch.bucketize(distances.contiguous(), boundaries)
    summaries = []
    for shell_id in range(len(SPATIAL_SHELL_BOUNDARIES) + 1):
        selected = pair_mask & (shell_ids == shell_id)
        occupancy = selected.to(distances.dtype).sum(dim=(1, 2)) / denominator
        contact_mean, contact_std = _masked_mean_std(contact, selected)
        distance_mean, _ = _masked_mean_std(expected_distance, selected)
        entropy_mean, _ = _masked_mean_std(entropy, selected)
        summaries.extend(
            (occupancy, contact_mean, contact_std, distance_mean, entropy_mean)
        )
    return torch.stack(summaries, dim=-1)


class E2EMSAInterfacePairFusionV3(MSAInterfacePairPoolExpertV18):
    """A V18-derived pocket-MSA core with one deployable prediction graph.

    Stage one warms up the core with the zero-initialized fusion head frozen.
    Stage two jointly fine-tunes this same core and head online. The forward
    pass requires neither historical predictions nor postprocessors.
    """

    version = VERSION
    FUSION_HEAD_PREFIX = "fusion_head."

    def __init__(
        self,
        *args,
        fusion_hidden_width=256,
        fusion_dropout=0.10,
        fusion_max_delta=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.affinity_latent_width = int(self.affinity_head[0].in_features)
        self.fusion_feature_width = (
            1 + 2 * self.affinity_latent_width + SPATIAL_FEATURE_WIDTH
        )
        self.fusion_head = ResidualFusionHeadV3(
            self.fusion_feature_width,
            hidden_width=fusion_hidden_width,
            dropout=fusion_dropout,
            max_delta=fusion_max_delta,
        )

    def freeze_fusion_head(self):
        for parameter in self.fusion_head.parameters():
            parameter.requires_grad_(False)

    def unfreeze_fusion_head(self):
        for parameter in self.fusion_head.parameters():
            parameter.requires_grad_(True)

    def forward(self, batch):
        core = super().forward(batch)
        if core.latent.ndim != 2 or core.latent.shape[1] != 2 * self.affinity_latent_width:
            raise RuntimeError("V18 base/pair latent layout changed")
        spatial = spatial_summary_v3(core, batch)
        features = torch.cat(
            (core.prediction[:, None], core.latent, spatial), dim=-1
        )
        residual = self.fusion_head(features)
        return E2EMSAInterfaceOutputV3(
            prediction=core.prediction + residual,
            latent=core.latent,
            contact_logits=core.contact_logits,
            residue_contact_logits=core.residue_contact_logits,
            interface_distance_logits=core.interface_distance_logits,
            interface_contact_logits=core.interface_contact_logits,
            match_logits=core.match_logits,
            distogram_logits=core.distogram_logits,
            affinity_class_logits=core.affinity_class_logits,
            msa_gate=core.msa_gate,
            fusion_features=features,
        )


def architecture_summary():
    return {
        "version": VERSION,
        "core": "V13 pocket-MSA/structure affinity trunk with V18 contact-aware pair-token pooling",
        "fusion_inputs": [
            "core affinity prediction",
            "base and pair-token latents",
            "six distance-shell contact/distance summaries from the same forward pass",
        ],
        "fusion_head": "zero-initialized bounded residual MLP inside the model graph",
        "training_stages": [
            "warm up the V18-style core while the zero-initialized fusion head is frozen",
            "jointly fine-tune the same core and fusion head online with the full-model affinity prediction",
            "use the official validation split for epoch selection and the sealed V20 no-regression gate",
        ],
        "training_protocol": "same_trunk_two_stage_v1",
        "feature_training": "online_same_forward_no_cross_fold_latent_cache",
        "msa_scope": "pocket-aligned MSA rows; not a full-chain MSA Transformer",
        "prediction_aggregation": "complete_prediction_per_msa_subset_then_mean",
        "historical_prediction_inputs": False,
        "pca_or_tree_postprocessor": False,
        "test_data_used_during_training": False,
    }


__all__ = [
    "E2EMSAInterfaceOutputV3",
    "E2EMSAInterfacePairFusionV3",
    "ResidualFusionHeadV3",
    "SPATIAL_FEATURE_WIDTH",
    "architecture_summary",
    "spatial_summary_v3",
]
