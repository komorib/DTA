"""V29 sealed V22 optional-structure backbone with a bounded 3D residual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from dta.optional_structure_staged_v22.model import (
    DISTANCE_BINS,
    OptionalStructureStagedDTA,
)


V22_VERSION = "optional-structure-staged-v22"


@dataclass
class OptionalStructureSealedResidualOutput:
    prediction: torch.Tensor
    backbone_prediction: torch.Tensor
    base_prediction: torch.Tensor
    base_latent: torch.Tensor
    structure_latent: torch.Tensor
    structure_gate: torch.Tensor
    structure_delta: torch.Tensor
    backbone_structure_gate: torch.Tensor
    backbone_structure_delta: torch.Tensor
    effective_has_3d: torch.Tensor
    contact_logits: Optional[torch.Tensor]
    distance_logits: Optional[torch.Tensor]


class OptionalStructureSealedResidualDTA(nn.Module):
    """Preserve the complete V22 path and learn only a bounded structured residual."""

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
        adapter_width=192,
        dropout=0.10,
        residual_width=192,
        maximum_residual=0.75,
    ):
        super().__init__()
        self.maximum_residual = float(maximum_residual)
        if self.maximum_residual <= 0.0:
            raise ValueError("maximum_residual must be positive")
        self.backbone = OptionalStructureStagedDTA(
            protein_input_width=protein_input_width,
            width=width,
            heads=heads,
            protein_layers=protein_layers,
            ligand_layers=ligand_layers,
            cross_layers=cross_layers,
            latent_width=latent_width,
            pair_width=pair_width,
            structure_self_layers=structure_self_layers,
            structure_pair_layers=structure_pair_layers,
            structure_token_layers=structure_token_layers,
            structure_top_pairs=structure_top_pairs,
            maximum_protein_tokens=maximum_protein_tokens,
            dataset_count=dataset_count,
            adapter_width=adapter_width,
            dropout=dropout,
        )
        feature_width = 3 * int(latent_width) + 2
        self.residual_features = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, int(residual_width)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_width), int(residual_width)),
            nn.SiLU(),
        )
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, int(residual_width)),
            nn.SiLU(),
            nn.Linear(int(residual_width), 1),
        )
        self.residual_head = nn.Linear(int(residual_width), 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def backbone_kwargs(self):
        backbone = self.backbone
        return {
            "protein_input_width": int(backbone.protein_encoder.adapter[1].in_features),
            "width": int(backbone.protein_pool.score[1].in_features),
            "heads": int(backbone.protein_encoder.encoder.layers[0].self_attn.num_heads),
            "protein_layers": int(len(backbone.protein_encoder.encoder.layers)),
            "ligand_layers": int(len(backbone.ligand_encoder.blocks)),
            "cross_layers": int(len(backbone.cross_blocks)),
            "latent_width": int(backbone.base_heads[0][0].in_features),
            "pair_width": int(backbone.structure_encoder.pocket_pair.out_features),
            "structure_self_layers": int(len(backbone.structure_encoder.pocket_blocks)),
            "structure_pair_layers": int(len(backbone.structure_encoder.pair_blocks)),
            "structure_token_layers": int(len(backbone.structure_encoder.token_encoder.layers)),
            "structure_top_pairs": int(backbone.structure_encoder.top_pairs),
            "maximum_protein_tokens": int(backbone.protein_encoder.maximum_tokens),
            "dataset_count": int(backbone.dataset_count),
            "adapter_width": int(backbone.task_adapters[0][1].out_features),
            "dropout": float(backbone.protein_encoder.adapter[3].p),
        }

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def trainable_parameter_count(self):
        return sum(int(parameter.numel()) for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, batch: Dict[str, torch.Tensor], force_no_3d=False):
        self.backbone.eval()
        with torch.no_grad():
            backbone_output = self.backbone(batch, force_no_3d=force_no_3d)
        base_prediction = backbone_output.base_prediction.detach()
        backbone_prediction = backbone_output.prediction.detach()
        effective = backbone_output.effective_has_3d.bool()
        residual_gate = backbone_prediction.new_zeros(backbone_prediction.shape)
        residual_delta = backbone_prediction.new_zeros(backbone_prediction.shape)
        if effective.any():
            base_latent = backbone_output.base_latent.detach()
            structure_latent = backbone_output.structure_latent.detach()
            features = torch.cat(
                (
                    base_latent,
                    structure_latent,
                    structure_latent - base_latent,
                    backbone_output.structure_gate.detach().to(base_latent.dtype).unsqueeze(-1),
                    backbone_output.structure_delta.detach().to(base_latent.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
            indices = torch.nonzero(effective, as_tuple=False).squeeze(-1)
            selected_features = features.index_select(0, indices)
            hidden = self.residual_features(selected_features)
            selected_gate = torch.sigmoid(self.residual_gate(selected_features).squeeze(-1))
            selected_raw = self.residual_head(hidden).squeeze(-1)
            selected_delta = selected_gate * self.maximum_residual * torch.tanh(selected_raw)
            residual_gate = residual_gate.index_copy(
                0, indices, selected_gate.to(dtype=residual_gate.dtype)
            )
            residual_delta = residual_delta.index_copy(
                0, indices, selected_delta.to(dtype=residual_delta.dtype)
            )
        prediction = backbone_prediction + residual_delta
        return OptionalStructureSealedResidualOutput(
            prediction=prediction,
            backbone_prediction=backbone_prediction,
            base_prediction=base_prediction,
            base_latent=backbone_output.base_latent.detach(),
            structure_latent=backbone_output.structure_latent.detach(),
            structure_gate=residual_gate,
            structure_delta=residual_delta,
            backbone_structure_gate=backbone_output.structure_gate.detach(),
            backbone_structure_delta=backbone_output.structure_delta.detach(),
            effective_has_3d=effective,
            contact_logits=backbone_output.contact_logits.detach()
            if backbone_output.contact_logits is not None
            else None,
            distance_logits=backbone_output.distance_logits.detach()
            if backbone_output.distance_logits is not None
            else None,
        )


def load_sealed_v22_backbone(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("version") != V22_VERSION:
        raise ValueError("V29 requires a sealed V22 structure checkpoint")
    if checkpoint.get("phase") != "structure":
        raise ValueError("V29 requires the V22 structure phase checkpoint")
    expected = model.backbone_kwargs()
    observed = dict(checkpoint.get("model_kwargs", {}))
    if observed != expected:
        raise ValueError("V22 checkpoint model_kwargs differ from V29 sealed backbone")
    model.backbone.load_state_dict(checkpoint["model_state"], strict=True)
    model.freeze_backbone()
    return checkpoint


__all__ = [
    "DISTANCE_BINS",
    "OptionalStructureSealedResidualDTA",
    "OptionalStructureSealedResidualOutput",
    "V22_VERSION",
    "load_sealed_v22_backbone",
]
