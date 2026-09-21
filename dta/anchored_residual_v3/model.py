"""Teacher-anchored residual model isolated from the v2 implementation."""

import torch
from torch import nn


class AnchoredResidualHead(nn.Module):
    def __init__(self, latent_dimension, hidden_dimension=256, dropout=0.1):
        super().__init__()
        input_dimension = int(latent_dimension) + 8
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.output = nn.Linear(hidden_dimension, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        latents,
        gerna_prediction,
        interface_prediction,
        base_prediction,
        teacher_prediction,
    ):
        pooled = latents.mean(dim=1)
        difference = interface_prediction - gerna_prediction
        scalar = torch.cat(
            [
                gerna_prediction,
                interface_prediction,
                base_prediction,
                teacher_prediction,
                base_prediction - teacher_prediction,
                difference,
                difference.abs(),
                gerna_prediction * interface_prediction,
            ],
            dim=-1,
        ).to(dtype=pooled.dtype)
        return self.output(self.network(torch.cat([pooled, scalar], dim=-1)))


class AnchoredResidualModel(nn.Module):
    """Compose unchanged backbones/fusion with a zero-output residual head."""

    def __init__(
        self,
        gerna,
        interface,
        fusion,
        hidden_dimension=256,
        dropout=0.1,
    ):
        super().__init__()
        self.gerna = gerna
        self.interface = interface
        self.fusion = fusion
        latent_dimension = getattr(fusion, "dimension", None)
        if latent_dimension is None:
            raise ValueError("fusion must expose its latent dimension")
        self.residual = AnchoredResidualHead(
            latent_dimension, hidden_dimension=hidden_dimension, dropout=dropout
        )

    def forward(self, gerna_inputs, packed_interface, teacher_prediction):
        if not isinstance(gerna_inputs, dict):
            raise ValueError("gerna_inputs must be a mapping")
        if (
            not torch.is_tensor(teacher_prediction)
            or teacher_prediction.ndim != 2
            or teacher_prediction.shape[1] != 1
            or not bool(torch.isfinite(teacher_prediction).all())
        ):
            raise ValueError("teacher_prediction must be a finite [B,1] tensor")
        self.gerna.request_multiscale_affinity_features()
        try:
            gerna_output = self.gerna(**gerna_inputs)
            gerna_prediction = (
                gerna_output[0]
                if isinstance(gerna_output, (tuple, list))
                else gerna_output
            )
            gerna_features = self.gerna.take_multiscale_affinity_features()
        except BaseException:
            self.gerna.discard_multiscale_affinity_features()
            raise
        interface_prediction, interface_features = self.interface.forward_packed(
            packed_interface
        )
        if teacher_prediction.shape[0] != gerna_prediction.shape[0]:
            raise ValueError("teacher_prediction batch size does not match")
        fusion_inputs = dict(gerna_features)
        fusion_inputs.update(interface_features)
        fusion_inputs["gerna_prediction"] = gerna_prediction
        fusion_inputs["interface_prediction"] = interface_prediction
        fusion_output = self.fusion(fusion_inputs)
        if not isinstance(fusion_output, dict) or {
            "prediction",
            "latents",
            "expert_weights",
        }.difference(fusion_output):
            raise RuntimeError("fusion output is missing residual inputs")
        teacher_prediction = teacher_prediction.to(
            device=gerna_prediction.device,
            dtype=fusion_output["prediction"].dtype,
        )
        delta = self.residual(
            fusion_output["latents"],
            gerna_prediction,
            interface_prediction,
            fusion_output["prediction"],
            teacher_prediction,
        )
        result = dict(fusion_output)
        result.update(
            {
                "prediction": teacher_prediction + delta,
                "delta": delta,
                "base_prediction": fusion_output["prediction"],
                "teacher_prediction": teacher_prediction,
            }
        )
        return result
