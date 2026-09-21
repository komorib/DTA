"""Rank-unlocked gated Transformer over three frozen DTA experts."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import nn


_EXPERT_COUNT = 3
_SCALAR_WIDTH = 13


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _dropout(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("dropout must be finite and in [0, 1)")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result >= 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return result


def _latent_widths(value):
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError("latent widths must contain three values") from error
    if len(result) != _EXPERT_COUNT:
        raise ValueError("latent widths must contain three values")
    return tuple(_positive_integer(width, "latent width") for width in result)


def _floating_tensor(value, name, ndim):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError(name + " must be a floating tensor")
    if value.ndim != ndim:
        raise ValueError(name + " shape does not match")
    return value


def _validate_batch(batch, latent_widths):
    if not isinstance(batch, dict):
        raise ValueError("batch must be a dictionary")
    try:
        base = _floating_tensor(batch["base_predictions"], "base predictions", 1)
        raw = _floating_tensor(batch["raw_predictions"], "raw predictions", 2)
        standard = _floating_tensor(
            batch["standardized_predictions"], "standardized predictions", 2
        )
        available = batch["available"]
        latents = tuple(batch["latents"])
    except KeyError as error:
        raise ValueError("batch is missing " + str(error.args[0])) from error
    row_count = base.shape[0]
    expected = (row_count, _EXPERT_COUNT)
    if row_count == 0 or raw.shape != expected or standard.shape != expected:
        raise ValueError("expert predictions must have shape [B, 3]")
    if (
        not torch.is_tensor(available)
        or available.dtype != torch.bool
        or available.shape != expected
    ):
        raise ValueError("available must be boolean with shape [B, 3]")
    if len(latents) != _EXPERT_COUNT:
        raise ValueError("batch must contain three latent tensors")
    for index, (latent, width) in enumerate(zip(latents, latent_widths)):
        latent = _floating_tensor(latent, "latent %d" % index, 2)
        if latent.shape != (row_count, width):
            raise ValueError("latent %d shape does not match" % index)
    tensors = (base, raw, standard) + latents
    device = base.device
    dtype = base.dtype
    if any(tensor.device != device or tensor.dtype != dtype for tensor in tensors):
        raise ValueError("all floating batch tensors must share device and dtype")
    if available.device != device:
        raise ValueError("availability mask must share the batch device")
    if not available.any(dim=1).all():
        raise ValueError("every row must contain at least one available expert")
    if not torch.stack([torch.isfinite(tensor).all() for tensor in tensors]).all():
        raise ValueError("batch tensors must be finite")
    return base, raw, standard, available, latents


def scalar_features(base, raw, standard, available):
    """Return global scalar context used by the fusion token."""
    fallback_raw = torch.where(available, raw, base[:, None])
    fallback_standard = torch.where(available, standard, torch.zeros_like(standard))
    disagreements = torch.stack(
        (
            torch.abs(fallback_raw[:, 0] - fallback_raw[:, 1]),
            torch.abs(fallback_raw[:, 0] - fallback_raw[:, 2]),
            torch.abs(fallback_raw[:, 1] - fallback_raw[:, 2]),
        ),
        dim=-1,
    )
    return torch.cat(
        (
            base[:, None],
            fallback_raw,
            fallback_standard,
            available.to(dtype=base.dtype),
            disagreements,
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class FusionOutput:
    prediction: torch.Tensor
    anchor: torch.Tensor
    dynamic_mix: torch.Tensor
    residual: torch.Tensor
    gate_weights: torch.Tensor


class RankUnlockedFusionV7(nn.Module):
    """Fuse frozen expert latents while permitting rank corrections."""

    def __init__(
        self,
        latent_widths=(1024, 640, 1024),
        model_width=256,
        transformer_layers=3,
        attention_heads=8,
        dropout=0.10,
    ):
        super().__init__()
        self.latent_widths = _latent_widths(latent_widths)
        self.model_width = _positive_integer(model_width, "model width")
        layer_count = _positive_integer(transformer_layers, "transformer layers")
        head_count = _positive_integer(attention_heads, "attention heads")
        if self.model_width % head_count:
            raise ValueError("model width must be divisible by attention heads")
        self.dropout = _dropout(dropout)
        adapter_hidden = 2 * self.model_width
        self.latent_adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, adapter_hidden),
                nn.LayerNorm(adapter_hidden),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(adapter_hidden, self.model_width),
                nn.LayerNorm(self.model_width),
                nn.SiLU(),
            )
            for width in self.latent_widths
        )
        self.expert_scalar_adapters = nn.ModuleList(
            nn.Linear(4, self.model_width) for _ in range(_EXPERT_COUNT)
        )
        self.scalar_adapter = nn.Sequential(
            nn.Linear(_SCALAR_WIDTH, self.model_width),
            nn.LayerNorm(self.model_width),
            nn.SiLU(),
        )
        self.expert_embedding = nn.Parameter(
            torch.zeros(_EXPERT_COUNT, self.model_width)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_width,
            nhead=head_count,
            dim_feedforward=4 * self.model_width,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layer_count, norm=nn.LayerNorm(self.model_width)
        )
        self.gate_head = nn.Linear(self.model_width, _EXPERT_COUNT)
        self.mix_scale_logit = nn.Parameter(torch.tensor(-4.0))
        self.residual_head = nn.Sequential(
            nn.Linear(2 * self.model_width, self.model_width),
            nn.LayerNorm(self.model_width),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_width, self.model_width // 2),
            nn.SiLU(),
            nn.Linear(self.model_width // 2, 1),
        )
        nn.init.normal_(self.expert_embedding, std=0.02)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, batch, return_details=False):
        base, raw, standard, available, latents = _validate_batch(
            batch, self.latent_widths
        )
        global_token = self.scalar_adapter(
            scalar_features(base, raw, standard, available)
        )
        expert_tokens = []
        for index, (latent, adapter, scalar_adapter) in enumerate(
            zip(latents, self.latent_adapters, self.expert_scalar_adapters)
        ):
            expert_scalar = torch.stack(
                (
                    raw[:, index],
                    standard[:, index],
                    raw[:, index] - base,
                    available[:, index].to(dtype=base.dtype),
                ),
                dim=-1,
            )
            token = adapter(latent) + scalar_adapter(expert_scalar)
            token = token + self.expert_embedding[index].to(dtype=token.dtype)
            expert_tokens.append(token)
        tokens = torch.stack((global_token,) + tuple(expert_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros((base.shape[0], 1), dtype=torch.bool, device=base.device),
                ~available,
            ),
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        global_state = encoded[:, 0]
        encoded_experts = encoded[:, 1:]
        gate_logits = self.gate_head(global_state)
        gate_logits = gate_logits.masked_fill(~available, torch.finfo(base.dtype).min)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        dynamic_mix = (gate_weights * raw).sum(dim=-1)
        expert_state = (gate_weights[:, :, None] * encoded_experts).sum(dim=1)
        residual = self.residual_head(
            torch.cat((global_state, expert_state), dim=-1)
        ).squeeze(-1)
        mix_scale = torch.sigmoid(self.mix_scale_logit).to(dtype=base.dtype)
        prediction = base + mix_scale * (dynamic_mix - base) + residual
        if not return_details:
            return prediction
        return FusionOutput(
            prediction=prediction,
            anchor=base,
            dynamic_mix=dynamic_mix,
            residual=residual,
            gate_weights=gate_weights,
        )


__all__ = ["FusionOutput", "RankUnlockedFusionV7", "scalar_features"]
