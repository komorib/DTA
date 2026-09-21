"""Conditional calibration network for rank-locked calibration v6."""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


_EXPERT_COUNT = 3
_SCALAR_WIDTH = 13


def _finite_tensor(value, name, *, ndim=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(name + " must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(name + " shape does not match")
    if not torch.isfinite(value).all():
        raise ValueError(name + " must be finite")
    return value


def scalar_features(batch):
    if not isinstance(batch, Mapping):
        raise ValueError("batch must be a mapping")
    base = _finite_tensor(
        batch.get("base_predictions"), "base predictions", ndim=1
    )
    raw = _finite_tensor(
        batch.get("raw_predictions"), "raw predictions", ndim=2
    )
    standard = _finite_tensor(
        batch.get("standardized_predictions"),
        "standardized predictions",
        ndim=2,
    )
    available = batch.get("available")
    if not isinstance(available, torch.Tensor) or available.dtype != torch.bool:
        raise ValueError("available must be a boolean tensor")
    expected = (base.shape[0], _EXPERT_COUNT)
    if raw.shape != expected or standard.shape != expected:
        raise ValueError("expert scalar features must have [batch, 3] shape")
    if available.shape != expected:
        raise ValueError("available must have [batch, 3] shape")
    if raw.device != standard.device or raw.device != base.device:
        raise ValueError("scalar feature devices must match")
    available_values = available.to(device=raw.device, dtype=raw.dtype)
    disagreement = torch.stack(
        (
            torch.abs(raw[:, 0] - raw[:, 1]),
            torch.abs(raw[:, 0] - raw[:, 2]),
            torch.abs(raw[:, 1] - raw[:, 2]),
        ),
        dim=-1,
    )
    result = torch.cat(
        (
            base.to(raw.dtype)[:, None],
            raw,
            standard.to(raw.dtype),
            available_values,
            disagreement,
        ),
        dim=-1,
    )
    if result.shape != (base.shape[0], _SCALAR_WIDTH):
        raise ValueError("conditional scalar feature width must be 13")
    return result


def _latent_widths(value):
    try:
        widths = tuple(value)
    except TypeError as error:
        raise ValueError("latent widths must contain exactly three values") from error
    if len(widths) != _EXPERT_COUNT:
        raise ValueError("latent widths must contain exactly three values")
    if any(
        isinstance(width, bool)
        or not isinstance(width, Integral)
        or int(width) <= 0
        for width in widths
    ):
        raise ValueError("latent widths must be positive integers")
    return tuple(int(width) for width in widths)


def _finite_scalar(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        raise ValueError(name + " must be finite")
    return float(value)


class ConditionalCalibrator(nn.Module):
    def __init__(self, latent_widths, affine_slope, affine_intercept):
        super().__init__()
        self.latent_widths = _latent_widths(latent_widths)
        slope = _finite_scalar(affine_slope, "affine slope", positive=True)
        intercept = _finite_scalar(affine_intercept, "affine intercept")
        self.register_buffer(
            "affine_slope", torch.tensor(slope, dtype=torch.float32)
        )
        self.register_buffer(
            "affine_intercept", torch.tensor(intercept, dtype=torch.float32)
        )
        self.latent_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, 64),
                    nn.LayerNorm(64),
                    nn.SiLU(),
                )
                for width in self.latent_widths
            ]
        )
        self.hidden = nn.Sequential(
            nn.Linear(_SCALAR_WIDTH + 3 * 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.output_layer = nn.Linear(64, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def config(self):
        return {
            "schema_version": 1,
            "latent_widths": list(self.latent_widths),
            "affine_slope": float(self.affine_slope.detach().cpu()),
            "affine_intercept": float(self.affine_intercept.detach().cpu()),
            "adapter_width": 64,
            "hidden_widths": [256, 128, 64],
            "dropout": 0.10,
            "scalar_width": _SCALAR_WIDTH,
        }

    def forward(self, batch):
        scalar = scalar_features(batch)
        latents = batch.get("latents")
        if not isinstance(latents, (tuple, list)) or len(latents) != 3:
            raise ValueError("latents must contain exactly three tensors")
        available = batch["available"].to(device=scalar.device)
        projected = []
        for index, (latent, width, adapter) in enumerate(
            zip(latents, self.latent_widths, self.latent_adapters)
        ):
            latent = _finite_tensor(
                latent, "expert latent", ndim=2
            )
            if latent.shape != (scalar.shape[0], width):
                raise ValueError("expert latent shape does not match")
            if latent.device != scalar.device:
                raise ValueError("expert latent device does not match")
            state = adapter(latent.to(dtype=scalar.dtype))
            state = state * available[:, index : index + 1].to(state.dtype)
            projected.append(state)
        hidden = self.hidden(torch.cat((scalar,) + tuple(projected), dim=-1))
        delta = self.output_layer(hidden).squeeze(-1)
        slope = self.affine_slope.to(dtype=delta.dtype)
        intercept = self.affine_intercept.to(dtype=delta.dtype)
        anchor = slope * batch["base_predictions"].to(delta.dtype) + intercept
        return anchor + delta


def calibration_loss(prediction, label):
    prediction = _finite_tensor(prediction, "prediction", ndim=1)
    label = _finite_tensor(label, "label", ndim=1)
    if prediction.shape != label.shape:
        raise ValueError("prediction and label shapes must match")
    if prediction.device != label.device:
        raise ValueError("prediction and label devices must match")
    label = label.to(dtype=prediction.dtype)
    mse = F.mse_loss(prediction, label)
    huber = F.smooth_l1_loss(prediction, label, beta=1.0)
    prediction_std = torch.sqrt(
        torch.var(prediction, unbiased=False) + 1e-8
    )
    label_std = torch.sqrt(torch.var(label, unbiased=False) + 1e-8)
    moment = (
        (prediction.mean() - label.mean()).square()
        + (prediction_std - label_std).square()
    )
    return 0.60 * mse + 0.30 * huber + 0.10 * moment


__all__ = [
    "ConditionalCalibrator",
    "calibration_loss",
    "scalar_features",
]
