"""Bounded residual model for anchored expert fusion v5."""

from decimal import Decimal, InvalidOperation
from numbers import Integral, Number, Real
import math

import torch
from torch import nn
from torch.nn import functional as F


_EXPERT_COUNT = 3
_ADAPTER_WIDTH = 64
_SCALAR_WIDTH = 12
_FEATURE_WIDTH = _SCALAR_WIDTH + _EXPERT_COUNT * _ADAPTER_WIDTH
_ALLOWED_BUDGETS = {
    Decimal("0.10"): 0.10,
    Decimal("0.25"): 0.25,
    Decimal("0.50"): 0.50,
}


def _validated_budget(value):
    if isinstance(value, bool) or not isinstance(value, Number):
        raise ValueError("budget must be one of 0.10, 0.25, or 0.50")
    try:
        finite_value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("budget must be one of 0.10, 0.25, or 0.50")
    if not math.isfinite(finite_value):
        raise ValueError("budget must be one of 0.10, 0.25, or 0.50")
    for exact_value, canonical_value in _ALLOWED_BUDGETS.items():
        if value == exact_value:
            return canonical_value
    try:
        decimal_value = Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation):
        raise ValueError("budget must be one of 0.10, 0.25, or 0.50")
    if decimal_value not in _ALLOWED_BUDGETS:
        raise ValueError("budget must be one of 0.10, 0.25, or 0.50")
    return _ALLOWED_BUDGETS[decimal_value]


def _validated_dropout(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("dropout must be finite and in [0, 1)")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return value


def _validated_latent_widths(widths):
    try:
        result = tuple(widths)
    except TypeError as error:
        raise ValueError("latent_widths must contain three positive integers") from error
    if len(result) != _EXPERT_COUNT or any(
        isinstance(width, bool)
        or not isinstance(width, Integral)
        or width <= 0
        for width in result
    ):
        raise ValueError("latent_widths must contain three positive integers")
    return tuple(int(width) for width in result)


def _floating_tensor(value, name):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError(name + " must be a floating tensor")
    return value


def _validate_batch_values(base, raw, standardized, available, latents):
    named_tensors = (
        ("base_predictions", base),
        ("raw_predictions", raw),
        ("standardized_predictions", standardized),
        *(("latent %d" % index, latent) for index, latent in enumerate(latents)),
    )
    # One host synchronization on valid batches; diagnostics are slow-path only.
    checks = [torch.isfinite(tensor).all() for _, tensor in named_tensors]
    checks.append(available[:, 0].all())
    if torch.stack(checks).all().item():
        return
    for name, tensor in named_tensors:
        if not torch.isfinite(tensor).all().item():
            raise ValueError(name + " must be finite")
    if not available[:, 0].all().item():
        raise ValueError("Upper-Bound must be available for every row")


def _validated_batch(batch, latent_adapters):
    if not isinstance(batch, dict):
        raise ValueError("batch must be a dictionary")
    try:
        base = _floating_tensor(batch["base_predictions"], "base_predictions")
        raw = _floating_tensor(batch["raw_predictions"], "raw_predictions")
        standardized = _floating_tensor(
            batch["standardized_predictions"], "standardized_predictions"
        )
        available = batch["available"]
        latents = batch["latents"]
    except KeyError as error:
        raise ValueError("batch is missing " + str(error.args[0])) from error

    if base.ndim != 1 or base.numel() == 0:
        raise ValueError("base_predictions must have nonempty shape [B]")
    row_count = base.shape[0]
    if raw.shape != (row_count, _EXPERT_COUNT):
        raise ValueError("raw_predictions must have shape [B, 3]")
    if standardized.shape != (row_count, _EXPERT_COUNT):
        raise ValueError("standardized_predictions must have shape [B, 3]")
    if not torch.is_tensor(available) or available.dtype != torch.bool:
        raise ValueError("available must be a boolean tensor")
    if available.shape != (row_count, _EXPERT_COUNT):
        raise ValueError("available must have shape [B, 3]")

    try:
        adapter_count = len(latent_adapters)
        latent_count = len(latents)
    except TypeError as error:
        raise ValueError("batch must contain three latents") from error
    if adapter_count != _EXPERT_COUNT:
        raise ValueError("latent_adapters must contain three Linear adapters")
    if latent_count != _EXPERT_COUNT:
        raise ValueError("batch must contain three latents")

    expected_device = base.device
    expected_dtype = base.dtype
    for name, tensor in (
        ("raw_predictions", raw),
        ("standardized_predictions", standardized),
    ):
        if tensor.device != expected_device:
            raise ValueError(name + " must be on the base_predictions device")
        if tensor.dtype != expected_dtype:
            raise ValueError(name + " must match the base_predictions dtype")
    if available.device != expected_device:
        raise ValueError("available must be on the base_predictions device")

    validated_latents = []
    for expert, (latent, adapter) in enumerate(zip(latents, latent_adapters)):
        latent = _floating_tensor(latent, "latent %d" % expert)
        if not isinstance(adapter, nn.Linear) or adapter.out_features != _ADAPTER_WIDTH:
            raise ValueError("latent_adapters must contain three Linear(., 64) adapters")
        if latent.shape != (row_count, adapter.in_features):
            raise ValueError(
                "latent %d must have shape [B, %d]" % (expert, adapter.in_features)
            )
        if latent.device != expected_device or adapter.weight.device != expected_device:
            raise ValueError("latent %d and its adapter must be on the batch device" % expert)
        if latent.dtype != expected_dtype or adapter.weight.dtype != expected_dtype:
            raise ValueError("latent %d and its adapter must match the batch dtype" % expert)
        validated_latents.append(latent)
    validated_latents = tuple(validated_latents)
    _validate_batch_values(
        base, raw, standardized, available, validated_latents
    )
    return base, raw, standardized, available, validated_latents


class GatedBlock(nn.Module):
    """SiLU/sigmoid gated projection followed by normalization and dropout."""

    def __init__(self, input_width, output_width, dropout=0.10):
        super().__init__()
        if (
            isinstance(input_width, bool)
            or not isinstance(input_width, Integral)
            or input_width <= 0
            or isinstance(output_width, bool)
            or not isinstance(output_width, Integral)
            or output_width <= 0
        ):
            raise ValueError("block widths must be positive integers")
        dropout = _validated_dropout(dropout)
        self.projection = nn.Linear(int(input_width), 2 * int(output_width))
        self.normalization = nn.LayerNorm(int(output_width))
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs):
        value, gate = self.projection(inputs).chunk(2, dim=-1)
        hidden = F.silu(value) * torch.sigmoid(gate)
        return self.dropout_layer(self.normalization(hidden))


def build_residual_features(batch, latent_adapters):
    """Build the fixed-width scalar and masked latent feature tensor."""
    base, raw, standardized, available, latents = _validated_batch(
        batch, latent_adapters
    )
    del base
    fallback_raw = torch.where(available, raw, raw[:, :1])
    fallback_standardized = torch.where(
        available, standardized, standardized[:, :1]
    )
    disagreements = torch.stack(
        (
            fallback_raw[:, 0] - fallback_raw[:, 1],
            fallback_raw[:, 0] - fallback_raw[:, 2],
            fallback_raw[:, 1] - fallback_raw[:, 2],
        ),
        dim=1,
    )
    scalar_features = torch.cat(
        (
            fallback_raw,
            fallback_standardized,
            disagreements,
            available.to(dtype=raw.dtype),
        ),
        dim=1,
    )
    adapted = []
    for expert, (adapter, latent) in enumerate(zip(latent_adapters, latents)):
        mask = available[:, expert : expert + 1].to(dtype=raw.dtype)
        adapted.append(F.silu(adapter(latent)) * mask)
    return torch.cat((scalar_features, *adapted), dim=1)


class AnchoredResidualModel(nn.Module):
    """Predict a bounded correction around immutable base predictions."""

    def __init__(
        self,
        latent_widths=(1024, 640, 1024),
        budget=0.25,
        dropout=0.10,
    ):
        super().__init__()
        self._latent_widths = _validated_latent_widths(latent_widths)
        self._budget = _validated_budget(budget)
        self._dropout = _validated_dropout(dropout)
        self.latent_adapters = nn.ModuleList(
            nn.Linear(width, _ADAPTER_WIDTH) for width in self._latent_widths
        )
        self.blocks = nn.ModuleList(
            (
                GatedBlock(_FEATURE_WIDTH, 256, self._dropout),
                GatedBlock(256, 128, self._dropout),
                GatedBlock(128, 64, self._dropout),
            )
        )
        self.output_layer = nn.Linear(64, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @property
    def latent_widths(self):
        return self._latent_widths

    @property
    def budget(self):
        return self._budget

    @property
    def dropout(self):
        return self._dropout

    @property
    def adapter_width(self):
        return _ADAPTER_WIDTH

    @property
    def feature_width(self):
        return _FEATURE_WIDTH

    def forward(self, batch):
        features = build_residual_features(batch, self.latent_adapters)
        hidden = features
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.output_layer(hidden).squeeze(-1)
        return batch["base_predictions"] + self._budget * torch.tanh(delta)


__all__ = ["GatedBlock", "build_residual_features", "AnchoredResidualModel"]
