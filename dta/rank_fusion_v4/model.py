"""Cross-expert Transformer for LP-PDBbind Rank Fusion v4."""

import torch
import torch.nn.functional as F
from torch import nn


EXPERT_COUNT = 3


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, value):
        if self.probability == 0.0 or not self.training:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = value.new_empty(shape).bernoulli_(keep)
        return value * mask / keep


class GEGLU(nn.Module):
    def __init__(self, dimension, hidden_dimension, dropout):
        super().__init__()
        self.input = nn.Linear(dimension, int(hidden_dimension) * 2)
        self.output = nn.Linear(int(hidden_dimension), dimension)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, value):
        value, gate = self.input(value).chunk(2, dim=-1)
        return self.output(self.dropout(value * F.gelu(gate)))


class TransformerBlock(nn.Module):
    def __init__(self, dimension, heads, dropout, drop_path, feed_forward_width):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.feed_forward = GEGLU(
            dimension, feed_forward_width, dropout
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens, key_padding_mask):
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.drop_path(attended)
        return tokens + self.drop_path(
            self.feed_forward(self.feed_forward_norm(tokens))
        )


class ExpertAdapter(nn.Module):
    def __init__(self, latent_dimension, model_dimension):
        super().__init__()
        self.latent = nn.Sequential(
            nn.LayerNorm(int(latent_dimension)),
            nn.Linear(int(latent_dimension), model_dimension),
            nn.GELU(),
            nn.Linear(model_dimension, model_dimension),
        )
        self.scalar = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, model_dimension),
            nn.GELU(),
            nn.Linear(model_dimension, model_dimension),
        )

    def forward(self, latent, scalar_features):
        return self.latent(latent) + self.scalar(scalar_features)


class RankFusionV4(nn.Module):
    def __init__(
        self,
        latent_dims,
        model_dim=256,
        layers=4,
        heads=8,
        dropout=0.20,
        drop_path_rate=0.10,
    ):
        super().__init__()
        if len(tuple(latent_dims)) != EXPERT_COUNT:
            raise ValueError("latent_dims must contain three experts")
        if int(model_dim) % int(heads):
            raise ValueError("model_dim must be divisible by heads")
        if int(layers) <= 0:
            raise ValueError("layers must be positive")
        self.latent_dims = tuple(int(value) for value in latent_dims)
        if any(value <= 0 for value in self.latent_dims):
            raise ValueError("latent dimensions must be positive")
        self.model_dim = int(model_dim)
        self.adapters = nn.ModuleList(
            [ExpertAdapter(width, self.model_dim) for width in self.latent_dims]
        )
        self.expert_identity = nn.Parameter(
            torch.empty(EXPERT_COUNT, self.model_dim)
        )
        self.missing_tokens = nn.Parameter(
            torch.empty(EXPERT_COUNT, self.model_dim)
        )
        self.fusion_token = nn.Parameter(torch.empty(1, 1, self.model_dim))
        drop_paths = torch.linspace(0.0, float(drop_path_rate), int(layers)).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.model_dim,
                    int(heads),
                    float(dropout),
                    drop_paths[index],
                    feed_forward_width=1024,
                )
                for index in range(int(layers))
            ]
        )
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.mixture_head = nn.Linear(self.model_dim, 1)
        self.register_parameter(
            "base_mixture_logits",
            nn.Parameter(torch.tensor([8.0, -8.0, -8.0])),
        )
        self.calibration_scale = nn.Parameter(torch.ones(EXPERT_COUNT))
        self.calibration_bias = nn.Parameter(torch.zeros(EXPERT_COUNT))
        self.route_coefficient = nn.Parameter(torch.zeros(()))
        self.surrogate_gradient_scale = 1.0
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.model_dim + EXPERT_COUNT),
            nn.Linear(self.model_dim + EXPERT_COUNT, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.expert_identity, std=0.02)
        nn.init.normal_(self.missing_tokens, std=0.02)
        nn.init.normal_(self.fusion_token, std=0.02)
        nn.init.zeros_(self.mixture_head.weight)
        nn.init.zeros_(self.mixture_head.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _validate(self, latents, raw, standardized, available):
        if not isinstance(latents, (tuple, list)) or len(latents) != EXPERT_COUNT:
            raise ValueError("latents must contain three expert latent tensors")
        if raw.ndim != 2 or raw.shape[1] != EXPERT_COUNT:
            raise ValueError("raw_predictions must have shape [B,3]")
        if standardized.shape != raw.shape or available.shape != raw.shape:
            raise ValueError("prediction and availability shapes do not match")
        if available.dtype != torch.bool:
            raise ValueError("available must be boolean")
        if not bool(available[:, 0].all()):
            raise ValueError("Upper-Bound must be available for every row")
        batch_size = raw.shape[0]
        for index, (latent, width) in enumerate(zip(latents, self.latent_dims)):
            if latent.shape != (batch_size, width):
                raise ValueError(f"expert latent {index} has an invalid shape")
        if not torch.isfinite(raw).all() or not torch.isfinite(standardized).all():
            raise ValueError("expert predictions contain nonfinite values")
        if any(not torch.isfinite(latent).all() for latent in latents):
            raise ValueError("expert latents contain nonfinite values")

    def _scalar_features(self, raw, standardized, available, expert_index):
        prediction = raw[:, expert_index : expert_index + 1]
        standardized_prediction = standardized[:, expert_index : expert_index + 1]
        disagreements = prediction - raw
        return torch.cat(
            (
                prediction,
                standardized_prediction,
                disagreements,
                available.float(),
            ),
            dim=1,
        )

    def forward(
        self,
        latents,
        raw_predictions,
        standardized_predictions,
        available,
    ):
        self._validate(
            latents,
            raw_predictions,
            standardized_predictions,
            available,
        )
        raw = raw_predictions.float()
        standardized = standardized_predictions.float()
        batch_size = raw.shape[0]
        expert_tokens = []
        for index, adapter in enumerate(self.adapters):
            token = adapter(
                latents[index].float(),
                self._scalar_features(raw, standardized, available, index),
            )
            token = token + self.expert_identity[index]
            token = torch.where(
                available[:, index : index + 1],
                token,
                self.missing_tokens[index].expand_as(token),
            )
            expert_tokens.append(token)
        expert_tokens = torch.stack(expert_tokens, dim=1)
        fusion = self.fusion_token.expand(batch_size, -1, -1)
        tokens = torch.cat((fusion, expert_tokens), dim=1)
        key_padding_mask = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1), device=available.device, dtype=torch.bool
                ),
                ~available,
            ),
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
        tokens = self.final_norm(tokens)
        fusion_state = tokens[:, 0]
        expert_states = tokens[:, 1:]
        calibrated = raw * self.calibration_scale + self.calibration_bias
        mixture_logits = (
            self.mixture_head(expert_states).squeeze(-1)
            + self.base_mixture_logits
        )
        mixture_logits = mixture_logits.masked_fill(~available, float("-inf"))
        mixture_weights = torch.softmax(mixture_logits, dim=1)
        mixture_prediction = (mixture_weights * calibrated).sum(dim=1)
        disagreement_summary = (raw - raw[:, :1]).abs()
        rank_residual = self.residual_head(
            torch.cat((fusion_state, disagreement_summary), dim=1)
        ).reshape(-1)
        upper_bound = raw[:, 0]
        delta = mixture_prediction - upper_bound
        surrogate_delta = delta - delta.detach()
        prediction = (
            upper_bound
            + self.route_coefficient * delta
            + self.surrogate_gradient_scale * surrogate_delta
            + rank_residual
        )
        return {
            "prediction": prediction,
            "mixture_prediction": mixture_prediction,
            "rank_residual": rank_residual,
            "mixture_weights": mixture_weights,
            "calibrated_predictions": calibrated,
        }
