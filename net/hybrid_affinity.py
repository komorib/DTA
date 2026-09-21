import math

import torch
from torch import nn


SOURCE_DIMS = {
    "gerna_2d": 16384,
    "gerna_3d": 16384,
    "gerna_head": 128,
    "interface_protein": 128,
    "interface_ligand": 96,
    "interface_local": 128,
    "interface_head": 128,
}
SOURCE_ORDER = tuple(SOURCE_DIMS)
GERNA_SOURCES = SOURCE_ORDER[:3]
INTERFACE_SOURCES = SOURCE_ORDER[3:]


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim, token_dim=128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.token_dim),
            nn.SiLU(),
        )

    def forward(self, value):
        return self.net(value)


class MLPTokenFusion(nn.Module):
    def __init__(self, source_count, token_dim, dropout):
        super().__init__()
        input_dim = int(source_count) * int(token_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(512, int(token_dim)),
        )

    def forward(self, tokens):
        return self.net(tokens.flatten(start_dim=1))


class TransformerTokenFusion(nn.Module):
    def __init__(self, token_dim, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(token_dim),
            nhead=8,
            dim_feedforward=512,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(int(token_dim)),
        )
        self.pool_query = nn.Parameter(torch.empty(int(token_dim)))
        self.pool_norm = nn.LayerNorm(int(token_dim))
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, tokens):
        encoded = self.encoder(tokens)
        scores = torch.matmul(encoded, self.pool_query)
        scores = scores / math.sqrt(encoded.shape[-1])
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        return self.pool_norm(pooled)


class HybridAffinityFusion(nn.Module):
    FUSION_TYPES = {"mlp", "transformer"}

    def __init__(self, fusion_type, token_dim=128, dropout=0.1):
        super().__init__()
        if fusion_type not in self.FUSION_TYPES:
            raise ValueError(
                "fusion_type must be one of "
                + ", ".join(sorted(self.FUSION_TYPES))
            )
        self.fusion_type = str(fusion_type)
        self.token_dim = int(token_dim)
        if self.token_dim != 128:
            raise ValueError("token_dim must be exactly 128")

        self.adapters = nn.ModuleDict(
            {
                name: FeatureAdapter(dimension, self.token_dim)
                for name, dimension in SOURCE_DIMS.items()
            }
        )
        self.source_embedding = nn.Parameter(
            torch.empty(len(SOURCE_ORDER), self.token_dim)
        )
        self.branch_embedding = nn.Parameter(
            torch.empty(2, self.token_dim)
        )
        nn.init.normal_(self.source_embedding, std=0.02)
        nn.init.normal_(self.branch_embedding, std=0.02)
        self.register_buffer(
            "source_branch",
            torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
            persistent=False,
        )

        if self.fusion_type == "mlp":
            self.fusion = MLPTokenFusion(
                len(SOURCE_ORDER), self.token_dim, dropout
            )
        else:
            self.fusion = TransformerTokenFusion(
                self.token_dim, dropout
            )

        self.context_dim = 3 * self.token_dim + 3
        self.mlp_expert = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, 256),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 1),
        )
        self.bilinear_expert = nn.Bilinear(
            self.token_dim, self.token_dim, 1
        )
        self.calibrated_expert = nn.Sequential(
            nn.LayerNorm(self.context_dim + 2),
            nn.Linear(self.context_dim + 2, 256),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 1),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    @staticmethod
    def _validate_prediction(
        value, name, batch_size, device, dtype
    ):
        if not torch.is_tensor(value):
            raise ValueError(f"{name} must be a tensor")
        if value.shape != (batch_size, 1):
            raise ValueError(
                f"{name} must have shape [{batch_size}, 1]"
            )
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")
        return value.to(dtype=dtype)

    @staticmethod
    def _validate_source(
        value, name, dimension, batch_size, device, dtype
    ):
        if not torch.is_tensor(value):
            raise ValueError(f"{name} must be a tensor")
        if value.shape != (batch_size, dimension):
            raise ValueError(
                f"{name} must have shape [{batch_size}, {dimension}]"
            )
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")
        return value.to(dtype=dtype)

    def _adapt_tokens(self, gerna_features, interface_features):
        if not isinstance(gerna_features, dict):
            raise ValueError("gerna_features must be a dict")
        if not isinstance(interface_features, dict):
            raise ValueError("interface_features must be a dict")
        merged = dict(gerna_features)
        overlap = set(merged).intersection(interface_features)
        if overlap:
            raise ValueError(
                "feature dictionaries overlap: "
                + ", ".join(sorted(overlap))
            )
        merged.update(interface_features)
        missing = [name for name in SOURCE_ORDER if name not in merged]
        if missing:
            raise ValueError(
                "missing affinity features: " + ", ".join(missing)
            )

        first = merged[SOURCE_ORDER[0]]
        if not torch.is_tensor(first) or first.ndim != 2:
            raise ValueError("gerna_2d must be a two-dimensional tensor")
        batch_size = int(first.shape[0])
        device = self.source_embedding.device
        dtype = self.source_embedding.dtype
        tokens = []
        for index, name in enumerate(SOURCE_ORDER):
            value = self._validate_source(
                merged[name],
                name,
                SOURCE_DIMS[name],
                batch_size,
                device,
                dtype,
            )
            token = self.adapters[name](value)
            token = token + self.source_embedding[index]
            token = token + self.branch_embedding[
                self.source_branch[index]
            ]
            tokens.append(token)
        return torch.stack(tokens, dim=1), batch_size

    def forward(
        self,
        gerna_features,
        interface_features,
        gerna_prediction,
        interface_prediction,
    ):
        tokens, batch_size = self._adapt_tokens(
            gerna_features, interface_features
        )
        y_g = self._validate_prediction(
            gerna_prediction,
            "gerna_prediction",
            batch_size,
            self.source_embedding.device,
            self.source_embedding.dtype,
        )
        y_i = self._validate_prediction(
            interface_prediction,
            "interface_prediction",
            batch_size,
            self.source_embedding.device,
            self.source_embedding.dtype,
        )

        z_g = tokens[:, : len(GERNA_SOURCES)].mean(dim=1)
        z_i = tokens[:, len(GERNA_SOURCES) :].mean(dim=1)
        pooled = self.fusion(tokens)
        context = torch.cat(
            [
                pooled,
                z_g * z_i,
                torch.abs(z_g - z_i),
                y_g,
                y_i,
                y_i - y_g,
            ],
            dim=-1,
        )
        expert_values = torch.cat(
            [
                self.mlp_expert(context),
                self.bilinear_expert(z_g, z_i),
                self.calibrated_expert(
                    torch.cat([context, y_g, y_i], dim=-1)
                ),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.gate(context), dim=-1)
        prediction = torch.sum(
            weights * expert_values, dim=-1, keepdim=True
        )
        return {
            "prediction": prediction,
            "branch_predictions": torch.cat([y_g, y_i], dim=-1),
            "expert_predictions": expert_values,
            "expert_weights": weights,
            "diagnostics": {
                "tokens": tokens,
                "pooled": pooled,
                "gerna_summary": z_g,
                "interface_summary": z_i,
                "context": context,
            },
        }


class HybridAffinityModel(nn.Module):
    def __init__(self, gerna, interface, fusion):
        super().__init__()
        self.gerna = gerna
        self.interface = interface
        self.fusion = fusion

    def forward(self, gerna_inputs, interface_item):
        if not isinstance(gerna_inputs, dict):
            raise ValueError("gerna_inputs must be a dict")
        self.gerna.request_affinity_features()
        try:
            gerna_device_type = next(
                self.gerna.parameters()
            ).device.type
            with torch.autocast(
                device_type=gerna_device_type, enabled=False
            ):
                gerna_output = self.gerna(**gerna_inputs)
                if (
                    not isinstance(gerna_output, (tuple, list))
                    or len(gerna_output) < 1
                ):
                    raise RuntimeError(
                        "GerNA forward must return prediction first"
                    )
                gerna_prediction = gerna_output[0]
                gerna_features = self.gerna.take_affinity_features()
        except BaseException:
            try:
                self.gerna.discard_affinity_features()
            except BaseException:
                pass
            raise
        interface_prediction, interface_features = self.interface(
            interface_item, return_features=True
        )
        return self.fusion(
            gerna_features,
            interface_features,
            gerna_prediction,
            interface_prediction,
        )
