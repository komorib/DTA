"""High-capacity multiscale fusion used by the LP-PDBBind upper-bound run."""

import math

import torch
import torch.nn.functional as F
from torch import nn


MODEL_DIM = 512
NUM_HEADS = 8
LATENT_COUNT = 32


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
    def __init__(self, dimension, multiplier=4, dropout=0.0):
        super().__init__()
        hidden = int(dimension) * int(multiplier)
        self.input = nn.Linear(dimension, hidden * 2)
        self.output = nn.Linear(hidden, dimension)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, value):
        projected, gate = self.input(value).chunk(2, dim=-1)
        return self.output(self.dropout(projected * F.gelu(gate)))


class BiasedMultiheadAttention(nn.Module):
    """SDPA attention with an optional per-head additive source bias."""

    def __init__(self, dimension=MODEL_DIM, heads=NUM_HEADS, dropout=0.0):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dim = self.dimension // self.heads
        self.to_q = nn.Linear(self.dimension, self.dimension, bias=False)
        self.to_k = nn.Linear(self.dimension, self.dimension, bias=False)
        self.to_v = nn.Linear(self.dimension, self.dimension, bias=False)
        self.output = nn.Linear(self.dimension, self.dimension)
        self.dropout = float(dropout)

    def forward(
        self,
        query,
        source,
        source_mask=None,
        source_bias=None,
        return_attention=False,
    ):
        batch_size, query_count, _ = query.shape
        source_count = source.shape[1]
        q = self.to_q(query).view(
            batch_size, query_count, self.heads, self.head_dim
        ).transpose(1, 2)
        k = self.to_k(source).view(
            batch_size, source_count, self.heads, self.head_dim
        ).transpose(1, 2)
        v = self.to_v(source).view(
            batch_size, source_count, self.heads, self.head_dim
        ).transpose(1, 2)
        additive_mask = None
        if source_bias is not None:
            if source_bias.shape != (batch_size, self.heads, 1, source_count):
                raise ValueError("source_bias has an invalid shape")
            additive_mask = source_bias.to(dtype=q.dtype)
        if source_mask is not None:
            if source_mask.shape != (batch_size, source_count):
                raise ValueError("source_mask has an invalid shape")
            padding = torch.zeros(
                (batch_size, 1, 1, source_count),
                device=q.device,
                dtype=q.dtype,
            )
            padding = padding.masked_fill(
                ~source_mask[:, None, None, :], torch.finfo(q.dtype).min
            )
            additive_mask = padding if additive_mask is None else additive_mask + padding
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, query_count, self.dimension
        )
        result = self.output(attended)
        if not return_attention:
            return result, None
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1))
        logits = logits / math.sqrt(self.head_dim)
        if additive_mask is not None:
            logits = logits + additive_mask.float()
        attention = torch.softmax(logits, dim=-1).to(dtype=result.dtype)
        return result, attention


class AxialInteractionEncoder(nn.Module):
    def __init__(self, input_dim=128, dimension=MODEL_DIM, summaries=8):
        super().__init__()
        layer_args = {
            "d_model": int(input_dim),
            "nhead": 8,
            "dim_feedforward": 512,
            "dropout": 0.0,
            "batch_first": True,
            "activation": "gelu",
            "norm_first": True,
        }
        self.row_encoder = nn.TransformerEncoderLayer(**layer_args)
        self.column_encoder = nn.TransformerEncoderLayer(**layer_args)
        self.summary_queries = nn.Parameter(
            torch.empty(int(summaries), int(input_dim))
        )
        self.summary_projection = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(dimension)),
            nn.GELU(),
            nn.Linear(int(dimension), int(dimension)),
        )
        nn.init.normal_(self.summary_queries, std=0.02)

    def forward(self, matrix):
        if matrix.ndim != 3 or matrix.shape[1:] != (128, 128):
            raise ValueError("interaction matrix must have shape [B,128,128]")
        rows = self.row_encoder(matrix)
        columns = self.column_encoder(matrix.transpose(1, 2)).transpose(1, 2)
        encoded = 0.5 * (rows + columns)
        summary = encoded.mean(dim=1, keepdim=True)
        summary = summary + self.summary_queries[None, :, :]
        return self.summary_projection(summary)


class PerceiverBlock(nn.Module):
    def __init__(self, dimension, heads, dropout, drop_path):
        super().__init__()
        self.cross_norm = nn.LayerNorm(dimension)
        self.source_norm = nn.LayerNorm(dimension)
        self.cross_attention = BiasedMultiheadAttention(
            dimension, heads, dropout
        )
        self.self_norm = nn.LayerNorm(dimension)
        self.self_attention = BiasedMultiheadAttention(
            dimension, heads, dropout
        )
        self.ff_norm = nn.LayerNorm(dimension)
        self.feed_forward = GEGLU(dimension, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, latents, sources, source_mask, source_bias):
        cross, attention = self.cross_attention(
            self.cross_norm(latents),
            self.source_norm(sources),
            source_mask=source_mask,
            source_bias=source_bias,
            return_attention=True,
        )
        latents = latents + self.drop_path(cross)
        self_attended, _ = self.self_attention(
            self.self_norm(latents), self.self_norm(latents)
        )
        latents = latents + self.drop_path(self_attended)
        latents = latents + self.drop_path(
            self.feed_forward(self.ff_norm(latents))
        )
        return latents, attention


class LatentBlock(nn.Module):
    def __init__(self, dimension, heads, dropout, drop_path):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = BiasedMultiheadAttention(dimension, heads, dropout)
        self.ff_norm = nn.LayerNorm(dimension)
        self.feed_forward = GEGLU(dimension, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, latents):
        attended, _ = self.attention(
            self.attention_norm(latents), self.attention_norm(latents)
        )
        latents = latents + self.drop_path(attended)
        return latents + self.drop_path(
            self.feed_forward(self.ff_norm(latents))
        )


class TwoLayerAdapter(nn.Module):
    def __init__(self, input_dim, output_dim=MODEL_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(output_dim)),
            nn.GELU(),
            nn.Linear(int(output_dim), int(output_dim)),
        )

    def forward(self, value):
        return self.network(value)


class HybridUpperBoundFusion(nn.Module):
    """Coordinate-biased Perceiver with an unconstrained eight-expert head."""

    FIXED_SOURCES = (
        ("gerna_2d_molecule", 128, 0),
        ("gerna_2d_protein", 128, 0),
        ("gerna_3d_molecule", 128, 0),
        ("gerna_3d_protein", 128, 0),
        ("gerna_head", 128, 0),
        ("interface_ligand_summary", 96, 1),
        ("interface_protein_summary", 128, 1),
        ("interface_local_summary", 128, 1),
        ("interface_head", 128, 1),
        ("interface_projection", 64, 1),
        ("interface_score", 1, 1),
    )
    NODE_SOURCES = (
        ("interface_ligand_nodes", "interface_ligand_coords", "interface_ligand_mask", 96),
        ("interface_protein_nodes", "interface_protein_coords", "interface_protein_mask", 96),
        ("interface_residue_nodes", "interface_residue_coords", "interface_residue_mask", 128),
    )

    def __init__(self, dropout=0.1, drop_path_rate=0.10):
        super().__init__()
        self.dimension = MODEL_DIM
        self.heads = NUM_HEADS
        self.fixed_adapters = nn.ModuleDict(
            {
                name: TwoLayerAdapter(input_dim)
                for name, input_dim, _ in self.FIXED_SOURCES
            }
        )
        self.node_adapters = nn.ModuleDict(
            {
                name: TwoLayerAdapter(input_dim)
                for name, _, _, input_dim in self.NODE_SOURCES
            }
        )
        self.matrix_encoders = nn.ModuleDict(
            {
                "gerna_2d_interaction": AxialInteractionEncoder(),
                "gerna_3d_interaction": AxialInteractionEncoder(),
            }
        )
        source_types = len(self.FIXED_SOURCES) + len(self.NODE_SOURCES) + 2
        self.source_embedding = nn.Embedding(source_types, self.dimension)
        self.branch_embedding = nn.Embedding(2, self.dimension)
        self.latent_tokens = nn.Parameter(
            torch.empty(LATENT_COUNT, self.dimension)
        )
        self.rbf_centers = nn.Parameter(torch.linspace(0.0, 16.0, 32))
        self.rbf_width = nn.Parameter(torch.tensor(1.0))
        self.coordinate_bias = nn.Linear(96, self.heads, bias=False)
        rates = torch.linspace(0.0, float(drop_path_rate), 12).tolist()
        self.perceiver_blocks = nn.ModuleList(
            [
                PerceiverBlock(self.dimension, self.heads, dropout, rates[index])
                for index in range(8)
            ]
        )
        self.latent_blocks = nn.ModuleList(
            [
                LatentBlock(self.dimension, self.heads, dropout, rates[8 + index])
                for index in range(4)
            ]
        )
        context_dim = self.dimension + 7
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(context_dim),
                    nn.Linear(context_dim, self.dimension),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.dimension, 1),
                )
                for _ in range(8)
            ]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, 256),
            nn.GELU(),
            nn.Linear(256, 8),
        )
        nn.init.normal_(self.latent_tokens, std=0.02)

    @staticmethod
    def _require(batch, name, dimensions=None):
        if name not in batch or not torch.is_tensor(batch[name]):
            raise ValueError(f"batch is missing tensor {name}")
        value = batch[name]
        if dimensions is not None and value.ndim != dimensions:
            raise ValueError(f"{name} has invalid rank")
        if not value.is_floating_point() and value.dtype != torch.bool:
            raise ValueError(f"{name} must be floating point or bool")
        return value

    @staticmethod
    def _masked_centroid(coords, mask):
        weights = mask.to(dtype=coords.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (coords * weights).sum(dim=1) / denominator

    def _append_source(
        self, tokens, masks, coordinates, coordinate_masks, branches, types,
        value, mask, coordinate, coordinate_mask, branch, source_type,
    ):
        tokens.append(value + self.source_embedding.weight[source_type] + self.branch_embedding.weight[branch])
        masks.append(mask)
        coordinates.append(coordinate)
        coordinate_masks.append(coordinate_mask)
        branches.append(torch.full_like(mask, branch, dtype=torch.long))
        types.append(torch.full_like(mask, source_type, dtype=torch.long))

    def _build_sources(self, batch):
        first = self._require(batch, "gerna_2d_molecule", 2)
        batch_size = first.shape[0]
        device = first.device
        dtype = next(self.parameters()).dtype
        tokens, masks, coordinates, coordinate_masks, branches, types = [], [], [], [], [], []
        zero_coordinate = torch.zeros((batch_size, 1, 3), device=device, dtype=dtype)
        false_coordinate = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        true_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        source_type = 0
        for name, _, branch in self.FIXED_SOURCES:
            value = self._require(batch, name, 2).to(dtype=dtype)
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} batch size does not match")
            self._append_source(tokens, masks, coordinates, coordinate_masks, branches, types, self.fixed_adapters[name](value).unsqueeze(1), true_mask, zero_coordinate, false_coordinate, branch, source_type)
            source_type += 1
        for matrix_name in ("gerna_2d_interaction", "gerna_3d_interaction"):
            matrix = self._require(batch, matrix_name, 3).to(dtype=dtype)
            if matrix.shape != (batch_size, 128, 128):
                raise ValueError(f"{matrix_name} must have shape [B,128,128]")
            value = self.matrix_encoders[matrix_name](matrix)
            count = value.shape[1]
            self._append_source(tokens, masks, coordinates, coordinate_masks, branches, types, value, torch.ones((batch_size, count), device=device, dtype=torch.bool), zero_coordinate.expand(-1, count, -1), false_coordinate.expand(-1, count), 0, source_type)
            source_type += 1
        centroids = []
        for name, coord_name, mask_name, _ in self.NODE_SOURCES:
            node = self._require(batch, name, 3).to(dtype=dtype)
            coords = self._require(batch, coord_name, 3).to(dtype=dtype)
            mask = self._require(batch, mask_name, 2).to(device=device, dtype=torch.bool)
            if node.shape[:2] != mask.shape or coords.shape != (*node.shape[:2], 3):
                raise ValueError(f"{name} nodes, coordinates, and mask do not align")
            value = self.node_adapters[name](node)
            branch = 1
            self._append_source(tokens, masks, coordinates, coordinate_masks, branches, types, value, mask, coords, mask, branch, source_type)
            centroids.append(self._masked_centroid(coords, mask))
            source_type += 1
        sources = torch.cat(tokens, dim=1)
        source_mask = torch.cat(masks, dim=1)
        source_coordinates = torch.cat(coordinates, dim=1)
        coordinate_mask = torch.cat(coordinate_masks, dim=1)
        branch_ids = torch.cat(branches, dim=1)
        type_ids = torch.cat(types, dim=1)
        interface_centroid = torch.stack(centroids, dim=1).mean(dim=1)
        centroid_stack = torch.stack(
            [centroids[0], centroids[1], interface_centroid], dim=1
        )
        distances = torch.linalg.vector_norm(
            source_coordinates[:, :, None, :] - centroid_stack[:, None, :, :], dim=-1
        )
        width = self.rbf_width.abs().clamp_min(1e-4)
        rbf = torch.exp(
            -width * (distances[..., None] - self.rbf_centers[None, None, None, :]).square()
        ).flatten(2)
        source_bias = self.coordinate_bias(rbf).permute(0, 2, 1)
        source_bias = source_bias * coordinate_mask[:, None, :].to(dtype=source_bias.dtype)
        return sources, source_mask, source_bias.unsqueeze(2), branch_ids, type_ids

    def forward(self, batch):
        if not isinstance(batch, dict):
            raise ValueError("batch must be a mapping")
        sources, source_mask, source_bias, branch_ids, _ = self._build_sources(batch)
        batch_size = sources.shape[0]
        latents = self.latent_tokens[None, :, :].expand(batch_size, -1, -1)
        attention = None
        for block in self.perceiver_blocks:
            latents, attention = block(latents, sources, source_mask, source_bias)
        for block in self.latent_blocks:
            latents = block(latents)
        source_attention = attention.mean(dim=(1, 2))
        source_attention = source_attention * source_mask.to(dtype=source_attention.dtype)
        branch_mass = torch.stack(
            [
                (source_attention * (branch_ids == branch).to(source_attention.dtype)).sum(dim=1)
                for branch in (0, 1)
            ],
            dim=1,
        )
        pooled = latents.mean(dim=1)
        gerna_prediction = self._require(batch, "gerna_prediction", 2).to(dtype=pooled.dtype)
        interface_prediction = self._require(batch, "interface_prediction", 2).to(dtype=pooled.dtype)
        if gerna_prediction.shape != (batch_size, 1) or interface_prediction.shape != (batch_size, 1):
            raise ValueError("backbone predictions must have shape [B,1]")
        difference = interface_prediction - gerna_prediction
        context = torch.cat(
            [pooled, gerna_prediction, interface_prediction, difference, difference.abs(), gerna_prediction * interface_prediction, branch_mass],
            dim=-1,
        )
        expert_predictions = torch.cat(
            [expert(context) for expert in self.experts], dim=-1
        )
        expert_weights = torch.softmax(self.gate(context), dim=-1)
        prediction = (expert_predictions * expert_weights).sum(dim=-1, keepdim=True)
        return {
            "prediction": prediction,
            "expert_predictions": expert_predictions,
            "expert_weights": expert_weights,
            "latents": latents,
            "source_attention": source_attention,
        }



class HybridUpperBoundEndToEndModel(nn.Module):
    """Joint GerNA/Interface wrapper for the upper-bound fusion model."""

    def __init__(self, gerna, interface, fusion):
        super().__init__()
        self.gerna = gerna
        self.interface = interface
        self.fusion = fusion

    def forward(self, gerna_inputs, packed_interface):
        if not isinstance(gerna_inputs, dict):
            raise ValueError("gerna_inputs must be a mapping")
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
        fusion_inputs = dict(gerna_features)
        fusion_inputs.update(interface_features)
        fusion_inputs["gerna_prediction"] = gerna_prediction
        fusion_inputs["interface_prediction"] = interface_prediction
        return self.fusion(fusion_inputs)
