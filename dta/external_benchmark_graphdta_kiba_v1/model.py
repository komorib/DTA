from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDTAGCN(nn.Module):
    """Stable GraphDTA-GCN variant with a binary logit head.

    This is the original GraphDTA GCN family (one molecular GCN stack and a
    sequence encoder) adapted for binary labels.  The sequence encoder uses
    masked-free mean/max pooling followed by a projection, deliberately avoiding
    Conv1d/unfold kernels that are unstable on the target cuDNN installation.
    """

    def __init__(self, sequence_vocab: int = 25, embed_dim: int = 128,
                 output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        # Native PyTorch message passing.  This is algebraically a GCN update,
        # but avoids the torch_scatter/PyG CUDA kernels that are incompatible
        # with the cluster's installed CUDA extension.
        self.gcn1 = nn.Linear(78 * 2, 156)
        self.gcn2 = nn.Linear(156 * 2, 312)
        self.gcn3 = nn.Linear(312 * 2, 312)
        self.graph_fc1 = nn.Linear(312 * 2, 1500)
        self.graph_fc2 = nn.Linear(1500, output_dim)
        self.embedding = nn.Embedding(sequence_vocab + 1, embed_dim, padding_idx=0)
        self.sequence_projection = nn.Linear(embed_dim * 2, output_dim)
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        def message_passing(node_features, layer):
            source, destination = data.edge_index
            aggregate = torch.zeros_like(node_features)
            aggregate.index_add_(0, destination, node_features[source])
            degree = torch.zeros(node_features.shape[0], 1, device=node_features.device, dtype=node_features.dtype)
            degree.index_add_(0, destination, torch.ones(destination.shape[0], 1, device=node_features.device, dtype=node_features.dtype))
            aggregate = aggregate / degree.clamp_min(1.0)
            return F.relu(layer(torch.cat([node_features, aggregate], dim=1)))

        node_features = message_passing(data.x, self.gcn1)
        node_features = message_passing(node_features, self.gcn2)
        node_features = message_passing(node_features, self.gcn3)
        batch_size = int(data.batch.max().item()) + 1
        graph_sum = torch.zeros(batch_size, node_features.shape[1], device=node_features.device, dtype=node_features.dtype)
        graph_sum.index_add_(0, data.batch, node_features)
        graph_count = torch.bincount(data.batch, minlength=batch_size).to(node_features.dtype).unsqueeze(1).clamp_min(1.0)
        graph_mean = graph_sum / graph_count
        graph_max = torch.full_like(graph_sum, -torch.inf)
        graph_max.scatter_reduce_(0, data.batch[:, None].expand_as(node_features), node_features, reduce="amax", include_self=True)
        graph_features = torch.cat([graph_max, graph_mean], dim=1)
        graph_features = self.dropout(F.relu(self.graph_fc1(graph_features)))
        graph_features = self.graph_fc2(graph_features)

        sequence = self.embedding(data.target)
        sequence = torch.cat([sequence.mean(dim=1), sequence.amax(dim=1)], dim=1)
        sequence = F.relu(self.sequence_projection(sequence))

        fused = torch.cat([graph_features, sequence], dim=1)
        fused = self.dropout(F.relu(self.fc1(fused)))
        fused = self.dropout(F.relu(self.fc2(fused)))
        return self.out(fused).squeeze(-1)
