from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSICHICSequenceEdgeClassifier(nn.Module):
    """PSICHIC-style protein/ligand graph interaction classifier.

    Protein edges are deterministic sequence +/-1 and +/-2 edges instead of
    ESM contact edges.  The model retains residue and atom message passing,
    ligand-to-protein multi-head interaction attention, and a binary head.
    """

    def __init__(self, protein_width=1280, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        self.protein_in = nn.Linear(protein_width, hidden)
        self.ligand_in = nn.Linear(78, hidden)
        self.protein_layers = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(2)])
        self.ligand_layers = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(2)])
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    @staticmethod
    def _message(nodes, edge_index, layer):
        if edge_index.numel() == 0:
            aggregate = torch.zeros_like(nodes)
        else:
            source, destination = edge_index
            aggregate = torch.zeros_like(nodes)
            aggregate.index_add_(0, destination, nodes[source])
            degree = torch.zeros(nodes.shape[0], 1, device=nodes.device, dtype=nodes.dtype)
            degree.index_add_(0, destination, torch.ones(destination.shape[0], 1, device=nodes.device, dtype=nodes.dtype))
            aggregate = aggregate / degree.clamp_min(1.0)
        return F.gelu(layer(torch.cat([nodes, aggregate], dim=1)))

    def forward_one(self, item):
        protein = self.protein_in(item["protein_embedding"])
        ligand = self.ligand_in(item["ligand_features"])
        for layer in self.protein_layers:
            protein = self._message(protein, item["protein_edge_index"], layer)
        for layer in self.ligand_layers:
            ligand = self._message(ligand, item["ligand_edge_index"], layer)
        attended, _ = self.cross_attention(ligand[None], protein[None], protein[None])
        ligand_pool = self.norm(ligand + attended[0]).mean(dim=0)
        protein_pool = protein.mean(dim=0)
        fused = torch.cat([ligand_pool, protein_pool, ligand_pool * protein_pool, (ligand_pool - protein_pool).abs()])
        return self.output(fused).reshape(())

    def forward(self, items):
        return torch.stack([self.forward_one(item) for item in items])

