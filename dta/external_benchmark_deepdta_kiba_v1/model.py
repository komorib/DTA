from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepDTAClassifier(nn.Module):
    """DeepDTA categorical CNN encoder with a binary interaction head."""

    def __init__(self, drug_vocab: int, protein_vocab: int, embedding_dim: int = 128,
                 filters: int = 32, drug_kernel: int = 4, protein_kernel: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.drug_embedding = nn.Embedding(drug_vocab + 1, embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.drug_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters, drug_kernel),
            nn.Conv1d(filters, filters * 2, drug_kernel),
            nn.Conv1d(filters * 2, filters * 3, drug_kernel),
        ])
        self.protein_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters, protein_kernel),
            nn.Conv1d(filters, filters * 2, protein_kernel),
            nn.Conv1d(filters * 2, filters * 3, protein_kernel),
        ])
        representation_dim = filters * 3 * 2
        self.fc1 = nn.Linear(representation_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _encode(sequence, embedding, convolutions):
        hidden = embedding(sequence).transpose(1, 2)
        for convolution in convolutions:
            hidden = F.relu(convolution(hidden))
        return F.adaptive_max_pool1d(hidden, 1).squeeze(-1)

    def forward(self, drug, protein):
        drug_hidden = self._encode(drug, self.drug_embedding, self.drug_convs)
        protein_hidden = self._encode(protein, self.protein_embedding, self.protein_convs)
        hidden = torch.cat([drug_hidden, protein_hidden], dim=1)
        hidden = self.dropout(F.relu(self.fc1(hidden)))
        hidden = self.dropout(F.relu(self.fc2(hidden)))
        hidden = self.dropout(F.relu(self.fc3(hidden)))
        return self.out(hidden).squeeze(-1)

