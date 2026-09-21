import torch
import torch.nn as nn
import torch.nn.functional as F


class BoltzInterfaceHead(nn.Module):
    """Pool frozen Boltz pocket/ligand embeddings into a local-plus-global latent."""

    def __init__(self, emb_dim=384, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.pocket_proj = nn.Sequential(nn.Linear(self.emb_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        self.ligand_proj = nn.Sequential(nn.Linear(self.emb_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        self.pair_value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pair_score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.pair_delta_gate = nn.Parameter(torch.ones(self.hidden_dim) * 0.1)
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(self.emb_dim * 2),
            nn.Linear(self.emb_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    @property
    def output_dim(self):
        return self.hidden_dim * 2

    def forward(self, batch):
        pocket_emb = batch["pocket_emb"].float()
        ligand_emb = batch["ligand_emb"].float()
        pocket_mask = batch["pocket_mask"].bool()
        ligand_mask = batch["ligand_mask"].bool()
        global_feat = batch["global_feat"].float()

        if pocket_emb.ndim != 3 or ligand_emb.ndim != 3:
            raise ValueError("Boltz interface embeddings must have shape [B, N, D]")
        if pocket_emb.shape[-1] != self.emb_dim or ligand_emb.shape[-1] != self.emb_dim:
            raise ValueError(
                f"Boltz interface embedding dim must be {self.emb_dim}: "
                f"pocket={pocket_emb.shape[-1]} ligand={ligand_emb.shape[-1]}"
            )
        if global_feat.shape[-1] != self.emb_dim * 2:
            raise ValueError(
                f"Boltz interface global feature dim must be {self.emb_dim * 2}, got {global_feat.shape[-1]}"
            )

        pocket = self.pocket_proj(pocket_emb) * pocket_mask.unsqueeze(-1).to(pocket_emb.dtype)
        ligand = self.ligand_proj(ligand_emb) * ligand_mask.unsqueeze(-1).to(ligand_emb.dtype)
        pair = pocket.unsqueeze(2) * ligand.unsqueeze(1)
        pair_value = self.pair_value(pair)
        pair_mask = pocket_mask.unsqueeze(2) & ligand_mask.unsqueeze(1)

        pair_logits = self.pair_score(pair).squeeze(-1).masked_fill(~pair_mask, float("-inf"))
        pair_attention = torch.softmax(pair_logits.flatten(1), dim=-1).view_as(pair_logits)
        pair_attention = torch.nan_to_num(pair_attention, nan=0.0)
        attended_pair = (pair_attention.unsqueeze(-1) * pair_value).sum(dim=(1, 2))

        mask_float = pair_mask.unsqueeze(-1).to(pair_value.dtype)
        mean_pair = (pair_value * mask_float).sum(dim=(1, 2)) / mask_float.sum(dim=(1, 2)).clamp_min(1.0)
        local = mean_pair + self.pair_delta_gate * (attended_pair - mean_pair)
        return torch.cat((local, self.global_mlp(global_feat)), dim=-1)
