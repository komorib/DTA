"""Original depth-4 FullChain plus one early MoLFormer K/V token."""

import torch
from torch import nn

from dta.lp_pdbbind_fullchain_affinity_depth4_v1.model import FullChainAffinityDepth4V1
from . import VERSION


class FullChainMoLFormerEarlyKVV1(FullChainAffinityDepth4V1):
    """Only early K/V fusion; no late molecule branch is present."""

    version = VERSION

    def __init__(self, *, molecule_kv_enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.molecule_kv_enabled = bool(molecule_kv_enabled)
        width = int(self.fullchain_attention.embed_dim)
        with torch.random.fork_rng(devices=[]):
            self.early_projection = nn.Sequential(
                nn.LayerNorm(768), nn.Linear(768, width), nn.LayerNorm(width)
            )

    def _prepare_context_memory(self, source, full_mask, batch):
        source, full_mask = super()._prepare_context_memory(source, full_mask, batch)
        if not self.molecule_kv_enabled:
            return source, full_mask
        value = batch.get("molformer_molecule")
        valid = (torch.is_tensor(value) and value.shape == (source.shape[0], 768)
                 and value.dtype == torch.float32 and value.device == source.device
                 and bool(torch.isfinite(value).all()))
        if not valid:
            raise ValueError("molformer_molecule must be finite FP32 [B,768]")
        token = self.early_projection(value).to(source.dtype)
        memory = torch.cat((source, token[:, None, :]), dim=1)
        token_mask = torch.ones((source.shape[0], 1), dtype=torch.bool,
                                device=full_mask.device)
        return memory, torch.cat((full_mask, token_mask), dim=1)


__all__ = ["FullChainMoLFormerEarlyKVV1"]
