"""Directly expose non-pocket full-chain ESM+MSA context to geometry tokens."""

from __future__ import annotations

import torch
from torch import nn

from dta.lp_pdbbind_fullchain_msa_replacement_v1.model import FullChainMSAReplacementV1
from . import VERSION


class FullChainESMMSACrossV1(FullChainMSAReplacementV1):
    version = VERSION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = int(kwargs.get("model_width", 256))
        heads = int(kwargs.get("heads", 8))
        self.full_esm_projection = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, width))
        self.full_msa_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.full_fusion_norm = nn.LayerNorm(width)
        self.nonpocket_attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.nonpocket_norm = nn.LayerNorm(width)

    def _tokens(self, batch, evolutionary):
        states, mask, token_types, gate = super()._tokens(batch, evolutionary)
        full_mask = batch.get("fullchain_mask")
        nonpocket = batch.get("fullchain_nonpocket_mask")
        esm = batch.get("fullchain_esm")
        msa = batch.get("fullchain_msa")
        if (full_mask is None or nonpocket is None or esm is None or msa is None
                or esm.shape[:2] != full_mask.shape or msa.shape[:2] != full_mask.shape
                or esm.shape[-1] != 640 or msa.shape[-1] != 768
                or nonpocket.shape != full_mask.shape
                or bool(torch.any(nonpocket & ~full_mask))):
            raise ValueError("complete-chain ESM/MSA context is missing or misaligned")
        if not torch.isfinite(esm).all() or not torch.isfinite(msa).all():
            raise ValueError("complete-chain ESM/MSA context contains nonfinite values")
        source = self.full_fusion_norm(
            self.full_esm_projection(esm.float()) + self.full_msa_projection(msa.float())
        )
        # A complex may have no residues outside its structural pocket. Keep a
        # safe dummy key for attention, then zero that sample's context.
        has_context = nonpocket.any(dim=1)
        safe = nonpocket.clone()
        safe[~has_context, 0] = True
        source = source.masked_fill(~nonpocket[..., None], 0.0)
        context, _weights = self.nonpocket_attention(
            states, source, source, key_padding_mask=~safe, need_weights=False,
        )
        context = context * has_context[:, None, None].to(context.dtype)
        states = (states + context * mask[..., None].to(context.dtype))
        return self.nonpocket_norm(states) * mask[..., None].to(states.dtype), mask, token_types, gate


__all__ = ["FullChainESMMSACrossV1"]
