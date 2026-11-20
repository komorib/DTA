"""Read MSA only through attention over every complete-chain residue."""

from __future__ import annotations

import torch

from dta.lp_pdbbind_fullchain_esm_msa_cross_v1.model import FullChainESMMSACrossV1
from dta.lp_pdbbind_fullchain_msa_replacement_v1.model import (
    FullChainEvolutionaryEncoderV1,
    FullChainMSAReplacementV1,
)

from . import VERSION


class FullChainESMMSAAllV2(FullChainESMMSACrossV1):
    """Keep local geometry, but make its evolutionary-like path ESM-only.

    The upstream windowed MSA embedding enters only through attention to all
    complete-chain residues. There is no gathered pocket-MSA tensor or weight
    transfer from the historical pocket-MSA model.
    """

    version = VERSION
    context_attention_name = "fullchain_attention"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_pair_encoder = FullChainEvolutionaryEncoderV1(
            width=int(kwargs.get("msa_width", 128)),
            layers=int(kwargs.get("msa_layers", 4)),
            heads=int(kwargs.get("msa_heads", 4)),
            pair_factor=int(kwargs.get("pair_factor", 8)),
            pair_width=int(kwargs.get("pair_width", 64)),
            dropout=float(kwargs.get("dropout", 0.10)),
            interface_blocks=int(kwargs.get("interface_blocks", 2)),
            fullchain_width=640,
            input_key="pocket_esm",
        )
        self.fullchain_attention = self.nonpocket_attention
        del self.nonpocket_attention

    def _prepare_context_memory(self, source, full_mask, batch):
        """Identity by default; opt-in subclasses may extend only the memory."""
        return source, full_mask

    def _tokens(self, batch, evolutionary):
        states, mask, token_types, gate = FullChainMSAReplacementV1._tokens(
            self, batch, evolutionary,
        )
        full_mask = batch.get("fullchain_mask")
        esm = batch.get("fullchain_esm")
        msa = batch.get("fullchain_msa")
        if (full_mask is None or esm is None or msa is None
                or full_mask.ndim != 2 or full_mask.dtype != torch.bool
                or esm.shape[:2] != full_mask.shape
                or msa.shape[:2] != full_mask.shape
                or esm.shape[-1] != 640 or msa.shape[-1] != 768
                or not bool(full_mask.any(dim=1).all())):
            raise ValueError("complete-chain ESM/MSA context is missing or misaligned")
        if not torch.isfinite(esm).all() or not torch.isfinite(msa).all():
            raise ValueError("complete-chain ESM/MSA context contains nonfinite values")
        source = self.full_fusion_norm(
            self.full_esm_projection(esm.float()) + self.full_msa_projection(msa.float())
        )
        source = source.masked_fill(~full_mask[..., None], 0.0)
        source, memory_mask = self._prepare_context_memory(source, full_mask, batch)
        context, _weights = self.fullchain_attention(
            states, source, source, key_padding_mask=~memory_mask,
            need_weights=False,
        )
        states = states + context * mask[..., None].to(context.dtype)
        return self.nonpocket_norm(states) * mask[..., None].to(states.dtype), mask, token_types, gate

    def forward(self, batch):
        if "fullchain_msa_pocket" in batch:
            raise ValueError("gathered pocket MSA is forbidden in the all-chain model")
        return super().forward(batch)


__all__ = ["FullChainESMMSAAllV2"]
