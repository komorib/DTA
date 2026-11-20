"""Initialize exactly as the eight-block v3, then remove only its last four main blocks."""

from __future__ import annotations

from torch import nn

from dta.lp_pdbbind_fullchain_esm_msa_all_v3.model import FullChainESMMSAAllV3

from . import CANONICAL_LAYERS, MAIN_LAYERS, VERSION


class FullChainAffinityDepth4V1(FullChainESMMSAAllV3):
    version = VERSION

    def __init__(self, **kwargs):
        if kwargs.get("layers") != MAIN_LAYERS:
            raise ValueError("depth4 model requires layers=4 in its saved model_kwargs")
        # Constructing the complete random graph first preserves the initial
        # weights of all retained modules, including heads built after blocks.
        super().__init__(**{**kwargs, "layers": CANONICAL_LAYERS})
        if len(self.blocks) != CANONICAL_LAYERS:
            raise ValueError("canonical model did not create eight main blocks")
        self.blocks = nn.ModuleList(list(self.blocks[:MAIN_LAYERS]))


__all__ = ["FullChainAffinityDepth4V1"]
