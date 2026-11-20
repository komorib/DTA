"""The v2 all-residue model graph with a separately identified training run."""

from dta.lp_pdbbind_fullchain_esm_msa_all_v2.model import FullChainESMMSAAllV2

from . import VERSION


class FullChainESMMSAAllV3(FullChainESMMSAAllV2):
    version = VERSION


__all__ = ["FullChainESMMSAAllV3"]
