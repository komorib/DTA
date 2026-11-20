"""Minimal MSA Transformer feature-extraction smoke path."""

from .msa import (
    ENCODER_NAME,
    ESM_SOURCE_REVISION,
    PreparedMSA,
    collect_targets,
    load_msa_manifest,
    prepare_msa,
)

__all__ = [
    "ENCODER_NAME",
    "ESM_SOURCE_REVISION",
    "PreparedMSA",
    "collect_targets",
    "load_msa_manifest",
    "prepare_msa",
]
