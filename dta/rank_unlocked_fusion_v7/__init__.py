"""Rank-unlocked expert fusion for LP-PDBbind v7."""

VERSION = "rank-unlocked-fusion-v7"
EXPERT_ORDER = ("upper_bound", "boltz_pairwise", "three_dprot")
LATENT_WIDTHS = (1024, 640, 1024)
META_TRAIN_COUNT = 2340
TEST_COUNT = 4751


__all__ = [
    "EXPERT_ORDER",
    "LATENT_WIDTHS",
    "META_TRAIN_COUNT",
    "TEST_COUNT",
    "VERSION",
]
