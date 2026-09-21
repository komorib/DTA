"""Repeated grouped-OOF rank fusion for LP-PDBbind v8."""

VERSION = "repeated-rank-fusion-v8"
FOLD_REPEATS = (
    {"name": "repeat_137", "fold_seed": 137, "model_seed": 87},
    {"name": "repeat_271", "fold_seed": 271, "model_seed": 137},
    {"name": "repeat_389", "fold_seed": 389, "model_seed": 271},
)
STACKER_REGULARIZATIONS = (0.01, 0.05, 0.10, 0.50)


__all__ = [
    "FOLD_REPEATS",
    "STACKER_REGULARIZATIONS",
    "VERSION",
]
