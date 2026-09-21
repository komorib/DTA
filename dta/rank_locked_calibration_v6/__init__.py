"""Rank-locked numeric calibration over the immutable v5 static anchor."""

VERSION = "rank-locked-calibration-v6"
EXPERT_ORDER = ("upper_bound", "boltz_pairwise", "three_dprot")
FORMAL_COUNTS = {"train": 11037, "valid": 2340, "test": 4751}
CANDIDATE_NAMES = (
    "identity",
    "affine_mse",
    "affine_huber",
    "spline_8_knots",
    "spline_16_knots",
    "conditional_seed_1",
    "conditional_seed_7",
    "conditional_seed_21",
    "conditional_ensemble",
)
VALIDATION_GATE = {"CI": 0.713374, "RMSE": 1.35, "MAE": 1.05}
TEST_GATE = {"CI": 0.708700, "RMSE": 1.497357, "MAE": 1.174794}


__all__ = [
    "VERSION",
    "EXPERT_ORDER",
    "FORMAL_COUNTS",
    "CANDIDATE_NAMES",
    "VALIDATION_GATE",
    "TEST_GATE",
]
