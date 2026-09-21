import numpy as np

from dta.rank_unlocked_fusion_v7.selection import (
    apply_selection,
    build_oof_selection,
    verify_oof_coverage,
)


def test_oof_selection_prefers_rank_improving_residual():
    labels = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    anchor = np.asarray([1.0, 3.0, 2.0, 4.0], dtype=np.float64)
    seed_predictions = {
        1: np.asarray([1.0, 2.1, 2.9, 4.0], dtype=np.float64),
        7: np.asarray([1.1, 2.0, 3.0, 3.9], dtype=np.float64),
        21: np.asarray([0.9, 2.0, 3.1, 4.0], dtype=np.float64),
    }

    selection = build_oof_selection(anchor, seed_predictions, labels)
    prediction = apply_selection(anchor, seed_predictions, selection)

    assert selection["oof_metrics"]["CI"] > selection["anchor_metrics"]["CI"]
    assert np.all(np.diff(prediction) > 0.0)
    assert selection["residual_scale"] > 0.0


def test_oof_coverage_requires_one_prediction_per_row_and_seed():
    coverage = {
        1: np.ones(5, dtype=np.int64),
        7: np.ones(5, dtype=np.int64),
        21: np.ones(5, dtype=np.int64),
    }
    verify_oof_coverage(coverage, row_count=5)

    coverage[7][2] = 2
    try:
        verify_oof_coverage(coverage, row_count=5)
    except ValueError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("duplicate OOF coverage was accepted")
