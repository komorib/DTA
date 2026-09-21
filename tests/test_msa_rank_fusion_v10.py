import numpy as np

from dta.msa_rank_fusion_v10.run_pipeline import (
    apply_standardized_blend,
    select_residual_model,
    select_standardized_blend,
)


def test_standardized_blend_round_trip():
    labels = np.asarray([1.0, 2.0, 4.0, 7.0, 8.0])
    v8 = labels + np.asarray([0.2, -0.2, 0.1, -0.1, 0.0])
    v9 = labels + np.asarray([-0.3, 0.1, -0.2, 0.3, 0.1])
    selected = select_standardized_blend(v8, v9, labels)
    applied = apply_standardized_blend(v8, v9, selected)
    np.testing.assert_allclose(applied, selected["predictions"])
    assert 0.0 <= selected["weight"] <= 1.0


def test_residual_selection_produces_complete_oof_predictions():
    rng = np.random.RandomState(17)
    features = rng.normal(size=(60, 8))
    labels = features[:, 0] + 0.2 * features[:, 1]
    anchor = labels + 0.4 * features[:, 0]
    folds = np.arange(60) % 5
    selected = select_residual_model(
        features,
        labels,
        anchor,
        folds,
        estimators=8,
        n_jobs=1,
        seed=11,
    )
    assert selected["predictions"].shape == labels.shape
    assert np.isfinite(selected["predictions"]).all()
    assert selected["budget"] in (0.0, 0.025, 0.05, 0.10, 0.20, 0.30)
