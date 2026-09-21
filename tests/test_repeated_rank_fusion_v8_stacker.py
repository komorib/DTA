import numpy as np

from dta.repeated_rank_fusion_v8.stacker import (
    cross_fit_rank_stacker,
    fit_positive_rank_stacker,
    predict_rank_stacker,
)


def test_positive_rank_stacker_prefers_concordant_predictor():
    labels = np.linspace(1.0, 8.0, 24)
    features = np.stack(
        (
            labels + 0.1 * np.sin(labels),
            labels[::-1],
            np.full(labels.shape, labels.mean()),
        ),
        axis=1,
    )
    model = fit_positive_rank_stacker(
        features, labels, regularization=0.05, maximum_pairs=4096, seed=7
    )
    predictions = predict_rank_stacker(features, model)

    assert model["weights"][0] > model["weights"][1]
    assert np.corrcoef(labels, predictions)[0, 1] > 0.99
    assert np.all(np.asarray(model["weights"]) >= 0.0)


def test_cross_fit_rank_stacker_has_exact_oof_coverage():
    labels = np.linspace(1.0, 10.0, 30)
    features = np.stack(
        (labels + 0.2, labels + np.sin(labels), labels[::-1]), axis=1
    )
    fold_ids = np.arange(labels.size) % 5

    result = cross_fit_rank_stacker(
        features,
        labels,
        fold_ids,
        regularization=0.10,
        maximum_pairs=4096,
        seed=21,
    )

    assert result["predictions"].shape == labels.shape
    assert np.isfinite(result["predictions"]).all()
    assert np.all(result["coverage"] == 1)
    assert len(result["fold_models"]) == 5
