import numpy as np

from dta.optional_structure_sealed_residual_v29.binary_route import (
    affinity_only_metrics,
    frozen_binary_metrics,
    load_aligned_binary_route,
)


def _route_arrays():
    sample_ids = np.asarray(("lp-0", "lp-1", "d-0", "d-1", "k-0", "k-1"))
    labels = np.asarray((5.0, 6.0, 7.0, 8.0, 11.0, 13.0), dtype=np.float64)
    dataset_index = np.asarray((0, 0, 1, 1, 2, 2), dtype=np.int8)
    affinity = np.linspace(0.0, 1.0, sample_ids.size)
    scores = np.asarray((np.nan, np.nan, 0.1, 0.9, 0.2, 0.8))
    predictions = np.asarray((-1, -1, 0, 1, 0, 1), dtype=np.int8)
    return sample_ids, labels, dataset_index, affinity, scores, predictions


def test_frozen_binary_route_aligns_and_preserves_v26_decisions(tmp_path):
    sample_ids, labels, dataset_index, affinity, scores, predictions = _route_arrays()
    path = tmp_path / "v26_anchor.npz"
    np.savez_compressed(
        path,
        sample_ids=sample_ids,
        labels=labels,
        dataset_index=dataset_index,
        predictions=affinity,
        binary_scores=scores,
        binary_predictions=predictions,
    )
    order = np.asarray((5, 2, 0, 4, 3, 1))
    route = load_aligned_binary_route(
        path,
        sample_ids[order],
        labels[order].astype(np.float32).astype(np.float64),
        dataset_index[order],
    )
    np.testing.assert_allclose(route["scores"], scores[order], equal_nan=True)
    np.testing.assert_array_equal(route["predictions"], predictions[order])


def test_frozen_binary_metrics_use_saved_decisions_not_affinity_thresholds():
    _, labels, dataset_index, _, scores, predictions = _route_arrays()
    metrics = frozen_binary_metrics(labels, dataset_index, scores, predictions)
    for name in ("DAVIS", "KIBA"):
        for key in ("ACC", "Precision", "Recall", "F1", "MCC", "ROC-AUC", "PR-AUC"):
            assert metrics[name][key] == 1.0


def test_affinity_only_metrics_removes_derived_binary_block():
    metrics = {
        "DAVIS": {
            "CI": 0.7,
            "RMSE": 1.0,
            "MAE": 0.8,
            "Pearson": 0.6,
            "Spearman": 0.5,
            "binary": {"PR-AUC": 0.1},
        }
    }
    assert affinity_only_metrics(metrics) == {
        "DAVIS": {
            "CI": 0.7,
            "RMSE": 1.0,
            "MAE": 0.8,
            "Pearson": 0.6,
            "Spearman": 0.5,
        }
    }
