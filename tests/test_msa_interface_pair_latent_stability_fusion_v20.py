import json
from argparse import Namespace

import numpy as np
import pytest

from dta.msa_interface_pair_latent_stability_fusion_v20.run_fusion import (
    MAX_FEATURE_FRACTIONS,
    MINIMUM_LEAVES,
    RESIDUAL_BUDGETS,
    TREE_SEED_STRIDE,
    _fit_transform,
    _fold_ids,
    _gate,
    run,
)


def _base(path, keys, labels, predictions, fold_ids=None):
    values = {
        "keys": keys,
        "labels": labels,
        "v12_base_predictions": predictions + 0.05,
        "v15_anchor_predictions": predictions - 0.03,
        "predictions": predictions,
    }
    if fold_ids is not None:
        values["fold_ids"] = fold_ids
    np.savez_compressed(path, **values)


def _benchmark(path, keys, labels, predictions, fold_ids=None):
    values = {
        "keys": keys,
        "labels": labels,
        "v16_anchor_predictions": predictions - 0.04,
        "predictions": predictions,
    }
    if fold_ids is not None:
        values["fold_ids"] = fold_ids
    np.savez_compressed(path, **values)


def _features(path, keys, labels, predictions, latents):
    np.savez_compressed(
        path,
        keys=keys,
        labels=labels,
        v18_predictions=predictions,
        pair_latents=np.asarray(latents, dtype=np.float32),
    )


def _args(base, benchmark, features, output, selection=None, minimum_gain=-1.0):
    return Namespace(
        base_root=str(base),
        benchmark_root=str(benchmark),
        feature_root=str(features),
        output=str(output),
        validation_only=selection is None,
        selection_result=None if selection is None else str(selection),
        limit=None,
        pca_components=4,
        minimum_validation_ci_gain=minimum_gain,
        minimum_spearman_gain=-1.0,
        maximum_rmse_increase=999.0,
        minimum_positive_folds=0,
        allow_zero_budget=False,
        oof_estimators=2,
        final_estimators=3,
        n_jobs=1,
        seed=23,
    )


def _validation_inputs(tmp_path, seed=29):
    generator = np.random.RandomState(seed)
    count = 75
    keys = np.asarray(["v%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    latents = generator.normal(size=(count, 8))
    base_predictions = labels + 0.9 * latents[:, 0] + generator.normal(
        0.0, 0.1, size=count
    )
    benchmark_predictions = base_predictions + generator.normal(0.0, 0.1, size=count)
    expert_predictions = labels + generator.normal(0.0, 0.5, size=count)
    base = tmp_path / "base"
    benchmark = tmp_path / "benchmark"
    features = tmp_path / "features"
    base.mkdir()
    benchmark.mkdir()
    features.mkdir()
    fold_values = np.asarray([2, 5, 9, 14, 22], dtype=np.int64)
    folds = fold_values[np.arange(count) % fold_values.size]
    _base(base / "valid_predictions.npz", keys, labels, base_predictions, folds)
    _benchmark(
        benchmark / "valid_predictions.npz",
        keys,
        labels,
        benchmark_predictions,
        folds,
    )
    _features(
        features / "valid_features.npz",
        keys,
        labels,
        expert_predictions,
        latents,
    )
    return generator, base, benchmark, features


def test_v20_search_space_is_frozen_to_validation_audit():
    assert MINIMUM_LEAVES == (2, 3, 5, 7, 10, 15, 20, 30, 40)
    assert MAX_FEATURE_FRACTIONS == (0.5, 0.7, 0.85, 1.0)
    assert RESIDUAL_BUDGETS[0] == 0.0
    assert RESIDUAL_BUDGETS[-1] == 1.0
    assert 0.375 in RESIDUAL_BUDGETS
    assert len(RESIDUAL_BUDGETS) == 41
    assert TREE_SEED_STRIDE == 100


def test_raw_pair_pca_and_non_contiguous_fold_mapping():
    labels = np.arange(6, dtype=np.float64)
    base = {
        "labels": labels,
        "v12_base_predictions": labels + 0.1,
        "v15_anchor_predictions": labels - 0.2,
        "predictions": labels + 0.3,
    }
    features = {
        "labels": labels,
        "v18_predictions": labels - 0.4,
        "pair_latents": np.arange(48, dtype=np.float64).reshape(6, 8),
    }
    transform, matrix = _fit_transform(base, features, 3, 31)
    assert transform["components"] == 3
    assert matrix.shape == (6, 7)
    assert np.isfinite(matrix).all()
    folds = _fold_ids(np.asarray([3, 8, 3, 21, 8, 21]), (6,))
    np.testing.assert_array_equal(folds, np.asarray([0, 1, 0, 2, 1, 2]))


def test_gate_compares_candidate_with_v19_benchmark():
    labels = np.tile(np.arange(5, dtype=np.float64), 5)
    benchmark = np.tile(np.asarray([0.0, 2.0, 1.0, 4.0, 3.0]), 5)
    candidate = labels.copy()
    folds = np.repeat(np.arange(5, dtype=np.int64), 5)
    args = Namespace(
        allow_zero_budget=False,
        minimum_validation_ci_gain=0.01,
        minimum_spearman_gain=0.0,
        maximum_rmse_increase=0.0,
        minimum_positive_folds=5,
    )
    assert _gate(labels, benchmark, candidate, folds, 0.1, args)["passed"]
    assert not _gate(labels, benchmark, candidate, folds, 0.0, args)["passed"]


def test_validation_failure_does_not_read_test(tmp_path):
    _, base, benchmark, features = _validation_inputs(tmp_path)
    output = tmp_path / "failed"
    result = run(_args(base, benchmark, features, output, minimum_gain=2.0))
    assert result["status"] == "validation_failed"
    assert not (base / "test_predictions.npz").exists()
    assert not (benchmark / "test_predictions.npz").exists()
    assert not (features / "test_features.npz").exists()
    assert not (output / "test_predictions.npz").exists()


def test_validation_selection_and_final_test_are_separate(tmp_path):
    generator, base, benchmark, features = _validation_inputs(tmp_path)
    selection_output = tmp_path / "selection"
    selection = run(_args(base, benchmark, features, selection_output))
    assert selection["status"] == "validation_passed"
    assert selection["selected_residual"]["budget"] > 0.0
    assert not (selection_output / "test_predictions.npz").exists()

    count = 35
    keys = np.asarray(["t%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    latents = generator.normal(size=(count, 8))
    base_predictions = labels + 0.9 * latents[:, 0] + generator.normal(
        0.0, 0.1, size=count
    )
    _base(base / "test_predictions.npz", keys, labels, base_predictions)
    _benchmark(
        benchmark / "test_predictions.npz",
        keys,
        labels,
        base_predictions + generator.normal(0.0, 0.1, size=count),
    )
    _features(
        features / "test_features.npz",
        keys,
        labels,
        labels + generator.normal(0.0, 0.5, size=count),
        latents,
    )
    final_output = tmp_path / "final"
    final = run(
        _args(
            base,
            benchmark,
            features,
            final_output,
            selection_output / "result.json",
        )
    )
    assert final["status"] in ("passed", "not_improved")
    assert final["count"] == count
    assert (final_output / "model.joblib").is_file()
    assert (final_output / "test_predictions.npz").is_file()
    with np.load(final_output / "test_predictions.npz") as archive:
        assert set(archive.files) == {
            "keys",
            "labels",
            "v16_base_predictions",
            "v19_benchmark_predictions",
            "predictions",
        }


def test_final_rejects_changed_validation_sha_before_test(tmp_path):
    _, base, benchmark, features = _validation_inputs(tmp_path)
    selection_output = tmp_path / "selection"
    run(_args(base, benchmark, features, selection_output))
    result_path = selection_output / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["inputs"]["v19_features_valid_sha256"] = "0" * 64
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        run(
            _args(
                base,
                benchmark,
                features,
                tmp_path / "final",
                result_path,
            )
        )
    assert not (features / "test_features.npz").exists()
