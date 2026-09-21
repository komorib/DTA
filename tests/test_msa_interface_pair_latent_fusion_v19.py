from argparse import Namespace
import json

import numpy as np
import pytest

from dta.msa_interface_pair_latent_fusion_v19.run_features import (
    FEATURE_VERSION,
    _pair_latents,
    _run_test_only,
)
from dta.msa_interface_pair_latent_fusion_v19.run_fusion import (
    _promotion_gate,
    run,
)


def _anchor(path, keys, labels, predictions, fold_ids=None):
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


def _features(path, keys, labels, predictions, latents):
    np.savez_compressed(
        path,
        keys=keys,
        labels=labels,
        v18_predictions=predictions,
        pair_latents=latents,
    )


def _args(anchor, features, output, selection=None, minimum_gain=-1.0):
    return Namespace(
        anchor_root=str(anchor),
        v19_root=str(features),
        output=str(output),
        selection_result=None if selection is None else str(selection),
        limit=None,
        pca_components=4,
        minimum_validation_ci_gain=minimum_gain,
        minimum_spearman_gain=-1.0,
        maximum_rmse_increase=999.0,
        minimum_positive_folds=0,
        oof_estimators=8,
        final_estimators=12,
        n_jobs=1,
        seed=17,
    )


def _validation_inputs(tmp_path, seed=11):
    generator = np.random.RandomState(seed)
    count = 60
    keys = np.asarray(["v%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    latents = generator.normal(size=(count, 8))
    anchor_predictions = labels + 0.8 * latents[:, 0] + generator.normal(
        0.0, 0.15, size=count
    )
    expert_predictions = labels + generator.normal(0.0, 0.5, size=count)
    anchor = tmp_path / "anchor"
    features = tmp_path / "features"
    anchor.mkdir()
    features.mkdir()
    fold_values = np.asarray([2, 5, 9, 14, 22], dtype=np.int64)
    _anchor(
        anchor / "valid_predictions.npz",
        keys,
        labels,
        anchor_predictions,
        fold_values[np.arange(count) % 5],
    )
    _features(
        features / "valid_features.npz",
        keys,
        labels,
        expert_predictions,
        latents,
    )
    return generator, anchor, features


def test_pair_latents_extract_only_second_half():
    values = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    extracted = _pair_latents(values, 4)
    np.testing.assert_array_equal(extracted, values[:, 4:])
    with pytest.raises(ValueError, match="latent width"):
        _pair_latents(values[:, :7], 4)


def test_validation_failure_does_not_read_test(tmp_path):
    _, anchor, features = _validation_inputs(tmp_path)
    output = tmp_path / "failed"
    result = run(_args(anchor, features, output, minimum_gain=2.0))
    assert result["status"] == "validation_failed"
    assert not (anchor / "test_predictions.npz").exists()
    assert not (features / "test_features.npz").exists()
    assert not (output / "test_predictions.npz").exists()


def test_validation_selection_and_final_test_are_separate(tmp_path):
    generator, anchor, features = _validation_inputs(tmp_path)
    selection_output = tmp_path / "selection"
    selection = run(_args(anchor, features, selection_output))
    assert selection["status"] == "validation_passed"
    assert selection["selected_residual"]["budget"] > 0.0
    assert not (selection_output / "test_predictions.npz").exists()

    count = 30
    keys = np.asarray(["t%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    latents = generator.normal(size=(count, 8))
    _anchor(
        anchor / "test_predictions.npz",
        keys,
        labels,
        labels + 0.8 * latents[:, 0] + generator.normal(0.0, 0.15, size=count),
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
            anchor,
            features,
            final_output,
            selection_output / "result.json",
        )
    )
    assert final["status"] in ("passed", "not_improved")
    assert final["count"] == count
    assert (final_output / "model.joblib").is_file()
    assert (final_output / "test_predictions.npz").is_file()


def test_final_stage_rejects_changed_selection_sha(tmp_path):
    _, anchor, features = _validation_inputs(tmp_path)
    selection_output = tmp_path / "selection"
    selection = run(_args(anchor, features, selection_output))
    assert selection["status"] == "validation_passed"
    result_path = selection_output / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["inputs"]["v19_valid_sha256"] = "0" * 64
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        run(_args(anchor, features, tmp_path / "final", result_path))
    assert not any((tmp_path / "final").iterdir())


def test_test_inference_rejects_wrong_selection_version(tmp_path):
    output = tmp_path / "features"
    output.mkdir()
    valid_path = output / "valid_features.npz"
    _features(
        valid_path,
        np.asarray(["a", "b"]),
        np.asarray([1.0, 2.0]),
        np.asarray([1.1, 1.9]),
        np.zeros((2, 4), dtype=np.float32),
    )
    (output / "result.json").write_text(
        json.dumps({"status": "validation_complete", "version": FEATURE_VERSION}),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"status": "validation_passed", "version": "wrong"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        _run_test_only(Namespace(output=str(output), selection_result=str(selection)))


def test_promotion_gate_enforces_all_criteria():
    labels = np.tile(np.arange(5, dtype=np.float64), 5)
    anchor = np.tile(np.asarray([0.0, 2.0, 1.0, 4.0, 3.0]), 5)
    candidate = labels.copy()
    folds = np.repeat(np.arange(5, dtype=np.int64), 5)
    values = {
        "minimum_validation_ci_gain": 0.01,
        "minimum_spearman_gain": 0.0,
        "maximum_rmse_increase": 0.0,
        "minimum_positive_folds": 5,
    }
    assert _promotion_gate(
        labels, anchor, candidate, folds, 0.1, Namespace(**values)
    )["passed"]
    assert not _promotion_gate(
        labels, anchor, candidate, folds, 0.0, Namespace(**values)
    )["passed"]
    for name, value in (
        ("minimum_validation_ci_gain", 1.0),
        ("minimum_spearman_gain", 1.0),
        ("maximum_rmse_increase", -2.0),
        ("minimum_positive_folds", 6),
    ):
        changed = dict(values)
        changed[name] = value
        assert not _promotion_gate(
            labels, anchor, candidate, folds, 0.1, Namespace(**changed)
        )["passed"]
