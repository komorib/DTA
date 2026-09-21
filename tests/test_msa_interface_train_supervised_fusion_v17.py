import json
from argparse import Namespace

import numpy as np
import pytest

from dta.msa_interface_train_supervised_fusion_v17 import FEATURE_VERSION, VERSION
from dta.msa_interface_train_supervised_fusion_v17 import run_features as feature_runner
from dta.msa_interface_train_supervised_fusion_v17.run_fusion import (
    BLEND_WEIGHTS,
    FAMILIES,
    HEAD_SPECS,
    _family_raw,
    _fit_transform,
    _gate,
    _normalize_fold_ids,
    run,
)
from utils.experiment_integrity import sha256_file


def _features(path, keys, labels, pretrain, affinity):
    np.savez_compressed(
        path,
        keys=np.asarray(keys),
        labels=np.asarray(labels, dtype=np.float64),
        pretrain_features=np.asarray(pretrain, dtype=np.float32),
        affinity_features=np.asarray(affinity, dtype=np.float32),
    )


def _anchor(path, keys, labels, predictions, folds=None):
    values = {
        "keys": np.asarray(keys),
        "labels": np.asarray(labels, dtype=np.float64),
        "v12_base_predictions": np.asarray(predictions, dtype=np.float64) + 0.03,
        "v15_anchor_predictions": np.asarray(predictions, dtype=np.float64) - 0.02,
        "predictions": np.asarray(predictions, dtype=np.float64),
    }
    if folds is not None:
        values["fold_ids"] = np.asarray(folds, dtype=np.int64)
    np.savez_compressed(path, **values)


def _args(features, anchor, output, *, selection=None, minimum_gain=-1.0):
    return Namespace(
        feature_root=str(features),
        anchor_root=str(anchor),
        output=str(output),
        validation_only=selection is None,
        selection_result=None if selection is None else str(selection),
        limit_train=None,
        limit_valid=None,
        limit_test=None,
        pca_components=[4, 6],
        minimum_validation_ci_gain=minimum_gain,
        minimum_spearman_gain=-1.0,
        maximum_rmse_increase=100.0,
        minimum_positive_folds=0,
        selection_estimators=5,
        final_estimators=7,
        hist_max_iter=12,
        n_jobs=1,
        seed=1701,
    )


def _synthetic_artifacts(tmp_path):
    generator = np.random.RandomState(1711)
    train_count, valid_count = 90, 50
    train_x = generator.normal(size=(train_count, 10))
    valid_x = generator.normal(size=(valid_count, 10))
    train_y = 6.0 + 1.3 * train_x[:, 0] - 0.9 * train_x[:, 1]
    valid_y = 6.0 + 1.3 * valid_x[:, 0] - 0.9 * valid_x[:, 1]
    train_y += generator.normal(0.0, 0.05, size=train_count)
    valid_y += generator.normal(0.0, 0.05, size=valid_count)
    train_keys = np.asarray(["tr%d" % i for i in range(train_count)])
    valid_keys = np.asarray(["va%d" % i for i in range(valid_count)])
    features = tmp_path / "features"
    anchor = tmp_path / "anchor"
    features.mkdir()
    anchor.mkdir()
    _features(
        features / "train_features.npz",
        train_keys,
        train_y,
        train_x,
        train_x + generator.normal(0.0, 0.15, size=train_x.shape),
    )
    _features(
        features / "valid_features.npz",
        valid_keys,
        valid_y,
        valid_x,
        valid_x + generator.normal(0.0, 0.15, size=valid_x.shape),
    )
    folds = np.asarray([3, 8, 14, 21, 34])[np.arange(valid_count) % 5]
    _anchor(
        anchor / "valid_predictions.npz",
        valid_keys,
        valid_y,
        valid_y + generator.normal(0.0, 0.8, size=valid_count),
        folds,
    )
    return generator, features, anchor


def test_v17_inventory_feature_families_and_folds():
    assert FAMILIES == ("pretrain", "affinity", "delta", "joint")
    assert tuple(spec["kind"] for spec in HEAD_SPECS) == (
        "extra_trees",
        "extra_trees",
        "hist_gradient",
        "pls",
    )
    assert BLEND_WEIGHTS[0] == 0.0 and BLEND_WEIGHTS[-1] == 1.0
    pretrain = np.arange(24, dtype=np.float32).reshape(6, 4)
    affinity = pretrain + 2.0
    values = {"pretrain_features": pretrain, "affinity_features": affinity}
    np.testing.assert_array_equal(_family_raw(values, "pretrain"), pretrain)
    np.testing.assert_array_equal(_family_raw(values, "affinity"), affinity)
    np.testing.assert_allclose(_family_raw(values, "delta"), 2.0)
    assert _family_raw(values, "joint").shape == (6, 8)
    folds = _normalize_fold_ids(np.asarray([3, 8, 3, 21, 8, 21]), (6,))
    np.testing.assert_array_equal(folds, np.asarray([0, 1, 0, 2, 1, 2]))


def test_transform_is_fitted_on_train_only():
    train = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    valid = np.asarray([[100.0, 200.0], [101.0, 201.0]], dtype=np.float32)
    transform, train_z, valid_z = _fit_transform(train, valid, 2, 17)
    np.testing.assert_allclose(transform["scaler"].mean_, np.asarray([2.0, 3.0]))
    assert train_z.shape == (3, 2)
    assert valid_z.shape == (2, 2)


def test_gate_requires_positive_weight_and_all_conditions():
    selected = {
        "head_weight": 0.3,
        "positive_folds": 4,
        "metrics": {"CI": 0.702, "Spearman": 0.601, "RMSE": 1.202},
    }
    anchor = {"CI": 0.700, "Spearman": 0.600, "RMSE": 1.200}
    args = Namespace(
        minimum_validation_ci_gain=0.001,
        minimum_spearman_gain=0.0,
        maximum_rmse_increase=0.003,
        minimum_positive_folds=4,
    )
    assert _gate(selected, anchor, args)["passed"]
    assert not _gate(dict(selected, head_weight=0.0), anchor, args)["passed"]
    assert not _gate(dict(selected, positive_folds=3), anchor, args)["passed"]
    changed = dict(selected, metrics=dict(selected["metrics"], CI=0.7005))
    assert not _gate(changed, anchor, args)["passed"]


def test_validation_failure_never_reads_test(tmp_path):
    _, features, anchor = _synthetic_artifacts(tmp_path)
    output = tmp_path / "failed"
    result = run(_args(features, anchor, output, minimum_gain=2.0))
    assert result["status"] == "validation_failed"
    assert not (features / "test_features.npz").exists()
    assert not (anchor / "test_predictions.npz").exists()
    assert not (output / "test_predictions.npz").exists()


def test_validation_then_sealed_final_refit(tmp_path):
    generator, features, anchor = _synthetic_artifacts(tmp_path)
    selection_output = tmp_path / "selection"
    selection = run(_args(features, anchor, selection_output))
    assert selection["status"] == "validation_passed"
    assert selection["selected"]["head_weight"] > 0.0
    assert not (selection_output / "test_predictions.npz").exists()

    test_count = 28
    test_x = generator.normal(size=(test_count, 10))
    test_y = 6.0 + 1.3 * test_x[:, 0] - 0.9 * test_x[:, 1]
    test_y += generator.normal(0.0, 0.05, size=test_count)
    test_keys = np.asarray(["te%d" % i for i in range(test_count)])
    _features(
        features / "test_features.npz",
        test_keys,
        test_y,
        test_x,
        test_x + generator.normal(0.0, 0.15, size=test_x.shape),
    )
    _anchor(
        anchor / "test_predictions.npz",
        test_keys,
        test_y,
        test_y + generator.normal(0.0, 0.8, size=test_count),
    )
    with (features / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": FEATURE_VERSION,
                "status": "complete",
                "selection_result_sha256": sha256_file(
                    selection_output / "result.json"
                ),
                "test_feature_sha256": sha256_file(
                    features / "test_features.npz"
                ),
            },
            handle,
        )
    final_output = tmp_path / "final"
    final = run(
        _args(
            features,
            anchor,
            final_output,
            selection=selection_output / "result.json",
        )
    )
    assert final["status"] in ("passed", "not_improved")
    assert final["counts"]["test"] == test_count
    assert final["counts"]["final_refit"] == 140
    assert (final_output / "model.joblib").is_file()
    assert (final_output / "test_predictions.npz").is_file()


def test_wrong_selection_sha_is_rejected_before_test(tmp_path):
    _, features, anchor = _synthetic_artifacts(tmp_path)
    selection_output = tmp_path / "selection"
    run(_args(features, anchor, selection_output))
    selection_path = selection_output / "result.json"
    selection = json.loads(selection_path.read_text())
    selection["inputs"]["v17_train_features_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="SHA256"):
        run(_args(features, anchor, tmp_path / "rejected", selection=selection_path))
    assert not (anchor / "test_predictions.npz").exists()


def test_feature_validation_constructs_only_train_and_valid(tmp_path, monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, manifest, cache, row_cache, limit):
            calls.append((manifest, cache))
            self.manifest = manifest

        def __len__(self):
            return 3

    def fake_values(pretrain, affinity, dataset, args):
        base = 0.0 if "train" in dataset.manifest else 1.0
        return {
            "keys": tuple("%s_%d" % (dataset.manifest, i) for i in range(3)),
            "labels": np.asarray([4.0, 5.0, 6.0]),
            "pretrain_features": np.full((3, 5), base, dtype=np.float32),
            "affinity_features": np.full((3, 5), base + 0.5, dtype=np.float32),
        }

    monkeypatch.setattr(feature_runner, "MSAPairComplexDataset", FakeDataset)
    monkeypatch.setattr(
        feature_runner,
        "_load_and_validate_sources",
        lambda args: (
            object(),
            object(),
            {
                "pretrain": {"sha256": "1" * 64},
                "affinity": {"sha256": "2" * 64},
            },
        ),
    )
    monkeypatch.setattr(feature_runner, "extract_dataset_features", fake_values)
    args = Namespace(
        inference_subsets=1,
        test_only=False,
        output=str(tmp_path / "feature_output"),
        train_manifest="train_manifest",
        valid_manifest="valid_manifest",
        test_manifest="test_manifest",
        train_cache="train_cache",
        valid_cache="valid_cache",
        test_cache="test_cache",
        row_cache="rows",
        row_depth=4,
        limit_train=None,
        limit_valid=None,
        limit_test=None,
    )
    result = feature_runner.run(args)
    assert result["status"] == "validation_ready"
    assert calls == [
        ("train_manifest", "train_cache"),
        ("valid_manifest", "valid_cache"),
    ]
    assert not (tmp_path / "feature_output" / "test_features.npz").exists()


def test_feature_test_only_rejects_wrong_train_sha_before_dataset(
    tmp_path, monkeypatch
):
    output = tmp_path / "features"
    output.mkdir()
    keys = np.asarray(["a", "b", "c"])
    labels = np.asarray([1.0, 2.0, 3.0])
    values = np.zeros((3, 4), dtype=np.float32)
    _features(output / "train_features.npz", keys, labels, values, values)
    _features(output / "valid_features.npz", keys, labels, values, values)
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": FEATURE_VERSION,
                "status": "validation_ready",
                "train_feature_sha256": sha256_file(
                    output / "train_features.npz"
                ),
                "valid_feature_sha256": sha256_file(
                    output / "valid_features.npz"
                ),
            },
            handle,
        )
    selection_path = tmp_path / "selection.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": VERSION,
                "status": "validation_passed",
                "inputs": {
                    "v17_train_features_sha256": "0" * 64,
                    "v17_valid_features_sha256": sha256_file(
                        output / "valid_features.npz"
                    ),
                },
            },
            handle,
        )
    monkeypatch.setattr(
        feature_runner,
        "MSAPairComplexDataset",
        lambda *args, **kwargs: pytest.fail("test dataset was constructed"),
    )
    with pytest.raises(ValueError, match="train_features_sha256"):
        feature_runner._run_test_only(
            Namespace(output=str(output), selection_result=str(selection_path))
        )
