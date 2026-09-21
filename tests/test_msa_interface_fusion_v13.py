from argparse import Namespace

import json

import numpy as np
import pytest

from dta.msa_interface_pretrained_v13.run_fusion import run
from dta.msa_interface_pretrained_v13.run_training import _run_test_only


def _anchor(path, keys, labels, predictions, fold_ids=None):
    values = {
        "keys": keys,
        "labels": labels,
        "v11_fusion_predictions": predictions + 0.03,
        "v12_predictions": predictions - 0.04,
        "predictions": predictions,
    }
    if fold_ids is not None:
        values["fold_ids"] = fold_ids
    np.savez_compressed(path, **values)


def _expert(path, keys, labels, predictions, latents):
    np.savez_compressed(
        path,
        keys=keys,
        labels=labels,
        predictions=predictions,
        latents=latents,
    )


def test_v13_fusion_runs_end_to_end(tmp_path):
    generator = np.random.RandomState(31)
    valid_count, test_count = 60, 30
    valid_labels = generator.normal(6.5, 1.2, size=valid_count)
    test_labels = generator.normal(6.5, 1.2, size=test_count)
    valid_latents = generator.normal(size=(valid_count, 12))
    test_latents = generator.normal(size=(test_count, 12))
    valid_anchor = valid_labels + generator.normal(0.0, 0.45, size=valid_count)
    test_anchor = test_labels + generator.normal(0.0, 0.45, size=test_count)
    valid_expert = valid_labels + 0.15 * valid_latents[:, 0] + generator.normal(
        0.0, 0.35, size=valid_count
    )
    test_expert = test_labels + 0.15 * test_latents[:, 0] + generator.normal(
        0.0, 0.35, size=test_count
    )
    valid_keys = np.asarray(["v%d" % index for index in range(valid_count)])
    test_keys = np.asarray(["t%d" % index for index in range(test_count)])
    anchor = tmp_path / "v11_fusion"
    v13 = tmp_path / "v13"
    output = tmp_path / "output"
    anchor.mkdir()
    v13.mkdir()
    _anchor(
        anchor / "valid_predictions.npz",
        valid_keys,
        valid_labels,
        valid_anchor,
        np.asarray([2, 4, 7, 9, 11], dtype=np.int64)[
            np.arange(valid_count) % 5
        ],
    )
    _anchor(anchor / "test_predictions.npz", test_keys, test_labels, test_anchor)
    _expert(
        v13 / "valid_predictions.npz",
        valid_keys,
        valid_labels,
        valid_expert,
        valid_latents,
    )
    _expert(
        v13 / "test_predictions.npz",
        test_keys,
        test_labels,
        test_expert,
        test_latents,
    )
    result = run(
        Namespace(
            anchor_root=str(anchor),
            v13_root=str(v13),
            output=str(output),
            limit=None,
            pca_components=6,
            minimum_validation_ci_gain=-1.0,
            oof_estimators=8,
            final_estimators=12,
            n_jobs=1,
            seed=7,
        )
    )
    assert result["count"] == test_count
    assert result["status"] in ("passed", "not_improved")
    assert (output / "result.json").is_file()
    assert (output / "model.joblib").is_file()



def _fusion_args(anchor, v13, output, minimum_gain):
    return Namespace(
        anchor_root=str(anchor),
        v13_root=str(v13),
        output=str(output),
        limit=None,
        pca_components=6,
        minimum_validation_ci_gain=minimum_gain,
        oof_estimators=8,
        final_estimators=12,
        n_jobs=1,
        seed=13,
    )


def _validation_artifacts(tmp_path, seed=53):
    generator = np.random.RandomState(seed)
    count = 45
    keys = np.asarray(["v%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    latents = generator.normal(size=(count, 10))
    anchor_predictions = labels + generator.normal(0.0, 0.5, size=count)
    expert_predictions = labels + generator.normal(0.0, 0.4, size=count)
    anchor = tmp_path / "anchor"
    v13 = tmp_path / "v13"
    anchor.mkdir()
    v13.mkdir()
    fold_values = np.asarray([3, 8, 14, 21, 34], dtype=np.int64)
    _anchor(
        anchor / "valid_predictions.npz",
        keys,
        labels,
        anchor_predictions,
        fold_values[np.arange(count) % fold_values.size],
    )
    _expert(
        v13 / "valid_predictions.npz",
        keys,
        labels,
        expert_predictions,
        latents,
    )
    return generator, anchor, v13


def test_validation_failed_does_not_require_or_read_test_artifacts(tmp_path):
    _, anchor, v13 = _validation_artifacts(tmp_path)
    output = tmp_path / "failed"
    result = run(_fusion_args(anchor, v13, output, minimum_gain=2.0))
    assert result["status"] == "validation_failed"
    assert not (output / "test_predictions.npz").exists()
    assert not (output / "model.joblib").exists()
    assert not (anchor / "test_predictions.npz").exists()
    assert not (v13 / "test_predictions.npz").exists()


def test_validation_selection_then_final_test_are_separate_stages(tmp_path):
    generator, anchor, v13 = _validation_artifacts(tmp_path)
    selection_output = tmp_path / "selection"
    selection = run(_fusion_args(anchor, v13, selection_output, minimum_gain=-1.0))
    assert selection["status"] == "validation_passed"
    assert not (selection_output / "test_predictions.npz").exists()

    count = 24
    keys = np.asarray(["t%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    latents = generator.normal(size=(count, 10))
    _anchor(
        anchor / "test_predictions.npz",
        keys,
        labels,
        labels + generator.normal(0.0, 0.5, size=count),
    )
    _expert(
        v13 / "test_predictions.npz",
        keys,
        labels,
        labels + generator.normal(0.0, 0.4, size=count),
        latents,
    )
    final_output = tmp_path / "final"
    final = run(_fusion_args(anchor, v13, final_output, minimum_gain=-1.0))
    assert final["status"] in ("passed", "not_improved")
    assert final["count"] == count
    assert (final_output / "model.joblib").is_file()
    assert (final_output / "test_predictions.npz").is_file()


def test_test_only_rejects_selection_with_wrong_validation_sha(tmp_path):
    output = tmp_path / "expert"
    output.mkdir()
    (output / "best.pt").write_bytes(b"placeholder")
    np.savez_compressed(
        output / "valid_predictions.npz",
        keys=np.asarray(["a", "b"]),
        labels=np.asarray([1.0, 2.0]),
        predictions=np.asarray([1.1, 1.9]),
        latents=np.zeros((2, 3), dtype=np.float32),
    )
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "validation_complete"}, handle)
    selection_path = tmp_path / "selection.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "validation_passed",
                "inputs": {"v13_valid_sha256": "0" * 64},
            },
            handle,
        )
    with pytest.raises(ValueError, match="SHA256"):
        _run_test_only(
            Namespace(output=str(output), selection_result=str(selection_path))
        )
