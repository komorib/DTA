import json
from types import SimpleNamespace

import numpy as np
import pytest

from dta.optional_structure_sealed_residual_v29.data import DATASET_INDEX
from dta.optional_structure_sealed_residual_v29.run_test import run as run_test
from dta.optional_structure_sealed_residual_v29.selection import (
    apply_sealed_blends,
    load_aligned_predictions,
    select_validation_blends,
)
from dta.optional_structure_sealed_residual_v29.training import VERSION


def _validation_arrays(count=120):
    labels, anchor, candidate, sample_ids, dataset_index = [], [], [], [], []
    for name, index in DATASET_INDEX.items():
        target = np.linspace(-2.0, 2.0, count)
        noise = 0.30 * np.sin(np.arange(count) * 1.7 + index)
        labels.append(target)
        anchor.append(target + noise)
        if name == "LP_PDBBind":
            candidate.append(target + 0.02 * np.cos(np.arange(count)))
        else:
            candidate.append(-target)
        sample_ids.extend("%s:%04d" % (name, value) for value in range(count))
        dataset_index.extend([index] * count)
    return (
        np.asarray(sample_ids),
        np.concatenate(labels),
        np.concatenate(anchor),
        np.concatenate(candidate),
        np.asarray(dataset_index),
    )


def test_validation_only_blend_promotes_only_lp_structure_residual():
    sample_ids, labels, anchor, candidate, dataset_index = _validation_arrays()
    result = select_validation_blends(
        sample_ids,
        labels,
        dataset_index,
        anchor,
        candidate,
        minimum_ci_gain=0.0001,
        minimum_positive_folds=3,
    )
    assert result["any_promoted"]
    assert result["eligible_datasets"] == ["LP_PDBBind"]
    assert result["datasets"]["LP_PDBBind"]["passed"]
    for name in ("DAVIS", "KIBA"):
        assert result["datasets"][name]["effective_v29_weight"] == 0.0
        assert result["datasets"][name]["reason"] == "outside_v29_lp_structure_scope"
    applied = apply_sealed_blends(dataset_index, anchor, candidate, result)
    np.testing.assert_allclose(applied, result["predictions"])
    for name, index in DATASET_INDEX.items():
        if name != "LP_PDBBind":
            np.testing.assert_array_equal(applied[dataset_index == index], anchor[dataset_index == index])


def test_failed_lp_candidate_falls_back_to_anchor_exactly():
    sample_ids, labels, anchor, _, dataset_index = _validation_arrays()
    candidate = -labels
    result = select_validation_blends(sample_ids, labels, dataset_index, anchor, candidate)
    assert not result["any_promoted"]
    assert all(
        entry["effective_v29_weight"] == 0.0
        for entry in result["datasets"].values()
    )
    np.testing.assert_array_equal(result["predictions"], anchor)


def test_prediction_alignment_accepts_smoke_subset_and_checks_labels(tmp_path):
    sample_ids, labels, anchor, _, dataset_index = _validation_arrays(count=10)
    path = tmp_path / "anchor.npz"
    np.savez_compressed(
        path,
        sample_ids=sample_ids,
        labels=labels,
        dataset_index=dataset_index,
        predictions=anchor,
    )
    order = np.asarray([0, 3, 12, 24])
    aligned = load_aligned_predictions(
        path, sample_ids[order], labels[order], dataset_index[order]
    )
    np.testing.assert_allclose(aligned, anchor[order])
    wrong = labels[order].copy()
    wrong[0] += 1.0
    with pytest.raises(ValueError, match="labels differ"):
        load_aligned_predictions(path, sample_ids[order], wrong, dataset_index[order])


def test_prediction_alignment_accepts_float32_label_round_trip(tmp_path):
    sample_ids, labels, anchor, _, dataset_index = _validation_arrays(count=10)
    source_labels = labels + np.where(np.arange(labels.size) % 2, 4.8e-7, -4.8e-7)
    path = tmp_path / "float64_anchor.npz"
    np.savez_compressed(
        path,
        sample_ids=sample_ids,
        labels=source_labels,
        dataset_index=dataset_index,
        predictions=anchor,
    )
    aligned = load_aligned_predictions(
        path,
        sample_ids,
        labels.astype(np.float32).astype(np.float64),
        dataset_index,
    )
    np.testing.assert_allclose(aligned, anchor)


def test_validation_failure_prevents_test_input_access(tmp_path):
    output = tmp_path / "selection"
    output.mkdir()
    (output / "selection.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "phase": "validation_selection",
                "status": "validation_failed",
                "validation_selected": False,
            }
        ),
        encoding="utf-8",
    )
    arguments = SimpleNamespace(
        selection_output=output,
        test_manifest=tmp_path / "must_not_be_read.jsonl",
        anchor_test=tmp_path / "must_not_be_read.npz",
        v22_test=tmp_path / "must_not_be_read_v22.npz",
        device="cpu",
        num_workers=0,
        evaluation_batch_size=1,
        limit_test=1,
    )
    with pytest.raises(ValueError, match="validation gate did not pass"):
        run_test(arguments)
