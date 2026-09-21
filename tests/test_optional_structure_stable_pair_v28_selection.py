import json
from types import SimpleNamespace

import numpy as np
import pytest

from dta.optional_structure_stable_pair_v28.data import DATASET_INDEX
from dta.optional_structure_stable_pair_v28.run_test import run as run_test
from dta.optional_structure_stable_pair_v28.selection import (
    apply_sealed_blends,
    load_aligned_predictions,
    select_validation_blends,
)
from dta.optional_structure_stable_pair_v28.training import VERSION


def _validation_arrays(count=120):
    labels, anchor, candidate, sample_ids, dataset_index = [], [], [], [], []
    for name, index in DATASET_INDEX.items():
        target = np.linspace(-2.0, 2.0, count)
        noise = 0.30 * np.sin(np.arange(count) * 1.7 + index)
        labels.append(target)
        anchor.append(target + noise)
        candidate.append(target + 0.02 * np.cos(np.arange(count)))
        sample_ids.extend("%s:%04d" % (name, value) for value in range(count))
        dataset_index.extend([index] * count)
    return (
        np.asarray(sample_ids),
        np.concatenate(labels),
        np.concatenate(anchor),
        np.concatenate(candidate),
        np.asarray(dataset_index),
    )


def test_validation_only_blend_promotes_a_consistently_better_v28():
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
    assert all(entry["passed"] for entry in result["datasets"].values())
    applied = apply_sealed_blends(dataset_index, anchor, candidate, result)
    np.testing.assert_allclose(applied, result["predictions"])


def test_failed_candidate_falls_back_to_anchor_exactly():
    sample_ids, labels, anchor, _, dataset_index = _validation_arrays()
    candidate = -labels
    result = select_validation_blends(sample_ids, labels, dataset_index, anchor, candidate)
    assert not result["any_promoted"]
    assert all(entry["effective_v28_weight"] == 0.0 for entry in result["datasets"].values())
    np.testing.assert_array_equal(result["predictions"], anchor)


def test_prediction_alignment_accepts_a_smoke_subset_but_checks_labels(tmp_path):
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
        device="cpu",
        num_workers=0,
        evaluation_batch_size=1,
        limit_test=1,
    )
    with pytest.raises(ValueError, match="validation gate did not pass"):
        run_test(arguments)
