import json
from types import SimpleNamespace

import numpy as np
import pytest

from dta.optional_structure_collaborative_router_v23 import VERSION
from dta.optional_structure_collaborative_router_v23.core import (
    RANKS,
    apply_binary_selection,
    atomic_joblib,
    collaborative_predictions,
    collaborative_validation_gate,
    fit_decontaminated_bias_svd,
    hash_fold_ids,
    load_v20_lp_predictions,
    lp_validation_gate,
    select_binary_blend,
    sha256_file,
)
from dta.optional_structure_collaborative_router_v23.run_selection import build_parser
from dta.optional_structure_collaborative_router_v23.run_test import (
    _verify_selection,
    run as run_test,
)


def _row(dataset, split, index, sequence_id, ligand_id, label, **extra):
    row = {
        "sample_id": "%s:%s:%d" % (dataset, split, index),
        "dataset": dataset,
        "split": split,
        "sequence_id": sequence_id,
        "ligand_id": ligand_id,
        "pair_id": "%s|%s" % (sequence_id, ligand_id),
        "label": float(label),
        "has_3d": False,
    }
    row.update(extra)
    return row


def _davis_grid():
    rows = []
    index = 0
    for sequence_index in range(3):
        for ligand_index in range(3):
            rows.append(
                _row(
                    "DAVIS",
                    "train",
                    index,
                    "s%d" % sequence_index,
                    "l%d" % ligand_index,
                    sequence_index + ligand_index / 10.0,
                )
            )
            index += 1
    return rows


def test_exact_evaluation_pair_is_removed_and_unknown_entity_falls_back():
    train = _davis_grid()
    evaluation = [
        _row("DAVIS", "valid", 0, "s0", "l0", 0.0),
        _row("DAVIS", "valid", 1, "unknown", "l1", 0.0),
    ]
    model, audit = fit_decontaminated_bias_svd(train, evaluation, seed=7)
    assert audit["excluded_exact_evaluation_pairs"] == 1
    assert audit["fit_uses_evaluation_labels"] is False
    fallback = np.asarray([91.0, 92.0])
    prediction, eligible = collaborative_predictions(model, evaluation, fallback, rank=8)
    assert eligible.tolist() == [True, False]
    assert prediction[1] == fallback[1]
    assert np.isfinite(prediction[0])


def test_binary_threshold_is_selected_on_validation_and_reused_unchanged():
    labels = np.asarray([5.0, 5.2, 6.0, 7.0, 8.0, 8.2, 9.0, 10.0])
    base = np.full(labels.shape, 7.0)
    collaborative = labels + np.asarray([0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.1, 0.0])
    by_rank = {rank: collaborative for rank in RANKS}
    selected, _, _, _ = select_binary_blend(
        labels, base, by_rank, affinity_threshold=7.522878745280337, affinity_scale=0.5
    )
    sealed_threshold = selected["decision_threshold"]
    test_base = np.asarray([6.0, 9.0])
    test_collaborative = {rank: np.asarray([5.0, 10.0]) for rank in RANKS}
    score = apply_binary_selection(
        selected,
        test_base,
        test_collaborative,
        affinity_threshold=7.522878745280337,
        affinity_scale=0.5,
    )
    prediction = score >= sealed_threshold
    assert prediction.tolist() == [False, True]
    assert selected["decision_threshold"] == sealed_threshold


def test_hash_fold_gate_requires_consistent_affinity_and_pr_improvement():
    sample_ids = ["sample-%d" % index for index in range(100)]
    folds = hash_fold_ids(sample_ids)
    labels = np.zeros(100, dtype=np.float64)
    for fold in range(5):
        indices = np.flatnonzero(folds == fold)
        labels[indices[1::2]] = 1.0
    base_prediction = 1.0 - labels
    selected_prediction = labels.copy()
    gate = collaborative_validation_gate(
        sample_ids,
        labels,
        labels.astype(np.int8),
        base_prediction,
        selected_prediction,
        base_prediction,
        selected_prediction,
        minimum_ci_gain=0.01,
        minimum_pr_auc_gain=0.02,
        minimum_positive_folds=4,
    )
    assert gate["passed"]
    assert gate["positive_ci_folds"] == 5
    assert gate["positive_pr_auc_folds"] == 5
    failed = collaborative_validation_gate(
        sample_ids,
        labels,
        labels.astype(np.int8),
        selected_prediction,
        selected_prediction,
        selected_prediction,
        selected_prediction,
        minimum_ci_gain=0.01,
        minimum_pr_auc_gain=0.02,
        minimum_positive_folds=4,
    )
    assert not failed["passed"]


def test_lp_gate_never_promotes_below_v20_ci():
    anchor = {"CI": 0.736, "RMSE": 1.20}
    assert lp_validation_gate(anchor, {"CI": 0.736, "RMSE": 1.202})["passed"]
    assert not lp_validation_gate(anchor, {"CI": 0.7359, "RMSE": 1.19})["passed"]


def test_v20_structure_keys_align_and_missing_structure_falls_back(tmp_path):
    rows = [
        _row(
            "LP_PDBBind",
            "valid",
            0,
            "s0",
            "l0",
            7.0,
            has_3d=True,
            structure_key="key0",
        ),
        _row("LP_PDBBind", "valid", 1, "s1", "l1", 8.0),
    ]
    path = tmp_path / "v20.npz"
    np.savez_compressed(
        path,
        keys=np.asarray(["key0"]),
        labels=np.asarray([7.0]),
        predictions=np.asarray([7.5]),
    )
    _, predictions, available = load_v20_lp_predictions(path, rows)
    assert available.tolist() == [True, False]
    assert predictions[0] == 7.5
    assert np.isnan(predictions[1])


def test_selection_cli_has_no_test_input():
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--test-manifest" not in option_strings
    assert "--v22-test" not in option_strings
    assert "--v20-test" not in option_strings


def test_validation_failure_prevents_any_test_input_access(tmp_path):
    output = tmp_path / "failed"
    output.mkdir()
    (output / "selection.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "phase": "validation_selection_sealed",
                "status": "validation_failed",
                "validation_selected": False,
                "test_inputs_read": False,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        selection_output=output,
        test_manifest=tmp_path / "must-not-be-read.jsonl",
        v22_test=tmp_path / "must-not-be-read-v22.npz",
        v20_test=tmp_path / "must-not-be-read-v20.npz",
    )
    with pytest.raises(ValueError, match="validation gate did not pass"):
        run_test(args)


def test_bad_selection_bundle_checksum_is_rejected_before_test(tmp_path):
    output = tmp_path / "sealed"
    output.mkdir()
    bundle = output / "selection_bundle.joblib"
    atomic_joblib(
        bundle,
        {
            "version": VERSION,
            "phase": "validation_selection_sealed",
        },
    )
    valid = output / "valid_predictions.npz"
    valid.write_bytes(b"valid")
    inputs = {}
    for name in ("train_manifest", "valid_manifest", "v22_valid", "v20_valid"):
        path = output / name
        path.write_bytes(name.encode("ascii"))
        inputs[name] = str(path)
        inputs[name + "_sha256"] = sha256_file(path)
    (output / "selection.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "phase": "validation_selection_sealed",
                "status": "validation_passed",
                "validation_selected": True,
                "test_inputs_read": False,
                "selection_bundle": str(bundle),
                "selection_bundle_sha256": "bad",
                "valid_predictions": str(valid),
                "valid_predictions_sha256": sha256_file(valid),
                "inputs": inputs,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        _verify_selection(output)
