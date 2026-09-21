import json
from types import SimpleNamespace

import numpy as np
import pytest

from dta.optional_structure_sequence_router_v24 import VERSION
from dta.optional_structure_sequence_router_v24.run_selection import build_parser
from dta.optional_structure_sequence_router_v24.run_test import run as run_test
from dta.optional_structure_sequence_router_v24.sequence_expert import (
    _decontaminated_rows,
    apply_affinity_expression,
    apply_binary_expression,
    select_affinity_expression,
    select_binary_expression,
)


def _row(split, index, sequence_id, ligand_id, label):
    return {
        "sample_id": "DAVIS:%s:%d" % (split, index),
        "dataset": "DAVIS",
        "split": split,
        "sequence_id": sequence_id,
        "ligand_id": ligand_id,
        "pair_id": "%s|%s" % (sequence_id, ligand_id),
        "sequence": "ACDEFGHIK",
        "smiles": "CCO",
        "label": float(label),
    }


def test_sequence_expert_excludes_exact_evaluation_pairs_without_labels():
    train = [
        _row("train", 0, "s0", "l0", 1.0),
        _row("train", 1, "s0", "l1", 2.0),
        _row("train", 2, "s1", "l0", 3.0),
        _row("train", 3, "s1", "l1", 4.0),
        _row("train", 4, "s2", "l2", 5.0),
    ]
    evaluation = [_row("valid", 0, "s0", "l0", 999.0)]
    filtered, excluded = _decontaminated_rows(train, evaluation)
    assert excluded == 1
    assert all(
        (row["sequence_id"], row["ligand_id"]) != ("s0", "l0")
        for row in filtered
    )
    evaluation[0]["label"] = -999.0
    filtered_again, excluded_again = _decontaminated_rows(train, evaluation)
    assert excluded_again == excluded
    assert [row["sample_id"] for row in filtered_again] == [
        row["sample_id"] for row in filtered
    ]


def test_affinity_expression_selection_and_application_are_identical():
    labels = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    predictions = {
        "v22": labels[::-1],
        "reg_all": labels.copy(),
        "bias_svd_rank8": labels + 0.1,
    }
    selected, expected, candidates = select_affinity_expression(labels, predictions)
    actual = apply_affinity_expression(selected, predictions)
    np.testing.assert_allclose(actual, expected)
    assert selected["metrics"]["CI"] == 1.0
    assert len(candidates) > len(predictions)


def test_binary_expression_and_threshold_are_validation_selected():
    labels = np.asarray([5.0, 5.5, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0])
    scores = {
        "v22": np.asarray([0.4] * 8),
        "cls_all": np.asarray([0.01, 0.02, 0.03, 0.04, 0.8, 0.85, 0.9, 0.95]),
        "bias_svd_rank8": np.asarray([0.1, 0.1, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9]),
    }
    selected, expected, _ = select_binary_expression(
        labels, scores, affinity_threshold=7.522878745280337
    )
    actual = apply_binary_expression(selected, scores)
    np.testing.assert_allclose(actual, expected)
    assert selected["PR-AUC"] == 1.0
    assert "decision_threshold" in selected


def test_selection_cli_cannot_accept_test_inputs():
    parser = build_parser()
    options = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--test-manifest" not in options
    assert "--v22-test" not in options
    assert "--v20-test" not in options


def test_failed_validation_gate_blocks_test_before_paths_are_read(tmp_path):
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
        test_manifest=tmp_path / "missing-test.jsonl",
        v22_test=tmp_path / "missing-v22.npz",
        v20_test=tmp_path / "missing-v20.npz",
    )
    with pytest.raises(ValueError, match="validation gate did not pass"):
        run_test(args)
