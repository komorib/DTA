"""Verify the V26 validation seal, then report test without reselection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.sequence_smiles_binary_probe_v1.core import binary_metrics

from . import VERSION
from .core import (
    apply_lp_selection,
    atomic_json,
    atomic_npz,
    dataset_indices,
    group_counts,
    load_v20_lp_predictions,
    load_v22_predictions,
    load_v25_anchor_predictions,
    metrics_by_dataset,
    read_manifest,
    rows_for_dataset,
    safe_average_precision,
    safe_roc_auc,
    sha256_file,
)
from .sequence_expert import (
    apply_affinity_expression,
    apply_binary_expression,
    fit_sequence_expert,
    predict_sequence_expert,
)


def _verify_selection(selection_output):
    output = Path(selection_output).expanduser().resolve()
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != VERSION:
        raise ValueError("V26 selection version mismatch")
    if selection.get("phase") != "validation_selection_sealed":
        raise ValueError("V26 selection phase mismatch")
    if selection.get("status") != "validation_passed" or not selection.get(
        "validation_selected"
    ):
        raise ValueError("V26 validation gate did not pass")
    if selection.get("test_inputs_read") is not False:
        raise ValueError("V26 selection does not assert validation-only access")
    bundle_path = Path(selection["selection_bundle"])
    if sha256_file(bundle_path) != selection["selection_bundle_sha256"]:
        raise ValueError("sealed V26 selection bundle checksum mismatch")
    valid_predictions = Path(selection["valid_predictions"])
    if sha256_file(valid_predictions) != selection["valid_predictions_sha256"]:
        raise ValueError("sealed V26 validation predictions checksum mismatch")
    for name in ("train_manifest", "valid_manifest", "v22_valid", "v20_valid", "v25_valid"):
        path = Path(selection["inputs"][name])
        if sha256_file(path) != selection["inputs"][name + "_sha256"]:
            raise ValueError("sealed V26 %s checksum mismatch" % name)
    bundle = joblib.load(bundle_path)
    if bundle.get("version") != VERSION:
        raise ValueError("V26 bundle version mismatch")
    if bundle.get("phase") != "validation_selection_sealed":
        raise ValueError("V26 bundle phase mismatch")
    return selection_path, selection, bundle


def run(args):
    output = Path(args.selection_output).expanduser().resolve()
    selection_path, selection, bundle = _verify_selection(output)
    result_path = output / "result.json"
    prediction_path = output / "test_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite an existing V26 test report")

    train_path, train_rows = read_manifest(selection["inputs"]["train_manifest"])
    if sha256_file(train_path) != selection["inputs"]["train_manifest_sha256"]:
        raise ValueError("V26 train manifest changed after seal verification")

    print("V26 validation seal verified; reading test inputs now", flush=True)
    test_path, test_rows = read_manifest(args.test_manifest)
    if any(str(row.get("split", "test")) != "test" for row in test_rows):
        raise ValueError("V26 test manifest contains a non-test row")
    v22_path, v22 = load_v22_predictions(args.v22_test, test_rows)
    v25_path, v25 = load_v25_anchor_predictions(args.v25_test, test_rows)

    combined_predictions = np.asarray(v22["base_predictions"], dtype=np.float64).copy()
    binary_scores = np.full(len(test_rows), np.nan, dtype=np.float64)
    binary_predictions = np.full(len(test_rows), -1, dtype=np.int8)
    collaborative_eligible = np.zeros(len(test_rows), dtype=bool)
    dataset_results = {}

    for dataset in ("DAVIS", "KIBA"):
        train_dataset = rows_for_dataset(train_rows, dataset)
        test_dataset = rows_for_dataset(test_rows, dataset)
        indices = dataset_indices(test_rows, dataset)
        config = bundle["dataset_config"][dataset]
        base = np.asarray(v22["base_predictions"], dtype=np.float64)[indices]
        print("[V26 %s] refitting sealed train-only expert for test report" % dataset, flush=True)
        fitted, fit_audit = fit_sequence_expert(
            train_dataset,
            test_dataset,
            seed=int(config["fit_seed"]),
            max_iter=int(config["max_iter"]),
        )
        fitted["affinity_scale"] = float(config["affinity_scale"])
        affinity_predictions, score_predictions, eligible = predict_sequence_expert(
            fitted, test_dataset, base
        )
        affinity_prediction = apply_affinity_expression(
            config["affinity"], affinity_predictions
        )
        score = apply_binary_expression(config["binary"], score_predictions)
        threshold = float(config["binary"]["decision_threshold"])
        labels = np.asarray([row["label"] for row in test_dataset], dtype=np.float64)
        binary_label = (
            labels >= float(config["affinity_threshold"])
        ).astype(np.int8)
        combined_predictions[indices] = affinity_prediction
        binary_scores[indices] = score
        binary_predictions[indices] = (score >= threshold).astype(np.int8)
        collaborative_eligible[indices] = eligible
        dataset_results[dataset] = {
            "fit_audit": fit_audit,
            "entity_coverage": {
                "bias_svd_eligible_both_entities_seen": int(eligible.sum()),
                "bias_svd_fallback_unknown_entity": int((~eligible).sum()),
            },
            "selected_affinity_config": config["affinity"],
            "selected_binary_config": config["binary"],
            "affinity_metrics": affinity_metrics(labels, affinity_prediction),
            "v22_base_affinity_metrics": affinity_metrics(
                labels, affinity_predictions["v22"]
            ),
            "binary_metrics": binary_metrics(binary_label, score, threshold),
            "v22_base_binary_ranking": {
                "PR-AUC": safe_average_precision(
                    binary_label, score_predictions["v22"]
                ),
                "ROC-AUC": safe_roc_auc(binary_label, score_predictions["v22"]),
            },
            "v25_anchor_affinity_metrics": affinity_metrics(
                labels, v25["predictions"][indices]
            ),
            "v25_anchor_binary_ranking": {
                "PR-AUC": safe_average_precision(
                    binary_label, v25["binary_scores"][indices]
                ),
                "ROC-AUC": safe_roc_auc(
                    binary_label, v25["binary_scores"][indices]
                ),
            },
            "test_configuration_reselected": False,
        }
        del fitted

    lp_indices = dataset_indices(test_rows, "LP_PDBBind")
    lp_rows = rows_for_dataset(test_rows, "LP_PDBBind")
    v20_path, v20_predictions, v20_available = load_v20_lp_predictions(
        args.v20_test, lp_rows
    )
    lp_config = bundle["lp_config"]["affinity"]
    lp_prediction = apply_lp_selection(
        lp_config,
        v22["predictions"][lp_indices],
        v22["base_predictions"][lp_indices],
        v20_predictions,
        v20_available,
    )
    combined_predictions[lp_indices] = lp_prediction
    lp_labels = np.asarray([row["label"] for row in lp_rows], dtype=np.float64)
    lp_anchor = np.asarray(v22["base_predictions"], dtype=np.float64)[lp_indices].copy()
    lp_anchor[v20_available] = v20_predictions[v20_available]
    dataset_results["LP_PDBBind"] = {
        "v20_available": int(v20_available.sum()),
        "v22_fallback": int((~v20_available).sum()),
        "selected_affinity_config": lp_config,
        "affinity_metrics": affinity_metrics(lp_labels, lp_prediction),
        "v20_route_anchor_metrics": affinity_metrics(lp_labels, lp_anchor),
        "v25_anchor_affinity_metrics": affinity_metrics(
            lp_labels, v25["predictions"][lp_indices]
        ),
        "test_configuration_reselected": False,
    }

    atomic_npz(
        prediction_path,
        sample_ids=np.asarray([row["sample_id"] for row in test_rows]),
        labels=np.asarray([row["label"] for row in test_rows], dtype=np.float64),
        dataset_index=np.asarray(
            [
                0
                if row["dataset"] == "LP_PDBBind"
                else 1
                if row["dataset"] == "DAVIS"
                else 2
                for row in test_rows
            ],
            dtype=np.int8,
        ),
        predictions=combined_predictions,
        v22_base_predictions=v22["base_predictions"],
        v22_full_predictions=v22["predictions"],
        binary_scores=binary_scores,
        binary_predictions=binary_predictions,
        collaborative_eligible=collaborative_eligible,
    )
    result = {
        "version": VERSION,
        "phase": "test_report",
        "status": "reported",
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selection_bundle_sha256": selection["selection_bundle_sha256"],
        "test_manifest": str(test_path),
        "test_manifest_sha256": sha256_file(test_path),
        "v22_test": str(v22_path),
        "v22_test_sha256": sha256_file(v22_path),
        "v20_test": str(v20_path),
        "v20_test_sha256": sha256_file(v20_path),
        "v25_test": str(v25_path),
        "v25_test_sha256": sha256_file(v25_path),
        "counts": {"test": group_counts(test_rows)},
        "dataset_results": dataset_results,
        "test_metrics": metrics_by_dataset(test_rows, combined_predictions),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "test_used_for_selection": False,
        "test_configuration_reselected": False,
        "exact_pair_decontamination_uses_test_labels": False,
    }
    atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--v22-test", type=Path, required=True)
    parser.add_argument("--v20-test", type=Path, required=True)
    parser.add_argument("--v25-test", type=Path, required=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
