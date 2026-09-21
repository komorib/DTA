"""Select and seal the V26 sparse simplex router on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dta.rank_fusion_v4.metrics import affinity_metrics

from . import VERSION
from .core import (
    BINARY_THRESHOLDS,
    atomic_joblib,
    atomic_json,
    atomic_npz,
    collaborative_validation_gate,
    dataset_indices,
    group_counts,
    load_v20_lp_predictions,
    load_v22_predictions,
    load_v25_anchor_predictions,
    lp_validation_gate,
    metrics_by_dataset,
    read_manifest,
    rows_for_dataset,
    safe_average_precision,
    safe_roc_auc,
    select_lp_blend,
    sha256_file,
)
from .sequence_expert import (
    AFFINITY_COMPONENTS,
    BINARY_COMPONENTS,
    SIMPLEX_UNITS,
    fit_sequence_expert,
    predict_sequence_expert,
    select_affinity_expression,
    select_binary_expression,
)


def _prepare_output(path):
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty V26 output: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(args):
    output = _prepare_output(args.output)
    train_path, train_rows = read_manifest(args.train_manifest)
    valid_path, valid_rows = read_manifest(args.valid_manifest)
    if any(str(row.get("split", "train")) != "train" for row in train_rows):
        raise ValueError("V26 train manifest contains a non-train row")
    if any(str(row.get("split", "valid")) != "valid" for row in valid_rows):
        raise ValueError("V26 validation manifest contains a non-validation row")
    v22_path, v22 = load_v22_predictions(args.v22_valid, valid_rows)
    v25_path, v25 = load_v25_anchor_predictions(args.v25_valid, valid_rows)

    combined_predictions = np.asarray(v22["base_predictions"], dtype=np.float64).copy()
    binary_scores = np.full(len(valid_rows), np.nan, dtype=np.float64)
    binary_predictions = np.full(len(valid_rows), -1, dtype=np.int8)
    collaborative_eligible = np.zeros(len(valid_rows), dtype=bool)
    dataset_results = {}
    sealed_dataset_config = {}

    for offset, dataset in enumerate(("DAVIS", "KIBA")):
        train_dataset = rows_for_dataset(train_rows, dataset)
        valid_dataset = rows_for_dataset(valid_rows, dataset)
        indices = dataset_indices(valid_rows, dataset)
        base = np.asarray(v22["base_predictions"], dtype=np.float64)[indices]
        fit_seed = int(args.seed) + offset * 100
        print("[V26 %s] fitting train-only sequence/SMILES experts" % dataset, flush=True)
        fitted, fit_audit = fit_sequence_expert(
            train_dataset,
            valid_dataset,
            seed=fit_seed,
            max_iter=args.max_iter,
        )
        affinity_predictions, score_predictions, eligible = predict_sequence_expert(
            fitted, valid_dataset, base
        )
        labels = np.asarray([row["label"] for row in valid_dataset], dtype=np.float64)
        affinity_selection, affinity_prediction, affinity_candidates = select_affinity_expression(
            labels, affinity_predictions, dataset
        )
        binary_selection, selected_score, binary_candidates = select_binary_expression(
            labels,
            score_predictions,
            BINARY_THRESHOLDS[dataset],
            dataset,
        )
        binary_label = (
            labels >= float(BINARY_THRESHOLDS[dataset])
        ).astype(np.int8)
        gate = collaborative_validation_gate(
            [row["sample_id"] for row in valid_dataset],
            labels,
            binary_label,
            v25["predictions"][indices],
            affinity_prediction,
            v25["binary_scores"][indices],
            selected_score,
            minimum_ci_gain=args.minimum_ci_gain,
            minimum_pr_auc_gain=args.minimum_pr_auc_gain,
            minimum_positive_folds=args.minimum_positive_folds,
        )
        combined_predictions[indices] = affinity_prediction
        binary_scores[indices] = selected_score
        binary_predictions[indices] = (
            selected_score >= float(binary_selection["decision_threshold"])
        ).astype(np.int8)
        collaborative_eligible[indices] = eligible
        dataset_results[dataset] = {
            "fit_audit": fit_audit,
            "entity_coverage": {
                "bias_svd_eligible_both_entities_seen": int(eligible.sum()),
                "bias_svd_fallback_unknown_entity": int((~eligible).sum()),
            },
            "v22_base_affinity_metrics": affinity_metrics(
                labels, affinity_predictions["v22"]
            ),
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
            "selected_affinity": affinity_selection,
            "affinity_candidate_count": int(len(affinity_candidates)),
            "affinity_candidates": affinity_candidates,
            "selected_binary": binary_selection,
            "binary_candidate_count": int(len(binary_candidates)),
            "binary_candidates": binary_candidates,
            "validation_gate": gate,
        }
        sealed_dataset_config[dataset] = {
            "affinity": affinity_selection,
            "binary": binary_selection,
            "affinity_threshold": float(BINARY_THRESHOLDS[dataset]),
            "affinity_scale": float(fitted["affinity_scale"]),
            "fit_seed": fit_seed,
            "max_iter": int(args.max_iter),
        }
        del fitted

    lp_indices = dataset_indices(valid_rows, "LP_PDBBind")
    lp_rows = rows_for_dataset(valid_rows, "LP_PDBBind")
    v20_path, v20_predictions, v20_available = load_v20_lp_predictions(
        args.v20_valid, lp_rows
    )
    lp_labels = np.asarray([row["label"] for row in lp_rows], dtype=np.float64)
    lp_selection, lp_prediction, lp_candidates, lp_anchor_metrics = select_lp_blend(
        lp_labels,
        v22["predictions"][lp_indices],
        v22["base_predictions"][lp_indices],
        v20_predictions,
        v20_available,
    )
    lp_gate = lp_validation_gate(
        lp_anchor_metrics,
        lp_selection["metrics"],
        maximum_rmse_increase=args.maximum_lp_rmse_increase,
    )
    combined_predictions[lp_indices] = lp_prediction
    dataset_results["LP_PDBBind"] = {
        "v20_available": int(v20_available.sum()),
        "v22_fallback": int((~v20_available).sum()),
        "v20_route_anchor_metrics": lp_anchor_metrics,
        "selected_affinity": lp_selection,
        "affinity_candidates": lp_candidates,
        "validation_gate": lp_gate,
    }

    passed = all(
        dataset_results[name]["validation_gate"]["passed"]
        for name in ("LP_PDBBind", "DAVIS", "KIBA")
    )
    prediction_path = output / "valid_predictions.npz"
    atomic_npz(
        prediction_path,
        sample_ids=np.asarray([row["sample_id"] for row in valid_rows]),
        labels=np.asarray([row["label"] for row in valid_rows], dtype=np.float64),
        dataset_index=np.asarray(
            [
                0
                if row["dataset"] == "LP_PDBBind"
                else 1
                if row["dataset"] == "DAVIS"
                else 2
                for row in valid_rows
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
    bundle_path = output / "selection_bundle.joblib"
    search_space = {
        "sequence_expert": "protein char 1-3mer SVD + molecule Morgan/SVD + descriptors",
        "target_statistics": "train-only leave-one-out for training rows",
        "heads": ["reg_all", "reg_stats", "cls_all", "cls_stats"],
        "bias_svd_ranks": [8, 16, 32],
        "simplex_units": int(SIMPLEX_UNITS),
        "affinity_components": {
            name: list(values) for name, values in AFFINITY_COMPONENTS.items()
        },
        "binary_components": {
            name: list(values) for name, values in BINARY_COMPONENTS.items()
        },
        "affinity_selection": "validation CI, then RMSE and Spearman",
        "binary_selection": "validation PR-AUC, then ROC-AUC",
        "decision_threshold": "validation MCC",
        "seed": int(args.seed),
        "max_iter": int(args.max_iter),
        "test_input_available_to_selection": False,
    }
    bundle = {
        "version": VERSION,
        "phase": "validation_selection_sealed",
        "dataset_config": sealed_dataset_config,
        "lp_config": {"affinity": lp_selection},
        "search_space": search_space,
    }
    atomic_joblib(bundle_path, bundle)
    inputs = {
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "valid_manifest": str(valid_path),
        "valid_manifest_sha256": sha256_file(valid_path),
        "v22_valid": str(v22_path),
        "v22_valid_sha256": sha256_file(v22_path),
        "v20_valid": str(v20_path),
        "v20_valid_sha256": sha256_file(v20_path),
        "v25_valid": str(v25_path),
        "v25_valid_sha256": sha256_file(v25_path),
    }
    result = {
        "version": VERSION,
        "phase": "validation_selection_sealed",
        "status": "validation_passed" if passed else "validation_failed",
        "validation_selected": bool(passed),
        "output": str(output),
        "counts": {"train": group_counts(train_rows), "valid": group_counts(valid_rows)},
        "inputs": inputs,
        "search_space": search_space,
        "dataset_results": dataset_results,
        "selected_validation_metrics": metrics_by_dataset(valid_rows, combined_predictions),
        "selection_bundle": str(bundle_path),
        "selection_bundle_sha256": sha256_file(bundle_path),
        "valid_predictions": str(prediction_path),
        "valid_predictions_sha256": sha256_file(prediction_path),
        "test_inputs_read": False,
    }
    atomic_json(output / "selection.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--v22-valid", type=Path, required=True)
    parser.add_argument("--v20-valid", type=Path, required=True)
    parser.add_argument("--v25-valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--minimum-ci-gain", type=float, default=0.001)
    parser.add_argument("--minimum-pr-auc-gain", type=float, default=0.003)
    parser.add_argument("--minimum-positive-folds", type=int, default=3)
    parser.add_argument("--maximum-lp-rmse-increase", type=float, default=0.003)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
