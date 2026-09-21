"""Select and seal V23 using train/validation inputs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dta.sequence_smiles_binary_probe_v1.core import affinity_to_score

from . import VERSION
from .core import (
    BINARY_THRESHOLDS,
    BLEND_WEIGHTS,
    COLLABORATIVE_DATASETS,
    RANKS,
    atomic_joblib,
    atomic_json,
    atomic_npz,
    collaborative_predictions,
    collaborative_validation_gate,
    dataset_indices,
    fit_decontaminated_bias_svd,
    group_counts,
    load_v20_lp_predictions,
    load_v22_predictions,
    lp_validation_gate,
    metrics_by_dataset,
    pair_key,
    read_manifest,
    rows_for_dataset,
    select_affinity_blend,
    select_binary_blend,
    select_lp_blend,
    sha256_file,
)


def _prepare_output(path):
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty V23 output: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _limit_rows(rows, limit):
    if limit is None:
        return rows
    counts = {"LP_PDBBind": 0, "DAVIS": 0, "KIBA": 0}
    selected = []
    for row in rows:
        dataset = row["dataset"]
        if counts[dataset] < int(limit):
            selected.append(row)
            counts[dataset] += 1
    return selected


def _selection_search_space(args):
    return {
        "bias_svd_ranks": list(RANKS),
        "blend_weights": list(BLEND_WEIGHTS),
        "affinity_selection": "maximum validation CI, then RMSE and Spearman",
        "binary_selection": "maximum validation PR-AUC, then ROC-AUC",
        "binary_decision_threshold": "maximum validation MCC",
        "unknown_entity_policy": "hard fallback to V22 neural base",
        "exact_pair_policy": "remove evaluation entity pairs before Bias-SVD fit",
        "validation_hash_folds": 5,
        "seed": int(args.seed),
    }


def run(args):
    output = _prepare_output(args.output)
    train_path, train_rows = read_manifest(args.train_manifest)
    valid_path, valid_rows = read_manifest(args.valid_manifest)
    train_rows = _limit_rows(train_rows, args.limit_per_dataset)
    valid_rows = _limit_rows(valid_rows, args.limit_per_dataset)
    if any(str(row.get("split", "train")) != "train" for row in train_rows):
        raise ValueError("V23 train manifest contains a non-train row")
    if any(str(row.get("split", "valid")) != "valid" for row in valid_rows):
        raise ValueError("V23 validation manifest contains a non-validation row")

    v22_path, v22 = load_v22_predictions(args.v22_valid, valid_rows)
    combined_predictions = np.asarray(v22["base_predictions"], dtype=np.float64).copy()
    binary_scores = np.full(len(valid_rows), np.nan, dtype=np.float64)
    binary_predictions = np.full(len(valid_rows), -1, dtype=np.int8)
    collaborative_eligible = np.zeros(len(valid_rows), dtype=bool)
    dataset_results = {}
    sealed_dataset_config = {}

    for dataset_offset, dataset in enumerate(COLLABORATIVE_DATASETS):
        train_dataset = rows_for_dataset(train_rows, dataset)
        valid_dataset = rows_for_dataset(valid_rows, dataset)
        indices = dataset_indices(valid_rows, dataset)
        if not train_dataset or not valid_dataset:
            raise ValueError("V23 requires train/validation rows for %s" % dataset)
        model, fit_audit = fit_decontaminated_bias_svd(
            train_dataset,
            valid_dataset,
            seed=int(args.seed) + dataset_offset * 100,
        )
        labels = np.asarray([row["label"] for row in valid_dataset], dtype=np.float64)
        base = np.asarray(v22["base_predictions"], dtype=np.float64)[indices]
        collaborative_by_rank = {}
        eligible = None
        for rank in RANKS:
            collaborative, rank_eligible = collaborative_predictions(
                model, valid_dataset, base, rank
            )
            collaborative_by_rank[rank] = collaborative
            if eligible is None:
                eligible = rank_eligible
            elif not np.array_equal(eligible, rank_eligible):
                raise RuntimeError("V23 collaborative eligibility changed across ranks")
        affinity_selection, affinity_prediction, affinity_candidates = select_affinity_blend(
            labels, base, collaborative_by_rank
        )
        excluded_pairs = {pair_key(row) for row in valid_dataset}
        fitted_labels = np.asarray(
            [row["label"] for row in train_dataset if pair_key(row) not in excluded_pairs],
            dtype=np.float64,
        )
        affinity_scale = max(0.25, float(np.std(fitted_labels)) * 0.35)
        affinity_threshold = float(BINARY_THRESHOLDS[dataset])
        binary_selection, selected_score, binary_candidates, base_binary_metrics = select_binary_blend(
            labels,
            base,
            collaborative_by_rank,
            affinity_threshold,
            affinity_scale,
        )
        binary_label = (labels >= affinity_threshold).astype(np.int8)
        base_score = affinity_to_score(base, affinity_threshold, affinity_scale)
        gate = collaborative_validation_gate(
            [row["sample_id"] for row in valid_dataset],
            labels,
            binary_label,
            base,
            affinity_prediction,
            base_score,
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
                "eligible_both_entities_seen": int(eligible.sum()),
                "fallback_unknown_entity": int((~eligible).sum()),
            },
            "affinity_scale": affinity_scale,
            "affinity_threshold": affinity_threshold,
            "v22_base_affinity_metrics": affinity_candidates[0]["metrics"],
            "selected_affinity": affinity_selection,
            "affinity_candidates": affinity_candidates,
            "v22_base_binary_metrics": base_binary_metrics,
            "selected_binary": binary_selection,
            "binary_candidates": binary_candidates,
            "validation_gate": gate,
        }
        sealed_dataset_config[dataset] = {
            "affinity": affinity_selection,
            "binary": binary_selection,
            "affinity_scale": affinity_scale,
            "affinity_threshold": affinity_threshold,
            "fit_seed": int(args.seed) + dataset_offset * 100,
        }

    lp_indices = dataset_indices(valid_rows, "LP_PDBBind")
    lp_rows = rows_for_dataset(valid_rows, "LP_PDBBind")
    if lp_indices.size == 0:
        raise ValueError("V23 requires LP-PDBBind validation rows")
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
        bool(dataset_results[name]["validation_gate"]["passed"])
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
    bundle = {
        "version": VERSION,
        "phase": "validation_selection_sealed",
        "dataset_config": sealed_dataset_config,
        "lp_config": {"affinity": lp_selection},
        "search_space": _selection_search_space(args),
        "gate_criteria": {
            "minimum_ci_gain": float(args.minimum_ci_gain),
            "minimum_pr_auc_gain": float(args.minimum_pr_auc_gain),
            "minimum_positive_folds": int(args.minimum_positive_folds),
            "maximum_lp_rmse_increase": float(args.maximum_lp_rmse_increase),
        },
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
    }
    result = {
        "version": VERSION,
        "phase": "validation_selection_sealed",
        "status": "validation_passed" if passed else "validation_failed",
        "validation_selected": bool(passed),
        "output": str(output),
        "counts": {
            "train": group_counts(train_rows),
            "valid": group_counts(valid_rows),
        },
        "inputs": inputs,
        "search_space": bundle["search_space"],
        "dataset_results": dataset_results,
        "selected_validation_metrics": metrics_by_dataset(valid_rows, combined_predictions),
        "selection_bundle": str(bundle_path),
        "selection_bundle_sha256": sha256_file(bundle_path),
        "valid_predictions": str(prediction_path),
        "valid_predictions_sha256": sha256_file(prediction_path),
        "limit_per_dataset": args.limit_per_dataset,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--minimum-ci-gain", type=float, default=0.01)
    parser.add_argument("--minimum-pr-auc-gain", type=float, default=0.02)
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--maximum-lp-rmse-increase", type=float, default=0.003)
    parser.add_argument("--limit-per-dataset", type=int)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
