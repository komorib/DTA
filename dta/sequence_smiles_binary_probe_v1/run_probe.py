from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from .core import (
    DEFAULT_THRESHOLDS,
    BiasSVDModel,
    RepresentationEncoder,
    TargetStatisticsEncoder,
    best_decision_threshold,
    binary_metrics,
    expression_score,
    fit_models,
    load_bundle,
    load_split,
    predict_base_scores,
    save_bundle,
    select_score_expression,
    sha256_file,
    split_fingerprints,
    write_json,
)


VERSION = "sequence-smiles-binary-probe-v1"


def _dataset_name(dataset_dir: Path, requested: str | None) -> str:
    name = (requested or dataset_dir.name).upper()
    if name not in DEFAULT_THRESHOLDS:
        raise ValueError(f"dataset must be one of {sorted(DEFAULT_THRESHOLDS)}, got {name}")
    return name


def run_select(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_name = _dataset_name(dataset_dir, args.dataset_name)
    affinity_threshold = float(
        args.affinity_threshold
        if args.affinity_threshold is not None
        else DEFAULT_THRESHOLDS[dataset_name]
    )

    print(f"[{dataset_name}] reading train and validation only", flush=True)
    train = load_split(dataset_dir, "train")
    valid = load_split(dataset_dir, "valid")
    train_binary = (train.affinity >= affinity_threshold).astype(np.int8)
    valid_binary = (valid.affinity >= affinity_threshold).astype(np.int8)

    representation = RepresentationEncoder(
        protein_hash_dim=args.protein_hash_dim,
        protein_svd_dim=args.protein_svd_dim,
        molecule_bits=args.molecule_bits,
        molecule_svd_dim=args.molecule_svd_dim,
        interaction_dim=args.interaction_dim,
        seed=args.seed,
    ).fit(train.smiles, train.proteins)
    statistics = TargetStatisticsEncoder().fit(
        train.smiles, train.proteins, train.affinity, train_binary
    )

    print(f"[{dataset_name}] building train/validation CPU features", flush=True)
    train_representation = representation.transform(train.smiles, train.proteins)
    valid_representation = representation.transform(valid.smiles, valid.proteins)
    train_statistics = statistics.transform(
        train.smiles,
        train.proteins,
        affinity=train.affinity,
        binary=train_binary,
        leave_one_out=True,
    )
    valid_statistics = statistics.transform(valid.smiles, valid.proteins)
    train_features = np.concatenate([train_representation, train_statistics], axis=1)
    valid_features = np.concatenate([valid_representation, valid_statistics], axis=1)

    print(f"[{dataset_name}] fitting CPU regressors/classifiers and bias-SVD", flush=True)
    models = fit_models(
        train_features,
        statistics_width=train_statistics.shape[1],
        affinity=train.affinity,
        binary=train_binary,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    bias_svd = BiasSVDModel(rank=32, seed=args.seed).fit(
        train.smiles, train.proteins, train.affinity
    )
    affinity_scale = max(0.25, float(np.std(train.affinity)) * 0.35)
    valid_base_scores = predict_base_scores(
        models,
        bias_svd,
        valid_features,
        train_statistics.shape[1],
        valid.smiles,
        valid.proteins,
        affinity_threshold,
        affinity_scale,
    )
    expression, candidates = select_score_expression(valid_binary, valid_base_scores)
    valid_score = expression_score(expression, valid_base_scores)
    decision_threshold, threshold_objective = best_decision_threshold(
        valid_binary, valid_score, objective=args.decision_objective
    )
    valid_metrics = binary_metrics(valid_binary, valid_score, decision_threshold)
    valid_novel_mask = statistics.novel_pair_mask(valid.smiles, valid.proteins)
    valid_novel_metrics = binary_metrics(
        valid_binary[valid_novel_mask], valid_score[valid_novel_mask], decision_threshold
    )

    fingerprints = split_fingerprints(dataset_dir, ("train", "valid"))
    bundle_path = output / "selection_bundle.joblib"
    bundle = {
        "version": VERSION,
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "affinity_threshold": affinity_threshold,
        "affinity_scale": affinity_scale,
        "representation": representation,
        "statistics": statistics,
        "models": models,
        "bias_svd": bias_svd,
        "statistics_width": train_statistics.shape[1],
        "score_expression": expression,
        "decision_threshold": decision_threshold,
        "decision_objective": args.decision_objective,
        "selection_input_sha256": fingerprints,
        "seed": args.seed,
    }
    save_bundle(bundle_path, bundle)
    bundle_sha256 = sha256_file(bundle_path)
    np.savez_compressed(
        output / "valid_predictions.npz",
        affinity=valid.affinity,
        label=valid_binary,
        score=valid_score,
        prediction=(valid_score >= decision_threshold).astype(np.int8),
        novel_pair=valid_novel_mask,
    )
    result = {
        "version": VERSION,
        "phase": "validation_selection_sealed",
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "output": str(output),
        "affinity_definition": {
            "positive_when": "affinity >= threshold",
            "threshold": affinity_threshold,
            "Davis_interpretation": "pKd >= 7.522878745280337 (Kd <= 30 nM)"
            if dataset_name == "DAVIS" and args.affinity_threshold is None
            else None,
        },
        "counts": {
            "train": len(train),
            "train_positive": int(train_binary.sum()),
            "valid": len(valid),
            "valid_positive": int(valid_binary.sum()),
            "valid_novel_pair": int(valid_novel_mask.sum()),
        },
        "feature_dimensions": {
            "representation": int(train_representation.shape[1]),
            "target_statistics": int(train_statistics.shape[1]),
            "total": int(train_features.shape[1]),
        },
        "selection_metric": "validation AUPRC, then validation AUROC",
        "selected_score_expression": expression,
        "selected_decision_threshold": decision_threshold,
        "decision_objective": args.decision_objective,
        "threshold_validation_objective": threshold_objective,
        "validation_metrics": valid_metrics,
        "validation_novel_pair_metrics": valid_novel_metrics,
        "candidate_ranking": candidates,
        "selection_input_sha256": fingerprints,
        "bundle": str(bundle_path),
        "bundle_sha256": bundle_sha256,
        "test_read_during_selection": False,
        "status": "sealed",
    }
    write_json(output / "selection.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def run_test(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    selection_path = output / "selection.json"
    bundle_path = output / "selection_bundle.joblib"
    if not selection_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError("sealed selection.json and selection_bundle.joblib are required")
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("phase") != "validation_selection_sealed" or selection.get("status") != "sealed":
        raise ValueError("selection is not sealed")
    if sha256_file(bundle_path) != selection.get("bundle_sha256"):
        raise ValueError("selection bundle checksum does not match the sealed selection")
    bundle = load_bundle(bundle_path)
    if bundle.get("version") != VERSION:
        raise ValueError(f"unexpected bundle version: {bundle.get('version')}")
    dataset_dir = Path(bundle["dataset_dir"])
    current_fingerprints = split_fingerprints(dataset_dir, ("train", "valid"))
    if current_fingerprints != bundle["selection_input_sha256"]:
        raise ValueError("train/validation inputs changed after selection was sealed")

    dataset_name = str(bundle["dataset_name"])
    print(f"[{dataset_name}] sealed selection verified; reading test now", flush=True)
    test = load_split(dataset_dir, "test")
    test_binary = (test.affinity >= float(bundle["affinity_threshold"])).astype(np.int8)
    representation = bundle["representation"]
    statistics = bundle["statistics"]
    test_representation = representation.transform(test.smiles, test.proteins)
    test_statistics = statistics.transform(test.smiles, test.proteins)
    test_features = np.concatenate([test_representation, test_statistics], axis=1)
    test_base_scores = predict_base_scores(
        bundle["models"],
        bundle["bias_svd"],
        test_features,
        int(bundle["statistics_width"]),
        test.smiles,
        test.proteins,
        float(bundle["affinity_threshold"]),
        float(bundle["affinity_scale"]),
    )
    test_score = expression_score(bundle["score_expression"], test_base_scores)
    decision_threshold = float(bundle["decision_threshold"])
    novel_mask = statistics.novel_pair_mask(test.smiles, test.proteins)
    raw_metrics = binary_metrics(test_binary, test_score, decision_threshold)
    novel_metrics = binary_metrics(test_binary[novel_mask], test_score[novel_mask], decision_threshold)

    prediction_path = output / "test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        affinity=test.affinity,
        label=test_binary,
        score=test_score,
        prediction=(test_score >= decision_threshold).astype(np.int8),
        novel_pair=novel_mask,
    )
    test_fingerprints = split_fingerprints(dataset_dir, ("test",))
    result = {
        "version": VERSION,
        "phase": "test_report",
        "dataset": dataset_name,
        "selection": str(selection_path),
        "selection_bundle_sha256": selection["bundle_sha256"],
        "selected_score_expression": bundle["score_expression"],
        "decision_threshold_selected_on_validation": decision_threshold,
        "affinity_threshold": float(bundle["affinity_threshold"]),
        "test_metrics": raw_metrics,
        "novel_pair_test_metrics": novel_metrics,
        "test_pair_audit": {
            "total": len(test),
            "seen_in_train": int((~novel_mask).sum()),
            "novel_pair": int(novel_mask.sum()),
        },
        "test_input_sha256": test_fingerprints,
        "predictions": str(prediction_path),
        "test_used_for_selection": False,
        "status": "reported",
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict validation-selected sequence/SMILES binary probe for Davis/KIBA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="fit on train and select only on validation")
    select.add_argument("--dataset-dir", type=Path, required=True)
    select.add_argument("--dataset-name", choices=sorted(DEFAULT_THRESHOLDS))
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--affinity-threshold", type=float)
    select.add_argument("--protein-hash-dim", type=int, default=4096)
    select.add_argument("--protein-svd-dim", type=int, default=96)
    select.add_argument("--molecule-bits", type=int, default=2048)
    select.add_argument("--molecule-svd-dim", type=int, default=96)
    select.add_argument("--interaction-dim", type=int, default=8)
    select.add_argument("--max-iter", type=int, default=300)
    select.add_argument("--decision-objective", choices=("mcc", "f1"), default="mcc")
    select.add_argument("--seed", type=int, default=20260902)
    select.set_defaults(func=run_select)
    test = subparsers.add_parser("test", help="verify sealed validation selection, then read test")
    test.add_argument("--output", type=Path, required=True)
    test.set_defaults(func=run_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
