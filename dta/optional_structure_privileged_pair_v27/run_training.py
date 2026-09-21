"""Train V27 and seal validation-only V26/V27 selection without reading test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dta.optional_structure_staged_v22.training import _dataset_metrics

from .binary_route import (
    affinity_only_metrics,
    frozen_binary_metrics,
    load_aligned_binary_route,
)
from .cli import add_config_arguments, config_from_args
from .data import OptionalStructureDataset, manifest_fingerprint, sha256_file
from .fit import fit
from .selection import load_aligned_predictions, select_validation_blends
from .training import VERSION, structure_validation_gate


def run(args):
    output = args.output.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    valid_manifest = args.valid_manifest.expanduser().resolve()
    protein_cache = args.protein_cache.expanduser().resolve()
    ligand_cache = args.ligand_cache.expanduser().resolve()
    warm_checkpoint = args.warm_checkpoint.expanduser().resolve() if args.warm_checkpoint else None
    anchor_valid = args.anchor_valid.expanduser().resolve()
    train = OptionalStructureDataset(
        train_manifest, protein_cache, ligand_cache, with_structure=True, limit=args.limit_train
    )
    valid = OptionalStructureDataset(
        valid_manifest, protein_cache, ligand_cache, with_structure=True, limit=args.limit_valid
    )
    detected_width = int(train[0]["protein_embeddings"].shape[1])
    config = config_from_args(args, detected_width)
    print(
        json.dumps(
            {
                "event": "v27_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid)},
                "warm_checkpoint": str(warm_checkpoint) if warm_checkpoint else None,
                "anchor_valid": str(anchor_valid),
                "test_manifest_argument_exists": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _, checkpoint, valid_result, history = fit(
        train,
        valid,
        output,
        config,
        protein_cache,
        ligand_cache,
        warm_checkpoint=warm_checkpoint,
        resume=args.resume,
    )
    structure_gate = structure_validation_gate(valid_result, config)
    anchor_predictions = load_aligned_predictions(
        anchor_valid,
        valid_result["sample_ids"],
        valid_result["labels"],
        valid_result["dataset_index"],
    )
    binary_route = load_aligned_binary_route(
        anchor_valid,
        valid_result["sample_ids"],
        valid_result["labels"],
        valid_result["dataset_index"],
    )
    blend = select_validation_blends(
        valid_result["sample_ids"],
        valid_result["labels"],
        valid_result["dataset_index"],
        anchor_predictions,
        valid_result["predictions"],
        minimum_ci_gain=args.minimum_anchor_ci_gain,
        maximum_rmse_increase=args.maximum_anchor_rmse_increase,
        minimum_spearman_gain=args.minimum_anchor_spearman_gain,
        minimum_positive_folds=args.minimum_positive_folds,
    )
    final_metrics = affinity_only_metrics(
        _dataset_metrics(
            valid_result["labels"], blend["predictions"], valid_result["dataset_index"]
        )
    )
    selected_prediction_path = output / "selected_valid_predictions.npz"
    np.savez_compressed(
        selected_prediction_path,
        sample_ids=np.asarray(valid_result["sample_ids"]),
        labels=valid_result["labels"],
        dataset_index=valid_result["dataset_index"],
        predictions=blend["predictions"],
        anchor_predictions=anchor_predictions,
        v27_predictions=valid_result["predictions"],
        v27_base_predictions=valid_result["base_predictions"],
        binary_scores=binary_route["scores"],
        binary_predictions=binary_route["predictions"],
    )
    passed = bool(structure_gate["passed"] and blend["any_promoted"])
    checkpoint_path = output / "best.pt"
    valid_prediction_path = output / "valid_predictions.npz"
    selection = {
        "version": VERSION,
        "phase": "validation_selection",
        "status": "validation_passed" if passed else "validation_failed",
        "validation_selected": passed,
        "output": str(output),
        "best_epoch": int(checkpoint["epoch"]),
        "selected_variant": checkpoint["variant"],
        "selection_rule": checkpoint["selection_rule"],
        "standalone_validation_metrics": valid_result["metrics"],
        "standalone_base_validation_metrics": valid_result["base_metrics"],
        "selected_validation_metrics": final_metrics,
        "frozen_v26_binary_validation_metrics": frozen_binary_metrics(
            valid_result["labels"],
            valid_result["dataset_index"],
            binary_route["scores"],
            binary_route["predictions"],
        ),
        "binary_route": "V26 binary scores and decisions are copied unchanged; V27 selection is affinity-only",
        "structure_validation_gate": structure_gate,
        "anchor_blend_selection": {key: value for key, value in blend.items() if key != "predictions"},
        "counts": {"train": len(train), "valid": len(valid)},
        "history_length": len(history),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": manifest_fingerprint(train_manifest),
        "valid_manifest": str(valid_manifest),
        "valid_manifest_sha256": manifest_fingerprint(valid_manifest),
        "protein_cache": str(protein_cache),
        "ligand_cache": str(ligand_cache),
        "anchor_valid": str(anchor_valid),
        "anchor_valid_sha256": sha256_file(anchor_valid),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "valid_predictions": str(valid_prediction_path),
        "valid_predictions_sha256": sha256_file(valid_prediction_path),
        "selected_valid_predictions": str(selected_prediction_path),
        "selected_valid_predictions_sha256": sha256_file(selected_prediction_path),
        "sources": checkpoint["sources"],
        "test_manifest_read": False,
        "test_metrics": None,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, sort_keys=True), flush=True)
    return selection


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--protein-cache", type=Path, required=True)
    parser.add_argument("--ligand-cache", type=Path, required=True)
    parser.add_argument("--anchor-valid", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--minimum-anchor-ci-gain", type=float, default=0.0005)
    parser.add_argument("--maximum-anchor-rmse-increase", type=float, default=0.003)
    parser.add_argument("--minimum-anchor-spearman-gain", type=float, default=0.0)
    parser.add_argument("--minimum-positive-folds", type=int, default=3)
    return add_config_arguments(parser)


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
