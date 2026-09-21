"""Verify the V27 validation seal, then run one frozen test report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dta.optional_structure_staged_v22.training import _dataset_metrics, cache_lineage

from .binary_route import (
    affinity_only_metrics,
    frozen_binary_metrics,
    load_aligned_binary_route,
)
from .data import OptionalStructureDataset, manifest_fingerprint, sha256_file
from .model import OptionalStructurePrivilegedPairDTA
from .selection import apply_sealed_blends, load_aligned_predictions
from .training import TrainingConfig, VERSION, predict


def _verify_selection(output):
    output = output.expanduser().resolve()
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != VERSION:
        raise ValueError("V27 selection version mismatch")
    if selection.get("phase") != "validation_selection":
        raise ValueError("V27 selection phase mismatch")
    if selection.get("status") != "validation_passed" or not selection.get("validation_selected"):
        raise ValueError("V27 validation gate did not pass")
    checks = (
        ("checkpoint", "checkpoint_sha256"),
        ("valid_predictions", "valid_predictions_sha256"),
        ("selected_valid_predictions", "selected_valid_predictions_sha256"),
        ("anchor_valid", "anchor_valid_sha256"),
    )
    for path_name, hash_name in checks:
        if sha256_file(selection[path_name]) != selection[hash_name]:
            raise ValueError("V27 sealed %s checksum mismatch" % path_name)
    if manifest_fingerprint(selection["train_manifest"]) != selection["train_manifest_sha256"]:
        raise ValueError("V27 training manifest changed after validation")
    if manifest_fingerprint(selection["valid_manifest"]) != selection["valid_manifest_sha256"]:
        raise ValueError("V27 validation manifest changed after selection")
    checkpoint = torch.load(str(selection["checkpoint"]), map_location="cpu")
    if checkpoint.get("version") != VERSION or checkpoint.get("phase") != "privileged_pair":
        raise ValueError("V27 sealed checkpoint phase mismatch")
    if checkpoint.get("train_manifest_sha256") != selection["train_manifest_sha256"]:
        raise ValueError("V27 checkpoint training lineage mismatch")
    if checkpoint.get("valid_manifest_sha256") != selection["valid_manifest_sha256"]:
        raise ValueError("V27 checkpoint validation lineage mismatch")
    if checkpoint.get("protein_cache") != cache_lineage(selection["protein_cache"]):
        raise ValueError("V27 protein cache lineage mismatch")
    if checkpoint.get("ligand_cache") != cache_lineage(selection["ligand_cache"]):
        raise ValueError("V27 ligand cache lineage mismatch")
    return selection_path, selection, checkpoint


def run(args):
    output = args.selection_output.expanduser().resolve()
    selection_path, selection, checkpoint = _verify_selection(output)
    result_path = output / "result.json"
    prediction_path = output / "test_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite an existing V27 test report")
    config_values = dict(checkpoint["config"])
    config_values["device"] = args.device or config_values["device"]
    if args.num_workers is not None:
        config_values["num_workers"] = int(args.num_workers)
    if args.evaluation_batch_size is not None:
        config_values["evaluation_batch_size"] = int(args.evaluation_batch_size)
    config = TrainingConfig(**config_values)
    print("V27 validation seal verified; reading test inputs now", flush=True)
    test_manifest = args.test_manifest.expanduser().resolve()
    anchor_test = args.anchor_test.expanduser().resolve()
    test = OptionalStructureDataset(
        test_manifest,
        selection["protein_cache"],
        selection["ligand_cache"],
        with_structure=True,
        limit=args.limit_test,
    )
    model = OptionalStructurePrivilegedPairDTA(**checkpoint["model_kwargs"]).to(config.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    v27 = predict(model, test, config, checkpoint["label_statistics"])
    anchor_predictions = load_aligned_predictions(
        anchor_test, v27["sample_ids"], v27["labels"], v27["dataset_index"]
    )
    binary_route = load_aligned_binary_route(
        anchor_test, v27["sample_ids"], v27["labels"], v27["dataset_index"]
    )
    final_predictions = apply_sealed_blends(
        v27["dataset_index"],
        anchor_predictions,
        v27["predictions"],
        selection["anchor_blend_selection"],
    )
    final_metrics = affinity_only_metrics(
        _dataset_metrics(v27["labels"], final_predictions, v27["dataset_index"])
    )
    anchor_metrics = affinity_only_metrics(
        _dataset_metrics(v27["labels"], anchor_predictions, v27["dataset_index"])
    )
    np.savez_compressed(
        prediction_path,
        sample_ids=np.asarray(v27["sample_ids"]),
        labels=v27["labels"],
        dataset_index=v27["dataset_index"],
        predictions=final_predictions,
        anchor_predictions=anchor_predictions,
        v27_predictions=v27["predictions"],
        v27_base_predictions=v27["base_predictions"],
        structure_gate=v27["structure_gate"],
        binary_scores=binary_route["scores"],
        binary_predictions=binary_route["predictions"],
    )
    deltas = {
        name: {
            key: float(final_metrics[name][key] - anchor_metrics[name][key])
            for key in ("CI", "RMSE", "MAE", "Pearson", "Spearman")
        }
        for name in final_metrics
    }
    result = {
        "version": VERSION,
        "phase": "test_report",
        "status": "reported",
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": manifest_fingerprint(test_manifest),
        "anchor_test": str(anchor_test),
        "anchor_test_sha256": sha256_file(anchor_test),
        "test_metrics": final_metrics,
        "anchor_test_metrics": anchor_metrics,
        "frozen_v26_binary_test_metrics": frozen_binary_metrics(
            v27["labels"],
            v27["dataset_index"],
            binary_route["scores"],
            binary_route["predictions"],
        ),
        "standalone_v27_test_metrics": v27["metrics"],
        "standalone_v27_base_test_metrics": v27["base_metrics"],
        "delta_vs_anchor": deltas,
        "counts": {"test": len(test)},
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "test_used_for_selection": False,
        "test_configuration_reselected": False,
        "binary_route": "V26 binary scores and decisions are copied unchanged; V27 selection changes affinity only",
        "inference_config": asdict(config),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--anchor-test", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--limit-test", type=int)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
