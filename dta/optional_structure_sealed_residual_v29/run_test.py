"""Verify the V29 validation seal, then run one frozen LP-residual test report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dta.optional_structure_staged_v22.training import _dataset_metrics, cache_lineage

from .binary_route import affinity_only_metrics, frozen_binary_metrics, load_aligned_binary_route
from .data import OptionalStructureDataset, manifest_fingerprint, sha256_file
from .model import OptionalStructureSealedResidualDTA, load_sealed_v22_backbone
from .run_training import _backbone_equivalence
from .selection import apply_sealed_blends, load_aligned_predictions
from .training import TrainingConfig, VERSION, predict


def _verify_selection(output):
    output = output.expanduser().resolve()
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != VERSION:
        raise ValueError("V29 selection version mismatch")
    if selection.get("phase") != "validation_selection":
        raise ValueError("V29 selection phase mismatch")
    if selection.get("status") != "validation_passed" or not selection.get("validation_selected"):
        raise ValueError("V29 validation gate did not pass")
    checks = (
        ("checkpoint", "checkpoint_sha256"),
        ("valid_predictions", "valid_predictions_sha256"),
        ("selected_valid_predictions", "selected_valid_predictions_sha256"),
        ("sealed_v22_checkpoint", "sealed_v22_checkpoint_sha256"),
        ("v22_valid", "v22_valid_sha256"),
        ("anchor_valid", "anchor_valid_sha256"),
    )
    for path_name, hash_name in checks:
        if sha256_file(selection[path_name]) != selection[hash_name]:
            raise ValueError("V29 sealed %s checksum mismatch" % path_name)
    if manifest_fingerprint(selection["train_manifest"]) != selection["train_manifest_sha256"]:
        raise ValueError("V29 training manifest changed after validation")
    if manifest_fingerprint(selection["valid_manifest"]) != selection["valid_manifest_sha256"]:
        raise ValueError("V29 validation manifest changed after selection")
    checkpoint = torch.load(str(selection["checkpoint"]), map_location="cpu")
    if checkpoint.get("version") != VERSION or checkpoint.get("phase") != "sealed_residual":
        raise ValueError("V29 sealed checkpoint phase mismatch")
    if checkpoint.get("train_manifest_sha256") != selection["train_manifest_sha256"]:
        raise ValueError("V29 checkpoint training lineage mismatch")
    if checkpoint.get("valid_manifest_sha256") != selection["valid_manifest_sha256"]:
        raise ValueError("V29 checkpoint validation lineage mismatch")
    if checkpoint.get("protein_cache") != cache_lineage(selection["protein_cache"]):
        raise ValueError("V29 protein cache lineage mismatch")
    if checkpoint.get("ligand_cache") != cache_lineage(selection["ligand_cache"]):
        raise ValueError("V29 ligand cache lineage mismatch")
    return selection_path, selection, checkpoint


def run(args):
    output = args.selection_output.expanduser().resolve()
    selection_path, selection, checkpoint = _verify_selection(output)
    result_path = output / "result.json"
    prediction_path = output / "test_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite an existing V29 test report")
    config_values = dict(checkpoint["config"])
    config_values["device"] = args.device or config_values["device"]
    if args.num_workers is not None:
        config_values["num_workers"] = int(args.num_workers)
    if args.evaluation_batch_size is not None:
        config_values["evaluation_batch_size"] = int(args.evaluation_batch_size)
    config = TrainingConfig(**config_values)
    print("V29 validation seal verified; reading test inputs now", flush=True)
    test_manifest = args.test_manifest.expanduser().resolve()
    anchor_test = args.anchor_test.expanduser().resolve()
    v22_test = args.v22_test.expanduser().resolve()
    test = OptionalStructureDataset(
        test_manifest,
        selection["protein_cache"],
        selection["ligand_cache"],
        with_structure=True,
        limit=args.limit_test,
    )
    model = OptionalStructureSealedResidualDTA(**checkpoint["model_kwargs"]).to(config.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    # Re-seal after loading the V29 artifact.  This is deliberately repeated
    # at test time so an older EMA checkpoint can never alter the frozen V22
    # reference path used for the validation-selected residual.
    load_sealed_v22_backbone(model, selection["sealed_v22_checkpoint"])
    v29 = predict(model, test, config, checkpoint["label_statistics"])
    frozen_v22 = load_aligned_predictions(
        v22_test, v29["sample_ids"], v29["labels"], v29["dataset_index"]
    )
    equivalence = _backbone_equivalence(v29["backbone_predictions"], frozen_v22)
    if not equivalence["passed"]:
        raise RuntimeError(
            "V29 frozen V22 test backbone mismatch: %.8f"
            % equivalence["maximum_absolute_difference"]
        )
    anchor_predictions = load_aligned_predictions(
        anchor_test, v29["sample_ids"], v29["labels"], v29["dataset_index"]
    )
    binary_route = load_aligned_binary_route(
        anchor_test, v29["sample_ids"], v29["labels"], v29["dataset_index"]
    )
    final_predictions = apply_sealed_blends(
        v29["dataset_index"],
        anchor_predictions,
        v29["predictions"],
        selection["anchor_blend_selection"],
    )
    final_metrics = affinity_only_metrics(
        _dataset_metrics(v29["labels"], final_predictions, v29["dataset_index"])
    )
    anchor_metrics = affinity_only_metrics(
        _dataset_metrics(v29["labels"], anchor_predictions, v29["dataset_index"])
    )
    np.savez_compressed(
        prediction_path,
        sample_ids=np.asarray(v29["sample_ids"]),
        labels=v29["labels"],
        dataset_index=v29["dataset_index"],
        predictions=final_predictions,
        anchor_predictions=anchor_predictions,
        v29_predictions=v29["predictions"],
        v29_base_predictions=v29["base_predictions"],
        v22_backbone_predictions=v29["backbone_predictions"],
        structure_gate=v29["structure_gate"],
        structure_delta=v29["structure_delta"],
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
        "v22_test": str(v22_test),
        "v22_test_sha256": sha256_file(v22_test),
        "backbone_equivalence": equivalence,
        "test_metrics": final_metrics,
        "anchor_test_metrics": anchor_metrics,
        "frozen_v26_binary_test_metrics": frozen_binary_metrics(
            v29["labels"],
            v29["dataset_index"],
            binary_route["scores"],
            binary_route["predictions"],
        ),
        "standalone_v29_test_metrics": v29["metrics"],
        "standalone_v29_base_test_metrics": v29["base_metrics"],
        "standalone_v22_backbone_test_metrics": v29["backbone_metrics"],
        "delta_vs_anchor": deltas,
        "counts": {"test": len(test)},
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "test_used_for_selection": False,
        "test_configuration_reselected": False,
        "binary_route": "V26 binary scores and decisions are copied unchanged; V29 selection changes LP affinity only",
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
    parser.add_argument("--v22-test", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--limit-test", type=int)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
