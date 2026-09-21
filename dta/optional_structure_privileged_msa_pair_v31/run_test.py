"""Verify the V31 validation seal, then run one frozen teacher-free test report."""

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
from .model import OptionalStructurePrivilegedMSAPairDTA, load_sealed_v22_backbone
from .run_training import _backbone_equivalence
from .selection import apply_sealed_blends, load_aligned_predictions
from .training import TrainingConfig, VERSION, predict


def _verify_selection(output):
    output = output.expanduser().resolve()
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != VERSION or selection.get("phase") != "validation_selection":
        raise ValueError("V31 selection version/phase mismatch")
    if selection.get("status") != "validation_passed" or not selection.get("validation_selected"):
        raise ValueError("V31 validation gate did not pass")
    if not selection.get("formal_validation_selection"):
        raise ValueError("V31 selection is a smoke/relaxed validation run and cannot unlock test")
    limits = selection.get("limits", {})
    if limits.get("train") is not None or limits.get("valid") is not None:
        raise ValueError("V31 formal test requires an unrestricted validation selection")
    policy = selection.get("formal_validation_policy", {})
    if not policy.get("passed") or not all(policy.get("checks", {}).values()):
        raise ValueError("V31 formal validation policy is incomplete or relaxed")
    expected_inputs = selection.get("expected_test_inputs")
    if not isinstance(expected_inputs, dict) or set(expected_inputs) != {
        "test_manifest",
        "anchor_test",
        "v22_test",
    }:
        raise ValueError("V31 selection lacks pre-registered test input paths")
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
            raise ValueError("V31 sealed %s checksum mismatch" % path_name)
    if manifest_fingerprint(selection["train_manifest"]) != selection["train_manifest_sha256"]:
        raise ValueError("V31 training manifest changed after validation")
    if manifest_fingerprint(selection["valid_manifest"]) != selection["valid_manifest_sha256"]:
        raise ValueError("V31 validation manifest changed after selection")
    if selection.get("teacher_valid_manifest_read") or selection.get("teacher_test_manifest_read"):
        raise ValueError("V31 teacher lineage is not train-only")
    checkpoint = torch.load(str(selection["checkpoint"]), map_location="cpu")
    if checkpoint.get("version") != VERSION or checkpoint.get("phase") != "privileged_msa":
        raise ValueError("V31 sealed checkpoint phase mismatch")
    if checkpoint.get("train_manifest_sha256") != selection["train_manifest_sha256"]:
        raise ValueError("V31 checkpoint training lineage mismatch")
    if checkpoint.get("valid_manifest_sha256") != selection["valid_manifest_sha256"]:
        raise ValueError("V31 checkpoint validation lineage mismatch")
    if checkpoint.get("protein_cache") != cache_lineage(selection["protein_cache"]):
        raise ValueError("V31 protein cache lineage mismatch")
    if checkpoint.get("ligand_cache") != cache_lineage(selection["ligand_cache"]):
        raise ValueError("V31 ligand cache lineage mismatch")
    return selection_path, selection, checkpoint


def _verify_requested_test_configuration(args, selection, checkpoint):
    """Reject test-time subsets, geometry changes, and anchor substitution."""

    if getattr(args, "limit_test", None) is not None:
        raise ValueError("V31 formal test does not permit --limit-test")
    expected_batch_size = int(checkpoint["config"]["evaluation_batch_size"])
    if (
        args.evaluation_batch_size is not None
        and int(args.evaluation_batch_size) != expected_batch_size
    ):
        raise ValueError(
            "V31 test evaluation_batch_size must match sealed validation geometry"
        )
    expected = selection["expected_test_inputs"]
    requested = {
        "test_manifest": args.test_manifest.expanduser().resolve(),
        "anchor_test": args.anchor_test.expanduser().resolve(),
        "v22_test": args.v22_test.expanduser().resolve(),
    }
    for name, actual in requested.items():
        if actual != Path(expected[name]).expanduser().resolve():
            raise ValueError("V31 test %s differs from its validation-sealed path" % name)
    return requested


def _verify_reported_anchor_test_lineage(valid_archive, test_archive, test_manifest, name):
    """Verify that a fixed test archive belongs to its validation anchor run.

    This is deliberately invoked only after the V31 validation seal has
    passed.  It reads prior-model report metadata, not V31 selection inputs,
    and prevents a same-shaped arbitrary prediction archive from replacing
    the frozen V22/V26 anchor at test time.
    """

    result_path = Path(valid_archive).expanduser().resolve().with_name("result.json")
    if not result_path.is_file():
        raise FileNotFoundError("V31 %s anchor lacks result.json" % name)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if report.get("phase") != "test_report" or report.get("status") != "reported":
        raise ValueError("V31 %s anchor result is not a sealed test report" % name)
    if report.get("test_used_for_selection") is not False:
        raise ValueError("V31 %s anchor test report was not frozen" % name)
    if (
        "test_configuration_reselected" in report
        and report.get("test_configuration_reselected") is not False
    ):
        raise ValueError("V31 %s anchor test configuration was reselected" % name)
    if Path(report.get("predictions", "")).expanduser().resolve() != Path(test_archive).expanduser().resolve():
        raise ValueError("V31 %s test archive differs from its anchor result" % name)
    if report.get("predictions_sha256") != sha256_file(test_archive):
        raise ValueError("V31 %s test archive checksum differs from anchor result" % name)
    if report.get("test_manifest_sha256") != manifest_fingerprint(test_manifest):
        raise ValueError("V31 %s test manifest differs from anchor result" % name)
    return {
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "predictions_sha256": sha256_file(test_archive),
        "test_manifest_sha256": manifest_fingerprint(test_manifest),
    }


def run(args):
    output = args.selection_output.expanduser().resolve()
    selection_path, selection, checkpoint = _verify_selection(output)
    requested = _verify_requested_test_configuration(args, selection, checkpoint)
    result_path = output / "result.json"
    prediction_path = output / "test_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite an existing V31 test report")
    config_values = dict(checkpoint["config"])
    config_values["device"] = args.device or config_values["device"]
    if args.num_workers is not None:
        config_values["num_workers"] = int(args.num_workers)
    if args.evaluation_batch_size is not None:
        config_values["evaluation_batch_size"] = int(args.evaluation_batch_size)
    config = TrainingConfig(**config_values)
    # The only point at which test paths are resolved.  No teacher-cache path
    # exists in this CLI or in model.forward.
    print("V31 validation seal verified; reading test inputs now", flush=True)
    test_manifest = requested["test_manifest"]
    anchor_test = requested["anchor_test"]
    v22_test = requested["v22_test"]
    anchor_lineage = _verify_reported_anchor_test_lineage(
        selection["anchor_valid"], anchor_test, test_manifest, "V26"
    )
    v22_lineage = _verify_reported_anchor_test_lineage(
        selection["v22_valid"], v22_test, test_manifest, "V22"
    )
    test = OptionalStructureDataset(
        test_manifest,
        selection["protein_cache"],
        selection["ligand_cache"],
        with_structure=True,
        limit=None,
    )
    model = OptionalStructurePrivilegedMSAPairDTA(**checkpoint["model_kwargs"]).to(config.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    load_sealed_v22_backbone(model, selection["sealed_v22_checkpoint"])
    v31 = predict(model, test, config, checkpoint["label_statistics"])
    frozen_v22 = load_aligned_predictions(
        v22_test, v31["sample_ids"], v31["labels"], v31["dataset_index"]
    )
    equivalence = _backbone_equivalence(v31["backbone_predictions"], frozen_v22)
    if not equivalence["passed"]:
        raise RuntimeError("V31 frozen V22 test backbone mismatch: %.8f" % equivalence["maximum_absolute_difference"])
    anchor_predictions = load_aligned_predictions(
        anchor_test, v31["sample_ids"], v31["labels"], v31["dataset_index"]
    )
    binary_route = load_aligned_binary_route(
        anchor_test, v31["sample_ids"], v31["labels"], v31["dataset_index"]
    )
    final_predictions = apply_sealed_blends(
        v31["dataset_index"], anchor_predictions, v31["predictions"], selection["anchor_blend_selection"]
    )
    final_metrics = affinity_only_metrics(_dataset_metrics(v31["labels"], final_predictions, v31["dataset_index"]))
    anchor_metrics = affinity_only_metrics(_dataset_metrics(v31["labels"], anchor_predictions, v31["dataset_index"]))
    np.savez_compressed(
        prediction_path,
        sample_ids=np.asarray(v31["sample_ids"]),
        labels=v31["labels"],
        dataset_index=v31["dataset_index"],
        predictions=final_predictions,
        anchor_predictions=anchor_predictions,
        v31_predictions=v31["predictions"],
        v31_base_predictions=v31["base_predictions"],
        v22_backbone_predictions=v31["backbone_predictions"],
        teacher_head_predictions=v31["teacher_predictions"],
        structure_gate=v31["structure_gate"],
        structure_delta=v31["structure_delta"],
        teacher_delta=v31["teacher_delta"],
        teacher_mix=v31["teacher_mix"],
        binary_scores=binary_route["scores"],
        binary_predictions=binary_route["predictions"],
    )
    deltas = {
        name: {key: float(final_metrics[name][key] - anchor_metrics[name][key]) for key in ("CI", "RMSE", "MAE", "Pearson", "Spearman")}
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
        "frozen_v26_binary_test_metrics": frozen_binary_metrics(v31["labels"], v31["dataset_index"], binary_route["scores"], binary_route["predictions"]),
        "standalone_v31_test_metrics": v31["metrics"],
        "standalone_v31_base_test_metrics": v31["base_metrics"],
        "standalone_v22_backbone_test_metrics": v31["backbone_metrics"],
        "teacher_head_test_metrics": v31["teacher_head_metrics"],
        "delta_vs_anchor": deltas,
        "counts": {"test": len(test)},
        "limits": {"test": None},
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "test_used_for_selection": False,
        "test_configuration_reselected": False,
        "formal_test_report": True,
        "anchor_test_lineage": anchor_lineage,
        "v22_test_lineage": v22_lineage,
        "teacher_runtime_dependency": False,
        "teacher_initialization_policies": selection.get(
            "teacher_initialization_policies", []
        ),
        "teacher_full_ancestry_train_only": bool(
            selection.get("teacher_full_ancestry_train_only", False)
        ),
        "binary_route": "V26 binary scores and decisions are copied unchanged; V31 selection changes LP affinity only",
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
    return parser


def main(argv=None):
    return 0 if run(build_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    main()
