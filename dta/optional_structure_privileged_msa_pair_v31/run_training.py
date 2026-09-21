"""Train V31 and seal LP-only validation selection without reading test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dta.optional_structure_staged_v22.model import OptionalStructureStagedDTA
from dta.optional_structure_staged_v22.training import _amp_dtype, _dataset_metrics

from .binary_route import affinity_only_metrics, frozen_binary_metrics, load_aligned_binary_route
from .cli import add_config_arguments, config_from_args
from .data import OptionalStructureDataset, manifest_fingerprint, move_batch_to_device, sha256_file
from .fit import fit
from .model import OptionalStructurePrivilegedMSAPairDTA
from .selection import load_aligned_predictions, select_validation_blends
from .teacher_cache import load_train_teacher_cache
from .training import (
    VERSION,
    TrainingConfig,
    build_loader,
    structure_validation_gate,
    student_validation_gate,
)


def _backbone_equivalence(student_backbone, v22_predictions):
    difference = np.asarray(student_backbone, dtype=np.float64) - np.asarray(v22_predictions, dtype=np.float64)
    maximum = float(np.max(np.abs(difference)))
    return {
        "passed": bool(maximum <= 1.0e-4),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "tolerance": 1.0e-4,
    }


def _verify_teacher_v22_lineage(teacher, sealed_v22_checkpoint):
    """Require the OOF residual target and frozen student to share V22 state.

    The scalar teacher is explicitly defined relative to V22 full train
    predictions.  A different V22 checkpoint at student time would change the
    residual's meaning even though both artifacts might be individually
    well-formed, so accept only an identical sealed checkpoint digest.
    """

    metadata = teacher.get("metadata", {})
    recorded_path = str(metadata.get("sealed_v22_checkpoint", ""))
    recorded_sha256 = str(metadata.get("sealed_v22_checkpoint_sha256", ""))
    actual_path = Path(sealed_v22_checkpoint).expanduser().resolve()
    actual_sha256 = sha256_file(actual_path)
    if not recorded_path or not recorded_sha256:
        raise ValueError("V31 teacher cache lacks frozen V22 train-export lineage")
    if recorded_sha256 != actual_sha256:
        raise ValueError("V31 teacher V22 checkpoint differs from frozen student backbone")
    return {
        "teacher_v22_checkpoint": recorded_path,
        "teacher_v22_checkpoint_sha256": recorded_sha256,
        "student_v22_checkpoint": str(actual_path),
        "student_v22_checkpoint_sha256": actual_sha256,
        "matched": True,
    }


def _verify_validation_anchor_archive(path, name, version, phase, test_read_field):
    """Check that a validation archive is its named model's sealed artifact.

    This reads only the sibling validation selection JSON and never opens a
    test manifest or test prediction.  It prevents silently substituting a
    same-shaped prediction file as the V22 or V26 validation anchor.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("V31 %s validation archive is missing" % name)
    selection_path = path.with_name("selection.json")
    if not selection_path.is_file():
        raise FileNotFoundError("V31 %s validation archive lacks selection.json" % name)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != version or selection.get("phase") != phase:
        raise ValueError("V31 %s validation selection version/phase mismatch" % name)
    if selection.get("status") != "validation_passed" or not selection.get("validation_selected"):
        raise ValueError("V31 %s validation selection did not pass" % name)
    if selection.get(test_read_field) is not False:
        raise ValueError("V31 %s validation selection accessed test inputs" % name)
    selected_path = Path(selection.get("valid_predictions", "")).expanduser().resolve()
    if selected_path != path:
        raise ValueError("V31 %s validation archive differs from sealed selection" % name)
    observed_sha256 = sha256_file(path)
    if selection.get("valid_predictions_sha256") != observed_sha256:
        raise ValueError("V31 %s validation archive checksum differs from sealed selection" % name)
    return {
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "validation_predictions": str(path),
        "validation_predictions_sha256": observed_sha256,
        "test_read_field": str(test_read_field),
        "test_inputs_read": False,
    }


def _same_batch_backbone_equivalence(v31_model, v22_model, batch, config):
    device = torch.device(config.device)
    amp_dtype = _amp_dtype(config.amp_dtype)
    v31_model.eval()
    v22_model.eval()
    moved = move_batch_to_device(batch, device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
            reference = v22_model(moved)
            wrapped = v31_model(moved)
    result = _backbone_equivalence(
        wrapped.backbone_prediction.float().cpu().numpy(), reference.prediction.float().cpu().numpy()
    )
    result.update(
        {
            "mode": "same_padded_batch_independent_v22_reference",
            "count": int(wrapped.backbone_prediction.numel()),
            "structured_count": int(batch.get("has_3d", torch.zeros(0)).sum().item()),
            "sample_ids": [str(value) for value in batch.get("sample_ids", ())],
        }
    )
    return result


def _validation_batch_backbone_equivalence(v31_model, valid_dataset, config, sealed_v22_checkpoint):
    checkpoint = torch.load(str(sealed_v22_checkpoint), map_location="cpu")
    if checkpoint.get("version") != "optional-structure-staged-v22" or checkpoint.get("phase") != "structure":
        raise ValueError("V31 same-batch reference is not a V22 structure checkpoint")
    if dict(checkpoint["model_kwargs"]) != v31_model.backbone_kwargs():
        raise ValueError("V31 same-batch V22 model kwargs differ")
    reference = OptionalStructureStagedDTA(**checkpoint["model_kwargs"]).to(config.device)
    reference.load_state_dict(checkpoint["model_state"], strict=True)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    maximum, absolute_sum, count, structured_count, worst = 0.0, 0.0, 0, 0, None
    for batch in build_loader(valid_dataset, config):
        audit = _same_batch_backbone_equivalence(v31_model, reference, batch, config)
        value = float(audit["maximum_absolute_difference"])
        if worst is None or value > maximum:
            maximum, worst = value, audit["sample_ids"]
        absolute_sum += float(audit["mean_absolute_difference"]) * int(audit["count"])
        count += int(audit["count"])
        structured_count += int(audit["structured_count"])
    if count == 0 or structured_count == 0:
        raise RuntimeError("V31 same-batch audit has no validation/structured examples")
    return {
        "passed": bool(maximum <= 1.0e-4),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": absolute_sum / float(count),
        "tolerance": 1.0e-4,
        "mode": "same_padded_batch_independent_v22_reference",
        "count": count,
        "structured_count": structured_count,
        "worst_batch_sample_ids": worst,
    }


def _archive_equivalence_policy(args, config, sealed_v22_checkpoint):
    sealed = torch.load(str(sealed_v22_checkpoint), map_location="cpu")
    v22_batch_size = int(sealed["config"]["evaluation_batch_size"])
    if args.limit_valid is not None:
        return {
            "required": False,
            "reason": "limited_validation_rebatches_bfloat16_dynamic_padding",
            "v22_evaluation_batch_size": v22_batch_size,
            "v31_evaluation_batch_size": int(config.evaluation_batch_size),
        }
    if int(config.evaluation_batch_size) != v22_batch_size:
        return {
            "required": False,
            "reason": "evaluation_batch_size_differs_from_v22_archive",
            "v22_evaluation_batch_size": v22_batch_size,
            "v31_evaluation_batch_size": int(config.evaluation_batch_size),
        }
    return {
        "required": True,
        "reason": "full_validation_uses_v22_archive_batch_geometry",
        "v22_evaluation_batch_size": v22_batch_size,
        "v31_evaluation_batch_size": int(config.evaluation_batch_size),
    }


def _formal_validation_policy(args, config, archive_policy):
    """Decide whether a selection is eligible to unlock the formal test CLI.

    Smoke runs intentionally use limited manifests and relaxed gates.  They
    may still write a useful selection artifact, but must never be able to
    unlock the held-out test set merely because those relaxed gates pass.
    Stricter-than-default gates are allowed for a formal run; weaker gates are
    not.
    """

    default = TrainingConfig()
    checks = {
        "full_train_manifest": args.limit_train is None,
        "full_validation_manifest": args.limit_valid is None,
        "v22_archive_batch_geometry": bool(archive_policy["required"]),
        "structure_ci_gate_not_relaxed": float(config.minimum_structure_ci_gain)
        >= float(default.minimum_structure_ci_gain),
        "structure_rmse_gate_not_relaxed": float(config.maximum_structure_rmse_increase)
        <= float(default.maximum_structure_rmse_increase),
        "structure_spearman_gate_not_relaxed": float(config.minimum_structure_spearman_gain)
        >= float(default.minimum_structure_spearman_gain),
        "student_ci_gate_not_relaxed": float(config.minimum_student_ci_gain)
        >= float(default.minimum_student_ci_gain),
        "student_rmse_gate_not_relaxed": float(config.maximum_student_rmse_increase)
        <= float(default.maximum_student_rmse_increase),
        "student_spearman_gate_not_relaxed": float(config.minimum_student_spearman_gain)
        >= float(default.minimum_student_spearman_gain),
        "anchor_ci_gate_not_relaxed": float(args.minimum_anchor_ci_gain) >= 0.0005,
        "anchor_rmse_gate_not_relaxed": float(args.maximum_anchor_rmse_increase) <= 0.003,
        "anchor_spearman_gate_not_relaxed": float(args.minimum_anchor_spearman_gain) >= 0.0,
        "anchor_fold_gate_not_relaxed": int(args.minimum_positive_folds) >= 3,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "limits": {"train": args.limit_train, "valid": args.limit_valid},
        "baseline_criteria": {
            "minimum_structure_ci_gain": float(default.minimum_structure_ci_gain),
            "maximum_structure_rmse_increase": float(default.maximum_structure_rmse_increase),
            "minimum_structure_spearman_gain": float(default.minimum_structure_spearman_gain),
            "minimum_student_ci_gain": float(default.minimum_student_ci_gain),
            "maximum_student_rmse_increase": float(default.maximum_student_rmse_increase),
            "minimum_student_spearman_gain": float(default.minimum_student_spearman_gain),
            "minimum_anchor_ci_gain": 0.0005,
            "maximum_anchor_rmse_increase": 0.003,
            "minimum_anchor_spearman_gain": 0.0,
            "minimum_positive_folds": 3,
        },
    }


def run(args):
    output = args.output.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    valid_manifest = args.valid_manifest.expanduser().resolve()
    protein_cache = args.protein_cache.expanduser().resolve()
    ligand_cache = args.ligand_cache.expanduser().resolve()
    sealed_v22_checkpoint = args.sealed_v22_checkpoint.expanduser().resolve()
    v22_valid = args.v22_valid.expanduser().resolve()
    anchor_valid = args.anchor_valid.expanduser().resolve()
    teacher_cache = args.teacher_cache.expanduser().resolve()
    # Validate the train-only artifact before model construction.  This loader
    # has no valid/test-manifest parameter by design.
    teacher = load_train_teacher_cache(teacher_cache, train_manifest)
    teacher_v22_lineage = _verify_teacher_v22_lineage(teacher, sealed_v22_checkpoint)
    v22_validation_lineage = _verify_validation_anchor_archive(
        v22_valid,
        "V22",
        "optional-structure-staged-v22",
        "structure_validation_selection",
        "test_manifest_read",
    )
    anchor_validation_lineage = _verify_validation_anchor_archive(
        anchor_valid,
        "V26",
        "optional-structure-simplex-router-v26",
        "validation_selection_sealed",
        "test_inputs_read",
    )
    train = OptionalStructureDataset(
        train_manifest,
        protein_cache,
        ligand_cache,
        with_structure=True,
        limit=args.limit_train,
        teacher_predictions=teacher["predictions_by_sample_id"],
    )
    valid = OptionalStructureDataset(
        valid_manifest, protein_cache, ligand_cache, with_structure=True, limit=args.limit_valid
    )
    detected_width = int(train[0]["protein_embeddings"].shape[1])
    config = config_from_args(args, detected_width)
    print(
        json.dumps(
            {
                "event": "v31_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid)},
                "sealed_v22_checkpoint": str(sealed_v22_checkpoint),
                "v22_valid": str(v22_valid),
                "anchor_valid": str(anchor_valid),
                "teacher_cache": str(teacher_cache),
                "teacher_labels_scope": teacher["metadata"]["labels_scope"],
                "test_manifest_argument_exists": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    selected_model, checkpoint, valid_result, history = fit(
        train,
        valid,
        output,
        config,
        protein_cache,
        ligand_cache,
        sealed_v22_checkpoint=sealed_v22_checkpoint,
        teacher_cache=teacher_cache,
        resume=args.resume,
    )
    same_batch_equivalence = _validation_batch_backbone_equivalence(
        selected_model, valid, config, sealed_v22_checkpoint
    )
    if not same_batch_equivalence["passed"]:
        raise RuntimeError("V31 same-batch frozen V22 mismatch: %.8f" % same_batch_equivalence["maximum_absolute_difference"])
    v22_predictions = load_aligned_predictions(
        v22_valid, valid_result["sample_ids"], valid_result["labels"], valid_result["dataset_index"]
    )
    archive_equivalence = _backbone_equivalence(valid_result["backbone_predictions"], v22_predictions)
    archive_policy = _archive_equivalence_policy(args, config, sealed_v22_checkpoint)
    formal_policy = _formal_validation_policy(args, config, archive_policy)
    equivalence = {
        "passed": bool(same_batch_equivalence["passed"] and (not archive_policy["required"] or archive_equivalence["passed"])),
        "mode": "same_batch_and_full_archive" if archive_policy["required"] else "same_batch_reference_limited_archive_diagnostic",
        "same_batch_reference": same_batch_equivalence,
        "archive": dict(archive_equivalence, **archive_policy),
    }
    if archive_policy["required"] and not archive_equivalence["passed"]:
        raise RuntimeError("V31 frozen V22 full-archive mismatch: %.8f" % archive_equivalence["maximum_absolute_difference"])
    structure_gate = structure_validation_gate(valid_result, config)
    student_gate = student_validation_gate(valid_result, config)
    anchor_predictions = load_aligned_predictions(
        anchor_valid, valid_result["sample_ids"], valid_result["labels"], valid_result["dataset_index"]
    )
    binary_route = load_aligned_binary_route(
        anchor_valid, valid_result["sample_ids"], valid_result["labels"], valid_result["dataset_index"]
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
        _dataset_metrics(valid_result["labels"], blend["predictions"], valid_result["dataset_index"])
    )
    selected_prediction_path = output / "selected_valid_predictions.npz"
    np.savez_compressed(
        selected_prediction_path,
        sample_ids=np.asarray(valid_result["sample_ids"]),
        labels=valid_result["labels"],
        dataset_index=valid_result["dataset_index"],
        predictions=blend["predictions"],
        anchor_predictions=anchor_predictions,
        v31_predictions=valid_result["predictions"],
        v22_backbone_predictions=valid_result["backbone_predictions"],
        v31_base_predictions=valid_result["base_predictions"],
        teacher_head_predictions=valid_result["teacher_predictions"],
        binary_scores=binary_route["scores"],
        binary_predictions=binary_route["predictions"],
    )
    passed = bool(equivalence["passed"] and structure_gate["passed"] and student_gate["passed"] and blend["any_promoted"])
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
        "standalone_v22_backbone_validation_metrics": valid_result["backbone_metrics"],
        "teacher_head_validation_metrics": valid_result["teacher_head_metrics"],
        "selected_validation_metrics": final_metrics,
        "frozen_v26_binary_validation_metrics": frozen_binary_metrics(
            valid_result["labels"], valid_result["dataset_index"], binary_route["scores"], binary_route["predictions"]
        ),
        "binary_route": "V26 binary scores and decisions are copied unchanged; V31 selection is LP affinity-only",
        "backbone_equivalence": equivalence,
        "structure_validation_gate": structure_gate,
        "student_validation_gate": student_gate,
        "anchor_blend_selection": {key: value for key, value in blend.items() if key != "predictions"},
        "counts": {"train": len(train), "valid": len(valid)},
        "limits": {"train": args.limit_train, "valid": args.limit_valid},
        "formal_validation_selection": bool(formal_policy["passed"]),
        "formal_validation_policy": formal_policy,
        "history_length": len(history),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": manifest_fingerprint(train_manifest),
        "valid_manifest": str(valid_manifest),
        "valid_manifest_sha256": manifest_fingerprint(valid_manifest),
        "protein_cache": str(protein_cache),
        "ligand_cache": str(ligand_cache),
        "sealed_v22_checkpoint": str(sealed_v22_checkpoint),
        "sealed_v22_checkpoint_sha256": sha256_file(sealed_v22_checkpoint),
        "v22_valid": str(v22_valid),
        "v22_valid_sha256": sha256_file(v22_valid),
        "anchor_valid": str(anchor_valid),
        "anchor_valid_sha256": sha256_file(anchor_valid),
        # These are pre-registered by deterministic sibling derivation.  No
        # test manifest or test prediction archive is opened during training
        # or validation selection.  The test CLI later refuses substitutions.
        "expected_test_inputs": {
            "test_manifest": str(valid_manifest.with_name("test.jsonl")),
            "anchor_test": str(anchor_valid.with_name("test_predictions.npz")),
            "v22_test": str(v22_valid.with_name("test_predictions.npz")),
        },
        "teacher_cache": teacher["cache_dir"],
        "teacher_cache_archive": teacher["archive"],
        "teacher_cache_archive_sha256": teacher["archive_sha256"],
        "teacher_cache_metadata_sha256": teacher["metadata_sha256"],
        "teacher_target_regime": teacher["metadata"]["teacher_target_regime"],
        "teacher_initialization_policies": teacher["metadata"].get(
            "initialization_policies", []
        ),
        "teacher_full_ancestry_train_only": bool(
            teacher["metadata"].get("full_ancestry_train_only", False)
        ),
        "teacher_v22_train_predictions_sha256": teacher["metadata"].get(
            "v22_train_predictions_sha256"
        ),
        "teacher_v22_lineage": teacher_v22_lineage,
        "v22_validation_lineage": v22_validation_lineage,
        "anchor_validation_lineage": anchor_validation_lineage,
        "teacher_valid_manifest_read": False,
        "teacher_test_manifest_read": False,
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
    (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selection, sort_keys=True), flush=True)
    return selection


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--protein-cache", type=Path, required=True)
    parser.add_argument("--ligand-cache", type=Path, required=True)
    parser.add_argument("--sealed-v22-checkpoint", type=Path, required=True)
    parser.add_argument("--v22-valid", type=Path, required=True)
    parser.add_argument("--anchor-valid", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minimum-anchor-ci-gain", type=float, default=0.0005)
    parser.add_argument("--maximum-anchor-rmse-increase", type=float, default=0.003)
    parser.add_argument("--minimum-anchor-spearman-gain", type=float, default=0.0)
    parser.add_argument("--minimum-positive-folds", type=int, default=3)
    add_config_arguments(parser)
    return parser


def main(argv=None):
    return 0 if run(build_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    main()
