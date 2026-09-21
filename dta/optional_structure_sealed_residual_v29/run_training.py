"""Train V29 and seal LP-only validation selection without reading test."""

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
from .selection import load_aligned_predictions, select_validation_blends
from .training import VERSION, build_loader, residual_validation_gate, structure_validation_gate


def _backbone_equivalence(v29_backbone, v22_predictions):
    difference = np.asarray(v29_backbone, dtype=np.float64) - np.asarray(
        v22_predictions, dtype=np.float64
    )
    maximum = float(np.max(np.abs(difference)))
    return {
        "passed": bool(maximum <= 1.0e-4),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "tolerance": 1.0e-4,
    }


def _same_batch_backbone_equivalence(v29_model, v22_model, batch, config):
    """Compare the sealed and wrapped V22 paths on exactly one padded batch."""
    device = torch.device(config.device)
    amp_dtype = _amp_dtype(config.amp_dtype)
    v29_model.eval()
    v22_model.eval()
    moved = move_batch_to_device(batch, device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
            reference = v22_model(moved)
            wrapped = v29_model(moved)
    result = _backbone_equivalence(
        wrapped.backbone_prediction.float().cpu().numpy(),
        reference.prediction.float().cpu().numpy(),
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


def _validation_batch_backbone_equivalence(v29_model, valid_dataset, config, sealed_v22_checkpoint):
    """Audit all validation batches without inheriting archive batch layout."""
    checkpoint = torch.load(str(sealed_v22_checkpoint), map_location="cpu")
    if checkpoint.get("version") != "optional-structure-staged-v22":
        raise ValueError("V29 same-batch reference is not a V22 checkpoint")
    if checkpoint.get("phase") != "structure":
        raise ValueError("V29 same-batch reference is not a V22 structure checkpoint")
    if dict(checkpoint["model_kwargs"]) != v29_model.backbone_kwargs():
        raise ValueError("V29 same-batch reference model kwargs differ from sealed backbone")
    reference = OptionalStructureStagedDTA(**checkpoint["model_kwargs"]).to(config.device)
    reference.load_state_dict(checkpoint["model_state"], strict=True)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    maximum, absolute_sum, count, structured_count = 0.0, 0.0, 0, 0
    worst_sample_ids = None
    for batch in build_loader(valid_dataset, config):
        audit = _same_batch_backbone_equivalence(v29_model, reference, batch, config)
        maximum_value = float(audit["maximum_absolute_difference"])
        if worst_sample_ids is None or maximum_value > maximum:
            maximum = maximum_value
            worst_sample_ids = audit["sample_ids"]
        absolute_sum += float(audit["mean_absolute_difference"]) * int(audit["count"])
        count += int(audit["count"])
        structured_count += int(audit["structured_count"])
    if count == 0:
        raise RuntimeError("V29 same-batch audit received no validation samples")
    if structured_count == 0:
        raise RuntimeError("V29 same-batch audit received no structured validation samples")
    return {
        "passed": bool(maximum <= 1.0e-4),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": absolute_sum / float(count),
        "tolerance": 1.0e-4,
        "mode": "same_padded_batch_independent_v22_reference",
        "count": count,
        "structured_count": structured_count,
        "worst_batch_sample_ids": worst_sample_ids,
    }


def _archive_equivalence_policy(args, config, sealed_v22_checkpoint):
    """The archived V22 values are exact only at the original eval batch geometry."""
    sealed = torch.load(str(sealed_v22_checkpoint), map_location="cpu")
    v22_batch_size = int(sealed["config"]["evaluation_batch_size"])
    if args.limit_valid is not None:
        return {
            "required": False,
            "reason": "limited_validation_rebatches_bfloat16_dynamic_padding",
            "v22_evaluation_batch_size": v22_batch_size,
            "v29_evaluation_batch_size": int(config.evaluation_batch_size),
        }
    if int(config.evaluation_batch_size) != v22_batch_size:
        return {
            "required": False,
            "reason": "evaluation_batch_size_differs_from_v22_archive",
            "v22_evaluation_batch_size": v22_batch_size,
            "v29_evaluation_batch_size": int(config.evaluation_batch_size),
        }
    return {
        "required": True,
        "reason": "full_validation_uses_v22_archive_batch_geometry",
        "v22_evaluation_batch_size": v22_batch_size,
        "v29_evaluation_batch_size": int(config.evaluation_batch_size),
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
                "event": "v29_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid)},
                "sealed_v22_checkpoint": str(sealed_v22_checkpoint),
                "v22_valid": str(v22_valid),
                "anchor_valid": str(anchor_valid),
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
        resume=args.resume,
    )
    same_batch_equivalence = _validation_batch_backbone_equivalence(
        selected_model, valid, config, sealed_v22_checkpoint
    )
    if not same_batch_equivalence["passed"]:
        raise RuntimeError(
            "V29 same-batch frozen V22 backbone mismatch: %.8f"
            % same_batch_equivalence["maximum_absolute_difference"]
        )
    v22_predictions = load_aligned_predictions(
        v22_valid,
        valid_result["sample_ids"],
        valid_result["labels"],
        valid_result["dataset_index"],
    )
    archive_equivalence = _backbone_equivalence(
        valid_result["backbone_predictions"], v22_predictions
    )
    archive_policy = _archive_equivalence_policy(args, config, sealed_v22_checkpoint)
    equivalence = {
        "passed": bool(
            same_batch_equivalence["passed"]
            and (not archive_policy["required"] or archive_equivalence["passed"])
        ),
        "mode": (
            "same_batch_and_full_archive"
            if archive_policy["required"]
            else "same_batch_reference_limited_archive_diagnostic"
        ),
        "same_batch_reference": same_batch_equivalence,
        "archive": dict(archive_equivalence, **archive_policy),
    }
    if archive_policy["required"] and not archive_equivalence["passed"]:
        raise RuntimeError(
            "V29 frozen V22 full-archive backbone mismatch: %.8f"
            % archive_equivalence["maximum_absolute_difference"]
        )
    structure_gate = structure_validation_gate(valid_result, config)
    residual_gate = residual_validation_gate(valid_result, config)
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
        v29_predictions=valid_result["predictions"],
        v22_backbone_predictions=valid_result["backbone_predictions"],
        v29_base_predictions=valid_result["base_predictions"],
        binary_scores=binary_route["scores"],
        binary_predictions=binary_route["predictions"],
    )
    passed = bool(
        equivalence["passed"]
        and structure_gate["passed"]
        and residual_gate["passed"]
        and blend["any_promoted"]
    )
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
        "selected_validation_metrics": final_metrics,
        "frozen_v26_binary_validation_metrics": frozen_binary_metrics(
            valid_result["labels"],
            valid_result["dataset_index"],
            binary_route["scores"],
            binary_route["predictions"],
        ),
        "binary_route": "V26 binary scores and decisions are copied unchanged; V29 selection is LP affinity-only",
        "backbone_equivalence": equivalence,
        "structure_validation_gate": structure_gate,
        "residual_validation_gate": residual_gate,
        "anchor_blend_selection": {
            key: value for key, value in blend.items() if key != "predictions"
        },
        "counts": {"train": len(train), "valid": len(valid)},
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
    parser.add_argument("--sealed-v22-checkpoint", type=Path, required=True)
    parser.add_argument("--v22-valid", type=Path, required=True)
    parser.add_argument("--anchor-valid", type=Path, required=True)
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
