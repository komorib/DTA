"""Stage B: protect the sealed V22 base and train a validation-selected 3D residual."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .cli import add_config_arguments, config_from_args
from .data import OptionalStructureDataset, manifest_fingerprint, sha256_file
from .training import VERSION, TrainingConfig, fit_structure, structure_validation_gate


def _verify_sequence_selection(path, train_manifest, valid_manifest):
    path = path.expanduser().resolve()
    selection_path = path / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != VERSION:
        raise ValueError("sequence selection version mismatch")
    if selection.get("phase") != "sequence_validation_sealed" or selection.get("status") != "sealed":
        raise ValueError("V22 sequence selection is not sealed")
    if sha256_file(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise ValueError("V22 sequence checkpoint checksum mismatch")
    if sha256_file(selection["valid_predictions"]) != selection["valid_predictions_sha256"]:
        raise ValueError("V22 sequence validation prediction checksum mismatch")
    if manifest_fingerprint(train_manifest) != selection["train_manifest_sha256"]:
        raise ValueError("train manifest differs from V22 sequence selection")
    if manifest_fingerprint(valid_manifest) != selection["valid_manifest_sha256"]:
        raise ValueError("validation manifest differs from V22 sequence selection")
    return selection_path, selection


def run(args):
    output = args.output.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    valid_manifest = args.valid_manifest.expanduser().resolve()
    protein_cache = args.protein_cache.expanduser().resolve()
    ligand_cache = args.ligand_cache.expanduser().resolve()
    sequence_selection_path, sequence_selection = _verify_sequence_selection(
        args.sequence_output, train_manifest, valid_manifest
    )
    warm_structure_checkpoint = (
        args.warm_structure_checkpoint.expanduser().resolve()
        if args.warm_structure_checkpoint is not None
        else None
    )
    train = OptionalStructureDataset(
        train_manifest,
        protein_cache,
        ligand_cache,
        with_structure=True,
        limit=args.limit_train,
    )
    valid = OptionalStructureDataset(
        valid_manifest,
        protein_cache,
        ligand_cache,
        with_structure=True,
        limit=args.limit_valid,
    )
    detected_width = int(train[0]["protein_embeddings"].shape[1])
    config = config_from_args(args, detected_width)
    print(
        json.dumps(
            {
                "event": "v22_structure_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid)},
                "sequence_selection": str(sequence_selection_path),
                "warm_structure_checkpoint": str(warm_structure_checkpoint) if warm_structure_checkpoint else None,
                "test_manifest_argument_exists": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _, checkpoint, valid_result, history = fit_structure(
        train,
        valid,
        output,
        config,
        protein_cache,
        ligand_cache,
        sequence_checkpoint=sequence_selection["checkpoint"],
        warm_structure_checkpoint=warm_structure_checkpoint,
        resume=args.resume,
    )
    gate = structure_validation_gate(valid_result, config)
    checkpoint_path = output / "best.pt"
    prediction_path = output / "valid_predictions.npz"
    selection = {
        "version": VERSION,
        "phase": "structure_validation_selection",
        "status": "validation_passed" if gate["passed"] else "validation_failed",
        "validation_selected": bool(gate["passed"]),
        "output": str(output),
        "best_epoch": int(checkpoint["epoch"]),
        "selected_variant": checkpoint["variant"],
        "selection_rule": checkpoint["selection_rule"],
        "validation_metrics": valid_result["metrics"],
        "validation_base_metrics": valid_result["base_metrics"],
        "validation_gate": gate,
        "counts": {"train": len(train), "valid": len(valid)},
        "history_length": len(history),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": manifest_fingerprint(train_manifest),
        "valid_manifest": str(valid_manifest),
        "valid_manifest_sha256": manifest_fingerprint(valid_manifest),
        "protein_cache": str(protein_cache),
        "ligand_cache": str(ligand_cache),
        "sequence_selection": str(sequence_selection_path),
        "sequence_selection_sha256": sha256_file(sequence_selection_path),
        "sequence_checkpoint_sha256": sequence_selection["checkpoint_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "valid_predictions": str(prediction_path),
        "valid_predictions_sha256": sha256_file(prediction_path),
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
    defaults = replace(
        TrainingConfig(),
        learning_rate=1.0e-4,
        maximum_epochs=40,
        steps_per_epoch=3000,
        patience=10,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-output", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--protein-cache", type=Path, required=True)
    parser.add_argument("--ligand-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-structure-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    return add_config_arguments(parser, defaults)


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
