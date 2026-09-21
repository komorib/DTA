"""Train and evaluate the isolated LP-PDBbind anchored residual v3 model."""

import argparse
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_utils.hybrid_affinity_dataset import LabelSpreadBatchSampler, collate_hybrid_group
from dta.anchored_residual_v3.data import TeacherAnchoredDataset
from dta.anchored_residual_v3.evaluate import (
    evaluate_selected_model,
    evaluate_teacher_lookup,
    write_test_artifact,
)
from dta.anchored_residual_v3.model import AnchoredResidualModel
from dta.anchored_residual_v3.teacher import load_teacher_artifact
from dta.anchored_residual_v3.training import fit_guarded_stages
from dta.hybrid_e2e_training_v2 import load_cached_fusion_checkpoint
from dta.train_hybrid_affinity import (
    build_gerna_backbone,
    build_interface_backbone,
    load_backbone_checkpoint,
)
from net.hybrid_upper_bound import HybridUpperBoundFusion
from utils.experiment_integrity import atomic_write_json, manifest_fingerprint, sha256_file


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train anchored residual E2E v3.")
    for split in ("train", "valid", "test"):
        parser.add_argument(f"--{split}-manifest", required=True)
        parser.add_argument(f"--{split}-native-cache", required=True)
        parser.add_argument(f"--teacher-{split}", required=True)
    parser.add_argument("--gerna-checkpoint", required=True)
    parser.add_argument("--interface-checkpoint", required=True)
    parser.add_argument("--cached-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--residual-epochs", type=int, default=5)
    parser.add_argument("--fusion-epochs", type=int, default=5)
    parser.add_argument("--selective-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _manifest(path, expected_split):
    fingerprint = manifest_fingerprint(path)
    if fingerprint["split"] != expected_split:
        raise ValueError(f"manifest split must be {expected_split}")
    payload = json.loads(Path(path).read_text())
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    keys = [str(sample["lmdb_key"]) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{expected_split} manifest keys must be unique")
    return payload, fingerprint


def _build_split_dataset(manifest_path, native_cache_path, expected_split):
    from data_utils.hybrid_affinity_dataset import HybridAffinityDataset
    from data_utils.lmdb_dataset import PDBbindLmdbDataset
    from data_utils.local_interface_dataset import NativeInterfaceDataset

    manifest, _ = _manifest(manifest_path, expected_split)
    gerna_dataset = PDBbindLmdbDataset(
        str(Path(manifest_path).resolve()),
        protein_3d_mode="ligand_pocket_residue_esm",
        protein_embedding_mode="esm",
        ligand_pocket_cutoff=8.0,
        disable_protein_2d=True,
    )
    interface_dataset = NativeInterfaceDataset(
        str(Path(manifest_path).resolve()), str(Path(native_cache_path).resolve())
    )
    return HybridAffinityDataset(manifest["samples"], gerna_dataset, interface_dataset)


def _load_teacher(directory, split, fingerprint, checkpoint_sha256):
    directory = Path(directory).expanduser().resolve()
    return load_teacher_artifact(
        directory / "predictions.npz",
        directory / "metadata.json",
        expected_split=split,
        expected_manifest_sha256=fingerprint["sha256"],
        expected_checkpoint_sha256=checkpoint_sha256,
    )


def _validate_teacher_dataset(lookup, dataset, split, limit=None):
    expected_keys = tuple(dataset.keys if limit is None else dataset.keys[:limit])
    expected_labels = dataset.labels if limit is None else dataset.labels[:limit]
    if expected_keys != lookup.keys:
        raise ValueError(f"{split} teacher keys do not match dataset order")
    if not np.allclose(expected_labels, lookup.labels, rtol=0.0, atol=1e-6):
        raise ValueError(f"{split} teacher labels do not match dataset")


def _atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("anchored residual formal training requires CUDA")
    if args.group_size < 2 or args.patience < 1:
        raise ValueError("group-size and patience are invalid")
    stage_epochs = {
        "residual": int(args.residual_epochs),
        "fusion": int(args.fusion_epochs),
        "selective_e2e": int(args.selective_epochs),
    }
    if any(value < 0 for value in stage_epochs.values()) or not any(stage_epochs.values()):
        raise ValueError("at least one nonnegative stage epoch count is required")
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError("v3 output is nonempty; pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    split_inputs = {}
    for split in ("train", "valid", "test"):
        manifest_path = getattr(args, f"{split}_manifest")
        _, fingerprint = _manifest(manifest_path, split)
        dataset = _build_split_dataset(
            manifest_path, getattr(args, f"{split}_native_cache"), split
        )
        split_inputs[split] = {"fingerprint": fingerprint, "dataset": dataset}
    train_keys = set(split_inputs["train"]["dataset"].keys)
    valid_keys = set(split_inputs["valid"]["dataset"].keys)
    test_keys = set(split_inputs["test"]["dataset"].keys)
    if train_keys & valid_keys or train_keys & test_keys or valid_keys & test_keys:
        raise ValueError("train, valid, and test splits overlap")

    limit = args.limit
    if limit is not None and (
        limit < 2
        or any(limit > len(split_inputs[name]["dataset"]) for name in ("train", "valid", "test"))
    ):
        raise ValueError("limit exceeds a split or is below two")
    cached_sha256 = sha256_file(args.cached_checkpoint)
    for split in ("train", "valid", "test"):
        lookup = _load_teacher(
            getattr(args, f"teacher_{split}"), split,
            split_inputs[split]["fingerprint"], cached_sha256,
        )
        _validate_teacher_dataset(
            lookup, split_inputs[split]["dataset"], split, limit=limit
        )
        split_inputs[split]["teacher"] = lookup
        source_dataset = split_inputs[split]["dataset"]
        if limit is not None:
            source_dataset = Subset(source_dataset, range(limit))
        split_inputs[split]["anchored"] = TeacherAnchoredDataset(
            source_dataset, lookup
        )

    train_dataset = split_inputs["train"]["anchored"]
    valid_dataset = split_inputs["valid"]["anchored"]
    test_dataset = split_inputs["test"]["anchored"]
    train_labels = np.asarray(split_inputs["train"]["dataset"].labels)
    if limit is not None:
        train_labels = train_labels[:limit]
    sampler = LabelSpreadBatchSampler(train_labels, args.group_size, args.seed, min_gap=0.05)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0,
                              collate_fn=collate_hybrid_group, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.group_size, shuffle=False,
                              num_workers=0, collate_fn=collate_hybrid_group, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.group_size, shuffle=False,
                             num_workers=0, collate_fn=collate_hybrid_group, pin_memory=True)

    gerna = build_gerna_backbone()
    interface = build_interface_backbone()
    backbones = {
        "gerna": load_backbone_checkpoint(gerna, args.gerna_checkpoint),
        "interface": load_backbone_checkpoint(interface, args.interface_checkpoint),
    }
    fusion = HybridUpperBoundFusion(dropout=args.dropout)
    cached = load_cached_fusion_checkpoint(fusion, args.cached_checkpoint)
    model = AnchoredResidualModel(
        gerna, interface, fusion, dropout=args.dropout
    ).to(device)
    provenance = {
        "version": 3,
        "seed": args.seed,
        "manifests": {split: split_inputs[split]["fingerprint"] for split in split_inputs},
        "backbones": backbones,
        "cached_teacher": cached,
        "limit": limit,
    }

    selected_checkpoint = output_dir / "Model_anchored_selected.pth"

    def on_accept(current_model, evaluation, variant, epoch):
        _atomic_torch_save({
            "model_state_dict": current_model.state_dict(),
            "metrics": dict(evaluation["metrics"]),
            "variant": variant,
            "epoch": epoch,
            "provenance": provenance,
        }, selected_checkpoint)
        np.savez_compressed(
            output_dir / "valid_predictions_selected.npz",
            keys=np.asarray(evaluation["keys"]),
            labels=evaluation["labels"],
            predictions=evaluation["predictions"],
        )

    fit = fit_guarded_stages(
        model, train_loader, valid_loader, device,
        stage_epochs=stage_epochs, patience=args.patience,
        amp_enabled=args.amp, on_accept=on_accept,
    )
    selected = fit["selected"]
    selected.restore(model)
    selection = {
        "status": "complete",
        "selected_variant": selected.variant,
        "selected_validation_metrics": selected.metrics,
        "teacher_validation_metrics": fit["baseline"]["teacher_metrics"],
        "improved_over_teacher": fit["improved_over_teacher"],
    }
    atomic_write_json(output_dir / "selection_status.json", selection)
    atomic_write_json(output_dir / "metrics_history.json", fit["history"])
    atomic_write_json(output_dir / "provenance.json", provenance)

    test_teacher = split_inputs["test"]["teacher"]
    teacher_evaluation = evaluate_teacher_lookup(test_teacher)
    if limit is not None:
        teacher_evaluation = {
            "keys": teacher_evaluation["keys"][:limit],
            "labels": teacher_evaluation["labels"][:limit],
            "predictions": teacher_evaluation["predictions"][:limit],
        }
    teacher_test = write_test_artifact(
        output_dir, teacher_evaluation, split="test", variant="cached_teacher",
        manifest_sha256=split_inputs["test"]["fingerprint"]["sha256"],
        checkpoint_sha256=cached_sha256,
    )
    selected_evaluation = evaluate_selected_model(
        model, test_loader, device, amp_enabled=args.amp
    )
    selected_test = write_test_artifact(
        output_dir, selected_evaluation, split="test", variant="selected",
        manifest_sha256=split_inputs["test"]["fingerprint"]["sha256"],
        checkpoint_sha256=sha256_file(selected_checkpoint),
    )
    result = {
        "status": "complete",
        "selection": selection,
        "cached_teacher_test": teacher_test["metrics"],
        "selected_test": selected_test["metrics"],
        "output": str(output_dir),
    }
    atomic_write_json(output_dir / "summary.json", result)
    return result


def main(argv=None):
    result = run(build_arg_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
