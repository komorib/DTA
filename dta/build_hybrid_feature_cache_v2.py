"""Build a provenance-bound frozen multiscale feature cache."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import torch

from data_utils.hybrid_feature_cache_v2 import HybridFeatureCacheWriter
from dta.hybrid_affinity_training import prepare_hybrid_sample
from dta.train_hybrid_affinity import (
    build_gerna_backbone,
    build_hybrid_dataset,
    build_interface_backbone,
    build_preflight_binding,
    load_backbone_checkpoint,
    validate_manifest_split,
)
from utils.experiment_integrity import manifest_fingerprint


GERNA_FEATURE_FIELDS = (
    "gerna_2d_molecule",
    "gerna_2d_protein",
    "gerna_3d_molecule",
    "gerna_3d_protein",
    "gerna_2d_interaction",
    "gerna_3d_interaction",
    "gerna_head",
)

INTERFACE_FEATURE_FIELDS = (
    "interface_ligand_nodes",
    "interface_protein_nodes",
    "interface_residue_nodes",
    "interface_ligand_coords",
    "interface_protein_coords",
    "interface_residue_coords",
    "interface_ligand_summary",
    "interface_protein_summary",
    "interface_local_summary",
    "interface_head",
    "interface_score",
    "interface_projection",
)

_BATCHED_FEATURE_FIELDS = frozenset(
    {
        "gerna_2d_molecule",
        "gerna_2d_protein",
        "gerna_3d_molecule",
        "gerna_3d_protein",
        "gerna_2d_interaction",
        "gerna_3d_interaction",
        "gerna_head",
        "interface_ligand_summary",
        "interface_protein_summary",
        "interface_local_summary",
        "interface_head",
        "interface_score",
        "interface_projection",
    }
)


def _single_batch_tensor(value, name):
    if not torch.is_tensor(value) or value.shape[0] != 1:
        raise ValueError(f"{name} must have leading batch size one")
    return value.detach().squeeze(0).cpu()


def _single_prediction(value, name):
    if not torch.is_tensor(value) or value.shape != (1, 1):
        raise ValueError(f"{name} must have shape [1,1]")
    return value.detach().reshape(1).cpu()


def build_cache_record(
    key,
    target,
    gerna_features,
    gerna_prediction,
    interface_features,
    interface_prediction,
):
    if not isinstance(gerna_features, dict):
        raise ValueError("gerna_features must be a mapping")
    if not isinstance(interface_features, dict):
        raise ValueError("interface_features must be a mapping")
    missing = [
        name
        for name in GERNA_FEATURE_FIELDS
        if name not in gerna_features
    ] + [
        name
        for name in INTERFACE_FEATURE_FIELDS
        if name not in interface_features
    ]
    if missing:
        raise ValueError("backbone features are missing: " + ", ".join(missing))
    if not torch.is_tensor(target) or target.numel() != 1:
        raise ValueError("target must contain exactly one value")

    record = {"key": str(key)}
    for name in GERNA_FEATURE_FIELDS:
        record[name] = _single_batch_tensor(gerna_features[name], name)
    for name in INTERFACE_FEATURE_FIELDS:
        value = interface_features[name]
        if name in _BATCHED_FEATURE_FIELDS:
            record[name] = _single_batch_tensor(value, name)
        else:
            if not torch.is_tensor(value):
                raise ValueError(f"{name} must be a tensor")
            record[name] = value.detach().cpu()
    record["gerna_prediction"] = _single_prediction(
        gerna_prediction, "gerna_prediction"
    )
    record["interface_prediction"] = _single_prediction(
        interface_prediction, "interface_prediction"
    )
    record["label"] = target.detach().reshape(1).cpu()
    return record


def _current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPOSITORY_ROOT),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_metadata(args, manifest, binding, checkpoint_artifacts, expected_keys):
    return {
        "dataset": "LP-PDBBind",
        "split": str(args.split),
        "source_manifest": manifest,
        "source_native_cache_binding": binding,
        "manifest_sha256": manifest["sha256"],
        "native_metadata_sha256": binding["native_metadata_sha256"],
        "gerna_checkpoint_sha256": checkpoint_artifacts["gerna"]["sha256"],
        "interface_checkpoint_sha256": checkpoint_artifacts["interface"]["sha256"],
        "gerna_checkpoint_path": checkpoint_artifacts["gerna"]["path"],
        "interface_checkpoint_path": checkpoint_artifacts["interface"]["path"],
        "git_commit": _current_git_commit(),
        "smoke_limit": int(args.limit) if args.limit else None,
        "source_sample_count": int(manifest["sample_count"]),
        "selected_sample_count": len(expected_keys),
    }


def build_feature_cache(args):
    manifest_payload = validate_manifest_split(args.manifest, args.split)
    manifest = manifest_fingerprint(args.manifest)
    binding = build_preflight_binding(args.manifest, args.native_cache)
    dataset = build_hybrid_dataset(
        args.manifest, args.native_cache, args.split
    )
    expected_keys = [str(sample["lmdb_key"]) for sample in manifest_payload["samples"]]
    if args.limit is not None:
        expected_keys = expected_keys[: int(args.limit)]
    if not expected_keys:
        raise ValueError("cache selection is empty")
    key_to_index = {key: index for index, key in enumerate(dataset.keys)}
    if any(key not in key_to_index for key in expected_keys):
        raise RuntimeError("dataset keys do not match manifest keys")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA cache build requested but CUDA is unavailable")
    gerna = build_gerna_backbone().to(device).eval()
    interface = build_interface_backbone().to(device).eval()
    checkpoint_artifacts = {
        "gerna": load_backbone_checkpoint(gerna, args.gerna_checkpoint),
        "interface": load_backbone_checkpoint(interface, args.interface_checkpoint),
    }
    metadata = _build_metadata(
        args, manifest, binding, checkpoint_artifacts, expected_keys
    )
    writer = HybridFeatureCacheWriter(
        args.output,
        expected_keys=expected_keys,
        metadata=metadata,
        resume=bool(args.resume),
        map_size=int(float(args.map_size_gb) * 1024 ** 3),
    )
    written = 0
    try:
        for key in list(writer.missing_keys):
            sample = dataset[key_to_index[key]]
            gerna_inputs, interface_item, target = prepare_hybrid_sample(
                sample, device
            )
            with torch.inference_mode():
                gerna.request_multiscale_affinity_features()
                try:
                    with torch.autocast(device_type=device.type, enabled=False):
                        gerna_output = gerna(**gerna_inputs)
                    gerna_prediction = gerna_output[0]
                    gerna_features = gerna.take_multiscale_affinity_features()
                except BaseException:
                    gerna.discard_multiscale_affinity_features()
                    raise
                interface_prediction, interface_features = interface(
                    interface_item, return_multiscale_features=True
                )
            writer.write(
                build_cache_record(
                    key,
                    target,
                    gerna_features,
                    gerna_prediction,
                    interface_features,
                    interface_prediction,
                )
            )
            written += 1
            if written % 25 == 0:
                print(json.dumps({"written": written, "key": key}))
        completed = writer.finalize()
    finally:
        writer.close()
    print(json.dumps({"status": "complete", "output": str(args.output), "written": written}))
    return completed


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Cache frozen GerNA and Interface multiscale features."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-cache", required=True)
    parser.add_argument("--split", required=True, choices=("train", "valid"))
    parser.add_argument("--gerna-checkpoint", required=True)
    parser.add_argument("--interface-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--map-size-gb", type=float, default=32.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.map_size_gb <= 0.0:
        raise ValueError("map-size-gb must be positive")
    build_feature_cache(args)


if __name__ == "__main__":
    main()
