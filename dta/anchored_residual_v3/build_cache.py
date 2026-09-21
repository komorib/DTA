"""Build isolated v3 hybrid feature caches, including the test split."""

import argparse
import json

import torch

from data_utils.hybrid_feature_cache_v2 import HybridFeatureCacheWriter
from dta.anchored_residual_v3.train import _build_split_dataset, _manifest
from dta.build_hybrid_feature_cache_v2 import build_cache_record
from dta.hybrid_affinity_training import prepare_hybrid_sample
from dta.train_hybrid_affinity import (
    build_gerna_backbone,
    build_interface_backbone,
    build_preflight_binding,
    load_backbone_checkpoint,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build anchored residual v3 feature cache.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-cache", required=True)
    parser.add_argument("--split", required=True, choices=("train", "valid", "test"))
    parser.add_argument("--gerna-checkpoint", required=True)
    parser.add_argument("--interface-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--map-size-gb", type=float, default=40.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args):
    manifest_payload, fingerprint = _manifest(args.manifest, args.split)
    dataset = _build_split_dataset(args.manifest, args.native_cache, args.split)
    binding = build_preflight_binding(args.manifest, args.native_cache)
    expected_keys = [str(sample["lmdb_key"]) for sample in manifest_payload["samples"]]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        expected_keys = expected_keys[: args.limit]
    key_to_index = {key: index for index, key in enumerate(dataset.keys)}
    if not expected_keys or any(key not in key_to_index for key in expected_keys):
        raise RuntimeError("dataset keys do not match cache selection")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA cache build requested but unavailable")
    gerna = build_gerna_backbone().to(device).eval()
    interface = build_interface_backbone().to(device).eval()
    checkpoints = {
        "gerna": load_backbone_checkpoint(gerna, args.gerna_checkpoint),
        "interface": load_backbone_checkpoint(interface, args.interface_checkpoint),
    }
    metadata = {
        "dataset": "LP-PDBBind",
        "split": args.split,
        "manifest_sha256": fingerprint["sha256"],
        "native_metadata_sha256": binding["native_metadata_sha256"],
        "gerna_checkpoint_sha256": checkpoints["gerna"]["sha256"],
        "interface_checkpoint_sha256": checkpoints["interface"]["sha256"],
        "source_sample_count": fingerprint["sample_count"],
        "selected_sample_count": len(expected_keys),
        "smoke_limit": args.limit,
        "version": 3,
    }
    writer = HybridFeatureCacheWriter(
        args.output,
        expected_keys=expected_keys,
        metadata=metadata,
        resume=args.resume,
        map_size=int(args.map_size_gb * 1024 ** 3),
    )
    written = 0
    try:
        for key in list(writer.missing_keys):
            sample = dataset[key_to_index[key]]
            gerna_inputs, interface_item, target = prepare_hybrid_sample(sample, device)
            with torch.inference_mode():
                gerna.request_multiscale_affinity_features()
                try:
                    gerna_output = gerna(**gerna_inputs)
                    gerna_prediction = gerna_output[0]
                    gerna_features = gerna.take_multiscale_affinity_features()
                except BaseException:
                    gerna.discard_multiscale_affinity_features()
                    raise
                interface_prediction, interface_features = interface(
                    interface_item, return_multiscale_features=True
                )
            writer.write(build_cache_record(
                key, target, gerna_features, gerna_prediction,
                interface_features, interface_prediction,
            ))
            written += 1
            if written % 25 == 0:
                print(json.dumps({"written": written, "key": key}))
        completed = writer.finalize()
    finally:
        writer.close()
    print(json.dumps({"status": "complete", "output": str(args.output), "written": written}))
    return completed


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.map_size_gb <= 0:
        raise ValueError("map-size-gb must be positive")
    run(args)


if __name__ == "__main__":
    main()
