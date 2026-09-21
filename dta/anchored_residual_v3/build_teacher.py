"""CLI for generating v3 cached-teacher prediction artifacts."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data_utils.hybrid_feature_cache_v2 import (
    CachedHybridFeatureDataset,
    collate_cached_hybrid_features,
)
from dta.anchored_residual_v3.teacher import generate_teacher_predictions
from net.hybrid_upper_bound import HybridUpperBoundFusion
from utils.experiment_integrity import hash_ordered_keys, manifest_fingerprint


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate anchored v3 teacher predictions.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=("train", "valid", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser


def run(args):
    fingerprint = manifest_fingerprint(args.manifest)
    if fingerprint["split"] != args.split:
        raise ValueError("manifest split does not match --split")
    manifest = json.loads(Path(args.manifest).read_text())
    manifest_keys = [str(sample["lmdb_key"]) for sample in manifest["samples"]]
    if args.limit is not None:
        manifest_keys = manifest_keys[: args.limit]
    dataset = CachedHybridFeatureDataset(args.cache)
    if dataset.metadata.get("manifest_sha256") != fingerprint["sha256"]:
        raise ValueError("cache manifest SHA256 does not match")
    if dataset.metadata.get("ordered_key_sha256") != hash_ordered_keys(dataset.keys):
        raise ValueError("cache ordered keys are invalid")
    if dataset.keys != manifest_keys:
        raise ValueError("cache keys do not match manifest order")
    if args.limit is not None:
        if args.limit <= 0 or args.limit != len(dataset):
            raise ValueError("limit is outside the cache size")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_cached_hybrid_features,
    )
    model = HybridUpperBoundFusion(dropout=0.0, drop_path_rate=0.0)
    return generate_teacher_predictions(
        model, loader, args.checkpoint, args.output,
        split=args.split, manifest_sha256=fingerprint["sha256"],
        device=torch.device(args.device),
    )


def main(argv=None):
    result = run(build_arg_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
