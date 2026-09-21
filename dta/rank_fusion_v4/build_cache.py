"""Command-line entry point for Rank Fusion v4 keyed caches."""

import argparse
import json
from pathlib import Path

from .cache import EXPERT_ORDER, build_fusion_cache


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build a keyed Rank Fusion v4 cache")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=("train", "valid", "test"))
    parser.add_argument("--normalization-output")
    parser.add_argument("--normalization-input")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    artifact_dirs = {
        expert: artifact_root / args.split / expert for expert in EXPERT_ORDER
    }
    metadata = build_fusion_cache(
        args.manifest,
        artifact_dirs,
        args.output,
        split=args.split,
        normalization_output=args.normalization_output,
        normalization_input=args.normalization_input,
        limit=args.limit,
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
