"""Validation-only Rank Fusion v4 selection CLI."""

import argparse
import json

from .cache import load_fusion_cache
from .selection import select_validation_package


def _run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be SEED=PATH")
    seed, path = value.split("=", 1)
    if not seed or not path:
        raise argparse.ArgumentTypeError("run must be SEED=PATH")
    return seed, path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Select and seal Rank Fusion v4 on validation"
    )
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--run", action="append", type=_run, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, default=2340)
    parser.add_argument("--expected-seed", action="append")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError("run seeds must be unique")
    cache = load_fusion_cache(args.valid_cache, expected_split="valid")
    selection = select_validation_package(
        cache,
        runs,
        args.output,
        expected_count=args.expected_count,
        expected_seeds=(
            tuple(args.expected_seed)
            if args.expected_seed
            else ("1", "7", "21", "42", "87")
        ),
    )
    print(json.dumps(selection, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
