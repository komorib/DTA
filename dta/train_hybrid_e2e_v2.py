import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from dta.hybrid_e2e_training_v2 import run_e2e_training


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Jointly fine-tune the LP-PDBBind upper-bound fusion."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--train-native-cache", required=True)
    parser.add_argument("--valid-native-cache", required=True)
    parser.add_argument("--gerna-checkpoint", required=True)
    parser.add_argument("--interface-checkpoint", required=True)
    parser.add_argument("--cached-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.epochs <= 0 or args.epochs > 10:
        raise ValueError("epochs must be in [1, 10]")
    if args.patience <= 0 or args.group_size < 2:
        raise ValueError("patience must be positive and group-size must be at least two")
    if args.limit is not None and args.limit < 2:
        raise ValueError("limit must be at least two")
    result = run_e2e_training(args)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
