import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from dta.hybrid_cached_training_v2 import train_cached_fusion


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train the cached LP-PDBBind upper-bound fusion."
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.batch_size <= 1:
        raise ValueError("batch-size must be greater than one")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    result = train_cached_fusion(args)
    print(result)


if __name__ == "__main__":
    main()
