from __future__ import annotations

import argparse

import torch

from .train import fit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GraphDTA-GAT-GCN on the isolated KIBA binary benchmark.")
    parser.add_argument("--data-root", default="data/ExternalBenchmarkKIBA_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--limit-test", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    fit(args)


if __name__ == "__main__":
    main()
