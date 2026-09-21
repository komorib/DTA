"""Train one Rank Fusion v4 seed using train and validation caches only."""

import argparse
import json

from .cache import load_fusion_cache
from .model import RankFusionV4
from .training import fit_one_seed


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train one Rank Fusion v4 seed")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--drop-path-rate", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    train_cache = load_fusion_cache(args.train_cache, expected_split="train")
    valid_cache = load_fusion_cache(args.valid_cache, expected_split="valid")
    latent_dims = (
        train_cache.upper_bound_latents.shape[1],
        train_cache.boltz_latents.shape[1],
        train_cache.three_dprot_latents.shape[1],
    )
    model = RankFusionV4(
        latent_dims,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
    )
    summary = fit_one_seed(
        model,
        train_cache,
        valid_cache,
        args.output,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        resume=args.resume,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
