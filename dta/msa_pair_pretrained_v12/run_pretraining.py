"""Run V12 masked-MSA and distogram pretraining."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .dataset import MSAPairComplexDataset
from .pretraining import PretrainingConfig, fit_msa_pair_pretrainer


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V12 pretraining output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    train = MSAPairComplexDataset(
        args.train_manifest, args.train_cache, args.row_cache, args.limit_train
    )
    valid = MSAPairComplexDataset(
        args.valid_manifest, args.valid_cache, args.row_cache, args.limit_valid
    )
    config = PretrainingConfig(
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        num_workers=args.num_workers,
        amp_dtype=args.amp_dtype,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        msa_width=args.msa_width,
        msa_layers=args.msa_layers,
        msa_heads=args.msa_heads,
        pair_factor=args.pair_factor,
        pair_width=args.pair_width,
        dropout=args.dropout,
        row_depth=args.row_depth,
        mask_probability=args.mask_probability,
        distogram_weight=args.distogram_weight,
        contact_weight=args.contact_weight,
    )
    print(
        json.dumps(
            {
                "event": "msa_v12_pretraining_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid)},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    checkpoint, history = fit_msa_pair_pretrainer(train, valid, output, config)
    result = {
        "status": "complete",
        "best_epoch": checkpoint["epoch"],
        "selected_variant": checkpoint["variant"],
        "validation_metrics": checkpoint["metrics"],
        "history_length": len(history),
        "output": str(output),
        "checkpoint": str(output / "best_pretrain.pt"),
    }
    with (output / "pretrain_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "valid"):
        parser.add_argument("--%s-manifest" % split, required=True)
        parser.add_argument("--%s-cache" % split, required=True)
        parser.add_argument("--limit-%s" % split, type=int)
    parser.add_argument("--row-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--msa-width", type=int, default=192)
    parser.add_argument("--msa-layers", type=int, default=4)
    parser.add_argument("--msa-heads", type=int, default=6)
    parser.add_argument("--pair-factor", type=int, default=8)
    parser.add_argument("--pair-width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--row-depth", type=int, default=64)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--distogram-weight", type=float, default=1.0)
    parser.add_argument("--contact-weight", type=float, default=0.20)
    return parser


def main(argv=None):
    return 0 if run(build_arg_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
