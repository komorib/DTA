"""Fine-tune the pretrained V12 MSA pair expert and evaluate it."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .dataset import MSAPairComplexDataset
from .training import TrainingConfig, fit_msa_pair_expert, predict_model
from utils.experiment_integrity import sha256_file


def _save_predictions(path, result):
    np.savez_compressed(
        path,
        keys=np.asarray(result["keys"]),
        labels=result["labels"],
        predictions=result["predictions"],
        latents=result["latents"],
    )


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V12 training output already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    train = MSAPairComplexDataset(
        args.train_manifest, args.train_cache, args.row_cache, args.limit_train
    )
    valid = MSAPairComplexDataset(
        args.valid_manifest, args.valid_cache, args.row_cache, args.limit_valid
    )
    test = MSAPairComplexDataset(
        args.test_manifest, args.test_cache, args.row_cache, args.limit_test
    )
    config = TrainingConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
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
        model_width=args.model_width,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        latent_width=args.latent_width,
        msa_width=args.msa_width,
        msa_layers=args.msa_layers,
        msa_heads=args.msa_heads,
        pair_factor=args.pair_factor,
        pair_width=args.pair_width,
        row_depth=args.row_depth,
        inference_subsets=args.inference_subsets,
        rank_weight=args.rank_weight,
        mse_weight=args.mse_weight,
        contact_weight=args.contact_weight,
        residue_contact_weight=args.residue_contact_weight,
        affinity_class_weight=args.affinity_class_weight,
        distogram_weight=args.distogram_weight,
        encoder_learning_rate_scale=args.encoder_learning_rate_scale,
    )
    print(
        json.dumps(
            {
                "event": "msa_v12_training_start",
                "config": asdict(config),
                "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    model, checkpoint, history = fit_msa_pair_expert(train, valid, output, config)
    valid_result = predict_model(
        model, valid, config, checkpoint["label_mean"], checkpoint["label_std"]
    )
    test_result = predict_model(
        model, test, config, checkpoint["label_mean"], checkpoint["label_std"]
    )
    _save_predictions(output / "valid_predictions.npz", valid_result)
    _save_predictions(output / "test_predictions.npz", test_result)
    result = {
        "status": "complete",
        "best_epoch": checkpoint["epoch"],
        "selected_variant": checkpoint["variant"],
        "validation_metrics": dict(valid_result["metrics"]),
        "test_metrics": dict(test_result["metrics"]),
        "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "history_length": len(history),
        "inference_subsets": config.inference_subsets,
        "pretrained_checkpoint": str(
            Path(config.pretrained_checkpoint).expanduser().resolve()
        ),
        "pretrained_checkpoint_sha256": sha256_file(config.pretrained_checkpoint),
        "output": str(output),
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "valid", "test"):
        parser.add_argument("--%s-manifest" % split, required=True)
        parser.add_argument("--%s-cache" % split, required=True)
        parser.add_argument("--limit-%s" % split, type=int)
    parser.add_argument("--row-cache", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--latent-width", type=int, default=1024)
    parser.add_argument("--msa-width", type=int, default=128)
    parser.add_argument("--msa-layers", type=int, default=4)
    parser.add_argument("--msa-heads", type=int, default=4)
    parser.add_argument("--pair-factor", type=int, default=8)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--row-depth", type=int, default=64)
    parser.add_argument("--inference-subsets", type=int, default=2)
    parser.add_argument("--rank-weight", type=float, default=0.40)
    parser.add_argument("--mse-weight", type=float, default=0.15)
    parser.add_argument("--contact-weight", type=float, default=0.05)
    parser.add_argument("--residue-contact-weight", type=float, default=0.03)
    parser.add_argument("--affinity-class-weight", type=float, default=0.05)
    parser.add_argument("--distogram-weight", type=float, default=0.05)
    parser.add_argument("--encoder-learning-rate-scale", type=float, default=0.30)
    return parser


def main(argv=None):
    return 0 if run(build_arg_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
