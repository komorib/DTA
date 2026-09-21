"""Fine-tune the pretrained V18 MSA pair expert and evaluate it."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import MSAPairComplexDataset
from .model import MSAInterfacePairPoolExpertV18
from .training import TrainingConfig, fit_msa_interface_expert, predict_model
from utils.experiment_integrity import sha256_file


def _save_predictions(path, result):
    np.savez_compressed(
        path,
        keys=np.asarray(result["keys"]),
        labels=result["labels"],
        predictions=result["predictions"],
        latents=result["latents"],
    )


def _run_test_only(args):
    output = Path(args.output).expanduser().resolve()
    checkpoint_path = output / "best.pt"
    result_path = output / "result.json"
    valid_path = output / "valid_predictions.npz"
    test_path = output / "test_predictions.npz"
    if not checkpoint_path.is_file() or not result_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("V18 validation artifacts are incomplete")
    if test_path.exists():
        raise ValueError("V18 test predictions already exist")
    if not args.selection_result:
        raise ValueError("--selection-result is required for V18 test-only inference")
    selection_path = Path(args.selection_result).expanduser().resolve()
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("version") != "msa-interface-pair-pooling-fusion-v18":
        raise ValueError("V18 selection version is invalid")
    if selection.get("status") != "validation_passed":
        raise ValueError("V18 validation promotion gate did not pass")
    expected_valid_sha = selection.get("inputs", {}).get("v18_valid_sha256")
    if expected_valid_sha != sha256_file(valid_path):
        raise ValueError("V18 validation prediction SHA256 does not match selection")
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "validation_complete":
        raise ValueError("V18 training result is not awaiting test inference")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if int(checkpoint.get("schema_version", -1)) != 1:
        raise ValueError("V18 affinity checkpoint schema is invalid")
    config = TrainingConfig(**checkpoint["config"])
    config = replace(
        config,
        device=args.device,
        evaluation_batch_size=args.evaluation_batch_size,
        num_workers=args.num_workers,
    )
    model = MSAInterfacePairPoolExpertV18(**config.model_kwargs()).to(torch.device(config.device))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test = MSAPairComplexDataset(
        args.test_manifest, args.test_cache, args.row_cache, args.limit_test
    )
    test_result = predict_model(
        model, test, config, checkpoint["label_mean"], checkpoint["label_std"]
    )
    _save_predictions(test_path, test_result)
    result["status"] = "complete"
    result["test_metrics"] = dict(test_result["metrics"])
    result["counts"]["test"] = len(test)
    result["selection_result"] = str(selection_path)
    result["selection_result_sha256"] = sha256_file(selection_path)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run(args):
    if args.test_only:
        return _run_test_only(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V18 training output already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    train = MSAPairComplexDataset(
        args.train_manifest, args.train_cache, args.row_cache, args.limit_train
    )
    valid = MSAPairComplexDataset(
        args.valid_manifest, args.valid_cache, args.row_cache, args.limit_valid
    )
    test = None
    if not args.skip_test:
        test = MSAPairComplexDataset(
            args.test_manifest, args.test_cache, args.row_cache, args.limit_test
        )
    config = TrainingConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        affinity_initial_checkpoint=args.affinity_initial_checkpoint,
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
        interface_blocks=args.interface_blocks,
        pair_token_width=args.pair_token_width,
        pair_token_heads=args.pair_token_heads,
        pair_token_layers=args.pair_token_layers,
        pair_token_count=args.pair_token_count,
        row_depth=args.row_depth,
        inference_subsets=args.inference_subsets,
        rank_weight=args.rank_weight,
        mse_weight=args.mse_weight,
        contact_weight=args.contact_weight,
        residue_contact_weight=args.residue_contact_weight,
        affinity_class_weight=args.affinity_class_weight,
        distogram_weight=args.distogram_weight,
        interface_distogram_weight=args.interface_distogram_weight,
        interface_contact_weight=args.interface_contact_weight,
        match_weight=args.match_weight,
        interface_mask_probability=args.interface_mask_probability,
        encoder_learning_rate_scale=args.encoder_learning_rate_scale,
    )
    print(
        json.dumps(
            {
                "event": "msa_interface_v18_training_start",
                "config": asdict(config),
                "counts": {
                    "train": len(train),
                    "valid": len(valid),
                    "test": None if test is None else len(test),
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    model, checkpoint, history = fit_msa_interface_expert(train, valid, output, config)
    valid_result = predict_model(
        model, valid, config, checkpoint["label_mean"], checkpoint["label_std"]
    )
    _save_predictions(output / "valid_predictions.npz", valid_result)
    test_result = None
    if test is not None:
        test_result = predict_model(
            model, test, config, checkpoint["label_mean"], checkpoint["label_std"]
        )
        _save_predictions(output / "test_predictions.npz", test_result)
    result = {
        "status": "complete" if test_result is not None else "validation_complete",
        "version": "msa-interface-pair-pooling-v18",
        "best_epoch": checkpoint["epoch"],
        "selected_variant": checkpoint["variant"],
        "validation_metrics": dict(valid_result["metrics"]),
        "counts": {
            "train": len(train),
            "valid": len(valid),
            "test": None if test is None else len(test),
        },
        "history_length": len(history),
        "inference_subsets": config.inference_subsets,
        "pretrained_checkpoint": str(
            Path(config.pretrained_checkpoint).expanduser().resolve()
        ),
        "pretrained_checkpoint_sha256": sha256_file(config.pretrained_checkpoint),
        "affinity_initial_checkpoint": str(
            Path(config.affinity_initial_checkpoint).expanduser().resolve()
        ),
        "affinity_initial_checkpoint_sha256": sha256_file(
            config.affinity_initial_checkpoint
        ),
        "output": str(output),
    }
    if test_result is not None:
        result["test_metrics"] = dict(test_result["metrics"])
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
    parser.add_argument("--affinity-initial-checkpoint", required=True)
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
    parser.add_argument("--interface-blocks", type=int, default=2)
    parser.add_argument("--pair-token-width", type=int, default=256)
    parser.add_argument("--pair-token-heads", type=int, default=8)
    parser.add_argument("--pair-token-layers", type=int, default=2)
    parser.add_argument("--pair-token-count", type=int, default=96)
    parser.add_argument("--row-depth", type=int, default=64)
    parser.add_argument("--inference-subsets", type=int, default=2)
    parser.add_argument("--rank-weight", type=float, default=0.40)
    parser.add_argument("--mse-weight", type=float, default=0.15)
    parser.add_argument("--contact-weight", type=float, default=0.05)
    parser.add_argument("--residue-contact-weight", type=float, default=0.03)
    parser.add_argument("--affinity-class-weight", type=float, default=0.05)
    parser.add_argument("--distogram-weight", type=float, default=0.02)
    parser.add_argument("--interface-distogram-weight", type=float, default=0.08)
    parser.add_argument("--interface-contact-weight", type=float, default=0.05)
    parser.add_argument("--match-weight", type=float, default=0.02)
    parser.add_argument("--interface-mask-probability", type=float, default=0.15)
    parser.add_argument("--encoder-learning-rate-scale", type=float, default=0.20)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--selection-result")
    return parser


def main(argv=None):
    return 0 if run(build_arg_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
