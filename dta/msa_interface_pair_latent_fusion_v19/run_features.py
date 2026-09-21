"""Seal V18 pair latents for validation, then run gated test inference."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from dta.msa_interface_pair_pooling_v18.dataset import MSAPairComplexDataset
from dta.msa_interface_pair_pooling_v18.model import MSAInterfacePairPoolExpertV18
from dta.msa_interface_pair_pooling_v18.training import TrainingConfig, predict_model
from dta.msa_rank_fusion_v10.run_pipeline import _atomic_json, _atomic_npz
from utils.experiment_integrity import sha256_file

from . import VERSION as FUSION_VERSION


FEATURE_VERSION = "msa-interface-pair-latent-extraction-v19"
SOURCE_VERSION = "msa-interface-pair-pooling-v18"


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_npz(path, required):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("missing V19 source archive: %s" % path)
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("V19 source archive is missing: %s" % ", ".join(missing))
        values = {name: np.asarray(archive[name]) for name in required}
    return path, values


def _checkpoint_metadata(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("V18 checkpoint does not exist: %s" % checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or int(checkpoint.get("schema_version", -1)) != 1:
        raise ValueError("V18 checkpoint schema is invalid")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("V18 checkpoint has no training config")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("V18 checkpoint has no model state")
    config = TrainingConfig(**checkpoint["config"])
    return checkpoint_path, checkpoint, config


def _pair_latents(latents, latent_width):
    latents = np.asarray(latents, dtype=np.float32)
    expected_width = 2 * int(latent_width)
    if latents.ndim != 2 or latents.shape[1] != expected_width:
        raise ValueError(
            "V18 latent width does not match checkpoint: expected %d, got %s"
            % (expected_width, tuple(latents.shape))
        )
    pair = np.asarray(latents[:, int(latent_width) :], dtype=np.float32)
    if not np.isfinite(pair).all():
        raise RuntimeError("V19 pair latents contain non-finite values")
    return pair


def _save_features(path, keys, labels, predictions, pair_latents):
    _atomic_npz(
        path,
        keys=np.asarray(keys),
        labels=np.asarray(labels, dtype=np.float64),
        v18_predictions=np.asarray(predictions, dtype=np.float64),
        pair_latents=np.asarray(pair_latents, dtype=np.float32),
    )


def _load_model(checkpoint, config, device):
    model = MSAInterfacePairPoolExpertV18(**config.model_kwargs()).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval()


def _run_test_only(args):
    output = Path(args.output).expanduser().resolve()
    result_path = output / "result.json"
    valid_path = output / "valid_features.npz"
    test_path = output / "test_features.npz"
    if not result_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("V19 validation feature artifacts are incomplete")
    if test_path.exists():
        raise ValueError("V19 test features already exist")
    if not args.selection_result:
        raise ValueError("--selection-result is required for V19 test-only inference")
    selection_path = Path(args.selection_result).expanduser().resolve()
    selection = _load_json(selection_path)
    if selection.get("version") != FUSION_VERSION:
        raise ValueError("V19 selection version is invalid")
    if selection.get("status") != "validation_passed":
        raise ValueError("V19 validation promotion gate did not pass")
    if selection.get("inputs", {}).get("v19_valid_sha256") != sha256_file(valid_path):
        raise ValueError("V19 validation feature SHA256 does not match selection")

    result = _load_json(result_path)
    if result.get("version") != FEATURE_VERSION:
        raise ValueError("V19 feature result version is invalid")
    if result.get("status") != "validation_complete":
        raise ValueError("V19 feature result is not awaiting test inference")
    checkpoint_path, checkpoint, config = _checkpoint_metadata(result["source_checkpoint"])
    if sha256_file(checkpoint_path) != result["source_checkpoint_sha256"]:
        raise ValueError("V18 checkpoint changed after V19 validation selection")
    source_valid_path = Path(result["source_valid_predictions"])
    if sha256_file(source_valid_path) != result["source_valid_predictions_sha256"]:
        raise ValueError("V18 validation predictions changed after V19 selection")

    device = torch.device(args.device)
    config = replace(
        config,
        device=str(device),
        evaluation_batch_size=int(args.evaluation_batch_size),
        num_workers=int(args.num_workers),
    )
    model = _load_model(checkpoint, config, device)
    test = MSAPairComplexDataset(
        args.test_manifest, args.test_cache, args.row_cache, args.limit_test
    )
    values = predict_model(
        model, test, config, checkpoint["label_mean"], checkpoint["label_std"]
    )
    pair = _pair_latents(values["latents"], config.latent_width)
    _save_features(
        test_path, values["keys"], values["labels"], values["predictions"], pair
    )
    result["status"] = "complete"
    result["counts"]["test"] = len(test)
    result["test_features"] = str(test_path)
    result["test_feature_sha256"] = sha256_file(test_path)
    result["selection_result"] = str(selection_path)
    result["selection_result_sha256"] = sha256_file(selection_path)
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run(args):
    if args.test_only:
        return _run_test_only(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("V19 feature output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    v18_root = Path(args.v18_root).expanduser().resolve()
    source_result_path = v18_root / "result.json"
    source_result = _load_json(source_result_path)
    if source_result.get("version") != SOURCE_VERSION:
        raise ValueError("V19 source is not a V18 expert")
    if source_result.get("status") != "validation_complete":
        raise ValueError("V18 source is not a validation-only artifact")
    checkpoint_path, checkpoint, config = _checkpoint_metadata(v18_root / "best.pt")
    valid_source_path, values = _load_npz(
        v18_root / "valid_predictions.npz",
        ("keys", "labels", "predictions", "latents"),
    )
    if values["labels"].ndim != 1 or values["labels"].size != values["keys"].size:
        raise ValueError("V18 validation rows are malformed")
    pair = _pair_latents(values["latents"], config.latent_width)
    valid_path = output / "valid_features.npz"
    _save_features(
        valid_path, values["keys"], values["labels"], values["predictions"], pair
    )
    result = {
        "status": "validation_complete",
        "version": FEATURE_VERSION,
        "counts": {"valid": int(values["labels"].size), "test": None},
        "pair_latent_dimension": int(pair.shape[1]),
        "source_v18_result": str(source_result_path.resolve()),
        "source_v18_result_sha256": sha256_file(source_result_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "source_checkpoint_variant": str(checkpoint.get("variant", "unknown")),
        "source_valid_predictions": str(valid_source_path),
        "source_valid_predictions_sha256": sha256_file(valid_source_path),
        "valid_features": str(valid_path),
        "valid_feature_sha256": sha256_file(valid_path),
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-root", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--row-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--selection-result")
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
