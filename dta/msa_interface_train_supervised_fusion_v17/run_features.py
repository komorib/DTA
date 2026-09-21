"""Export V17 train/validation features, then gated test features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dta.msa_interface_pretrained_v13.dataset import (
    MSAPairCollator,
    MSAPairComplexDataset,
    move_batch_to_device,
)
from dta.msa_rank_fusion_v10.run_pipeline import (
    _atomic_json,
    _atomic_npz,
    _load_npz,
)
from utils.experiment_integrity import sha256_file

from . import FEATURE_VERSION, VERSION
from .features import extract_spatial_interface_features, load_interface_encoders


FEATURE_FIELDS = (
    "keys",
    "labels",
    "pretrain_features",
    "affinity_features",
)


def _autocast_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError("amp_dtype must be bfloat16 or float16")


def _loader(dataset, batch_size, row_depth, subset_index, num_workers):
    workers = int(num_workers)
    arguments = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "collate_fn": MSAPairCollator(
            row_depth=int(row_depth), training=False, subset_index=int(subset_index)
        ),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        arguments["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(**arguments)


def extract_dataset_features(pretrain_encoder, affinity_encoder, dataset, args):
    device = torch.device(args.device)
    dtype = _autocast_dtype(args.amp_dtype)
    pretrain_encoder = pretrain_encoder.to(device).eval()
    affinity_encoder = affinity_encoder.to(device).eval()
    reference_keys = None
    reference_labels = None
    pretrain_runs = []
    affinity_runs = []
    with torch.no_grad():
        for subset_index in range(int(args.inference_subsets)):
            keys = []
            labels = []
            pretrain_values = []
            affinity_values = []
            loader = _loader(
                dataset,
                args.evaluation_batch_size,
                args.row_depth,
                subset_index,
                args.num_workers,
            )
            for batch in loader:
                keys.extend(batch["keys"])
                labels.append(batch["labels"].numpy().astype(np.float64))
                moved = move_batch_to_device(batch, device)
                with torch.cuda.amp.autocast(
                    enabled=device.type == "cuda", dtype=dtype
                ):
                    pretrain_features = extract_spatial_interface_features(
                        pretrain_encoder, moved
                    )
                    affinity_features = extract_spatial_interface_features(
                        affinity_encoder, moved
                    )
                pretrain_values.append(
                    pretrain_features.float().cpu().numpy().astype(np.float32)
                )
                affinity_values.append(
                    affinity_features.float().cpu().numpy().astype(np.float32)
                )
            labels = np.concatenate(labels)
            pretrain_values = np.concatenate(pretrain_values)
            affinity_values = np.concatenate(affinity_values)
            if reference_keys is None:
                reference_keys = tuple(keys)
                reference_labels = labels
            elif tuple(keys) != reference_keys or not np.allclose(
                labels, reference_labels, atol=1e-6
            ):
                raise ValueError("V17 inference subset order changed")
            pretrain_runs.append(pretrain_values)
            affinity_runs.append(affinity_values)
    pretrain_features = np.mean(np.stack(pretrain_runs), axis=0).astype(np.float32)
    affinity_features = np.mean(np.stack(affinity_runs), axis=0).astype(np.float32)
    if pretrain_features.shape != affinity_features.shape:
        raise ValueError("V17 pretrain and affinity feature shapes do not match")
    if not np.isfinite(pretrain_features).all() or not np.isfinite(
        affinity_features
    ).all():
        raise RuntimeError("V17 extracted features contain non-finite values")
    return {
        "keys": reference_keys,
        "labels": reference_labels,
        "pretrain_features": pretrain_features,
        "affinity_features": affinity_features,
    }


def _save_features(path, values):
    _atomic_npz(
        path,
        keys=np.asarray(values["keys"]),
        labels=np.asarray(values["labels"], dtype=np.float64),
        pretrain_features=np.asarray(values["pretrain_features"], dtype=np.float32),
        affinity_features=np.asarray(values["affinity_features"], dtype=np.float32),
    )


def _load_and_validate_sources(args, expected=None):
    pretrain, affinity, lineage = load_interface_encoders(
        args.pretrain_checkpoint, args.affinity_checkpoint
    )
    if expected is not None:
        for name in ("pretrain", "affinity"):
            if lineage[name]["sha256"] != expected[name]["sha256"]:
                raise ValueError(
                    "V17 %s checkpoint changed after validation" % name
                )
    return pretrain, affinity, lineage


def _load_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("V17 %s is malformed" % label) from error


def _run_test_only(args):
    output = Path(args.output).expanduser().resolve()
    result_path = output / "result.json"
    train_path = output / "train_features.npz"
    valid_path = output / "valid_features.npz"
    test_path = output / "test_features.npz"
    if not result_path.is_file() or not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("V17 train/validation feature artifacts are incomplete")
    if test_path.exists():
        raise ValueError("V17 test features already exist")
    if not args.selection_result:
        raise ValueError("--selection-result is required for V17 test-only extraction")
    selection_path = Path(args.selection_result).expanduser().resolve()
    selection = _load_json(selection_path, "selection result")
    if selection.get("version") != VERSION or selection.get("status") != "validation_passed":
        raise ValueError("V17 validation promotion gate did not pass")
    expected_inputs = {
        "v17_train_features_sha256": sha256_file(train_path),
        "v17_valid_features_sha256": sha256_file(valid_path),
    }
    for name, expected in expected_inputs.items():
        if selection.get("inputs", {}).get(name) != expected:
            raise ValueError("V17 %s does not match selection" % name)
    result = _load_json(result_path, "feature result")
    if result.get("version") != FEATURE_VERSION or result.get("status") != "validation_ready":
        raise ValueError("V17 feature result is not awaiting test extraction")
    if result.get("train_feature_sha256") != expected_inputs["v17_train_features_sha256"]:
        raise ValueError("V17 train feature SHA256 changed")
    if result.get("valid_feature_sha256") != expected_inputs["v17_valid_features_sha256"]:
        raise ValueError("V17 validation feature SHA256 changed")

    pretrain, affinity, lineage = _load_and_validate_sources(
        args, expected=result["source_lineage"]
    )
    # The test dataset is intentionally constructed only after every seal above passes.
    test = MSAPairComplexDataset(
        args.test_manifest, args.test_cache, args.row_cache, args.limit_test
    )
    values = extract_dataset_features(pretrain, affinity, test, args)
    _save_features(test_path, values)
    result["status"] = "complete"
    result["counts"]["test"] = len(test)
    result["test_feature_sha256"] = sha256_file(test_path)
    result["selection_result"] = str(selection_path)
    result["selection_result_sha256"] = sha256_file(selection_path)
    result["source_lineage"] = lineage
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run(args):
    if int(args.inference_subsets) <= 0:
        raise ValueError("inference_subsets must be positive")
    if args.test_only:
        return _run_test_only(args)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    train_path = output / "train_features.npz"
    valid_path = output / "valid_features.npz"
    has_existing = any(output.iterdir())

    if has_existing:
        if not result_path.is_file() or not train_path.is_file() or valid_path.exists():
            raise ValueError("V17 feature output is not a resumable train-only artifact")
        result = _load_json(result_path, "partial feature result")
        if result.get("version") != FEATURE_VERSION or result.get("status") != "train_ready":
            raise ValueError("V17 partial feature result is not resumable")
        if result.get("train_feature_sha256") != sha256_file(train_path):
            raise ValueError("V17 partial train feature SHA256 changed")
        if result.get("inference_subsets") != int(args.inference_subsets):
            raise ValueError("V17 inference subset count changed during resume")
        if result.get("row_depth") != int(args.row_depth):
            raise ValueError("V17 row depth changed during resume")
        pretrain, affinity, lineage = _load_and_validate_sources(
            args, expected=result["source_lineage"]
        )
        _, train_values = _load_npz(train_path, FEATURE_FIELDS)
        train_count = int(np.asarray(train_values["labels"]).size)
    else:
        pretrain, affinity, lineage = _load_and_validate_sources(args)
        train = MSAPairComplexDataset(
            args.train_manifest, args.train_cache, args.row_cache, args.limit_train
        )
        train_values = extract_dataset_features(pretrain, affinity, train, args)
        _save_features(train_path, train_values)
        train_count = len(train)
        result = {
            "status": "train_ready",
            "version": FEATURE_VERSION,
            "counts": {"train": train_count, "valid": None, "test": None},
            "feature_dimension": int(train_values["pretrain_features"].shape[1]),
            "inference_subsets": int(args.inference_subsets),
            "row_depth": int(args.row_depth),
            "source_lineage": lineage,
            "train_manifest": str(Path(args.train_manifest).expanduser().resolve()),
            "valid_manifest": str(Path(args.valid_manifest).expanduser().resolve()),
            "train_features": str(train_path),
            "valid_features": str(valid_path),
            "train_feature_sha256": sha256_file(train_path),
            "valid_feature_sha256": None,
            "output": str(output),
        }
        _atomic_json(result_path, result)

    valid = MSAPairComplexDataset(
        args.valid_manifest, args.valid_cache, args.row_cache, args.limit_valid
    )
    valid_values = extract_dataset_features(pretrain, affinity, valid, args)
    _save_features(valid_path, valid_values)
    if train_values["pretrain_features"].shape[1] != valid_values["pretrain_features"].shape[1]:
        raise ValueError("V17 train/validation feature widths differ")
    result["status"] = "validation_ready"
    result["counts"] = {"train": train_count, "valid": len(valid), "test": None}
    result["valid_feature_sha256"] = sha256_file(valid_path)
    result["source_lineage"] = lineage
    _atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--row-cache", required=True)
    parser.add_argument("--pretrain-checkpoint", required=True)
    parser.add_argument("--affinity-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--row-depth", type=int, default=64)
    parser.add_argument("--inference-subsets", type=int, default=2)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--selection-result")
    return parser


def main(argv=None):
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
