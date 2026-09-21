"""Extract frozen V13 interface features for validation, then gated test inference."""

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
from dta.msa_rank_fusion_v10.run_pipeline import _atomic_json, _atomic_npz
from utils.experiment_integrity import sha256_file

from .features import extract_spatial_interface_features, load_interface_encoders


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
                raise ValueError("V15 inference subset order changed")
            pretrain_runs.append(pretrain_values)
            affinity_runs.append(affinity_values)
    pretrain_features = np.mean(np.stack(pretrain_runs), axis=0).astype(np.float32)
    affinity_features = np.mean(np.stack(affinity_runs), axis=0).astype(np.float32)
    if pretrain_features.shape != affinity_features.shape:
        raise ValueError("V15 pretrain and affinity feature shapes do not match")
    if not np.isfinite(pretrain_features).all() or not np.isfinite(
        affinity_features
    ).all():
        raise RuntimeError("V15 extracted features contain non-finite values")
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
        if lineage["pretrain"]["sha256"] != expected["pretrain"]["sha256"]:
            raise ValueError("V15 pretrain checkpoint changed after validation")
        if lineage["affinity"]["sha256"] != expected["affinity"]["sha256"]:
            raise ValueError("V15 affinity checkpoint changed after validation")
    return pretrain, affinity, lineage


def _run_test_only(args):
    output = Path(args.output).expanduser().resolve()
    result_path = output / "result.json"
    valid_path = output / "valid_features.npz"
    test_path = output / "test_features.npz"
    if not result_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("V15 validation feature artifacts are incomplete")
    if test_path.exists():
        raise ValueError("V15 test features already exist")
    if not args.selection_result:
        raise ValueError("--selection-result is required for V15 test-only extraction")
    selection_path = Path(args.selection_result).expanduser().resolve()
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "validation_passed":
        raise ValueError("V15 validation promotion gate did not pass")
    expected_valid_sha = selection.get("inputs", {}).get("v15_valid_sha256")
    if expected_valid_sha != sha256_file(valid_path):
        raise ValueError("V15 validation feature SHA256 does not match selection")
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "validation_complete":
        raise ValueError("V15 feature result is not awaiting test extraction")

    pretrain, affinity, lineage = _load_and_validate_sources(
        args, expected=result["source_lineage"]
    )
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
    if output.exists() and any(output.iterdir()):
        raise ValueError("V15 feature output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    pretrain, affinity, lineage = _load_and_validate_sources(args)
    valid = MSAPairComplexDataset(
        args.valid_manifest, args.valid_cache, args.row_cache, args.limit_valid
    )
    values = extract_dataset_features(pretrain, affinity, valid, args)
    valid_path = output / "valid_features.npz"
    _save_features(valid_path, values)
    result = {
        "status": "validation_complete",
        "version": "msa-interface-spatial-feature-extraction-v15",
        "counts": {"valid": len(valid), "test": None},
        "feature_dimension": int(values["pretrain_features"].shape[1]),
        "inference_subsets": int(args.inference_subsets),
        "row_depth": int(args.row_depth),
        "source_lineage": lineage,
        "valid_manifest": str(Path(args.valid_manifest).expanduser().resolve()),
        "valid_features": str(valid_path),
        "valid_feature_sha256": sha256_file(valid_path),
        "output": str(output),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--row-cache", required=True)
    parser.add_argument("--pretrain-checkpoint", required=True)
    parser.add_argument("--affinity-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
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
