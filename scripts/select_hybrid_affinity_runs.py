#!/usr/bin/env python
"""Validation-only checkpoint selection for hybrid affinity runs."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np
import torch

from dta.hybrid_affinity_training import (
    atomic_torch_save,
    atomic_write_predictions,
    evaluate_hybrid_affinity,
    validation_candidate_is_better,
)
from utils.experiment_integrity import (
    atomic_write_json,
    hash_ordered_keys,
    sha256_file,
)


REQUIRED_METRICS = ("RMSE", "MAE", "Pearson", "Spearman", "CI")


class IncompleteFullStageError(RuntimeError):
    pass


def _load_json(path, name):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"{name} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _reject_test_fields(value, location="artifact"):
    if isinstance(value, dict):
        for key, nested in value.items():
            if "test" in str(key).lower():
                raise ValueError(
                    f"{location} contains forbidden test field {key}"
                )
            _reject_test_fields(nested, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_test_fields(nested, f"{location}[{index}]")


def load_prediction_archive(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"keys", "labels", "predictions"}:
                raise ValueError(
                    f"prediction archive fields differ: {path}"
                )
            keys = np.asarray(archive["keys"])
            labels = np.asarray(archive["labels"], dtype=np.float64)
            predictions = np.asarray(
                archive["predictions"], dtype=np.float64
            )
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(path) in str(error):
            raise
        raise ValueError(
            f"prediction archive is unreadable: {path}"
        ) from error

    if keys.ndim != 1 or labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            f"prediction archive arrays must be one-dimensional: {path}"
        )
    if not (len(keys) == len(labels) == len(predictions)) or not len(keys):
        raise ValueError(
            f"prediction archive arrays are empty or misaligned: {path}"
        )
    key_list = [str(key) for key in keys.tolist()]
    if any(not key for key in key_list):
        raise ValueError(f"prediction archive contains empty keys: {path}")
    if len(key_list) != len(set(key_list)):
        raise ValueError(
            f"prediction archive contains duplicate keys: {path}"
        )
    if not np.isfinite(labels).all() or not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            f"prediction archive contains nonfinite values: {path}"
        )
    return {
        "path": str(path),
        "keys": key_list,
        "labels": labels,
        "predictions": predictions,
    }


def align_prediction_archives(archives, label_tolerance=1e-6):
    archives = list(archives)
    if not archives:
        raise ValueError("at least one prediction archive is required")
    reference = archives[0]
    keys = list(reference["keys"])
    labels = np.asarray(reference["labels"], dtype=np.float64)
    columns = []
    for archive in archives:
        actual_keys = list(archive["keys"])
        if actual_keys != keys:
            raise ValueError(
                "prediction key order differs: "
                + str(archive.get("path", "<memory>"))
            )
        actual_labels = np.asarray(
            archive["labels"], dtype=np.float64
        )
        if actual_labels.shape != labels.shape or not np.allclose(
            actual_labels,
            labels,
            rtol=0.0,
            atol=float(label_tolerance),
        ):
            raise ValueError(
                "prediction labels differ: "
                + str(archive.get("path", "<memory>"))
            )
        predictions = np.asarray(
            archive["predictions"], dtype=np.float64
        )
        if predictions.shape != labels.shape or not np.isfinite(
            predictions
        ).all():
            raise ValueError(
                "prediction values are invalid: "
                + str(archive.get("path", "<memory>"))
            )
        columns.append(predictions)
    return {
        "keys": keys,
        "labels": labels.copy(),
        "matrix": np.stack(columns, axis=1),
        "prediction_key_sha256": hash_ordered_keys(keys),
    }


def candidate_is_better(candidate, incumbent, ci_tolerance=1e-6):
    return validation_candidate_is_better(
        candidate, incumbent, ci_tolerance=ci_tolerance
    )


def average_state_dicts(state_dicts):
    states = list(state_dicts)
    if not states:
        raise ValueError("at least one state dict is required")
    if not all(isinstance(state, dict) for state in states):
        raise ValueError("state dicts must be mappings")
    names = list(states[0])
    expected_names = set(names)
    for state in states[1:]:
        if set(state) != expected_names:
            raise ValueError("state dict keys differ")

    averaged = {}
    for name in names:
        values = [state[name] for state in states]
        reference = values[0]
        if not torch.is_tensor(reference):
            raise ValueError(f"state value is not a tensor: {name}")
        for value in values[1:]:
            if not torch.is_tensor(value):
                raise ValueError(f"state value is not a tensor: {name}")
            if value.shape != reference.shape:
                raise ValueError(f"state shape differs: {name}")
            if value.dtype != reference.dtype:
                raise ValueError(f"state dtype differs: {name}")
        if reference.is_floating_point() or reference.is_complex():
            if not all(
                bool(torch.isfinite(value).all()) for value in values
            ):
                raise ValueError(f"state contains nonfinite values: {name}")
            stacked = torch.stack(
                [value.detach().cpu() for value in values], dim=0
            )
            averaged[name] = stacked.mean(dim=0)
        else:
            if not all(
                torch.equal(reference, value) for value in values[1:]
            ):
                raise ValueError(
                    f"nonfloating state values differ: {name}"
                )
            averaged[name] = reference.detach().cpu().clone()
    return averaged


def _validated_metrics(value, context):
    if not isinstance(value, dict) or not all(
        name in value for name in REQUIRED_METRICS
    ):
        raise ValueError(f"{context} metrics are incomplete")
    result = {}
    for name in REQUIRED_METRICS:
        metric = float(value[name])
        if not math.isfinite(metric):
            raise ValueError(f"{context} metric is nonfinite: {name}")
        result[name] = metric
    return result


def _validated_candidate(value, context, require_state=True):
    if not isinstance(value, dict):
        raise ValueError(f"{context} candidate must be a mapping")
    try:
        epoch = int(value["epoch"])
        stage = str(value["stage"])
        variant = str(value["variant"])
        keys = [str(key) for key in value["keys"]]
        labels = np.asarray(value["labels"], dtype=np.float64)
        predictions = np.asarray(
            value["predictions"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} candidate is malformed") from error
    if epoch < 0 or not stage or not variant:
        raise ValueError(f"{context} candidate metadata is invalid")
    if (
        not keys
        or len(keys) != len(set(keys))
        or labels.shape != (len(keys),)
        or predictions.shape != (len(keys),)
    ):
        raise ValueError(f"{context} candidate arrays are misaligned")
    if not np.isfinite(labels).all() or not np.isfinite(
        predictions
    ).all():
        raise ValueError(f"{context} candidate contains nonfinite values")
    state = value.get("model_state_dict")
    if require_state and not isinstance(state, dict):
        raise ValueError(f"{context} candidate has no model state")
    return {
        "epoch": epoch,
        "stage": stage,
        "variant": variant,
        "metrics": _validated_metrics(value.get("metrics"), context),
        "keys": keys,
        "labels": labels,
        "predictions": predictions,
        "model_state_dict": state,
    }


def _states_equal(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if set(left) != set(right):
        return False
    return all(
        torch.is_tensor(left[name])
        and torch.is_tensor(right[name])
        and left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and torch.equal(left[name], right[name])
        for name in left
    )


def _candidate_metadata(candidate):
    return {
        "epoch": int(candidate["epoch"]),
        "stage": str(candidate["stage"]),
        "variant": str(candidate["variant"]),
        "metrics": {
            name: float(candidate["metrics"][name])
            for name in REQUIRED_METRICS
        },
    }


def _default_average_evaluator(
    run_dir,
    run_config,
    provenance,
    state_dict,
    device="cuda:0",
):
    from torch.utils.data import DataLoader

    from data_utils.hybrid_affinity_dataset import collate_hybrid_group
    from dta.train_hybrid_affinity import (
        build_gerna_backbone,
        build_hybrid_dataset,
        build_interface_backbone,
    )
    from net.hybrid_affinity import (
        HybridAffinityFusion,
        HybridAffinityModel,
    )

    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required to evaluate averaged checkpoints"
            )
        torch.cuda.set_device(device)
    gerna = build_gerna_backbone()
    interface = build_interface_backbone()
    fusion = HybridAffinityFusion(
        fusion_type=str(run_config["fusion_type"]),
        token_dim=128,
        dropout=0.1,
    )
    model = HybridAffinityModel(gerna, interface, fusion)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    splits = provenance.get("splits", {})
    valid = splits.get("valid", {})
    manifest = valid.get("manifest_path")
    native_cache = valid.get("native_cache_path")
    if not manifest or not native_cache:
        raise ValueError(
            f"run provenance has no validation split: {run_dir}"
        )
    dataset = build_hybrid_dataset(
        manifest, native_cache, expected_split="valid"
    )
    try:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_hybrid_group,
            pin_memory=False,
            persistent_workers=False,
        )
        return evaluate_hybrid_affinity(
            model,
            loader,
            device,
            amp_enabled=device.type == "cuda",
        )
    finally:
        for name in ("gerna_dataset", "interface_dataset"):
            owner = getattr(dataset, name, None)
            close = getattr(owner, "close", None)
            if callable(close):
                close()


def _require_full_stage(checkpoint, run_dir):
    training_config = checkpoint.get("training_config")
    stage_epochs = (
        training_config.get("stage_epochs")
        if isinstance(training_config, dict)
        else None
    )
    training_state = checkpoint.get("training_state")
    if (
        not isinstance(stage_epochs, dict)
        or int(stage_epochs.get("full", 0)) <= 0
        or not isinstance(training_state, dict)
        or training_state.get("complete") is not True
    ):
        raise IncompleteFullStageError(
            f"run has not completed the full stage: {run_dir}"
        )


def _expected_provenance_hashes(provenance):
    try:
        initial = provenance["initial_checkpoints"]
        splits = provenance["splits"]
        result = {
            "gerna_checkpoint_sha256": initial["gerna"]["sha256"],
            "interface_checkpoint_sha256": initial["interface"][
                "sha256"
            ],
        }
        for split in ("train", "valid"):
            binding = splits[split]
            for field in (
                "manifest_sha256",
                "native_metadata_sha256",
                "ordered_key_sha256",
            ):
                result[f"{split}_{field}"] = binding[field]
            result[f"{split}_sample_count"] = int(
                binding["sample_count"]
            )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "run provenance does not contain frozen hashes"
        ) from error
    return result


def _validate_checkpoint_bindings(
    checkpoint, run_config, provenance, run_dir
):
    training_config = checkpoint.get("training_config")
    saved_run_config = (
        training_config.get("run_config")
        if isinstance(training_config, dict)
        else None
    )
    if saved_run_config != run_config:
        raise RuntimeError(
            f"training checkpoint training config differs: {run_dir}"
        )
    expected_hashes = _expected_provenance_hashes(provenance)
    if checkpoint.get("provenance_hashes") != expected_hashes:
        raise RuntimeError(
            f"training checkpoint provenance hashes differ: {run_dir}"
        )


def select_run(
    run_dir,
    *,
    evaluate_state_dict=None,
    device="cuda:0",
    require_full_stage=False,
):
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(str(run_dir))
    run_config = _load_json(run_dir / "run_config.json", "run config")
    provenance = _load_json(run_dir / "provenance.json", "provenance")
    _reject_test_fields(run_config, "run config")
    _reject_test_fields(provenance, "provenance")

    prediction_path = run_dir / "valid_predictions.npz"
    saved_prediction = load_prediction_archive(prediction_path)
    training_checkpoint_path = run_dir / "training_checkpoint.pth"
    if not training_checkpoint_path.is_file():
        raise FileNotFoundError(str(training_checkpoint_path))
    checkpoint = torch.load(
        os.fspath(training_checkpoint_path), map_location="cpu"
    )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != 1
    ):
        raise RuntimeError(
            f"unsupported training checkpoint: {training_checkpoint_path}"
        )
    _validate_checkpoint_bindings(
        checkpoint, run_config, provenance, run_dir
    )
    if require_full_stage:
        _require_full_stage(checkpoint, run_dir)
    top_three_value = checkpoint.get("top_three")
    if not isinstance(top_three_value, list) or len(top_three_value) != 3:
        raise RuntimeError(
            f"training checkpoint must contain validation top three: {run_dir}"
        )
    top_three = [
        _validated_candidate(value, f"raw rank {index + 1}")
        for index, value in enumerate(top_three_value)
    ]
    training_state = checkpoint.get("training_state")
    best_ema_value = (
        training_state.get("best_ema")
        if isinstance(training_state, dict)
        else None
    )
    best_ema = _validated_candidate(best_ema_value, "EMA best")
    raw_best = top_three[0]

    base_selected = raw_best
    if candidate_is_better(best_ema, base_selected):
        base_selected = best_ema
    if saved_prediction["keys"] != base_selected["keys"]:
        raise ValueError(
            f"saved validation prediction keys differ: {prediction_path}"
        )
    if not np.allclose(
        saved_prediction["labels"],
        base_selected["labels"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"saved validation prediction labels differ: {prediction_path}"
        )
    if not np.array_equal(
        saved_prediction["predictions"],
        base_selected["predictions"],
    ):
        raise ValueError(
            f"saved validation predictions differ: {prediction_path}"
        )

    raw_path = run_dir / "validation_rank_1.pth"
    if not raw_path.is_file():
        raise FileNotFoundError(str(raw_path))
    raw_payload = torch.load(os.fspath(raw_path), map_location="cpu")
    if not isinstance(raw_payload, dict) or not _states_equal(
        raw_payload.get("model_state_dict"),
        raw_best["model_state_dict"],
    ):
        raise RuntimeError(
            f"raw validation checkpoint differs: {raw_path}"
        )

    ema_path = run_dir / "Model_hybrid_valid_ema.pth"
    atomic_torch_save(
        {
            "format_version": 1,
            "selected": _candidate_metadata(best_ema),
            "model_state_dict": best_ema["model_state_dict"],
        },
        ema_path,
    )

    average_state = average_state_dicts(
        [candidate["model_state_dict"] for candidate in top_three]
    )
    if evaluate_state_dict is None:
        evaluation = _default_average_evaluator(
            run_dir,
            run_config,
            provenance,
            average_state,
            device=device,
        )
    else:
        evaluation = evaluate_state_dict(
            run_dir, run_config, provenance, average_state
        )
    average = _validated_candidate(
        {
            "epoch": max(candidate["epoch"] for candidate in top_three),
            "stage": "averaged",
            "variant": "top3_average",
            "metrics": evaluation["metrics"],
            "keys": evaluation["keys"],
            "labels": evaluation["labels"],
            "predictions": evaluation["predictions"],
            "model_state_dict": average_state,
        },
        "top3 average",
    )
    if average["keys"] != raw_best["keys"] or not np.allclose(
        average["labels"],
        raw_best["labels"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"averaged validation evaluation is misaligned: {run_dir}"
        )
    average_path = run_dir / "Model_hybrid_valid_top3_average.pth"
    atomic_torch_save(
        {
            "format_version": 1,
            "selected": _candidate_metadata(average),
            "source_epochs": [
                int(candidate["epoch"]) for candidate in top_three
            ],
            "model_state_dict": average_state,
        },
        average_path,
    )

    candidates = (
        (raw_best, raw_path),
        (best_ema, ema_path),
        (average, average_path),
    )
    selected, selected_model_path = candidates[0]
    for challenger, challenger_path in candidates[1:]:
        if candidate_is_better(challenger, selected):
            selected = challenger
            selected_model_path = challenger_path

    selected_prediction_path = (
        run_dir / "selected_valid_predictions.npz"
    )
    atomic_write_predictions(
        selected_prediction_path,
        {
            "keys": selected["keys"],
            "labels": selected["labels"],
            "predictions": selected["predictions"],
        },
    )
    selected_path = run_dir / "selected_candidate.json"
    result = {
        "schema_version": 1,
        "dataset": "LP-PDBBind",
        "selection_split": "valid",
        "run_dir": str(run_dir),
        "fusion_type": str(run_config["fusion_type"]),
        "rank_weight": float(run_config["rank_weight"]),
        "seed": int(run_config["seed"]),
        "run_config": run_config,
        "provenance_hashes": checkpoint["provenance_hashes"],
        "selected_variant": str(selected["variant"]),
        "epoch": int(selected["epoch"]),
        "stage": str(selected["stage"]),
        "validation_metrics": {
            name: float(selected["metrics"][name])
            for name in REQUIRED_METRICS
        },
        "model_path": str(selected_model_path.resolve()),
        "model_sha256": sha256_file(selected_model_path),
        "prediction_path": str(selected_prediction_path.resolve()),
        "prediction_sha256": sha256_file(selected_prediction_path),
        "prediction_key_sha256": hash_ordered_keys(selected["keys"]),
        "training_checkpoint": str(
            training_checkpoint_path.resolve()
        ),
        "training_checkpoint_sha256": sha256_file(
            training_checkpoint_path
        ),
    }
    _reject_test_fields(result, "selected candidate")
    atomic_write_json(selected_path, result)
    result["selected_candidate_path"] = str(selected_path.resolve())
    result["selected_candidate_sha256"] = sha256_file(selected_path)
    return result


def _selection_score(candidate):
    return {
        "epoch": int(candidate["epoch"]),
        "metrics": dict(candidate["validation_metrics"]),
    }


def select_runs(
    run_root,
    *,
    top_k,
    output_path,
    evaluate_state_dict=None,
    device="cuda:0",
    require_full_stage=False,
):
    run_root = Path(run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(str(run_root))
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(str(output_path))
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    run_dirs = sorted(
        path.parent
        for path in run_root.glob("*/run_config.json")
        if path.is_file()
    )
    if len(run_dirs) < top_k:
        raise ValueError("fewer completed runs than requested top_k")
    ranked = []
    for run_dir in run_dirs:
        try:
            selected = select_run(
                run_dir,
                evaluate_state_dict=evaluate_state_dict,
                device=device,
                require_full_stage=require_full_stage,
            )
        except IncompleteFullStageError:
            if require_full_stage:
                continue
            raise
        inserted = False
        for index, existing in enumerate(ranked):
            if candidate_is_better(
                _selection_score(selected),
                _selection_score(existing),
            ):
                ranked.insert(index, selected)
                inserted = True
                break
        if not inserted:
            ranked.append(selected)
    if len(ranked) < top_k:
        raise ValueError(
            "fewer full-stage runs than requested top_k"
        )

    entries = []
    for selected in ranked[:top_k]:
        entries.append(
            {
                "fusion_type": selected["fusion_type"],
                "rank_weight": selected["rank_weight"],
                "seed": selected["seed"],
                "run_dir": selected["run_dir"],
                "training_checkpoint": selected[
                    "training_checkpoint"
                ],
                "training_checkpoint_sha256": selected[
                    "training_checkpoint_sha256"
                ],
                "selected_candidate": selected[
                    "selected_candidate_path"
                ],
                "selected_candidate_sha256": selected[
                    "selected_candidate_sha256"
                ],
                "selected_model": selected["model_path"],
                "selected_model_sha256": selected["model_sha256"],
                "validation_metrics": selected[
                    "validation_metrics"
                ],
                "selected_variant": selected["selected_variant"],
                "epoch": selected["epoch"],
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": "LP-PDBBind",
        "selection_split": "valid",
        "require_full_stage": bool(require_full_stage),
        "run_root": str(run_root),
        "entries": entries,
    }
    _reject_test_fields(payload, "selection")
    atomic_write_json(output_path, payload)
    return payload


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Select hybrid affinity checkpoints on validation data."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-full-stage", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = select_runs(
        args.run_root,
        top_k=args.top_k,
        output_path=args.output,
        device=args.device,
        require_full_stage=args.require_full_stage,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
