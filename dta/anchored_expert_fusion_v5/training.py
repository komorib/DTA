"""Cache-backed residual training for anchored expert fusion v5."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import tempfile
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import DataLoader

from dta.anchored_expert_fusion_v5.data import (
    AnchoredFusionDataset,
    ImmutableCacheRef,
    collate_anchored_fusion,
    fallback_predictions,
)
from dta.anchored_expert_fusion_v5.losses import anchored_residual_loss
from dta.anchored_expert_fusion_v5.model import (
    AnchoredResidualModel,
    _validated_budget,
)
from dta.anchored_expert_fusion_v5.static_fusion import rank_percentiles
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import atomic_write_json, sha256_file


_IDENTITY_FILES = (
    "features.npz",
    "metadata.json",
    "coverage.json",
    "normalization.json",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(name + " must be a lowercase SHA256 hash")
    return value


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _identity(value, split):
    if not isinstance(value, Mapping):
        raise ValueError(split + " cache identity must be a dictionary")
    required = {
        "split",
        "count",
        "manifest_sha256",
        "ordered_key_sha256",
        "files",
    }
    if set(value) != required:
        raise ValueError(split + " cache identity fields do not match")
    if value["split"] != split:
        raise ValueError(split + " cache identity split does not match")
    count = value["count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, Integral)
        or int(count) < 0
    ):
        raise ValueError(split + " cache identity count is invalid")
    files = value["files"]
    if not isinstance(files, Mapping) or set(files) != set(_IDENTITY_FILES):
        raise ValueError(split + " cache identity files do not match")
    immutable_files = MappingProxyType(
        {
            name: _sha256(files[name], split + " " + name + " hash")
            for name in _IDENTITY_FILES
        }
    )
    return MappingProxyType(
        {
            "split": split,
            "count": int(count),
            "manifest_sha256": _sha256(
                value["manifest_sha256"], split + " manifest hash"
            ),
            "ordered_key_sha256": _sha256(
                value["ordered_key_sha256"], split + " ordered key hash"
            ),
            "files": immutable_files,
        }
    )


def _plain_identity(value):
    return {
        "split": value["split"],
        "count": value["count"],
        "manifest_sha256": value["manifest_sha256"],
        "ordered_key_sha256": value["ordered_key_sha256"],
        "files": {
            name: value["files"][name] for name in _IDENTITY_FILES
        },
    }


@dataclass(frozen=True)
class StaticSpec:
    family: str
    deploy_weights: tuple
    rank_slope: float
    rank_intercept: float
    cross_fit_report_hash: str
    train_cache_identity: dict
    valid_cache_identity: dict

    def __post_init__(self):
        if self.family not in ("raw", "rank"):
            raise ValueError("family must be raw or rank")
        try:
            weights = tuple(self.deploy_weights)
        except TypeError as error:
            raise ValueError("deploy_weights must contain exactly three values") from error
        if len(weights) != 3:
            raise ValueError("deploy_weights must contain exactly three values")
        normalized = tuple(
            _finite_real(value, "deploy_weights") for value in weights
        )
        if any(value < 0.0 for value in normalized):
            raise ValueError("deploy_weights must be nonnegative")
        if not math.isclose(
            sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("deploy_weights must sum to one")
        slope = _finite_real(self.rank_slope, "rank_slope")
        intercept = _finite_real(self.rank_intercept, "rank_intercept")
        if self.family == "raw" and slope != 1.0:
            raise ValueError("raw family rank_slope must be 1")
        if self.family == "raw" and intercept != 0.0:
            raise ValueError("raw family rank_intercept must be 0")
        object.__setattr__(self, "deploy_weights", normalized)
        object.__setattr__(self, "rank_slope", slope)
        object.__setattr__(self, "rank_intercept", intercept)
        object.__setattr__(
            self,
            "cross_fit_report_hash",
            _sha256(self.cross_fit_report_hash, "cross-fit report hash"),
        )
        object.__setattr__(
            self,
            "train_cache_identity",
            _identity(self.train_cache_identity, "train"),
        )
        object.__setattr__(
            self,
            "valid_cache_identity",
            _identity(self.valid_cache_identity, "valid"),
        )

    def to_dict(self):
        return {
            "family": self.family,
            "deploy_weights": list(self.deploy_weights),
            "rank_slope": self.rank_slope,
            "rank_intercept": self.rank_intercept,
            "cross_fit_report_hash": self.cross_fit_report_hash,
            "train_cache_identity": _plain_identity(
                self.train_cache_identity
            ),
            "valid_cache_identity": _plain_identity(
                self.valid_cache_identity
            ),
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("static spec must be a dictionary")
        required = {
            "family",
            "deploy_weights",
            "rank_slope",
            "rank_intercept",
            "cross_fit_report_hash",
            "train_cache_identity",
            "valid_cache_identity",
        }
        if set(payload) != required:
            raise ValueError("static spec fields do not match")
        return cls(**payload)

    def __hash__(self):
        return int(static_spec_sha256(self)[:16], 16)


def static_spec_sha256(spec):
    if not isinstance(spec, StaticSpec):
        raise ValueError("spec must be a StaticSpec")
    payload = json.dumps(
        spec.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def static_predictions(cache, spec):
    if not isinstance(spec, StaticSpec):
        raise ValueError("spec must be a StaticSpec")
    experts = fallback_predictions(cache.predictions, cache.available)
    weights = np.asarray(spec.deploy_weights, dtype=np.float64)
    if spec.family == "raw":
        result = experts.astype(np.float64) @ weights
    else:
        ranked = np.column_stack(
            [
                rank_percentiles(experts[:, index])
                for index in range(experts.shape[1])
            ]
        )
        result = (
            ranked @ weights
        ) * spec.rank_slope + spec.rank_intercept
    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 1 or result.shape[0] != len(cache.keys):
        raise ValueError("static predictions do not match cache rows")
    if not np.isfinite(result).all():
        raise ValueError("static predictions must be finite")
    return result


class NonfiniteTrainingError(RuntimeError):
    pass


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _seed_value(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def _device_value(value):
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device is invalid") from error
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    return device


def _unwrap_cache(value):
    if isinstance(value, ImmutableCacheRef):
        return value.cache, value.identity
    return value, None


def _validate_cache(cache, sealed_identity, split, recorded_identity=None):
    required = (
        "keys",
        "labels",
        "predictions",
        "standardized_predictions",
        "available",
        "upper_bound_latents",
        "boltz_latents",
        "three_dprot_latents",
        "metadata",
    )
    if any(not hasattr(cache, name) for name in required):
        raise ValueError(split + " cache is invalid")
    row_count = len(cache.keys)
    if row_count <= 0 or len(set(cache.keys)) != row_count:
        raise ValueError(split + " cache key count is invalid")
    runtime = {
        "split": cache.metadata.get("split"),
        "count": cache.metadata.get("count"),
        "manifest_sha256": cache.metadata.get("manifest_sha256"),
        "ordered_key_sha256": cache.metadata.get("ordered_key_sha256"),
    }
    expected_runtime = {
        name: sealed_identity[name]
        for name in (
            "split",
            "count",
            "manifest_sha256",
            "ordered_key_sha256",
        )
    }
    if runtime != expected_runtime or row_count != sealed_identity["count"]:
        raise ValueError(split + " cache identity does not match StaticSpec")
    if recorded_identity is not None:
        if _plain_identity(recorded_identity) != _plain_identity(sealed_identity):
            raise ValueError(split + " cache file identity does not match StaticSpec")

    numeric = (
        ("labels", cache.labels, (row_count,)),
        ("predictions", cache.predictions, (row_count, 3)),
        (
            "standardized_predictions",
            cache.standardized_predictions,
            (row_count, 3),
        ),
    )
    for name, value, shape in numeric:
        array = np.asarray(value)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(split + " cache " + name + " is invalid")
    available = np.asarray(cache.available)
    if available.shape != (row_count, 3) or available.dtype != np.bool_:
        raise ValueError(split + " cache available is invalid")
    if not available[:, 0].all():
        raise ValueError(split + " cache Upper-Bound is incomplete")
    widths = []
    for name in (
        "upper_bound_latents",
        "boltz_latents",
        "three_dprot_latents",
    ):
        value = np.asarray(getattr(cache, name))
        if (
            value.ndim != 2
            or value.shape[0] != row_count
            or value.shape[1] <= 0
            or not np.isfinite(value).all()
        ):
            raise ValueError(split + " cache " + name + " is invalid")
        widths.append(int(value.shape[1]))
    return tuple(widths)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _move_batch(batch, device):
    return {
        "keys": batch["keys"],
        "labels": batch["labels"].to(device),
        "base_predictions": batch["base_predictions"].to(device),
        "raw_predictions": batch["raw_predictions"].to(device),
        "standardized_predictions": batch["standardized_predictions"].to(
            device
        ),
        "available": batch["available"].to(device),
        "latents": [value.to(device) for value in batch["latents"]],
    }


def _model_config(latent_widths, budget):
    return {
        "class": "AnchoredResidualModel",
        "latent_widths": list(latent_widths),
        "budget": float(budget),
        "dropout": 0.10,
        "adapter_width": 64,
        "hidden_widths": [256, 128, 64],
        "dtype": "float32",
    }


def _run_config(
    spec,
    latent_widths,
    budget,
    seed,
    maximum_epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
    gradient_clip,
):
    return {
        "schema_version": 1,
        "static_spec_sha256": static_spec_sha256(spec),
        "static_spec": spec.to_dict(),
        "family": spec.family,
        "deploy_weights": list(spec.deploy_weights),
        "rank_calibration": {
            "slope": spec.rank_slope,
            "intercept": spec.rank_intercept,
        },
        "cache_lineage": {
            "train": _plain_identity(spec.train_cache_identity),
            "valid": _plain_identity(spec.valid_cache_identity),
        },
        "budget": float(budget),
        "seed": seed,
        "model": _model_config(latent_widths, budget),
        "optimizer": {
            "class": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "schedule": "fixed",
        },
        "batch_size": batch_size,
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "gradient_clip": gradient_clip,
    }


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_torch(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _validate_resume_lineage(actual, expected):
    if actual != expected:
        raise ValueError("resume lineage does not match")


def _load_completed_result(result_path, output_path, config):
    try:
        result = _load_json(result_path)
        checkpoint_path = output_path / "checkpoint_best.pt"
        prediction_path = output_path / "predictions_best.npz"
        if not isinstance(result, dict) or result.get("status") != "complete":
            raise ValueError("completed result is invalid")
        _validate_resume_lineage(result.get("run_config"), config)
        if result.get("checkpoint_path") != str(checkpoint_path):
            raise ValueError("completed checkpoint path does not match")
        if result.get("prediction_path") != str(prediction_path):
            raise ValueError("completed prediction path does not match")
        if not checkpoint_path.is_file() or not prediction_path.is_file():
            raise ValueError("completed artifacts are missing")
        if result.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError("completed checkpoint hash does not match")
        if result.get("prediction_sha256") != sha256_file(prediction_path):
            raise ValueError("completed prediction hash does not match")
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("resume lineage does not match") from error
    return result


def _cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _checkpoint_payload(model, config, epoch, metrics, spec, budget, seed):
    return {
        "model_config": config["model"],
        "model_state_dict": _cpu_state(model),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "budget": float(budget),
        "seed": int(seed),
        "static_spec_sha256": static_spec_sha256(spec),
        "cache_lineage": config["cache_lineage"],
        "run_config": config,
    }


def _prediction_payload(cache, prediction, epoch, variant):
    return {
        "keys": np.asarray(cache.keys, dtype=np.str_),
        "labels": np.asarray(cache.labels, dtype=np.float32),
        "predictions": np.asarray(prediction, dtype=np.float32),
        "epoch": np.asarray(epoch, dtype=np.int64),
        "variant": np.asarray(variant, dtype=np.str_),
    }


def _tensor_statistics(value):
    value = value.detach().float()
    finite = torch.isfinite(value)
    selected = value[finite]
    return {
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "finite_count": int(finite.sum().item()),
        "minimum": float(selected.min().item()) if selected.numel() else None,
        "maximum": float(selected.max().item()) if selected.numel() else None,
    }


def _batch_tensor_items(batch):
    if batch is None:
        return ()
    values = [
        ("labels", batch["labels"]),
        ("base_predictions", batch["base_predictions"]),
        ("raw_predictions", batch["raw_predictions"]),
        ("standardized_predictions", batch["standardized_predictions"]),
    ]
    values.extend(
        ("latent_" + str(index), value)
        for index, value in enumerate(batch["latents"])
    )
    return tuple(values)


def _write_nonfinite_failure(
    output_path,
    stage,
    epoch,
    batch_index,
    global_step,
    batch,
    learning_rate,
    model,
    extra_tensors=(),
):
    tensor_items = list(_batch_tensor_items(batch)) + list(extra_tensors)
    payload = {
        "status": "failed_nonfinite",
        "stage": stage,
        "epoch": int(epoch),
        "batch_index": (
            None if batch_index is None else int(batch_index)
        ),
        "global_step": int(global_step),
        "sample_keys": [] if batch is None else list(batch["keys"]),
        "learning_rate": float(learning_rate),
        "tensor_stats": {
            name: _tensor_statistics(value) for name, value in tensor_items
        },
        "gradient_stats": {
            name: _tensor_statistics(parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
        "parameter_stats": {
            name: _tensor_statistics(parameter)
            for name, parameter in model.named_parameters()
        },
    }
    atomic_write_json(output_path / "failure.json", payload)
    atomic_write_json(
        output_path / "status.json",
        {
            "status": "failed_nonfinite",
            "stage": stage,
            "epoch": int(epoch),
        },
    )
    raise NonfiniteTrainingError(
        "nonfinite " + stage + " at epoch " + str(epoch)
    )


def _all_finite(values):
    checks = [torch.isfinite(value).all() for value in values]
    return not checks or bool(torch.stack(checks).all().item())


@torch.inference_mode()
def evaluate_residual(model, valid_dataset, device, batch_size=256):
    batch_size = _positive_integer(batch_size, "batch_size")
    device = _device_value(device)
    if not isinstance(valid_dataset, AnchoredFusionDataset):
        raise ValueError("valid_dataset must be an AnchoredFusionDataset")
    loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anchored_fusion,
    )
    model = model.to(device=device, dtype=torch.float32).eval()
    keys = []
    predictions = []
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch)
        keys.extend(batch["keys"])
        predictions.append(output.detach().float())
    prediction_tensor = torch.cat(predictions)
    if not bool(torch.isfinite(prediction_tensor).all().item()):
        raise NonfiniteTrainingError("nonfinite validation output")
    prediction_array = prediction_tensor.cpu().numpy().astype(
        np.float32, copy=False
    )
    cache = valid_dataset.cache
    return {
        "keys": tuple(keys),
        "labels": np.asarray(cache.labels, dtype=np.float32).copy(),
        "predictions": prediction_array,
        "metrics": affinity_metrics(cache.labels, prediction_array),
    }


def is_better_validation(candidate, current):
    if current is None:
        return True
    for name, direction in (
        ("CI", 1.0),
        ("Spearman", 1.0),
        ("RMSE", -1.0),
    ):
        difference = direction * (
            float(candidate[name]) - float(current[name])
        )
        if difference > 0.0:
            return True
        if difference < 0.0:
            return False
    return False


def _rng_state():
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state):
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _resume_payload(
    model,
    optimizer,
    epoch,
    stale,
    best,
    history,
    config,
    global_step,
):
    return {
        "model_state_dict": _cpu_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "stale": int(stale),
        "best": best,
        "history": history,
        "run_config": config,
        "global_step": int(global_step),
        "rng_state": _rng_state(),
    }


def _fit_one_residual_impl(
    train_cache,
    valid_cache,
    static_spec,
    budget,
    seed,
    output_dir,
    device,
    maximum_epochs=60,
    patience=8,
    batch_size=256,
    learning_rate=1e-4,
    weight_decay=1e-3,
    gradient_clip=1.0,
    resume=False,
    stop_after_epoch=None,
):
    output_path = Path(output_dir).expanduser().resolve()
    try:
        if not isinstance(static_spec, StaticSpec):
            raise ValueError("static_spec must be a StaticSpec")
        maximum_epochs = _positive_integer(
            maximum_epochs, "maximum_epochs"
        )
        patience = _positive_integer(patience, "patience")
        batch_size = _positive_integer(batch_size, "batch_size")
        seed = _seed_value(seed)
        learning_rate = _finite_real(learning_rate, "learning_rate")
        weight_decay = _finite_real(weight_decay, "weight_decay")
        gradient_clip = _finite_real(gradient_clip, "gradient_clip")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if not isinstance(resume, bool):
            raise ValueError("resume must be boolean")
        if stop_after_epoch is not None and (
            isinstance(stop_after_epoch, bool)
            or not isinstance(stop_after_epoch, Integral)
            or int(stop_after_epoch) < 0
        ):
            raise ValueError("stop_after_epoch must be nonnegative")
        device = _device_value(device)
        train_cache, train_recorded = _unwrap_cache(train_cache)
        valid_cache, valid_recorded = _unwrap_cache(valid_cache)
        train_widths = _validate_cache(
            train_cache,
            static_spec.train_cache_identity,
            "train",
            train_recorded,
        )
        valid_widths = _validate_cache(
            valid_cache,
            static_spec.valid_cache_identity,
            "valid",
            valid_recorded,
        )
        if train_widths != valid_widths:
            raise ValueError("train and valid latent widths do not match")
        budget = _validated_budget(budget)
        config = _run_config(
            static_spec,
            train_widths,
            budget,
            seed,
            maximum_epochs,
            patience,
            batch_size,
            learning_rate,
            weight_decay,
            gradient_clip,
        )
    except (ValueError, RuntimeError) as error:
        if resume:
            raise ValueError("resume lineage does not match") from error
        raise

    run_config_path = output_path / "run_config.json"
    result_path = output_path / "result.json"
    resume_path = output_path / "resume_state.pt"
    if output_path.exists() and any(output_path.iterdir()):
        if not resume:
            raise FileExistsError("output is nonempty; pass resume=True")
        if not run_config_path.is_file():
            raise ValueError("resume lineage does not match")
        _validate_resume_lineage(_load_json(run_config_path), config)
        if result_path.is_file():
            return _load_completed_result(
                result_path, output_path, config
            )
        if not resume_path.is_file():
            status_path = output_path / "status.json"
            existing = {path.name for path in output_path.iterdir()}
            restartable_pre_snapshot_artifacts = (
                {"run_config.json", "status.json"},
                {
                    "checkpoint_best.pt",
                    "checkpoint_epoch_minus_one.pt",
                    "history.json",
                    "predictions_best.npz",
                    "predictions_epoch_minus_one.npz",
                    "run_config.json",
                    "status.json",
                },
            )
            if (
                existing not in restartable_pre_snapshot_artifacts
                or not status_path.is_file()
                or _load_json(status_path) != {"status": "running"}
            ):
                raise ValueError("resume lineage does not match")
            resume = False
    elif resume:
        raise ValueError("resume lineage does not match")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_config_path, config)
        atomic_write_json(
            output_path / "status.json", {"status": "running"}
        )

    _seed_everything(seed)
    train_base = static_predictions(train_cache, static_spec)
    valid_base = static_predictions(valid_cache, static_spec)
    train_dataset = AnchoredFusionDataset(train_cache, train_base)
    valid_dataset = AnchoredFusionDataset(valid_cache, valid_base)
    model = AnchoredResidualModel(
        train_widths, budget=budget, dropout=0.10
    ).to(device=device, dtype=torch.float32)
    parameters = tuple(model.parameters())

    try:
        baseline = evaluate_residual(
            model, valid_dataset, device, batch_size
        )
    except NonfiniteTrainingError as error:
        _write_nonfinite_failure(
            output_path, "validation", -1, None, 0, None,
            learning_rate, model
        )
        raise error
    if not np.array_equal(baseline["predictions"], valid_base):
        raise RuntimeError(
            "epoch-minus-one predictions do not exactly equal static base"
        )
    baseline_row = {
        "epoch": -1,
        "loss": None,
        "gradient_norm": None,
        "metrics": baseline["metrics"],
    }
    baseline_checkpoint = _checkpoint_payload(
        model,
        config,
        -1,
        baseline["metrics"],
        static_spec,
        budget,
        seed,
    )

    if not resume:
        _atomic_npz(
            output_path / "predictions_epoch_minus_one.npz",
            **_prediction_payload(
                valid_cache,
                baseline["predictions"],
                -1,
                "epoch_minus_one",
            )
        )
        _atomic_torch_save(
            output_path / "checkpoint_epoch_minus_one.pt",
            baseline_checkpoint,
        )
        _atomic_torch_save(
            output_path / "checkpoint_best.pt", baseline_checkpoint
        )
        _atomic_npz(
            output_path / "predictions_best.npz",
            **_prediction_payload(
                valid_cache, baseline["predictions"], -1, "best"
            )
        )
        history = [baseline_row]
        best = {"epoch": -1, "metrics": baseline["metrics"]}
        stale = 0
        start_epoch = 0
        global_step = 0
        atomic_write_json(output_path / "history.json", history)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if resume:
        state = _load_torch(resume_path)
        _validate_resume_lineage(state.get("run_config"), config)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _optimizer_to(optimizer, device)
        start_epoch = int(state["next_epoch"])
        stale = int(state["stale"])
        best = state["best"]
        history = list(state["history"])
        global_step = int(state["global_step"])
        _restore_rng_state(state["rng_state"])

    for epoch in range(start_epoch, maximum_epochs):
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate_anchored_fusion,
        )
        model.train()
        loss_total = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        gradient_total = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        batch_count = 0
        last_batch = None
        last_batch_index = None
        for batch_index, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            last_batch = batch
            last_batch_index = batch_index
            optimizer.zero_grad(set_to_none=True)
            try:
                final = model(batch)
                loss, _components = anchored_residual_loss(
                    final,
                    batch["base_predictions"],
                    batch["labels"],
                )
            except ValueError as error:
                if "finite" not in str(error):
                    raise
                _write_nonfinite_failure(
                    output_path,
                    "loss",
                    epoch,
                    batch_index,
                    global_step,
                    batch,
                    learning_rate,
                    model,
                )
            loss.backward()
            try:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    gradient_clip,
                    error_if_nonfinite=True,
                )
            except RuntimeError:
                _write_nonfinite_failure(
                    output_path,
                    "gradient",
                    epoch,
                    batch_index,
                    global_step,
                    batch,
                    learning_rate,
                    model,
                    (("loss", loss),),
                )
            optimizer.step()
            loss_total.add_(loss.detach().float())
            gradient_total.add_(gradient_norm.detach().float())
            batch_count += 1
            global_step += 1

        if not _all_finite(parameters):
            _write_nonfinite_failure(
                output_path,
                "parameter",
                epoch,
                last_batch_index,
                global_step,
                last_batch,
                learning_rate,
                model,
                (
                    ("loss_total", loss_total),
                    ("gradient_total", gradient_total),
                ),
            )
        epoch_loss, epoch_gradient_norm = (
            torch.stack((loss_total, gradient_total))
            .div(batch_count)
            .detach()
            .cpu()
            .tolist()
        )
        try:
            validation = evaluate_residual(
                model, valid_dataset, device, batch_size
            )
        except NonfiniteTrainingError:
            _write_nonfinite_failure(
                output_path,
                "validation",
                epoch,
                None,
                global_step,
                None,
                learning_rate,
                model,
            )
        row = {
            "epoch": epoch,
            "loss": epoch_loss,
            "gradient_norm": epoch_gradient_norm,
            "metrics": validation["metrics"],
        }
        history.append(row)
        if is_better_validation(
            validation["metrics"], best["metrics"]
        ):
            stale = 0
            best = {
                "epoch": epoch,
                "metrics": validation["metrics"],
            }
            _atomic_torch_save(
                output_path / "checkpoint_best.pt",
                _checkpoint_payload(
                    model,
                    config,
                    epoch,
                    validation["metrics"],
                    static_spec,
                    budget,
                    seed,
                ),
            )
            _atomic_npz(
                output_path / "predictions_best.npz",
                **_prediction_payload(
                    valid_cache,
                    validation["predictions"],
                    epoch,
                    "best",
                )
            )
        else:
            stale += 1

        atomic_write_json(output_path / "history.json", history)
        _atomic_torch_save(
            resume_path,
            _resume_payload(
                model,
                optimizer,
                epoch,
                stale,
                best,
                history,
                config,
                global_step,
            ),
        )
        atomic_write_json(
            output_path / "status.json",
            {
                "status": "running",
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "stale": stale,
            },
        )
        if stop_after_epoch is not None and epoch >= int(stop_after_epoch):
            interrupted = {
                "status": "interrupted",
                "next_epoch": epoch + 1,
                "history": history,
            }
            atomic_write_json(
                output_path / "status.json", interrupted
            )
            return interrupted
        if stale >= patience:
            break

    checkpoint_path = output_path / "checkpoint_best.pt"
    prediction_path = output_path / "predictions_best.npz"
    stop_reason = "patience" if stale >= patience else "maximum_epochs"
    result = {
        "status": "complete",
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "history": history,
        "stop_reason": stop_reason,
        "run_config": config,
    }
    atomic_write_json(result_path, result)
    atomic_write_json(
        output_path / "status.json",
        {
            "status": "complete",
            "best_epoch": int(best["epoch"]),
            "stop_reason": stop_reason,
        },
    )
    return result


def _fit_one_residual(
    train_cache,
    valid_cache,
    static_spec,
    budget,
    seed,
    output_dir,
    device,
    maximum_epochs=60,
    patience=8,
    batch_size=256,
    learning_rate=1e-4,
    weight_decay=1e-3,
    gradient_clip=1.0,
    resume=False,
    stop_after_epoch=None,
):
    deterministic_algorithms = (
        torch.are_deterministic_algorithms_enabled()
    )
    try:
        return _fit_one_residual_impl(
            train_cache=train_cache,
            valid_cache=valid_cache,
            static_spec=static_spec,
            budget=budget,
            seed=seed,
            output_dir=output_dir,
            device=device,
            maximum_epochs=maximum_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            resume=resume,
            stop_after_epoch=stop_after_epoch,
        )
    finally:
        torch.use_deterministic_algorithms(deterministic_algorithms)


def fit_one_residual(
    train_cache,
    valid_cache,
    static_spec,
    budget,
    seed,
    output_dir,
    device,
    maximum_epochs=60,
    patience=8,
    batch_size=256,
    learning_rate=1e-4,
    weight_decay=1e-3,
    gradient_clip=1.0,
    resume=False,
):
    return _fit_one_residual(
        train_cache=train_cache,
        valid_cache=valid_cache,
        static_spec=static_spec,
        budget=budget,
        seed=seed,
        output_dir=output_dir,
        device=device,
        maximum_epochs=maximum_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        resume=resume,
    )


__all__ = [
    "StaticSpec",
    "NonfiniteTrainingError",
    "static_spec_sha256",
    "static_predictions",
    "evaluate_residual",
    "is_better_validation",
    "fit_one_residual",
]
