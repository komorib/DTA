"""Atomic keyed fusion caches with train-only normalization."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from utils.experiment_integrity import atomic_write_json, sha256_file

from .artifact import load_expert_artifact
from .manifest import load_manifest


SCHEMA_VERSION = 1
NORMALIZATION_SCHEMA_VERSION = 1
EXPERT_ORDER = ("upper_bound", "boltz_pairwise", "three_dprot")
LATENT_FIELDS = {
    "upper_bound": "upper_bound_latents",
    "boltz_pairwise": "boltz_latents",
    "three_dprot": "three_dprot_latents",
}
FEATURE_NAMES = frozenset(
    {
        "keys",
        "labels",
        "predictions",
        "standardized_predictions",
        "available",
        "upper_bound_latents",
        "boltz_latents",
        "three_dprot_latents",
    }
)


@dataclass(frozen=True)
class FusionCache:
    keys: tuple
    labels: np.ndarray
    predictions: np.ndarray
    standardized_predictions: np.ndarray
    available: np.ndarray
    upper_bound_latents: np.ndarray
    boltz_latents: np.ndarray
    three_dprot_latents: np.ndarray
    metadata: dict


def _atomic_write_npz(path, arrays):
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".features-", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _ordered_key_sha256(keys):
    payload = json.dumps(
        list(keys), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_artifacts(artifact_dirs, manifest, split):
    if not isinstance(artifact_dirs, dict):
        raise ValueError("artifact_dirs must map expert names to directories")
    if set(artifact_dirs) != set(EXPERT_ORDER):
        raise ValueError("artifact_dirs must contain exactly " + ", ".join(EXPERT_ORDER))
    artifacts = {}
    manifest_labels = dict(zip(manifest.keys, manifest.labels))
    for expert in EXPERT_ORDER:
        artifact = load_expert_artifact(
            artifact_dirs[expert],
            expected_expert=expert,
            expected_split=split,
            expected_manifest_sha256=manifest.sha256,
        )
        unknown = sorted(set(artifact.keys).difference(manifest_labels))
        if unknown:
            raise ValueError(f"{expert} artifact contains keys outside manifest")
        for key, label in zip(artifact.keys, artifact.labels):
            if not np.isclose(
                float(label), float(manifest_labels[key]), rtol=0.0, atol=1e-5
            ):
                raise ValueError(f"{expert} artifact label disagrees for {key}")
        artifacts[expert] = artifact
    return artifacts


def _join_artifacts(manifest, artifacts, count):
    keys = manifest.keys[:count]
    labels = manifest.labels[:count].copy()
    predictions = np.zeros((count, len(EXPERT_ORDER)), dtype=np.float32)
    available = np.zeros((count, len(EXPERT_ORDER)), dtype=np.bool_)
    latent_arrays = {}
    missing_by_expert = {}

    for expert_index, expert in enumerate(EXPERT_ORDER):
        artifact = artifacts[expert]
        width = int(artifact.latents.shape[1])
        joined_latents = np.zeros((count, width), dtype=np.float32)
        rows = {key: index for index, key in enumerate(artifact.keys)}
        missing = []
        for target_index, key in enumerate(keys):
            source_index = rows.get(key)
            if source_index is None or not bool(artifact.available[source_index]):
                missing.append(key)
                continue
            available[target_index, expert_index] = True
            predictions[target_index, expert_index] = artifact.predictions[source_index]
            joined_latents[target_index] = artifact.latents[source_index]
        latent_arrays[expert] = joined_latents
        missing_by_expert[expert] = missing

    if not available[:, 0].all():
        raise ValueError("Upper-Bound must be available for every target row")
    return keys, labels, predictions, available, latent_arrays, missing_by_expert


def _statistics(values, mask):
    selected = values[mask]
    if selected.shape[0] == 0:
        raise ValueError("cannot fit normalization without available training rows")
    mean = selected.astype(np.float64).mean(axis=0)
    std = selected.astype(np.float64).std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _fit_normalization(predictions, available, latent_arrays):
    experts = {}
    for expert_index, expert in enumerate(EXPERT_ORDER):
        mask = available[:, expert_index]
        prediction_mean, prediction_std = _statistics(
            predictions[:, expert_index : expert_index + 1], mask
        )
        latent_mean, latent_std = _statistics(latent_arrays[expert], mask)
        experts[expert] = {
            "prediction_mean": prediction_mean.tolist(),
            "prediction_std": prediction_std.tolist(),
            "latent_mean": latent_mean.tolist(),
            "latent_std": latent_std.tolist(),
            "available_count": int(mask.sum()),
        }
    return {
        "status": "complete",
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "expert_order": list(EXPERT_ORDER),
        "experts": experts,
    }


def _validate_normalization(payload, latent_arrays):
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("normalization is incomplete")
    if payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("normalization schema version does not match")
    if payload.get("expert_order") != list(EXPERT_ORDER):
        raise ValueError("normalization expert order does not match")
    experts = payload.get("experts")
    if not isinstance(experts, dict) or set(experts) != set(EXPERT_ORDER):
        raise ValueError("normalization experts do not match")
    normalized = {}
    for expert in EXPERT_ORDER:
        item = experts[expert]
        if not isinstance(item, dict):
            raise ValueError(f"normalization {expert} entry is invalid")
        width = latent_arrays[expert].shape[1]
        arrays = {
            "prediction_mean": np.asarray(item.get("prediction_mean"), dtype=np.float32),
            "prediction_std": np.asarray(item.get("prediction_std"), dtype=np.float32),
            "latent_mean": np.asarray(item.get("latent_mean"), dtype=np.float32),
            "latent_std": np.asarray(item.get("latent_std"), dtype=np.float32),
        }
        if arrays["prediction_mean"].shape != (1,) or arrays["prediction_std"].shape != (1,):
            raise ValueError(f"normalization {expert} prediction shape is invalid")
        if arrays["latent_mean"].shape != (width,) or arrays["latent_std"].shape != (width,):
            raise ValueError(f"normalization {expert} latent shape is invalid")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"normalization {expert} contains nonfinite values")
        if np.any(arrays["prediction_std"] < 1e-6) or np.any(arrays["latent_std"] < 1e-6):
            raise ValueError(f"normalization {expert} std is too small")
        normalized[expert] = arrays
    return normalized


def _apply_normalization(predictions, available, latent_arrays, statistics):
    standardized_predictions = np.zeros_like(predictions, dtype=np.float32)
    normalized_latents = {}
    for expert_index, expert in enumerate(EXPERT_ORDER):
        mask = available[:, expert_index]
        item = statistics[expert]
        standardized_predictions[mask, expert_index] = (
            predictions[mask, expert_index] - item["prediction_mean"][0]
        ) / item["prediction_std"][0]
        values = np.zeros_like(latent_arrays[expert], dtype=np.float32)
        values[mask] = (
            latent_arrays[expert][mask] - item["latent_mean"]
        ) / item["latent_std"]
        normalized_latents[expert] = values
    return standardized_predictions, normalized_latents


def _normalization_for_split(
    split,
    normalization_output,
    normalization_input,
    predictions,
    available,
    latent_arrays,
):
    if split == "train":
        if normalization_output is None or normalization_input is not None:
            raise ValueError(
                "train cache requires normalization_output and forbids normalization_input"
            )
        payload = _fit_normalization(predictions, available, latent_arrays)
        normalization_path = Path(normalization_output).expanduser().resolve()
        atomic_write_json(normalization_path, payload)
    else:
        if normalization_input is None or normalization_output is not None:
            raise ValueError(
                "non-train cache requires normalization_input and forbids normalization_output"
            )
        normalization_path = Path(normalization_input).expanduser().resolve()
        if not normalization_path.is_file():
            raise FileNotFoundError(normalization_path)
        with normalization_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    statistics = _validate_normalization(payload, latent_arrays)
    return payload, statistics, normalization_path


def build_fusion_cache(
    manifest_path,
    artifact_dirs,
    output_dir,
    *,
    split,
    normalization_output=None,
    normalization_input=None,
    limit=None,
):
    """Join keyed artifacts, normalize them, and atomically write one cache."""
    if split == "train":
        if normalization_output is None or normalization_input is not None:
            raise ValueError(
                "train cache requires normalization_output and forbids normalization_input"
            )
    elif normalization_input is None or normalization_output is not None:
        raise ValueError(
            "non-train cache requires normalization_input and forbids normalization_output"
        )
    manifest = load_manifest(manifest_path, expected_split=split)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        if not 1 <= limit <= len(manifest.keys):
            raise ValueError("limit must be between 1 and manifest count")
    count = len(manifest.keys) if limit is None else limit
    artifacts = _load_artifacts(artifact_dirs, manifest, split)
    keys, labels, predictions, available, latent_arrays, missing = _join_artifacts(
        manifest, artifacts, count
    )
    normalization, statistics, normalization_path = _normalization_for_split(
        split,
        normalization_output,
        normalization_input,
        predictions,
        available,
        latent_arrays,
    )
    standardized_predictions, normalized_latents = _apply_normalization(
        predictions, available, latent_arrays, statistics
    )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    normalization_snapshot = output_path / "normalization.json"
    atomic_write_json(normalization_snapshot, normalization)
    normalization_sha256 = sha256_file(normalization_snapshot)
    coverage = {
        "status": "complete",
        "split": split,
        "target_count": count,
        "experts": {
            expert: {
                "available_count": int(available[:, index].sum()),
                "missing_count": int((~available[:, index]).sum()),
                "missing_key_sample": missing[expert][:100],
            }
            for index, expert in enumerate(EXPERT_ORDER)
        },
    }
    atomic_write_json(output_path / "coverage.json", coverage)
    arrays = {
        "keys": np.asarray(keys, dtype=np.str_),
        "labels": labels,
        "predictions": predictions,
        "standardized_predictions": standardized_predictions,
        "available": available,
        "upper_bound_latents": normalized_latents["upper_bound"],
        "boltz_latents": normalized_latents["boltz_pairwise"],
        "three_dprot_latents": normalized_latents["three_dprot"],
    }
    _atomic_write_npz(output_path / "features.npz", arrays)
    metadata = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "count": count,
        "source_count": len(manifest.keys),
        "manifest_sha256": manifest.sha256,
        "ordered_key_sha256": _ordered_key_sha256(keys),
        "expert_order": list(EXPERT_ORDER),
        "expert_checkpoint_sha256": {
            expert: artifacts[expert].metadata["checkpoint_sha256"]
            for expert in EXPERT_ORDER
        },
        "latent_dimensions": {
            expert: int(latent_arrays[expert].shape[1]) for expert in EXPERT_ORDER
        },
        "normalization_sha256": normalization_sha256,
        "normalization_source": str(normalization_path),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def _read_cache_arrays(path):
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            names = frozenset(archive.files)
            if names != FEATURE_NAMES:
                raise ValueError("fusion cache feature schema does not match")
            return {name: archive[name] for name in FEATURE_NAMES}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "schema" in str(error):
            raise
        raise ValueError("fusion cache features are invalid") from error


def load_fusion_cache(
    output_dir, *, expected_split, expected_manifest_sha256=None
):
    """Validate and return a complete fusion cache."""
    output_path = Path(output_dir).expanduser().resolve()
    metadata_path = output_path / "metadata.json"
    features_path = output_path / "features.npz"
    coverage_path = output_path / "coverage.json"
    normalization_path = output_path / "normalization.json"
    if not metadata_path.is_file():
        raise RuntimeError("fusion cache metadata is missing")
    for path in (features_path, coverage_path, normalization_path):
        if not path.is_file():
            raise RuntimeError(f"fusion cache {path.name} is missing")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete" or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("fusion cache metadata is incomplete")
    if metadata.get("split") != expected_split:
        raise ValueError("fusion cache split does not match")
    if expected_manifest_sha256 is not None and metadata.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("fusion cache manifest_sha256 does not match")
    if metadata.get("expert_order") != list(EXPERT_ORDER):
        raise ValueError("fusion cache expert order does not match")
    if metadata.get("normalization_sha256") != sha256_file(normalization_path):
        raise ValueError("fusion cache normalization hash does not match")
    arrays = _read_cache_arrays(features_path)
    keys = tuple(str(value) for value in arrays["keys"].tolist())
    count = len(keys)
    expected_shapes = {
        "labels": (count,),
        "predictions": (count, 3),
        "standardized_predictions": (count, 3),
        "available": (count, 3),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"fusion cache {name} shape does not match")
    if arrays["available"].dtype != np.bool_:
        raise ValueError("fusion cache available must be boolean")
    for name in LATENT_FIELDS.values():
        if arrays[name].ndim != 2 or arrays[name].shape[0] != count:
            raise ValueError(f"fusion cache {name} shape does not match")
    if count != metadata.get("count") or len(keys) != len(set(keys)):
        raise ValueError("fusion cache key count does not match")
    if metadata.get("ordered_key_sha256") != _ordered_key_sha256(keys):
        raise ValueError("fusion cache ordered key hash does not match")
    numeric_names = FEATURE_NAMES.difference({"keys", "available"})
    if any(not np.isfinite(arrays[name]).all() for name in numeric_names):
        raise ValueError("fusion cache contains nonfinite values")
    unavailable = ~arrays["available"]
    if np.any(arrays["predictions"][unavailable] != 0.0) or np.any(
        arrays["standardized_predictions"][unavailable] != 0.0
    ):
        raise ValueError("fusion cache unavailable predictions must be zero")
    for expert_index, expert in enumerate(EXPERT_ORDER):
        values = arrays[LATENT_FIELDS[expert]]
        if np.any(values[~arrays["available"][:, expert_index]] != 0.0):
            raise ValueError("fusion cache unavailable latents must be zero")
    return FusionCache(
        keys=keys,
        labels=np.asarray(arrays["labels"], dtype=np.float32),
        predictions=np.asarray(arrays["predictions"], dtype=np.float32),
        standardized_predictions=np.asarray(
            arrays["standardized_predictions"], dtype=np.float32
        ),
        available=np.asarray(arrays["available"], dtype=np.bool_),
        upper_bound_latents=np.asarray(arrays["upper_bound_latents"], dtype=np.float32),
        boltz_latents=np.asarray(arrays["boltz_latents"], dtype=np.float32),
        three_dprot_latents=np.asarray(arrays["three_dprot_latents"], dtype=np.float32),
        metadata=metadata,
    )
