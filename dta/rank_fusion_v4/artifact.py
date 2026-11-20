"""Atomic, provenance-bound expert feature artifacts."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


SCHEMA_VERSION = 1
FEATURE_NAMES = frozenset(
    {"keys", "labels", "predictions", "latents", "available"}
)
METADATA_NAMES = frozenset(
    {
        "status",
        "schema_version",
        "expert",
        "split",
        "count",
        "manifest_sha256",
        "ordered_key_sha256",
        "checkpoint_sha256",
        "latent_dimension",
        "adapter_version",
        "missing_count",
    }
)


@dataclass(frozen=True)
class ExpertArtifact:
    keys: tuple
    labels: np.ndarray
    predictions: np.ndarray
    latents: np.ndarray
    available: np.ndarray
    metadata: dict


def _required_text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _normalize_keys(values):
    values = list(values)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("keys must be a nonempty sequence of nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError("keys contains duplicate values")
    return tuple(values)


def _ordered_key_sha256(keys):
    payload = json.dumps(
        list(keys), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _numeric_array(value, name, dimensions):
    array = np.asarray(value)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must have rank {dimensions}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must be numeric")
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalize_arrays(keys, labels, predictions, latents, available):
    keys = _normalize_keys(keys)
    labels = _numeric_array(labels, "labels", 1)
    predictions = _numeric_array(predictions, "predictions", 1)
    latents = _numeric_array(latents, "latents", 2)
    available_array = np.asarray(available)
    if available_array.ndim != 1:
        raise ValueError("available must have rank 1")
    if not np.issubdtype(available_array.dtype, np.bool_):
        raise ValueError("available must be boolean")
    available_array = np.ascontiguousarray(available_array, dtype=np.bool_)

    count = len(keys)
    lengths = {
        "labels": len(labels),
        "predictions": len(predictions),
        "latents": len(latents),
        "available": len(available_array),
    }
    mismatched = sorted(name for name, length in lengths.items() if length != count)
    if mismatched:
        raise ValueError("array length mismatch: " + ", ".join(mismatched))
    if not 1 <= latents.shape[1] <= 1024:
        raise ValueError("latents width must be between 1 and 1024")
    if not np.isfinite(labels).all():
        raise ValueError("labels contains nonfinite values")

    if not np.isfinite(predictions[available_array]).all():
        raise ValueError("predictions contains nonfinite available values")
    if not np.isfinite(latents[available_array]).all():
        raise ValueError("latents contains nonfinite available values")
    unavailable = ~available_array
    if np.any(predictions[unavailable] != 0.0):
        raise ValueError("unavailable predictions must be zero")
    if np.any(latents[unavailable] != 0.0):
        raise ValueError("unavailable latents must be zero")
    return keys, labels, predictions, latents, available_array


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    missing = sorted(METADATA_NAMES.difference(metadata))
    if missing:
        raise ValueError("metadata is missing " + ", ".join(missing))
    if metadata["status"] != "complete":
        raise ValueError("metadata status must be complete")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("metadata schema_version does not match")
    for name in (
        "expert",
        "split",
        "manifest_sha256",
        "ordered_key_sha256",
        "checkpoint_sha256",
        "adapter_version",
    ):
        _required_text(metadata[name], f"metadata {name}")
    for name in ("count", "latent_dimension", "missing_count"):
        if isinstance(metadata[name], bool) or not isinstance(metadata[name], int):
            raise ValueError(f"metadata {name} must be an integer")
    return metadata


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


def _atomic_write_json(path, payload):
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".metadata-", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_expert_artifact(
    output_dir,
    *,
    keys,
    labels,
    predictions,
    latents,
    available,
    expert,
    split,
    manifest_sha256,
    checkpoint_sha256,
    adapter_version,
):
    """Validate arrays, atomically write the artifact, and return metadata."""
    expert = _required_text(expert, "expert")
    split = _required_text(split, "split")
    manifest_sha256 = _required_text(manifest_sha256, "manifest_sha256")
    checkpoint_sha256 = _required_text(checkpoint_sha256, "checkpoint_sha256")
    adapter_version = _required_text(adapter_version, "adapter_version")
    keys, labels, predictions, latents, available = _normalize_arrays(
        keys, labels, predictions, latents, available
    )
    metadata = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "expert": expert,
        "split": split,
        "count": len(keys),
        "manifest_sha256": manifest_sha256,
        "ordered_key_sha256": _ordered_key_sha256(keys),
        "checkpoint_sha256": checkpoint_sha256,
        "latent_dimension": int(latents.shape[1]),
        "adapter_version": adapter_version,
        "missing_count": int((~available).sum()),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    feature_path = output_path / "features.npz"
    metadata_path = output_path / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    _atomic_write_npz(
        feature_path,
        {
            "keys": np.asarray(keys, dtype=np.str_),
            "labels": labels,
            "predictions": predictions,
            "latents": latents,
            "available": available,
        },
    )
    _atomic_write_json(metadata_path, metadata)
    return metadata


def load_expert_artifact(
    output_dir,
    *,
    expected_expert,
    expected_split,
    expected_manifest_sha256=None,
    expected_checkpoint_sha256=None,
):
    """Validate provenance and arrays, then return an ExpertArtifact."""
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.json"
    feature_path = output_path / "features.npz"
    if not metadata_path.is_file():
        raise RuntimeError("artifact metadata is missing")
    if not feature_path.is_file():
        raise RuntimeError("artifact features are missing")
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("artifact metadata is invalid") from error
    _validate_metadata(metadata)

    expected = {
        "expert": _required_text(expected_expert, "expected_expert"),
        "split": _required_text(expected_split, "expected_split"),
    }
    if expected_manifest_sha256 is not None:
        expected["manifest_sha256"] = _required_text(
            expected_manifest_sha256, "expected_manifest_sha256"
        )
    if expected_checkpoint_sha256 is not None:
        expected["checkpoint_sha256"] = _required_text(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        )
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"artifact {name} does not match expected value")

    try:
        with np.load(str(feature_path), allow_pickle=False) as payload:
            names = frozenset(payload.files)
            if names != FEATURE_NAMES:
                missing = sorted(FEATURE_NAMES.difference(names))
                extra = sorted(names.difference(FEATURE_NAMES))
                raise ValueError(
                    "artifact features schema mismatch: missing "
                    + ", ".join(missing)
                    + "; extra "
                    + ", ".join(extra)
                )
            keys = payload["keys"]
            if keys.ndim != 1 or keys.dtype.kind != "U":
                raise ValueError("artifact keys must be a Unicode vector")
            normalized = _normalize_arrays(
                keys.tolist(),
                payload["labels"],
                payload["predictions"],
                payload["latents"],
                payload["available"],
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("artifact features are invalid") from error
    keys, labels, predictions, latents, available = normalized

    derived = {
        "count": len(keys),
        "ordered_key_sha256": _ordered_key_sha256(keys),
        "latent_dimension": int(latents.shape[1]),
        "missing_count": int((~available).sum()),
    }
    for name, value in derived.items():
        if metadata.get(name) != value:
            raise ValueError(f"metadata {name} does not match artifact features")
    return ExpertArtifact(
        keys=keys,
        labels=labels,
        predictions=predictions,
        latents=latents,
        available=available,
        metadata=dict(metadata),
    )
