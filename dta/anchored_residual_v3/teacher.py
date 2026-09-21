"""Immutable keyed teacher-prediction artifacts for residual training."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np


_REQUIRED_METADATA = {
    "status",
    "split",
    "manifest_sha256",
    "teacher_checkpoint_sha256",
}


@dataclass(frozen=True)
class TeacherPredictionLookup:
    keys: tuple
    labels: np.ndarray
    predictions: np.ndarray
    metadata: dict

    def __post_init__(self):
        object.__setattr__(
            self, "_by_key", dict(zip(self.keys, self.predictions.tolist()))
        )

    def __len__(self):
        return len(self.keys)

    def value_for(self, key):
        key = str(key)
        try:
            return float(self._by_key[key])
        except KeyError:
            raise KeyError(f"teacher prediction is missing key {key}")


def _validate_hash(value, name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 string")


def _validated_arrays(keys, labels, predictions):
    keys = tuple(np.asarray(keys).astype(str).reshape(-1).tolist())
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not keys or len(keys) != len(labels) or len(keys) != len(predictions):
        raise ValueError("teacher keys, labels, and predictions must have equal nonzero length")
    if len(set(keys)) != len(keys):
        raise ValueError("teacher keys must be unique")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("teacher labels and predictions must be finite")
    return keys, labels, predictions


def load_teacher_artifact(
    predictions_path,
    metadata_path,
    *,
    expected_split,
    expected_manifest_sha256=None,
    expected_checkpoint_sha256=None,
):
    predictions_path = Path(predictions_path).expanduser().resolve()
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not predictions_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("teacher predictions and metadata are required")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or not _REQUIRED_METADATA.issubset(metadata):
        raise ValueError("teacher metadata schema is incomplete")
    if metadata["status"] != "complete":
        raise ValueError("teacher metadata status is not complete")
    if metadata["split"] != str(expected_split):
        raise ValueError("teacher split does not match expected split")
    _validate_hash(metadata["manifest_sha256"], "manifest_sha256")
    _validate_hash(
        metadata["teacher_checkpoint_sha256"], "teacher_checkpoint_sha256"
    )
    if (
        expected_manifest_sha256 is not None
        and metadata["manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("teacher manifest SHA256 does not match")
    if (
        expected_checkpoint_sha256 is not None
        and metadata["teacher_checkpoint_sha256"] != expected_checkpoint_sha256
    ):
        raise ValueError("teacher checkpoint SHA256 does not match")
    with np.load(predictions_path, allow_pickle=False) as archive:
        if set(archive.files) != {"keys", "labels", "predictions"}:
            raise ValueError("teacher prediction archive schema is invalid")
        keys, labels, predictions = _validated_arrays(
            archive["keys"], archive["labels"], archive["predictions"]
        )
    return TeacherPredictionLookup(keys, labels, predictions, dict(metadata))


def write_teacher_artifact(
    output_dir,
    *,
    keys,
    labels,
    predictions,
    split,
    manifest_sha256,
    teacher_checkpoint_sha256,
):
    keys, labels, predictions = _validated_arrays(keys, labels, predictions)
    _validate_hash(manifest_sha256, "manifest_sha256")
    _validate_hash(teacher_checkpoint_sha256, "teacher_checkpoint_sha256")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "complete",
        "split": str(split),
        "manifest_sha256": manifest_sha256,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "count": len(keys),
    }
    npz_fd, npz_temporary = tempfile.mkstemp(
        prefix="teacher.", suffix=".tmp", dir=str(output_dir)
    )
    os.close(npz_fd)
    metadata_fd, metadata_temporary = tempfile.mkstemp(
        prefix="metadata.", suffix=".tmp", dir=str(output_dir)
    )
    os.close(metadata_fd)
    try:
        with open(npz_temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                keys=np.asarray(keys),
                labels=labels,
                predictions=predictions,
            )
        with open(metadata_temporary, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        os.replace(npz_temporary, output_dir / "predictions.npz")
        os.replace(metadata_temporary, output_dir / "metadata.json")
    finally:
        for temporary in (npz_temporary, metadata_temporary):
            if os.path.exists(temporary):
                os.unlink(temporary)
    return metadata


def generate_teacher_predictions(
    model,
    loader,
    checkpoint_path,
    output_dir,
    *,
    split,
    manifest_sha256,
    device,
):
    """Load a cached fusion checkpoint and persist ordered predictions."""
    import torch

    from utils.experiment_integrity import sha256_file

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError("teacher checkpoint is missing model_state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    keys, labels, predictions = [], [], []

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {name: move(item) for name, item in value.items()}
        return value

    with torch.inference_mode():
        for batch in loader:
            batch_keys = [str(key) for key in batch["keys"]]
            moved = move(batch)
            output = model(moved)
            prediction = output.get("prediction") if isinstance(output, dict) else None
            if not torch.is_tensor(prediction):
                raise RuntimeError("teacher model output is missing prediction")
            target = moved.get("label")
            if not torch.is_tensor(target):
                raise RuntimeError("teacher batch is missing label")
            keys.extend(batch_keys)
            labels.extend(target.reshape(-1).float().cpu().tolist())
            predictions.extend(prediction.reshape(-1).float().cpu().tolist())
    return write_teacher_artifact(
        output_dir,
        keys=keys,
        labels=labels,
        predictions=predictions,
        split=split,
        manifest_sha256=manifest_sha256,
        teacher_checkpoint_sha256=sha256_file(checkpoint_path),
    )
