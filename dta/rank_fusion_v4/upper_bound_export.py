"""Keyed feature export for the frozen Upper-Bound seed87 expert."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.hybrid_feature_cache_v2 import (
    CachedHybridFeatureDataset,
    collate_cached_hybrid_features,
)
from dta.hybrid_e2e_training_v2 import load_cached_fusion_checkpoint
from net.hybrid_upper_bound import HybridUpperBoundFusion
from utils.experiment_integrity import atomic_write_json, manifest_fingerprint

from .artifact import write_expert_artifact


ADAPTER_VERSION = "upper-bound-seed87-v1"
PROJECTION_INPUT_DIMENSION = 1024
PROJECTION_OUTPUT_DIMENSION = 1014
DIAGNOSTIC_NAMES = tuple(
    [f"expert_weight_{index}" for index in range(8)]
    + ["gerna_prediction", "interface_prediction"]
)


def _load_manifest(path, expected_split):
    path = Path(path).expanduser().resolve()
    fingerprint = manifest_fingerprint(path)
    if fingerprint["split"] != expected_split:
        raise ValueError("manifest split does not match requested split")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a nonempty list")
    keys = []
    labels = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("manifest sample must be a mapping")
        key = sample.get("lmdb_key")
        if not isinstance(key, str) or not key:
            raise ValueError("manifest lmdb_key must be nonempty")
        try:
            label = float(sample["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"manifest label is invalid for {key}") from error
        if not math.isfinite(label):
            raise ValueError(f"manifest label is nonfinite for {key}")
        keys.append(key)
        labels.append(label)
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate keys")
    return fingerprint, keys, np.asarray(labels, dtype=np.float32)


def _validate_cache(dataset, keys, labels, split):
    metadata = dataset.metadata
    if metadata.get("status") != "complete":
        raise RuntimeError("Upper-Bound cache is not complete")
    if metadata.get("split") != split:
        raise ValueError("Upper-Bound cache split does not match manifest")
    if list(dataset.keys) != list(keys):
        raise ValueError("Upper-Bound cache key order does not match manifest")
    if len(dataset) != len(keys):
        raise ValueError("Upper-Bound cache count does not match manifest")

    cache_labels = []
    for index, expected_key in enumerate(keys):
        record = dataset[index]
        if record.get("key") != expected_key:
            raise ValueError("Upper-Bound cache key order does not match records")
        label = record.get("label")
        if not torch.is_tensor(label) or label.numel() != 1:
            raise ValueError(f"Upper-Bound cache label is invalid for {expected_key}")
        cache_labels.append(float(label.reshape(-1)[0]))
    cache_labels = np.asarray(cache_labels, dtype=np.float32)
    if not np.isfinite(cache_labels).all() or not np.allclose(
        cache_labels, labels, rtol=0.0, atol=1e-5
    ):
        raise ValueError("Upper-Bound cache label does not match manifest")


def _projection(checkpoint_sha256):
    digest = hashlib.sha256(str(checkpoint_sha256).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
    generator = np.random.RandomState(seed)
    matrix = generator.standard_normal(
        (PROJECTION_INPUT_DIMENSION, PROJECTION_OUTPUT_DIMENSION)
    ).astype(np.float32)
    matrix /= np.float32(math.sqrt(PROJECTION_OUTPUT_DIMENSION))
    return seed, matrix, hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {name: _move_to_device(item, device) for name, item in value.items()}
    return value


def _export_batches(model, loader, device, projection):
    keys = []
    labels = []
    predictions = []
    packed_latents = []
    projection_tensor = torch.from_numpy(projection).to(device=device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch_keys = list(batch["keys"])
            batch = _move_to_device(batch, device)
            output = model(batch)
            latents = output.get("latents")
            weights = output.get("expert_weights")
            prediction = output.get("prediction")
            if (
                not torch.is_tensor(latents)
                or latents.ndim != 3
                or latents.shape[1:] != (32, 512)
            ):
                raise ValueError("Upper-Bound latents must have shape [B,32,512]")
            batch_size = latents.shape[0]
            if not torch.is_tensor(weights) or weights.shape != (batch_size, 8):
                raise ValueError("Upper-Bound expert_weights must have shape [B,8]")
            if not torch.is_tensor(prediction) or prediction.shape != (batch_size, 1):
                raise ValueError("Upper-Bound prediction must have shape [B,1]")
            gerna = batch["gerna_prediction"]
            interface = batch["interface_prediction"]
            if gerna.shape != (batch_size, 1) or interface.shape != (batch_size, 1):
                raise ValueError("Upper-Bound backbone predictions must have shape [B,1]")
            pooled = torch.cat(
                (latents.float().mean(dim=1), latents.float().amax(dim=1)), dim=-1
            )
            projected = pooled @ projection_tensor
            packed = torch.cat(
                (projected, weights.float(), gerna.float(), interface.float()), dim=-1
            )
            if packed.shape != (batch_size, 1024):
                raise RuntimeError("Upper-Bound packed latent width is not 1024")
            keys.extend(batch_keys)
            labels.extend(batch["label"].reshape(-1).float().cpu().tolist())
            predictions.extend(prediction.reshape(-1).float().cpu().tolist())
            packed_latents.append(packed.cpu().numpy().astype(np.float32, copy=False))
    return (
        keys,
        np.asarray(labels, dtype=np.float32),
        np.asarray(predictions, dtype=np.float32),
        np.concatenate(packed_latents, axis=0),
    )


def export_upper_bound(
    cache_dir,
    manifest_path,
    checkpoint_path,
    output_dir,
    *,
    split,
    device,
    batch_size=64,
    limit=None,
):
    """Export one complete keyed Upper-Bound artifact and return metadata."""
    if not isinstance(split, str) or not split:
        raise ValueError("split must be nonempty")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    fingerprint, manifest_keys, manifest_labels = _load_manifest(
        manifest_path, split
    )
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        if limit <= 0 or limit > len(manifest_keys):
            raise ValueError("limit must be between 1 and manifest count")

    dataset = CachedHybridFeatureDataset(cache_dir)
    try:
        _validate_cache(dataset, manifest_keys, manifest_labels, split)
        selected_count = len(manifest_keys) if limit is None else limit
        selected = torch.utils.data.Subset(dataset, range(selected_count))
        loader = DataLoader(
            selected,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_cached_hybrid_features,
        )

        model = HybridUpperBoundFusion(dropout=0.0, drop_path_rate=0.0)
        checkpoint = load_cached_fusion_checkpoint(model, checkpoint_path)
        checkpoint_sha256 = checkpoint.get("sha256")
        if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
            raise RuntimeError("Upper-Bound checkpoint SHA256 is missing")
        model = model.to(torch.device(device))
        seed, projection, projection_sha256 = _projection(checkpoint_sha256)
        keys, labels, predictions, packed_latents = _export_batches(
            model, loader, torch.device(device), projection
        )

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        adapter_metadata = {
            "status": "complete",
            "adapter_version": ADAPTER_VERSION,
            "expert": "upper_bound",
            "split": split,
            "manifest_sha256": fingerprint["sha256"],
            "source_count": len(manifest_keys),
            "selected_count": selected_count,
            "checkpoint_sha256": checkpoint_sha256,
            "projection_seed": seed,
            "projection_input_dimension": PROJECTION_INPUT_DIMENSION,
            "projection_output_dimension": PROJECTION_OUTPUT_DIMENSION,
            "projection_sha256": projection_sha256,
            "appended_diagnostics": list(DIAGNOSTIC_NAMES),
        }
        artifact_metadata = write_expert_artifact(
            output_path,
            keys=keys,
            labels=labels,
            predictions=predictions,
            latents=packed_latents,
            available=np.ones(selected_count, dtype=np.bool_),
            expert="upper_bound",
            split=split,
            manifest_sha256=fingerprint["sha256"],
            checkpoint_sha256=checkpoint_sha256,
            adapter_version=ADAPTER_VERSION,
        )
        atomic_write_json(output_path / "adapter_metadata.json", adapter_metadata)
        return artifact_metadata
    finally:
        dataset.close()
