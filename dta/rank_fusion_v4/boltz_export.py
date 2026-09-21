"""Keyed feature export for the frozen Boltz pairwise expert."""

import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from utils.experiment_integrity import atomic_write_json, manifest_fingerprint, sha256_file

from .artifact import write_expert_artifact


ADAPTER_VERSION = "boltz-pairwise-v2"
SOURCE_LABEL_ROUNDING_TOLERANCE = 0.0050001
_EXTERNAL_PACKAGE = "_rank_fusion_v4_boltz_dta"


def forward_pairwise_features(model, batch):
    """Reproduce PairwiseTwoBranch.forward and expose its pooled states."""
    pocket = model.pocket_proj(batch["pocket_emb"])
    ligand = model.ligand_proj(batch["ligand_emb"])
    interactions = pocket.unsqueeze(2) * ligand.unsqueeze(1)
    pair_values = model.pair_value(interactions)
    pair_scores = model.pair_score(interactions).squeeze(-1)
    mask = batch["pocket_mask"].unsqueeze(2) & batch["ligand_mask"].unsqueeze(1)
    pair_scores = pair_scores.masked_fill(~mask, float("-inf"))
    attention = torch.softmax(pair_scores.flatten(1), dim=1).view_as(pair_scores)
    attention = torch.nan_to_num(attention, nan=0.0)
    pair_attention = (attention.unsqueeze(-1) * pair_values).sum(dim=(1, 2))
    weights = mask.unsqueeze(-1).to(dtype=pair_values.dtype)
    pair_mean = (pair_values * weights).sum(dim=(1, 2)) / weights.sum(
        dim=(1, 2)
    ).clamp_min(1.0)
    local = pair_mean + model.pair_delta_gate * (pair_attention - pair_mean)
    global_state = model.global_mlp(batch["global_feat"])
    fused = torch.cat((local, global_state), dim=-1)
    head_hidden = model.head[:4](fused)
    standardized = model.head[4](head_hidden).reshape(-1)
    latent = torch.cat(
        (pair_mean, pair_attention, local, global_state, head_hidden), dim=-1
    )
    return standardized, latent


def _build_dependencies():
    root = Path(__file__).resolve().parents[2] / "external_benchmarks" / "imported_index_20260721" / "boltz_dta" / "dta"
    if _EXTERNAL_PACKAGE not in sys.modules:
        specification = importlib.util.spec_from_file_location(
            _EXTERNAL_PACKAGE,
            str(root / "__init__.py"),
            submodule_search_locations=[str(root)],
        )
        if specification is None or specification.loader is None:
            raise ImportError("cannot load the external Boltz DTA package")
        package = importlib.util.module_from_spec(specification)
        sys.modules[_EXTERNAL_PACKAGE] = package
        specification.loader.exec_module(package)
    data_module = importlib.import_module(_EXTERNAL_PACKAGE + ".data")
    model_module = importlib.import_module(_EXTERNAL_PACKAGE + ".models")

    def build(config):
        return model_module.build_model(
            "pairwise",
            dim=int(config.get("dim", 128)),
            dropout=float(config.get("dropout", 0.3)),
            no_local=bool(config.get("no_local", False)),
            no_global=bool(config.get("no_global", False)),
            no_pair_delta=bool(config.get("no_pair_delta", False)),
        )

    return data_module.BoltzDTADataset, build


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
    keys, labels = [], []
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


def _load_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location="cpu")
    required = {"state_dict", "y_mu", "y_sd", "config"}
    missing = sorted(required.difference(payload if isinstance(payload, dict) else {}))
    if missing:
        raise ValueError("Boltz checkpoint is missing " + ", ".join(missing))
    if not isinstance(payload["state_dict"], dict) or not isinstance(payload["config"], dict):
        raise ValueError("Boltz checkpoint state_dict and config must be mappings")
    for name in ("y_mu", "y_sd"):
        try:
            value = float(payload[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Boltz checkpoint {name} is invalid") from error
        if not math.isfinite(value):
            raise ValueError(f"Boltz checkpoint {name} is nonfinite")
    if float(payload["y_sd"]) <= 0.0:
        raise ValueError("Boltz checkpoint y_sd must be positive")
    config = payload["config"]
    if config.get("model") != "pairwise":
        raise ValueError("Boltz checkpoint model must be pairwise")
    if any(bool(config.get(name, False)) for name in ("no_local", "no_global", "no_pair_delta")):
        raise ValueError("Boltz pairwise checkpoint must enable local, global, and pair delta")
    return path, payload


def _collate(records):
    if not records:
        raise ValueError("Boltz record batch must be nonempty")
    names = set(records[0])
    if any(set(record) != names for record in records):
        raise ValueError("Boltz record fields do not match")
    return {name: torch.stack([record[name] for record in records], dim=0) for name in names}


def _move(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _flush(model, records, row_indices, device, y_mu, y_sd, predictions, latents):
    batch = _move(_collate(records), device)
    with torch.inference_mode():
        standardized, hidden = forward_pairwise_features(model, batch)
    output = standardized.float() * float(y_sd) + float(y_mu)
    if hidden.ndim != 2 or hidden.shape[0] != len(row_indices):
        raise RuntimeError("Boltz latent batch has an invalid shape")
    if hidden.shape[1] > 1024:
        raise RuntimeError("Boltz latent width exceeds 1024")
    if not torch.isfinite(output).all() or not torch.isfinite(hidden).all():
        raise FloatingPointError("Boltz model produced nonfinite output")
    for offset, row_index in enumerate(row_indices):
        predictions[row_index] = float(output[offset].cpu())
        latents[row_index] = hidden[offset].float().cpu().numpy()


def export_boltz_pairwise(
    manifest_path,
    checkpoint_path,
    output_dir,
    *,
    split,
    device,
    batch_size=64,
    limit=None,
    lmdb_dir=None,
    index_path=None,
):
    """Export a target-manifest-aligned Boltz pairwise artifact."""
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    fingerprint, keys, labels = _load_manifest(manifest_path, split)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(keys):
            raise ValueError("limit must be between 1 and manifest count")
        keys, labels = keys[:limit], labels[:limit]
    checkpoint_path, checkpoint = _load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    selected_lmdb = lmdb_dir or config.get("merged_lmdb_dir")
    selected_index = index_path or config.get("index_path")
    if not selected_lmdb or not selected_index:
        raise ValueError("Boltz checkpoint config is missing LMDB or index path")

    Dataset, build_model = _build_dependencies()
    dataset_kwargs = {
        "pocket_radius": float(config.get("pocket_radius", 10.0)),
        "min_pocket_residues": int(config.get("min_pocket_residues", 16)),
        "max_pocket_residues": int(config.get("max_pocket_residues", 64)),
        "max_ligand_atoms": int(config.get("max_ligand_atoms", 64)),
        "edge_cutoff": float(config.get("edge_cutoff", 5.0)),
    }
    dataset = Dataset(
        selected_lmdb,
        selected_index,
        split=split,
        mode="pairwise",
        **dataset_kwargs,
    )
    dataset_keys = list(dataset.keys)
    if len(dataset_keys) != len(set(dataset_keys)):
        raise ValueError("Boltz dataset contains duplicate keys")
    key_to_index = {key: index for index, key in enumerate(dataset_keys)}
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(torch.device(device)).eval()
    latent_width = int(config.get("dim", 128)) * 5
    if latent_width > 1024:
        raise ValueError("Boltz latent width exceeds 1024")
    predictions = np.zeros(len(keys), dtype=np.float32)
    latents = np.zeros((len(keys), latent_width), dtype=np.float32)
    available = np.zeros(len(keys), dtype=np.bool_)
    missing = []
    source_label_mismatches = []
    source_label_max_abs_delta = 0.0
    pending_records, pending_rows = [], []

    def flush_pending():
        if not pending_records:
            return
        _flush(
            model,
            list(pending_records),
            list(pending_rows),
            torch.device(device),
            checkpoint["y_mu"],
            checkpoint["y_sd"],
            predictions,
            latents,
        )
        for row_index in pending_rows:
            available[row_index] = True
        pending_records.clear()
        pending_rows.clear()

    for row_index, key in enumerate(keys):
        dataset_index = key_to_index.get(key)
        if dataset_index is None:
            missing.append({"key": key, "reason": "not_in_boltz_dataset"})
            continue
        dataset_label = float(dataset.y[dataset_index])
        if not math.isfinite(dataset_label):
            raise ValueError(f"Boltz label is nonfinite for {key}")
        manifest_label = float(labels[row_index])
        label_delta = abs(dataset_label - manifest_label)
        source_label_max_abs_delta = max(source_label_max_abs_delta, label_delta)
        if label_delta > SOURCE_LABEL_ROUNDING_TOLERANCE:
            source_label_mismatches.append(
                {
                    "key": key,
                    "manifest_label": manifest_label,
                    "boltz_source_label": dataset_label,
                    "absolute_delta": label_delta,
                }
            )
        try:
            record = dataset[dataset_index]
        except (ValueError, KeyError, IndexError) as error:
            missing.append(
                {"key": key, "reason": "record_error", "error": repr(error)}
            )
            continue
        pending_records.append(record)
        pending_rows.append(row_index)
        if len(pending_records) >= int(batch_size):
            flush_pending()
    flush_pending()

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    metadata = write_expert_artifact(
        output_path,
        keys=keys,
        labels=labels,
        predictions=predictions,
        latents=latents,
        available=available,
        expert="boltz_pairwise",
        split=split,
        manifest_sha256=fingerprint["sha256"],
        checkpoint_sha256=checkpoint_sha256,
        adapter_version=ADAPTER_VERSION,
    )
    atomic_write_json(
        output_path / "adapter_metadata.json",
        {
            "status": "complete",
            "adapter_version": ADAPTER_VERSION,
            "expert": "boltz_pairwise",
            "split": split,
            "manifest_sha256": fingerprint["sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "source_count": fingerprint["sample_count"],
            "selected_count": len(keys),
            "available_count": int(available.sum()),
            "missing_count": int((~available).sum()),
            "missing": missing,
            "source_label_policy": "manifest_is_canonical",
            "source_label_rounding_tolerance": SOURCE_LABEL_ROUNDING_TOLERANCE,
            "source_label_mismatch_count": len(source_label_mismatches),
            "source_label_mismatches": source_label_mismatches,
            "source_label_max_abs_delta": source_label_max_abs_delta,
            "latent_dimension": latent_width,
            "lmdb_dir": str(selected_lmdb),
            "index_path": str(selected_index),
        },
    )
    return metadata
