"""Persistent multiscale feature cache for the upper-bound hybrid model."""

import json
import math
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch

from utils.experiment_integrity import atomic_write_json, hash_ordered_keys


CACHE_SCHEMA_VERSION = 2

FP16_FEATURE_FIELDS = (
    "gerna_2d_molecule",
    "gerna_2d_protein",
    "gerna_3d_molecule",
    "gerna_3d_protein",
    "gerna_2d_interaction",
    "gerna_3d_interaction",
    "gerna_head",
    "interface_ligand_nodes",
    "interface_protein_nodes",
    "interface_residue_nodes",
    "interface_ligand_summary",
    "interface_protein_summary",
    "interface_local_summary",
    "interface_head",
    "interface_projection",
)

FP32_FIELDS = (
    "interface_ligand_coords",
    "interface_protein_coords",
    "interface_residue_coords",
    "interface_score",
    "gerna_prediction",
    "interface_prediction",
    "label",
)

FIXED_FEATURE_SHAPES = {
    "gerna_2d_molecule": (128,),
    "gerna_2d_protein": (128,),
    "gerna_3d_molecule": (128,),
    "gerna_3d_protein": (128,),
    "gerna_2d_interaction": (128, 128),
    "gerna_3d_interaction": (128, 128),
    "gerna_head": (128,),
    "interface_ligand_summary": (96,),
    "interface_protein_summary": (128,),
    "interface_local_summary": (128,),
    "interface_head": (128,),
    "interface_projection": (64,),
}

NODE_FEATURE_DIMS = {
    "interface_ligand_nodes": 96,
    "interface_protein_nodes": 96,
    "interface_residue_nodes": 128,
}

COORDINATE_FIELDS = {
    "interface_ligand_nodes": "interface_ligand_coords",
    "interface_protein_nodes": "interface_protein_coords",
    "interface_residue_nodes": "interface_residue_coords",
}

SCALAR_FIELDS = (
    "interface_score",
    "gerna_prediction",
    "interface_prediction",
    "label",
)

RECORD_FIELDS = frozenset(FP16_FEATURE_FIELDS + FP32_FIELDS)
_METADATA_KEY = b"__metadata__"
_KEYS_KEY = b"__keys__"
_RECORD_PREFIX = b"record:"


def _as_unique_keys(expected_keys):
    keys = [str(key) for key in expected_keys]
    if not keys or any(not key for key in keys):
        raise ValueError("expected_keys must be a nonempty sequence of strings")
    if len(keys) != len(set(keys)):
        raise ValueError("expected_keys contains duplicate keys")
    return keys


def _record_key(key):
    return _RECORD_PREFIX + str(key).encode("utf-8")


def _serialize(value):
    return pickle.dumps(value, protocol=4)


def _deserialize(value):
    return pickle.loads(value)


def _metadata_sidecar(path):
    return Path(path) / "metadata.json"


def _open_environment(path, readonly, map_size=None):
    path = os.path.abspath(os.fspath(path))
    if readonly and not os.path.isdir(path):
        raise FileNotFoundError(path)
    if not readonly:
        os.makedirs(path, exist_ok=True)
    kwargs = {
        "subdir": True,
        "readonly": bool(readonly),
        "lock": not readonly,
        "readahead": False,
        "max_readers": 256,
    }
    if map_size is not None:
        kwargs["map_size"] = int(map_size)
    return lmdb.open(path, **kwargs)


def _initial_metadata(metadata, expected_keys):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    result = dict(metadata)
    result.update(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "incomplete",
            "sample_count": len(expected_keys),
            "ordered_key_sha256": hash_ordered_keys(expected_keys),
        }
    )
    return result


def _validate_metadata_match(actual, expected):
    immutable = set(expected).union(
        {
            "schema_version",
            "sample_count",
            "ordered_key_sha256",
        }
    )
    mismatch = sorted(
        field
        for field in immutable
        if actual.get(field) != expected.get(field)
    )
    if mismatch:
        raise ValueError(
            "cache provenance does not match: " + ", ".join(mismatch)
        )


def _as_finite_array(value, name, dtype):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _validate_and_normalize_record(record, expected_key=None):
    if not isinstance(record, dict):
        raise ValueError("cache record must be a mapping")
    try:
        key = str(record["key"])
    except (KeyError, TypeError) as error:
        raise ValueError("cache record is missing key") from error
    if not key:
        raise ValueError("cache record key must be nonempty")
    if expected_key is not None and key != expected_key:
        raise ValueError(f"cache record key mismatch: {key} != {expected_key}")

    missing = sorted(RECORD_FIELDS.difference(record))
    extra = sorted(set(record).difference(RECORD_FIELDS.union({"key"})))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("invalid cache fields: " + "; ".join(details))

    normalized = {"key": key}
    for name in FP16_FEATURE_FIELDS:
        normalized[name] = _as_finite_array(record[name], name, np.float16)
    for name in FP32_FIELDS:
        normalized[name] = _as_finite_array(record[name], name, np.float32)

    for name, shape in FIXED_FEATURE_SHAPES.items():
        if normalized[name].shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {normalized[name].shape}"
            )
    for name, feature_dim in NODE_FEATURE_DIMS.items():
        array = normalized[name]
        minimum_count = 0 if name == "interface_protein_nodes" else 1
        if (
            array.ndim != 2
            or array.shape[0] < minimum_count
            or array.shape[1] != feature_dim
        ):
            qualifier = "N >= 0" if minimum_count == 0 else "N > 0"
            raise ValueError(
                f"{name} must have shape [N,{feature_dim}] with {qualifier}"
            )
        coordinate_name = COORDINATE_FIELDS[name]
        coordinates = normalized[coordinate_name]
        if coordinates.shape != (array.shape[0], 3):
            raise ValueError(
                f"{coordinate_name} must align with {name}"
            )
    for name in SCALAR_FIELDS:
        if normalized[name].shape != (1,):
            raise ValueError(f"{name} must have shape (1,)")
    return normalized


def read_cache_metadata(path):
    environment = _open_environment(path, readonly=True)
    try:
        with environment.begin() as transaction:
            payload = transaction.get(_METADATA_KEY)
        if payload is None:
            raise RuntimeError("cache metadata is missing")
        metadata = _deserialize(payload)
    finally:
        environment.close()
    if not isinstance(metadata, dict):
        raise RuntimeError("cache metadata must be a mapping")
    return metadata


def cache_matches_provenance(metadata, expected):
    if not isinstance(metadata, dict) or metadata.get("status") != "complete":
        return False
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    if not isinstance(expected, dict):
        return False
    return all(metadata.get(name) == value for name, value in expected.items())


class HybridFeatureCacheWriter:
    """Write a single provenance-bound cache, safely resumable by key."""

    def __init__(
        self,
        path,
        expected_keys,
        metadata,
        resume=False,
        map_size=32 * 1024 ** 3,
    ):
        self.path = Path(path).expanduser().resolve()
        self.keys = _as_unique_keys(expected_keys)
        self._expected_key_set = set(self.keys)
        self._expected_metadata = _initial_metadata(metadata, self.keys)
        existing = self.path.exists() and any(self.path.iterdir())
        if existing and not resume:
            raise FileExistsError(
                f"cache already exists; pass resume=True: {self.path}"
            )
        self.environment = _open_environment(
            self.path, readonly=False, map_size=map_size
        )
        try:
            with self.environment.begin(write=True) as transaction:
                raw_metadata = transaction.get(_METADATA_KEY)
                raw_keys = transaction.get(_KEYS_KEY)
                if raw_metadata is None:
                    transaction.put(
                        _METADATA_KEY, _serialize(self._expected_metadata)
                    )
                    transaction.put(_KEYS_KEY, _serialize(self.keys))
                    self.metadata = dict(self._expected_metadata)
                else:
                    self.metadata = _deserialize(raw_metadata)
                    stored_keys = _deserialize(raw_keys) if raw_keys else None
                    if stored_keys != self.keys:
                        raise ValueError("cache key order does not match")
                    _validate_metadata_match(
                        self.metadata, self._expected_metadata
                    )
                    if self.metadata.get("status") not in {
                        "incomplete",
                        "complete",
                    }:
                        raise RuntimeError("cache status is invalid")
        except BaseException:
            self.environment.close()
            raise
        self.missing_keys = self._find_missing_keys()

    def _find_missing_keys(self):
        with self.environment.begin() as transaction:
            return [
                key
                for key in self.keys
                if transaction.get(_record_key(key)) is None
            ]

    def write(self, record):
        normalized = _validate_and_normalize_record(record)
        key = normalized["key"]
        if key not in self._expected_key_set:
            raise ValueError(f"cache key is not expected: {key}")
        with self.environment.begin(write=True) as transaction:
            if transaction.get(_record_key(key)) is not None:
                raise ValueError(f"duplicate cache record: {key}")
            transaction.put(_record_key(key), _serialize(normalized))
        self.missing_keys = [item for item in self.missing_keys if item != key]

    def finalize(self):
        missing = self._find_missing_keys()
        if missing:
            raise RuntimeError(
                "cache is incomplete; missing keys: " + ", ".join(missing[:5])
            )
        completed = dict(self._expected_metadata)
        completed["status"] = "complete"
        with self.environment.begin(write=True) as transaction:
            transaction.put(_METADATA_KEY, _serialize(completed))
        self.metadata = completed
        self.missing_keys = []
        atomic_write_json(_metadata_sidecar(self.path), completed)
        return completed

    def close(self):
        if getattr(self, "environment", None) is not None:
            self.environment.close()
            self.environment = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class CachedHybridFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.metadata = read_cache_metadata(self.path)
        if self.metadata.get("status") != "complete":
            raise RuntimeError("cache is not complete")
        self.environment = _open_environment(self.path, readonly=True)
        try:
            with self.environment.begin() as transaction:
                payload = transaction.get(_KEYS_KEY)
            if payload is None:
                raise RuntimeError("cache keys are missing")
            self.keys = _deserialize(payload)
        except BaseException:
            self.environment.close()
            raise
        if not isinstance(self.keys, list) or self.keys != _as_unique_keys(self.keys):
            raise RuntimeError("cache keys are invalid")
        if len(self.keys) != int(self.metadata.get("sample_count", -1)):
            raise RuntimeError("cache sample count does not match key list")
        if self.metadata.get("ordered_key_sha256") != hash_ordered_keys(self.keys):
            raise RuntimeError("cache key hash does not match key list")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        with self.environment.begin() as transaction:
            payload = transaction.get(_record_key(key))
        if payload is None:
            raise RuntimeError(f"cache record is missing: {key}")
        record = _validate_and_normalize_record(_deserialize(payload), key)
        result = {"key": key}
        for name in FP16_FEATURE_FIELDS:
            result[name] = torch.from_numpy(record[name].copy())
        for name in FP32_FIELDS:
            result[name] = torch.from_numpy(record[name].copy())
        return result

    def close(self):
        if getattr(self, "environment", None) is not None:
            self.environment.close()
            self.environment = None

    def __del__(self):
        self.close()


def _pad_variable_field(batch, name):
    values = [sample[name] for sample in batch]
    length = max(int(value.shape[0]) for value in values)
    feature_dim = int(values[0].shape[1])
    padded = values[0].new_zeros((len(values), length, feature_dim))
    mask = torch.zeros((len(values), length), dtype=torch.bool)
    for index, value in enumerate(values):
        padded[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return padded, mask


def collate_cached_hybrid_features(batch):
    if not batch:
        raise ValueError("cached feature batch must be nonempty")
    result = {"keys": [sample["key"] for sample in batch]}
    if len(result["keys"]) != len(set(result["keys"])):
        raise ValueError("cached feature batch contains duplicate keys")
    for name in FIXED_FEATURE_SHAPES:
        result[name] = torch.stack([sample[name] for sample in batch], dim=0)
    for node_name, coord_name in COORDINATE_FIELDS.items():
        result[node_name], result[node_name.replace("_nodes", "_mask")] = (
            _pad_variable_field(batch, node_name)
        )
        result[coord_name], coord_mask = _pad_variable_field(batch, coord_name)
        if not torch.equal(
            result[node_name.replace("_nodes", "_mask")], coord_mask
        ):
            raise RuntimeError(f"{coord_name} padding does not match {node_name}")
    for name in SCALAR_FIELDS:
        result[name] = torch.stack([sample[name] for sample in batch], dim=0)
    return result

