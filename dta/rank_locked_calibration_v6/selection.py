"""Validation gates and tamper-evident package for calibration v6."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cmp_to_key
import hashlib
from io import BytesIO
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType

import numpy as np
import torch

from utils.experiment_integrity import sha256_file

from . import CANDIDATE_NAMES, VALIDATION_GATE


_METRIC_NAMES = ("CI", "MAE", "RMSE", "Pearson", "Spearman")
_CANDIDATE_TYPES = frozenset(("identity", "analytic", "conditional", "ensemble"))
_CORE_FILES = (
    "selection.json",
    "candidate_manifest.json",
    "cache_identities.json",
    "normalization.json",
    "fold_map.npz",
    "selected_model.pt",
    "valid_predictions.npz",
    "rollback.json",
)
PACKAGE_FILES = _CORE_FILES + ("selection_seal.json",)
_ROLLBACK_FIELDS = frozenset(
    (
        "schema_version",
        "promotion_status",
        "v5_validation_package",
        "v5_selection_seal_sha256",
        "v5_official_test_result",
        "v5_official_test_result_sha256",
    )
)


def _canonical_json_bytes(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("payload must be canonical JSON compatible") from error


def _plain_json(value):
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(name + " must be finite")
    return float(value)


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def _prediction_hash(predictions):
    digest = hashlib.sha256()
    digest.update(str(predictions.shape).encode("ascii"))
    digest.update(predictions.tobytes(order="C"))
    return digest.hexdigest()


def _expected_type(name):
    if name == "identity":
        return "identity"
    if name.startswith("affine_") or name.startswith("spline_"):
        return "analytic"
    if name == "conditional_ensemble":
        return "ensemble"
    return "conditional"


@dataclass(frozen=True)
class ValidationCandidate:
    name: str
    candidate_type: str
    validation_metrics: Mapping
    parameter_count: int
    predictions: np.ndarray
    model_descriptor: Mapping
    oof_summary: Mapping
    status: str

    def __post_init__(self):
        if self.name not in CANDIDATE_NAMES:
            raise ValueError("candidate name is outside the frozen inventory")
        if (
            self.candidate_type not in _CANDIDATE_TYPES
            or self.candidate_type != _expected_type(self.name)
        ):
            raise ValueError("candidate type does not match name")
        if not isinstance(self.validation_metrics, Mapping) or set(
            self.validation_metrics
        ) != set(_METRIC_NAMES):
            raise ValueError("validation metrics fields do not match")
        metrics = {
            name: _finite(self.validation_metrics[name], name)
            for name in _METRIC_NAMES
        }
        if (
            isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, Integral)
            or int(self.parameter_count) < 0
        ):
            raise ValueError("parameter count must be a nonnegative integer")
        if not isinstance(self.predictions, np.ndarray) or self.predictions.dtype != np.float64:
            raise ValueError("candidate predictions must be a float64 array")
        predictions = np.asarray(self.predictions)
        if predictions.ndim != 1 or predictions.size == 0:
            raise ValueError("candidate predictions must be a nonempty vector")
        if not np.isfinite(predictions).all():
            raise ValueError("candidate predictions must be finite")
        if not isinstance(self.model_descriptor, Mapping):
            raise ValueError("model descriptor must be a mapping")
        if not isinstance(self.oof_summary, Mapping):
            raise ValueError("OOF summary must be a mapping")
        if self.status not in (
            "complete",
            "failed_nonfinite",
            "failed_missing_seed",
        ):
            raise ValueError("candidate status is invalid")
        predictions = predictions.copy()
        predictions.setflags(write=False)
        object.__setattr__(self, "validation_metrics", MappingProxyType(metrics))
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(
            self, "model_descriptor", MappingProxyType(_plain_json(self.model_descriptor))
        )
        object.__setattr__(
            self, "oof_summary", MappingProxyType(_plain_json(self.oof_summary))
        )

    @property
    def metrics(self):
        return self.validation_metrics

    @property
    def eligible(self):
        return (
            self.status == "complete"
            and self.validation_metrics["CI"] >= VALIDATION_GATE["CI"]
            and self.validation_metrics["RMSE"] <= VALIDATION_GATE["RMSE"]
            and self.validation_metrics["MAE"] <= VALIDATION_GATE["MAE"]
        )

    def to_dict(self):
        return {
            "name": self.name,
            "candidate_type": self.candidate_type,
            "validation_metrics": dict(self.validation_metrics),
            "parameter_count": self.parameter_count,
            "status": self.status,
            "eligible": self.eligible,
            "prediction_sha256": _prediction_hash(self.predictions),
            "model_descriptor": dict(self.model_descriptor),
            "oof_summary": dict(self.oof_summary),
        }


def validate_candidate_inventory(candidates):
    try:
        records = tuple(candidates)
    except TypeError as error:
        raise ValueError("candidate inventory must be iterable") from error
    if any(not isinstance(record, ValidationCandidate) for record in records):
        raise ValueError("candidate inventory contains an invalid record")
    names = tuple(record.name for record in records)
    if (
        len(names) != len(CANDIDATE_NAMES)
        or len(set(names)) != len(names)
        or set(names) != set(CANDIDATE_NAMES)
    ):
        raise ValueError("candidate inventory does not match frozen names")
    by_name = {record.name: record for record in records}
    return tuple(by_name[name] for name in CANDIDATE_NAMES)


def _candidate_compare(left, right):
    left_rmse = left.validation_metrics["RMSE"]
    right_rmse = right.validation_metrics["RMSE"]
    if abs(left_rmse - right_rmse) + 1e-12 >= 0.005:
        if left_rmse != right_rmse:
            return -1 if left_rmse < right_rmse else 1
    else:
        left_mae = left.validation_metrics["MAE"]
        right_mae = right.validation_metrics["MAE"]
        if left_mae != right_mae:
            return -1 if left_mae < right_mae else 1
    if left.parameter_count != right.parameter_count:
        return -1 if left.parameter_count < right.parameter_count else 1
    if left.name == right.name:
        return 0
    return -1 if left.name < right.name else 1


def select_candidate(candidates):
    try:
        records = tuple(candidates)
    except TypeError as error:
        raise ValueError("candidates must be iterable") from error
    if not records or any(not isinstance(value, ValidationCandidate) for value in records):
        raise ValueError("candidates must contain validation records")
    names = tuple(value.name for value in records)
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    eligible = tuple(value for value in records if value.eligible)
    if not eligible:
        return None
    return sorted(eligible, key=cmp_to_key(_candidate_compare))[0]


def _atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _npz_bytes(**values):
    buffer = BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def _torch_bytes(value):
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load_canonical_json_bytes(value, name):
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(name + " is malformed") from error
    if value != _canonical_json_bytes(decoded):
        raise ValueError(name + " is not canonical")
    return decoded


def _validated_rollback(value, *, verify_external):
    if not isinstance(value, Mapping) or set(value) != _ROLLBACK_FIELDS:
        raise ValueError("rollback fields do not match")
    result = _plain_json(dict(value))
    if result["schema_version"] != 1 or result["promotion_status"] != "not_promoted":
        raise ValueError("rollback policy does not match")
    package = Path(result["v5_validation_package"])
    official = Path(result["v5_official_test_result"])
    if not package.is_absolute() or not official.is_absolute():
        raise ValueError("rollback paths must be absolute")
    _sha256(result["v5_selection_seal_sha256"], "v5 selection seal hash")
    _sha256(result["v5_official_test_result_sha256"], "v5 result hash")
    if verify_external:
        seal = package / "selection_seal.json"
        if (
            not package.is_dir()
            or seal.is_symlink()
            or not seal.is_file()
            or official.is_symlink()
            or not official.is_file()
            or sha256_file(seal) != result["v5_selection_seal_sha256"]
            or sha256_file(official) != result["v5_official_test_result_sha256"]
        ):
            raise ValueError("rollback artifacts do not match")
    return result


def _validated_fold_map(value):
    if not isinstance(value, Mapping) or set(value) != {"keys", "fold_ids"}:
        raise ValueError("fold map fields do not match")
    keys = np.asarray(value["keys"])
    folds = np.asarray(value["fold_ids"])
    if keys.ndim != 1 or keys.size == 0 or folds.shape != keys.shape:
        raise ValueError("fold map arrays do not align")
    keys = np.asarray([str(item) for item in keys], dtype=np.str_)
    folds = np.asarray(folds, dtype=np.int64)
    if len(set(keys.tolist())) != keys.shape[0] or np.any(folds < 0):
        raise ValueError("fold map arrays are invalid")
    return keys, folds


def _validated_prediction_arrays(keys, labels, predictions, expected_predictions):
    keys = np.asarray(keys)
    labels = np.asarray(labels)
    if not isinstance(predictions, np.ndarray) or predictions.dtype != np.float64:
        raise ValueError("valid predictions must be float64")
    predictions = np.asarray(predictions)
    if labels.dtype != np.float64:
        labels = labels.astype(np.float64)
    if keys.ndim != 1 or labels.shape != keys.shape or predictions.shape != keys.shape:
        raise ValueError("valid prediction arrays do not align")
    if keys.size == 0 or len(set(str(value) for value in keys)) != keys.size:
        raise ValueError("valid keys are invalid")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("valid prediction arrays must be finite")
    if not np.array_equal(predictions, expected_predictions):
        raise ValueError("selected valid predictions do not match candidate")
    return (
        np.asarray([str(value) for value in keys], dtype=np.str_),
        np.asarray(labels, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
    )


def _preflight_package(package):
    source = Path(package).expanduser()
    if source.is_symlink():
        raise ValueError("validation package must not be a symlink")
    package = source.resolve()
    if package.exists():
        if not package.is_dir() or any(package.iterdir()):
            raise ValueError("validation package output is not empty")
    else:
        package.mkdir(parents=True)
    return package


def seal_validation_package(
    package,
    *,
    selected,
    candidates,
    cache_identities,
    normalization,
    fold_map,
    selected_model,
    valid_keys,
    valid_labels,
    valid_predictions,
    rollback,
):
    records = validate_candidate_inventory(candidates)
    if not isinstance(selected, ValidationCandidate):
        raise ValueError("selected candidate is invalid")
    expected = select_candidate(records)
    if expected is None or selected.name != expected.name:
        raise ValueError("selected candidate does not match validation policy")
    if not isinstance(cache_identities, Mapping) or set(cache_identities) != {"train", "valid"}:
        raise ValueError("cache identities must contain train and valid")
    if not isinstance(normalization, Mapping):
        raise ValueError("normalization must be a mapping")
    if not isinstance(selected_model, Mapping):
        raise ValueError("selected model must be a mapping")
    if selected_model.get("name") != selected.name:
        raise ValueError("selected model name does not match")
    keys, folds = _validated_fold_map(fold_map)
    valid_keys, valid_labels, valid_predictions = _validated_prediction_arrays(
        valid_keys, valid_labels, valid_predictions, selected.predictions
    )
    rollback = _validated_rollback(rollback, verify_external=True)
    package = _preflight_package(package)

    selection = {
        "schema_version": 1,
        "status": "passed",
        "selected_name": selected.name,
        "selected_type": selected.candidate_type,
        "selected_validation_metrics": dict(selected.validation_metrics),
        "validation_gate": dict(VALIDATION_GATE),
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "candidate_names": list(CANDIDATE_NAMES),
        "records": [record.to_dict() for record in records],
    }
    payloads = {
        "selection.json": _canonical_json_bytes(selection),
        "candidate_manifest.json": _canonical_json_bytes(manifest),
        "cache_identities.json": _canonical_json_bytes(dict(cache_identities)),
        "normalization.json": _canonical_json_bytes(dict(normalization)),
        "fold_map.npz": _npz_bytes(keys=keys, fold_ids=folds),
        "selected_model.pt": _torch_bytes(dict(selected_model)),
        "valid_predictions.npz": _npz_bytes(
            keys=valid_keys,
            labels=valid_labels,
            predictions=valid_predictions,
            selected_name=np.asarray(selected.name, dtype=np.str_),
        ),
        "rollback.json": _canonical_json_bytes(rollback),
    }
    for name in _CORE_FILES:
        _atomic_bytes(package / name, payloads[name])
    hashes = {name: sha256_file(package / name) for name in _CORE_FILES}
    seal = {
        "schema_version": 1,
        "status": "sealed",
        "files": hashes,
        "files_sha256": hashlib.sha256(_canonical_json_bytes(hashes)).hexdigest(),
    }
    _atomic_bytes(package / "selection_seal.json", _canonical_json_bytes(seal))
    for name in PACKAGE_FILES:
        os.chmod(package / name, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    verify_validation_package(package)
    return seal


def _flat_inventory(package):
    source = Path(package).expanduser()
    if source.is_symlink():
        raise ValueError("validation package must not be a symlink")
    package = source.resolve()
    if not package.is_dir():
        raise ValueError("validation package is unavailable")
    entries = tuple(package.iterdir())
    if {path.name for path in entries} != set(PACKAGE_FILES) or len(entries) != len(PACKAGE_FILES):
        raise ValueError("validation package inventory does not match")
    for path in entries:
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("validation package contains a symlink")
        if not stat.S_ISREG(mode):
            raise ValueError("validation package entries must be regular files")
        if mode & 0o222:
            raise ValueError("validation package contains a writable file")
    return package


def verify_validation_package(package):
    package = _flat_inventory(package)
    seal_bytes = (package / "selection_seal.json").read_bytes()
    seal = _load_canonical_json_bytes(seal_bytes, "selection seal")
    if (
        not isinstance(seal, dict)
        or set(seal) != {"schema_version", "status", "files", "files_sha256"}
        or seal["schema_version"] != 1
        or seal["status"] != "sealed"
        or not isinstance(seal["files"], dict)
        or set(seal["files"]) != set(_CORE_FILES)
    ):
        raise ValueError("selection seal is malformed")
    for digest in seal["files"].values():
        _sha256(digest, "sealed file hash")
    expected_map_hash = hashlib.sha256(_canonical_json_bytes(seal["files"])).hexdigest()
    if seal["files_sha256"] != expected_map_hash:
        raise ValueError("selection seal hash map does not match")
    for name in _CORE_FILES:
        if sha256_file(package / name) != seal["files"][name]:
            raise ValueError("sealed file hash does not match")

    selection = _load_canonical_json_bytes(
        (package / "selection.json").read_bytes(), "selection"
    )
    manifest = _load_canonical_json_bytes(
        (package / "candidate_manifest.json").read_bytes(), "candidate manifest"
    )
    _load_canonical_json_bytes(
        (package / "cache_identities.json").read_bytes(), "cache identities"
    )
    _load_canonical_json_bytes(
        (package / "normalization.json").read_bytes(), "normalization"
    )
    rollback = _load_canonical_json_bytes(
        (package / "rollback.json").read_bytes(), "rollback"
    )
    if (
        not isinstance(selection, dict)
        or selection.get("schema_version") != 1
        or selection.get("status") != "passed"
        or set(selection) != {
            "schema_version",
            "status",
            "selected_name",
            "selected_type",
            "selected_validation_metrics",
            "validation_gate",
        }
        or selection.get("validation_gate") != dict(VALIDATION_GATE)
    ):
        raise ValueError("selection payload does not match")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("candidate_names") != list(CANDIDATE_NAMES)
        or not isinstance(manifest.get("records"), list)
        or tuple(record.get("name") for record in manifest["records"]) != CANDIDATE_NAMES
    ):
        raise ValueError("candidate inventory does not match")
    selected_records = [
        record for record in manifest["records"]
        if record.get("name") == selection["selected_name"]
    ]
    if (
        len(selected_records) != 1
        or not selected_records[0].get("eligible")
        or selected_records[0].get("candidate_type") != selection["selected_type"]
        or selected_records[0].get("validation_metrics")
        != selection["selected_validation_metrics"]
    ):
        raise ValueError("selected candidate does not match manifest")
    with np.load(str(package / "fold_map.npz"), allow_pickle=False) as archive:
        if set(archive.files) != {"keys", "fold_ids"}:
            raise ValueError("fold map fields do not match")
        _validated_fold_map({"keys": archive["keys"], "fold_ids": archive["fold_ids"]})
    with np.load(str(package / "valid_predictions.npz"), allow_pickle=False) as archive:
        if set(archive.files) != {"keys", "labels", "predictions", "selected_name"}:
            raise ValueError("valid prediction fields do not match")
        predictions = archive["predictions"]
        if predictions.dtype != np.float64 or str(archive["selected_name"].item()) != selection["selected_name"]:
            raise ValueError("valid predictions do not match selection")
        keys = archive["keys"]
        labels = archive["labels"]
        if keys.ndim != 1 or labels.shape != keys.shape or predictions.shape != keys.shape:
            raise ValueError("valid prediction arrays do not align")
        if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
            raise ValueError("valid prediction arrays must be finite")
    try:
        model = torch.load(str(package / "selected_model.pt"), map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(str(package / "selected_model.pt"), map_location="cpu")
    if not isinstance(model, Mapping) or model.get("name") != selection["selected_name"]:
        raise ValueError("selected model does not match selection")
    _validated_rollback(rollback, verify_external=True)
    return selection


__all__ = [
    "PACKAGE_FILES",
    "ValidationCandidate",
    "seal_validation_package",
    "select_candidate",
    "validate_candidate_inventory",
    "verify_validation_package",
]
