"""Validation-only candidate policy and tamper-evident package seal for v5."""

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import tempfile
from types import MappingProxyType

import numpy as np
import torch

from dta.anchored_expert_fusion_v5.static_fusion import rank_percentiles
from dta.rank_fusion_v4.metrics import affinity_metrics


FORMAL_BUDGETS = (0.10, 0.25, 0.50)
FORMAL_SEEDS = (1, 7, 21, 42, 87)
EXPERT_NAMES = ("upper_bound", "boltz_pairwise", "three_dprot")
PACKAGE_FILES = (
    "selection.json",
    "fold_map.npz",
    "raw_cross_fit_report.json",
    "rank_cross_fit_report.json",
    "static_specs.json",
    "candidate_manifest.json",
    "selected_references.json",
    "cache_identities.json",
    "selection_seal.json",
)
_CORE_FILES = frozenset(PACKAGE_FILES[:-1])
_METRICS = ("CI", "MAE", "RMSE", "Pearson", "Spearman")
_TYPES = frozenset((
    "expert", "static", "residual", "raw_ensemble", "rank_ensemble"
))
_STATIC_LINEAGE_KEYS = frozenset((
    "train_manifest",
    "train_ordered_keys",
    "valid_manifest",
    "valid_ordered_keys",
    "cross_fit_report",
    "static_spec_sha256",
))
_ENSEMBLE_LINEAGE_KEYS = _STATIC_LINEAGE_KEYS | frozenset(("member_set",))
_EXPERT_LINEAGE_KEYS = frozenset((
    "valid_manifest",
    "valid_ordered_keys",
    "prediction_source",
))
_HEX = frozenset("0123456789abcdef")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " must be finite") from error
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def _safe_name(value):
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError("candidate name is unsafe")
    return value


def validate_relative_path(value, name="path"):
    """Return one canonical, output-root-relative POSIX path."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(name + " must be a nonempty relative path")
    if "\x00" in value or "\\" in value:
        raise ValueError(name + " is unsafe")
    if value.startswith("//") or _URI_SCHEME.match(value):
        raise ValueError(name + " is unsafe")
    windows = PureWindowsPath(value)
    if windows.drive or windows.root:
        raise ValueError(name + " is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(name + " is unsafe")
    if path.as_posix() != value:
        raise ValueError(name + " is not canonical")
    return value


def _jsonable(value):
    if isinstance(value, MappingProxyType):
        return _jsonable(dict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value):
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_verified_bytes(path, *, expected_sha256=None, expected_size=None,
                        name="artifact"):
    """Read, size-check, and hash the bytes from one opened regular file."""
    if expected_size is not None and (
        isinstance(expected_size, bool) or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ValueError(name + " expected size is invalid")
    if expected_sha256 is not None:
        expected_sha256 = _sha256(expected_sha256, name + " hash")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as error:
        raise ValueError(name + " is not safely readable") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(name + " must be a regular non-symlink file")
        initial_size = int(info.st_size)
        if expected_size is not None and initial_size != expected_size:
            raise ValueError(name + " size does not match")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        value = b"".join(chunks)
        final_size = int(os.fstat(descriptor).st_size)
        if len(value) != initial_size or final_size != initial_size:
            raise ValueError(name + " changed while being read")
    except OSError as error:
        raise ValueError(name + " could not be read") from error
    finally:
        os.close(descriptor)
    if expected_size is not None and len(value) != expected_size:
        raise ValueError(name + " size does not match")
    if expected_sha256 is not None and _hash_bytes(value) != expected_sha256:
        raise ValueError(name + " hash does not match")
    return value


def load_verified_npz_bytes(value, *, name="NPZ artifact"):
    try:
        with np.load(BytesIO(bytes(value)), allow_pickle=False) as archive:
            return {
                key: np.asarray(archive[key]).copy()
                for key in archive.files
            }
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(name + " is invalid") from error


def load_verified_torch_bytes(value, *, map_location="cpu",
                              name="checkpoint"):
    try:
        return torch.load(
            BytesIO(bytes(value)), map_location=map_location,
            weights_only=True,
        )
    except TypeError as error:
        raise ValueError(name + " loader does not support weights_only") from error
    except Exception as error:
        raise ValueError(name + " is invalid or unsafe") from error


def load_verified_npz(path, *, expected_sha256=None, expected_size=None,
                      name="NPZ artifact"):
    value = read_verified_bytes(
        path, expected_sha256=expected_sha256, expected_size=expected_size,
        name=name,
    )
    return load_verified_npz_bytes(value, name=name)


def load_verified_torch(path, *, expected_sha256=None, expected_size=None,
                        map_location="cpu", name="checkpoint"):
    value = read_verified_bytes(
        path, expected_sha256=expected_sha256, expected_size=expected_size,
        name=name,
    )
    return load_verified_torch_bytes(
        value, map_location=map_location, name=name
    )


def _hash_file(path):
    return _hash_bytes(read_verified_bytes(path, name="file"))


def _atomic_bytes(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + "-",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    _atomic_bytes(path, _canonical_json(value) + b"\n")


def _readonly_predictions(value):
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError) as error:
        raise ValueError("candidate predictions must be finite") from error
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("candidate predictions must be finite and nonempty")
    result.setflags(write=False)
    return result


def _hash_mapping(value, name, require_nonempty=True):
    try:
        items = dict(value or {})
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be a hash mapping") from error
    if require_nonempty and not items:
        raise ValueError(name + " must be nonempty")
    normalized = {}
    for key, digest in items.items():
        if not isinstance(key, str) or not key:
            raise ValueError(name + " keys must be nonempty")
        normalized[key] = _sha256(digest, name + " " + key)
    return MappingProxyType(normalized)


def _budget(value, required):
    if value is None and not required:
        return None
    supplied = _finite(value, "budget")
    for expected in FORMAL_BUDGETS:
        if supplied == expected:
            return expected
    raise ValueError("budget must be one of the formal budgets")


def _seed(value, required):
    if value is None and not required:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) not in FORMAL_SEEDS
    ):
        raise ValueError("seed must be one of the formal seeds")
    return int(value)


def _validate_lineage_shape(candidate_type, lineage):
    if candidate_type == "expert":
        return
    if candidate_type in ("static", "residual"):
        expected = _STATIC_LINEAGE_KEYS
    else:
        expected = _ENSEMBLE_LINEAGE_KEYS
    supplied = set(lineage)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "candidate lineage fields do not match: " + "; ".join(details)
        )


def _same_static_lineage(left, right):
    return all(
        left.lineage_hashes[key] == right.lineage_hashes[key]
        for key in _STATIC_LINEAGE_KEYS
    )


@dataclass(frozen=True)
class CandidateRecord:
    """Immutable validation candidate and complete artifact provenance."""

    name: str
    type: str
    family: str
    prediction_path: str
    prediction_sha256: str
    validation_metrics: dict
    parent_static_name: object = None
    checkpoint_path: object = None
    checkpoint_sha256: object = None
    lineage_hashes: dict = None
    member_hashes: dict = None
    budget: object = None
    seed: object = None
    members: tuple = ()
    ensemble_rule: object = None
    predictions: object = None

    def __post_init__(self):
        object.__setattr__(self, "name", _safe_name(self.name))
        if self.type not in _TYPES:
            raise ValueError("candidate type is invalid")
        expected_family = "expert" if self.type == "expert" else ("raw", "rank")
        if self.family not in (
            (expected_family,) if isinstance(expected_family, str)
            else expected_family
        ):
            raise ValueError("candidate family is invalid")
        object.__setattr__(
            self,
            "prediction_path",
            validate_relative_path(self.prediction_path, "prediction_path"),
        )
        object.__setattr__(
            self,
            "prediction_sha256",
            _sha256(self.prediction_sha256, "prediction hash"),
        )
        try:
            supplied_metrics = dict(self.validation_metrics)
        except (TypeError, ValueError) as error:
            raise ValueError("validation metrics must be a dictionary") from error
        if set(supplied_metrics) != set(_METRICS):
            raise ValueError("validation metrics fields do not match")
        normalized_metrics = {
            name: _finite(supplied_metrics[name], "candidate " + name)
            for name in _METRICS
        }
        object.__setattr__(
            self, "validation_metrics", MappingProxyType(normalized_metrics)
        )
        lineage = _hash_mapping(self.lineage_hashes, "lineage hashes")
        _validate_lineage_shape(self.type, lineage)
        object.__setattr__(self, "lineage_hashes", lineage)
        members = tuple(self.members or ())
        if any(not isinstance(item, str) or not item for item in members):
            raise ValueError("members must contain candidate names")
        object.__setattr__(self, "members", members)
        object.__setattr__(
            self,
            "member_hashes",
            _hash_mapping(
                self.member_hashes,
                "member hashes",
                require_nonempty=self.type in ("raw_ensemble", "rank_ensemble"),
            ),
        )
        object.__setattr__(
            self, "predictions", _readonly_predictions(self.predictions)
        )

        learned = self.type in ("residual", "raw_ensemble", "rank_ensemble")
        object.__setattr__(self, "budget", _budget(self.budget, learned))
        object.__setattr__(
            self, "seed", _seed(self.seed, self.type == "residual")
        )
        if self.type == "residual":
            if (
                not isinstance(self.parent_static_name, str)
                or self.parent_static_name != "static_" + self.family
            ):
                raise ValueError("residual parent static is invalid")
            object.__setattr__(
                self,
                "checkpoint_path",
                validate_relative_path(
                    self.checkpoint_path, "checkpoint_path"
                ),
            )
            object.__setattr__(
                self,
                "checkpoint_sha256",
                _sha256(self.checkpoint_sha256, "checkpoint hash"),
            )
            if members or self.member_hashes or self.ensemble_rule is not None:
                raise ValueError("residual ensemble fields must be empty")
        elif self.type in ("raw_ensemble", "rank_ensemble"):
            if (
                not isinstance(self.parent_static_name, str)
                or self.parent_static_name != "static_" + self.family
            ):
                raise ValueError("ensemble parent static is invalid")
            if self.checkpoint_path is not None or self.checkpoint_sha256 is not None:
                raise ValueError("ensemble checkpoint must be null")
            if self.seed is not None:
                raise ValueError("ensemble seed must be null")
            if len(members) not in (2, 3) or len(set(members)) != len(members):
                raise ValueError("ensemble must contain two or three unique members")
            if set(self.member_hashes) != set(members):
                raise ValueError("ensemble member hashes do not match members")
            expected_rule = (
                "arithmetic_mean"
                if self.type == "raw_ensemble"
                else "stable_average_percentile_mean"
            )
            if self.ensemble_rule != expected_rule:
                raise ValueError("ensemble rule is invalid")
        else:
            if self.parent_static_name is not None:
                raise ValueError("candidate parent static must be null")
            if self.checkpoint_path is not None or self.checkpoint_sha256 is not None:
                raise ValueError("candidate checkpoint must be null")
            if self.budget is not None or self.seed is not None:
                raise ValueError("candidate budget and seed must be null")
            if members or self.member_hashes or self.ensemble_rule is not None:
                raise ValueError("candidate ensemble fields must be empty")

    @property
    def candidate_type(self):
        return self.type

    @property
    def metrics(self):
        return self.validation_metrics

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "family": self.family,
            "prediction_path": self.prediction_path,
            "prediction_sha256": self.prediction_sha256,
            "validation_metrics": dict(self.validation_metrics),
            "parent_static_name": self.parent_static_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "lineage_hashes": dict(self.lineage_hashes),
            "member_hashes": dict(self.member_hashes),
            "budget": self.budget,
            "seed": self.seed,
            "members": list(self.members),
            "ensemble_rule": self.ensemble_rule,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("candidate record must be a dictionary")
        required = {
            "name", "type", "family", "prediction_path",
            "prediction_sha256", "validation_metrics",
            "parent_static_name", "checkpoint_path", "checkpoint_sha256",
            "lineage_hashes", "member_hashes", "budget", "seed",
            "members", "ensemble_rule",
        }
        if set(value) != required:
            raise ValueError("candidate record fields do not match")
        return cls(**value)


ValidationCandidate = CandidateRecord


def _coerce_candidate(value):
    return value if isinstance(value, CandidateRecord) else CandidateRecord.from_dict(value)


def validation_candidate_key(candidate):
    record = _coerce_candidate(candidate)
    metrics = record.validation_metrics
    return (
        -metrics["CI"],
        -metrics["Spearman"],
        metrics["RMSE"],
        record.name,
    )


def _normalize_families(values):
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError("expected families must be raw and/or rank") from error
    if (
        not result
        or len(result) != len(set(result))
        or any(value not in ("raw", "rank") for value in result)
    ):
        raise ValueError("expected families must be unique raw and/or rank")
    return tuple(family for family in ("raw", "rank") if family in result)


def _normalize_budgets(values):
    try:
        supplied = tuple(values)
    except TypeError as error:
        raise ValueError("expected budgets are invalid") from error
    normalized = tuple(_budget(value, True) for value in supplied)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("expected budgets must be unique and nonempty")
    return tuple(value for value in FORMAL_BUDGETS if value in normalized)


def _normalize_seeds(values, minimum=2):
    try:
        supplied = tuple(values)
    except TypeError as error:
        raise ValueError("expected seeds are invalid") from error
    normalized = tuple(_seed(value, True) for value in supplied)
    if len(normalized) < minimum or len(normalized) != len(set(normalized)):
        raise ValueError("expected seeds are duplicated or insufficient")
    return tuple(value for value in FORMAL_SEEDS if value in normalized)


def _validate_residual_grid(residuals, families, budgets, seeds):
    records = tuple(_coerce_candidate(item) for item in residuals)
    if any(item.type != "residual" for item in records):
        raise ValueError("residual inventory contains a non-residual candidate")
    expected = {
        (family, budget, seed)
        for family in families
        for budget in budgets
        for seed in seeds
    }
    actual = [(item.family, item.budget, item.seed) for item in records]
    if len(actual) != len(set(actual)):
        raise ValueError("residual inventory contains duplicate combinations")
    if set(actual) != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise ValueError(
            "residual inventory combinations do not match: %s %s"
            % (missing, extra)
        )
    if len(records) != len(expected):
        raise ValueError("residual inventory count does not match")
    return records


def _parent_ci(candidate, statics):
    for static in statics:
        if static.name == candidate.parent_static_name:
            return static.validation_metrics["CI"]
    return None


def _eligible(candidate, statics):
    if candidate.type not in ("residual", "raw_ensemble", "rank_ensemble"):
        return True
    parent_ci = _parent_ci(candidate, statics)
    return (
        parent_ci is not None
        and candidate.validation_metrics["CI"] > parent_ci
    )


def select_validation_candidate(experts, statics, residuals, ensembles):
    experts = tuple(_coerce_candidate(item) for item in experts)
    statics = tuple(_coerce_candidate(item) for item in statics)
    residuals = tuple(_coerce_candidate(item) for item in residuals)
    ensembles = tuple(_coerce_candidate(item) for item in ensembles)
    if not statics or any(item.type != "static" for item in statics):
        raise ValueError("at least one static candidate is required")
    if any(item.type != "expert" for item in experts):
        raise ValueError("expert candidates are invalid")
    if any(item.type != "residual" for item in residuals):
        raise ValueError("residual candidates are invalid")
    if any(
        item.type not in ("raw_ensemble", "rank_ensemble")
        for item in ensembles
    ):
        raise ValueError("ensemble candidates are invalid")
    eligible = experts + statics + tuple(
        item for item in residuals + ensembles if _eligible(item, statics)
    )
    names = [item.name for item in eligible]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    return min(eligible, key=validation_candidate_key)


def _prediction_arrays(keys, labels, predictions):
    try:
        ordered_keys = tuple(str(key) for key in keys)
    except TypeError as error:
        raise ValueError("validation keys are invalid") from error
    label_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if (
        not ordered_keys
        or len(set(ordered_keys)) != len(ordered_keys)
        or len(ordered_keys) != label_array.size
        or label_array.shape != prediction_array.shape
        or not np.isfinite(label_array).all()
        or not np.isfinite(prediction_array).all()
    ):
        raise ValueError("prediction artifact arrays do not align")
    return ordered_keys, label_array, prediction_array


def persist_prediction_artifact(output_root, name, keys, labels, predictions):
    """Atomically persist one candidate's validation predictions."""
    safe_name = _safe_name(name)
    ordered_keys, label_array, prediction_array = _prediction_arrays(
        keys, labels, predictions
    )
    relative = "candidate_predictions/" + safe_name + ".npz"
    destination = Path(output_root).expanduser().resolve() / relative
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        keys=np.asarray(ordered_keys, dtype=np.str_),
        labels=label_array.astype(np.float32),
        predictions=prediction_array.astype(np.float32),
    )
    artifact_bytes = buffer.getvalue()
    _atomic_bytes(destination, artifact_bytes)
    return relative, _hash_bytes(artifact_bytes)


def _prediction_metrics(labels, predictions):
    label_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if (
        label_array.shape != prediction_array.shape
        or label_array.size == 0
        or not np.isfinite(label_array).all()
        or not np.isfinite(prediction_array).all()
    ):
        raise ValueError("ensemble labels and predictions do not align")
    result = affinity_metrics(label_array, prediction_array)
    return {name: _finite(result[name], "ensemble " + name) for name in _METRICS}


def build_residual_ensembles(
    residuals,
    labels,
    validation_keys,
    output_root,
    *,
    expected_families,
    expected_budgets=FORMAL_BUDGETS,
    expected_seeds=FORMAL_SEEDS,
    rank_calibrations=None,
):
    """Validate the exact residual grid and persist two ensembles per group."""
    families = _normalize_families(expected_families)
    budgets = _normalize_budgets(expected_budgets)
    seeds = _normalize_seeds(expected_seeds)
    records = _validate_residual_grid(
        residuals, families, budgets, seeds
    )
    result = []
    member_count = min(3, len(seeds))
    for family in families:
        for budget in budgets:
            group = tuple(
                item for item in records
                if item.family == family and item.budget == budget
            )
            if any(item.predictions is None for item in group):
                raise ValueError("residual predictions are required for ensembles")
            shapes = {item.predictions.shape for item in group}
            if len(shapes) != 1:
                raise ValueError("residual prediction shapes do not match")
            top = tuple(sorted(group, key=validation_candidate_key)[:member_count])
            member_predictions = tuple(
                np.asarray(item.predictions, dtype=np.float32)
                for item in top
            )
            stack = np.stack(member_predictions, axis=0)
            raw_prediction = stack.mean(axis=0)
            rank_prediction = np.stack(
                [rank_percentiles(value) for value in member_predictions],
                axis=0,
            ).mean(axis=0)
            calibrations = dict(rank_calibrations or {})
            slope, intercept = calibrations.get(family, (1.0, 0.0))
            slope = _finite(slope, family + " rank slope")
            intercept = _finite(intercept, family + " rank intercept")
            if family == "rank":
                rank_prediction = slope * rank_prediction + intercept
            budget_name = "%02d" % int(round(budget * 100))
            member_names = tuple(item.name for item in top)
            member_hashes = {
                item.name: item.prediction_sha256 for item in top
            }
            lineage = dict(top[0].lineage_hashes)
            lineage["member_set"] = _hash_bytes(
                _canonical_json(member_hashes)
            )
            for kind, rule, prediction in (
                ("raw_ensemble", "arithmetic_mean", raw_prediction),
                (
                    "rank_ensemble",
                    "stable_average_percentile_mean",
                    rank_prediction,
                ),
            ):
                suffix = "raw" if kind == "raw_ensemble" else "rank"
                name = "ensemble_%s_%s_%s" % (
                    family, budget_name, suffix
                )
                path, digest = persist_prediction_artifact(
                    output_root,
                    name,
                    validation_keys,
                    labels,
                    prediction,
                )
                result.append(CandidateRecord(
                    name=name,
                    type=kind,
                    family=family,
                    prediction_path=path,
                    prediction_sha256=digest,
                    validation_metrics=_prediction_metrics(
                        labels, prediction
                    ),
                    parent_static_name="static_" + family,
                    checkpoint_path=None,
                    checkpoint_sha256=None,
                    lineage_hashes=lineage,
                    member_hashes=member_hashes,
                    budget=budget,
                    seed=None,
                    members=member_names,
                    ensemble_rule=rule,
                    predictions=prediction,
                ))
    return tuple(result)


def build_candidate_inventory(
    experts,
    statics,
    residuals,
    ensembles,
    *,
    expected_families,
    expected_budgets=FORMAL_BUDGETS,
    expected_seeds=FORMAL_SEEDS,
):
    """Validate and return the exact expert/static/residual/ensemble inventory."""
    families = _normalize_families(expected_families)
    budgets = _normalize_budgets(expected_budgets)
    seeds = _normalize_seeds(expected_seeds)
    experts = tuple(_coerce_candidate(item) for item in experts)
    statics = tuple(_coerce_candidate(item) for item in statics)
    residuals = _validate_residual_grid(
        residuals, families, budgets, seeds
    )
    ensembles = tuple(_coerce_candidate(item) for item in ensembles)

    if (
        tuple(item.name for item in experts) != EXPERT_NAMES
        or any(item.type != "expert" for item in experts)
    ):
        raise ValueError("expert inventory does not match")
    expected_static_names = tuple("static_" + family for family in families)
    if (
        tuple(item.name for item in statics) != expected_static_names
        or any(item.type != "static" for item in statics)
    ):
        raise ValueError("static inventory does not match")
    expected_ensembles = {
        (family, budget, kind)
        for family in families
        for budget in budgets
        for kind in ("raw_ensemble", "rank_ensemble")
    }
    actual_ensembles = [
        (item.family, item.budget, item.type) for item in ensembles
    ]
    if (
        len(actual_ensembles) != len(set(actual_ensembles))
        or set(actual_ensembles) != expected_ensembles
    ):
        raise ValueError("ensemble inventory combinations do not match")
    statics_by_name = {item.name: item for item in statics}
    for child in residuals + ensembles:
        parent = statics_by_name.get(child.parent_static_name)
        if (
            parent is None
            or parent.family != child.family
            or not _same_static_lineage(parent, child)
        ):
            raise ValueError(child.name + " parent static lineage does not match")

    for ensemble in ensembles:
        group = tuple(
            item for item in residuals
            if item.family == ensemble.family
            and item.budget == ensemble.budget
        )
        expected_members = tuple(
            item.name
            for item in sorted(group, key=validation_candidate_key)[
                : min(3, len(seeds))
            ]
        )
        if ensemble.members != expected_members:
            raise ValueError("ensemble members are not the best residual seeds")
        if ensemble.parent_static_name != "static_" + ensemble.family:
            raise ValueError("ensemble parent static does not match")
        expected_member_set = _hash_bytes(
            _canonical_json(dict(ensemble.member_hashes))
        )
        if ensemble.lineage_hashes["member_set"] != expected_member_set:
            raise ValueError("ensemble member lineage hash does not match")
        for member_name in ensemble.members:
            member = next(
                (item for item in group if item.name == member_name), None
            )
            if (
                member is None
                or not _same_static_lineage(member, ensemble)
                or ensemble.member_hashes[member_name]
                != member.prediction_sha256
            ):
                raise ValueError(ensemble.name + " member lineage does not match")

    inventory = experts + statics + residuals + ensembles
    expected_count = 3 + len(families) * (
        1 + len(budgets) * len(seeds) + len(budgets) * 2
    )
    if len(inventory) != expected_count:
        raise ValueError("candidate inventory count does not match")
    names = [item.name for item in inventory]
    if len(names) != len(set(names)):
        raise ValueError("candidate inventory names are duplicated")
    return inventory


def _fold_map_bytes(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, (str, Path)):
        return Path(value).read_bytes()
    if not isinstance(value, dict):
        raise ValueError("fold map is invalid")
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        keys=np.asarray(value.get("keys", ()), dtype=np.str_),
        fold_ids=np.asarray(value.get("fold_ids", ()), dtype=np.int64),
    )
    return buffer.getvalue()


def _payload_files(payload, files):
    if files is not None:
        result = dict(files)
        if "selection.json" not in result:
            result["selection.json"] = payload
        return result
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")
    return {
        "selection.json": payload.get("selection", payload),
        "fold_map.npz": payload.get("fold_map", {}),
        "raw_cross_fit_report.json": payload.get("raw_report", {}),
        "rank_cross_fit_report.json": payload.get("rank_report", {}),
        "static_specs.json": payload.get("static_specs", {}),
        "candidate_manifest.json": payload.get("candidate_manifest", []),
        "selected_references.json": payload.get("selected_references", {}),
        "cache_identities.json": payload.get("cache_identities", {}),
    }


def _file_bytes(name, value):
    if name == "fold_map.npz":
        return _fold_map_bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, (str, Path)):
        return read_verified_bytes(value, name=name)
    return _canonical_json(value) + b"\n"


def _resolve_reference(root, relative, field):
    safe = validate_relative_path(relative, field)
    root_path = Path(root).expanduser()
    if root_path.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    root_path = root_path.resolve()
    candidate = root_path / safe
    try:
        mode = os.lstat(str(candidate)).st_mode
    except OSError as error:
        raise ValueError(field + " artifact is missing") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(field + " artifact must be a regular non-symlink file")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as error:
        raise ValueError(field + " leaves the output root") from error
    return candidate


def _validate_reference_file(root, path, digest, field, allow_null=False):
    if path is None or digest is None:
        if allow_null and path is None and digest is None:
            return
        raise ValueError(field + " path and hash are required")
    resolved = _resolve_reference(root, path, field + "_path")
    expected = _sha256(digest, field + " hash")
    read_verified_bytes(
        resolved, expected_sha256=expected, name=field + " artifact"
    )


def _validate_future_context(selection):
    if not isinstance(selection, dict):
        raise ValueError("selection must be a dictionary")
    context = selection.get("future_test_context")
    if not isinstance(context, dict) or set(context) != {
        "target_ci", "comparison"
    }:
        raise ValueError("future context is missing or malformed")
    if _finite(context["target_ci"], "future target CI") != 0.706858:
        raise ValueError("future target CI does not match")
    comparison = context["comparison"]
    if (
        not isinstance(comparison, str)
        or "strictly greater" not in comparison.lower()
    ):
        raise ValueError("future comparison language is not strict")
    for key in selection:
        if "test" in key.lower() and key != "future_test_context":
            raise ValueError("selection contains forbidden future data")


def _validate_package_payload(root, file_values):
    selection = file_values["selection.json"]
    manifest = file_values["candidate_manifest.json"]
    references = file_values["selected_references.json"]
    identities = file_values["cache_identities.json"]
    _validate_future_context(selection)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("candidate manifest must be nonempty")
    records = tuple(CandidateRecord.from_dict(item) for item in manifest)
    names = [item.name for item in records]
    if len(names) != len(set(names)):
        raise ValueError("candidate manifest names are duplicated")
    for record in records:
        _validate_reference_file(
            root,
            record.prediction_path,
            record.prediction_sha256,
            record.name + " prediction",
        )
        _validate_reference_file(
            root,
            record.checkpoint_path,
            record.checkpoint_sha256,
            record.name + " checkpoint",
            allow_null=record.type != "residual",
        )
    if not isinstance(references, dict) or set(references) != {
        "name", "prediction_path", "prediction_sha256",
        "checkpoint_path", "checkpoint_sha256",
    }:
        raise ValueError("selected references fields do not match")
    selected = {item.name: item for item in records}.get(references["name"])
    if selected is None:
        raise ValueError("selected candidate is missing from manifest")
    expected_references = {
        "name": selected.name,
        "prediction_path": selected.prediction_path,
        "prediction_sha256": selected.prediction_sha256,
        "checkpoint_path": selected.checkpoint_path,
        "checkpoint_sha256": selected.checkpoint_sha256,
    }
    if references != expected_references:
        raise ValueError("selected references do not match candidate manifest")
    _validate_reference_file(
        root,
        references["prediction_path"],
        references["prediction_sha256"],
        "selected prediction",
    )
    _validate_reference_file(
        root,
        references["checkpoint_path"],
        references["checkpoint_sha256"],
        "selected checkpoint",
        allow_null=selected.type != "residual",
    )
    if selection.get("selected_name") != selected.name:
        raise ValueError("selection name does not match selected references")
    if not isinstance(identities, dict) or set(identities) != {"train", "valid"}:
        raise ValueError("cache identities must contain train and valid only")


def _package_directory(value, create):
    package = Path(value).expanduser()
    if package.is_symlink():
        raise ValueError("validation package must not be a symlink")
    if package.exists():
        try:
            mode = os.lstat(str(package)).st_mode
        except OSError as error:
            raise ValueError("validation package is inaccessible") from error
        if not stat.S_ISDIR(mode):
            raise ValueError("validation package must be a directory")
    elif create:
        package.mkdir(parents=True, exist_ok=False)
    else:
        raise ValueError("validation package is missing")
    return package.resolve()


def _preflight_package_write(package):
    for path in package.iterdir():
        if path.name not in _CORE_FILES:
            raise ValueError("validation package contains an unexpected path")
        mode = os.lstat(str(path)).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(
                "validation package entries must be regular non-symlink files"
            )


def _validate_flat_inventory(package, expected):
    entries = tuple(package.iterdir())
    names = {path.name for path in entries}
    if len(entries) != len(names) or names != set(expected):
        raise ValueError("validation package inventory does not match")
    for path in entries:
        mode = os.lstat(str(path)).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(
                "validation package entries must be regular non-symlink files"
            )


def _flat_hashes(package):
    _validate_flat_inventory(package, _CORE_FILES)
    return {
        name: _hash_file(package / name) for name in sorted(_CORE_FILES)
    }


def seal_validation_package(
    package, payload, files=None, *, artifact_root=None
):
    """Write and seal exactly the predefined flat validation package."""
    package = _package_directory(package, create=True)
    _preflight_package_write(package)
    file_values = _payload_files(payload, files)
    if set(file_values) != _CORE_FILES:
        raise ValueError("core package file inventory does not match")
    artifacts = (
        package.parent
        if artifact_root is None
        else Path(artifact_root).expanduser()
    )
    _validate_package_payload(artifacts, file_values)
    for name in sorted(_CORE_FILES):
        _atomic_bytes(package / name, _file_bytes(name, file_values[name]))
    hashes = _flat_hashes(package)
    seal = {
        "status": "sealed",
        "schema_version": 1,
        "files": hashes,
        "files_sha256": _hash_bytes(_canonical_json(hashes)),
    }
    _atomic_json(package / "selection_seal.json", seal)
    _validate_flat_inventory(package, PACKAGE_FILES)
    verify_validation_seal(package, artifact_root=artifacts)
    return seal


def _load_json(path, name):
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw.decode("utf-8")), raw
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(name + " is malformed") from error


def verify_validation_seal(package, *, artifact_root=None):
    """Verify exact flat inventory before hashes or artifact loading."""
    package = _package_directory(package, create=False)
    _validate_flat_inventory(package, PACKAGE_FILES)
    seal, raw = _load_json(
        package / "selection_seal.json", "selection seal"
    )
    if not isinstance(seal, dict) or set(seal) != {
        "status", "schema_version", "files", "files_sha256"
    }:
        raise ValueError("selection seal is malformed")
    if seal["status"] != "sealed" or seal["schema_version"] != 1:
        raise ValueError("selection seal is malformed")
    hashes = seal["files"]
    if not isinstance(hashes, dict) or set(hashes) != _CORE_FILES:
        raise ValueError("selection seal hash map is malformed")
    for name, digest in hashes.items():
        validate_relative_path(name, "sealed file")
        _sha256(digest, "sealed file hash")
    if seal["files_sha256"] != _hash_bytes(_canonical_json(hashes)):
        raise ValueError("selection seal canonical hash is invalid")
    if raw != _canonical_json(seal) + b"\n":
        raise ValueError("selection seal is not canonical")
    verified_core = {
        name: read_verified_bytes(
            package / name, expected_sha256=hashes[name],
            name="sealed " + name,
        )
        for name in sorted(_CORE_FILES)
    }
    if {name: _hash_bytes(value) for name, value in verified_core.items()} != hashes:
        raise ValueError("selection seal does not match package files")

    file_values = {}
    for name, value in verified_core.items():
        if name == "fold_map.npz":
            file_values[name] = value
        else:
            try:
                file_values[name] = json.loads(value.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                raise ValueError(name + " is malformed") from error
    artifacts = (
        package.parent
        if artifact_root is None
        else Path(artifact_root).expanduser()
    )
    _validate_package_payload(artifacts, file_values)
    return file_values["selection.json"]


__all__ = [
    "CandidateRecord",
    "ValidationCandidate",
    "EXPERT_NAMES",
    "FORMAL_BUDGETS",
    "FORMAL_SEEDS",
    "PACKAGE_FILES",
    "build_candidate_inventory",
    "build_residual_ensembles",
    "persist_prediction_artifact",
    "read_verified_bytes",
    "load_verified_npz",
    "load_verified_npz_bytes",
    "load_verified_torch",
    "load_verified_torch_bytes",
    "select_validation_candidate",
    "seal_validation_package",
    "validate_relative_path",
    "validation_candidate_key",
    "verify_validation_seal",
]
