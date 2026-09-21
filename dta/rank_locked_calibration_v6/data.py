"""Immutable v4 cache and sealed v5 anchor boundary for calibration v6."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from dta.anchored_expert_fusion_v5.selection import verify_validation_seal
from dta.anchored_expert_fusion_v5.training import (
    StaticSpec,
    static_predictions,
)
from dta.rank_fusion_v4.cache import FusionCache, load_fusion_cache
from utils.experiment_integrity import sha256_file

from . import EXPERT_ORDER


CACHE_FILES = (
    "features.npz",
    "metadata.json",
    "coverage.json",
    "normalization.json",
)
VALIDATION_SPLITS = frozenset(("train", "valid"))
_REFERENCE_FIELDS = frozenset(
    (
        "checkpoint_path",
        "checkpoint_sha256",
        "name",
        "prediction_path",
        "prediction_sha256",
    )
)


def _canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _load_canonical_json(path, name):
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(name + " is malformed") from error
    if raw != _canonical_json_bytes(value):
        raise ValueError(name + " must be canonical JSON")
    return value


def _validate_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class ImmutableCacheRef:
    split: str
    directory: Path
    cache: FusionCache
    file_sha256: Mapping

    def __post_init__(self):
        if self.split not in VALIDATION_SPLITS:
            raise ValueError("cache split must be train or valid")
        directory = Path(self.directory).expanduser().resolve()
        if not isinstance(self.cache, FusionCache):
            raise ValueError("cache must be a FusionCache")
        if not isinstance(self.file_sha256, Mapping) or set(
            self.file_sha256
        ) != set(CACHE_FILES):
            raise ValueError("cache file hashes do not match")
        hashes = MappingProxyType(
            {
                name: _validate_sha256(
                    self.file_sha256[name], self.split + " " + name
                )
                for name in CACHE_FILES
            }
        )
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "file_sha256", hashes)

    @property
    def identity(self):
        metadata = self.cache.metadata
        required = (
            "split",
            "count",
            "manifest_sha256",
            "ordered_key_sha256",
        )
        if not isinstance(metadata, Mapping) or any(
            name not in metadata for name in required
        ):
            raise ValueError("fusion cache metadata identity is incomplete")
        return {
            "split": self.split,
            "count": len(self.cache.keys),
            "manifest_sha256": metadata["manifest_sha256"],
            "ordered_key_sha256": metadata["ordered_key_sha256"],
            "files": {name: self.file_sha256[name] for name in CACHE_FILES},
        }


def _finite_array(value, name, *, ndim=None, first_dimension=None):
    try:
        array = np.asarray(value)
        finite = np.isfinite(array).all()
    except (TypeError, ValueError) as error:
        raise ValueError(name + " contains nonfinite values") from error
    if ndim is not None and array.ndim != ndim:
        raise ValueError(name + " shape does not match")
    if first_dimension is not None and array.shape[0] != first_dimension:
        raise ValueError(name + " shape does not match")
    if not finite:
        raise ValueError(name + " contains nonfinite values")
    return array


def _validate_cache(cache, split, expected_count):
    if not isinstance(cache, FusionCache):
        raise ValueError(split + " cache must be a FusionCache")
    count = len(cache.keys)
    if count != expected_count:
        raise ValueError(split + " cache count does not match expected count")
    if len(set(cache.keys)) != count:
        raise ValueError(split + " cache keys must be unique")
    if not isinstance(cache.metadata, Mapping):
        raise ValueError(split + " cache metadata is invalid")
    if cache.metadata.get("split") != split:
        raise ValueError(split + " cache metadata split does not match")
    if cache.metadata.get("count") != count:
        raise ValueError(split + " cache metadata count does not match")

    _finite_array(cache.labels, "labels", ndim=1, first_dimension=count)
    _finite_array(
        cache.predictions, "predictions", ndim=2, first_dimension=count
    )
    _finite_array(
        cache.standardized_predictions,
        "standardized_predictions",
        ndim=2,
        first_dimension=count,
    )
    for name in (
        "upper_bound_latents",
        "boltz_latents",
        "three_dprot_latents",
    ):
        _finite_array(
            getattr(cache, name), name, ndim=2, first_dimension=count
        )

    available = cache.available
    if not isinstance(available, np.ndarray):
        raise ValueError("available must be present")
    if available.dtype != np.bool_ or available.shape != (count, 3):
        raise ValueError("available must be boolean with [rows, 3] shape")
    if np.asarray(cache.predictions).shape != (count, 3) or np.asarray(
        cache.standardized_predictions
    ).shape != (count, 3):
        raise ValueError("expert predictions must have [rows, 3] shape")
    if not available[:, 0].all():
        raise ValueError("Upper-Bound must be available for every cache row")


def load_validation_caches(
    cache_root,
    expected_counts,
    splits=("train", "valid"),
    expected_manifest_sha256_by_split=None,
):
    """Load train/valid joined caches and capture exact file identities."""
    if isinstance(splits, str):
        raise ValueError("splits must contain only train and valid")
    requested = tuple(splits)
    if not requested or any(split not in VALIDATION_SPLITS for split in requested):
        raise ValueError("splits must contain only train and valid")
    if len(set(requested)) != len(requested):
        raise ValueError("splits must not contain duplicate names")
    if not isinstance(expected_counts, Mapping):
        raise ValueError("expected counts must map train and valid")
    if expected_manifest_sha256_by_split is not None and not isinstance(
        expected_manifest_sha256_by_split, Mapping
    ):
        raise ValueError("expected manifests must map train and valid")

    root = Path(cache_root).expanduser().resolve()
    references = {}
    for split in requested:
        count = expected_counts.get(split)
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or int(count) < 0
        ):
            raise ValueError("expected count for split " + split + " is invalid")
        expected_manifest = None
        if expected_manifest_sha256_by_split is not None:
            expected_manifest = expected_manifest_sha256_by_split.get(split)
            if expected_manifest is not None:
                _validate_sha256(
                    expected_manifest, split + " expected manifest hash"
                )
        directory = root / "joined" / split
        cache = load_fusion_cache(
            directory,
            expected_split=split,
            expected_manifest_sha256=expected_manifest,
        )
        _validate_cache(cache, split, int(count))
        hashes = {
            name: sha256_file(directory / name) for name in CACHE_FILES
        }
        references[split] = ImmutableCacheRef(
            split=split,
            directory=directory,
            cache=cache,
            file_sha256=hashes,
        )
    return MappingProxyType(references)


@dataclass(frozen=True)
class V5Anchor:
    family: str
    selected_name: str
    selected_checkpoint_epoch: int
    static_spec: StaticSpec
    train_predictions: np.ndarray
    valid_predictions: np.ndarray
    stored_valid_predictions: np.ndarray
    selection: Mapping
    selected_references: Mapping


def _resolve_reference(root, relative_path, name):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(name + " path is invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(name + " path escapes the v5 validation root")
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(name + " path escapes the v5 validation root")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(name + " must be a regular file")
    return candidate


def _require_reference_hash(path, expected, name):
    _validate_sha256(expected, name + " hash")
    if sha256_file(path) != expected:
        raise ValueError(name + " hash does not match")


def _plain_identity(value):
    return {
        "split": value["split"],
        "count": value["count"],
        "manifest_sha256": value["manifest_sha256"],
        "ordered_key_sha256": value["ordered_key_sha256"],
        "files": dict(value["files"]),
    }


def reconstruct_v5_anchor(package, v5_validation_root, cache_references):
    """Verify and reproduce the selected v5 rank-family static anchor."""
    if not isinstance(cache_references, Mapping) or set(
        cache_references
    ) != VALIDATION_SPLITS:
        raise ValueError("cache references must contain train and valid")
    for split in ("train", "valid"):
        if not isinstance(cache_references[split], ImmutableCacheRef):
            raise ValueError(split + " cache reference is invalid")

    root = Path(v5_validation_root).expanduser().resolve()
    package_source = Path(package).expanduser()
    if package_source.is_symlink():
        raise ValueError("v5 validation package must not be a symlink")
    package_path = package_source.resolve()
    if root != package_path and root not in package_path.parents:
        raise ValueError("v5 package must be inside the validation root")
    selection = verify_validation_seal(package_path, artifact_root=root)
    if not isinstance(selection, Mapping):
        raise ValueError("v5 selection is malformed")
    if selection.get("selected_family") != "rank":
        raise ValueError("v5 selected family must be rank")

    static_specs = _load_canonical_json(
        package_path / "static_specs.json", "static specs"
    )
    references = _load_canonical_json(
        package_path / "selected_references.json", "selected references"
    )
    if not isinstance(static_specs, dict) or "rank" not in static_specs:
        raise ValueError("rank static spec is missing")
    if not isinstance(references, dict) or set(references) != _REFERENCE_FIELDS:
        raise ValueError("selected references fields do not match")
    if references["name"] != selection.get("selected_name"):
        raise ValueError("selected reference name does not match")

    spec = StaticSpec.from_dict(static_specs["rank"])
    if spec.family != "rank":
        raise ValueError("v5 static spec family must be rank")
    for split in ("train", "valid"):
        expected_identity = cache_references[split].identity
        spec_identity = _plain_identity(
            getattr(spec, split + "_cache_identity")
        )
        if spec_identity != expected_identity:
            raise ValueError(split + " cache identity does not match v5 static spec")

    checkpoint_path = _resolve_reference(
        root, references["checkpoint_path"], "selected checkpoint"
    )
    prediction_path = _resolve_reference(
        root, references["prediction_path"], "selected prediction"
    )
    _require_reference_hash(
        checkpoint_path, references["checkpoint_sha256"], "selected checkpoint"
    )
    _require_reference_hash(
        prediction_path, references["prediction_sha256"], "selected prediction"
    )
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as error:
        raise ValueError("selected checkpoint is malformed") from error
    if not isinstance(checkpoint, Mapping) or checkpoint.get("epoch") != -1:
        raise ValueError("selected checkpoint must have epoch -1")

    try:
        with np.load(str(prediction_path), allow_pickle=False) as archive:
            required = {"keys", "labels", "predictions", "epoch", "variant"}
            if set(archive.files) != required:
                raise ValueError("selected prediction fields do not match")
            stored_keys = tuple(str(value) for value in archive["keys"].tolist())
            stored_labels = np.asarray(archive["labels"], dtype=np.float32)
            stored_predictions = np.asarray(
                archive["predictions"], dtype=np.float64
            )
            stored_epoch = int(np.asarray(archive["epoch"]).item())
            stored_variant = str(np.asarray(archive["variant"]).item())
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and "selected prediction" in str(error):
            raise
        raise ValueError("selected prediction is malformed") from error

    valid_cache = cache_references["valid"].cache
    if stored_keys != tuple(valid_cache.keys):
        raise ValueError("selected prediction ordered keys do not match")
    if stored_labels.shape != valid_cache.labels.shape or not np.array_equal(
        stored_labels, np.asarray(valid_cache.labels, dtype=np.float32)
    ):
        raise ValueError("selected prediction labels do not match")
    if stored_epoch != -1 or stored_variant != "best":
        raise ValueError("selected prediction must be best epoch -1")
    if stored_predictions.shape != (len(valid_cache.keys),) or not np.isfinite(
        stored_predictions
    ).all():
        raise ValueError("selected predictions must be a finite vector")

    train_predictions = np.asarray(
        static_predictions(cache_references["train"].cache, spec),
        dtype=np.float64,
    )
    valid_predictions = np.asarray(
        static_predictions(valid_cache, spec), dtype=np.float64
    )
    if not np.allclose(
        valid_predictions, stored_predictions, rtol=0.0, atol=1e-10
    ):
        raise ValueError("stored selected prediction does not match static anchor")
    for values in (train_predictions, valid_predictions, stored_predictions):
        values.setflags(write=False)

    return V5Anchor(
        family="rank",
        selected_name=selection["selected_name"],
        selected_checkpoint_epoch=-1,
        static_spec=spec,
        train_predictions=train_predictions,
        valid_predictions=valid_predictions,
        stored_valid_predictions=stored_predictions,
        selection=MappingProxyType(dict(selection)),
        selected_references=MappingProxyType(dict(references)),
    )


class RankLockedCalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, cache, base_predictions):
        if not isinstance(cache, FusionCache):
            raise ValueError("cache must be a FusionCache")
        try:
            base = np.asarray(base_predictions, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("base predictions must be finite") from error
        if base.shape != (len(cache.keys),) or not np.isfinite(base).all():
            raise ValueError("base predictions must be a finite matching vector")
        self.cache = cache
        self.base_predictions = base.copy()

    def __len__(self):
        return len(self.cache.keys)

    def __getitem__(self, index):
        return {
            "key": self.cache.keys[index],
            "label": self.cache.labels[index],
            "base_prediction": self.base_predictions[index],
            "raw_predictions": self.cache.predictions[index],
            "standardized_predictions": self.cache.standardized_predictions[index],
            "available": self.cache.available[index],
            "latents": (
                self.cache.upper_bound_latents[index],
                self.cache.boltz_latents[index],
                self.cache.three_dprot_latents[index],
            ),
        }


def collate_rank_locked_calibration(records):
    if not records:
        raise ValueError("rank-locked calibration batch must be nonempty")
    return {
        "keys": tuple(record["key"] for record in records),
        "labels": torch.as_tensor(
            np.asarray([record["label"] for record in records]),
            dtype=torch.float32,
        ),
        "base_predictions": torch.as_tensor(
            np.asarray([record["base_prediction"] for record in records]),
            dtype=torch.float32,
        ),
        "raw_predictions": torch.as_tensor(
            np.stack([record["raw_predictions"] for record in records]),
            dtype=torch.float32,
        ),
        "standardized_predictions": torch.as_tensor(
            np.stack(
                [record["standardized_predictions"] for record in records]
            ),
            dtype=torch.float32,
        ),
        "available": torch.as_tensor(
            np.stack([record["available"] for record in records]),
            dtype=torch.bool,
        ),
        "latents": [
            torch.as_tensor(
                np.stack([record["latents"][index] for record in records]),
                dtype=torch.float32,
            )
            for index in range(len(EXPERT_ORDER))
        ],
    }


__all__ = [
    "CACHE_FILES",
    "ImmutableCacheRef",
    "RankLockedCalibrationDataset",
    "V5Anchor",
    "collate_rank_locked_calibration",
    "load_validation_caches",
    "reconstruct_v5_anchor",
]
