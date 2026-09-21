"""Immutable v4 cache boundary and data loading for anchored fusion v5."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from dta.rank_fusion_v4.cache import FusionCache, load_fusion_cache
from utils.experiment_integrity import sha256_file


EXPERT_ORDER = ("upper_bound", "boltz_pairwise", "three_dprot")
VALIDATION_SPLITS = frozenset(("train", "valid"))
_IDENTITY_FILES = (
    "features.npz",
    "metadata.json",
    "coverage.json",
    "normalization.json",
)


@dataclass(frozen=True)
class ImmutableCacheRef:
    cache: FusionCache
    identity: dict


def cache_identity(cache_dir, cache):
    """Capture provenance and content hashes for a validated v4 cache."""
    metadata = cache.metadata
    required = (
        "split",
        "count",
        "manifest_sha256",
        "ordered_key_sha256",
    )
    if any(name not in metadata for name in required):
        raise ValueError("fusion cache metadata identity is incomplete")
    path = Path(cache_dir).expanduser().resolve()
    return {
        "split": metadata["split"],
        "count": metadata["count"],
        "manifest_sha256": metadata["manifest_sha256"],
        "ordered_key_sha256": metadata["ordered_key_sha256"],
        "files": {
            name: sha256_file(path / name) for name in _IDENTITY_FILES
        },
    }


def load_immutable_v4_bundle(
    cache_root, expected_counts, splits=("train", "valid")
):
    """Load validated train/valid v4 caches and freeze their identities."""
    if isinstance(splits, str):
        raise ValueError("splits must contain only train and valid")
    requested_splits = tuple(splits)
    if not requested_splits:
        raise ValueError("splits must be nonempty")
    if len(set(requested_splits)) != len(requested_splits):
        raise ValueError("splits must not contain duplicate names")
    if any(split not in VALIDATION_SPLITS for split in requested_splits):
        raise ValueError("splits must contain only train and valid")

    root = Path(cache_root).expanduser().resolve()
    references = {}
    for split in requested_splits:
        try:
            expected_count = expected_counts[split]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"expected count is missing for split {split}"
            ) from error
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, Integral)
            or expected_count < 0
        ):
            raise ValueError(f"expected count for split {split} is invalid")

        cache_dir = root / "joined" / split
        cache = load_fusion_cache(cache_dir, expected_split=split)
        if len(cache.keys) != int(expected_count):
            raise ValueError(
                f"{split} cache count does not match expected count"
            )
        if not cache.available[:, 0].all():
            raise ValueError(
                "Upper-Bound must be available for every cache row"
            )
        references[split] = ImmutableCacheRef(
            cache=cache,
            identity=cache_identity(cache_dir, cache),
        )
    return references


def fallback_predictions(predictions, available):
    """Replace unavailable expert outputs with each row's Upper-Bound."""
    prediction_array = np.asarray(predictions)
    available_array = np.asarray(available)
    if (
        prediction_array.ndim != 2
        or prediction_array.shape[1:] != (len(EXPERT_ORDER),)
        or available_array.shape != prediction_array.shape
    ):
        raise ValueError(
            "predictions and available must have matching [rows, 3] shape"
        )
    if available_array.dtype != np.bool_:
        raise ValueError("available must be boolean")
    try:
        result = prediction_array.astype(np.float32, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("predictions must be finite") from error
    if not np.isfinite(result).all():
        raise ValueError("predictions must be finite")
    if not available_array[:, 0].all():
        raise ValueError(
            "Upper-Bound must be available for every prediction row"
        )
    return np.where(available_array, result, result[:, :1]).astype(
        np.float32, copy=False
    )


class AnchoredFusionDataset(torch.utils.data.Dataset):
    def __init__(self, cache, base_predictions):
        base_array = np.asarray(base_predictions)
        if (
            base_array.ndim != 1
            or base_array.shape[0] != len(cache.keys)
        ):
            raise ValueError(
                "base_predictions must be a vector matching the cache length"
            )
        try:
            base_array = base_array.astype(np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("base_predictions must be finite") from error
        if not np.isfinite(base_array).all():
            raise ValueError("base_predictions must be finite")
        self.cache = cache
        self.base_predictions = base_array

    def __len__(self):
        return len(self.cache.keys)

    def __getitem__(self, index):
        return {
            "key": self.cache.keys[index],
            "label": self.cache.labels[index],
            "base_prediction": self.base_predictions[index],
            "raw_predictions": self.cache.predictions[index],
            "standardized_predictions": (
                self.cache.standardized_predictions[index]
            ),
            "available": self.cache.available[index],
            "latents": (
                self.cache.upper_bound_latents[index],
                self.cache.boltz_latents[index],
                self.cache.three_dprot_latents[index],
            ),
        }


def collate_anchored_fusion(records):
    if not records:
        raise ValueError("anchored fusion batch must be nonempty")
    return {
        "keys": tuple(record["key"] for record in records),
        "labels": torch.as_tensor(
            np.asarray([record["label"] for record in records]),
            dtype=torch.float32,
        ),
        "base_predictions": torch.as_tensor(
            np.asarray(
                [record["base_prediction"] for record in records]
            ),
            dtype=torch.float32,
        ),
        "raw_predictions": torch.as_tensor(
            np.stack(
                [record["raw_predictions"] for record in records]
            ),
            dtype=torch.float32,
        ),
        "standardized_predictions": torch.as_tensor(
            np.stack(
                [
                    record["standardized_predictions"]
                    for record in records
                ]
            ),
            dtype=torch.float32,
        ),
        "available": torch.as_tensor(
            np.stack([record["available"] for record in records]),
            dtype=torch.bool,
        ),
        "latents": [
            torch.as_tensor(
                np.stack(
                    [record["latents"][expert] for record in records]
                ),
                dtype=torch.float32,
            )
            for expert in range(len(EXPERT_ORDER))
        ],
    }
