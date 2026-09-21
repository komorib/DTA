"""Strict LP-PDBbind manifest parsing for Rank Fusion v4."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from utils.experiment_integrity import manifest_fingerprint


@dataclass(frozen=True)
class ParsedManifest:
    path: Path
    split: str
    keys: tuple
    labels: np.ndarray
    sha256: str
    ordered_key_sha256: str


def load_manifest(path, *, expected_split):
    """Load unique keyed finite labels and verify the declared split."""
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
    return ParsedManifest(
        path=path,
        split=expected_split,
        keys=tuple(keys),
        labels=np.asarray(labels, dtype=np.float32),
        sha256=fingerprint["sha256"],
        ordered_key_sha256=fingerprint["ordered_key_sha256"],
    )
