"""Full-chain feature reader with no pocket-MSA rows or profile in the batch."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
import torch

from dta.msa_pocket_expert_v9.dataset import (
    MSAPocketComplexDataset,
    collate_msa_pocket_complexes,
)
from dta.msa_transformer_smoke_v1.msa import ENCODER_NAME, sha256_file
from dta.lp_pdbbind_windowed_msa_v1 import WINDOW_POLICY
from dta.lp_pdbbind_windowed_msa_v1.cache import (
    FEATURE_WIDTH,
    gather_pocket_features,
    load_source_manifest,
)


class FullChainOnlyDataset(MSAPocketComplexDataset):
    """Consume sealed whole-chain embeddings without opening the old row cache."""

    def __init__(self, manifest_path, complex_cache, feature_cache, mode,
                 source_manifest, targets_path, weight_sha256):
        super().__init__(manifest_path, complex_cache)
        if self.split not in ("train", "valid"):
            raise ValueError("full-chain replacement only accepts train/valid")
        self.feature_cache = Path(feature_cache).expanduser().resolve()
        self.mode = str(mode)
        self.weight_sha256 = str(weight_sha256)
        self.expected_sources = load_source_manifest(source_manifest)
        with Path(targets_path).open("r", encoding="utf-8") as handle:
            targets = json.load(handle)
        if not isinstance(targets, list):
            raise ValueError("sealed target inventory must be a list")
        self.queries = {str(row["sequence_id"]): str(row["query_sequence"]) for row in targets}
        if len(self.queries) != len(targets) or set(self.queries) != set(self.expected_sources):
            raise ValueError("target inventory differs from source manifest")
        with (self.feature_cache / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("status") != "built"
            or metadata.get("encoder") != ENCODER_NAME
            or metadata.get("feature_mode") != self.mode
            or metadata.get("weights_sha256") != self.weight_sha256
            or metadata.get("source_manifest_sha256") != sha256_file(source_manifest)
            or metadata.get("window_policy") != WINDOW_POLICY
            or metadata.get("target_count") != len(targets)
        ):
            raise ValueError("full-chain cache metadata differs from the requested input")
        if not (self.feature_cache / "proteins.lmdb").is_dir():
            raise FileNotFoundError("full-chain feature LMDB is missing")
        self._feature_env = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_feature_env"] = None
        return state

    def _feature_environment(self):
        if self._feature_env is None:
            self._feature_env = lmdb.open(
                str(self.feature_cache / "proteins.lmdb"), readonly=True,
                lock=False, readahead=False, meminit=False,
            )
        return self._feature_env

    def __getitem__(self, index):
        item = dict(super().__getitem__(index))
        # The historical complex cache carries a 50-channel pocket-MSA profile.
        # It is discarded before any model or collator sees the sample.
        item.pop("pocket_msa", None)
        profile_ids = tuple(str(value) for value in item["profile_ids"])
        used = {profile_ids[int(i)] for i in np.unique(item["pocket_chain_index"])}
        records = {}
        with self._feature_environment().begin(write=False) as transaction:
            for sid in used:
                raw = transaction.get(sid.encode("ascii"))
                if raw is None or sid not in self.queries:
                    raise KeyError("missing sealed full-chain feature for %s" % sid)
                records[sid] = pickle.loads(bytes(raw))
        item["fullchain_msa_pocket"] = gather_pocket_features(
            item, records, {sid: self.queries[sid] for sid in used},
            self.mode, self.weight_sha256, self.expected_sources,
        )
        return item


class FullChainOnlyCollator:
    """Reuse the established geometry padding, then erase its dummy MSA slot."""

    def __call__(self, items):
        if not items:
            raise ValueError("cannot collate an empty full-chain batch")
        selected = []
        for item in items:
            if "pocket_msa" in item or any(key.startswith("pocket_msa_") for key in item):
                raise ValueError("old pocket-MSA inputs reached the replacement collator")
            pocket_count = int(np.asarray(item["pocket_chain_index"]).size)
            dummy = dict(item)
            dummy["pocket_msa"] = np.zeros((pocket_count, 50), dtype=np.float32)
            selected.append(dummy)
        batch = collate_msa_pocket_complexes(selected)
        batch.pop("pocket_msa")
        features = torch.zeros(
            (len(items), batch["pocket_mask"].shape[1], FEATURE_WIDTH),
            dtype=torch.float32,
        )
        for index, item in enumerate(items):
            pocket_count = int(np.asarray(item["pocket_chain_index"]).size)
            value = np.asarray(item.get("fullchain_msa_pocket"))
            if value.shape != (pocket_count, FEATURE_WIDTH) or not np.isfinite(value).all():
                raise ValueError("full-chain pocket features are missing or invalid")
            features[index, :pocket_count] = torch.as_tensor(value, dtype=torch.float32)
        batch["fullchain_msa_pocket"] = features
        return batch


__all__ = ["FullChainOnlyDataset", "FullChainOnlyCollator"]
