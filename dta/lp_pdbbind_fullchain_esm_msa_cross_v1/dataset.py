"""Join complete-chain ESM and sealed MSA features without old pocket MSA."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
import torch

from data_utils.lmdb_dataset import ordered_chain_items
from dta.msa_pocket_expert_v9.dataset import MSAPocketComplexDataset
from dta.msa_pocket_expert_v9.sequence_manifest import normalize_query_sequence
from dta.lp_pdbbind_windowed_msa_v1.cache import validate_feature_record
from dta.lp_pdbbind_fullchain_msa_replacement_v1.dataset import (
    FullChainOnlyCollator, FullChainOnlyDataset,
)


class FullChainESMMSADataset(FullChainOnlyDataset):
    """Retain all protein residues for context, plus the structural pocket."""

    include_local_msa = True

    def __init__(self, manifest_path, complex_cache, feature_cache, mode,
                 source_manifest, targets_path, weight_sha256):
        super().__init__(manifest_path, complex_cache, feature_cache, mode,
                         source_manifest, targets_path, weight_sha256)
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        source_path = Path(manifest["lmdb_path"])
        if not source_path.is_absolute():
            source_path = Path(manifest_path).parent / source_path
        self.source_lmdb = source_path.resolve()
        if not self.source_lmdb.is_dir():
            raise FileNotFoundError("full-chain ESM source LMDB is missing")
        self._source_env = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_source_env"] = None
        return state

    def _source_environment(self):
        if self._source_env is None:
            self._source_env = lmdb.open(
                str(self.source_lmdb), readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return self._source_env

    def __getitem__(self, index):
        # Bypass the pocket-only gather in FullChainOnlyDataset.__getitem__.
        item = dict(MSAPocketComplexDataset.__getitem__(self, index))
        item.pop("pocket_msa", None)
        key = str(item["key"])
        with self._source_environment().begin(write=False) as transaction:
            raw = transaction.get(key.encode("utf-8"))
        if raw is None:
            raise KeyError("full-chain ESM source is missing for %s" % key)
        source = pickle.loads(bytes(raw))
        chains = list(ordered_chain_items(source["protein_features"]))
        profile_ids = tuple(str(value) for value in item["profile_ids"])
        if len(chains) != len(profile_ids):
            raise ValueError("source-chain count differs from structural cache")
        esm_parts, msa_parts, offsets, lengths = [], [], [], []
        cursor = 0
        with self._feature_environment().begin(write=False) as transaction:
            for (_chain_name, chain), sid in zip(chains, profile_ids):
                query = normalize_query_sequence(chain["sequence"])
                if self.queries.get(sid) != query or sid not in self.expected_sources:
                    raise ValueError("full-chain ESM/MSA sequence identity differs")
                esm = np.asarray(chain["esm2_embedding"], dtype=np.float32)
                record_raw = transaction.get(sid.encode("ascii"))
                if record_raw is None:
                    raise KeyError("full-chain MSA feature is missing for %s" % sid)
                msa = validate_feature_record(
                    pickle.loads(bytes(record_raw)), sid, query, self.mode,
                    self.weight_sha256, self.expected_sources[sid],
                )
                if esm.shape != (len(query), 640) or msa.shape != (len(query), 768):
                    raise ValueError("complete-chain ESM/MSA shapes differ")
                if not np.isfinite(esm).all() or not np.isfinite(msa).all():
                    raise ValueError("complete-chain ESM/MSA contains nonfinite values")
                esm_parts.append(esm)
                msa_parts.append(np.asarray(msa, dtype=np.float32))
                offsets.append(cursor)
                lengths.append(len(query))
                cursor += len(query)
        chain_index = np.asarray(item["pocket_chain_index"], dtype=np.int64)
        position = np.asarray(item["pocket_sequence_position"], dtype=np.int64)
        if (chain_index.ndim != 1 or position.shape != chain_index.shape
                or np.any(chain_index < 0) or np.any(chain_index >= len(chains))):
            raise ValueError("pocket-to-full-chain mapping is malformed")
        chain_lengths = np.asarray(lengths, dtype=np.int64)
        if np.any(position < 0) or np.any(position >= chain_lengths[chain_index]):
            raise ValueError("pocket position exceeds complete-chain length")
        full_indices = np.asarray(offsets, dtype=np.int64)[chain_index] + position
        full_esm = np.concatenate(esm_parts, axis=0)
        full_msa = np.concatenate(msa_parts, axis=0)
        if not np.array_equal(
            np.asarray(item["pocket_esm"], dtype=np.float16),
            full_esm[full_indices].astype(np.float16),
        ):
            raise ValueError("structural pocket ESM is not aligned to the complete chain")
        nonpocket = np.ones(cursor, dtype=np.bool_)
        nonpocket[full_indices] = False
        item["fullchain_esm"] = full_esm
        item["fullchain_msa"] = full_msa
        item["fullchain_nonpocket_mask"] = nonpocket
        if self.include_local_msa:
            item["fullchain_msa_pocket"] = full_msa[full_indices]
        return item


class FullChainESMMSACollator(FullChainOnlyCollator):
    def __call__(self, items):
        batch = super().__call__(items)
        maximum = max(len(item["fullchain_esm"]) for item in items)
        count = len(items)
        esm = torch.zeros((count, maximum, 640), dtype=torch.float32)
        msa = torch.zeros((count, maximum, 768), dtype=torch.float32)
        nonpocket = torch.zeros((count, maximum), dtype=torch.bool)
        mask = torch.zeros((count, maximum), dtype=torch.bool)
        for index, item in enumerate(items):
            length = len(item["fullchain_esm"])
            if (np.asarray(item["fullchain_esm"]).shape != (length, 640)
                    or np.asarray(item["fullchain_msa"]).shape != (length, 768)
                    or np.asarray(item["fullchain_nonpocket_mask"]).shape != (length,)):
                raise ValueError("complete-chain feature shape is invalid")
            esm[index, :length] = torch.as_tensor(item["fullchain_esm"], dtype=torch.float32)
            msa[index, :length] = torch.as_tensor(item["fullchain_msa"], dtype=torch.float32)
            nonpocket[index, :length] = torch.as_tensor(item["fullchain_nonpocket_mask"], dtype=torch.bool)
            mask[index, :length] = True
        batch["fullchain_esm"] = esm
        batch["fullchain_msa"] = msa
        batch["fullchain_mask"] = mask
        batch["fullchain_nonpocket_mask"] = nonpocket
        return batch
