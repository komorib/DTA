"""Join V9 complex geometry with cached multi-row, pocket-aligned MSAs."""

import hashlib
import pickle

import lmdb
import numpy as np
import torch

from dta.msa_pocket_expert_v9.dataset import (
    MSAPocketComplexDataset,
    collate_msa_pocket_complexes,
    move_batch_to_device,
)


MSA_PAD_INDEX = 22


def _stable_seed(value):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


class MSAPairComplexDataset(MSAPocketComplexDataset):
    def __init__(self, manifest_path, cache_dir, row_cache, limit=None):
        super().__init__(manifest_path, cache_dir, limit)
        self.row_cache = self.cache_dir.__class__(row_cache).expanduser().resolve()
        self._row_env = None

    def _row_environment(self):
        if self._row_env is None:
            self._row_env = lmdb.open(
                str(self.row_cache / "rows.lmdb"),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=64,
            )
        return self._row_env

    def __getstate__(self):
        state = super().__getstate__()
        state["_row_env"] = None
        return state

    def __getitem__(self, index):
        item = dict(super().__getitem__(index))
        chain_ids = np.asarray(item["pocket_chain_index"], dtype=np.int64)
        positions = np.asarray(item["pocket_sequence_position"], dtype=np.int64)
        profile_ids = tuple(item["profile_ids"])
        rows_by_chain = []
        with self._row_environment().begin(write=False) as transaction:
            for sequence_id in profile_ids:
                raw = transaction.get(str(sequence_id).encode("ascii"))
                if raw is None:
                    raise KeyError("MSA row cache is missing %s" % sequence_id)
                rows_by_chain.append(pickle.loads(bytes(raw)))
        maximum_depth = max(int(rows["alignment"].shape[0]) for rows in rows_by_chain)
        pocket_count = int(chain_ids.size)
        tokens = np.full((maximum_depth, pocket_count), MSA_PAD_INDEX, dtype=np.uint8)
        insertions = np.zeros((maximum_depth, pocket_count), dtype=np.uint8)
        row_features = np.zeros((maximum_depth, pocket_count, 2), dtype=np.float16)
        row_mask = np.zeros((maximum_depth, pocket_count), dtype=np.bool_)
        for chain_index, rows in enumerate(rows_by_chain):
            selected = np.flatnonzero(chain_ids == chain_index)
            if selected.size == 0:
                continue
            sequence_positions = positions[selected]
            alignment = np.asarray(rows["alignment"], dtype=np.uint8)
            deletion = np.asarray(rows["insertions"], dtype=np.uint8)
            if np.any(sequence_positions < 0) or np.any(sequence_positions >= alignment.shape[1]):
                raise ValueError("pocket sequence position is outside cached MSA")
            depth = int(alignment.shape[0])
            tokens[:depth, selected] = alignment[:, sequence_positions]
            insertions[:depth, selected] = deletion[:, sequence_positions]
            row_features[:depth, selected, 0] = np.asarray(rows["identity"], dtype=np.float16)[:, None]
            row_features[:depth, selected, 1] = np.asarray(rows["coverage"], dtype=np.float16)[:, None]
            row_mask[:depth, selected] = True
        item["pocket_msa_tokens"] = tokens
        item["pocket_msa_insertions"] = insertions
        item["pocket_msa_row_features"] = row_features
        item["pocket_msa_row_mask"] = row_mask
        return item


class MSAPairCollator:
    def __init__(self, row_depth=64, training=False, subset_index=0):
        if int(row_depth) <= 0:
            raise ValueError("row_depth must be positive")
        self.row_depth = int(row_depth)
        self.training = bool(training)
        self.subset_index = int(subset_index)

    def _indices(self, item, chain_index, available):
        available = np.asarray(available, dtype=np.int64)
        if available.size <= self.row_depth:
            return available
        query = available[:1]
        candidates = available[1:]
        if self.training:
            random_component = int(torch.randint(0, 2**31 - 1, ()).item())
            seed = (_stable_seed("%s:%d" % (item["key"], chain_index)) + random_component) % (2**63)
        else:
            seed = _stable_seed("%s:%d:%d" % (item["key"], chain_index, self.subset_index))
        generator = np.random.default_rng(seed)
        chosen = generator.choice(candidates, size=self.row_depth - 1, replace=False)
        return np.concatenate((query, chosen))

    def _subsample(self, item):
        pocket_count = int(item["pocket_msa_tokens"].shape[1])
        tokens = np.full((self.row_depth, pocket_count), MSA_PAD_INDEX, dtype=np.uint8)
        insertions = np.zeros((self.row_depth, pocket_count), dtype=np.uint8)
        features = np.zeros((self.row_depth, pocket_count, 2), dtype=np.float16)
        mask = np.zeros((self.row_depth, pocket_count), dtype=np.bool_)
        chain_ids = np.asarray(item["pocket_chain_index"], dtype=np.int64)
        source_mask = np.asarray(item["pocket_msa_row_mask"], dtype=np.bool_)
        for chain_index in np.unique(chain_ids):
            positions = np.flatnonzero(chain_ids == chain_index)
            available = np.flatnonzero(source_mask[:, positions[0]])
            chosen = self._indices(item, int(chain_index), available)
            count = int(chosen.size)
            tokens[:count, positions] = item["pocket_msa_tokens"][chosen][:, positions]
            insertions[:count, positions] = item["pocket_msa_insertions"][chosen][:, positions]
            features[:count, positions] = item["pocket_msa_row_features"][chosen][:, positions]
            mask[:count, positions] = True
        result = dict(item)
        result["pocket_msa_tokens"] = tokens
        result["pocket_msa_insertions"] = insertions
        result["pocket_msa_row_features"] = features
        result["pocket_msa_row_mask"] = mask
        return result

    def __call__(self, items):
        selected = [self._subsample(item) for item in items]
        batch = collate_msa_pocket_complexes(selected)
        batch_size = len(selected)
        maximum_pocket = int(batch["pocket_mask"].shape[1])
        tokens = torch.full(
            (batch_size, self.row_depth, maximum_pocket),
            MSA_PAD_INDEX,
            dtype=torch.long,
        )
        insertions = torch.zeros((batch_size, self.row_depth, maximum_pocket), dtype=torch.float32)
        features = torch.zeros((batch_size, self.row_depth, maximum_pocket, 2), dtype=torch.float32)
        mask = torch.zeros((batch_size, self.row_depth, maximum_pocket), dtype=torch.bool)
        for batch_index, item in enumerate(selected):
            pocket_count = int(item["pocket_msa_tokens"].shape[1])
            tokens[batch_index, :, :pocket_count] = torch.as_tensor(
                item["pocket_msa_tokens"], dtype=torch.long
            )
            insertions[batch_index, :, :pocket_count] = torch.as_tensor(
                item["pocket_msa_insertions"], dtype=torch.float32
            )
            features[batch_index, :, :pocket_count] = torch.as_tensor(
                item["pocket_msa_row_features"], dtype=torch.float32
            )
            mask[batch_index, :, :pocket_count] = torch.as_tensor(
                item["pocket_msa_row_mask"], dtype=torch.bool
            )
        batch.update(
            {
                "pocket_msa_tokens": tokens,
                "pocket_msa_insertions": torch.log1p(insertions),
                "pocket_msa_row_features": features,
                "pocket_msa_row_mask": mask,
            }
        )
        return batch


__all__ = [
    "MSA_PAD_INDEX",
    "MSAPairCollator",
    "MSAPairComplexDataset",
    "move_batch_to_device",
]
