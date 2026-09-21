"""V27 data wrappers and target-aware balanced batch sampling."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from dta.optional_structure_staged_v22.data import (
    DATASET_INDEX,
    OptionalStructureCollator as _V22Collator,
    OptionalStructureDataset as _V22Dataset,
    canonical_sequence,
    manifest_fingerprint,
    move_batch_to_device,
    read_jsonl,
    sha256_file,
    stable_id,
    write_jsonl,
)


class OptionalStructureDataset(_V22Dataset):
    """Adds immutable target/ligand identities required by relative training."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item["sequence_id"] = str(row["sequence_id"])
        item["ligand_id"] = str(row["ligand_id"])
        return item


class OptionalStructureCollator(_V22Collator):
    """Preserves identities and creates batch-local target group identifiers."""

    def __call__(self, items):
        batch = super().__call__(items)
        sequence_ids = tuple(str(item["sequence_id"]) for item in items)
        ligand_ids = tuple(str(item["ligand_id"]) for item in items)
        group_by_sequence = {}
        target_group = []
        for sequence_id in sequence_ids:
            if sequence_id not in group_by_sequence:
                group_by_sequence[sequence_id] = len(group_by_sequence)
            target_group.append(group_by_sequence[sequence_id])
        batch["sequence_ids"] = sequence_ids
        batch["ligand_ids"] = ligand_ids
        batch["target_group"] = torch.as_tensor(target_group, dtype=torch.long)
        return batch


class TargetAwareBalancedBatchSampler(torch.utils.data.Sampler):
    """Balances datasets while regularly constructing same-target ligand pairs."""

    def __init__(
        self,
        dataset,
        batch_size,
        steps_per_epoch,
        probabilities,
        pair_batch_fraction=0.70,
        seed=20260910,
    ):
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.pair_batch_fraction = float(pair_batch_fraction)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 2:
            raise ValueError("V27 relative training requires batch_size >= 2")
        if not 0.0 <= self.pair_batch_fraction <= 1.0:
            raise ValueError("pair_batch_fraction must be in [0, 1]")
        self.datasets = tuple(DATASET_INDEX)
        probability = np.asarray(
            [float(probabilities[name]) for name in self.datasets], dtype=np.float64
        )
        if np.any(probability < 0) or probability.sum() <= 0:
            raise ValueError("dataset sampling probabilities are invalid")
        self.probability = probability / probability.sum()
        self.indices = {
            name: np.asarray(dataset.dataset_to_indices[name], dtype=np.int64)
            for name in self.datasets
        }
        if any(values.size == 0 for values in self.indices.values()):
            raise ValueError("all configured datasets need training rows")
        grouped = {name: defaultdict(list) for name in self.datasets}
        for index, row in enumerate(dataset.rows):
            grouped[str(row["dataset"])][str(row["sequence_id"])].append(index)
        self.pair_groups = {
            name: tuple(
                np.asarray(values, dtype=np.int64)
                for values in grouped[name].values()
                if len(values) >= 2
            )
            for name in self.datasets
        }

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def _paired_batch(self, name, generator):
        groups = self.pair_groups[name]
        if not groups:
            return None
        result = []
        while len(result) + 1 < self.batch_size:
            group = groups[int(generator.integers(0, len(groups)))]
            selected = generator.choice(group, size=2, replace=False)
            result.extend(int(value) for value in selected)
        if len(result) < self.batch_size:
            result.append(int(generator.choice(self.indices[name])))
        return result

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            name = self.datasets[int(generator.choice(len(self.datasets), p=self.probability))]
            batch = None
            if generator.random() < self.pair_batch_fraction:
                batch = self._paired_batch(name, generator)
            if batch is None:
                batch = generator.choice(
                    self.indices[name], size=self.batch_size, replace=True
                ).tolist()
            yield batch


__all__ = [
    "DATASET_INDEX",
    "OptionalStructureCollator",
    "OptionalStructureDataset",
    "TargetAwareBalancedBatchSampler",
    "canonical_sequence",
    "manifest_fingerprint",
    "move_batch_to_device",
    "read_jsonl",
    "sha256_file",
    "stable_id",
    "write_jsonl",
]
