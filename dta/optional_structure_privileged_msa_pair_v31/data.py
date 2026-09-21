"""V31 datasets, train-only teacher attachment, and LP pair-aware sampling."""

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
    """Adds relative-training identities and optional train-only teacher labels."""

    def __init__(self, *args, teacher_predictions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_predictions = None
        if teacher_predictions is not None:
            if not isinstance(teacher_predictions, dict):
                raise TypeError("V31 teacher predictions must be a sample-ID mapping")
            missing = [
                str(row["sample_id"])
                for row in self.rows
                if str(row["dataset"]) == "LP_PDBBind"
                and str(row["sample_id"]) not in teacher_predictions
            ]
            if missing:
                raise ValueError("V31 train teacher misses LP samples: %s" % missing[:3])
            self.teacher_predictions = {
                str(key): float(value) for key, value in teacher_predictions.items()
            }

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item["sequence_id"] = str(row["sequence_id"])
        item["ligand_id"] = str(row["ligand_id"])
        if self.teacher_predictions is not None and str(row["dataset"]) == "LP_PDBBind":
            item["teacher_prediction"] = self.teacher_predictions[str(row["sample_id"])]
        return item


class OptionalStructureCollator(_V22Collator):
    """Preserves identities and attaches scalar teacher targets when present."""

    def __init__(self, maximum_ligand_atoms=128, maximum_pocket_tokens=256):
        super().__init__(maximum_ligand_atoms, maximum_pocket_tokens)

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
        available = torch.as_tensor(
            ["teacher_prediction" in item for item in items], dtype=torch.bool
        )
        teacher = torch.zeros(len(items), dtype=torch.float32)
        for index, item in enumerate(items):
            if "teacher_prediction" in item:
                teacher[index] = float(item["teacher_prediction"])
        batch.update(
            {
                "sequence_ids": sequence_ids,
                "ligand_ids": ligand_ids,
                "target_group": torch.as_tensor(target_group, dtype=torch.long),
                "teacher_predictions": teacher,
                "teacher_available": available,
            }
        )
        return batch


class TargetAwareBalancedBatchSampler(torch.utils.data.Sampler):
    """Creates LP batches with same-target ligand pairs for rank supervision."""

    def __init__(
        self,
        dataset,
        batch_size,
        steps_per_epoch,
        probabilities,
        pair_batch_fraction=0.70,
        seed=20260917,
    ):
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.pair_batch_fraction = float(pair_batch_fraction)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 2:
            raise ValueError("V31 relative training requires batch_size >= 2")
        if not 0.0 <= self.pair_batch_fraction <= 1.0:
            raise ValueError("V31 pair_batch_fraction must be in [0, 1]")
        self.datasets = tuple(DATASET_INDEX)
        probability = np.asarray(
            [float(probabilities[name]) for name in self.datasets], dtype=np.float64
        )
        if np.any(probability < 0.0) or probability.sum() <= 0.0:
            raise ValueError("V31 dataset sampling probabilities are invalid")
        self.probability = probability / probability.sum()
        self.indices = {
            name: np.asarray(dataset.dataset_to_indices[name], dtype=np.int64)
            for name in self.datasets
        }
        # A caller may deliberately train this LP-only residual on a reduced
        # manifest that omits Davis/KIBA.  Only a dataset with positive
        # sampling probability needs rows; zero-probability datasets are never
        # drawn by ``__iter__`` and must not make a valid LP smoke fail.
        missing = [
            name
            for position, name in enumerate(self.datasets)
            if self.probability[position] > 0.0 and self.indices[name].size == 0
        ]
        if missing:
            raise ValueError("V31 sampled datasets need training rows: %s" % missing)
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
            result.extend(
                int(value) for value in generator.choice(group, size=2, replace=False)
            )
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
