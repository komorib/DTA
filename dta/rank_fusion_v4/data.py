"""In-memory cache dataset and label-spread sampling for Rank Fusion v4."""

import math

import numpy as np
import torch
from torch.utils.data import Sampler


class FusionCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache.keys)

    def __getitem__(self, index):
        return {
            "key": self.cache.keys[index],
            "label": self.cache.labels[index],
            "raw_predictions": self.cache.predictions[index],
            "standardized_predictions": self.cache.standardized_predictions[index],
            "available": self.cache.available[index],
            "latents": (
                self.cache.upper_bound_latents[index],
                self.cache.boltz_latents[index],
                self.cache.three_dprot_latents[index],
            ),
        }


def collate_fusion_cache(records):
    if not records:
        raise ValueError("fusion cache batch must be nonempty")
    return {
        "keys": tuple(record["key"] for record in records),
        "labels": torch.as_tensor(
            np.asarray([record["label"] for record in records]),
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
                np.stack([record["latents"][expert] for record in records]),
                dtype=torch.float32,
            )
            for expert in range(3)
        ],
    }


class LabelSpreadBatchSampler(Sampler):
    """Interleave shuffled label quartiles and cover every row exactly once."""

    def __init__(self, labels, group_size, seed):
        labels = np.asarray(labels, dtype=np.float64)
        if labels.ndim != 1 or labels.size == 0 or not np.isfinite(labels).all():
            raise ValueError("labels must be a finite nonempty vector")
        if isinstance(group_size, bool) or int(group_size) < 2:
            raise ValueError("group_size must be at least two")
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.labels = labels.copy()
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def __len__(self):
        return int(math.ceil(len(self.labels) / float(self.group_size)))

    def __iter__(self):
        order = np.argsort(self.labels, kind="mergesort")
        bins = [part.tolist() for part in np.array_split(order, 4)]
        generator = np.random.RandomState(
            (self.seed + 1000003 * self.epoch) % (2 ** 32)
        )
        for part in bins:
            generator.shuffle(part)
        interleaved = []
        maximum = max(len(part) for part in bins)
        for offset in range(maximum):
            for part in bins:
                if offset < len(part):
                    interleaved.append(part[offset])
        for start in range(0, len(interleaved), self.group_size):
            yield interleaved[start : start + self.group_size]
