import math
import numbers

import numpy as np
import torch
from torch.utils.data import Sampler

from utils.experiment_integrity import hash_ordered_keys


__all__ = [
    "unique_key_index",
    "validate_inventory",
    "HybridAffinityDataset",
    "collate_hybrid_group",
    "LabelSpreadBatchSampler",
    "preflight_hybrid_dataset",
]


def unique_key_index(samples, source_name):
    index = {}
    for position, sample in enumerate(samples):
        key = str(sample["lmdb_key"])
        if key in index:
            raise ValueError(f"{source_name} contains duplicate key {key}")
        index[key] = position
    return index


def validate_inventory(manifest_samples, gerna_index, interface_index):
    manifest_index = unique_key_index(manifest_samples, "canonical manifest")
    keys = list(manifest_index)
    for key in keys:
        if key not in gerna_index:
            raise ValueError(f"GerNA is missing key {key}")
        if key not in interface_index:
            raise ValueError(f"interface is missing key {key}")
    return keys


def _scalar_label(value, source, key):
    if torch.is_tensor(value):
        if value.ndim != 0:
            raise ValueError(f"{source} label for key {key} must be scalar")
        value = value.item()
    elif not isinstance(value, numbers.Number):
        raise ValueError(f"{source} label for key {key} must be numeric")

    try:
        label = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{source} label for key {key} must be numeric") from error

    if not math.isfinite(label):
        raise ValueError(f"{source} label for key {key} is nonfinite")
    return label


class HybridAffinityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_samples,
        gerna_dataset,
        interface_dataset,
        label_tolerance=1e-6,
    ):
        try:
            tolerance = float(label_tolerance)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("label_tolerance must be finite and nonnegative") from error
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("label_tolerance must be finite and nonnegative")

        self.samples = list(manifest_samples)
        self.gerna_dataset = gerna_dataset
        self.interface_dataset = interface_dataset
        self.gerna_index = unique_key_index(gerna_dataset.samples, "GerNA")
        self.interface_index = unique_key_index(
            interface_dataset.samples, "interface"
        )
        self.keys = validate_inventory(
            self.samples, self.gerna_index, self.interface_index
        )

        labels = []
        for key, sample in zip(self.keys, self.samples):
            labels.append(
                _scalar_label(sample["label"], "canonical manifest", key)
            )
        self.labels = np.asarray(labels, dtype=np.float64)
        self.label_tolerance = tolerance

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        label = float(self.labels[index])
        gerna_item = self.gerna_dataset[self.gerna_index[key]]
        interface_item = self.interface_dataset[self.interface_index[key]]

        if not isinstance(gerna_item, (tuple, list)) or len(gerna_item) <= 9:
            raise ValueError(
                f"GerNA item for key {key} must be a tuple or list with label at index 9"
            )
        gerna_label = _scalar_label(gerna_item[9], "GerNA", key)

        if not isinstance(interface_item, dict):
            raise ValueError(f"interface item for key {key} must be a dict")
        if "key" not in interface_item or str(interface_item["key"]) != key:
            raise ValueError(f"interface returned the wrong key for {key}")
        if "label" not in interface_item:
            raise ValueError(f"interface label is missing for key {key}")
        interface_label = _scalar_label(interface_item["label"], "interface", key)

        for source, actual in (
            ("GerNA", gerna_label),
            ("interface", interface_label),
        ):
            if abs(actual - label) > self.label_tolerance:
                raise ValueError(
                    f"{source} label mismatch for key {key}: {actual} != {label}"
                )

        return {
            "key": key,
            "label": label,
            "gerna": gerna_item,
            "interface": interface_item,
        }


def collate_hybrid_group(batch):
    return list(batch)


def _validated_integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} must be an integer")
    return result


class LabelSpreadBatchSampler(Sampler):
    def __init__(self, labels, group_size, seed, min_gap):
        label_array = np.asarray(labels, dtype=np.float64)
        if label_array.ndim != 1 or label_array.size == 0:
            raise ValueError("labels must be a nonempty one-dimensional array")
        if not np.isfinite(label_array).all():
            raise ValueError("labels must contain only finite values")
        group_size = _validated_integer(group_size, "group_size", minimum=2)
        try:
            minimum_gap = float(min_gap)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("min_gap must be finite and nonnegative") from error
        if not math.isfinite(minimum_gap) or minimum_gap < 0.0:
            raise ValueError("min_gap must be finite and nonnegative")

        self.labels = label_array.copy()
        self.group_size = group_size
        self.seed = _validated_integer(seed, "seed", maximum=2 ** 32 - 1)
        self.min_gap = minimum_gap
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = _validated_integer(epoch, "epoch")

    def __len__(self):
        return int(math.ceil(len(self.labels) / float(self.group_size)))

    def __iter__(self):
        order = np.argsort(self.labels, kind="mergesort").tolist()
        pairs = []
        fallback = []
        left, right = 0, len(order) - 1
        while left < right:
            low_index, high_index = order[left], order[right]
            if self.labels[high_index] - self.labels[low_index] < self.min_gap:
                fallback.extend(order[left : right + 1])
                break
            pairs.append([low_index, high_index])
            left += 1
            right -= 1
        else:
            if left == right:
                fallback.append(order[left])

        effective_seed = (self.seed + 1000003 * self.epoch) % (2 ** 32)
        rng = np.random.RandomState(effective_seed)
        rng.shuffle(pairs)
        for pair in pairs:
            if rng.randint(0, 2):
                pair.reverse()
        rng.shuffle(fallback)

        spread_order = [index for pair in pairs for index in pair]
        spread_order.extend(fallback)

        for start in range(0, len(spread_order), self.group_size):
            yield spread_order[start : start + self.group_size]


REQUIRED_INTERFACE_FIELDS = frozenset(
    {
        "ligand_atom_features",
        "ligand_coords",
        "protein_coords",
        "residue_esm",
        "residue_ca_coords",
    }
)


def _numeric_size_and_validate(value, key, path):
    if torch.is_tensor(value):
        size = int(value.numel())
        if size and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite numeric value for {key}: {path}")
        return size
    if isinstance(value, np.ndarray):
        size = int(value.size)
        numeric = value.dtype.kind in {"b", "i", "u", "f", "c"}
        if value.dtype.kind == "O":
            return _validate_nested_numeric(value.tolist(), key, path)
        if size and numeric and not bool(np.isfinite(value).all()):
            raise ValueError(f"nonfinite numeric value for {key}: {path}")
        return size if numeric else 0
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, numbers.Integral):
        return 1
    if isinstance(value, numbers.Complex):
        try:
            real = float(value.real)
            imaginary = float(value.imag)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid numeric value for {key}: {path}") from error
        if not math.isfinite(real) or not math.isfinite(imaginary):
            raise ValueError(f"nonfinite numeric value for {key}: {path}")
        return 1
    if isinstance(value, numbers.Number):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid numeric value for {key}: {path}") from error
        if not finite:
            raise ValueError(f"nonfinite numeric value for {key}: {path}")
        return 1
    return None


def _validate_nested_numeric(value, key, path):
    numeric_size = _numeric_size_and_validate(value, key, path)
    if numeric_size is not None:
        return numeric_size
    if isinstance(value, dict):
        return sum(
            _validate_nested_numeric(nested, key, f"{path}.{name}")
            for name, nested in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _validate_nested_numeric(nested, key, f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _validate_nested_numeric(to_dict(), key, path)

    graph_size = 0
    graph_storage_found = False
    for storage_name in ("ndata", "edata"):
        storage = getattr(value, storage_name, None)
        if storage is None:
            continue
        items = getattr(storage, "items", None)
        if not callable(items):
            raise ValueError(
                f"unsupported graph storage for {key}: {path}.{storage_name}"
            )
        graph_storage_found = True
        graph_size += _validate_nested_numeric(
            dict(items()), key, f"{path}.{storage_name}"
        )
    if graph_storage_found:
        return graph_size

    if hasattr(value, "num_nodes"):
        raise ValueError(f"unsupported graph-like object for {key}: {path}")
    return 0


def _validate_required_interface_field(value, key, field):
    path = f"sample.interface.{field}"
    if not torch.is_tensor(value) and not isinstance(value, np.ndarray):
        raise ValueError(f"required numeric field is invalid for {key}: {path}")
    if isinstance(value, np.ndarray) and value.dtype.kind not in {
        "b", "i", "u", "f", "c"
    }:
        raise ValueError(f"required numeric field is invalid for {key}: {path}")
    size = _numeric_size_and_validate(value, key, path)
    if size is None or size <= 0:
        raise ValueError(f"required numeric field is empty for {key}: {path}")


def preflight_hybrid_dataset(dataset, largest_count=4):
    largest_count = _validated_integer(
        largest_count, "largest_count", minimum=1
    )

    keys = [str(key) for key in dataset.keys]
    if len(keys) != len(dataset):
        raise ValueError("dataset keys must align with dataset length")
    complexities = []
    for index, key in enumerate(keys):
        item = dataset[index]
        if not isinstance(item, dict):
            raise ValueError(f"hybrid sample must be a dict for {key}")
        interface = item.get("interface")
        if not isinstance(interface, dict):
            raise ValueError(f"interface sample must be a dict for {key}")
        for field in sorted(REQUIRED_INTERFACE_FIELDS):
            if field not in interface:
                raise ValueError(
                    f"required interface field is missing for {key}: {field}"
                )
            _validate_required_interface_field(interface[field], key, field)
        complexities.append(_validate_nested_numeric(item, key, "sample"))

    ranked = sorted(
        range(len(keys)),
        key=lambda index: (-complexities[index], index),
    )
    largest_indices = ranked[: min(largest_count, len(ranked))]
    return {
        "sample_count": len(keys),
        "ordered_key_sha256": hash_ordered_keys(keys),
        "largest_indices": largest_indices,
        "largest_keys": [keys[index] for index in largest_indices],
    }
