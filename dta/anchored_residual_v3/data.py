"""Teacher-aware dataset wrappers built on unchanged v2 packing."""

import math

import torch


class TeacherAnchoredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, teacher_lookup):
        self.dataset = dataset
        self.teacher_lookup = teacher_lookup
        raw_keys = getattr(dataset, "keys", None)
        if raw_keys is None:
            raw_keys = [dataset[index]["key"] for index in range(len(dataset))]
        self.keys = tuple(str(key) for key in raw_keys)
        if len(self.keys) != len(dataset) or len(set(self.keys)) != len(self.keys):
            raise ValueError("anchored dataset keys must be unique and aligned")
        for key in self.keys:
            teacher_lookup.value_for(key)
        if hasattr(dataset, "labels"):
            self.labels = dataset.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source = self.dataset[index]
        if not isinstance(source, dict) or "key" not in source:
            raise ValueError("anchored source sample must contain key")
        sample = dict(source)
        sample["teacher_prediction"] = self.teacher_lookup.value_for(sample["key"])
        return sample


def prepare_anchored_group(samples, device, *, prepare_fn=None):
    samples = list(samples)
    if not samples:
        raise ValueError("anchored group must be nonempty")
    teacher_values = []
    for sample in samples:
        key = str(sample.get("key", "<missing>")) if isinstance(sample, dict) else "<missing>"
        if not isinstance(sample, dict) or "teacher_prediction" not in sample:
            raise ValueError(f"sample {key} is missing teacher_prediction")
        try:
            value = float(sample["teacher_prediction"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"sample {key} teacher_prediction must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"sample {key} teacher_prediction must be finite")
        teacher_values.append(value)
    if prepare_fn is None:
        from data_utils.packed_hybrid_batch import prepare_packed_hybrid_group

        prepare_fn = prepare_packed_hybrid_group
    keys, gerna_inputs, packed_interface, targets = prepare_fn(samples, device)
    expected_keys = [str(sample["key"]) for sample in samples]
    if list(keys) != expected_keys:
        raise RuntimeError("packed group changed anchored sample order")
    teacher = torch.tensor(
        teacher_values, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    return keys, gerna_inputs, packed_interface, targets, teacher
