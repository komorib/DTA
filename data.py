"""Data and MoLFormer cache adapter used by the standalone runner."""

from pathlib import Path
import json

import numpy as np
import torch

from dta.lp_pdbbind_fullchain_esm_msa_all_v2.dataset import (
    FullChainESMMSAAllDataset, FullChainESMMSAAllCollator,
)


class MoleculeCache:
    """Load a cache with ``keys`` and FP32 ``embeddings`` arrays.

    Accepted files are ``embeddings.npz`` plus optional ``metadata.json``. The
    keys must match the structural manifest exactly; no fallback or truncation
    is performed.
    """

    def __init__(self, root):
        root = Path(root)
        with np.load(root / "embeddings.npz", allow_pickle=False) as data:
            ids = data["keys"].astype(str).tolist() if "keys" in data else data["unique_ids"].astype(str).tolist()
            vectors = np.asarray(data["embeddings"], dtype=np.float32)
        if vectors.shape != (len(ids), 768) or not np.isfinite(vectors).all():
            raise ValueError("molecule cache must contain finite FP32 [N,768]")
        if len(ids) != len(set(ids)):
            raise ValueError("molecule cache keys are duplicated")
        self.vectors = dict(zip(ids, vectors))
        self.key_to_id = {key: key for key in ids}
        metadata = root / "metadata.json"
        if metadata.exists():
            meta = json.loads(metadata.read_text())
            binding = meta.get("binding", {})
            inputs = binding.get("inputs", {}).get("path")
            if inputs:
                inventory_path = Path(inputs)
                if not inventory_path.is_absolute():
                    inventory_path = metadata.parent / inventory_path
                inventory = json.loads(inventory_path.read_text())["inventory"]
                self.key_to_id = {key: row["unique_id"] for key, row in inventory.items()}

    def get(self, key):
        unique_id = self.key_to_id.get(key, key)
        if unique_id not in self.vectors:
            raise KeyError("missing MoLFormer vector for %s" % key)
        return self.vectors[unique_id].copy()


class EarlyKVDataset(FullChainESMMSAAllDataset):
    def __init__(self, *args, molecule_cache, **kwargs):
        super().__init__(*args, **kwargs)
        self.molecule_cache = molecule_cache

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["molformer_molecule"] = self.molecule_cache.get(str(item["key"]))
        return item


class EarlyKVCollator:
    def __call__(self, items):
        vectors = np.stack([item["molformer_molecule"] for item in items])
        clean_items = [
            {key: value for key, value in item.items() if key != "molformer_molecule"}
            for item in items
        ]
        batch = FullChainESMMSAAllCollator()(clean_items)
        batch["molformer_molecule"] = torch.from_numpy(
            np.asarray(vectors, dtype=np.float32)
        )
        return batch
