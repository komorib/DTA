"""Complete-chain ESM/MSA loader without a gathered pocket-MSA field."""

from __future__ import annotations

import numpy as np
import torch

from dta.lp_pdbbind_fullchain_esm_msa_cross_v1.dataset import FullChainESMMSADataset
from dta.msa_pocket_expert_v9.dataset import collate_msa_pocket_complexes


class FullChainESMMSAAllDataset(FullChainESMMSADataset):
    include_local_msa = False


class FullChainESMMSAAllCollator:
    def __call__(self, items):
        if not items:
            raise ValueError("cannot collate an empty complete-chain batch")
        geometry_items = []
        for item in items:
            if ("fullchain_msa_pocket" in item or "pocket_msa" in item
                    or any(key.startswith("pocket_msa_") for key in item)):
                raise ValueError("local MSA input reached the all-chain collator")
            pocket_count = int(np.asarray(item["pocket_chain_index"]).size)
            geometry = dict(item)
            # The established geometry collator requires this legacy slot. It
            # is a temporary all-zero placeholder, removed before model input.
            geometry["pocket_msa"] = np.zeros((pocket_count, 50), dtype=np.float32)
            geometry_items.append(geometry)
        batch = collate_msa_pocket_complexes(geometry_items)
        batch.pop("pocket_msa")
        count = len(items)
        maximum = max(len(item["fullchain_esm"]) for item in items)
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
            nonpocket[index, :length] = torch.as_tensor(
                item["fullchain_nonpocket_mask"], dtype=torch.bool,
            )
            mask[index, :length] = True
        batch["fullchain_esm"] = esm
        batch["fullchain_msa"] = msa
        batch["fullchain_mask"] = mask
        batch["fullchain_nonpocket_mask"] = nonpocket
        return batch


__all__ = ["FullChainESMMSAAllDataset", "FullChainESMMSAAllCollator"]
