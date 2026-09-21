"""Packed graph batches for the opt-in end-to-end hybrid trainer."""

from dataclasses import dataclass

import torch


@dataclass
class PackedInterfaceBatch:
    keys: list
    ligand_atom_features: torch.Tensor
    ligand_coords: torch.Tensor
    ligand_edge_index: torch.Tensor
    ligand_edge_attr: torch.Tensor
    ligand_canonical_rank: torch.Tensor
    ligand_chiral_tag: torch.Tensor
    protein_coords: torch.Tensor
    protein_element: torch.Tensor
    protein_atom_name_class: torch.Tensor
    protein_residue_type: torch.Tensor
    protein_is_backbone: torch.Tensor
    protein_parent_residue: torch.Tensor
    residue_esm: torch.Tensor
    residue_ca_coords: torch.Tensor
    ligand_graph: torch.Tensor
    protein_graph: torch.Tensor
    residue_graph: torch.Tensor
    ligand_offsets: torch.Tensor
    protein_offsets: torch.Tensor
    residue_offsets: torch.Tensor

    @property
    def batch_size(self):
        return len(self.keys)

    def to(self, device):
        return PackedInterfaceBatch(
            **{
                name: (
                    value.to(device)
                    if torch.is_tensor(value)
                    else value
                )
                for name, value in self.__dict__.items()
            }
        )


_REQUIRED = (
    "key",
    "ligand_atom_features",
    "ligand_coords",
    "ligand_edge_index",
    "ligand_edge_attr",
    "ligand_canonical_rank",
    "ligand_chiral_tag",
    "protein_coords",
    "protein_element",
    "protein_atom_name_class",
    "protein_residue_type",
    "protein_is_backbone",
    "protein_parent_residue",
    "residue_esm",
    "residue_ca_coords",
)


def _require_item_tensor(item, name):
    value = item.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"interface item is missing tensor {name}")
    return value


def _offsets(counts, device):
    values = [0]
    for count in counts:
        values.append(values[-1] + int(count))
    return torch.tensor(values, dtype=torch.long, device=device)


def pack_interface_items(items):
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("items must be a nonempty sequence")
    if any(not isinstance(item, dict) for item in items):
        raise ValueError("every interface item must be a mapping")
    for item in items:
        missing = [name for name in _REQUIRED if name not in item]
        if missing:
            raise ValueError("interface item is missing " + ", ".join(missing))

    ligand_counts = [_require_item_tensor(item, "ligand_coords").shape[0] for item in items]
    protein_counts = [_require_item_tensor(item, "protein_coords").shape[0] for item in items]
    residue_counts = [_require_item_tensor(item, "residue_ca_coords").shape[0] for item in items]
    device = _require_item_tensor(items[0], "ligand_coords").device
    ligand_offsets = _offsets(ligand_counts, device)
    protein_offsets = _offsets(protein_counts, device)
    residue_offsets = _offsets(residue_counts, device)
    edge_indices = []
    parent_residues = []
    for index, item in enumerate(items):
        edge_index = _require_item_tensor(item, "ligand_edge_index").long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("ligand_edge_index must have shape [2,E]")
        if edge_index.numel() and (
            int(edge_index.min()) < 0
            or int(edge_index.max()) >= ligand_counts[index]
        ):
            raise ValueError("ligand edge index is out of range")
        edge_indices.append(edge_index + ligand_offsets[index])
        parents = _require_item_tensor(item, "protein_parent_residue").long()
        if parents.shape != (protein_counts[index],):
            raise ValueError("protein_parent_residue does not align with protein")
        if parents.numel() and (
            int(parents.min()) < 0 or int(parents.max()) >= residue_counts[index]
        ):
            raise ValueError("protein_parent_residue is out of range")
        parent_residues.append(parents + residue_offsets[index])

    def concatenate(name):
        values = [_require_item_tensor(item, name) for item in items]
        if any(value.device != device for value in values):
            raise ValueError(f"{name} is on inconsistent devices")
        return torch.cat(values, dim=0)

    return PackedInterfaceBatch(
        keys=[str(item["key"]) for item in items],
        ligand_atom_features=concatenate("ligand_atom_features"),
        ligand_coords=concatenate("ligand_coords"),
        ligand_edge_index=torch.cat(edge_indices, dim=1),
        ligand_edge_attr=concatenate("ligand_edge_attr"),
        ligand_canonical_rank=concatenate("ligand_canonical_rank"),
        ligand_chiral_tag=concatenate("ligand_chiral_tag"),
        protein_coords=concatenate("protein_coords"),
        protein_element=concatenate("protein_element"),
        protein_atom_name_class=concatenate("protein_atom_name_class"),
        protein_residue_type=concatenate("protein_residue_type"),
        protein_is_backbone=concatenate("protein_is_backbone"),
        protein_parent_residue=torch.cat(parent_residues, dim=0),
        residue_esm=concatenate("residue_esm"),
        residue_ca_coords=concatenate("residue_ca_coords"),
        ligand_graph=torch.cat(
            [torch.full((count,), index, dtype=torch.long, device=device) for index, count in enumerate(ligand_counts)]
        ),
        protein_graph=torch.cat(
            [torch.full((count,), index, dtype=torch.long, device=device) for index, count in enumerate(protein_counts)]
        ),
        residue_graph=torch.cat(
            [torch.full((count,), index, dtype=torch.long, device=device) for index, count in enumerate(residue_counts)]
        ),
        ligand_offsets=ligand_offsets,
        protein_offsets=protein_offsets,
        residue_offsets=residue_offsets,
    )



def prepare_packed_hybrid_group(
    samples,
    device,
    *,
    collate_fn=None,
    unpack_fn=None,
):
    """Build one GerNA batch and one packed Interface batch from a group."""
    samples = list(samples)
    if not samples:
        raise ValueError("hybrid group must be nonempty")
    required = ("key", "label", "gerna", "interface")
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError("every hybrid sample must be a mapping")
    for sample in samples:
        missing = [name for name in required if name not in sample]
        if missing:
            raise ValueError("hybrid sample is missing " + ", ".join(missing))
        if str(sample["interface"].get("key")) != str(sample["key"]):
            raise ValueError("interface item key does not match hybrid sample")
    if collate_fn is None:
        from data_utils.dataset import custom_collate_fn

        collate_fn = custom_collate_fn
    if unpack_fn is None:
        from dta.train_dta import unpack_batch_for_device

        unpack_fn = unpack_batch_for_device
    from dta.hybrid_affinity_training import GERNA_INPUT_INDICES

    unpacked = unpack_fn(collate_fn([sample["gerna"] for sample in samples]), device)
    if not isinstance(unpacked, (tuple, list)) or len(unpacked) != 23:
        raise RuntimeError("unpacked GerNA batch must contain 23 fields")
    gerna_inputs = {
        name: unpacked[index]
        for name, index in GERNA_INPUT_INDICES.items()
    }
    packed = pack_interface_items(
        [sample["interface"] for sample in samples]
    ).to(device)
    targets = torch.tensor(
        [float(sample["label"]) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    if not bool(torch.isfinite(targets).all()):
        raise FloatingPointError("hybrid group contains a nonfinite label")
    return [str(sample["key"]) for sample in samples], gerna_inputs, packed, targets
