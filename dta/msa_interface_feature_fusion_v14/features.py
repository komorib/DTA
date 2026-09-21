"""Frozen multi-scale interface features from V13 pretrain and affinity encoders."""

from pathlib import Path

import torch
from torch.nn import functional as F

from dta.msa_interface_pretrained_v13.model import PocketMSAInterfaceV13
from utils.experiment_integrity import sha256_file


FEATURE_VERSION = "v14-interface-multiscale-1"


def _encoder_kwargs(config):
    return {
        "width": int(config["msa_width"]),
        "layers": int(config["msa_layers"]),
        "heads": int(config["msa_heads"]),
        "pair_factor": int(config["pair_factor"]),
        "pair_width": int(config["pair_width"]),
        "interface_blocks": int(config["interface_blocks"]),
        "dropout": float(config["dropout"]),
    }


def _load_checkpoint(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V14 {label} checkpoint does not exist: {path}")
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or int(checkpoint.get("schema_version", -1)) != 1:
        raise ValueError(f"V14 {label} checkpoint schema is invalid")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError(f"V14 {label} checkpoint has no config")
    return path, checkpoint


def load_interface_encoders(pretrain_checkpoint, affinity_checkpoint):
    pretrain_path, pretrain = _load_checkpoint(pretrain_checkpoint, "pretrain")
    affinity_path, affinity = _load_checkpoint(affinity_checkpoint, "affinity")
    pretrain_kwargs = _encoder_kwargs(pretrain["config"])
    affinity_kwargs = _encoder_kwargs(affinity["config"])
    if pretrain_kwargs != affinity_kwargs:
        raise ValueError("V14 V13 encoder configurations do not match")

    pretrain_state = pretrain.get("encoder_state")
    if not isinstance(pretrain_state, dict) or not pretrain_state:
        raise ValueError("V14 pretrain checkpoint has no encoder_state")
    affinity_model_state = affinity.get("model_state")
    if not isinstance(affinity_model_state, dict) or not affinity_model_state:
        raise ValueError("V14 affinity checkpoint has no model_state")
    prefix = "msa_pair_encoder."
    affinity_state = {
        name[len(prefix) :]: value
        for name, value in affinity_model_state.items()
        if name.startswith(prefix)
    }
    if not affinity_state:
        raise ValueError("V14 affinity checkpoint has no interface encoder state")

    pretrain_encoder = PocketMSAInterfaceV13(**pretrain_kwargs)
    affinity_encoder = PocketMSAInterfaceV13(**affinity_kwargs)
    try:
        pretrain_encoder.load_state_dict(pretrain_state, strict=True)
        affinity_encoder.load_state_dict(affinity_state, strict=True)
    except RuntimeError as error:
        raise ValueError("V14 V13 encoder checkpoint architecture mismatch") from error
    lineage = {
        "feature_version": FEATURE_VERSION,
        "encoder_kwargs": pretrain_kwargs,
        "pretrain": {
            "path": str(pretrain_path),
            "sha256": sha256_file(pretrain_path),
            "epoch": int(pretrain.get("epoch", -1)),
            "variant": str(pretrain.get("variant", "unknown")),
            "metrics": pretrain.get("metrics"),
        },
        "affinity": {
            "path": str(affinity_path),
            "sha256": sha256_file(affinity_path),
            "epoch": int(affinity.get("epoch", -1)),
            "variant": str(affinity.get("variant", "unknown")),
            "metrics": affinity.get("metrics"),
        },
    }
    return pretrain_encoder, affinity_encoder, lineage


def _masked_moments(value, mask, dimensions):
    if value.shape[:-1] != mask.shape:
        raise ValueError("V14 feature mask shape does not match values")
    expanded = mask[..., None]
    weight = expanded.to(value.dtype)
    count = weight.sum(dim=dimensions).clamp_min(1.0)
    mean = (value * weight).sum(dim=dimensions) / count
    centered = value - mean[(slice(None),) + (None,) * (value.ndim - 2) + (slice(None),)]
    variance = (centered.square() * weight).sum(dim=dimensions) / count
    maximum = value.masked_fill(~expanded, -torch.inf).amax(dim=dimensions)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, torch.sqrt(variance.clamp_min(0.0) + 1e-8), maximum


def _scalar_side_statistics(values, mask):
    return torch.cat(_masked_moments(values[..., None], mask, (1,)), dim=-1)


def interface_feature_dimension(pair_width):
    return 9 * int(pair_width) + 39


def extract_interface_features(encoder, batch):
    evolutionary = encoder(batch)
    pair_mask = (
        batch["contact_mask"]
        & batch["pocket_mask"][:, :, None]
        & batch["ligand_mask"][:, None, :]
    )
    residue_mask = batch["pocket_mask"] & pair_mask.any(dim=2)
    ligand_mask = batch["ligand_mask"] & pair_mask.any(dim=1)

    pair_statistics = torch.cat(
        _masked_moments(evolutionary.interface_pair, pair_mask, (1, 2)), dim=-1
    )
    residue_statistics = torch.cat(
        _masked_moments(evolutionary.interface_summary, residue_mask, (1,)), dim=-1
    )
    ligand_statistics = torch.cat(
        _masked_moments(evolutionary.ligand_summary, ligand_mask, (1,)), dim=-1
    )

    distance_probability = F.softmax(
        evolutionary.interface_distance_logits.float(), dim=-1
    )
    distance_statistics = torch.cat(
        _masked_moments(distance_probability, pair_mask, (1, 2)), dim=-1
    )
    distance_entropy = -(
        distance_probability * distance_probability.clamp_min(1e-8).log()
    ).sum(dim=-1)
    entropy_statistics = _scalar_side_statistics(distance_entropy.flatten(1), pair_mask.flatten(1))

    contact_probability = torch.sigmoid(
        evolutionary.interface_contact_logits.float()
    )
    contact_statistics = torch.cat(
        _masked_moments(contact_probability[..., None], pair_mask, (1, 2)), dim=-1
    )
    residue_contact = contact_probability.masked_fill(~pair_mask, -torch.inf).amax(dim=2)
    ligand_contact = contact_probability.masked_fill(~pair_mask, -torch.inf).amax(dim=1)
    residue_contact = torch.where(
        torch.isfinite(residue_contact), residue_contact, torch.zeros_like(residue_contact)
    )
    ligand_contact = torch.where(
        torch.isfinite(ligand_contact), ligand_contact, torch.zeros_like(ligand_contact)
    )
    residue_contact_statistics = _scalar_side_statistics(residue_contact, residue_mask)
    ligand_contact_statistics = _scalar_side_statistics(ligand_contact, ligand_mask)

    features = torch.cat(
        (
            pair_statistics,
            residue_statistics,
            ligand_statistics,
            distance_statistics,
            entropy_statistics,
            contact_statistics,
            residue_contact_statistics,
            ligand_contact_statistics,
        ),
        dim=-1,
    )
    expected = interface_feature_dimension(encoder.pair_width)
    if features.shape[-1] != expected:
        raise RuntimeError(
            f"V14 interface feature width is {features.shape[-1]}, expected {expected}"
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("V14 interface features contain non-finite values")
    return features


__all__ = [
    "FEATURE_VERSION",
    "extract_interface_features",
    "interface_feature_dimension",
    "load_interface_encoders",
]
