"""Frozen multiscale spatial and chemical interface features from V13 encoders."""

from pathlib import Path

import torch
from torch.nn import functional as F

from dta.msa_interface_pretrained_v13.model import PocketMSAInterfaceV13
from utils.experiment_integrity import sha256_file


FEATURE_VERSION = "v15-interface-spatial-chemical-1"
DISTANCE_SHELL_BOUNDARIES = (4.5, 6.0, 8.0, 12.0, 24.0)
TOP_CONTACT_COUNTS = (1, 4, 16)
RESIDUE_GROUP_IDS = (
    (1, 5, 8, 10, 11, 18, 19, 20),
    (2, 12, 14, 16, 17),
    (7, 9, 15),
    (3, 4),
    (6, 13),
    (0,),
)
LIGAND_GROUP_ATOMIC_NUMBERS = (
    (6,),
    (7,),
    (8,),
    (15, 16),
    (9, 17, 35, 53),
    (),
)


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
        raise FileNotFoundError(f"V15 {label} checkpoint does not exist: {path}")
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or int(checkpoint.get("schema_version", -1)) != 1:
        raise ValueError(f"V15 {label} checkpoint schema is invalid")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError(f"V15 {label} checkpoint has no config")
    return path, checkpoint


def load_interface_encoders(pretrain_checkpoint, affinity_checkpoint):
    pretrain_path, pretrain = _load_checkpoint(pretrain_checkpoint, "pretrain")
    affinity_path, affinity = _load_checkpoint(affinity_checkpoint, "affinity")
    pretrain_kwargs = _encoder_kwargs(pretrain["config"])
    affinity_kwargs = _encoder_kwargs(affinity["config"])
    if pretrain_kwargs != affinity_kwargs:
        raise ValueError("V15 V13 encoder configurations do not match")

    pretrain_state = pretrain.get("encoder_state")
    if not isinstance(pretrain_state, dict) or not pretrain_state:
        raise ValueError("V15 pretrain checkpoint has no encoder_state")
    affinity_model_state = affinity.get("model_state")
    if not isinstance(affinity_model_state, dict) or not affinity_model_state:
        raise ValueError("V15 affinity checkpoint has no model_state")
    prefix = "msa_pair_encoder."
    affinity_state = {
        name[len(prefix) :]: value
        for name, value in affinity_model_state.items()
        if name.startswith(prefix)
    }
    if not affinity_state:
        raise ValueError("V15 affinity checkpoint has no interface encoder state")

    pretrain_encoder = PocketMSAInterfaceV13(**pretrain_kwargs)
    affinity_encoder = PocketMSAInterfaceV13(**affinity_kwargs)
    try:
        pretrain_encoder.load_state_dict(pretrain_state, strict=True)
        affinity_encoder.load_state_dict(affinity_state, strict=True)
    except RuntimeError as error:
        raise ValueError("V15 V13 encoder checkpoint architecture mismatch") from error
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
        raise ValueError("V15 feature mask shape does not match values")
    expanded = mask[..., None]
    weight = expanded.to(value.dtype)
    count = weight.sum(dim=dimensions).clamp_min(1.0)
    mean = (value * weight).sum(dim=dimensions) / count
    centered = value - mean[
        (slice(None),) + (None,) * (value.ndim - 2) + (slice(None),)
    ]
    variance = (centered.square() * weight).sum(dim=dimensions) / count
    maximum = value.masked_fill(~expanded, -torch.inf).amax(dim=dimensions)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, torch.sqrt(variance.clamp_min(0.0) + 1e-8), maximum


def _scalar_statistics(values, mask, dimensions):
    return torch.cat(_masked_moments(values[..., None], mask, dimensions), dim=-1)


def _membership(values, members):
    selected = torch.zeros_like(values, dtype=torch.bool)
    for member in members:
        selected |= values == int(member)
    return selected


def _occupancy(mask, total):
    count = mask.to(torch.float32).sum(dim=(1, 2), keepdim=False)
    return (count / total).unsqueeze(-1)


def _weighted_pair_moments(pair, scores, mask):
    flat_pair = pair.flatten(1, 2).float()
    flat_scores = scores.flatten(1).float()
    flat_mask = mask.flatten(1)
    weights = flat_scores * flat_mask.to(flat_scores.dtype)
    total = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mean = (flat_pair * weights[..., None]).sum(dim=1) / total
    variance = (
        (flat_pair - mean[:, None, :]).square() * weights[..., None]
    ).sum(dim=1) / total
    return mean, torch.sqrt(variance.clamp_min(0.0) + 1e-8)


def _top_contact_features(pair, scores, mask):
    flat_pair = pair.flatten(1, 2).float()
    flat_scores = scores.flatten(1).float()
    flat_mask = mask.flatten(1)
    ranked = flat_scores.masked_fill(~flat_mask, -torch.inf)
    pair_features = []
    score_features = []
    for requested in TOP_CONTACT_COUNTS:
        count = min(int(requested), int(ranked.shape[1]))
        indices = torch.topk(ranked, count, dim=1).indices
        selected_mask = flat_mask.gather(1, indices)
        selected_count = selected_mask.to(flat_pair.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        selected_pair = flat_pair.gather(
            1, indices[..., None].expand(-1, -1, flat_pair.shape[-1])
        )
        pair_features.append(
            (selected_pair * selected_mask[..., None].to(flat_pair.dtype)).sum(dim=1)
            / selected_count
        )
        selected_scores = flat_scores.gather(1, indices)
        score_features.append(
            (
                selected_scores
                * selected_mask.to(selected_scores.dtype)
            ).sum(dim=1, keepdim=True)
            / selected_count
        )
    return torch.cat(pair_features, dim=-1), torch.cat(score_features, dim=-1)


def _base_features(evolutionary, pair_mask, residue_mask, ligand_mask):
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
    entropy_statistics = _scalar_statistics(
        distance_entropy.flatten(1), pair_mask.flatten(1), (1,)
    )

    contact_probability = torch.sigmoid(
        evolutionary.interface_contact_logits.float()
    )
    contact_statistics = _scalar_statistics(contact_probability, pair_mask, (1, 2))
    residue_contact = contact_probability.masked_fill(~pair_mask, -torch.inf).amax(dim=2)
    ligand_contact = contact_probability.masked_fill(~pair_mask, -torch.inf).amax(dim=1)
    residue_contact = torch.where(
        torch.isfinite(residue_contact), residue_contact, torch.zeros_like(residue_contact)
    )
    ligand_contact = torch.where(
        torch.isfinite(ligand_contact), ligand_contact, torch.zeros_like(ligand_contact)
    )
    residue_contact_statistics = _scalar_statistics(
        residue_contact, residue_mask, (1,)
    )
    ligand_contact_statistics = _scalar_statistics(
        ligand_contact, ligand_mask, (1,)
    )
    return (
        torch.cat(
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
        ),
        distance_entropy,
        contact_probability,
    )


def _shell_features(pair, distances, entropy, contact, pair_mask):
    boundaries = torch.as_tensor(
        DISTANCE_SHELL_BOUNDARIES,
        device=distances.device,
        dtype=distances.dtype,
    )
    shell_ids = torch.bucketize(distances, boundaries)
    total = pair_mask.to(torch.float32).sum(dim=(1, 2)).clamp_min(1.0)
    pair_features = []
    scalar_features = []
    for shell in range(len(DISTANCE_SHELL_BOUNDARIES) + 1):
        selected = pair_mask & (shell_ids == shell)
        pair_features.extend(_masked_moments(pair, selected, (1, 2)))
        scalar_features.append(_occupancy(selected, total))
        scalar_features.append(_scalar_statistics(contact, selected, (1, 2)))
        scalar_features.append(_scalar_statistics(entropy, selected, (1, 2)))
    return torch.cat(pair_features, dim=-1), torch.cat(scalar_features, dim=-1)


def _group_features(pair, pair_mask, pocket_aa, atomic_number):
    total = pair_mask.to(torch.float32).sum(dim=(1, 2)).clamp_min(1.0)
    residue_features = []
    residue_occupancy = []
    for members in RESIDUE_GROUP_IDS:
        selected = pair_mask & _membership(pocket_aa, members)[:, :, None]
        moments = _masked_moments(pair, selected, (1, 2))
        residue_features.extend(moments[:2])
        residue_occupancy.append(_occupancy(selected, total))

    ligand_features = []
    ligand_occupancy = []
    known_ligand = torch.zeros_like(atomic_number, dtype=torch.bool)
    for members in LIGAND_GROUP_ATOMIC_NUMBERS[:-1]:
        known_ligand |= _membership(atomic_number, members)
    ligand_groups = LIGAND_GROUP_ATOMIC_NUMBERS[:-1] + ((),)
    for index, members in enumerate(ligand_groups):
        if index + 1 == len(ligand_groups):
            member_mask = ~known_ligand
        else:
            member_mask = _membership(atomic_number, members)
        selected = pair_mask & member_mask[:, None, :]
        moments = _masked_moments(pair, selected, (1, 2))
        ligand_features.extend(moments[:2])
        ligand_occupancy.append(_occupancy(selected, total))

    return (
        torch.cat(residue_features, dim=-1),
        torch.cat(ligand_features, dim=-1),
        torch.cat(residue_occupancy + ligand_occupancy, dim=-1),
    )


def spatial_feature_dimension(pair_width):
    return 56 * int(pair_width) + 96


def extract_spatial_interface_features(encoder, batch):
    evolutionary = encoder(batch)
    pair_mask = (
        batch["contact_mask"]
        & batch["pocket_mask"][:, :, None]
        & batch["ligand_mask"][:, None, :]
    )
    residue_mask = batch["pocket_mask"] & pair_mask.any(dim=2)
    ligand_mask = batch["ligand_mask"] & pair_mask.any(dim=1)
    base, entropy, contact = _base_features(
        evolutionary, pair_mask, residue_mask, ligand_mask
    )
    shell_pair, shell_scalar = _shell_features(
        evolutionary.interface_pair,
        batch["residue_ligand_distance"],
        entropy,
        contact,
        pair_mask,
    )
    atomic_number = batch["ligand_categorical"][..., 0].long()
    residue_group, ligand_group, group_occupancy = _group_features(
        evolutionary.interface_pair,
        pair_mask,
        batch["pocket_aa"].long(),
        atomic_number,
    )
    weighted = torch.cat(
        _weighted_pair_moments(evolutionary.interface_pair, contact, pair_mask),
        dim=-1,
    )
    top_pair, top_scores = _top_contact_features(
        evolutionary.interface_pair, contact, pair_mask
    )
    features = torch.cat(
        tuple(
            value.float()
            for value in (
                base,
                shell_pair,
                shell_scalar,
                residue_group,
                ligand_group,
                group_occupancy,
                weighted,
                top_pair,
                top_scores,
            )
        ),
        dim=-1,
    )
    expected = spatial_feature_dimension(encoder.pair_width)
    if features.shape[-1] != expected:
        raise RuntimeError(
            f"V15 spatial feature width is {features.shape[-1]}, expected {expected}"
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("V15 spatial interface features contain non-finite values")
    return features


__all__ = [
    "DISTANCE_SHELL_BOUNDARIES",
    "FEATURE_VERSION",
    "LIGAND_GROUP_ATOMIC_NUMBERS",
    "RESIDUE_GROUP_IDS",
    "TOP_CONTACT_COUNTS",
    "extract_spatial_interface_features",
    "load_interface_encoders",
    "spatial_feature_dimension",
]
