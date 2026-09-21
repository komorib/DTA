from copy import deepcopy
from types import SimpleNamespace

import torch

from dta.optional_structure_sealed_residual_v29.model import (
    OptionalStructureSealedResidualDTA,
    load_sealed_v22_backbone,
)
from dta.optional_structure_sealed_residual_v29.fit import (
    _reseal_checkpoint_state,
    _update_residual_ema,
)
from dta.optional_structure_sealed_residual_v29.run_training import (
    _archive_equivalence_policy,
    _same_batch_backbone_equivalence,
)
from dta.optional_structure_sealed_residual_v29.training import TrainingConfig


def _small_model():
    return OptionalStructureSealedResidualDTA(
        protein_input_width=16,
        width=32,
        heads=4,
        protein_layers=1,
        ligand_layers=1,
        cross_layers=1,
        latent_width=48,
        pair_width=32,
        structure_self_layers=1,
        structure_pair_layers=1,
        structure_token_layers=1,
        structure_top_pairs=6,
        maximum_protein_tokens=16,
        dataset_count=3,
        adapter_width=16,
        dropout=0.0,
        residual_width=16,
        maximum_residual=0.50,
    )


def test_sealed_v22_loader_is_strict_and_freezes_entire_backbone(tmp_path):
    source = _small_model()
    with torch.no_grad():
        source.backbone.delta_heads[0][-1].weight.fill_(0.03125)
    checkpoint_path = tmp_path / "v22.pt"
    torch.save(
        {
            "version": "optional-structure-staged-v22",
            "phase": "structure",
            "model_kwargs": source.backbone_kwargs(),
            "model_state": source.backbone.state_dict(),
        },
        checkpoint_path,
    )
    target = _small_model()
    checkpoint = load_sealed_v22_backbone(target, checkpoint_path)
    assert checkpoint["phase"] == "structure"
    assert not any(parameter.requires_grad for parameter in target.backbone.parameters())
    assert not target.backbone.training
    for key, value in source.backbone.state_dict().items():
        torch.testing.assert_close(value, target.backbone.state_dict()[key])
    target.train()
    assert not target.backbone.training


def test_sealed_loader_rejects_architecture_or_phase_mismatch(tmp_path):
    source = _small_model()
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "version": "optional-structure-staged-v22",
            "phase": "sequence",
            "model_kwargs": source.backbone_kwargs(),
            "model_state": source.backbone.state_dict(),
        },
        path,
    )
    try:
        load_sealed_v22_backbone(_small_model(), path)
    except ValueError as error:
        assert "structure phase" in str(error)
    else:
        raise AssertionError("V29 accepted a non-structure V22 checkpoint")


def test_v29_defaults_are_bounded_and_lp_residual_only():
    config = TrainingConfig()
    assert config.maximum_residual == 0.75
    assert config.learning_rate == 2.0e-4
    assert config.backbone_anchor_weight == 0.08
    assert config.minimum_residual_ci_gain == -0.001
    assert config.loss_config().delta_l2_weight == 0.01
    assert config.model_kwargs()["maximum_residual"] == 0.75


def test_residual_only_ema_preserves_sealed_backbone_bitwise(tmp_path):
    source = _small_model()
    checkpoint_path = tmp_path / "v22.pt"
    torch.save(
        {
            "version": "optional-structure-staged-v22",
            "phase": "structure",
            "model_kwargs": source.backbone_kwargs(),
            "model_state": source.backbone.state_dict(),
        },
        checkpoint_path,
    )
    raw, ema = _small_model(), _small_model()
    sealed = load_sealed_v22_backbone(raw, checkpoint_path)
    load_sealed_v22_backbone(ema, checkpoint_path)
    with torch.no_grad():
        raw.residual_head.bias.fill_(1.0)
        ema.residual_head.bias.zero_()
    for _ in range(3):
        _update_residual_ema(ema, raw, decay=0.5)
    for name, value in sealed["model_state"].items():
        assert torch.equal(value, raw.backbone.state_dict()[name])
        assert torch.equal(value, ema.backbone.state_dict()[name])
    torch.testing.assert_close(
        ema.residual_head.bias,
        torch.full_like(ema.residual_head.bias, 0.875),
    )


def test_legacy_ema_checkpoint_is_resealed_before_v29_inference(tmp_path):
    source = _small_model()
    checkpoint_path = tmp_path / "v22.pt"
    torch.save(
        {
            "version": "optional-structure-staged-v22",
            "phase": "structure",
            "model_kwargs": source.backbone_kwargs(),
            "model_state": source.backbone.state_dict(),
        },
        checkpoint_path,
    )
    sealed = load_sealed_v22_backbone(_small_model(), checkpoint_path)
    legacy_state = {
        "backbone." + name: value.detach().clone()
        for name, value in sealed["model_state"].items()
    }
    perturbed = "backbone.protein_encoder.adapter.1.weight"
    legacy_state[perturbed].view(-1)[0] += 1.0e-4
    checkpoint = {"model_state": legacy_state}
    assert _reseal_checkpoint_state(checkpoint, sealed)
    for name, value in sealed["model_state"].items():
        assert torch.equal(value, checkpoint["model_state"]["backbone." + name])
    assert not _reseal_checkpoint_state(checkpoint, sealed)


def test_same_padded_batch_audit_matches_independent_v22_reference():
    model = _small_model().eval()
    model.freeze_backbone()
    reference = deepcopy(model.backbone).eval()
    batch = {
        "sample_ids": ("LP_PDBBind:a", "LP_PDBBind:b"),
        "labels": torch.tensor((0.0, 0.0)),
        "dataset_index": torch.tensor((0, 0)),
        "has_3d": torch.tensor((True, True)),
        "protein_embeddings": torch.randn(2, 5, 16),
        "protein_mask": torch.ones(2, 5, dtype=torch.bool),
        "ligand_categorical": torch.randint(0, 2, (2, 4, 8)),
        "ligand_continuous": torch.randn(2, 4, 4),
        "bond_features": torch.randn(2, 4, 4, 7),
        "ligand_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool),
        "pocket_esm": torch.randn(2, 3, 640),
        "pocket_scalar": torch.randn(2, 3, 15),
        "pocket_aa": torch.randint(0, 21, (2, 3)),
        "pocket_coords": torch.randn(2, 3, 3),
        "pocket_mask": torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool),
        "structure_ligand_categorical": torch.randn(2, 4, 19),
        "structure_ligand_continuous": torch.randn(2, 4, 2),
        "structure_ligand_coords": torch.randn(2, 4, 3),
        "structure_ligand_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool),
    }
    batch["residue_ligand_distance"] = torch.cdist(
        batch["pocket_coords"], batch["structure_ligand_coords"]
    )
    audit = _same_batch_backbone_equivalence(
        model, reference, batch, TrainingConfig(device="cpu")
    )
    assert audit["passed"]
    assert audit["count"] == 2
    assert audit["structured_count"] == 2
    assert audit["sample_ids"] == ["LP_PDBBind:a", "LP_PDBBind:b"]


def test_archive_equivalence_is_required_only_for_original_full_batch_geometry(tmp_path):
    checkpoint_path = tmp_path / "v22.pt"
    torch.save({"config": {"evaluation_batch_size": 4}}, checkpoint_path)
    config = TrainingConfig(device="cpu", evaluation_batch_size=4)
    full = _archive_equivalence_policy(
        SimpleNamespace(limit_valid=None), config, checkpoint_path
    )
    limited = _archive_equivalence_policy(
        SimpleNamespace(limit_valid=64), config, checkpoint_path
    )
    different_batch = _archive_equivalence_policy(
        SimpleNamespace(limit_valid=None),
        TrainingConfig(device="cpu", evaluation_batch_size=2),
        checkpoint_path,
    )
    assert full["required"]
    assert not limited["required"]
    assert limited["reason"] == "limited_validation_rebatches_bfloat16_dynamic_padding"
    assert not different_batch["required"]
    assert different_batch["reason"] == "evaluation_batch_size_differs_from_v22_archive"
