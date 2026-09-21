from collections import defaultdict

import torch

from dta.optional_structure_sealed_residual_v29.data import (
    DATASET_INDEX,
    TargetAwareBalancedBatchSampler,
)
from dta.optional_structure_sealed_residual_v29.losses import LossConfig, sealed_residual_loss
from dta.optional_structure_sealed_residual_v29.model import OptionalStructureSealedResidualDTA


def _model():
    torch.manual_seed(29)
    model = OptionalStructureSealedResidualDTA(
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
    with torch.no_grad():
        model.backbone.delta_heads[0][-1].weight.normal_(mean=0.0, std=0.02)
        model.backbone.delta_heads[0][-1].bias.fill_(0.01)
    model.freeze_backbone()
    return model


def _base_batch(batch_size=4):
    torch.manual_seed(11)
    return {
        "protein_embeddings": torch.randn(batch_size, 5, 16),
        "protein_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "ligand_categorical": torch.randint(0, 2, (batch_size, 4, 8)),
        "ligand_continuous": torch.randn(batch_size, 4, 4),
        "bond_features": torch.randn(batch_size, 4, 4, 7),
        "ligand_mask": torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.bool),
        "dataset_index": torch.tensor([0] * batch_size),
        "target_group": torch.tensor([0, 0, 1, 1]),
        "has_3d": torch.zeros(batch_size, dtype=torch.bool),
    }


def _add_structure(batch):
    batch = dict(batch)
    batch_size = batch["protein_embeddings"].shape[0]
    torch.manual_seed(13)
    pocket_coords = torch.randn(batch_size, 3, 3)
    ligand_coords = torch.randn(batch_size, 4, 3)
    pocket_mask = torch.tensor([[1, 1, 0]] * batch_size, dtype=torch.bool)
    ligand_mask = torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.bool)
    batch.update(
        {
            "has_3d": torch.tensor([True, True] + [False] * (batch_size - 2)),
            "pocket_esm": torch.randn(batch_size, 3, 640),
            "pocket_scalar": torch.randn(batch_size, 3, 15),
            "pocket_aa": torch.randint(0, 21, (batch_size, 3)),
            "pocket_coords": pocket_coords,
            "pocket_mask": pocket_mask,
            "structure_ligand_categorical": torch.randn(batch_size, 4, 19),
            "structure_ligand_continuous": torch.randn(batch_size, 4, 2),
            "structure_ligand_coords": ligand_coords,
            "structure_ligand_mask": ligand_mask,
            "residue_ligand_distance": torch.cdist(pocket_coords, ligand_coords),
        }
    )
    return batch


def test_sequence_only_path_never_accesses_structure_and_equals_v22_base():
    model = _model().eval()
    batch = _base_batch()
    batch.pop("has_3d")
    output = model(batch)
    forced = model(batch, force_no_3d=True)
    assert not output.effective_has_3d.any()
    assert output.contact_logits is None
    torch.testing.assert_close(output.prediction, output.base_prediction)
    torch.testing.assert_close(output.prediction, output.backbone_prediction)
    torch.testing.assert_close(output.prediction, forced.prediction)


def test_zero_initialized_residual_preserves_full_v22_structure_path():
    model = _model().eval()
    output = model(_add_structure(_base_batch()))
    assert output.effective_has_3d.tolist() == [True, True, False, False]
    assert output.contact_logits.shape == (4, 3, 4)
    assert output.distance_logits.shape == (4, 3, 4, 16)
    torch.testing.assert_close(output.prediction, output.backbone_prediction)
    assert torch.count_nonzero(output.backbone_prediction[:2] - output.base_prediction[:2]) > 0
    torch.testing.assert_close(output.structure_delta, torch.zeros_like(output.structure_delta))
    assert torch.all(output.structure_gate[:2] > 0.0)
    assert torch.count_nonzero(output.structure_gate[2:]) == 0


def test_residual_is_strictly_bounded_and_never_applies_without_structure():
    model = _model().eval()
    with torch.no_grad():
        model.residual_head.weight.fill_(10.0)
        model.residual_head.bias.fill_(10.0)
    output = model(_add_structure(_base_batch()))
    assert float(output.structure_delta[:2].abs().max()) <= 0.50 + 1.0e-6
    torch.testing.assert_close(output.structure_delta[2:], torch.zeros_like(output.structure_delta[2:]))
    torch.testing.assert_close(output.prediction[2:], output.backbone_prediction[2:])


def test_only_residual_parameters_receive_finite_gradients():
    model = _model().train()
    batch = _add_structure(_base_batch())
    output = model(batch)
    loss, components = sealed_residual_loss(
        output,
        torch.tensor([0.8, -0.5, 0.6, -0.4]),
        batch,
        LossConfig(),
    )
    assert torch.isfinite(loss)
    assert components["relative_pairs"] == 1.0
    loss.backward()
    assert model.residual_head.weight.grad is not None
    assert torch.isfinite(model.residual_head.weight.grad).all()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_structure_path_remains_rigid_transform_invariant():
    model = _model().eval()
    batch = _add_structure(_base_batch())
    baseline = model(batch)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([4.0, -2.0, 7.0])
    transformed = dict(batch)
    transformed["pocket_coords"] = batch["pocket_coords"] @ rotation.T + translation
    transformed["structure_ligand_coords"] = (
        batch["structure_ligand_coords"] @ rotation.T + translation
    )
    transformed["residue_ligand_distance"] = torch.cdist(
        transformed["pocket_coords"], transformed["structure_ligand_coords"]
    )
    moved = model(transformed)
    torch.testing.assert_close(baseline.backbone_prediction, moved.backbone_prediction, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(baseline.prediction, moved.prediction, atol=2e-6, rtol=2e-6)


def test_target_aware_sampler_constructs_distinct_same_target_pairs():
    class Dataset:
        rows = []
        dataset_to_indices = defaultdict(list)

    dataset = Dataset()
    index = 0
    for name in DATASET_INDEX:
        for target in range(2):
            for ligand in range(3):
                dataset.rows.append(
                    {
                        "dataset": name,
                        "sequence_id": "%s-target-%d" % (name, target),
                        "ligand_id": "%s-ligand-%d-%d" % (name, target, ligand),
                    }
                )
                dataset.dataset_to_indices[name].append(index)
                index += 1
    sampler = TargetAwareBalancedBatchSampler(
        dataset,
        batch_size=4,
        steps_per_epoch=12,
        probabilities={"LP_PDBBind": 1.0, "DAVIS": 0.0, "KIBA": 0.0},
        pair_batch_fraction=1.0,
        seed=19,
    )
    for batch in sampler:
        assert len(batch) == 4
        rows = [dataset.rows[value] for value in batch]
        assert {row["dataset"] for row in rows} == {"LP_PDBBind"}
        for offset in (0, 2):
            assert rows[offset]["sequence_id"] == rows[offset + 1]["sequence_id"]
            assert batch[offset] != batch[offset + 1]
