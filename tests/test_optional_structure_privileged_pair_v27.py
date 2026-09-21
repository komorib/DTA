from collections import defaultdict

import torch

from dta.optional_structure_privileged_pair_v27.data import (
    DATASET_INDEX,
    TargetAwareBalancedBatchSampler,
)
from dta.optional_structure_privileged_pair_v27.losses import (
    LossConfig,
    privileged_pair_loss,
)
from dta.optional_structure_privileged_pair_v27.model import (
    OptionalStructurePrivilegedPairDTA,
)


def _model():
    torch.manual_seed(7)
    return OptionalStructurePrivilegedPairDTA(
        protein_input_width=16,
        width=32,
        heads=4,
        protein_layers=1,
        ligand_layers=1,
        cross_layers=1,
        latent_width=48,
        pair_width=32,
        pair_factor=8,
        sequence_pair_layers=1,
        sequence_top_pairs=6,
        structure_self_layers=1,
        structure_pair_layers=1,
        structure_token_layers=1,
        structure_top_pairs=6,
        maximum_protein_tokens=16,
        dataset_count=3,
        adapter_width=16,
        dropout=0.0,
    )


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
    pair_mask = pocket_mask[:, :, None] & ligand_mask[:, None, :]
    masked = torch.zeros_like(pair_mask)
    masked[:, 0, 0] = True
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
            "contact_label": torch.cdist(pocket_coords, ligand_coords) <= 1.5,
            "geometry_masked": masked & pair_mask,
            "geometry_visible": pair_mask & ~masked,
        }
    )
    return batch


def test_sequence_only_path_never_accesses_structure_and_is_exact_base():
    model = _model().eval()
    batch = _base_batch()
    batch.pop("has_3d")
    output = model(batch)
    forced = model(batch, force_no_3d=True)
    assert not output.effective_has_3d.any()
    assert output.contact_logits is None
    torch.testing.assert_close(output.prediction, output.base_prediction)
    torch.testing.assert_close(output.prediction, forced.prediction)


def test_optional_geometry_is_fused_in_pair_latent_and_starts_prediction_safe():
    model = _model().eval()
    output = model(_add_structure(_base_batch()))
    assert output.effective_has_3d.tolist() == [True, True, False, False]
    assert output.contact_logits.shape == (4, 3, 4)
    assert output.distance_logits.shape == (4, 3, 4, 16)
    assert output.full_pair_latent.shape == output.sequence_pair_latent.shape
    assert torch.count_nonzero(
        output.full_pair_latent[:2] - output.sequence_pair_latent[:2]
    ) > 0
    assert torch.count_nonzero(
        output.full_pair_latent[2:] - output.sequence_pair_latent[2:]
    ) == 0
    torch.testing.assert_close(output.prediction, output.base_prediction)
    torch.testing.assert_close(output.structure_gate[:2], torch.ones(2))


def test_relative_head_is_antisymmetric_by_construction():
    model = _model().eval()
    left = torch.randn(3, 48)
    right = torch.randn(3, 48)
    dataset = torch.tensor([0, 1, 2])
    forward = model.relative_difference(left, right, dataset)
    reverse = model.relative_difference(right, left, dataset)
    torch.testing.assert_close(forward, -reverse)


def test_masked_geometry_and_privileged_distillation_backward_are_finite():
    model = _model().train()
    batch = _add_structure(_base_batch())
    output = model(batch)
    thresholds = torch.tensor([float("nan"), 0.0, 0.0])
    positive_weights = torch.ones(3)
    loss, components = privileged_pair_loss(
        model,
        output,
        torch.tensor([0.8, -0.5, 0.6, -0.4]),
        batch,
        thresholds,
        positive_weights,
        LossConfig(),
    )
    assert torch.isfinite(loss)
    assert components["relative_pairs"] == 2
    assert components["distance"] > 0
    assert components["pair_distillation"] > 0
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.sequence_pair_encoder.contact_head.weight.grad is not None
    assert model.structure_encoder.distance_head.weight.grad is not None


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
    torch.testing.assert_close(baseline.full_pair_latent, moved.full_pair_latent, atol=2e-6, rtol=2e-6)
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
        probabilities={name: 1.0 for name in DATASET_INDEX},
        pair_batch_fraction=1.0,
        seed=19,
    )
    for batch in sampler:
        assert len(batch) == 4
        rows = [dataset.rows[value] for value in batch]
        assert len({row["dataset"] for row in rows}) == 1
        for offset in (0, 2):
            assert rows[offset]["sequence_id"] == rows[offset + 1]["sequence_id"]
            assert batch[offset] != batch[offset + 1]
