import torch

from dta.optional_structure_multitask_v21.losses import optional_structure_loss
from dta.optional_structure_multitask_v21.model import OptionalStructureDTA


def _model():
    torch.manual_seed(7)
    return OptionalStructureDTA(
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
        dropout=0.0,
    )


def _base_batch(batch_size=3):
    torch.manual_seed(11)
    return {
        "protein_embeddings": torch.randn(batch_size, 5, 16),
        "protein_mask": torch.tensor([[1, 1, 1, 1, 1]] * batch_size, dtype=torch.bool),
        "ligand_categorical": torch.randint(0, 2, (batch_size, 4, 8)),
        "ligand_continuous": torch.randn(batch_size, 4, 4),
        "bond_features": torch.randn(batch_size, 4, 4, 7),
        "ligand_mask": torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.bool),
        "dataset_index": torch.arange(batch_size) % 3,
    }


def _add_structure(batch):
    batch = dict(batch)
    batch_size = batch["protein_embeddings"].shape[0]
    torch.manual_seed(13)
    pocket_coords = torch.randn(batch_size, 3, 3)
    ligand_coords = torch.randn(batch_size, 4, 3)
    batch.update(
        {
            "has_3d": torch.tensor([True] + [False] * (batch_size - 1)),
            "pocket_esm": torch.randn(batch_size, 3, 640),
            "pocket_scalar": torch.randn(batch_size, 3, 15),
            "pocket_aa": torch.randint(0, 21, (batch_size, 3)),
            "pocket_coords": pocket_coords,
            "pocket_mask": torch.tensor([[1, 1, 0]] * batch_size, dtype=torch.bool),
            "structure_ligand_categorical": torch.randn(batch_size, 4, 19),
            "structure_ligand_continuous": torch.randn(batch_size, 4, 2),
            "structure_ligand_coords": ligand_coords,
            "structure_ligand_mask": torch.tensor(
                [[1, 1, 1, 0]] * batch_size, dtype=torch.bool
            ),
            "residue_ligand_distance": torch.cdist(pocket_coords, ligand_coords),
        }
    )
    return batch


def test_sequence_only_batch_never_requires_structure_fields():
    model = _model().eval()
    batch = _base_batch()
    output = model(batch)
    forced = model(batch, force_no_3d=True)
    assert not output.effective_has_3d.any()
    assert output.contact_logits is None
    torch.testing.assert_close(output.prediction, output.base_prediction)
    torch.testing.assert_close(output.prediction, forced.prediction)


def test_mixed_batch_hard_routes_and_zero_initialized_delta_is_safe():
    model = _model().eval()
    batch = _add_structure(_base_batch())
    output = model(batch)
    assert output.effective_has_3d.tolist() == [True, False, False]
    assert output.contact_logits.shape == (3, 3, 4)
    assert output.distance_logits.shape[:3] == (3, 3, 4)
    torch.testing.assert_close(output.prediction, output.base_prediction)
    assert torch.count_nonzero(output.structure_latent[1:]) == 0
    assert torch.count_nonzero(output.structure_gate[1:]) == 0


def test_structure_loss_backward_is_finite():
    model = _model().train()
    batch = _add_structure(_base_batch())
    output = model(batch)
    loss, components = optional_structure_loss(
        output, torch.tensor([0.5, -0.3, 0.9]), batch
    )
    loss.backward()
    assert all(torch.isfinite(torch.tensor(value)) for value in components.values())
    gradients = [
        parameter.grad
        for parameter in model.structure_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mixed_batch_cpu_bfloat16_autocast_routes_and_backpropagates():
    class PassThroughTransformer(torch.nn.Module):
        def forward(self, values, **kwargs):
            return values

    model = _model().train()
    model.protein_encoder.encoder = PassThroughTransformer()
    model.structure_encoder.token_encoder = PassThroughTransformer()
    batch = _add_structure(_base_batch())
    routed_latent = torch.randn(3, 48, requires_grad=True)
    dataset_index = torch.tensor([0, 1, 2])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        routed = model._dataset_head(routed_latent, dataset_index, model.base_heads)
        output = model(batch)
        loss, _ = optional_structure_loss(
            output, torch.tensor([0.5, -0.3, 0.9]), batch
        )
        loss = loss + routed.square().mean()
    assert routed.dtype == routed_latent.dtype == torch.float32
    assert output.prediction.dtype == output.base_latent.dtype
    assert output.contact_logits.dtype == output.base_latent.dtype
    assert output.distance_logits.dtype == output.base_latent.dtype
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert routed_latent.grad is not None
    assert torch.isfinite(routed_latent.grad).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)



def test_structure_branch_is_rigid_transform_invariant():
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
    torch.testing.assert_close(baseline.structure_latent, moved.structure_latent, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(baseline.prediction, moved.prediction, atol=2e-6, rtol=2e-6)
