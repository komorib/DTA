import copy

import torch

from dta.msa_pocket_expert_v9.model import MSAPocketExpertV9


def _batch():
    batch_size, pocket, ligand = 2, 5, 3
    generator = torch.Generator().manual_seed(7)
    pocket_coords = torch.randn(batch_size, pocket, 3, generator=generator)
    ligand_coords = torch.randn(batch_size, ligand, 3, generator=generator)
    cross = torch.cdist(pocket_coords, ligand_coords)
    return {
        "pocket_msa": torch.randn(batch_size, pocket, 50, generator=generator),
        "pocket_esm": torch.randn(batch_size, pocket, 640, generator=generator),
        "pocket_scalar": torch.randn(batch_size, pocket, 15, generator=generator),
        "pocket_aa": torch.randint(0, 21, (batch_size, pocket), generator=generator),
        "pocket_coords": pocket_coords,
        "pocket_chain": torch.zeros(batch_size, pocket, dtype=torch.long),
        "pocket_position": torch.arange(pocket)[None].expand(batch_size, -1),
        "pocket_mask": torch.ones(batch_size, pocket, dtype=torch.bool),
        "ligand_categorical": torch.zeros(batch_size, ligand, 19),
        "ligand_continuous": torch.randn(batch_size, ligand, 2, generator=generator),
        "ligand_coords": ligand_coords,
        "ligand_mask": torch.ones(batch_size, ligand, dtype=torch.bool),
        "residue_ligand_distance": cross,
        "bond_features": torch.zeros(batch_size, ligand, ligand, 7),
    }


def test_msa_pocket_model_shapes_and_finite_outputs():
    model = MSAPocketExpertV9(model_width=64, layers=2, heads=4, latent_width=128)
    output = model(_batch())
    assert output.prediction.shape == (2,)
    assert output.latent.shape == (2, 128)
    assert output.contact_logits.shape == (2, 5, 3)
    assert torch.isfinite(output.prediction).all()


def test_msa_pocket_model_is_rigid_motion_invariant():
    model = MSAPocketExpertV9(model_width=64, layers=1, heads=4, latent_width=128)
    model.eval()
    batch = _batch()
    moved = copy.deepcopy(batch)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([10.0, -4.0, 3.0])
    moved["pocket_coords"] = batch["pocket_coords"] @ rotation.T + translation
    moved["ligand_coords"] = batch["ligand_coords"] @ rotation.T + translation
    with torch.no_grad():
        original = model(batch).prediction
        transformed = model(moved).prediction
    torch.testing.assert_close(original, transformed, rtol=1e-5, atol=1e-5)
