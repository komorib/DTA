import torch

from dta.msa_pocket_expert_v9.model import MSAPocketExpertV9
from dta.msa_pocket_expert_v9.training import TrainingConfig, msa_pocket_loss


def test_msa_pocket_training_step_is_finite():
    batch_size, pocket, ligand = 4, 6, 4
    generator = torch.Generator().manual_seed(11)
    pocket_coords = torch.randn(batch_size, pocket, 3, generator=generator)
    ligand_coords = torch.randn(batch_size, ligand, 3, generator=generator)
    batch = {
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
        "residue_ligand_distance": torch.cdist(pocket_coords, ligand_coords),
        "contact_label": torch.rand(batch_size, pocket, ligand, generator=generator) > 0.7,
        "contact_mask": torch.ones(batch_size, pocket, ligand, dtype=torch.bool),
        "bond_features": torch.zeros(batch_size, ligand, ligand, 7),
    }
    labels = torch.tensor([-1.0, -0.2, 0.4, 1.1])
    config = TrainingConfig(
        device="cpu", model_width=64, layers=1, heads=4, latent_width=128
    )
    model = MSAPocketExpertV9(**config.model_kwargs())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(batch)
    loss, components = msa_pocket_loss(output, labels, batch, config)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in components.values())
    loss.backward()
    optimizer.step()
