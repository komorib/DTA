import copy

import numpy as np
import torch

from dta.msa_pair_expert_v11.dataset import MSAPairCollator
from dta.msa_pair_expert_v11.model import MSAPairExpertV11
from dta.msa_pair_expert_v11.row_cache import parse_and_select_a3m
from dta.msa_pair_expert_v11.training import TrainingConfig, msa_pair_loss


def _batch():
    batch_size, depth, pocket, ligand = 2, 6, 5, 3
    generator = torch.Generator().manual_seed(17)
    pocket_coords = torch.randn(batch_size, pocket, 3, generator=generator)
    ligand_coords = torch.randn(batch_size, ligand, 3, generator=generator)
    cross = torch.cdist(pocket_coords, ligand_coords)
    row_mask = torch.ones(batch_size, depth, pocket, dtype=torch.bool)
    row_mask[1, 4:, 3:] = False
    return {
        "labels": torch.tensor([5.0, 7.0]),
        "pocket_msa": torch.randn(batch_size, pocket, 50, generator=generator),
        "pocket_msa_tokens": torch.randint(0, 22, (batch_size, depth, pocket), generator=generator),
        "pocket_msa_insertions": torch.rand(batch_size, depth, pocket, generator=generator),
        "pocket_msa_row_features": torch.rand(
            batch_size, depth, pocket, 2, generator=generator
        ),
        "pocket_msa_row_mask": row_mask,
        "pocket_esm": torch.randn(batch_size, pocket, 640, generator=generator),
        "pocket_scalar": torch.randn(batch_size, pocket, 15, generator=generator),
        "pocket_aa": torch.randint(0, 21, (batch_size, pocket), generator=generator),
        "pocket_coords": pocket_coords,
        "pocket_chain": torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]),
        "pocket_position": torch.tensor([[1, 7, 12, 2, 9], [1, 7, 12, 2, 9]]),
        "pocket_mask": torch.ones(batch_size, pocket, dtype=torch.bool),
        "ligand_categorical": torch.zeros(batch_size, ligand, 19),
        "ligand_continuous": torch.randn(batch_size, ligand, 2, generator=generator),
        "ligand_coords": ligand_coords,
        "ligand_mask": torch.ones(batch_size, ligand, dtype=torch.bool),
        "residue_ligand_distance": cross,
        "contact_label": cross <= 4.5,
        "contact_mask": torch.ones(batch_size, pocket, ligand, dtype=torch.bool),
        "bond_features": torch.zeros(batch_size, ligand, ligand, 7),
    }


def _small_model():
    return MSAPairExpertV11(
        model_width=64,
        layers=1,
        heads=4,
        latent_width=96,
        msa_width=32,
        msa_layers=1,
        msa_heads=4,
        pair_factor=4,
        pair_width=16,
        dropout=0.0,
    )


def test_a3m_row_selection_keeps_query_and_limits_depth():
    raw = (
        b">query\nACDE\n"
        b">a\nAC-E\n"
        b">b\nA-DE\n"
        b">c\n----\n"
        b">d\nACDE\n"
        b">e\nAcCDE\n"
    )
    selected = parse_and_select_a3m(raw, "ACDE", maximum_rows=4)
    assert selected["alignment"].shape == (4, 4)
    np.testing.assert_array_equal(selected["alignment"][0], np.asarray([0, 1, 2, 3]))
    assert selected["original_depth"] == 6
    assert np.all(selected["coverage"] >= 0.0)


def test_pair_collator_subsamples_each_chain_independently():
    collator = MSAPairCollator(row_depth=3, training=False, subset_index=0)
    item = {
        "key": "sample",
        "pocket_chain_index": np.asarray([0, 0, 1, 1]),
        "pocket_msa_tokens": np.arange(5 * 4, dtype=np.uint8).reshape(5, 4),
        "pocket_msa_insertions": np.zeros((5, 4), dtype=np.uint8),
        "pocket_msa_row_features": np.zeros((5, 4, 2), dtype=np.float16),
        "pocket_msa_row_mask": np.asarray(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.bool_,
        ),
    }
    selected = collator._subsample(item)
    assert selected["pocket_msa_tokens"].shape == (3, 4)
    assert selected["pocket_msa_row_mask"][:, :2].all()
    assert selected["pocket_msa_row_mask"][:, 2:].all()
    np.testing.assert_array_equal(selected["pocket_msa_tokens"][0], item["pocket_msa_tokens"][0])


def test_msa_pair_model_shapes_loss_and_finite_gradient():
    model = _small_model()
    batch = _batch()
    output = model(batch)
    assert output.prediction.shape == (2,)
    assert output.latent.shape == (2, 96)
    assert output.contact_logits.shape == (2, 5, 3)
    assert output.residue_contact_logits.shape == (2, 5, 5)
    normalized = torch.tensor([-1.0, 1.0])
    loss, components = msa_pair_loss(output, normalized, batch, TrainingConfig())
    assert torch.isfinite(loss)
    assert set(components) == {
        "smooth_l1",
        "mse",
        "rank",
        "contact",
        "residue_contact",
        "affinity_class",
    }
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_msa_pair_model_is_rigid_motion_invariant():
    model = _small_model().eval()
    batch = _batch()
    moved = copy.deepcopy(batch)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([8.0, -3.0, 5.0])
    moved["pocket_coords"] = batch["pocket_coords"] @ rotation.T + translation
    moved["ligand_coords"] = batch["ligand_coords"] @ rotation.T + translation
    with torch.no_grad():
        original = model(batch).prediction
        transformed = model(moved).prediction
    torch.testing.assert_close(original, transformed, rtol=1e-5, atol=1e-5)
