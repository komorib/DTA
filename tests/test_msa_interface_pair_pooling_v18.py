import copy
from types import SimpleNamespace

import numpy as np
import torch

from dta.msa_pair_expert_v11.dataset import MSAPairCollator
from dta.msa_pair_expert_v11.row_cache import parse_and_select_a3m
from dta.msa_pair_pretrained_v12.model import MSAPairExpertV12
from dta.msa_pair_pretrained_v12.pretraining import MSAPairPretrainerV12
from dta.msa_interface_pair_pooling_v18.model import (
    ContactAwarePairTokenPoolV18,
    MSAInterfacePairPoolExpertV18,
    PocketMSAInterfaceV13,
)
from dta.msa_interface_pretrained_v13 import pretraining as pretraining_module
from dta.msa_interface_pretrained_v13.pretraining import (
    MSAInterfacePretrainerV13,
    PretrainingConfig,
    _load_initial_msa_encoder,
    pretraining_loss,
)
from dta.msa_interface_pair_pooling_v18.training import (
    TrainingConfig,
    _load_affinity_initialization,
    _load_pretrained_encoder,
    msa_interface_loss,
)


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
    return MSAInterfacePairPoolExpertV18(
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
        pair_token_width=32,
        pair_token_heads=4,
        pair_token_layers=1,
        pair_token_count=8,
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


def test_msa_interface_model_shapes_loss_and_finite_gradient():
    model = _small_model()
    batch = _batch()
    output = model(batch)
    assert output.prediction.shape == (2,)
    assert output.latent.shape == (2, 192)
    assert output.contact_logits.shape == (2, 5, 3)
    assert output.residue_contact_logits.shape == (2, 5, 5)
    assert output.distogram_logits.shape == (2, 5, 5, 10)
    assert output.interface_distance_logits.shape == (2, 5, 3, 9)
    assert output.interface_contact_logits.shape == (2, 5, 3)
    assert output.match_logits.shape == (2, 2)
    normalized = torch.tensor([-1.0, 1.0])
    loss, components = msa_interface_loss(
        output, normalized, batch, TrainingConfig(pretrained_checkpoint="unused")
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "smooth_l1",
        "mse",
        "rank",
        "contact",
        "residue_contact",
        "affinity_class",
        "distogram",
        "interface_distogram",
        "interface_contact",
        "match",
    }
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_msa_interface_model_is_rigid_motion_invariant():
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


def _small_encoder_kwargs():
    return {
        "width": 32,
        "layers": 1,
        "heads": 4,
        "pair_factor": 4,
        "pair_width": 16,
        "dropout": 0.0,
    }


def test_masked_msa_interface_pretraining_has_finite_gradients():
    torch.manual_seed(23)
    model = MSAInterfacePretrainerV13(_small_encoder_kwargs(), mask_probability=0.20)
    batch = _batch()
    output = model(batch)
    assert output.token_logits.shape == (2, 6, 5, 22)
    assert output.token_mask.any()
    assert output.evolutionary.distogram_logits.shape == (2, 5, 5, 10)
    config = PretrainingConfig(
        device="cpu",
        msa_width=32,
        msa_layers=1,
        msa_heads=4,
        pair_factor=4,
        pair_width=16,
    )
    loss, components = pretraining_loss(output, batch, config)
    assert torch.isfinite(loss)
    assert set(components) == {
        "masked_msa",
        "token_accuracy",
        "distogram",
        "distogram_accuracy",
        "residue_contact",
        "interface_distogram",
        "interface_distogram_accuracy",
        "interface_contact",
        "match",
    }
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_pretraining_checkpoint_loads_strictly_into_affinity_encoder(tmp_path):
    torch.manual_seed(29)
    source = MSAInterfacePretrainerV13(_small_encoder_kwargs())
    checkpoint_path = tmp_path / "best_pretrain.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 3,
            "variant": "ema",
            "metrics": {"loss": 1.25},
            "encoder_state": source.encoder.state_dict(),
        },
        checkpoint_path,
    )
    target = _small_model()
    lineage = _load_pretrained_encoder(target, checkpoint_path)
    assert lineage["epoch"] == 3
    assert lineage["variant"] == "ema"
    assert len(lineage["sha256"]) == 64
    for name, expected in source.encoder.state_dict().items():
        torch.testing.assert_close(target.msa_pair_encoder.state_dict()[name], expected)


def test_pretraining_validation_does_not_reset_training_rng(monkeypatch):
    class Config:
        device = "cpu"

    monkeypatch.setattr(
        pretraining_module,
        "_evaluate_with_fixed_masks",
        lambda model, dataset, config, seed: torch.rand(8),
    )
    torch.manual_seed(101)
    expected = torch.rand(4)
    torch.manual_seed(101)
    pretraining_module._evaluate(None, None, Config(), 999)
    actual = torch.rand(4)
    torch.testing.assert_close(actual, expected)



def test_masked_interface_distances_are_not_visible_to_interface_head():
    torch.manual_seed(37)
    encoder = PocketMSAInterfaceV13(**_small_encoder_kwargs()).eval()
    batch = _batch()
    batch["interface_distance_visible"] = torch.zeros_like(
        batch["contact_mask"], dtype=torch.bool
    )
    changed = copy.deepcopy(batch)
    changed["residue_ligand_distance"] = batch["residue_ligand_distance"] + 50.0
    with torch.no_grad():
        original = encoder(batch)
        altered = encoder(changed)
    torch.testing.assert_close(
        original.interface_distance_logits,
        altered.interface_distance_logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        original.interface_contact_logits,
        altered.interface_contact_logits,
        rtol=0.0,
        atol=0.0,
    )


def test_v12_pretraining_parameters_warm_start_v13_encoder(tmp_path):
    torch.manual_seed(41)
    source = MSAPairPretrainerV12(_small_encoder_kwargs())
    checkpoint_path = tmp_path / "v12_pretrain.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 2,
            "variant": "ema",
            "metrics": {"loss": 2.0},
            "encoder_state": source.encoder.state_dict(),
        },
        checkpoint_path,
    )
    target = PocketMSAInterfaceV13(**_small_encoder_kwargs())
    lineage = _load_initial_msa_encoder(target, checkpoint_path)
    assert lineage["loaded_parameter_count"] == len(source.encoder.state_dict())
    for name, expected in source.encoder.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], expected)


def test_v12_affinity_trunk_warm_starts_v13_without_old_encoder(tmp_path):
    torch.manual_seed(43)
    source = MSAPairExpertV12(
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
    checkpoint_path = tmp_path / "v12_affinity.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 5,
            "variant": "raw",
            "metrics": {"CI": 0.7},
            "model_state": source.state_dict(),
        },
        checkpoint_path,
    )
    target = _small_model()
    encoder_before = {
        name: value.detach().clone()
        for name, value in target.msa_pair_encoder.state_dict().items()
    }
    lineage = _load_affinity_initialization(target, checkpoint_path)
    assert lineage["loaded_parameter_count"] > 0
    for name, expected in source.state_dict().items():
        if not name.startswith("msa_pair_encoder."):
            torch.testing.assert_close(target.state_dict()[name], expected)
    for name, expected in encoder_before.items():
        torch.testing.assert_close(target.msa_pair_encoder.state_dict()[name], expected)



def test_pair_token_padding_and_topk_are_stable():
    torch.manual_seed(47)
    pool = ContactAwarePairTokenPoolV18(
        pair_width=16,
        hidden_width=32,
        heads=4,
        layers=1,
        token_count=5,
        dropout=0.0,
    ).eval()
    pair = torch.randn(2, 3, 3, 16)
    distances = torch.tensor(
        [
            [[2.0, 4.0, 9.0], [3.0, 8.0, 10.0], [7.0, 11.0, 12.0]],
            [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]],
        ]
    )
    contact_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    contact_mask[0, 0, :2] = True
    contact_mask[0, 1, 0] = True
    batch = {
        "contact_mask": contact_mask,
        "pocket_mask": torch.tensor([[True, True, False], [False, False, False]]),
        "ligand_mask": torch.tensor([[True, True, False], [False, False, False]]),
        "residue_ligand_distance": distances,
        "pocket_msa_tokens": torch.tensor(
            [[[1, 2, 3]], [[4, 5, 6]]], dtype=torch.long
        ),
        "ligand_categorical": torch.tensor(
            [
                [[6.0] + [0.0] * 18, [8.0] + [0.0] * 18, [7.0] + [0.0] * 18],
                [[6.0] + [0.0] * 18, [6.0] + [0.0] * 18, [6.0] + [0.0] * 18],
            ]
        ),
    }
    changed_pair = pair.clone()
    invalid = ~contact_mask
    changed_pair[invalid] = changed_pair[invalid] + 1000.0
    with torch.no_grad():
        original = pool(SimpleNamespace(interface_pair=pair), batch)
        changed = pool(SimpleNamespace(interface_pair=changed_pair), batch)
    assert original.shape == (2, 32)
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original[0], changed[0], rtol=0.0, atol=0.0)


def test_pair_readout_receives_gradient_and_encoder_runs_once():
    torch.manual_seed(59)
    model = _small_model().train()
    calls = {"count": 0}

    def count_forward(_module, _inputs, _output):
        calls["count"] += 1

    handle = model.msa_pair_encoder.register_forward_hook(count_forward)
    output = model(_batch())
    handle.remove()
    loss = output.prediction.square().mean()
    loss.backward()
    gradient = model.pair_affinity_head[-1].weight.grad
    assert calls["count"] == 1
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0
