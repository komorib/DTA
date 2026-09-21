import json
from types import SimpleNamespace

import pytest
import torch

from dta.optional_structure_multitask_v21.model import OptionalStructureDTA
from dta.optional_structure_staged_v22.data import DATASET_INDEX
from dta.optional_structure_staged_v22.losses import (
    SequenceLossConfig,
    StructureLossConfig,
    sequence_only_loss,
    structure_residual_loss,
)
from dta.optional_structure_staged_v22.model import OptionalStructureStagedDTA
from dta.optional_structure_staged_v22.run_test import run as run_test
from dta.optional_structure_staged_v22.training import (
    STRUCTURE_PREFIXES,
    TrainingConfig,
    VERSION,
    _compatible_state,
    configure_structure_trainable_parameters,
    structure_validation_gate,
)


def _model():
    torch.manual_seed(7)
    return OptionalStructureStagedDTA(
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
    torch.testing.assert_close(output.structure_gate[0], torch.tensor(1.0))


def test_structure_loss_backward_is_finite():
    model = _model().train()
    batch = _add_structure(_base_batch())
    output = model(batch)
    loss, components = structure_residual_loss(
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
        loss, _ = structure_residual_loss(
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


def test_task_adapters_are_zero_initialized_and_dataset_routed():
    model = _model().eval()
    latent = torch.randn(3, 48)
    dataset_index = torch.tensor([0, 1, 2])
    torch.testing.assert_close(model._dataset_adapter(latent, dataset_index), latent)
    with torch.no_grad():
        model.task_adapters[1][-1].bias.fill_(0.5)
    routed = model._dataset_adapter(latent, dataset_index)
    torch.testing.assert_close(routed[0], latent[0])
    torch.testing.assert_close(routed[2], latent[2])
    torch.testing.assert_close(routed[1], latent[1] + 0.5)


def test_sequence_binary_loss_pushes_both_predictions_across_fixed_cutoff():
    prediction = torch.tensor([0.4, 0.6], requires_grad=True)
    output = SimpleNamespace(base_prediction=prediction)
    labels = torch.tensor([1.5, -0.5])
    batch = {"dataset_index": torch.tensor([DATASET_INDEX["DAVIS"]] * 2)}
    thresholds = torch.tensor([float("nan"), 0.5, float("nan")])
    positive_weights = torch.ones(3)
    config = SequenceLossConfig(
        smooth_weight=0.0,
        mse_weight=0.0,
        rank_weight=0.0,
        binary_weight=1.0,
        binary_temperature=0.5,
    )
    for _ in range(12):
        loss, _ = sequence_only_loss(
            output, labels, batch, thresholds, positive_weights, config
        )
        gradient = torch.autograd.grad(loss, prediction)[0]
        with torch.no_grad():
            prediction -= gradient
    assert prediction[0] > thresholds[1]
    assert prediction[1] < thresholds[1]


def test_structure_phase_gradients_and_updates_are_isolated_from_davis_kiba():
    model = _model()
    structure_parameters, lp_parameters = configure_structure_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        [{"params": structure_parameters}, {"params": lp_parameters}], lr=1.0e-2
    )
    sequence_batch = _base_batch(batch_size=2)
    sequence_batch["dataset_index"] = torch.tensor(
        [DATASET_INDEX["DAVIS"], DATASET_INDEX["KIBA"]]
    )
    model.eval()
    before = model(sequence_batch).prediction.detach().clone()
    model.train()
    batch = _add_structure(_base_batch())
    output = model(batch)
    loss, _ = structure_residual_loss(
        output, torch.tensor([0.5, -0.3, 0.9]), batch, StructureLossConfig()
    )
    loss.backward()
    gradient_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradient_names
    assert all(
        name.startswith(STRUCTURE_PREFIXES)
        or name.startswith("task_adapters.0.")
        or name.startswith("base_heads.0.")
        for name in gradient_names
    )
    optimizer.step()
    model.eval()
    after = model(sequence_batch).prediction.detach()
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_structure_only_warm_start_cannot_modify_sealed_base(tmp_path):
    source = _model()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.startswith("structure_encoder."):
                parameter.add_(1.0)
            elif name.startswith("protein_encoder."):
                parameter.add_(2.0)
    checkpoint_path = tmp_path / "warm.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint_path)
    target = _model()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    information = _compatible_state(
        target, checkpoint_path, prefixes=("structure_encoder.",)
    )
    after = target.state_dict()
    assert information["loaded_parameter_groups"] > 0
    assert any(
        not torch.equal(after[name], before[name])
        for name in after
        if name.startswith("structure_encoder.") and after[name].is_floating_point()
    )
    assert all(
        torch.equal(after[name], before[name])
        for name in after
        if not name.startswith("structure_encoder.")
    )


def _gate_result(full_ci, base_ci, full_rmse, base_rmse, full_spearman, base_spearman):
    return {
        "metrics": {
            "LP_PDBBind": {
                "CI": full_ci,
                "RMSE": full_rmse,
                "Spearman": full_spearman,
            }
        },
        "base_metrics": {
            "LP_PDBBind": {
                "CI": base_ci,
                "RMSE": base_rmse,
                "Spearman": base_spearman,
            }
        },
    }


def test_structure_validation_gate_uses_only_full_vs_base_validation():
    config = TrainingConfig(
        minimum_structure_ci_gain=0.005,
        maximum_structure_rmse_increase=0.01,
        minimum_structure_spearman_gain=0.0,
    )
    passed = structure_validation_gate(
        _gate_result(0.706, 0.700, 1.009, 1.000, 0.61, 0.61), config
    )
    failed = structure_validation_gate(
        _gate_result(0.704, 0.700, 0.990, 1.000, 0.62, 0.61), config
    )
    assert passed["passed"]
    assert not failed["passed"]


def test_validation_failure_prevents_test_manifest_access(tmp_path):
    selection_output = tmp_path / "selection"
    selection_output.mkdir()
    (selection_output / "selection.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "phase": "structure_validation_selection",
                "status": "validation_failed",
                "validation_selected": False,
            }
        ),
        encoding="utf-8",
    )
    missing_test = tmp_path / "must_not_be_read.jsonl"
    arguments = SimpleNamespace(
        selection_output=selection_output,
        test_manifest=missing_test,
        device="cpu",
        num_workers=0,
        evaluation_batch_size=1,
        limit_test=1,
    )
    with pytest.raises(ValueError, match="validation gate did not pass"):
        run_test(arguments)


def test_bad_checkpoint_checksum_is_rejected_before_test_access(tmp_path):
    selection_output = tmp_path / "selection"
    selection_output.mkdir()
    checkpoint = selection_output / "best.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    valid_predictions = selection_output / "valid_predictions.npz"
    valid_predictions.write_bytes(b"prediction")
    sequence_selection = selection_output / "sequence.json"
    sequence_selection.write_text("{}", encoding="utf-8")
    (selection_output / "selection.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "phase": "structure_validation_selection",
                "status": "validation_passed",
                "validation_selected": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": "wrong",
                "valid_predictions": str(valid_predictions),
                "valid_predictions_sha256": "wrong",
                "sequence_selection": str(sequence_selection),
                "sequence_selection_sha256": "wrong",
            }
        ),
        encoding="utf-8",
    )
    arguments = SimpleNamespace(
        selection_output=selection_output,
        test_manifest=tmp_path / "must_not_be_read.jsonl",
        device="cpu",
        num_workers=0,
        evaluation_batch_size=1,
        limit_test=1,
    )
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        run_test(arguments)


def test_v21_base_warm_start_preserves_sequence_predictions(tmp_path):
    torch.manual_seed(7)
    source = OptionalStructureDTA(
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
    ).eval()
    checkpoint_path = tmp_path / "v21.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint_path)
    target = _model().eval()
    information = _compatible_state(
        target,
        checkpoint_path,
        excluded_prefixes=STRUCTURE_PREFIXES,
    )
    batch = _base_batch()
    with torch.no_grad():
        expected = source(batch).base_prediction
        actual = target(batch).base_prediction
    assert information["loaded_parameter_groups"] > 0
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
