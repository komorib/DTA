from types import SimpleNamespace

import torch

from dta.optional_structure_stable_pair_v28.losses import normalized_pair_distillation
from dta.optional_structure_stable_pair_v28.model import OptionalStructureStablePairDTA
from dta.optional_structure_stable_pair_v28.training import (
    NEW_PREFIXES,
    STRUCTURE_PREFIXES,
    TrainingConfig,
    _set_pretrained_trainable,
    pair_distillation_scale,
)


def _small_model():
    return OptionalStructureStablePairDTA(
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


def test_normalized_pair_distillation_is_scale_shift_invariant_and_bounded():
    torch.manual_seed(28)
    student = torch.randn(5, 32, requires_grad=True)
    teacher = torch.randn(5, 32, requires_grad=True)
    baseline, baseline_cosine, baseline_regression = normalized_pair_distillation(
        student, teacher
    )
    transformed, _, _ = normalized_pair_distillation(
        1000.0 * student + 17.0,
        250.0 * teacher - 91.0,
    )
    torch.testing.assert_close(baseline, transformed, atol=2e-5, rtol=2e-5)
    assert 0.0 <= float(baseline_cosine) <= 2.0
    assert 0.0 <= float(baseline_regression) <= 2.0
    assert float(baseline) < 2.0
    baseline.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_pair_distillation_schedule_waits_for_unfreeze_then_warms_up():
    config = SimpleNamespace(
        pretrained_freeze_epochs=2,
        pair_distillation_warmup_epochs=5,
    )
    values = [pair_distillation_scale(epoch, config) for epoch in range(8)]
    assert values == [0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]


def test_pretrained_freeze_does_not_disable_new_or_structure_modules():
    model = _small_model()
    count = _set_pretrained_trainable(model, False)
    assert count > 0
    for name, parameter in model.named_parameters():
        is_new = name.startswith(NEW_PREFIXES)
        is_structure = name.startswith(STRUCTURE_PREFIXES)
        if is_new or is_structure:
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad
    _set_pretrained_trainable(model, True)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_v28_defaults_bound_pair_transfer_and_protect_base():
    config = TrainingConfig()
    assert config.pretrained_freeze_epochs == 2
    assert config.pretrained_learning_rate_scale == 0.05
    assert config.pair_distillation_weight == 0.03
    assert config.pair_distillation_warmup_epochs == 5
    assert config.maximum_pair_distillation == 4.0
    assert config.base_anchor_distillation_weight == 0.50
    assert config.loss_config().pair_distillation_weight == 0.03

