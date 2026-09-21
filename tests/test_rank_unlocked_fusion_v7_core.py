import torch


from dta.rank_unlocked_fusion_v7.losses import (
    pairwise_logistic_loss,
    rank_unlocked_objective,
)
from dta.rank_unlocked_fusion_v7.model import RankUnlockedFusionV7


def _batch():
    return {
        "base_predictions": torch.tensor([0.0, 1.0]),
        "raw_predictions": torch.tensor(
            [[2.0, 0.5, 0.0], [-1.0, 0.5, 1.0]]
        ),
        "standardized_predictions": torch.tensor(
            [[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]]
        ),
        "available": torch.tensor(
            [[True, True, False], [True, False, True]]
        ),
        "latents": [
            torch.zeros(2, 4),
            torch.zeros(2, 3),
            torch.zeros(2, 5),
        ],
    }


def _model():
    return RankUnlockedFusionV7(
        latent_widths=(4, 3, 5),
        model_width=16,
        transformer_layers=1,
        attention_heads=4,
        dropout=0.0,
    )


def test_missing_experts_receive_zero_gate_weight():
    model = _model().eval()
    with torch.no_grad():
        output = model(_batch(), return_details=True)

    assert output.gate_weights.shape == (2, 3)
    assert output.gate_weights[0, 2].item() == 0.0
    assert output.gate_weights[1, 1].item() == 0.0
    torch.testing.assert_close(
        output.gate_weights.sum(dim=1), torch.ones(2)
    )


def test_model_can_change_anchor_ordering():
    model = _model().eval()
    with torch.no_grad():
        model.gate_head.weight.zero_()
        model.gate_head.bias.copy_(torch.tensor([20.0, -20.0, -20.0]))
        model.mix_scale_logit.fill_(20.0)
        model.residual_head[-1].weight.zero_()
        model.residual_head[-1].bias.zero_()
        output = model(_batch(), return_details=True)

    assert output.prediction[0] > output.prediction[1]
    assert output.anchor[0] < output.anchor[1]


def test_pairwise_loss_rewards_correct_order():
    labels = torch.tensor([3.0, 2.0, 1.0])
    correct, correct_count = pairwise_logistic_loss(
        torch.tensor([2.0, 1.0, 0.0]), labels
    )
    reversed_loss, reversed_count = pairwise_logistic_loss(
        torch.tensor([0.0, 1.0, 2.0]), labels
    )

    assert correct_count == reversed_count == 3
    assert correct < reversed_loss


def test_objective_combines_regression_ranking_and_residual_terms():
    prediction = torch.tensor([2.8, 2.1, 1.2], requires_grad=True)
    labels = torch.tensor([3.0, 2.0, 1.0])
    anchor = torch.tensor([2.5, 2.0, 1.5])

    total, components = rank_unlocked_objective(
        prediction, labels, anchor
    )
    total.backward()

    assert torch.isfinite(total)
    assert prediction.grad is not None
    assert set(components) == {
        "smooth_l1",
        "mse",
        "pairwise",
        "hard_pairwise",
        "residual",
        "pair_count",
        "hard_pair_count",
    }
