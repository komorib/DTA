import unittest

import torch

from dta.rank_fusion_v4.losses import (
    hard_pair_logistic_loss,
    listmle_loss,
    mine_hard_pairs,
    mixture_regularization,
    rank_fusion_loss,
    soft_spearman_loss,
)


class RankFusionLossTest(unittest.TestCase):
    def test_pair_miner_excludes_small_gaps_and_prioritizes_wrong_order(self):
        labels = torch.tensor([4.0, 3.95, 3.0, 2.0, 1.0])
        prediction = torch.tensor([0.0, 3.9, 3.0, 2.0, 1.0])
        experts = torch.tensor(
            [
                [4.0, 0.0, 4.0],
                [3.9, 3.9, 3.9],
                [3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        )
        available = torch.ones_like(experts, dtype=torch.bool)
        left, right = mine_hard_pairs(
            prediction,
            labels,
            experts,
            available,
            min_label_gap=0.10,
            max_pairs=2,
        )
        pairs = list(zip(left.tolist(), right.tolist()))
        self.assertTrue(all(labels[l] - labels[r] >= 0.10 for l, r in pairs))
        self.assertNotIn((0, 1), pairs)
        self.assertTrue(all(0 in pair for pair in pairs))

    def test_pair_miner_uses_expert_disagreement_and_caps_pair_count(self):
        labels = torch.arange(65, 0, -1, dtype=torch.float32)
        prediction = labels.clone()
        experts = labels[:, None].repeat(1, 3)
        experts[0, 1] = -100.0
        experts[-1, 1] = 100.0
        available = torch.ones_like(experts, dtype=torch.bool)
        left, right = mine_hard_pairs(
            prediction,
            labels,
            experts,
            available,
            min_label_gap=0.10,
            max_pairs=2048,
        )
        self.assertLessEqual(len(left), 2048)
        self.assertEqual((int(left[0]), int(right[0])), (0, 64))

    def test_loss_components_use_float32_and_have_finite_gradients(self):
        labels = torch.tensor([4.0, 1.0, 3.0, 2.0], dtype=torch.float16)
        base_prediction = torch.tensor([1.0, 4.0, 2.0, 3.0], dtype=torch.float16)
        expert_predictions = torch.tensor(
            [[4.0, 3.0, 4.5], [1.0, 2.0, 0.5], [3.0, 1.0, 3.5], [2.0, 4.0, 1.5]],
            dtype=torch.float16,
        )
        available = torch.ones((4, 3), dtype=torch.bool)

        for function in (hard_pair_logistic_loss, listmle_loss, soft_spearman_loss):
            prediction = base_prediction.clone().requires_grad_(True)
            if function is hard_pair_logistic_loss:
                loss = function(
                    prediction, labels, expert_predictions, available
                )
            else:
                loss = function(prediction, labels)
            self.assertEqual(loss.dtype, torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(float(prediction.grad.float().abs().sum()), 0.0)

        prediction = base_prediction.clone().requires_grad_(True)
        smooth = torch.nn.functional.smooth_l1_loss(
            prediction.float(), labels.float()
        )
        smooth.backward()
        self.assertGreater(float(prediction.grad.float().abs().sum()), 0.0)

        logits = torch.tensor(
            [[2.0, 0.0, -1.0], [0.0, 1.0, -2.0]], requires_grad=True
        )
        weights = torch.softmax(logits, dim=-1)
        regularization = mixture_regularization(
            weights, torch.ones_like(weights, dtype=torch.bool)
        )
        self.assertEqual(regularization.dtype, torch.float32)
        regularization.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_mixture_regularization_matches_available_entropy_expression(self):
        weights = torch.tensor([[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]])
        available = torch.tensor([[True, True, False], [True, False, False]])
        expected = (
            0.75 * torch.log(torch.tensor(0.75))
            + 0.25 * torch.log(torch.tensor(0.25))
        ) / 2.0
        actual = mixture_regularization(weights, available)
        torch.testing.assert_close(actual, expected)

    def test_total_loss_uses_exact_approved_weights(self):
        labels = torch.tensor([4.0, 1.0, 3.0, 2.0])
        prediction = torch.tensor([1.0, 4.0, 2.0, 3.0], requires_grad=True)
        expert_predictions = torch.tensor(
            [[4.0, 3.0, 4.5], [1.0, 2.0, 0.5], [3.0, 1.0, 3.5], [2.0, 4.0, 1.5]]
        )
        available = torch.ones((4, 3), dtype=torch.bool)
        logits = torch.randn(4, 3, requires_grad=True)
        weights = torch.softmax(logits, dim=-1)
        components = rank_fusion_loss(
            prediction,
            labels,
            expert_predictions,
            weights,
            available,
        )
        expected = (
            components["hard_pair_logistic"]
            + 0.30 * components["listmle"]
            + 0.20 * components["soft_spearman"]
            + 0.10 * components["smooth_l1"]
            + 0.01 * components["mixture_regularization"]
        )
        torch.testing.assert_close(components["loss"], expected)
        components["loss"].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_listmle_tie_break_is_deterministic_by_original_index(self):
        prediction = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        labels = torch.tensor([4.0, 4.0, 1.0])
        first = listmle_loss(prediction, labels)
        second = listmle_loss(prediction, labels)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
