import unittest

import torch


class TestHybridCachedTrainingV2(unittest.TestCase):
    def test_loss_matches_declared_components_and_prefers_correct_ranking(self):
        from dta.hybrid_cached_training_v2 import upper_bound_affinity_loss

        target = torch.tensor([[1.0], [3.0], [2.0]])
        weights = torch.full((3, 8), 0.125)
        correct = upper_bound_affinity_loss(
            torch.tensor([[1.0], [3.0], [2.0]], requires_grad=True),
            target,
            weights,
        )
        incorrect = upper_bound_affinity_loss(
            torch.tensor([[3.0], [1.0], [2.0]], requires_grad=True),
            target,
            weights,
        )
        expected = (
            correct["smooth_l1"]
            + 0.05 * correct["mse"]
            + 0.75 * correct["pairwise"]
            + 0.05 * correct["listmle"]
            + 0.01 * correct["load_balance"]
        )
        torch.testing.assert_close(correct["loss"], expected)
        self.assertLess(correct["pairwise"], incorrect["pairwise"])
        self.assertLess(correct["listmle"], incorrect["listmle"])
        correct["loss"].backward()

    def test_ci_selection_prioritizes_ci_then_spearman_then_rmse(self):
        from dta.hybrid_cached_training_v2 import is_better_validation

        current = {"CI": 0.70, "Spearman": 0.50, "RMSE": 1.2}
        self.assertTrue(
            is_better_validation(
                {"CI": 0.71, "Spearman": 0.0, "RMSE": 9.0}, current
            )
        )
        self.assertTrue(
            is_better_validation(
                {"CI": 0.7000001, "Spearman": 0.51, "RMSE": 2.0}, current
            )
        )
        self.assertTrue(
            is_better_validation(
                {"CI": 0.70, "Spearman": 0.50, "RMSE": 1.1}, current
            )
        )
        self.assertFalse(
            is_better_validation(
                {"CI": 0.70, "Spearman": 0.49, "RMSE": 0.1}, current
            )
        )

    def test_preflight_retries_smaller_batches_after_oom(self):
        from dta.hybrid_cached_training_v2 import select_preflight_batch_size

        attempted = []

        def probe(batch_size):
            attempted.append(batch_size)
            if batch_size > 64:
                raise RuntimeError("CUDA out of memory")
            return True

        self.assertEqual(
            select_preflight_batch_size([256, 128, 64, 32], probe), 64
        )
        self.assertEqual(attempted, [256, 128, 64])


if __name__ == "__main__":
    unittest.main()
