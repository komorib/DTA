import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestGerNADTATrainingStrategy(unittest.TestCase):
    def test_regression_checkpoint_selection_can_prioritize_ci_with_rmse_tiebreak(self):
        from dta.train_dta import is_better_regression_checkpoint

        best = {
            "RMSE": 1.3, "MAE": 1.0, "Pearson": 0.4, "Spearman": 0.5,
            "CI": 0.70, "Epoch": 7,
        }
        higher_ci = {
            "RMSE": 1.5, "MAE": 1.1, "Pearson": 0.3, "Spearman": 0.4,
            "CI": 0.71, "Epoch": 8,
        }
        tied_ci_lower_rmse = {
            "RMSE": 1.2, "MAE": 1.1, "Pearson": 0.4, "Spearman": 0.5,
            "CI": 0.7000001, "Epoch": 8,
        }
        tied_ci_rmse_lower_mae = {
            "RMSE": 1.3, "MAE": 0.9, "Pearson": 0.4, "Spearman": 0.5,
            "CI": 0.7000000001, "Epoch": 8,
        }
        tied_metrics_earlier_epoch = {
            "RMSE": 1.3, "MAE": 1.0, "Pearson": 0.4, "Spearman": 0.5,
            "CI": 0.70, "Epoch": 6,
        }
        tied_metrics_later_epoch = dict(tied_metrics_earlier_epoch, Epoch=8)
        lower_ci = {
            "RMSE": 1.1, "MAE": 0.8, "Pearson": 0.5, "Spearman": 0.6,
            "CI": 0.69, "Epoch": 6,
        }
        within_delta_lower_ci = dict(tied_ci_lower_rmse, CI=0.6999999)

        self.assertTrue(is_better_regression_checkpoint(higher_ci, best, "CI", min_delta=1e-6))
        self.assertTrue(is_better_regression_checkpoint(tied_ci_lower_rmse, best, "CI", min_delta=1e-6))
        self.assertTrue(is_better_regression_checkpoint(within_delta_lower_ci, best, "CI", min_delta=1e-6))
        self.assertTrue(is_better_regression_checkpoint(tied_ci_rmse_lower_mae, best, "CI", min_delta=1e-6))
        self.assertTrue(is_better_regression_checkpoint(tied_metrics_earlier_epoch, best, "CI", min_delta=1e-6))
        self.assertFalse(is_better_regression_checkpoint(tied_metrics_later_epoch, best, "CI", min_delta=1e-6))
        self.assertFalse(is_better_regression_checkpoint(lower_ci, best, "CI", min_delta=1e-6))

    def test_ci_checkpoint_tiebreak_treats_nonfinite_errors_as_worst(self):
        from dta.train_dta import is_better_regression_checkpoint

        best = {
            "RMSE": 1.2, "MAE": 0.9, "Pearson": 0.5, "Spearman": 0.5,
            "CI": 0.70, "Epoch": 4,
        }
        nan_candidate = dict(best, RMSE=np.nan, MAE=0.1, Epoch=3)
        nan_best = dict(best, RMSE=np.nan)
        finite_candidate = dict(best, RMSE=1.3, Epoch=5)

        self.assertFalse(
            is_better_regression_checkpoint(nan_candidate, best, "CI", min_delta=1e-6)
        )
        self.assertTrue(
            is_better_regression_checkpoint(finite_candidate, nan_best, "CI", min_delta=1e-6)
        )

    def test_dta_pairwise_rank_loss_prefers_correct_order(self):
        from dta.train_dta import pairwise_rank_loss

        target = torch.tensor([5.0, 6.0, 7.0])
        pred_good = torch.tensor([5.1, 6.0, 7.2])
        pred_bad = torch.tensor([7.0, 6.0, 5.0])

        good = pairwise_rank_loss(pred_good, target, min_gap=0.1, temperature=0.5)
        bad = pairwise_rank_loss(pred_bad, target, min_gap=0.1, temperature=0.5)

        self.assertLess(float(good), float(bad))

    def test_write_prediction_file_uses_prediction_then_label_columns(self):
        from dta.train_dta import write_prediction_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid_pred.npy"
            write_prediction_file(path, labels=np.array([1.0, 2.0]), preds=np.array([1.5, 2.5]))

            arr = np.load(path)

        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr[:, 0].tolist(), [1.5, 2.5])
        self.assertEqual(arr[:, 1].tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
