import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / "external_benchmarks" / "imported_index_20260721" / "boltz_dta"
sys.path.insert(0, str(ROOT))


class TestBoltzDTACIPriority(unittest.TestCase):
    def test_pairwise_rank_loss_prefers_correct_order(self):
        from dta.train import pairwise_rank_loss

        target_raw = torch.tensor([5.0, 6.0, 7.0])
        pred_good = torch.tensor([-1.0, 0.0, 1.0])
        pred_bad = torch.tensor([1.0, 0.0, -1.0])

        good = pairwise_rank_loss(pred_good, target_raw, min_gap=0.1, temperature=0.5)
        bad = pairwise_rank_loss(pred_bad, target_raw, min_gap=0.1, temperature=0.5)

        self.assertLess(float(good), float(bad))

    def test_select_metric_ci_tiebreaks_with_rmse(self):
        from dta.train import is_better_checkpoint

        best = {"ci": 0.70, "rmse": 1.30}
        better_ci = {"ci": 0.71, "rmse": 1.50}
        tied_ci_better_rmse = {"ci": 0.7000000001, "rmse": 1.20}
        worse = {"ci": 0.69, "rmse": 1.10}

        self.assertTrue(is_better_checkpoint(better_ci, best, "ci", min_delta=1e-6))
        self.assertTrue(is_better_checkpoint(tied_ci_better_rmse, best, "ci", min_delta=1e-6))
        self.assertFalse(is_better_checkpoint(worse, best, "ci", min_delta=1e-6))

    def test_pairwise_model_exposes_delta_readout_gate(self):
        from dta.models.pairwise import PairwiseTwoBranch

        model = PairwiseTwoBranch(emb_dim=4, global_dim=8, dim=6, dropout=0.0)

        self.assertTrue(hasattr(model, "pair_delta_gate"))
        self.assertEqual(tuple(model.pair_delta_gate.shape), (6,))
        self.assertTrue(torch.allclose(model.pair_delta_gate.detach(), torch.full((6,), 0.1)))

    def test_pairwise_delta_readout_can_be_disabled(self):
        from dta.models.pairwise import PairwiseTwoBranch

        model = PairwiseTwoBranch(emb_dim=4, global_dim=8, dim=6, dropout=0.0, use_pair_delta=False)

        self.assertFalse(model.use_pair_delta)


if __name__ == "__main__":
    unittest.main()
