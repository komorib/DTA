import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestBoltzEnsemble(unittest.TestCase):
    def test_rank_normalize_returns_centered_ranks(self):
        from scripts.ensemble_boltz_predictions import rank_normalize

        out = rank_normalize(np.array([10.0, 30.0, 20.0]))

        self.assertAlmostEqual(float(out.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(out.std()), 1.0, places=7)
        self.assertGreater(out[1], out[2])
        self.assertGreater(out[2], out[0])

    def test_simplex_weights_sum_to_one(self):
        from scripts.ensemble_boltz_predictions import simplex_weights

        weights = simplex_weights(3, step=0.5)

        self.assertIn((1.0, 0.0, 0.0), weights)
        self.assertIn((0.5, 0.5, 0.0), weights)
        self.assertIn((0.0, 0.0, 1.0), weights)
        for w in weights:
            self.assertAlmostEqual(sum(w), 1.0, places=7)

    def test_load_run_predictions_and_optimize_weights(self):
        from scripts.ensemble_boltz_predictions import load_runs, optimize_weights

        y = np.array([1.0, 2.0, 3.0, 4.0])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            r1 = base / "r1"
            r2 = base / "r2"
            r1.mkdir()
            r2.mkdir()
            np.save(r1 / "valid_pred.npy", np.stack([np.array([1.0, 2.0, 3.0, 4.0]), y], axis=1))
            np.save(r1 / "test_pred.npy", np.stack([np.array([1.0, 2.0, 3.0, 4.0]), y], axis=1))
            np.save(r2 / "valid_pred.npy", np.stack([np.array([4.0, 3.0, 2.0, 1.0]), y], axis=1))
            np.save(r2 / "test_pred.npy", np.stack([np.array([4.0, 3.0, 2.0, 1.0]), y], axis=1))

            runs = load_runs([str(r1), str(r2)])
            result = optimize_weights(runs, step=0.5, prediction_mode="rank", metric="ci")

        self.assertEqual(result["weights"], [1.0, 0.0])
        self.assertAlmostEqual(result["valid"]["ci"], 1.0, places=7)
        self.assertAlmostEqual(result["test"]["ci"], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
