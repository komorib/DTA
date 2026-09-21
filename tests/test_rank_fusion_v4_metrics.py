import inspect
import unittest

import numpy as np

from dta.hybrid_cached_training_v2 import _concordance_index
from dta.rank_fusion_v4.metrics import affinity_metrics, fast_concordance_index


class RankFusionMetricsTest(unittest.TestCase):
    def test_fast_ci_matches_repository_semantics_on_random_inputs(self):
        generator = np.random.RandomState(17)
        for count in (1, 2, 7, 31, 127):
            for _ in range(10):
                labels = generator.randint(0, 8, size=count).astype(np.float64)
                predictions = generator.randint(-4, 5, size=count).astype(np.float64)
                expected = _concordance_index(labels, predictions)
                actual = fast_concordance_index(labels, predictions)
                self.assertAlmostEqual(actual, expected, places=12)

    def test_fast_ci_handles_explicit_ties_and_edge_cases(self):
        cases = [
            ([1.0, 1.0], [0.0, 1.0]),
            ([1.0, 2.0], [3.0, 3.0]),
            ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
            ([1.0, 2.0], [1.0, 2.0]),
            ([2.0], [7.0]),
        ]
        for labels, predictions in cases:
            with self.subTest(labels=labels, predictions=predictions):
                self.assertEqual(
                    fast_concordance_index(labels, predictions),
                    _concordance_index(labels, predictions),
                )

    def test_implementation_uses_grouped_fenwick_queries(self):
        source = inspect.getsource(fast_concordance_index)
        self.assertIn("Fenwick", source)
        self.assertNotIn("[:, None]", source)
        self.assertNotIn("for right in", source)

    def test_affinity_metrics_match_stable_repository_ranks_and_are_finite(self):
        labels = np.asarray([1.0, 1.0, 3.0, 2.0], dtype=np.float64)
        predictions = np.asarray([2.0, 2.0, 1.0, 4.0], dtype=np.float64)
        metrics = affinity_metrics(labels, predictions)
        self.assertEqual(set(metrics), {"CI", "MAE", "RMSE", "Pearson", "Spearman"})
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["CI"], _concordance_index(labels, predictions))
        label_order = np.argsort(labels, kind="mergesort")
        prediction_order = np.argsort(predictions, kind="mergesort")
        label_rank = np.empty(4, dtype=np.float64)
        prediction_rank = np.empty(4, dtype=np.float64)
        label_rank[label_order] = np.arange(4, dtype=np.float64)
        prediction_rank[prediction_order] = np.arange(4, dtype=np.float64)
        expected_spearman = np.corrcoef(label_rank, prediction_rank)[0, 1]
        self.assertAlmostEqual(metrics["Spearman"], expected_spearman, places=12)

    def test_metrics_reject_nonfinite_or_mismatched_inputs(self):
        with self.assertRaisesRegex(ValueError, "same nonzero length"):
            affinity_metrics([1.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            affinity_metrics([1.0, np.nan], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
