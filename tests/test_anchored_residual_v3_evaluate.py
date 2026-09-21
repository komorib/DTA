import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestTestOnlyEvaluation(unittest.TestCase):
    def test_writes_test_predictions_metrics_and_provenance(self):
        from dta.anchored_residual_v3.evaluate import write_test_artifact

        evaluation = {
            "keys": ["a", "b", "c"],
            "labels": np.asarray([1.0, 2.0, 3.0]),
            "predictions": np.asarray([1.1, 1.9, 3.2]),
            "metrics": {
                "CI": 1.0, "MAE": 0.1, "RMSE": 0.15,
                "Pearson": 0.99, "Spearman": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            result = write_test_artifact(
                directory,
                evaluation,
                split="test",
                variant="cached_teacher",
                manifest_sha256="a" * 64,
                checkpoint_sha256="b" * 64,
            )
            self.assertEqual(result["split"], "test")
            archive = np.load(Path(directory) / "test_predictions_cached_teacher.npz")
            self.assertEqual(set(archive.files), {"keys", "labels", "predictions"})
            metadata = json.loads(
                (Path(directory) / "test_metrics_cached_teacher.json").read_text()
            )
            self.assertEqual(metadata["manifest_sha256"], "a" * 64)
            self.assertEqual(metadata["metrics"]["CI"], 1.0)

    def test_rejects_non_test_split(self):
        from dta.anchored_residual_v3.evaluate import write_test_artifact

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "test"):
                write_test_artifact(
                    directory,
                    {"keys": ["a"], "labels": [1.0], "predictions": [1.0], "metrics": {}},
                    split="valid",
                    variant="raw",
                    manifest_sha256="a" * 64,
                    checkpoint_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
