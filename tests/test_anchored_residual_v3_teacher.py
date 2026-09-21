import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TestTeacherPredictionArtifact(unittest.TestCase):
    def _write(self, root, *, keys=("a", "b"), predictions=(1.25, 2.5), split="train"):
        root = Path(root)
        np.savez_compressed(
            root / "predictions.npz",
            keys=np.asarray(keys),
            labels=np.asarray([1.0, 2.0]),
            predictions=np.asarray(predictions),
        )
        (root / "metadata.json").write_text(json.dumps({
            "status": "complete",
            "split": split,
            "manifest_sha256": "a" * 64,
            "teacher_checkpoint_sha256": "b" * 64,
        }))
        return root / "predictions.npz", root / "metadata.json"

    def test_loads_complete_keyed_artifact(self):
        from dta.anchored_residual_v3.teacher import load_teacher_artifact

        with tempfile.TemporaryDirectory() as directory:
            predictions, metadata = self._write(directory)
            lookup = load_teacher_artifact(
                predictions,
                metadata,
                expected_split="train",
                expected_manifest_sha256="a" * 64,
            )
            self.assertEqual(lookup.value_for("a"), 1.25)
            self.assertEqual(lookup.keys, ("a", "b"))
            with self.assertRaisesRegex(KeyError, "missing"):
                lookup.value_for("missing")

    def test_rejects_duplicate_nonfinite_and_provenance_mismatch(self):
        from dta.anchored_residual_v3.teacher import load_teacher_artifact

        with tempfile.TemporaryDirectory() as directory:
            predictions, metadata = self._write(directory, keys=("a", "a"))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_teacher_artifact(predictions, metadata, expected_split="train")
        with tempfile.TemporaryDirectory() as directory:
            predictions, metadata = self._write(directory, predictions=(1.0, np.nan))
            with self.assertRaisesRegex(ValueError, "finite"):
                load_teacher_artifact(predictions, metadata, expected_split="train")
        with tempfile.TemporaryDirectory() as directory:
            predictions, metadata = self._write(directory)
            with self.assertRaisesRegex(ValueError, "split"):
                load_teacher_artifact(predictions, metadata, expected_split="valid")
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_teacher_artifact(
                    predictions,
                    metadata,
                    expected_split="train",
                    expected_manifest_sha256="c" * 64,
                )

    def test_writer_commits_complete_metadata_last(self):
        from dta.anchored_residual_v3.teacher import (
            load_teacher_artifact,
            write_teacher_artifact,
        )

        with tempfile.TemporaryDirectory() as directory:
            result = write_teacher_artifact(
                directory,
                keys=["a", "b"],
                labels=[1.0, 2.0],
                predictions=[1.25, 2.5],
                split="valid",
                manifest_sha256="a" * 64,
                teacher_checkpoint_sha256="b" * 64,
            )
            self.assertEqual(result["status"], "complete")
            lookup = load_teacher_artifact(
                Path(directory) / "predictions.npz",
                Path(directory) / "metadata.json",
                expected_split="valid",
            )
            self.assertEqual(len(lookup), 2)


class TestTeacherPredictionGeneration(unittest.TestCase):
    def test_generation_loads_checkpoint_and_preserves_order(self):
        from dta.anchored_residual_v3.teacher import (
            generate_teacher_predictions,
            load_teacher_artifact,
        )

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.0))

            def forward(self, batch):
                return {"prediction": batch["feature"] * self.weight}

        loader = [{
            "keys": ["b", "a"],
            "feature": torch.tensor([[2.0], [1.0]]),
            "label": torch.tensor([[2.0], [1.0]]),
        }]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            source = Model()
            source.weight.data.fill_(1.5)
            torch.save({"model_state_dict": source.state_dict()}, checkpoint)
            output = Path(directory) / "teacher"
            result = generate_teacher_predictions(
                Model(), loader, checkpoint, output,
                split="valid", manifest_sha256="a" * 64, device="cpu",
            )
            self.assertEqual(result["count"], 2)
            lookup = load_teacher_artifact(
                output / "predictions.npz", output / "metadata.json",
                expected_split="valid",
            )
            self.assertEqual(lookup.keys, ("b", "a"))
            np.testing.assert_allclose(lookup.predictions, [3.0, 1.5])

    def test_generation_requires_model_state_dict(self):
        from dta.anchored_residual_v3.teacher import generate_teacher_predictions

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pth"
            torch.save({"weights": {}}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "model_state_dict"):
                generate_teacher_predictions(
                    nn.Linear(1, 1), [], checkpoint, Path(directory) / "teacher",
                    split="train", manifest_sha256="a" * 64, device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
