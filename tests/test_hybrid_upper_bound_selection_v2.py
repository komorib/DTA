import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestValidationCandidates(unittest.TestCase):
    def _write_candidate(self, root, name, keys, labels, predictions, metadata):
        root = Path(root) / name
        root.mkdir()
        np.savez_compressed(
            root / "valid_predictions.npz",
            keys=np.asarray(keys),
            labels=np.asarray(labels),
            predictions=np.asarray(predictions),
        )
        (root / "provenance.json").write_text(json.dumps(metadata))
        return root

    def test_candidate_alignment_rejects_label_or_split_mismatch(self):
        from scripts.select_hybrid_upper_bound_v2 import (
            align_validation_candidates,
            load_validation_candidate,
        )

        metadata = {
            "split": "valid",
            "model_sha256": "a" * 64,
            "valid_manifest_sha256": "m" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            first = self._write_candidate(
                directory, "first", ["a", "b"], [1.0, 2.0], [1.1, 1.9], metadata
            )
            second = self._write_candidate(
                directory, "second", ["a", "b"], [1.0, 3.0], [1.1, 2.9], metadata
            )
            loaded_first = load_validation_candidate("first", first)
            loaded_second = load_validation_candidate("second", second)
            with self.assertRaisesRegex(ValueError, "labels differ"):
                align_validation_candidates([loaded_first, loaded_second])
            invalid = self._write_candidate(
                directory, "invalid", ["a", "b"], [1.0, 2.0], [1.1, 1.9],
                {**metadata, "split": "test"},
            )
            with self.assertRaisesRegex(ValueError, "valid"):
                load_validation_candidate("invalid", invalid)


class TestRealWeightEnsemble(unittest.TestCase):
    def test_powell_finds_negative_weight_ensemble_that_beats_members(self):
        from scripts.select_hybrid_upper_bound_v2 import fit_real_weight_ensemble

        labels = np.arange(6, dtype=np.float64)
        first = np.asarray([0.0, 2.0, 1.0, 3.0, 5.0, 4.0])
        second = 2.0 * first - labels
        result = fit_real_weight_ensemble(
            labels,
            np.stack([first, second], axis=1),
            seed=13,
            random_starts=12,
        )
        self.assertGreaterEqual(
            result["metrics"]["CI"],
            max(result["member_metrics"][0]["CI"], result["member_metrics"][1]["CI"]),
        )
        self.assertTrue(np.any(result["weights"] < 0.0))
        self.assertAlmostEqual(result["metrics"]["CI"], 1.0)
        self.assertAlmostEqual(result["predictions"].mean(), labels.mean(), places=8)
        self.assertLess(result["metrics"]["MAE"], 0.5)

    def test_positive_affine_calibration_restores_label_scale(self):
        from scripts.select_hybrid_upper_bound_v2 import _positive_affine_calibration

        labels = np.arange(8, dtype=np.float64)
        raw_score = 224.0 + 0.004 * labels
        slope, intercept, predictions = _positive_affine_calibration(
            labels, raw_score
        )

        self.assertGreater(slope, 0.0)
        self.assertAlmostEqual(predictions.mean(), labels.mean(), places=8)
        np.testing.assert_allclose(predictions, labels, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()



class TestSelectionPackage(unittest.TestCase):
    def test_package_embeds_member_states_and_reproduces_selected_prediction(self):
        import hashlib
        import torch

        from scripts.select_hybrid_upper_bound_v2 import (
            fit_real_weight_ensemble,
            load_validation_candidate,
            write_selection_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            roots = []
            for index, prediction in enumerate(([0.0, 2.0, 1.0], [0.0, 3.0, 0.0])):
                root = Path(directory) / f"candidate-{index}"
                root.mkdir()
                checkpoint = root / "model.pth"
                torch.save({"model_state_dict": {"weight": torch.tensor([index])}}, checkpoint)
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                np.savez_compressed(
                    root / "valid_predictions.npz",
                    keys=np.asarray(["a", "b", "c"]),
                    labels=np.asarray([0.0, 1.0, 2.0]),
                    predictions=np.asarray(prediction),
                )
                (root / "provenance.json").write_text(json.dumps({
                    "split": "valid",
                    "model_sha256": digest,
                    "valid_manifest_sha256": "m" * 64,
                }))
                roots.append((root, checkpoint))
            candidates = [
                load_validation_candidate(f"c{index}", root)
                for index, (root, _) in enumerate(roots)
            ]
            fitted = fit_real_weight_ensemble(
                candidates[0].labels,
                np.stack([candidate.predictions for candidate in candidates], axis=1),
                random_starts=2,
            )
            output = Path(directory) / "package"
            package = write_selection_package(
                output,
                candidates,
                fitted,
                [checkpoint for _, checkpoint in roots],
            )
            self.assertTrue((output / "Model_hybrid_upper_bound.pth").is_file())
            self.assertEqual(len(package["members"]), 2)
            archive = np.load(output / "valid_predictions.npz")
            np.testing.assert_allclose(archive["predictions"], fitted["predictions"])



class TestSelectionCli(unittest.TestCase):
    def test_cli_requires_explicit_validation_candidate_artifacts(self):
        from scripts.select_hybrid_upper_bound_v2 import build_arg_parser

        args = build_arg_parser().parse_args([
            "--candidate", "cached=cache.npz=candidate.json=model.pth",
            "--candidate", "e2e=raw.npz=raw.json=raw.pth",
            "--output", "selection",
        ])
        self.assertEqual(len(args.candidate), 2)
        self.assertEqual(args.seed, 42)
        self.assertFalse(hasattr(args, "test_manifest"))



class TestSelectorEntrypoint(unittest.TestCase):
    def test_selector_runs_from_repository_root(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/select_hybrid_upper_bound_v2.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--candidate", result.stdout)
