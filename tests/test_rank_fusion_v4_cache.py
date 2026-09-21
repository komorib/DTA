import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dta.rank_fusion_v4.artifact import write_expert_artifact
from dta.rank_fusion_v4.cache import build_fusion_cache, load_fusion_cache


EXPERTS = ("upper_bound", "boltz_pairwise", "three_dprot")


class FusionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = self._manifest(
            "train",
            [
                ("a_protein", 1.0),
                ("b_protein", 2.0),
                ("c_protein", 3.0),
            ],
        )
        self.manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.artifacts = {}
        self._artifact(
            "upper_bound",
            ["c_protein", "a_protein", "b_protein"],
            [3.0, 1.0, 2.0],
            [30.0, 10.0, 20.0],
            [[30.0, 300.0], [10.0, 100.0], [20.0, 200.0]],
            [True, True, True],
        )
        self._artifact(
            "boltz_pairwise",
            ["b_protein", "a_protein"],
            [2.0, 1.0],
            [200.0, 0.0],
            [[20.0], [0.0]],
            [True, False],
        )
        self._artifact(
            "three_dprot",
            ["c_protein", "a_protein", "b_protein"],
            [3.0, 1.0, 2.0],
            [3000.0, 1000.0, 0.0],
            [[30.0, 3.0, 0.3], [10.0, 1.0, 0.1], [0.0, 0.0, 0.0]],
            [True, True, False],
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _manifest(self, split, rows, name=None):
        path = self.root / (name or (split + "_manifest.json"))
        path.write_text(
            json.dumps(
                {
                    "split": split,
                    "samples": [
                        {"lmdb_key": key, "label": label} for key, label in rows
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _artifact(
        self,
        expert,
        keys,
        labels,
        predictions,
        latents,
        available,
        *,
        manifest_sha=None,
        split="train",
        suffix="",
    ):
        path = self.root / (expert + suffix)
        write_expert_artifact(
            path,
            keys=keys,
            labels=np.asarray(labels, dtype=np.float32),
            predictions=np.asarray(predictions, dtype=np.float32),
            latents=np.asarray(latents, dtype=np.float32),
            available=np.asarray(available, dtype=np.bool_),
            expert=expert,
            split=split,
            manifest_sha256=manifest_sha or self.manifest_sha,
            checkpoint_sha256=expert + "-checkpoint",
            adapter_version="test-v1",
        )
        self.artifacts[expert] = path
        return path

    def test_keyed_join_preserves_manifest_denominator_and_fits_train_statistics(self):
        normalization = self.root / "normalization.json"
        metadata = build_fusion_cache(
            self.manifest,
            self.artifacts,
            self.root / "cache",
            split="train",
            normalization_output=normalization,
        )
        cache = load_fusion_cache(
            self.root / "cache",
            expected_split="train",
            expected_manifest_sha256=self.manifest_sha,
        )

        self.assertEqual(cache.keys, ("a_protein", "b_protein", "c_protein"))
        np.testing.assert_allclose(cache.labels, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            cache.predictions,
            [[10.0, 0.0, 1000.0], [20.0, 200.0, 0.0], [30.0, 0.0, 3000.0]],
        )
        np.testing.assert_array_equal(
            cache.available,
            [[True, False, True], [True, True, False], [True, False, True]],
        )
        np.testing.assert_allclose(
            cache.standardized_predictions[:, 0], [-1.2247449, 0.0, 1.2247449]
        )
        np.testing.assert_array_equal(
            cache.standardized_predictions[~cache.available], 0.0
        )
        np.testing.assert_allclose(cache.boltz_latents[1], [0.0])
        np.testing.assert_array_equal(cache.boltz_latents[[0, 2]], 0.0)
        self.assertEqual(metadata["normalization_sha256"], hashlib.sha256(normalization.read_bytes()).hexdigest())
        self.assertEqual(cache.metadata["expert_order"], list(EXPERTS))
        coverage = json.loads((self.root / "cache" / "coverage.json").read_text())
        self.assertEqual(coverage["target_count"], 3)
        self.assertEqual(coverage["experts"]["upper_bound"]["available_count"], 3)
        self.assertEqual(coverage["experts"]["boltz_pairwise"]["missing_count"], 2)

    def test_valid_cache_reuses_frozen_train_normalization(self):
        normalization = self.root / "normalization.json"
        build_fusion_cache(
            self.manifest,
            self.artifacts,
            self.root / "train-cache",
            split="train",
            normalization_output=normalization,
        )
        valid_manifest = self._manifest(
            "valid", [("v_protein", 4.0)], name="valid.json"
        )
        valid_sha = hashlib.sha256(valid_manifest.read_bytes()).hexdigest()
        valid_artifacts = {}
        for expert, prediction, latent in (
            ("upper_bound", 40.0, [40.0, 400.0]),
            ("boltz_pairwise", 400.0, [40.0]),
            ("three_dprot", 4000.0, [40.0, 4.0, 0.4]),
        ):
            valid_artifacts[expert] = self._artifact(
                expert,
                ["v_protein"],
                [4.0],
                [prediction],
                [latent],
                [True],
                manifest_sha=valid_sha,
                split="valid",
                suffix="-valid",
            )
        build_fusion_cache(
            valid_manifest,
            valid_artifacts,
            self.root / "valid-cache",
            split="valid",
            normalization_input=normalization,
        )
        valid = load_fusion_cache(
            self.root / "valid-cache", expected_split="valid"
        )
        self.assertAlmostEqual(valid.standardized_predictions[0, 0], 2.4494898, places=5)
        self.assertGreater(valid.standardized_predictions[0, 2], 1.0)

    def test_rejects_bad_manifest_or_artifact_provenance(self):
        duplicate = self._manifest(
            "train", [("a_protein", 1.0), ("a_protein", 1.0)], name="duplicate.json"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_fusion_cache(
                duplicate,
                self.artifacts,
                self.root / "duplicate-cache",
                split="train",
                normalization_output=self.root / "duplicate-normalization.json",
            )

        bad_path = self._artifact(
            "boltz_pairwise",
            ["a_protein"],
            [99.0],
            [1.0],
            [[1.0]],
            [True],
            suffix="-bad-label",
        )
        bad_artifacts = dict(self.artifacts, boltz_pairwise=bad_path)
        with self.assertRaisesRegex(ValueError, "label"):
            build_fusion_cache(
                self.manifest,
                bad_artifacts,
                self.root / "bad-label-cache",
                split="train",
                normalization_output=self.root / "bad-label-normalization.json",
            )

        wrong_hash = self._artifact(
            "boltz_pairwise",
            ["a_protein"],
            [1.0],
            [1.0],
            [[1.0]],
            [True],
            manifest_sha="wrong-hash",
            suffix="-wrong-hash",
        )
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            build_fusion_cache(
                self.manifest,
                dict(self.artifacts, boltz_pairwise=wrong_hash),
                self.root / "wrong-hash-cache",
                split="train",
                normalization_output=self.root / "wrong-hash-normalization.json",
            )

    def test_requires_complete_upper_bound_and_frozen_normalization_policy(self):
        incomplete_upper = self._artifact(
            "upper_bound",
            ["a_protein", "b_protein"],
            [1.0, 2.0],
            [1.0, 2.0],
            [[1.0], [2.0]],
            [True, True],
            suffix="-incomplete",
        )
        with self.assertRaisesRegex(ValueError, "Upper-Bound.*every"):
            build_fusion_cache(
                self.manifest,
                dict(self.artifacts, upper_bound=incomplete_upper),
                self.root / "incomplete-cache",
                split="train",
                normalization_output=self.root / "incomplete-normalization.json",
            )
        with self.assertRaisesRegex(ValueError, "normalization_output"):
            build_fusion_cache(
                self.manifest,
                self.artifacts,
                self.root / "no-normalization-cache",
                split="train",
            )
        with self.assertRaisesRegex(ValueError, "normalization_input"):
            build_fusion_cache(
                self.manifest,
                self.artifacts,
                self.root / "wrong-policy-cache",
                split="valid",
                normalization_output=self.root / "should-not-fit.json",
            )

    def test_loader_rejects_partial_cache_and_wrong_normalization_hash(self):
        output = self.root / "partial"
        output.mkdir()
        np.savez(output / "features.npz", keys=np.asarray(["x"]))
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            load_fusion_cache(output, expected_split="train")

        normalization = self.root / "normalization.json"
        build_fusion_cache(
            self.manifest,
            self.artifacts,
            self.root / "cache",
            split="train",
            normalization_output=normalization,
        )
        metadata_path = self.root / "cache" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["normalization_sha256"] = "tampered"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "normalization"):
            load_fusion_cache(self.root / "cache", expected_split="train")


if __name__ == "__main__":
    unittest.main()
