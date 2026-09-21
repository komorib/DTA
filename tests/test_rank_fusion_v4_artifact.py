import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dta.rank_fusion_v4.artifact import (
    ExpertArtifact,
    load_expert_artifact,
    write_expert_artifact,
)


class ExpertArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary_directory.name) / "artifact"
        self.payload = {
            "keys": ["1abc_protein", "2def_protein", "3ghi_protein"],
            "labels": np.asarray([6.5, 7.0, 4.5], dtype=np.float64),
            "predictions": np.asarray([6.4, 0.0, 4.7], dtype=np.float64),
            "latents": np.asarray(
                [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype=np.float64
            ),
            "available": np.asarray([True, False, True]),
            "expert": "upper_bound",
            "split": "valid",
            "manifest_sha256": "manifest-fixture",
            "checkpoint_sha256": "checkpoint-fixture",
            "adapter_version": "v4-test",
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write(self, **updates):
        arguments = dict(self.payload)
        arguments.update(updates)
        return write_expert_artifact(self.output, **arguments)

    def test_round_trip_normalizes_types_and_records_provenance(self):
        metadata = self.write()
        artifact = load_expert_artifact(
            self.output,
            expected_expert="upper_bound",
            expected_split="valid",
            expected_manifest_sha256="manifest-fixture",
            expected_checkpoint_sha256="checkpoint-fixture",
        )

        self.assertIsInstance(artifact, ExpertArtifact)
        self.assertEqual(artifact.keys, tuple(self.payload["keys"]))
        self.assertEqual(artifact.labels.dtype, np.dtype(np.float32))
        self.assertEqual(artifact.predictions.dtype, np.dtype(np.float32))
        self.assertEqual(artifact.latents.dtype, np.dtype(np.float32))
        self.assertEqual(artifact.available.dtype, np.dtype(np.bool_))
        self.assertEqual(metadata, artifact.metadata)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["count"], 3)
        self.assertEqual(metadata["latent_dimension"], 2)
        self.assertEqual(metadata["missing_count"], 1)
        self.assertEqual(len(metadata["ordered_key_sha256"]), 64)
        self.assertTrue((self.output / "features.npz").is_file())
        self.assertTrue((self.output / "metadata.json").is_file())

    def test_metadata_is_replaced_after_features(self):
        real_replace = os.replace
        replacements = []

        def recording_replace(source, destination):
            replacements.append(Path(destination).name)
            return real_replace(source, destination)

        with mock.patch("dta.rank_fusion_v4.artifact.os.replace", recording_replace):
            self.write()

        self.assertEqual(replacements, ["features.npz", "metadata.json"])

    def test_loader_rejects_partial_or_incomplete_artifacts(self):
        self.output.mkdir(parents=True)
        np.savez_compressed(
            self.output / "features.npz",
            keys=np.asarray(["1abc_protein"]),
            labels=np.asarray([1.0], dtype=np.float32),
            predictions=np.asarray([1.0], dtype=np.float32),
            latents=np.asarray([[1.0]], dtype=np.float32),
            available=np.asarray([True]),
        )
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            load_expert_artifact(
                self.output,
                expected_expert="upper_bound",
                expected_split="valid",
            )

        (self.output / "metadata.json").write_text(
            json.dumps({"status": "incomplete"}), encoding="utf-8"
        )
        with self.assertRaisesRegex((RuntimeError, ValueError), "complete|metadata"):
            load_expert_artifact(
                self.output,
                expected_expert="upper_bound",
                expected_split="valid",
            )

    def test_writer_rejects_invalid_keys_shapes_and_latent_width(self):
        cases = [
            ({"keys": []}, "nonempty"),
            ({"keys": ["same", "same", "third"]}, "duplicate"),
            ({"keys": ["", "second", "third"]}, "nonempty"),
            ({"labels": np.ones((3, 1))}, "labels"),
            ({"predictions": np.ones((3, 1))}, "predictions"),
            ({"available": np.ones((3, 1), dtype=bool)}, "available"),
            ({"latents": np.ones((3, 2, 1))}, "latents"),
            ({"latents": np.ones((3, 1025))}, "1024"),
            ({"labels": np.ones(2)}, "length"),
        ]
        for updates, message in cases:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, message):
                    self.write(**updates)

    def test_writer_rejects_nonfinite_and_nonzero_unavailable_rows(self):
        cases = [
            ({"labels": [1.0, np.nan, 2.0]}, "labels.*nonfinite"),
            ({"predictions": [1.0, 0.0, np.inf]}, "predictions.*nonfinite"),
            (
                {"latents": [[1.0, 2.0], [0.0, 0.0], [np.nan, 1.0]]},
                "latents.*nonfinite",
            ),
            ({"predictions": [6.4, 0.1, 4.7]}, "unavailable.*zero"),
            (
                {"latents": [[1.0, 2.0], [0.0, 0.1], [3.0, 4.0]]},
                "unavailable.*zero",
            ),
        ]
        for updates, message in cases:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, message):
                    self.write(**updates)

    def test_writer_rejects_empty_provenance(self):
        for field in (
            "expert",
            "split",
            "manifest_sha256",
            "checkpoint_sha256",
            "adapter_version",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    self.write(**{field: ""})

    def test_loader_revalidates_arrays_and_expected_provenance(self):
        self.write()
        mismatches = [
            ({"expected_expert": "boltz_pairwise"}, "expert"),
            ({"expected_split": "test"}, "split"),
            ({"expected_manifest_sha256": "wrong"}, "manifest_sha256"),
            ({"expected_checkpoint_sha256": "wrong"}, "checkpoint_sha256"),
        ]
        defaults = {
            "expected_expert": "upper_bound",
            "expected_split": "valid",
        }
        for updates, message in mismatches:
            arguments = dict(defaults)
            arguments.update(updates)
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, message):
                    load_expert_artifact(self.output, **arguments)

        metadata_path = self.output / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata["adapter_version"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata.*adapter_version"):
            load_expert_artifact(self.output, **defaults)

    def test_loader_rejects_tampered_features(self):
        self.write()
        feature_path = self.output / "features.npz"
        with np.load(str(feature_path), allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        arrays["predictions"] = arrays["predictions"].copy()
        arrays["predictions"][1] = 1.0
        np.savez_compressed(str(feature_path), **arrays)

        with self.assertRaisesRegex(ValueError, "unavailable.*zero"):
            load_expert_artifact(
                self.output,
                expected_expert="upper_bound",
                expected_split="valid",
            )


if __name__ == "__main__":
    unittest.main()
