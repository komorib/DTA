import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dta.rank_fusion_v4.artifact import load_expert_artifact
from dta.rank_fusion_v4.upper_bound_export import export_upper_bound


class FakeDataset(torch.utils.data.Dataset):
    records = []
    metadata_template = {}

    def __init__(self, path):
        self.path = Path(path)
        self.records = list(type(self).records)
        self.keys = [record["key"] for record in self.records]
        self.metadata = dict(type(self).metadata_template)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def close(self):
        return None


def fake_collate(samples):
    return {
        "keys": [sample["key"] for sample in samples],
        "label": torch.stack([sample["label"] for sample in samples]),
        "gerna_prediction": torch.stack(
            [sample["gerna_prediction"] for sample in samples]
        ),
        "interface_prediction": torch.stack(
            [sample["interface_prediction"] for sample in samples]
        ),
        "sample_index": torch.stack([sample["sample_index"] for sample in samples]),
    }


class FakeFusion(torch.nn.Module):
    instances = []

    def __init__(self, dropout, drop_path_rate):
        super().__init__()
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.calls = 0
        type(self).instances.append(self)

    def forward(self, batch):
        self.calls += 1
        indices = batch["sample_index"].reshape(-1).float()
        batch_size = len(indices)
        base = torch.arange(512, dtype=torch.float32)[None, None, :]
        latents = base.expand(batch_size, 32, -1).clone()
        offsets = torch.arange(32, dtype=torch.float32)[None, :, None]
        latents += indices[:, None, None] + offsets
        weights = torch.arange(8, dtype=torch.float32)[None, :].expand(
            batch_size, -1
        )
        weights = weights + indices[:, None] * 10.0
        return {
            "prediction": (indices + 6.0).reshape(-1, 1),
            "latents": latents,
            "expert_weights": weights,
        }


class UpperBoundExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.manifest = self.root / "manifest.json"
        self.samples = [
            {"lmdb_key": "1abc_protein", "label": 6.0, "pdb_id": "1abc"},
            {"lmdb_key": "2def_protein", "label": 7.0, "pdb_id": "2def"},
            {"lmdb_key": "3ghi_protein", "label": 8.0, "pdb_id": "3ghi"},
        ]
        self.manifest.write_text(
            json.dumps({"split": "valid", "samples": self.samples}),
            encoding="utf-8",
        )
        FakeDataset.records = [
            {
                "key": sample["lmdb_key"],
                "label": torch.tensor([sample["label"]], dtype=torch.float32),
                "gerna_prediction": torch.tensor(
                    [10.0 + index], dtype=torch.float32
                ),
                "interface_prediction": torch.tensor(
                    [20.0 + index], dtype=torch.float32
                ),
                "sample_index": torch.tensor(index, dtype=torch.long),
            }
            for index, sample in enumerate(self.samples)
        ]
        FakeDataset.metadata_template = {
            "status": "complete",
            "split": "valid",
            "sample_count": 3,
        }
        FakeFusion.instances = []

    def tearDown(self):
        self.temporary_directory.cleanup()

    def patches(self):
        return mock.patch.multiple(
            "dta.rank_fusion_v4.upper_bound_export",
            CachedHybridFeatureDataset=FakeDataset,
            collate_cached_hybrid_features=fake_collate,
            HybridUpperBoundFusion=FakeFusion,
            load_cached_fusion_checkpoint=mock.Mock(
                return_value={"sha256": "checkpoint-fixture"}
            ),
        )

    def export(self, name="output", **updates):
        arguments = {
            "cache_dir": self.cache,
            "manifest_path": self.manifest,
            "checkpoint_path": self.checkpoint,
            "output_dir": self.root / name,
            "split": "valid",
            "device": "cpu",
            "batch_size": 2,
        }
        arguments.update(updates)
        with self.patches():
            return export_upper_bound(**arguments)

    def test_exports_keyed_predictions_and_packed_latents(self):
        metadata = self.export(limit=2)
        artifact = load_expert_artifact(
            self.root / "output",
            expected_expert="upper_bound",
            expected_split="valid",
            expected_checkpoint_sha256="checkpoint-fixture",
        )

        expected_manifest_hash = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.assertEqual(metadata["manifest_sha256"], expected_manifest_hash)
        self.assertEqual(artifact.keys, ("1abc_protein", "2def_protein"))
        np.testing.assert_allclose(artifact.labels, [6.0, 7.0])
        np.testing.assert_allclose(artifact.predictions, [6.0, 7.0])
        self.assertEqual(artifact.latents.shape, (2, 1024))
        np.testing.assert_allclose(
            artifact.latents[:, -10:-2],
            [[float(value) for value in range(8)], [10.0 + value for value in range(8)]],
        )
        np.testing.assert_allclose(artifact.latents[:, -2], [10.0, 11.0])
        np.testing.assert_allclose(artifact.latents[:, -1], [20.0, 21.0])
        self.assertTrue(artifact.available.all())

        adapter = json.loads(
            (self.root / "output" / "adapter_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(adapter["status"], "complete")
        self.assertEqual(adapter["selected_count"], 2)
        self.assertEqual(adapter["source_count"], 3)
        self.assertEqual(adapter["projection_input_dimension"], 1024)
        self.assertEqual(adapter["projection_output_dimension"], 1014)
        self.assertEqual(len(adapter["projection_sha256"]), 64)
        self.assertEqual(
            adapter["appended_diagnostics"],
            [
                "expert_weight_0",
                "expert_weight_1",
                "expert_weight_2",
                "expert_weight_3",
                "expert_weight_4",
                "expert_weight_5",
                "expert_weight_6",
                "expert_weight_7",
                "gerna_prediction",
                "interface_prediction",
            ],
        )

    def test_projection_is_reproducible_for_checkpoint(self):
        self.export(name="first")
        self.export(name="second", batch_size=1)
        first = load_expert_artifact(
            self.root / "first",
            expected_expert="upper_bound",
            expected_split="valid",
        )
        second = load_expert_artifact(
            self.root / "second",
            expected_expert="upper_bound",
            expected_split="valid",
        )
        np.testing.assert_allclose(first.latents, second.latents, rtol=1e-4, atol=1e-3)
        first_adapter = json.loads(
            (self.root / "first" / "adapter_metadata.json").read_text()
        )
        second_adapter = json.loads(
            (self.root / "second" / "adapter_metadata.json").read_text()
        )
        self.assertEqual(
            first_adapter["projection_sha256"],
            second_adapter["projection_sha256"],
        )

    def test_key_order_mismatch_fails_before_checkpoint_or_inference(self):
        FakeDataset.records = list(reversed(FakeDataset.records))
        checkpoint_loader = mock.Mock(return_value={"sha256": "checkpoint-fixture"})
        with mock.patch.multiple(
            "dta.rank_fusion_v4.upper_bound_export",
            CachedHybridFeatureDataset=FakeDataset,
            collate_cached_hybrid_features=fake_collate,
            HybridUpperBoundFusion=FakeFusion,
            load_cached_fusion_checkpoint=checkpoint_loader,
        ):
            with self.assertRaisesRegex(ValueError, "key order"):
                export_upper_bound(
                    self.cache,
                    self.manifest,
                    self.checkpoint,
                    self.root / "mismatch",
                    split="valid",
                    device="cpu",
                )
        checkpoint_loader.assert_not_called()
        self.assertEqual(FakeFusion.instances, [])

    def test_label_mismatch_fails_before_inference(self):
        FakeDataset.records[1] = dict(FakeDataset.records[1])
        FakeDataset.records[1]["label"] = torch.tensor([99.0])
        with self.patches():
            with self.assertRaisesRegex(ValueError, "label"):
                export_upper_bound(
                    self.cache,
                    self.manifest,
                    self.checkpoint,
                    self.root / "bad-label",
                    split="valid",
                    device="cpu",
                )
        self.assertEqual(FakeFusion.instances, [])

    def test_rejects_wrong_split_and_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "split"):
            self.export(split="test")
        for limit in (0, 4):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "limit"):
                    self.export(name="limit-%s" % limit, limit=limit)



    def test_adapter_metadata_is_not_complete_when_artifact_write_fails(self):
        output = self.root / "failed-output"
        with self.patches(), mock.patch(
            "dta.rank_fusion_v4.upper_bound_export.write_expert_artifact",
            side_effect=RuntimeError("artifact write failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "artifact write failed"):
                export_upper_bound(
                    self.cache,
                    self.manifest,
                    self.checkpoint,
                    output,
                    split="valid",
                    device="cpu",
                )
        self.assertFalse((output / "adapter_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
