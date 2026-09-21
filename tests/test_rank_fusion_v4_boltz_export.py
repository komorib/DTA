import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dta.rank_fusion_v4.artifact import load_expert_artifact
from dta.rank_fusion_v4.boltz_export import (
    _build_dependencies,
    export_boltz_pairwise,
    forward_pairwise_features,
)


class TinyPairwise(torch.nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.pocket_proj = torch.nn.Sequential(torch.nn.Linear(3, dim), torch.nn.LayerNorm(dim))
        self.ligand_proj = torch.nn.Sequential(torch.nn.Linear(3, dim), torch.nn.LayerNorm(dim))
        self.pair_value = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.GELU())
        self.pair_score = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.GELU(), torch.nn.Linear(dim, 1))
        self.pair_delta_gate = torch.nn.Parameter(torch.ones(dim) * 0.1)
        self.global_mlp = torch.nn.Sequential(torch.nn.LayerNorm(6), torch.nn.Linear(6, dim), torch.nn.GELU())
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(dim * 2), torch.nn.Linear(dim * 2, dim), torch.nn.GELU(), torch.nn.Identity(), torch.nn.Linear(dim, 1))

    def forward(self, batch):
        return forward_pairwise_features(self, batch)[0]


def batch(size=2):
    torch.manual_seed(4)
    return {
        "pocket_emb": torch.randn(size, 3, 3),
        "ligand_emb": torch.randn(size, 2, 3),
        "pocket_mask": torch.tensor([[True, True, False]] * size),
        "ligand_mask": torch.tensor([[True, True]] * size),
        "global_feat": torch.randn(size, 6),
        "y": torch.arange(size, dtype=torch.float32) + 5.0,
    }


class FakeDataset:
    records = {}

    def __init__(self, lmdb_dir, index_path, split, mode, **kwargs):
        self.keys = ["1abc_protein", "3ghi_protein", "bad_protein"]
        self.y = np.asarray([6.0, 8.0, 9.0], dtype=np.float32)

    def __getitem__(self, index):
        key = self.keys[index]
        if key == "bad_protein":
            raise ValueError("broken record")
        item = batch(1)
        return {name: value[0] for name, value in item.items()}


class FatalDataset(FakeDataset):
    def __getitem__(self, index):
        raise RuntimeError("LMDB subsystem failed")


class SourceLabelDifferenceDataset(FakeDataset):
    def __init__(self, lmdb_dir, index_path, split, mode, **kwargs):
        self.keys = ["rounded_protein", "historical_protein"]
        self.y = np.asarray([6.3979400086720375, 8.045757490560675], dtype=np.float32)

    def __getitem__(self, index):
        item = batch(1)
        return {name: value[0] for name, value in item.items()}


class BoltzExportTest(unittest.TestCase):
    def test_feature_forward_matches_normal_readout_and_latent_layout(self):
        model = TinyPairwise(dim=4).eval()
        inputs = batch()
        standardized, latent = forward_pairwise_features(model, inputs)
        self.assertEqual(latent.shape, (2, 20))
        self.assertTrue(torch.isfinite(latent).all())
        expected = model.head(torch.cat([latent[:, 8:12], latent[:, 12:16]], dim=-1)).reshape(-1)
        torch.testing.assert_close(standardized, expected, rtol=1e-6, atol=1e-6)


    def test_real_checkpoint_feature_forward_matches_upstream_model(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint_path = root / "external_benchmarks/results/boltz_pairwise_rank010_pairDelta_save/model.pt"
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        config = checkpoint["config"]
        Dataset, build_model = _build_dependencies()
        dataset = Dataset(
            config["merged_lmdb_dir"],
            config["index_path"],
            split="valid",
            mode="pairwise",
            pocket_radius=config["pocket_radius"],
            min_pocket_residues=config["min_pocket_residues"],
            max_pocket_residues=config["max_pocket_residues"],
            max_ligand_atoms=config["max_ligand_atoms"],
            edge_cutoff=config["edge_cutoff"],
        )
        record = dataset[0]
        inputs = {name: value.unsqueeze(0) for name, value in record.items()}
        model = build_model(config).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        with torch.inference_mode():
            expected = model(inputs)
            actual, latent = forward_pairwise_features(model, inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(latent.shape, (1, 640))

    def test_export_preserves_target_rows_and_masks_missing_or_broken_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"split": "valid", "samples": [
                {"lmdb_key": "1abc_protein", "label": 6.0},
                {"lmdb_key": "2def_protein", "label": 7.0},
                {"lmdb_key": "bad_protein", "label": 9.0},
                {"lmdb_key": "3ghi_protein", "label": 8.0},
            ]}), encoding="utf-8")
            checkpoint = root / "model.pt"
            model = TinyPairwise(dim=4)
            torch.save({
                "state_dict": model.state_dict(), "y_mu": 6.0, "y_sd": 2.0,
                "config": {"model": "pairwise", "dim": 4, "dropout": 0.0,
                           "no_local": False, "no_global": False, "no_pair_delta": False,
                           "merged_lmdb_dir": "lmdb", "index_path": "index"},
            }, checkpoint)
            with mock.patch("dta.rank_fusion_v4.boltz_export._build_dependencies", return_value=(FakeDataset, lambda config: TinyPairwise(dim=4))):
                export_boltz_pairwise(manifest, checkpoint, root / "output", split="valid", device="cpu", batch_size=2)
            artifact = load_expert_artifact(root / "output", expected_expert="boltz_pairwise", expected_split="valid")
            self.assertEqual(artifact.keys, ("1abc_protein", "2def_protein", "bad_protein", "3ghi_protein"))
            np.testing.assert_array_equal(artifact.available, [True, False, False, True])
            self.assertEqual(artifact.latents.shape, (4, 20))
            np.testing.assert_array_equal(artifact.predictions[~artifact.available], 0.0)
            np.testing.assert_array_equal(artifact.latents[~artifact.available], 0.0)
            adapter = json.loads((root / "output" / "adapter_metadata.json").read_text())
            self.assertEqual(adapter["missing_count"], 2)
            self.assertEqual({item["key"] for item in adapter["missing"]}, {"2def_protein", "bad_protein"})

    def test_manifest_labels_are_canonical_and_source_differences_are_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"split": "train", "samples": [
                {"lmdb_key": "rounded_protein", "label": 6.4},
                {"lmdb_key": "historical_protein", "label": 6.7},
            ]}), encoding="utf-8")
            checkpoint = root / "model.pt"
            model = TinyPairwise(dim=4)
            torch.save({
                "state_dict": model.state_dict(), "y_mu": 6.0, "y_sd": 2.0,
                "config": {"model": "pairwise", "dim": 4, "dropout": 0.0,
                           "no_local": False, "no_global": False, "no_pair_delta": False,
                           "merged_lmdb_dir": "lmdb", "index_path": "index"},
            }, checkpoint)
            with mock.patch(
                "dta.rank_fusion_v4.boltz_export._build_dependencies",
                return_value=(SourceLabelDifferenceDataset, lambda config: TinyPairwise(dim=4)),
            ):
                export_boltz_pairwise(
                    manifest, checkpoint, root / "output",
                    split="train", device="cpu", batch_size=2,
                )

            artifact = load_expert_artifact(
                root / "output", expected_expert="boltz_pairwise", expected_split="train"
            )
            np.testing.assert_allclose(artifact.labels, [6.4, 6.7], rtol=0.0, atol=1e-6)
            np.testing.assert_array_equal(artifact.available, [True, True])
            adapter = json.loads((root / "output" / "adapter_metadata.json").read_text())
            self.assertEqual(adapter["source_label_mismatch_count"], 1)
            self.assertEqual(adapter["source_label_mismatches"][0]["key"], "historical_protein")
            self.assertAlmostEqual(
                adapter["source_label_max_abs_delta"],
                abs(float(np.float32(8.045757490560675)) - float(np.float32(6.7))),
            )

    def test_rejects_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"split": "valid", "samples": [{"lmdb_key": "x_protein", "label": 1.0}]}))
            checkpoint = root / "bad.pt"
            torch.save({"state_dict": {}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "checkpoint.*config"):
                export_boltz_pairwise(manifest, checkpoint, root / "output", split="valid", device="cpu")


    def test_runtime_record_error_terminates_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"split": "valid", "samples": [
                {"lmdb_key": "1abc_protein", "label": 6.0}
            ]}))
            checkpoint = root / "model.pt"
            model = TinyPairwise(dim=4)
            torch.save({
                "state_dict": model.state_dict(), "y_mu": 6.0, "y_sd": 2.0,
                "config": {"model": "pairwise", "dim": 4, "dropout": 0.0,
                           "no_local": False, "no_global": False, "no_pair_delta": False,
                           "merged_lmdb_dir": "lmdb", "index_path": "index"},
            }, checkpoint)
            with mock.patch(
                "dta.rank_fusion_v4.boltz_export._build_dependencies",
                return_value=(FatalDataset, lambda config: TinyPairwise(dim=4)),
            ):
                with self.assertRaisesRegex(RuntimeError, "LMDB subsystem failed"):
                    export_boltz_pairwise(
                        manifest, checkpoint, root / "output",
                        split="valid", device="cpu",
                    )

if __name__ == "__main__":
    unittest.main()
