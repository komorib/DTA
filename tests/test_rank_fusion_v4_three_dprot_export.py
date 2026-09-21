import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from dta.rank_fusion_v4.artifact import load_expert_artifact
from dta.rank_fusion_v4.three_dprot_export import (
    _build_dependencies,
    export_three_dprot,
    forward_three_dprot_features,
)


class FakeTrial:
    def __init__(self, hp):
        self.hp = hp


class FakeThreeDProt(torch.nn.Module):
    def __init__(self, trial=None, widths=(3, 4, 5, 6)):
        super().__init__()
        ecfp_width, ligand_width, protein_width, joint_width = widths
        self.chem_ecfp_post_fc = torch.nn.Linear(2, ecfp_width)
        self.chem_graph_encoder = torch.nn.Linear(2, ligand_width)
        self.prot_graph_encoder = torch.nn.Linear(2, protein_width)
        self.fc = torch.nn.Linear(
            ecfp_width + ligand_width + protein_width, joint_width
        )
        self.out = torch.nn.Linear(joint_width, 1)

    def forward(self, data):
        ecfp = self.chem_ecfp_post_fc(data["e1_fp"])
        ligand = self.chem_graph_encoder(data["e1_graph"])
        protein = self.prot_graph_encoder(data["e2_graph"])
        return self.out(self.fc(torch.cat((ecfp, ligand, protein), dim=1)))


class WideThreeDProt(FakeThreeDProt):
    def __init__(self, trial=None):
        super().__init__(trial=trial, widths=(280, 290, 300, 310))


class FakeCustomDataLoader:
    def __init__(
        self,
        df,
        batch_size,
        device,
        e1_key_to_graph,
        e2_key_to_graph,
        e1_key_to_fp,
        shuffle=False,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.e1_key_to_graph = e1_key_to_graph
        self.e2_key_to_graph = e2_key_to_graph
        self.e1_key_to_fp = e1_key_to_fp
        self.shuffle = shuffle

    def __iter__(self):
        for start in range(0, len(self.df), self.batch_size):
            rows = self.df.iloc[start : start + self.batch_size]
            ligand_keys = rows["ligand"].tolist()
            protein_keys = rows["protein"].tolist()
            data = {
                "e1_fp": torch.stack(
                    [torch.as_tensor(self.e1_key_to_fp[key]) for key in ligand_keys]
                ).float().to(self.device),
                "e1_graph": torch.stack(
                    [torch.as_tensor(self.e1_key_to_graph[key]) for key in ligand_keys]
                ).float().to(self.device),
                "e2_graph": torch.stack(
                    [torch.as_tensor(self.e2_key_to_graph[key]) for key in protein_keys]
                ).float().to(self.device),
            }
            labels = torch.as_tensor(rows["label"].to_numpy()).float()
            yield labels.reshape(-1, 1).to(self.device), data


class ThreeDProtExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.prepared = self.root / "prepared"
        (self.prepared / "folds").mkdir(parents=True)
        pd.DataFrame(
            [
                {"ligand": "lig-1", "protein": "1abc", "label": 6.0},
                {"ligand": "lig-3", "protein": "3ghi", "label": 8.0},
                {"ligand": "lig-bad", "protein": "bad", "label": 9.0},
            ]
        ).to_csv(self.prepared / "full.csv", index=False)
        (self.prepared / "folds" / "train_fold_setting1.txt").write_text(
            "[[]]", encoding="utf-8"
        )
        (self.prepared / "folds" / "test_fold_setting1.txt").write_text(
            "[]", encoding="utf-8"
        )
        self.ligand_graphs = {
            "lig-1": np.asarray([1.0, 2.0], dtype=np.float32),
            "lig-3": np.asarray([3.0, 4.0], dtype=np.float32),
        }
        self.protein_graphs = {
            "1abc": np.asarray([2.0, 1.0], dtype=np.float32),
            "3ghi": np.asarray([4.0, 3.0], dtype=np.float32),
        }
        self.ecfp = {
            "lig-1": np.asarray([0.5, 1.5], dtype=np.float32),
            "lig-3": np.asarray([2.5, 3.5], dtype=np.float32),
        }
        for filename, payload in (
            ("ligand_to_graph.pkl", self.ligand_graphs),
            ("protein_to_graph.pkl", self.protein_graphs),
            ("ligand_to_ecfp.pkl", self.ecfp),
        ):
            with (self.prepared / filename).open("wb") as handle:
                pickle.dump(payload, handle)

        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "split": "valid",
                    "samples": [
                        {"lmdb_key": "3GHI_protein", "label": 8.0},
                        {"lmdb_key": "2DEF_protein", "label": 7.0},
                        {"lmdb_key": "1ABC_protein", "label": 6.0},
                        {"lmdb_key": "BAD_protein", "label": 9.0},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def checkpoint(self, model_class=FakeThreeDProt, name="model.pt"):
        path = self.root / name
        torch.manual_seed(17)
        model = model_class(trial=FakeTrial({}))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": {"data_dir": str(self.prepared)},
            },
            path,
        )
        return path, model

    def dependencies(self, model_class=FakeThreeDProt):
        return FakeCustomDataLoader, FakeTrial, model_class, {}

    def test_feature_forward_matches_normal_model_and_layout(self):
        _, model = self.checkpoint()
        data = {
            "e1_fp": torch.randn(2, 2),
            "e1_graph": torch.randn(2, 2),
            "e2_graph": torch.randn(2, 2),
        }
        model.eval()
        with torch.inference_mode():
            prediction, parts = forward_three_dprot_features(model, data)
            expected = model(data).reshape(-1)
        torch.testing.assert_close(prediction, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual([part.shape[1] for part in parts], [3, 4, 5, 6])

    def test_export_preserves_manifest_order_and_masks_missing_records(self):
        checkpoint, _ = self.checkpoint()
        with mock.patch(
            "dta.rank_fusion_v4.three_dprot_export._build_dependencies",
            return_value=self.dependencies(),
        ):
            export_three_dprot(
                self.manifest,
                checkpoint,
                self.root / "output",
                split="valid",
                device="cpu",
                batch_size=2,
            )

        artifact = load_expert_artifact(
            self.root / "output",
            expected_expert="three_dprot",
            expected_split="valid",
        )
        self.assertEqual(
            artifact.keys,
            ("3GHI_protein", "2DEF_protein", "1ABC_protein", "BAD_protein"),
        )
        np.testing.assert_array_equal(artifact.available, [True, False, True, False])
        self.assertEqual(artifact.latents.shape, (4, 18))
        np.testing.assert_array_equal(artifact.predictions[~artifact.available], 0.0)
        np.testing.assert_array_equal(artifact.latents[~artifact.available], 0.0)
        adapter = json.loads(
            (self.root / "output" / "adapter_metadata.json").read_text()
        )
        self.assertEqual(adapter["available_count"], 2)
        self.assertEqual(adapter["missing_count"], 2)
        self.assertEqual(
            {item["key"] for item in adapter["missing"]},
            {"2DEF_protein", "BAD_protein"},
        )

    def test_wide_latent_projection_is_reproducible_across_batch_sizes(self):
        checkpoint, _ = self.checkpoint(WideThreeDProt, name="wide.pt")
        for name, batch_size in (("first", 1), ("second", 3)):
            with mock.patch(
                "dta.rank_fusion_v4.three_dprot_export._build_dependencies",
                return_value=self.dependencies(WideThreeDProt),
            ):
                export_three_dprot(
                    self.manifest,
                    checkpoint,
                    self.root / name,
                    split="valid",
                    device="cpu",
                    batch_size=batch_size,
                )
        first = load_expert_artifact(
            self.root / "first",
            expected_expert="three_dprot",
            expected_split="valid",
        )
        second = load_expert_artifact(
            self.root / "second",
            expected_expert="three_dprot",
            expected_split="valid",
        )
        self.assertEqual(first.latents.shape, (4, 1024))
        np.testing.assert_allclose(first.latents, second.latents, rtol=1e-5, atol=1e-5)
        first_metadata = json.loads(
            (self.root / "first" / "adapter_metadata.json").read_text()
        )
        second_metadata = json.loads(
            (self.root / "second" / "adapter_metadata.json").read_text()
        )
        self.assertEqual(first_metadata["projection_input_dimension"], 1180)
        self.assertEqual(first_metadata["projection_output_dimension"], 1024)
        self.assertEqual(
            first_metadata["projection_sha256"],
            second_metadata["projection_sha256"],
        )

    def test_missing_prepared_directory_fails_before_model_construction(self):
        checkpoint, _ = self.checkpoint()
        dependencies = mock.Mock(return_value=self.dependencies())
        with mock.patch(
            "dta.rank_fusion_v4.three_dprot_export._build_dependencies",
            dependencies,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "prepared 3DProt"):
                export_three_dprot(
                    self.manifest,
                    checkpoint,
                    self.root / "missing-output",
                    split="valid",
                    device="cpu",
                    data_dir=self.root / "does-not-exist",
                )
        dependencies.assert_not_called()

    def test_incomplete_checkpoint_terminates_export(self):
        checkpoint = self.root / "bad.pt"
        torch.save({"args": {"data_dir": str(self.prepared)}}, checkpoint)
        with self.assertRaisesRegex(ValueError, "model_state_dict"):
            export_three_dprot(
                self.manifest,
                checkpoint,
                self.root / "bad-output",
                split="valid",
                device="cpu",
            )

    def test_adapter_metadata_is_written_only_after_artifact(self):
        checkpoint, _ = self.checkpoint()
        output = self.root / "failed-output"
        with mock.patch(
            "dta.rank_fusion_v4.three_dprot_export._build_dependencies",
            return_value=self.dependencies(),
        ), mock.patch(
            "dta.rank_fusion_v4.three_dprot_export.write_expert_artifact",
            side_effect=RuntimeError("artifact failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "artifact failed"):
                export_three_dprot(
                    self.manifest,
                    checkpoint,
                    output,
                    split="valid",
                    device="cpu",
                )
        self.assertFalse((output / "adapter_metadata.json").exists())

    def test_real_checkpoint_feature_forward_matches_upstream_model(self):
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            project_root
            / "external_benchmarks/results/3dprot_dta_lp_official_ckpt/best_model.pt"
        )
        data_dir = (
            project_root
            / "external_benchmarks/3d-prot-dta/data/lp_pdbbind_smoke"
        )
        if not checkpoint_path.is_file() or not data_dir.is_dir():
            self.skipTest("real 3DProt checkpoint fixture is unavailable")
        CustomDataLoader, CustomTrial, Model, hp = _build_dependencies()
        with (data_dir / "ligand_to_graph.pkl").open("rb") as handle:
            ligand_graphs = pickle.load(handle)
        with (data_dir / "protein_to_graph.pkl").open("rb") as handle:
            protein_graphs = pickle.load(handle)
        with (data_dir / "ligand_to_ecfp.pkl").open("rb") as handle:
            ecfp = pickle.load(handle)
        frame = pd.read_csv(data_dir / "full.csv").iloc[:1]
        loader = CustomDataLoader(
            df=frame[["ligand", "protein", "label"]].copy(),
            batch_size=1,
            device=torch.device("cpu"),
            e1_key_to_graph=ligand_graphs,
            e2_key_to_graph=protein_graphs,
            e1_key_to_fp=ecfp,
            shuffle=False,
        )
        _, data = next(iter(loader))
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        model = Model(trial=CustomTrial(hp=hp)).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        with torch.inference_mode():
            expected = model(data).reshape(-1)
            actual, parts = forward_three_dprot_features(model, data)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(sum(part.shape[1] for part in parts), 2816)


if __name__ == "__main__":
    unittest.main()
