import unittest
import tempfile
from pathlib import Path


class TestThreeDProtLPAdapter(unittest.TestCase):
    def test_ligand_and_protein_graph_shapes_match_official_model_inputs(self):
        from scripts.train_3dprot_dta_lp import (
            build_ligand_graph,
            build_morgan_ecfp,
            build_protein_graph_from_residues,
        )

        ligand_nodes, ligand_edges, ligand_edge_features = build_ligand_graph("CCO")
        self.assertEqual(ligand_nodes.shape[1], 23)
        self.assertEqual(ligand_edges.shape[1], 2)
        self.assertEqual(ligand_edge_features.shape[1], 6)

        ecfp = build_morgan_ecfp("CCO")
        self.assertEqual(ecfp.shape, (2048,))

        residues = [
            {"resname": "ALA", "chain": "A", "resseq": 1, "ca": (0.0, 0.0, 0.0)},
            {"resname": "GLY", "chain": "A", "resseq": 2, "ca": (3.8, 0.0, 0.0)},
            {"resname": "LYS", "chain": "A", "resseq": 3, "ca": (7.6, 0.0, 0.0)},
        ]
        protein_nodes, protein_edges, protein_edge_features = build_protein_graph_from_residues(residues)
        self.assertEqual(protein_nodes.shape[1], 41)
        self.assertEqual(protein_edges.shape[1], 2)
        self.assertEqual(protein_edge_features.shape[1], 10)
        self.assertGreaterEqual(protein_edges.shape[0], 4)


    def test_epoch_outputs_are_written_incrementally(self):
        from scripts.train_3dprot_dta_lp import write_epoch_outputs

        epoch_to_metrics = {
            10: {
                "metrics_train": {"mse": 1.0, "mae": 0.8, "pearsonr": 0.5},
                "metrics_val": {"mse": 2.0, "mae": 1.2, "pearsonr": 0.4, "ci": 0.61},
                "loss_train": 1.0,
                "loss_val": 2.0,
            },
            20: {
                "metrics_train": {"mse": 0.9, "mae": 0.7, "pearsonr": 0.6},
                "metrics_val": {"mse": 1.8, "mae": 1.1, "pearsonr": 0.45, "ci": 0.64},
                "loss_train": 0.9,
                "loss_val": 1.8,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            write_epoch_outputs(
                out_dir=out_dir,
                eval_split="val",
                epoch=10,
                epoch_to_metrics={10: epoch_to_metrics[10]},
                best_epoch=10,
                best_record=epoch_to_metrics[10],
            )
            write_epoch_outputs(
                out_dir=out_dir,
                eval_split="val",
                epoch=20,
                epoch_to_metrics=epoch_to_metrics,
                best_epoch=20,
                best_record=epoch_to_metrics[20],
            )

            history = (out_dir / "history.csv").read_text(encoding="utf-8")
            self.assertIn("epoch,train_loss,val_loss,val_mse,val_mae,val_pearsonr,val_ci", history)
            self.assertIn("10,1.0,2.0,2.0,1.2,0.4,0.61", history)
            self.assertIn("20,0.9,1.8,1.8,1.1,0.45,0.64", history)
            self.assertTrue((out_dir / "official_3dprot_dta_lp_val_metrics.json").exists())
            self.assertTrue((out_dir / "official_3dprot_dta_lp_val_metrics.pkl").exists())
            self.assertIn("\"epoch\": 20", (out_dir / "best_metrics.json").read_text(encoding="utf-8"))


    def test_model_checkpoints_are_written_for_best_and_last(self):
        import torch
        from scripts.train_3dprot_dta_lp import write_model_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            record10 = {"metrics_val": {"ci": 0.61}, "loss_val": 2.0}
            record20 = {"metrics_val": {"ci": 0.60}, "loss_val": 2.1}

            write_model_checkpoints(out_dir, model, optimizer, epoch=10, best_epoch=10, best_record=record10, is_best=True, args={"epochs": 20})
            write_model_checkpoints(out_dir, model, optimizer, epoch=20, best_epoch=10, best_record=record10, is_best=False, args={"epochs": 20})

            self.assertTrue((out_dir / "best_model.pt").exists())
            self.assertTrue((out_dir / "last_model.pt").exists())
            self.assertTrue((out_dir / "training_checkpoint.pt").exists())
            best = torch.load(out_dir / "best_model.pt", map_location="cpu")
            last = torch.load(out_dir / "last_model.pt", map_location="cpu")
            checkpoint = torch.load(out_dir / "training_checkpoint.pt", map_location="cpu")
            self.assertEqual(best["epoch"], 10)
            self.assertEqual(last["epoch"], 20)
            self.assertEqual(checkpoint["epoch"], 20)
            self.assertEqual(checkpoint["best_epoch"], 10)
            self.assertEqual(checkpoint["best_record"], record10)

    def test_eval_only_metrics_are_written_separately(self):
        import json
        from scripts.train_3dprot_dta_lp import write_eval_only_outputs

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            metrics = {"mse": 2.4, "mae": 1.2, "pearsonr": 0.45, "ci": 0.69}
            output_path = write_eval_only_outputs(
                out_dir=out_dir,
                eval_split="test",
                checkpoint_path="/tmp/best_model.pt",
                checkpoint_epoch=90,
                checkpoint_best_epoch=90,
                metrics=metrics,
                loss=2.4,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output_path.name, "official_3dprot_dta_lp_test_eval_only_metrics.json")
            self.assertEqual(payload["split"], "test")
            self.assertEqual(payload["checkpoint_epoch"], 90)
            self.assertEqual(payload["checkpoint_best_epoch"], 90)
            self.assertEqual(payload["metrics"]["ci"], 0.69)
            self.assertEqual(payload["loss"], 2.4)

    def test_spearman_is_added_to_official_3dprot_metrics(self):
        import torch
        from scripts.train_3dprot_dta_lp import add_spearman_metric

        metrics = {"mse": 1.0, "pearsonr": 0.5}
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([1.0, 3.0, 2.0, 4.0])

        add_spearman_metric(metrics, y_true, y_pred)

        self.assertIn("spearman", metrics)
        self.assertAlmostEqual(metrics["spearman"], 0.8, places=6)



if __name__ == "__main__":
    unittest.main()
