import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1] / "external_benchmarks" / "imported_index_20260721" / "boltz_dta"
sys.path.insert(0, str(ROOT))


def make_args(**overrides):
    values = {
        "model": "pairwise",
        "out_dir": "/tmp/boltz-test",
        "select_metric": "ci",
        "min_delta": 1e-6,
        "dim": 8,
        "hidden": 8,
        "pair_dim": 4,
        "num_blocks": 1,
        "dropout": 0.1,
        "no_local": False,
        "no_global": False,
        "no_geom": False,
        "no_pair_delta": False,
        "pocket_radius": 10.0,
        "min_pocket_residues": 16,
        "max_pocket_residues": 64,
        "max_ligand_atoms": 64,
        "edge_cutoff": 5.0,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "epochs": 1,
        "early_stop": 0,
        "device": "cpu",
        "seed": 0,
        "log_every": 1,
        "rank_loss_weight": 0.0,
        "rank_min_gap": 0.1,
        "rank_temperature": 0.5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestBoltzDTACheckpointEval(unittest.TestCase):
    def test_save_best_checkpoint_writes_recoverable_payload_and_metrics(self):
        from dta.train import save_best_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            args = make_args(out_dir=str(out_dir))
            model = torch.nn.Linear(2, 1)
            best = {
                "epoch": 3,
                "valid": {"rmse": 1.2, "pearson": 0.5, "spearman": 0.4, "ci": 0.71},
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }

            save_best_checkpoint(out_dir, model, args, best, ymu=6.0, ysd=1.5, n_params=3, n_by_split={"valid": 2})

            checkpoint = torch.load(out_dir / "model.pt", map_location="cpu")
            metrics = (out_dir / "best_metrics.json").read_text(encoding="utf-8")
            self.assertEqual(checkpoint["best_val_epoch"], 3)
            self.assertEqual(checkpoint["best_valid_metrics"]["ci"], 0.71)
            self.assertEqual(checkpoint["config"]["select_metric"], "ci")
            self.assertIn("state_dict", checkpoint)
            self.assertIn('"best_val_epoch": 3', metrics)

    def test_export_predictions_from_checkpoint_writes_ensemble_inputs(self):
        from dta.train import export_predictions_from_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            args = make_args(out_dir=str(out_dir))
            checkpoint = {
                "state_dict": {},
                "y_mu": 5.0,
                "y_sd": 2.0,
                "best_val_epoch": 4,
                "best_valid_metrics": {"ci": 0.72, "rmse": 1.1},
                "config": vars(args),
            }

            def predict(split):
                if split == "valid":
                    return np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0])
                return np.array([3.0, 2.0, 1.0]), np.array([3.0, 1.5, 1.0])

            result = export_predictions_from_checkpoint(
                out_dir=out_dir,
                checkpoint=checkpoint,
                args=args,
                predict_fn=predict,
                model_params=0,
                n_by_split={"valid": 3, "test": 3},
            )

            self.assertTrue((out_dir / "valid_pred.npy").exists())
            self.assertTrue((out_dir / "test_pred.npy").exists())
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertEqual(result["best_val_epoch"], 4)
            self.assertEqual(np.load(out_dir / "valid_pred.npy").shape, (3, 2))
            self.assertIn("ci", result["test"])

    def test_old_pairwise_checkpoint_without_delta_gate_uses_old_readout(self):
        from dta.train import apply_checkpoint_config

        args = make_args(no_pair_delta=False)
        checkpoint = {
            "config": {"model": "pairwise"},
            "state_dict": {"head.0.weight": torch.zeros(1)},
        }

        out = apply_checkpoint_config(args, checkpoint)

        self.assertTrue(out.no_pair_delta)


if __name__ == "__main__":
    unittest.main()
