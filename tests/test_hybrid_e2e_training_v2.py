import unittest

import torch
from torch import nn


def fake_gerna_features(value, batch_size):
    return {
        "gerna_2d_molecule": value.expand(batch_size, 128),
        "gerna_2d_protein": value.expand(batch_size, 128),
        "gerna_3d_molecule": value.expand(batch_size, 128),
        "gerna_3d_protein": value.expand(batch_size, 128),
        "gerna_2d_interaction": value.expand(batch_size, 128, 128),
        "gerna_3d_interaction": value.expand(batch_size, 128, 128),
        "gerna_head": value.expand(batch_size, 128),
    }


class FakeGerNA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.pending = False

    def request_multiscale_affinity_features(self):
        self.pending = True

    def discard_multiscale_affinity_features(self):
        self.pending = False

    def forward(self, batch_size, **unused):
        return self.weight.expand(batch_size, 1),

    def take_multiscale_affinity_features(self):
        if not self.pending:
            raise RuntimeError("features were not requested")
        self.pending = False
        return fake_gerna_features(self.weight.reshape(1, 1), 2)


class FakeInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward_packed(self, packed):
        batch_size = packed.batch_size
        value = self.weight
        return value.expand(batch_size, 1), {
            "interface_ligand_nodes": value.expand(batch_size, 2, 96),
            "interface_protein_nodes": value.expand(batch_size, 2, 96),
            "interface_residue_nodes": value.expand(batch_size, 2, 128),
            "interface_ligand_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            "interface_protein_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            "interface_residue_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            "interface_ligand_coords": value.expand(batch_size, 2, 3),
            "interface_protein_coords": value.expand(batch_size, 2, 3),
            "interface_residue_coords": value.expand(batch_size, 2, 3),
            "interface_ligand_summary": value.expand(batch_size, 96),
            "interface_protein_summary": value.expand(batch_size, 128),
            "interface_local_summary": value.expand(batch_size, 128),
            "interface_head": value.expand(batch_size, 128),
            "interface_score": value.expand(batch_size, 1),
            "interface_projection": value.expand(batch_size, 64),
        }


class PackedStub:
    batch_size = 2


class TestHybridUpperBoundEndToEnd(unittest.TestCase):
    def test_wrapper_routes_packed_features_and_backpropagates_to_backbones(self):
        from net.hybrid_upper_bound import (
            HybridUpperBoundEndToEndModel,
            HybridUpperBoundFusion,
        )

        model = HybridUpperBoundEndToEndModel(
            FakeGerNA(), FakeInterface(),
            HybridUpperBoundFusion(dropout=0.0, drop_path_rate=0.0),
        )
        output = model({"batch_size": 2}, PackedStub())
        self.assertEqual(tuple(output["prediction"].shape), (2, 1))
        output["prediction"].sum().backward()
        self.assertIsNotNone(model.gerna.weight.grad)
        self.assertIsNotNone(model.interface.weight.grad)
        self.assertTrue(torch.isfinite(model.gerna.weight.grad))
        self.assertTrue(torch.isfinite(model.interface.weight.grad))


if __name__ == "__main__":
    unittest.main()



class TestJointOptimizer(unittest.TestCase):
    def test_optimizer_covers_all_trainable_backbones_with_layerwise_decay(self):
        from dta.hybrid_e2e_training_v2 import build_e2e_optimizer
        from net.hybrid_upper_bound import (
            HybridUpperBoundEndToEndModel,
            HybridUpperBoundFusion,
        )

        model = HybridUpperBoundEndToEndModel(
            FakeGerNA(), FakeInterface(),
            nn.Sequential(nn.Linear(1, 1)),
        )
        optimizer = build_e2e_optimizer(model)

        maximum = {}
        parameter_ids = set()
        for group in optimizer.param_groups:
            component = group["component"]
            maximum[component] = max(
                maximum.get(component, 0.0), group["lr"]
            )
            parameter_ids.update(id(parameter) for parameter in group["params"])
        self.assertEqual(
            maximum,
            {"fusion": 3e-5, "interface": 3e-6, "gerna": 1e-6},
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        self.assertEqual(
            parameter_ids,
            {id(parameter) for parameter in model.parameters()},
        )
        for component in ("interface", "gerna"):
            learning_rates = sorted(
                {
                    group["lr"]
                    for group in optimizer.param_groups
                    if group["component"] == component
                },
                reverse=True,
            )
            self.assertGreaterEqual(len(learning_rates), 1)
            for higher, lower in zip(learning_rates, learning_rates[1:]):
                self.assertAlmostEqual(lower / higher, 0.85, places=8)



class TestE2EValidationPolicy(unittest.TestCase):
    def test_raw_ci_controls_ema_evaluation_and_patience(self):
        from dta.hybrid_e2e_training_v2 import E2EValidationTracker

        tracker = E2EValidationTracker(patience=2)
        self.assertTrue(
            tracker.record_raw({"CI": 0.60, "Spearman": 0.2, "RMSE": 2.0})
        )
        self.assertEqual(tracker.stale_epochs, 0)
        self.assertFalse(
            tracker.record_raw({"CI": 0.59, "Spearman": 0.9, "RMSE": 1.0})
        )
        self.assertEqual(tracker.stale_epochs, 1)
        self.assertFalse(tracker.should_stop)
        self.assertFalse(
            tracker.record_raw({"CI": 0.59, "Spearman": 1.0, "RMSE": 0.5})
        )
        self.assertEqual(tracker.stale_epochs, 2)
        self.assertTrue(tracker.should_stop)



class TestCachedFusionTransfer(unittest.TestCase):
    def test_strictly_loads_cached_fusion_state_only(self):
        import tempfile
        from pathlib import Path

        from dta.hybrid_e2e_training_v2 import load_cached_fusion_checkpoint

        source = nn.Linear(3, 1)
        target = nn.Linear(3, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.pth"
            torch.save({"model_state_dict": source.state_dict()}, path)
            artifact = load_cached_fusion_checkpoint(target, path)
        self.assertIn("sha256", artifact)
        for expected, actual in zip(source.parameters(), target.parameters()):
            torch.testing.assert_close(expected, actual)



class TinyJointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, gerna_inputs, packed_interface):
        prediction = self.weight.expand(2, 1)
        return {
            "prediction": prediction,
            "expert_weights": torch.full(
                (2, 8), 0.125, device=prediction.device
            ),
        }


class TestJointGroupTraining(unittest.TestCase):
    def test_group_step_updates_packed_joint_model(self):
        from dta.hybrid_e2e_training_v2 import train_e2e_group

        model = TinyJointModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()

        def prepare(group, device):
            return ["one", "two"], {}, object(), torch.tensor([1.0, 2.0])

        result = train_e2e_group(
            model,
            optimizer,
            [{"key": "one"}, {"key": "two"}],
            "cpu",
            prepare_fn=prepare,
        )
        self.assertTrue(result["optimizer_step"])
        self.assertEqual(result["keys"], ["one", "two"])
        self.assertNotEqual(float(before), float(model.weight.detach()))
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))



class TestJointEvaluation(unittest.TestCase):
    def test_evaluation_collects_packed_predictions_and_restores_mode(self):
        from dta.hybrid_e2e_training_v2 import evaluate_e2e

        model = TinyJointModel().train()
        groups = [[{"key": "one"}, {"key": "two"}]]

        def prepare(group, device):
            return ["one", "two"], {}, object(), torch.tensor([1.0, 2.0])

        result = evaluate_e2e(model, groups, "cpu", prepare_fn=prepare)
        self.assertTrue(model.training)
        self.assertEqual(result["keys"], ["one", "two"])
        self.assertEqual(tuple(result["labels"].shape), (2,))
        self.assertEqual(tuple(result["predictions"].shape), (2,))
        self.assertIn("CI", result["metrics"])



class TestJointFitLoop(unittest.TestCase):
    def test_fit_loop_records_raw_and_conditional_ema_validation(self):
        from dta.hybrid_e2e_training_v2 import fit_e2e

        model = TinyJointModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        groups = [[{"key": "one"}, {"key": "two"}]]

        def prepare(group, device):
            return ["one", "two"], {}, object(), torch.tensor([1.0, 2.0])

        result = fit_e2e(
            model,
            optimizer,
            groups,
            groups,
            "cpu",
            epochs=3,
            patience=2,
            prepare_fn=prepare,
        )
        self.assertGreaterEqual(len(result["history"]), 1)
        self.assertLessEqual(len(result["history"]), 3)
        self.assertIn("raw", result["best"])
        self.assertEqual(result["history"][0]["ema_evaluated"], True)



class TestE2ECli(unittest.TestCase):
    def test_cli_exposes_only_train_valid_joint_inputs(self):
        from dta.train_hybrid_e2e_v2 import build_arg_parser

        args = build_arg_parser().parse_args([
            "--train-manifest", "train.json",
            "--valid-manifest", "valid.json",
            "--train-native-cache", "train-cache",
            "--valid-native-cache", "valid-cache",
            "--gerna-checkpoint", "gerna.pth",
            "--interface-checkpoint", "interface.pth",
            "--cached-checkpoint", "cached.pth",
            "--output", "output",
            "--seed", "42",
            "--device", "cuda:0",
            "--limit", "16",
        ])
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.group_size, 16)
        self.assertFalse(args.amp)
        self.assertEqual(args.limit, 16)
        self.assertFalse(hasattr(args, "test_manifest"))
