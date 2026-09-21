import unittest

import torch
from torch import nn


class Component(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gerna = Component()
        self.interface = Component()
        self.fusion = nn.Linear(2, 2)
        self.residual = nn.Linear(2, 1)


class TestStageOptimizer(unittest.TestCase):
    def _trainable(self, model):
        return {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    def test_stage_trainable_sets_are_guarded(self):
        from dta.anchored_residual_v3.training import configure_stage

        model = Model()
        optimizer, metadata = configure_stage(model, "residual")
        self.assertTrue(self._trainable(model))
        self.assertTrue(all(name.startswith("residual.") for name in self._trainable(model)))
        self.assertEqual(metadata["stage"], "residual")

        optimizer, metadata = configure_stage(model, "fusion")
        self.assertTrue(all(
            name.startswith(("residual.", "fusion.")) for name in self._trainable(model)
        ))

        optimizer, metadata = configure_stage(
            model, "selective_e2e", gerna_top_owners=1, interface_top_owners=1
        )
        trainable = self._trainable(model)
        self.assertIn("gerna.layers.1.weight", trainable)
        self.assertNotIn("gerna.layers.0.weight", trainable)
        self.assertIn("interface.layers.1.weight", trainable)
        self.assertNotIn("interface.layers.0.weight", trainable)
        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))


class TestAnchoredLoss(unittest.TestCase):
    def test_loss_contains_consistency_and_l2_sp(self):
        from dta.anchored_residual_v3.training import anchored_affinity_loss

        parameter = nn.Parameter(torch.tensor([2.0]))
        output = {
            "prediction": torch.tensor([[1.0], [3.0]], requires_grad=True),
            "delta": torch.tensor([[0.5], [-0.5]], requires_grad=True),
            "expert_weights": torch.full((2, 8), 0.125),
        }
        result = anchored_affinity_loss(
            output,
            torch.tensor([1.5, 2.5]),
            consistency_weight=0.2,
            named_parameters={"weight": parameter},
            reference_parameters={"weight": torch.tensor([1.0])},
            l2_sp_weight=0.1,
        )
        self.assertAlmostEqual(float(result["consistency"]), 0.25)
        self.assertAlmostEqual(float(result["l2_sp"]), 1.0)
        result["loss"].backward()
        self.assertIsNotNone(output["delta"].grad)
        self.assertIsNotNone(parameter.grad)


class TestValidationGuard(unittest.TestCase):
    def test_rejects_worse_state_and_restores_baseline(self):
        from dta.anchored_residual_v3.training import ValidationGuard

        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        guard = ValidationGuard(
            model,
            {"CI": 0.70, "Spearman": 0.5, "RMSE": 1.3},
            variant="cached_teacher",
        )
        model.weight.data.fill_(2.0)
        accepted = guard.consider(
            model, {"CI": 0.69, "Spearman": 0.9, "RMSE": 1.0}, "residual"
        )
        self.assertFalse(accepted)
        guard.restore(model)
        self.assertEqual(float(model.weight), 1.0)
        model.weight.data.fill_(3.0)
        self.assertTrue(guard.consider(
            model, {"CI": 0.71, "Spearman": 0.4, "RMSE": 1.4}, "residual"
        ))
        self.assertEqual(guard.variant, "residual")


if __name__ == "__main__":
    unittest.main()
