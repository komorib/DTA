import unittest

import torch
from torch import nn


SOURCE_DIMS = {
    "gerna_2d": 16384,
    "gerna_3d": 16384,
    "gerna_head": 128,
    "interface_protein": 128,
    "interface_ligand": 96,
    "interface_local": 128,
    "interface_head": 128,
}


def make_features(batch_size=2, requires_grad=True):
    torch.manual_seed(307)
    gerna = {}
    interface = {}
    for name, dimension in SOURCE_DIMS.items():
        value = torch.randn(
            batch_size, dimension, requires_grad=requires_grad
        )
        if name.startswith("gerna_"):
            gerna[name] = value
        else:
            interface[name] = value
    return gerna, interface


class TestHybridAffinityFusion(unittest.TestCase):
    def test_mlp_and_transformer_outputs_are_finite_and_fully_differentiable(self):
        from net.hybrid_affinity import HybridAffinityFusion

        for fusion_type in ("mlp", "transformer"):
            with self.subTest(fusion_type=fusion_type):
                torch.manual_seed(311)
                model = HybridAffinityFusion(
                    fusion_type=fusion_type,
                    dropout=0.0,
                )
                gerna, interface = make_features()
                y_g = torch.tensor([[5.0], [6.0]], requires_grad=True)
                y_i = torch.tensor([[7.0], [8.0]], requires_grad=True)

                output = model(gerna, interface, y_g, y_i)

                self.assertEqual(output["prediction"].shape, (2, 1))
                self.assertEqual(
                    output["branch_predictions"].shape, (2, 2)
                )
                self.assertEqual(
                    output["expert_predictions"].shape, (2, 3)
                )
                self.assertEqual(output["expert_weights"].shape, (2, 3))
                self.assertTrue(
                    torch.isfinite(output["prediction"]).all()
                )
                self.assertTrue(
                    torch.isfinite(output["expert_predictions"]).all()
                )
                self.assertTrue(
                    torch.isfinite(output["expert_weights"]).all()
                )
                torch.testing.assert_close(
                    output["expert_weights"].sum(dim=-1),
                    torch.ones(2),
                )

                output["prediction"].sum().backward()
                for name, value in {
                    **gerna,
                    **interface,
                    "gerna_prediction": y_g,
                    "interface_prediction": y_i,
                }.items():
                    self.assertIsNotNone(value.grad, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)
                required_prefixes = (
                    "adapters.",
                    "source_embedding",
                    "branch_embedding",
                    "fusion.",
                    "mlp_expert.",
                    "bilinear_expert.",
                    "calibrated_expert.",
                    "gate.",
                )
                for name, parameter in model.named_parameters():
                    if name.startswith(required_prefixes):
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(
                            torch.isfinite(parameter.grad).all(), name
                        )

    def test_experts_can_predict_outside_both_branch_predictions(self):
        from net.hybrid_affinity import HybridAffinityFusion

        model = HybridAffinityFusion(
            fusion_type="mlp", dropout=0.0
        ).eval()
        gerna, interface = make_features(
            batch_size=1, requires_grad=False
        )
        with torch.no_grad():
            for expert in (
                model.mlp_expert,
                model.calibrated_expert,
            ):
                for parameter in expert.parameters():
                    parameter.zero_()
            model.mlp_expert[-1].bias.fill_(20.0)
            model.calibrated_expert[-1].bias.fill_(30.0)
            model.bilinear_expert.weight.zero_()
            model.bilinear_expert.bias.fill_(40.0)

        output = model(
            gerna,
            interface,
            torch.tensor([[5.0]]),
            torch.tensor([[7.0]]),
        )

        self.assertGreater(float(output["prediction"]), 7.0)
        torch.testing.assert_close(
            output["expert_predictions"],
            torch.tensor([[20.0, 40.0, 30.0]]),
        )

    def test_source_dimension_mismatch_names_the_source(self):
        from net.hybrid_affinity import HybridAffinityFusion

        model = HybridAffinityFusion(fusion_type="mlp")
        gerna, interface = make_features(requires_grad=False)
        gerna["gerna_2d"] = torch.randn(2, 8)

        with self.assertRaisesRegex(ValueError, "gerna_2d"):
            model(
                gerna,
                interface,
                torch.randn(2, 1),
                torch.randn(2, 1),
            )

    def test_float64_inputs_are_normalized_to_fusion_dtype(self):
        from net.hybrid_affinity import HybridAffinityFusion

        model = HybridAffinityFusion("mlp", dropout=0.0)
        gerna, interface = make_features(
            batch_size=1, requires_grad=False
        )
        gerna = {
            name: value.double() for name, value in gerna.items()
        }
        interface = {
            name: value.double() for name, value in interface.items()
        }

        output = model(
            gerna,
            interface,
            torch.tensor([[5.0]], dtype=torch.float64),
            torch.tensor([[7.0]], dtype=torch.float64),
        )

        self.assertEqual(output["prediction"].dtype, torch.float32)

    def test_unknown_fusion_type_is_rejected(self):
        from net.hybrid_affinity import HybridAffinityFusion

        with self.assertRaisesRegex(ValueError, "fusion_type"):
            HybridAffinityFusion(fusion_type="cnn")

    def test_token_dimension_is_fixed_at_128(self):
        from net.hybrid_affinity import HybridAffinityFusion

        with self.assertRaisesRegex(ValueError, "token_dim"):
            HybridAffinityFusion(fusion_type="mlp", token_dim=64)


class FakeGerNA(nn.Module):
    def __init__(self, features, prediction):
        super().__init__()
        self.features = features
        self.register_buffer("base_prediction", prediction)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.requested = False
        self.captured = None
        self.take_count = 0
        self.discard_count = 0
        self.failure_mode = None

    def request_affinity_features(self):
        if self.requested:
            raise RuntimeError("already requested")
        self.requested = True

    def forward(self, **inputs):
        if not self.requested:
            raise RuntimeError("features were not requested")
        if self.failure_mode == "raise":
            self.failure_mode = None
            raise RuntimeError("synthetic GerNA failure")
        self.captured = {
            name: value * self.scale
            for name, value in self.features.items()
        }
        prediction = self.base_prediction * self.scale
        if self.failure_mode == "malformed":
            self.failure_mode = None
            return prediction
        return prediction, inputs["pairwise"]

    def take_affinity_features(self):
        if not self.requested:
            raise RuntimeError("features were not requested")
        self.requested = False
        self.take_count += 1
        features = self.captured
        self.captured = None
        return features

    def discard_affinity_features(self):
        self.requested = False
        self.captured = None
        self.discard_count += 1


class FakeInterface(nn.Module):
    def __init__(self, features, prediction):
        super().__init__()
        self.features = features
        self.register_buffer("base_prediction", prediction)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, item, return_features=False):
        if not return_features:
            raise RuntimeError("expanded features were not requested")
        return self.base_prediction * self.scale, {
            name: value * self.scale
            for name, value in self.features.items()
        }


class TestHybridAffinityModel(unittest.TestCase):
    def test_wrapper_requests_and_consumes_backbone_features(self):
        from net.hybrid_affinity import (
            HybridAffinityFusion,
            HybridAffinityModel,
        )

        gerna_features, interface_features = make_features(
            batch_size=1, requires_grad=False
        )
        gerna = FakeGerNA(gerna_features, torch.tensor([[5.0]]))
        interface = FakeInterface(
            interface_features, torch.tensor([[7.0]])
        )
        model = HybridAffinityModel(
            gerna,
            interface,
            HybridAffinityFusion("mlp", dropout=0.0),
        )

        output = model(
            {"pairwise": torch.ones(1, 1, 1)},
            {"key": "sample-a"},
        )

        self.assertEqual(output["prediction"].shape, (1, 1))
        self.assertEqual(gerna.take_count, 1)
        self.assertFalse(gerna.requested)
        torch.testing.assert_close(
            output["branch_predictions"],
            torch.tensor([[5.0, 7.0]]),
        )
        output["prediction"].sum().backward()
        self.assertIsNotNone(gerna.scale.grad)
        self.assertIsNotNone(interface.scale.grad)

    def test_wrapper_runs_only_gerna_outside_cpu_autocast(self):
        from net.hybrid_affinity import (
            HybridAffinityFusion,
            HybridAffinityModel,
        )

        gerna_features, interface_features = make_features(
            batch_size=1, requires_grad=False
        )

        class AutocastAwareGerNA(FakeGerNA):
            def forward(self, **inputs):
                self.autocast_enabled = (
                    torch.is_autocast_cpu_enabled()
                )
                return super().forward(**inputs)

        class AutocastAwareInterface(FakeInterface):
            def forward(self, item, return_features=False):
                self.autocast_enabled = (
                    torch.is_autocast_cpu_enabled()
                )
                return super().forward(
                    item, return_features=return_features
                )

        gerna = AutocastAwareGerNA(
            gerna_features, torch.tensor([[5.0]])
        )
        interface = AutocastAwareInterface(
            interface_features, torch.tensor([[7.0]])
        )
        model = HybridAffinityModel(
            gerna,
            interface,
            HybridAffinityFusion("mlp", dropout=0.0),
        )

        with torch.autocast(
            device_type="cpu", dtype=torch.bfloat16
        ):
            output = model(
                {"pairwise": torch.ones(1, 1, 1)},
                {"key": "sample-a"},
            )

        self.assertFalse(gerna.autocast_enabled)
        self.assertTrue(interface.autocast_enabled)
        self.assertTrue(torch.isfinite(output["prediction"]).all())

    def test_wrapper_recovers_after_gerna_failures(self):
        from net.hybrid_affinity import (
            HybridAffinityFusion,
            HybridAffinityModel,
        )

        gerna_features, interface_features = make_features(
            batch_size=1, requires_grad=False
        )
        gerna = FakeGerNA(gerna_features, torch.tensor([[5.0]]))
        interface = FakeInterface(
            interface_features, torch.tensor([[7.0]])
        )
        model = HybridAffinityModel(
            gerna,
            interface,
            HybridAffinityFusion("mlp", dropout=0.0),
        )
        inputs = {"pairwise": torch.ones(1, 1, 1)}

        for failure_mode, message in (
            ("raise", "synthetic GerNA failure"),
            ("malformed", "return prediction first"),
        ):
            with self.subTest(failure_mode=failure_mode):
                gerna.failure_mode = failure_mode
                with self.assertRaisesRegex(RuntimeError, message):
                    model(inputs, {"key": "sample-a"})
                self.assertFalse(gerna.requested)
                self.assertIsNone(gerna.captured)
                recovered = model(inputs, {"key": "sample-a"})
                self.assertEqual(recovered["prediction"].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
