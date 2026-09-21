import copy
from decimal import Decimal
from fractions import Fraction
import math
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from dta.anchored_expert_fusion_v5.model import (
    AnchoredResidualModel,
    GatedBlock,
    build_residual_features,
)


class AnchoredResidualModelTest(unittest.TestCase):
    @staticmethod
    def batch():
        return {
            "base_predictions": torch.tensor([7.0, 8.0], dtype=torch.float32),
            "raw_predictions": torch.tensor(
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                dtype=torch.float32,
            ),
            "standardized_predictions": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=torch.float32,
            ),
            "available": torch.tensor(
                [[True, False, True], [True, True, False]],
                dtype=torch.bool,
            ),
            "latents": [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
                torch.tensor(
                    [[11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]]
                ),
            ],
        }

    @staticmethod
    def model(budget=0.25, dropout=0.10):
        return AnchoredResidualModel(
            latent_widths=(2, 3, 4), budget=budget, dropout=dropout
        )

    @staticmethod
    def count_item_calls(function):
        original_item = torch.Tensor.item
        call_count = 0

        def counted_item(tensor, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_item(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "item", counted_item):
            result = function()
        return result, call_count

    def test_feature_width_scalar_values_and_architecture(self):
        model = self.model()
        with torch.no_grad():
            for adapter in model.latent_adapters:
                adapter.weight.zero_()
                adapter.bias.zero_()
        features = build_residual_features(self.batch(), model.latent_adapters)

        expected_scalars = torch.tensor(
            [
                [10, 10, 30, 1, 1, 3, 0, -20, -20, 1, 0, 1],
                [40, 50, 40, 4, 5, 4, -10, 0, 10, 1, 1, 0],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(tuple(features.shape), (2, 204))
        torch.testing.assert_close(features[:, :12], expected_scalars, rtol=0, atol=0)
        torch.testing.assert_close(
            features[:, 12:], torch.zeros(2, 192), rtol=0, atol=0
        )
        self.assertEqual(model.feature_width, 204)
        self.assertEqual(model.adapter_width, 64)
        self.assertEqual(model.latent_widths, (2, 3, 4))
        self.assertEqual(model.budget, 0.25)
        self.assertEqual(model.dropout, 0.10)
        self.assertEqual(
            [(a.in_features, a.out_features) for a in model.latent_adapters],
            [(2, 64), (3, 64), (4, 64)],
        )
        self.assertTrue(all(isinstance(block, GatedBlock) for block in model.blocks))
        self.assertEqual(
            [(b.projection.in_features, b.normalization.normalized_shape[0]) for b in model.blocks],
            [(204, 256), (256, 128), (128, 64)],
        )
        self.assertEqual(
            [b.projection.out_features for b in model.blocks], [512, 256, 128]
        )
        self.assertIsInstance(model.output_layer, nn.Linear)
        self.assertEqual((model.output_layer.in_features, model.output_layer.out_features), (64, 1))

    def test_gated_block_matches_manual_math_when_dropout_is_zero(self):
        block = GatedBlock(3, 2, dropout=0.0)
        with torch.no_grad():
            block.projection.weight.copy_(
                torch.tensor(
                    [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2],
                    ]
                )
            )
            block.projection.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
            block.normalization.weight.copy_(torch.tensor([1.5, 0.5]))
            block.normalization.bias.copy_(torch.tensor([-0.25, 0.75]))
        inputs = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]])
        projected = F.linear(
            inputs, block.projection.weight, block.projection.bias
        )
        value, gate = projected.chunk(2, dim=-1)
        gated = F.silu(value) * torch.sigmoid(gate)
        expected = F.layer_norm(
            gated,
            (2,),
            block.normalization.weight,
            block.normalization.bias,
            block.normalization.eps,
        )

        block.train()
        torch.testing.assert_close(block(inputs), expected, rtol=1e-6, atol=1e-7)
        block.eval()
        torch.testing.assert_close(block(inputs), expected, rtol=1e-6, atol=1e-7)

    def test_gated_block_configures_dropout_and_obeys_train_eval_mode(self):
        torch.manual_seed(23)
        block = GatedBlock(3, 2, dropout=0.4)
        inputs = torch.arange(48, dtype=torch.float32).reshape(16, 3) / 10.0
        projected = F.linear(
            inputs, block.projection.weight, block.projection.bias
        )
        value, gate = projected.chunk(2, dim=-1)
        normalized = F.layer_norm(
            F.silu(value) * torch.sigmoid(gate),
            (2,),
            block.normalization.weight,
            block.normalization.bias,
            block.normalization.eps,
        )

        self.assertEqual(block.dropout_layer.p, 0.4)
        block.eval()
        self.assertFalse(block.dropout_layer.training)
        torch.testing.assert_close(block(inputs), normalized, rtol=1e-6, atol=1e-7)

        block.train()
        self.assertTrue(block.dropout_layer.training)
        torch.manual_seed(91)
        actual = block(inputs)
        torch.manual_seed(91)
        expected = F.dropout(normalized, p=0.4, training=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_unavailable_predictions_and_latents_are_fully_masked_after_adapters(self):
        model = self.model()
        with torch.no_grad():
            for index, adapter in enumerate(model.latent_adapters):
                adapter.weight.fill_(0.125 * (index + 1))
                adapter.bias.fill_(2.0 + index)
        original = self.batch()
        changed = copy.deepcopy(original)
        unavailable = ~changed["available"]
        changed["raw_predictions"][unavailable] = 1.0e20
        changed["standardized_predictions"][unavailable] = -1.0e20
        for expert, latent in enumerate(changed["latents"]):
            latent[~changed["available"][:, expert]] = 1.0e20

        first = build_residual_features(original, model.latent_adapters)
        second = build_residual_features(changed, model.latent_adapters)

        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for expert in range(3):
            start = 12 + expert * 64
            available_rows = original["available"][:, expert]
            expected_available = F.silu(
                F.linear(
                    original["latents"][expert][available_rows],
                    model.latent_adapters[expert].weight,
                    model.latent_adapters[expert].bias,
                )
            )
            torch.testing.assert_close(
                first[available_rows, start : start + 64],
                expected_available,
                rtol=1e-6,
                atol=1e-7,
            )
            missing_rows = ~available_rows
            torch.testing.assert_close(
                first[missing_rows, start : start + 64],
                torch.zeros(int(missing_rows.sum()), 64),
                rtol=0,
                atol=0,
            )

    def test_zero_initialized_output_is_bit_exact_base_in_train_and_eval(self):
        model = self.model(dropout=0.7)
        batch = self.batch()
        self.assertEqual(torch.count_nonzero(model.output_layer.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(model.output_layer.bias).item(), 0)

        model.train()
        train_output = model(batch)
        model.eval()
        eval_output = model(batch)

        torch.testing.assert_close(
            train_output, batch["base_predictions"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            eval_output, batch["base_predictions"], rtol=0, atol=0
        )

    def test_correction_is_bounded_for_every_approved_budget_and_bias_sign(self):
        for budget in (0.10, 0.25, 0.50):
            for bias in (-100.0, 100.0):
                with self.subTest(budget=budget, bias=bias):
                    model = self.model(budget=budget, dropout=0.0)
                    with torch.no_grad():
                        model.output_layer.bias.fill_(bias)
                    base = self.batch()["base_predictions"]
                    correction = model(self.batch()) - base
                    self.assertTrue(torch.all(correction.abs() <= budget + 1e-6))
                    self.assertTrue(torch.all(torch.sign(correction) == math.copysign(1.0, bias)))

    def test_rejects_invalid_budget_values(self):
        for budget in (True, False, 0.0, 0.1001, 0.2, 1.0, float("nan"), float("inf")):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "budget"):
                    self.model(budget=budget)
        for budget in (0.1, 0.25, 0.5):
            self.model(budget=budget)

    def test_accepts_exact_numeric_budget_equivalents(self):
        equivalents = (
            (Decimal("0.10"), 0.10),
            (Fraction(1, 4), 0.25),
            (Fraction(1, 2), 0.50),
        )
        for supplied, expected in equivalents:
            with self.subTest(supplied=supplied):
                self.assertEqual(self.model(budget=supplied).budget, expected)

    def test_rejects_invalid_batch_shapes_dtypes_and_latents(self):
        cases = []
        batch = self.batch()
        batch["base_predictions"] = batch["base_predictions"][:, None]
        cases.append((batch, "base_predictions"))
        batch = self.batch()
        batch["raw_predictions"] = batch["raw_predictions"][:, :2]
        cases.append((batch, "raw_predictions"))
        batch = self.batch()
        batch["standardized_predictions"] = batch["standardized_predictions"][:1]
        cases.append((batch, "standardized_predictions"))
        batch = self.batch()
        batch["available"] = batch["available"].float()
        cases.append((batch, "available"))
        batch = self.batch()
        batch["latents"] = batch["latents"][:2]
        cases.append((batch, "three latents"))
        batch = self.batch()
        batch["latents"][1] = batch["latents"][1][:, :2]
        cases.append((batch, "latent 1"))
        batch = self.batch()
        batch["raw_predictions"] = batch["raw_predictions"].long()
        cases.append((batch, "floating"))

        model = self.model()
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model(invalid)

    def test_rejects_mismatched_float_dtype_and_device(self):
        model = self.model()
        batch = self.batch()
        batch["raw_predictions"] = batch["raw_predictions"].double()
        with self.assertRaisesRegex(ValueError, "dtype"):
            model(batch)

        if torch.cuda.is_available():
            model = self.model().cuda()
            batch = self.batch()
            batch["base_predictions"] = batch["base_predictions"].cuda()
            batch["standardized_predictions"] = batch[
                "standardized_predictions"
            ].cuda()
            batch["available"] = batch["available"].cuda()
            batch["latents"] = [latent.cuda() for latent in batch["latents"]]
            with self.assertRaisesRegex(ValueError, "device"):
                model(batch)

    def test_valid_feature_build_uses_one_scalar_synchronization(self):
        model = self.model()
        features, item_calls = self.count_item_calls(
            lambda: build_residual_features(self.batch(), model.latent_adapters)
        )
        self.assertEqual(tuple(features.shape), (2, 204))
        self.assertEqual(item_calls, 1)

    def test_rejects_nonfinite_float_inputs_and_incomplete_upper_bound(self):
        fields = (
            ("base_predictions", (0,)),
            ("raw_predictions", (0, 1)),
            ("standardized_predictions", (1, 2)),
        )
        model = self.model()
        for field, index in fields:
            batch = self.batch()
            batch[field][index] = float("nan")
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError, field + ".*finite"
                ):
                    model(batch)
        batch = self.batch()
        batch["latents"][2][1, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "latent 2.*finite"):
            model(batch)
        batch = self.batch()
        batch["available"][1, 0] = False
        with self.assertRaisesRegex(ValueError, "Upper-Bound"):
            model(batch)

    def test_float32_output_shape_gradients_and_input_immutability(self):
        model = self.model(dropout=0.0)
        with torch.no_grad():
            model.output_layer.weight.copy_(
                torch.linspace(-0.05, 0.05, model.output_layer.in_features)[None, :]
            )
            model.output_layer.bias.fill_(0.1)
        batch = self.batch()
        originals = {
            key: value.clone() if torch.is_tensor(value) else [item.clone() for item in value]
            for key, value in batch.items()
        }

        output = model(batch)
        loss = output.square().mean()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2,))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        upstream_parameters = list(model.latent_adapters.parameters()) + list(
            model.blocks.parameters()
        )
        self.assertTrue(
            any(torch.count_nonzero(parameter.grad).item() > 0 for parameter in upstream_parameters)
        )
        for key in ("base_predictions", "raw_predictions", "standardized_predictions", "available"):
            torch.testing.assert_close(batch[key], originals[key], rtol=0, atol=0)
        for actual, expected in zip(batch["latents"], originals["latents"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(hasattr(model, "base_prediction"))


if __name__ == "__main__":
    unittest.main()
