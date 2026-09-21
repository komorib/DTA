import unittest

import numpy as np
import torch

from dta.rank_locked_calibration_v6.calibrators import (
    MonotonicSplineCalibrator,
    PositiveAffineCalibrator,
)
from dta.rank_locked_calibration_v6.model import (
    ConditionalCalibrator,
    calibration_loss,
    scalar_features,
)


def _rmse(labels, predictions):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(labels, dtype=np.float64)
                    - np.asarray(predictions, dtype=np.float64)
                )
                ** 2
            )
        )
    )


def _batch(batch_size=4, latent_widths=(8, 6, 10)):
    generator = torch.Generator().manual_seed(42)
    return {
        "base_predictions": torch.linspace(4.0, 7.0, batch_size),
        "raw_predictions": torch.randn(
            batch_size, 3, generator=generator
        ),
        "standardized_predictions": torch.randn(
            batch_size, 3, generator=generator
        ),
        "available": torch.ones(batch_size, 3, dtype=torch.bool),
        "latents": [
            torch.randn(batch_size, width, generator=generator)
            for width in latent_widths
        ],
    }


class RankLockedV6AnalyticCalibratorTest(unittest.TestCase):
    def test_affine_recovers_shift_and_scale(self):
        score = np.linspace(2.0, 10.0, 200)
        label = 0.63 * score + 2.95
        model = PositiveAffineCalibrator.fit(score, label, objective="mse")
        self.assertGreater(model.slope, 0.0)
        self.assertLess(_rmse(label, model.predict(score)), 1e-8)

    def test_huber_is_less_sensitive_to_large_outlier(self):
        score = np.linspace(0.0, 10.0, 101)
        clean = 0.7 * score + 2.0
        label = clean.copy()
        label[-1] += 100.0
        mse = PositiveAffineCalibrator.fit(score, label, objective="mse")
        huber = PositiveAffineCalibrator.fit(score, label, objective="huber")
        self.assertLess(
            _rmse(clean[:-1], huber.predict(score[:-1])),
            0.25 * _rmse(clean[:-1], mse.predict(score[:-1])),
        )

    def test_spline_improves_nonlinear_compression(self):
        score = np.linspace(0.0, 1.0, 500)
        label = 3.0 + 6.0 * score ** 2
        affine = PositiveAffineCalibrator.fit(
            score, label, objective="mse"
        ).predict(score)
        spline = MonotonicSplineCalibrator.fit(
            score, label, knot_count=16
        ).predict(score)
        self.assertLess(_rmse(label, spline), 0.5 * _rmse(label, affine))

    def test_spline_x_knots_are_unique_train_quantiles(self):
        score = np.linspace(0.0, 1.0, 500)
        label = 3.0 + 6.0 * score ** 2
        model = MonotonicSplineCalibrator.fit(
            score, label, knot_count=8
        )
        expected = np.unique(
            np.quantile(score, np.linspace(0.0, 1.0, 8))
        )
        np.testing.assert_array_equal(model.x_knots, expected)

    def test_spline_deduplicates_knots_and_extrapolates_positively(self):
        score = np.repeat(np.asarray([1.0, 2.0, 4.0, 8.0]), 20)
        label = np.repeat(np.asarray([3.0, 4.0, 6.0, 9.0]), 20)
        model = MonotonicSplineCalibrator.fit(
            score, label, knot_count=8
        )
        self.assertTrue(np.all(np.diff(model.x_knots) > 0.0))
        self.assertTrue(np.all(np.diff(model.y_knots) > 0.0))
        outside = model.predict(np.asarray([-1.0, 10.0]))
        self.assertLess(outside[0], model.y_knots[0])
        self.assertGreater(outside[1], model.y_knots[-1])

    def test_analytic_serialization_round_trip_is_exact(self):
        score = np.linspace(1.0, 9.0, 100)
        label = 2.0 + 0.5 * score + 0.1 * np.sin(score)
        models = (
            PositiveAffineCalibrator.fit(score, label, objective="mse"),
            PositiveAffineCalibrator.fit(score, label, objective="huber"),
            MonotonicSplineCalibrator.fit(score, label, knot_count=8),
        )
        probe = np.asarray([-1.0, 1.5, 6.0, 11.0], dtype=np.float64)
        for model in models:
            with self.subTest(model=type(model).__name__):
                restored = type(model).from_dict(model.to_dict())
                np.testing.assert_array_equal(
                    restored.predict(probe), model.predict(probe)
                )

    def test_analytic_inputs_and_payloads_are_validated(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            PositiveAffineCalibrator.fit(
                np.ones(2), np.ones(3), objective="mse"
            )
        with self.assertRaisesRegex(ValueError, "objective"):
            PositiveAffineCalibrator.fit(
                np.arange(3.0), np.arange(3.0), objective="bad"
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            MonotonicSplineCalibrator.fit(
                np.asarray([1.0, np.nan]), np.ones(2), knot_count=8
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            MonotonicSplineCalibrator.fit(
                np.ones(20), np.ones(20), knot_count=8
            )
        with self.assertRaisesRegex(ValueError, "fields"):
            PositiveAffineCalibrator.from_dict({"slope": 1.0})


class RankLockedV6ConditionalCalibratorTest(unittest.TestCase):
    def test_scalar_features_have_exact_width_thirteen(self):
        batch = _batch()
        features = scalar_features(batch)
        self.assertEqual(tuple(features.shape), (4, 13))
        torch.testing.assert_close(
            features[:, 7:10], batch["available"].to(features.dtype)
        )

    def test_conditional_starts_at_affine_anchor(self):
        model = ConditionalCalibrator((8, 6, 10), 0.63, 2.95)
        batch = _batch()
        torch.testing.assert_close(
            model(batch),
            0.63 * batch["base_predictions"] + 2.95,
        )

    def test_unavailable_expert_latent_is_fully_masked(self):
        model = ConditionalCalibrator((8, 6, 10), 0.63, 2.95)
        model.eval()
        with torch.no_grad():
            model.output_layer.weight.fill_(0.05)
        first = _batch()
        first["available"][:, 1] = False
        second = {
            key: (
                [value.clone() for value in values]
                if key == "latents"
                else values.clone()
            )
            for key, values in first.items()
        }
        second["latents"][1].fill_(1e6)
        with torch.no_grad():
            torch.testing.assert_close(model(first), model(second))

    def test_conditional_rejects_bad_latents_and_nonfinite_scalars(self):
        model = ConditionalCalibrator((8, 6, 10), 0.63, 2.95)
        wrong = _batch()
        wrong["latents"][0] = torch.zeros(4, 7)
        with self.assertRaisesRegex(ValueError, "latent"):
            model(wrong)
        missing = _batch()
        missing["latents"] = missing["latents"][:2]
        with self.assertRaisesRegex(ValueError, "three"):
            model(missing)
        nonfinite = _batch()
        nonfinite["raw_predictions"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(nonfinite)

    def test_calibration_loss_has_finite_backward_gradients(self):
        model = ConditionalCalibrator((8, 6, 10), 0.63, 2.95)
        prediction = model(_batch())
        label = torch.linspace(5.0, 8.0, prediction.shape[0])
        loss = calibration_loss(prediction, label)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))


if __name__ == "__main__":
    unittest.main()
