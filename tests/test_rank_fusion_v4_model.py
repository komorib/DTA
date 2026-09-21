import unittest

import torch

from dta.rank_fusion_v4.model import RankFusionV4


def inputs(batch_size=5):
    torch.manual_seed(9)
    return (
        [
            torch.randn(batch_size, 7),
            torch.randn(batch_size, 5),
            torch.randn(batch_size, 6),
        ],
        torch.tensor(
            [
                [6.0, 6.5, 5.5],
                [7.0, 6.0, 8.0],
                [5.0, 4.0, 5.5],
                [8.0, 8.5, 7.5],
                [4.0, 3.0, 4.5],
            ]
        )[:batch_size],
        torch.randn(batch_size, 3),
        torch.tensor(
            [
                [True, True, True],
                [True, False, True],
                [True, True, False],
                [True, False, False],
                [True, True, True],
            ]
        )[:batch_size],
    )


class RankFusionModelTest(unittest.TestCase):
    def model(self):
        return RankFusionV4(
            (7, 5, 6),
            model_dim=32,
            layers=2,
            heads=4,
            dropout=0.0,
            drop_path_rate=0.0,
        )

    def test_initialization_is_exact_upper_bound_with_identity_calibration(self):
        model = self.model().eval()
        latents, raw, standardized, available = inputs()
        with torch.no_grad():
            output = model(latents, raw, standardized, available)
        self.assertEqual(
            set(output),
            {
                "prediction",
                "mixture_prediction",
                "rank_residual",
                "mixture_weights",
                "calibrated_predictions",
            },
        )
        torch.testing.assert_close(output["prediction"], raw[:, 0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            output["calibrated_predictions"], raw, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(model.calibration_scale, torch.ones(3))
        torch.testing.assert_close(model.calibration_bias, torch.zeros(3))
        torch.testing.assert_close(
            model.base_mixture_logits, torch.tensor([8.0, -8.0, -8.0])
        )
        self.assertEqual(float(model.route_coefficient.detach()), 0.0)
        self.assertEqual(model.surrogate_gradient_scale, 1.0)
        self.assertTrue(torch.equal(output["rank_residual"], torch.zeros(5)))

    def test_masked_weights_are_exact_zero_and_available_weights_sum_to_one(self):
        model = self.model().eval()
        latents, raw, standardized, available = inputs()
        output = model(latents, raw, standardized, available)
        self.assertTrue(torch.equal(output["mixture_weights"][~available], torch.zeros_like(output["mixture_weights"][~available])))
        torch.testing.assert_close(
            output["mixture_weights"].sum(dim=1),
            torch.ones(len(raw)),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(torch.isfinite(output["prediction"]).all())

    def test_surrogate_path_trains_mixture_while_forward_delta_is_zero(self):
        model = self.model().train()
        latents, raw, standardized, available = inputs()
        output = model(latents, raw, standardized, available)
        torch.testing.assert_close(output["prediction"], raw[:, 0], rtol=0.0, atol=0.0)
        output["prediction"].sum().backward()
        self.assertIsNotNone(model.mixture_head.weight.grad)
        self.assertGreater(float(model.mixture_head.weight.grad.abs().sum()), 0.0)

    def test_learned_route_and_residual_can_move_away_from_baseline(self):
        model = self.model().eval()
        with torch.no_grad():
            model.route_coefficient.fill_(1.0)
            model.residual_head[-1].bias.fill_(0.25)
        latents, raw, standardized, available = inputs()
        output = model(latents, raw, standardized, available)
        self.assertFalse(torch.equal(output["prediction"], raw[:, 0]))
        torch.testing.assert_close(output["rank_residual"], torch.full((5,), 0.25))

    def test_rejects_bad_shapes_and_missing_upper_bound(self):
        model = self.model()
        latents, raw, standardized, available = inputs()
        with self.assertRaisesRegex(ValueError, "Upper-Bound"):
            bad_available = available.clone()
            bad_available[0, 0] = False
            model(latents, raw, standardized, bad_available)
        with self.assertRaisesRegex(ValueError, "latent"):
            model(latents[:2], raw, standardized, available)

    def test_default_architecture_has_four_blocks_and_eight_heads(self):
        model = RankFusionV4((1024, 640, 1024))
        self.assertEqual(len(model.blocks), 4)
        self.assertEqual(model.blocks[0].attention.num_heads, 8)
        self.assertEqual(model.model_dim, 256)


if __name__ == "__main__":
    unittest.main()
