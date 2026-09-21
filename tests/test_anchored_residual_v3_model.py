import unittest

import torch
from torch import nn


class FakeGerNA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.pending = False

    def request_multiscale_affinity_features(self):
        self.pending = True

    def discard_multiscale_affinity_features(self):
        self.pending = False

    def forward(self, batch_size):
        return self.weight.expand(batch_size, 1),

    def take_multiscale_affinity_features(self):
        if not self.pending:
            raise RuntimeError("features not requested")
        self.pending = False
        return {"gerna_feature": self.weight.expand(2, 1)}


class FakeInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward_packed(self, packed):
        prediction = self.weight.expand(packed.batch_size, 1)
        return prediction, {"interface_feature": prediction}


class FakeFusion(nn.Module):
    dimension = 4

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))

    def forward(self, batch):
        base = 0.5 * (batch["gerna_prediction"] + batch["interface_prediction"])
        latents = self.weight.reshape(1, 1, 4).expand(base.shape[0], 2, 4)
        return {
            "prediction": base,
            "latents": latents,
            "expert_weights": base.new_full((base.shape[0], 8), 0.125),
        }


class Packed:
    batch_size = 2


class TestAnchoredResidualModel(unittest.TestCase):
    def test_zero_init_exactly_reproduces_teacher_and_has_gradient(self):
        from dta.anchored_residual_v3.model import AnchoredResidualModel

        model = AnchoredResidualModel(
            FakeGerNA(), FakeInterface(), FakeFusion(),
            hidden_dimension=16, dropout=0.0,
        )
        teacher = torch.tensor([[4.0], [6.0]])
        output = model({"batch_size": 2}, Packed(), teacher)
        torch.testing.assert_close(output["prediction"], teacher, rtol=0, atol=0)
        torch.testing.assert_close(output["delta"], torch.zeros_like(teacher))
        self.assertEqual(tuple(output["base_prediction"].shape), (2, 1))
        loss = (output["prediction"] - torch.tensor([[5.0], [7.0]])).square().mean()
        loss.backward()
        gradient = model.residual.output.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_rejects_invalid_teacher_shape(self):
        from dta.anchored_residual_v3.model import AnchoredResidualModel

        model = AnchoredResidualModel(FakeGerNA(), FakeInterface(), FakeFusion())
        with self.assertRaisesRegex(ValueError, "teacher_prediction"):
            model({"batch_size": 2}, Packed(), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
