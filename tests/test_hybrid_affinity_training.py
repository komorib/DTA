import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TinyGerNA(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Linear(2, 2)
        self.high = nn.Linear(2, 2)

    def high_level_affinity_parameters(self):
        return list(self.high.parameters())


class TinyLocalPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.interface_blocks = nn.ModuleList(
            [nn.Linear(2, 2) for _ in range(3)]
        )
        self.interface_pool = nn.Linear(2, 2)


class TinyInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_path = TinyLocalPath()
        self.protein_global = nn.Linear(2, 2)
        self.protein_conditioned_pool = nn.Linear(2, 2)
        self.ligand_conditioned_pool = nn.Linear(2, 2)
        self.affinity_head = nn.Linear(2, 1)


class TinyHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(2, 4), nn.SiLU(), nn.Linear(4, 1))
        self.interface = TinyInterface()
        self.gerna = TinyGerNA()


class TestHybridAffinityLoss(unittest.TestCase):
    def test_correct_order_has_lower_pairwise_loss(self):
        from dta.hybrid_affinity_training import (
            pairwise_logistic_rank_loss,
        )

        target = torch.tensor([1.0, 2.0, 3.0])
        correct = pairwise_logistic_rank_loss(
            torch.tensor([1.0, 2.0, 3.0]), target
        )
        reversed_order = pairwise_logistic_rank_loss(
            torch.tensor([3.0, 2.0, 1.0]), target
        )

        self.assertLess(float(correct), float(reversed_order))

    def test_small_target_gaps_return_differentiable_zero(self):
        from dta.hybrid_affinity_training import (
            pairwise_logistic_rank_loss,
        )

        prediction = torch.tensor(
            [1.0, 2.0, 3.0], requires_grad=True
        )
        loss = pairwise_logistic_rank_loss(
            prediction,
            torch.tensor([1.00, 1.04, 1.09]),
            min_gap=0.1,
        )
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        torch.testing.assert_close(
            prediction.grad, torch.zeros_like(prediction)
        )

    def test_total_loss_matches_manual_formula(self):
        from dta.hybrid_affinity_training import (
            hybrid_affinity_loss,
            pairwise_logistic_rank_loss,
        )

        prediction = torch.tensor([[1.0], [2.5], [4.0]])
        target = torch.tensor([1.5, 2.0, 3.0])
        rank_weight = 0.2
        result = hybrid_affinity_loss(
            prediction, target, rank_weight=rank_weight
        )
        smooth = torch.nn.functional.smooth_l1_loss(
            prediction.reshape(-1), target
        )
        mse = torch.nn.functional.mse_loss(
            prediction.reshape(-1), target
        )
        rank = pairwise_logistic_rank_loss(prediction, target)

        torch.testing.assert_close(result["smooth_l1"], smooth)
        torch.testing.assert_close(result["mse"], mse)
        torch.testing.assert_close(result["rank"], rank)
        torch.testing.assert_close(
            result["loss"], smooth + 0.10 * mse + rank_weight * rank
        )

    def test_rank_mask_includes_exact_threshold_only(self):
        from dta.hybrid_affinity_training import (
            pairwise_logistic_rank_loss,
        )

        prediction = torch.tensor([0.0, 100.0, 0.2])
        target = torch.tensor([0.0, 0.05, 0.1])
        actual = pairwise_logistic_rank_loss(
            prediction, target, min_gap=0.1, temperature=0.5
        )
        expected = torch.nn.functional.softplus(torch.tensor(-0.4))

        torch.testing.assert_close(actual, expected)

    def test_zero_min_gap_excludes_tied_targets(self):
        from dta.hybrid_affinity_training import pairwise_logistic_rank_loss
        actual = pairwise_logistic_rank_loss(
            torch.tensor([100.0, -100.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0]), min_gap=0.0,
        )
        expected = (
            torch.nn.functional.softplus(torch.tensor(198.0))
            + torch.nn.functional.softplus(torch.tensor(-202.0))
        ) / 2.0
        torch.testing.assert_close(actual, expected)


class TestTrainingStages(unittest.TestCase):
    def test_stage_trainable_sets_and_learning_rates(self):
        from dta.hybrid_affinity_training import (
            build_hybrid_optimizer,
            configure_training_stage,
        )

        model = TinyHybrid()
        optimizer = build_hybrid_optimizer(model, stage="head")
        expected_lrs = {
            "head": {
                "fusion": 3e-4,
                "interface": 0.0,
                "gerna": 0.0,
            },
            "high": {
                "fusion": 1e-4,
                "interface": 3e-5,
                "gerna": 1e-5,
            },
            "full": {
                "fusion": 5e-5,
                "interface": 5e-6,
                "gerna": 2e-6,
            },
        }
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["fusion", "interface", "gerna"],
        )
        self.assertEqual(
            {group["name"]: group["lr"] for group in optimizer.param_groups},
            expected_lrs["head"],
        )
        self.assertTrue(
            all(
                group["weight_decay"] == 1e-4
                for group in optimizer.param_groups
            )
        )

        configure_training_stage(model, "head", optimizer)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.fusion.parameters())
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.interface.parameters())
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.gerna.parameters())
        )

        configure_training_stage(model, "high", optimizer)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.gerna.high.parameters())
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.gerna.low.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for block in model.interface.local_path.interface_blocks[-2:]
                for parameter in block.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.interface.local_path.stem.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.interface.local_path.interface_blocks[0].parameters()
            )
        )
        high_modules = (
            model.interface.local_path.interface_pool,
            model.interface.protein_global,
            model.interface.protein_conditioned_pool,
            model.interface.ligand_conditioned_pool,
            model.interface.affinity_head,
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in high_modules
                for parameter in module.parameters()
            )
        )
        self.assertEqual(
            {group["name"]: group["lr"] for group in optimizer.param_groups},
            expected_lrs["high"],
        )

        configure_training_stage(model, "full", optimizer)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        self.assertEqual(
            {group["name"]: group["lr"] for group in optimizer.param_groups},
            expected_lrs["full"],
        )

    def test_stage_changes_keep_optimizer_state(self):
        from dta.hybrid_affinity_training import (
            build_hybrid_optimizer,
            configure_training_stage,
        )

        model = TinyHybrid()
        optimizer = build_hybrid_optimizer(model, stage="head")
        loss = model.fusion(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        tracked = next(model.fusion.parameters())
        state_before = optimizer.state[tracked]

        configure_training_stage(model, "high", optimizer)
        configure_training_stage(model, "full", optimizer)

        self.assertIs(optimizer.state[tracked], state_before)

    def test_refreezing_clears_stale_gradients(self):
        from dta.hybrid_affinity_training import (
            build_hybrid_optimizer, configure_training_stage,
        )
        model = TinyHybrid()
        optimizer = build_hybrid_optimizer(model, stage="full")
        frozen = next(model.gerna.low.parameters())
        frozen.grad = torch.ones_like(frozen)
        before = frozen.detach().clone()
        configure_training_stage(model, "high", optimizer)
        optimizer.step()
        self.assertIsNone(frozen.grad)
        torch.testing.assert_close(frozen, before)


class TestModelEMA(unittest.TestCase):
    def test_update_serialization_and_temporary_application(self):
        from dta.hybrid_affinity_training import ModelEMA

        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(2.0)
        ema = ModelEMA(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(4.0)
        ema.update(model)
        raw_weight = model.weight.detach().clone()

        with ema.average_parameters(model):
            torch.testing.assert_close(
                model.weight, torch.full_like(model.weight, 2.0)
            )
            torch.testing.assert_close(
                model.bias, torch.full_like(model.bias, 3.0)
            )
        torch.testing.assert_close(model.weight, raw_weight)

        clone = ModelEMA(model, decay=0.9)
        clone.load_state_dict(ema.state_dict())
        with clone.average_parameters(model):
            torch.testing.assert_close(
                model.weight, torch.full_like(model.weight, 2.0)
            )

        default_ema = ModelEMA(model)
        self.assertEqual(default_ema.decay, 0.999)
        malformed = default_ema.state_dict()
        malformed["shadow"]["weight"] = malformed["shadow"]["weight"].long()
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            default_ema.load_state_dict(malformed)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class TinyScaler:
    def __init__(self, scale):
        self.scale = float(scale)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state_dict):
        self.scale = float(state_dict["scale"])


class OverflowOnceScaler:
    def __init__(self, scale=8.0, overflow_attempts=1):
        self.current_scale = float(scale)
        self.overflow_attempts = int(overflow_attempts)
        self.unscale_calls = 0
        self.step_calls = 0
        self._found_inf = False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        self.unscale_calls += 1
        self._found_inf = self.unscale_calls <= self.overflow_attempts
        if self._found_inf:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.fill_(float("inf"))
                        return

    def get_scale(self):
        return self.current_scale

    def step(self, optimizer):
        self.step_calls += 1
        if not self._found_inf:
            optimizer.step()

    def update(self):
        if self._found_inf:
            self.current_scale /= 2.0


class CountingEMA:
    def __init__(self):
        self.update_calls = 0

    def update(self, model):
        self.update_calls += 1


class TinyGroupedModel(nn.Module):
    def __init__(self, weight=0.5, nonfinite=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(weight)))
        self.nonfinite = bool(nonfinite)

    def forward(self, gerna_inputs, interface_item):
        prediction = self.weight * gerna_inputs["x"].reshape(1, 1)
        if self.nonfinite:
            prediction = prediction * torch.tensor(float("nan"))
        return {"prediction": prediction}


def prepare_tiny_sample(sample, device):
    return (
        {"x": sample["x"].to(device)},
        {"unused": sample["x"].to(device)},
        torch.tensor([sample["label"]], dtype=torch.float32, device=device),
    )


class TestHybridGroupedTraining(unittest.TestCase):
    def test_prepare_sample_maps_collated_gerna_fields(self):
        from dta.hybrid_affinity_training import prepare_hybrid_sample

        calls = []

        def fake_collate(batch):
            calls.append(batch)
            return "collated"

        unpacked = tuple(torch.tensor(float(index)) for index in range(23))

        def fake_unpack(batch, device):
            self.assertEqual(batch, "collated")
            self.assertEqual(device, torch.device("cpu"))
            return unpacked

        sample = {
            "key": "sample-a",
            "label": 7.25,
            "gerna": ("raw-gerna",),
            "interface": {
                "key": "sample-a",
                "nested": [torch.tensor([3.0])],
            },
        }
        gerna_inputs, interface_item, target = prepare_hybrid_sample(
            sample,
            torch.device("cpu"),
            collate_fn=fake_collate,
            unpack_fn=fake_unpack,
        )

        self.assertEqual(calls, [[sample["gerna"]]])
        expected_indices = {
            "RNA_repre": 0,
            "Seq_mask": 1,
            "RNA_Graph": 3,
            "Mol_Graph": 2,
            "RNA_feats": 4,
            "RNA_C4_coors": 5,
            "RNA_coors": 6,
            "RNA_mask": 7,
            "Mol_feats": 8,
            "Mol_coors": 9,
            "Mol_mask": 10,
            "Mol_LAS_dis": 11,
            "DTI_protein_tokens": 13,
            "DTI_protein_mask": 14,
            "DTI_global": 15,
            "Ligand_unimol_tokens": 16,
            "Ligand_unimol_mask": 17,
            "Ligand_unimol2_tokens": 18,
            "Ligand_unimol2_mask": 19,
            "PBCNet_global": 20,
            "PBCNet_graph": 21,
            "Boltz_interface": 22,
        }
        self.assertEqual(set(gerna_inputs), set(expected_indices))
        for name, index in expected_indices.items():
            torch.testing.assert_close(gerna_inputs[name], unpacked[index])
        torch.testing.assert_close(interface_item["nested"][0], torch.tensor([3.0]))
        torch.testing.assert_close(target, torch.tensor([7.25]))

    def test_group_uses_one_step_and_one_joint_rank_loss(self):
        from dta.hybrid_affinity_training import (
            pairwise_logistic_rank_loss,
            train_hybrid_group,
        )

        model = TinyGroupedModel(weight=0.5)
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        group = [
            {"key": f"k{index}", "x": torch.tensor([value]), "label": label}
            for index, (value, label) in enumerate(
                [(1.0, 1.0), (2.0, 3.0), (3.0, 2.0), (4.0, 4.0)]
            )
        ]
        result = train_hybrid_group(
            model,
            optimizer,
            group,
            torch.device("cpu"),
            rank_weight=0.2,
            prepare_fn=prepare_tiny_sample,
            max_grad_norm=None,
        )

        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(result["keys"], ["k0", "k1", "k2", "k3"])
        self.assertEqual(result["predictions"].shape, (4,))
        expected_rank = pairwise_logistic_rank_loss(
            torch.tensor([0.5, 1.0, 1.5, 2.0]),
            torch.tensor([1.0, 3.0, 2.0, 4.0]),
        )
        self.assertAlmostEqual(result["rank"], float(expected_rank), places=7)

    def test_amp_gradient_overflow_skips_step_and_reduces_scale(self):
        from dta.hybrid_affinity_training import train_hybrid_group

        model = TinyGroupedModel(weight=0.5)
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        scaler = OverflowOnceScaler(scale=8.0)
        ema = CountingEMA()
        group = [
            {"key": "overflow", "x": torch.tensor([1.0]), "label": 2.0}
        ]

        result = train_hybrid_group(
            model,
            optimizer,
            group,
            torch.device("cpu"),
            rank_weight=0.2,
            prepare_fn=prepare_tiny_sample,
            scaler=scaler,
            max_grad_norm=None,
            ema=ema,
        )

        self.assertFalse(result["optimizer_step"])
        self.assertEqual(result["loss_scale_before"], 8.0)
        self.assertEqual(result["loss_scale_after"], 4.0)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(ema.update_calls, 0)
        self.assertIsNone(model.weight.grad)

    def test_amp_overflow_retries_same_group_until_optimizer_step(self):
        from dta.hybrid_affinity_training import (
            train_hybrid_group_with_amp_retries,
        )

        model = TinyGroupedModel(weight=0.5)
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        scaler = OverflowOnceScaler(scale=8.0)
        ema = CountingEMA()
        group = [
            {"key": "retry-key", "x": torch.tensor([1.0]), "label": 2.0}
        ]
        overflow_events = []

        result = train_hybrid_group_with_amp_retries(
            model,
            optimizer,
            group,
            torch.device("cpu"),
            rank_weight=0.2,
            prepare_fn=prepare_tiny_sample,
            scaler=scaler,
            max_grad_norm=None,
            ema=ema,
            overflow_callback=overflow_events.append,
        )

        self.assertTrue(result["optimizer_step"])
        self.assertEqual(result["amp_overflow_retries"], 1)
        self.assertEqual(scaler.unscale_calls, 2)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(ema.update_calls, 1)
        self.assertEqual(len(overflow_events), 1)
        self.assertEqual(overflow_events[0]["sample_keys"], ["retry-key"])
        self.assertEqual(overflow_events[0]["loss_scale_before"], 8.0)
        self.assertEqual(overflow_events[0]["loss_scale_after"], 4.0)

    def test_amp_overflow_retry_limit_raises_diagnostic_error(self):
        from dta.hybrid_affinity_training import (
            train_hybrid_group_with_amp_retries,
        )

        model = TinyGroupedModel(weight=0.5)
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        scaler = OverflowOnceScaler(scale=8.0, overflow_attempts=3)
        ema = CountingEMA()
        group = [
            {"key": "persistent", "x": torch.tensor([1.0]), "label": 2.0}
        ]

        with self.assertRaisesRegex(
            FloatingPointError, "after 1 retries"
        ) as raised:
            train_hybrid_group_with_amp_retries(
                model,
                optimizer,
                group,
                torch.device("cpu"),
                rank_weight=0.2,
                prepare_fn=prepare_tiny_sample,
                scaler=scaler,
                max_grad_norm=None,
                ema=ema,
                max_amp_overflow_retries=1,
            )

        self.assertEqual(scaler.unscale_calls, 2)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(ema.update_calls, 0)
        self.assertEqual(raised.exception.diagnostics["phase"], "gradient")
        self.assertEqual(raised.exception.diagnostics["keys"], ["persistent"])

    def test_evaluation_preserves_order_and_returns_regression_metrics(self):
        from dta.hybrid_affinity_training import evaluate_hybrid_affinity

        model = TinyGroupedModel(weight=2.0)
        groups = [
            [
                {"key": "a", "x": torch.tensor([1.0]), "label": 1.5},
                {"key": "b", "x": torch.tensor([2.0]), "label": 3.5},
            ],
            [{"key": "c", "x": torch.tensor([3.0]), "label": 5.5}],
        ]
        result = evaluate_hybrid_affinity(
            model,
            groups,
            torch.device("cpu"),
            prepare_fn=prepare_tiny_sample,
        )

        self.assertEqual(result["keys"], ["a", "b", "c"])
        np.testing.assert_array_equal(result["labels"], [1.5, 3.5, 5.5])
        np.testing.assert_array_equal(result["predictions"], [2.0, 4.0, 6.0])
        self.assertEqual(
            set(result["metrics"]),
            {"RMSE", "MAE", "Pearson", "Spearman", "CI"},
        )
        self.assertAlmostEqual(result["metrics"]["RMSE"], 0.5)
        self.assertAlmostEqual(result["metrics"]["MAE"], 0.5)
        self.assertAlmostEqual(result["metrics"]["Pearson"], 1.0)
        self.assertAlmostEqual(result["metrics"]["Spearman"], 1.0)
        self.assertAlmostEqual(result["metrics"]["CI"], 1.0)

        with self.assertRaisesRegex(ValueError, "AMP"):
            evaluate_hybrid_affinity(
                model,
                groups,
                torch.device("cpu"),
                prepare_fn=prepare_tiny_sample,
                amp_enabled=True,
            )


        from unittest.mock import patch

        with patch(
            "utils.metrics.get_score_metrics",
            return_value=(0.5, 0.5, float("nan"), 1.0, 1.0),
        ):
            with self.assertRaisesRegex(
                FloatingPointError, "metrics"
            ) as metric_error:
                evaluate_hybrid_affinity(
                    model,
                    groups,
                    torch.device("cpu"),
                    prepare_fn=prepare_tiny_sample,
                )
        self.assertEqual(
            metric_error.exception.diagnostics["phase"], "metrics"
        )

    def test_nonfinite_prediction_reports_sample_keys(self):
        from dta.hybrid_affinity_training import train_hybrid_group

        model = TinyGroupedModel(nonfinite=True)
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        group = [{"key": "bad-key", "x": torch.tensor([1.0]), "label": 1.0}]

        with self.assertRaisesRegex(
            FloatingPointError, "bad-key"
        ) as raised:
            train_hybrid_group(
                model,
                optimizer,
                group,
                torch.device("cpu"),
                rank_weight=0.2,
                prepare_fn=prepare_tiny_sample,
            )
        self.assertEqual(
            raised.exception.diagnostics["phase"], "prediction"
        )
        self.assertEqual(
            raised.exception.diagnostics["keys"], ["bad-key"]
        )
        self.assertEqual(
            raised.exception.diagnostics["tensors"]["prediction"][
                "finite_count"
            ],
            0,
        )

    def test_nonfinite_loss_and_gradient_report_sample_keys(self):
        from dta.hybrid_affinity_training import train_hybrid_group

        overflow_model = TinyGroupedModel(weight=1e20)
        overflow_optimizer = CountingSGD(
            overflow_model.parameters(), lr=0.01
        )
        overflow_group = [
            {
                "key": "overflow-key",
                "x": torch.tensor([1.0]),
                "label": -1e20,
            }
        ]
        with self.assertRaisesRegex(FloatingPointError, "overflow-key"):
            train_hybrid_group(
                overflow_model,
                overflow_optimizer,
                overflow_group,
                torch.device("cpu"),
                rank_weight=0.2,
                prepare_fn=prepare_tiny_sample,
            )

        class NonfiniteGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * torch.tensor(float("nan"))

        class NonfiniteGradientModel(TinyGroupedModel):
            def forward(self, gerna_inputs, interface_item):
                value = self.weight * gerna_inputs["x"].reshape(1, 1)
                return {"prediction": NonfiniteGradient.apply(value)}

        gradient_model = NonfiniteGradientModel()
        gradient_optimizer = CountingSGD(
            gradient_model.parameters(), lr=0.01
        )
        gradient_group = [
            {
                "key": "gradient-key",
                "x": torch.tensor([1.0]),
                "label": 1.0,
            }
        ]
        with self.assertRaisesRegex(FloatingPointError, "gradient-key"):
            train_hybrid_group(
                gradient_model,
                gradient_optimizer,
                gradient_group,
                torch.device("cpu"),
                rank_weight=0.2,
                prepare_fn=prepare_tiny_sample,
            )


class StochasticGroupedModel(TinyGroupedModel):
    def forward(self, gerna_inputs, interface_item):
        python_noise = random.random()
        numpy_noise = float(np.random.random())
        torch_noise = torch.rand_like(gerna_inputs["x"])
        value = gerna_inputs["x"] + python_noise + numpy_noise + torch_noise
        return {"prediction": self.weight * value.reshape(1, 1)}


class TestHybridCheckpointResume(unittest.TestCase):

    def test_stage_extension_only_allows_unstarted_suffix(self):
        from dta.hybrid_affinity_training import (
            _training_configs_match_for_resume,
        )

        saved = {
            "stage_epochs": {"head": 1, "high": 0, "full": 1},
            "rank_weight": 0.2,
        }
        expected = {
            "stage_epochs": {"head": 1, "high": 1, "full": 1},
            "rank_weight": 0.2,
        }
        self.assertFalse(
            _training_configs_match_for_resume(
                saved,
                expected,
                allow_stage_extension=True,
            )
        )
        trailing = {
            "stage_epochs": {"head": 1, "high": 0, "full": 0},
            "rank_weight": 0.2,
        }
        self.assertTrue(
            _training_configs_match_for_resume(
                trailing,
                expected,
                allow_stage_extension=True,
            )
        )


    def test_checkpoint_loading_defaults_to_cpu(self):
        import inspect

        from dta.hybrid_affinity_training import load_training_checkpoint

        default = inspect.signature(load_training_checkpoint).parameters[
            "map_location"
        ].default
        self.assertEqual(default, "cpu")

    def test_resume_restores_exact_next_update_and_all_rng_states(self):
        from dta.hybrid_affinity_training import (
            ModelEMA,
            load_training_checkpoint,
            save_training_checkpoint,
            train_hybrid_group,
        )

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        generator = torch.Generator().manual_seed(987)
        model = StochasticGroupedModel(weight=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.8
        )
        ema = ModelEMA(model, decay=0.9)
        scaler = TinyScaler(8.0)
        first_group = [
            {"key": "first", "x": torch.tensor([1.0]), "label": 2.0}
        ]
        second_group = [
            {"key": "second", "x": torch.tensor([2.0]), "label": 3.0}
        ]
        train_hybrid_group(
            model,
            optimizer,
            first_group,
            torch.device("cpu"),
            rank_weight=0.2,
            prepare_fn=prepare_tiny_sample,
            ema=ema,
        )
        scheduler.step()
        torch.rand((), generator=generator)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "training_checkpoint.pth"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                schedulers={"head": scheduler},
                scaler=scaler,
                ema=ema,
                training_state={
                    "stage": "head",
                    "next_epoch": 1,
                    "global_step": 1,
                },
                top_three=[{"epoch": 0, "metrics": {"CI": 0.5}}],
                history=[{"epoch": 0, "train_loss": 1.0}],
                data_loader_generator=generator,
                provenance_hashes={"manifest": "abc123"},
                training_config={"rank_weight": 0.2},
            )
            self.assertTrue(checkpoint_path.is_file())
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

            expected_rng = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
                torch.rand(3, generator=generator),
            )
            train_hybrid_group(
                model,
                optimizer,
                second_group,
                torch.device("cpu"),
                rank_weight=0.2,
                prepare_fn=prepare_tiny_sample,
                ema=ema,
            )
            scheduler.step()
            expected_model = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            expected_ema = ema.state_dict()

            resumed_model = StochasticGroupedModel(weight=-9.0)
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.parameters(), lr=0.5
            )
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(
                resumed_optimizer, step_size=1, gamma=0.8
            )
            resumed_ema = ModelEMA(resumed_model, decay=0.5)
            resumed_scaler = TinyScaler(1.0)
            resumed_generator = torch.Generator().manual_seed(1)
            loaded = load_training_checkpoint(
                checkpoint_path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                schedulers={"head": resumed_scheduler},
                scaler=resumed_scaler,
                ema=resumed_ema,
                data_loader_generator=resumed_generator,
                expected_provenance_hashes={"manifest": "abc123"},
                expected_training_config={"rank_weight": 0.2},
                map_location="cpu",
            )
            self.assertEqual(loaded["training_state"]["next_epoch"], 1)
            self.assertEqual(len(loaded["top_three"]), 1)
            self.assertEqual(len(loaded["history"]), 1)
            self.assertEqual(resumed_scaler.scale, 8.0)

            actual_rng = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
                torch.rand(3, generator=resumed_generator),
            )
            self.assertEqual(actual_rng[0], expected_rng[0])
            self.assertEqual(actual_rng[1], expected_rng[1])
            torch.testing.assert_close(
                actual_rng[2], expected_rng[2], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_rng[3], expected_rng[3], rtol=0, atol=0
            )

            train_hybrid_group(
                resumed_model,
                resumed_optimizer,
                second_group,
                torch.device("cpu"),
                rank_weight=0.2,
                prepare_fn=prepare_tiny_sample,
                ema=resumed_ema,
            )
            resumed_scheduler.step()
            for name, expected in expected_model.items():
                torch.testing.assert_close(
                    resumed_model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )
            self.assertEqual(
                resumed_scheduler.state_dict(), scheduler.state_dict()
            )
            for name, expected in expected_ema["shadow"].items():
                torch.testing.assert_close(
                    resumed_ema.state_dict()["shadow"][name],
                    expected,
                    rtol=0,
                    atol=0,
                )

            rejected_model = StochasticGroupedModel(weight=-7.0)
            rejected_optimizer = torch.optim.AdamW(
                rejected_model.parameters(), lr=0.5
            )
            rejected_scheduler = torch.optim.lr_scheduler.StepLR(
                rejected_optimizer, step_size=1, gamma=0.8
            )
            rejected_ema = ModelEMA(rejected_model, decay=0.5)
            rejected_scaler = TinyScaler(1.0)
            with self.assertRaisesRegex(RuntimeError, "training config"):
                load_training_checkpoint(
                    checkpoint_path,
                    model=rejected_model,
                    optimizer=rejected_optimizer,
                    schedulers={"head": rejected_scheduler},
                    scaler=rejected_scaler,
                    ema=rejected_ema,
                    data_loader_generator=torch.Generator(),
                    expected_provenance_hashes={"manifest": "abc123"},
                    expected_training_config={"rank_weight": 0.4},
                    map_location="cpu",
                )
            torch.testing.assert_close(
                rejected_model.weight,
                torch.tensor(-7.0),
                rtol=0,
                atol=0,
            )


class TinyFitModel(TinyHybrid):
    def forward(self, gerna_inputs, interface_item):
        value = gerna_inputs["x"].reshape(1, 1)
        pair = torch.cat([value, value], dim=-1)
        prediction = self.fusion(pair)
        prediction = prediction + 0.01 * self.gerna.high(pair).mean(
            dim=-1, keepdim=True
        )
        prediction = prediction + 0.01 * self.gerna.low(pair).mean(
            dim=-1, keepdim=True
        )
        prediction = prediction + 0.01 * self.interface.affinity_head(pair)
        return {"prediction": prediction}


class TestHybridFit(unittest.TestCase):

    def test_fit_uses_amp_retry_wrapper_and_logs_overflow(self):
        from unittest import mock

        import dta.hybrid_affinity_training as training

        model = TinyFitModel()
        train_loader = [[
            {"key": "t0", "x": torch.tensor([1.0]), "label": 1.5},
            {"key": "t1", "x": torch.tensor([2.0]), "label": 2.5},
        ]]
        valid_loader = [[
            {"key": "v0", "x": torch.tensor([1.5]), "label": 2.0},
            {"key": "v1", "x": torch.tensor([2.5]), "label": 3.0},
        ]]

        def fake_retry(*args, overflow_callback=None, **kwargs):
            overflow_callback({
                "event": "amp_overflow",
                "attempt": 1,
                "sample_keys": ["t0", "t1"],
                "loss_scale_before": 65536.0,
                "loss_scale_after": 32768.0,
            })
            kwargs.pop("max_amp_overflow_retries", None)
            result = training.train_hybrid_group(*args, **kwargs)
            result["amp_overflow_retries"] = 1
            return result

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                training,
                "train_hybrid_group_with_amp_retries",
                side_effect=fake_retry,
            ) as retry:
                training.fit_hybrid_affinity(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    output_dir=temporary,
                    device=torch.device("cpu"),
                    rank_weight=0.2,
                    run_config={"seed": 42, "fusion_type": "mlp"},
                    provenance={"train_manifest": "train.json"},
                    provenance_hashes={"manifest": "abc123"},
                    data_loader_generator=torch.Generator().manual_seed(42),
                    prepare_fn=prepare_tiny_sample,
                    stage_epochs={"head": 1, "high": 0, "full": 0},
                    amp_enabled=False,
                )

            self.assertEqual(retry.call_count, 1)
            lines = [
                json.loads(line)
                for line in (Path(temporary) / "train.log")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            overflow = [line for line in lines if line["event"] == "amp_overflow"]
            self.assertEqual(len(overflow), 1)
            self.assertEqual(overflow[0]["global_step"], 0)
            self.assertEqual(overflow[0]["stage"], "head")
            checkpoint = torch.load(
                Path(temporary) / "training_checkpoint.pth",
                map_location="cpu",
            )
            self.assertEqual(checkpoint["training_state"]["global_step"], 1)

    def test_exact_resume_rejects_worker_and_independent_loader_rng(self):
        from torch.utils.data import DataLoader

        from dta.hybrid_affinity_training import (
            _validate_exact_resume_loader,
        )

        shared = torch.Generator().manual_seed(42)
        worker_loader = DataLoader([0, 1], num_workers=1)
        with self.assertRaisesRegex(ValueError, "num_workers=0"):
            _validate_exact_resume_loader(
                worker_loader,
                shared,
                loader_name="train_loader",
            )

        independent = torch.Generator().manual_seed(7)
        independent_loader = DataLoader(
            [0, 1],
            num_workers=0,
            generator=independent,
        )
        with self.assertRaisesRegex(ValueError, "generator"):
            _validate_exact_resume_loader(
                independent_loader,
                shared,
                loader_name="valid_loader",
            )

    def test_fit_writes_nonfinite_diagnostic_before_stopping(self):
        from dta.hybrid_affinity_training import fit_hybrid_affinity

        model = TinyFitModel()
        with torch.no_grad():
            model.fusion[0].weight.fill_(float("nan"))
        train_loader = [[{
            "key": "nonfinite-fit-key",
            "x": torch.tensor([1.0]),
            "label": 1.0,
        }]]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                FloatingPointError, "nonfinite-fit-key"
            ):
                fit_hybrid_affinity(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=train_loader,
                    output_dir=temporary,
                    device=torch.device("cpu"),
                    rank_weight=0.2,
                    run_config={"seed": 42, "fusion_type": "mlp"},
                    provenance={"train_manifest": "train.json"},
                    provenance_hashes={"manifest": "abc123"},
                    data_loader_generator=torch.Generator().manual_seed(42),
                    prepare_fn=prepare_tiny_sample,
                    stage_epochs={"head": 1, "high": 0, "full": 0},
                    amp_enabled=False,
                )
            output = Path(temporary)
            with (output / "training_failure.json").open(
                encoding="utf-8"
            ) as handle:
                failure = json.load(handle)
            self.assertEqual(
                failure["sample_keys"], ["nonfinite-fit-key"]
            )
            self.assertEqual(failure["stage"], "head")
            self.assertEqual(failure["epoch"], 0)
            self.assertEqual(
                failure["diagnostics"]["phase"], "prediction"
            )
            self.assertFalse(
                (output / "Model_hybrid_best.pth").exists()
            )

        class EvalNonfiniteModel(TinyFitModel):
            def forward(self, gerna_inputs, interface_item):
                output = super().forward(
                    gerna_inputs, interface_item
                )
                if not self.training:
                    output["prediction"] = (
                        output["prediction"]
                        * torch.tensor(float("nan"))
                    )
                return output

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                FloatingPointError, "nonfinite-fit-key"
            ):
                fit_hybrid_affinity(
                    model=EvalNonfiniteModel(),
                    train_loader=train_loader,
                    valid_loader=train_loader,
                    output_dir=temporary,
                    device=torch.device("cpu"),
                    rank_weight=0.2,
                    run_config={"seed": 42, "fusion_type": "mlp"},
                    provenance={"train_manifest": "train.json"},
                    provenance_hashes={"manifest": "abc123"},
                    data_loader_generator=torch.Generator().manual_seed(42),
                    prepare_fn=prepare_tiny_sample,
                    stage_epochs={"head": 1, "high": 0, "full": 0},
                    amp_enabled=False,
                )
            with (
                Path(temporary) / "training_failure.json"
            ).open(encoding="utf-8") as handle:
                validation_failure = json.load(handle)
            self.assertEqual(
                validation_failure["validation_variant"], "raw"
            )
            self.assertEqual(
                validation_failure["diagnostics"]["phase"],
                "prediction",
            )


    def test_resume_log_preserves_existing_lines(self):
        from dta.hybrid_affinity_training import _initial_log_lines

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            original = ['{"event": "start"}', '{"event": "epoch"}']
            (output / "train.log").write_text(
                "\n".join(original) + "\n",
                encoding="utf-8",
            )
            resumed = _initial_log_lines(output, resume=True)
            self.assertEqual(resumed[:2], original)
            self.assertEqual(json.loads(resumed[-1])["event"], "resume")

    def test_validation_order_uses_ci_spearman_rmse_and_epoch(self):
        from dta.hybrid_affinity_training import (
            validation_candidate_is_better,
        )

        baseline = {
            "epoch": 5,
            "metrics": {
                "CI": 0.70,
                "Spearman": 0.60,
                "RMSE": 1.5,
            },
        }
        self.assertTrue(
            validation_candidate_is_better(
                {
                    "epoch": 9,
                    "metrics": {
                        "CI": 0.71,
                        "Spearman": 0.1,
                        "RMSE": 9.0,
                    },
                },
                baseline,
            )
        )
        self.assertTrue(
            validation_candidate_is_better(
                {
                    "epoch": 9,
                    "metrics": {
                        "CI": 0.7000005,
                        "Spearman": 0.61,
                        "RMSE": 9.0,
                    },
                },
                baseline,
            )
        )
        self.assertTrue(
            validation_candidate_is_better(
                {
                    "epoch": 9,
                    "metrics": {
                        "CI": 0.70,
                        "Spearman": 0.60,
                        "RMSE": 1.4,
                    },
                },
                baseline,
            )
        )
        self.assertTrue(
            validation_candidate_is_better(
                {
                    "epoch": 4,
                    "metrics": dict(baseline["metrics"]),
                },
                baseline,
            )
        )


    def test_fit_stage_extension_matches_continuous_training(self):
        from dta.hybrid_affinity_training import fit_hybrid_affinity

        torch.manual_seed(202)
        base = TinyFitModel()
        base_state = {
            name: value.detach().clone()
            for name, value in base.state_dict().items()
        }
        train_loader = [[
            {"key": "t0", "x": torch.tensor([1.0]), "label": 1.5},
            {"key": "t1", "x": torch.tensor([2.0]), "label": 2.5},
        ]]
        valid_loader = [[
            {"key": "v0", "x": torch.tensor([1.5]), "label": 2.0},
            {"key": "v1", "x": torch.tensor([2.5]), "label": 3.0},
        ]]
        common = {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "device": torch.device("cpu"),
            "rank_weight": 0.2,
            "run_config": {"seed": 42, "fusion_type": "mlp"},
            "provenance": {"train_manifest": "train.json"},
            "provenance_hashes": {"manifest": "abc123"},
            "prepare_fn": prepare_tiny_sample,
            "amp_enabled": False,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous_model = TinyFitModel()
            continuous_model.load_state_dict(base_state)
            random.seed(91)
            np.random.seed(91)
            torch.manual_seed(91)
            fit_hybrid_affinity(
                model=continuous_model,
                output_dir=root / "continuous",
                data_loader_generator=torch.Generator().manual_seed(91),
                stage_epochs={"head": 1, "high": 0, "full": 1},
                **common,
            )

            staged_model = TinyFitModel()
            staged_model.load_state_dict(base_state)
            random.seed(91)
            np.random.seed(91)
            torch.manual_seed(91)
            fit_hybrid_affinity(
                model=staged_model,
                output_dir=root / "staged",
                data_loader_generator=torch.Generator().manual_seed(91),
                stage_epochs={"head": 1, "high": 0, "full": 0},
                **common,
            )
            resumed_model = TinyFitModel()
            resumed = fit_hybrid_affinity(
                model=resumed_model,
                output_dir=root / "staged",
                data_loader_generator=torch.Generator().manual_seed(1),
                stage_epochs={"head": 1, "high": 0, "full": 1},
                resume=True,
                **common,
            )
            self.assertEqual(len(resumed["history"]), 2)

            continuous = torch.load(
                root / "continuous" / "Model_hybrid_last.pth",
                map_location="cpu",
            )
            staged = torch.load(
                root / "staged" / "Model_hybrid_last.pth",
                map_location="cpu",
            )
            for name, expected in continuous[
                "model_state_dict"
            ].items():
                torch.testing.assert_close(
                    staged["model_state_dict"][name],
                    expected,
                    rtol=0,
                    atol=0,
                )
            for name, expected in continuous[
                "ema_state_dict"
            ].items():
                torch.testing.assert_close(
                    staged["ema_state_dict"][name],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_fit_writes_atomic_validation_only_artifacts(self):
        from dta.hybrid_affinity_training import fit_hybrid_affinity

        torch.manual_seed(17)
        model = TinyFitModel()
        train_loader = [
            [
                {
                    "key": "t0",
                    "x": torch.tensor([1.0]),
                    "label": 1.5,
                },
                {
                    "key": "t1",
                    "x": torch.tensor([2.0]),
                    "label": 2.5,
                },
            ],
            [
                {
                    "key": "t2",
                    "x": torch.tensor([3.0]),
                    "label": 3.5,
                },
                {
                    "key": "t3",
                    "x": torch.tensor([4.0]),
                    "label": 4.5,
                },
            ],
        ]
        valid_loader = [
            [
                {
                    "key": "v0",
                    "x": torch.tensor([1.5]),
                    "label": 2.0,
                },
                {
                    "key": "v1",
                    "x": torch.tensor([2.5]),
                    "label": 3.0,
                },
            ]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = fit_hybrid_affinity(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                output_dir=temporary,
                device=torch.device("cpu"),
                rank_weight=0.2,
                run_config={"seed": 42, "fusion_type": "mlp"},
                provenance={"train_manifest": "train.json"},
                provenance_hashes={"manifest": "abc123"},
                data_loader_generator=torch.Generator().manual_seed(42),
                prepare_fn=prepare_tiny_sample,
                stage_epochs={"head": 1, "high": 1, "full": 1},
                amp_enabled=False,
            )

            expected = {
                "Model_hybrid_best.pth",
                "Model_hybrid_last.pth",
                "training_checkpoint.pth",
                "run_config.json",
                "provenance.json",
                "metrics_history.csv",
                "valid_predictions.npz",
                "best_metrics.json",
                "train.log",
                "validation_rank_1.pth",
                "validation_rank_2.pth",
                "validation_rank_3.pth",
            }
            output = Path(temporary)
            self.assertTrue(
                expected.issubset({path.name for path in output.iterdir()})
            )
            self.assertEqual(list(output.glob("*.tmp")), [])
            self.assertFalse((output / "test_predictions.npz").exists())
            self.assertEqual(len(result["history"]), 3)
            with (output / "run_config.json").open(
                encoding="utf-8"
            ) as handle:
                self.assertEqual(json.load(handle)["seed"], 42)
            with np.load(output / "valid_predictions.npz") as predictions:
                self.assertEqual(
                    predictions["keys"].tolist(), ["v0", "v1"]
                )
                self.assertEqual(
                    predictions["labels"].tolist(), [2.0, 3.0]
                )
                self.assertTrue(
                    np.isfinite(predictions["predictions"]).all()
                )
            checkpoint = torch.load(
                output / "training_checkpoint.pth",
                map_location="cpu",
            )
            self.assertTrue(
                checkpoint["training_state"]["complete"]
            )

            run_config_before = (output / "run_config.json").read_bytes()
            provenance_before = (output / "provenance.json").read_bytes()
            rejected_config_model = TinyFitModel()
            rejected_config_values = {
                name: value.detach().clone()
                for name, value in rejected_config_model.state_dict().items()
            }
            rejected_config_trainable = {
                name: parameter.requires_grad
                for name, parameter in rejected_config_model.named_parameters()
            }
            with self.assertRaisesRegex(RuntimeError, "training config"):
                fit_hybrid_affinity(
                    model=rejected_config_model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    output_dir=temporary,
                    device=torch.device("cpu"),
                    rank_weight=0.4,
                    run_config={"seed": 42, "fusion_type": "mlp"},
                    provenance={"train_manifest": "train.json"},
                    provenance_hashes={"manifest": "abc123"},
                    data_loader_generator=torch.Generator().manual_seed(42),
                    prepare_fn=prepare_tiny_sample,
                    stage_epochs={"head": 1, "high": 1, "full": 1},
                    amp_enabled=False,
                    resume=True,
                )
            self.assertEqual(
                (output / "run_config.json").read_bytes(),
                run_config_before,
            )
            for name, expected in rejected_config_values.items():
                torch.testing.assert_close(
                    rejected_config_model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )
            self.assertEqual(
                {
                    name: parameter.requires_grad
                    for name, parameter
                    in rejected_config_model.named_parameters()
                },
                rejected_config_trainable,
            )
            self.assertEqual(
                (output / "provenance.json").read_bytes(),
                provenance_before,
            )

            with self.assertRaises(FileExistsError):
                fit_hybrid_affinity(
                    model=TinyFitModel(),
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    output_dir=temporary,
                    device=torch.device("cpu"),
                    rank_weight=0.2,
                    run_config={
                        "seed": 99,
                        "fusion_type": "transformer",
                    },
                    provenance={"train_manifest": "wrong.json"},
                    provenance_hashes={"manifest": "abc123"},
                    data_loader_generator=torch.Generator().manual_seed(42),
                    prepare_fn=prepare_tiny_sample,
                    stage_epochs={"head": 1, "high": 1, "full": 1},
                    amp_enabled=False,
                    resume=False,
                )
            self.assertEqual(
                (output / "run_config.json").read_bytes(),
                run_config_before,
            )
            self.assertEqual(
                (output / "provenance.json").read_bytes(),
                provenance_before,
            )


            (output / "Model_hybrid_best.pth").unlink()
            (output / "valid_predictions.npz").unlink()
            (output / "train.log").write_text(
                '{"event": "start", "resume": false}\n',
                encoding="utf-8",
            )
            recovered = fit_hybrid_affinity(
                model=TinyFitModel(),
                train_loader=train_loader,
                valid_loader=valid_loader,
                output_dir=temporary,
                device=torch.device("cpu"),
                rank_weight=0.2,
                run_config={"seed": 42, "fusion_type": "mlp"},
                provenance={"train_manifest": "train.json"},
                provenance_hashes={"manifest": "abc123"},
                data_loader_generator=torch.Generator().manual_seed(42),
                prepare_fn=prepare_tiny_sample,
                stage_epochs={"head": 1, "high": 1, "full": 1},
                amp_enabled=False,
                resume=True,
            )
            self.assertEqual(len(recovered["history"]), 3)
            self.assertTrue(
                (output / "Model_hybrid_best.pth").is_file()
            )
            self.assertTrue(
                (output / "valid_predictions.npz").is_file()
            )
            recovered_log = (
                output / "train.log"
            ).read_text(encoding="utf-8")
            self.assertIn('"event": "epoch"', recovered_log)

            (output / "train.log").write_text(
                "not-json\n", encoding="utf-8"
            )
            rejected_log_model = TinyFitModel()
            rejected_log_before = {
                name: value.detach().clone()
                for name, value in rejected_log_model.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "train.log"):
                fit_hybrid_affinity(
                    model=rejected_log_model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    output_dir=temporary,
                    device=torch.device("cpu"),
                    rank_weight=0.2,
                    run_config={"seed": 42, "fusion_type": "mlp"},
                    provenance={"train_manifest": "train.json"},
                    provenance_hashes={"manifest": "abc123"},
                    data_loader_generator=torch.Generator().manual_seed(42),
                    prepare_fn=prepare_tiny_sample,
                    stage_epochs={"head": 1, "high": 1, "full": 1},
                    amp_enabled=False,
                    resume=True,
                )
            for name, expected in rejected_log_before.items():
                torch.testing.assert_close(
                    rejected_log_model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )



if __name__ == "__main__":
    unittest.main()
