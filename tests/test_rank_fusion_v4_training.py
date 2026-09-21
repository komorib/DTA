import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dta.rank_fusion_v4.cache import FusionCache
from dta.rank_fusion_v4.data import LabelSpreadBatchSampler
from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.rank_fusion_v4.model import RankFusionV4
from dta.rank_fusion_v4.train import build_arg_parser
from dta.rank_fusion_v4.training import (
    NonfiniteTrainingError,
    evaluate_rank_fusion,
    fit_one_seed,
    is_better_validation,
)


def synthetic_cache(split, count=24, normalization_hash="normalization-hash"):
    generator = np.random.RandomState(11 if split == "train" else 13)
    labels = np.linspace(3.0, 9.0, count, dtype=np.float32)
    predictions = np.stack(
        (
            labels + generator.normal(0, 0.8, count),
            labels + generator.normal(0, 0.5, count),
            labels + generator.normal(0, 0.6, count),
        ),
        axis=1,
    ).astype(np.float32)
    available = np.ones((count, 3), dtype=np.bool_)
    available[1::5, 1] = False
    available[2::7, 2] = False
    predictions[~available] = 0.0
    standardized = predictions.copy()
    for expert in range(3):
        mask = available[:, expert]
        values = standardized[mask, expert]
        standardized[mask, expert] = (values - values.mean()) / max(values.std(), 1e-6)
    standardized[~available] = 0.0

    def latent(width, expert):
        values = generator.normal(size=(count, width)).astype(np.float32)
        values[~available[:, expert]] = 0.0
        return values

    keys = tuple(f"{split}_{index:04d}_protein" for index in range(count))
    return FusionCache(
        keys=keys,
        labels=labels,
        predictions=predictions,
        standardized_predictions=standardized,
        available=available,
        upper_bound_latents=latent(7, 0),
        boltz_latents=latent(5, 1),
        three_dprot_latents=latent(6, 2),
        metadata={
            "status": "complete",
            "split": split,
            "count": count,
            "manifest_sha256": split + "-manifest",
            "ordered_key_sha256": split + "-keys",
            "normalization_sha256": normalization_hash,
            "latent_dimensions": {
                "upper_bound": 7,
                "boltz_pairwise": 5,
                "three_dprot": 6,
            },
        },
    )


def tiny_model():
    return RankFusionV4(
        (7, 5, 6),
        model_dim=16,
        layers=1,
        heads=4,
        dropout=0.0,
        drop_path_rate=0.0,
    )


class RankFusionTrainingTest(unittest.TestCase):
    def test_sampler_is_deterministic_covers_once_and_spreads_labels(self):
        labels = np.arange(257, dtype=np.float64)
        first = LabelSpreadBatchSampler(labels, group_size=64, seed=42)
        second = LabelSpreadBatchSampler(labels, group_size=64, seed=42)
        first.set_epoch(3)
        second.set_epoch(3)
        groups = list(first)
        self.assertEqual(groups, list(second))
        flattened = [index for group in groups for index in group]
        self.assertEqual(len(flattened), len(labels))
        self.assertEqual(sorted(flattened), list(range(len(labels))))
        self.assertEqual(len(groups[-1]), 1)
        quartile = np.digitize(labels, np.quantile(labels, [0.25, 0.5, 0.75]))
        for group in groups[:-1]:
            self.assertGreaterEqual(len(set(quartile[group].tolist())), 2)
        first.set_epoch(4)
        self.assertNotEqual(groups, list(first))

    def test_initial_evaluation_is_exact_upper_bound_and_preserves_order(self):
        cache = synthetic_cache("valid")
        result = evaluate_rank_fusion(tiny_model().eval(), cache, "cpu", batch_size=7)
        self.assertEqual(result["keys"], cache.keys)
        np.testing.assert_array_equal(result["predictions"], cache.predictions[:, 0])
        self.assertEqual(
            result["metrics"],
            affinity_metrics(cache.labels, cache.predictions[:, 0]),
        )

    def test_one_seed_training_updates_parameters_and_writes_complete_outputs(self):
        train = synthetic_cache("train")
        valid = synthetic_cache("valid")
        model = tiny_model()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            summary = fit_one_seed(
                model,
                train,
                valid,
                output,
                seed=7,
                device="cpu",
                epochs=2,
                patience=2,
                batch_size=8,
                learning_rate=1e-3,
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(summary["validation_count"], len(valid.keys))
            self.assertEqual(len(summary["history"]), 2)
            self.assertEqual(
                set(summary["history"][0]["losses"]),
                {
                    "loss",
                    "hard_pair_logistic",
                    "listmle",
                    "soft_spearman",
                    "smooth_l1",
                    "mixture_regularization",
                },
            )
            for name in (
                "checkpoint_last.pt",
                "checkpoint_best.pt",
                "valid_predictions_best.npz",
                "best_metrics.json",
                "summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            with np.load(output / "valid_predictions_best.npz") as archive:
                self.assertEqual(tuple(archive["keys"].tolist()), valid.keys)
                self.assertEqual(archive["mixture_weights"].shape, (len(valid.keys), 3))
            self.assertTrue(
                any(
                    not torch.equal(before[name], value.detach())
                    for name, value in model.named_parameters()
                )
            )

    def test_resume_and_completed_reuse_require_matching_provenance(self):
        train = synthetic_cache("train", count=16)
        valid = synthetic_cache("valid", count=16)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            first = fit_one_seed(
                tiny_model(), train, valid, output,
                seed=42, device="cpu", epochs=1, patience=1, batch_size=8,
            )
            reused = fit_one_seed(
                tiny_model(), train, valid, output,
                seed=42, device="cpu", epochs=1, patience=1, batch_size=8,
                resume=True,
            )
            self.assertEqual(reused["best_checkpoint_sha256"], first["best_checkpoint_sha256"])
            mismatched = synthetic_cache("valid", count=16, normalization_hash="other")
            with self.assertRaisesRegex(ValueError, "normalization"):
                fit_one_seed(
                    tiny_model(), train, mismatched, output,
                    seed=42, device="cpu", epochs=1, patience=1, batch_size=8,
                    resume=True,
                )

    def test_nonfinite_diagnostics_include_epoch_keys_and_tensor_statistics(self):
        train = synthetic_cache("train", count=8)
        train.upper_bound_latents[0, 0] = np.nan
        valid = synthetic_cache("valid", count=8)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(NonfiniteTrainingError) as caught:
                fit_one_seed(
                    tiny_model(), train, valid, Path(temporary) / "run",
                    seed=1, device="cpu", epochs=1, patience=1, batch_size=8,
                )
        message = str(caught.exception)
        self.assertIn("epoch=0", message)
        self.assertIn("train_0000_protein", message)
        self.assertIn("nonfinite", message)
        self.assertIn("shape", message)

    def test_validation_selection_order_is_ci_spearman_then_rmse(self):
        current = {"CI": 0.7, "Spearman": 0.5, "RMSE": 1.2}
        self.assertTrue(is_better_validation({"CI": 0.71, "Spearman": 0.0, "RMSE": 9.0}, current))
        self.assertTrue(is_better_validation({"CI": 0.7, "Spearman": 0.51, "RMSE": 9.0}, current))
        self.assertTrue(is_better_validation({"CI": 0.7, "Spearman": 0.5, "RMSE": 1.1}, current))
        self.assertFalse(is_better_validation({"CI": 0.69, "Spearman": 1.0, "RMSE": 0.1}, current))

    def test_train_cli_has_no_test_arguments(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--train-cache", "train",
                "--valid-cache", "valid",
                "--output", "run",
                "--seed", "42",
                "--device", "cpu",
            ]
        )
        self.assertEqual(args.seed, 42)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--train-cache", "train",
                    "--valid-cache", "valid",
                    "--output", "run",
                    "--seed", "42",
                    "--device", "cpu",
                    "--test-cache", "test",
                ]
            )


if __name__ == "__main__":
    unittest.main()
