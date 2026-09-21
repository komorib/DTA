import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dta.rank_locked_calibration_v6.training import (
    ConditionalTrainingConfig,
    FinalFitResult,
    FoldFitResult,
    NonfiniteTrainingError,
    build_run_lineage,
    cross_fit_analytic,
    cross_fit_conditional,
    ensemble_then_project,
    final_epoch_from_best_epochs,
    fit_conditional_fold,
    fit_conditional_inventory,
    fit_conditional_seed,
    fit_final_conditional,
    load_completed_run,
    predict_conditional_checkpoint,
    save_completed_run,
    validate_resume_lineage,
)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, count=25):
        self.keys = tuple("sample_%02d" % index for index in range(count))
        self.base = np.linspace(2.0, 10.0, count, dtype=np.float64)
        self.labels = 0.65 * self.base + 2.25
        generator = np.random.RandomState(42)
        self.raw = generator.normal(size=(count, 3)).astype(np.float32)
        self.standardized = generator.normal(size=(count, 3)).astype(np.float32)
        self.available = np.ones((count, 3), dtype=np.bool_)
        self.latents = (
            generator.normal(size=(count, 4)).astype(np.float32),
            generator.normal(size=(count, 3)).astype(np.float32),
            generator.normal(size=(count, 5)).astype(np.float32),
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return {
            "key": self.keys[index],
            "label": self.labels[index],
            "base_prediction": self.base[index],
            "raw_predictions": self.raw[index],
            "standardized_predictions": self.standardized[index],
            "available": self.available[index],
            "latents": tuple(value[index] for value in self.latents),
        }


def _lineage(config=None):
    if config is None:
        config = ConditionalTrainingConfig(maximum_epochs=7, patience=2)
    return build_run_lineage(
        candidate="conditional_seed_1",
        seed=1,
        config=config,
        source_revision="a" * 40,
        train_identity={"split": "train", "count": 25},
        valid_identity={"split": "valid", "count": 10},
        v5_package_sha256="b" * 64,
        v5_anchor_sha256="c" * 64,
        fold_map_sha256="d" * 64,
        normalization_sha256="e" * 64,
    )


class RankLockedV6CrossFitTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _TinyDataset()
        self.fold_ids = np.arange(len(self.dataset), dtype=np.int64) % 5
        self.config = ConditionalTrainingConfig(
            maximum_epochs=7,
            patience=2,
            batch_size=8,
            device="cpu",
        )

    def test_formal_training_defaults_are_frozen(self):
        config = ConditionalTrainingConfig()
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 1e-3)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.maximum_epochs, 200)
        self.assertEqual(config.patience, 20)
        self.assertEqual(config.gradient_clip, 1.0)

    def test_analytic_oof_fits_only_nonheld_rows(self):
        result = cross_fit_analytic(
            self.dataset, self.fold_ids, candidate="affine_mse"
        )
        np.testing.assert_array_equal(
            result.held_out_count, np.ones(len(self.dataset), dtype=np.int64)
        )
        np.testing.assert_allclose(
            result.predictions, self.dataset.labels, rtol=0.0, atol=1e-12
        )
        self.assertEqual(len(result.fold_results), 5)
        for fold_result in result.fold_results:
            held = set(np.flatnonzero(self.fold_ids == fold_result.fold))
            self.assertTrue(held.isdisjoint(set(fold_result.train_indices)))
            self.assertEqual(held, set(fold_result.holdout_indices))

    def test_conditional_oof_covers_each_row_once_without_holdout_leakage(self):
        calls = []

        def fit_spy(**kwargs):
            calls.append(kwargs)
            held = np.asarray(kwargs["holdout_indices"], dtype=np.int64)
            return FoldFitResult(
                fold=kwargs["fold"],
                predictions=self.dataset.labels[held],
                best_epoch=kwargs["fold"] + 1,
                best_loss=float(kwargs["fold"] + 1),
                model_state={"weight": torch.tensor([kwargs["fold"]])},
                history=({"epoch": kwargs["fold"] + 1},),
            )

        result = cross_fit_conditional(
            self.dataset,
            self.fold_ids,
            seed=1,
            config=self.config,
            fit_fold_fn=fit_spy,
        )
        np.testing.assert_array_equal(
            result.held_out_count, np.ones(len(self.dataset), dtype=np.int64)
        )
        np.testing.assert_array_equal(result.predictions, self.dataset.labels)
        self.assertEqual(result.best_epochs, (1, 2, 3, 4, 5))
        self.assertEqual(len(calls), 5)
        for fold, call in enumerate(calls):
            held = set(np.flatnonzero(self.fold_ids == fold))
            self.assertTrue(held.isdisjoint(set(call["train_indices"])))
            self.assertEqual(held, set(call["holdout_indices"]))
            self.assertEqual(call["seed"], 1)

    def test_conditional_oof_is_deterministic_for_the_same_seed(self):
        def seeded_fit(**kwargs):
            held = np.asarray(kwargs["holdout_indices"], dtype=np.int64)
            generator = np.random.RandomState(
                kwargs["seed"] * 1009 + kwargs["fold"]
            )
            return FoldFitResult(
                fold=kwargs["fold"],
                predictions=generator.normal(size=held.shape[0]),
                best_epoch=2,
                best_loss=0.5,
                model_state={},
                history=(),
            )

        first = cross_fit_conditional(
            self.dataset,
            self.fold_ids,
            seed=7,
            config=self.config,
            fit_fold_fn=seeded_fit,
        )
        second = cross_fit_conditional(
            self.dataset,
            self.fold_ids,
            seed=7,
            config=self.config,
            fit_fold_fn=seeded_fit,
        )
        np.testing.assert_array_equal(first.predictions, second.predictions)

    def test_final_epoch_is_ceiling_median_clamped_to_formal_range(self):
        self.assertEqual(final_epoch_from_best_epochs((1, 2, 4, 7, 9)), 4)
        self.assertEqual(final_epoch_from_best_epochs((0, 0, 0, 0, 0)), 1)
        self.assertEqual(
            final_epoch_from_best_epochs((201, 250, 300, 350, 400)), 200
        )

    def test_real_fold_fit_selects_lowest_heldout_loss(self):
        train_indices = np.arange(5, len(self.dataset), dtype=np.int64)
        holdout_indices = np.arange(5, dtype=np.int64)
        result = fit_conditional_fold(
            dataset=self.dataset,
            train_indices=train_indices,
            holdout_indices=holdout_indices,
            seed=1,
            fold=0,
            config=ConditionalTrainingConfig(
                maximum_epochs=2,
                patience=2,
                batch_size=8,
                device="cpu",
            ),
        )
        self.assertEqual(result.predictions.shape, (5,))
        self.assertTrue(np.isfinite(result.predictions).all())
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, 2)
        self.assertAlmostEqual(
            result.best_loss,
            min(row["holdout_loss"] for row in result.history),
        )

    def test_final_fit_uses_exact_fixed_epoch_count(self):
        result = fit_final_conditional(
            dataset=self.dataset,
            seed=1,
            epochs=2,
            config=ConditionalTrainingConfig(
                maximum_epochs=7,
                patience=2,
                batch_size=8,
                device="cpu",
            ),
        )
        self.assertIsInstance(result, FinalFitResult)
        self.assertEqual(result.epoch, 2)
        self.assertEqual(len(result.history), 2)
        self.assertTrue(np.isfinite(result.final_loss))
        self.assertIn("output_layer.weight", result.model_state)

    def test_checkpoint_prediction_exports_complete_float64_vector(self):
        from dta.rank_locked_calibration_v6.model import ConditionalCalibrator

        model = ConditionalCalibrator((4, 3, 5), 0.65, 2.25)
        checkpoint = {
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
        }
        predictions = predict_conditional_checkpoint(
            checkpoint,
            self.dataset,
            config=ConditionalTrainingConfig(
                maximum_epochs=2,
                patience=1,
                batch_size=8,
                device="cpu",
            ),
        )
        self.assertEqual(predictions.dtype, np.float64)
        self.assertEqual(predictions.shape, (len(self.dataset),))
        np.testing.assert_allclose(
            predictions,
            0.65 * self.dataset.base + 2.25,
            rtol=0.0,
            atol=2e-6,
        )


class RankLockedV6IsolationAndProjectionTest(unittest.TestCase):
    def test_ensemble_averages_unprojected_seeds_before_one_projection(self):
        base = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        members = {
            1: np.asarray([1.0, 5.0, 2.0]),
            7: np.asarray([3.0, 1.0, 4.0]),
            21: np.asarray([2.0, 3.0, 6.0]),
        }
        calls = []

        def project_spy(base_values, proposed_values):
            calls.append((base_values.copy(), proposed_values.copy()))
            return proposed_values + 10.0, {"projected": True}

        projected, report, mean = ensemble_then_project(
            base, members, project_fn=project_spy
        )
        expected = np.mean(np.stack(tuple(members.values())), axis=0)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], base)
        np.testing.assert_array_equal(calls[0][1], expected)
        np.testing.assert_array_equal(mean, expected)
        np.testing.assert_array_equal(projected, expected + 10.0)
        self.assertEqual(report, {"projected": True})

    def test_nonfinite_conditional_failure_isolated_to_one_seed(self):
        base = np.linspace(1.0, 4.0, 6)

        def fit_seed(seed):
            if seed == 7:
                raise NonfiniteTrainingError(
                    "nonfinite gradient", diagnostics={"seed": seed}
                )
            return base + seed / 100.0

        result = fit_conditional_inventory(base, fit_seed_fn=fit_seed)
        self.assertEqual(set(result.seed_predictions), {1, 21})
        self.assertEqual(set(result.failures), {7})
        self.assertEqual(result.failures[7]["seed"], 7)
        self.assertIsNone(result.ensemble_predictions)
        self.assertIsNone(result.ensemble_unprojected)

    def test_nonfinite_returned_prediction_isolated_to_one_seed(self):
        base = np.linspace(1.0, 4.0, 6)

        def fit_seed(seed):
            result = base + seed / 100.0
            if seed == 7:
                result = result.copy()
                result[0] = np.nan
            return result

        result = fit_conditional_inventory(base, fit_seed_fn=fit_seed)
        self.assertEqual(set(result.seed_predictions), {1, 21})
        self.assertEqual(set(result.failures), {7})
        self.assertEqual(result.failures[7]["status"], "failed_nonfinite")
        self.assertIsNone(result.ensemble_predictions)


class RankLockedV6ResumeTest(unittest.TestCase):
    def test_resume_rejects_changed_anchor_fold_or_normalization(self):
        expected = _lineage()
        for field in (
            "v5_anchor_sha256",
            "fold_map_sha256",
            "normalization_sha256",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(expected)
                changed[field] = "f" * 64
                with self.assertRaisesRegex(ValueError, "resume lineage"):
                    validate_resume_lineage(changed, expected)

    def test_completed_run_is_atomic_hash_checked_and_reusable(self):
        lineage = _lineage()
        predictions = np.linspace(3.0, 8.0, 25, dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed_1"
            save_completed_run(
                output,
                lineage=lineage,
                checkpoint_last={"epoch": 3, "value": torch.tensor([1.0])},
                checkpoint_best={"epoch": 2, "value": torch.tensor([2.0])},
                result={"status": "complete", "final_epoch": 3},
                diagnostics={"finite": True},
                predictions=predictions,
            )
            expected_names = {
                "lineage.json",
                "checkpoint_last.pt",
                "checkpoint_best.pt",
                "predictions.npz",
                "result.json",
                "diagnostics.json",
                "complete.json",
            }
            self.assertEqual(
                {path.name for path in output.iterdir()}, expected_names
            )
            completed = load_completed_run(output, lineage)
            self.assertTrue(completed.reused)
            self.assertEqual(completed.result["final_epoch"], 3)
            np.testing.assert_array_equal(completed.predictions, predictions)
            self.assertEqual(completed.checkpoint_best["epoch"], 2)
            self.assertEqual(completed.checkpoint_last["epoch"], 3)

    def test_completed_run_rejects_tampered_artifact(self):
        lineage = _lineage()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed_1"
            save_completed_run(
                output,
                lineage=lineage,
                checkpoint_last={"epoch": 1},
                checkpoint_best={"epoch": 1},
                result={"status": "complete"},
                diagnostics={"finite": True},
                predictions=np.arange(5.0),
            )
            (output / "result.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed run"):
                load_completed_run(output, lineage)

    def test_conditional_seed_final_fit_and_complete_resume(self):
        dataset = _TinyDataset()
        fold_ids = np.arange(len(dataset), dtype=np.int64) % 5
        config = ConditionalTrainingConfig(
            maximum_epochs=7,
            patience=2,
            batch_size=8,
            device="cpu",
        )
        lineage = _lineage(config)
        calls = {"fold": 0, "final": 0}

        def fit_fold(**kwargs):
            calls["fold"] += 1
            held = np.asarray(kwargs["holdout_indices"], dtype=np.int64)
            return FoldFitResult(
                fold=kwargs["fold"],
                predictions=dataset.labels[held],
                best_epoch=kwargs["fold"] + 1,
                best_loss=0.1 + kwargs["fold"],
                model_state={},
                history=({"epoch": kwargs["fold"] + 1},),
            )

        def fit_final(**kwargs):
            calls["final"] += 1
            self.assertEqual(kwargs["epochs"], 3)
            return FinalFitResult(
                epoch=kwargs["epochs"],
                final_loss=0.25,
                model_config={
                    "latent_widths": [4, 3, 5],
                    "affine_slope": 0.65,
                    "affine_intercept": 2.25,
                },
                model_state={"weight": torch.tensor([1.0])},
                history=({"epoch": 3, "train_loss": 0.25},),
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed_1"
            first = fit_conditional_seed(
                dataset,
                fold_ids,
                seed=1,
                config=config,
                output=output,
                lineage=lineage,
                resume=False,
                fit_fold_fn=fit_fold,
                fit_final_fn=fit_final,
            )
            self.assertFalse(first.reused)
            self.assertEqual(first.result["final_epoch"], 3)
            np.testing.assert_array_equal(first.predictions, dataset.labels)
            self.assertEqual(calls, {"fold": 5, "final": 1})

            def unexpected(**kwargs):
                raise AssertionError("completed resume must not train")

            resumed = fit_conditional_seed(
                dataset,
                fold_ids,
                seed=1,
                config=config,
                output=output,
                lineage=lineage,
                resume=True,
                fit_fold_fn=unexpected,
                fit_final_fn=unexpected,
            )
            self.assertTrue(resumed.reused)
            self.assertEqual(calls, {"fold": 5, "final": 1})

    def test_nonfinite_seed_writes_atomic_failure_diagnostic(self):
        dataset = _TinyDataset()
        fold_ids = np.arange(len(dataset), dtype=np.int64) % 5
        config = ConditionalTrainingConfig(
            maximum_epochs=7,
            patience=2,
            batch_size=8,
            device="cpu",
        )
        lineage = _lineage(config)

        def fail_fold(**kwargs):
            raise NonfiniteTrainingError(
                "nonfinite loss",
                diagnostics={
                    "seed": kwargs["seed"],
                    "fold": kwargs["fold"],
                    "epoch": 1,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed_1"
            with self.assertRaisesRegex(
                NonfiniteTrainingError, "nonfinite loss"
            ):
                fit_conditional_seed(
                    dataset,
                    fold_ids,
                    seed=1,
                    config=config,
                    output=output,
                    lineage=lineage,
                    resume=False,
                    fit_fold_fn=fail_fold,
                )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"lineage.json", "diagnostics.json"},
            )
            diagnostic = __import__("json").loads(
                (output / "diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostic["status"], "failed_nonfinite")
            self.assertEqual(diagnostic["seed"], 1)
            self.assertEqual(diagnostic["fold"], 0)
            self.assertFalse((output / "complete.json").exists())

            changed = copy.deepcopy(lineage)
            changed["fold_map_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "resume lineage"):
                fit_conditional_seed(
                    dataset,
                    fold_ids,
                    seed=1,
                    config=config,
                    output=output,
                    lineage=changed,
                    resume=True,
                    fit_fold_fn=fail_fold,
                )

            def successful_fold(**kwargs):
                held = np.asarray(kwargs["holdout_indices"], dtype=np.int64)
                return FoldFitResult(
                    fold=kwargs["fold"],
                    predictions=dataset.labels[held],
                    best_epoch=2,
                    best_loss=0.1,
                    model_state={},
                    history=({"epoch": 2},),
                )

            def successful_final(**kwargs):
                return FinalFitResult(
                    epoch=kwargs["epochs"],
                    final_loss=0.2,
                    model_config={
                        "latent_widths": [4, 3, 5],
                        "affine_slope": 0.65,
                        "affine_intercept": 2.25,
                    },
                    model_state={"weight": torch.tensor([1.0])},
                    history=({"epoch": kwargs["epochs"]},),
                )

            resumed = fit_conditional_seed(
                dataset,
                fold_ids,
                seed=1,
                config=config,
                output=output,
                lineage=lineage,
                resume=True,
                fit_fold_fn=successful_fold,
                fit_final_fn=successful_final,
            )
            self.assertFalse(resumed.reused)
            self.assertTrue((output / "complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
