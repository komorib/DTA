import dataclasses
import hashlib
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from dta.anchored_expert_fusion_v5.data import (
    AnchoredFusionDataset,
    ImmutableCacheRef,
)
from dta.anchored_expert_fusion_v5.losses import (
    anchored_residual_loss,
    mine_targeted_pairs,
)



from dta.anchored_expert_fusion_v5.model import AnchoredResidualModel
from dta.anchored_expert_fusion_v5 import training as training_module
from dta.anchored_expert_fusion_v5.training import (
    NonfiniteTrainingError,
    _atomic_npz,
    _atomic_torch_save,
    _fit_one_residual,
    StaticSpec,
    evaluate_residual,
    fit_one_residual,
    is_better_validation,
    static_predictions,
    static_spec_sha256,
)
from dta.rank_fusion_v4.cache import FusionCache
from dta.rank_fusion_v4.metrics import affinity_metrics


def _identity(split, count, marker):
    digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()
    return {
        "split": split,
        "count": count,
        "manifest_sha256": digest,
        "ordered_key_sha256": hashlib.sha256(
            (marker + "-keys").encode("utf-8")
        ).hexdigest(),
        "files": {
            name: hashlib.sha256(
                (marker + "-" + name).encode("utf-8")
            ).hexdigest()
            for name in (
                "features.npz",
                "metadata.json",
                "coverage.json",
                "normalization.json",
            )
        },
    }


def _cache(split, keys, labels=None):
    count = len(keys)
    if labels is None:
        labels = np.linspace(1.0, float(count), count, dtype=np.float32)
    predictions = np.asarray(
        [
            [1.0 + row, 10.0 - row, 3.0 + 2.0 * row]
            for row in range(count)
        ],
        dtype=np.float32,
    )
    available = np.ones((count, 3), dtype=np.bool_)
    standardized = predictions / np.float32(5.0)
    identity = _identity(split, count, split)
    metadata = {
        key: identity[key]
        for key in ("split", "count", "manifest_sha256", "ordered_key_sha256")
    }
    return FusionCache(
        keys=tuple(keys),
        labels=np.asarray(labels, dtype=np.float32),
        predictions=predictions,
        standardized_predictions=standardized,
        available=available,
        upper_bound_latents=np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        boltz_latents=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        three_dprot_latents=np.arange(count * 4, dtype=np.float32).reshape(count, 4),
        metadata=metadata,
    )


def _spec(family="raw", train_count=4, valid_count=4, marker_suffix=""):
    return StaticSpec(
        family=family,
        deploy_weights=(0.5, 0.3, 0.2),
        rank_slope=1.0 if family == "raw" else 2.5,
        rank_intercept=0.0 if family == "raw" else -0.75,
        cross_fit_report_hash=hashlib.sha256(
            ("report" + marker_suffix).encode("utf-8")
        ).hexdigest(),
        train_cache_identity=_identity(
            "train", train_count, "train" + marker_suffix
        ),
        valid_cache_identity=_identity(
            "valid", valid_count, "valid" + marker_suffix
        ),
    )


class StaticTrainingContractTest(unittest.TestCase):
    def test_static_spec_roundtrips_with_deterministic_hash_and_deep_immutability(self):
        spec = _spec("rank")
        payload = spec.to_dict()
        roundtripped = StaticSpec.from_dict(
            json.loads(json.dumps(payload, sort_keys=True))
        )

        self.assertEqual(roundtripped, spec)
        self.assertEqual(roundtripped.to_dict(), payload)
        self.assertEqual(static_spec_sha256(roundtripped), static_spec_sha256(spec))
        self.assertEqual(len(static_spec_sha256(spec)), 64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.family = "raw"
        with self.assertRaises(TypeError):
            spec.train_cache_identity["count"] = 99
        with self.assertRaises(TypeError):
            spec.train_cache_identity["files"]["features.npz"] = "changed"

    def test_static_spec_rejects_invalid_family_weights_calibration_and_hashes(self):
        valid = _spec().to_dict()
        changes = (
            ({"family": "other"}, "family"),
            ({"deploy_weights": [0.5, 0.5]}, "three"),
            ({"deploy_weights": [0.5, -0.1, 0.6]}, "nonnegative"),
            ({"deploy_weights": [0.5, float("nan"), 0.5]}, "finite"),
            ({"deploy_weights": [0.5, 0.4, 0.2]}, "sum"),
            ({"rank_slope": 2.0}, "raw.*slope"),
            ({"rank_intercept": 1.0}, "raw.*intercept"),
            ({"cross_fit_report_hash": "short"}, "hash"),
        )
        for change, message in changes:
            with self.subTest(change=change):
                payload = dict(valid)
                payload.update(change)
                with self.assertRaisesRegex(ValueError, message):
                    StaticSpec.from_dict(payload)

        bad_identity = dict(valid)
        bad_identity["train_cache_identity"] = dict(
            bad_identity["train_cache_identity"]
        )
        bad_identity["train_cache_identity"]["split"] = "valid"
        with self.assertRaisesRegex(ValueError, "train.*split"):
            StaticSpec.from_dict(bad_identity)

        bad_file_hash = dict(valid)
        bad_file_hash["valid_cache_identity"] = dict(
            bad_file_hash["valid_cache_identity"]
        )
        bad_file_hash["valid_cache_identity"]["files"] = dict(
            bad_file_hash["valid_cache_identity"]["files"]
        )
        bad_file_hash["valid_cache_identity"]["files"]["features.npz"] = "bad"
        with self.assertRaisesRegex(ValueError, "hash"):
            StaticSpec.from_dict(bad_file_hash)

    def test_static_predictions_raw_uses_missing_fallback_and_ignores_labels(self):
        cache = _cache("valid", ("a", "b", "c", "d"))
        cache.available[1, 1] = False
        cache.available[2, 2] = False
        cache.predictions[~cache.available] = 0.0
        changed_labels = dataclasses.replace(
            cache, labels=np.full(4, -999.0, dtype=np.float32)
        )
        expected_experts = np.where(
            cache.available, cache.predictions, cache.predictions[:, :1]
        )
        expected = (expected_experts @ np.asarray((0.5, 0.3, 0.2))).astype(
            np.float32
        )

        actual = static_predictions(cache, _spec("raw"))

        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (4,))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            static_predictions(changed_labels, _spec("raw")), actual
        )

    def test_static_predictions_rank_uses_complete_population_then_sealed_affine(self):
        cache = _cache("valid", ("a", "b", "c", "d"))
        cache.predictions[:] = np.asarray(
            [[4.0, 8.0, 1.0], [1.0, 8.0, 3.0], [3.0, 2.0, 2.0], [2.0, 6.0, 4.0]],
            dtype=np.float32,
        )
        expected_ranks = np.asarray(
            [
                [1.0, 5.0 / 6.0, 0.0],
                [0.0, 5.0 / 6.0, 2.0 / 3.0],
                [2.0 / 3.0, 0.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected = (
            (expected_ranks @ np.asarray((0.5, 0.3, 0.2))) * 2.5 - 0.75
        ).astype(np.float32)

        actual = static_predictions(cache, _spec("rank"))

        np.testing.assert_array_equal(actual, expected)

    def test_public_training_signatures_have_no_test_concept(self):
        signatures = {
            "fit": inspect.signature(fit_one_residual),
            "evaluate": inspect.signature(evaluate_residual),
            "static": inspect.signature(static_predictions),
        }
        for name, signature in signatures.items():
            with self.subTest(name=name):
                self.assertFalse(
                    any("test" in parameter.lower() for parameter in signature.parameters)
                )
        self.assertEqual(
            tuple(signatures["fit"].parameters),
            (
                "train_cache",
                "valid_cache",
                "static_spec",
                "budget",
                "seed",
                "output_dir",
                "device",
                "maximum_epochs",
                "patience",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "resume",
            ),
        )
        self.assertEqual(
            tuple(signatures["static"].parameters), ("cache", "spec")
        )


class CacheBackedResidualTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train_cache = _cache(
            "train", ("train-a", "train-b", "train-c", "train-d")
        )
        self.valid_cache = _cache(
            "valid", ("valid-d", "valid-b", "valid-a", "valid-c")
        )
        self.spec = _spec()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _load_torch(path):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    @staticmethod
    def _npz(path):
        with np.load(str(path), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def _fit(self, name, **changes):
        arguments = {
            "train_cache": self.train_cache,
            "valid_cache": self.valid_cache,
            "static_spec": self.spec,
            "budget": 0.25,
            "seed": 17,
            "output_dir": self.root / name,
            "device": "cpu",
            "maximum_epochs": 1,
            "patience": 8,
            "batch_size": 4,
        }
        arguments.update(changes)
        return fit_one_residual(**arguments)

    def test_tiny_fit_saves_exact_epoch_minus_one_and_readable_atomic_artifacts(self):
        result = self._fit("baseline")
        output = self.root / "baseline"
        expected_files = {
            "run_config.json",
            "history.json",
            "checkpoint_best.pt",
            "checkpoint_epoch_minus_one.pt",
            "predictions_epoch_minus_one.npz",
            "predictions_best.npz",
            "resume_state.pt",
            "result.json",
            "status.json",
        }

        self.assertEqual(result["status"], "complete")
        self.assertTrue(expected_files.issubset({path.name for path in output.iterdir()}))
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(history[0]["epoch"], -1)
        baseline = self._npz(output / "predictions_epoch_minus_one.npz")
        expected = static_predictions(self.valid_cache, self.spec)
        np.testing.assert_array_equal(baseline["predictions"], expected)
        np.testing.assert_array_equal(
            baseline["keys"], np.asarray(self.valid_cache.keys, dtype=np.str_)
        )
        np.testing.assert_array_equal(baseline["labels"], self.valid_cache.labels)
        self.assertEqual(int(baseline["epoch"]), -1)
        self.assertEqual(str(baseline["variant"]), "epoch_minus_one")

        checkpoint = self._load_torch(output / "checkpoint_epoch_minus_one.pt")
        self.assertEqual(checkpoint["epoch"], -1)
        torch.testing.assert_close(
            checkpoint["model_state_dict"]["output_layer.weight"],
            torch.zeros_like(checkpoint["model_state_dict"]["output_layer.weight"]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            checkpoint["model_state_dict"]["output_layer.bias"],
            torch.zeros_like(checkpoint["model_state_dict"]["output_layer.bias"]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            json.loads((output / "result.json").read_text(encoding="utf-8")),
            result,
        )
        self.assertEqual(
            json.loads((output / "status.json").read_text(encoding="utf-8"))["status"],
            "complete",
        )
        self.assertEqual(
            self._load_torch(output / "resume_state.pt")["next_epoch"], 1
        )

    def test_one_epoch_updates_finite_model_and_preserves_order_metrics_and_bound(self):
        result = self._fit("update")
        output = self.root / "update"
        baseline = self._load_torch(output / "checkpoint_epoch_minus_one.pt")
        resumed = self._load_torch(output / "resume_state.pt")
        changed = [
            not torch.equal(value, resumed["model_state_dict"][name])
            for name, value in baseline["model_state_dict"].items()
        ]
        self.assertTrue(any(changed))
        for value in resumed["model_state_dict"].values():
            self.assertTrue(torch.isfinite(value).all())

        best = self._npz(output / "predictions_best.npz")
        np.testing.assert_array_equal(
            best["keys"], np.asarray(self.valid_cache.keys, dtype=np.str_)
        )
        self.assertTrue(np.isfinite(best["predictions"]).all())
        expected_metrics = affinity_metrics(
            self.valid_cache.labels, best["predictions"]
        )
        self.assertEqual(set(result["best_metrics"]), set(expected_metrics))
        for name in expected_metrics:
            self.assertAlmostEqual(
                result["best_metrics"][name],
                expected_metrics[name],
                delta=1e-12,
            )
        static = static_predictions(self.valid_cache, self.spec)
        self.assertLessEqual(
            float(np.max(np.abs(best["predictions"] - static))),
            0.25 + 1e-6,
        )
        self.assertTrue(np.isfinite(result["history"][1]["loss"]))
        self.assertTrue(np.isfinite(result["history"][1]["gradient_norm"]))

    def test_epoch_minus_one_can_remain_best_and_selection_is_lexicographic(self):
        high = {"CI": 0.9, "Spearman": 0.8, "RMSE": 0.1, "MAE": 0.1, "Pearson": 0.8}
        low = {"CI": 0.8, "Spearman": 0.9, "RMSE": 0.01, "MAE": 0.01, "Pearson": 0.9}
        with mock.patch.object(
            training_module, "affinity_metrics", side_effect=[high, low]
        ):
            result = self._fit("static-best")
        self.assertEqual(result["best_epoch"], -1)
        best = self._npz(self.root / "static-best" / "predictions_best.npz")
        np.testing.assert_array_equal(
            best["predictions"], static_predictions(self.valid_cache, self.spec)
        )

        base = {"CI": 0.8, "Spearman": 0.7, "RMSE": 1.0}
        self.assertTrue(
            is_better_validation(
                {"CI": 0.9, "Spearman": 0.0, "RMSE": 9.0}, base
            )
        )
        self.assertTrue(
            is_better_validation(
                {"CI": 0.8, "Spearman": 0.8, "RMSE": 9.0}, base
            )
        )
        self.assertTrue(
            is_better_validation(
                {"CI": 0.8, "Spearman": 0.7, "RMSE": 0.9}, base
            )
        )
        self.assertFalse(is_better_validation(dict(base), base))

        epsilon = 5e-13
        self.assertTrue(
            is_better_validation(
                {**base, "CI": base["CI"] + epsilon}, base
            )
        )
        self.assertFalse(
            is_better_validation(
                {
                    **base,
                    "CI": base["CI"] - epsilon,
                    "Spearman": 1.0,
                    "RMSE": 0.0,
                },
                base,
            )
        )
        self.assertTrue(
            is_better_validation(
                {**base, "Spearman": base["Spearman"] + epsilon}, base
            )
        )
        self.assertFalse(
            is_better_validation(
                {
                    **base,
                    "Spearman": base["Spearman"] - epsilon,
                    "RMSE": 0.0,
                },
                base,
            )
        )
        self.assertTrue(
            is_better_validation(
                {**base, "RMSE": base["RMSE"] - epsilon}, base
            )
        )
        self.assertFalse(
            is_better_validation(
                {**base, "RMSE": base["RMSE"] + epsilon}, base
            )
        )

    def test_same_seed_produces_identical_history_predictions_and_model_tensors(self):
        first = self._fit("deterministic-a")
        second = self._fit("deterministic-b")
        self.assertEqual(first["history"], second["history"])
        first_predictions = self._npz(
            self.root / "deterministic-a" / "predictions_best.npz"
        )
        second_predictions = self._npz(
            self.root / "deterministic-b" / "predictions_best.npz"
        )
        for name in first_predictions:
            np.testing.assert_array_equal(
                first_predictions[name], second_predictions[name]
            )
        first_state = self._load_torch(
            self.root / "deterministic-a" / "resume_state.pt"
        )["model_state_dict"]
        second_state = self._load_torch(
            self.root / "deterministic-b" / "resume_state.pt"
        )["model_state_dict"]
        self.assertEqual(tuple(first_state), tuple(second_state))
        for name in first_state:
            torch.testing.assert_close(
                first_state[name], second_state[name], rtol=0, atol=0
            )

    def test_fit_preserves_prior_deterministic_setting_on_success_and_error(self):
        original_setting = torch.are_deterministic_algorithms_enabled()
        original_evaluate = training_module.evaluate_residual
        try:
            for enabled in (False, True):
                with self.subTest(enabled=enabled, outcome="success"):
                    torch.use_deterministic_algorithms(enabled)
                    observed = []

                    def recording_evaluate(*args, **kwargs):
                        observed.append(
                            torch.are_deterministic_algorithms_enabled()
                        )
                        return original_evaluate(*args, **kwargs)

                    with mock.patch.object(
                        training_module,
                        "evaluate_residual",
                        recording_evaluate,
                    ):
                        self._fit(
                            "deterministic-setting-" + str(enabled)
                        )
                    self.assertTrue(observed)
                    self.assertTrue(all(observed))
                    self.assertEqual(
                        torch.are_deterministic_algorithms_enabled(),
                        enabled,
                    )

            for enabled in (False, True):
                with self.subTest(enabled=enabled, outcome="error"):
                    torch.use_deterministic_algorithms(enabled)
                    with mock.patch.object(
                        training_module,
                        "evaluate_residual",
                        side_effect=RuntimeError("injected evaluation failure"),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "injected evaluation failure"
                        ):
                            self._fit(
                                "deterministic-error-" + str(enabled)
                            )
                    self.assertEqual(
                        torch.are_deterministic_algorithms_enabled(),
                        enabled,
                    )
        finally:
            torch.use_deterministic_algorithms(original_setting)

    def test_resume_restarts_failure_before_first_snapshot(self):
        output = self.root / "resume-before-first-snapshot"
        with mock.patch.object(
            training_module,
            "evaluate_residual",
            side_effect=RuntimeError("injected pre-snapshot failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "injected pre-snapshot failure"
            ):
                self._fit("resume-before-first-snapshot")
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"run_config.json", "status.json"},
        )

        result = self._fit(
            "resume-before-first-snapshot", resume=True
        )
        self.assertEqual(result["status"], "complete")
        self.assertTrue((output / "resume_state.pt").is_file())
        self.assertTrue((output / "result.json").is_file())

    def test_resume_restarts_first_batch_failure_after_baseline_artifacts(self):
        output = self.root / "resume-after-baseline"
        with mock.patch.object(
            training_module,
            "anchored_residual_loss",
            side_effect=RuntimeError("injected first-batch failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "injected first-batch failure"
            ):
                self._fit("resume-after-baseline")
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                "checkpoint_best.pt",
                "checkpoint_epoch_minus_one.pt",
                "history.json",
                "predictions_best.npz",
                "predictions_epoch_minus_one.npz",
                "run_config.json",
                "status.json",
            },
        )

        result = self._fit("resume-after-baseline", resume=True)

        self.assertEqual(result["status"], "complete")
        self.assertTrue((output / "resume_state.pt").is_file())
        self.assertTrue((output / "result.json").is_file())

    def test_patience_counts_completed_non_improving_epochs(self):
        baseline = {"CI": 0.9, "Spearman": 0.9, "RMSE": 0.1, "MAE": 0.1, "Pearson": 0.9}
        worse = {"CI": 0.8, "Spearman": 0.8, "RMSE": 0.2, "MAE": 0.2, "Pearson": 0.8}
        with mock.patch.object(
            training_module,
            "affinity_metrics",
            side_effect=[baseline, worse, worse],
        ):
            result = self._fit(
                "patience", maximum_epochs=5, patience=2
            )
        self.assertEqual(
            [row["epoch"] for row in result["history"]], [-1, 0, 1]
        )
        self.assertEqual(result["stop_reason"], "patience")

    def test_interrupted_resume_matches_uninterrupted_contents(self):
        full = self._fit(
            "resume-full", maximum_epochs=3, patience=8
        )
        common = {
            "train_cache": self.train_cache,
            "valid_cache": self.valid_cache,
            "static_spec": self.spec,
            "budget": 0.25,
            "seed": 17,
            "output_dir": self.root / "resume-split",
            "device": "cpu",
            "maximum_epochs": 3,
            "patience": 8,
            "batch_size": 4,
        }
        interrupted = _fit_one_residual(**common, stop_after_epoch=0)
        self.assertEqual(interrupted["status"], "interrupted")
        resumed = fit_one_residual(**common, resume=True)

        self.assertEqual(full["history"], resumed["history"])
        full_predictions = self._npz(
            self.root / "resume-full" / "predictions_best.npz"
        )
        resumed_predictions = self._npz(
            self.root / "resume-split" / "predictions_best.npz"
        )
        for name in full_predictions:
            np.testing.assert_array_equal(
                full_predictions[name], resumed_predictions[name]
            )
        full_state = self._load_torch(
            self.root / "resume-full" / "resume_state.pt"
        )["model_state_dict"]
        resumed_state = self._load_torch(
            self.root / "resume-split" / "resume_state.pt"
        )["model_state_dict"]
        for name in full_state:
            torch.testing.assert_close(
                full_state[name], resumed_state[name], rtol=0, atol=0
            )

    def test_resume_replays_after_fault_between_best_artifacts(self):
        def metrics(ci):
            return {
                "CI": ci,
                "Spearman": ci,
                "RMSE": 1.0 - ci,
                "MAE": 1.0 - ci,
                "Pearson": ci,
            }

        common = {
            "train_cache": self.train_cache,
            "valid_cache": self.valid_cache,
            "static_spec": self.spec,
            "budget": 0.25,
            "seed": 17,
            "device": "cpu",
            "maximum_epochs": 2,
            "patience": 8,
            "batch_size": 4,
        }
        control_arguments = {
            **common,
            "output_dir": self.root / "fault-control",
        }
        with mock.patch.object(
            training_module,
            "affinity_metrics",
            side_effect=[metrics(0.1), metrics(0.2), metrics(0.3)],
        ):
            control = fit_one_residual(**control_arguments)

        fault_arguments = {
            **common,
            "output_dir": self.root / "fault-replay",
        }
        original_atomic_npz = training_module._atomic_npz
        best_prediction_writes = 0

        def fail_third_best_prediction(path, **arrays):
            nonlocal best_prediction_writes
            if Path(path).name == "predictions_best.npz":
                best_prediction_writes += 1
                if best_prediction_writes == 3:
                    raise RuntimeError("injected best prediction failure")
            return original_atomic_npz(path, **arrays)

        with mock.patch.object(
            training_module,
            "affinity_metrics",
            side_effect=[metrics(0.1), metrics(0.2), metrics(0.3)],
        ), mock.patch.object(
            training_module,
            "_atomic_npz",
            fail_third_best_prediction,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "injected best prediction failure"
            ):
                fit_one_residual(**fault_arguments)

        fault_output = self.root / "fault-replay"
        self.assertEqual(
            self._load_torch(fault_output / "resume_state.pt")["epoch"], 0
        )
        self.assertEqual(
            self._load_torch(fault_output / "checkpoint_best.pt")["epoch"], 1
        )
        self.assertEqual(
            int(self._npz(fault_output / "predictions_best.npz")["epoch"]),
            0,
        )

        with mock.patch.object(
            training_module,
            "affinity_metrics",
            side_effect=[metrics(0.1), metrics(0.3)],
        ):
            resumed = fit_one_residual(**fault_arguments, resume=True)

        resumed_checkpoint = self._load_torch(
            fault_output / "checkpoint_best.pt"
        )
        resumed_prediction = self._npz(
            fault_output / "predictions_best.npz"
        )
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["best_epoch"], 1)
        self.assertEqual(resumed_checkpoint["epoch"], 1)
        self.assertEqual(int(resumed_prediction["epoch"]), 1)
        self.assertEqual(
            resumed["checkpoint_sha256"],
            hashlib.sha256(
                (fault_output / "checkpoint_best.pt").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            resumed["prediction_sha256"],
            hashlib.sha256(
                (fault_output / "predictions_best.npz").read_bytes()
            ).hexdigest(),
        )

        control_checkpoint = self._load_torch(
            self.root / "fault-control" / "checkpoint_best.pt"
        )
        control_prediction = self._npz(
            self.root / "fault-control" / "predictions_best.npz"
        )
        self.assertEqual(control["best_epoch"], resumed["best_epoch"])
        for name, value in control_checkpoint["model_state_dict"].items():
            torch.testing.assert_close(
                value,
                resumed_checkpoint["model_state_dict"][name],
                rtol=0,
                atol=0,
            )
        for name, value in control_prediction.items():
            np.testing.assert_array_equal(
                value, resumed_prediction[name]
            )

    def test_resume_rejects_all_changed_lineage_fields(self):
        common = {
            "train_cache": self.train_cache,
            "valid_cache": self.valid_cache,
            "static_spec": self.spec,
            "budget": 0.25,
            "seed": 17,
            "output_dir": self.root / "lineage",
            "device": "cpu",
            "maximum_epochs": 3,
            "patience": 8,
            "batch_size": 4,
        }
        _fit_one_residual(**common, stop_after_epoch=0)
        changed_static = dataclasses.replace(
            self.spec,
            cross_fit_report_hash=hashlib.sha256(b"changed").hexdigest(),
        )
        changed_cache = dataclasses.replace(
            self.train_cache,
            metadata={
                **self.train_cache.metadata,
                "manifest_sha256": hashlib.sha256(b"changed-cache").hexdigest(),
            },
        )
        changed_model = dataclasses.replace(
            self.train_cache,
            upper_bound_latents=np.zeros((4, 5), dtype=np.float32),
        )
        cases = (
            {"train_cache": changed_cache},
            {"static_spec": changed_static},
            {"budget": 0.5},
            {"seed": 18},
            {"train_cache": changed_model},
            {"learning_rate": 2e-4},
            {"weight_decay": 2e-3},
            {"batch_size": 2},
            {"maximum_epochs": 4},
            {"patience": 7},
            {"gradient_clip": 0.5},
        )
        for change in cases:
            with self.subTest(change=tuple(change)):
                arguments = dict(common)
                arguments.update(change)
                with self.assertRaises(ValueError) as context:
                    fit_one_residual(**arguments, resume=True)
                self.assertEqual(
                    str(context.exception), "resume lineage does not match"
                )

    def test_resume_rejects_one_changed_cache_file_hash_with_exact_message(self):
        common = {
            "train_cache": ImmutableCacheRef(
                self.train_cache, self.spec.train_cache_identity
            ),
            "valid_cache": ImmutableCacheRef(
                self.valid_cache, self.spec.valid_cache_identity
            ),
            "static_spec": self.spec,
            "budget": 0.25,
            "seed": 17,
            "output_dir": self.root / "file-hash-lineage",
            "device": "cpu",
            "maximum_epochs": 3,
            "patience": 8,
            "batch_size": 4,
        }
        _fit_one_residual(**common, stop_after_epoch=0)

        changed_files = dict(self.spec.train_cache_identity["files"])
        changed_files["features.npz"] = hashlib.sha256(
            b"changed-features"
        ).hexdigest()
        changed_identity = dict(self.spec.train_cache_identity)
        changed_identity["files"] = changed_files
        changed_spec = dataclasses.replace(
            self.spec, train_cache_identity=changed_identity
        )

        arguments = dict(common)
        arguments["static_spec"] = changed_spec
        with self.assertRaises(ValueError) as context:
            fit_one_residual(**arguments, resume=True)
        self.assertEqual(
            str(context.exception), "resume lineage does not match"
        )

    def test_fit_constructs_one_model_per_run(self):
        with mock.patch.object(
            training_module,
            "AnchoredResidualModel",
            wraps=AnchoredResidualModel,
        ) as constructor:
            self._fit("single-model")
        self.assertEqual(constructor.call_count, 1)

    def test_completed_resume_is_idempotent(self):
        first = self._fit("complete-resume")
        path = self.root / "complete-resume" / "result.json"
        before = path.read_bytes()
        second = self._fit("complete-resume", resume=True)
        self.assertEqual(second, first)
        self.assertEqual(path.read_bytes(), before)

    def test_completed_resume_rejects_artifact_and_result_tampering(self):
        source = self.root / "complete-integrity-source"
        self._fit("complete-integrity-source")
        mutations = (
            "delete-checkpoint",
            "delete-prediction",
            "tamper-checkpoint",
            "tamper-prediction",
            "checkpoint-path",
            "prediction-path",
            "checkpoint-hash",
            "prediction-hash",
            "run-config",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                output = self.root / ("complete-" + mutation)
                shutil.copytree(source, output)
                result_path = output / "result.json"
                result = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                checkpoint_path = output / "checkpoint_best.pt"
                prediction_path = output / "predictions_best.npz"
                result["checkpoint_path"] = str(checkpoint_path)
                result["prediction_path"] = str(prediction_path)
                result_path.write_text(
                    json.dumps(result), encoding="utf-8"
                )

                if mutation == "delete-checkpoint":
                    checkpoint_path.unlink()
                elif mutation == "delete-prediction":
                    prediction_path.unlink()
                elif mutation == "tamper-checkpoint":
                    checkpoint_path.write_bytes(
                        checkpoint_path.read_bytes() + b"tamper"
                    )
                elif mutation == "tamper-prediction":
                    prediction_path.write_bytes(
                        prediction_path.read_bytes() + b"tamper"
                    )
                elif mutation == "checkpoint-path":
                    result["checkpoint_path"] = str(
                        output / "other-checkpoint.pt"
                    )
                elif mutation == "prediction-path":
                    result["prediction_path"] = str(
                        output / "other-prediction.npz"
                    )
                elif mutation == "checkpoint-hash":
                    result["checkpoint_sha256"] = "0" * 64
                elif mutation == "prediction-hash":
                    result["prediction_sha256"] = "0" * 64
                elif mutation == "run-config":
                    result["run_config"]["seed"] += 1

                if mutation not in (
                    "delete-checkpoint",
                    "delete-prediction",
                    "tamper-checkpoint",
                    "tamper-prediction",
                ):
                    result_path.write_text(
                        json.dumps(result), encoding="utf-8"
                    )

                with self.assertRaises(ValueError) as context:
                    self._fit("complete-" + mutation, resume=True)
                self.assertEqual(
                    str(context.exception),
                    "resume lineage does not match",
                )

    def test_nonfinite_gradient_writes_detailed_failure_and_raises(self):
        original = training_module.anchored_residual_loss

        def poisoned(final, base, labels, **kwargs):
            final.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            return original(final, base, labels, **kwargs)

        with mock.patch.object(
            training_module, "anchored_residual_loss", poisoned
        ):
            with self.assertRaises(NonfiniteTrainingError):
                self._fit("nonfinite")
        failure = json.loads(
            (self.root / "nonfinite" / "failure.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(failure["status"], "failed_nonfinite")
        self.assertEqual(failure["stage"], "gradient")
        self.assertEqual(failure["epoch"], 0)
        self.assertEqual(failure["batch_index"], 0)
        self.assertEqual(failure["global_step"], 0)
        self.assertCountEqual(
            failure["sample_keys"], list(self.train_cache.keys)
        )
        self.assertEqual(failure["learning_rate"], 1e-4)
        self.assertTrue(failure["tensor_stats"])
        self.assertTrue(failure["gradient_stats"])
        self.assertTrue(failure["parameter_stats"])

    def test_atomic_helpers_replace_outputs_and_leave_no_temporary_files(self):
        output = self.root / "atomic"
        output.mkdir()
        torch_path = output / "value.pt"
        npz_path = output / "value.npz"
        _atomic_torch_save(torch_path, {"value": torch.tensor([1.0])})
        _atomic_torch_save(torch_path, {"value": torch.tensor([2.0])})
        _atomic_npz(npz_path, value=np.asarray([1.0], dtype=np.float32))
        _atomic_npz(npz_path, value=np.asarray([2.0], dtype=np.float32))

        torch.testing.assert_close(
            self._load_torch(torch_path)["value"], torch.tensor([2.0])
        )
        np.testing.assert_array_equal(
            self._npz(npz_path)["value"], np.asarray([2.0], dtype=np.float32)
        )
        self.assertEqual(
            [path.name for path in output.iterdir() if path.name.startswith(".")],
            [],
        )
        self.assertFalse(any(path.suffix == ".tmp" for path in output.iterdir()))

    def test_rejects_invalid_device_configuration_caches_and_overwrite(self):
        invalid_changes = (
            ({"maximum_epochs": 0}, "maximum_epochs"),
            ({"patience": 0}, "patience"),
            ({"batch_size": 0}, "batch_size"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"weight_decay": -1.0}, "weight_decay"),
            ({"gradient_clip": float("nan")}, "gradient_clip"),
            ({"seed": True}, "seed"),
            ({"budget": 0.2}, "budget"),
            ({"device": "meta"}, "device"),
        )
        for index, (change, message) in enumerate(invalid_changes):
            with self.subTest(change=change):
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    self._fit("invalid-" + str(index), **change)

        wrong_split = dataclasses.replace(
            self.train_cache,
            metadata={**self.train_cache.metadata, "split": "valid"},
        )
        with self.assertRaisesRegex(ValueError, "train.*identity"):
            self._fit("wrong-split", train_cache=wrong_split)

        wrong_width = dataclasses.replace(
            self.valid_cache,
            boltz_latents=np.zeros((4, 9), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "latent width"):
            self._fit("wrong-width", valid_cache=wrong_width)

        nonempty = self.root / "nonempty"
        nonempty.mkdir()
        (nonempty / "foreign.txt").write_text("foreign", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "nonempty"):
            self._fit("nonempty")

        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                self._fit("cuda", device="cuda")

    def test_evaluate_transfers_the_validation_population_to_cpu_once(self):
        base = static_predictions(self.valid_cache, self.spec)
        dataset = AnchoredFusionDataset(self.valid_cache, base)
        model = AnchoredResidualModel((2, 3, 4), budget=0.25)
        original_cpu = torch.Tensor.cpu
        cpu_calls = 0

        def counted_cpu(tensor, *args, **kwargs):
            nonlocal cpu_calls
            cpu_calls += 1
            return original_cpu(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "cpu", counted_cpu):
            result = evaluate_residual(
                model, dataset, "cpu", batch_size=1
            )

        self.assertEqual(result["keys"], self.valid_cache.keys)
        self.assertEqual(cpu_calls, 1)

    def test_training_hot_path_avoids_per_batch_full_scans_and_host_totals(self):
        source = Path(training_module.__file__).read_text(
            encoding="utf-8"
        )
        parameters_position = source.index(
            "parameters = tuple(model.parameters())"
        )
        batch_loop_position = source.index(
            "for batch_index, batch in enumerate(loader):"
        )
        self.assertLess(parameters_position, batch_loop_position)
        self.assertEqual(source.count("_all_finite(parameters)"), 1)
        self.assertNotIn("_all_finite(gradients)", source)
        self.assertNotIn("torch.isfinite(loss)", source)
        self.assertNotIn("float(loss.detach().cpu())", source)
        self.assertIn("error_if_nonfinite=True", source)

    def test_evaluate_preserves_dataset_order_and_source_has_no_mixed_precision_tokens(self):
        base = static_predictions(self.valid_cache, self.spec)
        dataset = AnchoredFusionDataset(self.valid_cache, base)
        torch.manual_seed(5)
        model = AnchoredResidualModel((2, 3, 4), budget=0.25)
        result = evaluate_residual(model, dataset, "cpu", batch_size=3)
        self.assertEqual(result["keys"], self.valid_cache.keys)
        np.testing.assert_array_equal(result["predictions"], base)
        source = Path(training_module.__file__).read_text(encoding="utf-8")
        for forbidden in ("autocast", "GradScaler", "cuda.amp"):
            self.assertNotIn(forbidden, source)



class TargetedResidualLossTest(unittest.TestCase):
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

    def test_pair_order_is_exact_for_wrong_ties_and_correct_margins(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])

        pairs = mine_targeted_pairs(labels, base, minimum_label_gap=0.5)

        expected = torch.tensor(
            [[1, 0], [1, 3], [1, 2], [2, 0], [3, 0], [3, 2]],
            dtype=torch.long,
        )
        torch.testing.assert_close(pairs, expected, rtol=0, atol=0)
        self.assertEqual(pairs.dtype, torch.long)
        self.assertEqual(pairs.device, labels.device)

    def test_zero_gap_includes_equal_labels_in_original_pair_orientation(self):
        labels = torch.tensor([2.0, 2.0, 1.0])
        base = torch.tensor([0.0, 1.0, 2.0])

        pairs = mine_targeted_pairs(labels, base, minimum_label_gap=0.0)

        self.assertTrue(torch.any(torch.all(pairs == torch.tensor([0, 1]), dim=1)))
        self.assertFalse(torch.any(torch.all(pairs == torch.tensor([1, 0]), dim=1)))

    def test_pair_at_exact_positive_label_gap_is_included(self):
        labels = torch.tensor([1.0, 1.5, 3.0])
        base = torch.tensor([0.0, 1.0, 2.0])

        pairs = mine_targeted_pairs(labels, base, minimum_label_gap=0.5)

        self.assertTrue(torch.any(torch.all(pairs == torch.tensor([1, 0]), dim=1)))

    def test_pair_mining_is_repeatable_and_truncates_after_sorting(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])
        expected = torch.tensor([[1, 0], [1, 3], [1, 2]], dtype=torch.long)

        first = mine_targeted_pairs(labels, base, maximum_pairs=3)
        second = mine_targeted_pairs(labels, base, maximum_pairs=3)

        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    def test_pair_mining_never_sorts_boolean_keys(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])
        expected = torch.tensor(
            [[1, 0], [1, 3], [1, 2], [2, 0], [3, 0], [3, 2]],
            dtype=torch.long,
        )
        original_argsort = torch.argsort

        def reject_boolean_sort(key, *args, **kwargs):
            if key.dtype == torch.bool:
                raise RuntimeError(
                    "Sort currently does not support bool dtype on CUDA"
                )
            return original_argsort(key, *args, **kwargs)

        with mock.patch.object(
            torch, "argsort", side_effect=reject_boolean_sort
        ):
            pairs = mine_targeted_pairs(
                labels, base, minimum_label_gap=0.5
            )

        torch.testing.assert_close(pairs, expected, rtol=0, atol=0)

    def test_pair_mining_rejects_mixed_float_dtypes(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        for labels, base in (
            (values.float(), values.double()),
            (values.double(), values.float()),
        ):
            with self.subTest(labels=labels.dtype, base=base.dtype):
                with self.assertRaisesRegex(ValueError, "dtype"):
                    mine_targeted_pairs(labels, base)

    def test_composite_loss_rejects_mixed_float_dtypes_before_backward(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        cases = (
            (torch.float64, torch.float32, torch.float32),
            (torch.float32, torch.float64, torch.float32),
            (torch.float32, torch.float32, torch.float64),
        )
        for final_dtype, base_dtype, labels_dtype in cases:
            final = values.to(final_dtype).requires_grad_()
            base = values.to(base_dtype)
            labels = values.to(labels_dtype)
            with self.subTest(
                final=final_dtype, base=base_dtype, labels=labels_dtype
            ):
                with self.assertRaisesRegex(ValueError, "dtype"):
                    anchored_residual_loss(final, base, labels)
                self.assertIsNone(final.grad)

    def test_public_pair_mining_uses_one_scalar_synchronization(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])

        pairs, item_calls = self.count_item_calls(
            lambda: mine_targeted_pairs(labels, base)
        )

        self.assertGreater(pairs.shape[0], 0)
        self.assertEqual(item_calls, 1)

    def test_composite_loss_uses_one_total_scalar_synchronization(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])
        final = torch.tensor([1.5, 2.5, 2.0, 3.0], requires_grad=True)

        (total, components), item_calls = self.count_item_calls(
            lambda: anchored_residual_loss(final, base, labels)
        )

        self.assertTrue(torch.isfinite(total))
        self.assertGreater(components["pair_count"], 0)
        self.assertEqual(item_calls, 1)

    def test_no_pair_shape_and_pair_component_is_differentiable_zero(self):
        labels = torch.tensor([1.0, 1.2, 1.4])
        base = torch.tensor([3.0, 2.0, 1.0])
        final = torch.tensor([2.5, 2.0, 1.5], requires_grad=True)

        pairs = mine_targeted_pairs(labels, base, minimum_label_gap=0.5)
        total, components = anchored_residual_loss(
            final, base, labels, minimum_label_gap=0.5
        )
        components["targeted_pair"].backward(retain_graph=True)

        self.assertEqual(tuple(pairs.shape), (0, 2))
        self.assertEqual(pairs.dtype, torch.long)
        self.assertEqual(components["pair_count"], 0)
        torch.testing.assert_close(
            components["targeted_pair"], torch.zeros(()), rtol=0, atol=0
        )
        self.assertIsNotNone(final.grad)
        torch.testing.assert_close(final.grad, torch.zeros_like(final), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(total))

    def test_loss_matches_exact_formula_and_components(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])
        final = torch.tensor([1.5, 2.5, 2.0, 3.0], requires_grad=True)
        pairs = mine_targeted_pairs(labels, base, maximum_pairs=4)
        high, low = pairs[:, 0], pairs[:, 1]
        expected_pair = F.softplus(-(final[high] - final[low])).mean()
        expected_regression = F.smooth_l1_loss(final, labels)
        expected_anchor = ((final - base) ** 2).mean()
        expected_total = expected_pair + 0.25 * expected_regression + 0.50 * expected_anchor

        total, components = anchored_residual_loss(
            final, base, labels, maximum_pairs=4
        )

        torch.testing.assert_close(total, expected_total, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(components["targeted_pair"], expected_pair)
        torch.testing.assert_close(components["smooth_l1"], expected_regression)
        torch.testing.assert_close(components["anchor"], expected_anchor)
        self.assertEqual(components["pair_count"], 4)
        self.assertIsInstance(components["pair_count"], int)

    def test_loss_pair_selection_depends_on_base_not_final_predictions(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])
        final_a = torch.tensor([-100.0, 100.0, -50.0, 50.0])
        final_b = -final_a
        fixed_pairs = mine_targeted_pairs(labels, base, maximum_pairs=3)
        high, low = fixed_pairs[:, 0], fixed_pairs[:, 1]

        _, components_a = anchored_residual_loss(
            final_a, base, labels, maximum_pairs=3
        )
        _, components_b = anchored_residual_loss(
            final_b, base, labels, maximum_pairs=3
        )

        expected_a = F.softplus(-(final_a[high] - final_a[low])).mean()
        expected_b = F.softplus(-(final_b[high] - final_b[low])).mean()
        torch.testing.assert_close(components_a["targeted_pair"], expected_a)
        torch.testing.assert_close(components_b["targeted_pair"], expected_b)
        self.assertEqual(components_a["pair_count"], fixed_pairs.shape[0])
        self.assertEqual(components_b["pair_count"], fixed_pairs.shape[0])

    def test_loss_gradient_is_finite(self):
        labels = torch.tensor([1.0, 4.0, 2.0, 3.0])
        base = torch.tensor([2.0, 0.0, 1.0, 2.0])
        final = torch.tensor([1.5, 2.5, 2.0, 3.0], requires_grad=True)

        total, _ = anchored_residual_loss(final, base, labels)
        total.backward()

        self.assertIsNotNone(final.grad)
        self.assertTrue(torch.isfinite(final.grad).all())

    def test_pair_mining_rejects_invalid_tensors_and_arguments(self):
        valid = torch.tensor([1.0, 2.0])
        tensor_cases = (
            (valid[:, None], valid, "1D"),
            (valid, valid[:1], "same"),
            (torch.tensor([]), torch.tensor([]), "nonempty"),
            (torch.tensor([1.0, float("nan")]), valid, "labels.*finite"),
            (valid, torch.tensor([1.0, float("inf")]), "base.*finite"),
            (torch.tensor([1, 2]), torch.tensor([1, 2]), "floating"),
        )
        for labels, base, message in tensor_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    mine_targeted_pairs(labels, base)
        for maximum_pairs in (True, False, 0, -1, 1.5):
            with self.subTest(maximum_pairs=maximum_pairs):
                with self.assertRaisesRegex(ValueError, "maximum_pairs"):
                    mine_targeted_pairs(valid, valid, maximum_pairs=maximum_pairs)
        for gap in (True, False, -0.1, float("nan"), float("inf")):
            with self.subTest(gap=gap):
                with self.assertRaisesRegex(ValueError, "minimum_label_gap"):
                    mine_targeted_pairs(valid, valid, minimum_label_gap=gap)

    def test_loss_rejects_invalid_shape_match_finiteness_and_device(self):
        valid = torch.tensor([1.0, 2.0])
        cases = (
            (valid[:, None], valid, valid, "1D"),
            (valid, valid[:1], valid, "matching"),
            (
                valid,
                valid,
                torch.tensor([1.0, float("nan")]),
                "labels.*finite",
            ),
            (torch.tensor([1, 2]), valid, valid, "floating"),
        )
        for final, base, labels, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    anchored_residual_loss(final, base, labels)
        if torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "device"):
                anchored_residual_loss(valid.cuda(), valid, valid)


if __name__ == "__main__":
    unittest.main()
