import os
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dta.rank_locked_calibration_v6 import CANDIDATE_NAMES
from dta.rank_locked_calibration_v6.calibrators import (
    MonotonicSplineCalibrator,
    PositiveAffineCalibrator,
)
from dta.rank_locked_calibration_v6.selection import (
    PACKAGE_FILES,
    ValidationCandidate,
    seal_validation_package,
    select_candidate,
    validate_candidate_inventory,
    verify_validation_package,
)
from dta.rank_locked_calibration_v6.run_validation import (
    DefaultValidationHooks,
    ValidationMaterial,
    build_arg_parser,
    run_validation,
)
import dta.rank_locked_calibration_v6.run_validation as validation_module
from dta.rank_locked_calibration_v6.evaluate_test import (
    OFFICIAL_TEST_IDENTITY,
    evaluate_official_test,
    predict_selected_model,
    reserve_evaluation,
    test_status,
)
import dta.rank_locked_calibration_v6.evaluate_test as evaluation_module
from utils.experiment_integrity import sha256_file


def _candidate(
    name,
    *,
    ci=0.72,
    rmse=1.30,
    mae=1.00,
    parameter_count=10,
    predictions=None,
    status="complete",
):
    if predictions is None:
        predictions = np.asarray([3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    if name == "identity":
        candidate_type = "identity"
    elif name.startswith("affine") or name.startswith("spline"):
        candidate_type = "analytic"
    elif name == "conditional_ensemble":
        candidate_type = "ensemble"
    else:
        candidate_type = "conditional"
    return ValidationCandidate(
        name=name,
        candidate_type=candidate_type,
        validation_metrics={
            "CI": ci,
            "MAE": mae,
            "RMSE": rmse,
            "Pearson": 0.60,
            "Spearman": 0.59,
        },
        parameter_count=parameter_count,
        predictions=predictions,
        model_descriptor={"type": candidate_type, "name": name},
        oof_summary={"status": "complete"},
        status=status,
    )


def _inventory():
    return tuple(
        _candidate(name, parameter_count=index)
        for index, name in enumerate(CANDIDATE_NAMES)
    )


class RankLockedV6CandidateSelectionTest(unittest.TestCase):
    def test_all_three_validation_gates_are_required(self):
        passing = _candidate(
            "affine_mse", ci=0.713374, rmse=1.34, mae=1.04
        )
        bad_ci = _candidate(
            "affine_huber", ci=0.713373, rmse=1.20, mae=0.90
        )
        bad_rmse = _candidate(
            "spline_8_knots", ci=0.72, rmse=1.36, mae=0.90
        )
        bad_mae = _candidate(
            "spline_16_knots", ci=0.72, rmse=1.20, mae=1.06
        )
        selected = select_candidate((bad_ci, bad_rmse, bad_mae, passing))
        self.assertEqual(selected.name, "affine_mse")

    def test_rmse_within_point_zero_zero_five_uses_mae(self):
        left = _candidate("affine_mse", rmse=1.300, mae=1.02)
        right = _candidate("affine_huber", rmse=1.304, mae=0.99)
        self.assertEqual(
            select_candidate((left, right)).name, "affine_huber"
        )

    def test_rmse_difference_at_boundary_uses_lower_rmse(self):
        left = _candidate("affine_mse", rmse=1.300, mae=1.02)
        right = _candidate("affine_huber", rmse=1.305, mae=0.90)
        self.assertEqual(select_candidate((left, right)).name, "affine_mse")

    def test_parameter_count_then_name_are_final_tie_breakers(self):
        larger = _candidate(
            "conditional_seed_1", parameter_count=100
        )
        smaller = _candidate(
            "conditional_seed_7", parameter_count=50
        )
        self.assertEqual(
            select_candidate((larger, smaller)).name,
            "conditional_seed_7",
        )
        alpha = _candidate("affine_huber", parameter_count=5)
        beta = _candidate("affine_mse", parameter_count=5)
        self.assertEqual(
            select_candidate((alpha, beta)).name, "affine_huber"
        )

    def test_no_passing_candidate_returns_none_for_external_v5_fallback(self):
        failed = _candidate("identity", ci=0.70, rmse=1.40, mae=1.10)
        self.assertIsNone(select_candidate((failed,)))

    def test_failed_conditional_seed_is_never_eligible(self):
        failed = _candidate(
            "conditional_seed_1",
            ci=0.90,
            rmse=0.50,
            mae=0.40,
            status="failed_nonfinite",
        )
        passing = _candidate("affine_mse", ci=0.72, rmse=1.30, mae=1.00)
        self.assertEqual(select_candidate((failed, passing)).name, "affine_mse")

    def test_inventory_is_exactly_the_frozen_nine_names(self):
        records = _inventory()
        self.assertEqual(validate_candidate_inventory(records), records)
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_candidate_inventory(records[:-1])
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_candidate_inventory(records + (records[0],))
        extra = _candidate("identity")
        object.__setattr__(extra, "name", "unexpected")
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_candidate_inventory(records[:-1] + (extra,))

    def test_candidate_rejects_nonfinite_metrics_and_non_float64_predictions(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _candidate("affine_mse", ci=float("nan"))
        with self.assertRaisesRegex(ValueError, "float64"):
            _candidate(
                "affine_mse",
                predictions=np.ones(4, dtype=np.float32),
            )


class RankLockedV6OfficialEvaluationPolicyTest(unittest.TestCase):
    @staticmethod
    def _conditional_dataset(count=8):
        base = np.linspace(2.0, 9.0, count, dtype=np.float64)
        records = []
        for index in range(count):
            records.append(
                {
                    "key": "test_%02d" % index,
                    "label": 0.7 * base[index] + 1.5,
                    "base_prediction": base[index],
                    "raw_predictions": np.zeros(3, dtype=np.float32),
                    "standardized_predictions": np.zeros(3, dtype=np.float32),
                    "available": np.ones(3, dtype=np.bool_),
                    "latents": (
                        np.zeros(4, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                        np.zeros(5, dtype=np.float32),
                    ),
                }
            )
        return base, records

    @staticmethod
    def _conditional_checkpoint(slope, intercept):
        from dta.rank_locked_calibration_v6.model import ConditionalCalibrator

        model = ConditionalCalibrator((4, 3, 5), slope, intercept)
        return {
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
        }

    def test_test_status_requires_all_strict_gates(self):
        self.assertEqual(
            test_status({"CI": 0.708700, "RMSE": 1.49, "MAE": 1.17}),
            "passed",
        )
        self.assertEqual(
            test_status({"CI": 0.708699, "RMSE": 1.40, "MAE": 1.00}),
            "failed_gate",
        )
        self.assertEqual(
            test_status({"CI": 0.72, "RMSE": 1.497357, "MAE": 1.00}),
            "failed_gate",
        )
        self.assertEqual(
            test_status({"CI": 0.72, "RMSE": 1.40, "MAE": 1.174794}),
            "failed_gate",
        )

    def test_official_test_identity_freezes_count_order_and_four_files(self):
        self.assertEqual(OFFICIAL_TEST_IDENTITY["count"], 4751)
        self.assertEqual(
            OFFICIAL_TEST_IDENTITY["manifest_sha256"],
            "f8741d40837568c552ca0bd3725e0b079ba855d56d2a663c0bb95c4bc09c95b8",
        )
        self.assertEqual(
            OFFICIAL_TEST_IDENTITY["ordered_key_sha256"],
            "432db77b96cc1c5ff8eff4de655b64b4a487c4b448913ee7db33cdaf886b303f",
        )
        self.assertEqual(
            OFFICIAL_TEST_IDENTITY["files"],
            {
                "features.npz": "b8ded10b2499f1d4d81b9321a914d5d8146f1296528c89d095369843c928abd3",
                "metadata.json": "9a539a21e71d2b3f093a43d43762a78bd5df1c92937d8559c66155a68e173fbe",
                "coverage.json": "26672db86031ad4d6dfedf3be873867ece9537d45909c556737b2dd776c6e280",
                "normalization.json": "65c5b00a0cc5952c8da3ce3be0e8a96b299ad07fb1acebda15c16b1e7d3ea340",
            },
        )

    def test_evaluation_reservation_is_exclusive_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "official"
            start = reserve_evaluation(output, {"selection_seal_sha256": "a" * 64})
            self.assertTrue(start.is_file())
            with self.assertRaisesRegex(ValueError, "already reserved"):
                reserve_evaluation(
                    output, {"selection_seal_sha256": "a" * 64}
                )
            (output / "result.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already"):
                reserve_evaluation(
                    output, {"selection_seal_sha256": "a" * 64}
                )

    def test_analytic_selected_model_is_reconstructed_without_labels(self):
        base = np.linspace(2.0, 10.0, 20, dtype=np.float64)
        labels = 2.0 + 0.6 * base
        model = PositiveAffineCalibrator.fit(base, labels, objective="mse")
        proposed = predict_selected_model(
            {
                "schema_version": 1,
                "name": "affine_mse",
                "candidate_type": "analytic",
                "status": "complete",
                "model": model.to_dict(),
            },
            dataset=None,
            base_predictions=base,
            device="cpu",
        )
        self.assertEqual(proposed.dtype, np.float64)
        np.testing.assert_allclose(proposed, labels, rtol=0.0, atol=1e-12)

    def test_analytic_model_must_match_selected_inventory_name(self):
        base = np.linspace(2.0, 10.0, 20, dtype=np.float64)
        model = PositiveAffineCalibrator.fit(
            base, 2.0 + 0.6 * base, objective="huber"
        )
        with self.assertRaisesRegex(ValueError, "selected model name"):
            predict_selected_model(
                {
                    "schema_version": 1,
                    "name": "affine_mse",
                    "candidate_type": "analytic",
                    "status": "complete",
                    "model": model.to_dict(),
                },
                dataset=None,
                base_predictions=base,
                device="cpu",
            )

    def test_conditional_selected_model_reconstructs_checkpoint(self):
        base, dataset = self._conditional_dataset()
        proposed = predict_selected_model(
            {
                "schema_version": 1,
                "name": "conditional_seed_1",
                "candidate_type": "conditional",
                "status": "complete",
                "model": self._conditional_checkpoint(0.5, 1.25),
            },
            dataset=dataset,
            base_predictions=base,
            device="cpu",
        )
        self.assertEqual(proposed.dtype, np.float64)
        np.testing.assert_allclose(
            proposed, 0.5 * base + 1.25, rtol=0.0, atol=2e-6
        )

    def test_ensemble_averages_three_checkpoints_before_projection(self):
        base, dataset = self._conditional_dataset()
        members = {
            "1": self._conditional_checkpoint(0.5, 1.0),
            "7": self._conditional_checkpoint(1.0, 2.0),
            "21": self._conditional_checkpoint(1.5, 3.0),
        }
        proposed = predict_selected_model(
            {
                "schema_version": 1,
                "name": "conditional_ensemble",
                "candidate_type": "ensemble",
                "status": "complete",
                "model": {
                    "type": "conditional_ensemble",
                    "rule": "arithmetic_mean_before_one_rank_projection",
                    "members": members,
                },
            },
            dataset=dataset,
            base_predictions=base,
            device="cpu",
        )
        self.assertEqual(proposed.dtype, np.float64)
        np.testing.assert_allclose(proposed, base + 2.0, rtol=0.0, atol=2e-6)

    def test_ensemble_requires_the_frozen_aggregation_rule(self):
        base, dataset = self._conditional_dataset()
        checkpoint = self._conditional_checkpoint(1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "ensemble selected model"):
            predict_selected_model(
                {
                    "schema_version": 1,
                    "name": "conditional_ensemble",
                    "candidate_type": "ensemble",
                    "status": "complete",
                    "model": {
                        "type": "conditional_ensemble",
                        "rule": "mean_after_projection",
                        "members": {
                            "1": checkpoint,
                            "7": checkpoint,
                            "21": checkpoint,
                        },
                    },
                },
                dataset=dataset,
                base_predictions=base,
                device="cpu",
            )

    def test_official_cache_rejects_changed_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(OFFICIAL_TEST_IDENTITY["files"]):
                (root / name).write_bytes((name + str(index)).encode("ascii"))
            identity = {
                "split": "test",
                "count": 2,
                "manifest_sha256": "a" * 64,
                "ordered_key_sha256": "b" * 64,
                "files": {
                    name: sha256_file(root / name)
                    for name in OFFICIAL_TEST_IDENTITY["files"]
                },
            }
            cache = SimpleNamespace(
                keys=("first", "second"),
                metadata={
                    "split": "test",
                    "count": 2,
                    "manifest_sha256": "a" * 64,
                    "ordered_key_sha256": "b" * 64,
                },
            )
            with patch.object(
                evaluation_module, "OFFICIAL_TEST_IDENTITY", identity
            ):
                evaluation_module._validate_official_cache(root, cache)
                (root / "features.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash"):
                    evaluation_module._validate_official_cache(root, cache)

    def test_official_evaluation_writes_float64_atomic_inventory_and_preserves_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            cache_root = root / "test_cache"
            output = root / "official_test"
            package.mkdir()
            cache_root.mkdir()
            (package / "selection_seal.json").write_text(
                '{"status":"sealed"}\n', encoding="utf-8"
            )
            normalization = {"schema_version": 1, "source": "train"}
            for path in (
                package / "normalization.json",
                cache_root / "normalization.json",
            ):
                path.write_text(json.dumps(normalization) + "\n", encoding="utf-8")
            rollback_package = root / "v5_package"
            rollback_package.mkdir()
            rollback_result = root / "v5_result.json"
            rollback_marker = rollback_package / "selection_seal.json"
            rollback_marker.write_bytes(b"immutable-v5-package\n")
            rollback_result.write_bytes(b"immutable-v5-result\n")
            before = {
                str(path): sha256_file(path)
                for path in (rollback_marker, rollback_result)
            }
            base = np.asarray([7.0, 2.0, 9.0, 4.0], dtype=np.float64)
            labels = np.asarray([7.2, 2.5, 8.4, 4.1], dtype=np.float64)
            cache = SimpleNamespace(
                keys=("a", "b", "c", "d"),
                labels=labels,
                metadata={"split": "test"},
            )
            selection = {
                "selected_name": "identity",
                "selected_type": "identity",
                "selected_validation_metrics": {
                    "CI": 0.72,
                    "MAE": 1.0,
                    "RMSE": 1.2,
                    "Pearson": 0.6,
                    "Spearman": 0.6,
                },
            }
            selected_model = {
                "schema_version": 1,
                "name": "identity",
                "candidate_type": "identity",
                "status": "complete",
                "model": {"type": "v5_static_anchor"},
            }
            fallback_metrics = evaluation_module.affinity_metrics(labels, base)
            rollback = {
                "v5_validation_package": str(rollback_package),
                "v5_official_test_result": str(rollback_result),
            }
            with patch.object(
                evaluation_module,
                "verify_validation_package",
                return_value=selection,
            ), patch.object(
                evaluation_module, "load_fusion_cache", return_value=cache
            ), patch.object(
                evaluation_module, "_validate_official_cache"
            ), patch.object(
                evaluation_module,
                "_load_selected_model",
                return_value=selected_model,
            ), patch.object(
                evaluation_module,
                "_load_v5_base",
                return_value=(base, fallback_metrics, rollback),
            ), patch.object(
                evaluation_module,
                "RankLockedCalibrationDataset",
                return_value=[],
            ):
                result = evaluate_official_test(
                    package, cache_root, output, device="cpu"
                )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "evaluation.start",
                    "predictions.npz",
                    "projection_report.json",
                    "result.json",
                    "evaluation.complete",
                },
            )
            with np.load(output / "predictions.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["base_predictions"].dtype, np.float64)
                self.assertEqual(archive["proposed_predictions"].dtype, np.float64)
                self.assertEqual(archive["predictions"].dtype, np.float64)
                np.testing.assert_array_equal(
                    np.argsort(archive["predictions"], kind="mergesort"),
                    np.argsort(base, kind="mergesort"),
                )
            complete = json.loads(
                (output / "evaluation.complete").read_text(encoding="utf-8")
            )
            self.assertEqual(
                complete["result_sha256"], sha256_file(output / "result.json")
            )
            self.assertEqual(
                complete["projection_report_sha256"],
                sha256_file(output / "projection_report.json"),
            )
            self.assertEqual(
                result["prediction_sha256"], complete["prediction_sha256"]
            )
            self.assertEqual(
                before,
                {
                    str(path): sha256_file(path)
                    for path in (rollback_marker, rollback_result)
                },
            )


class RankLockedV6ValidationPackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.v5_package = self.root / "v5_package"
        self.v5_package.mkdir()
        self.v5_seal = self.v5_package / "selection_seal.json"
        self.v5_seal.write_text('{"status":"sealed"}\n', encoding="utf-8")
        self.v5_result = self.root / "v5_result.json"
        self.v5_result.write_text('{"status":"passed"}\n', encoding="utf-8")
        self.rollback = {
            "schema_version": 1,
            "promotion_status": "not_promoted",
            "v5_validation_package": str(self.v5_package.resolve()),
            "v5_selection_seal_sha256": sha256_file(self.v5_seal),
            "v5_official_test_result": str(self.v5_result.resolve()),
            "v5_official_test_result_sha256": sha256_file(self.v5_result),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _seal(self):
        package = self.root / "validation_package"
        candidates = _inventory()
        selected = select_candidate(candidates)
        seal_validation_package(
            package,
            selected=selected,
            candidates=candidates,
            cache_identities={
                "train": {"count": 8, "manifest_sha256": "a" * 64},
                "valid": {"count": 4, "manifest_sha256": "b" * 64},
            },
            normalization={"schema_version": 1, "mean": [0.0], "std": [1.0]},
            fold_map={
                "keys": np.asarray(
                    ["train_%02d" % index for index in range(8)],
                    dtype=np.str_,
                ),
                "fold_ids": np.arange(8, dtype=np.int64) % 5,
            },
            selected_model={"schema_version": 1, "name": selected.name},
            valid_keys=np.asarray(["a", "b", "c", "d"], dtype=np.str_),
            valid_labels=np.asarray([3.0, 4.0, 5.0, 6.0], dtype=np.float64),
            valid_predictions=selected.predictions,
            rollback=self.rollback,
        )
        return package, selected

    def test_seal_has_exact_read_only_inventory_and_verifies(self):
        package, selected = self._seal()
        self.assertEqual({path.name for path in package.iterdir()}, set(PACKAGE_FILES))
        for path in package.iterdir():
            self.assertEqual(os.lstat(path).st_mode & 0o222, 0)
        verified = verify_validation_package(package)
        self.assertEqual(verified["selected_name"], selected.name)
        self.assertEqual(verified["status"], "passed")

    def test_verifier_rejects_writable_tampered_extra_and_symlink_files(self):
        package, _ = self._seal()
        selection = package / "selection.json"
        selection.chmod(stat.S_IRUSR | stat.S_IWUSR)
        with self.assertRaisesRegex(ValueError, "writable"):
            verify_validation_package(package)

        selection.chmod(stat.S_IRUSR | stat.S_IWUSR)
        selection.write_text("{}\n", encoding="utf-8")
        selection.chmod(stat.S_IRUSR)
        with self.assertRaisesRegex(ValueError, "hash"):
            verify_validation_package(package)

        package, _ = self._seal_fresh("extra_package")
        (package / "extra.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_validation_package(package)

        package, _ = self._seal_fresh("symlink_package")
        model = package / "selected_model.pt"
        model.chmod(stat.S_IRUSR | stat.S_IWUSR)
        model.unlink()
        model.symlink_to(self.v5_result)
        with self.assertRaisesRegex(ValueError, "symlink"):
            verify_validation_package(package)

    def _seal_fresh(self, name):
        original_root = self.root
        target = self.root / name
        candidates = _inventory()
        selected = select_candidate(candidates)
        seal_validation_package(
            target,
            selected=selected,
            candidates=candidates,
            cache_identities={"train": {"count": 8}, "valid": {"count": 4}},
            normalization={"schema_version": 1},
            fold_map={
                "keys": np.asarray([str(index) for index in range(8)]),
                "fold_ids": np.arange(8, dtype=np.int64) % 5,
            },
            selected_model={"schema_version": 1, "name": selected.name},
            valid_keys=np.asarray(["a", "b", "c", "d"]),
            valid_labels=np.arange(4.0),
            valid_predictions=selected.predictions,
            rollback=self.rollback,
        )
        self.root = original_root
        return target, selected

    def test_verifier_rejects_changed_v5_rollback_artifact(self):
        package, _ = self._seal()
        self.v5_result.write_text('{"status":"changed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "rollback"):
            verify_validation_package(package)


class RankLockedV6ValidationWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.v5_package = self.root / "v5_package"
        self.v5_package.mkdir()
        self.v5_seal = self.v5_package / "selection_seal.json"
        self.v5_seal.write_text('{"status":"sealed"}\n', encoding="utf-8")
        self.v5_result = self.root / "v5_result.json"
        self.v5_result.write_text('{"status":"passed"}\n', encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _rollback(self):
        return {
            "schema_version": 1,
            "promotion_status": "not_promoted",
            "v5_validation_package": str(self.v5_package.resolve()),
            "v5_selection_seal_sha256": sha256_file(self.v5_seal),
            "v5_official_test_result": str(self.v5_result.resolve()),
            "v5_official_test_result_sha256": sha256_file(self.v5_result),
        }

    def _material(self, candidates):
        return ValidationMaterial(
            candidates=tuple(candidates),
            cache_identities={"train": {"count": 8}, "valid": {"count": 4}},
            normalization={"schema_version": 1},
            fold_map={
                "keys": np.asarray([str(index) for index in range(8)]),
                "fold_ids": np.arange(8, dtype=np.int64) % 5,
            },
            selected_models={
                candidate.name: {"schema_version": 1, "name": candidate.name}
                for candidate in candidates
            },
            valid_keys=np.asarray(["a", "b", "c", "d"]),
            valid_labels=np.arange(3.0, 7.0, dtype=np.float64),
            rollback=self._rollback(),
        )

    def _hooks(self, candidates, events):
        output_root = self.root / "run"

        class Hooks:
            def load_train(self, args):
                events.append("load_train")
                return {"train": True}

            def fit_train(self, train_context, args):
                events.append("fit_train")
                return {"models": True}

            def load_valid(self, train_context, train_artifacts, args):
                self.assert_inventory_lock(output_root)
                events.append("load_valid")
                return {"valid": True}

            def assert_inventory_lock(self, root):
                if not (root / "candidate_inventory_lock.json").is_file():
                    raise AssertionError("valid loaded before inventory lock")

            def evaluate(self, train_context, train_artifacts, valid_context, args):
                events.append("evaluate")
                return self_material

        self_material = self._material(candidates)
        return Hooks(), output_root

    def test_inventory_is_locked_before_valid_and_formal_pass_is_sealed(self):
        events = []
        hooks, output_root = self._hooks(_inventory(), events)
        result = run_validation(
            SimpleNamespace(output_root=str(output_root), limit=None),
            hooks=hooks,
        )
        self.assertEqual(
            events, ["load_train", "fit_train", "load_valid", "evaluate"]
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue((output_root / "validation_package").is_dir())
        verify_validation_package(output_root / "validation_package")

    def test_limit_run_never_creates_formal_package(self):
        events = []
        hooks, output_root = self._hooks(_inventory(), events)
        result = run_validation(
            SimpleNamespace(output_root=str(output_root), limit=8),
            hooks=hooks,
        )
        self.assertEqual(result["status"], "smoke_complete")
        self.assertFalse((output_root / "validation_package").exists())
        self.assertTrue((output_root / "smoke_result.json").is_file())

    def test_no_passing_candidate_writes_failure_without_package(self):
        failed = tuple(
            _candidate(
                name,
                ci=0.70,
                rmse=1.40,
                mae=1.10,
                parameter_count=index,
            )
            for index, name in enumerate(CANDIDATE_NAMES)
        )
        events = []
        hooks, output_root = self._hooks(failed, events)
        result = run_validation(
            SimpleNamespace(output_root=str(output_root), limit=None),
            hooks=hooks,
        )
        self.assertEqual(result["status"], "validation_failed")
        self.assertFalse(result["official_test_allowed"])
        self.assertFalse((output_root / "validation_package").exists())
        self.assertTrue((output_root / "validation_failed.json").is_file())

    def test_cli_has_no_test_input_option(self):
        parser = build_arg_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertFalse(any("test" in option for option in option_strings))

    def test_default_evaluate_keeps_failed_seeds_but_preserves_anchor_rank(self):
        base = np.linspace(2.0, 10.0, 40, dtype=np.float64)
        labels = 2.0 + 0.6 * base + 0.1 * np.sin(base)
        analytic = {
            "affine_mse": PositiveAffineCalibrator.fit(
                base, labels, objective="mse"
            ).to_dict(),
            "affine_huber": PositiveAffineCalibrator.fit(
                base, labels, objective="huber"
            ).to_dict(),
            "spline_8_knots": MonotonicSplineCalibrator.fit(
                base, labels, knot_count=8
            ).to_dict(),
            "spline_16_knots": MonotonicSplineCalibrator.fit(
                base, labels, knot_count=16
            ).to_dict(),
        }
        artifacts = {
            "identity": {
                "status": "complete",
                "model_descriptor": {"type": "identity"},
                "oof_summary": {"status": "complete"},
            }
        }
        for name, descriptor in analytic.items():
            artifacts[name] = {
                "status": "complete",
                "model_descriptor": descriptor,
                "oof_summary": {"status": "complete"},
            }
        for seed in (1, 7, 21):
            artifacts["conditional_seed_%d" % seed] = {
                "status": "failed_nonfinite",
                "model_descriptor": {"type": "conditional", "seed": seed},
                "oof_summary": {"status": "failed_nonfinite"},
            }
        artifacts["conditional_ensemble"] = {
            "status": "failed_missing_seed",
            "model_descriptor": {"type": "ensemble"},
            "oof_summary": {"status": "failed_missing_seed"},
        }
        train_context = {
            "cache_identities": {"train": {"count": 40}, "valid": {"count": 40}},
            "normalization": {"schema_version": 1},
            "fold_map": {
                "keys": np.asarray([str(index) for index in range(40)]),
                "fold_ids": np.arange(40, dtype=np.int64) % 5,
            },
            "rollback": self._rollback(),
        }
        valid_context = {
            "keys": np.asarray(["valid_%02d" % index for index in range(40)]),
            "labels": labels,
            "base_predictions": base,
            "dataset": None,
        }
        material = DefaultValidationHooks().evaluate(
            train_context,
            artifacts,
            valid_context,
            SimpleNamespace(device="cpu"),
        )
        self.assertEqual(
            tuple(candidate.name for candidate in material.candidates),
            CANDIDATE_NAMES,
        )
        identity = material.candidates[0]
        for candidate in material.candidates[:5]:
            self.assertEqual(
                candidate.validation_metrics["CI"],
                identity.validation_metrics["CI"],
            )
            self.assertEqual(
                candidate.validation_metrics["Spearman"],
                identity.validation_metrics["Spearman"],
            )
        self.assertTrue(
            all(
                candidate.status != "complete" and not candidate.eligible
                for candidate in material.candidates[5:]
            )
        )

    def test_default_fit_train_uses_exact_three_seeds_and_nine_names(self):
        count = 25
        base = np.linspace(2.0, 10.0, count, dtype=np.float64)
        labels = 2.25 + 0.65 * base
        dataset = [
            {"label": labels[index], "base_prediction": base[index]}
            for index in range(count)
        ]
        calls = []

        def conditional_runner(*args, **kwargs):
            seed = kwargs["seed"]
            calls.append(seed)
            return SimpleNamespace(
                predictions=labels.copy(),
                result={"oof_metrics": {"RMSE": 0.0}, "final_epoch": 2},
                diagnostics={"status": "finite"},
                checkpoint_best={
                    "model_config": {
                        "latent_widths": [4, 3, 5],
                        "affine_slope": 0.65,
                        "affine_intercept": 2.25,
                    },
                    "model_state_dict": {},
                },
            )

        hooks = DefaultValidationHooks(conditional_runner=conditional_runner)
        artifacts = hooks.fit_train(
            {
                "dataset": dataset,
                "fold_ids": np.arange(count, dtype=np.int64) % 5,
                "base_predictions": base,
                "labels": labels,
                "anchor_descriptor": {"type": "identity"},
                "training_config": object(),
                "conditional_output_root": self.root / "conditional",
                "lineages": {seed: {"seed": seed} for seed in (1, 7, 21)},
                "resume": False,
            },
            SimpleNamespace(),
        )
        self.assertEqual(calls, [1, 7, 21])
        self.assertEqual(tuple(artifacts), CANDIDATE_NAMES)
        self.assertTrue(
            all(
                artifacts["conditional_seed_%d" % seed]["status"]
                == "complete"
                for seed in (1, 7, 21)
            )
        )
        self.assertEqual(
            artifacts["conditional_ensemble"]["status"], "complete"
        )

    def test_default_loaders_limit_after_identity_checks_and_bind_lineage(self):
        train_keys = tuple("train_%02d" % index for index in range(12))
        valid_keys = tuple("valid_%02d" % index for index in range(11))

        class Cache:
            def __init__(self, split, keys):
                self.split = split
                self.keys = keys
                self.labels = np.arange(len(keys), dtype=np.float32) + 3.0

        class Reference:
            def __init__(self, split, keys, directory):
                self.cache = Cache(split, keys)
                self.directory = directory
                self.file_sha256 = {"normalization.json": "9" * 64}
                self.identity = {
                    "split": split,
                    "count": len(keys),
                    "manifest_sha256": ("a" if split == "train" else "b") * 64,
                    "ordered_key_sha256": ("c" if split == "train" else "d") * 64,
                    "files": {"normalization.json": "9" * 64},
                }

        train_directory = self.root / "train_cache"
        valid_directory = self.root / "valid_cache"
        train_directory.mkdir()
        valid_directory.mkdir()
        (train_directory / "normalization.json").write_text(
            '{"schema_version":1}\n', encoding="utf-8"
        )
        references = {
            "train": Reference("train", train_keys, train_directory),
            "valid": Reference("valid", valid_keys, valid_directory),
        }

        class StaticSpec:
            def to_dict(self):
                return {"family": "rank"}

        anchor = SimpleNamespace(
            train_predictions=np.linspace(2.0, 9.0, len(train_keys)),
            valid_predictions=np.linspace(3.0, 8.0, len(valid_keys)),
            stored_valid_predictions=np.linspace(3.0, 8.0, len(valid_keys)),
            static_spec=StaticSpec(),
            selected_name="residual_rank_10_1",
        )

        def dataset_factory(cache, base_predictions):
            return [
                {
                    "key": cache.keys[index],
                    "label": cache.labels[index],
                    "base_prediction": base_predictions[index],
                }
                for index in range(len(cache.keys))
            ]

        assignment = SimpleNamespace(
            by_key={key: index % 5 for index, key in enumerate(train_keys[:10])},
            sha256="e" * 64,
        )
        lineages = []

        def lineage_builder(**kwargs):
            lineages.append(kwargs)
            return {"seed": kwargs["seed"], "fold_map_sha256": kwargs["fold_map_sha256"]}

        args = SimpleNamespace(
            cache_root=str(self.root / "cache_root"),
            metadata_csv=str(self.root / "metadata.csv"),
            v5_validation_root=str(self.v5_package),
            v5_package=str(self.v5_package),
            v5_fallback_result=str(self.v5_result),
            output_root=str(self.root / "output"),
            source_revision="f" * 40,
            device="cpu",
            limit=10,
            resume=False,
            smoke_epochs=2,
        )
        hooks = DefaultValidationHooks()
        with patch.object(
            validation_module,
            "load_validation_caches",
            return_value=references,
            create=True,
        ) as load_caches, patch.object(
            validation_module,
            "reconstruct_v5_anchor",
            return_value=anchor,
            create=True,
        ), patch.object(
            validation_module,
            "RankLockedCalibrationDataset",
            side_effect=dataset_factory,
            create=True,
        ), patch.object(
            validation_module,
            "load_fold_rows",
            return_value=tuple(train_keys[:10]),
            create=True,
        ), patch.object(
            validation_module,
            "assign_grouped_folds",
            return_value=assignment,
            create=True,
        ), patch.object(
            validation_module,
            "build_run_lineage",
            side_effect=lineage_builder,
            create=True,
        ):
            train_context = hooks.load_train(args)
            self.assertEqual(len(train_context["dataset"]), 10)
            self.assertEqual(
                train_context["fold_ids"].tolist(),
                [assignment.by_key[key] for key in train_keys[:10]],
            )
            self.assertEqual(train_context["training_config"].maximum_epochs, 2)
            self.assertEqual(len(lineages), 3)
            self.assertEqual([value["seed"] for value in lineages], [1, 7, 21])
            self.assertTrue(
                all(value["fold_map_sha256"] == "e" * 64 for value in lineages)
            )
            valid_context = hooks.load_valid(train_context, {}, args)
            self.assertEqual(len(valid_context["dataset"]), 10)
            self.assertEqual(tuple(valid_context["keys"]), valid_keys[:10])
        load_caches.assert_called_once()

    def test_complete_conditional_checkpoint_is_separate_from_json_manifest(self):
        from dta.rank_locked_calibration_v6.model import ConditionalCalibrator

        count = 20
        base = np.linspace(2.0, 10.0, count, dtype=np.float64)
        labels = 0.65 * base + 2.25
        dataset = []
        for index in range(count):
            dataset.append(
                {
                    "key": "valid_%02d" % index,
                    "label": labels[index],
                    "base_prediction": base[index],
                    "raw_predictions": np.zeros(3, dtype=np.float32),
                    "standardized_predictions": np.zeros(3, dtype=np.float32),
                    "available": np.ones(3, dtype=np.bool_),
                    "latents": (
                        np.zeros(4, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                        np.zeros(5, dtype=np.float32),
                    ),
                }
            )
        analytic = {
            "affine_mse": PositiveAffineCalibrator.fit(
                base, labels, objective="mse"
            ).to_dict(),
            "affine_huber": PositiveAffineCalibrator.fit(
                base, labels, objective="huber"
            ).to_dict(),
            "spline_8_knots": MonotonicSplineCalibrator.fit(
                base, labels, knot_count=8
            ).to_dict(),
            "spline_16_knots": MonotonicSplineCalibrator.fit(
                base, labels, knot_count=16
            ).to_dict(),
        }
        model = ConditionalCalibrator((4, 3, 5), 0.65, 2.25)
        checkpoint = {
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
        }
        artifacts = {
            "identity": {
                "status": "complete",
                "model_descriptor": {"type": "identity"},
                "oof_summary": {"status": "complete"},
            }
        }
        for name, descriptor in analytic.items():
            artifacts[name] = {
                "status": "complete",
                "model_descriptor": descriptor,
                "oof_summary": {"status": "complete"},
            }
        for seed in (1, 7, 21):
            artifacts["conditional_seed_%d" % seed] = {
                "status": "complete",
                "model_descriptor": checkpoint,
                "oof_summary": {"status": "complete"},
            }
        artifacts["conditional_ensemble"] = {
            "status": "complete",
            "model_descriptor": {"type": "conditional_ensemble"},
            "oof_summary": {"status": "complete"},
        }
        train_context = {
            "cache_identities": {"train": {"count": count}, "valid": {"count": count}},
            "normalization": {"schema_version": 1},
            "fold_map": {
                "keys": np.asarray([str(index) for index in range(count)]),
                "fold_ids": np.arange(count, dtype=np.int64) % 5,
            },
            "rollback": self._rollback(),
        }
        material = DefaultValidationHooks().evaluate(
            train_context,
            artifacts,
            {
                "keys": np.asarray([record["key"] for record in dataset]),
                "labels": labels,
                "base_predictions": base,
                "dataset": dataset,
            },
            SimpleNamespace(device="cpu"),
        )
        self.assertTrue(all(candidate.status == "complete" for candidate in material.candidates))
        for candidate in material.candidates:
            json.dumps(dict(candidate.model_descriptor), allow_nan=False)
        conditional_model = material.selected_models["conditional_seed_1"]["model"]
        self.assertIn("model_state_dict", conditional_model)
        self.assertTrue(
            any(
                hasattr(value, "numel")
                for value in conditional_model["model_state_dict"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
