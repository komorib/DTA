import dataclasses
import hashlib
import inspect
import unittest
import numpy as np
from dta.anchored_expert_fusion_v5.static_fusion import (
    GateResult, SearchResult, _grid_points, _validate_search_result,
    assign_folds, cross_fit_static, deploy_weights,
    evaluate_gate, fold_map_sha256, project_simplex, rank_percentiles,
    report_sha256, search_simplex_weights,
)


class StaticFusionTest(unittest.TestCase):
    @staticmethod
    def keys():
        return tuple("v%02d" % i for i in range(25))

    @classmethod
    def problem(cls):
        keys = cls.keys()
        y_train = np.linspace(2.0, 9.0, 19)
        y_valid = np.linspace(1.0, 10.0, len(keys))
        return dict(
            train_labels=y_train,
            train_predictions=np.column_stack(
                (y_train, y_train[::-1], np.full(19, 5.0))
            ),
            train_available=np.ones((19, 3), dtype=np.bool_),
            validation_keys=keys,
            validation_labels=y_valid,
            validation_predictions=np.column_stack(
                (y_valid, y_valid[::-1], np.full(len(keys), 5.0))
            ),
            validation_available=np.ones((len(keys), 3), dtype=np.bool_),
        )

    @staticmethod
    def result(weights=(1, 0, 0), slope=1, intercept=0):
        return SearchResult(
            np.asarray(weights, dtype=np.float64),
            dict(CI=1.0, Spearman=1.0, RMSE=0.0, MAE=0.0, Pearson=1.0),
            float(slope), float(intercept), (1, 0, 0), 1,
        )

    def test_folds_are_keyed_deterministic_order_independent_and_hashed(self):
        keys = self.keys()
        folds = assign_folds(keys, 42, 5)
        expected = np.asarray([
            int(hashlib.sha256(("42:" + key).encode()).hexdigest()[:16], 16) % 5
            for key in keys
        ], dtype=np.int64)
        np.testing.assert_array_equal(folds, expected)
        np.testing.assert_array_equal(folds, assign_folds(keys))
        np.testing.assert_array_equal(
            folds, assign_folds(tuple(reversed(keys)))[::-1]
        )
        digest = fold_map_sha256(keys, folds)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, fold_map_sha256(keys, folds))
        self.assertNotEqual(digest, fold_map_sha256(keys[::-1], folds[::-1]))

    def test_folds_reject_invalid_keys_counts_and_empty_fold(self):
        cases = (
            (("a", "a", "b", "c", "d"), 5, "unique"),
            ((), 5, "nonempty"),
            (("a", "b", "c", "d", "e"), 1, "fold_count"),
            (("a", "b", "c"), 5, "rows"),
            (("", "a", "b", "c", "d"), 5, "nonempty"),
            (("a", "b", "c", "d", "e"), 5, "empty fold"),
        )
        for keys, count, message in cases:
            with self.subTest(keys=keys):
                with self.assertRaisesRegex(ValueError, message):
                    assign_folds(keys, fold_count=count)

    def test_average_tie_percentiles_and_validation(self):
        actual = rank_percentiles([30, 10, 20, 20, 40])
        np.testing.assert_array_equal(actual, [0.75, 0, 0.375, 0.375, 1])
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(rank_percentiles([7]), [0])
        for values in ([], [[1]], [1, np.nan], [1, np.inf]):
            with self.assertRaises(ValueError):
                rank_percentiles(values)

    def test_simplex_projection_and_deploy_median(self):
        projected = project_simplex([0.8, -0.2, 0.8])
        np.testing.assert_allclose(projected, [0.5, 0, 0.5], atol=1e-15)
        self.assertEqual(projected.dtype, np.float64)
        self.assertAlmostEqual(float(projected.sum()), 1.0, places=15)
        weights = np.asarray([
            [0.55, .30, .15], [.60, .30, .10], [.575, .30, .125],
            [.58, .29, .13], [.57, .31, .12],
        ])
        actual = deploy_weights(weights)
        np.testing.assert_array_equal(
            actual, project_simplex(np.median(weights, axis=0))
        )
        np.testing.assert_allclose(actual, [.575, .30, .125], atol=.05)
        for invalid in ([], [[1]], [np.nan, 1], [np.inf]):
            with self.assertRaises(ValueError):
                project_simplex(invalid)
        for invalid in (weights[:4], np.ones((5, 2)), weights * np.nan):
            with self.assertRaises(ValueError):
                deploy_weights(invalid)

    def test_search_metric_tiebreak_chain_including_lexicographic(self):
        module = __import__(
            "dta.anchored_expert_fusion_v5.static_fusion", fromlist=["x"]
        )
        original = module.affinity_metrics
        labels, predictions = np.arange(4.0), np.eye(4, 3)
        def result(ci, sp, rmse):
            return dict(CI=ci, Spearman=sp, RMSE=rmse, MAE=0., Pearson=0.)
        def run(fn):
            module.affinity_metrics = fn
            try:
                return search_simplex_weights(labels, predictions).weights
            finally:
                module.affinity_metrics = original
        np.testing.assert_array_equal(
            run(lambda _y, p: result(float(p[0]), float(p[1]), 9)), [1, 0, 0]
        )
        np.testing.assert_array_equal(
            run(lambda _y, p: result(.5, float(p[1]), 9)), [0, 1, 0]
        )
        np.testing.assert_array_equal(
            run(lambda _y, p: result(.5, .5, float(p[2]))), [0, 1, 0]
        )
        np.testing.assert_array_equal(
            run(lambda _y, _p: result(.5, .5, 1)), [0, 0, 1]
        )

    def test_search_repeatability_refinement_and_diagnostic_neighborhood(self):
        rng = np.random.RandomState(7)
        x = rng.normal(size=(101, 3))
        target = np.asarray([.577, .298, .125])
        first = search_simplex_weights(x @ target, x)
        second = search_simplex_weights(x @ target, x)
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_allclose(first.weights, target, atol=.001)
        self.assertEqual(first.stage_counts[0], 861)
        self.assertGreater(first.stage_counts[1], 0)
        self.assertGreater(first.stage_counts[2], 0)
        self.assertEqual(first.searched_count, sum(first.stage_counts))
        self.assertLess(first.searched_count, 10000)

    def test_search_rank_affine_is_train_only_and_degenerate_is_deterministic(self):
        train_y = np.asarray([1, 2, 4, 7, 8, 14.], dtype=float)
        kwargs = dict(
            predictions=np.full((5, 3), 9.),
            family="rank",
            train_labels=train_y,
            train_predictions=np.full((6, 3), 3.),
        )
        first = search_simplex_weights(labels=np.arange(5.), **kwargs)
        second = search_simplex_weights(labels=np.arange(5.)[::-1] * 100, **kwargs)
        self.assertEqual((first.slope, first.intercept), (0., train_y.mean()))
        self.assertEqual(
            (first.slope, first.intercept), (second.slope, second.intercept)
        )

    def test_search_validates_family_shapes_and_finiteness(self):
        y, x = np.arange(5.), np.ones((5, 3))
        cases = (
            (dict(labels=y, predictions=x, family="bad"), "family"),
            (dict(labels=y[:-1], predictions=x), "matching"),
            (dict(labels=y, predictions=x[:, :2]), "shape"),
            (dict(labels=y, predictions=x * np.nan), "finite"),
        )
        for kwargs, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                search_simplex_weights(**kwargs)

    def test_cross_fit_search_excludes_held_out_labels(self):
        problem, calls = self.problem(), []
        def spy(**kwargs):
            calls.append(kwargs)
            return self.result()
        report = cross_fit_static(**problem, family="raw", search_fn=spy)
        self.assertEqual(len(calls), 5)
        for fold, call in enumerate(calls):
            mask = report.fold_ids != fold
            np.testing.assert_array_equal(
                call["labels"], problem["validation_labels"][mask]
            )
            self.assertEqual(call["labels"].size, int(mask.sum()))
            self.assertNotIn("held_out_labels", call)
            self.assertNotIn("test_labels", call)

    def test_rank_uses_complete_population_ranks_and_separate_train_labels(self):
        problem, calls = self.problem(), []
        def spy(**kwargs):
            calls.append(kwargs)
            return self.result(slope=2, intercept=-1)
        cross_fit_static(**problem, family="rank", search_fn=spy)
        train_rank = np.column_stack([
            rank_percentiles(problem["train_predictions"][:, i]) for i in range(3)
        ])
        valid_rank = np.column_stack([
            rank_percentiles(problem["validation_predictions"][:, i])
            for i in range(3)
        ])
        folds = assign_folds(problem["validation_keys"])
        for fold, call in enumerate(calls):
            np.testing.assert_array_equal(call["train_labels"], problem["train_labels"])
            np.testing.assert_array_equal(call["train_predictions"], train_rank)
            np.testing.assert_array_equal(
                call["predictions"], valid_rank[folds != fold]
            )
            self.assertTrue(call["features_prepared"])

    def test_held_out_labels_cannot_change_that_folds_rank_calibration(self):
        problem = self.problem()
        a = np.linspace(-2, 2, len(problem["train_labels"]))
        b = np.linspace(-3, 3, len(problem["validation_labels"]))
        problem["train_predictions"] = np.column_stack((a, a, a))
        problem["validation_predictions"] = np.column_stack((b, b, b))
        first = cross_fit_static(**problem, family="rank")
        changed = dict(problem)
        changed["validation_labels"] = problem["validation_labels"].copy()
        changed["validation_labels"][first.fold_ids == 2] += 10000
        second = cross_fit_static(**changed, family="rank")
        self.assertEqual(first.folds[2].slope, second.folds[2].slope)
        self.assertEqual(first.folds[2].intercept, second.folds[2].intercept)

    def test_cross_fit_order_coverage_fallback_availability_and_raw_upper_bound(self):
        problem = self.problem()
        problem["validation_available"][1, 1] = False
        problem["validation_available"][2, 2] = False
        problem["validation_predictions"][1, 1] = -999
        problem["validation_predictions"][2, 2] = -888
        report = cross_fit_static(
            **problem, family="raw",
            search_fn=lambda **_kw: self.result((0, .5, .5))
        )
        fallback = problem["validation_predictions"].copy()
        fallback[1, 1], fallback[2, 2] = fallback[1, 0], fallback[2, 0]
        np.testing.assert_allclose(report.predictions, fallback @ [0, .5, .5])
        np.testing.assert_array_equal(
            report.held_out_count, np.ones(len(report.fold_ids), dtype=np.int64)
        )
        self.assertEqual(tuple(report.validation_keys), problem["validation_keys"])
        self.assertTrue(report.coverage_complete)
        np.testing.assert_array_equal(
            report.validation_available, problem["validation_available"]
        )
        expected_rmse = np.sqrt(np.mean(np.square(
            problem["validation_labels"] - problem["validation_predictions"][:, 0]
        )))
        self.assertAlmostEqual(
            report.upper_bound_pooled_metrics["RMSE"], expected_rmse
        )

    def test_raw_rank_are_separate_and_unknown_family_rejected(self):
        p = self.problem()
        raw = cross_fit_static(
            **p, family="raw",
            search_fn=lambda **_kw: self.result((.5, .5, 0))
        )
        rank = cross_fit_static(
            **p, family="rank",
            search_fn=lambda **_kw: self.result((.5, .5, 0), 3, 2)
        )
        self.assertFalse(np.array_equal(raw.predictions, rank.predictions))
        with self.assertRaisesRegex(ValueError, "family"):
            cross_fit_static(**p, family="unknown")

    def test_gate_thresholds_and_all_stable_reasons(self):
        folds = tuple({"CI": .6} for _ in range(5))
        upper_folds = tuple({"CI": .5} for _ in range(5))
        defaults = dict(
            pooled_metrics={"CI": .705},
            upper_bound_pooled_metrics={"CI": .700},
            fold_metrics=folds,
            upper_bound_fold_metrics=upper_folds,
            fold_weights=np.tile([.6, .3, .1], (5, 1)),
            held_out_count=np.ones(25, dtype=np.int64),
        )
        self.assertEqual(
            evaluate_gate(**defaults), GateResult(True, ())
        )
        failures = (
            ({"pooled_metrics": {"CI": .704999}}, "pooled_ci_gain_below_0.005"),
            ({"fold_metrics": ({"CI": .4}, {"CI": .4}) + folds[2:]},
             "fewer_than_4_of_5_folds_improved"),
            ({"fold_weights": np.asarray([
                [.4, .5, .1], [.61, .29, .1], [.6, .3, .1],
                [.6, .3, .1], [.6, .3, .1],
            ])}, "fold_weight_range_exceeds_0.20"),
            ({"held_out_count": np.asarray([1, 1, 0, 2])},
             "held_out_coverage_not_exactly_once"),
        )
        for override, reason in failures:
            args = dict(defaults)
            args.update(override)
            self.assertEqual(evaluate_gate(**args).reasons, (reason,))
        args = dict(defaults)
        for override, _ in failures:
            args.update(override)
        self.assertEqual(evaluate_gate(**args).reasons, tuple(x[1] for x in failures))

    def test_report_hash_repeats_and_api_has_no_test_inputs(self):
        p = self.problem()
        fn = lambda **_kw: self.result((.575, .300, .125))
        first = cross_fit_static(**p, family="raw", search_fn=fn)
        second = cross_fit_static(**p, family="raw", search_fn=fn)
        self.assertEqual(first.fold_map_hash, second.fold_map_hash)
        self.assertEqual(first.report_hash, second.report_hash)
        self.assertEqual(report_sha256(first), report_sha256(second))
        for function in (
            assign_folds, rank_percentiles, project_simplex,
            search_simplex_weights, cross_fit_static, deploy_weights,
        ):
            names = inspect.signature(function).parameters
            self.assertFalse(any("test" in name.lower() for name in names))
            self.assertFalse(any("cache" in name.lower() for name in names))

    def test_gate_contractual_float_boundaries_have_no_relaxation(self):
        folds = tuple({"CI": 1.0} for _ in range(5))
        upper_folds = tuple({"CI": 0.0} for _ in range(5))
        coverage = np.ones(5, dtype=np.int64)
        stable = np.asarray([
            [0.0, 0.5, 0.5],
            [0.2, 0.4, 0.4],
            [0.1, 0.45, 0.45],
            [0.1, 0.45, 0.45],
            [0.1, 0.45, 0.45],
        ])
        def gate(gain, candidate_folds=folds, weights=stable):
            return evaluate_gate(
                {"CI": gain}, {"CI": 0.0},
                candidate_folds, upper_folds, weights, coverage,
            )
        self.assertNotIn(
            "pooled_ci_gain_below_0.005", gate(0.005).reasons
        )
        below = np.nextafter(0.005, -np.inf)
        self.assertIn(
            "pooled_ci_gain_below_0.005", gate(below).reasons
        )

        equal_folds = ({"CI": 0.0}, {"CI": 0.0}) + folds[2:]
        self.assertIn(
            "fewer_than_4_of_5_folds_improved",
            gate(0.005, equal_folds).reasons,
        )
        next_folds = (
            {"CI": np.nextafter(0.0, np.inf)},
            {"CI": 0.0},
        ) + folds[2:]
        self.assertNotIn(
            "fewer_than_4_of_5_folds_improved",
            gate(0.005, next_folds).reasons,
        )

        self.assertNotIn(
            "fold_weight_range_exceeds_0.20",
            gate(0.005, weights=stable).reasons,
        )
        unstable = stable.copy()
        unstable[1, 0] = np.nextafter(0.20, np.inf)
        self.assertIn(
            "fold_weight_range_exceeds_0.20",
            gate(0.005, weights=unstable).reasons,
        )

    def test_callback_result_validation_rejects_every_malformed_category(self):
        valid = self.result()
        _validate_search_result(
            dataclasses.replace(
                valid, stage_counts=(np.int64(2), 3), searched_count=np.int64(5)
            )
        )
        malformed = [
            object(),
            dataclasses.replace(
                valid,
                metrics={
                    "CI": 1.0, "Spearman": 1.0, "RMSE": 0.0,
                    "MAE": 0.0,
                },
            ),
            dataclasses.replace(
                valid,
                metrics=dict(valid.metrics, CI=np.nan),
            ),
            dataclasses.replace(
                valid, weights=np.asarray([0.5, 0.5, 0.5])
            ),
            dataclasses.replace(valid, slope=np.inf),
            dataclasses.replace(valid, intercept=np.nan),
            dataclasses.replace(valid, stage_counts=()),
            dataclasses.replace(valid, stage_counts=None),
            dataclasses.replace(valid, stage_counts=(True,)),
            dataclasses.replace(valid, stage_counts=(-1,)),
            dataclasses.replace(valid, stage_counts=(1.5,)),
            dataclasses.replace(valid, searched_count=True),
            dataclasses.replace(valid, searched_count=-1),
            dataclasses.replace(valid, searched_count=2),
        ]
        problem = self.problem()
        for index, result in enumerate(malformed):
            with self.subTest(category=index):
                with self.assertRaisesRegex(ValueError, "search_fn"):
                    cross_fit_static(
                        **problem,
                        family="raw",
                        search_fn=lambda **_kwargs: result,
                    )

    def test_grid_points_are_exact_complete_sets_without_duplicates(self):
        def expected(step, center=None, radius=None):
            points = set()
            for first in range(0, 1001, step):
                for second in range(0, 1001 - first, step):
                    point = (first, second, 1000 - first - second)
                    if center is None or max(
                        abs(point[i] - center[i]) for i in range(3)
                    ) <= radius:
                        points.add(point)
            return points

        coarse_list = list(_grid_points(25))
        self.assertEqual(set(coarse_list), expected(25))
        self.assertEqual(len(coarse_list), len(set(coarse_list)))
        self.assertEqual(len(coarse_list), 861)

        for step, radius in ((5, 50), (1, 10)):
            for center in (
                (575, 300, 125),
                (1000, 0, 0),
                (600, 400, 0),
            ):
                with self.subTest(step=step, center=center):
                    actual_list = list(_grid_points(step, center, radius))
                    self.assertEqual(
                        set(actual_list), expected(step, center, radius)
                    )
                    self.assertEqual(
                        len(actual_list), len(set(actual_list))
                    )

    def test_search_stage_counts_match_exact_deduplicated_grid_sets(self):
        module = __import__(
            "dta.anchored_expert_fusion_v5.static_fusion", fromlist=["x"]
        )
        original = module.affinity_metrics
        target = np.asarray([0.577, 0.298, 0.125])
        def fixed_order(labels, predictions):
            difference = labels - predictions
            return {
                "CI": 0.5,
                "Spearman": 0.5,
                "RMSE": float(np.linalg.norm(difference)),
                "MAE": float(np.abs(difference).mean()),
                "Pearson": 0.0,
            }
        module.affinity_metrics = fixed_order
        try:
            result = search_simplex_weights(target, np.eye(3))
        finally:
            module.affinity_metrics = original

        coarse = set(_grid_points(25))
        coarse_winner = (575, 300, 125)
        fine = set(_grid_points(5, coarse_winner, 50))
        fine_winner = coarse_winner
        finest = set(_grid_points(1, fine_winner, 10))
        expected_counts = (
            len(coarse),
            len(fine - coarse),
            len(finest - coarse - fine),
        )
        self.assertEqual(result.stage_counts, expected_counts)
        self.assertEqual(result.searched_count, sum(expected_counts))
        np.testing.assert_array_equal(result.weights, target)

    def test_rank_affine_is_recomputed_per_candidate_and_selected_fit_is_train_only(self):
        module = __import__(
            "dta.anchored_expert_fusion_v5.static_fusion", fromlist=["x"]
        )
        original = module._fit_affine
        calls = []
        def recording_fit(labels, blend):
            calls.append((labels.copy(), blend.copy()))
            return original(labels, blend)
        train_predictions = np.column_stack((
            np.linspace(-2.0, 3.0, 11),
            np.asarray([0, 3, 1, 7, 4, 8, 2, 10, 5, 9, 6.], dtype=float),
            np.linspace(5.0, -4.0, 11),
        ))
        train_labels = np.asarray(
            [1, 4, 2, 8, 3, 10, 5, 11, 7, 9, 6.], dtype=float
        )
        valid_predictions = np.column_stack((
            np.linspace(-1.0, 2.0, 9),
            np.asarray([2, 0, 5, 1, 8, 4, 7, 3, 6.], dtype=float),
            np.linspace(4.0, -3.0, 9),
        ))
        valid_labels = np.asarray(
            [2, 1, 6, 3, 9, 5, 8, 4, 7.], dtype=float
        )
        module._fit_affine = recording_fit
        try:
            result = search_simplex_weights(
                valid_labels,
                valid_predictions,
                family="rank",
                train_labels=train_labels,
                train_predictions=train_predictions,
            )
        finally:
            module._fit_affine = original

        self.assertEqual(len(calls), result.searched_count)
        self.assertGreater(
            len({tuple(np.round(blend, 14)) for _labels, blend in calls}), 1
        )
        train_rank = np.column_stack([
            rank_percentiles(train_predictions[:, i]) for i in range(3)
        ])
        expected = original(
            train_labels, train_rank @ result.weights
        )
        self.assertEqual((result.slope, result.intercept), expected)
        for labels, _blend in calls:
            np.testing.assert_array_equal(labels, train_labels)

    def test_negative_train_affine_controls_rank_scoring_direction(self):
        ascending_train = np.arange(9.0)
        descending_train = np.arange(9.0, 0.0, -1.0)
        ascending_valid = np.arange(7.0)
        descending_valid = np.linspace(9.0, 1.0, 7)
        result = search_simplex_weights(
            descending_valid,
            np.column_stack(
                (ascending_valid, ascending_valid, ascending_valid)
            ),
            family="rank",
            train_labels=descending_train,
            train_predictions=np.column_stack((
                ascending_train, ascending_train, ascending_train
            )),
        )
        self.assertLess(result.slope, 0.0)
        self.assertEqual(result.metrics["CI"], 1.0)
        self.assertAlmostEqual(result.metrics["Spearman"], 1.0, places=15)
        self.assertAlmostEqual(result.metrics["RMSE"], 0.0, places=12)

    def test_cross_fit_rank_applies_custom_calibration_to_held_out_rows(self):
        problem = self.problem()
        weights = np.asarray([0.2, 0.3, 0.5])
        slope, intercept = 2.75, -0.4
        report = cross_fit_static(
            **problem,
            family="rank",
            search_fn=lambda **_kwargs: self.result(
                weights, slope, intercept
            ),
        )
        valid_rank = np.column_stack([
            rank_percentiles(problem["validation_predictions"][:, i])
            for i in range(3)
        ])
        expected = slope * (valid_rank @ weights) + intercept
        np.testing.assert_allclose(report.predictions, expected, atol=0.0)

    def test_report_hash_is_sensitive_to_every_selection_category(self):
        problem = self.problem()
        report = cross_fit_static(
            **problem,
            family="raw",
            search_fn=lambda **_kwargs: self.result((0.6, 0.3, 0.1)),
        )
        original_hash = report_sha256(report)
        def assert_changed(changed):
            self.assertNotEqual(original_hash, report_sha256(changed))

        weights = report.fold_weights.copy()
        weights[0, 0] = np.nextafter(weights[0, 0], np.inf)
        assert_changed(dataclasses.replace(report, fold_weights=weights))

        deployed = report.deploy_weights.copy()
        deployed[0] = np.nextafter(deployed[0], np.inf)
        assert_changed(dataclasses.replace(report, deploy_weights=deployed))

        fold = dataclasses.replace(
            report.folds[0],
            slope=np.nextafter(report.folds[0].slope, np.inf),
        )
        assert_changed(
            dataclasses.replace(report, folds=(fold,) + report.folds[1:])
        )

        fold = dataclasses.replace(
            report.folds[0],
            search_metrics=dict(report.folds[0].search_metrics, CI=0.25),
        )
        assert_changed(
            dataclasses.replace(report, folds=(fold,) + report.folds[1:])
        )

        fold = dataclasses.replace(
            report.folds[0],
            stage_counts=(2, 3),
            searched_count=5,
        )
        assert_changed(
            dataclasses.replace(report, folds=(fold,) + report.folds[1:])
        )

        fold = dataclasses.replace(
            report.folds[0],
            candidate_metrics=dict(
                report.folds[0].candidate_metrics, CI=0.25
            ),
        )
        assert_changed(
            dataclasses.replace(report, folds=(fold,) + report.folds[1:])
        )

        availability = report.validation_available.copy()
        availability[0, 1] = not availability[0, 1]
        assert_changed(
            dataclasses.replace(report, validation_available=availability)
        )

        counts = report.held_out_count.copy()
        counts[0] = 2
        assert_changed(dataclasses.replace(report, held_out_count=counts))

        gate = dataclasses.replace(
            report.gate,
            passed=not report.gate.passed,
            reasons=report.gate.reasons + ("changed",),
        )
        assert_changed(dataclasses.replace(report, gate=gate))

    def test_gate_rejects_nonfinite_pooled_and_fold_metrics(self):
        folds = tuple({"CI": 0.6} for _ in range(5))
        upper_folds = tuple({"CI": 0.5} for _ in range(5))
        defaults = dict(
            pooled_metrics={"CI": 0.705},
            upper_bound_pooled_metrics={"CI": 0.700},
            fold_metrics=folds,
            upper_bound_fold_metrics=upper_folds,
            fold_weights=np.tile([0.6, 0.3, 0.1], (5, 1)),
            held_out_count=np.ones(5, dtype=np.int64),
        )
        for field in ("pooled_metrics", "upper_bound_pooled_metrics"):
            for value in (np.nan, np.inf, -np.inf, True, "0.7"):
                with self.subTest(field=field, value=value):
                    arguments = dict(defaults)
                    arguments[field] = {"CI": value}
                    result = evaluate_gate(**arguments)
                    self.assertFalse(result.passed)
                    self.assertIn("nonfinite_gate_metrics", result.reasons)

        for field in ("fold_metrics", "upper_bound_fold_metrics"):
            for fold in range(5):
                for value in (np.nan, np.inf, -np.inf, True, "0.5"):
                    with self.subTest(field=field, fold=fold, value=value):
                        candidates = [dict(metric) for metric in folds]
                        upper = [dict(metric) for metric in upper_folds]
                        target = candidates if field == "fold_metrics" else upper
                        target[fold]["CI"] = value
                        arguments = dict(defaults)
                        arguments["fold_metrics"] = tuple(candidates)
                        arguments["upper_bound_fold_metrics"] = tuple(upper)
                        result = evaluate_gate(**arguments)
                        self.assertFalse(result.passed)
                        self.assertIn("nonfinite_gate_metrics", result.reasons)

    def test_callback_calibration_requires_finite_real_nonbool_values(self):
        problem = self.problem()
        cases = (
            {"slope": True}, {"slope": "1.0"}, {"slope": object()},
            {"intercept": False}, {"intercept": "0.0"},
            {"intercept": object()},
        )
        for override in cases:
            with self.subTest(override=override):
                malformed = dataclasses.replace(self.result(), **override)
                with self.assertRaisesRegex(
                    ValueError, "^search_fn returned invalid calibration$"
                ):
                    cross_fit_static(
                        **problem,
                        family="raw",
                        search_fn=lambda **_kwargs: malformed,
                    )

    def test_cross_fit_asserts_stored_report_hash_matches_recomputation(self):
        module = __import__(
            "dta.anchored_expert_fusion_v5.static_fusion", fromlist=["x"]
        )
        original = module.report_sha256
        calls = []
        def changing_hash(_report):
            calls.append(None)
            return "digest-%d" % len(calls)
        module.report_sha256 = changing_hash
        try:
            with self.assertRaises(AssertionError):
                cross_fit_static(
                    **self.problem(),
                    family="raw",
                    search_fn=lambda **_kwargs: self.result(),
                )
        finally:
            module.report_sha256 = original
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
