import hashlib
import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dta.anchored_expert_fusion_v5.data import ImmutableCacheRef
from dta.anchored_expert_fusion_v5.selection import (
    CandidateRecord,
    build_candidate_inventory,
    build_residual_ensembles,
    seal_validation_package,
    select_validation_candidate,
    validation_candidate_key,
    verify_validation_seal,
)
from dta.rank_fusion_v4.cache import FusionCache, _ordered_key_sha256
from dta.anchored_expert_fusion_v5.static_fusion import assign_folds


FORMAL_BUDGETS = (0.10, 0.25, 0.50)
FORMAL_SEEDS = (1, 7, 21, 42, 87)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed_fold_map(reports):
    from dta.anchored_expert_fusion_v5 import evaluate_test

    source = reports["raw"]
    fold_ids = np.asarray(source["fold_ids"], dtype=np.int64)
    fold_ids.setflags(write=False)
    return evaluate_test.SealedFoldMap(
        keys=tuple(source["validation_keys"]),
        fold_ids=fold_ids,
        sha256=source["fold_map_hash"],
    )


def metrics(ci=0.70, spearman=0.70, rmse=1.0):
    return {
        "CI": float(ci),
        "MAE": 0.5,
        "RMSE": float(rmse),
        "Pearson": 0.6,
        "Spearman": float(spearman),
    }


def fake_residual_training_result(output, valid_ref, spec, budget, seed, prediction):
    from dta.anchored_expert_fusion_v5 import training as training_module
    from dta.rank_fusion_v4.metrics import affinity_metrics

    prediction = np.asarray(prediction, dtype=np.float32)
    prediction_path = output / "predictions_best.npz"
    checkpoint_path = output / "checkpoint_best.pt"
    np.savez_compressed(
        prediction_path,
        keys=np.asarray(valid_ref.cache.keys),
        labels=np.asarray(valid_ref.cache.labels, dtype=np.float32),
        predictions=prediction,
        epoch=np.asarray(0, dtype=np.int64),
        variant=np.asarray("best"),
    )
    raw_metrics = affinity_metrics(
        np.asarray(valid_ref.cache.labels, dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
    )
    best_metrics = {key: float(raw_metrics[key]) for key in
                    ("CI", "MAE", "RMSE", "Pearson", "Spearman")}
    run_config = {"fixture": True, "family": spec.family,
                  "budget": float(budget), "seed": int(seed)}
    spec_payload = spec.to_dict()
    checkpoint_payload = {
        "epoch": 0,
        "metrics": best_metrics,
        "run_config": run_config,
        "budget": float(budget),
        "seed": int(seed),
        "static_spec_sha256": training_module.static_spec_sha256(spec),
        "cache_lineage": {
            "train": spec_payload["train_cache_identity"],
            "valid": spec_payload["valid_cache_identity"],
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    return {
        "status": "complete",
        "best_epoch": 0,
        "best_metrics": best_metrics,
        "run_config": run_config,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256(prediction_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def lineage(family="raw"):
    if family == "expert":
        return {
            "valid_manifest": HASH_A,
            "valid_ordered_keys": HASH_B,
            "prediction_source": HASH_C,
        }
    return {
        "train_manifest": HASH_A,
        "train_ordered_keys": HASH_B,
        "valid_manifest": HASH_A,
        "valid_ordered_keys": HASH_B,
        "cross_fit_report": HASH_C if family == "raw" else HASH_B,
        "static_spec_sha256": HASH_A,
    }


def candidate(name, kind, family="raw", ci=0.70, spearman=0.70,
              rmse=1.0, parent=None, budget=None, seed=None,
              predictions=None, members=(), member_hashes=None):
    checkpoint_path = None
    checkpoint_hash = None
    if kind == "residual":
        checkpoint_path = "residual_runs/%s/checkpoint_best.pt" % name
        checkpoint_hash = HASH_B
    record_lineage = lineage(family)
    if kind in ("raw_ensemble", "rank_ensemble"):
        member_hashes = member_hashes or {
            member: HASH_A for member in members
        }
        member_payload = json.dumps(
            member_hashes, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        record_lineage["member_set"] = hashlib.sha256(
            member_payload
        ).hexdigest()
    return CandidateRecord(
        name=name,
        type=kind,
        family=family,
        prediction_path="candidate_predictions/%s.npz" % name,
        prediction_sha256=HASH_A,
        validation_metrics=metrics(ci, spearman, rmse),
        parent_static_name=parent,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        lineage_hashes=record_lineage,
        member_hashes=member_hashes or {},
        budget=budget,
        seed=seed,
        members=members,
        ensemble_rule=(
            "arithmetic_mean" if kind == "raw_ensemble"
            else "stable_average_percentile_mean"
            if kind == "rank_ensemble" else None
        ),
        predictions=predictions,
    )


def residual_grid(families=("raw",), budgets=FORMAL_BUDGETS,
                  seeds=FORMAL_SEEDS):
    values = []
    base = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0])
    for family_index, family in enumerate(families):
        for budget_index, budget in enumerate(budgets):
            for seed_index, seed in enumerate(seeds):
                name = "residual_%s_%02d_%d" % (
                    family, int(round(budget * 100)), seed
                )
                values.append(candidate(
                    name,
                    "residual",
                    family=family,
                    ci=.75 - seed_index * .001,
                    parent="static_" + family,
                    budget=budget,
                    seed=seed,
                    predictions=base + family_index + budget_index / 10.0
                    + seed_index / 100.0,
                ))
    return values


def expert_candidates():
    return [
        candidate("upper_bound", "expert", family="expert", ci=.68),
        candidate("boltz_pairwise", "expert", family="expert", ci=.67),
        candidate("three_dprot", "expert", family="expert", ci=.66),
    ]


def static_candidates(families):
    return [
        candidate("static_" + family, "static", family=family, ci=.71)
        for family in families
    ]


def identity(split, count, marker, ordered_keys=None):
    digest = hashlib.sha256(marker.encode("ascii")).hexdigest()
    ordered_key_sha256 = (
        _ordered_key_sha256(ordered_keys)
        if ordered_keys is not None else hashlib.sha256(
            (marker + "-keys").encode("ascii")
        ).hexdigest()
    )
    return {
        "split": split,
        "count": count,
        "manifest_sha256": digest,
        "ordered_key_sha256": ordered_key_sha256,
        "files": {
            name: hashlib.sha256(
                (marker + "-" + name).encode("ascii")
            ).hexdigest()
            for name in (
                "features.npz", "metadata.json", "coverage.json",
                "normalization.json",
            )
        },
    }


def cache_ref(split, count=25):
    prefix = "v" if split == "valid" else "t"
    keys = tuple(prefix + "%02d" % index for index in range(count))
    labels = np.linspace(1.0, 5.0, count, dtype=np.float32)
    predictions = np.column_stack((
        labels + .15,
        labels[::-1] + .3,
        labels * .8 + .5,
    )).astype(np.float32)
    available = np.ones((count, 3), dtype=np.bool_)
    cache = FusionCache(
        keys=keys,
        labels=labels,
        predictions=predictions,
        standardized_predictions=(predictions / 5.0).astype(np.float32),
        available=available,
        upper_bound_latents=np.ones((count, 2), dtype=np.float32),
        boltz_latents=np.ones((count, 3), dtype=np.float32),
        three_dprot_latents=np.ones((count, 4), dtype=np.float32),
        metadata={
            key: value for key, value in identity(
                split, count, split, ordered_keys=keys
            ).items() if key != "files"
        },
    )
    return ImmutableCacheRef(
        cache=cache,
        identity=identity(split, count, split, ordered_keys=keys),
    )


def report(family, valid_ref, passed=True):
    from dta.anchored_expert_fusion_v5 import static_fusion

    keys = tuple(valid_ref.cache.keys)
    fold_ids = assign_folds(keys)
    held_out = np.ones(len(keys), dtype=np.int64)
    weights = np.tile(np.asarray([.6, .3, .1]), (5, 1))
    candidate_metrics = metrics(.72 if passed else .70)
    upper_metrics = metrics(.70)
    slope = 1.0 if family == "raw" else 2.0
    intercept = 0.0 if family == "raw" else -.5
    folds = tuple(
        static_fusion.FoldResult(
            fold_id=index,
            held_out_indices=np.flatnonzero(fold_ids == index),
            selected_weights=weights[index].copy(),
            candidate_metrics=dict(candidate_metrics),
            upper_bound_metrics=dict(upper_metrics),
            search_metrics=dict(candidate_metrics),
            slope=slope,
            intercept=intercept,
            stage_counts=(1, 1, 1),
            searched_count=3,
        )
        for index in range(5)
    )
    gate = static_fusion.evaluate_gate(
        candidate_metrics,
        upper_metrics,
        [fold.candidate_metrics for fold in folds],
        [fold.upper_bound_metrics for fold in folds],
        weights,
        held_out,
    )
    if gate.passed != passed:
        raise AssertionError("canonical report fixture gate is inconsistent")
    value = static_fusion.CrossFitReport(
        family=family,
        validation_keys=keys,
        fold_ids=fold_ids,
        fold_map_hash=static_fusion.fold_map_sha256(keys, fold_ids),
        held_out_count=held_out,
        predictions=(
            valid_ref.cache.labels.astype(np.float64)
            + (.02 if family == "raw" else .03)
        ),
        pooled_metrics=dict(candidate_metrics),
        upper_bound_pooled_metrics=dict(upper_metrics),
        folds=folds,
        fold_weights=weights,
        deploy_weights=static_fusion.deploy_weights(weights),
        gate=gate,
        validation_available=valid_ref.cache.available.copy(),
    )
    return replace(
        value,
        report_hash=static_fusion.report_sha256(value),
    )


class CandidatePolicyTest(unittest.TestCase):
    def test_metric_order_and_equal_ci_residual_falls_back_to_static(self):
        static = candidate("static_raw", "static", ci=.71, spearman=.8,
                           rmse=.8)
        residual = candidate(
            "residual_raw_025_1", "residual", ci=.71, spearman=.99,
            rmse=.1, parent="static_raw", budget=.25, seed=1,
        )
        selected = select_validation_candidate(
            [], [static], [residual], []
        )
        self.assertEqual(selected.name, "static_raw")
        ordered = sorted([
            candidate("z", "expert", family="expert", ci=.7,
                      spearman=.8, rmse=.4),
            candidate("b", "expert", family="expert", ci=.7,
                      spearman=.8, rmse=.3),
            candidate("a", "expert", family="expert", ci=.7,
                      spearman=.8, rmse=.3),
            candidate("c", "expert", family="expert", ci=.7,
                      spearman=.9, rmse=9),
            candidate("d", "expert", family="expert", ci=.8,
                      spearman=.1, rmse=9),
        ], key=validation_candidate_key)
        self.assertEqual([item.name for item in ordered],
                         ["d", "c", "a", "b", "z"])

    def test_both_ensemble_types_require_strict_parent_ci(self):
        static = candidate("static_raw", "static", ci=.71)
        members = ("a", "b", "c")
        hashes = {name: HASH_A for name in members}
        raw = candidate(
            "ensemble_raw_025_raw", "raw_ensemble", ci=.71,
            parent="static_raw", budget=.25, members=members,
            member_hashes=hashes,
        )
        rank = candidate(
            "ensemble_raw_025_rank", "rank_ensemble", ci=.711,
            parent="static_raw", budget=.25, members=members,
            member_hashes=hashes,
        )
        selected = select_validation_candidate([], [static], [], [raw, rank])
        self.assertEqual(selected.name, rank.name)
        equal_rank = candidate(
            "ensemble_raw_025_rank_equal", "rank_ensemble", ci=.71,
            parent="static_raw", budget=.25, members=members,
            member_hashes=hashes,
        )
        selected = select_validation_candidate(
            [], [static], [], [raw, equal_rank]
        )
        self.assertEqual(selected.name, static.name)

    def test_candidate_record_is_type_aware_strict_and_immutable(self):
        valid = candidate("upper_bound", "expert", family="expert")
        with self.assertRaises(Exception):
            valid.validation_metrics["CI"] = 0.0
        cases = [
            dict(prediction_path=None),
            dict(prediction_sha256=None),
            dict(lineage_hashes={}),
            dict(type="residual", family="raw", parent_static_name=None,
                 budget=.25, seed=1, checkpoint_path="x.pt",
                 checkpoint_sha256=HASH_A),
            dict(type="residual", family="raw",
                 parent_static_name="static_raw", budget=.25, seed=1,
                 checkpoint_path=None, checkpoint_sha256=None),
            dict(type="static", family="raw",
                 parent_static_name="static_raw"),
            dict(type="static", family="raw",
                 checkpoint_path="x.pt", checkpoint_sha256=HASH_A),
            dict(type="raw_ensemble", family="raw",
                 parent_static_name="static_raw", budget=.25,
                 members=("a", "b", "c"), member_hashes={}),
        ]
        payload = valid.to_dict()
        for change in cases:
            with self.subTest(change=change):
                changed = dict(payload)
                changed.update(change)
                with self.assertRaises(ValueError):
                    CandidateRecord.from_dict(changed)
        changed = dict(payload)
        changed["validation_metrics"] = dict(
            changed["validation_metrics"], CI=float("nan")
        )
        with self.assertRaises(ValueError):
            CandidateRecord.from_dict(changed)
        static_payload = candidate(
            "static_raw", "static", family="raw"
        ).to_dict()
        static_payload["lineage_hashes"].pop("static_spec_sha256")
        with self.assertRaisesRegex(ValueError, "static_spec_sha256"):
            CandidateRecord.from_dict(static_payload)

    def test_candidate_record_rejects_all_unsafe_path_forms(self):
        unsafe = (
            "", "/absolute/a.npz", "../a.npz", "a/../../b.npz",
            r"C:\temp\a.npz", r"\\server\share\a.npz",
            "//server/share/a.npz", "file://host/a.npz",
            "https://host/a.npz",
        )
        payload = candidate("upper_bound", "expert",
                            family="expert").to_dict()
        for value in unsafe:
            with self.subTest(value=value):
                changed = dict(payload, prediction_path=value)
                with self.assertRaises(ValueError):
                    CandidateRecord.from_dict(changed)


class InventoryTest(unittest.TestCase):
    def _build(self, root, families, seeds=FORMAL_SEEDS):
        residuals = residual_grid(families=families, seeds=seeds)
        ensembles = build_residual_ensembles(
            residuals,
            labels=np.asarray([0., 1., 2., 3., 5.]),
            validation_keys=("a", "b", "c", "d", "e"),
            output_root=root,
            expected_families=families,
            expected_budgets=FORMAL_BUDGETS,
            expected_seeds=seeds,
        )
        inventory = build_candidate_inventory(
            expert_candidates(),
            static_candidates(families),
            residuals,
            ensembles,
            expected_families=families,
            expected_budgets=FORMAL_BUDGETS,
            expected_seeds=seeds,
        )
        return residuals, ensembles, inventory

    def test_exact_formal_inventory_counts_and_combinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for families, expected in ((("raw", "rank"), 47),
                                       (("raw",), 25)):
                with self.subTest(families=families):
                    residuals, ensembles, inventory = self._build(
                        root / ("-".join(families)), families
                    )
                    self.assertEqual(len(inventory), expected)
                    self.assertEqual(
                        len(residuals), len(families) * 3 * 5
                    )
                    self.assertEqual(
                        len(ensembles), len(families) * 3 * 2
                    )
                    combinations = {
                        (item.family, item.budget, item.seed)
                        for item in residuals
                    }
                    self.assertEqual(combinations, {
                        (family, budget, seed)
                        for family in families
                        for budget in FORMAL_BUDGETS
                        for seed in FORMAL_SEEDS
                    })
                    for item in ensembles:
                        path = root / ("-".join(families)) / item.prediction_path
                        self.assertTrue(path.is_file())
                        self.assertEqual(item.prediction_sha256, sha256(path))
                        self.assertEqual(len(item.members), 3)
                        self.assertEqual(
                            set(item.member_hashes), set(item.members)
                        )

    def test_ensemble_builder_rejects_missing_duplicate_and_extraneous_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = residual_grid(
                families=("raw",), budgets=(.10,), seeds=(1, 7, 21)
            )
            kwargs = dict(
                labels=np.asarray([0., 1., 2., 3., 5.]),
                validation_keys=("a", "b", "c", "d", "e"),
                output_root=root,
                expected_families=("raw",),
                expected_budgets=(.10,),
                expected_seeds=(1, 7, 21),
            )
            malformed = {
                "missing": base[:-1],
                "duplicate": base + [base[0]],
                "family": base + [
                    candidate(
                        "extra_family", "residual", family="rank",
                        parent="static_rank", budget=.10, seed=1,
                        predictions=np.arange(5.0),
                    )
                ],
                "budget": base + [
                    candidate(
                        "extra_budget", "residual", family="raw",
                        parent="static_raw", budget=.25, seed=42,
                        predictions=np.arange(5.0),
                    )
                ],
                "seed": base + [
                    candidate(
                        "extra_seed", "residual", family="raw",
                        parent="static_raw", budget=.10, seed=42,
                        predictions=np.arange(5.0),
                    )
                ],
            }
            for label, values in malformed.items():
                with self.subTest(label=label):
                    with self.assertRaises(ValueError):
                        build_residual_ensembles(values, **kwargs)

    def test_ensembles_use_best_three_raw_and_stable_percentile_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            residuals = residual_grid(
                families=("raw",), budgets=(.25,),
                seeds=(1, 7, 21, 42)
            )
            ensembles = build_residual_ensembles(
                residuals,
                labels=np.asarray([0., 1., 2., 3., 5.]),
                validation_keys=("a", "b", "c", "d", "e"),
                output_root=temporary,
                expected_families=("raw",),
                expected_budgets=(.25,),
                expected_seeds=(1, 7, 21, 42),
            )
            raw, rank = ensembles
            self.assertEqual(raw.members, tuple(
                item.name for item in residuals[:3]
            ))
            np.testing.assert_allclose(
                raw.predictions,
                np.mean([item.predictions for item in residuals[:3]], axis=0),
            )
            self.assertEqual(rank.predictions[0], 0.0)
            self.assertEqual(rank.predictions[-1], 1.0)

    def test_ensemble_artifacts_use_persisted_float32_member_values(self):
        from dta.anchored_expert_fusion_v5.selection import rank_percentiles

        quantized = np.asarray(
            [7.7012854, 7.7012854, 6.609538, 5.652175, 7.0566683],
            dtype=np.float32,
        )
        in_memory = quantized.astype(np.float64)
        in_memory[1] += 1e-8
        residuals = [
            candidate(
                "residual_raw_10_%d" % seed,
                "residual",
                family="raw",
                ci=.80 - index * .01,
                parent="static_raw",
                budget=.10,
                seed=seed,
                predictions=in_memory.copy(),
            )
            for index, seed in enumerate((1, 7, 21))
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, rank = build_residual_ensembles(
                residuals,
                labels=np.asarray([1., 2., 3., 4., 5.]),
                validation_keys=("a", "b", "c", "d", "e"),
                output_root=root,
                expected_families=("raw",),
                expected_budgets=(.10,),
                expected_seeds=(1, 7, 21),
            )
            with np.load(root / raw.prediction_path) as archive:
                persisted_raw = archive["predictions"]
            with np.load(root / rank.prediction_path) as archive:
                persisted_rank = archive["predictions"]

        persisted_members = np.stack([quantized] * 3, axis=0)
        expected_raw = persisted_members.mean(axis=0).astype(np.float32)
        expected_rank = np.stack(
            [rank_percentiles(value) for value in persisted_members], axis=0
        ).mean(axis=0).astype(np.float32)
        np.testing.assert_array_equal(persisted_raw, expected_raw)
        np.testing.assert_array_equal(persisted_rank, expected_rank)


class SealTest(unittest.TestCase):
    def _payload(self, root):
        artifact = root / "candidate_predictions" / "upper_bound.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            keys=np.asarray(["a", "b"]),
            labels=np.asarray([1., 2.]),
            predictions=np.asarray([1.1, 1.9]),
        )
        record = candidate(
            "upper_bound", "expert", family="expert"
        ).to_dict()
        record["prediction_path"] = artifact.relative_to(root).as_posix()
        record["prediction_sha256"] = sha256(artifact)
        references = {
            "name": "upper_bound",
            "prediction_path": record["prediction_path"],
            "prediction_sha256": record["prediction_sha256"],
            "checkpoint_path": None,
            "checkpoint_sha256": None,
        }
        return {
            "selection": {
                "status": "complete",
                "selected_name": "upper_bound",
                "future_test_context": {
                    "target_ci": 0.706858,
                    "comparison": (
                        "A future evaluation compares CI strictly greater "
                        "than target_ci."
                    ),
                },
            },
            "fold_map": {"keys": ["a", "b"], "fold_ids": [0, 1]},
            "raw_report": {"family": "raw"},
            "rank_report": {"family": "rank"},
            "static_specs": {},
            "candidate_manifest": [record],
            "selected_references": references,
            "cache_identities": {
                "train": {"split": "train"},
                "valid": {"split": "valid"},
            },
        }

    def _seal(self, root, payload):
        package = root / "validation_package"
        seal = seal_validation_package(
            package, payload, artifact_root=root
        )
        return package, seal

    def test_seal_has_exact_flat_inventory_and_selected_reference_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._payload(root)
            package, seal = self._seal(root, payload)
            self.assertEqual(
                set(seal["files"]),
                {
                    "selection.json", "fold_map.npz",
                    "raw_cross_fit_report.json",
                    "rank_cross_fit_report.json", "static_specs.json",
                    "candidate_manifest.json",
                    "selected_references.json", "cache_identities.json",
                },
            )
            self.assertEqual(
                {path.name for path in package.iterdir()},
                set(seal["files"]) | {"selection_seal.json"},
            )
            selection = verify_validation_seal(
                package, artifact_root=root
            )
            refs = json.loads(
                (package / "selected_references.json").read_text()
            )
            selected_path = root / refs["prediction_path"]
            self.assertTrue(selected_path.is_file())
            self.assertEqual(refs["prediction_sha256"], sha256(selected_path))
            self.assertEqual(
                selection["future_test_context"]["target_ci"], 0.706858
            )

    def test_seal_rejects_unsafe_paths_in_manifest_and_selected_refs(self):
        unsafe = (
            "", "/absolute/a.npz", "../a.npz", r"C:\temp\a.npz",
            r"\\server\share\a.npz", "//server/share/a.npz",
            "file://host/a.npz", "https://host/a.npz",
        )
        for location in ("candidate_manifest", "selected_references"):
            for value in unsafe:
                with self.subTest(location=location, value=value):
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        payload = self._payload(root)
                        if location == "candidate_manifest":
                            payload[location][0]["prediction_path"] = value
                        else:
                            payload[location]["prediction_path"] = value
                        with self.assertRaises(ValueError):
                            seal_validation_package(
                                root / "validation_package",
                                payload,
                                artifact_root=root,
                            )

    def test_verifier_rejects_modified_missing_extra_and_bad_reference_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._payload(root)
            package, _ = self._seal(root, payload)
            (package / "selection.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_validation_seal(package, artifact_root=root)

            missing_root = root / "missing-root"
            missing_root.mkdir()
            payload = self._payload(missing_root)
            package, _ = self._seal(missing_root, payload)
            (package / "fold_map.npz").unlink()
            with self.assertRaises(ValueError):
                verify_validation_seal(package, artifact_root=missing_root)

            extra_root = root / "extra-root"
            extra_root.mkdir()
            payload = self._payload(extra_root)
            package, _ = self._seal(extra_root, payload)
            (package / "extra.bin").write_bytes(b"x")
            with self.assertRaises(ValueError):
                verify_validation_seal(package, artifact_root=extra_root)

            bad_root = root / "bad-root"
            bad_root.mkdir()
            payload = self._payload(bad_root)
            payload["selected_references"]["prediction_sha256"] = HASH_A
            with self.assertRaises(ValueError):
                seal_validation_package(
                    bad_root / "validation_package",
                    payload,
                    artifact_root=bad_root,
                )


class FlatPackageHardeningTest(unittest.TestCase):
    def _setup(self, root):
        output = root / "output"
        output.mkdir()
        payload = SealTest()._payload(output)
        package = output / "validation_package"
        return output, package, payload

    def _seal(self, output, package, payload):
        return seal_validation_package(
            package, payload, artifact_root=output
        )

    def test_seal_rejects_preexisting_extra_file_directory_and_nested_file(self):
        cases = ("file", "directory", "nested")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                output, package, payload = self._setup(Path(temporary))
                package.mkdir()
                if case == "file":
                    (package / "extra.bin").write_bytes(b"x")
                elif case == "directory":
                    (package / "extra").mkdir()
                else:
                    nested = package / "nested"
                    nested.mkdir()
                    (nested / "extra.bin").write_bytes(b"x")
                with self.assertRaises(ValueError):
                    self._seal(output, package, payload)
                self.assertFalse((package / "selection_seal.json").exists())

    def test_seal_rejects_required_file_symlink_to_inside_and_outside(self):
        for location in ("inside", "outside"):
            with self.subTest(location=location), \
                 tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, package, payload = self._setup(root)
                package.mkdir()
                inside = package / "static_specs.json"
                inside.write_text("{}", encoding="utf-8")
                outside = root / "outside.json"
                outside.write_text("{}", encoding="utf-8")
                target = inside if location == "inside" else outside
                (package / "selection.json").symlink_to(target)
                with self.assertRaises(ValueError):
                    self._seal(output, package, payload)
                self.assertFalse((package / "selection_seal.json").exists())

    def test_verify_rejects_required_file_symlink_to_inside_and_outside(self):
        for location in ("inside", "outside"):
            with self.subTest(location=location), \
                 tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, package, payload = self._setup(root)
                self._seal(output, package, payload)
                selection = package / "selection.json"
                selection.unlink()
                target = (
                    package / "static_specs.json"
                    if location == "inside"
                    else root / "outside.json"
                )
                if location == "outside":
                    target.write_text("{}", encoding="utf-8")
                selection.symlink_to(target)
                with self.assertRaises(ValueError):
                    verify_validation_seal(
                        package, artifact_root=output
                    )

    def test_verify_rejects_directory_nested_file_fifo_and_extra_file(self):
        cases = ("directory", "nested", "fifo", "file")
        for case in cases:
            with self.subTest(case=case), \
                 tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, package, payload = self._setup(root)
                self._seal(output, package, payload)
                if case == "directory":
                    (package / "extra").mkdir()
                elif case == "nested":
                    nested = package / "nested"
                    nested.mkdir()
                    (nested / "extra.bin").write_bytes(b"x")
                elif case == "fifo":
                    import os
                    os.mkfifo(str(package / "extra.fifo"))
                else:
                    (package / "extra.bin").write_bytes(b"x")
                with self.assertRaises(ValueError):
                    verify_validation_seal(
                        package, artifact_root=output
                    )

    def test_seal_rejects_referenced_prediction_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, package, payload = self._setup(root)
            artifact = output / payload["selected_references"]["prediction_path"]
            real = output / "real-prediction.npz"
            artifact.replace(real)
            artifact.symlink_to(real)
            payload["candidate_manifest"][0]["prediction_sha256"] = sha256(real)
            payload["selected_references"]["prediction_sha256"] = sha256(real)
            with self.assertRaises(ValueError):
                self._seal(output, package, payload)

    def test_seal_rejects_referenced_checkpoint_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            prediction = output / "residual_runs" / "residual" / "predictions_best.npz"
            checkpoint = output / "residual_runs" / "residual" / "checkpoint_best.pt"
            prediction.parent.mkdir(parents=True)
            np.savez_compressed(
                prediction,
                keys=np.asarray(["a", "b"]),
                labels=np.asarray([1., 2.]),
                predictions=np.asarray([1.1, 1.9]),
            )
            real_checkpoint = output / "real-checkpoint.pt"
            real_checkpoint.write_bytes(b"checkpoint")
            checkpoint.symlink_to(real_checkpoint)
            record = candidate(
                "residual_raw_025_1",
                "residual",
                family="raw",
                parent="static_raw",
                budget=.25,
                seed=1,
            ).to_dict()
            record["prediction_path"] = prediction.relative_to(output).as_posix()
            record["prediction_sha256"] = sha256(prediction)
            record["checkpoint_path"] = checkpoint.relative_to(output).as_posix()
            record["checkpoint_sha256"] = sha256(real_checkpoint)
            references = {
                "name": record["name"],
                "prediction_path": record["prediction_path"],
                "prediction_sha256": record["prediction_sha256"],
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
            payload = SealTest()._payload(output)
            payload["selection"]["selected_name"] = record["name"]
            payload["candidate_manifest"] = [record]
            payload["selected_references"] = references
            with self.assertRaises(ValueError):
                self._seal(
                    output, output / "validation_package", payload
                )


class OrchestrationTest(unittest.TestCase):
    def args(self, output, **changes):
        values = dict(
            train_cache_root="train",
            valid_cache_root="valid",
            train_count=25,
            valid_count=25,
            output=str(output),
            device="cpu",
            maximum_epochs=1,
            patience=1,
            batch_size=4,
            learning_rate=1e-4,
            weight_decay=1e-3,
            gradient_clip=1.0,
            resume=False,
            seeds=(1, 7),
            limit=10,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_parser_and_formal_smoke_seed_rules_have_no_selection_target(self):
        from dta.anchored_expert_fusion_v5 import run_validation

        parser = run_validation.build_arg_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--limit", options)
        self.assertIn("--seeds", options)
        self.assertNotIn("--budgets", options)
        self.assertFalse(any("test" in option.lower() for option in options))
        self.assertFalse(any("target" in option.lower() for option in options))
        for parameter in inspect.signature(
            run_validation.run_validation
        ).parameters:
            self.assertNotIn("test", parameter.lower())
            self.assertNotIn("target", parameter.lower())

        formal = SimpleNamespace(limit=None, seeds=FORMAL_SEEDS)
        self.assertEqual(
            run_validation._validated_inventory_controls(formal),
            ("formal", FORMAL_BUDGETS, FORMAL_SEEDS),
        )
        for seeds in ((1, 7), (1, 1, 7, 21, 42), FORMAL_SEEDS + (99,)):
            with self.subTest(formal_seeds=seeds):
                with self.assertRaises(ValueError):
                    run_validation._validated_inventory_controls(
                        SimpleNamespace(limit=None, seeds=seeds)
                    )
        self.assertEqual(
            run_validation._validated_inventory_controls(
                SimpleNamespace(limit=64, seeds=(7, 1))
            ),
            ("smoke", FORMAL_BUDGETS, (1, 7)),
        )
        for seeds in ((1,), (1, 1), (1, 99)):
            with self.subTest(smoke_seeds=seeds):
                with self.assertRaises(ValueError):
                    run_validation._validated_inventory_controls(
                        SimpleNamespace(limit=64, seeds=seeds)
                    )

    def test_failed_static_gate_prevents_fit_and_seal(self):
        from dta.anchored_expert_fusion_v5 import run_validation

        raw = SimpleNamespace(
            gate_passed=False, gate_reasons=("raw_reason",)
        )
        rank = SimpleNamespace(
            gate_passed=False, gate_reasons=("rank_reason",)
        )
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(
                 run_validation, "_load_validation_inputs",
                 return_value={"train": mock.Mock(), "valid": mock.Mock()},
             ), mock.patch.object(
                 run_validation, "_cross_fit",
                 return_value={"raw": raw, "rank": rank},
             ), mock.patch.object(
                 run_validation.training_module, "fit_one_residual"
             ) as fit, mock.patch.object(
                 run_validation, "seal_validation_package"
             ) as seal:
            result = run_validation.run_validation(
                self.args(temporary)
            )
        self.assertEqual(result["status"], "failed_static_gate")
        fit.assert_not_called()
        seal.assert_not_called()

    def test_resume_only_applies_to_residual_runs_that_already_started(self):
        from dta.anchored_expert_fusion_v5 import run_validation

        train_ref = cache_ref("train")
        valid_ref = cache_ref("valid")
        reports = {
            family: report(family, valid_ref) for family in ("raw", "rank")
        }
        observed_resume = {}

        def fake_fit(**kwargs):
            output = Path(kwargs["output_dir"])
            observed_resume[output.name] = kwargs["resume"]
            output.mkdir(parents=True, exist_ok=True)
            spec = kwargs["static_spec"]
            prediction = (
                valid_ref.cache.labels.astype(np.float64)
                + .04 + kwargs["budget"] / 100.0
                + kwargs["seed"] / 10000.0
                + (.001 if spec.family == "rank" else 0.0)
            )
            return fake_residual_training_result(
                output,
                valid_ref,
                spec,
                kwargs["budget"],
                kwargs["seed"],
                prediction,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            started = root / "residual_runs" / "raw_0.10_1"
            started.mkdir(parents=True)
            (started / "status.json").write_text(
                "{\"status\": \"running\"}", encoding="utf-8"
            )
            (root / "residual_runs" / "raw_0.10_7").mkdir()
            with mock.patch.object(
                run_validation,
                "_load_validation_inputs",
                return_value={"train": train_ref, "valid": valid_ref},
            ), mock.patch.object(
                run_validation, "_cross_fit", return_value=reports,
            ), mock.patch.object(
                run_validation.training_module,
                "fit_one_residual",
                side_effect=fake_fit,
            ):
                run_validation.run_validation(
                    self.args(root, resume=True)
                )

        self.assertEqual(
            {name for name, resume in observed_resume.items() if resume},
            {"raw_0.10_1"},
        )

    def test_smoke_orchestration_persists_real_candidate_artifacts_and_refs(self):
        from dta.anchored_expert_fusion_v5 import run_validation

        train_ref = cache_ref("train")
        valid_ref = cache_ref("valid")
        reports = {
            family: report(family, valid_ref) for family in ("raw", "rank")
        }

        def fake_fit(**kwargs):
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            spec = kwargs["static_spec"]
            family = spec.family
            seed = kwargs["seed"]
            budget = kwargs["budget"]
            prediction = (
                valid_ref.cache.labels.astype(np.float64)
                + .04 + budget / 100.0 + seed / 10000.0
                + (0.001 if family == "rank" else 0.0)
            )
            return fake_residual_training_result(
                output, valid_ref, spec, budget, seed, prediction,
            )

        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(
                 run_validation, "_load_validation_inputs",
                 return_value={"train": train_ref, "valid": valid_ref},
             ), mock.patch.object(
                 run_validation, "_cross_fit", return_value=reports,
             ), mock.patch.object(
                 run_validation.training_module, "fit_one_residual",
                 side_effect=fake_fit,
             ):
            root = Path(temporary)
            result = run_validation.run_validation(self.args(root))
            self.assertEqual(result["inventory_mode"], "smoke")
            self.assertEqual(result["candidate_count"], 29)
            package = root / "validation_package"
            self.assertEqual(Path(result["package"]), package)
            verify_validation_seal(package, artifact_root=root)
            manifest = json.loads(
                (package / "candidate_manifest.json").read_text()
            )
            self.assertEqual(len(manifest), 29)
            self.assertEqual(
                {record["name"] for record in manifest},
                set(result["candidate_names"]),
            )
            for record in manifest:
                prediction_path = root / record["prediction_path"]
                self.assertTrue(prediction_path.is_file(), record["name"])
                self.assertEqual(
                    record["prediction_sha256"], sha256(prediction_path)
                )
                self.assertFalse(Path(record["prediction_path"]).is_absolute())
                if record["type"] == "residual":
                    checkpoint_path = root / record["checkpoint_path"]
                    self.assertTrue(checkpoint_path.is_file())
                    self.assertEqual(
                        record["checkpoint_sha256"],
                        sha256(checkpoint_path),
                    )
                else:
                    self.assertIsNone(record["checkpoint_path"])
                    self.assertIsNone(record["checkpoint_sha256"])
            references = json.loads(
                (package / "selected_references.json").read_text()
            )
            selected_prediction = root / references["prediction_path"]
            self.assertTrue(selected_prediction.is_file())
            self.assertEqual(
                references["prediction_sha256"],
                sha256(selected_prediction),
            )
            selection = json.loads(
                (package / "selection.json").read_text()
            )
            context = selection["future_test_context"]
            self.assertEqual(
                set(context), {"target_ci", "comparison"}
            )
            self.assertEqual(context["target_ci"], 0.706858)
            self.assertIn("strictly greater", context["comparison"])
            best = min(
                (CandidateRecord.from_dict(record) for record in manifest),
                key=validation_candidate_key,
            )
            self.assertEqual(selection["selected_name"], best.name)


    def _run_formal_producer(self, root):
        from dta.anchored_expert_fusion_v5 import run_validation

        train_ref = cache_ref("train")
        valid_ref = cache_ref("valid")
        reports = {
            family: report(family, valid_ref)
            for family in ("raw", "rank")
        }

        def fake_fit(**kwargs):
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            spec = kwargs["static_spec"]
            family = spec.family
            seed = kwargs["seed"]
            budget = kwargs["budget"]
            prediction = (
                valid_ref.cache.labels.astype(np.float64)
                + .04 + budget / 100.0 + seed / 10000.0
                + (.001 if family == "rank" else 0.0)
            )
            return fake_residual_training_result(
                output, valid_ref, spec, budget, seed, prediction,
            )

        with mock.patch.object(
            run_validation,
            "_load_validation_inputs",
            return_value={"train": train_ref, "valid": valid_ref},
        ), mock.patch.object(
            run_validation,
            "_cross_fit",
            return_value=reports,
        ), mock.patch.object(
            run_validation.training_module,
            "fit_one_residual",
            side_effect=fake_fit,
        ):
            result = run_validation.run_validation(self.args(
                root,
                limit=None,
                seeds=FORMAL_SEEDS,
            ))
        return result

    def _task6_preflight(self, package):
        from dta.anchored_expert_fusion_v5 import evaluate_test

        loaded = evaluate_test._package(package)
        (_, _, selection, manifest, references, specs,
         identities, reports) = loaded
        validated = evaluate_test._validate_formal_package(
            selection,
            manifest,
            references,
            identities,
            reports,
            specs,
            evaluate_test._load_fold_map(package),
        )
        return loaded, validated

    def test_formal_producer_package_passes_task6_preflight(self):
        from dta.anchored_expert_fusion_v5 import training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._run_formal_producer(root)
            self.assertEqual(result["inventory_mode"], "formal")
            self.assertEqual(result["candidate_count"], 47)
            package = root / "validation_package"
            loaded, validated = self._task6_preflight(package)
            (_, _, _, manifest, references, specs,
             _, reports) = loaded
            records, selected, _, _, passing = validated
            self.assertEqual(passing, ("raw", "rank"))
            self.assertEqual(len(records), 47)
            self.assertEqual(references["name"], selected.name)
            for family in passing:
                self.assertIn("folds", reports[family])
                self.assertIn("validation_available", reports[family])
                spec = training.StaticSpec.from_dict(specs[family])
                spec_hash = training.static_spec_sha256(spec)
                family_records = [
                    record for record in records
                    if record.family == family
                ]
                self.assertTrue(family_records)
                self.assertTrue(all(
                    record.lineage_hashes["static_spec_sha256"]
                    == spec_hash
                    for record in family_records
                ))
            selected_manifest = next(
                item for item in manifest
                if item["name"] == references["name"]
            )
            self.assertEqual(
                selected_manifest["prediction_sha256"],
                references["prediction_sha256"],
            )

    def test_resealed_producer_report_spec_and_lineage_mutations_fail_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run_formal_producer(root)
            package = root / "validation_package"
            payload = {
                "selection": json.loads(
                    (package / "selection.json").read_text()
                ),
                "fold_map": (package / "fold_map.npz").read_bytes(),
                "raw_report": json.loads(
                    (package / "raw_cross_fit_report.json").read_text()
                ),
                "rank_report": json.loads(
                    (package / "rank_cross_fit_report.json").read_text()
                ),
                "static_specs": json.loads(
                    (package / "static_specs.json").read_text()
                ),
                "candidate_manifest": json.loads(
                    (package / "candidate_manifest.json").read_text()
                ),
                "selected_references": json.loads(
                    (package / "selected_references.json").read_text()
                ),
                "cache_identities": json.loads(
                    (package / "cache_identities.json").read_text()
                ),
            }

            def clone():
                return {
                    key: value if isinstance(value, bytes) else
                    json.loads(json.dumps(value))
                    for key, value in payload.items()
                }

            report_changed = clone()
            report_changed["raw_report"]["predictions"][0] += .01
            spec_changed = clone()
            spec_changed["static_specs"]["raw"][
                "deploy_weights"
            ] = [.4, .4, .2]
            lineage_changed = clone()
            static_raw = next(
                item for item in lineage_changed["candidate_manifest"]
                if item["name"] == "static_raw"
            )
            static_raw["lineage_hashes"][
                "static_spec_sha256"
            ] = HASH_C

            for name, changed in (
                ("report", report_changed),
                ("spec", spec_changed),
                ("lineage", lineage_changed),
            ):
                with self.subTest(mutation=name):
                    (package / "selection_seal.json").unlink()
                    seal_validation_package(
                        package,
                        changed,
                        artifact_root=root,
                    )
                    verify_validation_seal(package, artifact_root=root)
                    with self.assertRaises(ValueError):
                        self._task6_preflight(package)


class OfficialTestEvaluatorRedTest(unittest.TestCase):
    def test_strict_target_status_rejects_bool_and_nonfinite(self):
        from dta.anchored_expert_fusion_v5.evaluate_test import TARGET_CI, test_status
        self.assertEqual(TARGET_CI, 0.706858)
        self.assertEqual(test_status(TARGET_CI + 1e-12), "passed")
        self.assertEqual(test_status(TARGET_CI), "complete_below_target")
        self.assertEqual(test_status(TARGET_CI - 1e-12), "complete_below_target")
        for value in (True, False, float("nan"), float("inf"), None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    test_status(value)

    def test_validation_seal_is_verified_before_test_loader(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        with tempfile.TemporaryDirectory() as temporary:
            events = []
            with mock.patch.object(
                evaluate_test.selection_module, "verify_validation_seal",
                side_effect=lambda *args, **kwargs: (events.append("seal"), (_ for _ in ()).throw(ValueError("bad seal")))[1],
            ), mock.patch.object(
                evaluate_test.cache_module, "load_fusion_cache",
                side_effect=lambda *args, **kwargs: events.append("load"),
            ):
                with self.assertRaisesRegex(ValueError, "bad seal"):
                    evaluate_test._evaluate_impl(
                        temporary, "unused-test-cache", temporary,
                        device="cpu", expected_identity=evaluate_test.ExpectedTestIdentity(
                            2, HASH_A, HASH_B
                        ),
                    )
            self.assertEqual(events, ["seal"])


class OfficialEvaluatorFormalContractTest(unittest.TestCase):
    def _records(self, families):
        train = identity("train", 50, "train-formal")
        valid = identity("valid", 5, "valid-formal")
        reports = {
            "raw": {
                "family": "raw", "report_hash": HASH_A,
                "gate": {"passed": "raw" in families, "reasons": []},
            },
            "rank": {
                "family": "rank", "report_hash": HASH_B,
                "gate": {"passed": "rank" in families, "reasons": []},
            },
        }
        def with_lineage(record, values):
            payload = record.to_dict()
            payload["lineage_hashes"] = values
            return CandidateRecord.from_dict(payload)
        experts = []
        for item in expert_candidates():
            experts.append(with_lineage(item, {
                "valid_manifest": valid["manifest_sha256"],
                "valid_ordered_keys": valid["ordered_key_sha256"],
                "prediction_source": valid["files"]["features.npz"],
            }))
        statics = []
        residuals = []
        ensembles = []
        for family in families:
            cross_hash = reports[family]["report_hash"]
            common = {
                "train_manifest": train["manifest_sha256"],
                "train_ordered_keys": train["ordered_key_sha256"],
                "valid_manifest": valid["manifest_sha256"],
                "valid_ordered_keys": valid["ordered_key_sha256"],
                "cross_fit_report": cross_hash,
                "static_spec_sha256": HASH_A,
            }
            statics.append(with_lineage(
                candidate("static_" + family, "static", family=family),
                common,
            ))
            family_residuals = []
            for item in residual_grid(families=(family,)):
                converted = with_lineage(item, common)
                residuals.append(converted)
                family_residuals.append(converted)
            for budget in FORMAL_BUDGETS:
                group = [item for item in family_residuals if item.budget == budget]
                members = tuple(
                    item.name for item in sorted(group, key=validation_candidate_key)[:3]
                )
                hashes = {name: HASH_A for name in members}
                member_payload = json.dumps(
                    hashes, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                ensemble_lineage = dict(common)
                ensemble_lineage["member_set"] = hashlib.sha256(
                    member_payload
                ).hexdigest()
                budget_name = "%02d" % int(round(budget * 100))
                for kind, suffix in (("raw_ensemble", "raw"),
                                     ("rank_ensemble", "rank")):
                    item = candidate(
                        "ensemble_%s_%s_%s" % (family, budget_name, suffix),
                        kind, family=family, parent="static_" + family,
                        budget=budget, members=members,
                        member_hashes=hashes,
                    )
                    ensembles.append(with_lineage(item, ensemble_lineage))
        records = tuple(experts + statics + residuals + ensembles)
        return records, train, valid, reports

    def _selection(self, records, selected=None):
        selected = records[0] if selected is None else selected
        return {
            "status": "complete",
            "selected_name": selected.name,
            "selected_type": selected.type,
            "selected_family": selected.family,
            "selected_validation_metrics": dict(selected.validation_metrics),
            "candidate_names": [item.name for item in records],
            "candidate_count": len(records),
            "inventory_mode": "formal",
            "inventory_controls": {
                "mode": "formal",
                "budgets": list(FORMAL_BUDGETS),
                "seeds": list(FORMAL_SEEDS),
                "limit": None,
            },
        }

    def _refs(self, selected):
        return {
            "name": selected.name,
            "prediction_path": selected.prediction_path,
            "prediction_sha256": selected.prediction_sha256,
            "checkpoint_path": selected.checkpoint_path,
            "checkpoint_sha256": selected.checkpoint_sha256,
        }

    def test_exact_formal_25_and_47_candidate_packages_are_accepted(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        fixture = OfficialEvaluatorCrossFitBindingTest()
        raw = fixture._formal("raw")
        rank = fixture._formal("rank")
        one_cases = [(raw, 25, ("raw",))]
        raw_records, _, _, identities, raw_reports, raw_specs = raw
        rank_records, _, _, _, rank_reports, rank_specs = rank
        both_records = tuple(raw_records) + tuple(rank_records[3:])
        both_selection = self._selection(both_records)
        both = (
            both_records, both_selection, self._refs(both_records[0]),
            identities,
            {"raw": raw_reports["raw"], "rank": rank_reports["rank"]},
            {"raw": raw_specs["raw"], "rank": rank_specs["rank"]},
        )
        for values, count, families in one_cases + [(both, 47, ("raw", "rank"))]:
            records, selection, refs, identities, reports, specs = values
            with self.subTest(families=families):
                actual, selected, _, _, passing = evaluate_test._validate_formal_package(
                    selection, [item.to_dict() for item in records], refs,
                    identities, reports, specs, sealed_fold_map(reports),
                )
                self.assertEqual(len(actual), count)
                self.assertEqual(selected.name, "upper_bound")
                self.assertEqual(passing, families)

    def test_formal_package_rejects_smoke_and_inventory_or_lineage_drift(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        records, base, refs, identities, reports, specs = (
            OfficialEvaluatorCrossFitBindingTest()._formal("raw")
        )
        cases = {}
        smoke = json.loads(json.dumps(base))
        smoke["inventory_mode"] = "smoke"
        smoke["inventory_controls"]["mode"] = "smoke"
        smoke["inventory_controls"]["limit"] = 10
        cases["smoke"] = (smoke, list(records))
        cases["missing"] = (base, list(records[:-1]))
        cases["duplicate"] = (base, list(records) + [records[-1]])
        extra = candidate("extra", "expert", family="expert")
        cases["extra"] = (base, list(records) + [extra])
        broken = [item.to_dict() for item in records]
        broken[3]["lineage_hashes"] = {
            "valid_manifest": identities["valid"]["manifest_sha256"]
        }
        cases["missing_lineage"] = (base, broken)
        for name, (selection, values) in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                manifest = [
                    item.to_dict() if isinstance(item, CandidateRecord) else item
                    for item in values
                ]
                evaluate_test._validate_formal_package(
                    selection, manifest, refs, identities, reports, specs, sealed_fold_map(reports),
                )

    def test_official_constants_public_signature_and_cli_are_fixed(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        self.assertEqual(evaluate_test.OFFICIAL_TEST_COUNT, 4751)
        self.assertEqual(
            evaluate_test.OFFICIAL_TEST_MANIFEST_SHA256,
            "f8741d40837568c552ca0bd3725e0b079ba855d56d2a663c0bb95c4bc09c95b8",
        )
        self.assertEqual(
            evaluate_test.OFFICIAL_TEST_ORDERED_KEY_SHA256,
            "432db77b96cc1c5ff8eff4de655b64b4a487c4b448913ee7db33cdaf886b303f",
        )
        parameters = inspect.signature(
            evaluate_test.evaluate_sealed_selection
        ).parameters
        self.assertEqual(tuple(parameters), ("package", "test_cache", "output", "device"))
        options = {
            option for action in evaluate_test.build_arg_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            options,
            {"-h", "--help", "--package", "--test-cache", "--output", "--device"},
        )
        self.assertFalse(any("count" in item or "hash" in item or "target" in item
                             for item in options))

    def test_private_test_identity_checks_count_manifest_order_coverage_and_finite(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        cache = cache_ref("test", count=5).cache
        metadata = dict(cache.metadata)
        metadata["ordered_key_sha256"] = hashlib.sha256(
            json.dumps(list(cache.keys), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        metadata["latent_dimensions"] = {
            "upper_bound": 2, "boltz_pairwise": 3, "three_dprot": 4,
        }
        cache = replace(cache, metadata=metadata)
        expected = evaluate_test.ExpectedTestIdentity(
            count=5,
            manifest_sha256=metadata["manifest_sha256"],
            ordered_key_sha256=metadata["ordered_key_sha256"],
        )
        _, keys = evaluate_test._validate_test_cache(cache, expected)
        self.assertEqual(keys, cache.keys)
        cases = {
            "count": replace(expected, count=4),
            "manifest": replace(expected, manifest_sha256=HASH_A),
            "order": replace(expected, ordered_key_sha256=HASH_B),
        }
        for name, changed in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate_test._validate_test_cache(cache, changed)
        available = cache.available.copy()
        available[0, 0] = False
        with self.assertRaises(ValueError):
            evaluate_test._validate_test_cache(
                replace(cache, available=available), expected
            )
        labels = cache.labels.copy()
        labels[0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_test._validate_test_cache(
                replace(cache, labels=labels), expected
            )

    def test_one_shot_reservation_errors_are_runtime_errors(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already evaluated"):
                evaluate_test._reserve(output)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "evaluation.start").write_text("started", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "in progress"):
                evaluate_test._reserve(output)


class OfficialEvaluatorReconstructionTest(unittest.TestCase):
    def _cache(self):
        cache = cache_ref("test", count=5).cache
        metadata = dict(cache.metadata)
        metadata["ordered_key_sha256"] = hashlib.sha256(
            json.dumps(list(cache.keys), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        metadata["latent_dimensions"] = {
            "upper_bound": 2, "boltz_pairwise": 3, "three_dprot": 4,
        }
        return replace(cache, metadata=metadata)

    def _specs(self):
        from dta.anchored_expert_fusion_v5.training import StaticSpec
        train = identity("train", 50, "train-reconstruct")
        valid = identity("valid", 25, "valid-reconstruct")
        specs = {
            "raw": StaticSpec(
                family="raw", deploy_weights=(.5, .3, .2),
                rank_slope=1.0, rank_intercept=0.0,
                cross_fit_report_hash=HASH_A,
                train_cache_identity=train, valid_cache_identity=valid,
            ),
            "rank": StaticSpec(
                family="rank", deploy_weights=(.5, .3, .2),
                rank_slope=4.0, rank_intercept=1.0,
                cross_fit_report_hash=HASH_B,
                train_cache_identity=train, valid_cache_identity=valid,
            ),
        }
        return specs, train, valid

    def _prediction_artifact(self, root, name, keys):
        path = root / "candidate_predictions" / (name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.arange(len(keys), dtype=np.float32)
        np.savez_compressed(
            path, keys=np.asarray(keys), labels=values,
            predictions=values + .1,
        )
        return path.relative_to(root).as_posix(), sha256(path)

    def _checkpoint(self, root, name, record_seed, budget, spec, valid=True):
        import torch
        from dta.anchored_expert_fusion_v5.model import AnchoredResidualModel
        from dta.anchored_expert_fusion_v5.training import static_spec_sha256
        path = root / "residual_runs" / name / "checkpoint_best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        widths = (2, 3, 4)
        model = AnchoredResidualModel(widths, budget=budget, dropout=.1)
        config = {
            "class": "AnchoredResidualModel",
            "latent_widths": list(widths),
            "budget": budget,
            "dropout": .1,
            "adapter_width": 64,
            "hidden_widths": [256, 128, 64],
            "dtype": "float32",
        }
        lineage = {
            "train": spec.to_dict()["train_cache_identity"],
            "valid": spec.to_dict()["valid_cache_identity"],
        }
        run_config = {
            "schema_version": 1,
            "static_spec_sha256": static_spec_sha256(spec),
            "static_spec": spec.to_dict(),
            "family": spec.family,
            "deploy_weights": list(spec.deploy_weights),
            "rank_calibration": {
                "slope": spec.rank_slope,
                "intercept": spec.rank_intercept,
            },
            "cache_lineage": lineage,
            "budget": budget,
            "seed": record_seed,
            "model": config,
        }
        payload = {
            "model_config": config,
            "model_state_dict": model.state_dict(),
            "budget": budget,
            "seed": record_seed,
            "static_spec_sha256": static_spec_sha256(spec),
            "cache_lineage": lineage,
            "run_config": run_config,
        }
        if not valid:
            payload.pop("run_config")
        torch.save(payload, path)
        return path.relative_to(root).as_posix(), sha256(path)

    def _record(self, root, keys, name, kind, family, *, spec=None,
                seed=None, budget=None, members=(), member_hashes=None,
                valid_checkpoint=True):
        prediction_path, prediction_hash = self._prediction_artifact(
            root, name, keys
        )
        checkpoint_path = checkpoint_hash = None
        parent = None
        if kind == "residual":
            parent = "static_" + family
            checkpoint_path, checkpoint_hash = self._checkpoint(
                root, name, seed, budget, spec,
                valid=valid_checkpoint,
            )
        elif kind in ("raw_ensemble", "rank_ensemble"):
            parent = "static_" + family
        if kind == "expert":
            record_lineage = {"sealed": HASH_A}
        else:
            if spec is None:
                raise ValueError("reconstruction fixture requires a static spec")
            from dta.anchored_expert_fusion_v5.training import static_spec_sha256
            record_lineage = {
                "train_manifest": spec.train_cache_identity[
                    "manifest_sha256"
                ],
                "train_ordered_keys": spec.train_cache_identity[
                    "ordered_key_sha256"
                ],
                "valid_manifest": spec.valid_cache_identity[
                    "manifest_sha256"
                ],
                "valid_ordered_keys": spec.valid_cache_identity[
                    "ordered_key_sha256"
                ],
                "cross_fit_report": spec.cross_fit_report_hash,
                "static_spec_sha256": static_spec_sha256(spec),
            }
            if kind in ("raw_ensemble", "rank_ensemble"):
                member_payload = json.dumps(
                    dict(member_hashes or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                record_lineage["member_set"] = hashlib.sha256(
                    member_payload
                ).hexdigest()
        return CandidateRecord(
            name=name, type=kind, family=family,
            prediction_path=prediction_path,
            prediction_sha256=prediction_hash,
            validation_metrics=metrics(.72),
            parent_static_name=parent,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            lineage_hashes=record_lineage,
            member_hashes=member_hashes or {},
            budget=budget, seed=seed, members=members,
            ensemble_rule=(
                "arithmetic_mean" if kind == "raw_ensemble" else
                "stable_average_percentile_mean"
                if kind == "rank_ensemble" else None
            ),
        )

    def test_expert_static_residual_raw_and_rank_reconstruction(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        from dta.anchored_expert_fusion_v5.training import static_predictions
        cache = self._cache()
        keys = tuple("v%d" % index for index in range(5))
        specs, train, valid = self._specs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "validation_package"
            package.mkdir()
            expert = self._record(
                root, keys, "upper_bound", "expert", "expert"
            )
            static = self._record(
                root, keys, "static_raw", "static", "raw",
                spec=specs["raw"],
            )
            residuals = tuple(
                self._record(
                    root, keys, "residual_raw_25_%d" % seed,
                    "residual", "raw", spec=specs["raw"],
                    seed=seed, budget=.25,
                ) for seed in (1, 7)
            )
            raw_hashes = {
                item.name: item.prediction_sha256 for item in residuals
            }
            raw = self._record(
                root, keys, "ensemble_raw_25_raw", "raw_ensemble",
                "raw", spec=specs["raw"], budget=.25,
                members=tuple(item.name for item in residuals),
                member_hashes=raw_hashes,
            )
            rank_residuals = tuple(
                self._record(
                    root, keys, "residual_rank_25_%d" % seed,
                    "residual", "rank", spec=specs["rank"],
                    seed=seed, budget=.25,
                ) for seed in (1, 7)
            )
            rank_hashes = {
                item.name: item.prediction_sha256 for item in rank_residuals
            }
            rank = self._record(
                root, keys, "ensemble_rank_25_rank", "rank_ensemble",
                "rank", spec=specs["rank"], budget=.25,
                members=tuple(item.name for item in rank_residuals),
                member_hashes=rank_hashes,
            )
            spec_payload = {name: value.to_dict()
                            for name, value in specs.items()}
            predicted_expert = evaluate_test._predict(
                package, (expert,), expert, spec_payload,
                train, valid, cache, torch.device("cpu"), keys,
            )
            np.testing.assert_array_equal(
                predicted_expert,
                cache.predictions[:, 0].astype(np.float64),
            )
            predicted_static = evaluate_test._predict(
                package, (static,), static, spec_payload,
                train, valid, cache, torch.device("cpu"), keys,
            )
            raw_base = static_predictions(cache, specs["raw"])
            np.testing.assert_allclose(predicted_static, raw_base)
            predicted_residual = evaluate_test._predict(
                package, residuals, residuals[0], spec_payload,
                train, valid, cache, torch.device("cpu"), keys,
            )
            np.testing.assert_allclose(predicted_residual, raw_base)
            predicted_raw = evaluate_test._predict(
                package, residuals + (raw,), raw, spec_payload,
                train, valid, cache, torch.device("cpu"), keys,
            )
            np.testing.assert_allclose(predicted_raw, raw_base)
            predicted_rank = evaluate_test._predict(
                package, rank_residuals + (rank,), rank, spec_payload,
                train, valid, cache, torch.device("cpu"), keys,
            )
            rank_base = static_predictions(cache, specs["rank"])
            expected = evaluate_test.rank_percentiles(rank_base) * 4.0 + 1.0
            np.testing.assert_allclose(predicted_rank, expected)

    def test_checkpoint_requires_complete_config_lineage_and_strict_state(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        cache = self._cache()
        keys = tuple("v%d" % index for index in range(5))
        specs, train, valid = self._specs()
        del train, valid
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "validation_package"
            package.mkdir()
            incomplete = self._record(
                root, keys, "residual_raw_25_1", "residual", "raw",
                spec=specs["raw"], seed=1, budget=.25,
                valid_checkpoint=False,
            )
            checkpoint = root / incomplete.checkpoint_path
            with self.assertRaisesRegex(ValueError, "reconstruction metadata"):
                evaluate_test._model(
                    checkpoint, incomplete, specs["raw"], (2, 3, 4),
                    torch.device("cpu"),
                )
            valid = self._record(
                root, keys, "residual_raw_25_7", "residual", "raw",
                spec=specs["raw"], seed=7, budget=.25,
            )
            payload = torch.load(
                root / valid.checkpoint_path, map_location="cpu",
                weights_only=False,
            )
            payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
            torch.save(payload, root / valid.checkpoint_path)
            with self.assertRaisesRegex(ValueError, "state_dict"):
                evaluate_test._model(
                    root / valid.checkpoint_path, valid, specs["raw"],
                    (2, 3, 4), torch.device("cpu"),
                )

    def test_prediction_does_not_depend_on_test_labels(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        cache = self._cache()
        changed = replace(cache, labels=cache.labels[::-1].copy())
        keys = tuple("v%d" % index for index in range(5))
        specs, train, valid = self._specs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "validation_package"
            package.mkdir()
            selected = self._record(
                root, keys, "static_raw", "static", "raw",
                spec=specs["raw"],
            )
            payload = {name: value.to_dict() for name, value in specs.items()}
            first = evaluate_test._predict(
                package, (selected,), selected, payload,
                train, valid, cache, torch.device("cpu"), keys,
            )
            second = evaluate_test._predict(
                package, (selected,), selected, payload,
                train, valid, changed, torch.device("cpu"), keys,
            )
            self.assertTrue(np.array_equal(first, second))
            self.assertNotEqual(
                evaluate_test._metrics(cache.labels, first, "first"),
                evaluate_test._metrics(changed.labels, second, "second"),
            )

    def test_artifact_tamper_symlink_and_nonregular_are_rejected(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"sealed")
            digest = sha256(artifact)
            self.assertEqual(
                evaluate_test._ref(root, "artifact.bin", digest, "test"),
                artifact,
            )
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                evaluate_test._ref(root, "artifact.bin", digest, "test")
            target = root / "target.bin"
            target.write_bytes(b"sealed")
            link = root / "link.bin"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                evaluate_test._ref(root, "link.bin", digest, "test")
            directory = root / "directory"
            directory.mkdir()
            with self.assertRaisesRegex(ValueError, "regular"):
                evaluate_test._ref(root, "directory", digest, "test")

    def _evaluation_mocks(self, root, cache, predictions):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        package = root / "validation_package"
        package.mkdir()
        (package / "selection_seal.json").write_bytes(b"sealed\n")
        selected = candidate("upper_bound", "expert", family="expert")
        selection = {
            "selected_validation_metrics": dict(selected.validation_metrics)
        }
        package_result = (
            package, {}, selection, [selected.to_dict()], {},
            {"raw": {}}, {}, {},
        )
        validated = (
            (selected,), selected, identity("train", 1, "out-train"),
            identity("valid", 1, "out-valid"), ("raw",),
        )
        return (
            mock.patch.object(evaluate_test, "_package", return_value=package_result),
            mock.patch.object(
                evaluate_test, "_validate_formal_package",
                return_value=validated,
            ),
            mock.patch.object(evaluate_test, "_fold_keys", return_value=("v",)),
            mock.patch.object(
                evaluate_test.cache_module, "load_fusion_cache",
                return_value=cache,
            ),
            mock.patch.object(
                evaluate_test, "_predict",
                return_value=np.asarray(predictions, dtype=np.float64),
            ),
        )

    def test_one_shot_atomic_outputs_result_fields_key_order_and_second_call(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        cache = self._cache()
        expected = evaluate_test.ExpectedTestIdentity(
            len(cache.keys), cache.metadata["manifest_sha256"],
            cache.metadata["ordered_key_sha256"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "official_output"
            patches = self._evaluation_mocks(
                root, cache, cache.predictions[:, 1]
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = evaluate_test._evaluate_impl(
                    "package", "cache", output,
                    expected_identity=expected,
                )
                with self.assertRaisesRegex(RuntimeError, "already evaluated"):
                    evaluate_test._evaluate_impl(
                        "package", "cache", output,
                        expected_identity=expected,
                    )
            required = {
                "status", "count", "metrics", "coverage",
                "selected_validation_metrics", "upper_bound_test_metrics",
                "delta_ci_vs_upper_bound", "target_ci",
                "delta_ci_vs_target", "selection_seal_sha256",
                "selected_name", "selected_type", "test_manifest_sha256",
                "test_ordered_key_sha256", "prediction_sha256",
            }
            self.assertEqual(set(result), required)
            self.assertEqual(result["count"], len(cache.keys))
            self.assertEqual(result["coverage"], 1.0)
            self.assertAlmostEqual(
                result["delta_ci_vs_target"],
                result["metrics"]["CI"] - evaluate_test.TARGET_CI,
            )
            self.assertAlmostEqual(
                result["delta_ci_vs_upper_bound"],
                result["metrics"]["CI"] -
                result["upper_bound_test_metrics"]["CI"],
            )
            with np.load(output / "predictions.npz", allow_pickle=False) as archive:
                self.assertEqual(
                    tuple(str(value) for value in archive["keys"].tolist()),
                    cache.keys,
                )
            self.assertTrue((output / "result.json").is_file())
            self.assertTrue((output / "evaluation.complete").is_file())
            self.assertFalse((output / "evaluation.start").exists())
            self.assertFalse(any(path.name.endswith(".tmp")
                                 for path in output.iterdir()))

    def test_failed_cache_preflight_cleans_marker_and_is_retryable(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        cache = self._cache()
        expected = evaluate_test.ExpectedTestIdentity(
            len(cache.keys), cache.metadata["manifest_sha256"],
            cache.metadata["ordered_key_sha256"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "official_output"
            patches = self._evaluation_mocks(
                root, cache, cache.predictions[:, 0]
            )
            patches = list(patches)
            patches[3] = mock.patch.object(
                evaluate_test.cache_module, "load_fusion_cache",
                side_effect=[ValueError("bad cache"), cache],
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(ValueError, "bad cache"):
                    evaluate_test._evaluate_impl(
                        "package", "cache", output,
                        expected_identity=expected,
                    )
                self.assertFalse((output / "result.json").exists())
                self.assertFalse((output / "evaluation.complete").exists())
                self.assertFalse((output / "evaluation.start").exists())
                result = evaluate_test._evaluate_impl(
                    "package", "cache", output,
                    expected_identity=expected,
                )
                self.assertEqual(result["count"], len(cache.keys))


class OfficialEvaluatorCrossFitBindingTest(unittest.TestCase):
    def _report(self, family, passed=True):
        from dta.anchored_expert_fusion_v5 import static_fusion
        keys = tuple("v%d" % index for index in range(5))
        fold_ids = np.arange(5, dtype=np.int64)
        held_out = np.ones(5, dtype=np.int64)
        weights = np.tile(np.asarray([.5, .3, .2]), (5, 1))
        candidate_metrics = metrics(.72 if passed else .70)
        upper_metrics = metrics(.70)
        slope = 1.0 if family == "raw" else 2.0
        intercept = 0.0 if family == "raw" else -.5
        folds = tuple(
            static_fusion.FoldResult(
                fold_id=index,
                held_out_indices=np.asarray([index], dtype=np.int64),
                selected_weights=weights[index].copy(),
                candidate_metrics=dict(candidate_metrics),
                upper_bound_metrics=dict(upper_metrics),
                search_metrics=dict(candidate_metrics),
                slope=slope, intercept=intercept,
                stage_counts=(1, 1, 1), searched_count=3,
            )
            for index in range(5)
        )
        gate = static_fusion.evaluate_gate(
            candidate_metrics, upper_metrics,
            [dict(candidate_metrics) for _ in folds],
            [dict(upper_metrics) for _ in folds],
            weights, held_out,
        )
        self.assertEqual(gate.passed, passed)
        report = static_fusion.CrossFitReport(
            family=family, validation_keys=keys,
            fold_ids=fold_ids,
            fold_map_hash=static_fusion.fold_map_sha256(keys, fold_ids),
            held_out_count=held_out,
            predictions=np.linspace(1.0, 5.0, 5),
            pooled_metrics=dict(candidate_metrics),
            upper_bound_pooled_metrics=dict(upper_metrics),
            folds=folds, fold_weights=weights,
            deploy_weights=static_fusion.deploy_weights(weights),
            gate=gate,
            validation_available=np.ones((5, 3), dtype=np.bool_),
        )
        report = replace(
            report, report_hash=static_fusion.report_sha256(report)
        )
        payload = {
            "family": report.family,
            "validation_keys": list(report.validation_keys),
            "fold_ids": report.fold_ids.tolist(),
            "fold_map_hash": report.fold_map_hash,
            "held_out_count": report.held_out_count.tolist(),
            "predictions": report.predictions.tolist(),
            "pooled_metrics": dict(report.pooled_metrics),
            "upper_bound_pooled_metrics": dict(
                report.upper_bound_pooled_metrics
            ),
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "held_out_indices": fold.held_out_indices.tolist(),
                    "selected_weights": fold.selected_weights.tolist(),
                    "candidate_metrics": dict(fold.candidate_metrics),
                    "upper_bound_metrics": dict(fold.upper_bound_metrics),
                    "search_metrics": dict(fold.search_metrics),
                    "slope": fold.slope,
                    "intercept": fold.intercept,
                    "stage_counts": list(fold.stage_counts),
                    "searched_count": fold.searched_count,
                }
                for fold in report.folds
            ],
            "fold_weights": report.fold_weights.tolist(),
            "deploy_weights": report.deploy_weights.tolist(),
            "gate": {
                "passed": report.gate.passed,
                "reasons": list(report.gate.reasons),
            },
            "validation_available": report.validation_available.tolist(),
            "report_hash": report.report_hash,
        }
        return report, payload

    def _formal(self, family):
        from dta.anchored_expert_fusion_v5 import training
        base = OfficialEvaluatorFormalContractTest()
        records, train, valid, _ = base._records([family])
        report, payload = self._report(family, passed=True)
        valid = json.loads(json.dumps(valid))
        valid["count"] = len(report.validation_keys)
        valid["ordered_key_sha256"] = hashlib.sha256(
            json.dumps(
                list(report.validation_keys), separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        spec = training.StaticSpec(
            family=family,
            deploy_weights=tuple(report.deploy_weights.tolist()),
            rank_slope=(1.0 if family == "raw" else 2.0),
            rank_intercept=(0.0 if family == "raw" else -.5),
            cross_fit_report_hash=report.report_hash,
            train_cache_identity=train,
            valid_cache_identity=valid,
        )
        spec_hash = training.static_spec_sha256(spec)
        converted = []
        for record in records:
            item = record.to_dict()
            lineage = dict(item["lineage_hashes"])
            lineage["valid_manifest"] = valid["manifest_sha256"]
            lineage["valid_ordered_keys"] = valid["ordered_key_sha256"]
            if record.type == "expert":
                lineage["prediction_source"] = valid["files"]["features.npz"]
            else:
                lineage["train_manifest"] = train["manifest_sha256"]
                lineage["train_ordered_keys"] = train["ordered_key_sha256"]
                lineage["cross_fit_report"] = report.report_hash
                lineage["static_spec_sha256"] = spec_hash
            item["lineage_hashes"] = lineage
            converted.append(CandidateRecord.from_dict(item))
        records = tuple(converted)
        selection = base._selection(records)
        refs = base._refs(records[0])
        failed_family = "rank" if family == "raw" else "raw"
        _, failed = self._report(failed_family, passed=False)
        reports = {family: payload, failed_family: failed}
        return (
            records, selection, refs, {"train": train, "valid": valid},
            reports, {family: spec.to_dict()},
        )

    def test_exact_task2_report_hash_and_static_binding_accept(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        for family in ("raw", "rank"):
            with self.subTest(family=family):
                records, selection, refs, identities, reports, specs = self._formal(family)
                actual = evaluate_test._validate_formal_package(
                    selection, [item.to_dict() for item in records], refs,
                    identities, reports, specs, sealed_fold_map(reports),
                )
                self.assertEqual(len(actual[0]), 25)

    def test_report_and_static_binding_mutations_fail_closed(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        from dta.anchored_expert_fusion_v5 import training
        for family in ("raw", "rank"):
            records, selection, refs, identities, reports, specs = self._formal(family)
            cases = []
            changed = json.loads(json.dumps(reports))
            changed[family]["predictions"][0] += .01
            cases.append(("report_contents", changed, specs, identities))
            changed = json.loads(json.dumps(reports))
            changed[family]["report_hash"] = HASH_C
            cases.append(("report_hash", changed, specs, identities))
            changed_specs = json.loads(json.dumps(specs))
            changed_specs[family]["cross_fit_report_hash"] = HASH_C
            cases.append(("spec_crossfit", reports, changed_specs, identities))
            changed_specs = json.loads(json.dumps(specs))
            changed_specs[family]["deploy_weights"] = [.4, .4, .2]
            cases.append(("weights", reports, changed_specs, identities))
            if family == "rank":
                changed_specs = json.loads(json.dumps(specs))
                changed_specs[family]["rank_slope"] += .1
                cases.append(("rank_slope", reports, changed_specs, identities))
                changed_specs = json.loads(json.dumps(specs))
                changed_specs[family]["rank_intercept"] += .1
                cases.append(("rank_intercept", reports, changed_specs, identities))
            changed_identities = json.loads(json.dumps(identities))
            changed_identities["train"]["manifest_sha256"] = HASH_C
            cases.append(("identity", reports, specs, changed_identities))
            missing = json.loads(json.dumps(reports))
            missing[family].pop("folds")
            cases.append(("missing_folds", missing, specs, identities))
            for name, report_values, spec_values, identity_values in cases:
                with self.subTest(family=family, mutation=name), \
                     self.assertRaises(ValueError):
                    evaluate_test._validate_formal_package(
                        selection, [item.to_dict() for item in records], refs,
                        identity_values, report_values, spec_values, sealed_fold_map(reports),
                    )

    def test_missing_static_spec_hash_lineage_is_fatal(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        records, selection, refs, identities, reports, specs = self._formal("raw")
        manifest = [item.to_dict() for item in records]
        manifest[3]["lineage_hashes"].pop("static_spec_sha256")
        with self.assertRaisesRegex(ValueError, "static_spec"):
            evaluate_test._validate_formal_package(
                selection, manifest, refs, identities, reports, specs, sealed_fold_map(reports),
            )


class OfficialEvaluatorConcurrentReservationTest(unittest.TestCase):
    def test_two_real_threads_allow_exactly_one_evaluation(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from dta.anchored_expert_fusion_v5 import evaluate_test
        fixture = OfficialEvaluatorReconstructionTest()
        cache = fixture._cache()
        expected = evaluate_test.ExpectedTestIdentity(
            len(cache.keys), cache.metadata["manifest_sha256"],
            cache.metadata["ordered_key_sha256"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "official_output"
            patches = list(fixture._evaluation_mocks(
                root, cache, cache.predictions[:, 0]
            ))
            barrier = threading.Barrier(2)
            patches[2] = mock.patch.object(
                evaluate_test, "_fold_keys",
                side_effect=lambda package: (barrier.wait(), ("v",))[1],
            )
            def run_once():
                try:
                    return evaluate_test._evaluate_impl(
                        "package", "cache", output,
                        expected_identity=expected,
                    )
                except RuntimeError as error:
                    return error
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    outcomes = list(executor.map(lambda _: run_once(), range(2)))
            successes = [value for value in outcomes if isinstance(value, dict)]
            failures = [value for value in outcomes if isinstance(value, RuntimeError)]
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(failures), 1)
            self.assertRegex(str(failures[0]), "in progress|already evaluated")
            self.assertTrue((output / "result.json").is_file())
            self.assertTrue((output / "evaluation.complete").is_file())
            self.assertFalse((output / "evaluation.start").exists())
            persisted = json.loads((output / "result.json").read_text())
            self.assertEqual(persisted, successes[0])


class OfficialEvaluatorIdentityFoldBindingTest(unittest.TestCase):
    def _values(self, root, family="raw"):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        values = OfficialEvaluatorCrossFitBindingTest()._formal(family)
        records, selection, refs, identities, reports, specs = values
        source = reports["raw"]
        package = root / "validation_package"
        package.mkdir()
        np.savez_compressed(
            package / "fold_map.npz",
            keys=np.asarray(source["validation_keys"], dtype=np.str_),
            fold_ids=np.asarray(source["fold_ids"], dtype=np.int64),
        )
        fold_map = evaluate_test._load_fold_map(package)
        return values, fold_map

    def test_unchanged_task1_identity_and_fold_map_are_accepted(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        with tempfile.TemporaryDirectory() as temporary:
            values, fold_map = self._values(Path(temporary))
            records, selection, refs, identities, reports, specs = values
            actual = evaluate_test._validate_formal_package(
                selection, [item.to_dict() for item in records], refs,
                identities, reports, specs, fold_map,
            )
            self.assertEqual(len(actual[0]), 25)
            self.assertEqual(fold_map.keys, tuple(reports["raw"]["validation_keys"]))
            self.assertTrue(np.array_equal(
                fold_map.fold_ids,
                np.asarray(reports["raw"]["fold_ids"], dtype=np.int64),
            ))

    def test_identity_schema_count_and_every_hash_are_strict(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        with tempfile.TemporaryDirectory() as temporary:
            values, fold_map = self._values(Path(temporary))
            records, selection, refs, identities, reports, specs = values
            cases = []
            for split in ("train", "valid"):
                changed = json.loads(json.dumps(identities))
                changed[split]["split"] = "valid" if split == "train" else "train"
                cases.append((split + "_split", changed))
                for count in (True, 0, -1):
                    changed = json.loads(json.dumps(identities))
                    changed[split]["count"] = count
                    cases.append((split + "_count_" + str(count), changed))
                changed = json.loads(json.dumps(identities))
                changed[split]["count"] += 1
                cases.append((split + "_changed_count", changed))
                for field in ("manifest_sha256", "ordered_key_sha256"):
                    changed = json.loads(json.dumps(identities))
                    changed[split][field] = "Z" * 64
                    cases.append((split + "_" + field, changed))
                for file_name in (
                    "features.npz", "metadata.json", "coverage.json",
                    "normalization.json",
                ):
                    changed = json.loads(json.dumps(identities))
                    changed[split]["files"][file_name] = "g" * 64
                    cases.append((split + "_" + file_name, changed))
                changed = json.loads(json.dumps(identities))
                changed[split]["files"].pop("coverage.json")
                cases.append((split + "_missing_file", changed))
                changed = json.loads(json.dumps(identities))
                changed[split]["files"]["extra.bin"] = HASH_A
                cases.append((split + "_extra_file", changed))
            for name, changed in cases:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    evaluate_test._validate_formal_package(
                        selection, [item.to_dict() for item in records], refs,
                        changed, reports, specs, fold_map,
                    )

    def test_fold_map_keys_ids_hash_and_report_binding_are_strict(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        from dta.anchored_expert_fusion_v5 import static_fusion
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values, fold_map = self._values(root)
            records, selection, refs, identities, reports, specs = values
            package = root / "validation_package"
            keys = list(fold_map.keys)
            ids = fold_map.fold_ids.copy()
            mutations = {
                "reordered_keys": (list(reversed(keys)), ids),
                "changed_ids": (keys, np.roll(ids, 1)),
            }
            for name, (changed_keys, changed_ids) in mutations.items():
                with self.subTest(name=name):
                    np.savez_compressed(
                        package / "fold_map.npz",
                        keys=np.asarray(changed_keys, dtype=np.str_),
                        fold_ids=np.asarray(changed_ids, dtype=np.int64),
                    )
                    changed_fold_map = evaluate_test._load_fold_map(package)
                    with self.assertRaises(ValueError):
                        evaluate_test._validate_formal_package(
                            selection, [item.to_dict() for item in records],
                            refs, identities, reports, specs,
                            changed_fold_map,
                        )
            changed_reports = json.loads(json.dumps(reports))
            report_object = evaluate_test._load_crossfit_report(
                changed_reports["rank"], "rank"
            )
            changed_hash = HASH_C
            changed_object = replace(
                report_object, fold_map_hash=changed_hash, report_hash=""
            )
            changed_reports["rank"]["fold_map_hash"] = changed_hash
            changed_reports["rank"]["report_hash"] = (
                static_fusion.report_sha256(changed_object)
            )
            with self.assertRaises(ValueError):
                evaluate_test._validate_formal_package(
                    selection, [item.to_dict() for item in records], refs,
                    identities, changed_reports, specs, fold_map,
                )

    def test_fold_map_npz_exact_schema_and_population(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            cases = {
                "extra": {
                    "keys": np.asarray(["a"]),
                    "fold_ids": np.asarray([0]),
                    "extra": np.asarray([1]),
                },
                "duplicate": {
                    "keys": np.asarray(["a", "a"]),
                    "fold_ids": np.asarray([0, 1]),
                },
                "length": {
                    "keys": np.asarray(["a", "b"]),
                    "fold_ids": np.asarray([0]),
                },
                "five_folds": {
                    "keys": np.asarray(["a", "b", "c", "d", "e"]),
                    "fold_ids": np.asarray([0, 1, 2, 3, 3]),
                },
            }
            for name, arrays in cases.items():
                with self.subTest(name=name):
                    np.savez_compressed(package / "fold_map.npz", **arrays)
                    with self.assertRaises(ValueError):
                        evaluate_test._load_fold_map(package)

class Task56QualityFixesTest(OfficialEvaluatorReconstructionTest):
    def test_verified_bytes_are_fd_bound_and_hash_checked(self):
        from dta.anchored_expert_fusion_v5 import selection
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "payload.bin"
            path.write_bytes(b"original")
            expected = hashlib.sha256(b"original").hexdigest()
            original_read = selection.os.read
            replaced = {"done": False}

            def replace_after_read(fd, size):
                value = original_read(fd, size)
                if value and not replaced["done"]:
                    replacement = root / "replacement.bin"
                    replacement.write_bytes(b"replacement")
                    replacement.replace(path)
                    replaced["done"] = True
                return value

            with mock.patch.object(selection.os, "read", side_effect=replace_after_read):
                value = selection.read_verified_bytes(
                    path, expected_sha256=expected, expected_size=8
                )
            self.assertEqual(value, b"original")
            link = root / "link.bin"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                selection.read_verified_bytes(link, expected_sha256=expected)

    def test_verified_torch_load_enforces_weights_only_and_never_executes_pickle(self):
        from dta.anchored_expert_fusion_v5 import selection
        import pickle
        side_effect = Path(tempfile.gettempdir()) / "v5-quality-pickle-side-effect"
        try:
            side_effect.unlink()
        except FileNotFoundError:
            pass

        class Malicious:
            def __reduce__(self):
                return (Path.write_text, (side_effect, "executed"))

        payload = pickle.dumps(Malicious())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with self.assertRaises(ValueError):
                selection.load_verified_torch(
                    path, expected_sha256=digest, map_location="cpu"
                )
        self.assertFalse(side_effect.exists())

    def test_rank_ensemble_artifact_uses_sealed_affine_calibration(self):
        from dta.anchored_expert_fusion_v5 import selection
        with tempfile.TemporaryDirectory() as temporary:
            records = []
            labels = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
            keys = tuple("v%d" % index for index in range(5))
            for seed, values in ((1, [0, 0, 1, 1, 2]), (7, [0, 1, 1, 2, 2])):
                records.append(candidate(
                    "residual_rank_25_%d" % seed, "residual", family="rank",
                    parent="static_rank", budget=.25, seed=seed,
                    predictions=np.asarray(values, dtype=np.float64),
                ))
            result = selection.build_residual_ensembles(
                records, labels, keys, temporary,
                expected_families=("rank",), expected_budgets=(.25,),
                expected_seeds=(1, 7), rank_calibrations={"rank": (4.0, 1.0)},
            )
            rank = next(item for item in result if item.type == "rank_ensemble")
            with np.load(Path(temporary) / rank.prediction_path, allow_pickle=False) as archive:
                actual = np.asarray(archive["predictions"], dtype=np.float32)
            top = sorted(records, key=selection.validation_candidate_key)
            expected = ((selection.rank_percentiles(top[0].predictions) +
                         selection.rank_percentiles(top[1].predictions)) / 2.0) * 4.0 + 1.0
            np.testing.assert_array_equal(actual, expected.astype(np.float32))
            self.assertEqual(
                rank.validation_metrics,
                selection._prediction_metrics(labels, actual.astype(np.float64)),
            )

    def test_persisted_float32_predictions_and_result_digest_are_exact(self):
        from dta.anchored_expert_fusion_v5 import evaluate_test
        cache = self._cache()
        expected = evaluate_test.ExpectedTestIdentity(
            len(cache.keys), cache.metadata["manifest_sha256"],
            cache.metadata["ordered_key_sha256"],
        )
        predictions = np.asarray(cache.predictions[:, 1], dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = list(self._evaluation_mocks(root, cache, predictions))
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = evaluate_test._evaluate_impl(
                    "package", "cache", root / "official_output",
                    expected_identity=expected,
                )
            with np.load(root / "official_output" / "predictions.npz", allow_pickle=False) as archive:
                persisted = np.asarray(archive["predictions"])
            self.assertEqual(persisted.dtype, np.float32)
            self.assertEqual(
                result["prediction_sha256"],
                hashlib.sha256((root / "official_output" / "predictions.npz").read_bytes()).hexdigest(),
            )
            recomputed = evaluate_test._metrics(cache.labels, persisted, "persisted")
            self.assertEqual(result["metrics"], recomputed)

if __name__ == "__main__":
    unittest.main()
