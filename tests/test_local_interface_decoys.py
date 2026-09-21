import copy
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_contact_stats(jaccard, ratio=0.8, coverage_retained=True):
    return {
        "native_contact_mass": 10.0,
        "decoy_contact_mass": 10.0 * ratio,
        "contact_mass_ratio": ratio,
        "soft_jaccard": jaccard,
        "native_neighbor_count": 2,
        "decoy_neighbor_count": 2,
        "coverage_retained": coverage_retained,
        "contact_pattern_nonempty": True,
    }


def make_overlap_stats(hard_clash=False):
    return {
        "minimum_distance_ratio": 0.9,
        "max_overlap_severity": 0.1,
        "sum_overlap_severity": 0.1,
        "count_below_0_9": 0,
        "count_below_0_8": 0,
        "has_hard_clash": hard_clash,
    }


def make_transform(index):
    return {
        "center": [0.0, 0.0, 0.0],
        "rotation_matrix": np.eye(3).tolist(),
        "translation": [float(index), 0.0, 0.0],
    }


def make_decoy(stratum, index):
    metrics = {
        "low": (0.7, 0.80),
        "medium": (1.3, 0.65),
        "high": (2.0, 0.50),
    }
    rmsd, jaccard = metrics[stratum]
    return {
        "transform": make_transform(index),
        "rmsd_sym": rmsd,
        "contact_stats": make_contact_stats(jaccard),
        "overlap_stats": make_overlap_stats(),
        "stratum": stratum,
        "failure_reason": None,
    }


def make_valid_decoy_set():
    decoys = []
    index = 1
    for stratum in ("low", "medium", "high"):
        for _ in range(4):
            decoys.append(make_decoy(stratum, index))
            index += 1
    return {
        "schema_version": 1,
        "key": "complex_a",
        "seed": 42,
        "minimum_total": 12,
        "minimum_per_stratum": 4,
        "native_contact_mass": 10.0,
        "decoys": decoys,
    }


def make_grid_record(key="grid"):
    grid = np.asarray(
        [
            [x, y, 3.8]
            for x in np.arange(-20.0, 20.1, 2.3)
            for y in np.arange(-20.0, 20.1, 2.3)
        ],
        dtype=np.float64,
    )
    return {
        "key": key,
        "ligand_coords": np.array([[0.0, 0.0, 0.0]]),
        "ligand_atomic_numbers": np.array([6]),
        "ligand_automorphisms": np.array([[0]]),
        "protein_coords": grid,
        "protein_element_symbol": np.array(["C"] * len(grid)),
    }


def write_native_records(path, records):
    environment = lmdb.open(
        str(path),
        map_size=16 * 1024 * 1024,
        subdir=True,
    )
    try:
        with environment.begin(write=True) as transaction:
            for key, record in records.items():
                transaction.put(
                    key.encode("utf-8"),
                    pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
                )
    finally:
        environment.close()


def write_decoy_manifest(path, native_lmdb, keys, split="train"):
    manifest = {
        "dataset_type": "pdbbind_lmdb",
        "task": "dta_regression",
        "split": split,
        "split_method": "unit_test",
        "lmdb_path": str(native_lmdb),
        "samples": [
            {"lmdb_key": key, "label": float(index + 1)}
            for index, key in enumerate(keys)
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


class TestSymmetryCorrectedGeometry(unittest.TestCase):
    def test_symmetry_rmsd_permits_atom_mapping_but_not_realignment(self):
        from data_utils.local_interface_decoys import symmetry_corrected_rmsd

        native = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        swapped = native[[1, 0]]
        mappings = np.array([[0, 1], [1, 0]], dtype=np.int64)

        self.assertAlmostEqual(
            symmetry_corrected_rmsd(native, swapped, mappings), 0.0
        )
        self.assertGreater(
            symmetry_corrected_rmsd(native, native + 2.0, mappings), 1.0
        )

    def test_symmetry_rmsd_rejects_invalid_coordinates_and_mappings(self):
        from data_utils.local_interface_decoys import symmetry_corrected_rmsd

        coords = np.zeros((2, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "finite"):
            symmetry_corrected_rmsd(
                coords, np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                [[0, 1]],
            )
        with self.assertRaisesRegex(ValueError, "permutations"):
            symmetry_corrected_rmsd(coords, coords, [[0, 0]])

    def test_random_transform_is_deterministic_rigid_and_reconstructable(self):
        from data_utils.local_interface_decoys import (
            apply_rigid_transform,
            random_rigid_transform,
        )

        coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        left, left_metadata = random_rigid_transform(coords, seed=17)
        right, right_metadata = random_rigid_transform(coords, seed=17)

        np.testing.assert_allclose(left, right)
        self.assertEqual(left_metadata, right_metadata)
        np.testing.assert_allclose(
            np.linalg.norm(coords[:, None] - coords[None, :], axis=-1),
            np.linalg.norm(left[:, None] - left[None, :], axis=-1),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            apply_rigid_transform(coords, left_metadata), left, atol=1e-7
        )


class TestContactAndOverlapStatistics(unittest.TestCase):
    def test_soft_contacts_report_mass_jaccard_and_retained_coverage(self):
        from data_utils.local_interface_decoys import soft_contact_statistics

        protein = np.array([[0.0, 0.0, 0.0]])
        native = np.array([[5.0, 0.0, 0.0], [5.5, 0.0, 0.0]])
        decoy = np.array([[5.5, 0.0, 0.0], [7.0, 0.0, 0.0]])

        stats = soft_contact_statistics(protein, native, decoy)

        self.assertAlmostEqual(stats["native_contact_mass"], 1.5, places=6)
        self.assertAlmostEqual(stats["decoy_contact_mass"], 0.5, places=6)
        self.assertAlmostEqual(stats["contact_mass_ratio"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(stats["soft_jaccard"], 1.0 / 3.0, places=6)
        self.assertEqual(stats["native_neighbor_count"], 2)
        self.assertEqual(stats["decoy_neighbor_count"], 1)
        self.assertFalse(stats["coverage_retained"])
        self.assertTrue(stats["contact_pattern_nonempty"])

    def test_coverage_only_requires_atoms_having_native_weighted_contacts(self):
        from data_utils.local_interface_decoys import soft_contact_statistics

        protein = np.array([[0.0, 0.0, 0.0]])
        native = np.array([[5.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        decoy = np.array([[5.5, 0.0, 0.0], [20.0, 0.0, 0.0]])

        stats = soft_contact_statistics(protein, native, decoy)

        self.assertTrue(stats["coverage_retained"])
        self.assertEqual(stats["native_neighbor_count"], 1)
        self.assertEqual(stats["decoy_neighbor_count"], 1)

    def test_contact_cutoff_at_exactly_six_angstrom_is_zero(self):
        from data_utils.local_interface_decoys import soft_contact_statistics

        protein = np.array([[0.0, 0.0, 0.0]])
        native = np.array([[5.0, 0.0, 0.0]])
        decoy = np.array([[6.0, 0.0, 0.0]])

        stats = soft_contact_statistics(protein, native, decoy)

        self.assertEqual(stats["native_contact_mass"], 1.0)
        self.assertEqual(stats["decoy_contact_mass"], 0.0)
        self.assertFalse(stats["contact_pattern_nonempty"])
        self.assertFalse(stats["coverage_retained"])

    def test_vdw_statistics_separate_hard_filter_from_probe_thresholds(self):
        from data_utils.local_interface_decoys import vdw_overlap_statistics

        carbon_radius = 1.7
        protein = np.array([[0.0, 0.0, 0.0]])
        ligand = np.array(
            [
                [2.0 * carbon_radius * 0.70, 0.0, 0.0],
                [2.0 * carbon_radius * 0.85, 0.0, 0.0],
                [2.0 * carbon_radius * 0.95, 0.0, 0.0],
            ]
        )

        stats = vdw_overlap_statistics(
            protein, ["C"], ligand, [6, 6, 6]
        )

        self.assertTrue(stats["has_hard_clash"])
        self.assertAlmostEqual(stats["minimum_distance_ratio"], 0.70, places=6)
        self.assertAlmostEqual(stats["max_overlap_severity"], 0.30, places=6)
        self.assertAlmostEqual(stats["sum_overlap_severity"], 0.50, places=6)
        self.assertEqual(stats["count_below_0_9"], 2)
        self.assertEqual(stats["count_below_0_8"], 1)

        boundary = vdw_overlap_statistics(
            protein,
            ["C"],
            np.array([[2.0 * carbon_radius * 0.75, 0.0, 0.0]]),
            [6],
        )
        self.assertFalse(boundary["has_hard_clash"])


class TestFrozenDecoyClassification(unittest.TestCase):
    def test_exact_stratum_boundaries_are_frozen(self):
        from data_utils.local_interface_decoys import classify_decoy

        cases = [
            (0.4, 0.68, "low"),
            (0.999999, 0.949999, "low"),
            (1.0, 0.52, "medium"),
            (1.599999, 0.819999, "medium"),
            (1.6, 0.32, "high"),
            (2.5, 0.679999, "high"),
        ]
        for rmsd, jaccard, expected in cases:
            with self.subTest(rmsd=rmsd, jaccard=jaccard):
                result = classify_decoy(
                    rmsd,
                    make_contact_stats(jaccard),
                    make_overlap_stats(),
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["stratum"], expected)
                self.assertIsNone(result["failure_reason"])

        rejected = [(1.0, 0.82), (1.6, 0.68), (2.500001, 0.50)]
        for rmsd, jaccard in rejected:
            with self.subTest(rejected=(rmsd, jaccard)):
                self.assertFalse(
                    classify_decoy(
                        rmsd,
                        make_contact_stats(jaccard),
                        make_overlap_stats(),
                    )["accepted"]
                )

    def test_contact_ratio_boundaries_are_inclusive(self):
        from data_utils.local_interface_decoys import classify_decoy

        for ratio in (0.6, 1.2):
            with self.subTest(ratio=ratio):
                result = classify_decoy(
                    0.7, make_contact_stats(0.8, ratio=ratio), make_overlap_stats()
                )
                self.assertTrue(result["accepted"])

    def test_classification_rejects_ratio_clash_coverage_and_empty_pattern(self):
        from data_utils.local_interface_decoys import classify_decoy

        ratio = classify_decoy(
            0.7, make_contact_stats(0.8, ratio=0.59), make_overlap_stats()
        )
        self.assertEqual(ratio["failure_reason"], "contact_mass_ratio")

        clash = classify_decoy(
            0.7, make_contact_stats(0.8), make_overlap_stats(hard_clash=True)
        )
        self.assertEqual(clash["failure_reason"], "vdw_hard_clash")

        coverage = classify_decoy(
            0.7,
            make_contact_stats(0.8, coverage_retained=False),
            make_overlap_stats(),
        )
        self.assertEqual(coverage["failure_reason"], "ligand_neighbor_coverage")

        empty_stats = make_contact_stats(0.8)
        empty_stats["contact_pattern_nonempty"] = False
        empty = classify_decoy(0.7, empty_stats, make_overlap_stats())
        self.assertEqual(empty["failure_reason"], "empty_contact_pattern")

    def test_symmetric_chemically_equivalent_pose_is_not_a_decoy(self):
        from data_utils.local_interface_decoys import (
            classify_decoy,
            symmetry_corrected_rmsd,
        )

        native = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        equivalent = native[[1, 0]]
        mappings = np.array([[0, 1], [1, 0]])
        rmsd = symmetry_corrected_rmsd(native, equivalent, mappings)

        result = classify_decoy(
            rmsd, make_contact_stats(0.8), make_overlap_stats()
        )

        self.assertEqual(rmsd, 0.0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["failure_reason"], "outside_frozen_strata")


class TestDecoySetContract(unittest.TestCase):
    def test_validator_requires_twelve_and_four_per_stratum(self):
        from data_utils.local_interface_decoys import validate_decoy_set

        valid = make_valid_decoy_set()
        self.assertIs(validate_decoy_set(valid), valid)

        missing = copy.deepcopy(valid)
        missing["decoys"] = missing["decoys"][:-1]
        with self.assertRaisesRegex(ValueError, "at least 12"):
            validate_decoy_set(missing)

        unbalanced = copy.deepcopy(valid)
        unbalanced["decoys"][0] = make_decoy("medium", 50)
        with self.assertRaisesRegex(ValueError, "at least 4.*low"):
            validate_decoy_set(unbalanced)

    def test_validator_rejects_duplicate_transforms_and_atom_tensors(self):
        from data_utils.local_interface_decoys import validate_decoy_set

        duplicated = make_valid_decoy_set()
        duplicated["decoys"][1]["transform"] = copy.deepcopy(
            duplicated["decoys"][0]["transform"]
        )
        with self.assertRaisesRegex(ValueError, "duplicate transform"):
            validate_decoy_set(duplicated)

        for field in ("ligand_coords", "ligand_mask", "bond_index"):
            atom_tensor = make_valid_decoy_set()
            atom_tensor["decoys"][0][field] = [0]
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError, "transform metadata and statistics only"
                ):
                    validate_decoy_set(atom_tensor)

    def test_validator_rejects_inconsistent_derived_statistics(self):
        from data_utils.local_interface_decoys import validate_decoy_set

        bad_ratio = make_valid_decoy_set()
        bad_ratio["decoys"][0]["contact_stats"]["contact_mass_ratio"] = 0.9
        with self.assertRaisesRegex(ValueError, "contact mass ratio"):
            validate_decoy_set(bad_ratio)

        bad_clash_flag = make_valid_decoy_set()
        overlap = bad_clash_flag["decoys"][0]["overlap_stats"]
        overlap["minimum_distance_ratio"] = 0.70
        overlap["max_overlap_severity"] = 0.30
        overlap["sum_overlap_severity"] = 0.30
        with self.assertRaisesRegex(ValueError, "hard-clash flag"):
            validate_decoy_set(bad_clash_flag)

        bad_jaccard = make_valid_decoy_set()
        bad_jaccard["decoys"][0]["contact_stats"]["soft_jaccard"] = 1.1
        with self.assertRaisesRegex(ValueError, "soft Jaccard"):
            validate_decoy_set(bad_jaccard)

        bad_native_mass = make_valid_decoy_set()
        bad_native_mass["native_contact_mass"] = 9.0
        with self.assertRaisesRegex(ValueError, "native contact mass"):
            validate_decoy_set(bad_native_mass)

    def test_validator_recomputes_statistics_from_native_geometry(self):
        from data_utils.local_interface_decoys import (
            generate_decoy_set,
            validate_decoy_set,
        )

        record = make_grid_record()
        decoy_set = generate_decoy_set(record, seed=42)
        changed = copy.deepcopy(decoy_set)
        changed["decoys"][0]["transform"]["translation"][0] += 0.1

        with self.assertRaisesRegex(ValueError, "geometry"):
            validate_decoy_set(changed, native_record=record)

    def test_validator_rejects_nonfinite_cached_values(self):
        from data_utils.local_interface_decoys import validate_decoy_set

        bad_transform = make_valid_decoy_set()
        bad_transform["decoys"][0]["transform"]["translation"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_decoy_set(bad_transform)

        bad_stat = make_valid_decoy_set()
        bad_stat["decoys"][0]["contact_stats"]["soft_jaccard"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_decoy_set(bad_stat)

    def test_validator_rejects_boolean_numeric_statistics(self):
        from data_utils.local_interface_decoys import validate_decoy_set

        decoy_set = make_valid_decoy_set()
        decoy_set["native_contact_mass"] = True
        for entry in decoy_set["decoys"]:
            stats = entry["contact_stats"]
            stats["native_contact_mass"] = True
            stats["decoy_contact_mass"] = 0.8
            stats["contact_mass_ratio"] = 0.8

        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_decoy_set(decoy_set)

    def test_validator_rejects_float_counts(self):
        from data_utils.local_interface_decoys import validate_decoy_set

        decoy_set = make_valid_decoy_set()
        decoy_set["decoys"][0]["contact_stats"][
            "native_neighbor_count"
        ] = 2.0

        with self.assertRaisesRegex(ValueError, "integer"):
            validate_decoy_set(decoy_set)

    def test_validator_rejects_fractional_minimum_counts(self):
        from data_utils.local_interface_decoys import validate_decoy_set

        decoy_set = make_valid_decoy_set()
        decoy_set["minimum_total"] = 12.9

        with self.assertRaisesRegex(ValueError, "minimum_total.*integer"):
            validate_decoy_set(decoy_set)

    def test_epoch_sampler_is_deterministic_fixed_k_and_stratified(self):
        from data_utils.local_interface_decoys import select_epoch_decoys

        decoy_set = make_valid_decoy_set()
        left = select_epoch_decoys(decoy_set, "complex_a", epoch=4, seed=42)
        right = select_epoch_decoys(decoy_set, "complex_a", epoch=4, seed=42)

        self.assertEqual(left, right)
        self.assertEqual(len(left), 3)
        self.assertEqual(
            [entry["stratum"] for entry in left],
            ["low", "medium", "high"],
        )

        incomplete = copy.deepcopy(decoy_set)
        incomplete["decoys"] = [
            entry for entry in incomplete["decoys"] if entry["stratum"] != "high"
        ]
        with self.assertRaisesRegex(ValueError, "missing stratum high"):
            select_epoch_decoys(incomplete, "complex_a", epoch=4, seed=42)

    def test_generator_builds_deterministic_fixed_stratum_set(self):
        from data_utils.local_interface_decoys import generate_decoy_set

        record = make_grid_record()

        left = generate_decoy_set(record, seed=42)
        right = generate_decoy_set(record, seed=42)
        other_key = generate_decoy_set(make_grid_record("other"), seed=42)

        self.assertEqual(left, right)
        self.assertNotEqual(left["decoys"], other_key["decoys"])
        self.assertEqual(len(left["decoys"]), 12)
        self.assertEqual(
            [entry["stratum"] for entry in left["decoys"]],
            ["low"] * 4 + ["medium"] * 4 + ["high"] * 4,
        )
        self.assertTrue(
            all(entry["failure_reason"] is None for entry in left["decoys"])
        )
        self.assertFalse(any("ligand_coords" in entry for entry in left["decoys"]))

    def test_generator_uses_four_pooled_sixty_four_candidate_budgets(self):
        from data_utils.local_interface_decoys import (
            DecoyGenerationError,
            generate_decoy_set,
        )

        rejected = {
            "accepted": False,
            "stratum": None,
            "failure_reason": "outside_frozen_strata",
        }
        with patch(
            "data_utils.local_interface_decoys.classify_decoy",
            return_value=rejected,
        ) as classifier:
            with self.assertRaises(DecoyGenerationError):
                generate_decoy_set(make_grid_record(), seed=42)

        self.assertEqual(classifier.call_count, 4 * 64)

    def test_native_contact_mass_threshold_is_inclusive(self):
        from data_utils.local_interface_decoys import (
            DecoyGenerationError,
            generate_decoy_set,
        )

        record = {
            "key": "mass_boundary",
            "ligand_coords": np.array([[0.0, 0.0, 0.0]]),
            "ligand_atomic_numbers": np.array([6]),
            "ligand_automorphisms": np.array([[0]]),
            "protein_coords": np.repeat([[5.0, 0.0, 0.0]], 8, axis=0),
            "protein_element_symbol": np.array(["C"] * 8),
        }
        try:
            generate_decoy_set(record, seed=42)
        except DecoyGenerationError:
            pass

    def test_generate_rejects_native_below_contact_mass_threshold(self):
        from data_utils.local_interface_decoys import generate_decoy_set

        record = {
            "key": "empty",
            "ligand_coords": np.array([[0.0, 0.0, 0.0]]),
            "ligand_atomic_numbers": np.array([6]),
            "ligand_automorphisms": np.array([[0]]),
            "protein_coords": np.array([[20.0, 0.0, 0.0]]),
            "protein_element_symbol": np.array(["C"]),
        }

        with self.assertRaisesRegex(ValueError, "native contact mass"):
            generate_decoy_set(record, seed=42)


class TestDecoyCacheBuilder(unittest.TestCase):
    def test_builder_rejects_non_train_manifest(self):
        from scripts.build_local_interface_decoys import (
            build_arg_parser,
            build_decoy_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "valid_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_decoy_manifest(
                manifest_path, root / "native.lmdb", ["valid_a"], split="valid"
            )
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )
            args = build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--native-lmdb",
                    str(root / "native.lmdb"),
                    "--output-lmdb",
                    str(root / "decoys.lmdb"),
                    "--diagnostic-keys",
                    str(diagnostic_path),
                ]
            )

            with self.assertRaisesRegex(ValueError, "split=train"):
                build_decoy_cache(args)

    def test_builder_excludes_diagnostic_keys_and_writes_report(self):
        from scripts.build_local_interface_decoys import (
            build_arg_parser,
            build_decoy_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.lmdb"
            output_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_native_records(
                native_path,
                {
                    "drop": make_grid_record("drop"),
                    "keep": make_grid_record("keep"),
                },
            )
            write_decoy_manifest(
                manifest_path, native_path, ["drop", "keep"]
            )
            diagnostic_path.write_text(
                json.dumps({"keys": ["drop"]}), encoding="utf-8"
            )
            args = build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--native-lmdb",
                    str(native_path),
                    "--output-lmdb",
                    str(output_path),
                    "--diagnostic-keys",
                    str(diagnostic_path),
                    "--seed",
                    "42",
                ]
            )

            report = build_decoy_cache(args)

            self.assertEqual(report["counts"]["diagnostic_excluded"], 1)
            self.assertEqual(report["counts"]["written"], 1)
            self.assertEqual(
                json.loads(
                    (output_path / "decoy_report.json").read_text(
                        encoding="utf-8"
                    )
                ),
                report,
            )
            environment = lmdb.open(
                str(output_path), readonly=True, lock=False, subdir=True
            )
            try:
                with environment.begin(write=False) as transaction:
                    self.assertIsNone(transaction.get(b"drop"))
                    self.assertIsNotNone(transaction.get(b"keep"))
            finally:
                environment.close()

    def test_builder_rejects_invalid_generated_decoy_set(self):
        from scripts.build_local_interface_decoys import (
            build_arg_parser,
            build_decoy_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.lmdb"
            output_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_native_records(
                native_path, {"keep": make_grid_record("keep")}
            )
            write_decoy_manifest(manifest_path, native_path, ["keep"])
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )
            args = build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--native-lmdb",
                    str(native_path),
                    "--output-lmdb",
                    str(output_path),
                    "--diagnostic-keys",
                    str(diagnostic_path),
                ]
            )
            invalid = {
                "schema_version": 1,
                "key": "keep",
                "seed": 42,
                "native_contact_mass": 10.0,
                "decoys": [],
            }

            with patch(
                "scripts.build_local_interface_decoys.generate_decoy_set",
                return_value=invalid,
            ):
                report = build_decoy_cache(args)

            self.assertEqual(report["counts"]["written"], 0)
            self.assertEqual(report["counts"]["errors"], 1)
            environment = lmdb.open(
                str(output_path), readonly=True, lock=False, subdir=True
            )
            try:
                with environment.begin(write=False) as transaction:
                    self.assertIsNone(transaction.get(b"keep"))
            finally:
                environment.close()

    def test_builder_resume_rejects_replaced_native_record(self):
        from scripts import build_local_interface_decoys as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.lmdb"
            output_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_native_records(
                native_path, {"a": make_grid_record("a")}
            )
            write_decoy_manifest(manifest_path, native_path, ["a"])
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )
            parser = builder.build_arg_parser()
            args = parser.parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--native-lmdb",
                    str(native_path),
                    "--output-lmdb",
                    str(output_path),
                    "--diagnostic-keys",
                    str(diagnostic_path),
                ]
            )
            decoy_set = make_valid_decoy_set()
            decoy_set["key"] = "a"
            with patch.object(
                builder, "generate_decoy_set", return_value=decoy_set
            ), patch.object(
                builder, "validate_decoy_set", return_value=decoy_set
            ):
                builder.build_decoy_cache(args)

            replacement = make_grid_record("a")
            replacement["protein_coords"] = replacement[
                "protein_coords"
            ].copy()
            replacement["protein_coords"][0, 0] += 0.25
            write_native_records(native_path, {"a": replacement})

            with self.assertRaisesRegex(
                ValueError, "resume provenance"
            ):
                builder.build_decoy_cache(
                    parser.parse_args(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--native-lmdb",
                            str(native_path),
                            "--output-lmdb",
                            str(output_path),
                            "--diagnostic-keys",
                            str(diagnostic_path),
                            "--resume",
                        ]
                    )
                )

    def test_builder_resume_rejects_changed_max_samples(self):
        from scripts import build_local_interface_decoys as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.lmdb"
            output_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_native_records(
                native_path,
                {
                    "a": make_grid_record("a"),
                    "b": make_grid_record("b"),
                },
            )
            write_decoy_manifest(
                manifest_path, native_path, ["a", "b"]
            )
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )
            parser = builder.build_arg_parser()
            first = parser.parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--native-lmdb",
                    str(native_path),
                    "--output-lmdb",
                    str(output_path),
                    "--diagnostic-keys",
                    str(diagnostic_path),
                    "--max-samples",
                    "1",
                ]
            )
            decoy_set = make_valid_decoy_set()
            decoy_set["key"] = "a"
            with patch.object(
                builder, "generate_decoy_set", return_value=decoy_set
            ), patch.object(
                builder, "validate_decoy_set", return_value=decoy_set
            ):
                builder.build_decoy_cache(first)

            with self.assertRaisesRegex(
                ValueError, "resume provenance"
            ):
                builder.build_decoy_cache(
                    parser.parse_args(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--native-lmdb",
                            str(native_path),
                            "--output-lmdb",
                            str(output_path),
                            "--diagnostic-keys",
                            str(diagnostic_path),
                            "--max-samples",
                            "2",
                            "--resume",
                        ]
                    )
                )

    def test_builder_map_full_is_fatal(self):
        from scripts import build_local_interface_decoys as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.lmdb"
            output_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_native_records(
                native_path, {"a": make_grid_record("a")}
            )
            write_decoy_manifest(manifest_path, native_path, ["a"])
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )
            args = builder.build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--native-lmdb",
                    str(native_path),
                    "--output-lmdb",
                    str(output_path),
                    "--diagnostic-keys",
                    str(diagnostic_path),
                ]
            )
            decoy_set = make_valid_decoy_set()
            decoy_set["key"] = "a"

            with patch.object(
                builder, "generate_decoy_set", return_value=decoy_set
            ), patch.object(
                builder, "validate_decoy_set", return_value=decoy_set
            ), patch.object(
                builder,
                "_write_record",
                create=True,
                side_effect=lmdb.MapFullError("map full"),
            ):
                with self.assertRaises(lmdb.MapFullError):
                    builder.build_decoy_cache(args)

    def test_builder_failure_row_preserves_rejections_and_contact_mass(self):
        from data_utils.local_interface_decoys import DecoyGenerationError
        from scripts import build_local_interface_decoys as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.lmdb"
            output_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_native_records(
                native_path, {"a": make_grid_record("a")}
            )
            write_decoy_manifest(manifest_path, native_path, ["a"])
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )
            args = builder.build_arg_parser().parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--native-lmdb",
                    str(native_path),
                    "--output-lmdb",
                    str(output_path),
                    "--diagnostic-keys",
                    str(diagnostic_path),
                ]
            )
            failure = DecoyGenerationError(
                "insufficient legal transforms",
                {"outside_frozen_strata": 17},
            )

            with patch.object(
                builder, "generate_decoy_set", side_effect=failure
            ):
                report = builder.build_decoy_cache(args)

            row = report["excluded_samples"][0]
            self.assertEqual(
                row["generation_rejection_counts"],
                {"outside_frozen_strata": 17},
            )
            self.assertGreater(row["native_contact_mass"], 0.0)

    def test_builder_resume_clears_transient_failure(self):
        from data_utils.local_interface_decoys import DecoyGenerationError
        from scripts import build_local_interface_decoys as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_path = root / "native.lmdb"
            output_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            write_native_records(
                native_path, {"a": make_grid_record("a")}
            )
            write_decoy_manifest(manifest_path, native_path, ["a"])
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )
            parser = builder.build_arg_parser()
            common = [
                "--manifest",
                str(manifest_path),
                "--native-lmdb",
                str(native_path),
                "--output-lmdb",
                str(output_path),
                "--diagnostic-keys",
                str(diagnostic_path),
            ]
            with patch.object(
                builder,
                "generate_decoy_set",
                side_effect=DecoyGenerationError(
                    "transient", {"outside_frozen_strata": 3}
                ),
            ):
                initial = builder.build_decoy_cache(
                    parser.parse_args(common)
                )
            self.assertEqual(initial["counts"]["errors"], 1)

            decoy_set = make_valid_decoy_set()
            decoy_set["key"] = "a"
            with patch.object(
                builder, "generate_decoy_set", return_value=decoy_set
            ), patch.object(
                builder, "validate_decoy_set", return_value=decoy_set
            ):
                resumed = builder.build_decoy_cache(
                    parser.parse_args(common + ["--resume"])
                )

            self.assertEqual(resumed["counts"]["errors"], 0)
            self.assertEqual(resumed["excluded_samples"], [])
            self.assertEqual(resumed["errors_preview"], [])
            self.assertEqual(
                resumed["generation_rejection_counts"], {}
            )


if __name__ == "__main__":
    unittest.main()
