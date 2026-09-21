import copy
import hashlib
import json
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.interface_decoy_audit import (  # noqa: E402
    HELDOUT_FEATURE_NAMES,
    extract_coarse_matching_features,
    extract_heldout_interface_features,
)
from data_utils import local_interface_decoys_v2 as decoys_v2  # noqa: E402
from scripts import build_local_interface_decoys_v2 as builder_v2  # noqa: E402
from utils.experiment_integrity import (  # noqa: E402
    hash_ordered_keys,
    manifest_fingerprint,
    sha256_file,
)
from utils.interface_diagnostics import extract_shortcut_features  # noqa: E402


def axis_angle_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def make_audit_record():
    ligand = np.asarray(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]],
        dtype=np.float64,
    )
    protein = np.asarray(
        [[0.0, 0.0, 3.0], [1.0, 0.0, 3.2], [0.0, 1.0, 4.0], [2.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    return {
        "ligand_coords": ligand,
        "ligand_heavy_atom_count": 3,
        "ligand_edge_index": np.asarray(
            [[0, 1, 0, 2], [1, 0, 2, 0]], dtype=np.int64
        ),
        "ligand_atomic_numbers": np.asarray([6, 7, 8], dtype=np.int64),
        "protein_coords": protein,
        "protein_parent_residue": np.asarray([0, 0, 1, 2], dtype=np.int64),
        "protein_residue_type": np.asarray([2, 2, 5, 7], dtype=np.int64),
        "protein_element_symbol": np.asarray(["C", "N", "O", "C"]),
        "residue_count": 3,
    }, ligand.copy()


def rotate_record(record, pose, rotation):
    rotated = copy.deepcopy(record)
    rotated["ligand_coords"] = np.asarray(record["ligand_coords"]) @ rotation.T
    rotated["protein_coords"] = np.asarray(record["protein_coords"]) @ rotation.T
    return rotated, np.asarray(pose) @ rotation.T


def permute_aligned_atoms(record, pose):
    ligand_order = np.asarray([2, 0, 1], dtype=np.int64)
    protein_order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    inverse_ligand = np.empty(len(ligand_order), dtype=np.int64)
    inverse_ligand[ligand_order] = np.arange(len(ligand_order))

    permuted = copy.deepcopy(record)
    for field in ("ligand_coords", "ligand_atomic_numbers"):
        permuted[field] = np.asarray(record[field])[ligand_order]
    permuted["ligand_edge_index"] = inverse_ligand[
        np.asarray(record["ligand_edge_index"])
    ]
    for field in (
        "protein_coords",
        "protein_parent_residue",
        "protein_residue_type",
        "protein_element_symbol",
    ):
        permuted[field] = np.asarray(record[field])[protein_order]
    return permuted, np.asarray(pose)[ligand_order]


class InterfaceDecoyHeldoutFeatureTest(unittest.TestCase):
    def test_exact_names_dimension_and_finite_values(self):
        record, pose = make_audit_record()
        values = extract_heldout_interface_features(record, pose)

        self.assertEqual(len(HELDOUT_FEATURE_NAMES), 30)
        self.assertEqual(values.shape, (30,))
        self.assertEqual(values.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(
            tuple(HELDOUT_FEATURE_NAMES[7:27]),
            tuple(f"contact_residue_type_{index}" for index in range(20)),
        )
        self.assertTrue(np.all(np.diff(values[-3:]) <= 0.0))

    def test_heldout_features_are_rotation_and_permutation_invariant(self):
        record, pose = make_audit_record()
        original = extract_heldout_interface_features(record, pose)

        rotation = axis_angle_rotation([1.0, 2.0, 3.0], 0.7)
        rotated_record, rotated_pose = rotate_record(record, pose, rotation)
        np.testing.assert_allclose(
            original,
            extract_heldout_interface_features(rotated_record, rotated_pose),
            atol=1e-10,
            rtol=1e-10,
        )

        permuted_record, permuted_pose = permute_aligned_atoms(record, pose)
        np.testing.assert_allclose(
            original,
            extract_heldout_interface_features(permuted_record, permuted_pose),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_single_atom_single_residue_and_bondless_definitions(self):
        record = {
            "ligand_coords": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            "ligand_heavy_atom_count": 1,
            "ligand_edge_index": np.empty((2, 0), dtype=np.int64),
            "ligand_atomic_numbers": np.asarray([6], dtype=np.int64),
            "protein_coords": np.asarray([[0.0, 0.0, 3.0]], dtype=np.float64),
            "protein_parent_residue": np.asarray([0], dtype=np.int64),
            "protein_residue_type": np.asarray([4], dtype=np.int64),
            "protein_element_symbol": np.asarray(["C"]),
            "residue_count": 1,
        }
        values = extract_heldout_interface_features(
            record, record["ligand_coords"]
        )

        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[2], 1.0)
        self.assertEqual(values[3], 0.0)
        self.assertEqual(values[4], 0.0)
        self.assertEqual(values[6], 1.0)
        composition = values[7:27]
        self.assertEqual(composition[4], 1.0)
        self.assertAlmostEqual(float(np.sum(composition)), 1.0)
        np.testing.assert_allclose(values[-3:], [1.0, 0.0, 0.0], atol=1e-12)

    def test_malformed_parent_and_residue_classes_are_rejected(self):
        record, pose = make_audit_record()
        cases = []
        value = copy.deepcopy(record)
        value["protein_parent_residue"] = np.asarray(
            [0.0, 0.0, 1.0, 2.0], dtype=np.float64
        )
        cases.append((value, "parent"))
        value = copy.deepcopy(record)
        value["protein_parent_residue"][0] = -1
        cases.append((value, "parent"))
        value = copy.deepcopy(record)
        value["protein_parent_residue"][0] = value["residue_count"]
        cases.append((value, "parent"))
        value = copy.deepcopy(record)
        value["protein_residue_type"][1] = 3
        cases.append((value, "consistent"))
        value = copy.deepcopy(record)
        value["protein_residue_type"][0] = 20
        cases.append((value, "residue"))

        for malformed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    extract_heldout_interface_features(malformed, pose)

    def test_zero_distance_contact_and_bad_edges_are_rejected(self):
        record, pose = make_audit_record()
        record["protein_coords"][0] = pose[0]
        with self.assertRaisesRegex(ValueError, "zero-distance"):
            extract_heldout_interface_features(record, pose)

        record, pose = make_audit_record()
        record["ligand_edge_index"][0, 0] = 99
        with self.assertRaisesRegex(ValueError, "edge"):
            extract_heldout_interface_features(record, pose)

    def test_coarse_wrapper_is_exact_v1_compatibility(self):
        record, pose = make_audit_record()
        np.testing.assert_array_equal(
            extract_coarse_matching_features(record, pose),
            extract_shortcut_features(record, pose),
        )


def make_probe_rows(seed=137):
    generator = np.random.default_rng(seed)
    rows = []
    labels = []
    groups = []
    for group_index in range(10):
        group_shift = generator.normal(scale=0.15, size=3)
        for pose_index in range(4):
            label = int(pose_index == 0)
            rows.append(
                group_shift
                + generator.normal(scale=0.35, size=3)
                + np.asarray([1.0 * label, 0.4 * label, 0.0])
            )
            labels.append(label)
            groups.append(f"target-{group_index}")
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


def make_data_report(matched_auroc, random_auroc, complete_count, requested=100):
    return {
        "cohorts": {
            "paired_intersection": {
                "random_physical": {
                    "coarse": {
                        "nonlinear_cv_auroc": random_auroc,
                    },
                    "heldout": {"nonlinear_cv_auroc": 1.0},
                    "interface": {"auroc": 0.0},
                },
                "matched": {
                    "coarse": {
                        "nonlinear_cv_auroc": matched_auroc,
                    },
                    "heldout": {"nonlinear_cv_auroc": 1.0},
                    "interface": {"auroc": 0.0},
                },
            }
        },
        "bank_completion": {
            "requested_denominator": requested,
            "matched": {"complete_count": complete_count},
        },
    }



def diagnostic_transform(x_offset, y_offset=0.0):
    return {
        "center": [0.0, 0.0, 0.0],
        "rotation_matrix": np.eye(3, dtype=np.float64).tolist(),
        "translation": [float(x_offset), float(y_offset), 0.0],
    }


def diagnostic_fingerprint(transform):
    return hashlib.sha256(
        json.dumps(transform, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def diagnostic_native_record(key, index, split="train"):
    shift = float(index) * 0.2
    ligand = np.asarray(
        [
            [1.5 + shift, -0.2, 0.0],
            [2.4 + shift, -0.2, 0.0],
            [1.9 + shift, 0.7, 0.0],
        ],
        dtype=np.float64,
    )
    protein = np.asarray(
        [
            [0.0 + shift, -1.0, 0.0],
            [0.0 + shift, 0.0, 0.0],
            [0.0 + shift, 1.0, 0.0],
            [0.4 + shift, -0.5, 1.2],
            [0.4 + shift, 0.5, 1.2],
            [-0.4 + shift, -0.5, -1.2],
            [-0.4 + shift, 0.5, -1.2],
            [0.7 + shift, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    return {
        "key": key,
        "split": str(split),
        "target_signature": "target_group_%02d" % index,
        "ligand_coords": ligand,
        "ligand_atomic_numbers": np.asarray([6, 7, 8], dtype=np.int64),
        "ligand_automorphisms": np.asarray([[0, 1, 2]], dtype=np.int64),
        "ligand_heavy_atom_count": 3,
        "ligand_edge_index": np.asarray(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
        ),
        "protein_coords": protein,
        "protein_element_symbol": np.asarray(["C"] * len(protein)),
        "protein_parent_residue": np.arange(len(protein), dtype=np.int64),
        "protein_residue_type": np.asarray(
            [(index + offset) % 20 for offset in range(len(protein))],
            dtype=np.int64,
        ),
        "residue_count": int(len(protein)),
    }


def diagnostic_match_values(match_pass):
    if match_pass:
        return {
            "contact_mass_ratio": 1.0,
            "delta_centroid_distance": 0.0,
            "delta_minimum_distance": 0.0,
            "delta_max_overlap_severity": 0.0,
            "delta_sum_overlap_per_ligand_atom": 0.0,
            "delta_count_below_0_9_per_ligand_atom": 0.0,
            "delta_count_below_0_8_per_ligand_atom": 0.0,
        }
    return {
        "contact_mass_ratio": 1.30,
        "delta_centroid_distance": 0.0,
        "delta_minimum_distance": 0.0,
        "delta_max_overlap_severity": 0.0,
        "delta_sum_overlap_per_ligand_atom": 0.0,
        "delta_count_below_0_9_per_ligand_atom": 0.0,
        "delta_count_below_0_8_per_ligand_atom": 0.0,
    }


def diagnostic_decoy_entry(bank, stratum, index, match_pass):
    stratum_base = {"low": 0.45, "medium": 1.05, "high": 1.7}[stratum]
    jaccard = {"low": 0.80, "medium": 0.62, "high": 0.44}[stratum]
    transform = diagnostic_transform(
        stratum_base + 0.03 * index,
        0.01 * (1 if bank == "matched" else 2),
    )
    values = diagnostic_match_values(match_pass)
    return {
        "transform": transform,
        "transform_fingerprint": diagnostic_fingerprint(transform),
        "rmsd_sym": stratum_base,
        "contact_stats": {
            "native_contact_mass": 10.0,
            "decoy_contact_mass": 10.0 * values["contact_mass_ratio"],
            "contact_mass_ratio": values["contact_mass_ratio"],
            "soft_jaccard": jaccard,
            "native_neighbor_count": 3,
            "decoy_neighbor_count": 3,
            "coverage_retained": True,
            "contact_pattern_nonempty": True,
        },
        "overlap_stats": {
            "minimum_distance_ratio": 1.0,
            "max_overlap_severity": 0.0,
            "sum_overlap_severity": 0.0,
            "count_below_0_9": 0,
            "count_below_0_8": 0,
            "has_hard_clash": False,
        },
        "stratum": stratum,
        "failure_reason": None,
        "match_values": values,
        "match_distance": decoys_v2.normalized_match_distance(values, stratum),
        "match_pass": bool(match_pass),
    }


def diagnostic_decoy_set(record, seed, random_complete=True, matched_complete=True):
    banks = {}
    for bank, complete, entry_match_pass in (
        ("random_physical", random_complete, matched_complete),
        ("matched", matched_complete, True),
    ):
        rows = []
        if complete:
            for stratum in decoys_v2.STRATUM_ORDER:
                rows.extend(
                    diagnostic_decoy_entry(bank, stratum, index, entry_match_pass)
                    for index in range(decoys_v2.RETAINED_PER_STRATUM)
                )
        banks[bank] = {
            "complete": bool(complete),
            "failure_reason": (
                None
                if complete
                else "insufficient_%s_candidates:low,medium,high" % bank
            ),
            "decoys": rows,
        }

    proposal_counts = {}
    rejection_counts = {}
    for stratum in decoys_v2.STRATUM_ORDER:
        physical = decoys_v2.RETAINED_PER_STRATUM if random_complete else 0
        matched = decoys_v2.RETAINED_PER_STRATUM if matched_complete else 0
        proposal_counts[stratum] = {
            "proposed": decoys_v2.PROPOSALS_PER_STRATUM,
            "physical_accepted": physical,
            "match_accepted": matched,
            "shortlisted": min(decoys_v2.SHORTLIST_SIZE, matched),
            "random_physical_retained": (
                decoys_v2.RETAINED_PER_STRATUM if random_complete else 0
            ),
            "matched_retained": (
                decoys_v2.RETAINED_PER_STRATUM if matched_complete else 0
            ),
        }
        rejection_counts[stratum] = {}
        physical_rejections = decoys_v2.PROPOSALS_PER_STRATUM - physical
        if physical_rejections:
            rejection_counts[stratum]["outside_frozen_strata"] = physical_rejections
        match_rejections = physical - matched
        if match_rejections:
            rejection_counts[stratum]["match_contact_mass_ratio"] = match_rejections

    value = {
        "schema_version": decoys_v2.DECOY_SCHEMA_VERSION,
        "key": record["key"],
        "seed": int(seed),
        "proposal_count_per_stratum": decoys_v2.PROPOSALS_PER_STRATUM,
        "automorphism_count": len(record["ligand_automorphisms"]),
        "automorphism_sha256": decoys_v2.automorphism_digest(
            record["ligand_automorphisms"]
        ),
        "proposal_fingerprint_sha256": "a" * 64,
        "banks": banks,
        "proposal_counts": proposal_counts,
        "rejection_counts": rejection_counts,
    }
    decoys_v2.validate_decoy_set_v2(value)
    return value


class DiagnosticV2EvaluatorFixture:
    def __init__(
        self,
        key_count=7,
        complete_count=5,
        max_samples=None,
        record_schema_version=decoys_v2.DECOY_SCHEMA_VERSION,
        native_split="train",
        nontrain_native_index=None,
        wrong_native_key_index=None,
        malformed_matched_only=False,
    ):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "train_manifest.json"
        self.native_path = self.root / "native_lmdb"
        self.diagnostic_path = self.root / "diagnostic_keys.json"
        self.decoy_path = self.root / "decoys_v2"
        self.output_path = self.root / "evaluation"
        self.seed = 23
        self.keys = ["diag_%02d" % index for index in range(key_count)]
        self.complete_count = int(complete_count)
        self.max_samples = max_samples
        self.record_schema_version = int(record_schema_version)
        self.native_split = str(native_split)
        self.nontrain_native_index = nontrain_native_index
        self.wrong_native_key_index = wrong_native_key_index
        self.malformed_matched_only = bool(malformed_matched_only)
        self.records = {
            key: diagnostic_native_record(
                key,
                index,
                split=(
                    "valid"
                    if index == self.nontrain_native_index
                    else self.native_split
                ),
            )
            for index, key in enumerate(self.keys)
        }
        if self.wrong_native_key_index is not None:
            wrong_key = self.keys[self.wrong_native_key_index]
            self.records[wrong_key]["key"] = "wrong_native_record_key"
        self._write_json(
            self.manifest_path,
            {
                "split": "train",
                "samples": [
                    {"lmdb_key": key, "split": "train", "label": float(index)}
                    for index, key in enumerate(self.keys)
                ],
            },
        )
        self._write_native()
        self._write_json(
            self.diagnostic_path,
            {
                "schema_version": 1,
                "manifest": manifest_fingerprint(self.manifest_path),
                "keys": list(self.keys),
            },
        )
        self._build_decoys()

    def cleanup(self):
        self.temporary.cleanup()

    def _write_json(self, path, payload):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_native(self):
        environment = lmdb.open(
            str(self.native_path), map_size=32 * 1024 * 1024, subdir=True
        )
        try:
            with environment.begin(write=True) as transaction:
                for key in self.keys:
                    transaction.put(
                        key.encode("utf-8"),
                        pickle.dumps(
                            self.records[key], protocol=pickle.HIGHEST_PROTOCOL
                        ),
                    )
        finally:
            environment.close()
        self._write_json(
            self.native_path / "metadata.json",
            {"schema_version": 1, "fixture": "native_diagnostic"},
        )

    def _build_decoys(self):
        fixture = self

        def generate(record, seed):
            key = next(
                key
                for key, candidate in fixture.records.items()
                if candidate["target_signature"]
                == record["target_signature"]
            )
            index = fixture.keys.index(key)
            decoy_record = copy.deepcopy(record)
            decoy_record["key"] = key
            both_complete = index < fixture.complete_count
            random_only = index == fixture.complete_count
            value = diagnostic_decoy_set(
                decoy_record,
                seed,
                random_complete=(both_complete or random_only),
                matched_complete=both_complete,
            )
            if (
                fixture.malformed_matched_only
                and index == fixture.complete_count + 1
            ):
                value = diagnostic_decoy_set(
                    decoy_record,
                    seed,
                    random_complete=True,
                    matched_complete=True,
                )
                value["banks"]["random_physical"] = {
                    "complete": False,
                    "failure_reason": (
                        "insufficient_random_physical_candidates:"
                        "low,medium,high"
                    ),
                    "decoys": [],
                }
                for stratum in decoys_v2.STRATUM_ORDER:
                    value["proposal_counts"][stratum][
                        "random_physical_retained"
                    ] = 0
            value["schema_version"] = fixture.record_schema_version
            return value

        args = SimpleNamespace(
            manifest=str(self.manifest_path),
            native_lmdb=str(self.native_path),
            diagnostic_keys=str(self.diagnostic_path),
            cohort="diagnostic",
            output_lmdb=str(self.decoy_path),
            seed=self.seed,
            map_size_gb=0.01,
            max_samples=self.max_samples,
            progress_every=0,
            max_error_preview=3,
            resume=False,
        )
        with patch.object(
            builder_v2, "generate_decoy_set_v2", side_effect=generate
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            self.builder_report = builder_v2.build_decoy_cache_v2(args)

    def evaluator_args(self, require_pass=False):
        return SimpleNamespace(
            manifest=str(self.manifest_path),
            native_lmdb=str(self.native_path),
            diagnostic_decoy_lmdb=str(self.decoy_path),
            diagnostic_keys=str(self.diagnostic_path),
            output_dir=str(self.output_path),
            seed=self.seed,
            require_pass=bool(require_pass),
        )


def assert_json_bounded(test_case, value):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            test_case.assertNotIn("oof", lowered)
            test_case.assertNotIn("prediction", lowered)
            assert_json_bounded(test_case, child)
    elif isinstance(value, list):
        for child in value:
            assert_json_bounded(test_case, child)
    else:
        test_case.assertNotIsInstance(value, np.ndarray)


class DecoyV2EvaluatorTest(unittest.TestCase):
    def tearDown(self):
        fixture = getattr(self, "fixture", None)
        if fixture is not None:
            fixture.cleanup()

    def make_fixture(self, **kwargs):
        self.fixture = DiagnosticV2EvaluatorFixture(**kwargs)
        return self.fixture

    def test_schema_invariant_rejects_matched_only_bank(self):
        record = diagnostic_native_record("matched_only", 0)
        with self.assertRaisesRegex(
            ValueError, "match-accepted|random-physical retained"
        ):
            diagnostic_decoy_set(
                record,
                seed=23,
                random_complete=False,
                matched_complete=True,
            )

    def test_evaluator_reports_exact_cohorts_hash_and_bounded_npz_rows(self):
        from scripts.evaluate_local_interface_decoys_v2 import evaluate_decoy_data

        fixture = self.make_fixture()
        report = evaluate_decoy_data(fixture.evaluator_args())

        self.assertEqual(
            set(report["cohorts"]),
            {"random_physical_full", "matched_full", "paired_intersection"},
        )
        paired = report["cohorts"]["paired_intersection"]
        self.assertEqual(paired["keys"], fixture.keys[:5])
        self.assertEqual(
            paired["ordered_key_sha256"], hash_ordered_keys(fixture.keys[:5])
        )
        self.assertEqual(
            report["cohorts"]["random_physical_full"]["keys"], fixture.keys[:6]
        )
        self.assertEqual(
            report["cohorts"]["matched_full"]["keys"], fixture.keys[:5]
        )
        self.assertEqual(
            paired["random_physical"]["labels_sha256"],
            paired["matched"]["labels_sha256"],
        )
        self.assertEqual(
            paired["random_physical"]["groups_sha256"],
            paired["matched"]["groups_sha256"],
        )
        self.assertEqual(report["gate"]["thresholds"]["maximum_matched_auroc"], 0.80)
        assert_json_bounded(self, report)

        json_path = fixture.output_path / "decoy_v2_data_diagnostic.json"
        npz_path = fixture.output_path / "decoy_v2_data_rows.npz"
        self.assertTrue(json_path.is_file())
        self.assertTrue(npz_path.is_file())
        persisted = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted, report)
        self.assertEqual(
            report["artifacts"]["rows_npz"]["sha256"],
            sha256_file(npz_path),
        )

        with np.load(str(npz_path), allow_pickle=False) as rows:
            expected_count = 4 * (6 + 5 + 5 + 5)
            for name in (
                "cohort",
                "keys",
                "labels",
                "groups",
                "coarse_features",
                "heldout_features",
                "bank",
                "stratum",
                "transform_fingerprint",
            ):
                self.assertIn(name, rows.files)
                self.assertEqual(len(rows[name]), expected_count)
            self.assertEqual(
                rows["coarse_features"].shape, (expected_count, 9)
            )
            self.assertEqual(
                rows["heldout_features"].shape, (expected_count, 30)
            )
            self.assertEqual(rows["labels"][:4].astype(int).tolist(), [1, 0, 0, 0])
            self.assertEqual(rows["stratum"][:4].tolist(), ["native", "low", "medium", "high"])
            self.assertEqual(rows["bank"][:4].tolist(), ["random_physical"] * 4)
            self.assertEqual(rows["keys"][:4].tolist(), [fixture.keys[0]] * 4)
            self.assertEqual(
                rows["groups"][:4].tolist(), ["target_group_00"] * 4
            )
            self.assertEqual(rows["transform_fingerprint"][0], "native")
            self.assertTrue(np.all(np.isfinite(rows["coarse_features"])))
            self.assertTrue(np.all(np.isfinite(rows["heldout_features"])))
    def test_rejects_native_record_stored_under_the_wrong_key(self):
        from scripts.evaluate_local_interface_decoys_v2 import (
            evaluate_decoy_data,
        )

        fixture = self.make_fixture(wrong_native_key_index=0)
        with self.assertRaisesRegex(ValueError, "native.*key"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_closes_native_environment_when_decoy_open_fails(self):
        from scripts.evaluate_local_interface_decoys_v2 import (
            evaluate_decoy_data,
        )

        fixture = self.make_fixture()
        native_environment = Mock()
        with patch.object(
            builder_v2,
            "_open_readonly_lmdb",
            side_effect=[
                native_environment,
                RuntimeError("decoy open failed"),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "decoy open"):
                evaluate_decoy_data(fixture.evaluator_args())
        native_environment.close.assert_called_once_with()

    def test_row_extraction_rejects_nonfinite_features(self):
        from scripts import evaluate_local_interface_decoys_v2 as evaluator

        record = diagnostic_native_record("finite_check", 0)
        decoy_set = diagnostic_decoy_set(
            record,
            seed=23,
            random_complete=True,
            matched_complete=True,
        )
        with patch.object(
            evaluator,
            "extract_coarse_matching_features",
            return_value=np.full(9, np.nan, dtype=np.float64),
        ), patch.object(
            evaluator,
            "extract_heldout_interface_features",
            return_value=np.zeros(30, dtype=np.float64),
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                evaluator._row_set(
                    record,
                    decoy_set,
                    key="finite_check",
                    bank="matched",
                    seed=23,
                )

    def test_rejects_nontrain_native_records_even_with_valid_provenance(self):
        from scripts.evaluate_local_interface_decoys_v2 import (
            evaluate_decoy_data,
        )

        fixture = self.make_fixture(native_split="valid")
        with self.assertRaisesRegex(ValueError, "native.*train|split"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_rejects_nontrain_native_record_even_when_decoy_is_excluded(self):
        from scripts.evaluate_local_interface_decoys_v2 import (
            evaluate_decoy_data,
        )

        fixture = self.make_fixture(nontrain_native_index=6)
        with self.assertRaisesRegex(ValueError, "native.*train|split"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_evaluator_enforces_schema_against_matched_only_record(self):
        from scripts.evaluate_local_interface_decoys_v2 import (
            evaluate_decoy_data,
        )

        fixture = self.make_fixture(malformed_matched_only=True)
        with self.assertRaisesRegex(
            ValueError, "match-accepted|random-physical retained"
        ):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_rejects_official_valid_test_and_nontrain_entry_manifests(self):
        from scripts.evaluate_local_interface_decoys_v2 import evaluate_decoy_data

        fixture = self.make_fixture()
        payload = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
        payload["split"] = "valid"
        fixture.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "split=train|valid|test"):
            evaluate_decoy_data(fixture.evaluator_args())

        payload["split"] = "train"
        payload["samples"][0]["split"] = "valid"
        fixture.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "entries.*train"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_rejects_schema_provenance_completion_and_diagnostic_key_failures(self):
        from scripts.evaluate_local_interface_decoys_v2 import evaluate_decoy_data

        fixture = self.make_fixture()
        with self.assertRaisesRegex(ValueError, "seed|provenance"):
            args = fixture.evaluator_args()
            args.seed += 1
            evaluate_decoy_data(args)

        report_path = fixture.decoy_path / "decoy_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["status"] = "in_progress"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completion|incomplete"):
            evaluate_decoy_data(fixture.evaluator_args())

        fixture.cleanup()
        self.fixture = DiagnosticV2EvaluatorFixture()
        fixture = self.fixture
        diagnostic = json.loads(fixture.diagnostic_path.read_text(encoding="utf-8"))
        diagnostic["keys"][0] = "unknown_key"
        fixture.diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown|not present"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_supports_authorized_max_samples_prefix_and_rejects_prefix_mismatch(self):
        from scripts.evaluate_local_interface_decoys_v2 import evaluate_decoy_data

        fixture = self.make_fixture(max_samples=5)
        report = evaluate_decoy_data(fixture.evaluator_args())
        self.assertEqual(report["cohorts"]["paired_intersection"]["keys"], fixture.keys[:5])

        diagnostic = json.loads(fixture.diagnostic_path.read_text(encoding="utf-8"))
        diagnostic["keys"] = [
            fixture.keys[0],
            fixture.keys[2],
            fixture.keys[3],
            fixture.keys[4],
            fixture.keys[5],
            fixture.keys[6],
        ]
        fixture.diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prefix"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_requires_at_least_five_paired_target_groups(self):
        from scripts.evaluate_local_interface_decoys_v2 import evaluate_decoy_data

        fixture = self.make_fixture(key_count=5, complete_count=4)
        with self.assertRaisesRegex(ValueError, "five target groups"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_rejects_schema_one_decoy_records(self):
        from scripts.evaluate_local_interface_decoys_v2 import evaluate_decoy_data

        fixture = self.make_fixture(record_schema_version=1)
        with self.assertRaisesRegex(ValueError, "schema version 2"):
            evaluate_decoy_data(fixture.evaluator_args())

    def test_require_pass_cli_writes_artifacts_before_exit_status_two(self):
        fixture = self.make_fixture()
        command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_local_interface_decoys_v2.py"),
            "--manifest",
            str(fixture.manifest_path),
            "--native-lmdb",
            str(fixture.native_path),
            "--diagnostic-decoy-lmdb",
            str(fixture.decoy_path),
            "--diagnostic-keys",
            str(fixture.diagnostic_path),
            "--output-dir",
            str(fixture.output_path),
            "--seed",
            str(fixture.seed),
            "--require-pass",
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertTrue((fixture.output_path / "decoy_v2_data_diagnostic.json").is_file())
        self.assertTrue((fixture.output_path / "decoy_v2_data_rows.npz").is_file())


class InterfaceDecoyGroupedProbeTest(unittest.TestCase):
    def test_grouped_probes_are_deterministic_and_complete(self):
        from utils.interface_decoy_audit import fit_grouped_pose_probes

        features, labels, groups = make_probe_rows()
        names = ("feature_a", "feature_b", "feature_c")
        first = fit_grouped_pose_probes(
            features, labels, groups, names
        )
        second = fit_grouped_pose_probes(
            features, labels, groups, names
        )

        self.assertEqual(first["feature_names"], list(names))
        self.assertEqual(first["row_count"], len(labels))
        self.assertEqual(first["target_group_count"], 10)
        self.assertEqual(first["cv_splits"], 5)
        self.assertEqual(len(first["univariate_cv_auroc"]), len(names))
        self.assertEqual(
            [row["feature"] for row in first["univariate_cv_auroc"]],
            list(names),
        )
        for field in ("linear_cv_auroc", "nonlinear_cv_auroc"):
            self.assertTrue(np.isfinite(first[field]))
            self.assertEqual(first[field], second[field])
        for field in (
            "oof_linear_prediction",
            "oof_nonlinear_prediction",
        ):
            self.assertEqual(first[field].shape, labels.shape)
            self.assertTrue(np.all(np.isfinite(first[field])))
            np.testing.assert_array_equal(first[field], second[field])
        self.assertEqual(
            first["univariate_cv_auroc"],
            second["univariate_cv_auroc"],
        )

    def test_grouped_probe_validates_features_labels_and_groups(self):
        from utils.interface_decoy_audit import fit_grouped_pose_probes

        features, labels, groups = make_probe_rows()
        cases = (
            (
                features[:, :2],
                labels,
                groups,
                ("a", "b", "c"),
                "feature",
            ),
            (
                features,
                labels,
                groups,
                ("a", "a", "c"),
                "feature",
            ),
            (
                features.copy(),
                labels,
                groups,
                ("a", "b", "c"),
                "finite",
            ),
            (
                features,
                np.zeros_like(labels),
                groups,
                ("a", "b", "c"),
                "classes",
            ),
            (
                features,
                labels,
                np.asarray(["same"] * len(groups)),
                ("a", "b", "c"),
                "five target groups",
            ),
        )
        cases[2][0][0, 0] = np.nan
        for values, target, target_groups, names, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fit_grouped_pose_probes(
                        values, target, target_groups, names
                    )


class InterfaceDecoyDataGateTest(unittest.TestCase):
    def test_v2_data_gate_has_three_and_only_three_hard_conditions(self):
        from utils.interface_decoy_audit import evaluate_v2_data_gate

        matched = np.nextafter(0.80, 0.0)
        random_auroc = np.nextafter(matched + 0.08, np.inf)
        report = make_data_report(
            matched_auroc=matched,
            random_auroc=random_auroc,
            complete_count=31,
        )
        result = evaluate_v2_data_gate(report)

        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(
            set(result["thresholds"]),
            {
                "maximum_matched_auroc",
                "minimum_random_minus_matched_auroc",
                "minimum_matched_requested_fraction",
            },
        )
        self.assertNotIn("heldout", str(result).lower())
        self.assertNotIn("interface", str(result).lower())

    def test_each_gate_boundary_fails_exactly(self):
        from utils.interface_decoy_audit import evaluate_v2_data_gate

        cases = (
            (
                make_data_report(0.80, 0.90, 31),
                "matched_coarse_auroc_too_high",
            ),
            (
                make_data_report(
                    0.70, np.nextafter(0.78, 0.0), 31
                ),
                "paired_coarse_auroc_reduction_too_small",
            ),
            (
                make_data_report(0.70, 0.90, 30),
                "matched_requested_fraction_too_low",
            ),
        )
        for report, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                result = evaluate_v2_data_gate(report)
                self.assertFalse(result["passed"])
                self.assertEqual(result["reasons"], [expected_reason])

    def test_gate_rejects_nonfinite_and_invalid_completion_counts(self):
        from utils.interface_decoy_audit import evaluate_v2_data_gate

        cases = (
            make_data_report(np.nan, 0.9, 31),
            make_data_report(0.7, np.inf, 31),
            make_data_report(0.7, 0.9, -1),
            make_data_report(0.7, 0.9, 101),
            make_data_report(0.7, 0.9, 0, requested=0),
        )
        for report in cases:
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    evaluate_v2_data_gate(report)


class InterfaceDecoyContractOracleTest(unittest.TestCase):
    def test_complete_feature_names_and_nondegenerate_numeric_oracle(self):
        expected_names = (
            "ligand_contact_entropy_normalized",
            "ligand_contact_max_share",
            "ligand_effective_contact_atom_count",
            "ligand_covalent_edge_total_variation",
            "residue_contact_entropy_normalized",
            "residue_contact_max_share",
            "residue_effective_contact_count",
            *tuple(
                f"contact_residue_type_{index}"
                for index in range(20)
            ),
            "contact_direction_eigenvalue_0",
            "contact_direction_eigenvalue_1",
            "contact_direction_eigenvalue_2",
        )
        self.assertEqual(HELDOUT_FEATURE_NAMES, expected_names)

        ligand = np.asarray(
            [[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
            dtype=np.float64,
        )
        record = {
            "ligand_coords": ligand,
            "ligand_heavy_atom_count": 2,
            "ligand_edge_index": np.asarray(
                [[0, 1], [1, 0]], dtype=np.int64
            ),
            "ligand_atomic_numbers": np.asarray(
                [6, 6], dtype=np.int64
            ),
            "protein_coords": np.asarray(
                [
                    [0.0, 0.0, 3.0],
                    [5.5, 0.0, 3.0],
                    [0.0, 3.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "protein_parent_residue": np.asarray(
                [0, 1, 0], dtype=np.int64
            ),
            "protein_residue_type": np.asarray(
                [2, 5, 2], dtype=np.int64
            ),
            "protein_element_symbol": np.asarray(
                ["C", "C", "C"]
            ),
            "residue_count": 2,
        }
        probabilities = np.asarray([2.0 / 3.0, 1.0 / 3.0])
        entropy = float(
            -np.sum(probabilities * np.log(probabilities))
        )
        composition = np.zeros(20, dtype=np.float64)
        composition[2] = 2.0 / 3.0
        composition[5] = 1.0 / 3.0
        expected = np.concatenate(
            [
                np.asarray(
                    [
                        entropy / np.log(2.0),
                        2.0 / 3.0,
                        np.exp(entropy),
                        1.0 / 3.0,
                        entropy / np.log(2.0),
                        2.0 / 3.0,
                        np.exp(entropy),
                    ]
                ),
                composition,
                np.asarray([2.0 / 3.0, 1.0 / 3.0, 0.0]),
            ]
        )

        actual = extract_heldout_interface_features(record, ligand)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )

    def test_probe_folds_are_explicitly_target_disjoint(self):
        from unittest.mock import patch

        from sklearn.model_selection import GroupKFold
        from utils.interface_decoy_audit import fit_grouped_pose_probes

        features, labels, groups = make_probe_rows()
        order = np.asarray(
            [
                group_index * 4 + pose_index
                for pose_index in range(4)
                for group_index in range(10)
            ],
            dtype=np.int64,
        )
        features = features[order]
        labels = labels[order]
        groups = groups[order]
        observed = []
        test_case = self

        class AuditedGroupKFold:
            def __init__(self, n_splits):
                self.splitter = GroupKFold(n_splits=n_splits)

            def split(self, values, target, groups=None):
                test_case.assertIsNotNone(groups)
                for train_index, test_index in self.splitter.split(
                    values, target, groups=groups
                ):
                    train_groups = set(groups[train_index].tolist())
                    test_groups = set(groups[test_index].tolist())
                    test_case.assertTrue(
                        train_groups.isdisjoint(test_groups)
                    )
                    observed.append(
                        (train_groups, test_groups)
                    )
                    yield train_index, test_index

        with patch(
            "sklearn.model_selection.GroupKFold",
            AuditedGroupKFold,
        ):
            fit_grouped_pose_probes(
                features,
                labels,
                groups,
                ("feature_a", "feature_b", "feature_c"),
            )

        self.assertEqual(len(observed), 5)
        self.assertEqual(
            set().union(*(test for _, test in observed)),
            set(groups.tolist()),
        )

    def test_paired_reduction_exact_equality_passes(self):
        from utils.interface_decoy_audit import evaluate_v2_data_gate

        report = make_data_report(
            matched_auroc=0.0,
            random_auroc=0.08,
            complete_count=31,
        )
        result = evaluate_v2_data_gate(report)

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["metrics"][
                "random_minus_matched_coarse_auroc"
            ],
            0.08,
        )


if __name__ == "__main__":
    unittest.main()
