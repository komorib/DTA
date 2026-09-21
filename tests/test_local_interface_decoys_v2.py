import copy
import hashlib
import json
import itertools
import io
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import local_interface_decoys_v2 as decoys_v2  # noqa: E402
from data_utils.local_interface import ligand_automorphisms  # noqa: E402
from data_utils.local_interface_decoys import (  # noqa: E402
    DecoyGenerationError,
    apply_rigid_transform,
)
from scripts import build_local_interface_decoys_v2 as builder_v2  # noqa: E402
from utils.experiment_integrity import (  # noqa: E402
    manifest_fingerprint,
    sha256_file,
)
from data_utils.local_interface_decoys_v2 import (  # noqa: E402
    BANK_NAMES,
    CENTROID_LIMITS,
    CONTACT_MASS_RATIO_RANGE,
    DECOY_SCHEMA_VERSION,
    HARD_VDW_RATIO,
    MATCH_LIMITS,
    PROPOSAL_RANGES,
    PROPOSALS_PER_STRATUM,
    RETAINED_PER_STRATUM,
    SHORTLIST_SIZE,
    STRATUM_ORDER,
    automorphism_digest,
    generate_proposal_pool,
)


GOLDEN_MEDIUM_SEED_42_FINGERPRINTS = (
    "58e5d83d467da48ea7558cd980a8f017b48e4a3b92a2a52a8c73e3402ebcdf77",
    "4fb76e47276d0b0e6473a823a4158fc4daff8059d7890d5cf12ed4120ce7515d",
    "5627151ca707fdf4df387a9062d138933006cd4d6a75664ad6d0e87333b922f0",
    "0da361ed85df84f327145db859ed2b703866c432bc27b838c2aa2044af7191a1",
    "9ca0c7bd14083ea637deaa68cfac9e81a7f1dabea083a266577f995d282bc0e5",
)


def make_native_record(key="complex_a"):
    return {
        "key": key,
        "ligand_coords": np.asarray(
            [
                [1.5, -0.5, 0.0],
                [2.5, -0.5, 0.0],
                [2.0, 0.8, 0.0],
            ],
            dtype=np.float64,
        ),
        "protein_coords": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }


def pocket_outward_unit_vector(record):
    direction = (
        np.mean(record["ligand_coords"], axis=0)
        - np.mean(record["protein_coords"], axis=0)
    )
    return direction / np.linalg.norm(direction)


def pairwise_distances(coordinates):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(differences, axis=-1)


class RotationDominantV2CoreTest(unittest.TestCase):
    def test_configuration_is_frozen(self):
        self.assertEqual(DECOY_SCHEMA_VERSION, 2)
        self.assertEqual(BANK_NAMES, ("random_physical", "matched"))
        self.assertEqual(STRATUM_ORDER, ("low", "medium", "high"))
        self.assertEqual(PROPOSALS_PER_STRATUM, 512)
        self.assertEqual(RETAINED_PER_STRATUM, 4)
        self.assertEqual(SHORTLIST_SIZE, 20)
        self.assertEqual(HARD_VDW_RATIO, 0.75)
        self.assertEqual(CONTACT_MASS_RATIO_RANGE, (0.80, 1.15))
        self.assertEqual(decoys_v2.RNG_BIT_GENERATOR, "PCG64")
        self.assertEqual(
            PROPOSAL_RANGES,
            {
                "low": {
                    "rotation_degrees": (5.0, 35.0),
                    "translation": (0.0, 0.75),
                },
                "medium": {
                    "rotation_degrees": (15.0, 80.0),
                    "translation": (0.0, 1.25),
                },
                "high": {
                    "rotation_degrees": (30.0, 150.0),
                    "translation": (0.0, 2.0),
                },
            },
        )
        self.assertEqual(
            CENTROID_LIMITS,
            {"low": 0.50, "medium": 1.00, "high": 1.50},
        )
        self.assertEqual(
            MATCH_LIMITS,
            {
                "minimum_distance": 0.50,
                "maximum_overlap_severity": 0.10,
                "sum_overlap_per_ligand_atom": 0.05,
                "count_below_0_9_per_ligand_atom": 0.12,
                "count_below_0_8_per_ligand_atom": 0.08,
            },
        )

    def test_v2_does_not_reexport_ligand_automorphisms(self):
        self.assertFalse(hasattr(decoys_v2, "ligand_automorphisms"))

    def test_proposal_pool_is_exact_deterministic_and_stratified(self):
        record = make_native_record()

        first = generate_proposal_pool(record, seed=42, stratum="medium")
        second = generate_proposal_pool(record, seed=42, stratum="medium")

        self.assertEqual(len(first), 512)
        self.assertEqual(
            [row["transform_fingerprint"] for row in first],
            [row["transform_fingerprint"] for row in second],
        )
        angles = np.asarray([row["rotation_degrees"] for row in first])
        translations = np.asarray(
            [row["translation_distance"] for row in first]
        )
        self.assertTrue(np.all((angles >= 15.0) & (angles <= 80.0)))
        self.assertTrue(
            np.all((translations >= 0.0) & (translations <= 1.25))
        )
        self.assertTrue(
            all(
                {
                    "transform",
                    "rotation_degrees",
                    "translation_distance",
                    "translation_direction",
                    "transform_fingerprint",
                }.issubset(row)
                for row in first
            )
        )

    def test_proposal_fingerprints_match_frozen_pcg64_stream(self):
        rows = generate_proposal_pool(
            make_native_record(), seed=42, stratum="medium"
        )

        self.assertEqual(
            tuple(
                row["transform_fingerprint"]
                for row in rows[: len(GOLDEN_MEDIUM_SEED_42_FINGERPRINTS)]
            ),
            GOLDEN_MEDIUM_SEED_42_FINGERPRINTS,
        )

    def test_seed_and_key_have_independent_proposal_streams(self):
        baseline = generate_proposal_pool(
            make_native_record(), seed=42, stratum="medium"
        )
        changed_seed = generate_proposal_pool(
            make_native_record(), seed=43, stratum="medium"
        )
        changed_key = generate_proposal_pool(
            make_native_record(key="complex_b"), seed=42, stratum="medium"
        )

        baseline_fingerprints = [
            row["transform_fingerprint"] for row in baseline
        ]
        self.assertNotEqual(
            baseline_fingerprints,
            [row["transform_fingerprint"] for row in changed_seed],
        )
        self.assertNotEqual(
            baseline_fingerprints,
            [row["transform_fingerprint"] for row in changed_key],
        )

    def test_seed_must_be_an_integer_without_truncation(self):
        for invalid in (1.9, True, "42"):
            with self.subTest(seed=invalid):
                with self.assertRaisesRegex(ValueError, "seed must be an integer"):
                    generate_proposal_pool(
                        make_native_record(), seed=invalid, stratum="low"
                    )

        expected = generate_proposal_pool(
            make_native_record(), seed=42, stratum="low"
        )
        actual = generate_proposal_pool(
            make_native_record(), seed=np.int64(42), stratum="low"
        )
        self.assertEqual(
            [row["transform_fingerprint"] for row in actual],
            [row["transform_fingerprint"] for row in expected],
        )

    def test_proposals_are_proper_rigid_transforms(self):
        record = make_native_record()
        native_pairwise = pairwise_distances(record["ligand_coords"])
        rows = generate_proposal_pool(record, seed=42, stratum="medium")

        for row in rows:
            rotation = np.asarray(
                row["transform"]["rotation_matrix"], dtype=np.float64
            )
            np.testing.assert_allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-12
            )
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
            transformed = apply_rigid_transform(
                record["ligand_coords"], row["transform"]
            )
            np.testing.assert_allclose(
                pairwise_distances(transformed), native_pairwise, atol=1e-12
            )

    def test_translation_has_no_pocket_outward_concentration(self):
        record = make_native_record()

        rows = generate_proposal_pool(record, seed=11, stratum="high")

        directions = np.asarray(
            [row["translation_direction"] for row in rows]
        )
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=1), np.ones(len(rows)), atol=1e-12
        )
        outward = pocket_outward_unit_vector(record)
        self.assertLess(abs(float(np.mean(directions @ outward))), 0.10)
        target_second_moment = np.eye(3) / 3.0
        np.testing.assert_allclose(
            directions.T @ directions / len(directions),
            target_second_moment,
            atol=0.05,
        )
        np.testing.assert_allclose(
            np.cov(directions, rowvar=False, bias=True),
            target_second_moment,
            atol=0.05,
        )

    def test_automorphism_digest_has_frozen_bytes(self):
        mappings = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        expected = hashlib.sha256(
            mappings.astype("<i8").tobytes(order="C")
        ).hexdigest()

        self.assertEqual(automorphism_digest(mappings), expected)
        self.assertEqual(automorphism_digest(mappings.astype(">i8")), expected)

        for invalid in (np.asarray([0, 1]), np.empty((0, 2), dtype=np.int64)):
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "nonempty \\[M,L\\]"):
                    automorphism_digest(invalid)

    def test_ligand_automorphism_limit_accepts_cap_and_rejects_cap_plus_one(self):
        matches = tuple(itertools.islice(itertools.permutations(range(8)), 4097))
        mol = Mock()
        mol.GetNumAtoms.return_value = 8
        mol.GetSubstructMatches.return_value = matches[:4096]

        mappings = ligand_automorphisms(mol, max_matches=4096)

        self.assertEqual(mappings.shape, (4096, 8))
        mol.GetSubstructMatches.assert_called_once_with(
            mol,
            uniquify=False,
            useChirality=True,
            maxMatches=4097,
        )

        overflow_mol = Mock()
        overflow_mol.GetNumAtoms.return_value = 8
        overflow_mol.GetSubstructMatches.return_value = matches
        with self.assertRaisesRegex(ValueError, "exceeds max_matches=4096"):
            ligand_automorphisms(overflow_mol, max_matches=4096)


_REAL_DECOY_CACHE = {}


def make_geometry_record(key="geometry_a"):
    protein = np.asarray(
        [
            [x, y, 3.8]
            for x in np.arange(-20.0, 20.1, 2.3)
            for y in np.arange(-20.0, 20.1, 2.3)
        ],
        dtype=np.float64,
    )
    return {
        "key": key,
        "ligand_coords": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        "ligand_atomic_numbers": np.asarray([6], dtype=np.int64),
        "ligand_automorphisms": np.asarray([[0]], dtype=np.int64),
        "ligand_heavy_atom_count": 1,
        "protein_coords": protein,
        "protein_element_symbol": np.asarray(["C"] * len(protein)),
    }


def make_transform(translation):
    return {
        "center": [0.0, 0.0, 0.0],
        "rotation_matrix": np.eye(3, dtype=np.float64).tolist(),
        "translation": [float(value) for value in translation],
    }


def fingerprint_transform(transform):
    return hashlib.sha256(
        json.dumps(transform, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def make_candidate(index=0, match_distance=None, match_pass=True):
    return {
        "transform_fingerprint": f"{int(index):064x}",
        "match_distance": (
            float(index) if match_distance is None else float(match_distance)
        ),
        "match_pass": bool(match_pass),
    }


def make_contact_stats(
    jaccard=0.80,
    ratio=1.0,
    coverage_retained=True,
    contact_pattern_nonempty=True,
):
    return {
        "native_contact_mass": 10.0,
        "decoy_contact_mass": 10.0 * ratio,
        "contact_mass_ratio": ratio,
        "soft_jaccard": jaccard,
        "native_neighbor_count": 1,
        "decoy_neighbor_count": 1,
        "coverage_retained": coverage_retained,
        "contact_pattern_nonempty": contact_pattern_nonempty,
    }


def make_overlap_stats(hard_clash=False):
    return {
        "minimum_distance_ratio": 0.7 if hard_clash else 1.0,
        "max_overlap_severity": 0.3 if hard_clash else 0.0,
        "sum_overlap_severity": 0.3 if hard_clash else 0.0,
        "count_below_0_9": 1 if hard_clash else 0,
        "count_below_0_8": 1 if hard_clash else 0,
        "has_hard_clash": bool(hard_clash),
    }


def make_match_values(**overrides):
    values = {
        "contact_mass_ratio": 1.0,
        "delta_centroid_distance": 0.0,
        "delta_minimum_distance": 0.0,
        "delta_max_overlap_severity": 0.0,
        "delta_sum_overlap_per_ligand_atom": 0.0,
        "delta_count_below_0_9_per_ligand_atom": 0.0,
        "delta_count_below_0_8_per_ligand_atom": 0.0,
    }
    values.update(overrides)
    return values


def make_full_candidate(index, stratum, match_pass=True):
    transform = make_transform((float(index + 1) * 1e-6, 0.0, 0.0))
    jaccard = {"low": 0.80, "medium": 0.65, "high": 0.50}[stratum]
    return {
        "transform": transform,
        "transform_fingerprint": fingerprint_transform(transform),
        "rmsd_sym": {"low": 0.7, "medium": 1.3, "high": 2.0}[stratum],
        "contact_stats": make_contact_stats(jaccard=jaccard),
        "overlap_stats": make_overlap_stats(),
        "stratum": stratum,
        "failure_reason": None,
        "match_values": make_match_values(),
        "match_distance": 0.0,
        "match_pass": bool(match_pass),
    }


def controlled_proposal_pool(record, seed, stratum):
    stratum_offset = STRATUM_ORDER.index(stratum) * 10000
    rows = []
    for candidate_index in range(PROPOSALS_PER_STRATUM):
        unique_index = stratum_offset + candidate_index
        transform = make_transform((float(unique_index + 1) * 1e-6, 0.0, 0.0))
        rows.append(
            {
                "transform": transform,
                "rotation_degrees": 10.0,
                "translation_distance": abs(transform["translation"][0]),
                "translation_direction": [1.0, 0.0, 0.0],
                "transform_fingerprint": fingerprint_transform(transform),
                "candidate_index": candidate_index,
                "unique_index": unique_index,
            }
        )
    return rows


def controlled_evaluator(
    physical_count=8, matched_high_count=8, accept_all=False
):
    def evaluate(native_geometry, proposal, target_stratum, seen_fingerprints):
        index = int(proposal["candidate_index"])
        if not accept_all and index >= physical_count:
            return None, "outside_frozen_strata", None
        fingerprint = proposal["transform_fingerprint"]
        if fingerprint in seen_fingerprints:
            return None, "duplicate_transform", None
        seen_fingerprints.add(fingerprint)
        match_pass = not (
            target_stratum == "high" and index >= matched_high_count
        )
        candidate = make_full_candidate(
            int(proposal["unique_index"]), target_stratum, match_pass=match_pass
        )
        candidate["transform"] = copy.deepcopy(proposal["transform"])
        candidate["transform_fingerprint"] = fingerprint
        return (
            candidate,
            None,
            None if match_pass else "match_delta_centroid_distance",
        )

    return evaluate


def make_real_decoy_set():
    if "value" not in _REAL_DECOY_CACHE:
        record = make_geometry_record()
        value = decoys_v2.generate_decoy_set_v2(record, seed=42)
        _REAL_DECOY_CACHE["record"] = record
        _REAL_DECOY_CACHE["value"] = value
    return (
        _REAL_DECOY_CACHE["record"],
        copy.deepcopy(_REAL_DECOY_CACHE["value"]),
    )


def find_unselected_real_candidate(record, value, bank_name, stratum):
    native_geometry = decoys_v2._prepare_native_geometry(record)
    native_geometry["native_overlap"] = decoys_v2.vdw_overlap_statistics(
        native_geometry["protein"],
        native_geometry["protein_symbols"],
        native_geometry["native"],
        native_geometry["ligand_atomic_numbers"],
    )
    native_geometry["native_centroid"] = decoys_v2.centroid_distance(
        native_geometry["protein"], native_geometry["native"]
    )
    native_geometry["native_minimum"] = decoys_v2.minimum_pair_distance(
        native_geometry["protein"], native_geometry["native"]
    )
    selected = {
        row["transform_fingerprint"]
        for row in value["banks"][bank_name]["decoys"]
        if row["stratum"] == stratum
    }
    seen_fingerprints = set()
    for current_stratum in STRATUM_ORDER:
        proposals = generate_proposal_pool(
            record, seed=value["seed"], stratum=current_stratum
        )
        for proposal in proposals:
            candidate, _, _ = decoys_v2._evaluate_proposal(
                native_geometry,
                proposal,
                current_stratum,
                seen_fingerprints,
            )
            if (
                current_stratum == stratum
                and candidate is not None
                and candidate["transform_fingerprint"] not in selected
                and (
                    bank_name == "random_physical"
                    or candidate["match_pass"]
                )
            ):
                return candidate
        if current_stratum == stratum:
            break
    raise AssertionError("fixture has no unselected valid candidate")


class RotationDominantV2BankTest(unittest.TestCase):
    def test_retain_matched_samples_four_from_top_twenty(self):
        candidates = [
            make_candidate(
                index=100 - index,
                match_distance=float(index),
                match_pass=True,
            )
            for index in range(30)
        ]

        retained = decoys_v2.retain_matched(
            candidates, seed=42, key="x", stratum="low"
        )

        self.assertEqual(len(retained), 4)
        self.assertTrue(all(row["match_distance"] < 20.0 for row in retained))
        self.assertEqual(
            retained,
            decoys_v2.retain_matched(
                candidates, seed=42, key="x", stratum="low"
            ),
        )

    def test_retain_random_physical_ignores_matching_limits(self):
        candidates = [
            make_candidate(index=index, match_pass=False) for index in range(8)
        ]

        retained = decoys_v2.retain_random_physical(
            candidates, seed=42, key="x", stratum="low"
        )

        self.assertEqual(len(retained), 4)
        self.assertTrue(all(not row["match_pass"] for row in retained))

    def test_bank_selection_uses_independent_rng_namespaces(self):
        candidates = [make_candidate(index=index) for index in range(12)]

        random_rows = decoys_v2.retain_random_physical(
            candidates, seed=17, key="x", stratum="low"
        )
        matched_rows = decoys_v2.retain_matched(
            candidates, seed=17, key="x", stratum="low"
        )

        self.assertNotEqual(
            [row["transform_fingerprint"] for row in random_rows],
            [row["transform_fingerprint"] for row in matched_rows],
        )

    def test_centroid_and_minimum_pair_distance_validate_finite_shapes(self):
        protein = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        ligand = np.asarray([[1.0, 3.0, 0.0]])

        self.assertAlmostEqual(
            decoys_v2.centroid_distance(protein, ligand), 3.0
        )
        self.assertAlmostEqual(
            decoys_v2.minimum_pair_distance(protein, ligand),
            np.sqrt(10.0),
        )
        for invalid in (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([[np.nan, 0.0, 0.0]]),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "shape|coordinates|finite"):
                    decoys_v2.centroid_distance(invalid, ligand)
                with self.assertRaisesRegex(ValueError, "shape|coordinates|finite"):
                    decoys_v2.minimum_pair_distance(protein, invalid)

    def test_candidate_match_values_use_exact_native_relative_formulas(self):
        record = make_geometry_record()
        transformed = record["ligand_coords"] + np.asarray([[0.7, 0.0, 0.0]])
        contact = decoys_v2.soft_contact_statistics(
            record["protein_coords"], record["ligand_coords"], transformed
        )
        overlap = decoys_v2.vdw_overlap_statistics(
            record["protein_coords"],
            record["protein_element_symbol"],
            transformed,
            record["ligand_atomic_numbers"],
        )

        values = decoys_v2.candidate_match_values(
            record, transformed, contact, overlap
        )

        self.assertEqual(
            set(values),
            {
                "contact_mass_ratio",
                "delta_centroid_distance",
                "delta_minimum_distance",
                "delta_max_overlap_severity",
                "delta_sum_overlap_per_ligand_atom",
                "delta_count_below_0_9_per_ligand_atom",
                "delta_count_below_0_8_per_ligand_atom",
            },
        )
        native_overlap = decoys_v2.vdw_overlap_statistics(
            record["protein_coords"],
            record["protein_element_symbol"],
            record["ligand_coords"],
            record["ligand_atomic_numbers"],
        )
        self.assertEqual(values["contact_mass_ratio"], contact["contact_mass_ratio"])
        self.assertAlmostEqual(
            values["delta_centroid_distance"],
            decoys_v2.centroid_distance(
                record["protein_coords"], transformed
            )
            - decoys_v2.centroid_distance(
                record["protein_coords"], record["ligand_coords"]
            ),
        )
        self.assertAlmostEqual(
            values["delta_minimum_distance"],
            decoys_v2.minimum_pair_distance(
                record["protein_coords"], transformed
            )
            - decoys_v2.minimum_pair_distance(
                record["protein_coords"], record["ligand_coords"]
            ),
        )
        self.assertAlmostEqual(
            values["delta_max_overlap_severity"],
            overlap["max_overlap_severity"]
            - native_overlap["max_overlap_severity"],
        )

    def test_physical_classification_ignores_all_matching_limits(self):
        contact = make_contact_stats(jaccard=0.80, ratio=100.0)
        overlap = make_overlap_stats()

        physical = decoys_v2.classify_physical_candidate(
            0.7,
            contact,
            overlap,
            target_stratum="low",
            transform_fingerprint="unique",
            seen_fingerprints=set(),
        )
        matched = decoys_v2.classify_matched_candidate(
            make_match_values(
                contact_mass_ratio=100.0,
                delta_centroid_distance=100.0,
                delta_minimum_distance=100.0,
                delta_max_overlap_severity=100.0,
                delta_sum_overlap_per_ligand_atom=100.0,
                delta_count_below_0_9_per_ligand_atom=100.0,
                delta_count_below_0_8_per_ligand_atom=100.0,
            ),
            stratum="low",
        )

        self.assertTrue(physical["accepted"])
        self.assertFalse(matched["match_pass"])

    def test_physical_classification_checks_each_frozen_requirement(self):
        good_contact = make_contact_stats(jaccard=0.80)
        good_overlap = make_overlap_stats()
        cases = [
            (
                "nonfinite_metrics",
                np.nan,
                good_contact,
                good_overlap,
                "low",
                "new",
                set(),
            ),
            (
                "vdw_hard_clash",
                0.7,
                good_contact,
                make_overlap_stats(hard_clash=True),
                "low",
                "new",
                set(),
            ),
            (
                "empty_contact_pattern",
                0.7,
                make_contact_stats(
                    jaccard=0.80, contact_pattern_nonempty=False
                ),
                good_overlap,
                "low",
                "new",
                set(),
            ),
            (
                "ligand_neighbor_coverage",
                0.7,
                make_contact_stats(jaccard=0.80, coverage_retained=False),
                good_overlap,
                "low",
                "new",
                set(),
            ),
            (
                "outside_frozen_strata",
                0.2,
                good_contact,
                good_overlap,
                "low",
                "new",
                set(),
            ),
            (
                "wrong_stratum",
                1.3,
                make_contact_stats(jaccard=0.65),
                good_overlap,
                "low",
                "new",
                set(),
            ),
            (
                "duplicate_transform",
                0.7,
                good_contact,
                good_overlap,
                "low",
                "duplicate",
                {"duplicate"},
            ),
        ]
        for expected, rmsd, contact, overlap, stratum, fingerprint, seen in cases:
            with self.subTest(expected=expected):
                result = decoys_v2.classify_physical_candidate(
                    rmsd,
                    contact,
                    overlap,
                    target_stratum=stratum,
                    transform_fingerprint=fingerprint,
                    seen_fingerprints=seen,
                )
                self.assertFalse(result["accepted"])
                self.assertEqual(result["failure_reason"], expected)

    def test_matched_limits_and_normalized_distance_are_exact(self):
        fixed_limits = {
            "delta_minimum_distance": MATCH_LIMITS["minimum_distance"],
            "delta_max_overlap_severity": MATCH_LIMITS[
                "maximum_overlap_severity"
            ],
            "delta_sum_overlap_per_ligand_atom": MATCH_LIMITS[
                "sum_overlap_per_ligand_atom"
            ],
            "delta_count_below_0_9_per_ligand_atom": MATCH_LIMITS[
                "count_below_0_9_per_ligand_atom"
            ],
            "delta_count_below_0_8_per_ligand_atom": MATCH_LIMITS[
                "count_below_0_8_per_ligand_atom"
            ],
        }
        for stratum in STRATUM_ORDER:
            values = make_match_values(
                contact_mass_ratio=CONTACT_MASS_RATIO_RANGE[0],
                delta_centroid_distance=CENTROID_LIMITS[stratum],
                **fixed_limits,
            )
            with self.subTest(stratum=stratum):
                result = decoys_v2.classify_matched_candidate(values, stratum)
                self.assertTrue(result["match_pass"])
                expected = (
                    abs(np.log(values["contact_mass_ratio"])) / np.log(1.15)
                    + abs(values["delta_centroid_distance"])
                    / CENTROID_LIMITS[stratum]
                    + abs(values["delta_minimum_distance"]) / 0.50
                    + abs(values["delta_max_overlap_severity"]) / 0.10
                    + abs(values["delta_sum_overlap_per_ligand_atom"]) / 0.05
                    + abs(values["delta_count_below_0_9_per_ligand_atom"])
                    / 0.12
                    + abs(values["delta_count_below_0_8_per_ligand_atom"])
                    / 0.08
                )
                self.assertAlmostEqual(result["match_distance"], expected)

        for ratio in (0.799999, 1.150001):
            result = decoys_v2.classify_matched_candidate(
                make_match_values(contact_mass_ratio=ratio), "low"
            )
            self.assertEqual(
                result["failure_reason"], "match_contact_mass_ratio"
            )
        limits_by_field = {
            "delta_centroid_distance": CENTROID_LIMITS["low"],
            **fixed_limits,
        }
        for field, limit in limits_by_field.items():
            with self.subTest(field=field):
                result = decoys_v2.classify_matched_candidate(
                    make_match_values(**{field: limit + 1e-9}), "low"
                )
                self.assertFalse(result["match_pass"])
                self.assertIn(field, result["failure_reason"])

    def test_strict_validator_rejects_deterministic_bank_selection_tamper(self):
        record, value = make_real_decoy_set()
        replacement = find_unselected_real_candidate(
            record, value, bank_name="matched", stratum="low"
        )
        tampered = copy.deepcopy(value)
        low_index = next(
            index
            for index, row in enumerate(
                tampered["banks"]["matched"]["decoys"]
            )
            if row["stratum"] == "low"
        )
        tampered["banks"]["matched"]["decoys"][low_index] = replacement

        with self.assertRaisesRegex(
            ValueError, "retained|selection|fingerprint sequence"
        ):
            decoys_v2.validate_decoy_set_v2(
                tampered, native_record=record
            )

    def test_strict_validator_rejects_exact_count_tampering(self):
        record, value = make_real_decoy_set()

        bad_proposal_counts = copy.deepcopy(value)
        bad_proposal_counts["proposal_counts"]["low"][
            "physical_accepted"
        ] += 1
        with self.assertRaisesRegex(
            ValueError, "proposal counts|physical rejection|match rejection"
        ):
            decoys_v2.validate_decoy_set_v2(
                bad_proposal_counts, native_record=record
            )

        bad_rejection_counts = copy.deepcopy(value)
        low_rejections = bad_rejection_counts["rejection_counts"]["low"]
        reason = next(iter(low_rejections), "outside_frozen_strata")
        low_rejections[reason] = low_rejections.get(reason, 0) + 1
        with self.assertRaisesRegex(
            ValueError, "rejection counts|physical rejection|match rejection"
        ):
            decoys_v2.validate_decoy_set_v2(
                bad_rejection_counts, native_record=record
            )

    def test_native_hard_clash_precedes_low_contact_exclusion(self):
        from data_utils.local_interface_decoys import DecoyGenerationError

        record = make_geometry_record(key="low_contact_native_clash")
        record["protein_coords"] = np.asarray(
            [[0.0, 0.0, 2.0]], dtype=np.float64
        )
        record["protein_element_symbol"] = np.asarray(["C"])

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=AssertionError("proposals must not be generated"),
        ):
            with self.assertRaises(DecoyGenerationError) as raised:
                decoys_v2.generate_decoy_set_v2(record, seed=42)

        self.assertEqual(
            raised.exception.rejection_counts, {"native_hard_clash": 1}
        )
        self.assertIn("native_hard_clash", str(raised.exception))

    def test_low_native_contact_raises_typed_exclusion_after_validation(self):
        record = make_geometry_record(key="typed_low_contact")
        record["protein_coords"] = np.asarray(
            [[20.0, 0.0, 0.0]], dtype=np.float64
        )
        record["protein_element_symbol"] = np.asarray(["C"])

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=AssertionError("proposals must not be generated"),
        ):
            with self.assertRaises(
                decoys_v2.NativeContactMassExclusion
            ) as raised:
                decoys_v2.generate_decoy_set_v2(record, seed=42)

        self.assertEqual(raised.exception.native_contact_mass, 0.0)
        self.assertEqual(raised.exception.minimum_contact_mass, 8.0)
        self.assertIn("native contact mass", str(raised.exception))

    def test_retained_banks_do_not_alias_shared_candidates(self):
        record = make_geometry_record(key="bank_aliasing")

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=controlled_proposal_pool,
        ), patch.object(
            decoys_v2,
            "_evaluate_proposal",
            side_effect=controlled_evaluator(physical_count=4),
        ), patch.object(
            decoys_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            value = decoys_v2.generate_decoy_set_v2(record, seed=42)

        random_by_fingerprint = {
            row["transform_fingerprint"]: row
            for row in value["banks"]["random_physical"]["decoys"]
        }
        matched_by_fingerprint = {
            row["transform_fingerprint"]: row
            for row in value["banks"]["matched"]["decoys"]
        }
        shared = sorted(
            set(random_by_fingerprint) & set(matched_by_fingerprint)
        )
        self.assertTrue(shared)
        fingerprint = shared[0]
        random_entry = random_by_fingerprint[fingerprint]
        matched_entry = matched_by_fingerprint[fingerprint]

        self.assertIsNot(random_entry, matched_entry)
        for field in (
            "transform",
            "contact_stats",
            "overlap_stats",
            "match_values",
        ):
            self.assertIsNot(random_entry[field], matched_entry[field])

        matched_before = copy.deepcopy(matched_entry)
        random_entry["transform"]["translation"][0] += 1.0
        random_entry["contact_stats"]["decoy_contact_mass"] += 1.0
        random_entry["match_values"]["delta_minimum_distance"] += 1.0
        self.assertEqual(matched_entry, matched_before)

    def test_native_hard_clash_excludes_before_proposals_with_explicit_reason(self):
        from data_utils.local_interface_decoys import DecoyGenerationError

        record = make_geometry_record(key="native_clash")
        record["protein_coords"] = record["protein_coords"].copy()
        record["protein_coords"][0] = [0.0, 0.0, 2.0]

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=AssertionError("proposals must not be generated"),
        ):
            with self.assertRaises(DecoyGenerationError) as raised:
                decoys_v2.generate_decoy_set_v2(record, seed=42)

        self.assertEqual(
            raised.exception.rejection_counts, {"native_hard_clash": 1}
        )
        self.assertIn("native_hard_clash", str(raised.exception))

    def test_generator_evaluates_all_512_times_three_after_enough_survivors(self):
        record = make_geometry_record(key="all_proposals")
        evaluator = Mock(side_effect=controlled_evaluator(accept_all=True))

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=controlled_proposal_pool,
        ), patch.object(
            decoys_v2, "_evaluate_proposal", evaluator
        ), patch.object(
            decoys_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            value = decoys_v2.generate_decoy_set_v2(record, seed=42)

        self.assertEqual(evaluator.call_count, 512 * 3)
        self.assertTrue(value["banks"]["random_physical"]["complete"])
        self.assertTrue(value["banks"]["matched"]["complete"])

    def test_formal_generator_preserves_complete_and_incomplete_banks(self):
        record = make_geometry_record(key="mixed_completion")
        evaluator = controlled_evaluator(
            physical_count=8, matched_high_count=3
        )

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=controlled_proposal_pool,
        ), patch.object(
            decoys_v2, "_evaluate_proposal", side_effect=evaluator
        ), patch.object(
            decoys_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            value = decoys_v2.generate_decoy_set_v2(record, seed=42)

        random_bank = value["banks"]["random_physical"]
        matched_bank = value["banks"]["matched"]
        self.assertTrue(random_bank["complete"])
        self.assertIsNone(random_bank["failure_reason"])
        self.assertEqual(len(random_bank["decoys"]), 12)
        self.assertFalse(matched_bank["complete"])
        self.assertIn("high", matched_bank["failure_reason"])
        self.assertEqual(len(matched_bank["decoys"]), 11)
        self.assertEqual(
            [row["stratum"] for row in matched_bank["decoys"]].count("high"),
            3,
        )

    def test_complete_banks_retain_four_per_stratum_from_shared_pool(self):
        record = make_geometry_record(key="shared_pool")
        pools = {
            stratum: controlled_proposal_pool(record, 42, stratum)
            for stratum in STRATUM_ORDER
        }

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=lambda record, seed, stratum: pools[stratum],
        ), patch.object(
            decoys_v2,
            "_evaluate_proposal",
            side_effect=controlled_evaluator(physical_count=8),
        ), patch.object(
            decoys_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            value = decoys_v2.generate_decoy_set_v2(record, seed=42)

        shared_fingerprints = {
            row["transform_fingerprint"]
            for rows in pools.values()
            for row in rows
        }
        self.assertEqual(
            value["proposal_fingerprint_sha256"],
            decoys_v2.proposal_fingerprint_digest(pools),
        )
        for bank_name in BANK_NAMES:
            bank = value["banks"][bank_name]
            self.assertTrue(bank["complete"])
            for stratum in STRATUM_ORDER:
                retained = [
                    row for row in bank["decoys"]
                    if row["stratum"] == stratum
                ]
                self.assertEqual(len(retained), 4)
                self.assertTrue(
                    all(
                        row["transform_fingerprint"] in shared_fingerprints
                        for row in retained
                    )
                )

    def test_native_free_validator_enforces_exact_proposal_count_equations(self):
        _, value = make_real_decoy_set()

        physical_overflow = copy.deepcopy(value)
        physical_overflow["proposal_counts"]["low"][
            "physical_accepted"
        ] = PROPOSALS_PER_STRATUM + 1
        with self.assertRaisesRegex(
            ValueError, "physical-accepted|proposal counts"
        ):
            decoys_v2.validate_decoy_set_v2(physical_overflow)

        matched_overflow = copy.deepcopy(value)
        matched_overflow["proposal_counts"]["low"]["match_accepted"] = (
            matched_overflow["proposal_counts"]["low"][
                "physical_accepted"
            ]
            + 1
        )
        with self.assertRaisesRegex(
            ValueError, "match-accepted|proposal counts"
        ):
            decoys_v2.validate_decoy_set_v2(matched_overflow)

        random_under_retained = copy.deepcopy(value)
        random_counts = random_under_retained["proposal_counts"]["low"]
        random_counts["random_physical_retained"] = 3
        random_bank = random_under_retained["banks"]["random_physical"]
        random_low_index = next(
            index
            for index, row in enumerate(random_bank["decoys"])
            if row["stratum"] == "low"
        )
        del random_bank["decoys"][random_low_index]
        random_bank["complete"] = False
        random_bank["failure_reason"] = (
            "insufficient_random_physical_candidates:low"
        )
        with self.assertRaisesRegex(ValueError, "retained proposal count"):
            decoys_v2.validate_decoy_set_v2(random_under_retained)

        matched_under_retained = copy.deepcopy(value)
        matched_counts = matched_under_retained["proposal_counts"]["low"]
        matched_counts["matched_retained"] = 3
        matched_bank = matched_under_retained["banks"]["matched"]
        matched_low_index = next(
            index
            for index, row in enumerate(matched_bank["decoys"])
            if row["stratum"] == "low"
        )
        del matched_bank["decoys"][matched_low_index]
        matched_bank["complete"] = False
        matched_bank["failure_reason"] = (
            "insufficient_matched_candidates:low"
        )
        with self.assertRaisesRegex(ValueError, "retained proposal count"):
            decoys_v2.validate_decoy_set_v2(matched_under_retained)

    def test_native_free_validator_enforces_rejection_partition_equations(self):
        _, value = make_real_decoy_set()

        bad_physical = copy.deepcopy(value)
        physical_rejections = bad_physical["rejection_counts"]["low"]
        physical_reason = next(
            (
                reason
                for reason in physical_rejections
                if not reason.startswith("match_")
            ),
            "outside_frozen_strata",
        )
        physical_rejections[physical_reason] = (
            physical_rejections.get(physical_reason, 0) + 1
        )
        with self.assertRaisesRegex(
            ValueError, "physical rejection|total rejection"
        ):
            decoys_v2.validate_decoy_set_v2(bad_physical)

        bad_match = copy.deepcopy(value)
        match_rejections = bad_match["rejection_counts"]["low"]
        match_reason = next(
            (
                reason
                for reason in match_rejections
                if reason.startswith("match_")
            ),
            "match_contact_mass_ratio",
        )
        match_rejections[match_reason] = (
            match_rejections.get(match_reason, 0) + 1
        )
        with self.assertRaisesRegex(
            ValueError, "match rejection|total rejection"
        ):
            decoys_v2.validate_decoy_set_v2(bad_match)

    def test_select_epoch_returns_deep_independent_copies(self):
        record, value = make_real_decoy_set()
        original = copy.deepcopy(value)

        with patch.object(
            decoys_v2,
            "generate_proposal_pool",
            side_effect=AssertionError(
                "native-free selector validation must not regenerate geometry"
            ),
        ):
            first = decoys_v2.select_epoch_decoys_v2(
                value, record["key"], epoch=4, seed=23, bank="matched"
            )
            second = decoys_v2.select_epoch_decoys_v2(
                value, record["key"], epoch=4, seed=23, bank="matched"
            )

        self.assertEqual(first, second)
        for first_entry, second_entry in zip(first, second):
            source_entry = next(
                row
                for row in value["banks"]["matched"]["decoys"]
                if row["transform_fingerprint"]
                == first_entry["transform_fingerprint"]
            )
            self.assertIsNot(first_entry, second_entry)
            self.assertIsNot(first_entry, source_entry)
            for field in (
                "transform",
                "contact_stats",
                "overlap_stats",
                "match_values",
            ):
                self.assertIsNot(first_entry[field], second_entry[field])
                self.assertIsNot(first_entry[field], source_entry[field])

        first[0]["transform"]["translation"][0] += 1.0
        first[0]["contact_stats"]["decoy_contact_mass"] += 1.0
        first[0]["overlap_stats"]["sum_overlap_severity"] += 1.0
        first[0]["match_values"]["delta_minimum_distance"] += 1.0
        self.assertEqual(value, original)
        self.assertNotEqual(first, second)

    def test_select_epoch_rejects_malformed_schema_with_value_error(self):
        record, value = make_real_decoy_set()

        malformed_banks = copy.deepcopy(value)
        malformed_banks["banks"] = []
        with self.assertRaises(ValueError):
            decoys_v2.select_epoch_decoys_v2(
                malformed_banks,
                record["key"],
                epoch=0,
                seed=42,
                bank="matched",
            )

        partial_entries = copy.deepcopy(value)
        for entry in partial_entries["banks"]["matched"]["decoys"]:
            del entry["match_values"]
        with self.assertRaises(ValueError):
            decoys_v2.select_epoch_decoys_v2(
                partial_entries,
                record["key"],
                epoch=0,
                seed=42,
                bank="matched",
            )

    def test_select_epoch_requires_complete_explicit_bank_and_is_deterministic(self):
        record, value = make_real_decoy_set()

        first = decoys_v2.select_epoch_decoys_v2(
            value, record["key"], epoch=3, seed=19, bank="matched"
        )
        second = decoys_v2.select_epoch_decoys_v2(
            value, record["key"], epoch=3, seed=19, bank="matched"
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [row["stratum"] for row in first], list(STRATUM_ORDER)
        )
        incomplete = copy.deepcopy(value)
        incomplete["banks"]["matched"]["complete"] = False
        incomplete["banks"]["matched"]["failure_reason"] = (
            "insufficient_matched_candidates:high"
        )
        with self.assertRaisesRegex(ValueError, "complete"):
            decoys_v2.select_epoch_decoys_v2(
                incomplete,
                record["key"],
                epoch=3,
                seed=19,
                bank="matched",
            )
        with self.assertRaisesRegex(ValueError, "bank"):
            decoys_v2.select_epoch_decoys_v2(
                value, record["key"], epoch=3, seed=19, bank=None
            )

    def test_validator_rejects_schema_one_and_noncanonical_embedded_tensors(self):
        with self.assertRaisesRegex(ValueError, "schema version 2"):
            decoys_v2.validate_decoy_set_v2({"schema_version": 1})

        record, value = make_real_decoy_set()
        value["banks"]["matched"]["decoys"][0]["ligand_coords"] = np.zeros(
            (1, 3)
        )
        with self.assertRaisesRegex(
            ValueError, "canonical|schema fields|metadata"
        ):
            decoys_v2.validate_decoy_set_v2(value, native_record=record)

    def test_validator_rejects_duplicate_fingerprints_and_bad_digests(self):
        record, value = make_real_decoy_set()
        duplicate = copy.deepcopy(value)
        duplicate["banks"]["matched"]["decoys"][1] = copy.deepcopy(
            duplicate["banks"]["matched"]["decoys"][0]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            decoys_v2.validate_decoy_set_v2(
                duplicate, native_record=record
            )

        wrong_automorphism = copy.deepcopy(value)
        wrong_automorphism["automorphism_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "automorphism"):
            decoys_v2.validate_decoy_set_v2(
                wrong_automorphism, native_record=record
            )

        wrong_proposals = copy.deepcopy(value)
        wrong_proposals["proposal_fingerprint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "proposal fingerprint"):
            decoys_v2.validate_decoy_set_v2(
                wrong_proposals, native_record=record
            )

    def test_validator_recomputes_rmsd_contact_overlap_and_match_values(self):
        record, value = make_real_decoy_set()

        bad_rmsd = copy.deepcopy(value)
        bad_rmsd["banks"]["matched"]["decoys"][0]["rmsd_sym"] += 0.05
        with self.assertRaisesRegex(ValueError, "RMSD"):
            decoys_v2.validate_decoy_set_v2(bad_rmsd, native_record=record)

        bad_contact = copy.deepcopy(value)
        entry = bad_contact["banks"]["matched"]["decoys"][0]
        entry["contact_stats"]["decoy_contact_mass"] += 0.25
        entry["contact_stats"]["contact_mass_ratio"] = (
            entry["contact_stats"]["decoy_contact_mass"]
            / entry["contact_stats"]["native_contact_mass"]
        )
        with self.assertRaisesRegex(ValueError, "contact"):
            decoys_v2.validate_decoy_set_v2(
                bad_contact, native_record=record
            )

        bad_overlap = copy.deepcopy(value)
        bad_overlap["banks"]["matched"]["decoys"][0][
            "overlap_stats"
        ]["sum_overlap_severity"] += 0.01
        with self.assertRaisesRegex(ValueError, "overlap"):
            decoys_v2.validate_decoy_set_v2(
                bad_overlap, native_record=record
            )

        bad_match = copy.deepcopy(value)
        match_entry = bad_match["banks"]["matched"]["decoys"][0]
        match_entry["match_values"]["delta_minimum_distance"] += 0.1
        match_entry["match_distance"] = decoys_v2.normalized_match_distance(
            match_entry["match_values"], match_entry["stratum"]
        )
        with self.assertRaisesRegex(ValueError, "match"):
            decoys_v2.validate_decoy_set_v2(
                bad_match, native_record=record
            )


class RotationDominantV2BuilderTest(unittest.TestCase):
    MANIFEST_KEYS = ("train_a", "diag_b", "train_c", "diag_a", "train_d")
    DIAGNOSTIC_KEYS = ("diag_a", "diag_b")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "train_manifest.json"
        self.native_path = self.root / "native"
        self.diagnostic_path = self.root / "diagnostic.json"
        self.output_index = 0
        self.generation_modes = {}
        self._write_manifest("train", self.MANIFEST_KEYS)
        self._write_native_records(self.MANIFEST_KEYS)
        self._write_diagnostic(self.DIAGNOSTIC_KEYS)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_json(self, path, payload):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_manifest(self, split, keys):
        samples = [
            {
                "lmdb_key": key,
                "label": float(index + 1),
            }
            for index, key in enumerate(keys)
        ]
        self._write_json(
            self.manifest_path,
            {"split": split, "samples": samples},
        )

    def _native_record(self, key, marker=0):
        return {
            "key": key,
            "ligand_coords": np.asarray(
                [[1.5 + marker * 0.01, 0.0, 0.0]], dtype=np.float64
            ),
            "ligand_atomic_numbers": np.asarray([6], dtype=np.int64),
            "ligand_automorphisms": np.asarray([[0]], dtype=np.int64),
            "ligand_heavy_atom_count": 1,
            "protein_coords": np.asarray(
                [
                    [0.0, float(index) * 0.1, 0.0]
                    for index in range(8)
                ],
                dtype=np.float64,
            ),
            "protein_element_symbol": np.asarray(["C"] * 8),
            "protein_atom_count": 8,
        }

    def _write_native_records(self, keys):
        environment = lmdb.open(
            str(self.native_path),
            map_size=16 * 1024 * 1024,
            subdir=True,
        )
        try:
            with environment.begin(write=True) as transaction:
                for index, key in enumerate(keys):
                    transaction.put(
                        key.encode("utf-8"),
                        pickle.dumps(
                            self._native_record(key, marker=index),
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                    )
        finally:
            environment.close()
        self._write_json(
            self.native_path / "metadata.json",
            {"schema_version": 1, "fixture": "native"},
        )

    def _write_diagnostic(self, keys, manifest=None):
        self._write_json(
            self.diagnostic_path,
            {
                "schema_version": 1,
                "manifest": (
                    manifest_fingerprint(self.manifest_path)
                    if manifest is None
                    else manifest
                ),
                "keys": list(keys),
            },
        )

    def _args(self, cohort="diagnostic", output=None, **overrides):
        if output is None:
            self.output_index += 1
            output = self.root / f"output_{self.output_index}"
        values = {
            "manifest": str(self.manifest_path),
            "native_lmdb": str(self.native_path),
            "diagnostic_keys": str(self.diagnostic_path),
            "cohort": cohort,
            "output_lmdb": str(output),
            "seed": 42,
            "map_size_gb": 0.01,
            "max_samples": None,
            "progress_every": 0,
            "max_error_preview": 2,
            "resume": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _metric_entry(key, bank, stratum, index):
        value = float(index + 1)
        return {
            "transform": {
                "center": [0.0, 0.0, 0.0],
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [value * 0.01, 0.0, 0.0],
            },
            "transform_fingerprint": hashlib.sha256(
                f"{key}:{bank}:{stratum}:{index}".encode("utf-8")
            ).hexdigest(),
            "rmsd_sym": value,
            "contact_stats": {
                "native_contact_mass": 10.0,
                "decoy_contact_mass": 10.0 + value,
                "contact_mass_ratio": 1.0 + value / 10.0,
                "soft_jaccard": value / 10.0,
                "native_neighbor_count": 1,
                "decoy_neighbor_count": 1,
                "coverage_retained": True,
                "contact_pattern_nonempty": True,
            },
            "overlap_stats": {
                "minimum_distance_ratio": 0.75 + value / 100.0,
                "max_overlap_severity": value / 100.0,
                "sum_overlap_severity": value / 50.0,
                "count_below_0_9": index,
                "count_below_0_8": index + 1,
                "has_hard_clash": False,
            },
            "stratum": stratum,
            "failure_reason": None,
            "match_values": {
                "contact_mass_ratio": 1.0 + value / 10.0,
                "delta_centroid_distance": value / 20.0,
                "delta_minimum_distance": -value / 25.0,
                "delta_max_overlap_severity": value / 100.0,
                "delta_sum_overlap_per_ligand_atom": value / 50.0,
                "delta_count_below_0_9_per_ligand_atom": value / 40.0,
                "delta_count_below_0_8_per_ligand_atom": value / 30.0,
            },
            "match_distance": value / 5.0,
            "match_pass": bank == "matched",
        }

    def _fake_decoy_set(self, record, seed):
        mode = self.generation_modes.get(record["key"], (True, True))
        if isinstance(mode, BaseException):
            raise mode
        random_complete, matched_complete = mode
        banks = {}
        for bank, complete in (
            ("random_physical", random_complete),
            ("matched", matched_complete),
        ):
            reason = (
                None
                if complete
                else f"insufficient_{bank}_candidates:low,medium,high"
            )
            banks[bank] = {
                "complete": complete,
                "failure_reason": reason,
                "decoys": (
                    [
                        self._metric_entry(
                            record["key"], bank, stratum, index
                        )
                        for stratum in STRATUM_ORDER
                        for index in range(RETAINED_PER_STRATUM)
                    ]
                    if complete
                    else []
                ),
            }
        return {
            "schema_version": DECOY_SCHEMA_VERSION,
            "key": record["key"],
            "seed": seed,
            "proposal_count_per_stratum": PROPOSALS_PER_STRATUM,
            "automorphism_count": len(record["ligand_automorphisms"]),
            "automorphism_sha256": automorphism_digest(
                record["ligand_automorphisms"]
            ),
            "proposal_fingerprint_sha256": "a" * 64,
            "banks": banks,
            "proposal_counts": {
                stratum: {
                    "proposed": PROPOSALS_PER_STRATUM,
                    "physical_accepted": 8,
                    "match_accepted": 6,
                    "shortlisted": 6,
                    "random_physical_retained": (
                        RETAINED_PER_STRATUM if random_complete else 0
                    ),
                    "matched_retained": (
                        RETAINED_PER_STRATUM if matched_complete else 0
                    ),
                }
                for stratum in STRATUM_ORDER
            },
            "rejection_counts": {
                stratum: {
                    "outside_frozen_strata": PROPOSALS_PER_STRATUM - 8,
                    "match_delta_centroid_distance": 2,
                }
                for stratum in STRATUM_ORDER
            },
        }

    def _build(self, args, validator=None):
        if validator is None:
            validator = lambda value, native_record=None: value
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=validator,
        ):
            return builder_v2.build_decoy_cache_v2(args)

    @staticmethod
    def _read_output(output):
        output = Path(output)
        environment = lmdb.open(
            str(output),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            report, metadata, _ = (
                builder_v2._verify_completed_cache(
                    output, environment
                )
            )
            keys = builder_v2._output_record_keys(environment)
        finally:
            environment.close()
        return report, keys, metadata

    def test_builder_freezes_diagnostic_and_train_cohort_order(self):
        diagnostic_args = self._args(cohort="diagnostic")
        diagnostic = self._build(diagnostic_args)
        self.assertEqual(diagnostic["cohort"], "diagnostic")
        self.assertEqual(
            diagnostic["requested_keys"], ["diag_a", "diag_b"]
        )
        self.assertEqual(
            diagnostic["cohort_selection"]["full_cohort_count"], 2
        )

        train_args = self._args(cohort="train")
        train = self._build(train_args)
        self.assertEqual(train["cohort"], "train")
        self.assertEqual(
            train["requested_keys"], ["train_a", "train_c", "train_d"]
        )
        self.assertTrue(
            set(train["requested_keys"]).isdisjoint(self.DIAGNOSTIC_KEYS)
        )

    def test_max_samples_is_post_selection_prefix_and_frozen_for_evaluators(self):
        diagnostic_args = self._args(
            cohort="diagnostic", max_samples=1
        )
        diagnostic = self._build(diagnostic_args)
        self.assertEqual(diagnostic["requested_keys"], ["diag_a"])
        self.assertEqual(
            diagnostic["cohort_selection"],
            {
                "full_cohort_count": 2,
                "requested_count": 1,
                "max_samples": 1,
                "max_samples_applied": True,
                "truncated_by_max_samples": True,
            },
        )
        self.assertEqual(diagnostic["provenance"]["max_samples"], 1)
        self.assertEqual(
            diagnostic["provenance"]["selected_key_count"], 1
        )

        train = self._build(
            self._args(cohort="train", max_samples=2)
        )
        self.assertEqual(
            train["requested_keys"], ["train_a", "train_c"]
        )
        self.assertEqual(
            train["cohort_selection"]["full_cohort_count"], 3
        )

    def test_builder_rejects_nontrain_bad_diagnostic_and_invalid_limits(self):
        self._write_manifest("valid", self.MANIFEST_KEYS)
        with self.assertRaisesRegex(ValueError, "split=train"):
            self._build(self._args())
        self._write_manifest("train", self.MANIFEST_KEYS)
        self._write_diagnostic(self.DIAGNOSTIC_KEYS)

        cases = [
            (
                "duplicate diagnostic",
                lambda: self._write_diagnostic(("diag_a", "diag_a")),
                "duplicates",
            ),
            (
                "unknown diagnostic",
                lambda: self._write_diagnostic(("diag_a", "unknown")),
                "not present",
            ),
            (
                "diagnostic manifest mismatch",
                lambda: self._write_diagnostic(
                    self.DIAGNOSTIC_KEYS,
                    manifest={
                        **manifest_fingerprint(self.manifest_path),
                        "sha256": "0" * 64,
                    },
                ),
                "different manifest",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(label=label):
                self._write_diagnostic(self.DIAGNOSTIC_KEYS)
                mutate()
                with self.assertRaisesRegex(ValueError, message):
                    self._build(self._args())

        self._write_diagnostic(self.DIAGNOSTIC_KEYS)
        invalid_limits = (
            ({"max_samples": 0}, "max_samples"),
            ({"max_samples": -1}, "max_samples"),
            ({"progress_every": -1}, "progress_every"),
            ({"max_error_preview": -1}, "max_error_preview"),
            ({"map_size_gb": 0}, "map_size_gb"),
            ({"map_size_gb": -1}, "map_size_gb"),
        )
        for overrides, message in invalid_limits:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self._build(self._args(**overrides))

    def test_writes_when_either_bank_is_complete_and_uses_requested_denominator(self):
        self.generation_modes = {
            "train_a": (True, True),
            "train_c": (True, False),
            "train_d": (False, False),
        }
        args = self._args(cohort="train")
        report = self._build(args)
        persisted, keys, metadata = self._read_output(args.output_lmdb)

        self.assertEqual(keys, ["train_a", "train_c"])
        completion = report["bank_completion"]
        self.assertEqual(completion["requested_denominator"], 3)
        self.assertEqual(
            completion["random_physical"],
            {"complete_count": 2, "requested_fraction": 2.0 / 3.0},
        )
        self.assertEqual(
            completion["matched"],
            {"complete_count": 1, "requested_fraction": 1.0 / 3.0},
        )
        self.assertEqual(
            completion["intersection"],
            {"complete_count": 1, "requested_fraction": 1.0 / 3.0},
        )
        self.assertEqual(len(report["excluded_samples"]), 1)
        excluded = report["excluded_samples"][0]
        self.assertEqual(excluded["key"], "train_d")
        self.assertEqual(excluded["error_type"], "IncompleteBanksError")
        self.assertEqual(
            set(excluded["bank_reasons"]),
            {"random_physical", "matched"},
        )
        self.assertEqual(
            report["bank_exclusion_reasons"]["matched"][
                "insufficient_matched_candidates:low,medium,high"
            ],
            2,
        )
        self.assertEqual(persisted, report)
        self.assertEqual(metadata, persisted)


    def test_exact_resume_skips_existing_and_retries_ordinary_failures(self):
        self.generation_modes = {
            "diag_a": (True, True),
            "diag_b": DecoyGenerationError(
                "native_hard_clash: temporary diagnostic failure",
                {"native_hard_clash": 1},
            ),
        }
        args = self._args(cohort="diagnostic")
        first = self._build(args)
        self.assertEqual(first["invocation_counts"]["written"], 1)
        self.assertEqual(first["invocation_counts"]["errors"], 1)

        self.generation_modes = {}
        resume_args = self._args(
            cohort="diagnostic",
            output=args.output_lmdb,
            resume=True,
        )
        resumed = self._build(resume_args)
        self.assertEqual(
            resumed["invocation_counts"],
            {
                "processed": 2,
                "written": 1,
                "skipped_existing": 1,
                "errors": 0,
            },
        )
        self.assertEqual(resumed["cache_counts"]["cached_total"], 2)
        self.assertEqual(resumed["cache_counts"]["excluded_total"], 0)
        self.assertEqual(resumed["included_keys"], ["diag_a", "diag_b"])

    def test_resume_rejects_every_provenance_change_before_output_open(self):
        args = self._args(cohort="diagnostic")
        baseline = self._build(args)
        output = Path(args.output_lmdb)
        report_path = output / "decoy_report.json"
        baseline_report_bytes = report_path.read_bytes()

        def rejected(changed_args, context=None):
            context = context or patch.object(
                builder_v2, "PROPOSALS_PER_STRATUM", PROPOSALS_PER_STRATUM
            )
            with context, patch.object(
                builder_v2,
                "_open_output_lmdb",
                side_effect=AssertionError("output LMDB opened"),
            ):
                with self.assertRaisesRegex(ValueError, "resume provenance"):
                    self._build(changed_args)
            self.assertEqual(report_path.read_bytes(), baseline_report_bytes)

        rejected(
            self._args(
                cohort="diagnostic",
                output=output,
                resume=True,
                seed=43,
            )
        )
        rejected(
            self._args(
                cohort="train",
                output=output,
                resume=True,
            )
        )
        rejected(
            self._args(
                cohort="diagnostic",
                output=output,
                resume=True,
                max_samples=1,
            )
        )

        manifest_bytes = self.manifest_path.read_bytes()
        diagnostic_bytes = self.diagnostic_path.read_bytes()
        payload = json.loads(manifest_bytes)
        payload["samples"][0], payload["samples"][2] = (
            payload["samples"][2],
            payload["samples"][0],
        )
        self._write_json(self.manifest_path, payload)
        self._write_diagnostic(self.DIAGNOSTIC_KEYS)
        rejected(
            self._args(
                cohort="diagnostic", output=output, resume=True
            )
        )
        self.manifest_path.write_bytes(manifest_bytes)
        self.diagnostic_path.write_bytes(diagnostic_bytes)

        environment = lmdb.open(str(self.native_path), subdir=True)
        try:
            with environment.begin(write=True) as transaction:
                original_raw = transaction.get(b"diag_a")
                changed_record = pickle.loads(original_raw)
                changed_record["resume_marker"] = "changed bytes"
                transaction.put(
                    b"diag_a",
                    pickle.dumps(
                        changed_record, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
        finally:
            environment.close()
        rejected(
            self._args(
                cohort="diagnostic", output=output, resume=True
            )
        )
        environment = lmdb.open(str(self.native_path), subdir=True)
        try:
            with environment.begin(write=True) as transaction:
                transaction.put(b"diag_a", original_raw)
        finally:
            environment.close()

        rejected(
            self._args(
                cohort="diagnostic", output=output, resume=True
            ),
            patch.object(
                builder_v2,
                "automorphism_digest",
                return_value="f" * 64,
            ),
        )

        semantic = json.loads(diagnostic_bytes)
        self.diagnostic_path.write_text(
            json.dumps(semantic, separators=(",", ":")),
            encoding="utf-8",
        )
        rejected(
            self._args(
                cohort="diagnostic", output=output, resume=True
            )
        )
        self.diagnostic_path.write_bytes(diagnostic_bytes)

        self._write_diagnostic(tuple(reversed(self.DIAGNOSTIC_KEYS)))
        rejected(
            self._args(
                cohort="diagnostic", output=output, resume=True
            )
        )
        self.diagnostic_path.write_bytes(diagnostic_bytes)

        rejected(
            self._args(
                cohort="diagnostic", output=output, resume=True
            ),
            patch.object(
                builder_v2,
                "PROPOSALS_PER_STRATUM",
                PROPOSALS_PER_STRATUM + 1,
            ),
        )
        rejected(
            self._args(
                cohort="diagnostic", output=output, resume=True
            ),
            patch.object(
                builder_v2,
                "MATCH_LIMITS",
                {**MATCH_LIMITS, "minimum_distance": 0.51},
            ),
        )

        provenance = baseline["provenance"]
        self.assertEqual(
            {
                "builder",
                "schema_version",
                "cohort",
                "seed",
                "max_samples",
                "manifest",
                "selected_key_count",
                "selected_key_sha256",
                "native_lmdb",
                "native_key_count",
                "native_key_sha256",
                "selected_native_record_sha256",
                "diagnostic_keys_path",
                "diagnostic_file_sha256",
                "diagnostic_ordered_key_sha256",
                "diagnostic_manifest",
                "generator_configuration",
                "rdkit_runtime_version",
                "selected_automorphism_sha256",
                "native_metadata_sha256",
                "native_schema_version",
            },
            set(provenance),
        )
        configuration = provenance["generator_configuration"]
        self.assertEqual(
            {
                "bank_names",
                "stratum_order",
                "proposal_ranges",
                "proposal_sampling",
                "physical_filters",
                "matching_rules",
                "native_eligibility",
                "geometry_tolerances",
                "continuous_contact_weights",
                "overlap_statistics",
                "proposals_per_stratum",
                "retained_per_stratum",
                "shortlist_size",
                "stratum_bounds",
                "hard_vdw_ratio",
                "contact_mass_ratio_range",
                "centroid_limits",
                "match_limits",
                "rng_bit_generator",
                "rng_purpose_namespace_templates",
                "symmetry",
                "shortlist_rule",
                "retention_rules",
            },
            set(configuration),
        )

    def test_provenance_freezes_exact_generation_semantics(self):
        report = self._build(self._args(cohort="diagnostic"))
        configuration = report["provenance"]["generator_configuration"]

        self.assertEqual(
            configuration["continuous_contact_weights"],
            {
                "distance_unit": "angstrom",
                "piecewise": [
                    {"condition": "d<=5.0", "weight": "1.0"},
                    {
                        "condition": "5.0<d<6.0",
                        "weight": (
                            "0.5*(cos(pi*(d-5.0))+1.0)"
                        ),
                    },
                    {"condition": "d>=6.0", "weight": "0.0"},
                ],
            },
        )
        self.assertEqual(
            configuration["overlap_statistics"],
            {
                "distance_ratio": "distance/vdw_radius_sum",
                "count_below_0_9": {
                    "comparison": "<",
                    "threshold": 0.9,
                },
                "count_below_0_8": {
                    "comparison": "<",
                    "threshold": 0.8,
                },
                "hard_clash": {
                    "comparison": "<",
                    "threshold": 0.75,
                },
                "severity": "max(0.0,1.0-distance_ratio)",
            },
        )
        self.assertEqual(
            configuration["matching_rules"]["distance"],
            {
                "aggregation": "sum",
                "terms": [
                    {
                        "numerator": (
                            "abs(log(contact_mass_ratio))"
                        ),
                        "divisor": {
                            "formula": "log(1.15)",
                            "reference": 1.15,
                        },
                    },
                    {
                        "numerator": (
                            "abs(delta_centroid_distance)"
                        ),
                        "divisor": {
                            "lookup": "centroid_limits[stratum]",
                            "values": {
                                "low": 0.5,
                                "medium": 1.0,
                                "high": 1.5,
                            },
                        },
                    },
                    {
                        "numerator": (
                            "abs(delta_minimum_distance)"
                        ),
                        "divisor": 0.5,
                    },
                    {
                        "numerator": (
                            "abs(delta_max_overlap_severity)"
                        ),
                        "divisor": 0.1,
                    },
                    {
                        "numerator": (
                            "abs(delta_sum_overlap_per_ligand_atom)"
                        ),
                        "divisor": 0.05,
                    },
                    {
                        "numerator": (
                            "abs(delta_count_below_0_9_per_ligand_atom)"
                        ),
                        "divisor": 0.12,
                    },
                    {
                        "numerator": (
                            "abs(delta_count_below_0_8_per_ligand_atom)"
                        ),
                        "divisor": 0.08,
                    },
                ],
            },
        )
        self.assertEqual(
            configuration["native_eligibility"][
                "minimum_native_contact_mass"
            ],
            "max(8.0,0.5*ligand_heavy_atom_count)",
        )
        self.assertEqual(
            configuration["rng_purpose_namespace_templates"][
                "epoch_selection"
            ],
            (
                "{seed}<NUL>{key}<NUL>{stratum}<NUL>"
                "epoch_selection:{epoch}:{bank}"
            ),
        )
        self.assertEqual(
            configuration["geometry_tolerances"],
            {"absolute": 1e-9, "relative": 1e-7},
        )

    def test_validator_and_generator_structural_errors_are_fatal(self):
        validator_args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=ValueError("invalid generated schema"),
        ):
            with self.assertRaisesRegex(
                ValueError, "invalid generated schema"
            ):
                builder_v2.build_decoy_cache_v2(validator_args)

        generator_args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=ValueError("unknown protein element"),
        ):
            with self.assertRaisesRegex(
                ValueError, "unknown protein element"
            ):
                builder_v2.build_decoy_cache_v2(generator_args)

    def test_generic_lmdb_and_serialization_errors_are_fatal(self):
        lmdb_args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ), patch.object(
            builder_v2,
            "_write_record",
            side_effect=lmdb.Error("generic LMDB failure"),
        ):
            with self.assertRaisesRegex(
                lmdb.Error, "generic LMDB failure"
            ):
                builder_v2.build_decoy_cache_v2(lmdb_args)

        real_dumps = pickle.dumps

        def fail_decoy_serialization(value, *args, **kwargs):
            if isinstance(value, dict) and "banks" in value:
                raise pickle.PicklingError("record serialization failed")
            return real_dumps(value, *args, **kwargs)

        serialization_args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ), patch.object(
            builder_v2.pickle,
            "dumps",
            side_effect=fail_decoy_serialization,
        ):
            with self.assertRaisesRegex(
                pickle.PicklingError, "record serialization failed"
            ):
                builder_v2.build_decoy_cache_v2(serialization_args)

    def test_resume_preflight_propagates_lmdb_errors(self):
        args = self._args(cohort="diagnostic")
        self._build(args)
        failure = lmdb.Error("resume preflight LMDB failure")
        resume_args = self._args(
            cohort="diagnostic",
            output=Path(args.output_lmdb),
            resume=True,
        )

        with patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=failure,
        ), patch.object(
            builder_v2,
            "_open_output_lmdb",
            side_effect=AssertionError(
                "writable output opened after preflight failure"
            ),
        ) as writable_open:
            with self.assertRaises(lmdb.Error) as raised:
                builder_v2.build_decoy_cache_v2(resume_args)

        self.assertIs(raised.exception, failure)
        writable_open.assert_not_called()

    def test_hard_clash_and_low_contact_are_explicit_exclusions(self):
        hard_clash_args = self._args(cohort="diagnostic")

        def hard_clash_generator(record, seed):
            if record["key"] == "diag_a":
                raise DecoyGenerationError(
                    "native_hard_clash: fixture",
                    {"native_hard_clash": 1},
                )
            return self._fake_decoy_set(record, seed)

        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=hard_clash_generator,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            hard_clash_report = builder_v2.build_decoy_cache_v2(
                hard_clash_args
            )

        hard_clash_row = hard_clash_report["excluded_samples"][0]
        self.assertEqual(
            hard_clash_row["exclusion_reason"], "native_hard_clash"
        )
        self.assertEqual(
            hard_clash_row["bank_reasons"],
            {
                "random_physical": "native_hard_clash",
                "matched": "native_hard_clash",
            },
        )
        for bank in BANK_NAMES:
            self.assertEqual(
                hard_clash_report["bank_exclusion_reasons"][bank],
                {"native_hard_clash": 1},
            )

        environment = lmdb.open(str(self.native_path), subdir=True)
        try:
            with environment.begin(write=True) as transaction:
                record = pickle.loads(transaction.get(b"diag_a"))
                record["protein_coords"] = np.asarray(
                    [[20.0, 0.0, 0.0]], dtype=np.float64
                )
                record["protein_element_symbol"] = np.asarray(["C"])
                record["protein_atom_count"] = 1
                transaction.put(
                    b"diag_a",
                    pickle.dumps(
                        record, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
        finally:
            environment.close()

        generated_keys = []

        def low_contact_generator(record, seed):
            generated_keys.append(record["key"])
            if record["key"] == "diag_a":
                return decoys_v2.generate_decoy_set_v2(record, seed)
            return self._fake_decoy_set(record, seed)

        low_contact_args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=low_contact_generator,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            low_contact_report = builder_v2.build_decoy_cache_v2(
                low_contact_args
            )

        low_contact_row = low_contact_report["excluded_samples"][0]
        self.assertEqual(
            low_contact_row["exclusion_reason"], "native_contact_mass"
        )
        self.assertEqual(
            low_contact_row["bank_reasons"],
            {
                "random_physical": "native_contact_mass",
                "matched": "native_contact_mass",
            },
        )
        self.assertIn("diag_a", generated_keys)
        for bank in BANK_NAMES:
            self.assertEqual(
                low_contact_report["bank_exclusion_reasons"][bank],
                {"native_contact_mass": 1},
            )

    def test_malformed_low_contact_native_records_are_fatal(self):
        cases = (
            (
                "missing atomic numbers",
                lambda record: record.pop("ligand_atomic_numbers"),
                "missing ligand_atomic_numbers",
            ),
            (
                "invalid element",
                lambda record: record.__setitem__(
                    "ligand_atomic_numbers", np.asarray([0])
                ),
                "ligand_atomic_numbers contains invalid atomic number",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                self._write_native_records(self.MANIFEST_KEYS)
                environment = lmdb.open(str(self.native_path), subdir=True)
                try:
                    with environment.begin(write=True) as transaction:
                        record = pickle.loads(transaction.get(b"diag_a"))
                        record["protein_coords"] = np.asarray(
                            [[20.0, 0.0, 0.0]], dtype=np.float64
                        )
                        record["protein_element_symbol"] = np.asarray(
                            ["C"]
                        )
                        record["protein_atom_count"] = 1
                        mutate(record)
                        transaction.put(
                            b"diag_a",
                            pickle.dumps(
                                record, protocol=pickle.HIGHEST_PROTOCOL
                            ),
                        )
                finally:
                    environment.close()

                def structural_generator(record, seed):
                    if record["key"] == "diag_a":
                        return decoys_v2.generate_decoy_set_v2(
                            record, seed
                        )
                    raise AssertionError(
                        "builder continued after malformed diag_a"
                    )

                args = self._args(cohort="diagnostic")
                with patch.object(
                    builder_v2,
                    "generate_decoy_set_v2",
                    side_effect=structural_generator,
                ):
                    with self.assertRaisesRegex(ValueError, expected):
                        builder_v2.build_decoy_cache_v2(args)

    def test_ordinary_exclusions_still_emit_bounded_progress(self):
        args = self._args(
            cohort="diagnostic",
            progress_every=1,
            max_error_preview=1,
        )

        def hard_clash_generator(record, seed):
            raise DecoyGenerationError(
                "native_hard_clash: fixture",
                {"native_hard_clash": 1},
            )

        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=hard_clash_generator,
        ), patch("builtins.print") as terminal_print:
            report = builder_v2.build_decoy_cache_v2(args)

        self.assertEqual(report["invocation_counts"]["errors"], 2)
        self.assertEqual(terminal_print.call_count, 2)
        payloads = [
            json.loads(call.args[0])
            for call in terminal_print.call_args_list
        ]
        self.assertEqual(
            [payload["progress"]["processed"] for payload in payloads],
            [1, 2],
        )
        self.assertTrue(
            all(len(payload["errors_preview"]) == 1 for payload in payloads)
        )

    def test_generated_records_use_native_free_validation_only(self):
        validator = Mock(
            side_effect=lambda value, native_record=None: value
        )
        args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            validator,
        ):
            builder_v2.build_decoy_cache_v2(args)

        self.assertEqual(validator.call_count, 2)
        for call in validator.call_args_list:
            self.assertEqual(len(call.args), 1)
            self.assertEqual(call.kwargs, {})

    def test_record_bytes_have_atomic_digest_keys_and_record_enumeration(self):
        args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ):
            builder_v2.build_decoy_cache_v2(args)

        environment = lmdb.open(
            str(args.output_lmdb),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            self.assertEqual(
                builder_v2._output_record_keys(environment),
                ["diag_a", "diag_b"],
            )
            with environment.begin(write=False) as transaction:
                for key in ("diag_a", "diag_b"):
                    raw = transaction.get(key.encode("utf-8"))
                    stored_digest = transaction.get(
                        b"__record_sha256__:" + key.encode("utf-8")
                    )
                    self.assertEqual(
                        stored_digest,
                        hashlib.sha256(raw).hexdigest().encode("ascii"),
                    )
        finally:
            environment.close()

    def test_resume_rejects_missing_orphan_and_mismatched_record_digests(self):
        cases = (
            (
                "missing",
                lambda transaction: transaction.delete(
                    b"__record_sha256__:diag_a"
                ),
                "missing record digest",
            ),
            (
                "orphan",
                lambda transaction: transaction.delete(b"diag_a"),
                "orphan record digest",
            ),
            (
                "mismatched",
                lambda transaction: transaction.put(
                    b"__record_sha256__:diag_a", b"0" * 64
                ),
                "record digest mismatch",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                args = self._args(cohort="diagnostic")
                self._build(args)
                output_path = Path(args.output_lmdb)
                report_path = output_path / "decoy_report.json"
                environment = lmdb.open(str(output_path), subdir=True)
                try:
                    with environment.begin(write=True) as transaction:
                        mutate(transaction)
                finally:
                    environment.close()
                report_bytes = report_path.read_bytes()

                resume_args = self._args(
                    cohort="diagnostic",
                    output=output_path,
                    resume=True,
                )
                with patch.object(
                    builder_v2,
                    "atomic_write_json",
                    side_effect=AssertionError(
                        "report mutated before digest preflight"
                    ),
                ), patch.object(
                    builder_v2,
                    "_open_output_lmdb",
                    side_effect=AssertionError(
                        "writable output opened before digest preflight"
                    ),
                ) as writable_open:
                    with self.assertRaisesRegex(ValueError, expected):
                        self._build(resume_args)
                writable_open.assert_not_called()
                self.assertEqual(report_path.read_bytes(), report_bytes)

    def test_resume_rejects_schema_valid_geometry_mutation_by_digest(self):
        _, template = make_real_decoy_set()

        def valid_generator(record, seed):
            value = copy.deepcopy(template)
            value["key"] = record["key"]
            value["seed"] = seed
            value["automorphism_count"] = len(
                record["ligand_automorphisms"]
            )
            value["automorphism_sha256"] = automorphism_digest(
                record["ligand_automorphisms"]
            )
            return value

        args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=valid_generator,
        ):
            builder_v2.build_decoy_cache_v2(args)

        output_path = Path(args.output_lmdb)
        report_path = output_path / "decoy_report.json"
        environment = lmdb.open(str(output_path), subdir=True)
        try:
            with environment.begin(write=True) as transaction:
                raw = transaction.get(b"diag_a")
                value = pickle.loads(raw)
                value["banks"]["matched"]["decoys"][0][
                    "rmsd_sym"
                ] += 1e-10
                decoys_v2.validate_decoy_set_v2(value)
                transaction.put(
                    b"diag_a",
                    pickle.dumps(
                        value, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
        finally:
            environment.close()
        report_bytes = report_path.read_bytes()

        validator = Mock(
            side_effect=AssertionError(
                "validator ran before record digest verification"
            )
        )
        resume_args = self._args(
            cohort="diagnostic",
            output=output_path,
            resume=True,
        )
        with patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            validator,
        ), patch.object(
            builder_v2,
            "atomic_write_json",
            side_effect=AssertionError(
                "report mutated before digest preflight"
            ),
        ), patch.object(
            builder_v2,
            "_open_output_lmdb",
            side_effect=AssertionError(
                "writable output opened before digest preflight"
            ),
        ) as writable_open:
            with self.assertRaisesRegex(
                ValueError, "record digest mismatch"
            ):
                builder_v2.build_decoy_cache_v2(resume_args)

        validator.assert_not_called()
        writable_open.assert_not_called()
        self.assertEqual(report_path.read_bytes(), report_bytes)

    def test_completion_verifier_rejects_record_inventory_corruption(self):
        def mutate_record(transaction, update_digest):
            value = pickle.loads(transaction.get(b"diag_a"))
            value["banks"]["matched"]["decoys"][0][
                "rmsd_sym"
            ] += 1e-10
            raw = pickle.dumps(
                value, protocol=pickle.HIGHEST_PROTOCOL
            )
            transaction.put(b"diag_a", raw)
            if update_digest:
                transaction.put(
                    b"__record_sha256__:diag_a",
                    hashlib.sha256(raw).hexdigest().encode("ascii"),
                )

        cases = (
            (
                "deleted record",
                lambda transaction: transaction.delete(b"diag_a"),
                "orphan record digest",
            ),
            (
                "raw mutation",
                lambda transaction: mutate_record(transaction, False),
                "record digest mismatch",
            ),
            (
                "digest mutation",
                lambda transaction: transaction.put(
                    b"__record_sha256__:diag_a", b"0" * 64
                ),
                "record digest mismatch",
            ),
            (
                "coordinated raw and digest mutation",
                lambda transaction: mutate_record(transaction, True),
                "cache integrity",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                args = self._args(cohort="diagnostic")
                self._build(args)
                output_path = Path(args.output_lmdb)
                environment = lmdb.open(str(output_path), subdir=True)
                try:
                    with environment.begin(write=True) as transaction:
                        mutate(transaction)
                finally:
                    environment.close()

                environment = lmdb.open(
                    str(output_path),
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    subdir=True,
                )
                try:
                    with self.assertRaisesRegex(ValueError, expected):
                        builder_v2._verify_completed_cache(
                            output_path, environment
                        )
                finally:
                    environment.close()

    def test_completion_verifier_rejects_report_inventory_mismatch(self):
        cases = ("included_keys", "cache_integrity")
        for label in cases:
            with self.subTest(label=label):
                args = self._args(cohort="diagnostic")
                self._build(args)
                output_path = Path(args.output_lmdb)
                report_path = output_path / "decoy_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if label == "included_keys":
                    report["included_keys"] = list(
                        reversed(report["included_keys"])
                    )
                else:
                    report["cache_integrity"]["record_count"] += 1
                builder_v2.atomic_write_json(report_path, report)

                environment = lmdb.open(
                    str(output_path),
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    subdir=True,
                )
                try:
                    with self.assertRaisesRegex(
                        ValueError, "cache integrity"
                    ):
                        builder_v2._verify_completed_cache(
                            output_path, environment
                        )
                finally:
                    environment.close()

    def test_completed_resume_rejects_coordinated_record_mutation(self):
        args = self._args(cohort="diagnostic")
        self._build(args)
        output_path = Path(args.output_lmdb)
        report_path = output_path / "decoy_report.json"
        report_bytes = report_path.read_bytes()

        environment = lmdb.open(str(output_path), subdir=True)
        try:
            with environment.begin(write=True) as transaction:
                value = pickle.loads(transaction.get(b"diag_a"))
                value["banks"]["matched"]["decoys"][0][
                    "rmsd_sym"
                ] += 1e-10
                raw = pickle.dumps(
                    value, protocol=pickle.HIGHEST_PROTOCOL
                )
                transaction.put(b"diag_a", raw)
                transaction.put(
                    b"__record_sha256__:diag_a",
                    hashlib.sha256(raw).hexdigest().encode("ascii"),
                )
        finally:
            environment.close()

        resume_args = self._args(
            cohort="diagnostic", output=output_path, resume=True
        )
        with patch.object(
            builder_v2,
            "atomic_write_json",
            side_effect=AssertionError(
                "report mutated before cache integrity preflight"
            ),
        ), patch.object(
            builder_v2,
            "_open_output_lmdb",
            side_effect=AssertionError(
                "writable output opened before cache integrity preflight"
            ),
        ) as writable_open:
            with self.assertRaisesRegex(ValueError, "cache integrity"):
                self._build(resume_args)

        writable_open.assert_not_called()
        self.assertEqual(report_path.read_bytes(), report_bytes)

    def test_in_progress_partial_cache_resumes_with_valid_digests(self):
        args = self._args(cohort="diagnostic")
        real_atomic_write = builder_v2.atomic_write_json

        def fail_final_report(path, payload):
            if payload.get("status") == "complete":
                raise OSError("simulated final JSON failure")
            return real_atomic_write(path, payload)

        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ), patch.object(
            builder_v2,
            "atomic_write_json",
            side_effect=fail_final_report,
        ):
            with self.assertRaisesRegex(
                OSError, "simulated final JSON failure"
            ):
                builder_v2.build_decoy_cache_v2(args)

        output_path = Path(args.output_lmdb)
        interrupted = json.loads(
            (output_path / "decoy_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(interrupted["status"], "in_progress")

        resumed = self._build(
            self._args(
                cohort="diagnostic", output=output_path, resume=True
            )
        )
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(
            resumed["invocation_counts"]["skipped_existing"], 2
        )

    def test_completion_marker_authorizes_report_and_is_overwritten(self):
        args = self._args(cohort="diagnostic")
        first_report = self._build(args)
        output_path = Path(args.output_lmdb)
        report_path = output_path / "decoy_report.json"

        environment = lmdb.open(
            str(output_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            verified, metadata, first_marker = (
                builder_v2._verify_completed_cache(
                    output_path, environment
                )
            )
        finally:
            environment.close()

        self.assertEqual(verified, first_report)
        self.assertEqual(metadata, first_report)
        self.assertEqual(
            first_marker["report_sha256"], sha256_file(report_path)
        )
        self.assertEqual(
            first_marker["provenance"], first_report["provenance"]
        )
        self.assertEqual(
            first_marker["provenance_sha256"],
            builder_v2._canonical_sha256(
                first_report["provenance"]
            ),
        )
        environment = lmdb.open(
            str(output_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            with environment.begin(write=False) as transaction:
                inventory_rows = [
                    {
                        "key": key,
                        "record_sha256": hashlib.sha256(
                            transaction.get(key.encode("utf-8"))
                        ).hexdigest(),
                    }
                    for key in first_report["included_keys"]
                ]
        finally:
            environment.close()
        expected_integrity = {
            "record_count": len(inventory_rows),
            "ordered_key_sha256": builder_v2.hash_ordered_keys(
                first_report["included_keys"]
            ),
            "ordered_record_digest_sha256": (
                builder_v2._canonical_sha256(inventory_rows)
            ),
        }
        self.assertEqual(
            first_report["cache_integrity"], expected_integrity
        )
        self.assertEqual(
            first_marker["cache_integrity"], expected_integrity
        )
        self.assertEqual(
            first_marker["cache_integrity_sha256"],
            builder_v2._canonical_sha256(expected_integrity),
        )

        resume_args = self._args(
            cohort="diagnostic",
            output=output_path,
            resume=True,
        )
        resumed = self._build(resume_args)
        environment = lmdb.open(
            str(output_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            verified_resume, metadata_resume, resumed_marker = (
                builder_v2._verify_completed_cache(
                    output_path, environment
                )
            )
        finally:
            environment.close()

        self.assertEqual(verified_resume, resumed)
        self.assertEqual(metadata_resume, resumed)
        self.assertNotEqual(
            resumed_marker["report_sha256"],
            first_marker["report_sha256"],
        )
        self.assertEqual(
            resumed_marker["report_sha256"], sha256_file(report_path)
        )

    def test_final_json_failure_commits_no_complete_metadata_or_marker(self):
        args = self._args(cohort="diagnostic")
        real_atomic_write = builder_v2.atomic_write_json

        def fail_final_report(path, payload):
            if payload.get("status") == "complete":
                raise OSError("simulated final JSON failure")
            return real_atomic_write(path, payload)

        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ), patch.object(
            builder_v2,
            "atomic_write_json",
            side_effect=fail_final_report,
        ):
            with self.assertRaisesRegex(
                OSError, "simulated final JSON failure"
            ):
                builder_v2.build_decoy_cache_v2(args)

        output_path = Path(args.output_lmdb)
        report = json.loads(
            (output_path / "decoy_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["status"], "in_progress")
        environment = lmdb.open(
            str(output_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            with environment.begin(write=False) as transaction:
                self.assertIsNone(transaction.get(b"__metadata__"))
                self.assertIsNone(transaction.get(b"__completion__"))
        finally:
            environment.close()

    def test_in_progress_report_invalidates_old_completion_marker(self):
        args = self._args(cohort="diagnostic")
        self._build(args)
        output_path = Path(args.output_lmdb)

        resume_args = self._args(
            cohort="diagnostic",
            output=output_path,
            resume=True,
        )
        with patch.object(
            builder_v2,
            "_open_output_lmdb",
            side_effect=RuntimeError("interrupt after in-progress report"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "interrupt after in-progress report"
            ):
                self._build(resume_args)

        environment = lmdb.open(
            str(output_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            with self.assertRaisesRegex(
                ValueError, "completion marker"
            ):
                builder_v2._verify_completed_cache(
                    output_path, environment
                )
        finally:
            environment.close()

    def test_resume_after_report_write_before_lmdb_creation(self):
        args = self._args(cohort="diagnostic")
        output_path = Path(args.output_lmdb)
        report_path = output_path / "decoy_report.json"

        with patch.object(
            builder_v2,
            "_open_output_lmdb",
            side_effect=RuntimeError("simulated interruption"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "simulated interruption"
            ):
                self._build(args)

        self.assertTrue(report_path.exists())
        self.assertFalse((output_path / "data.mdb").exists())
        interrupted_report = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        self.assertEqual(interrupted_report["status"], "in_progress")

        real_open_readonly = builder_v2._open_readonly_lmdb

        def guarded_open_readonly(path):
            if Path(path).resolve() == output_path.resolve():
                raise AssertionError(
                    "absent output LMDB opened read-only during resume"
                )
            return real_open_readonly(path)

        resume_args = self._args(
            cohort="diagnostic",
            output=output_path,
            resume=True,
        )
        with patch.object(
            builder_v2,
            "_open_readonly_lmdb",
            side_effect=guarded_open_readonly,
        ):
            report = self._build(resume_args)

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["invocation_counts"]["written"], 2)
        self.assertEqual(report["cache_counts"]["cached_total"], 2)
        self.assertTrue((output_path / "data.mdb").exists())

    def test_resume_preflights_existing_records_before_report_mutation(self):
        def mutate_pickle(raw, mutation):
            value = pickle.loads(raw)
            mutation(value)
            return pickle.dumps(
                value, protocol=pickle.HIGHEST_PROTOCOL
            )

        cases = (
            ("truncated", lambda raw: b"not-a-pickle"),
            (
                "malformed",
                lambda raw: mutate_pickle(
                    raw, lambda value: value.pop("schema_version")
                ),
            ),
            (
                "key",
                lambda raw: mutate_pickle(
                    raw,
                    lambda value: value.__setitem__(
                        "key", "different_key"
                    ),
                ),
            ),
            (
                "seed",
                lambda raw: mutate_pickle(
                    raw,
                    lambda value: value.__setitem__(
                        "seed", value["seed"] + 1
                    ),
                ),
            ),
            (
                "automorphism count",
                lambda raw: mutate_pickle(
                    raw,
                    lambda value: value.__setitem__(
                        "automorphism_count",
                        value["automorphism_count"] + 1,
                    ),
                ),
            ),
            (
                "automorphism digest",
                lambda raw: mutate_pickle(
                    raw,
                    lambda value: value.__setitem__(
                        "automorphism_sha256", "f" * 64
                    ),
                ),
            ),
        )

        for label, mutate in cases:
            with self.subTest(label=label):
                args = self._args(cohort="diagnostic")
                self._build(args)
                output_path = Path(args.output_lmdb)
                report_path = output_path / "decoy_report.json"

                environment = lmdb.open(str(output_path), subdir=True)
                try:
                    with environment.begin(write=True) as transaction:
                        raw = transaction.get(b"diag_a")
                        transaction.put(b"diag_a", mutate(raw))
                finally:
                    environment.close()

                report_bytes = report_path.read_bytes()

                def native_free_validator(value, native_record=None):
                    self.assertIsNone(native_record)
                    if "schema_version" not in value:
                        raise ValueError("malformed schema")
                    return value

                resume_args = self._args(
                    cohort="diagnostic",
                    output=output_path,
                    resume=True,
                )
                with patch.object(
                    builder_v2,
                    "atomic_write_json",
                    side_effect=AssertionError(
                        "report mutated before resume preflight"
                    ),
                ), patch.object(
                    builder_v2,
                    "_open_output_lmdb",
                    side_effect=AssertionError(
                        "writable output opened during preflight"
                    ),
                ) as writable_open:
                    with self.assertRaisesRegex(
                        ValueError,
                        "existing v2 decoy record is invalid: diag_a",
                    ):
                        self._build(
                            resume_args,
                            validator=native_free_validator,
                        )
                writable_open.assert_not_called()
                self.assertEqual(
                    report_path.read_bytes(), report_bytes
                )

    def test_resume_rejects_unexpected_keys_before_report_mutation(self):
        args = self._args(cohort="diagnostic")
        self._build(args)
        output_path = Path(args.output_lmdb)
        report_path = output_path / "decoy_report.json"

        environment = lmdb.open(str(output_path), subdir=True)
        try:
            with environment.begin(write=True) as transaction:
                transaction.put(
                    b"cohort_external",
                    pickle.dumps(
                        {"schema_version": 1},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                )
        finally:
            environment.close()
        report_bytes = report_path.read_bytes()

        resume_args = self._args(
            cohort="diagnostic",
            output=output_path,
            resume=True,
        )
        with patch.object(
            builder_v2,
            "atomic_write_json",
            side_effect=AssertionError(
                "report mutated before resume preflight"
            ),
        ), patch.object(
            builder_v2,
            "_open_output_lmdb",
            side_effect=AssertionError(
                "writable output opened during preflight"
            ),
        ) as writable_open:
            with self.assertRaisesRegex(
                ValueError,
                "existing v2 decoy cache contains unexpected keys",
            ):
                self._build(resume_args)

        writable_open.assert_not_called()
        self.assertEqual(report_path.read_bytes(), report_bytes)

    def test_map_full_is_fatal_and_ordinary_errors_are_bounded(self):
        fatal_args = self._args(cohort="diagnostic")
        with patch.object(
            builder_v2,
            "generate_decoy_set_v2",
            side_effect=self._fake_decoy_set,
        ), patch.object(
            builder_v2,
            "validate_decoy_set_v2",
            side_effect=lambda value, native_record=None: value,
        ), patch.object(
            builder_v2,
            "_write_record",
            side_effect=lmdb.MapFullError("fixture map full"),
        ):
            with self.assertRaises(lmdb.MapFullError):
                builder_v2.build_decoy_cache_v2(fatal_args)

        self.generation_modes = {
            key: DecoyGenerationError(
                f"native_hard_clash: full failure detail for {key}",
                {"native_hard_clash": 1},
            )
            for key in ("train_a", "train_c", "train_d")
        }
        error_args = self._args(
            cohort="train",
            progress_every=1,
            max_error_preview=1,
        )
        terminal = io.StringIO()
        with redirect_stdout(terminal):
            report = self._build(error_args)
        persisted = json.loads(
            (
                Path(error_args.output_lmdb) / "decoy_report.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(report["excluded_samples"]), 3)
        self.assertEqual(len(report["errors_preview"]), 1)
        self.assertEqual(len(persisted["excluded_samples"]), 3)
        self.assertNotIn("full failure detail for train_d", terminal.getvalue())

    def test_report_has_exact_bank_stratum_quantiles_and_all_metrics(self):
        report = self._build(self._args(cohort="diagnostic"))
        expected_metrics = {
            "rmsd_sym",
            "soft_jaccard",
            "contact_mass_ratio",
            "delta_centroid_distance",
            "delta_minimum_distance",
            "delta_max_overlap_severity",
            "delta_sum_overlap_per_ligand_atom",
            "delta_count_below_0_9_per_ligand_atom",
            "delta_count_below_0_8_per_ligand_atom",
            "minimum_distance_ratio",
            "max_overlap_severity",
            "sum_overlap_severity",
            "count_below_0_9",
            "count_below_0_8",
            "match_distance",
        }
        summary_fields = {
            "count",
            "min",
            "p10",
            "p25",
            "median",
            "p75",
            "p90",
            "max",
        }
        for bank in BANK_NAMES:
            self.assertEqual(
                set(report["distributions"][bank]), set(STRATUM_ORDER)
            )
            for stratum in STRATUM_ORDER:
                metrics = report["distributions"][bank][stratum]
                self.assertEqual(set(metrics), expected_metrics)
                for summary in metrics.values():
                    self.assertEqual(set(summary), summary_fields)

        values = np.asarray([1.0, 2.0, 3.0, 4.0] * 2)
        quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        rmsd = report["distributions"]["matched"]["low"]["rmsd_sym"]
        self.assertEqual(rmsd["count"], 8)
        self.assertEqual(rmsd["min"], 1.0)
        self.assertEqual(rmsd["max"], 4.0)
        for name, expected in zip(
            ("p10", "p25", "median", "p75", "p90"), quantiles
        ):
            self.assertAlmostEqual(rmsd[name], float(expected))

    def test_report_tracks_generation_counts_selection_bias_and_metadata(self):
        self.generation_modes = {
            "train_a": (True, True),
            "train_c": (True, False),
            "train_d": DecoyGenerationError(
                "native_hard_clash: ordinary failure",
                {"native_hard_clash": 1},
            ),
        }
        args = self._args(cohort="train")
        report = self._build(args)
        persisted, keys, metadata = self._read_output(args.output_lmdb)

        self.assertEqual(
            set(report["selection_bias"]), {"included", "excluded"}
        )
        for cohort in ("included", "excluded"):
            self.assertEqual(
                set(report["selection_bias"][cohort]),
                {
                    "sample_count",
                    "affinity",
                    "ligand_size",
                    "pocket_size",
                    "native_contact_mass",
                },
            )
        self.assertEqual(report["selection_bias"]["included"]["sample_count"], 2)
        self.assertEqual(report["selection_bias"]["excluded"]["sample_count"], 1)
        self.assertEqual(
            set(report["generation_counts"]),
            set(BANK_NAMES),
        )
        for bank in BANK_NAMES:
            self.assertEqual(
                set(report["generation_counts"][bank]),
                set(STRATUM_ORDER),
            )
            for stratum in STRATUM_ORDER:
                self.assertEqual(
                    set(report["generation_counts"][bank][stratum]),
                    {
                        "proposed",
                        "physical_accepted",
                        "match_accepted",
                        "shortlisted",
                        "retained",
                    },
                )
        self.assertIn(
            "outside_frozen_strata", report["generation_rejection_counts"]
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["cache_counts"]["cached_total"], len(keys))
        self.assertEqual(report, persisted)
        self.assertEqual(metadata, persisted)

    def test_cli_help_exposes_exact_builder_arguments(self):
        parser = builder_v2.build_arg_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertEqual(
            options,
            {
                "-h",
                "--help",
                "--manifest",
                "--native-lmdb",
                "--diagnostic-keys",
                "--cohort",
                "--output-lmdb",
                "--seed",
                "--map-size-gb",
                "--max-samples",
                "--progress-every",
                "--max-error-preview",
                "--resume",
            },
        )
        actions = {
            action.dest: action for action in parser._actions
        }
        for required in (
            "manifest",
            "native_lmdb",
            "diagnostic_keys",
            "cohort",
            "output_lmdb",
        ):
            self.assertTrue(actions[required].required)
        self.assertEqual(actions["cohort"].choices, ("diagnostic", "train"))
        self.assertEqual(actions["seed"].default, 42)
        self.assertEqual(actions["map_size_gb"].default, 16.0)
        self.assertEqual(actions["progress_every"].default, 25)
        self.assertEqual(actions["max_error_preview"].default, 50)
        help_text = parser.format_help()
        for option in options - {"-h", "--help"}:
            self.assertIn(option, help_text)


if __name__ == "__main__":
    unittest.main()
