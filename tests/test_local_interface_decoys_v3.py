import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import local_interface_decoys_v3 as decoys_v3  # noqa: E402
from scripts import build_local_interface_decoys_v3 as builder_v3  # noqa: E402


class PhysicsGuidedV3ConfigurationTest(unittest.TestCase):
    def test_configuration_is_frozen(self):
        self.assertEqual(decoys_v3.DECOY_SCHEMA_VERSION, 3)
        self.assertEqual(decoys_v3.BASE_ROTATIONS_PER_STRATUM, 512)
        self.assertEqual(decoys_v3.TRANSLATIONS_PER_ROTATION, 4)
        self.assertEqual(decoys_v3.PROPOSALS_PER_STRATUM, 2048)
        self.assertEqual(decoys_v3.RETAINED_PER_STRATUM, 4)
        self.assertEqual(decoys_v3.SHORTLIST_SIZE, 20)
        self.assertEqual(decoys_v3.HARD_VDW_RATIO, 0.75)
        self.assertEqual(decoys_v3.REPULSION_NEAR_RATIO, 1.10)
        self.assertEqual(decoys_v3.ZERO_DISTANCE_EPSILON, 1e-12)
        self.assertEqual(
            decoys_v3.TRANSLATION_GRIDS,
            {
                "low": (0.125, 0.250, 0.375, 0.500),
                "medium": (0.250, 0.500, 0.750, 1.000),
                "high": (0.375, 0.750, 1.125, 1.500),
            },
        )



def make_v3_record(key="complex_a"):
    ligand = np.asarray(
        [[1.5, -0.5, 0.0], [2.5, -0.5, 0.0], [2.0, 0.8, 0.0]],
        dtype=np.float64,
    )
    return {
        "key": key,
        "ligand_coords": ligand,
        "protein_coords": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        ),
        "protein_element_symbol": np.asarray(["C", "C", "C"]),
        "ligand_atomic_numbers": np.asarray([6, 6, 6]),
        "ligand_automorphisms": np.asarray([[0, 1, 2]], dtype=np.int64),
    }


def group_rows(rows, field):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[field], []).append(row)
    return grouped


class PhysicsGuidedV3DirectionTest(unittest.TestCase):
    def test_exact_overlap_is_finite_and_deterministic(self):
        args = dict(
            delta=np.zeros(3),
            ligand_centroid=np.zeros(3),
            protein_atom=np.zeros(3),
            namespace=(42, "k", "high", 7, 2, 5),
        )
        first = decoys_v3._safe_pair_unit_vector(**args)
        second = decoys_v3._safe_pair_unit_vector(**args)
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(np.linalg.norm(first), 1.0)

    def test_hard_pairs_override_many_weak_contacts(self):
        direction, source, zero_count = decoys_v3._local_repulsion_direction(
            ligand=np.asarray([[0.0, 0.0, 0.0]]),
            protein=np.asarray([[-1.0, 0.0, 0.0]] + [[2.8, 0.0, 0.0]] * 20),
            ligand_radii=np.asarray([1.7]),
            protein_radii=np.asarray([1.7] * 21),
            namespace=(42, "k", "high", 0),
        )
        self.assertEqual(source, "hard_clash_repulsion")
        self.assertGreater(direction[0], 0.99)
        self.assertEqual(zero_count, 0)

    def test_symmetric_hard_force_uses_native_interface_normal(self):
        direction, source, zero_count = decoys_v3._proposal_direction(
            rotated_ligand=np.asarray([[0.0, 0.0, 0.0]]),
            native_ligand=np.asarray([[0.0, 2.0, 0.0]]),
            protein=np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            protein_symbols=np.asarray(["C", "C"]),
            ligand_atomic_numbers=np.asarray([6]),
            axis=np.asarray([0.0, 0.0, 1.0]),
            maximum_distance=1.5,
            seed=42,
            key="k",
            stratum="high",
            base_rotation_id=0,
        )
        self.assertEqual(source, "native_interface_normal")
        np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-12)
        self.assertEqual(zero_count, 0)

    def test_signed_axis_never_chooses_worse_endpoint(self):
        axis, sign_rank = decoys_v3._choose_signed_axis(
            rotated_ligand=np.asarray([[0.0, 0.0, 0.0]]),
            protein=np.asarray([[-1.0, 0.0, 0.0]]),
            protein_symbols=np.asarray(["C"]),
            ligand_atomic_numbers=np.asarray([6]),
            axis=np.asarray([1.0, 0.0, 0.0]),
            maximum_distance=1.5,
        )
        self.assertEqual(sign_rank, 0)
        np.testing.assert_array_equal(axis, [1.0, 0.0, 0.0])

    def test_pool_contains_512_families_and_four_grid_translations(self):
        rows = decoys_v3.generate_proposal_pool(make_v3_record(), 42, "medium")
        self.assertEqual(len(rows), 2048)
        self.assertEqual(len({row["transform_fingerprint"] for row in rows}), 2048)
        self.assertEqual(len({row["base_rotation_fingerprint"] for row in rows}), 512)
        required = {
            "base_rotation_id", "base_rotation_fingerprint",
            "translation_index", "translation_distance",
            "translation_direction", "direction_source",
            "zero_distance_pair_count", "zero_translation_vdw_safe",
            "transform_fingerprint",
        }
        self.assertTrue(all(required.issubset(row) for row in rows))
        self.assertTrue(all(row["direction_source"] in decoys_v3.DIRECTION_SOURCES for row in rows))
        families = group_rows(rows, "base_rotation_id")
        self.assertEqual(set(families), set(range(512)))
        for rows in families.values():
            rows.sort(key=lambda row: row["translation_index"])
            self.assertEqual(
                [row["translation_distance"] for row in rows],
                list(decoys_v3.TRANSLATION_GRIDS["medium"]),
            )
            self.assertEqual(len({row["base_rotation_fingerprint"] for row in rows}), 1)
            self.assertEqual(len({tuple(row["translation_direction"]) for row in rows}), 1)

    def test_proposal_diagnostics_track_family_repair_states(self):
        rows = []
        evaluations = []
        for family, source, zero_safe in (
            (0, "hard_clash_repulsion", False),
            (1, "near_contact_repulsion", True),
        ):
            for translation_index in range(4):
                rows.append(
                    {
                        "base_rotation_id": family,
                        "translation_index": translation_index,
                        "direction_source": source,
                        "zero_translation_vdw_safe": zero_safe,
                        "zero_distance_pair_count": 0,
                    }
                )
                if family == 0 and translation_index == 3:
                    evaluations.append(({}, None, None))
                elif family == 1 and translation_index == 0:
                    evaluations.append((None, "wrong_stratum", None))
                else:
                    evaluations.append((None, "vdw_hard_clash", None))

        diagnostics = decoys_v3._summarize_proposal_diagnostics(
            rows, evaluations
        )
        self.assertEqual(
            diagnostics["family_funnel_counts"],
            {
                "base_rotation_families": 2,
                "zero_translation_vdw_safe": 1,
                "any_grid_vdw_safe": 2,
                "any_physical_target_candidate": 1,
                "any_matched_candidate": 1,
                "max_distance_still_hard_clash": 1,
            },
        )
        hard = diagnostics["direction_source_counts"][
            "hard_clash_repulsion"
        ]
        self.assertEqual(hard["proposed_families"], 1)
        self.assertEqual(hard["physical_accepted_transforms"], 1)
        self.assertEqual(hard["match_accepted_transforms"], 1)
        self.assertEqual(
            hard["families_with_any_physical_candidate"], 1
        )
        self.assertEqual(
            hard["families_with_any_matched_candidate"], 1
        )



def make_retention_candidate(family, translation, match_distance, match_pass=True):
    return {
        "base_rotation_id": family,
        "base_rotation_fingerprint": hashlib.sha256(
            f"family:{family}".encode("ascii")
        ).hexdigest(),
        "translation_index": translation,
        "translation_distance": float(translation + 1),
        "direction_source": "hard_clash_repulsion",
        "transform_fingerprint": hashlib.sha256(
            f"transform:{family}:{translation}".encode("ascii")
        ).hexdigest(),
        "match_distance": float(match_distance),
        "match_pass": bool(match_pass),
    }


class PhysicsGuidedV3RetentionTest(unittest.TestCase):
    def test_matched_deduplicates_family_before_top_twenty(self):
        rows = [
            make_retention_candidate(family, translation, family + translation / 10.0)
            for family in range(25)
            for translation in range(4)
        ]
        retained = decoys_v3.retain_matched(rows, 42, "k", "high")
        self.assertEqual(len(retained), 4)
        self.assertEqual(len({row["base_rotation_id"] for row in retained}), 4)
        self.assertTrue(all(row["translation_index"] == 0 for row in retained))
        self.assertTrue(all(row["base_rotation_id"] < 20 for row in retained))

    def test_random_is_family_balanced_and_ignores_matching(self):
        rows = [
            make_retention_candidate(family, translation, 1000.0, False)
            for family in range(8)
            for translation in range(4)
        ]
        first = decoys_v3.retain_random_physical(rows, 42, "k", "low")
        changed = copy.deepcopy(rows)
        for row in changed:
            row["match_distance"] = -999.0
            row["match_pass"] = True
        second = decoys_v3.retain_random_physical(changed, 42, "k", "low")
        self.assertEqual(
            [row["transform_fingerprint"] for row in first],
            [row["transform_fingerprint"] for row in second],
        )
        self.assertEqual(len({row["base_rotation_id"] for row in first}), 4)

    def test_distinct_family_validator_rejects_duplicates(self):
        rows = [
            {"stratum": "low", "base_rotation_id": value}
            for value in (0, 0, 1, 2)
        ]
        with self.assertRaisesRegex(ValueError, "distinct base rotation"):
            decoys_v3._validate_distinct_family_ids(rows, "matched")

    def test_v2_and_v3_schema_identities_are_mutually_rejected(self):
        from data_utils import local_interface_decoys_v2 as decoys_v2
        with self.assertRaisesRegex(ValueError, "schema version 3"):
            decoys_v3.validate_decoy_set_v3({"schema_version": 2})
        with self.assertRaisesRegex(ValueError, "schema version 2"):
            decoys_v2.validate_decoy_set_v2({"schema_version": 3})



class PhysicsGuidedV3BuilderContractTest(unittest.TestCase):
    def test_generator_configuration_records_family_protocol(self):
        value = builder_v3._generator_configuration()
        self.assertEqual(value["schema_version"], 3)
        self.assertEqual(value["base_rotations_per_stratum"], 512)
        self.assertEqual(value["translations_per_rotation"], 4)
        self.assertEqual(value["proposals_per_stratum"], 2048)
        self.assertEqual(
            value["proposal_direction"]["pair_priority"],
            ["q<0.75", "q<1.10"],
        )
        self.assertEqual(
            value["retention_rules"]["bank_complete"],
            "four_distinct_base_rotation_ids_per_stratum",
        )
        self.assertEqual(
            value["proposal_sampling"]["translation_direction"],
            "deterministic_local_repulsion_with_fallback_chain",
        )
        self.assertEqual(
            value["proposal_sampling"]["translation_distance"],
            "fixed_stratum_grid",
        )

    def test_smoke_selection_is_hash_sorted_and_disjoint(self):
        samples = [{"lmdb_key": f"k{i:04d}"} for i in range(1200)]
        diagnostic = [f"k{i:04d}" for i in range(100)]
        actual = builder_v3._select_cohort_samples(
            samples, diagnostic, "generator_smoke", None
        )
        expected = sorted(
            samples[100:],
            key=lambda row: hashlib.sha256(
                b"interface_first_v3_generator_smoke\0"
                + row["lmdb_key"].encode("utf-8")
            ).digest(),
        )[:64]
        self.assertEqual(actual, expected)
        self.assertTrue({row["lmdb_key"] for row in actual}.isdisjoint(diagnostic))

    def test_smoke_rejects_max_samples(self):
        with self.assertRaisesRegex(ValueError, "fixed 64-key"):
            builder_v3._select_cohort_samples(
                [{"lmdb_key": f"k{i}"} for i in range(80)], [],
                "generator_smoke", 8,
            )

    def test_cohort_selection_returns_full_and_requested_samples(self):
        samples = [
            {"lmdb_key": key}
            for key in ("train_a", "diag_a", "train_b", "train_c")
        ]
        full, requested = (
            builder_v3._select_full_and_requested_cohort_samples(
                samples, ["diag_a"], "train", 2
            )
        )
        self.assertEqual(
            [row["lmdb_key"] for row in full],
            ["train_a", "train_b", "train_c"],
        )
        self.assertEqual(
            [row["lmdb_key"] for row in requested],
            ["train_a", "train_b"],
        )



class PhysicsGuidedV3ReportTest(unittest.TestCase):
    def test_rank_bins_cover_every_requested_key_and_warn_only(self):
        keys = [f"k{i:02d}" for i in range(12)]
        properties = {
            key: {
                "ligand_size": float(index + 1),
                "pocket_size": float(100 + index),
                "native_contact_mass": float(200 + index),
            }
            for index, key in enumerate(keys)
        }
        result = builder_v3._rank_bin_completion(
            requested_keys=keys,
            properties_by_key=properties,
            eligible_keys=set(keys),
            complete_keys={
                "random_physical": set(keys),
                "matched": set(keys[3:]),
                "intersection": set(keys[3:]),
            },
        )
        for name in ("ligand_size", "pocket_size", "native_contact_mass"):
            bins = result["properties"][name]["bins"]
            flattened = [key for row in bins for key in row["keys"]]
            self.assertEqual(len(bins), 4)
            self.assertEqual(
                flattened,
                sorted(flattened, key=lambda key: properties[key][name]),
            )
            self.assertEqual(set(flattened), set(keys))
            self.assertTrue(
                all(
                    row["requested_count"] == len(row["keys"])
                    for row in bins
                )
            )
            self.assertTrue(
                all("eligible_count" in row for row in bins)
            )
            self.assertTrue(all("value_min" in row for row in bins))
            self.assertTrue(all("value_max" in row for row in bins))
        self.assertIn(
            "ligand_size_rank_bin_0_zero_matched_complete",
            result["warnings"],
        )

    def test_selection_bias_report_combines_summary_rank_bins_and_warnings(self):
        rows = [
            {
                "key": f"k{index}",
                "affinity": float(index),
                "ligand_size": float(index + 1),
                "pocket_size": float(index + 10),
                "native_contact_mass": float(index + 20),
            }
            for index in range(4)
        ]
        result = builder_v3._selection_bias_report(
            requested_keys=[row["key"] for row in rows],
            included_rows=rows[:2],
            excluded_rows=rows[2:],
            eligible_keys={"k0", "k1", "k2"},
            complete_keys={
                "random_physical": {"k0", "k1"},
                "matched": {"k0"},
                "intersection": {"k0"},
            },
        )
        self.assertEqual(result["summary"]["included"]["sample_count"], 2)
        self.assertEqual(result["eligible_requested_count"], 3)
        self.assertEqual(set(result["rank_bins"]), {
            "ligand_size", "pocket_size", "native_contact_mass"
        })
        self.assertTrue(result["warnings"])

    def test_rank_bins_reject_nonfinite_properties(self):
        complete = {
            "random_physical": set(), "matched": set(), "intersection": set()
        }
        with self.assertRaisesRegex(ValueError, "finite.*ligand_size"):
            builder_v3._rank_bin_completion(
                ["a"], {"a": {"ligand_size": np.nan}}, {"a"}, complete
            )

    def test_physical_and_match_conservation_are_separate(self):
        builder_v3._validate_stage_conservation(
            stratum="high", proposed=8, physical_accepted=5,
            match_accepted=3, physical_rejections={"vdw_hard_clash": 3},
            match_rejections={"match_contact_mass_ratio": 2},
        )
        with self.assertRaisesRegex(ValueError, "physical.*conservation"):
            builder_v3._validate_stage_conservation(
                "high", 8, 6, 3, {"vdw_hard_clash": 3},
                {"match_contact_mass_ratio": 3},
            )
        with self.assertRaisesRegex(ValueError, "match.*conservation"):
            builder_v3._validate_stage_conservation(
                "high", 8, 5, 4, {"vdw_hard_clash": 3},
                {"match_contact_mass_ratio": 2},
            )


    def test_aggregate_generation_separates_v3_rejection_stages(self):
        per_stratum = {
            "base_rotation_families": 512,
            "proposed": 2048,
            "physical_accepted": 2000,
            "physical_family_count": 500,
            "match_accepted": 1800,
            "match_family_count": 450,
            "shortlisted_families": 20,
            "random_physical_retained": 4,
            "matched_retained": 4,
        }
        value = {
            "proposal_counts": {
                stratum: dict(per_stratum) for stratum in decoys_v3.STRATUM_ORDER
            },
            "physical_rejection_counts": {
                stratum: {"vdw_hard_clash": 48}
                for stratum in decoys_v3.STRATUM_ORDER
            },
            "match_rejection_counts": {
                stratum: {"match_contact_mass_ratio": 200}
                for stratum in decoys_v3.STRATUM_ORDER
            },
            "direction_source_counts": {
                stratum: {
                    "hard_clash_repulsion": {
                        "proposed_families": 512,
                        "proposed_transforms": 2048,
                        "families_with_any_physical_candidate": 500,
                        "families_with_any_matched_candidate": 450,
                        "physical_accepted_transforms": 2000,
                        "match_accepted_transforms": 1800,
                    }
                }
                for stratum in decoys_v3.STRATUM_ORDER
            },
            "family_funnel_counts": {
                stratum: {
                    "base_rotation_families": 512,
                    "zero_translation_vdw_safe": 100,
                    "any_grid_vdw_safe": 510,
                    "any_physical_target_candidate": 500,
                    "any_matched_candidate": 450,
                    "max_distance_still_hard_clash": 2,
                }
                for stratum in decoys_v3.STRATUM_ORDER
            },
            "zero_distance_pair_count_by_stratum": {
                stratum: 3 for stratum in decoys_v3.STRATUM_ORDER
            },
        }
        counts, physical, matched, directions, families, zero = (
            builder_v3._aggregate_generation([value], [])
        )
        self.assertEqual(counts["matched"]["low"]["shortlisted"], 20)
        self.assertEqual(physical["low"], {"vdw_hard_clash": 48})
        self.assertEqual(matched["low"], {"match_contact_mass_ratio": 200})
        self.assertEqual(
            directions["low"]["hard_clash_repulsion"]
            ["physical_accepted_transforms"],
            2000,
        )
        self.assertEqual(
            families["high"]["max_distance_still_hard_clash"], 2
        )
        self.assertEqual(zero["medium"], 3)



class PhysicsGuidedV3WorkflowTest(unittest.TestCase):
    def test_workflows_are_train_only_and_gate_train_build(self):
        root = ROOT / "experiments/lp_pdbbind/interface_first_v3"
        files = [
            root / "build_generator_smoke.sh",
            root / "build_diagnostic_cache.sh",
            root / "build_train_cache.sh",
        ]
        for path in files:
            self.assertTrue(path.is_file(), str(path))
            text = path.read_text(encoding="utf-8")
            self.assertIn("train_manifest.json", text)
            self.assertIn("LP_PDBBind_InterfaceFirstGerNA_v3", text)
            self.assertNotIn("valid_manifest", text)
            self.assertNotIn("test_manifest", text)
            self.assertNotIn("InterfaceFirstGerNA_v2", text)
        self.assertIn("--require-pass", files[1].read_text(encoding="utf-8"))
        self.assertIn(
            "verify_published_v3_diagnostic",
            files[2].read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "data_diagnostic_report:",
            files[0].read_text(encoding="utf-8"),
        )


    def test_builder_cli_accepts_generator_smoke(self):
        cohort = next(
            action
            for action in builder_v3.build_arg_parser()._actions
            if action.dest == "cohort"
        )
        self.assertIn("generator_smoke", cohort.choices)

    def test_generator_smoke_passes_builder_argument_validation(self):
        args = builder_v3.build_arg_parser().parse_args(
            [
                "--manifest", "train_manifest.json",
                "--native-lmdb", "native_train",
                "--diagnostic-keys", "diagnostic_keys.json",
                "--cohort", "generator_smoke",
                "--output-lmdb", "decoys_generator_smoke_v3",
            ]
        )
        builder_v3._validate_args(args)
