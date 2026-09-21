import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import local_interface_decoys_v3 as v3  # noqa: E402
from data_utils import local_interface_translation_probe_v4 as probe  # noqa: E402


def make_probe_record(key="complex_a"):
    return {
        "key": key,
        "ligand_coords": np.asarray(
            [[1.5, -0.5, 0.0], [2.5, -0.5, 0.0], [2.0, 0.8, 0.0]],
            dtype=np.float64,
        ),
        "protein_coords": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        ),
        "protein_element_symbol": np.asarray(["C", "C", "C"]),
        "ligand_atomic_numbers": np.asarray([6, 6, 6]),
        "ligand_automorphisms": np.asarray([[0, 1, 2]], dtype=np.int64),
        "ligand_heavy_atom_count": 3,
    }


class TranslationProbeV4DirectionTest(unittest.TestCase):
    def test_configuration_is_frozen(self):
        np.testing.assert_allclose(
            probe.HIGH_DISTANCE_GRID,
            tuple(index * 0.15 for index in range(1, 11)),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(probe.MAX_GUIDED_DIRECTIONS, 22)
        self.assertEqual(probe.FIBONACCI_DIRECTION_COUNT, 64)
        self.assertEqual(probe.BASE_ROTATIONS, 512)

    def test_rotation_replay_matches_v3(self):
        record = make_probe_record()
        rotations = list(probe.iter_v3_high_rotations(record, seed=42))
        v3_rows = v3.generate_proposal_pool(record, 42, "high")
        first_by_family = {}
        for row in v3_rows:
            first_by_family.setdefault(row["base_rotation_id"], row)
        self.assertEqual(len(rotations), 512)
        for rotation in rotations:
            expected = first_by_family[rotation["base_rotation_id"]]
            self.assertEqual(
                rotation["base_rotation_fingerprint"],
                expected["base_rotation_fingerprint"],
            )
            np.testing.assert_array_equal(
                rotation["rotation_matrix"],
                expected["transform"]["rotation_matrix"],
            )

    def test_fibonacci_directions_are_deterministic_units(self):
        first = probe.fibonacci_directions()
        second = probe.fibonacci_directions()
        self.assertEqual(first.shape, (64, 3))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            np.linalg.norm(first, axis=1), np.ones(64), atol=1e-12
        )

    def test_symmetric_clashes_keep_independent_escape_directions(self):
        rows = probe.build_guided_directions(
            rotated_ligand=np.asarray([[0.0, 0.0, 0.0]]),
            native_ligand=np.asarray([[0.0, 2.0, 0.0]]),
            protein=np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            protein_symbols=np.asarray(["C", "C"]),
            ligand_atomic_numbers=np.asarray([6]),
            axis=np.asarray([0.0, 0.0, 1.0]),
            seed=42,
            key="symmetric",
            base_rotation_id=7,
        )
        sources = {row["direction_source"] for row in rows}
        self.assertIn("severe_pair", sources)
        self.assertLessEqual(len(rows), 22)
        self.assertGreater(len(rows), 1)
        directions = np.asarray([row["direction"] for row in rows])
        self.assertTrue(np.all(np.isfinite(directions)))
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=1), np.ones(len(rows)), atol=1e-12
        )
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                self.assertFalse(
                    np.allclose(
                        directions[left],
                        directions[right],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                )

    def test_ligand_atom_cluster_combines_noncollinear_pair_forces(self):
        clusters = probe._cluster_directions(
            [
                {
                    "ligand_atom_index": 0,
                    "q": 0.10,
                    "direction": np.asarray([1.0, 0.0, 0.0]),
                },
                {
                    "ligand_atom_index": 0,
                    "q": 0.20,
                    "direction": np.asarray([0.0, 1.0, 0.0]),
                },
            ]
        )
        self.assertEqual(len(clusters), 1)
        direction = clusters[0]["direction"]
        self.assertGreater(direction[0], 0.0)
        self.assertGreater(direction[1], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)


def make_identity_rotation():
    return {
        "base_rotation_id": 0,
        "base_rotation_fingerprint": "a" * 64,
        "center": [0.0, 0.0, 0.0],
        "axis": [0.0, 0.0, 1.0],
        "rotation_degrees": 0.0,
        "rotation_matrix": np.eye(3).tolist(),
    }


def make_direction(direction, source="test_direction", stage="guided", index=0):
    return {
        "direction_family_id": index,
        "direction_source": source,
        "direction": list(direction),
        "search_stage": stage,
    }


class TranslationProbeV4FamilySearchTest(unittest.TestCase):
    def test_fast_vdw_screen_matches_full_v3_overlap(self):
        ligand = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        protein = np.asarray([[3.0, 0.0, 0.0]], dtype=np.float64)
        symbols = np.asarray(["C"])
        atomic_numbers = np.asarray([6])
        translations = np.asarray(
            [[0.0, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]],
            dtype=np.float64,
        )
        fast = probe.fast_vdw_safe_mask(
            ligand, protein, symbols, atomic_numbers, translations
        )
        exact = []
        for translation in translations:
            stats = v3.vdw_overlap_statistics(
                protein,
                symbols,
                ligand + translation,
                atomic_numbers,
            )
            exact.append(not stats["has_hard_clash"])
        np.testing.assert_array_equal(fast, np.asarray(exact))

    def test_batch_calls_full_evaluator_only_for_fast_safe_transforms(self):
        record = make_probe_record()
        proposals = probe.build_translation_proposals(
            make_identity_rotation(),
            [make_direction([1.0, 0.0, 0.0])],
        )
        calls = []

        def evaluator(native_geometry, proposal, target_stratum, seen):
            calls.append(proposal["transform_fingerprint"])
            return None, "outside_frozen_strata", None

        outcomes = probe.evaluate_proposal_batch(
            native_geometry={
                "native": record["ligand_coords"],
                "protein": record["protein_coords"],
                "protein_symbols": record["protein_element_symbol"],
                "ligand_atomic_numbers": record["ligand_atomic_numbers"],
            },
            rotated_ligand=record["ligand_coords"],
            proposals=proposals,
            seen_fingerprints=set(),
            full_evaluator=evaluator,
        )
        safe_count = sum(row["fast_vdw_safe"] for row in outcomes)
        self.assertEqual(len(calls), safe_count)
        self.assertEqual(len(outcomes), len(proposals))

    def test_fibonacci_runs_only_when_guided_has_no_match(self):
        record = make_probe_record()
        geometry = {
            "native": record["ligand_coords"],
            "protein": np.asarray([[100.0, 0.0, 0.0]]),
            "protein_symbols": np.asarray(["C"]),
            "ligand_atomic_numbers": record["ligand_atomic_numbers"],
        }
        guided = [make_direction([1.0, 0.0, 0.0])]
        fallback = [
            make_direction([0.0, 1.0, 0.0], "fibonacci", "fibonacci", 0)
        ]

        def matched_evaluator(native_geometry, proposal, target_stratum, seen):
            candidate = dict(proposal)
            candidate.update({"match_pass": True, "match_distance": 0.1})
            return candidate, None, None

        guided_match = probe.evaluate_rotation_family(
            geometry,
            make_identity_rotation(),
            guided,
            fallback,
            full_evaluator=matched_evaluator,
        )
        self.assertTrue(guided_match["guided"]["any_matched"])
        self.assertEqual(guided_match["counts"]["fibonacci_proposed"], 0)

        def fallback_evaluator(native_geometry, proposal, target_stratum, seen):
            candidate = dict(proposal)
            passed = proposal["search_stage"] == "fibonacci"
            candidate.update({"match_pass": passed, "match_distance": 0.1})
            return candidate, None, None if passed else "match_failure"

        fallback_match = probe.evaluate_rotation_family(
            geometry,
            make_identity_rotation(),
            guided,
            fallback,
            full_evaluator=fallback_evaluator,
        )
        self.assertFalse(fallback_match["guided"]["any_matched"])
        self.assertTrue(fallback_match["expanded"]["any_matched"])
        self.assertGreater(fallback_match["counts"]["fibonacci_proposed"], 0)

    def test_physical_representative_ignores_matching_values(self):
        candidates = [
            {
                "transform_fingerprint": "b",
                "match_pass": False,
                "match_distance": 9.0,
            },
            {
                "transform_fingerprint": "a",
                "match_pass": True,
                "match_distance": 0.1,
            },
        ]
        first = probe.select_physical_representative(candidates)
        changed = [
            dict(row, match_pass=not row["match_pass"], match_distance=-1.0)
            for row in candidates
        ]
        second = probe.select_physical_representative(changed)
        self.assertEqual(first["transform_fingerprint"], "a")
        self.assertEqual(second["transform_fingerprint"], "a")



def make_eligible_probe_record(key="eligible_complex"):
    record = make_probe_record(key)
    record["ligand_coords"] = np.asarray(
        [[3.0, -0.5, 0.0], [3.5, 0.5, 0.0], [3.0, 1.5, 0.0]],
        dtype=np.float64,
    )
    return record


class TranslationProbeV4RecordTest(unittest.TestCase):
    def test_debug_record_has_requested_rotation_count_and_is_not_decision_eligible(self):
        value = probe.probe_record(
            make_eligible_probe_record(), seed=42, rotation_limit=2
        )
        self.assertEqual(value["schema"], probe.PROBE_SCHEMA)
        self.assertEqual(value["rotation_count"], 2)
        self.assertFalse(value["decision_eligible"])
        self.assertEqual(len(value["families"]), 2)
        probe.validate_probe_record(
            value,
            expected_key="eligible_complex",
            expected_rotation_count=2,
        )

    def test_decision_eligible_record_requires_all_512_rotations(self):
        value = {
            "schema": probe.PROBE_SCHEMA,
            "key": "k",
            "seed": 42,
            "target_stratum": "high",
            "rotation_count": 1,
            "decision_eligible": True,
            "sample_properties": {
                "ligand_size": 1,
                "pocket_size": 1,
                "native_contact_mass": 8.0,
            },
            "families": [],
            "family_counts": {},
            "rejection_counts": {},
            "direction_source_counts": {},
        }
        with self.assertRaisesRegex(ValueError, "512"):
            probe.validate_probe_record(value, expected_key="k")



def make_report_record(index, matched_complete=False, vdw_safe_families=512):
    key = "k%02d" % index
    matched_families = 4 if matched_complete else 0
    return {
        "schema": probe.PROBE_SCHEMA,
        "key": key,
        "seed": 42,
        "target_stratum": "high",
        "rotation_count": 512,
        "decision_eligible": True,
        "sample_properties": {
            "ligand_size": index + 1,
            "pocket_size": 100 + index,
            "native_contact_mass": 200.0 + index,
        },
        "families": [],
        "family_counts": {
            "guided_any_vdw_safe": 0,
            "guided_any_physical": 0,
            "guided_any_matched": 0,
            "expanded_any_vdw_safe": int(vdw_safe_families),
            "expanded_any_physical": int(vdw_safe_families),
            "expanded_any_matched": matched_families,
            "fibonacci_added_physical": int(vdw_safe_families),
            "fibonacci_added_matched": matched_families,
        },
        "rejection_counts": {},
        "direction_source_counts": {},
    }


def make_report_fixture(completed_indices, vdw_safe_families=512):
    requested = ["excluded%02d" % index for index in range(8)]
    requested += ["k%02d" % index for index in range(56)]
    exclusions = [
        {"key": key, "exclusion_reason": "native_hard_clash"}
        for key in requested[:8]
    ]
    records = [
        make_report_record(
            index,
            matched_complete=index in set(completed_indices),
            vdw_safe_families=vdw_safe_families,
        )
        for index in range(56)
    ]
    return requested, exclusions, records


class TranslationProbeV4ReportTest(unittest.TestCase):
    def test_report_keeps_requested_and_eligible_denominators(self):
        from scripts import probe_local_interface_translation_v4 as runner

        requested, exclusions, records = make_report_fixture(range(20))
        report = runner.aggregate_probe_report(
            requested,
            exclusions,
            records,
            decision_eligible=True,
        )
        self.assertEqual(report["requested_count"], 64)
        self.assertEqual(report["eligible_count"], 56)
        self.assertEqual(len(report["eligible_keys"]), 56)

    def test_rank_bins_cover_every_eligible_key_once(self):
        from scripts import probe_local_interface_translation_v4 as runner

        requested, exclusions, records = make_report_fixture(range(20))
        report = runner.aggregate_probe_report(
            requested,
            exclusions,
            records,
            decision_eligible=True,
        )
        for metric in (
            "ligand_size",
            "pocket_size",
            "native_contact_mass",
        ):
            bins = report["completion_rank_bins"][metric]["bins"]
            keys = [key for row in bins for key in row["keys"]]
            self.assertEqual(sorted(keys), sorted(report["eligible_keys"]))
            self.assertEqual([len(row["keys"]) for row in bins], [14] * 4)

    def test_twenty_balanced_completions_pass_probe_decision(self):
        from scripts import probe_local_interface_translation_v4 as runner

        completed = [0, 1, 2, 3, 4, 14, 15, 16, 17, 18]
        completed += [28, 29, 30, 31, 32, 42, 43, 44, 45, 46]
        requested, exclusions, records = make_report_fixture(completed)
        report = runner.aggregate_probe_report(
            requested,
            exclusions,
            records,
            decision_eligible=True,
        )
        decision = runner.evaluate_probe_decision(report)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["outcome"], "translation_search_go")
        self.assertEqual(
            decision["metrics"]["matched_complete_count"], 20
        )

    def test_over_ninety_percent_without_vdw_safe_is_translation_no_go(self):
        from scripts import probe_local_interface_translation_v4 as runner

        requested, exclusions, records = make_report_fixture(
            [], vdw_safe_families=51
        )
        report = runner.aggregate_probe_report(
            requested,
            exclusions,
            records,
            decision_eligible=True,
        )
        decision = runner.evaluate_probe_decision(report)
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["outcome"], "translation_search_no_go_vdw"
        )
        self.assertIn(
            "expanded_no_vdw_safe_fraction_above_0_90",
            decision["reasons"],
        )



def make_v3_smoke_report(status="complete"):
    requested, exclusions, records = make_report_fixture([])
    return {
        "status": status,
        "builder": "local_interface_decoy_cache_v3",
        "schema_version": 3,
        "cohort": "generator_smoke",
        "seed": 42,
        "requested_keys": requested,
        "excluded_samples": exclusions,
        "included_keys": [row["key"] for row in records[:15]],
        "cohort_selection": {"requested_count": 64},
        "provenance": {
            "generator_configuration": {
                "schema_version": 3,
                "base_rotations_per_stratum": 512,
            }
        },
    }


class TranslationProbeV4RunnerTest(unittest.TestCase):
    def test_runner_requires_complete_v3_smoke_report(self):
        from scripts import probe_local_interface_translation_v4 as runner

        with self.assertRaisesRegex(ValueError, "complete V3 smoke"):
            runner.validate_v3_smoke_report(
                make_v3_smoke_report(status="in_progress")
            )

    def test_runner_uses_exact_requested_and_native_exclusion_keys(self):
        from scripts import probe_local_interface_translation_v4 as runner

        report = make_v3_smoke_report()
        cohort = runner.cohort_from_v3_smoke_report(report)
        self.assertEqual(cohort["requested_keys"], report["requested_keys"])
        self.assertEqual(
            cohort["native_exclusion_keys"],
            [row["key"] for row in report["excluded_samples"]],
        )
        self.assertEqual(len(cohort["eligible_keys"]), 56)

    def test_resume_rejects_changed_provenance(self):
        import json
        import tempfile

        from scripts import probe_local_interface_translation_v4 as runner

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / runner.REPORT_FILENAME).write_text(
                json.dumps({"provenance": {"seed": 42}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.prepare_output(
                    output,
                    resume=True,
                    provenance={"seed": 43},
                )

    def test_debug_limits_disable_probe_decision(self):
        from scripts import probe_local_interface_translation_v4 as runner

        self.assertFalse(
            runner.is_decision_eligible(
                max_eligible_samples=1,
                rotation_limit=2,
                eligible_count=56,
            )
        )
        self.assertTrue(
            runner.is_decision_eligible(
                max_eligible_samples=None,
                rotation_limit=None,
                eligible_count=56,
            )
        )



class TranslationProbeV4WorkflowTest(unittest.TestCase):
    def test_workflow_is_train_only_v4_and_supports_safe_resume(self):
        workflow = (
            ROOT
            / "experiments"
            / "lp_pdbbind"
            / "interface_first_v4"
            / "run_translation_probe.sh"
        )
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("scripts/probe_local_interface_translation_v4.py", text)
        self.assertIn("train_manifest.json", text)
        self.assertIn("LP_PDBBind_InterfaceFirstGerNA_v1", text)
        self.assertIn("decoys_generator_smoke_v3/decoy_report.json", text)
        self.assertIn("LP_PDBBind_InterfaceFirstGerNA_v4", text)
        self.assertIn('set -- "$@" --resume', text)
        self.assertIn("MAX_ELIGIBLE_SAMPLES", text)
        self.assertIn("ROTATION_LIMIT", text)
        self.assertNotIn("val_manifest", text)
        self.assertNotIn("test_manifest", text)
        self.assertNotIn("build_diagnostic_cache.sh", text)
        self.assertNotIn("train_model.py", text)



class TranslationProbeV4ReportCounterTest(unittest.TestCase):
    def test_report_aggregates_rejection_and_direction_source_counts(self):
        from scripts import probe_local_interface_translation_v4 as runner

        requested, exclusions, records = make_report_fixture([])
        for record in records:
            record["rejection_counts"] = {"vdw_hard_clash": 2}
            record["direction_source_counts"] = {"severe_pair": 3}
        report = runner.aggregate_probe_report(
            requested,
            exclusions,
            records,
            decision_eligible=True,
        )
        self.assertEqual(
            report["rejection_counts"], {"vdw_hard_clash": 112}
        )
        self.assertEqual(
            report["direction_source_counts"], {"severe_pair": 168}
        )


if __name__ == "__main__":
    unittest.main()
