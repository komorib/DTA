import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import local_interface_decoys_v3 as v3  # noqa: E402
from data_utils import local_interface_rotation_probe_v4b as probe  # noqa: E402


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


class RotationProbeV4bGeometryTest(unittest.TestCase):
    def test_configuration_is_frozen(self):
        self.assertEqual(probe.PROBE_SCHEMA, "rotation_feasibility_probe_v4b")
        self.assertEqual(probe.TARGET_STRATUM, "high")
        self.assertEqual(probe.BASE_ROTATIONS, 512)
        self.assertEqual(probe.MAX_CENTROID_MOTION, 1.5)
        np.testing.assert_allclose(
            probe.SLIDE_GRID,
            tuple(float(index) * 0.15 for index in range(11)),
            rtol=0.0,
            atol=1e-15,
        )

    def test_raw_rotation_replay_matches_v3(self):
        record = make_probe_record()
        rotations = list(probe.iter_v3_high_rotation_seeds(record, seed=42))
        v3_rows = v3.generate_proposal_pool(record, 42, "high")
        first_by_family = {}
        for row in v3_rows:
            first_by_family.setdefault(row["base_rotation_id"], row)

        self.assertEqual(len(rotations), 512)
        for rotation in rotations:
            expected = first_by_family[rotation["base_rotation_id"]]
            self.assertEqual(
                rotation["v3_base_rotation_fingerprint"],
                expected["base_rotation_fingerprint"],
            )
            np.testing.assert_allclose(
                rotation["raw_rotation_matrix"],
                expected["transform"]["rotation_matrix"],
                rtol=0.0,
                atol=1e-12,
            )

    def test_tangent_axis_is_unit_and_perpendicular(self):
        outward = np.asarray([0.0, 0.0, 1.0])
        tangent = probe.tangent_axis(
            np.asarray([1.0, 2.0, 3.0]),
            outward,
            namespace=(42, "complex_a", 0),
        )
        self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0)
        self.assertAlmostEqual(float(np.dot(tangent, outward)), 0.0)

    def test_parallel_raw_axis_uses_deterministic_tangent_fallback(self):
        outward = np.asarray([0.0, 0.0, 1.0])
        first = probe.tangent_axis(
            outward,
            outward,
            namespace=(42, "complex_a", 0),
        )
        second = probe.tangent_axis(
            outward,
            outward,
            namespace=(42, "complex_a", 0),
        )
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0)
        self.assertAlmostEqual(float(np.dot(first, outward)), 0.0)

    def test_pivots_are_ordered_and_duplicates_are_removed(self):
        rows = probe.build_pivot_rows(
            native_ligand=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            protein=np.asarray([[0.0, 0.0, -1.0]]),
            outward=np.asarray([1.0, 0.0, 0.0]),
        )
        self.assertEqual(
            [row["pivot_id"] for row in rows],
            ["ligand_centroid", "exposed_atom"],
        )
        self.assertEqual(rows[-1]["pivot_atom_index"], 1)


def make_family_seed():
    return {
        "base_rotation_id": 0,
        "center": [0.0, 0.0, 0.0],
        "raw_axis": [0.0, 1.0, 0.0],
        "raw_rotation_degrees": 180.0,
        "raw_rotation_matrix": np.diag([-1.0, 1.0, -1.0]).tolist(),
        "v3_base_rotation_fingerprint": "a" * 64,
    }


def make_family_geometry():
    return {
        "native": np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64
        ),
        "protein": np.asarray([[100.0, 0.0, 0.0]], dtype=np.float64),
        "protein_symbols": np.asarray(["C"]),
        "ligand_atomic_numbers": np.asarray([6, 6]),
        "automorphisms": np.asarray([[0, 1]], dtype=np.int64),
    }


class RotationProbeV4bFamilyTest(unittest.TestCase):
    def test_coupled_proposals_reject_centroid_motion_before_vdw(self):
        variant = {
            **make_family_seed(),
            "rotation_variant_id": 0,
            "rotation_source": "tangent_axis",
            "tangent_axis": [0.0, 1.0, 0.0],
            "pivot_id": "exposed_atom",
            "pivot": [0.0, 0.0, 0.0],
            "pivot_atom_index": 0,
            "rotation_matrix": np.diag([-1.0, 1.0, -1.0]).tolist(),
        }
        rows = probe.build_coupled_proposals(
            native_ligand=make_family_geometry()["native"],
            outward=np.asarray([0.0, 0.0, 1.0]),
            variant=variant,
        )
        self.assertEqual(len(rows), len(probe.SLIDE_GRID))
        self.assertTrue(any(not row["motion_limit_pass"] for row in rows))
        self.assertEqual(
            [row["slide_index"] for row in rows],
            list(range(len(probe.SLIDE_GRID))),
        )

    def test_family_selects_one_match_and_calls_evaluator_only_for_vdw_safe(self):
        calls = []

        def evaluator(native_geometry, proposal, target_stratum, seen):
            calls.append(proposal["transform_fingerprint"])
            if (
                proposal["rotation_variant_id"] == 0
                and proposal["slide_index"] < 2
            ):
                candidate = dict(proposal)
                candidate.update({"match_pass": True, "match_distance": 0.2})
                return candidate, None, None
            return None, "outside_frozen_strata", None

        family = probe.evaluate_rotation_family(
            native_geometry=make_family_geometry(),
            family_seed=make_family_seed(),
            full_evaluator=evaluator,
        )
        self.assertEqual(family["counts"]["matched"], 2)
        self.assertEqual(family["best_matched"]["base_rotation_id"], 0)
        self.assertEqual(len(calls), family["counts"]["vdw_safe"])

    def test_physical_representative_ignores_matching_values(self):
        candidates = [
            {
                "rotation_variant_id": 1,
                "slide_index": 0,
                "transform_fingerprint": "b" * 64,
                "match_distance": 0.0,
            },
            {
                "rotation_variant_id": 0,
                "slide_index": 10,
                "transform_fingerprint": "a" * 64,
                "match_distance": 999.0,
            },
        ]
        selected = probe._select_physical_representative(candidates)
        self.assertEqual(selected["transform_fingerprint"], "a" * 64)


def make_eligible_probe_record(key="eligible_a"):
    record = make_probe_record(key)
    record["ligand_coords"] = np.asarray(
        [[4.0, -0.5, 0.0], [5.0, -0.5, 0.0], [4.5, 0.8, 0.0]],
        dtype=np.float64,
    )
    return record


def make_balanced_report():
    bins = [
        {"matched_complete_count": 1},
        {"matched_complete_count": 1},
        {"matched_complete_count": 9},
        {"matched_complete_count": 9},
    ]
    return {
        "eligible_count": 56,
        "total_rotation_families": 28672,
        "family_totals": {"any_vdw_safe": 2868},
        "sample_completion": {"matched_complete_count": 20},
        "completion_rank_bins": {
            "ligand_size": {"bins": bins, "upper_half_matched_complete_count": 18},
            "pocket_size": {"bins": bins, "upper_half_matched_complete_count": 18},
            "native_contact_mass": {"bins": bins, "upper_half_matched_complete_count": 18},
        },
    }


class RotationProbeV4bRecordReportTest(unittest.TestCase):
    def test_limited_record_cannot_be_decision_eligible(self):
        record = probe.probe_record(
            make_eligible_probe_record(),
            seed=42,
            rotation_limit=2,
        )
        self.assertEqual(record["rotation_count"], 2)
        self.assertFalse(record["decision_eligible"])
        invalid = dict(record)
        invalid["decision_eligible"] = True
        with self.assertRaisesRegex(ValueError, "512 rotations"):
            probe.validate_probe_record(
                invalid,
                expected_key="eligible_a",
                expected_rotation_count=2,
            )

    def test_aggregate_reports_per_pivot_funnels(self):
        record = probe.probe_record(
            make_eligible_probe_record(),
            seed=42,
            rotation_limit=2,
        )
        report = probe.aggregate_probe_report(
            records=[record],
            requested_keys=["eligible_a"],
            native_exclusions=[],
            decision_eligible=False,
        )
        self.assertEqual(report["eligible_count"], 1)
        self.assertTrue(report["pivot_totals"])
        for values in report["pivot_totals"].values():
            self.assertIn("proposed", values)
            self.assertIn("rejection_counts", values)

    def test_decision_requires_balanced_matches_and_vdw_feasibility(self):
        report = make_balanced_report()
        decision = probe.evaluate_probe_decision(report)
        self.assertTrue(decision["passed"])
        report["sample_completion"] = {"matched_complete_count": 19}
        self.assertFalse(probe.evaluate_probe_decision(report)["passed"])
        report = make_balanced_report()
        report["family_totals"] = {"any_vdw_safe": 2000}
        decision = probe.evaluate_probe_decision(report)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["outcome"], "rotation_probe_no_go_vdw")


class RotationProbeV4bRunnerWorkflowTest(unittest.TestCase):
    def test_resume_rejects_changed_v4_report_provenance(self):
        import json
        import tempfile

        from scripts import probe_local_interface_rotation_v4b as runner

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / runner.REPORT_FILENAME).write_text(
                json.dumps({"provenance": {"v4_report_sha256": "a" * 64}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.prepare_output(
                    output,
                    resume=True,
                    provenance={"v4_report_sha256": "b" * 64},
                )

    def test_v4_report_validation_rejects_missing_provenance(self):
        from scripts import probe_local_interface_rotation_v4b as runner

        with self.assertRaisesRegex(ValueError, "V4 report"):
            runner.validate_v4_report(
                {
                    "status": "complete",
                    "decision_eligible": True,
                    "requested_count": 64,
                    "eligible_count": 56,
                    "decision": {},
                }
            )

    def test_debug_limits_disable_decision_and_workflow_is_train_only(self):
        from scripts import probe_local_interface_rotation_v4b as runner

        self.assertFalse(
            runner.is_decision_eligible(
                max_eligible_samples=1,
                rotation_limit=2,
                eligible_count=56,
            )
        )
        workflow = (
            ROOT
            / "experiments"
            / "lp_pdbbind"
            / "interface_first_v4b"
            / "run_rotation_probe.sh"
        )
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("scripts/probe_local_interface_rotation_v4b.py", text)
        self.assertIn("translation_probe_high_seed", text)
        self.assertIn("MAX_ELIGIBLE_SAMPLES", text)
        self.assertIn("ROTATION_LIMIT", text)
        self.assertNotIn("train_model.py", text)


if __name__ == "__main__":
    unittest.main()
