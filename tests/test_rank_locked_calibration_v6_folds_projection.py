import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dta.rank_locked_calibration_v6.folds import (
    FoldRow,
    assign_grouped_folds,
    load_fold_rows,
    murcko_scaffold,
    normalize_target,
)
from dta.rank_locked_calibration_v6.projection import (
    rank_locked_project,
    weighted_pava,
)
from dta.rank_fusion_v4.metrics import affinity_metrics


class RankLockedV6FoldTest(unittest.TestCase):
    def test_connected_target_or_scaffold_stays_in_one_fold(self):
        rows = (
            FoldRow("a_protein", "AAAA", "scaf1"),
            FoldRow("b_protein", "AAAA", "scaf2"),
            FoldRow("c_protein", "CCCC", "scaf1"),
            FoldRow("d_protein", "DDDD", "scaf3"),
        )
        result = assign_grouped_folds(rows, fold_count=2, seed=42)
        self.assertEqual(result.by_key["a_protein"], result.by_key["b_protein"])
        self.assertEqual(result.by_key["a_protein"], result.by_key["c_protein"])
        self.assertNotEqual(result.by_key["a_protein"], result.by_key["d_protein"])
        self.assertEqual(sum(result.fold_sizes), 4)
        self.assertEqual(len(result.sha256), 64)
        self.assertFalse(result.fold_ids.flags.writeable)

    def test_assignment_is_input_order_invariant_and_balanced(self):
        rows = tuple(
            FoldRow(
                "%s_protein" % name,
                "TARGET_" + name,
                "SCAFFOLD_" + name,
            )
            for name in ("a", "b", "c", "d", "e", "f", "g")
        )
        forward = assign_grouped_folds(rows, fold_count=3, seed=87)
        reverse = assign_grouped_folds(tuple(reversed(rows)), fold_count=3, seed=87)
        self.assertEqual(dict(forward.by_key), dict(reverse.by_key))
        self.assertEqual(forward.sha256, reverse.sha256)
        self.assertEqual(forward.ordered_keys, reverse.ordered_keys)
        self.assertLessEqual(max(forward.fold_sizes) - min(forward.fold_sizes), 1)

    def test_empty_acyclic_scaffolds_do_not_connect_unrelated_rows(self):
        rows = (
            FoldRow("a_protein", "AAAA", ""),
            FoldRow("b_protein", "BBBB", ""),
            FoldRow("c_protein", "CCCC", ""),
        )
        result = assign_grouped_folds(rows, fold_count=3, seed=42)
        self.assertEqual(len(set(result.by_key.values())), 3)
        self.assertEqual(result.fold_sizes, (1, 1, 1))

    def test_normalization_and_murcko_scaffold(self):
        self.assertEqual(normalize_target(" aa\n bb : cc "), "AABB:CC")
        self.assertEqual(murcko_scaffold("c1ccccc1CCO"), "c1ccccc1")
        self.assertEqual(murcko_scaffold("CCO"), "")
        with self.assertRaisesRegex(ValueError, "target sequence is empty"):
            normalize_target(" \n ")
        with self.assertRaisesRegex(ValueError, "ligand SMILES is invalid"):
            murcko_scaffold("not-a-smiles")

    def test_csv_loader_maps_real_columns_and_cache_order(self):
        rows = (
            {
                "Unnamed: 0": "a",
                "smiles": "c1ccccc1CCO",
                "seq": "aa aa",
                "new_split": "train",
            },
            {
                "Unnamed: 0": "b",
                "smiles": "CCO",
                "seq": "bb:cc",
                "new_split": "train",
            },
        )
        with self._csv(rows) as path:
            loaded = load_fold_rows(
                path, ("b_protein", "a_protein"), expected_split="train"
            )
        self.assertEqual(
            tuple(row.key for row in loaded), ("b_protein", "a_protein")
        )
        self.assertEqual(loaded[0].target, "BB:CC")
        self.assertEqual(loaded[0].scaffold, "")
        self.assertEqual(loaded[1].scaffold, "c1ccccc1")

    def test_csv_loader_rejects_duplicate_missing_invalid_and_wrong_split(self):
        base = {
            "Unnamed: 0": "a",
            "smiles": "CCO",
            "seq": "AAAA",
            "new_split": "train",
        }
        cases = (
            ((base, dict(base)), ("a_protein",), "duplicate"),
            ((base,), ("a_protein", "b_protein"), "missing cache keys"),
            ((dict(base, seq=" "),), ("a_protein",), "target sequence"),
            ((dict(base, new_split="valid"),), ("a_protein",), "split"),
        )
        for rows, keys, message in cases:
            with self.subTest(message=message):
                with self._csv(rows) as path:
                    with self.assertRaisesRegex(ValueError, message):
                        load_fold_rows(path, keys, expected_split="train")

    def test_csv_loader_has_explicit_invalid_smiles_fallback_policy(self):
        invalid = "[Cl+]1C2C13[Cl+]C3C21C23Cl4Cl2C413"
        rows = (
            {
                "Unnamed: 0": "a",
                "smiles": invalid,
                "seq": "AAAA",
                "new_split": "train",
            },
            {
                "Unnamed: 0": "b",
                "smiles": invalid,
                "seq": "BBBB",
                "new_split": "train",
            },
        )
        with self._csv(rows) as path:
            loaded = load_fold_rows(
                path, ("a_protein", "b_protein"), expected_split="train"
            )
            with self.assertRaisesRegex(ValueError, "SMILES"):
                load_fold_rows(
                    path,
                    ("a_protein", "b_protein"),
                    expected_split="train",
                    allow_invalid_scaffold_fallback=False,
                )
        self.assertTrue(
            loaded[0].scaffold.startswith("__invalid_smiles__:")
        )
        self.assertEqual(loaded[0].scaffold, loaded[1].scaffold)
        assignment = assign_grouped_folds(
            loaded + (FoldRow("c_protein", "CCCC", "valid"),),
            fold_count=2,
            seed=42,
        )
        self.assertEqual(
            assignment.by_key["a_protein"],
            assignment.by_key["b_protein"],
        )

    def test_missing_smiles_fallback_is_key_specific(self):
        rows = (
            {
                "Unnamed: 0": "a",
                "smiles": "",
                "seq": "AAAA",
                "new_split": "train",
            },
            {
                "Unnamed: 0": "b",
                "smiles": "",
                "seq": "BBBB",
                "new_split": "train",
            },
        )
        with self._csv(rows) as path:
            loaded = load_fold_rows(
                path, ("a_protein", "b_protein"), expected_split="train"
            )
            with self.assertRaisesRegex(ValueError, "SMILES"):
                load_fold_rows(
                    path,
                    ("a_protein", "b_protein"),
                    expected_split="train",
                    allow_invalid_scaffold_fallback=False,
                )
        self.assertTrue(loaded[0].scaffold.startswith("__missing_smiles__:"))
        self.assertTrue(loaded[1].scaffold.startswith("__missing_smiles__:"))
        self.assertNotEqual(loaded[0].scaffold, loaded[1].scaffold)
        assignment = assign_grouped_folds(
            loaded, fold_count=2, seed=42
        )
        self.assertNotEqual(
            assignment.by_key["a_protein"],
            assignment.by_key["b_protein"],
        )

    def test_assignment_rejects_duplicates_and_too_few_components(self):
        duplicate = (
            FoldRow("a_protein", "AAAA", "x"),
            FoldRow("a_protein", "BBBB", "y"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assign_grouped_folds(duplicate, fold_count=2, seed=42)
        connected = (
            FoldRow("a_protein", "AAAA", "x"),
            FoldRow("b_protein", "AAAA", "y"),
            FoldRow("c_protein", "CCCC", "z"),
        )
        with self.assertRaisesRegex(ValueError, "components"):
            assign_grouped_folds(connected, fold_count=3, seed=42)
        with self.assertRaisesRegex(ValueError, "fold_count"):
            assign_grouped_folds(connected, fold_count=1, seed=42)

    class _CsvContext:
        def __init__(self, rows):
            self.rows = tuple(rows)
            self.temporary = None

        def __enter__(self):
            self.temporary = tempfile.TemporaryDirectory()
            path = Path(self.temporary.name) / "metadata.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("Unnamed: 0", "smiles", "seq", "new_split"),
                )
                writer.writeheader()
                writer.writerows(self.rows)
            return path

        def __exit__(self, exception_type, exception, traceback):
            self.temporary.cleanup()

    def _csv(self, rows):
        return self._CsvContext(rows)


class RankLockedV6ProjectionTest(unittest.TestCase):
    def test_projection_removes_inversions_and_preserves_base_ties(self):
        base = np.asarray([2.0, 1.0, 3.0, 2.0], dtype=np.float64)
        proposed = np.asarray([0.0, 4.0, -2.0, 8.0], dtype=np.float64)
        final, report = rank_locked_project(base, proposed, epsilon=1e-8)
        self.assertEqual(final.dtype, np.float64)
        self.assertEqual(final[0], final[3])
        self.assertLess(final[1], final[0])
        self.assertLess(final[0], final[2])
        self.assertEqual(report.inversion_count, 0)
        self.assertEqual(report.group_count, 3)
        self.assertEqual(report.tie_group_count, 1)
        self.assertGreaterEqual(report.minimum_distinct_gap, 1e-8)
        self.assertEqual(report.base_order_sha256, report.final_order_sha256)

    def test_projection_preserves_rank_metrics_exactly(self):
        labels = np.asarray([8.0, 4.0, 6.0, 7.0, 9.0], dtype=np.float64)
        base = np.asarray([3.0, 1.0, 2.0, 2.0, 4.0], dtype=np.float64)
        proposed = np.asarray([0.0, 9.0, -4.0, 8.0, 1.0], dtype=np.float64)
        final, _ = rank_locked_project(base, proposed)
        before = affinity_metrics(labels, base)
        after = affinity_metrics(labels, final)
        self.assertEqual(before["CI"], after["CI"])
        self.assertEqual(before["Spearman"], after["Spearman"])

    def test_weighted_pava_uses_block_mass(self):
        result = weighted_pava(
            np.asarray([3.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([1.0, 3.0, 1.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            result, np.asarray([1.5, 1.5, 2.0], dtype=np.float64)
        )

    def test_projection_handles_all_equal_base_and_is_deterministic(self):
        base = np.ones(5, dtype=np.float64)
        proposed = np.asarray([5.0, 1.0, 3.0, 9.0, 2.0], dtype=np.float64)
        first, first_report = rank_locked_project(base, proposed)
        second, second_report = rank_locked_project(base, proposed)
        np.testing.assert_array_equal(first, np.full(5, proposed.mean()))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_report, second_report)
        self.assertIsNone(first_report.minimum_distinct_gap)
        self.assertEqual(first_report.tie_group_count, 1)

    def test_projection_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            rank_locked_project(np.ones(2), np.ones(3))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            rank_locked_project(np.asarray([]), np.asarray([]))
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_locked_project(np.asarray([1.0, np.nan]), np.ones(2))
        with self.assertRaisesRegex(ValueError, "epsilon"):
            rank_locked_project(np.ones(2), np.ones(2), epsilon=0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_pava(np.ones(2), np.asarray([1.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "aligned"):
            weighted_pava(np.ones(2), np.ones(3))

    def test_float64_npz_round_trip_preserves_projection(self):
        base = np.asarray([1.0, 2.0, 2.0, 3.0], dtype=np.float64)
        proposed = np.asarray([4.0, -2.0, 8.0, 1.0], dtype=np.float64)
        final, report = rank_locked_project(base, proposed)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "projection.npz"
            np.savez_compressed(path, predictions=final)
            with np.load(path, allow_pickle=False) as archive:
                restored = archive["predictions"]
        self.assertEqual(restored.dtype, np.float64)
        np.testing.assert_array_equal(restored, final)
        repeated, repeated_report = rank_locked_project(base, proposed)
        np.testing.assert_array_equal(repeated, restored)
        self.assertEqual(report, repeated_report)


if __name__ == "__main__":
    unittest.main()
