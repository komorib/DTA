import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _write_manifest(path, split, samples):
    path.write_text(
        json.dumps({"split": split, "samples": samples}),
        encoding="utf-8",
    )


def _graph(value):
    return (
        np.asarray([[value, value + 1]], dtype=np.float32),
        np.asarray([[0, 0]], dtype=np.int64),
        np.asarray([[value]], dtype=np.float32),
    )


class TestRankFusionV4ThreeDProtPrepare(unittest.TestCase):
    def _fixture(self, root):
        raw_path = root / "lp.csv"
        pd.DataFrame(
            [
                {
                    "Unnamed: 0": "3ccc",
                    "smiles": "CCC",
                    "value": 8.3,
                    "new_split": "test",
                    "date": "2014-01-01",
                },
                {
                    "Unnamed: 0": "1AAA",
                    "smiles": "CC",
                    "value": 6.1,
                    "new_split": "train",
                    "date": "1999-01-01",
                },
                {
                    "Unnamed: 0": "2BbB",
                    "smiles": "CO",
                    "value": 7.2,
                    "new_split": "val",
                    "date": "2008-01-01",
                },
            ]
        ).to_csv(raw_path, index=False)
        manifests = {}
        for split, key, label in (
            ("train", "1aaa_protein", 6.1),
            ("valid", "2bbb_protein", 7.2),
            ("test", "3ccc_protein", 8.3),
        ):
            path = root / (split + "_manifest.json")
            _write_manifest(
                path,
                split,
                [{"lmdb_key": key, "pdb_id": key[:4], "label": label}],
            )
            manifests[split] = path
        return raw_path, manifests

    def test_official_rows_follow_manifest_order_and_validate_labels(self):
        from dta.rank_fusion_v4.three_dprot_prepare import load_official_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path, manifests = self._fixture(root)

            rows, provenance = load_official_rows(raw_path, manifests)

            self.assertEqual([row["pdb_id"] for row in rows], ["1aaa", "2bbb", "3ccc"])
            self.assertEqual([row["split"] for row in rows], ["train", "valid", "test"])
            self.assertEqual([row["label"] for row in rows], [6.1, 7.2, 8.3])
            self.assertEqual(provenance["counts"], {"train": 1, "valid": 1, "test": 1})

            payload = json.loads(manifests["valid"].read_text(encoding="utf-8"))
            payload["samples"][0]["label"] = 7.25
            manifests["valid"].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label mismatch"):
                load_official_rows(raw_path, manifests)

    def test_missing_raw_smiles_is_preserved_for_structure_fallback(self):
        from dta.rank_fusion_v4.three_dprot_prepare import load_official_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path, manifests = self._fixture(root)
            raw = pd.read_csv(raw_path)
            raw.loc[raw["Unnamed: 0"] == "1AAA", "smiles"] = np.nan
            raw.to_csv(raw_path, index=False)

            rows, provenance = load_official_rows(raw_path, manifests)

            self.assertEqual(len(rows), 3)
            self.assertIsNone(rows[0]["smiles"])
            self.assertEqual(provenance["missing_raw_smiles"], 1)

    def test_missing_raw_smiles_is_recovered_from_ligand_sdf(self):
        from rdkit import Chem

        from dta.rank_fusion_v4.three_dprot_prepare import resolve_ligand_smiles

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            protein_path = root / "1aaa_protein.pdb"
            protein_path.write_text("", encoding="utf-8")
            Chem.MolToMolFile(Chem.MolFromSmiles("CCO"), str(root / "1aaa_ligand.sdf"))

            smiles, source = resolve_ligand_smiles(
                {"pdb_id": "1aaa", "smiles": None}, protein_path
            )

            self.assertEqual(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), "CCO")
            self.assertEqual(source, "ligand_sdf")

    def test_complete_shard_is_reused_without_rebuilding(self):
        from dta.rank_fusion_v4.three_dprot_prepare import build_or_reuse_shard

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "shard_00000.pkl"
            rows = [
                {"pdb_id": "1aaa", "smiles": "CC", "label": 6.1, "split": "train", "date": "1999-01-01"},
                {"pdb_id": "2bbb", "smiles": "CO", "label": 7.2, "split": "valid", "date": "2008-01-01"},
            ]
            calls = []

            def build_record(row):
                calls.append(row["pdb_id"])
                return {
                    "pdb_id": row["pdb_id"],
                    "ligand": row["smiles"],
                    "label": row["label"],
                    "split": row["split"],
                    "ligand_graph": _graph(1.0),
                    "ligand_ecfp": np.ones(8, dtype=np.float32),
                    "protein_graph": _graph(2.0),
                }

            first = build_or_reuse_shard(
                shard_path, rows, signature="signature-a", build_record=build_record
            )
            second = build_or_reuse_shard(
                shard_path, rows, signature="signature-a", build_record=build_record
            )

            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(calls, ["1aaa", "2bbb"])
            self.assertFalse((Path(str(shard_path) + ".tmp")).exists())

            build_or_reuse_shard(
                shard_path, rows, signature="signature-b", build_record=build_record
            )
            self.assertEqual(calls, ["1aaa", "2bbb", "1aaa", "2bbb"])

    def test_merge_writes_official_order_folds_and_complete_graph_coverage(self):
        from dta.rank_fusion_v4.three_dprot_prepare import (
            build_or_reuse_shard,
            merge_prepared_shards,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                {"pdb_id": "1aaa", "smiles": "CC", "label": 6.1, "split": "train", "date": "1999-01-01"},
                {"pdb_id": "2bbb", "smiles": "CO", "label": 7.2, "split": "valid", "date": "2008-01-01"},
                {"pdb_id": "3ccc", "smiles": "CCC", "label": 8.3, "split": "test", "date": "2014-01-01"},
            ]

            def build_record(row):
                value = float(int(row["pdb_id"][0]))
                return {
                    "pdb_id": row["pdb_id"],
                    "ligand": row["smiles"],
                    "label": row["label"],
                    "split": row["split"],
                    "ligand_graph": _graph(value),
                    "ligand_ecfp": np.full(8, value, dtype=np.float32),
                    "protein_graph": _graph(value + 10),
                }

            shard_paths = []
            for index, subset in enumerate((rows[:2], rows[2:])):
                path = root / "shards" / ("shard_%05d.pkl" % index)
                build_or_reuse_shard(path, subset, "fixed", build_record)
                shard_paths.append(path)

            output = root / "prepared"
            summary = merge_prepared_shards(
                shard_paths,
                rows,
                output,
                provenance={"signature": "fixed"},
            )

            frame = pd.read_csv(output / "full.csv")
            self.assertEqual(frame["protein"].tolist(), ["1aaa", "2bbb", "3ccc"])
            self.assertEqual(frame["label"].tolist(), [6.1, 7.2, 8.3])
            self.assertEqual(
                json.loads((output / "folds/train_fold_setting1.txt").read_text()),
                [[1]],
            )
            self.assertEqual(
                json.loads((output / "folds/test_fold_setting1.txt").read_text()),
                [2],
            )
            with (output / "protein_to_graph.pkl").open("rb") as handle:
                self.assertEqual(set(pickle.load(handle)), {"1aaa", "2bbb", "3ccc"})
            with (output / "ligand_to_graph.pkl").open("rb") as handle:
                self.assertEqual(set(pickle.load(handle)), {"CC", "CO", "CCC"})
            self.assertEqual(summary["counts"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["status"], "complete")
            persisted = json.loads((output / "prepare_summary.json").read_text())
            self.assertEqual(persisted["status"], "complete")

    def test_merge_rejects_label_mismatch_before_committing_summary(self):
        from dta.rank_fusion_v4.three_dprot_prepare import (
            build_or_reuse_shard,
            merge_prepared_shards,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [{"pdb_id": "1aaa", "smiles": "CC", "label": 6.1, "split": "train", "date": "1999-01-01"}]

            def wrong_record(row):
                return {
                    "pdb_id": row["pdb_id"],
                    "ligand": row["smiles"],
                    "label": 9.9,
                    "split": row["split"],
                    "ligand_graph": _graph(1.0),
                    "ligand_ecfp": np.ones(8, dtype=np.float32),
                    "protein_graph": _graph(2.0),
                }

            shard = root / "shard.pkl"
            build_or_reuse_shard(shard, rows, "fixed", wrong_record)
            output = root / "prepared"
            output.mkdir()
            (output / "prepare_summary.json").write_text(
                json.dumps({"status": "complete", "signature": "stale"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "label mismatch"):
                merge_prepared_shards([shard], rows, output, {"signature": "fixed"})
            self.assertFalse((output / "prepare_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
