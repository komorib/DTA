import unittest

import pandas as pd


class TestTankBindLPAdapter(unittest.TestCase):
    def test_lp_rows_are_split_and_mapped_to_tankbind_native_rows(self):
        from scripts.train_tankbind_lp import load_lp_frames_from_dataframe, make_tankbind_frame

        frame = pd.DataFrame(
            {
                "pdb_id": ["1aaa", "2bbb", "3ccc", "4ddd"],
                "smiles": ["CCO", "CCN", "CCC", "CCCl"],
                "value": [6.1, 7.2, 5.3, 8.4],
                "new_split": ["train", "val", "test", None],
            }
        )

        splits = load_lp_frames_from_dataframe(frame)
        self.assertEqual(list(splits), ["train", "val", "test"])
        self.assertEqual(len(splits["train"]), 1)
        self.assertEqual(len(splits["val"]), 1)
        self.assertEqual(len(splits["test"]), 1)

        train_rows = make_tankbind_frame(splits["train"], group="train")
        self.assertEqual(
            list(train_rows.columns),
            [
                "protein_name",
                "compound_name",
                "pdb",
                "smiles",
                "affinity",
                "uid",
                "pocket_com",
                "use_compound_com",
                "use_whole_protein",
                "group",
            ],
        )
        self.assertEqual(train_rows.loc[0, "protein_name"], "1aaa")
        self.assertEqual(train_rows.loc[0, "compound_name"], "1aaa_ligand")
        self.assertTrue(bool(train_rows.loc[0, "use_compound_com"]))
        self.assertFalse(bool(train_rows.loc[0, "use_whole_protein"]))

    def test_metrics_include_ci_for_affinity_predictions(self):
        from scripts.train_tankbind_lp import compute_affinity_metrics

        metrics = compute_affinity_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["ci"], 1.0)
        self.assertLess(metrics["rmse"], 0.2)


if __name__ == "__main__":
    unittest.main()
