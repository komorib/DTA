import tempfile
import unittest
from pathlib import Path

import pandas as pd


class DummyDataset:
    def __init__(self, pair_indices):
        self.pair_indices = pair_indices


class TestCasterLPAdapter(unittest.TestCase):
    def test_lp_rows_are_converted_to_caster_dataframe(self):
        from scripts.train_caster_lp import make_caster_dataframe

        rows = pd.DataFrame(
            [
                {
                    "pdb_id": "1abc",
                    "seq": "ACD",
                    "smiles": "CCO",
                    "value": 6.5,
                    "new_split": "train",
                    "protein_file": "/tmp/1abc_protein.pdb",
                },
                {
                    "pdb_id": "2def",
                    "seq": "GG",
                    "smiles": "CCC",
                    "value": 7.1,
                    "new_split": "val",
                    "protein_file": "",
                },
            ]
        )

        usable, caster_df = make_caster_dataframe(rows)

        self.assertEqual(list(usable["pdb_id"]), ["1abc"])
        self.assertEqual(list(caster_df.columns), [
            "protein_id",
            "protein_sequence",
            "protein_file",
            "molecule_id",
            "molecule_smiles",
            "affinity_score",
            "split",
        ])
        self.assertEqual(caster_df.iloc[0]["protein_id"], "1abc")
        self.assertEqual(caster_df.iloc[0]["molecule_id"], "1abc")
        self.assertEqual(caster_df.iloc[0]["affinity_score"], 6.5)
        self.assertEqual(caster_df.iloc[0]["split"], "train")

    def test_official_model_kwargs_match_caster_training_defaults(self):
        from scripts.train_caster_lp import build_official_model_kwargs

        metadata = {
            "protein_node_features": (17, 3),
            "protein_edge_features": (32, 1),
            "protein_node_types": 20,
            "protein_edge_types": 1,
            "molecule_node_features": 41,
            "molecule_edge_features": 9,
            "molecule_node_types": 11,
            "molecule_edge_types": 5,
        }

        model_kwargs = build_official_model_kwargs(metadata)

        self.assertEqual(model_kwargs["protein_gnn_kwargs"]["base_conv"], "lbamodel")
        self.assertEqual(model_kwargs["protein_gnn_kwargs"]["hidden_channels"], (16, 4))
        self.assertEqual(model_kwargs["protein_gnn_kwargs"]["out_channels"], 64)
        self.assertEqual(model_kwargs["molecule_gnn_kwargs"]["base_conv"], "gine")
        self.assertEqual(model_kwargs["molecule_gnn_kwargs"]["hidden_channels"], 16)
        self.assertEqual(model_kwargs["joint_gnn_kwargs"]["pairwise_embedding_dim"], 512)
        self.assertEqual(model_kwargs["joint_gnn_kwargs"]["num_cross_attn_layers"], 1)

    def test_split_indices_are_derived_after_graph_filtering(self):
        from scripts.train_caster_lp import split_indices_from_dataset

        caster_df = pd.DataFrame(
            [
                {"protein_id": "a", "molecule_id": "a", "split": "train"},
                {"protein_id": "b", "molecule_id": "b", "split": "val"},
                {"protein_id": "c", "molecule_id": "c", "split": "test"},
            ]
        )
        dataset = DummyDataset({0: ("c", "c"), 1: ("a", "a")})

        split_indices = split_indices_from_dataset(dataset, caster_df)

        self.assertEqual(split_indices["train"], [1])
        self.assertEqual(split_indices["val"], [])
        self.assertEqual(split_indices["test"], [0])


if __name__ == "__main__":
    unittest.main()
