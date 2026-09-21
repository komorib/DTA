import tempfile
import unittest
from pathlib import Path

import torch


class TestDeepDTAGenLPAdapter(unittest.TestCase):
    def test_smile_graph_and_sequence_match_official_inputs(self):
        from scripts.train_deepdtagen_lp import sequence_to_array, smile_to_graph

        c_size, features, edge_index, edge_feats = smile_to_graph("CCO")
        self.assertEqual(c_size, 3)
        self.assertEqual(len(features[0]), 94)
        self.assertEqual(len(edge_index[0]), 2)
        self.assertEqual(len(edge_feats[0]), 5)

        seq = sequence_to_array("ACD:EFG*")
        self.assertEqual(seq.shape, (1000,))
        self.assertEqual(seq[0], 1)
        self.assertEqual(seq[1], 3)
        self.assertEqual(seq[2], 4)
        self.assertEqual(seq[5], 7)
        self.assertEqual(seq[6], 0)

    def test_checkpoints_are_written_for_best_and_last(self):
        from scripts.train_deepdtagen_lp import write_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            record = {"val": {"ci": 0.6}, "test": {"ci": 0.58}}
            write_checkpoints(out_dir, model, optimizer, epoch=3, best_record=record, is_best=True, args={"epochs": 3})

            self.assertTrue((out_dir / "best_model.pt").exists())
            self.assertTrue((out_dir / "last_model.pt").exists())
            self.assertTrue((out_dir / "training_checkpoint.pt").exists())
            payload = torch.load(out_dir / "training_checkpoint.pt", map_location="cpu")
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["best_record"], record)

    def test_best_selection_falls_back_to_loss_when_ci_is_nan(self):
        from scripts.train_deepdtagen_lp import is_better_validation

        improved, best_ci, best_loss = is_better_validation({"ci": float("nan"), "loss": 2.0}, -float("inf"), float("inf"))
        self.assertTrue(improved)
        self.assertEqual(best_ci, -float("inf"))
        self.assertEqual(best_loss, 2.0)

        improved, best_ci, best_loss = is_better_validation({"ci": 0.61, "loss": 3.0}, -float("inf"), 2.0)
        self.assertTrue(improved)
        self.assertEqual(best_ci, 0.61)
        self.assertEqual(best_loss, 3.0)


    def test_tokenized_smiles_is_clipped_to_deepdtagen_position_limit(self):
        from scripts.train_deepdtagen_lp import DEEPDTAGEN_TARGET_MAX_LEN, tokenize_smiles_for_model

        class FakeTokenizer:
            s2i = {"<eos>": 9}
            def parse(self, smiles):
                return list(range(200))

        tokens = tokenize_smiles_for_model(FakeTokenizer(), "long")
        self.assertEqual(tokens.shape[0], DEEPDTAGEN_TARGET_MAX_LEN)
        self.assertEqual(int(tokens[-1]), 9)


    def test_tokenized_smiles_can_be_dropped_when_too_long(self):
        from scripts.train_deepdtagen_lp import tokenize_smiles_for_model

        class FakeTokenizer:
            s2i = {"<eos>": 9}
            def parse(self, smiles):
                return list(range(200))

        self.assertIsNone(tokenize_smiles_for_model(FakeTokenizer(), "long", policy="drop"))
        kept = tokenize_smiles_for_model(FakeTokenizer(), "long", policy="truncate")
        self.assertIsNotNone(kept)



if __name__ == "__main__":
    unittest.main()
