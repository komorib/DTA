import tempfile
import unittest
from pathlib import Path

import torch


class TestAttentionMGTLPAdapter(unittest.TestCase):
    def test_compound_and_protein_graph_shapes_match_model_inputs(self):
        from scripts.train_attentionmgt_lp import build_compound_graph, build_protein_graph_from_residue_records

        compound = build_compound_graph("CCO")
        self.assertEqual(compound.ndata["atom"].shape[1], 44)
        self.assertEqual(compound.edata["bond"].shape[1], 10)
        self.assertEqual(compound.ndata["lap_pos_enc"].shape[1], 8)

        residues = [
            {
                "resname": "ALA",
                "chain": "A",
                "resseq": 1,
                "ca": (0.0, 0.0, 0.0),
                "coords": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                "embedding": [0.1] * 320,
                "angles": [0.0, 0.0, 0.0, 0.0],
            },
            {
                "resname": "GLY",
                "chain": "A",
                "resseq": 2,
                "ca": (3.8, 0.0, 0.0),
                "coords": [(3.8, 0.0, 0.0), (4.8, 0.0, 0.0)],
                "embedding": [0.2] * 320,
                "angles": [0.0, 0.0, 0.0, 0.0],
            },
        ]
        graph, embedding = build_protein_graph_from_residue_records(residues)
        self.assertEqual(graph.ndata["feats"].shape[1], 41)
        self.assertEqual(graph.edata["feats"].shape[1], 5)
        self.assertEqual(graph.ndata["lap_pos_enc"].shape[1], 8)
        self.assertEqual(embedding.shape, (2, 320))

    def test_checkpoints_are_written(self):
        from scripts.train_attentionmgt_lp import write_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            record = {"val": {"ci": 0.6}}
            write_checkpoints(out_dir, model, optimizer, epoch=1, best_record=record, is_best=True, args={"epochs": 1})

            for name in ("best_model.pt", "last_model.pt", "training_checkpoint.pt"):
                self.assertTrue((out_dir / name).exists())
            payload = torch.load(out_dir / "training_checkpoint.pt", map_location="cpu")
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["best_record"], record)


    def test_dgl_function_compatibility_patch(self):
        import dgl.function as fn
        from scripts.train_attentionmgt_lp import patch_dgl_function_compatibility

        patch_dgl_function_compatibility()
        self.assertTrue(hasattr(fn, "src_mul_edge"))
        self.assertTrue(hasattr(fn, "copy_edge"))

    def test_training_progress_message_contains_epoch_batch_and_loss(self):
        from scripts.train_attentionmgt_lp import format_train_progress

        message = format_train_progress(epoch=3, batch_idx=99, total_batches=2713, loss=1.23456)
        self.assertIn("epoch 3", message)
        self.assertIn("batch 100/2713", message)
        self.assertIn("loss 1.234560", message)



if __name__ == "__main__":
    unittest.main()
