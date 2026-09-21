import unittest

import torch


def make_batch():
    torch.manual_seed(601)
    batch_size = 2
    ligand_count = 3
    protein_count = 4
    residue_count = 5
    batch = {
        "gerna_2d_molecule": torch.randn(batch_size, 128),
        "gerna_2d_protein": torch.randn(batch_size, 128),
        "gerna_3d_molecule": torch.randn(batch_size, 128),
        "gerna_3d_protein": torch.randn(batch_size, 128),
        "gerna_2d_interaction": torch.randn(batch_size, 128, 128),
        "gerna_3d_interaction": torch.randn(batch_size, 128, 128),
        "gerna_head": torch.randn(batch_size, 128),
        "interface_ligand_nodes": torch.randn(batch_size, ligand_count, 96),
        "interface_protein_nodes": torch.randn(batch_size, protein_count, 96),
        "interface_residue_nodes": torch.randn(batch_size, residue_count, 128),
        "interface_ligand_coords": torch.randn(batch_size, ligand_count, 3),
        "interface_protein_coords": torch.randn(batch_size, protein_count, 3),
        "interface_residue_coords": torch.randn(batch_size, residue_count, 3),
        "interface_ligand_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        "interface_protein_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
        "interface_residue_mask": torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        ),
        "interface_ligand_summary": torch.randn(batch_size, 96),
        "interface_protein_summary": torch.randn(batch_size, 128),
        "interface_local_summary": torch.randn(batch_size, 128),
        "interface_head": torch.randn(batch_size, 128),
        "interface_score": torch.randn(batch_size, 1),
        "interface_projection": torch.randn(batch_size, 64),
        "gerna_prediction": torch.tensor([[1.0], [2.0]]),
        "interface_prediction": torch.tensor([[2.0], [3.0]]),
    }
    return batch


class TestHybridUpperBoundFusion(unittest.TestCase):
    @staticmethod
    def build_model():
        from net.hybrid_upper_bound import HybridUpperBoundFusion

        torch.manual_seed(607)
        return HybridUpperBoundFusion(dropout=0.0, drop_path_rate=0.0)

    def test_output_shapes_and_all_trainable_parameters_receive_gradients(self):
        model = self.build_model()
        model.train()
        output = model(make_batch())

        self.assertEqual(tuple(output["prediction"].shape), (2, 1))
        self.assertEqual(tuple(output["expert_predictions"].shape), (2, 8))
        self.assertEqual(tuple(output["expert_weights"].shape), (2, 8))
        self.assertEqual(tuple(output["latents"].shape), (2, 32, 512))
        self.assertEqual(
            len(model.perceiver_blocks), 8
        )
        self.assertEqual(len(model.latent_blocks), 4)
        self.assertTrue(torch.isfinite(output["prediction"]).all())
        output["prediction"].sum().backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

    def test_masked_padding_and_other_sample_cannot_change_a_prediction(self):
        model = self.build_model().eval()
        batch = make_batch()
        expected = model(batch)["prediction"]

        padded = {name: value.clone() if torch.is_tensor(value) else value for name, value in batch.items()}
        padded["interface_ligand_nodes"][0, 2:] = 1e6
        padded["interface_protein_nodes"][0, 2:] = -1e6
        padded["interface_residue_nodes"][0, 3:] = 1e6
        padded["interface_ligand_coords"][0, 2:] = 1e6
        padded["interface_protein_coords"][0, 2:] = -1e6
        padded["interface_residue_coords"][0, 3:] = 1e6
        torch.testing.assert_close(
            expected, model(padded)["prediction"], rtol=1e-5, atol=1e-6
        )

        changed = {name: value.clone() if torch.is_tensor(value) else value for name, value in batch.items()}
        for name, value in changed.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.dtype != torch.bool
            ):
                value[1] = torch.randn_like(value[1]) * 100.0
        changed["interface_ligand_mask"][1] = batch["interface_ligand_mask"][1]
        changed["interface_protein_mask"][1] = batch["interface_protein_mask"][1]
        changed["interface_residue_mask"][1] = batch["interface_residue_mask"][1]
        actual = model(changed)["prediction"]
        torch.testing.assert_close(
            expected[0], actual[0], rtol=1e-5, atol=1e-6
        )

    def test_output_is_not_clipped_to_backbone_prediction_range(self):
        model = self.build_model().eval()
        with torch.no_grad():
            for expert in model.experts:
                for parameter in expert.parameters():
                    parameter.zero_()
                expert[-1].bias.fill_(10.0)
            for parameter in model.gate.parameters():
                parameter.zero_()
        output = model(make_batch())["prediction"]
        self.assertTrue(torch.all(output > 9.0))


if __name__ == "__main__":
    unittest.main()
