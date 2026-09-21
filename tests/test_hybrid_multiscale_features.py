import unittest
from unittest import mock

import torch
import torch.nn.functional as F


class TestGerNAMultiscaleFeatures(unittest.TestCase):
    @staticmethod
    def build_model():
        from dta.model_dta import GerNADTA

        torch.manual_seed(401)
        return GerNADTA(
            [2, 1, 8, 128],
            trigonometry=False,
            mol_graph=True,
            coors=True,
            rna_repre=True,
            rna_graph=True,
            uncertainty=False,
        ).eval()

    @staticmethod
    def affinity_inputs():
        torch.manual_seed(409)
        return {
            "batch_size": 1,
            "RNA_1d_fea": torch.randn(1, 2, 8),
            "RNA_2d_fea": torch.randn(1, 2, 8),
            "Mol_2d_fea": torch.randn(1, 3, 8),
            "seq_mask": torch.ones(1, 2),
            "vertex_mask": torch.ones(1, 3),
            "RNA_feats": torch.randn(1, 2, 16),
            "Mol_feats": torch.randn(1, 3, 16),
            "RNA_mask": torch.ones(1, 2),
            "Mol_mask": torch.ones(1, 3),
            "pairwise_pred_2d": torch.zeros(1, 2, 3),
            "pairwise_pred_3d": torch.zeros(1, 2, 3),
        }

    def test_capture_preserves_prediction_and_exposes_factor_matrices(self):
        from dta.model_dta import GerNADTA

        model = self.build_model()
        two_d = (
            torch.linspace(0.1, 1.0, 128).unsqueeze(0).requires_grad_(),
            torch.linspace(1.0, 0.1, 128).unsqueeze(0).requires_grad_(),
        )
        three_d = (
            torch.linspace(-0.5, 0.5, 128).unsqueeze(0).requires_grad_(),
            torch.linspace(0.75, -0.25, 128).unsqueeze(0).requires_grad_(),
        )
        with mock.patch.object(
            GerNADTA, "dma_gru_2d", return_value=two_d
        ), mock.patch.object(
            GerNADTA, "dma_gru_3d", return_value=three_d
        ):
            legacy_prediction = model.Affinity_pred_module(
                **self.affinity_inputs()
            )

        model.request_multiscale_affinity_features()
        with mock.patch.object(
            GerNADTA, "dma_gru_2d", return_value=two_d
        ), mock.patch.object(
            GerNADTA, "dma_gru_3d", return_value=three_d
        ):
            prediction = model.Affinity_pred_module(
                **self.affinity_inputs()
            )
        features = model.take_multiscale_affinity_features()

        self.assertTrue(torch.equal(legacy_prediction, prediction))
        expected_shapes = {
            "gerna_2d_molecule": (1, 128),
            "gerna_2d_protein": (1, 128),
            "gerna_3d_molecule": (1, 128),
            "gerna_3d_protein": (1, 128),
            "gerna_2d_interaction": (1, 128, 128),
            "gerna_3d_interaction": (1, 128, 128),
            "gerna_head": (1, 128),
        }
        self.assertEqual(set(features), set(expected_shapes))
        for name, expected_shape in expected_shapes.items():
            self.assertEqual(tuple(features[name].shape), expected_shape)
            features[name].retain_grad()

        torch.testing.assert_close(
            features["gerna_2d_interaction"],
            F.leaky_relu(
                torch.einsum(
                    "bi,bj->bij",
                    features["gerna_2d_molecule"],
                    features["gerna_2d_protein"],
                ),
                negative_slope=0.1,
            ),
        )
        torch.testing.assert_close(
            features["gerna_3d_interaction"],
            F.leaky_relu(
                torch.einsum(
                    "bi,bj->bij",
                    features["gerna_3d_molecule"],
                    features["gerna_3d_protein"],
                ),
                negative_slope=0.1,
            ),
        )
        prediction.sum().backward()
        for name, feature in features.items():
            self.assertIsNotNone(feature.grad, name)
            self.assertTrue(torch.isfinite(feature.grad).all(), name)

    def test_capture_is_cleared_when_affinity_head_raises(self):
        model = self.build_model()
        model.request_multiscale_affinity_features()
        with mock.patch.object(
            model.W_out,
            "forward",
            side_effect=RuntimeError("synthetic affinity failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic affinity"):
                model.Affinity_pred_module(**self.affinity_inputs())
        with self.assertRaisesRegex(RuntimeError, "no captured GerNA"):
            model.take_multiscale_affinity_features()
        model.request_multiscale_affinity_features()
        model.discard_multiscale_affinity_features()


class TestInterfaceMultiscaleFeatures(unittest.TestCase):
    def test_multiscale_return_preserves_legacy_prediction_and_alignment(self):
        from net.interface_first import InterfaceFirstGerNADTA
        from tests.test_interface_first_model import make_interface_item

        torch.manual_seed(419)
        model = InterfaceFirstGerNADTA(
            ligand_gnn_depth=2,
            ligand_dropout=0.0,
            affinity_dropout=0.0,
        ).eval()
        item = make_interface_item()

        legacy_prediction = model(item)[0]
        prediction, features = model(
            item, return_multiscale_features=True
        )

        expected = {
            "interface_ligand_nodes",
            "interface_protein_nodes",
            "interface_residue_nodes",
            "interface_ligand_coords",
            "interface_protein_coords",
            "interface_residue_coords",
            "interface_ligand_summary",
            "interface_protein_summary",
            "interface_local_summary",
            "interface_head",
            "interface_score",
            "interface_projection",
        }
        self.assertEqual(set(features), expected)
        self.assertTrue(torch.equal(legacy_prediction, prediction))
        self.assertEqual(
            features["interface_ligand_nodes"].shape[0],
            features["interface_ligand_coords"].shape[0],
        )
        self.assertEqual(
            features["interface_protein_nodes"].shape[0],
            features["interface_protein_coords"].shape[0],
        )
        self.assertEqual(
            features["interface_residue_nodes"].shape[0],
            features["interface_residue_coords"].shape[0],
        )
        torch.testing.assert_close(
            prediction,
            model.affinity_head[-1](features["interface_head"]),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
