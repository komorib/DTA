import unittest
from unittest import mock

import torch


class TestGerNAAffinityFeatures(unittest.TestCase):
    @staticmethod
    def build_model():
        from dta.model_dta import GerNADTA

        torch.manual_seed(211)
        return GerNADTA(
            [2, 1, 8, 4],
            trigonometry=False,
            mol_graph=True,
            coors=True,
            rna_repre=True,
            rna_graph=True,
            uncertainty=False,
        ).eval()

    @staticmethod
    def affinity_inputs():
        torch.manual_seed(223)
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

    def test_capture_is_single_use_and_prediction_is_exactly_legacy_head(self):
        from dta.model_dta import GerNADTA

        model = self.build_model()
        with self.assertRaisesRegex(RuntimeError, "no captured GerNA"):
            model.take_affinity_features()
        parent_hook_calls = []
        parent_hook = model.W_out.register_forward_hook(
            lambda module, inputs, output: parent_hook_calls.append(output)
        )

        two_d = (
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0]], requires_grad=True
            ),
            torch.tensor(
                [[0.5, 1.5, 2.5, 3.5]], requires_grad=True
            ),
        )
        three_d = (
            torch.tensor(
                [[4.0, 3.0, 2.0, 1.0]], requires_grad=True
            ),
            torch.tensor(
                [[3.5, 2.5, 1.5, 0.5]], requires_grad=True
            ),
        )
        with mock.patch.object(
            GerNADTA, "dma_gru_2d", return_value=two_d
        ), mock.patch.object(
            GerNADTA, "dma_gru_3d", return_value=three_d
        ):
            legacy_prediction = model.Affinity_pred_module(
                **self.affinity_inputs()
            )
        with self.assertRaisesRegex(RuntimeError, "no captured GerNA"):
            model.take_affinity_features()

        model.request_affinity_features()
        with self.assertRaisesRegex(RuntimeError, "already requested"):
            model.request_affinity_features()
        with mock.patch.object(
            GerNADTA, "dma_gru_2d", return_value=two_d
        ), mock.patch.object(
            GerNADTA, "dma_gru_3d", return_value=three_d
        ):
            prediction = model.Affinity_pred_module(
                **self.affinity_inputs()
            )
        with mock.patch.object(
            GerNADTA, "dma_gru_2d", return_value=two_d
        ), mock.patch.object(
            GerNADTA, "dma_gru_3d", return_value=three_d
        ):
            with self.assertRaisesRegex(RuntimeError, "unread"):
                model.Affinity_pred_module(**self.affinity_inputs())
        parent_hook.remove()

        features = model.take_affinity_features()

        self.assertEqual(len(parent_hook_calls), 2)
        self.assertTrue(torch.equal(legacy_prediction, prediction))
        self.assertEqual(
            set(features), {"gerna_2d", "gerna_3d", "gerna_head"}
        )
        self.assertEqual(features["gerna_2d"].shape, (1, 16))
        self.assertEqual(features["gerna_3d"].shape, (1, 16))
        self.assertEqual(features["gerna_head"].shape, (1, 128))
        for feature in features.values():
            feature.retain_grad()
        combined = torch.cat(
            [features["gerna_2d"], features["gerna_3d"]], dim=-1
        )
        expected = model.W_out(combined)
        torch.testing.assert_close(
            prediction, expected, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            prediction,
            model.W_out[-1](features["gerna_head"]),
            rtol=0.0,
            atol=0.0,
        )
        prediction.sum().backward()
        for name, feature in features.items():
            self.assertIsNotNone(feature.grad, name)
            self.assertTrue(torch.isfinite(feature.grad).all(), name)
        with self.assertRaisesRegex(RuntimeError, "no captured GerNA"):
            model.take_affinity_features()

    def test_high_level_parameters_are_explicit_complete_and_unique(self):
        model = self.build_model()
        modules = [
            model.W_out,
            model.mol_2d_final,
            model.rna_1d_final,
            model.rna_2d_final,
            model.mc1,
            model.mp1,
            model.hc0,
            model.hp0,
            model.hc1,
            model.hp1,
            model.m_to_r_transform,
            model.r_to_m_transform,
            model.GRU_dma,
            model.mol_3d_final,
            model.rna_3d_final,
            model.mc1_3d,
            model.mp1_3d,
            model.hc0_3d,
            model.hp0_3d,
            model.hc1_3d,
            model.hp1_3d,
            model.m_to_r_transform_3d,
            model.r_to_m_transform_3d,
            model.GRU_dma_3d,
        ]
        expected = {
            id(parameter)
            for module in modules
            for parameter in module.parameters()
        }

        actual = model.high_level_affinity_parameters()
        actual_ids = [id(parameter) for parameter in actual]

        self.assertEqual(set(actual_ids), expected)
        self.assertEqual(len(actual_ids), len(set(actual_ids)))
        self.assertNotIn(
            id(next(model.GCN_mol.parameters())), set(actual_ids)
        )

    def test_pending_feature_request_can_be_discarded(self):
        model = self.build_model()

        model.request_affinity_features()
        model.discard_affinity_features()
        model.request_affinity_features()
        model.discard_affinity_features()

        with self.assertRaisesRegex(RuntimeError, "no captured GerNA"):
            model.take_affinity_features()


class TestInterfaceAffinityFeatures(unittest.TestCase):
    def test_optional_features_preserve_legacy_prediction_and_shapes(self):
        from net.interface_first import InterfaceFirstGerNADTA
        from tests.test_interface_first_model import make_interface_item

        torch.manual_seed(227)
        model = InterfaceFirstGerNADTA(
            ligand_gnn_depth=2,
            ligand_dropout=0.0,
            affinity_dropout=0.0,
        ).eval()
        item = make_interface_item()
        parent_hook_calls = []
        parent_hook = model.affinity_head.register_forward_hook(
            lambda module, inputs, output: parent_hook_calls.append(output)
        )

        legacy = model(item)
        prediction, features = model(item, return_features=True)
        parent_hook.remove()

        self.assertEqual(len(parent_hook_calls), 2)
        self.assertTrue(torch.equal(legacy[0], prediction))
        self.assertEqual(
            set(features),
            {
                "interface_protein",
                "interface_ligand",
                "interface_local",
                "interface_head",
            },
        )
        self.assertEqual(features["interface_protein"].shape, (1, 128))
        self.assertEqual(features["interface_ligand"].shape, (1, 96))
        self.assertEqual(features["interface_local"].shape, (1, 128))
        self.assertEqual(features["interface_head"].shape, (1, 128))
        torch.testing.assert_close(
            prediction,
            model.affinity_head[-1](features["interface_head"]),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
