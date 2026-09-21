import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_interface_item():
    torch.manual_seed(101)
    ligand_count = 4
    residue_count = 3
    protein_count = 5
    ligand_edges = torch.tensor(
        [
            [0, 1, 0, 2, 0, 3],
            [1, 0, 2, 0, 3, 0],
        ],
        dtype=torch.long,
    )
    return {
        "key": "synthetic_protein",
        "ligand_atom_features": torch.randn(ligand_count, 55),
        "ligand_edge_index": ligand_edges,
        "ligand_edge_attr": torch.randn(ligand_edges.shape[1], 13),
        "ligand_coords": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.0, 1.1, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "ligand_canonical_rank": torch.tensor([0, 1, 2, 3]),
        "ligand_chiral_tag": torch.tensor([1, 0, 0, 0]),
        "protein_coords": torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [2.5, 1.0, 0.0],
                [3.0, 0.0, 1.0],
                [4.0, 1.0, 0.5],
                [5.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "protein_element": torch.tensor([0, 1, 2, 3, 0]),
        "protein_atom_name_class": torch.tensor([1, 2, 3, 4, 6]),
        "protein_residue_type": torch.tensor([0, 0, 1, 1, 2]),
        "protein_is_backbone": torch.tensor(
            [True, True, True, False, False]
        ),
        "protein_parent_residue": torch.tensor([0, 0, 1, 1, 2]),
        "residue_esm": torch.randn(residue_count, 640),
        "residue_ca_coords": torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [3.8, 0.5, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }


class TestConditionedAttentionPool(unittest.TestCase):
    def test_masked_states_do_not_change_conditioned_pool(self):
        from net.interface_first import ConditionedAttentionPool

        torch.manual_seed(103)
        pool = ConditionedAttentionPool(
            input_dim=8,
            condition_dim=6,
            output_dim=5,
        ).eval()
        states = torch.randn(5, 8)
        condition = torch.randn(1, 6)
        mask = torch.tensor([True, True, True, False, False])
        expected = pool(states, condition, mask=mask)

        changed = states.clone()
        changed[~mask] = torch.randn(2, 8) * 10000.0
        actual = pool(changed, condition, mask=mask)

        self.assertEqual(expected.shape, (1, 5))
        torch.testing.assert_close(expected, actual)

    def test_pool_rejects_an_empty_mask(self):
        from net.interface_first import ConditionedAttentionPool

        pool = ConditionedAttentionPool(4, 3, 2)
        with self.assertRaisesRegex(ValueError, "at least one"):
            pool(
                torch.randn(2, 4),
                torch.randn(1, 3),
                mask=torch.zeros(2, dtype=torch.bool),
            )


class TestInterfaceFirstGerNADTA(unittest.TestCase):
    def test_forward_shapes_and_required_gradients_are_finite(self):
        from net.interface_first import InterfaceFirstGerNADTA

        model = InterfaceFirstGerNADTA(
            ligand_gnn_depth=2,
            ligand_dropout=0.0,
            affinity_dropout=0.0,
        )
        model.configure_stage2_full()
        prediction, latent, score, projection = model(
            make_interface_item()
        )

        self.assertEqual(prediction.shape, (1, 1))
        self.assertEqual(latent.shape, (1, 128))
        self.assertEqual(score.shape, (1, 1))
        self.assertEqual(projection.shape, (1, 64))
        for tensor in (prediction, latent, score, projection):
            self.assertTrue(torch.isfinite(tensor).all())

        (
            prediction.sum()
            + latent.sum()
            + score.sum()
            + projection.sum()
        ).backward()
        required = (
            "local_path.ligand_gnn.atom_encoder.weight",
            "local_path.protein_chemical_encoder.element_embedding.weight",
            "local_path.esm_atom_adapter.local_projection.1.weight",
            "local_path.interface_blocks.0.scalar_update.0.weight",
            "local_path.interface_pool.pool_score.weight",
            "local_path.projection_head.0.weight",
            "local_path.score_head.weight",
            "protein_global.esm_projection.1.weight",
            "protein_conditioned_pool.query.weight",
            "ligand_conditioned_pool.query.weight",
            "affinity_head.0.weight",
        )
        parameters = dict(model.named_parameters())
        for name in required:
            self.assertIn(name, parameters)
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)

    def test_stage1_contract_is_closed_and_checkpoint_is_complete(self):
        from net.interface_first import InterfaceFirstGerNADTA

        torch.manual_seed(107)
        source = InterfaceFirstGerNADTA(ligand_gnn_depth=2)
        source.configure_stage1()
        trainable = {
            name
            for name, parameter in source.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("local_path.") for name in trainable)
        )
        for prefix in (
            "local_path.ligand_gnn",
            "local_path.protein_chemical_encoder",
            "local_path.esm_atom_adapter",
            "local_path.interface_blocks",
            "local_path.interface_pool",
            "local_path.projection_head",
            "local_path.score_head",
        ):
            self.assertTrue(
                any(name.startswith(prefix) for name in trainable),
                prefix,
            )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in source.local_path.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in source.protein_global.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in source.affinity_head.parameters()
            )
        )

        checkpoint = source.stage1_state_dict()
        self.assertEqual(set(checkpoint), {"local_path"})
        self.assertEqual(
            set(checkpoint["local_path"]),
            set(source.local_path.state_dict()),
        )
        torch.manual_seed(109)
        target = InterfaceFirstGerNADTA(ligand_gnn_depth=2)
        global_before = {
            name: value.detach().clone()
            for name, value in target.protein_global.state_dict().items()
        }
        target.load_stage1_state_dict(checkpoint, strict=True)
        for name, expected in source.local_path.state_dict().items():
            torch.testing.assert_close(
                expected, target.local_path.state_dict()[name]
            )
        for name, expected in global_before.items():
            torch.testing.assert_close(
                expected, target.protein_global.state_dict()[name]
            )

        incomplete = {
            "local_path": dict(checkpoint["local_path"])
        }
        incomplete["local_path"].pop(next(iter(incomplete["local_path"])))
        with self.assertRaises(RuntimeError):
            target.load_stage1_state_dict(incomplete, strict=True)

    def test_stage2_modes_and_optimizer_groups_are_named(self):
        from net.interface_first import InterfaceFirstGerNADTA

        model = InterfaceFirstGerNADTA(ligand_gnn_depth=2)
        model.configure_stage2_warmup()
        warmup_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            all(
                name.startswith(("local_path.", "affinity_head."))
                for name in warmup_names
            )
        )
        warmup_groups = model.optimizer_parameter_groups(
            local_lr=1e-4,
            global_lr=2e-5,
            affinity_lr=3e-4,
        )
        self.assertEqual(
            [group["name"] for group in warmup_groups],
            ["local", "affinity"],
        )

        model.configure_stage2_full()
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        full_groups = model.optimizer_parameter_groups(
            local_lr=1e-4,
            global_lr=2e-5,
            affinity_lr=3e-4,
        )
        self.assertEqual(
            [group["name"] for group in full_groups],
            ["local", "global", "affinity"],
        )

    def test_import_does_not_replace_legacy_gerna_class(self):
        import dta.model_dta as legacy

        original_class = legacy.GerNADTA
        from net.interface_first import InterfaceFirstGerNADTA

        self.assertIs(legacy.GerNADTA, original_class)
        self.assertIsNot(InterfaceFirstGerNADTA, original_class)


if __name__ == "__main__":
    unittest.main()
