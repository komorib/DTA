import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rotation_z(angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def make_geometry():
    ligand = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.2, 1.3, 0.5],
        ],
        dtype=torch.float32,
    )
    protein = torch.tensor(
        [
            [2.0, 0.5, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    bonds = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )
    bond_attr = torch.ones((2, 1), dtype=torch.float32)
    return ligand, protein, bonds, bond_attr


class TestRadialAndSparseEdges(unittest.TestCase):
    def test_radial_basis_shape_and_cosine_cutoff_boundaries(self):
        from net.interface_blocks import RadialBasis, cosine_cutoff

        distances = torch.tensor([0.0, 3.0, 6.0])
        features = RadialBasis(channels=16, cutoff=6.0)(distances)

        self.assertEqual(features.shape, (3, 16))
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(
            cosine_cutoff(distances, cutoff=6.0),
            torch.tensor([1.0, 0.5, 0.0]),
        )
        self.assertEqual(
            cosine_cutoff(torch.tensor([6.1]), cutoff=6.0).item(),
            0.0,
        )

    def test_builder_emits_four_types_and_sparse_cross_edges(self):
        from net.interface_blocks import (
            CROSS_EDGE,
            LIGAND_COVALENT_EDGE,
            LIGAND_SPATIAL_EDGE,
            PROTEIN_SPATIAL_EDGE,
            build_typed_interface_edges,
        )

        ligand = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.8, 0.0, 0.0]]
        )
        protein = torch.tensor(
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
        )
        bonds = torch.tensor([[0, 1], [1, 0]])
        edges = build_typed_interface_edges(
            ligand,
            protein,
            bonds,
            torch.ones((2, 1)),
            ligand_spatial_cutoff=3.0,
            protein_spatial_cutoff=2.0,
            cross_cutoff=2.1,
        )

        self.assertEqual(
            set(edges.edge_type.tolist()),
            {
                LIGAND_COVALENT_EDGE,
                LIGAND_SPATIAL_EDGE,
                PROTEIN_SPATIAL_EDGE,
                CROSS_EDGE,
            },
        )
        self.assertEqual(edges.edge_index.shape[0], 2)
        self.assertEqual(edges.distance.shape[0], edges.edge_index.shape[1])
        self.assertEqual(edges.direction.shape, (edges.edge_index.shape[1], 3))
        cross_count = int((edges.edge_type == CROSS_EDGE).sum())
        self.assertLess(cross_count, 2 * len(ligand) * len(protein))
        self.assertFalse(
            torch.any(edges.edge_index[0] == edges.edge_index[1])
        )

    def test_scatter_softmax_normalizes_each_destination(self):
        from net.interface_blocks import scatter_softmax

        logits = torch.tensor([0.0, 1.0, -1.0, 2.0, 2.0])
        groups = torch.tensor([0, 0, 1, 2, 2])
        weights = scatter_softmax(logits, groups, dim_size=3)

        for group in range(3):
            self.assertAlmostEqual(
                weights[groups == group].sum().item(), 1.0, places=6
            )


class TestTypedScalarVectorBlock(unittest.TestCase):
    def make_block_inputs(self):
        from net.interface_blocks import (
            ScalarVectorState,
            build_typed_interface_edges,
        )

        torch.manual_seed(11)
        ligand, protein, bonds, bond_attr = make_geometry()
        edges = build_typed_interface_edges(
            ligand,
            protein,
            bonds,
            bond_attr,
            ligand_spatial_cutoff=3.0,
            protein_spatial_cutoff=3.0,
            cross_cutoff=3.0,
        )
        node_count = len(ligand) + len(protein)
        state = ScalarVectorState(
            scalar=torch.randn(node_count, 8),
            vector=torch.randn(node_count, 4, 3),
        )
        return ligand, protein, bonds, bond_attr, edges, state

    def test_scalar_is_invariant_and_vector_is_rotation_equivariant(self):
        from net.interface_blocks import (
            ScalarVectorState,
            TypedScalarVectorBlock,
            build_typed_interface_edges,
        )

        (
            ligand,
            protein,
            bonds,
            bond_attr,
            edges,
            state,
        ) = self.make_block_inputs()
        torch.manual_seed(23)
        block = TypedScalarVectorBlock(
            scalar_dim=8,
            vector_channels=4,
            radial_channels=16,
            bond_feature_dim=1,
        ).eval()
        self.assertEqual(block.edge_type_embedding.num_embeddings, 4)
        self.assertGreater(block.edge_type_embedding.embedding_dim, 0)
        pseudoscalar = torch.zeros((state.scalar.shape[0], 1))
        original = block(state, edges, pseudoscalar=pseudoscalar)

        rotation = rotation_z(0.71)
        translation = torch.tensor([4.0, -2.0, 1.5])
        transformed_ligand = ligand @ rotation.T + translation
        transformed_protein = protein @ rotation.T + translation
        transformed_edges = build_typed_interface_edges(
            transformed_ligand,
            transformed_protein,
            bonds,
            bond_attr,
            ligand_spatial_cutoff=3.0,
            protein_spatial_cutoff=3.0,
            cross_cutoff=3.0,
        )
        transformed_state = ScalarVectorState(
            scalar=state.scalar,
            vector=state.vector @ rotation.T,
        )
        transformed = block(
            transformed_state,
            transformed_edges,
            pseudoscalar=pseudoscalar,
        )

        torch.testing.assert_close(
            original.scalar, transformed.scalar, atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            original.vector @ rotation.T,
            transformed.vector,
            atol=2e-5,
            rtol=2e-5,
        )

    def test_autocast_accepts_low_precision_state_and_float32_geometry(self):
        from net.interface_blocks import (
            ScalarVectorState,
            TypedScalarVectorBlock,
        )

        _, _, _, _, edges, state = self.make_block_inputs()
        low_precision_state = ScalarVectorState(
            scalar=state.scalar.bfloat16(),
            vector=state.vector.bfloat16(),
        )
        block = TypedScalarVectorBlock(
            scalar_dim=8,
            vector_channels=4,
            radial_channels=16,
            bond_feature_dim=1,
        ).eval()

        with torch.autocast(
            device_type="cpu", dtype=torch.bfloat16
        ):
            output = block(low_precision_state, edges)

        self.assertEqual(output.scalar.dtype, torch.bfloat16)
        self.assertEqual(output.vector.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output.scalar).all())
        self.assertTrue(torch.isfinite(output.vector).all())

    def test_node_and_edge_permutation_does_not_change_output(self):
        from net.interface_blocks import (
            ScalarVectorState,
            TypedInterfaceEdges,
            TypedScalarVectorBlock,
        )

        _, _, _, _, edges, state = self.make_block_inputs()
        torch.manual_seed(29)
        block = TypedScalarVectorBlock(
            scalar_dim=8,
            vector_channels=4,
            radial_channels=16,
            bond_feature_dim=1,
        ).eval()
        pseudoscalar = torch.randn((state.scalar.shape[0], 1))
        expected = block(state, edges, pseudoscalar=pseudoscalar)

        permutation = torch.tensor([3, 0, 4, 1, 2])
        old_to_new = torch.argsort(permutation)
        edge_order = torch.arange(
            edges.edge_index.shape[1] - 1, -1, -1
        )
        permuted_edges = TypedInterfaceEdges(
            edge_index=old_to_new[edges.edge_index[:, edge_order]],
            edge_type=edges.edge_type[edge_order],
            distance=edges.distance[edge_order],
            direction=edges.direction[edge_order],
            bond_attr=edges.bond_attr[edge_order],
        )
        permuted_state = ScalarVectorState(
            scalar=state.scalar[permutation],
            vector=state.vector[permutation],
        )
        actual = block(
            permuted_state,
            permuted_edges,
            pseudoscalar=pseudoscalar[permutation],
        )

        torch.testing.assert_close(
            expected.scalar,
            actual.scalar[old_to_new],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            expected.vector,
            actual.vector[old_to_new],
            atol=2e-5,
            rtol=2e-5,
        )

    def test_chiral_mirror_changes_scalar_output(self):
        from net.interface_blocks import (
            ScalarVectorState,
            TypedScalarVectorBlock,
            build_typed_interface_edges,
            ligand_chiral_pseudoscalar,
        )

        ligand = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mirror = ligand.clone()
        mirror[:, 0] *= -1.0
        bonds = torch.tensor(
            [[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]]
        )
        ranks = torch.tensor([0, 1, 2, 3])
        tags = torch.tensor([1, 0, 0, 0])
        native_pseudo = ligand_chiral_pseudoscalar(
            ligand, bonds, ranks, tags
        )
        mirror_pseudo = ligand_chiral_pseudoscalar(
            mirror, bonds, ranks, tags
        )
        self.assertGreater(native_pseudo[0, 0].item(), 0.0)
        self.assertLess(mirror_pseudo[0, 0].item(), 0.0)
        inverted_tags = tags.clone()
        inverted_tags[0] = 2
        mirror_with_inverted_tag = ligand_chiral_pseudoscalar(
            mirror, bonds, ranks, inverted_tags
        )
        self.assertLess(
            mirror_with_inverted_tag[0, 0].item(),
            0.0,
        )

        empty_protein = torch.empty((0, 3))
        native_edges = build_typed_interface_edges(
            ligand, empty_protein, bonds, torch.ones((6, 1))
        )
        mirror_edges = build_typed_interface_edges(
            mirror, empty_protein, bonds, torch.ones((6, 1))
        )
        state = ScalarVectorState(
            scalar=torch.zeros((4, 8)),
            vector=torch.zeros((4, 4, 3)),
        )
        torch.manual_seed(31)
        block = TypedScalarVectorBlock(
            scalar_dim=8,
            vector_channels=4,
            bond_feature_dim=1,
        ).eval()
        native_out = block(state, native_edges, native_pseudo)
        mirror_out = block(state, mirror_edges, mirror_pseudo)

        self.assertFalse(
            torch.allclose(native_out.scalar, mirror_out.scalar)
        )


class TestLigandCentricPool(unittest.TestCase):
    def test_protein_neighbor_order_does_not_change_pooling(self):
        from net.interface_blocks import LigandCentricInterfacePool

        torch.manual_seed(41)
        ligand = torch.randn(3, 8)
        protein = torch.randn(4, 8)
        cross = torch.tensor(
            [[0, 1, 2, 3, 1], [0, 0, 1, 2, 2]],
            dtype=torch.long,
        )
        pool = LigandCentricInterfacePool(
            scalar_dim=8, latent_dim=6
        ).eval()

        expected_latent, expected_ligand = pool(
            ligand, protein, cross
        )
        order = torch.tensor([4, 2, 0, 3, 1])
        actual_latent, actual_ligand = pool(
            ligand, protein, cross[:, order]
        )

        self.assertEqual(expected_latent.shape, (1, 6))
        self.assertEqual(expected_ligand.shape, (3, 8))
        torch.testing.assert_close(
            expected_latent, actual_latent, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            expected_ligand, actual_ligand, atol=1e-6, rtol=1e-6
        )

    def test_projection_combines_attention_and_mean_summaries(self):
        from net.interface_blocks import LigandCentricInterfacePool

        pool = LigandCentricInterfacePool(
            scalar_dim=8, latent_dim=6
        )

        self.assertEqual(
            pool.latent_projection[0].in_features,
            2 * pool.scalar_dim,
        )


if __name__ == "__main__":
    unittest.main()
