import unittest

import torch


class TestPackedInterfaceBatch(unittest.TestCase):
    def test_packed_forward_matches_independent_forward(self):
        from data_utils.packed_hybrid_batch import pack_interface_items
        from net.interface_first import InterfaceFirstGerNADTA
        from tests.test_interface_first_model import make_interface_item

        torch.manual_seed(801)
        model = InterfaceFirstGerNADTA(
            ligand_gnn_depth=2,
            ligand_dropout=0.0,
            affinity_dropout=0.0,
        ).eval()
        first = make_interface_item()
        second = make_interface_item()
        second["ligand_coords"] = second["ligand_coords"] + 1000.0
        second["protein_coords"] = second["protein_coords"] + 1000.0
        second["residue_ca_coords"] = second["residue_ca_coords"] + 1000.0
        expected = torch.cat(
            [model(first)[0], model(second)[0]], dim=0
        )
        prediction, features = model.forward_packed(
            pack_interface_items([first, second])
        )

        torch.testing.assert_close(prediction, expected)
        self.assertEqual(
            tuple(features["interface_ligand_nodes"].shape),
            (2, 4, 96),
        )
        self.assertTrue(features["interface_ligand_mask"].all())
        self.assertEqual(
            tuple(features["interface_residue_nodes"].shape),
            (2, 3, 128),
        )

    def test_membership_filters_overlapping_cross_graph_edges(self):
        from net.interface_blocks import build_typed_interface_edges

        ligand_coords = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        protein_coords = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        edges = build_typed_interface_edges(
            ligand_coords,
            protein_coords,
            torch.empty((2, 0), dtype=torch.long),
            ligand_graph=torch.tensor([0, 1]),
            protein_graph=torch.tensor([0, 1]),
        )
        graph = torch.tensor([0, 1, 0, 1])
        self.assertTrue(
            torch.equal(
                graph[edges.edge_index[0]], graph[edges.edge_index[1]]
            )
        )

    def test_offsets_edges_and_tracks_graph_membership(self):
        from data_utils.packed_hybrid_batch import pack_interface_items
        from tests.test_interface_first_model import make_interface_item

        first = make_interface_item()
        second = make_interface_item()
        second["ligand_coords"] = second["ligand_coords"] + 1000.0
        second["protein_coords"] = second["protein_coords"] + 1000.0
        second["residue_ca_coords"] = second["residue_ca_coords"] + 1000.0
        packed = pack_interface_items([first, second])

        self.assertEqual(packed.batch_size, 2)
        self.assertEqual(
            tuple(packed.ligand_edge_index.shape), (2, 12)
        )
        self.assertTrue(
            torch.equal(
                packed.ligand_graph[packed.ligand_edge_index[0]],
                packed.ligand_graph[packed.ligand_edge_index[1]],
            )
        )
        self.assertTrue(
            torch.equal(
                packed.protein_graph,
                torch.tensor([0] * 5 + [1] * 5),
            )
        )
        self.assertTrue(
            torch.equal(
                packed.residue_graph,
                torch.tensor([0] * 3 + [1] * 3),
            )
        )
        self.assertEqual(
            packed.keys, ["synthetic_protein", "synthetic_protein"]
        )


if __name__ == "__main__":
    unittest.main()



class TestPackedGernaPreparation(unittest.TestCase):
    def test_prepares_one_gerna_batch_and_aligned_interface_graphs(self):
        from data_utils.packed_hybrid_batch import prepare_packed_hybrid_group
        from tests.test_interface_first_model import make_interface_item

        first = make_interface_item()
        second = make_interface_item()
        first["key"] = "first"
        second["key"] = "second"
        samples = [
            {"key": "first", "label": 1.5, "gerna": "g1", "interface": first},
            {"key": "second", "label": 2.5, "gerna": "g2", "interface": second},
        ]
        calls = []

        def collate(items):
            calls.append(list(items))
            return tuple(range(23))

        def unpack(batch, device):
            self.assertEqual(batch, tuple(range(23)))
            self.assertEqual(device, "cpu")
            return tuple(range(23))

        keys, gerna_inputs, packed, targets = prepare_packed_hybrid_group(
            samples,
            "cpu",
            collate_fn=collate,
            unpack_fn=unpack,
        )
        self.assertEqual(calls, [["g1", "g2"]])
        self.assertEqual(keys, ["first", "second"])
        self.assertEqual(gerna_inputs["RNA_repre"], 0)
        self.assertEqual(gerna_inputs["Boltz_interface"], 22)
        self.assertEqual(packed.keys, ["first", "second"])
        torch.testing.assert_close(targets, torch.tensor([1.5, 2.5]))
