import unittest

import numpy as np
import torch

from utils.experiment_integrity import hash_ordered_keys

from data_utils.hybrid_affinity_dataset import (
    HybridAffinityDataset,
    LabelSpreadBatchSampler,
    preflight_hybrid_dataset,
    collate_hybrid_group,
    unique_key_index,
    validate_inventory,
)


class FakeDataset:
    def __init__(self, samples, values):
        self.samples = list(samples)
        self.values = list(values)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.values[index]


def manifest_row(key, label):
    return {"lmdb_key": key, "label": label}


def gerna_item(key, label):
    return ("gerna-" + key,) + (None,) * 8 + (label,)


def interface_item(key, label):
    return {
        "key": key,
        "label": torch.tensor(label),
        "payload": "interface-" + key,
    }


def valid_inputs():
    manifest = [manifest_row("a", 1.0), manifest_row("b", 2.0)]
    gerna = FakeDataset(
        [manifest_row("b", 2.0), manifest_row("a", 1.0)],
        [gerna_item("b", 2.0), gerna_item("a", 1.0)],
    )
    interface = FakeDataset(
        [manifest_row("a", 1.0), manifest_row("b", 2.0)],
        [interface_item("a", 1.0), interface_item("b", 2.0)],
    )
    return manifest, gerna, interface


class TestHybridAffinityDataset(unittest.TestCase):
    def test_unique_key_index_maps_physical_positions(self):
        samples = [manifest_row("b", 2.0), manifest_row("a", 1.0)]

        self.assertEqual(
            unique_key_index(samples, "GerNA"),
            {"b": 0, "a": 1},
        )

    def test_validate_inventory_preserves_manifest_order(self):
        manifest, gerna, interface = valid_inputs()

        keys = validate_inventory(
            manifest,
            unique_key_index(gerna.samples, "GerNA"),
            unique_key_index(interface.samples, "interface"),
        )

        self.assertEqual(keys, ["a", "b"])

    def test_join_uses_lmdb_key_and_emits_manifest_order(self):
        manifest, gerna, interface = valid_inputs()

        dataset = HybridAffinityDataset(manifest, gerna, interface)
        items = [dataset[index] for index in range(len(dataset))]

        self.assertEqual([item["key"] for item in items], ["a", "b"])
        self.assertEqual(
            [item["gerna"][0] for item in items],
            ["gerna-a", "gerna-b"],
        )
        self.assertEqual(
            [item["interface"]["payload"] for item in items],
            ["interface-a", "interface-b"],
        )
        self.assertEqual(
            set(items[0]),
            {"key", "label", "gerna", "interface"},
        )

    def test_duplicate_manifest_key_is_rejected_with_key(self):
        manifest = [manifest_row("a", 1.0), manifest_row("a", 1.0)]
        gerna = FakeDataset([manifest_row("a", 1.0)], [gerna_item("a", 1.0)])
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "a"):
            HybridAffinityDataset(manifest, gerna, interface)

    def test_duplicate_gerna_key_is_rejected_with_key(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset(
            [manifest_row("a", 1.0), manifest_row("a", 1.0)],
            [gerna_item("a", 1.0), gerna_item("a", 1.0)],
        )
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "a"):
            HybridAffinityDataset(manifest, gerna, interface)

    def test_duplicate_interface_key_is_rejected_with_key(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset([manifest_row("a", 1.0)], [gerna_item("a", 1.0)])
        interface = FakeDataset(
            [manifest_row("a", 1.0), manifest_row("a", 1.0)],
            [interface_item("a", 1.0), interface_item("a", 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "a"):
            HybridAffinityDataset(manifest, gerna, interface)

    def test_missing_gerna_key_is_rejected_with_key(self):
        manifest, _, interface = valid_inputs()
        gerna = FakeDataset(
            [manifest_row("a", 1.0)],
            [gerna_item("a", 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "b"):
            HybridAffinityDataset(manifest, gerna, interface)

    def test_missing_interface_key_is_rejected_with_key(self):
        manifest, gerna, _ = valid_inputs()
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "b"):
            HybridAffinityDataset(manifest, gerna, interface)

    def test_nonfinite_manifest_label_is_rejected_with_key(self):
        manifest = [manifest_row("a", float("nan"))]
        gerna = FakeDataset([manifest_row("a", 1.0)], [gerna_item("a", 1.0)])
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "a"):
            HybridAffinityDataset(manifest, gerna, interface)

    def test_nonfinite_gerna_label_is_rejected_with_key(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset(
            [manifest_row("a", 1.0)],
            [gerna_item("a", float("nan"))],
        )
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "a"):
            HybridAffinityDataset(manifest, gerna, interface)[0]

    def test_nonfinite_interface_label_is_rejected_with_key(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset([manifest_row("a", 1.0)], [gerna_item("a", 1.0)])
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", float("inf"))],
        )

        with self.assertRaisesRegex(ValueError, "a"):
            HybridAffinityDataset(manifest, gerna, interface)[0]

    def test_gerna_label_mismatch_above_default_tolerance_has_key(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset(
            [manifest_row("a", 1.0)],
            [gerna_item("a", 1.000002)],
        )
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 1.0)],
        )
        dataset = HybridAffinityDataset(manifest, gerna, interface)

        with self.assertRaisesRegex(ValueError, "a"):
            dataset[0]

    def test_interface_scalar_tensor_label_mismatch_has_key(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset([manifest_row("a", 1.0)], [gerna_item("a", 1.0)])
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 1.000002)],
        )
        dataset = HybridAffinityDataset(manifest, gerna, interface)

        with self.assertRaisesRegex(ValueError, "a"):
            dataset[0]

    def test_interface_returned_key_mismatch_has_expected_key(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset([manifest_row("a", 1.0)], [gerna_item("a", 1.0)])
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("wrong", 1.0)],
        )
        dataset = HybridAffinityDataset(manifest, gerna, interface)

        with self.assertRaisesRegex(ValueError, "a"):
            dataset[0]

    def test_canonical_labels_preserve_float64_precision(self):
        precise_label = 1.00000005
        manifest = [manifest_row("a", precise_label)]
        gerna = FakeDataset(
            [manifest_row("a", precise_label)],
            [gerna_item("a", precise_label)],
        )
        interface = FakeDataset(
            [manifest_row("a", precise_label)],
            [interface_item("a", precise_label)],
        )

        dataset = HybridAffinityDataset(manifest, gerna, interface)

        self.assertEqual(dataset.labels.dtype, np.dtype(np.float64))
        self.assertEqual(dataset[0]["label"], precise_label)

    def test_default_tolerance_accepts_label_difference_below_one_e_minus_six(self):
        manifest = [manifest_row("a", 1.0)]
        gerna = FakeDataset(
            [manifest_row("a", 1.0)],
            [gerna_item("a", 1.0000005)],
        )
        interface = FakeDataset(
            [manifest_row("a", 1.0)],
            [interface_item("a", 0.9999995)],
        )

        item = HybridAffinityDataset(manifest, gerna, interface)[0]

        self.assertEqual(item["key"], "a")
        self.assertEqual(item["label"], 1.0)

    def test_collate_hybrid_group_returns_a_list_in_input_order(self):
        first = {"key": "a"}
        second = {"key": "b"}

        result = collate_hybrid_group((first, second))

        self.assertIsInstance(result, list)
        self.assertEqual(result, [first, second])
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)


class FakeGraph:
    def __init__(self, node_features, edge_features=None):
        self.ndata = {"x": node_features}
        self.edata = {} if edge_features is None else {"edge_attr": edge_features}

    def num_nodes(self):
        return int(self.ndata["x"].shape[0])


class FakeOpaqueGraph:
    def num_nodes(self):
        return 1


class FakePreflightDataset:
    def __init__(self, keys, values):
        self.keys = list(keys)
        self.values = list(values)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.values[index]


class TestLabelSpreadBatchSampler(unittest.TestCase):
    def test_sampler_covers_every_index_once_and_reports_length(self):
        sampler = LabelSpreadBatchSampler(
            labels=np.arange(10, dtype=np.float32),
            group_size=4,
            seed=7,
            min_gap=0.1,
        )
        groups = list(sampler)

        self.assertEqual(len(sampler), 3)
        self.assertTrue(all(1 <= len(group) <= 4 for group in groups))
        self.assertEqual(sorted(index for group in groups for index in group), list(range(10)))

    def test_sampler_is_deterministic_per_epoch_and_changes_across_epochs(self):
        sampler = LabelSpreadBatchSampler(
            labels=np.arange(12, dtype=np.float32),
            group_size=4,
            seed=11,
            min_gap=0.1,
        )
        epoch_zero = list(sampler)
        self.assertEqual(epoch_zero, list(sampler))

        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, list(sampler))

    def test_sampler_mixes_low_and_high_label_quantiles(self):
        labels = np.arange(8, dtype=np.float32)
        sampler = LabelSpreadBatchSampler(
            labels=labels,
            group_size=4,
            seed=42,
            min_gap=0.1,
        )

        for group in sampler:
            if len(group) == 4:
                values = labels[group]
                self.assertGreaterEqual(float(values.max() - values.min()), 3.0)

    def test_impossible_min_gap_preserves_every_sample(self):
        sampler = LabelSpreadBatchSampler(
            labels=np.arange(7, dtype=np.float64),
            group_size=3,
            seed=5,
            min_gap=100.0,
        )

        groups = list(sampler)
        expected_order = np.arange(7)
        np.random.RandomState(5).shuffle(expected_order)
        expected_groups = [
            expected_order[start : start + 3].tolist()
            for start in range(0, len(expected_order), 3)
        ]

        self.assertEqual(groups, expected_groups)
        self.assertEqual(
            sorted(index for group in groups for index in group),
            list(range(7)),
        )

    def test_sampler_rejects_invalid_arguments(self):
        invalid = (
            ([], 2, 1, 0.1),
            ([0.0, float("nan")], 2, 1, 0.1),
            ([0.0, 1.0], 1, 1, 0.1),
            ([0.0, 1.0], 2.0, 1, 0.1),
            ([0.0, 1.0], 2.5, 1, 0.1),
            ([0.0, 1.0], None, 1, 0.1),
            ([0.0, 1.0], float("inf"), 1, 0.1),
            ([0.0, 1.0], 2, True, 0.1),
            ([0.0, 1.0], 2, 1.5, 0.1),
            ([0.0, 1.0], 2, -1, 0.1),
            ([0.0, 1.0], 2, 2 ** 32, 0.1),
            ([0.0, 1.0], 2, 1, -0.1),
            ([0.0, 1.0], 2, 1, float("inf")),
        )
        for labels, group_size, seed, min_gap in invalid:
            with self.subTest(
                labels=labels,
                group_size=group_size,
                seed=seed,
                min_gap=min_gap,
            ):
                with self.assertRaises(ValueError):
                    LabelSpreadBatchSampler(
                        labels=labels,
                        group_size=group_size,
                        seed=seed,
                        min_gap=min_gap,
                    )

    def test_sampler_rejects_invalid_epoch(self):
        sampler = LabelSpreadBatchSampler(
            labels=[0.0, 1.0],
            group_size=2,
            seed=1,
            min_gap=0.1,
        )

        for epoch in (True, 1.5, -1, None, float("inf")):
            with self.subTest(epoch=epoch):
                with self.assertRaises(ValueError):
                    sampler.set_epoch(epoch)


class TestHybridDatasetPreflight(unittest.TestCase):
    @staticmethod
    def item(size, edge_index=None):
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return {
            "label": 1.0,
            "interface": {
                "ligand_atom_features": torch.ones((size, 3)),
                "ligand_coords": torch.ones((size, 3)),
                "protein_coords": torch.ones((size + 1, 3)),
                "residue_esm": torch.ones((size + 1, 8)),
                "residue_ca_coords": torch.ones((size + 1, 3)),
                "edge_index": edge_index,
            },
        }

    def test_preflight_rejects_nested_nonfinite_with_key_and_path(self):
        item = self.item(2)
        item["interface"]["nested"] = {
            "bad": torch.tensor([float("nan")])
        }
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(
            ValueError, "sample-a.*sample.interface.nested.bad"
        ):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_required_empty_dense_field(self):
        item = self.item(2)
        item["interface"]["ligand_atom_features"] = torch.empty((0, 3))
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(
            ValueError, "sample-a.*ligand_atom_features"
        ):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_missing_required_interface_field(self):
        item = self.item(2)
        del item["interface"]["residue_esm"]
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(ValueError, "sample-a.*residue_esm"):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_required_empty_list(self):
        item = self.item(2)
        item["interface"]["protein_coords"] = []
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(ValueError, "sample-a.*protein_coords"):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_required_nonnumeric_value(self):
        item = self.item(2)
        item["interface"]["residue_esm"] = None
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(ValueError, "sample-a.*residue_esm"):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_nonfinite_numpy_array_with_path(self):
        item = self.item(2)
        item["interface"]["ligand_coords"] = np.array(
            [[0.0, float("inf"), 0.0]]
        )
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(
            ValueError, "sample-a.*sample.interface.ligand_coords"
        ):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_nested_nonfinite_scalar_with_path(self):
        item = self.item(2)
        item["interface"]["nested_scalar"] = {"bad": float("nan")}
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(
            ValueError, "sample-a.*sample.interface.nested_scalar.bad"
        ):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_nonfinite_dgl_style_graph_storage(self):
        item = self.item(2)
        item["gerna_graph"] = FakeGraph(torch.tensor([[float("nan")]]))
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(
            ValueError, "sample-a.*sample.gerna_graph.ndata.x"
        ):
            preflight_hybrid_dataset(dataset)

    def test_preflight_rejects_opaque_graph_like_object(self):
        item = self.item(2)
        item["opaque_graph"] = FakeOpaqueGraph()
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(
            ValueError, "sample-a.*sample.opaque_graph"
        ):
            preflight_hybrid_dataset(dataset)

    def test_preflight_required_names_are_scoped_to_interface(self):
        item = self.item(2)
        item["optional_metadata"] = {"ligand_coords": []}
        dataset = FakePreflightDataset(["sample-a"], [item])

        report = preflight_hybrid_dataset(dataset)

        self.assertEqual(report["sample_count"], 1)

    def test_preflight_handles_finite_complex_and_large_integer_scalars(self):
        item = self.item(2)
        item["optional_metadata"] = {
            "complex": 1.0 + 2.0j,
            "large_integer": 10 ** 1000,
        }
        dataset = FakePreflightDataset(["sample-a"], [item])

        report = preflight_hybrid_dataset(dataset)

        self.assertEqual(report["sample_count"], 1)

    def test_preflight_rejects_nonfinite_complex_with_path(self):
        item = self.item(2)
        item["optional_metadata"] = {"complex": complex(float("inf"), 1.0)}
        dataset = FakePreflightDataset(["sample-a"], [item])

        with self.assertRaisesRegex(
            ValueError, "sample-a.*sample.optional_metadata.complex"
        ):
            preflight_hybrid_dataset(dataset)

    def test_preflight_accepts_optional_empty_edge_index_and_hashes_keys(self):
        dataset = FakePreflightDataset(
            ["a", "b"],
            [self.item(2), self.item(3)],
        )

        report = preflight_hybrid_dataset(dataset)

        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["ordered_key_sha256"], hash_ordered_keys(["a", "b"]))

    def test_preflight_ranks_complexity_with_stable_index_tie_break(self):
        dataset = FakePreflightDataset(
            ["a", "b", "c"],
            [self.item(3), self.item(3), self.item(5)],
        )

        report = preflight_hybrid_dataset(dataset, largest_count=3)

        self.assertEqual(report["largest_indices"], [2, 0, 1])
        self.assertEqual(report["largest_keys"], ["c", "a", "b"])

    def test_preflight_rejects_invalid_largest_count(self):
        dataset = FakePreflightDataset(["a"], [self.item(1)])
        for value in (0, -1, 1.0, 1.5, True, None, float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    preflight_hybrid_dataset(dataset, largest_count=value)


if __name__ == "__main__":
    unittest.main()
