import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def make_record(key, ligand_count, protein_count, residue_count):
    value = float(len(key))
    return {
        "key": key,
        "gerna_2d_molecule": np.full(128, value, dtype=np.float32),
        "gerna_2d_protein": np.full(128, value, dtype=np.float32),
        "gerna_3d_molecule": np.full(128, value, dtype=np.float32),
        "gerna_3d_protein": np.full(128, value, dtype=np.float32),
        "gerna_2d_interaction": np.full(
            (128, 128), value, dtype=np.float32
        ),
        "gerna_3d_interaction": np.full(
            (128, 128), value, dtype=np.float32
        ),
        "gerna_head": np.full(128, value, dtype=np.float32),
        "interface_ligand_nodes": np.full(
            (ligand_count, 96), value, dtype=np.float32
        ),
        "interface_protein_nodes": np.full(
            (protein_count, 96), value, dtype=np.float32
        ),
        "interface_residue_nodes": np.full(
            (residue_count, 128), value, dtype=np.float32
        ),
        "interface_ligand_coords": np.full(
            (ligand_count, 3), value, dtype=np.float32
        ),
        "interface_protein_coords": np.full(
            (protein_count, 3), value, dtype=np.float32
        ),
        "interface_residue_coords": np.full(
            (residue_count, 3), value, dtype=np.float32
        ),
        "interface_ligand_summary": np.full(96, value, dtype=np.float32),
        "interface_protein_summary": np.full(128, value, dtype=np.float32),
        "interface_local_summary": np.full(128, value, dtype=np.float32),
        "interface_head": np.full(128, value, dtype=np.float32),
        "interface_score": np.asarray([value], dtype=np.float32),
        "interface_projection": np.full(64, value, dtype=np.float32),
        "gerna_prediction": np.asarray([value], dtype=np.float32),
        "interface_prediction": np.asarray([value], dtype=np.float32),
        "label": np.asarray([value], dtype=np.float32),
    }


class TestHybridFeatureCacheV2(unittest.TestCase):
    def test_builder_flattens_single_sample_backbone_outputs(self):
        from dta.build_hybrid_feature_cache_v2 import build_cache_record

        gerna_features = {
            "gerna_2d_molecule": torch.zeros(1, 128),
            "gerna_2d_protein": torch.zeros(1, 128),
            "gerna_3d_molecule": torch.zeros(1, 128),
            "gerna_3d_protein": torch.zeros(1, 128),
            "gerna_2d_interaction": torch.zeros(1, 128, 128),
            "gerna_3d_interaction": torch.zeros(1, 128, 128),
            "gerna_head": torch.zeros(1, 128),
        }
        interface_features = {
            "interface_ligand_nodes": torch.zeros(2, 96),
            "interface_protein_nodes": torch.zeros(3, 96),
            "interface_residue_nodes": torch.zeros(4, 128),
            "interface_ligand_coords": torch.zeros(2, 3),
            "interface_protein_coords": torch.zeros(3, 3),
            "interface_residue_coords": torch.zeros(4, 3),
            "interface_ligand_summary": torch.zeros(1, 96),
            "interface_protein_summary": torch.zeros(1, 128),
            "interface_local_summary": torch.zeros(1, 128),
            "interface_head": torch.zeros(1, 128),
            "interface_score": torch.zeros(1, 1),
            "interface_projection": torch.zeros(1, 64),
        }
        record = build_cache_record(
            key="sample",
            target=torch.tensor([7.5]),
            gerna_features=gerna_features,
            gerna_prediction=torch.tensor([[7.25]]),
            interface_features=interface_features,
            interface_prediction=torch.tensor([[7.75]]),
        )
        self.assertEqual(record["key"], "sample")
        self.assertEqual(record["gerna_head"].shape, (128,))
        self.assertEqual(record["interface_ligand_nodes"].shape, (2, 96))
        self.assertEqual(record["interface_score"].shape, (1,))
        self.assertEqual(record["label"].shape, (1,))

    def test_round_trip_preserves_order_dtypes_and_padded_batches(self):
        from data_utils.hybrid_feature_cache_v2 import (
            CachedHybridFeatureDataset,
            HybridFeatureCacheWriter,
            collate_cached_hybrid_features,
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache"
            metadata = {
                "manifest_sha256": "manifest",
                "gerna_checkpoint_sha256": "gerna",
                "interface_checkpoint_sha256": "interface",
            }
            writer = HybridFeatureCacheWriter(
                cache_path,
                expected_keys=["a", "b"],
                metadata=metadata,
            )
            writer.write(make_record("a", 2, 3, 4))
            writer.write(make_record("b", 5, 6, 7))
            writer.finalize()

            dataset = CachedHybridFeatureDataset(cache_path)
            self.assertEqual(dataset.keys, ["a", "b"])
            self.assertEqual(
                dataset.metadata["status"], "complete"
            )
            self.assertEqual(
                dataset[0]["gerna_head"].dtype, torch.float16
            )
            self.assertEqual(
                dataset[0]["label"].dtype, torch.float32
            )

            batch = collate_cached_hybrid_features(
                [dataset[0], dataset[1]]
            )
            self.assertEqual(
                tuple(batch["gerna_2d_interaction"].shape),
                (2, 128, 128),
            )
            self.assertEqual(batch["keys"], ["a", "b"])
            self.assertEqual(
                batch["interface_ligand_nodes"].shape[0], 2
            )
            self.assertEqual(
                batch["interface_ligand_mask"].dtype, torch.bool
            )
            self.assertFalse(
                batch["interface_ligand_mask"][0, 2:].any()
            )
            self.assertTrue(
                batch["interface_ligand_mask"][1, :5].all()
            )

    def test_writer_rejects_duplicate_nonfinite_and_shape_mismatch(self):
        from data_utils.hybrid_feature_cache_v2 import (
            HybridFeatureCacheWriter,
        )

        with tempfile.TemporaryDirectory() as directory:
            writer = HybridFeatureCacheWriter(
                Path(directory) / "cache",
                expected_keys=["a", "b"],
                metadata={"manifest_sha256": "manifest"},
            )
            writer.write(make_record("a", 2, 3, 4))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                writer.write(make_record("a", 2, 3, 4))

            nonfinite = make_record("b", 2, 3, 4)
            nonfinite["gerna_head"][0] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                writer.write(nonfinite)

            mismatched = make_record("b", 2, 3, 4)
            mismatched["interface_ligand_coords"] = np.zeros(
                (3, 3), dtype=np.float32
            )
            with self.assertRaisesRegex(ValueError, "align"):
                writer.write(mismatched)
            writer.close()

    def test_empty_local_protein_nodes_are_cacheable(self):
        from data_utils.hybrid_feature_cache_v2 import (
            CachedHybridFeatureDataset,
            HybridFeatureCacheWriter,
            collate_cached_hybrid_features,
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache"
            record = make_record("a", 2, 1, 4)
            record["interface_protein_nodes"] = np.zeros(
                (0, 96), dtype=np.float32
            )
            record["interface_protein_coords"] = np.zeros(
                (0, 3), dtype=np.float32
            )
            with HybridFeatureCacheWriter(
                cache_path,
                expected_keys=["a"],
                metadata={"manifest_sha256": "manifest"},
            ) as writer:
                writer.write(record)
                writer.finalize()
            batch = collate_cached_hybrid_features(
                [CachedHybridFeatureDataset(cache_path)[0]]
            )
            self.assertEqual(
                tuple(batch["interface_protein_nodes"].shape),
                (1, 0, 96),
            )
            self.assertEqual(
                tuple(batch["interface_protein_mask"].shape), (1, 0)
            )

    def test_incomplete_and_provenance_mismatch_cannot_be_reused(self):
        from data_utils.hybrid_feature_cache_v2 import (
            HybridFeatureCacheWriter,
            cache_matches_provenance,
            read_cache_metadata,
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache"
            metadata = {
                "manifest_sha256": "manifest-a",
                "gerna_checkpoint_sha256": "gerna",
                "interface_checkpoint_sha256": "interface",
            }
            writer = HybridFeatureCacheWriter(
                cache_path, expected_keys=["a", "b"], metadata=metadata
            )
            writer.write(make_record("a", 2, 3, 4))
            writer.close()
            incomplete = read_cache_metadata(cache_path)
            self.assertEqual(incomplete["status"], "incomplete")
            self.assertFalse(cache_matches_provenance(incomplete, metadata))

            resumed = HybridFeatureCacheWriter(
                cache_path,
                expected_keys=["a", "b"],
                metadata=metadata,
                resume=True,
            )
            self.assertEqual(resumed.missing_keys, ["b"])
            resumed.write(make_record("b", 2, 3, 4))
            resumed.finalize()
            complete = read_cache_metadata(cache_path)
            self.assertTrue(cache_matches_provenance(complete, metadata))
            self.assertFalse(
                cache_matches_provenance(
                    complete,
                    {**metadata, "manifest_sha256": "manifest-b"},
                )
            )


if __name__ == "__main__":
    unittest.main()
