import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dta.anchored_expert_fusion_v5.data import (
    EXPERT_ORDER,
    VALIDATION_SPLITS,
    AnchoredFusionDataset,
    ImmutableCacheRef,
    cache_identity,
    collate_anchored_fusion,
    fallback_predictions,
    load_immutable_v4_bundle,
)


class AnchoredExpertFusionDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache_root = self.root / "cache-root"

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _ordered_key_sha256(keys):
        payload = json.dumps(
            list(keys), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _write_cache(self, split, keys, *, upper_bound_complete=True):
        output = self.cache_root / "joined" / split
        output.mkdir(parents=True, exist_ok=True)
        count = len(keys)
        labels = np.arange(1, count + 1, dtype=np.float32)
        predictions = np.asarray(
            [[10.0 + row, 20.0 + row, 30.0 + row] for row in range(count)],
            dtype=np.float32,
        )
        available = np.asarray(
            [
                [True, row % 2 == 1, row % 2 == 0]
                for row in range(count)
            ],
            dtype=np.bool_,
        )
        if not upper_bound_complete:
            available[-1, 0] = False
        predictions[~available] = 0.0
        standardized = predictions / np.float32(10.0)

        latent_widths = (2, 1, 3)
        latents = []
        for expert_index, width in enumerate(latent_widths):
            values = np.asarray(
                [
                    [
                        float(100 * (expert_index + 1) + 10 * row + column)
                        for column in range(width)
                    ]
                    for row in range(count)
                ],
                dtype=np.float32,
            )
            values[~available[:, expert_index]] = 0.0
            latents.append(values)

        np.savez_compressed(
            output / "features.npz",
            keys=np.asarray(keys, dtype=np.str_),
            labels=labels,
            predictions=predictions,
            standardized_predictions=standardized,
            available=available,
            upper_bound_latents=latents[0],
            boltz_latents=latents[1],
            three_dprot_latents=latents[2],
        )
        (output / "normalization.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        (output / "coverage.json").write_text(
            json.dumps({"status": "complete", "split": split}),
            encoding="utf-8",
        )
        metadata = {
            "status": "complete",
            "schema_version": 1,
            "split": split,
            "count": count,
            "manifest_sha256": split + "-manifest",
            "ordered_key_sha256": self._ordered_key_sha256(keys),
            "expert_order": list(EXPERT_ORDER),
            "normalization_sha256": self._sha256(
                output / "normalization.json"
            ),
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return output

    def _rewrite_features(self, cache_dir, transform):
        with np.load(str(cache_dir / "features.npz"), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        transform(arrays)
        np.savez_compressed(cache_dir / "features.npz", **arrays)

    def test_loads_valid_v4_caches_in_requested_order_with_file_identities(self):
        train_dir = self._write_cache(
            "train", ("train-a", "train-b")
        )
        valid_dir = self._write_cache("valid", ("valid-a",))

        bundle = load_immutable_v4_bundle(
            self.cache_root,
            {"train": 2, "valid": 1},
            splits=("valid", "train"),
        )

        self.assertEqual(EXPERT_ORDER, ("upper_bound", "boltz_pairwise", "three_dprot"))
        self.assertEqual(VALIDATION_SPLITS, frozenset(("train", "valid")))
        self.assertIsInstance(bundle, dict)
        self.assertEqual(tuple(bundle), ("valid", "train"))
        self.assertEqual(bundle["valid"].cache.keys, ("valid-a",))
        self.assertEqual(
            bundle["train"].cache.keys, ("train-a", "train-b")
        )
        for split, cache_dir in (
            ("valid", valid_dir),
            ("train", train_dir),
        ):
            ref = bundle[split]
            self.assertIsInstance(ref, ImmutableCacheRef)
            self.assertEqual(
                ref.identity["split"], ref.cache.metadata["split"]
            )
            self.assertEqual(ref.identity["count"], len(ref.cache.keys))
            self.assertEqual(
                ref.identity["manifest_sha256"],
                ref.cache.metadata["manifest_sha256"],
            )
            self.assertEqual(
                ref.identity["ordered_key_sha256"],
                ref.cache.metadata["ordered_key_sha256"],
            )
            self.assertEqual(
                ref.identity["files"],
                {
                    name: self._sha256(cache_dir / name)
                    for name in (
                        "features.npz",
                        "metadata.json",
                        "coverage.json",
                        "normalization.json",
                    )
                },
            )
            self.assertEqual(
                cache_identity(cache_dir, ref.cache), ref.identity
            )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bundle["valid"].cache = bundle["train"].cache

    def test_rejects_wrong_count_and_invalid_split_sequence(self):
        self._write_cache("train", ("train-a", "train-b"))

        with self.assertRaisesRegex(ValueError, "count"):
            load_immutable_v4_bundle(
                self.cache_root, {"train": 3}, splits=("train",)
            )
        with self.assertRaisesRegex(ValueError, "split"):
            load_immutable_v4_bundle(
                self.cache_root, {"test": 1}, splits=("test",)
            )
        with self.assertRaisesRegex(ValueError, "nonempty"):
            load_immutable_v4_bundle(
                self.cache_root, {}, splits=()
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_immutable_v4_bundle(
                self.cache_root,
                {"train": 2},
                splits=("train", "train"),
            )

    def test_rejects_invalid_or_modified_features_through_v4_loader(self):
        invalid_dir = self._write_cache("train", ("train-a",))
        (invalid_dir / "features.npz").write_bytes(b"not-an-npz")
        with self.assertRaisesRegex(ValueError, "features are invalid"):
            load_immutable_v4_bundle(
                self.cache_root, {"train": 1}, splits=("train",)
            )

        modified_dir = self._write_cache("train", ("train-a",))
        self._rewrite_features(
            modified_dir,
            lambda arrays: arrays["labels"].__setitem__(0, np.nan),
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_immutable_v4_bundle(
                self.cache_root, {"train": 1}, splits=("train",)
            )

    def test_rejects_incomplete_upper_bound(self):
        self._write_cache(
            "valid", ("valid-a", "valid-b"), upper_bound_complete=False
        )

        with self.assertRaisesRegex(ValueError, "Upper-Bound.*every"):
            load_immutable_v4_bundle(
                self.cache_root, {"valid": 2}, splits=("valid",)
            )

    def test_fallback_predictions_are_float32_and_do_not_mutate_inputs(self):
        predictions = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )
        available = np.asarray(
            [[True, False, True], [True, True, False]], dtype=np.bool_
        )
        original_predictions = predictions.copy()
        original_available = available.copy()

        result = fallback_predictions(predictions, available)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(
            result, [[1.0, 1.0, 3.0], [4.0, 5.0, 4.0]]
        )
        np.testing.assert_array_equal(predictions, original_predictions)
        np.testing.assert_array_equal(available, original_available)
        self.assertFalse(np.shares_memory(result, predictions))

    def test_fallback_predictions_validate_shapes_upper_bound_and_finiteness(self):
        valid_predictions = np.ones((2, 3), dtype=np.float32)
        valid_available = np.ones((2, 3), dtype=np.bool_)
        cases = (
            (
                valid_predictions[:, :2],
                valid_available[:, :2],
                "shape",
            ),
            (
                valid_predictions,
                valid_available[:1],
                "shape",
            ),
            (
                valid_predictions,
                np.asarray(
                    [[False, True, True], [True, True, True]],
                    dtype=np.bool_,
                ),
                "Upper-Bound",
            ),
            (
                np.asarray(
                    [[1.0, np.nan, 1.0], [1.0, 1.0, 1.0]],
                    dtype=np.float32,
                ),
                valid_available,
                "finite",
            ),
        )
        for predictions, available, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fallback_predictions(predictions, available)

    def test_dataset_and_collation_carry_anchored_features(self):
        self._write_cache("train", ("train-a", "train-b"))
        ref = load_immutable_v4_bundle(
            self.cache_root, {"train": 2}, splits=("train",)
        )["train"]
        base_predictions = np.asarray(
            [101.0, 202.0], dtype=np.float64
        )
        dataset = AnchoredFusionDataset(ref.cache, base_predictions)

        self.assertEqual(len(dataset), 2)
        first = dataset[0]
        self.assertEqual(first["key"], "train-a")
        self.assertEqual(float(first["label"]), 1.0)
        self.assertEqual(float(first["base_prediction"]), 101.0)
        self.assertEqual(len(first["latents"]), 3)

        batch = collate_anchored_fusion([dataset[1], dataset[0]])

        self.assertEqual(batch["keys"], ("train-b", "train-a"))
        self.assertEqual(batch["labels"].dtype, torch.float32)
        self.assertEqual(batch["base_predictions"].dtype, torch.float32)
        self.assertEqual(batch["base_predictions"].shape, (2,))
        self.assertEqual(batch["raw_predictions"].dtype, torch.float32)
        self.assertEqual(
            batch["standardized_predictions"].dtype, torch.float32
        )
        self.assertEqual(batch["available"].dtype, torch.bool)
        self.assertEqual(len(batch["latents"]), 3)
        for latent in batch["latents"]:
            self.assertEqual(latent.dtype, torch.float32)
        torch.testing.assert_close(
            batch["base_predictions"],
            torch.tensor([202.0, 101.0], dtype=torch.float32),
        )
        torch.testing.assert_close(
            batch["raw_predictions"][0],
            torch.as_tensor(ref.cache.predictions[1]),
        )

    def test_dataset_rejects_invalid_base_shapes_and_collate_rejects_empty(self):
        self._write_cache("train", ("train-a", "train-b"))
        cache = load_immutable_v4_bundle(
            self.cache_root, {"train": 2}, splits=("train",)
        )["train"].cache
        invalid_bases = (
            np.asarray([1.0], dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
            np.asarray(1.0, dtype=np.float32),
            np.ones((2, 1, 1), dtype=np.float32),
        )

        for base_predictions in invalid_bases:
            with self.subTest(shape=base_predictions.shape):
                with self.assertRaisesRegex(ValueError, "base_predictions"):
                    AnchoredFusionDataset(cache, base_predictions)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            collate_anchored_fusion([])


if __name__ == "__main__":
    unittest.main()
