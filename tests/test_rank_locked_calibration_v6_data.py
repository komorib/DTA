import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dta.anchored_expert_fusion_v5.training import (
    StaticSpec,
    static_predictions,
)
from dta.rank_fusion_v4.cache import FusionCache
from dta.rank_locked_calibration_v6 import (
    CANDIDATE_NAMES,
    EXPERT_ORDER,
    FORMAL_COUNTS,
    TEST_GATE,
    VALIDATION_GATE,
    VERSION,
)
from dta.rank_locked_calibration_v6.data import (
    ImmutableCacheRef,
    RankLockedCalibrationDataset,
    collate_rank_locked_calibration,
    load_validation_caches,
    reconstruct_v5_anchor,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(path, payload):
    Path(path).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _cache(split, *, count=3):
    keys = tuple("%s_%d_protein" % (split, index) for index in range(count))
    available = np.ones((count, 3), dtype=np.bool_)
    predictions = np.asarray(
        [[5.0 + index, 4.0 + index, 6.0 + index] for index in range(count)],
        dtype=np.float32,
    )
    return FusionCache(
        keys=keys,
        labels=np.linspace(5.5, 7.5, count, dtype=np.float32),
        predictions=predictions,
        standardized_predictions=predictions - 5.0,
        available=available,
        upper_bound_latents=np.full((count, 4), 1.0, dtype=np.float32),
        boltz_latents=np.full((count, 2), 2.0, dtype=np.float32),
        three_dprot_latents=np.full((count, 3), 3.0, dtype=np.float32),
        metadata={
            "split": split,
            "count": count,
            "manifest_sha256": ("1" if split == "train" else "2") * 64,
            "ordered_key_sha256": ("3" if split == "train" else "4") * 64,
        },
    )


class RankLockedV6DataTest(unittest.TestCase):
    def test_public_contract(self):
        self.assertEqual(VERSION, "rank-locked-calibration-v6")
        self.assertEqual(
            EXPERT_ORDER,
            ("upper_bound", "boltz_pairwise", "three_dprot"),
        )
        self.assertEqual(
            FORMAL_COUNTS,
            {"train": 11037, "valid": 2340, "test": 4751},
        )
        self.assertEqual(CANDIDATE_NAMES[-1], "conditional_ensemble")
        self.assertEqual(
            VALIDATION_GATE, {"CI": 0.713374, "RMSE": 1.35, "MAE": 1.05}
        )
        self.assertEqual(
            TEST_GATE, {"CI": 0.708700, "RMSE": 1.497357, "MAE": 1.174794}
        )

    def test_load_validation_caches_hashes_files_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            caches = {"train": _cache("train"), "valid": _cache("valid")}
            expected_hashes = {}
            for split in caches:
                directory = root / "joined" / split
                directory.mkdir(parents=True)
                expected_hashes[split] = {}
                for index, name in enumerate(
                    ("features.npz", "metadata.json", "coverage.json", "normalization.json")
                ):
                    path = directory / name
                    path.write_bytes((split + name + str(index)).encode("ascii"))
                    expected_hashes[split][name] = _sha256(path)

            def loader(path, *, expected_split, expected_manifest_sha256=None):
                self.assertEqual(Path(path), root.resolve() / "joined" / expected_split)
                expected_manifest = caches[expected_split].metadata["manifest_sha256"]
                self.assertEqual(expected_manifest_sha256, expected_manifest)
                return caches[expected_split]

            with mock.patch(
                "dta.rank_locked_calibration_v6.data.load_fusion_cache",
                side_effect=loader,
            ) as patched:
                references = load_validation_caches(
                    root,
                    {"train": 3, "valid": 3},
                    expected_manifest_sha256_by_split={
                        split: cache.metadata["manifest_sha256"]
                        for split, cache in caches.items()
                    },
                )

            self.assertEqual(patched.call_count, 2)
            self.assertEqual(dict(references["train"].file_sha256), expected_hashes["train"])
            with self.assertRaises(TypeError):
                references["train"].file_sha256["features.npz"] = "0" * 64

    def test_validation_loader_rejects_test_duplicates_and_bad_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "train and valid"):
                load_validation_caches(temporary, {"test": 1}, ("test",))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_validation_caches(
                    temporary, {"train": 1}, ("train", "train")
                )
            with self.assertRaisesRegex(ValueError, "expected count"):
                load_validation_caches(temporary, {}, ("train",))
            with self.assertRaisesRegex(ValueError, "expected count"):
                load_validation_caches(
                    temporary, {"train": True}, ("train",)
                )

    def test_loader_rejects_bad_availability_upper_bound_and_nonfinite(self):
        cases = []
        missing = replace(_cache("train"), available=None)
        cases.append((missing, "available"))
        wrong_dtype = replace(
            _cache("train"), available=np.ones((3, 3), dtype=np.int64)
        )
        cases.append((wrong_dtype, "available"))
        unavailable = _cache("train")
        unavailable.available[1, 0] = False
        cases.append((unavailable, "Upper-Bound"))
        for field in (
            "labels",
            "predictions",
            "standardized_predictions",
            "upper_bound_latents",
            "boltz_latents",
            "three_dprot_latents",
        ):
            cache = _cache("train")
            values = getattr(cache, field)
            values.flat[0] = np.nan
            cases.append((cache, "nonfinite"))

        for cache, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary) / "joined" / "train"
                    directory.mkdir(parents=True)
                    for name in (
                        "features.npz",
                        "metadata.json",
                        "coverage.json",
                        "normalization.json",
                    ):
                        (directory / name).write_bytes(name.encode("ascii"))
                    with mock.patch(
                        "dta.rank_locked_calibration_v6.data.load_fusion_cache",
                        return_value=cache,
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            load_validation_caches(
                                temporary, {"train": 3}, ("train",)
                            )


class RankLockedV6AnchorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.package = self.root / "validation_package"
        self.package.mkdir()
        self.train_ref = self._reference("train")
        self.valid_ref = self._reference("valid")
        self.references = {"train": self.train_ref, "valid": self.valid_ref}
        self.spec_payload = {
            "family": "rank",
            "deploy_weights": [0.5, 0.3, 0.2],
            "rank_slope": 5.0,
            "rank_intercept": 3.0,
            "cross_fit_report_hash": "a" * 64,
            "train_cache_identity": self.train_ref.identity,
            "valid_cache_identity": self.valid_ref.identity,
        }
        self.spec = StaticSpec.from_dict(self.spec_payload)
        _canonical_json(self.package / "static_specs.json", {"rank": self.spec_payload})
        self.checkpoint_path = self.root / "runs" / "checkpoint_best.pt"
        self.prediction_path = self.root / "runs" / "predictions_best.npz"
        self.checkpoint_path.parent.mkdir()
        self.checkpoint_path.write_bytes(b"checkpoint")
        self.stored_selected_predictions = static_predictions(
            self.valid_ref.cache, self.spec
        )
        self._write_predictions(
            self.valid_ref.cache.keys,
            self.valid_ref.cache.labels,
            self.stored_selected_predictions,
        )
        self._write_references()
        self.selection = {
            "selected_family": "rank",
            "selected_name": "residual_rank_10_1",
            "selected_type": "residual",
            "status": "complete",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _reference(self, split):
        directory = self.root / "joined" / split
        directory.mkdir(parents=True)
        hashes = {}
        for name in (
            "features.npz",
            "metadata.json",
            "coverage.json",
            "normalization.json",
        ):
            path = directory / name
            path.write_bytes((split + name).encode("ascii"))
            hashes[name] = _sha256(path)
        return ImmutableCacheRef(
            split=split,
            directory=directory,
            cache=_cache(split),
            file_sha256=hashes,
        )

    def _write_predictions(self, keys, labels, predictions):
        np.savez_compressed(
            self.prediction_path,
            keys=np.asarray(keys),
            labels=np.asarray(labels, dtype=np.float32),
            predictions=np.asarray(predictions, dtype=np.float32),
            epoch=np.asarray(-1, dtype=np.int64),
            variant=np.asarray("best"),
        )

    def _write_references(self):
        _canonical_json(
            self.package / "selected_references.json",
            {
                "checkpoint_path": str(self.checkpoint_path.relative_to(self.root)),
                "checkpoint_sha256": _sha256(self.checkpoint_path),
                "name": "residual_rank_10_1",
                "prediction_path": str(self.prediction_path.relative_to(self.root)),
                "prediction_sha256": _sha256(self.prediction_path),
            },
        )

    def _reconstruct(self, *, selection=None, checkpoint=None):
        if selection is None:
            selection = self.selection
        if checkpoint is None:
            checkpoint = {"epoch": -1}
        with mock.patch(
            "dta.rank_locked_calibration_v6.data.verify_validation_seal",
            return_value=selection,
        ), mock.patch(
            "dta.rank_locked_calibration_v6.data.torch.load",
            return_value=checkpoint,
        ):
            return reconstruct_v5_anchor(
                self.package, self.root, self.references
            )

    def test_reconstruct_anchor_requires_rank_epoch_minus_one(self):
        anchor = self._reconstruct()
        self.assertEqual(anchor.family, "rank")
        self.assertEqual(anchor.selected_checkpoint_epoch, -1)
        np.testing.assert_allclose(
            anchor.valid_predictions,
            self.stored_selected_predictions,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            anchor.train_predictions,
            static_predictions(self.train_ref.cache, self.spec),
            rtol=0.0,
            atol=1e-10,
        )

    def test_rejects_modified_seal_wrong_family_and_wrong_epoch(self):
        with mock.patch(
            "dta.rank_locked_calibration_v6.data.verify_validation_seal",
            side_effect=ValueError("selection seal does not match"),
        ):
            with self.assertRaisesRegex(ValueError, "seal"):
                reconstruct_v5_anchor(self.package, self.root, self.references)
        changed = dict(self.selection, selected_family="raw")
        with self.assertRaisesRegex(ValueError, "rank"):
            self._reconstruct(selection=changed)
        with self.assertRaisesRegex(ValueError, "epoch -1"):
            self._reconstruct(checkpoint={"epoch": 0})

    def test_rejects_symlinked_validation_package(self):
        package_link = self.root / "validation_package_link"
        package_link.symlink_to(self.package, target_is_directory=True)
        with mock.patch(
            "dta.rank_locked_calibration_v6.data.verify_validation_seal",
            return_value=self.selection,
        ), mock.patch(
            "dta.rank_locked_calibration_v6.data.torch.load",
            return_value={"epoch": -1},
        ):
            with self.assertRaisesRegex(ValueError, "symlink"):
                reconstruct_v5_anchor(
                    package_link, self.root, self.references
                )

    def test_rejects_wrong_ordered_keys_and_labels(self):
        self._write_predictions(
            tuple(reversed(self.valid_ref.cache.keys)),
            self.valid_ref.cache.labels,
            self.stored_selected_predictions,
        )
        self._write_references()
        with self.assertRaisesRegex(ValueError, "ordered keys"):
            self._reconstruct()

        labels = self.valid_ref.cache.labels.copy()
        labels[0] += 1.0
        self._write_predictions(
            self.valid_ref.cache.keys,
            labels,
            self.stored_selected_predictions,
        )
        self._write_references()
        with self.assertRaisesRegex(ValueError, "labels"):
            self._reconstruct()

    def test_rejects_noncanonical_metadata_and_prediction_mismatch(self):
        payload = json.loads(
            (self.package / "static_specs.json").read_text(encoding="utf-8")
        )
        (self.package / "static_specs.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            self._reconstruct()

        _canonical_json(self.package / "static_specs.json", {"rank": self.spec_payload})
        changed = self.stored_selected_predictions.copy()
        changed[0] += 0.01
        self._write_predictions(
            self.valid_ref.cache.keys,
            self.valid_ref.cache.labels,
            changed,
        )
        self._write_references()
        with self.assertRaisesRegex(ValueError, "static anchor"):
            self._reconstruct()

    def test_dataset_and_collator_expose_all_frozen_inputs(self):
        base = np.asarray([6.0, 7.0, 8.0], dtype=np.float64)
        dataset = RankLockedCalibrationDataset(self.train_ref.cache, base)
        batch = collate_rank_locked_calibration([dataset[0], dataset[2]])
        self.assertEqual(batch["keys"], ("train_0_protein", "train_2_protein"))
        self.assertEqual(tuple(batch["raw_predictions"].shape), (2, 3))
        self.assertEqual(tuple(batch["standardized_predictions"].shape), (2, 3))
        self.assertEqual(tuple(batch["available"].shape), (2, 3))
        self.assertEqual(
            [tuple(values.shape) for values in batch["latents"]],
            [(2, 4), (2, 2), (2, 3)],
        )
        self.assertEqual(batch["base_predictions"].dtype, torch.float32)
        self.assertEqual(batch["available"].dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
