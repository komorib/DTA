import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import lmdb
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_local_interface_cache import (  # noqa: E402
    make_chain,
    make_ligand,
    make_payload,
    write_test_lmdb,
    write_test_manifest,
)
from tests.test_local_interface_decoys import (  # noqa: E402
    make_valid_decoy_set,
)


def write_pickle_lmdb(path, records):
    environment = lmdb.open(
        str(path),
        map_size=16 * 1024 * 1024,
        subdir=True,
    )
    try:
        with environment.begin(write=True) as transaction:
            for key, value in records.items():
                transaction.put(
                    str(key).encode("utf-8"),
                    pickle.dumps(
                        value, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
    finally:
        environment.close()


def make_multichain_payload():
    chain_a = make_chain(
        "AC",
        [[2.0, 0.0, 0.0], [2.2, 0.0, 0.0]],
        residue_indices=[0, 1],
        atom_names=["CA", "CA"],
    )
    chain_b = make_chain(
        "G",
        [[2.4, 0.0, 0.0]],
        residue_indices=[0],
        atom_names=["CA"],
    )
    chain_a["esm2_embedding"] = np.stack(
        [
            np.full(640, 10.0, dtype=np.float32),
            np.full(640, 20.0, dtype=np.float32),
        ]
    )
    chain_b["esm2_embedding"] = np.full(
        (1, 640), 30.0, dtype=np.float32
    )
    return {
        "protein_features": {"B": chain_b, "A": chain_a},
        "ligand_features": make_ligand(),
    }


def make_native_record(payload, key):
    from data_utils.local_interface import extract_local_interface_record

    return extract_local_interface_record(payload, key, "train")


def make_decoy_set(key):
    decoy_set = make_valid_decoy_set()
    decoy_set["key"] = key
    return decoy_set


def make_v2_decoy_set(
    native_record,
    seed=42,
    random_complete=True,
    matched_complete=True,
):
    from tests.test_interface_decoy_audit import (
        diagnostic_decoy_set,
    )

    return diagnostic_decoy_set(
        native_record,
        seed=seed,
        random_complete=random_complete,
        matched_complete=matched_complete,
    )


class TestNativeInterfaceDataset(unittest.TestCase):
    def test_joins_exact_keys_and_preserves_residue_parent_indexing(self):
        from data_utils.local_interface_dataset import (
            NativeInterfaceDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            payload = make_multichain_payload()
            other = make_payload()
            write_test_lmdb(
                source_path, {"other": other, "target": payload}
            )
            write_pickle_lmdb(
                native_path,
                {
                    "target": make_native_record(payload, "target"),
                    "other": make_native_record(other, "other"),
                },
            )
            write_test_manifest(
                manifest_path, source_path, ["target", "other"]
            )

            dataset = NativeInterfaceDataset(
                manifest_path, native_path
            )
            try:
                item = dataset[0]
            finally:
                dataset.close()

            self.assertEqual(item["key"], "target")
            self.assertEqual(item["label"].dtype, torch.float32)
            self.assertEqual(item["residue_esm"].shape, (3, 640))
            self.assertEqual(
                item["residue_esm"][:, 0].tolist(),
                [10.0, 20.0, 30.0],
            )
            parents = item["protein_parent_residue"]
            atom_context = item["residue_esm"][parents]
            self.assertEqual(
                atom_context[:, 0].tolist(), [10.0, 20.0, 30.0]
            )
            self.assertEqual(
                item["residue_ca_coords"].shape, (3, 3)
            )
            self.assertEqual(
                item["ligand_atom_features"].dtype, torch.int64
            )
            self.assertEqual(
                item["protein_is_backbone"].dtype, torch.bool
            )

    def test_missing_or_mismatched_native_key_is_a_hard_error(self):
        from data_utils.local_interface_dataset import (
            NativeInterfaceDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            payload = make_payload()
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(native_path, {})
            write_test_manifest(manifest_path, source_path, ["a"])
            dataset = NativeInterfaceDataset(
                manifest_path, native_path
            )
            try:
                with self.assertRaisesRegex(
                    KeyError, "native interface.*a"
                ):
                    dataset[0]
            finally:
                dataset.close()

            bad_record = make_native_record(payload, "wrong")
            write_pickle_lmdb(native_path, {"a": bad_record})
            dataset = NativeInterfaceDataset(
                manifest_path, native_path
            )
            try:
                with self.assertRaisesRegex(
                    ValueError, "record key.*a"
                ):
                    dataset[0]
            finally:
                dataset.close()

    def test_rejects_misaligned_residue_esm(self):
        from data_utils.local_interface_dataset import (
            NativeInterfaceDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            payload = make_payload()
            payload["protein_features"]["A"][
                "esm2_embedding"
            ] = np.zeros((2, 640), dtype=np.float32)
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(
                native_path,
                {"a": make_native_record(make_payload(), "a")},
            )
            write_test_manifest(manifest_path, source_path, ["a"])
            dataset = NativeInterfaceDataset(
                manifest_path, native_path
            )
            try:
                with self.assertRaisesRegex(
                    ValueError, "ESM.*residue"
                ):
                    dataset[0]
            finally:
                dataset.close()


class TestPosePretrainDataset(unittest.TestCase):
    def test_lmdb_inventory_ignores_all_v2_reserved_keys(self):
        from data_utils.local_interface_dataset import _lmdb_keys

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decoys.lmdb"
            write_pickle_lmdb(
                path,
                {
                    "a": {"key": "a"},
                    "__metadata__": {"status": "complete"},
                    "__complete__": {"status": "complete"},
                    "__record_sha256__:a": "digest",
                },
            )

            self.assertEqual(_lmdb_keys(path), {"a"})

    def test_rejects_non_train_manifest(self):
        from data_utils.local_interface_dataset import (
            PosePretrainDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            decoy_path = root / "decoys.lmdb"
            manifest_path = root / "valid_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            payload = make_payload()
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(
                native_path,
                {"a": make_native_record(payload, "a")},
            )
            write_pickle_lmdb(
                decoy_path, {"a": make_decoy_set("a")}
            )
            write_test_manifest(
                manifest_path, source_path, ["a"], split="valid"
            )
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "split=train"):
                PosePretrainDataset(
                    manifest_path,
                    native_path,
                    decoy_path,
                    diagnostic_path,
                    seed=42,
                )

    def test_excludes_diagnostic_and_selects_fixed_stratified_decoys(self):
        from data_utils.local_interface_dataset import (
            PosePretrainDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            decoy_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            payloads = {"drop": make_payload(), "keep": make_payload()}
            write_test_lmdb(source_path, payloads)
            write_pickle_lmdb(
                native_path,
                {
                    key: make_native_record(payload, key)
                    for key, payload in payloads.items()
                },
            )
            write_pickle_lmdb(
                decoy_path,
                {
                    "drop": make_decoy_set("drop"),
                    "keep": make_decoy_set("keep"),
                },
            )
            write_test_manifest(
                manifest_path, source_path, ["drop", "keep"]
            )
            diagnostic_path.write_text(
                json.dumps({"keys": ["drop"]}), encoding="utf-8"
            )

            dataset = PosePretrainDataset(
                manifest_path,
                native_path,
                decoy_path,
                diagnostic_path,
                seed=42,
            )
            try:
                self.assertEqual(len(dataset), 1)
                dataset.set_epoch(7)
                left = dataset[0]
                right = dataset[0]
            finally:
                dataset.close()

            self.assertEqual(left["key"], "keep")
            self.assertNotIn("label", left)
            self.assertEqual(
                left["decoy_strata"], ("low", "medium", "high")
            )
            self.assertEqual(
                left["decoy_ligand_coords"].shape,
                (3, left["ligand_heavy_atom_count"], 3),
            )
            torch.testing.assert_close(
                left["decoy_ligand_coords"],
                right["decoy_ligand_coords"],
            )
            native = left["ligand_coords"]
            translations = left["decoy_transform_translation"]
            torch.testing.assert_close(
                left["decoy_ligand_coords"],
                native.unsqueeze(0) + translations[:, None, :],
            )

    def test_filters_generation_failures_not_present_in_decoy_cache(self):
        from data_utils.local_interface_dataset import (
            PosePretrainDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            decoy_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            payloads = {"failed": make_payload(), "keep": make_payload()}
            write_test_lmdb(source_path, payloads)
            write_pickle_lmdb(
                native_path,
                {
                    key: make_native_record(payload, key)
                    for key, payload in payloads.items()
                },
            )
            write_pickle_lmdb(
                decoy_path, {"keep": make_decoy_set("keep")}
            )
            write_test_manifest(
                manifest_path, source_path, ["failed", "keep"]
            )
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )

            dataset = PosePretrainDataset(
                manifest_path,
                native_path,
                decoy_path,
                diagnostic_path,
                seed=42,
            )
            try:
                self.assertEqual(
                    [sample["lmdb_key"] for sample in dataset.samples],
                    ["keep"],
                )
            finally:
                dataset.close()

    def test_v2_selects_explicit_banks_and_freezes_paired_key_order(self):
        from data_utils.local_interface_dataset import (
            PosePretrainDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            decoy_path = root / "decoys_v2.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            keys = ["both_a", "both_b", "random_only", "missing"]
            payloads = {key: make_payload() for key in keys}
            native_records = {
                key: make_native_record(payloads[key], key)
                for key in keys
            }
            write_test_lmdb(source_path, payloads)
            write_pickle_lmdb(native_path, native_records)
            write_pickle_lmdb(
                decoy_path,
                {
                    "both_a": make_v2_decoy_set(
                        native_records["both_a"]
                    ),
                    "both_b": make_v2_decoy_set(
                        native_records["both_b"]
                    ),
                    "random_only": make_v2_decoy_set(
                        native_records["random_only"],
                        matched_complete=False,
                    ),
                },
            )
            write_test_manifest(
                manifest_path, source_path, keys
            )
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )

            matched = PosePretrainDataset(
                manifest_path,
                native_path,
                decoy_path,
                diagnostic_path,
                seed=42,
                decoy_schema_version=2,
                decoy_bank="matched",
                require_complete_banks=("matched",),
            )
            paired_random = PosePretrainDataset(
                manifest_path,
                native_path,
                decoy_path,
                diagnostic_path,
                seed=42,
                decoy_schema_version=2,
                decoy_bank="random_physical",
                require_complete_banks=(
                    "random_physical",
                    "matched",
                ),
            )
            paired_matched = PosePretrainDataset(
                manifest_path,
                native_path,
                decoy_path,
                diagnostic_path,
                seed=42,
                decoy_schema_version=2,
                decoy_bank="matched",
                require_complete_banks=(
                    "random_physical",
                    "matched",
                ),
            )
            try:
                self.assertEqual(len(matched), 2)
                self.assertEqual(
                    matched.require_complete_banks, ("matched",)
                )
                self.assertEqual(len(paired_random), 2)
                self.assertEqual(len(paired_matched), 2)
                self.assertEqual(
                    paired_random.stage1_included_ordered_key_sha256,
                    paired_matched.stage1_included_ordered_key_sha256,
                )
                self.assertEqual(
                    paired_random.stage1_included_keys,
                    ["both_a", "both_b"],
                )
                self.assertEqual(
                    paired_random.stage1_missing_decoy_keys,
                    ["missing"],
                )
                self.assertEqual(
                    paired_random.stage1_incomplete_bank_keys,
                    ["random_only"],
                )
                item = paired_random[0]
            finally:
                paired_matched.close()
                paired_random.close()
                matched.close()

            self.assertEqual(item["decoy_bank"], "random_physical")
            self.assertEqual(item["decoy_schema_version"], 2)
            self.assertEqual(
                item["decoy_strata"], ("low", "medium", "high")
            )

    def test_v2_rejects_decoy_record_seed_mismatch(self):
        from data_utils.local_interface_dataset import (
            PosePretrainDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            decoy_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            payload = make_payload()
            native = make_native_record(payload, "a")
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(native_path, {"a": native})
            write_pickle_lmdb(
                decoy_path,
                {"a": make_v2_decoy_set(native, seed=43)},
            )
            write_test_manifest(manifest_path, source_path, ["a"])
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "seed.*does not match"
            ):
                PosePretrainDataset(
                    manifest_path,
                    native_path,
                    decoy_path,
                    diagnostic_path,
                    seed=42,
                    decoy_schema_version=2,
                    decoy_bank="matched",
                    require_complete_banks=("matched",),
                )

    def test_v2_rejects_schema_one_and_v1_rejects_explicit_bank(self):
        from data_utils.local_interface_dataset import (
            PosePretrainDataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            decoy_path = root / "decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            payload = make_payload()
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(
                native_path,
                {"a": make_native_record(payload, "a")},
            )
            write_pickle_lmdb(
                decoy_path, {"a": make_decoy_set("a")}
            )
            write_test_manifest(
                manifest_path, source_path, ["a"]
            )
            diagnostic_path.write_text(
                json.dumps({"keys": []}), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "schema version 2"
            ):
                PosePretrainDataset(
                    manifest_path,
                    native_path,
                    decoy_path,
                    diagnostic_path,
                    seed=42,
                    decoy_schema_version=2,
                    decoy_bank="matched",
                    require_complete_banks=("matched",),
                )
            with self.assertRaisesRegex(
                ValueError, "version 1.*decoy_bank"
            ):
                PosePretrainDataset(
                    manifest_path,
                    native_path,
                    decoy_path,
                    diagnostic_path,
                    seed=42,
                    decoy_schema_version=1,
                    decoy_bank="matched",
                )


class TestInterfaceBatchUtilities(unittest.TestCase):
    def test_collate_requires_exactly_one_sample(self):
        from data_utils.local_interface_dataset import (
            collate_single_interface,
        )

        sample = {"key": "a"}
        self.assertIs(collate_single_interface([sample]), sample)
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            collate_single_interface([])
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            collate_single_interface([sample, sample])

    def test_recursive_device_transfer_preserves_dtypes_and_structure(self):
        from data_utils.local_interface_dataset import recursive_to_device

        value = {
            "float": torch.tensor([1.0], dtype=torch.float64),
            "nested": [
                torch.tensor([1], dtype=torch.int16),
                (torch.tensor([True]), "keep"),
            ],
        }

        moved = recursive_to_device(value, torch.device("cpu"))

        self.assertEqual(moved["float"].dtype, torch.float64)
        self.assertEqual(moved["nested"][0].dtype, torch.int16)
        self.assertEqual(moved["nested"][1][0].dtype, torch.bool)
        self.assertIsInstance(moved["nested"], list)
        self.assertIsInstance(moved["nested"][1], tuple)
        self.assertEqual(moved["nested"][1][1], "keep")


class TestStage1Objectives(unittest.TestCase):
    def test_pose_set_loss_has_log_four_random_baseline_and_fixed_k(self):
        from dta.pretrain_local_interface import pose_set_loss

        native = torch.tensor(0.0, requires_grad=True)
        decoys = torch.zeros(3, requires_grad=True)
        loss = pose_set_loss(native, decoys, temperature=0.2)

        self.assertAlmostEqual(
            loss.item(), float(np.log(4.0)), places=6
        )
        loss.backward()
        self.assertTrue(torch.isfinite(native.grad))
        self.assertTrue(torch.isfinite(decoys.grad).all())
        with self.assertRaisesRegex(ValueError, "exactly K=3"):
            pose_set_loss(native, torch.zeros(2))

    def test_pretrain_objective_cannot_be_consistency_only(self):
        from dta.pretrain_local_interface import pretrain_objective

        native_projection = torch.tensor([[1.0, 0.0]])
        jitter_projection = torch.tensor([[1.0, 0.0]])
        total, pose, consistency = pretrain_objective(
            native_score=torch.tensor([[0.0]]),
            decoy_scores=torch.zeros(3, 1),
            native_projection=native_projection,
            jitter_projection=jitter_projection,
        )

        self.assertAlmostEqual(total.item(), np.log(4.0), places=6)
        self.assertAlmostEqual(pose.item(), np.log(4.0), places=6)
        self.assertEqual(consistency.item(), 0.0)
        with self.assertRaisesRegex(ValueError, "exactly K=3"):
            pretrain_objective(
                native_score=torch.tensor([[0.0]]),
                decoy_scores=torch.empty((0, 1)),
                native_projection=native_projection,
                jitter_projection=jitter_projection,
            )

    def test_model_stage1_forward_has_finite_local_only_gradients(self):
        from dta.pretrain_local_interface import stage1_forward_loss
        from net.interface_first import InterfaceFirstGerNADTA
        from tests.test_interface_first_model import (
            make_interface_item,
        )

        item = make_interface_item()
        translations = torch.tensor(
            [
                [0.10, 0.00, 0.00],
                [0.00, 0.10, 0.00],
                [0.00, 0.00, 0.10],
            ]
        )
        item["decoy_ligand_coords"] = (
            item["ligand_coords"].unsqueeze(0)
            + translations[:, None, :]
        )
        model = InterfaceFirstGerNADTA(
            ligand_gnn_depth=2,
            ligand_dropout=0.0,
        )
        model.configure_stage1()
        ligand_gnn_calls = []
        hook = model.local_path.ligand_gnn.register_forward_hook(
            lambda module, inputs, output: ligand_gnn_calls.append(1)
        )
        try:
            output = stage1_forward_loss(
                model,
                item,
                jitter_coords=item["ligand_coords"] + 0.01,
            )
        finally:
            hook.remove()

        self.assertTrue(torch.isfinite(output["total_loss"]))
        self.assertEqual(
            len(ligand_gnn_calls),
            1,
            "one pose set must share one static ligand encoding",
        )
        output["total_loss"].backward()
        local_gradients = [
            parameter.grad
            for parameter in model.local_path.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(local_gradients)
        self.assertTrue(
            all(torch.isfinite(value).all() for value in local_gradients)
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.protein_global.parameters()
            )
        )

    def test_ligand_jitter_is_rigid_valid_and_contact_preserving(self):
        from dta.pretrain_local_interface import (
            sample_valid_ligand_jitter,
        )

        payload = make_payload(core_atoms=4)
        payload["protein_features"]["A"]["atom_coords"][:] = np.array(
            [4.0, 0.0, 0.0], dtype=np.float32
        )
        record = make_native_record(payload, "jitter")
        result = sample_valid_ligand_jitter(
            record, seed=71, max_attempts=32
        )

        native = np.asarray(record["ligand_coords"])
        jittered = np.asarray(result["coords"])
        native_distances = np.linalg.norm(
            native[:, None, :] - native[None, :, :], axis=-1
        )
        jittered_distances = np.linalg.norm(
            jittered[:, None, :] - jittered[None, :, :], axis=-1
        )
        np.testing.assert_allclose(
            native_distances,
            jittered_distances,
            atol=1e-5,
            rtol=0.0,
        )
        self.assertLessEqual(
            result["translation_norm"], 0.10 + 1e-8
        )
        self.assertLessEqual(
            result["rotation_angle_degrees"], 3.0 + 1e-8
        )
        self.assertGreaterEqual(
            result["contact_mass_retention"], 0.95
        )
        self.assertFalse(result["overlap_statistics"]["has_hard_clash"])

    def test_native_hard_clash_uses_audited_identity_fallback(self):
        from dta.pretrain_local_interface import (
            sample_valid_ligand_jitter,
        )

        record = make_native_record(
            make_payload(core_atoms=4), "native_clash"
        )
        result = sample_valid_ligand_jitter(
            record, seed=73, max_attempts=2
        )

        np.testing.assert_allclose(
            result["coords"],
            record["ligand_coords"],
            atol=0.0,
            rtol=0.0,
        )
        self.assertTrue(result["fallback_identity"])
        self.assertTrue(result["native_hard_clash"])
        self.assertTrue(
            result["overlap_statistics"]["has_hard_clash"]
        )


class TestStage1Checkpoint(unittest.TestCase):
    def test_checkpoint_restores_complete_local_path_only(self):
        from dta.pretrain_local_interface import (
            load_stage1_checkpoint,
            save_stage1_checkpoint,
        )
        from net.interface_first import InterfaceFirstGerNADTA

        torch.manual_seed(127)
        source = InterfaceFirstGerNADTA(ligand_gnn_depth=2)
        source.configure_stage1()
        optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in source.parameters()
                if parameter.requires_grad
            ],
            lr=1e-4,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage1.pth"
            save_stage1_checkpoint(
                path,
                source,
                optimizer,
                epoch=7,
                metrics={"loss": 1.25},
                provenance={"train_manifest": "sha256:test"},
            )

            torch.manual_seed(131)
            target = InterfaceFirstGerNADTA(ligand_gnn_depth=2)
            global_before = {
                name: value.detach().clone()
                for name, value in target.protein_global.state_dict().items()
            }
            target.configure_stage1()
            target_optimizer = torch.optim.Adam(
                [
                    parameter
                    for parameter in target.parameters()
                    if parameter.requires_grad
                ],
                lr=1e-4,
            )
            payload = load_stage1_checkpoint(
                path,
                target,
                optimizer=target_optimizer,
                strict=True,
            )

        self.assertEqual(payload["epoch"], 7)
        self.assertEqual(payload["metrics"]["loss"], 1.25)
        self.assertEqual(
            payload["provenance"]["train_manifest"], "sha256:test"
        )
        for name, expected in source.local_path.state_dict().items():
            torch.testing.assert_close(
                expected, target.local_path.state_dict()[name]
            )
        for name, expected in global_before.items():
            torch.testing.assert_close(
                expected, target.protein_global.state_dict()[name]
            )


def passing_diagnostic_report():
    return {
        "score_probe": {"nonlinear_cv_r2": 0.20},
        "pose_probe": {"nonlinear_cv_auroc": 0.60},
        "interface": {
            "auroc": 0.85,
            "contact_mass_tertiles": [
                {"name": "low", "set_count": 120, "auroc": 0.80},
                {"name": "medium", "set_count": 120, "auroc": 0.82},
                {"name": "high", "set_count": 120, "auroc": 0.84},
            ],
        },
    }


class TestShortcutDiagnostics(unittest.TestCase):
    def test_each_shortcut_gate_failure_has_an_explicit_reason(self):
        from utils.interface_diagnostics import evaluate_shortcut_gate

        passed = evaluate_shortcut_gate(passing_diagnostic_report())
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["reasons"], [])

        cases = (
            (
                ("score_probe", "nonlinear_cv_r2"),
                0.50,
                "nonlinear_score_r2_too_high",
            ),
            (
                ("pose_probe", "nonlinear_cv_auroc"),
                0.80,
                "nonlinear_shortcut_auroc_too_high",
            ),
            (
                ("pose_probe", "nonlinear_cv_auroc"),
                0.76,
                "interface_shortcut_gap_too_small",
            ),
            (
                ("interface", "auroc"),
                0.74,
                "interface_auroc_too_low",
            ),
        )
        for path, value, reason in cases:
            report = passing_diagnostic_report()
            report[path[0]][path[1]] = value
            result = evaluate_shortcut_gate(report)
            self.assertFalse(result["passed"], reason)
            self.assertIn(reason, result["reasons"])

        report = passing_diagnostic_report()
        report["interface"]["contact_mass_tertiles"][1]["auroc"] = 0.64
        result = evaluate_shortcut_gate(report)
        self.assertFalse(result["passed"])
        self.assertIn(
            "interface_tertile_auroc_too_low:medium",
            result["reasons"],
        )

    def test_small_tertiles_do_not_trigger_the_hundred_set_gate(self):
        from utils.interface_diagnostics import evaluate_shortcut_gate

        report = passing_diagnostic_report()
        report["interface"]["contact_mass_tertiles"][0] = {
            "name": "low",
            "set_count": 99,
            "auroc": 0.10,
        }

        result = evaluate_shortcut_gate(report)

        self.assertTrue(result["passed"])

    def test_grouped_probe_report_contains_linear_and_nonlinear_metrics(self):
        from utils.interface_diagnostics import (
            SHORTCUT_FEATURE_NAMES,
            fit_shortcut_probes,
        )

        generator = np.random.default_rng(137)
        rows = []
        labels = []
        scores = []
        groups = []
        set_ids = []
        contact_mass = []
        for group_index in range(10):
            for pose_index in range(4):
                label = int(pose_index == 0)
                rows.append(
                    generator.normal(
                        size=len(SHORTCUT_FEATURE_NAMES)
                    )
                )
                labels.append(label)
                scores.append(
                    1.5 * label + generator.normal(scale=0.2)
                )
                groups.append(f"target-{group_index}")
                set_ids.append(f"set-{group_index}")
                contact_mass.append(float(group_index + 1))
        report = fit_shortcut_probes(
            np.asarray(rows),
            np.asarray(scores),
            np.asarray(labels),
            np.asarray(groups),
            np.asarray(set_ids),
            np.asarray(contact_mass),
        )

        self.assertEqual(report["cv_splits"], 5)
        self.assertEqual(
            report["feature_names"], list(SHORTCUT_FEATURE_NAMES)
        )
        self.assertIn("linear_cv_r2", report["score_probe"])
        self.assertIn("nonlinear_cv_r2", report["score_probe"])
        self.assertIn("linear_cv_auroc", report["pose_probe"])
        self.assertIn("nonlinear_cv_auroc", report["pose_probe"])
        self.assertEqual(
            len(report["score_probe"]["linear_coefficients"]),
            len(SHORTCUT_FEATURE_NAMES),
        )
        self.assertEqual(
            len(report["interface"]["contact_mass_tertiles"]),
            3,
        )


class TestStage1V2TrainerContract(unittest.TestCase):
    def _fixture_diagnostic_hash(self):
        from utils.experiment_integrity import hash_ordered_keys

        return hash_ordered_keys(["a"])

    def _validate_fixture_data(self, args):
        from dta.pretrain_local_interface import (
            validate_v2_data_diagnostic,
        )

        return validate_v2_data_diagnostic(
            args,
            expected_diagnostic_key_count=1,
            expected_diagnostic_ordered_key_sha256=(
                self._fixture_diagnostic_hash()
            ),
        )

    def _validate_fixture_training(self, args):
        from dta.pretrain_local_interface import (
            validate_v2_training_decoy_cache,
        )
        from utils.experiment_integrity import hash_ordered_keys

        return validate_v2_training_decoy_cache(
            args,
            expected_diagnostic_key_count=0,
            expected_diagnostic_ordered_key_sha256=(
                hash_ordered_keys([])
            ),
        )

    def _write_data_report_fixture(self, root, passed=True):
        from scripts.evaluate_local_interface_decoys_v2 import (
            EVALUATOR_NAME,
            REPORT_FILENAME,
            ROWS_FILENAME,
        )
        from scripts import build_local_interface_decoys_v2 as builder_v2
        from utils.experiment_integrity import (
            hash_ordered_keys,
            manifest_fingerprint,
            sha256_file,
        )
        from utils.interface_decoy_audit import evaluate_v2_data_gate

        source_path = root / "source.lmdb"
        native_path = root / "native"
        diagnostic_decoy_path = root / "diagnostic_decoys"
        diagnostic_path = root / "diagnostic.json"
        manifest_path = root / "train_manifest.json"
        report_dir = root / "diagnostic_report"
        report_dir.mkdir()
        payload = make_payload()
        native = make_native_record(payload, "a")
        write_test_lmdb(source_path, {"a": payload})
        write_pickle_lmdb(native_path, {"a": native})
        write_test_manifest(manifest_path, source_path, ["a"])
        diagnostic_path.write_text(
            json.dumps(
                {
                    "keys": ["a"],
                    "ordered_key_sha256": hash_ordered_keys(["a"]),
                    "manifest": manifest_fingerprint(manifest_path),
                }
            ),
            encoding="utf-8",
        )
        environment = builder_v2._open_output_lmdb(
            diagnostic_decoy_path, map_size_gb=1
        )
        try:
            builder_v2._write_record(
                environment,
                "a",
                pickle.dumps(
                    {"schema_version": 2, "key": "a"},
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
            )
            integrity = builder_v2._cache_integrity_summary(
                environment, ["a"]
            )
        finally:
            environment.close()
        native_environment = builder_v2._open_readonly_lmdb(
            native_path
        )
        try:
            native_keys = builder_v2._lmdb_keys(native_environment)
            selected_native_record_sha256 = (
                builder_v2._hash_ordered_lmdb_records(
                    native_environment, ["a"]
                )
            )
            selected_automorphism_sha256 = (
                builder_v2._automorphism_digest_aggregate(
                    native_environment, ["a"]
                )
            )
        finally:
            native_environment.close()
        native_metadata = builder_v2._native_metadata_summary(
            native_path
        )
        cache_provenance = {
            "builder": builder_v2.BUILDER_NAME,
            "schema_version": builder_v2.DECOY_SCHEMA_VERSION,
            "cohort": "diagnostic",
            "seed": 42,
            "max_samples": None,
            "manifest": manifest_fingerprint(manifest_path),
            "native_lmdb": str(native_path.resolve()),
            "native_key_count": len(native_keys),
            "native_key_sha256": hash_ordered_keys(native_keys),
            "selected_native_record_sha256": (
                selected_native_record_sha256
            ),
            "selected_automorphism_sha256": (
                selected_automorphism_sha256
            ),
            **native_metadata,
            "diagnostic_keys_path": str(diagnostic_path.resolve()),
            "diagnostic_file_sha256": sha256_file(diagnostic_path),
            "diagnostic_ordered_key_sha256": hash_ordered_keys(
                ["a"]
            ),
            "selected_key_count": 1,
            "selected_key_sha256": hash_ordered_keys(["a"]),
        }
        cache_report = {
            "builder": builder_v2.BUILDER_NAME,
            "schema_version": builder_v2.DECOY_SCHEMA_VERSION,
            "status": "complete",
            "output_lmdb": str(diagnostic_decoy_path.resolve()),
            "cohort": "diagnostic",
            "seed": 42,
            "max_samples": None,
            "included_keys": ["a"],
            "cache_integrity": integrity,
            "provenance": cache_provenance,
        }
        cache_report_path = diagnostic_decoy_path / "decoy_report.json"
        cache_report_path.write_text(
            json.dumps(cache_report), encoding="utf-8"
        )
        marker = {
            "builder": builder_v2.BUILDER_NAME,
            "schema_version": builder_v2.DECOY_SCHEMA_VERSION,
            "report_sha256": sha256_file(cache_report_path),
            "provenance": cache_provenance,
            "provenance_sha256": builder_v2._canonical_sha256(
                cache_provenance
            ),
            "cache_integrity": integrity,
            "cache_integrity_sha256": builder_v2._canonical_sha256(
                integrity
            ),
        }
        environment = builder_v2._open_output_lmdb(
            diagnostic_decoy_path, map_size_gb=1
        )
        try:
            with environment.begin(write=True) as transaction:
                transaction.put(
                    builder_v2.METADATA_KEY,
                    pickle.dumps(
                        cache_report,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                )
                transaction.put(
                    builder_v2.COMPLETION_KEY,
                    pickle.dumps(
                        marker,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                )
        finally:
            environment.close()
        rows_path = report_dir / ROWS_FILENAME
        np.savez_compressed(rows_path, labels=np.asarray([1, 0, 0, 0]))
        matched_auroc = 0.72 if passed else 0.90
        report = {
            "evaluator": EVALUATOR_NAME,
            "schema_version": 2,
            "status": "complete",
            "seed": 42,
            "inputs": {
                "manifest": manifest_fingerprint(manifest_path),
                "native_lmdb": str(native_path.resolve()),
                "diagnostic_decoy_lmdb": str(
                    diagnostic_decoy_path.resolve()
                ),
                "diagnostic_keys_path": str(diagnostic_path.resolve()),
                "diagnostic_file_sha256": sha256_file(diagnostic_path),
                "cache_report_sha256": sha256_file(cache_report_path),
            },
            "cache_provenance": cache_provenance,
            "bank_completion": {
                "requested_denominator": 1,
                "matched": {"complete_count": 1},
            },
            "cohorts": {
                "matched_full": {
                    "keys": ["a"],
                    "ordered_key_sha256": hash_ordered_keys(["a"]),
                    "matched": {
                        "coarse": {
                            "linear_cv_auroc": 0.70,
                            "nonlinear_cv_auroc": matched_auroc,
                        },
                        "heldout": {
                            "linear_cv_auroc": 0.68,
                            "nonlinear_cv_auroc": 0.71,
                        },
                    },
                },
                "paired_intersection": {
                    "keys": ["a"],
                    "ordered_key_sha256": hash_ordered_keys(["a"]),
                    "random_physical": {
                        "coarse": {
                            "nonlinear_cv_auroc": 0.90,
                        },
                    },
                    "matched": {
                        "coarse": {
                            "nonlinear_cv_auroc": matched_auroc,
                        },
                    },
                },
            },
            "artifacts": {
                "report_json": {"filename": REPORT_FILENAME},
                "rows_npz": {
                    "filename": ROWS_FILENAME,
                    "sha256": sha256_file(rows_path),
                },
            },
        }
        report["gate"] = evaluate_v2_data_gate(report)
        report_path = report_dir / REPORT_FILENAME
        report_path.write_text(
            json.dumps(report), encoding="utf-8"
        )
        args = SimpleNamespace(
            train_manifest=str(manifest_path),
            native_lmdb=str(native_path),
            diagnostic_lmdb=str(diagnostic_decoy_path),
            diagnostic_decoy_lmdb=str(diagnostic_decoy_path),
            diagnostic_keys=str(diagnostic_path),
            data_diagnostic_report=str(report_path),
            seed=42,
        )
        return args, report_path

    def test_initial_model_state_sha256_is_deterministic_and_sensitive(self):
        from dta.pretrain_local_interface import (
            initial_model_state_sha256,
        )

        left = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.BatchNorm1d(2),
        )
        right = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.BatchNorm1d(2),
        )
        right.load_state_dict(left.state_dict())

        left_digest = initial_model_state_sha256(left)
        self.assertEqual(left_digest, initial_model_state_sha256(right))
        with torch.no_grad():
            right[0].weight[0, 0].add_(1.0)
        self.assertNotEqual(
            left_digest, initial_model_state_sha256(right)
        )

    def test_stage2_ready_invalidation_removes_stale_marker(self):
        from dta.pretrain_local_interface import (
            invalidate_stage2_ready,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            ready_path = output_dir / "stage2_ready.json"
            ready_path.write_text(
                json.dumps({"status": "complete"}),
                encoding="utf-8",
            )

            returned = invalidate_stage2_ready(output_dir)

            self.assertEqual(returned, str(ready_path.resolve()))
            self.assertFalse(ready_path.exists())

    def test_formal_data_gate_rejects_shortened_diagnostic_holdout(self):
        from dta.pretrain_local_interface import (
            validate_v2_data_diagnostic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = self._write_data_report_fixture(root)

            with self.assertRaisesRegex(
                ValueError, "frozen holdout"
            ):
                validate_v2_data_diagnostic(args)

    def test_v2_data_report_must_be_passed_and_match_frozen_inputs(self):
        from dta.pretrain_local_interface import (
            validate_v2_data_diagnostic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, report_path = self._write_data_report_fixture(root)
            validated = self._validate_fixture_data(args)
            self.assertEqual(validated["report"]["gate"]["passed"], True)
            self.assertEqual(validated["path"], str(report_path.resolve()))
            self.assertEqual(len(validated["sha256"]), 64)

            failed_root = root / "failed"
            failed_root.mkdir()
            failed_args, _ = self._write_data_report_fixture(
                failed_root, passed=False
            )
            with self.assertRaisesRegex(
                ValueError, "data diagnostic gate did not pass"
            ):
                self._validate_fixture_data(failed_args)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["inputs"]["manifest"]["sha256"] = "0" * 64
            report_path.write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "manifest provenance"
            ):
                self._validate_fixture_data(args)

    def test_v2_data_report_recomputes_gate_and_rejects_reduced_cache(self):
        from dta.pretrain_local_interface import (
            validate_v2_data_diagnostic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, report_path = self._write_data_report_fixture(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["cohorts"]["paired_intersection"]["matched"][
                "coarse"
            ]["nonlinear_cv_auroc"] = 0.90
            report_path.write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "gate.*does not match"
            ):
                self._validate_fixture_data(args)

            reduced_root = root / "reduced"
            reduced_root.mkdir()
            args, report_path = self._write_data_report_fixture(
                reduced_root
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["cache_provenance"]["max_samples"] = 1
            report_path.write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "reduced|max_samples"
            ):
                self._validate_fixture_data(args)

    def test_v2_data_report_rejects_changed_native_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = self._write_data_report_fixture(root)
            self._validate_fixture_data(args)

            record = make_native_record(make_payload(), "a")
            record["ligand_coords"] = record["ligand_coords"].copy()
            record["ligand_coords"][0, 0] += 0.25
            write_pickle_lmdb(
                Path(args.native_lmdb), {"a": record}
            )

            with self.assertRaisesRegex(
                ValueError, "native.*provenance|native.*record"
            ):
                self._validate_fixture_data(args)

    def test_v2_data_report_rejects_changed_cache_records(self):
        from dta.pretrain_local_interface import (
            validate_v2_data_diagnostic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = self._write_data_report_fixture(root)
            self._validate_fixture_data(args)

            environment = lmdb.open(
                str(args.diagnostic_decoy_lmdb),
                map_size=1024 * 1024 * 1024,
                subdir=True,
            )
            try:
                with environment.begin(write=True) as transaction:
                    transaction.put(
                        b"a",
                        pickle.dumps(
                            {"schema_version": 2, "key": "tampered"},
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                    )
            finally:
                environment.close()

            with self.assertRaisesRegex(
                ValueError, "cache integrity|record digest"
            ):
                self._validate_fixture_data(args)

    def test_v2_training_cache_is_full_provenance_locked_and_untampered(self):
        from dta.pretrain_local_interface import (
            validate_v2_training_decoy_cache,
        )
        from scripts import build_local_interface_decoys_v2 as builder_v2
        from utils.experiment_integrity import (
            hash_ordered_keys,
            manifest_fingerprint,
            sha256_file,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native"
            cache_path = root / "train_decoys"
            manifest_path = root / "train_manifest.json"
            diagnostic_path = root / "diagnostic.json"
            payload = make_payload()
            native = make_native_record(payload, "a")
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(native_path, {"a": native})
            write_test_manifest(manifest_path, source_path, ["a"])
            diagnostic_path.write_text(
                json.dumps(
                    {
                        "keys": [],
                        "ordered_key_sha256": hash_ordered_keys([]),
                        "manifest": manifest_fingerprint(manifest_path),
                    }
                ),
                encoding="utf-8",
            )
            environment = builder_v2._open_output_lmdb(
                cache_path, map_size_gb=1
            )
            try:
                builder_v2._write_record(
                    environment,
                    "a",
                    pickle.dumps(
                        {"schema_version": 2, "key": "a", "seed": 42},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                )
                integrity = builder_v2._cache_integrity_summary(
                    environment, ["a"]
                )
            finally:
                environment.close()
            native_environment = builder_v2._open_readonly_lmdb(
                native_path
            )
            try:
                native_keys = builder_v2._lmdb_keys(native_environment)
                selected_native_record_sha256 = (
                    builder_v2._hash_ordered_lmdb_records(
                        native_environment, ["a"]
                    )
                )
                selected_automorphism_sha256 = (
                    builder_v2._automorphism_digest_aggregate(
                        native_environment, ["a"]
                    )
                )
            finally:
                native_environment.close()
            native_metadata = builder_v2._native_metadata_summary(
                native_path
            )
            provenance = {
                "builder": builder_v2.BUILDER_NAME,
                "schema_version": 2,
                "cohort": "train",
                "seed": 42,
                "max_samples": None,
                "manifest": manifest_fingerprint(manifest_path),
                "native_lmdb": str(native_path.resolve()),
                "native_key_count": len(native_keys),
                "native_key_sha256": hash_ordered_keys(native_keys),
                "selected_native_record_sha256": (
                    selected_native_record_sha256
                ),
                "selected_automorphism_sha256": (
                    selected_automorphism_sha256
                ),
                **native_metadata,
                "diagnostic_keys_path": str(
                    diagnostic_path.resolve()
                ),
                "diagnostic_file_sha256": sha256_file(
                    diagnostic_path
                ),
                "diagnostic_ordered_key_sha256": hash_ordered_keys(
                    []
                ),
                "selected_key_count": 1,
                "selected_key_sha256": hash_ordered_keys(["a"]),
            }
            cache_report = {
                "builder": builder_v2.BUILDER_NAME,
                "schema_version": 2,
                "status": "complete",
                "output_lmdb": str(cache_path.resolve()),
                "cohort": "train",
                "seed": 42,
                "max_samples": None,
                "included_keys": ["a"],
                "cache_integrity": integrity,
                "provenance": provenance,
            }
            cache_report_path = cache_path / "decoy_report.json"
            cache_report_path.write_text(
                json.dumps(cache_report), encoding="utf-8"
            )
            marker = {
                "builder": builder_v2.BUILDER_NAME,
                "schema_version": 2,
                "report_sha256": sha256_file(cache_report_path),
                "provenance": provenance,
                "provenance_sha256": builder_v2._canonical_sha256(
                    provenance
                ),
                "cache_integrity": integrity,
                "cache_integrity_sha256": builder_v2._canonical_sha256(
                    integrity
                ),
            }
            environment = builder_v2._open_output_lmdb(
                cache_path, map_size_gb=1
            )
            try:
                with environment.begin(write=True) as transaction:
                    transaction.put(
                        builder_v2.METADATA_KEY,
                        pickle.dumps(
                            cache_report,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                    )
                    transaction.put(
                        builder_v2.COMPLETION_KEY,
                        pickle.dumps(
                            marker,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                    )
            finally:
                environment.close()
            args = SimpleNamespace(
                train_manifest=str(manifest_path),
                native_lmdb=str(native_path),
                decoy_lmdb=str(cache_path),
                diagnostic_keys=str(diagnostic_path),
                seed=42,
            )

            validated = self._validate_fixture_training(args)
            self.assertEqual(
                validated["report"]["cohort"], "train"
            )
            self.assertEqual(len(validated["sha256"]), 64)

            args.seed = 43
            with self.assertRaisesRegex(
                ValueError, "identity|provenance"
            ):
                self._validate_fixture_training(args)
            args.seed = 42

            environment = lmdb.open(
                str(cache_path),
                map_size=1024 * 1024 * 1024,
                subdir=True,
            )
            try:
                with environment.begin(write=True) as transaction:
                    transaction.put(
                        b"a",
                        pickle.dumps(
                            {"schema_version": 2, "key": "tampered"},
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                    )
            finally:
                environment.close()
            with self.assertRaisesRegex(
                ValueError, "record digest|cache integrity"
            ):
                self._validate_fixture_training(args)

    def test_v2_provenance_locks_bank_cohort_and_diagnostic_artifacts(self):
        from dta.pretrain_local_interface import (
            build_stage1_provenance,
            validate_v2_data_diagnostic,
        )
        from utils.experiment_integrity import hash_ordered_keys

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = self._write_data_report_fixture(root)
            args.decoy_lmdb = args.diagnostic_decoy_lmdb
            args.decoy_schema_version = 2
            args.decoy_bank = "matched"
            args.require_paired_bank_intersection = True
            args.ligand_gnn_depth = 3
            args.ligand_dropout = 0.1
            args.epochs = 1
            args.lr = 1e-4
            args.weight_decay = 1e-5
            args.grad_clip = 1.0
            args.temperature = 0.2
            args.consistency_weight = 0.1
            dataset = mock.MagicMock()
            dataset.__len__.return_value = 2
            dataset.stage1_missing_decoy_keys = ["missing"]
            dataset.stage1_incomplete_bank_keys = ["incomplete"]
            dataset.stage1_included_keys = ["a", "b"]
            dataset.stage1_included_ordered_key_sha256 = (
                hash_ordered_keys(["a", "b"])
            )
            dataset.require_complete_banks = (
                "random_physical",
                "matched",
            )
            validated = self._validate_fixture_data(args)

            provenance = build_stage1_provenance(
                args,
                dataset,
                initial_state_sha256="a" * 64,
                v2_data_diagnostic=validated,
                v2_training_cache={
                    "path": "/frozen/train/decoy_report.json",
                    "sha256": "b" * 64,
                    "cache_integrity": {
                        "ordered_record_digest_sha256": "c" * 64,
                    },
                },
            )

            self.assertEqual(provenance["decoy_schema_version"], 2)
            self.assertEqual(provenance["decoy_bank"], "matched")
            self.assertEqual(
                provenance["required_complete_banks"],
                ["random_physical", "matched"],
            )
            self.assertEqual(
                provenance["stage1_included_ordered_key_sha256"],
                hash_ordered_keys(["a", "b"]),
            )
            self.assertEqual(
                provenance["model"]["initial_state_sha256"],
                "a" * 64,
            )
            self.assertEqual(
                provenance["data_diagnostic"]["sha256"],
                validated["sha256"],
            )
            self.assertEqual(
                provenance["diagnostic_decoy_lmdb"][
                    "decoy_report_sha256"
                ],
                validated["diagnostic_cache_report_sha256"],
            )
            self.assertEqual(
                provenance["training_decoy_cache"]["sha256"],
                "b" * 64,
            )
            self.assertEqual(
                provenance["training_decoy_cache"][
                    "cache_integrity"
                ]["ordered_record_digest_sha256"],
                "c" * 64,
            )

    def test_v2_model_diagnostic_uses_cached_poses_and_is_non_gating(self):
        from dta.pretrain_local_interface import (
            run_v2_model_diagnostics,
        )

        class FakeLocalPath:
            def encode_static(self, item):
                return None

            def forward_geometry(
                self, item, static_state, ligand_coords
            ):
                score = torch.zeros((), device=ligand_coords.device)
                return SimpleNamespace(interface_score=score)

        class FakeModel:
            def __init__(self):
                self.local_path = FakeLocalPath()

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            decoy_path = root / "diagnostic_decoys.lmdb"
            manifest_path = root / "train_manifest.json"
            output_dir = root / "output"
            payload = make_payload()
            native = make_native_record(payload, "a")
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(native_path, {"a": native})
            write_pickle_lmdb(
                decoy_path, {"a": make_v2_decoy_set(native)}
            )
            write_test_manifest(manifest_path, source_path, ["a"])
            data_report = {
                "cohorts": {
                    "matched_full": {
                        "keys": ["a"],
                        "matched": {
                            "coarse": {
                                "linear_cv_auroc": 0.99,
                                "nonlinear_cv_auroc": 0.99,
                            },
                            "heldout": {
                                "linear_cv_auroc": 0.99,
                                "nonlinear_cv_auroc": 0.99,
                            },
                        },
                    }
                }
            }

            with mock.patch(
                "data_utils.local_interface_decoys.generate_decoy_set",
                side_effect=AssertionError(
                    "v2 diagnostics must not generate decoys"
                ),
            ):
                report = run_v2_model_diagnostics(
                    FakeModel(),
                    train_manifest=manifest_path,
                    native_lmdb=native_path,
                    diagnostic_decoy_lmdb=decoy_path,
                    data_report=data_report,
                    output_dir=output_dir,
                    seed=42,
                    device=torch.device("cpu"),
                    bank="matched",
                    paired_intersection=False,
                )

            self.assertTrue(report["descriptive_only"])
            self.assertEqual(report["gate"]["passed"], True)
            self.assertEqual(report["gate"]["reasons"], [])
            self.assertLess(
                report["metrics"][
                    "interface_minus_coarse_nonlinear_auroc"
                ],
                0.0,
            )
            self.assertTrue(
                (output_dir / "model_diagnostic_report.json").is_file()
            )
            self.assertTrue(
                (output_dir / "model_diagnostic_rows.npz").is_file()
            )

            write_pickle_lmdb(
                decoy_path,
                {"a": make_v2_decoy_set(native, seed=43)},
            )
            with self.assertRaisesRegex(
                ValueError, "seed.*does not match"
            ):
                run_v2_model_diagnostics(
                    FakeModel(),
                    train_manifest=manifest_path,
                    native_lmdb=native_path,
                    diagnostic_decoy_lmdb=decoy_path,
                    data_report=data_report,
                    output_dir=output_dir / "seed_mismatch",
                    seed=42,
                    device=torch.device("cpu"),
                    bank="matched",
                    paired_intersection=False,
                )


class TestStage1CliContract(unittest.TestCase):
    def test_cli_is_train_only_and_defaults_to_thirty_epochs(self):
        from dta.pretrain_local_interface import build_arg_parser

        parser = build_arg_parser()
        destinations = {
            action.dest for action in parser._actions
        }
        self.assertIn("train_manifest", destinations)
        self.assertNotIn("valid_manifest", destinations)
        self.assertNotIn("test_manifest", destinations)
        args = parser.parse_args(
            [
                "--train-manifest",
                "train.json",
                "--native-lmdb",
                "native",
                "--decoy-lmdb",
                "decoys",
                "--diagnostic-keys",
                "diagnostic.json",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.decoy_schema_version, 1)
        self.assertIsNone(args.decoy_bank)
        self.assertFalse(args.require_paired_bank_intersection)
        self.assertIsNone(args.diagnostic_decoy_lmdb)
        self.assertIsNone(args.data_diagnostic_report)

        v2 = parser.parse_args(
            [
                "--train-manifest",
                "train.json",
                "--native-lmdb",
                "native",
                "--decoy-lmdb",
                "train-decoys",
                "--diagnostic-keys",
                "diagnostic.json",
                "--output-dir",
                "output",
                "--decoy-schema-version",
                "2",
                "--decoy-bank",
                "matched",
                "--require-paired-bank-intersection",
                "--diagnostic-decoy-lmdb",
                "diagnostic-decoys",
                "--data-diagnostic-report",
                "diagnostic-report.json",
            ]
        )
        self.assertEqual(v2.decoy_schema_version, 2)
        self.assertEqual(v2.decoy_bank, "matched")
        self.assertTrue(v2.require_paired_bank_intersection)
        self.assertEqual(v2.diagnostic_decoy_lmdb, "diagnostic-decoys")
        self.assertEqual(
            v2.data_diagnostic_report, "diagnostic-report.json"
        )

    def test_cli_help_runs_from_the_repository_root(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "dta" / "pretrain_local_interface.py"),
                "--help",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
