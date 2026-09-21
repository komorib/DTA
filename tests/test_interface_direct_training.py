import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dta.train_dta import build_arg_parser, validate_training_args
from tests.test_interface_pretraining import (
    make_native_record,
    write_pickle_lmdb,
)
from tests.test_local_interface_cache import (
    make_payload,
    write_test_lmdb,
    write_test_manifest,
)
from utils.experiment_integrity import hash_ordered_keys


def make_interface_args(interface_init_mode="scratch"):
    parser = build_arg_parser()
    return parser.parse_args([
        "--model_arch", "interface_first",
        "--interface_init_mode", interface_init_mode,
        "--interface_train_native_lmdb", "train.lmdb",
        "--interface_valid_native_lmdb", "valid.lmdb",
        "--interface_test_native_lmdb", "test.lmdb",
        "--batch_size", "1",
        "--early_stop_patience", "0",
        "--select_metric", "CI",
        "--interface_warmup_epochs", "0",
    ])


class TestScratchModeContract(unittest.TestCase):
    def setUp(self):
        self.parser = build_arg_parser()

    def _interface_args(self, *extra):
        return self.parser.parse_args([
            "--model_arch", "interface_first",
            "--interface_train_native_lmdb", "train.lmdb",
            "--interface_valid_native_lmdb", "valid.lmdb",
            "--interface_test_native_lmdb", "test.lmdb",
            "--batch_size", "1",
            "--early_stop_patience", "0",
            "--select_metric", "CI",
            *extra,
        ])

    def test_scratch_accepts_native_caches_without_stage1_artifacts(self):
        args = self._interface_args(
            "--interface_init_mode", "scratch",
            "--interface_warmup_epochs", "0",
        )

        validate_training_args(self.parser, args)

    def test_scratch_rejects_stage1_artifacts(self):
        args = self._interface_args(
            "--interface_init_mode", "scratch",
            "--interface_warmup_epochs", "0",
            "--interface_stage1_checkpoint", "stage1.pth",
        )

        with self.assertRaises(SystemExit):
            validate_training_args(self.parser, args)

    def test_scratch_requires_zero_warmup_epochs(self):
        args = self._interface_args(
            "--interface_init_mode", "scratch",
        )

        with self.assertRaises(SystemExit):
            validate_training_args(self.parser, args)

    def test_stage1_still_requires_stage1_artifacts(self):
        args = self._interface_args()

        with self.assertRaises(SystemExit):
            validate_training_args(self.parser, args)


class TestNativeInterfacePreflight(unittest.TestCase):
    def test_rejects_manifest_key_missing_from_native_lmdb(self):
        from data_utils.local_interface_dataset import (
            validate_native_interface_inventory,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            payload_a = make_payload()
            payload_b = make_payload()
            write_test_lmdb(
                source_path, {"a": payload_a, "b": payload_b}
            )
            write_pickle_lmdb(
                native_path, {"a": make_native_record(payload_a, "a")}
            )
            write_test_manifest(manifest_path, source_path, ["a", "b"])

            with self.assertRaisesRegex(
                ValueError, "missing native interface records: b"
            ):
                validate_native_interface_inventory(
                    manifest_path, native_path
                )

    def test_returns_ordered_key_hash_after_validating_every_record(self):
        from data_utils.local_interface_dataset import (
            validate_native_interface_inventory,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            payload_a = make_payload()
            payload_b = make_payload()
            write_test_lmdb(
                source_path, {"a": payload_a, "b": payload_b}
            )
            write_pickle_lmdb(
                native_path,
                {
                    "a": make_native_record(payload_a, "a"),
                    "b": make_native_record(payload_b, "b"),
                },
            )
            write_test_manifest(manifest_path, source_path, ["b", "a"])

            summary = validate_native_interface_inventory(
                manifest_path, native_path
            )

            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(
                summary["ordered_key_sha256"], hash_ordered_keys(["b", "a"])
            )

    def test_rejects_malformed_native_record(self):
        from data_utils.local_interface_dataset import (
            validate_native_interface_inventory,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.lmdb"
            native_path = root / "native.lmdb"
            manifest_path = root / "train_manifest.json"
            payload = make_payload()
            native = make_native_record(payload, "a")
            native["ligand_coords"] = native["ligand_coords"][:-1]
            write_test_lmdb(source_path, {"a": payload})
            write_pickle_lmdb(native_path, {"a": native})
            write_test_manifest(manifest_path, source_path, ["a"])

            with self.assertRaisesRegex(
                ValueError, "ligand coordinate shape"
            ):
                validate_native_interface_inventory(
                    manifest_path, native_path
                )


class TestScratchTrainingSetup(unittest.TestCase):
    def test_scratch_initialization_does_not_call_stage1_gate(self):
        import dta.interface_first_training as training

        args = make_interface_args(interface_init_mode="scratch")
        with mock.patch.object(
            training,
            "_validate_stage1_gate",
            side_effect=AssertionError("scratch must not read Stage-1"),
        ):
            initialization = training.build_interface_initialization(args)

        self.assertEqual(initialization, {"mode": "scratch"})

    def test_scratch_resume_rejects_stage1_provenance(self):
        from dta.interface_first_training import (
            validate_resume_provenance,
        )

        provenance = {"initialization": {"mode": "scratch"}}
        checkpoint = {
            "provenance": {
                "initialization": {
                    "mode": "stage1",
                    "artifacts": {},
                }
            }
        }

        with self.assertRaisesRegex(RuntimeError, "resume provenance"):
            validate_resume_provenance(checkpoint, provenance)
