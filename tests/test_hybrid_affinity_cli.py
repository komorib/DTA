from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


class TestHybridAffinityParser(unittest.TestCase):
    def test_requires_train_valid_inputs_and_exposes_no_test_path(self):
        from dta.train_hybrid_affinity import build_arg_parser

        parser = build_arg_parser()
        required = {
            action.dest
            for action in parser._actions
            if getattr(action, "required", False)
        }
        self.assertEqual(
            required,
            {
                "train_manifest",
                "valid_manifest",
                "train_native_cache",
                "valid_native_cache",
                "gerna_checkpoint",
                "interface_checkpoint",
                "output_dir",
                "fusion_type",
                "rank_weight",
                "seed",
                "device",
            },
        )
        self.assertNotIn(
            "test_path", {action.dest for action in parser._actions}
        )


class TestBackboneCheckpointLoading(unittest.TestCase):
    def test_loads_plain_state_dict_strictly_and_records_sha256(self):
        from dta.train_hybrid_affinity import load_backbone_checkpoint

        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plain.pth"
            torch.save(source.state_dict(), path)
            artifact = load_backbone_checkpoint(target, path)

            self.assertEqual(artifact["format"], "state_dict")
            self.assertEqual(
                artifact["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[name]))

    def test_loads_model_state_dict_payload_strictly(self):
        from dta.train_hybrid_affinity import load_backbone_checkpoint

        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.pth"
            torch.save({"model_state_dict": source.state_dict()}, path)
            artifact = load_backbone_checkpoint(target, path)

            self.assertEqual(artifact["format"], "model_state_dict")
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[name]))

    def test_rejects_checkpoint_schema_mismatch(self):
        from dta.train_hybrid_affinity import load_backbone_checkpoint

        target = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong.pth"
            torch.save({"wrong.weight": torch.ones(1)}, path)
            with self.assertRaises(RuntimeError):
                load_backbone_checkpoint(target, path)


class TestExactBackboneBuilders(unittest.TestCase):
    def test_builds_gerna_with_frozen_reference_configuration(self):
        from dta.train_hybrid_affinity import build_gerna_backbone

        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

        build_gerna_backbone(model_class=factory)
        self.assertEqual(calls[0][0], ([4, 2, 128, 128],))
        self.assertEqual(
            calls[0][1],
            {
                "trigonometry": True,
                "mol_graph": True,
                "rna_repre": True,
                "rna_graph": False,
                "coors": True,
                "coors_3_bead": True,
                "uncertainty": False,
                "rna_num_tokens": None,
                "rna_3d_input_dim": 640,
                "disable_protein_2d": True,
            },
        )

    def test_builds_interface_with_frozen_reference_configuration(self):
        from dta.train_hybrid_affinity import build_interface_backbone

        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return object()

        build_interface_backbone(model_class=factory)
        self.assertEqual(
            calls,
            [
                {
                    "ligand_gnn_depth": 3,
                    "ligand_dropout": 0.1,
                    "affinity_dropout": 0.2,
                    "residue_to_atom_mode": "gated_residual",
                }
            ],
        )


class TestHybridDatasetAssembly(unittest.TestCase):
    def test_builds_both_views_with_exact_gerna_options(self):
        from dta.train_hybrid_affinity import build_hybrid_dataset

        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "train.json"
            samples = [{"lmdb_key": "a", "label": 7.0}]
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_type": "pdbbind_lmdb",
                        "split": "train",
                        "samples": samples,
                    }
                ),
                encoding="utf-8",
            )
            calls = {}

            class GernaDataset:
                def __init__(self, *args, **kwargs):
                    calls["gerna"] = (args, kwargs)

            class InterfaceDataset:
                def __init__(self, *args, **kwargs):
                    calls["interface"] = (args, kwargs)

            sentinel = object()

            def hybrid_dataset(manifest_samples, gerna, interface):
                calls["hybrid"] = (manifest_samples, gerna, interface)
                return sentinel

            result = build_hybrid_dataset(
                manifest_path,
                Path(temporary) / "native",
                expected_split="train",
                gerna_dataset_class=GernaDataset,
                interface_dataset_class=InterfaceDataset,
                hybrid_dataset_class=hybrid_dataset,
            )

            self.assertIs(result, sentinel)
            self.assertEqual(
                calls["gerna"],
                (
                    (str(manifest_path.resolve()),),
                    {
                        "protein_3d_mode": "ligand_pocket_residue_esm",
                        "protein_embedding_mode": "esm",
                        "ligand_pocket_cutoff": 8.0,
                        "disable_protein_2d": True,
                    },
                ),
            )
            self.assertEqual(
                calls["interface"][0],
                (
                    str(manifest_path.resolve()),
                    str((Path(temporary) / "native").resolve()),
                ),
            )
            self.assertEqual(calls["hybrid"][0], samples)


class TestReusableDatasetPreflight(unittest.TestCase):
    def make_binding_fixture(self, root):
        from utils.experiment_integrity import manifest_fingerprint

        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset_type": "pdbbind_lmdb",
                    "split": "train",
                    "samples": [
                        {"lmdb_key": "a", "label": 1.0},
                        {"lmdb_key": "b", "label": 2.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        native_cache = root / "native"
        native_cache.mkdir()
        (native_cache / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "manifest": manifest_fingerprint(manifest_path),
                }
            ),
            encoding="utf-8",
        )
        return manifest_path, native_cache

    def test_match_requires_all_four_binding_fields(self):
        from dta.train_hybrid_affinity import preflight_report_matches

        binding = {
            "manifest_sha256": "a" * 64,
            "native_metadata_sha256": "b" * 64,
            "ordered_key_sha256": "c" * 64,
            "sample_count": 2,
        }
        report = {
            "status": "complete",
            "binding": dict(binding),
            "sample_count": 2,
            "ordered_key_sha256": "c" * 64,
            "largest_indices": [1, 0],
            "largest_keys": ["b", "a"],
        }
        self.assertTrue(preflight_report_matches(report, binding))
        incomplete = {"status": "complete", "binding": dict(binding)}
        self.assertFalse(
            preflight_report_matches(incomplete, binding)
        )
        contradictory = dict(report)
        contradictory["sample_count"] = 3
        self.assertFalse(
            preflight_report_matches(contradictory, binding)
        )
        invalid_largest = dict(report)
        invalid_largest["largest_indices"] = [0, 0]
        self.assertFalse(
            preflight_report_matches(invalid_largest, binding)
        )
        unhashable_largest = dict(report)
        unhashable_largest["largest_indices"] = [[0], 1]
        self.assertFalse(
            preflight_report_matches(unhashable_largest, binding)
        )
        for field in binding:
            changed = json.loads(json.dumps(report))
            changed["binding"][field] = (
                3 if field == "sample_count" else "d" * 64
            )
            self.assertFalse(preflight_report_matches(changed, binding))

    def test_reuses_only_a_matching_complete_report(self):
        from dta.train_hybrid_affinity import (
            run_or_reuse_dataset_preflight,
        )
        from utils.experiment_integrity import hash_ordered_keys

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, native_cache = self.make_binding_fixture(root)
            report_path = root / "preflight.json"
            calls = []

            def preflight(dataset):
                calls.append(dataset)
                return {
                    "sample_count": 2,
                    "ordered_key_sha256": hash_ordered_keys(["a", "b"]),
                    "largest_indices": [1, 0],
                    "largest_keys": ["b", "a"],
                }

            first = run_or_reuse_dataset_preflight(
                dataset=object(),
                manifest_path=manifest_path,
                native_cache_path=native_cache,
                report_path=report_path,
                preflight_fn=preflight,
            )
            second = run_or_reuse_dataset_preflight(
                dataset=object(),
                manifest_path=manifest_path,
                native_cache_path=native_cache,
                report_path=report_path,
                preflight_fn=lambda dataset: self.fail(
                    "matching report should be reused"
                ),
            )

            self.assertEqual(len(calls), 1)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(
                set(first["binding"]),
                {
                    "manifest_sha256",
                    "native_metadata_sha256",
                    "ordered_key_sha256",
                    "sample_count",
                },
            )

    def test_changed_native_metadata_forces_full_preflight(self):
        from dta.train_hybrid_affinity import (
            run_or_reuse_dataset_preflight,
        )
        from utils.experiment_integrity import hash_ordered_keys

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, native_cache = self.make_binding_fixture(root)
            report_path = root / "preflight.json"
            result = {
                "sample_count": 2,
                "ordered_key_sha256": hash_ordered_keys(["a", "b"]),
                "largest_indices": [0],
                "largest_keys": ["a"],
            }
            run_or_reuse_dataset_preflight(
                object(),
                manifest_path,
                native_cache,
                report_path,
                preflight_fn=lambda dataset: result,
            )
            metadata_path = native_cache / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["audit_note"] = "changed"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            calls = []
            refreshed = run_or_reuse_dataset_preflight(
                object(),
                manifest_path,
                native_cache,
                report_path,
                preflight_fn=lambda dataset: calls.append(dataset) or result,
            )

            self.assertEqual(len(calls), 1)
            self.assertFalse(refreshed["reused"])


class TestGpuMemoryPreflight(unittest.TestCase):
    class FakeCuda:
        def __init__(self, peaks):
            self.peaks = peaks
            self.current_group_size = None
            self.reset_calls = []
            self.empty_cache_calls = 0

        def reset_peak_memory_stats(self, device):
            self.reset_calls.append(str(device))

        def max_memory_allocated(self, device):
            return self.peaks[self.current_group_size][0]

        def max_memory_reserved(self, device):
            return self.peaks[self.current_group_size][1]

        def empty_cache(self):
            self.empty_cache_calls += 1

    def test_selects_group_four_when_probe_fits(self):
        from dta.train_hybrid_affinity import run_gpu_memory_preflight

        with tempfile.TemporaryDirectory() as temporary:
            cuda = self.FakeCuda({4: (100, 200)})
            calls = []

            def probe(model, samples, device):
                calls.append(len(samples))
                cuda.current_group_size = len(samples)

            result = run_gpu_memory_preflight(
                model=object(),
                dataset=list(range(8)),
                largest_indices=[7, 6, 5, 4],
                device="cuda:0",
                report_path=Path(temporary) / "memory.json",
                memory_budget_bytes=1000,
                probe_fn=probe,
                cuda_api=cuda,
                configure_fn=lambda model, stage: self.assertEqual(
                    stage, "full"
                ),
            )

            self.assertEqual(calls, [4])
            self.assertEqual(result["selected_group_size"], 4)
            self.assertEqual(result["optimizer"], "AdamW")
            self.assertTrue(result["optimizer_step"])
            self.assertTrue(result["grad_scaler"])
            self.assertEqual(result["ema_decay"], 0.999)
            self.assertEqual(result["attempts"][0]["status"], "complete")
            self.assertEqual(cuda.empty_cache_calls, 1)

    def test_oom_at_four_retries_and_selects_two(self):
        from dta.train_hybrid_affinity import run_gpu_memory_preflight

        with tempfile.TemporaryDirectory() as temporary:
            cuda = self.FakeCuda(
                {4: (900, 1000), 2: (90, 120)}
            )
            calls = []

            def probe(model, samples, device):
                size = len(samples)
                calls.append(size)
                cuda.current_group_size = size
                if size == 4:
                    raise torch.cuda.OutOfMemoryError("synthetic oom")

            result = run_gpu_memory_preflight(
                object(),
                list(range(8)),
                [7, 6, 5, 4],
                "cuda:0",
                Path(temporary) / "memory.json",
                memory_budget_bytes=1000,
                probe_fn=probe,
                cuda_api=cuda,
                configure_fn=lambda model, stage: None,
            )

            self.assertEqual(calls, [4, 2])
            self.assertEqual(result["selected_group_size"], 2)
            self.assertEqual(
                [attempt["status"] for attempt in result["attempts"]],
                ["oom", "complete"],
            )
            self.assertEqual(
                result["attempts"][0]["max_allocated_bytes"], 900
            )
            self.assertEqual(
                result["attempts"][0]["max_reserved_bytes"], 1000
            )
            self.assertEqual(cuda.empty_cache_calls, 2)

    def test_over_budget_at_four_retries_two(self):
        from dta.train_hybrid_affinity import run_gpu_memory_preflight

        with tempfile.TemporaryDirectory() as temporary:
            cuda = self.FakeCuda({4: (800, 1200), 2: (400, 600)})

            def probe(model, samples, device):
                cuda.current_group_size = len(samples)

            result = run_gpu_memory_preflight(
                object(),
                list(range(4)),
                [0, 1, 2, 3],
                "cuda:0",
                Path(temporary) / "memory.json",
                memory_budget_bytes=1000,
                probe_fn=probe,
                cuda_api=cuda,
                configure_fn=lambda model, stage: None,
            )

            self.assertEqual(
                [attempt["status"] for attempt in result["attempts"]],
                ["over_budget", "complete"],
            )
            self.assertEqual(result["selected_group_size"], 2)

    def test_failure_at_group_two_writes_report_and_stops(self):
        from dta.train_hybrid_affinity import run_gpu_memory_preflight

        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "memory.json"
            cuda = self.FakeCuda(
                {4: (950, 1000), 2: (700, 800)}
            )

            def always_oom(model, samples, device):
                cuda.current_group_size = len(samples)
                raise torch.cuda.OutOfMemoryError("synthetic oom")

            with self.assertRaises(RuntimeError):
                run_gpu_memory_preflight(
                    object(),
                    list(range(4)),
                    [0, 1, 2, 3],
                    "cuda:0",
                    report_path,
                    memory_budget_bytes=1000,
                    probe_fn=always_oom,
                    cuda_api=cuda,
                    configure_fn=lambda model, stage: None,
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(
                [attempt["group_size"] for attempt in report["attempts"]],
                [4, 2],
            )


class TestCliSafetyAndProvenance(unittest.TestCase):
    @staticmethod
    def parse_args(output_dir, extra=()):
        from dta.train_hybrid_affinity import build_arg_parser

        arguments = [
            "--train-manifest",
            "train.json",
            "--valid-manifest",
            "valid.json",
            "--train-native-cache",
            "native_train",
            "--valid-native-cache",
            "native_valid",
            "--gerna-checkpoint",
            "gerna.pth",
            "--interface-checkpoint",
            "interface.pth",
            "--output-dir",
            str(output_dir),
            "--fusion-type",
            "transformer",
            "--rank-weight",
            "0.2",
            "--seed",
            "42",
            "--device",
            "cuda:0",
        ]
        arguments.extend(extra)
        return build_arg_parser().parse_args(arguments)

    def test_rejects_cpu_and_nonzero_workers_for_formal_training(self):
        from dta.train_hybrid_affinity import validate_cli_args

        with tempfile.TemporaryDirectory() as temporary:
            cpu_args = self.parse_args(
                Path(temporary) / "cpu",
                extra=("--device", "cpu"),
            )
            with self.assertRaisesRegex(ValueError, "CUDA"):
                validate_cli_args(cpu_args)

            worker_args = self.parse_args(
                Path(temporary) / "workers",
                extra=("--num-workers", "1"),
            )
            with self.assertRaisesRegex(ValueError, "num_workers=0"):
                validate_cli_args(worker_args)

            budget_args = self.parse_args(
                Path(temporary) / "budget",
                extra=("--memory-budget-gb", "0"),
            )
            with self.assertRaisesRegex(ValueError, "memory budget"):
                validate_cli_args(budget_args)

    def test_rejects_nonempty_output_without_resume(self):
        from dta.train_hybrid_affinity import validate_cli_args

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "existing.txt").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                validate_cli_args(self.parse_args(output))

            resumed = self.parse_args(output, extra=("--resume",))
            validate_cli_args(resumed)

    def test_parser_rejects_unknown_fusion_and_rank_weight(self):
        from dta.train_hybrid_affinity import build_arg_parser

        base = [
            "--train-manifest",
            "train.json",
            "--valid-manifest",
            "valid.json",
            "--train-native-cache",
            "native_train",
            "--valid-native-cache",
            "native_valid",
            "--gerna-checkpoint",
            "gerna.pth",
            "--interface-checkpoint",
            "interface.pth",
            "--output-dir",
            "run",
            "--seed",
            "42",
            "--device",
            "cuda:0",
        ]
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_arg_parser().parse_args(
                    base
                    + [
                        "--fusion-type",
                        "unknown",
                        "--rank-weight",
                        "0.2",
                    ]
                )
            with self.assertRaises(SystemExit):
                build_arg_parser().parse_args(
                    base
                    + [
                        "--fusion-type",
                        "mlp",
                        "--rank-weight",
                        "0.3",
                    ]
                )

    def test_training_records_include_only_train_and_valid_bindings(self):
        from dta.train_hybrid_affinity import build_training_records

        with tempfile.TemporaryDirectory() as temporary:
            args = self.parse_args(Path(temporary) / "run")
            checkpoint_artifacts = {
                "gerna": {
                    "path": "/models/gerna.pth",
                    "sha256": "a" * 64,
                    "format": "state_dict",
                },
                "interface": {
                    "path": "/models/interface.pth",
                    "sha256": "b" * 64,
                    "format": "model_state_dict",
                },
            }
            preflights = {
                "train": {
                    "binding": {
                        "manifest_sha256": "c" * 64,
                        "native_metadata_sha256": "d" * 64,
                        "ordered_key_sha256": "e" * 64,
                        "sample_count": 10,
                    }
                },
                "valid": {
                    "binding": {
                        "manifest_sha256": "f" * 64,
                        "native_metadata_sha256": "1" * 64,
                        "ordered_key_sha256": "2" * 64,
                        "sample_count": 3,
                    }
                },
            }
            records = build_training_records(
                args,
                checkpoint_artifacts,
                preflights,
                group_size=4,
            )

            serialized = json.dumps(records, sort_keys=True)
            self.assertNotIn("test_manifest", serialized)
            self.assertNotIn("native_test", serialized)
            self.assertEqual(
                set(records["provenance"]["splits"]),
                {"train", "valid"},
            )
            self.assertEqual(
                records["provenance_hashes"]["gerna_checkpoint_sha256"],
                "a" * 64,
            )
            self.assertEqual(records["run_config"]["group_size"], 4)
            self.assertNotIn("stage_epochs", records["run_config"])
            short_args = self.parse_args(
                Path(temporary) / "short",
                extra=("--full-epochs", "0"),
            )
            short_records = build_training_records(
                short_args,
                checkpoint_artifacts,
                preflights,
                group_size=4,
            )
            self.assertEqual(
                records["run_config"], short_records["run_config"]
            )


class TestHybridTrainingEntrypoint(unittest.TestCase):
    def test_direct_script_help_runs_from_repository_root(self):
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [
                sys.executable,
                "dta/train_hybrid_affinity.py",
                "--help",
            ],
            cwd=str(repository),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--train-manifest", completed.stdout)
        self.assertNotIn("--test-path", completed.stdout)

    def test_builds_label_spread_train_and_ordered_valid_loaders(self):
        from dta.train_hybrid_affinity import build_hybrid_loaders

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, labels):
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                return {"key": str(index), "label": self.labels[index]}

        train_loader, valid_loader, generator = build_hybrid_loaders(
            Dataset([1.0, 2.0, 3.0, 4.0]),
            Dataset([5.0, 6.0]),
            group_size=2,
            seed=42,
            num_workers=0,
        )

        self.assertIsInstance(generator, torch.Generator)
        self.assertIs(train_loader.generator, generator)
        self.assertIs(valid_loader.generator, generator)
        self.assertEqual(train_loader.num_workers, 0)
        self.assertEqual(valid_loader.num_workers, 0)
        self.assertEqual(
            [sample["key"] for group in valid_loader for sample in group],
            ["0", "1"],
        )
        self.assertTrue(
            all(len(group) == 2 for group in train_loader)
        )

    def test_run_training_calls_fit_with_train_and_valid_only(self):
        import dta.train_hybrid_affinity as cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name in (
                "train_manifest",
                "valid_manifest",
                "gerna_checkpoint",
                "interface_checkpoint",
            ):
                path = root / (name + ".file")
                path.write_text("fixture", encoding="utf-8")
                paths[name] = path
            for split in ("train", "valid"):
                paths[f"{split}_manifest"].write_text(
                    json.dumps(
                        {
                            "dataset_type": "pdbbind_lmdb",
                            "split": split,
                            "samples": [],
                        }
                    ),
                    encoding="utf-8",
                )
            for name in ("train_native_cache", "valid_native_cache"):
                path = root / name
                path.mkdir()
                paths[name] = path
            output = root / "run"
            args = cli.build_arg_parser().parse_args(
                [
                    "--train-manifest",
                    str(paths["train_manifest"]),
                    "--valid-manifest",
                    str(paths["valid_manifest"]),
                    "--train-native-cache",
                    str(paths["train_native_cache"]),
                    "--valid-native-cache",
                    str(paths["valid_native_cache"]),
                    "--gerna-checkpoint",
                    str(paths["gerna_checkpoint"]),
                    "--interface-checkpoint",
                    str(paths["interface_checkpoint"]),
                    "--output-dir",
                    str(output),
                    "--fusion-type",
                    "transformer",
                    "--rank-weight",
                    "0.2",
                    "--seed",
                    "42",
                    "--device",
                    "cuda:0",
                ]
            )
            model = torch.nn.Linear(1, 1)
            artifacts = {
                "gerna": {
                    "path": str(paths["gerna_checkpoint"]),
                    "sha256": "a" * 64,
                    "format": "state_dict",
                },
                "interface": {
                    "path": str(paths["interface_checkpoint"]),
                    "sha256": "b" * 64,
                    "format": "model_state_dict",
                },
            }
            datasets = {
                "train": type("Dataset", (), {"labels": [1.0, 2.0]})(),
                "valid": type("Dataset", (), {"labels": [3.0]})(),
            }
            bindings = {
                "train": {
                    "binding": {
                        "manifest_sha256": "c" * 64,
                        "native_metadata_sha256": "d" * 64,
                        "ordered_key_sha256": "e" * 64,
                        "sample_count": 2,
                    },
                    "largest_indices": [0, 1, 0, 1],
                },
                "valid": {
                    "binding": {
                        "manifest_sha256": "f" * 64,
                        "native_metadata_sha256": "1" * 64,
                        "ordered_key_sha256": "2" * 64,
                        "sample_count": 1,
                    },
                    "largest_indices": [0],
                },
            }
            fit_calls = []

            with mock.patch.object(
                cli,
                "build_initialized_hybrid_model",
                return_value=(model, artifacts),
            ), mock.patch.object(
                cli,
                "build_hybrid_dataset",
                side_effect=[datasets["train"], datasets["valid"]],
            ), mock.patch.object(
                cli,
                "run_or_reuse_dataset_preflight",
                side_effect=[bindings["train"], bindings["valid"]],
            ), mock.patch.object(
                cli,
                "run_gpu_memory_preflight",
                return_value={"selected_group_size": 4},
            ), mock.patch.object(
                cli,
                "build_hybrid_loaders",
                return_value=("train_loader", "valid_loader", "generator"),
            ), mock.patch.object(
                torch.cuda, "is_available", return_value=True
            ), mock.patch.object(
                torch.cuda, "set_device"
            ), mock.patch.object(
                cli, "set_random_seeds"
            ) as seed_mock:
                with redirect_stdout(io.StringIO()):
                    result = cli.run_training(
                        args,
                        fit_fn=lambda **kwargs: fit_calls.append(kwargs)
                        or {"status": "complete"},
                    )

            self.assertEqual(result, {"status": "complete"})
            self.assertEqual(
                [call.args for call in seed_mock.call_args_list],
                [(42,), (42,)],
            )
            self.assertEqual(len(fit_calls), 1)
            call = fit_calls[0]
            self.assertEqual(call["train_loader"], "train_loader")
            self.assertEqual(call["valid_loader"], "valid_loader")
            self.assertFalse(
                any("test" in name.lower() for name in call)
            )


class TestSharedPreflightPaths(unittest.TestCase):
    def test_optional_shared_reports_override_per_run_defaults(self):
        from dta.train_hybrid_affinity import (
            build_arg_parser,
            resolve_dataset_preflight_paths,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            train_report = root / "shared" / "train.json"
            valid_report = root / "shared" / "valid.json"
            args = build_arg_parser().parse_args(
                [
                    "--train-manifest",
                    "train.json",
                    "--valid-manifest",
                    "valid.json",
                    "--train-native-cache",
                    "native_train",
                    "--valid-native-cache",
                    "native_valid",
                    "--gerna-checkpoint",
                    "gerna.pth",
                    "--interface-checkpoint",
                    "interface.pth",
                    "--output-dir",
                    str(output),
                    "--fusion-type",
                    "mlp",
                    "--rank-weight",
                    "0.1",
                    "--seed",
                    "42",
                    "--device",
                    "cuda:0",
                    "--train-preflight-report",
                    str(train_report),
                    "--valid-preflight-report",
                    str(valid_report),
                ]
            )

            paths = resolve_dataset_preflight_paths(args, output)
            self.assertEqual(paths["train"], train_report.resolve())
            self.assertEqual(paths["valid"], valid_report.resolve())


class TestMemoryPreflightIsolation(unittest.TestCase):
    def test_restores_parameters_buffers_and_training_mode(self):
        from dta.train_hybrid_affinity import (
            run_isolated_gpu_memory_preflight,
        )

        class StatefulModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))
                self.register_buffer("running", torch.tensor([2.0]))

        model = StatefulModel()
        model.train()
        original = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

        def mutate_runner(model, **kwargs):
            with torch.no_grad():
                model.weight.fill_(9.0)
                model.running.fill_(8.0)
            model.eval()
            return {"selected_group_size": 4}

        result = run_isolated_gpu_memory_preflight(
            model,
            runner=mutate_runner,
            dataset=[],
            largest_indices=[0, 1, 2, 3],
            device="cuda:0",
            report_path="unused.json",
        )

        self.assertEqual(result, {"selected_group_size": 4})
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))


    def test_default_probe_runs_autocast_forward_and_backward(self):
        from dta import hybrid_affinity_training as training
        from dta.train_hybrid_affinity import _run_gpu_probe_group

        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))
                self.forward_count = 0

            def to(self, device):
                return self

            def forward(self, gerna_inputs, interface_item):
                self.forward_count += 1
                return {
                    "prediction": self.weight * gerna_inputs["value"]
                }

        model = ProbeModel()
        backward_gradients = []
        model.weight.register_hook(
            lambda gradient: backward_gradients.append(
                gradient.detach().clone()
            )
        )
        samples = [
            {"key": "a", "label": 1.0},
            {"key": "b", "label": 2.0},
        ]

        def prepare(sample, device):
            value = torch.tensor([float(sample["label"])])
            return {"value": value}, {}, value

        amp_context = mock.MagicMock()
        amp_context.__enter__.return_value = None
        amp_context.__exit__.return_value = False
        scaler_events = []

        class FakeScaler:
            def scale(self, loss):
                scaler_events.append("scale")
                return loss

            def unscale_(self, optimizer):
                scaler_events.append("unscale")

            def step(self, optimizer):
                scaler_events.append("step")
                optimizer.step()

            def update(self):
                scaler_events.append("update")

        ema_events = []

        class FakeEMA:
            def __init__(self, model, decay):
                ema_events.append(("init", decay))

            def update(self, model):
                ema_events.append(("update", None))

        with mock.patch.object(
            training, "prepare_hybrid_sample", side_effect=prepare
        ), mock.patch.object(
            training, "ModelEMA", FakeEMA
        ), mock.patch.object(
            torch.cuda.amp,
            "autocast",
            return_value=amp_context,
        ) as autocast, mock.patch.object(
            torch.cuda.amp,
            "GradScaler",
            return_value=FakeScaler(),
        ) as scaler:
            _run_gpu_probe_group(model, samples, "cuda:0")

        autocast.assert_called_once_with(dtype=torch.float16)
        scaler.assert_called_once_with(enabled=True)
        self.assertEqual(
            scaler_events, ["scale", "unscale", "step", "update"]
        )
        self.assertEqual(
            ema_events, [("init", 0.999), ("update", None)]
        )
        self.assertEqual(model.forward_count, 2)
        self.assertEqual(len(backward_gradients), 1)
        self.assertIsNone(model.weight.grad)


class TestManifestSplitSafety(unittest.TestCase):
    def test_rejects_test_manifest_for_train_or_valid_role(self):
        from dta.train_hybrid_affinity import validate_manifest_split

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset_type": "pdbbind_lmdb",
                        "split": "test",
                        "samples": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected train"):
                validate_manifest_split(path, "train")
            with self.assertRaisesRegex(ValueError, "expected valid"):
                validate_manifest_split(path, "valid")

    def test_accepts_only_the_expected_manifest_role(self):
        from dta.train_hybrid_affinity import validate_manifest_split

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset_type": "pdbbind_lmdb",
                        "split": "train",
                        "samples": [],
                    }
                ),
                encoding="utf-8",
            )
            manifest = validate_manifest_split(path, "train")
            self.assertEqual(manifest["split"], "train")


    def test_rejects_train_valid_key_overlap(self):
        from dta.train_hybrid_affinity import (
            validate_train_valid_manifests,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for split, keys in (
                ("train", ["shared", "train_only"]),
                ("valid", ["shared", "valid_only"]),
            ):
                path = root / f"{split}.json"
                path.write_text(
                    json.dumps(
                        {
                            "dataset_type": "pdbbind_lmdb",
                            "split": split,
                            "samples": [
                                {"lmdb_key": key, "label": 1.0}
                                for key in keys
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                paths[split] = path
            with self.assertRaisesRegex(ValueError, "overlap.*shared"):
                validate_train_valid_manifests(
                    paths["train"], paths["valid"]
                )


class TestFinalHybridEvaluation(unittest.TestCase):
    @staticmethod
    def _sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _write_selection(self, root, valid_ci=0.71):
        root = Path(root)
        keys = ["a", "b", "c", "d"]
        from utils.experiment_integrity import hash_ordered_keys
        key_hash = hash_ordered_keys(keys)
        labels = np.asarray([1.0, 2.0, 3.0, 4.0])
        seeds = [1, 7, 21, 42, 87]
        model_paths = []
        model_hashes = []
        prediction_paths = []
        prediction_hashes = []
        provenance_hashes = {
            "gerna_checkpoint_sha256": "a" * 64,
            "interface_checkpoint_sha256": "b" * 64,
            "train_manifest_sha256": "c" * 64,
            "train_native_metadata_sha256": "d" * 64,
            "train_ordered_key_sha256": "e" * 64,
            "train_sample_count": 10,
            "valid_manifest_sha256": "f" * 64,
            "valid_native_metadata_sha256": "1" * 64,
            "valid_ordered_key_sha256": "2" * 64,
            "valid_sample_count": 4,
        }
        for index, seed in enumerate(seeds):
            run_dir = root / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            model_path = run_dir / "Model_hybrid_valid_ema.pth"
            torch.save(
                {"model_state_dict": {"weight": torch.tensor([seed])}},
                model_path,
            )
            prediction_path = run_dir / "selected_valid_predictions.npz"
            with prediction_path.open("wb") as handle:
                np.savez(
                    handle,
                    keys=np.asarray(keys),
                    labels=labels,
                    predictions=labels + 0.01 * index,
                )
            training_checkpoint = run_dir / "training_checkpoint.pth"
            training_checkpoint.write_bytes(
                f"training:{seed}".encode("ascii")
            )
            run_config = {
                "schema_version": 1,
                "fusion_type": "transformer",
                "rank_weight": 0.2,
                "seed": seed,
                "device": "cuda:0",
                "group_size": 4,
                "num_workers": 0,
                "amp": True,
            }
            candidate = {
                "schema_version": 1,
                "dataset": "LP-PDBBind",
                "selection_split": "valid",
                "run_dir": str(run_dir.resolve()),
                "fusion_type": "transformer",
                "rank_weight": 0.2,
                "seed": seed,
                "run_config": run_config,
                "provenance_hashes": provenance_hashes,
                "selected_variant": "ema",
                "epoch": 20,
                "stage": "full",
                "validation_metrics": {
                    "RMSE": 1.0,
                    "MAE": 0.8,
                    "Pearson": 0.7,
                    "Spearman": 0.7,
                    "CI": 0.71,
                },
                "model_path": str(model_path.resolve()),
                "model_sha256": self._sha256(model_path),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": self._sha256(prediction_path),
                "prediction_key_sha256": key_hash,
                "training_checkpoint": str(
                    training_checkpoint.resolve()
                ),
                "training_checkpoint_sha256": self._sha256(
                    training_checkpoint
                ),
            }
            (run_dir / "selected_candidate.json").write_text(
                json.dumps(candidate), encoding="utf-8"
            )
            model_paths.append(str(model_path.resolve()))
            model_hashes.append(self._sha256(model_path))
            prediction_paths.append(str(prediction_path.resolve()))
            prediction_hashes.append(self._sha256(prediction_path))

        ensemble_predictions = (
            root / "final_selection_valid_predictions.npz"
        )
        with ensemble_predictions.open("wb") as handle:
            np.savez(
                handle,
                keys=np.asarray(keys),
                labels=labels,
                predictions=labels,
            )
        selection = {
            "schema_version": 1,
            "dataset": "LP-PDBBind",
            "selection_split": "valid",
            "fusion_type": "transformer",
            "rank_weight": 0.2,
            "seeds": seeds,
            "model_paths": model_paths,
            "model_sha256": model_hashes,
            "prediction_paths": prediction_paths,
            "prediction_sha256": prediction_hashes,
            "weights": [0.2] * 5,
            "intercept": 0.0,
            "validation_metrics": {
                "RMSE": 0.9,
                "MAE": 0.7,
                "Pearson": 0.72,
                "Spearman": 0.71,
                "CI": float(valid_ci),
            },
            "prediction_key_sha256": key_hash,
            "validation_prediction_path": str(
                ensemble_predictions.resolve()
            ),
            "validation_prediction_sha256": self._sha256(
                ensemble_predictions
            ),
            "ensemble_candidate": "equal_weight",
            "l2_penalty": 0.001,
            "bounded_fit": {
                "objective": 0.1,
                "iterations": 10,
                "converged": True,
            },
        }
        selection_path = root / "final_selection.json"
        selection_path.write_text(
            json.dumps(selection), encoding="utf-8"
        )
        return selection_path, selection

    def _write_test_inputs(self, root):
        from utils.experiment_integrity import manifest_fingerprint

        root = Path(root)
        manifest_path = root / "test_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset_type": "pdbbind_lmdb",
                    "split": "test",
                    "samples": [
                        {
                            "lmdb_key": key,
                            "label": float(index + 1),
                        }
                        for index, key in enumerate(
                            ("a", "b", "c", "d")
                        )
                    ],
                }
            ),
            encoding="utf-8",
        )
        native_cache = root / "native_test"
        native_cache.mkdir()
        (native_cache / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "manifest": manifest_fingerprint(manifest_path),
                }
            ),
            encoding="utf-8",
        )
        return manifest_path, native_cache

    def test_final_parser_requires_explicit_test_confirmation(self):
        from dta.eval_hybrid_affinity import build_arg_parser

        parser = build_arg_parser()
        required = {
            action.dest
            for action in parser._actions
            if getattr(action, "required", False)
        }
        self.assertEqual(
            required,
            {
                "selection_manifest",
                "test_manifest",
                "test_native_cache",
                "output_dir",
                "device",
                "confirm_final_test",
            },
        )

    def test_final_request_rejects_missing_confirmation_and_nonempty_output(self):
        from dta.eval_hybrid_affinity import validate_final_request

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_path, _ = self._write_selection(root)
            manifest_path, native_cache = self._write_test_inputs(root)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "confirm"):
                validate_final_request(
                    selection_path,
                    manifest_path,
                    native_cache,
                    output,
                    confirm_final_test=False,
                    expected_test_manifest=manifest_path,
                    expected_test_native_cache=native_cache,
                )
            output.mkdir()
            (output / "old.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                validate_final_request(
                    selection_path,
                    manifest_path,
                    native_cache,
                    output,
                    confirm_final_test=True,
                    expected_test_manifest=manifest_path,
                    expected_test_native_cache=native_cache,
                )

    def test_selection_rejects_low_validation_ci_model_hash_and_config_change(self):
        from dta.eval_hybrid_affinity import (
            load_and_validate_selection,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_path, _ = self._write_selection(
                root / "low", valid_ci=0.69
            )
            with self.assertRaisesRegex(ValueError, "validation CI"):
                load_and_validate_selection(low_path)

            hash_path, hash_selection = self._write_selection(
                root / "hash"
            )
            Path(hash_selection["model_paths"][0]).write_bytes(
                b"changed"
            )
            with self.assertRaisesRegex(RuntimeError, "model hash"):
                load_and_validate_selection(hash_path)

            config_path, config_selection = self._write_selection(
                root / "config"
            )
            candidate_path = (
                Path(config_selection["model_paths"][0]).parent
                / "selected_candidate.json"
            )
            candidate = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
            candidate["run_config"]["rank_weight"] = 0.4
            candidate_path.write_text(
                json.dumps(candidate), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "config"):
                load_and_validate_selection(config_path)

    def test_one_time_evaluation_writes_exact_outputs_and_fixed_fusion_deltas(self):
        from dta.eval_hybrid_affinity import run_final_evaluation

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_path, _ = self._write_selection(root)
            manifest_path, native_cache = self._write_test_inputs(root)
            output = root / "final_output"
            calls = []

            def build_dataset(actual_manifest, actual_cache):
                self.assertEqual(
                    Path(actual_manifest), manifest_path.resolve()
                )
                self.assertEqual(
                    Path(actual_cache), native_cache.resolve()
                )
                return object()

            def evaluate_model(record, selection, dataset, device):
                calls.append(record["seed"])
                offset = 0.02 * calls[-1] / 87.0
                labels = np.asarray([1.0, 2.0, 3.0, 4.0])
                return {
                    "keys": ["a", "b", "c", "d"],
                    "labels": labels,
                    "predictions": labels + offset,
                }

            result = run_final_evaluation(
                selection_manifest_path=selection_path,
                test_manifest_path=manifest_path,
                test_native_cache_path=native_cache,
                output_dir=output,
                device="cuda:0",
                confirm_final_test=True,
                expected_test_manifest=manifest_path,
                expected_test_native_cache=native_cache,
                dataset_builder=build_dataset,
                model_evaluator=evaluate_model,
            )
            self.assertEqual(calls, [1, 7, 21, 42, 87])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "test_predictions.npz",
                    "final_test_metrics.json",
                    "final_selection_snapshot.json",
                },
            )
            metrics_payload = json.loads(
                (output / "final_test_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertAlmostEqual(
                metrics_payload[
                    "metric_deltas_vs_fixed_fusion"
                ]["CI"],
                metrics_payload["test_metrics"]["CI"] - 0.685273,
            )
            self.assertEqual(
                result["test_metrics"],
                metrics_payload["test_metrics"],
            )
            snapshot = json.loads(
                (output / "final_selection_snapshot.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                snapshot["selection_manifest_sha256"],
                self._sha256(selection_path),
            )
            with self.assertRaisesRegex(
                FileExistsError, "not empty"
            ):
                run_final_evaluation(
                    selection_manifest_path=selection_path,
                    test_manifest_path=manifest_path,
                    test_native_cache_path=native_cache,
                    output_dir=output,
                    device="cuda:0",
                    confirm_final_test=True,
                    expected_test_manifest=manifest_path,
                    expected_test_native_cache=native_cache,
                    dataset_builder=lambda *_: self.fail(
                        "dataset must not be reopened"
                    ),
                    model_evaluator=lambda *_: self.fail(
                        "model must not be reevaluated"
                    ),
                )


    def test_final_shell_requires_confirmation_and_fixed_gpu7_inputs(self):
        root = Path(__file__).resolve().parents[1]
        script = (
            root
            / "experiments/lp_pdbbind/performance_fusion_v1"
            / "evaluate_final_gpu7.sh"
        )
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        for required in (
            "GPU_ID=",
            ":-7",
            "CONFIRM_FINAL_TEST",
            "SELECTION_MANIFEST",
            "test_manifest.json",
            "native_test",
            "--confirm-final-test",
            "--device cuda:0",
        ):
            self.assertIn(required, text)
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
