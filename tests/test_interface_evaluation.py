from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestConcordanceIndexContract(unittest.TestCase):
    def test_ci_uses_fixed_prediction_tie_tolerance(self):
        from utils.metrics import concordance_index

        labels = np.array([1.0, 2.0, 3.0])
        predictions = np.array([0.0, 5e-9, 2.0])
        self.assertAlmostEqual(
            concordance_index(labels, predictions, tolerance=1e-8), 5.0 / 6.0
        )

    def test_ci_excludes_label_ties_within_tolerance(self):
        from utils.metrics import concordance_index

        labels = np.array([1.0, 1.0 + 5e-9, 2.0])
        predictions = np.array([0.0, 2.0, 3.0])
        self.assertEqual(concordance_index(labels, predictions, tolerance=1e-8), 1.0)

    def test_ci_rejects_invalid_inputs(self):
        from utils.metrics import concordance_index

        with self.assertRaisesRegex(ValueError, "same length"):
            concordance_index([1.0, 2.0], [1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            concordance_index([1.0, np.nan], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            concordance_index([1.0, 2.0], [1.0, np.inf])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            concordance_index([1.0, 2.0], [1.0, 2.0], tolerance=-1e-8)
        with self.assertRaisesRegex(ValueError, "finite"):
            concordance_index([1.0, 2.0], [1.0, 2.0], tolerance=np.nan)


class TestExperimentIntegrity(unittest.TestCase):
    def test_ordered_key_hash_is_stable_and_order_sensitive(self):
        from utils.experiment_integrity import hash_ordered_keys

        self.assertEqual(hash_ordered_keys(["a", "b"]), hash_ordered_keys(["a", "b"]))
        self.assertNotEqual(hash_ordered_keys(["a", "b"]), hash_ordered_keys(["b", "a"]))
        self.assertEqual(hash_ordered_keys(["a", "b"]), hashlib.sha256(b"a\0b\0").hexdigest())

    def test_sha256_file_matches_file_bytes(self):
        from utils.experiment_integrity import sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            payload = b"interface-first\0" * 100
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_atomic_write_json_creates_parents_and_replaces_existing_file(self):
        from utils.experiment_integrity import atomic_write_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "metadata.json"
            atomic_write_json(path, {"z": 1, "a": [2, 3]})
            first_text = path.read_text(encoding="utf-8")
            atomic_write_json(path, {"replacement": True})
            self.assertEqual(first_text, '{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}')
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"replacement": True})
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_atomic_write_json_removes_temp_file_after_serialization_error(self):
        from utils.experiment_integrity import atomic_write_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "metadata.json"
            with self.assertRaises(TypeError):
                atomic_write_json(path, {"not_serializable": object()})

            self.assertEqual(list(path.parent.iterdir()), [])

    def test_atomic_write_json_supports_concurrent_writers(self):
        from utils.experiment_integrity import atomic_write_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            barrier = threading.Barrier(2)
            real_dump = json.dump

            def synchronized_dump(payload, handle, **kwargs):
                barrier.wait(timeout=5)
                return real_dump(payload, handle, **kwargs)

            payloads = [{"writer": 1}, {"writer": 2}]
            with patch("utils.experiment_integrity.json.dump", side_effect=synchronized_dump):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(atomic_write_json, path, payload)
                        for payload in payloads
                    ]
                    for future in futures:
                        future.result(timeout=5)

            self.assertIn(json.loads(path.read_text(encoding="utf-8")), payloads)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_manifest_fingerprint_records_split_count_and_ordered_keys(self):
        from utils.experiment_integrity import hash_ordered_keys, manifest_fingerprint, sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = {
                "split": "train",
                "samples": [
                    {"lmdb_key": "sample_b", "label": 1.0},
                    {"lmdb_key": "sample_a", "label": 2.0},
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("builtins.open", wraps=open) as mocked_open:
                fingerprint = manifest_fingerprint(path)
            self.assertEqual(mocked_open.call_count, 1)
            self.assertEqual(fingerprint, {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "split": "train",
                "sample_count": 2,
                "ordered_key_sha256": hash_ordered_keys(["sample_b", "sample_a"]),
            })
            payload["samples"].reverse()
            path.write_text(json.dumps(payload), encoding="utf-8")
            reversed_fingerprint = manifest_fingerprint(path)
            self.assertNotEqual(
                fingerprint["ordered_key_sha256"],
                reversed_fingerprint["ordered_key_sha256"],
            )


def write_keyed_predictions(path, keys, labels, predictions):
    np.savez_compressed(
        path,
        keys=np.asarray(keys, dtype=str),
        labels=np.asarray(labels, dtype=np.float32),
        predictions=np.asarray(predictions, dtype=np.float32),
    )


class TestMatchedSeedBootstrap(unittest.TestCase):
    def test_fast_ci_matches_fixed_reference_with_ties(self):
        from scripts.summarize_interface_first_seeds import (
            _fast_concordance_index,
        )
        from utils.metrics import concordance_index

        labels = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0])
        predictions = np.asarray(
            [0.0, 1.0, 1.0 + 5e-9, 0.5, 4.0]
        )
        self.assertAlmostEqual(
            _fast_concordance_index(labels, predictions),
            concordance_index(labels, predictions),
        )

    def test_fast_ci_matches_reference_on_random_tied_arrays(self):
        from scripts.summarize_interface_first_seeds import (
            _fast_concordance_index,
        )
        from utils.metrics import concordance_index

        generator = np.random.RandomState(20260726)
        for size in (2, 3, 8, 31, 127):
            for tolerance in (0.0, 1e-8):
                for _ in range(20):
                    labels = generator.randint(-3, 4, size=size).astype(
                        np.float64
                    )
                    predictions = generator.randint(
                        -4, 5, size=size
                    ).astype(np.float64)
                    if tolerance:
                        offsets = generator.choice(
                            [0.0, 0.5e-8, 2.0e-8], size=size
                        )
                        labels += offsets
                        predictions -= offsets[::-1]
                    self.assertAlmostEqual(
                        _fast_concordance_index(
                            labels, predictions, tolerance=tolerance
                        ),
                        concordance_index(
                            labels, predictions, tolerance=tolerance
                        ),
                        places=14,
                    )

    def test_load_requires_unique_aligned_finite_rows(self):
        from scripts.summarize_interface_first_seeds import (
            load_keyed_predictions,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.npz"
            write_keyed_predictions(
                valid,
                ["a", "b", "c"],
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 2.9],
            )
            loaded = load_keyed_predictions(valid)
            self.assertEqual(loaded["keys"].tolist(), ["a", "b", "c"])
            self.assertEqual(loaded["labels"].dtype, np.float64)
            self.assertEqual(loaded["predictions"].dtype, np.float64)

            duplicate = root / "duplicate.npz"
            write_keyed_predictions(
                duplicate, ["a", "a"], [1.0, 2.0], [1.0, 2.0]
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_keyed_predictions(duplicate)

            nonfinite = root / "nonfinite.npz"
            write_keyed_predictions(
                nonfinite, ["a"], [1.0], [np.nan]
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                load_keyed_predictions(nonfinite)

    def test_seed_summary_requires_exact_key_and_label_alignment(self):
        from scripts.summarize_interface_first_seeds import (
            summarize_matched_seeds,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = {}
            interface = {}
            for seed, offset in ((42, 0.0), (43, 0.1)):
                baseline_path = root / f"baseline_{seed}.npz"
                interface_path = root / f"interface_{seed}.npz"
                write_keyed_predictions(
                    baseline_path,
                    ["a", "b", "c", "d"],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0 + offset, 4.0, 3.0],
                )
                write_keyed_predictions(
                    interface_path,
                    ["a", "b", "c", "d"],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0 + offset, 4.0],
                )
                baseline[seed] = baseline_path
                interface[seed] = interface_path

            summary = summarize_matched_seeds(baseline, interface)
            self.assertEqual(summary["seeds"], [42, 43])
            self.assertEqual(summary["sample_count"], 4)
            self.assertEqual(len(summary["per_seed"]), 2)
            self.assertIn("mean", summary["baseline"])
            self.assertIn("std", summary["interface_first"])
            self.assertIn("CI", summary["delta"]["mean"])

            reversed_path = root / "reversed.npz"
            write_keyed_predictions(
                reversed_path,
                ["d", "c", "b", "a"],
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
            )
            interface[43] = reversed_path
            with self.assertRaisesRegex(ValueError, "key alignment"):
                summarize_matched_seeds(baseline, interface)

    def test_paired_bootstrap_is_deterministic_and_truly_paired(self):
        from scripts.summarize_interface_first_seeds import (
            load_keyed_predictions,
            paired_bootstrap_delta_ci,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "same.npz"
            write_keyed_predictions(
                path,
                ["a", "b", "c", "d", "e"],
                [1, 2, 3, 4, 5],
                [1.2, 1.8, 3.1, 4.3, 4.7],
            )
            record = load_keyed_predictions(path)
            first = paired_bootstrap_delta_ci(
                [record, record],
                [record, record],
                n_bootstrap=64,
                seed=77,
            )
            second = paired_bootstrap_delta_ci(
                [record, record],
                [record, record],
                n_bootstrap=64,
                seed=77,
            )

        self.assertEqual(first, second)
        self.assertEqual(first["draw_count"], 64)
        self.assertTrue(
            np.array_equal(np.asarray(first["draws"]), np.zeros(64))
        )
        self.assertEqual(first["point_estimate"], 0.0)
        self.assertEqual(first["interval_95"], [0.0, 0.0])

    def test_group_draw_retains_all_complexes_in_each_sampled_group(self):
        from scripts.summarize_interface_first_seeds import (
            _sample_group_indices,
        )

        groups = np.asarray(["target_a", "target_a", "target_b"])
        generator = np.random.RandomState(11)
        for _ in range(20):
            indices = _sample_group_indices(groups, generator)
            self.assertEqual(
                int(np.sum(indices == 0)), int(np.sum(indices == 1))
            )
            self.assertGreater(len(indices), 0)

    def test_exact_target_group_is_a_chain_multiset_hash(self):
        from scripts.summarize_interface_first_seeds import (
            exact_target_groups,
        )

        keys = ["complex_a", "complex_b", "complex_c"]
        chain_sequences = {
            "complex_a": [" aa a ", "BBB"],
            "complex_b": ["bbb", "AAA"],
            "complex_c": ["AAA", "AAA"],
        }
        groups = exact_target_groups(keys, chain_sequences)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertEqual(len(groups[0]), 64)

    def test_related_target_clustering_freezes_mmseqs_thresholds(self):
        from scripts.summarize_interface_first_seeds import (
            run_mmseqs_target_clustering,
        )

        commands = []

        def fake_runner(command, **kwargs):
            commands.append(list(command))
            if command[1] == "--version":
                return SimpleNamespace(stdout="MMseqs2 15.6f452\n")
            fasta_path = Path(command[2])
            prefix = Path(command[3])
            sequence_ids = [
                line[1:].strip()
                for line in fasta_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.startswith(">")
            ]
            cluster_path = Path(str(prefix) + "_cluster.tsv")
            cluster_path.write_text(
                "".join(
                    f"{sequence_ids[0]}\t{sequence_id}\n"
                    for sequence_id in sequence_ids
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(stdout="")

        with tempfile.TemporaryDirectory() as temporary:
            result = run_mmseqs_target_clustering(
                ["a", "b"],
                {"a": ["AAAA"], "b": ["AAAT"]},
                temporary,
                runner=fake_runner,
            )
            metadata = json.loads(
                (Path(temporary) / "mmseqs_metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["group_ids"][0], result["group_ids"][1])
        easy_cluster = commands[1]
        self.assertIn("--min-seq-id", easy_cluster)
        self.assertEqual(
            easy_cluster[easy_cluster.index("--min-seq-id") + 1], "0.9"
        )
        self.assertIn("-c", easy_cluster)
        self.assertEqual(easy_cluster[easy_cluster.index("-c") + 1], "0.8")
        self.assertEqual(metadata["version"], "MMseqs2 15.6f452")

    def test_interpretation_is_conservative_for_target_uncertainty(self):
        from scripts.summarize_interface_first_seeds import (
            interpret_bootstrap,
        )

        interpretation = interpret_bootstrap(
            {"interval_95": [0.01, 0.04]},
            {"interval_95": [-0.01, 0.05]},
            {"interval_95": [0.005, 0.06]},
        )
        self.assertEqual(
            interpretation["status"], "target_repeat_driven"
        )
        self.assertFalse(
            interpretation["stable_cross_target_improvement"]
        )


class TestFrozenInterfaceFirstRunScripts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_root = (
            ROOT / "experiments" / "lp_pdbbind" / "interface_first_v1"
        )

    def read_script(self, name):
        path = self.run_root / name
        self.assertTrue(path.is_file(), f"missing run script: {path}")
        return path.read_text(encoding="utf-8")

    def test_all_frozen_run_scripts_exist(self):
        expected = {
            "build_caches.sh",
            "smoke_train_only.sh",
            "pretrain_stage1.sh",
            "finetune_seed.sh",
            "run_matched_seeds.sh",
            "summarize.sh",
        }
        self.assertEqual(
            {path.name for path in self.run_root.glob("*.sh")},
            expected,
        )

    def test_cache_script_uses_only_official_lp_manifests(self):
        text = self.read_script("build_caches.sh")
        for split in ("train", "valid", "test"):
            self.assertIn(
                "data/PDBbind_chain_esm_residue_DTA/"
                f"lp_pdbbind/{split}_manifest.json",
                text,
            )
        self.assertIn("build_local_interface_diagnostic_split.py", text)
        self.assertIn("build_local_interface_cache.py", text)
        self.assertIn("build_local_interface_decoys.py", text)
        self.assertIn("--shell-budget 128", text)
        self.assertIn("--resume", text)
        self.assertIn("metadata.json", text)
        self.assertIn("decoy_report.json", text)

    def test_cache_script_is_compatible_with_bash42_nounset(self):
        text = self.read_script("build_caches.sh")
        self.assertNotIn("resume_args=()", text)
        self.assertNotIn("decoy_resume_args=()", text)
        self.assertNotIn('${resume_args[@]}', text)
        self.assertNotIn('${decoy_resume_args[@]}', text)

    def test_train_only_smoke_never_references_official_valid_or_test(self):
        text = self.read_script("smoke_train_only.sh")
        self.assertIn("--train-manifest", text)
        self.assertIn("--diagnostic-keys", text)
        self.assertIn("--epochs 1", text)
        self.assertNotIn("valid_manifest", text)
        self.assertNotIn("test_manifest", text)
        self.assertNotIn("--amp", text)
        self.assertIn('GPU_ID="${GPU_ID:?', text)

    def test_stage1_script_freezes_complete_pretraining_contract(self):
        text = self.read_script("pretrain_stage1.sh")
        self.assertIn("--train-manifest", text)
        self.assertIn("--native-lmdb", text)
        self.assertIn("--decoy-lmdb", text)
        self.assertIn("--diagnostic-keys", text)
        self.assertIn("--epochs 30", text)
        self.assertIn("--temperature 0.2", text)
        self.assertIn("--consistency-weight 0.1", text)
        self.assertNotIn("valid_manifest", text)
        self.assertNotIn("test_manifest", text)
        self.assertNotIn("--amp", text)
        self.assertIn("Stage 1 is already complete", text)

    def test_stage2_script_freezes_lp_interface_first_contract(self):
        text = self.read_script("finetune_seed.sh")
        for split in ("train", "valid", "test"):
            self.assertIn(f"{split}_manifest.json", text)
            self.assertIn(
                f"--interface_{split}_native_lmdb", text
            )
        required_fragments = (
            "--model_arch interface_first",
            "--interface_stage1_checkpoint",
            "--interface_diagnostic_report",
            "--interface_stage2_ready",
            "--interface_warmup_epochs 3",
            "--batch_size 1",
            "--epoch 30",
            "--loss smooth_l1",
            "--early_stop_patience 0",
            "--rank_loss_weight 0",
            "--select_metric CI",
            "LP_PDBBind_InterfaceFirstGerNA_v1_seed${SEED}",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, text)
        self.assertNotIn("--amp", text)

    def test_matched_seed_and_summary_scripts_freeze_seeds_42_to_46(self):
        run_text = self.read_script("run_matched_seeds.sh")
        summary_text = self.read_script("summarize.sh")
        for text in (run_text, summary_text):
            self.assertIn("SEEDS=(42 43 44 45 46)", text)
        self.assertIn("pretrain_stage1.sh", run_text)
        self.assertIn("finetune_seed.sh", run_text)
        self.assertIn("stage2_ready.json", run_text)
        self.assertIn("skipping Stage 1", run_text)
        self.assertIn("--baseline-prediction", summary_text)
        self.assertIn("--interface-prediction", summary_text)
        self.assertIn("test_predictions.npz", summary_text)
        self.assertIn("--bootstrap-draws 2000", summary_text)

    def test_experiment_record_marks_results_as_pending(self):
        text = (
            ROOT / "docs" / "LP_PDBBind_experiment_record.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Interface-first GerNA v1", text)
        self.assertIn("尚未产生 validation/test CI", text)
        self.assertIn("做了什么", text)
        self.assertIn("结果怎么样", text)
        self.assertIn("结果分析", text)


class InterfaceFirstV2RunScriptContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_root = (
            ROOT / "experiments" / "lp_pdbbind" / "interface_first_v2"
        )

    def read_script(self, name):
        path = self.run_root / name
        self.assertTrue(path.is_file(), f"missing run script: {path}")
        return path.read_text(encoding="utf-8")

    def test_all_v2_scripts_and_smoke_comparator_exist(self):
        expected = {
            "build_diagnostic_cache.sh",
            "build_train_cache.sh",
            "smoke_paired.sh",
            "pretrain_stage1.sh",
            "finetune_seed.sh",
            "run_matched_seeds.sh",
            "summarize.sh",
        }
        self.assertEqual(
            {path.name for path in self.run_root.glob("*.sh")},
            expected,
        )
        comparator = ROOT / "scripts" / "compare_interface_v2_smokes.py"
        self.assertTrue(comparator.is_file())
        comparator_text = comparator.read_text(encoding="utf-8")
        for field in (
            "initial_state_sha256",
            "stage1_included_ordered_key_sha256",
            "optimizer",
            "paired_smoke_report.json",
        ):
            self.assertIn(field, comparator_text)

    def test_diagnostic_and_smoke_scripts_are_strictly_train_only(self):
        diagnostic = self.read_script("build_diagnostic_cache.sh")
        smoke = self.read_script("smoke_paired.sh")
        train_cache = self.read_script("build_train_cache.sh")
        for text in (diagnostic, smoke, train_cache):
            self.assertIn("train_manifest.json", text)
            self.assertNotIn("valid_manifest", text)
            self.assertNotIn("test_manifest", text)
        self.assertIn("--cohort diagnostic", diagnostic)
        self.assertIn("--require-pass", diagnostic)
        self.assertIn("--cohort train", train_cache)
        self.assertIn("decoy_v2_data_diagnostic.json", train_cache)
        self.assertIn("GPU_ID=\"$" "{GPU_ID:?", smoke)
        self.assertIn("CUDA_VISIBLE_DEVICES", smoke)
        self.assertIn("--device cuda:0", smoke)
        self.assertNotIn("--amp", smoke)

    def test_optional_resume_arguments_are_compatible_with_bash42_nounset(self):
        for name in (
            "build_diagnostic_cache.sh",
            "build_train_cache.sh",
            "pretrain_stage1.sh",
            "finetune_seed.sh",
        ):
            text = self.read_script(name)
            self.assertNotIn("resume_args=()", text, name)
            self.assertNotIn("${resume_args[@]}", text, name)

    def test_paired_smoke_and_formal_stage1_select_the_frozen_banks(self):
        smoke = self.read_script("smoke_paired.sh")
        formal = self.read_script("pretrain_stage1.sh")
        for bank in ("random_physical", "matched"):
            self.assertIn(f"--decoy-bank {bank}", smoke)
        self.assertIn("--require-paired-bank-intersection", smoke)
        self.assertIn("--decoy-schema-version 2", smoke)
        self.assertIn("--diagnostic-decoy-lmdb", smoke)
        self.assertIn("--data-diagnostic-report", smoke)
        self.assertIn("--decoy-schema-version 2", formal)
        self.assertIn("--decoy-bank matched", formal)
        self.assertNotIn("--require-paired-bank-intersection", formal)
        self.assertIn("--epochs 30", formal)
        self.assertIn("--device cuda:0", formal)
        self.assertIn("GPU_ID=\"$" "{GPU_ID:?", formal)
        self.assertNotIn("valid_manifest", formal)
        self.assertNotIn("test_manifest", formal)

    def test_formal_stage2_and_seed_scripts_preserve_lp_contract(self):
        finetune = self.read_script("finetune_seed.sh")
        run = self.read_script("run_matched_seeds.sh")
        summary = self.read_script("summarize.sh")
        for split in ("train", "valid", "test"):
            self.assertIn(f"{split}_manifest.json", finetune)
        for value in (
            "--model_arch interface_first",
            "--interface_stage1_checkpoint",
            "--interface_diagnostic_report",
            "--interface_stage2_ready",
            "--epoch 30",
            "--loss smooth_l1",
            "--early_stop_patience 0",
            "--rank_loss_weight 0",
            "--select_metric CI",
        ):
            self.assertIn(value, finetune)
        self.assertIn("SEEDS=(42 43 44 45 46)", run)
        self.assertIn("pretrain_stage1.sh", run)
        self.assertIn("finetune_seed.sh", run)
        self.assertIn("SEEDS=(42 43 44 45 46)", summary)
        self.assertIn("--bootstrap-draws 2000", summary)


class InterfaceFirstV2SmokeComparatorTest(unittest.TestCase):
    @staticmethod
    def _write_smoke_run(
        root, bank, initial_state_sha256="a" * 64
    ):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        arguments = {
            "seed": 42,
            "epochs": 1,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "num_workers": 0,
            "grad_clip": 1.0,
            "temperature": 0.2,
            "consistency_weight": 0.1,
            "ligand_gnn_depth": 3,
            "ligand_dropout": 0.1,
            "device": "cuda:0",
            "decoy_schema_version": 2,
            "decoy_bank": bank,
            "require_paired_bank_intersection": True,
        }
        provenance = {
            "decoy_schema_version": 2,
            "stage1_included_ordered_key_sha256": "b" * 64,
            "model": {
                "initial_state_sha256": initial_state_sha256,
                "ligand_gnn_depth": 3,
                "ligand_dropout": 0.1,
            },
            "training": {
                "epochs": 1,
                "seed": 42,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "gradient_clip": 1.0,
                "pose_temperature": 0.2,
                "consistency_weight": 0.1,
            },
            "required_complete_banks": [
                "random_physical",
                "matched",
            ],
        }
        (root / "run_config.json").write_text(
            json.dumps(
                {"arguments": arguments, "provenance": provenance}
            ),
            encoding="utf-8",
        )
        (root / "metrics_history.csv").write_text(
            "epoch,total_loss,pose_loss,consistency_loss,pose_accuracy,"
            "score_margin,gradient_norm,jitter_identity_fallbacks,"
            "samples,seconds\n"
            "1,0.5,0.4,0.1,0.8,0.6,1.0,0,10,2.0\n",
            encoding="utf-8",
        )
        metrics = {
            "interface_auroc": 0.7,
            "pose_accuracy": 0.8,
            "score_margin": 0.6,
            "coarse_linear_cv_auroc": 0.55,
            "coarse_nonlinear_cv_auroc": 0.60,
            "heldout_linear_cv_auroc": 0.52,
            "heldout_nonlinear_cv_auroc": 0.58,
            "interface_minus_coarse_linear_auroc": 0.15,
            "interface_minus_coarse_nonlinear_auroc": 0.10,
            "interface_minus_heldout_linear_auroc": 0.18,
            "interface_minus_heldout_nonlinear_auroc": 0.12,
        }
        (root / "model_diagnostic_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "complete",
                    "descriptive_only": True,
                    "seed": 42,
                    "bank": bank,
                    "cohort": "paired_intersection",
                    "paired_intersection": True,
                    "key_count": 2,
                    "ordered_key_sha256": "c" * 64,
                    "row_count": 8,
                    "metrics": metrics,
                    "gate": {"passed": True, "reasons": []},
                }
            ),
            encoding="utf-8",
        )

    def test_paired_comparator_writes_descriptive_delta_report(self):
        from scripts.compare_interface_v2_smokes import (
            compare_paired_smokes,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            random_root = root / "random"
            matched_root = root / "matched"
            self._write_smoke_run(random_root, "random_physical")
            self._write_smoke_run(matched_root, "matched")

            report = compare_paired_smokes(
                random_root, matched_root, root / "output"
            )

            report_path = root / "output" / "paired_smoke_report.json"
            self.assertTrue(report_path.is_file())
            self.assertEqual(
                report["paired_provenance"][
                    "initial_state_sha256"
                ],
                "a" * 64,
            )
            self.assertEqual(
                report["delta"]["stage1"]["total_loss"], 0.0
            )
            self.assertEqual(
                report["banks"]["matched"]["model_diagnostic"][
                    "interface_auroc"
                ],
                0.7,
            )

    def test_paired_comparator_rejects_mismatched_initial_state(self):
        from scripts.compare_interface_v2_smokes import (
            compare_paired_smokes,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_smoke_run(
                root / "random", "random_physical", "a" * 64
            )
            self._write_smoke_run(
                root / "matched", "matched", "b" * 64
            )

            with self.assertRaisesRegex(ValueError, "initial_state_sha256"):
                compare_paired_smokes(
                    root / "random", root / "matched", root / "output"
                )



if __name__ == "__main__":
    unittest.main()
