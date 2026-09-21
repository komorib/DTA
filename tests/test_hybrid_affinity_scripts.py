import hashlib
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    ROOT / "experiments" / "lp_pdbbind" / "performance_fusion_v1"
)
SCRIPTS = (
    "smoke_gpu7.sh",
    "run_seed42_grid_gpu7.sh",
    "continue_seed42_top2_gpu7.sh",
    "run_final_seeds_gpu7.sh",
)


class TestHybridAffinityExperimentScripts(unittest.TestCase):
    def read_script(self, name):
        return (SCRIPT_ROOT / name).read_text(encoding="utf-8")

    def embedded_python(self, name):
        text = self.read_script(name)
        marker = "<<'PY'\n"
        return text.split(marker, 1)[1].split("\nPY\n", 1)[0]

    def run_embedded_python(self, name, *arguments):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                self.embedded_python(name),
                *[str(argument) for argument in arguments],
            ],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_all_scripts_have_gpu7_guards_and_no_test_inputs(self):
        for name in SCRIPTS:
            with self.subTest(name=name):
                text = self.read_script(name)
                self.assertIn("set -euo pipefail", text)
                self.assertIn("GPU_ID=\"${GPU_ID:-7}\"", text)
                self.assertIn("fixed_inputs=(", text)
                self.assertIn(
                    "CUDA_VISIBLE_DEVICES=\"${GPU_ID}\"", text
                )
                self.assertIn("--device", text)
                self.assertIn("cuda:0", text)
                self.assertNotIn("test_manifest", text)
                self.assertNotIn("native_test", text)
                self.assertNotIn("--test", text)

    def test_smoke_uses_deterministic_subsets_and_short_stages(self):
        text = self.read_script("smoke_gpu7.sh")
        self.assertIn("SMOKE_SAMPLE_COUNT=16", text)
        self.assertIn("SMOKE_SEED=42", text)
        self.assertIn("--head-epochs 1", text)
        self.assertIn("--high-epochs 1", text)
        self.assertIn("--full-epochs 1", text)
        self.assertIn("--fusion-type transformer", text)
        self.assertIn("--rank-weight 0.2", text)
        self.assertIn("/tmp/", text)
        self.assertIn("memory_preflight.json", text)
        self.assertIn("best_metrics.json", text)

    def test_grid_runs_all_six_high_stage_candidates(self):
        text = self.read_script("run_seed42_grid_gpu7.sh")
        self.assertIn("for fusion in mlp transformer", text)
        self.assertIn("for rank_weight in 0.1 0.2 0.4", text)
        self.assertIn("--head-epochs 2", text)
        self.assertIn("--high-epochs 15", text)
        self.assertIn("--full-epochs 0", text)
        self.assertIn("--seed 42", text)
        self.assertIn("--train-preflight-report", text)
        self.assertIn("--valid-preflight-report", text)

    def test_continuation_requires_two_hashed_checkpoint_entries(self):
        text = self.read_script("continue_seed42_top2_gpu7.sh")
        self.assertIn("dev_selection_top2.json", text)
        self.assertIn("len(entries) != 2", text)
        self.assertIn("training_checkpoint_sha256", text)
        self.assertIn("--full-epochs 40", text)
        self.assertIn("--resume", text)

    def test_continuation_validator_accepts_checkpoint_after_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev_root = (
                root
                / "Model"
                / "LP_PDBBind_HybridPerformance_v1"
                / "dev_seed42"
            )
            entries = []
            checkpoints = []
            for fusion, rank_weight in (
                ("mlp", "0.1"),
                ("transformer", "0.2"),
            ):
                run_dir = dev_root / f"{fusion}_rank{rank_weight}"
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "training_checkpoint.pth"
                checkpoint.write_bytes(
                    f"{fusion}:{rank_weight}".encode("ascii")
                )
                (run_dir / "run_config.json").write_text(
                    json.dumps(
                        {
                            "fusion_type": fusion,
                            "rank_weight": float(rank_weight),
                            "seed": 42,
                        }
                    ),
                    encoding="utf-8",
                )
                entries.append(
                    {
                        "fusion_type": fusion,
                        "rank_weight": float(rank_weight),
                        "run_dir": str(run_dir),
                        "training_checkpoint": str(checkpoint),
                        "training_checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                )
                checkpoints.append(checkpoint)
            selection = root / "selection.json"
            selection.write_text(
                json.dumps({"entries": entries}), encoding="utf-8"
            )

            first = self.run_embedded_python(
                "continue_seed42_top2_gpu7.sh",
                selection,
                root,
                dev_root,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            checkpoints[0].write_bytes(b"checkpoint after one full epoch")
            second = self.run_embedded_python(
                "continue_seed42_top2_gpu7.sh",
                selection,
                root,
                dev_root,
            )
            self.assertEqual(second.returncode, 0, second.stderr)

    def test_smoke_rejects_source_cache_manifest_mismatch(self):
        from utils.experiment_integrity import manifest_fingerprint

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = []
            for split in ("train", "valid"):
                manifest = root / f"{split}.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "split": split,
                            "lmdb_path": str(root / "source.mdb"),
                            "samples": [
                                {
                                    "lmdb_key": f"{split}-{index}",
                                    "label": float(index),
                                }
                                for index in range(16)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                cache = root / f"source_{split}"
                cache.mkdir()
                (cache / "data.mdb").write_bytes(b"data")
                (cache / "lock.mdb").write_bytes(b"lock")
                fingerprint = manifest_fingerprint(manifest)
                if split == "train":
                    fingerprint = dict(fingerprint)
                    fingerprint["sha256"] = "0" * 64
                (cache / "metadata.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "manifest": fingerprint,
                        }
                    ),
                    encoding="utf-8",
                )
                arguments.extend((manifest, cache))

            completed = self.run_embedded_python(
                "smoke_gpu7.sh",
                arguments[0],
                arguments[1],
                arguments[2],
                arguments[3],
                root / "output_train.json",
                root / "output_native_train",
                root / "output_valid.json",
                root / "output_native_valid",
                16,
                42,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "source native cache manifest provenance does not match",
                completed.stderr,
            )

    def test_relative_selection_paths_are_rooted_at_repository(self):
        for name in (
            "continue_seed42_top2_gpu7.sh",
            "run_final_seeds_gpu7.sh",
        ):
            with self.subTest(name=name):
                text = self.read_script(name)
                self.assertIn(
                    r'SELECTION_JSON="${ROOT}/${SELECTION_JSON}"', text
                )

    def test_continuation_enters_root_before_repo_python_import(self):
        text = self.read_script("continue_seed42_top2_gpu7.sh")
        self.assertLess(
            text.index('cd "${ROOT}"'),
            text.index(
                '"${PY}" - "${SELECTION_JSON}" "${ROOT}" "${DEV_ROOT}"'
            ),
        )

    def test_final_script_uses_frozen_winner_and_five_seeds(self):
        text = self.read_script("run_final_seeds_gpu7.sh")
        self.assertIn("winner.json", text)
        self.assertIn("len(entries) != 1", text)
        self.assertIn("for seed in 1 7 21 42 87", text)
        self.assertIn("--head-epochs 2", text)
        self.assertIn("--high-epochs 15", text)
        self.assertIn("--full-epochs 40", text)
        self.assertIn("resume_args=(--resume)", text)

    def test_all_scripts_pass_bash_syntax_check(self):
        for name in SCRIPTS:
            with self.subTest(name=name):
                completed = subprocess.run(
                    ["bash", "-n", str(SCRIPT_ROOT / name)],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stderr
                )


if __name__ == "__main__":
    unittest.main()
