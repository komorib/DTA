import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    REPO_ROOT / "experiments" / "lp_pdbbind" / "performance_fusion_v8"
)
SCRIPT_NAMES = (
    "common.sh",
    "smoke_gpu7.sh",
    "run_repeated_rank_fusion_gpu7.sh",
)


class RepeatedRankFusionV8ScriptTest(unittest.TestCase):
    def read(self, name):
        return (SCRIPT_ROOT / name).read_text(encoding="utf-8")

    def test_scripts_exist_and_parse_with_bash(self):
        for name in SCRIPT_NAMES:
            path = SCRIPT_ROOT / name
            self.assertTrue(path.is_file(), name)
            subprocess.run(
                ["bash", "-n", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_scripts_pin_gpu7_and_keep_v7_read_only(self):
        combined = "\n".join(self.read(name) for name in SCRIPT_NAMES)
        self.assertIn('GPU_ID="${GPU_ID:-7}"', combined)
        self.assertIn('export CUDA_VISIBLE_DEVICES="${GPU_ID}"', combined)
        self.assertIn('DEVICE="cuda:0"', combined)
        self.assertIn('MEMORY_BUDGET_GB="40"', combined)
        self.assertIn("LP_PDBBind_RankUnlockedFusion_v7", combined)
        self.assertIn("LP_PDBBind_RepeatedRankFusion_v8", combined)
        self.assertNotIn('mkdir -p "${V7_ROOT}', combined)

    def test_smoke_uses_small_repeat_configuration(self):
        smoke = self.read("smoke_gpu7.sh")
        for argument in (
            "--limit 64",
            "--smoke",
            "--maximum-epochs 2",
            "--patience 2",
            "--batch-size 32",
            "--model-width 64",
            "--transformer-layers 1",
            "--attention-heads 4",
            "--maximum-pairs 4096",
        ):
            self.assertIn(argument, smoke)
        self.assertIn("/tmp/LP_PDBBind_RepeatedRankFusion_v8_", smoke)

    def test_formal_uses_safe_resume_and_full_configuration(self):
        formal = self.read("run_repeated_rank_fusion_gpu7.sh")
        self.assertIn('RESUME="${RESUME:-0}"', formal)
        self.assertIn('[[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]', formal)
        self.assertIn('if [[ "${RESUME}" == "1" ]]; then', formal)
        self.assertIn("command+=(--resume)", formal)
        self.assertIn('"${command[@]}"', formal)
        self.assertNotIn("resume_args", formal)
        for argument in (
            "--maximum-epochs 160",
            "--patience 20",
            "--batch-size 256",
            "--model-width 256",
            "--transformer-layers 3",
            "--attention-heads 8",
            "--maximum-pairs 200000",
        ):
            self.assertIn(argument, formal)

    def test_readme_documents_smoke_formal_resume_and_result(self):
        readme = self.read("README.md")
        for value in (
            "GPU_ID=7",
            "RESUME=0",
            "RESUME=1",
            "smoke_gpu7.sh",
            "run_repeated_rank_fusion_gpu7.sh",
            "Model/LP_PDBBind_RepeatedRankFusion_v8/run/result.json",
            "Model/LP_PDBBind_RankUnlockedFusion_v7/run",
        ):
            self.assertIn(value, readme)


if __name__ == "__main__":
    unittest.main()
