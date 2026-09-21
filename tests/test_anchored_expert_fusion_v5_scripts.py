import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    ROOT / "experiments" / "lp_pdbbind" / "performance_fusion_v5"
)
SCRIPT_NAMES = (
    "common.sh",
    "smoke_gpu7.sh",
    "run_anchored_fusion_gpu7.sh",
)


class AnchoredExpertFusionV5ScriptTest(unittest.TestCase):
    def read(self, name):
        return (SCRIPT_ROOT / name).read_text(encoding="utf-8")

    def test_scripts_parse_and_pin_gpu7_cuda0_and_40gb(self):
        for name in SCRIPT_NAMES:
            path = SCRIPT_ROOT / name
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("set -euo pipefail", self.read(name))
        combined = "\n".join(self.read(name) for name in SCRIPT_NAMES)
        self.assertIn('GPU_ID="${GPU_ID:-7}"', combined)
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPU_ID}"', combined)
        self.assertIn(
            'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
            self.read("common.sh"),
        )
        self.assertIn('DEVICE="cuda:0"', combined)
        self.assertIn('MEMORY_BUDGET_GB="40"', combined)

    def test_scripts_use_read_only_v4_cache_and_isolated_v5_outputs(self):
        common = self.read("common.sh")
        self.assertIn("data/LP_PDBBind_RankFusionCache_v4", common)
        self.assertIn("Model/LP_PDBBind_AnchoredExpertFusion_v5", common)
        self.assertIn('TRAIN_CACHE="${V4_CACHE_ROOT}/joined/train"', common)
        self.assertIn('VALID_CACHE="${V4_CACHE_ROOT}/joined/valid"', common)
        self.assertIn('TEST_CACHE="${V4_CACHE_ROOT}/joined/test"', common)
        self.assertNotIn("LP_PDBBind_RankFusion_v4", common)
        for name in ("smoke_gpu7.sh", "run_anchored_fusion_gpu7.sh"):
            text = self.read(name)
            self.assertNotIn("mkdir -p \"${V4_CACHE_ROOT}", text)
            self.assertNotIn("> \"${TRAIN_CACHE}", text)
            self.assertNotIn("> \"${VALID_CACHE}", text)
            self.assertNotIn("> \"${TEST_CACHE}", text)

    def test_scripts_do_not_expand_optional_empty_arrays_or_eval_strings(self):
        combined = "\n".join(self.read(name) for name in SCRIPT_NAMES)
        self.assertNotIn("limit_args[@]", combined)
        self.assertNotIn("resume_args[@]", combined)
        self.assertNotIn(" eval ", combined)
        self.assertNotIn("bash -c", combined)

    def test_smoke_is_validation_only_with_64_rows_and_two_formal_seeds(self):
        text = self.read("smoke_gpu7.sh")
        self.assertIn("dta.anchored_expert_fusion_v5.run_validation", text)
        self.assertIn("--limit 64", text)
        self.assertIn("--seeds 1 7", text)
        self.assertIn("--maximum-epochs 1", text)
        self.assertIn("--patience 1", text)
        self.assertNotIn("evaluate_test", text)
        self.assertNotIn("TEST_CACHE", text)
        self.assertNotIn("official_test", text)

    def test_formal_uses_one_command_array_then_verifies_and_evaluates_once(self):
        text = self.read("run_anchored_fusion_gpu7.sh")
        self.assertIn("validation_command=(", text)
        self.assertIn('validation_command+=(--resume)', text)
        self.assertIn('"${validation_command[@]}"', text)
        self.assertIn("--train-count 11037", text)
        self.assertIn("--valid-count 2340", text)
        self.assertIn("--seeds 1 7 21 42 87", text)
        validation = text.index('"${validation_command[@]}"')
        verification = text.index("verify_validation_seal")
        evaluation = text.index("dta.anchored_expert_fusion_v5.evaluate_test")
        self.assertLess(validation, verification)
        self.assertLess(verification, evaluation)
        self.assertEqual(
            text.count("dta.anchored_expert_fusion_v5.evaluate_test"), 1
        )
        self.assertIn('--test-cache "${TEST_CACHE}"', text)
        self.assertIn('--output "${OFFICIAL_TEST_ROOT}"', text)


if __name__ == "__main__":
    unittest.main()
