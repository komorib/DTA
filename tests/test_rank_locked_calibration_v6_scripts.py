import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    REPO_ROOT / "experiments" / "lp_pdbbind" / "performance_fusion_v6"
)
SCRIPT_NAMES = (
    "common.sh",
    "smoke_gpu7.sh",
    "run_rank_locked_calibration_gpu7.sh",
)


class RankLockedCalibrationV6ScriptTest(unittest.TestCase):
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

    def test_scripts_pin_gpu7_and_keep_old_roots_read_only(self):
        combined = "\n".join(self.read(name) for name in SCRIPT_NAMES)
        self.assertIn('GPU_ID="${GPU_ID:-7}"', combined)
        self.assertIn('export CUDA_VISIBLE_DEVICES="${GPU_ID}"', combined)
        self.assertIn('DEVICE="cuda:0"', combined)
        self.assertIn('MEMORY_BUDGET_GB="40"', combined)
        self.assertIn(
            'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
            combined,
        )
        self.assertNotIn('mkdir -p "${V4_CACHE_ROOT}', combined)
        self.assertNotIn('mkdir -p "${V5_MODEL_ROOT}', combined)

    def test_common_preflights_every_immutable_input(self):
        common = self.read("common.sh")
        for value in (
            "LP_PDBBind_RankFusionCache_v4",
            "LP_PDBBind_AnchoredExpertFusion_v5",
            "LP_PDBBind_RankLockedCalibration_v6",
            "data/LP_PDBBind.csv",
            "selection_seal.json",
            "official_test/result.json",
            "features.npz",
            "metadata.json",
            "coverage.json",
            "normalization.json",
        ):
            self.assertIn(value, common)
        self.assertIn("preflight_validation_inputs", common)
        self.assertIn("preflight_official_inputs", common)
        self.assertIn("git rev-parse HEAD", common)
        self.assertNotIn("git -C", common)

    def test_smoke_has_no_test_access_and_only_tmp_output(self):
        smoke = self.read("smoke_gpu7.sh")
        self.assertIn("--limit 64", smoke)
        self.assertIn("--smoke-epochs 1", smoke)
        self.assertIn("/tmp/LP_PDBBind_RankLockedCalibration_v6_smoke_", smoke)
        self.assertNotIn("TEST_CACHE", smoke)
        self.assertNotIn("evaluate_test", smoke)
        self.assertNotIn("OFFICIAL_TEST_ROOT", smoke)
        self.assertNotIn("MODEL_ROOT", smoke)

    def test_formal_uses_safe_resume_and_exactly_one_sealed_evaluation(self):
        formal = self.read("run_rank_locked_calibration_gpu7.sh")
        evaluator = "-m dta.rank_locked_calibration_v6.evaluate_test"
        verifier = "verify_validation_package"
        self.assertEqual(formal.count(evaluator), 1)
        self.assertIn(verifier, formal)
        self.assertLess(formal.index(verifier), formal.index(evaluator))
        self.assertIn('RESUME="${RESUME:-0}"', formal)
        self.assertIn(
            '[[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]', formal
        )
        self.assertIn('if [[ "${RESUME}" == "1" ]]; then', formal)
        self.assertIn("validation_command+=(--resume)", formal)
        self.assertIn('"${validation_command[@]}"', formal)
        self.assertIn('if [[ -e "${OFFICIAL_TEST_ROOT}" ]]; then', formal)
        self.assertLess(
            formal.index('if [[ -e "${OFFICIAL_TEST_ROOT}" ]]; then'),
            formal.index('"${validation_command[@]}"'),
        )
        self.assertNotIn("resume_args", formal)
        self.assertNotIn("limit_args", formal)

    def test_scripts_have_no_dynamic_shell_or_promotion_pointer(self):
        combined = "\n".join(self.read(name) for name in SCRIPT_NAMES)
        self.assertIsNone(re.search(r"(^|[;&|\s])eval(\s|$)", combined))
        self.assertNotIn("bash -c", combined)
        self.assertNotRegex(combined, r"\bln\s+-s")
        self.assertNotIn("active_model", combined)
        self.assertNotIn("best_model", combined)

    def test_readme_documents_launch_resume_results_and_read_only_rollback(self):
        readme = self.read("README.md")
        self.assertIn("GPU_ID=7 RESUME=0", readme)
        self.assertIn("GPU_ID=7 RESUME=1", readme)
        self.assertIn("run_rank_locked_calibration_gpu7.sh", readme)
        self.assertIn("smoke_gpu7.sh", readme)
        self.assertIn(
            "Model/LP_PDBBind_AnchoredExpertFusion_v5/validation/validation_package",
            readme,
        )
        self.assertIn(
            "Model/LP_PDBBind_AnchoredExpertFusion_v5/official_test/result.json",
            readme,
        )
        self.assertIn("validation_result.json", readme)
        self.assertIn("official_test/result.json", readme)
        for command in ("rm ", "cp ", "mv ", "ln "):
            self.assertNotIn(command, readme)


if __name__ == "__main__":
    unittest.main()
