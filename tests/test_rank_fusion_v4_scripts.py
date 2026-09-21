import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "experiments" / "lp_pdbbind" / "performance_fusion_v4"


class RankFusionScriptTest(unittest.TestCase):
    def read(self, name):
        return (SCRIPT_ROOT / name).read_text(encoding="utf-8")

    def test_shell_syntax_and_gpu_policy(self):
        for name in ("common.sh", "smoke_gpu7.sh", "run_rank_fusion_gpu7.sh"):
            path = SCRIPT_ROOT / name
            completed = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("set -euo pipefail", self.read(name))
        combined = self.read("common.sh") + self.read("smoke_gpu7.sh") + self.read("run_rank_fusion_gpu7.sh")
        self.assertIn('GPU_ID="${GPU_ID:-7}"', combined)
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPU_ID}"', combined)
        self.assertIn('DEVICE="cuda:0"', combined)
        self.assertIn('MEMORY_BUDGET_GB="40"', combined)
        self.assertIn('SEEDS=(1 7 21 42 87)', combined)
        self.assertIn('"${seeds[@]}"', combined)
        self.assertNotIn("resume[@]", combined)
        self.assertIn('resume_arg=""', combined)
        self.assertIn('${resume_arg:+"${resume_arg}"}', combined)

    def test_formal_helpers_accept_an_omitted_limit_under_nounset(self):
        environment = os.environ.copy()
        environment.update({
            "PYTHON": "/bin/true",
            "BOLTZ_LMDB_DIR": "/tmp/boltz",
            "BOLTZ_INDEX_PATH": "/tmp/index.pkl",
            "THREED_DATA_DIR": "/tmp/three_dprot",
        })
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; export_split train; build_join train',
                "rank-fusion-v4-test",
                str(SCRIPT_ROOT / "common.sh"),
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_common_pins_immutable_inputs_and_preflights_data(self):
        text = self.read("common.sh")
        required = (
            "data/PDBbind_chain_esm_residue_DTA/lp_pdbbind",
            "data/LP_PDBBind_HybridFeatureCache_v3",
            "Model/LP_PDBBind_HybridUpperBound_v2/cached_seed87/Model_cached_best.pth",
            "external_benchmarks/results/boltz_pairwise_rank010_pairDelta_save/model.pt",
            "external_benchmarks/results/3dprot_dta_lp_official_ckpt/best_model.pt",
            "data/LP_PDBBind_RankFusionCache_v4",
            "Model/LP_PDBBind_RankFusion_v4",
        )
        for value in required:
            self.assertIn(value, text)
        self.assertIn('p["config"]["merged_lmdb_dir"]', text)
        self.assertIn('p["config"]["index_path"]', text)
        self.assertIn('p["args"]["data_dir"]', text)
        self.assertIn("preflight_paths", text)

    def test_formal_pipeline_has_sealed_stage_order(self):
        text = self.read("run_rank_fusion_gpu7.sh")
        stages = (
            "export_split train",
            "build_join train",
            "export_split valid",
            "build_join valid",
            "train_seeds",
            "select_validation",
            "export_split test",
            "build_join test",
            "evaluate_test",
        )
        positions = [text.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        train_section = text[text.index("train_seeds") : text.index("select_validation")]
        selection_section = text[text.index("select_validation") : text.index("export_split test")]
        self.assertNotIn("test", train_section.lower())
        self.assertNotIn("test", selection_section.lower())

    def test_smoke_uses_64_rows_two_seeds_and_one_epoch(self):
        text = self.read("smoke_gpu7.sh")
        self.assertIn("SMOKE_SEEDS=(42 87)", text)
        self.assertIn('LIMIT="64"', text)
        self.assertIn('EPOCHS="1"', text)
        self.assertIn('PATIENCE="1"', text)
        self.assertIn("--expected-count 64", text)
        self.assertIn("--expected-seed 42", text)
        self.assertIn("--expected-seed 87", text)
        self.assertIn("external_benchmarks/3d-prot-dta/data/lp_pdbbind", text)
        self.assertNotIn("lp_pdbbind_smoke", text)
        self.assertLess(text.index("select_validation"), text.index("export_split test"))


if __name__ == "__main__":
    unittest.main()
