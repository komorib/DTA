import subprocess
import unittest
from pathlib import Path


class TestUpperBoundLaunchers(unittest.TestCase):
    def test_gpu7_launcher_contract(self):
        root = Path(__file__).resolve().parents[1]
        formal = root / "experiments/lp_pdbbind/performance_fusion_v2/run_upper_bound_gpu7.sh"
        smoke = root / "experiments/lp_pdbbind/performance_fusion_v2/smoke_gpu7.sh"
        formal_text = formal.read_text()
        smoke_text = smoke.read_text()
        for text in (formal_text, smoke_text):
            self.assertIn("GPU_ID=${GPU_ID:-7}", text)
            self.assertIn("MEMORY_BUDGET_GB=40", text)
            self.assertIn("/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python", text)
            self.assertNotIn("test_manifest", text)
        self.assertIn("SEEDS=(1 7 21 42 87)", formal_text)
        self.assertIn("rank_cached_seeds", formal_text)
        self.assertIn("\"metadata.json\"", formal_text)
        self.assertIn(".get(\"status\") == \"complete\"", formal_text)
        self.assertNotIn("resume[@]", formal_text)
        self.assertIn("--limit 16", smoke_text)
        for path in (formal, smoke):
            subprocess.run(["bash", "-n", str(path)], check=True)


if __name__ == "__main__":
    unittest.main()
