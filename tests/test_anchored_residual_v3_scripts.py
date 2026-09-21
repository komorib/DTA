import subprocess
import unittest
from pathlib import Path


class TestAnchoredResidualV3Scripts(unittest.TestCase):
    def test_launchers_are_isolated_and_cover_test(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "experiments/lp_pdbbind/performance_fusion_v3"
        smoke = directory / "smoke_gpu7.sh"
        formal = directory / "run_anchored_residual_gpu7.sh"
        for path in (smoke, formal):
            text = path.read_text()
            self.assertIn("GPU_ID=${GPU_ID:-7}", text)
            self.assertIn("MEMORY_BUDGET_GB=40", text)
            self.assertIn("test_manifest.json", text)
            self.assertIn("native_test", text)
            self.assertIn("cached_seed87/Model_cached_best.pth", text)
            self.assertIn("anchored_residual_v3", text)
            self.assertNotIn("performance_fusion_v2", text)
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertIn("HybridFeatureCache_v3", formal.read_text())
        self.assertIn("HybridResidual_v3", formal.read_text())
        self.assertIn("--limit 16", smoke.read_text())


if __name__ == "__main__":
    unittest.main()
