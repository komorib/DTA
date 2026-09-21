import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_local_interface_decoys_v3 as decoy_eval_v3  # noqa: E402


class PhysicsGuidedV3EvaluatorConfigurationTest(unittest.TestCase):
    def test_frozen_diagnostic_identity(self):
        self.assertEqual(decoy_eval_v3.EVALUATOR_NAME, "local_interface_decoy_data_diagnostic_v3")
        self.assertEqual(decoy_eval_v3.FROZEN_DIAGNOSTIC_KEY_COUNT, 1104)
        self.assertEqual(
            decoy_eval_v3.FROZEN_DIAGNOSTIC_ORDERED_KEY_SHA256,
            "43a7fa8b74f928fb8712713b80f0fd46aeb626aa0b2fec0392adf84f1f7d4005",
        )


    def test_publication_verifier_is_exposed(self):
        self.assertTrue(callable(decoy_eval_v3.verify_published_v3_diagnostic))
