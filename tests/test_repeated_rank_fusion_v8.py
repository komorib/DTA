import unittest

from tests import test_repeated_rank_fusion_v8_pipeline as pipeline
from tests import test_repeated_rank_fusion_v8_stacker as stacker
from tests import test_repeated_rank_fusion_v8_training as training


class RepeatedRankFusionV8Test(unittest.TestCase):
    def test_positive_rank_stacker(self):
        stacker.test_positive_rank_stacker_prefers_concordant_predictor()

    def test_stacker_oof_coverage(self):
        stacker.test_cross_fit_rank_stacker_has_exact_oof_coverage()

    def test_repeat_oof_coverage(self):
        training.test_repeat_training_covers_every_oof_row_once()

    def test_repeat_resume_configuration(self):
        training.test_repeat_resume_rejects_changed_training_configuration()

    def test_repeat_resume_data_lineage(self):
        training.test_repeat_resume_rejects_changed_data_lineage()

    def test_strict_v7_gate(self):
        pipeline.test_v8_gate_requires_strict_ci_improvement_over_v7()

    def test_pipeline_parser(self):
        pipeline.test_v8_parser_exposes_repeat_and_resume_controls()


if __name__ == "__main__":
    unittest.main()
