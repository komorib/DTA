import unittest

from tests import test_rank_unlocked_fusion_v7_core as core
from tests import test_rank_unlocked_fusion_v7_pipeline as pipeline
from tests import test_rank_unlocked_fusion_v7_selection as selection
from tests import test_rank_unlocked_fusion_v7_training as training


class RankUnlockedFusionV7Test(unittest.TestCase):
    def test_missing_expert_mask(self):
        core.test_missing_experts_receive_zero_gate_weight()

    def test_rank_can_change(self):
        core.test_model_can_change_anchor_ordering()

    def test_pairwise_loss(self):
        core.test_pairwise_loss_rewards_correct_order()

    def test_combined_objective(self):
        core.test_objective_combines_regression_ranking_and_residual_terms()

    def test_oof_selection(self):
        selection.test_oof_selection_prefers_rank_improving_residual()

    def test_oof_coverage(self):
        selection.test_oof_coverage_requires_one_prediction_per_row_and_seed()

    def test_checkpoint_training(self):
        training.test_fixed_epoch_checkpoint_predicts_every_requested_row()

    def test_official_gate(self):
        pipeline.test_official_gate_requires_strict_ci_improvement()

    def test_pipeline_parser(self):
        pipeline.test_parser_exposes_smoke_and_resume_controls()


if __name__ == "__main__":
    unittest.main()
