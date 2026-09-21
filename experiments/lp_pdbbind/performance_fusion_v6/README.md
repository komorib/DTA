# LP-PDBbind Rank-Locked Calibration v6

This directory launches the isolated v6 calibration pipeline. It reads the
existing v4 feature cache and sealed v5 result, and writes only to a new v6
output root.

## Smoke

The smoke run uses 64 train/validation rows, one epoch, GPU 7, and /tmp
outputs. It never loads or evaluates the official test split.

    GPU_ID=7 RUN_ID=rank_locked_v6_smoke_$(date +%Y%m%d_%H%M%S) \
    bash experiments/lp_pdbbind/performance_fusion_v6/smoke_gpu7.sh \
    2>&1 | tee /tmp/rank_locked_calibration_v6_smoke_gpu7.log

The result is written under:

    /tmp/LP_PDBBind_RankLockedCalibration_v6_smoke_<RUN_ID>/smoke_result.json

## Formal Run

Start a new formal validation and the single sealed official-test evaluation:

    GPU_ID=7 RESUME=0 \
    bash experiments/lp_pdbbind/performance_fusion_v6/run_rank_locked_calibration_gpu7.sh \
    2>&1 | tee /tmp/rank_locked_calibration_v6_full_gpu7.log

Resume only when validation was interrupted before any official-test output was
created:

    GPU_ID=7 RESUME=1 \
    bash experiments/lp_pdbbind/performance_fusion_v6/run_rank_locked_calibration_gpu7.sh \
    2>&1 | tee /tmp/rank_locked_calibration_v6_full_gpu7_resume.log

Formal outputs:

    Model/LP_PDBBind_RankLockedCalibration_v6/validation/validation_result.json
    Model/LP_PDBBind_RankLockedCalibration_v6/validation/validation_package
    Model/LP_PDBBind_RankLockedCalibration_v6/official_test/result.json

An existing v6 official-test directory is rejected. Test results never trigger
candidate reselection or a second evaluation, and this pipeline creates no
active or best model pointer.

## Rollback

v5 remains the unchanged rollback model. Its sealed package and official result
stay at:

    Model/LP_PDBBind_AnchoredExpertFusion_v5/validation/validation_package
    Model/LP_PDBBind_AnchoredExpertFusion_v5/official_test/result.json

No promotion or file-changing rollback step is performed by v6.
