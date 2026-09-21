# LP-PDBbind Repeated Rank Fusion v8

V8 leaves the complete V7 implementation and artifacts unchanged. It adds
three grouped-fold repeats, combines V5 plus all V7/V8 OOF predictors with a
positive pairwise rank stacker, seals the OOF selection, and only then evaluates
the 4,751-row test split. Persistent output is isolated under
`Model/LP_PDBBind_RepeatedRankFusion_v8/`.

The immutable V7 input is
`Model/LP_PDBBind_RankUnlockedFusion_v7/run`.

Smoke on physical GPU 7:

```bash
GPU_ID=7 \
  bash experiments/lp_pdbbind/performance_fusion_v8/smoke_gpu7.sh \
  2>&1 | tee /tmp/repeated_rank_fusion_v8_smoke_gpu7.log
```

Formal run:

```bash
GPU_ID=7 RESUME=0 \
  bash experiments/lp_pdbbind/performance_fusion_v8/run_repeated_rank_fusion_gpu7.sh \
  2>&1 | tee /tmp/repeated_rank_fusion_v8_full_gpu7.log
```

Resume an interrupted formal run:

```bash
GPU_ID=7 RESUME=1 \
  bash experiments/lp_pdbbind/performance_fusion_v8/run_repeated_rank_fusion_gpu7.sh \
  2>&1 | tee /tmp/repeated_rank_fusion_v8_full_gpu7_resume.log
```

The final result is written to
`Model/LP_PDBBind_RepeatedRankFusion_v8/run/result.json`. A formal result passes
only when its test CI is strictly greater than the V7 test CI.
