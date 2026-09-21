# LP-PDBbind Rank-Unlocked Fusion v7

V7 trains only the fusion network on the 2,340 original validation rows. The
three expert encoders and all V4/V5 artifacts are immutable. The 4,751-row test
split is evaluated after OOF selection is written.

Smoke on physical GPU 7:

```bash
GPU_ID=7 bash experiments/lp_pdbbind/performance_fusion_v7/smoke_gpu7.sh
```

Formal run:

```bash
GPU_ID=7 RESUME=0 \
  bash experiments/lp_pdbbind/performance_fusion_v7/run_rank_unlocked_fusion_gpu7.sh
```

Resume an interrupted formal run with `RESUME=1`.
