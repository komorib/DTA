# HCLBind PDBbind FT Supplemented Baseline

Use this after running:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/supplement_lmdb/01_extract_missing_esm.sh
bash experiments/hclbind_pdbbind_ft/supplement_lmdb/02_merge_and_rebuild_manifests.sh
```

Training:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/baseline_noProtein2D_fp32_supplemented/run.sh
```

Resume:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/baseline_noProtein2D_fp32_supplemented/resume.sh
```

Paper target for PDBbind test comparison: HCLBind C-Index `0.744`.
