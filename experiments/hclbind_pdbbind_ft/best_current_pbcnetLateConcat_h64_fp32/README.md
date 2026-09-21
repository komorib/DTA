# HCLBind PDBbind FT: Current Best Previous Variant

Chosen after scanning all existing `Model/**/best_metrics.json` files:

- Best previous test CI: `noProtein2D + PBCNet late concat h64 fp32`
- Previous fixed-split test CI: `0.769596`
- Previous fixed-split valid CI: `0.774785`

HCLBind aligned coverage for this run:

- train: `11573`
- valid: `643`
- test: `281`

This is lower than the non-fusion HCLBind subset because PBCNet global features are only available for part of the HCLBind CSV.

Run:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/best_current_pbcnetLateConcat_h64_fp32/run.sh
```

Resume:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/best_current_pbcnetLateConcat_h64_fp32/resume.sh
```

Paper target for PDBbind test comparison: HCLBind C-Index `0.744`.
