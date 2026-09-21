# HCLBind PDBbind FT: Optimized Trainable PBCNet Preinit

This is the optimized full-trainable PBCNet variant for the HCLBind PDBbind fine-tune split.

Changes from `pbcnetTrainablePreinit_h64_fp32`:

- PBCNet adapter hidden dim: `64 -> 32`
- Main learning rate: `1e-4`
- PBCNet encoder learning rate: `1e-5`
- Weight decay: `1e-4`
- Freeze pretrained PBCNet encoder for the first 5 epochs, then unfreeze
- Early stopping disabled with `--early_stop_patience 0`; bad-epoch count is still logged

Run:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/pbcnetTrainablePreinit_h32_wd1e4_pbcLR1e5_freeze5_fp32/run.sh
```

Resume:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/pbcnetTrainablePreinit_h32_wd1e4_pbcLR1e5_freeze5_fp32/resume.sh
```

Training artifacts and `train.log` are written inside this experiment directory.
