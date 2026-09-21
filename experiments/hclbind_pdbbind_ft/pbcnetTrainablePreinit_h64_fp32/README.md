# HCLBind PDBbind FT: Trainable PBCNet Preinit

This is the full trainable PBCNet variant:

- `--pbcnet_trainable`
- `--pbcnet_graph_lmdb_path data/PDBbind_pbcnet2_graph_fixed_seed42_filtered`
- `--pbcnet_init_checkpoint external/PBCNet2.0/PBCNet2.pth`

Expected HCLBind graph-aligned coverage:

- train: `11573`
- valid: `643`
- test: `281`

Caution: on the previous fixed split, this variant underperformed freeze PBCNet global late fusion and showed stronger overfitting risk.

Run:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/pbcnetTrainablePreinit_h64_fp32/run.sh
```

Resume:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/pbcnetTrainablePreinit_h64_fp32/resume.sh
```
