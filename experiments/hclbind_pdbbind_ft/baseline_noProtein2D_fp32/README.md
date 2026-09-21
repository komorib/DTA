# HCLBind PDBbind FT Baseline

Source split: `external/HCLBind/data/pdbbind/pdbbind_ft.csv`

GerNA manifests:

- `data/PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT/hclbind_pdbbind_ft/train_manifest.json`
- `data/PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT/hclbind_pdbbind_ft/valid_manifest.json`
- `data/PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT/hclbind_pdbbind_ft/test_manifest.json`

Manifest coverage from current GerNA LMDB:

- HCLBind train/val/test rows: 14990 / 887 / 345
- Kept rows: 14551 / 857 / 339
- Missing rows: 439 / 30 / 6
- Ignored rows: 1514 with `split=unknown`

Paper target for PDBbind test comparison:

- HCLBind: RMSE 1.309, PCC 0.698, C-Index 0.744

Run:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/baseline_noProtein2D_fp32/run.sh
```

Resume after interruption:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/baseline_noProtein2D_fp32/resume.sh
```
