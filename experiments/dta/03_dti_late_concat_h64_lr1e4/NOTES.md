# DTI global late fusion concat, hidden 64, lr 1e-4

## Purpose

Uses DrugCLIP/DTI pooled global feature only after GerNA pooling. Shape: DTI_global [B,512] -> [B,64], concat before final affinity head. No DTI front token fusion.

## Data

- train: `data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/train_manifest.json`
- valid: `data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/valid_manifest.json`
- test: `data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/test_manifest.json`
- split: fixed_seed42_filtered

## Output

- model dir: `Model/PDBbind_chain_esm_residue_DTA_noProtein2D_dtiLateConcat_h64_lr1e4_fixed_seed42_filtered/`
- terminal logs: `experiments/dta/03_dti_late_concat_h64_lr1e4/logs/`
- training command: `experiments/dta/03_dti_late_concat_h64_lr1e4/run.sh`

## Feature Configuration

- protein embedding: full-chain ESM, 640 dim
- protein 2D graph: disabled
- protein 3D: residue mode, CA coordinates
- ligand 2D: RDKit graph, 55-dim atom features
- ligand 3D: GerNA ligand atom ids + coordinates

## Result

Paste best valid/test metrics here after training.

```text
valid RMSE:
valid MAE:
valid Pearson:
valid Spearman:
valid CI:

test RMSE:
test MAE:
test Pearson:
test Spearman:
test CI:
```

## Notes

Record interruptions, OOM, unstable epochs, or code changes here.
