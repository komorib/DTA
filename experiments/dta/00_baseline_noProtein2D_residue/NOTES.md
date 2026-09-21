# Baseline: chain ESM + residue 3D, no protein 2D

## Purpose

Core fixed-split baseline. Full-chain ESM 1D + chain residue CA 3D; protein 2D graph disabled. No DTI, no Uni-Mol, no Uni-Mol2.

## Data

- train: `data/PDBbind_chain_esm_residue_DTA/fixed_seed42_filtered/train_manifest.json`
- valid: `data/PDBbind_chain_esm_residue_DTA/fixed_seed42_filtered/valid_manifest.json`
- test: `data/PDBbind_chain_esm_residue_DTA/fixed_seed42_filtered/test_manifest.json`
- split: fixed_seed42_filtered

## Output

- model dir: `Model/PDBbind_chain_esm_residue_DTA_noProtein2D_fixed_seed42_filtered/`
- terminal logs: `experiments/dta/00_baseline_noProtein2D_residue/logs/`
- training command: `experiments/dta/00_baseline_noProtein2D_residue/run.sh`

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
