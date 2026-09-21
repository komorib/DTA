# Old Uni-Mol ligand token fused into ligand 2D by concat

## Purpose

Tests old Uni-Mol atom tokens as ligand 2D augmentation only. Shape: Uni-Mol [B,A,512] -> hidden_size1, concat with RDKit ligand graph branch. DTI and Uni-Mol2 disabled.

## Data

- train: `data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/train_manifest.json`
- valid: `data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/valid_manifest.json`
- test: `data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/test_manifest.json`
- split: fixed_seed42_filtered

## Output

- model dir: `Model/PDBbind_chain_esm_residue_DTA_noProtein2D_ligandUniMol2DConcat_fixed_seed42_filtered/`
- terminal logs: `experiments/dta/01_ligand_unimol_2d_concat/logs/`
- training command: `experiments/dta/01_ligand_unimol_2d_concat/run.sh`

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
