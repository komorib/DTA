# Uni-Mol2 ligand token fused into ligand 3D by concat

## Purpose

Tests Uni-Mol2 atom tokens in GerNA atom order as ligand 3D augmentation. Shape: Uni-Mol2 [B,A,768] -> 16, concat with Equiformer ligand 3D [B,A,16]. DTI and old Uni-Mol disabled.

## Data

- train: `data/PDBbind_chain_esm_residue_DTA_unimol2Aligned/fixed_seed42_filtered/train_manifest.json`
- valid: `data/PDBbind_chain_esm_residue_DTA_unimol2Aligned/fixed_seed42_filtered/valid_manifest.json`
- test: `data/PDBbind_chain_esm_residue_DTA_unimol2Aligned/fixed_seed42_filtered/test_manifest.json`
- split: fixed_seed42_filtered

## Output

- model dir: `Model/PDBbind_chain_esm_residue_DTA_noProtein2D_ligandUniMol2_3DConcat_fixed_seed42_filtered/`
- terminal logs: `experiments/dta/02_ligand_unimol2_3d_concat/logs/`
- training command: `experiments/dta/02_ligand_unimol2_3d_concat/run.sh`

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
