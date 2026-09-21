# Uni-Mol2 ligand token fused before ligand Equiformer by concat

## Purpose

Tests Uni-Mol2 atom tokens in GerNA atom order as ligand 3D input augmentation before Equiformer. This is the cleaner 3D fusion path:

```text
GerNA ligand atom ids [B,A] -> Embedding(10,16)
Uni-Mol2 token [B,A,768] -> Linear(768,16)
concat -> Linear(32,16)
fused ligand atom state [B,A,16] + ligand coords [B,A,3] -> ligand Equiformer
```

DTI and old Uni-Mol 2D fusion are disabled.

## Data

- train: `data/PDBbind_chain_esm_residue_DTA_unimol2Aligned/fixed_seed42_filtered/train_manifest.json`
- valid: `data/PDBbind_chain_esm_residue_DTA_unimol2Aligned/fixed_seed42_filtered/valid_manifest.json`
- test: `data/PDBbind_chain_esm_residue_DTA_unimol2Aligned/fixed_seed42_filtered/test_manifest.json`
- split: fixed_seed42_filtered

## Output

- model dir: `Model/PDBbind_chain_esm_residue_DTA_noProtein2D_ligandUniMol2_3DPreConcat_fixed_seed42_filtered/`
- terminal logs: `experiments/dta/07_ligand_unimol2_3d_pre_concat/logs/`
- training command: `experiments/dta/07_ligand_unimol2_3d_pre_concat/run.sh`

## Feature Configuration

- protein embedding: full-chain ESM, 640 dim
- protein 2D graph: disabled
- protein 3D: residue mode, CA coordinates
- ligand 2D: RDKit graph, 55-dim atom features
- ligand 3D: GerNA ligand atom ids + coordinates, with Uni-Mol2 token fused before Equiformer

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
