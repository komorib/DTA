# Pocket Residue ESM

## Purpose

Test a cleaner pocket-focused protein 3D route:

```text
full-chain ESM -> take embeddings for pocket residues -> attach to pocket residue CA 3D nodes
```

This keeps ESM context from the complete chain while reducing 3D protein nodes to pocket residues only.

## Data

- feature LMDB: `data/PDBbind_chain_lmdb_esm_gernabind_v2`
- structure LMDB: `data/PDBbind_pocket_lmdb_gernabind_v2`
- output split: `data/PDBbind_chainESM_pocketResidueESM_DTA/fixed_seed42_filtered/`
- fixed split TSV: `data/pdbbind_pl_fixed_split_seed42_lmdb18862.tsv`

Prepare split first:

```bash
experiments/dta/06_pocket_residue_esm/prepare_split.sh
```

Then train:

```bash
GPU=3 experiments/dta/06_pocket_residue_esm/run.sh
```

## Feature Configuration

- protein 1D: full-chain ESM `[B, L_full, 640]`
- protein 2D graph: disabled
- protein 3D: pocket residue CA nodes with full-chain ESM features `[B, L_pocket, 640]`
- ligand 2D: RDKit graph, 55-dim atom features
- ligand 3D: GerNA ligand atom ids + coordinates
- external fusion: none

## Implementation Notes

Mode name: `pocket_residue_esm`

Alignment currently mirrors `pocket_pharmacophore_esm`:

```text
if pocket sequence length == full-chain sequence length: index alignment
else: greedy subsequence alignment from pocket sequence into full-chain sequence
```

This is useful for testing, but exact residue-id mapping would be more reliable for final production data.

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

Record split counts, OOM, unstable epochs, or follow-up changes here.
