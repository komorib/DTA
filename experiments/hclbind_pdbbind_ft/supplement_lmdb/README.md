# HCLBind PDBbind FT LMDB Supplement

Current missing rows from HCLBind finetune CSV against `data/PDBbind_chain_lmdb_esm_gernabind_v2`:

- total missing: 475
- local raw PDBbind files available: 147
- base LMDB extraction success: 131
- base LMDB extraction failed: 16
- local raw files unavailable: 328

The 16 local extraction failures are ligand atom feature failures caused by existing ring-size checks. They are left out to keep feature logic consistent with the original GerNA LMDB.

Run on a GPU node:

```bash
GPU_ID=7 bash experiments/hclbind_pdbbind_ft/supplement_lmdb/01_extract_missing_esm.sh
bash experiments/hclbind_pdbbind_ft/supplement_lmdb/02_merge_and_rebuild_manifests.sh
```

Expected supplemented HCLBind split coverage after merge:

- train: 14669
- valid: 869
- test: 340

Remaining unavailable after local supplement:

- train: 321
- valid: 18
- test: 5
