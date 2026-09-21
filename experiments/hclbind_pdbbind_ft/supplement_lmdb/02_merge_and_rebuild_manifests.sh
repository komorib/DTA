#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/wangzhiwen/GerNA-Bind"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"

cd "${ROOT}"

"${PY}" scripts/merge_lmdb_with_index.py \
  --primary-lmdb data/PDBbind_chain_lmdb_esm_gernabind_v2 \
  --primary-index data/PDBbind_chain_lmdb_esm_gernabind_v2/index.pkl \
  --supplement-lmdb data/PDBbind_chain_lmdb_hclbind_missing_esm \
  --supplement-index data/PDBbind_chain_lmdb_hclbind_missing_esm/index.pkl \
  --output-lmdb data/PDBbind_chain_lmdb_esm_gernabind_v2_hclbindFT_supplemented \
  --output-index data/PDBbind_chain_lmdb_esm_gernabind_v2_hclbindFT_supplemented/index.pkl \
  --summary-output data/PDBbind_chain_lmdb_esm_gernabind_v2_hclbindFT_supplemented/merge_summary.json \
  --map-size-gb 220

"${PY}" scripts/build_hclbind_pdbbind_manifests.py \
  --hclbind-csv external/HCLBind/data/pdbbind/pdbbind_ft.csv \
  --lmdb-path data/PDBbind_chain_lmdb_esm_gernabind_v2_hclbindFT_supplemented \
  --output-dir data/PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT/hclbind_pdbbind_ft_supplemented \
  --split-method hclbind_pdbbind_ft_supplemented \
  --protein-3d-mode residue
