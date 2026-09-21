#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
ROOT="/home/user/wangzhiwen/GerNA-Bind"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
ESM_MODEL="/home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D"

cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" data_utils/extract_lmdb_esm.py \
  --input_lmdb data/PDBbind_chain_lmdb_hclbind_missing_base \
  --index_pkl data/PDBbind_chain_lmdb_hclbind_missing_base/index.pkl \
  --output_lmdb data/PDBbind_chain_lmdb_hclbind_missing_esm \
  --output_index_pkl data/PDBbind_chain_lmdb_hclbind_missing_esm/index.pkl \
  --esm_model_path "${ESM_MODEL}" \
  --device cuda:0 \
  --sample_batch_size 4 \
  --esm_batch_size 4 \
  --expected_hidden_size 640 \
  --map_size 8 \
  --resume
