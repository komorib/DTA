#!/usr/bin/env bash
set -euo pipefail

# Required paths; override with environment variables.
STRUCTURE_ROOT=${LP_PDBBIND_STRUCTURE_ROOT:-data/LP-PDBbind/structures}
SPLIT_SOURCE=${LP_PDBBIND_SPLIT_SOURCE:-data/LP-PDBbind/splits}
AFFINITY_SOURCE=${LP_PDBBIND_AFFINITY_SOURCE:-data/PDBbind/index/INDEX_general_PL.2020R1.lst}
OUTPUT_LMDB=${LP_PDBBIND_LMDB_OUT:-data/LP-PDBbind/kdbnet_lmdb}
ESM_MODEL_PATH=${LP_PDBBIND_ESM_MODEL_PATH:-}

ESM_ARGS=(
  --esm_mode generate
  --esm_model_name esm2_t33_650M_UR50D
  --esm_repr_layer 33
  --esm_device cuda
  --embedding_key esm2_embedding
  --embedding_dim 1280
  --embedding_dtype float16
  --esm_cache_dir "${OUTPUT_LMDB}_esm_cache"
)
if [[ -n "${ESM_MODEL_PATH}" ]]; then
  ESM_ARGS+=(--esm_model_path "${ESM_MODEL_PATH}")
fi

python scripts/build_lp_pdbbind_lmdb.py \
  --structure_root "${STRUCTURE_ROOT}" \
  --split_source "${SPLIT_SOURCE}" \
  --affinity_source "${AFFINITY_SOURCE}" \
  --output_lmdb "${OUTPUT_LMDB}" \
  --protein_source auto \
  --path_mode absolute \
  --normalise_ligands \
  --map_size_gb 64 \
  --max_map_size_gb 512 \
  "${ESM_ARGS[@]}" \
  "$@"

python scripts/audit_lp_pdbbind_lmdb.py \
  --lmdb_dir "${OUTPUT_LMDB}" \
  --embedding_key esm2_embedding \
  --embedding_dim 1280 \
  --check_paths
