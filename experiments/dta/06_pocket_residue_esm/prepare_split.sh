#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_DIR/experiments/dta/06_pocket_residue_esm"
LOG_DIR="$EXP_DIR/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_ID}_split.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

{
  echo "experiment: 06_pocket_residue_esm"
  echo "step: prepare_split"
  echo "log_file: $LOG_FILE"
  echo "start_time: $(date '+%F %T')"

  /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/split_pdbbind_lmdb.py \
    --lmdb_path data/PDBbind_chain_lmdb_esm_gernabind_v2 \
    --index_pkl data/PDBbind_chain_lmdb_esm_gernabind_v2/index.pkl \
    --structure_lmdb_path data/PDBbind_pocket_lmdb_gernabind_v2 \
    --output_dir data/PDBbind_chainESM_pocketResidueESM_DTA/fixed_seed42_filtered \
    --split_method fixed \
    --fixed_split_tsv data/pdbbind_pl_fixed_split_seed42_lmdb18862.tsv \
    --protein_3d_mode pocket_residue_esm \
    --max_residue 512 \
    --max_protein_atom 512 \
    --max_ligand_atom 96 \
    --max_pair_tokens 35000 \
    --seed 99

  echo "end_time: $(date '+%F %T')"
} 2>&1 | tee "$LOG_FILE"
