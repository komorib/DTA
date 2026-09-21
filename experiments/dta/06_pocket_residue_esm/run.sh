#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_DIR/experiments/dta/06_pocket_residue_esm"
LOG_DIR="$EXP_DIR/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
MODEL_DIR="Model/PDBbind_chainESM_pocketResidueESM_DTA_noProtein2D_fixed_seed42_filtered"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

{
  echo "experiment: 06_pocket_residue_esm"
  echo "model_output_path: $MODEL_DIR"
  echo "gpu: $GPU"
  echo "log_file: $LOG_FILE"
  echo "start_time: $(date '+%F %T')"

  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
    --train_path data/PDBbind_chainESM_pocketResidueESM_DTA/fixed_seed42_filtered/train_manifest.json \
    --valid_path data/PDBbind_chainESM_pocketResidueESM_DTA/fixed_seed42_filtered/valid_manifest.json \
    --test_path data/PDBbind_chainESM_pocketResidueESM_DTA/fixed_seed42_filtered/test_manifest.json \
    --dataset PDBbind_chainESM_pocketResidueESM_DTA \
    --split_method fixed_seed42_filtered \
    --data_root data \
    --model_output_path "$MODEL_DIR"/ \
    --cuda 0 \
    --protein_embedding_mode esm \
    --protein_3d_mode pocket_residue_esm \
    --protein_edge_mode radius \
    --protein_contact_cutoff 6.0 \
    --disable_protein_2d \
    --batch_size 1 \
    --num_workers 2 \
    --epoch 30 \
    --loss smooth_l1 \
    --lr 3e-4 \
    --grad_clip 1.0 \
    --early_stop_patience 5

  echo "end_time: $(date '+%F %T')"
} 2>&1 | tee "$LOG_FILE"
