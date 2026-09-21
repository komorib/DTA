#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_DIR/experiments/dta/05_dti_front_concat_ligand_unimol_2d_concat"
LOG_DIR="$EXP_DIR/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
MODEL_DIR="Model/PDBbind_chain_esm_residue_DTA_noProtein2D_dtiFrontConcat_ligandUniMol2DConcat_fixed_seed42_filtered"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

{
  echo "experiment: 05_dti_front_concat_ligand_unimol_2d_concat"
  echo "model_output_path: $MODEL_DIR"
  echo "gpu: $GPU"
  echo "log_file: $LOG_FILE"
  echo "start_time: $(date '+%F %T')"

  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
    --train_path data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/train_manifest.json \
    --valid_path data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/valid_manifest.json \
    --test_path data/PDBbind_chain_esm_residue_DTA_lateAligned/fixed_seed42_filtered/test_manifest.json \
    --dataset PDBbind_chain_esm_residue_DTA \
    --split_method fixed_seed42_filtered \
    --data_root data \
    --model_output_path "$MODEL_DIR"/ \
    --cuda 0 \
    --protein_embedding_mode esm \
    --protein_edge_mode radius \
    --protein_contact_cutoff 6.0 \
    --disable_protein_2d \
    --batch_size 1 \
    --num_workers 2 \
    --epoch 30 \
    --loss smooth_l1 \
    --lr 3e-4 \
    --grad_clip 1.0 \
    --early_stop_patience 5 \
    --dti_fusion_lmdb_path data/PDBbind_dti_v12_fusion_lmdb_late_fixed_seed42_filtered \
    --dti_front_fusion concat \
    --dti_front_input_dim 256 \
    --ligand_unimol_lmdb_path data/PDBbind_ligand_unimol_token_gernaOrder_fixed_seed42_filtered.lmdb \
    --ligand_unimol_fusion concat \
    --ligand_unimol_input_dim 512

  echo "end_time: $(date '+%F %T')"
} 2>&1 | tee "$LOG_FILE"
