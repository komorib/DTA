#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
ROOT="/home/user/wangzhiwen/GerNA-Bind"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
OUT="${ROOT}/experiments/hclbind_pdbbind_ft/pbcnetTrainablePreinit_h64_fp32"

cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
"${PY}" dta/train_dta.py \
  --train_path data/PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT/hclbind_pdbbind_ft_pbcnetGraphAligned/train_manifest.json \
  --valid_path data/PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT/hclbind_pdbbind_ft_pbcnetGraphAligned/valid_manifest.json \
  --test_path data/PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT/hclbind_pdbbind_ft_pbcnetGraphAligned/test_manifest.json \
  --dataset PDBbind_chain_esm_residue_DTA_HCLBindPDBbindFT \
  --split_method hclbind_pdbbind_ft_pbcnetGraphAligned \
  --data_root data \
  --model_output_path "${OUT}/" \
  --resume_training_checkpoint "${OUT}/training_checkpoint.pth" \
  --cuda 0 \
  --protein_3d_mode residue \
  --protein_embedding_mode esm \
  --protein_edge_mode radius \
  --protein_contact_cutoff 6.0 \
  --disable_protein_2d \
  --batch_size 2 \
  --num_workers 4 \
  --epoch 60 \
  --loss smooth_l1 \
  --lr 1e-4 \
  --grad_clip 1.0 \
  --early_stop_patience 0 \
  --pbcnet_graph_lmdb_path data/PDBbind_pbcnet2_graph_fixed_seed42_filtered \
  --pbcnet_trainable \
  --pbcnet_init_checkpoint external/PBCNet2.0/PBCNet2.pth \
  --pbcnet_late_fusion concat \
  --pbcnet_input_dim 96 \
  --pbcnet_hidden_dim 64 \
  --pbcnet_model_hidden_dim 96
