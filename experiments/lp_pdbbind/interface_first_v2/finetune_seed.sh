#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index for Stage 2}"
SEED="${SEED:?Set SEED to one of 42,43,44,45,46}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
NATIVE_CACHE_ROOT="${NATIVE_CACHE_ROOT:-${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1}"
MANIFEST_ROOT="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind"
STAGE1_DIR="${ROOT}/Model/LP_PDBBind_InterfaceFirstGerNA_v2_stage1_seed${SEED}"
OUTPUT_DIR="${ROOT}/Model/LP_PDBBind_InterfaceFirstGerNA_v2_seed${SEED}"
RESUME_STAGE2="${RESUME_STAGE2:-0}"

case "${SEED}" in
  42|43|44|45|46) ;;
  *) echo "SEED must be one of 42,43,44,45,46" >&2; exit 1 ;;
esac

for path in \
  "${MANIFEST_ROOT}/train_manifest.json" \
  "${MANIFEST_ROOT}/valid_manifest.json" \
  "${MANIFEST_ROOT}/test_manifest.json" \
  "${NATIVE_CACHE_ROOT}/native_train" \
  "${NATIVE_CACHE_ROOT}/native_valid" \
  "${NATIVE_CACHE_ROOT}/native_test" \
  "${STAGE1_DIR}/stage1_final.pth" \
  "${STAGE1_DIR}/model_diagnostic_report.json" \
  "${STAGE1_DIR}/stage2_ready.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "required Stage-2 input is missing: ${path}" >&2
    exit 1
  fi
done

if [[ "${RESUME_STAGE2}" == "1" ]]; then
  checkpoint="${OUTPUT_DIR}/training_checkpoint.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "resume checkpoint is missing: ${checkpoint}" >&2
    exit 1
  fi
  set -- --resume_training_checkpoint "${checkpoint}"
else
  set --
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
"${PY}" -u dta/train_dta.py \
  --model_arch interface_first \
  --train_path "${MANIFEST_ROOT}/train_manifest.json" \
  --valid_path "${MANIFEST_ROOT}/valid_manifest.json" \
  --test_path "${MANIFEST_ROOT}/test_manifest.json" \
  --dataset PDBbind_chain_esm_residue_DTA \
  --split_method lp_pdbbind \
  --data_root data \
  --model_output_path "${OUTPUT_DIR}/" \
  --interface_train_native_lmdb "${NATIVE_CACHE_ROOT}/native_train" \
  --interface_valid_native_lmdb "${NATIVE_CACHE_ROOT}/native_valid" \
  --interface_test_native_lmdb "${NATIVE_CACHE_ROOT}/native_test" \
  --interface_stage1_checkpoint "${STAGE1_DIR}/stage1_final.pth" \
  --interface_diagnostic_report "${STAGE1_DIR}/model_diagnostic_report.json" \
  --interface_stage2_ready "${STAGE1_DIR}/stage2_ready.json" \
  --interface_local_lr 1e-4 \
  --interface_global_lr 3e-5 \
  --interface_affinity_lr 1e-4 \
  --interface_warmup_epochs 3 \
  --interface_shell_budget 128 \
  --interface_ligand_gnn_depth 3 \
  --interface_ligand_dropout 0.1 \
  --interface_affinity_dropout 0.2 \
  --interface_residue_to_atom_mode gated_residual \
  --cuda 0 \
  --seed "${SEED}" \
  --batch_size 1 \
  --num_workers 4 \
  --epoch 30 \
  --loss smooth_l1 \
  --weight_decay 0 \
  --lr_decay_step 10 \
  --lr_decay_gamma 0.5 \
  --grad_clip 5.0 \
  --early_stop_patience 0 \
  --rank_loss_weight 0 \
  --select_metric CI \
  --step_log_every 100 \
  "$@"

echo "best_checkpoint: ${OUTPUT_DIR}/Model_dta_best.pth"
echo "test_predictions: ${OUTPUT_DIR}/test_predictions.npz"
