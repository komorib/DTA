#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="$(hostname -s)"
if [[ "${HOST}" == ln* ]]; then
  echo "refusing to run V13 training on login node ${HOST}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ID="${RUN_ID:-msa_interface_pretrained_v13_full_$(date +%Y%m%d_%H%M%S)}"
V9_DATA_ROOT="${V9_DATA_ROOT:-${ROOT}/data/LP_PDBBind_MSA_v9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind}"
COMPLEX_CACHE_ROOT="${COMPLEX_CACHE_ROOT:-${V9_DATA_ROOT}/complex_cache_full}"
ROW_CACHE="${ROW_CACHE:-${ROOT}/data/LP_PDBBind_MSA_v11/uniref30_rows_128}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${ROOT}/Model/LP_PDBBind_MSAPairFusion_v12/msa_pair_pretrained_v12_full_20260826_144811}"
V12_PRETRAIN="${V12_PRETRAIN:-${ROOT}/Model/LP_PDBBind_MSAPairPretraining_v12/msa_pair_pretrained_v12_full_20260826_144811/best_pretrain.pt}"
V12_AFFINITY="${V12_AFFINITY:-${ROOT}/Model/LP_PDBBind_MSAPairExpert_v12/msa_pair_pretrained_v12_full_20260826_144811/best.pt}"
PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAInterfacePretraining_v13/${RUN_ID}}"
OUTPUT="${OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceExpert_v13/${RUN_ID}}"
FUSION_OUTPUT="${FUSION_OUTPUT:-${ROOT}/Model/LP_PDBBind_MSAInterfaceFusion_v13/${RUN_ID}}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:-${FUSION_OUTPUT}_validation_selection}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=40"
echo "row_cache=${ROW_CACHE}"
echo "anchor_root=${ANCHOR_ROOT}"
echo "pretrain_output=${PRETRAIN_OUTPUT}"
echo "output=${OUTPUT}"
echo "fusion_output=${FUSION_OUTPUT}"
cd "${ROOT}"

if [[ ! -f "${ROW_CACHE}/row_cache_report.json" ]]; then
  echo "missing V11 row cache: ${ROW_CACHE}" >&2
  exit 2
fi
echo "reusing_complete_row_cache=${ROW_CACHE}"
for required_file in \
  "${V12_PRETRAIN}" \
  "${V12_AFFINITY}" \
  "${ANCHOR_ROOT}/valid_predictions.npz" \
  "${ANCHOR_ROOT}/test_predictions.npz"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required V12 artifact: ${required_file}" >&2
    exit 2
  fi
done

if [[ -f "${PRETRAIN_OUTPUT}/best_pretrain.pt" ]]; then
  echo "reusing_v13_pretraining=${PRETRAIN_OUTPUT}/best_pretrain.pt"
else
  "${PYTHON}" -m dta.msa_interface_pretrained_v13.run_pretraining \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --row-cache "${ROW_CACHE}" \
    --initial-msa-checkpoint "${V12_PRETRAIN}" \
    --output "${PRETRAIN_OUTPUT}" \
    --device "${DEVICE}" \
    --seed 42 \
    --batch-size 4 \
    --evaluation-batch-size 6 \
    --num-workers 4 \
    --amp-dtype bfloat16 \
    --learning-rate 2e-4 \
    --weight-decay 1e-4 \
    --maximum-epochs 16 \
    --patience 5 \
    --msa-width 192 \
    --msa-layers 4 \
    --msa-heads 6 \
    --pair-factor 8 \
    --pair-width 96 \
    --interface-blocks 2 \
    --dropout 0.10 \
    --row-depth 64 \
    --mask-probability 0.15 \
    --interface-mask-probability 0.35 \
    --distogram-weight 0.35 \
    --contact-weight 0.10 \
    --interface-distogram-weight 1.0 \
    --interface-contact-weight 0.30 \
    --match-weight 0.10
fi

if [[ -f "${OUTPUT}/best.pt" && -f "${OUTPUT}/valid_predictions.npz" && -f "${OUTPUT}/result.json" ]]; then
  echo "reusing_v13_validation=${OUTPUT}/valid_predictions.npz"
else
  "${PYTHON}" -m dta.msa_interface_pretrained_v13.run_training \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --pretrained-checkpoint "${PRETRAIN_OUTPUT}/best_pretrain.pt" \
    --affinity-initial-checkpoint "${V12_AFFINITY}" \
    --output "${OUTPUT}" \
    --device "${DEVICE}" \
    --seed 42 \
    --batch-size 4 \
    --evaluation-batch-size 6 \
    --num-workers 4 \
    --amp-dtype bfloat16 \
    --learning-rate 1.5e-4 \
    --encoder-learning-rate-scale 0.20 \
    --weight-decay 1e-4 \
    --maximum-epochs 80 \
    --patience 14 \
    --model-width 512 \
    --layers 8 \
    --heads 8 \
    --dropout 0.10 \
    --latent-width 1024 \
    --msa-width 192 \
    --msa-layers 4 \
    --msa-heads 6 \
    --pair-factor 8 \
    --pair-width 96 \
    --interface-blocks 2 \
    --row-depth 64 \
    --inference-subsets 2 \
    --rank-weight 0.40 \
    --mse-weight 0.15 \
    --contact-weight 0.05 \
    --residue-contact-weight 0.03 \
    --affinity-class-weight 0.05 \
    --distogram-weight 0.02 \
    --interface-distogram-weight 0.08 \
    --interface-contact-weight 0.05 \
    --match-weight 0.02 \
    --interface-mask-probability 0.15 \
    --skip-test
fi

if [[ -f "${SELECTION_OUTPUT}/result.json" ]]; then
  echo "reusing_v13_validation_selection=${SELECTION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pretrained_v13.run_fusion \
    --anchor-root "${ANCHOR_ROOT}" \
    --v13-root "${OUTPUT}" \
    --output "${SELECTION_OUTPUT}" \
    --minimum-validation-ci-gain 0.003 \
    --pca-components 64 \
    --oof-estimators 300 \
    --final-estimators 700 \
    --n-jobs 8 \
    --seed 20260827
fi

if grep -q "\"status\": \"validation_failed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "v13_validation_gate=failed; test inference was not run"
  echo "v13_output=${OUTPUT}"
  echo "v13_selection_result=${SELECTION_OUTPUT}/result.json"
  exit 0
fi
if ! grep -q "\"status\": \"validation_passed\"" "${SELECTION_OUTPUT}/result.json"; then
  echo "unexpected V13 validation selection status" >&2
  exit 2
fi

if [[ -f "${OUTPUT}/test_predictions.npz" ]]; then
  echo "reusing_v13_test=${OUTPUT}/test_predictions.npz"
else
  "${PYTHON}" -m dta.msa_interface_pretrained_v13.run_training \
    --train-manifest "${MANIFEST_ROOT}/train_manifest.json" \
    --valid-manifest "${MANIFEST_ROOT}/valid_manifest.json" \
    --test-manifest "${MANIFEST_ROOT}/test_manifest.json" \
    --train-cache "${COMPLEX_CACHE_ROOT}/train" \
    --valid-cache "${COMPLEX_CACHE_ROOT}/valid" \
    --test-cache "${COMPLEX_CACHE_ROOT}/test" \
    --row-cache "${ROW_CACHE}" \
    --pretrained-checkpoint "${PRETRAIN_OUTPUT}/best_pretrain.pt" \
    --affinity-initial-checkpoint "${V12_AFFINITY}" \
    --output "${OUTPUT}" \
    --device "${DEVICE}" \
    --evaluation-batch-size 6 \
    --num-workers 4 \
    --test-only \
    --selection-result "${SELECTION_OUTPUT}/result.json"
fi

if [[ -f "${FUSION_OUTPUT}/result.json" ]]; then
  echo "reusing_v13_final_fusion=${FUSION_OUTPUT}/result.json"
else
  "${PYTHON}" -m dta.msa_interface_pretrained_v13.run_fusion \
    --anchor-root "${ANCHOR_ROOT}" \
    --v13-root "${OUTPUT}" \
    --output "${FUSION_OUTPUT}" \
    --minimum-validation-ci-gain 0.003 \
    --pca-components 64 \
    --oof-estimators 300 \
    --final-estimators 700 \
    --n-jobs 8 \
    --seed 20260827
fi

echo "v13_pretrain_output=${PRETRAIN_OUTPUT}"
echo "v13_output=${OUTPUT}"
echo "v13_result=${OUTPUT}/result.json"
echo "v13_selection_result=${SELECTION_OUTPUT}/result.json"
echo "v13_fusion_output=${FUSION_OUTPUT}"
echo "v13_fusion_result=${FUSION_OUTPUT}/result.json"
