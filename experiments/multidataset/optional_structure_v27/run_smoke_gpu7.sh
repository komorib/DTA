#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
RUN_ID="${RUN_ID:-optional_structure_privileged_pair_v27_smoke_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
PROTEIN_CACHE="${PROTEIN_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"
WARM_CHECKPOINT="${WARM_CHECKPOINT:-${ROOT}/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${ROOT}/Model/OptionalStructureSimplexRouter_v26/optional_structure_simplex_router_v26_20260910_161346}"
OUTPUT="${OUTPUT:-${ROOT}/Model/OptionalStructurePrivilegedPair_v27/${RUN_ID}}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  -m dta.optional_structure_privileged_pair_v27.run_training \
  --train-manifest "${MANIFEST_ROOT}/train.jsonl" \
  --valid-manifest "${MANIFEST_ROOT}/valid.jsonl" \
  --protein-cache "${PROTEIN_CACHE}" \
  --ligand-cache "${LIGAND_CACHE}" \
  --anchor-valid "${ANCHOR_ROOT}/valid_predictions.npz" \
  --warm-checkpoint "${WARM_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --limit-train 128 \
  --limit-valid 64 \
  --batch-size 4 \
  --evaluation-batch-size 4 \
  --gradient-accumulation 1 \
  --num-workers 0 \
  --maximum-epochs 1 \
  --steps-per-epoch 6 \
  --patience 1 \
  --width 384 \
  --heads 8 \
  --protein-layers 2 \
  --ligand-layers 4 \
  --cross-layers 3 \
  --latent-width 768 \
  --pair-width 192 \
  --pair-factor 16 \
  --sequence-pair-layers 2 \
  --sequence-top-pairs 128 \
  --structure-self-layers 2 \
  --structure-pair-layers 2 \
  --structure-token-layers 2 \
  --structure-top-pairs 128 \
  --maximum-pocket-tokens 256 \
  --adapter-width 192 \
  --minimum-structure-ci-gain -1.0 \
  --maximum-structure-rmse-increase 10.0 \
  --minimum-structure-spearman-gain -1.0 \
  --minimum-anchor-ci-gain -1.0 \
  --maximum-anchor-rmse-increase 10.0 \
  --minimum-anchor-spearman-gain -1.0 \
  --minimum-positive-folds 0

STATUS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${OUTPUT}/selection.json")"
if [[ "${STATUS}" == "validation_passed" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
    -m dta.optional_structure_privileged_pair_v27.run_test \
    --selection-output "${OUTPUT}" \
    --test-manifest "${MANIFEST_ROOT}/test.jsonl" \
    --anchor-test "${ANCHOR_ROOT}/test_predictions.npz" \
    --device cuda:0 \
    --num-workers 0 \
    --evaluation-batch-size 4 \
    --limit-test 8
else
  echo "v27_smoke_validation_gate=${STATUS}; test inference was not run"
fi

echo "v27_smoke_output=${OUTPUT}"
echo "v27_smoke_selection=${OUTPUT}/selection.json"
if [[ -f "${OUTPUT}/result.json" ]]; then
  echo "v27_smoke_result=${OUTPUT}/result.json"
fi
