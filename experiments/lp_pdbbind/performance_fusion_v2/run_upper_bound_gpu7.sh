#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python"
GPU_ID=${GPU_ID:-7}
MEMORY_BUDGET_GB=40
DEVICE=cuda:0
SEEDS=(1 7 21 42 87)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TRAIN_MANIFEST="${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
VALID_MANIFEST="${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/valid_manifest.json"
TRAIN_NATIVE_CACHE="${ROOT_DIR}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_train"
VALID_NATIVE_CACHE="${ROOT_DIR}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_valid"
GERNA_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth"
INTERFACE_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth"
CACHE_ROOT="${ROOT_DIR}/data/LP_PDBBind_HybridFeatureCache_v2"
MODEL_ROOT="${ROOT_DIR}/Model/LP_PDBBind_HybridUpperBound_v2"

cache_complete() {
  "${PYTHON}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]) / "metadata.json"; sys.exit(0 if p.is_file() and json.loads(p.read_text()).get("status") == "complete" else 1)' "$1"
}

build_or_resume_cache() {
  local split="$1" manifest="$2" native_cache="$3" output="$4"
  if cache_complete "${output}"; then
    echo "reusing complete ${split} cache: ${output}"
    return
  fi
  local resume_arg=""
  if [[ -d "${output}" ]] && [[ -n "$(ls -A "${output}" 2>/dev/null)" ]]; then
    resume_arg="--resume"
  fi
  "${PYTHON}" "${ROOT_DIR}/dta/build_hybrid_feature_cache_v2.py" \
    --manifest "${manifest}" --native-cache "${native_cache}" --split "${split}" \
    --gerna-checkpoint "${GERNA_CHECKPOINT}" --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
    --output "${output}" --device "${DEVICE}" --map-size-gb "${MEMORY_BUDGET_GB}" \
    ${resume_arg:+"${resume_arg}"}
}

run_cached_seed() {
  local seed="$1" output="${MODEL_ROOT}/cached_seed${seed}"
  if [[ -f "${output}/Model_cached_best.pth" ]] && [[ -f "${output}/candidate_cached.json" ]]; then
    echo "reusing cached seed ${seed}: ${output}"
    return
  fi
  local resume_arg=""
  if [[ -d "${output}" ]] && [[ -n "$(ls -A "${output}" 2>/dev/null)" ]]; then
    resume_arg="--resume"
  fi
  "${PYTHON}" "${ROOT_DIR}/dta/train_hybrid_cached_v2.py" \
    --train-cache "${CACHE_ROOT}/train" --valid-cache "${CACHE_ROOT}/valid" \
    --output "${output}" --seed "${seed}" --device "${DEVICE}" \
    --batch-size 64 --epochs 100 --patience 12 --learning-rate 2e-4 --dropout 0.1 \
    ${resume_arg:+"${resume_arg}"}
}

rank_cached_seeds() {
  "${PYTHON}" - "${MODEL_ROOT}" "${SEEDS[@]}" <<PY
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for raw_seed in sys.argv[2:]:
    seed = int(raw_seed)
    payload = json.loads((root / f"cached_seed{seed}" / "best_metrics.json").read_text())
    records.append((float(payload["CI"]), float(payload["Spearman"]), -float(payload["RMSE"]), seed))
if len(records) < 3:
    raise RuntimeError("fewer than three complete cached seeds")
for _, _, _, seed in sorted(records, reverse=True)[:3]:
    print(seed)
PY
}

run_e2e_seed() {
  local seed="$1" cached="${MODEL_ROOT}/cached_seed${seed}" output="${MODEL_ROOT}/e2e_seed${seed}"
  if [[ -f "${output}/Model_e2e_raw_best.pth" ]] && [[ -f "${output}/candidate_raw.json" ]]; then
    echo "reusing end-to-end seed ${seed}: ${output}"
    return
  fi
  local resume_arg=""
  if [[ -d "${output}" ]] && [[ -n "$(ls -A "${output}" 2>/dev/null)" ]]; then
    resume_arg="--resume"
  fi
  "${PYTHON}" "${ROOT_DIR}/dta/train_hybrid_e2e_v2.py" \
    --train-manifest "${TRAIN_MANIFEST}" --valid-manifest "${VALID_MANIFEST}" \
    --train-native-cache "${TRAIN_NATIVE_CACHE}" --valid-native-cache "${VALID_NATIVE_CACHE}" \
    --gerna-checkpoint "${GERNA_CHECKPOINT}" --interface-checkpoint "${INTERFACE_CHECKPOINT}" \
    --cached-checkpoint "${cached}/Model_cached_best.pth" --output "${output}" \
    --seed "${seed}" --device "${DEVICE}" --epochs 10 --patience 2 --group-size 16 --dropout 0.1 \
    ${resume_arg:+"${resume_arg}"}
}

cd "${ROOT_DIR}"
mkdir -p "${MODEL_ROOT}"
echo "physical_gpu=${GPU_ID} logical_device=${DEVICE} memory_budget_gb=${MEMORY_BUDGET_GB}"
build_or_resume_cache train "${TRAIN_MANIFEST}" "${TRAIN_NATIVE_CACHE}" "${CACHE_ROOT}/train"
build_or_resume_cache valid "${VALID_MANIFEST}" "${VALID_NATIVE_CACHE}" "${CACHE_ROOT}/valid"
for seed in "${SEEDS[@]}"; do
  run_cached_seed "${seed}"
done
mapfile -t TOP_CACHED_SEEDS < <(rank_cached_seeds)
printf top_cached_seeds=%sn "${TOP_CACHED_SEEDS[*]}"
for seed in "${TOP_CACHED_SEEDS[@]}"; do
  run_e2e_seed "${seed}"
done

SELECT_ARGS=()
for seed in "${TOP_CACHED_SEEDS[@]}"; do
  cached="${MODEL_ROOT}/cached_seed${seed}"
  e2e="${MODEL_ROOT}/e2e_seed${seed}"
  SELECT_ARGS+=(--candidate "cached_seed${seed}=${cached}/valid_predictions.npz=${cached}/candidate_cached.json=${cached}/Model_cached_best.pth")
  SELECT_ARGS+=(--candidate "e2e_raw_seed${seed}=${e2e}/valid_predictions_raw.npz=${e2e}/candidate_raw.json=${e2e}/Model_e2e_raw_best.pth")
  if [[ -f "${e2e}/candidate_ema.json" ]]; then
    SELECT_ARGS+=(--candidate "e2e_ema_seed${seed}=${e2e}/valid_predictions_ema.npz=${e2e}/candidate_ema.json=${e2e}/Model_e2e_ema_best.pth")
  fi
done
"${PYTHON}" "${ROOT_DIR}/scripts/select_hybrid_upper_bound_v2.py" \
  "${SELECT_ARGS[@]}" --output "${MODEL_ROOT}/selection" --seed 42 --random-starts 32
"${PYTHON}" - "${MODEL_ROOT}/selection/selection.json" <<PY
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
ci = float(payload["selected_metrics"]["CI"])
print("selected_type=" + payload["selected_type"])
print("selected_validation_CI=" + str(ci))
print("selected_CI_minus_0.697287=" + str(ci - 0.697287))
print("members=" + ",".join(payload["members"]))
PY
