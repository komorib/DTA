#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
DEVICE="cuda:0"
MEMORY_BUDGET_GB="40"
SEEDS=(1 7 21 42 87)

MANIFEST_ROOT="${ROOT_DIR}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind"
UPPER_CACHE_ROOT="${ROOT_DIR}/data/LP_PDBBind_HybridFeatureCache_v3"
UPPER_CHECKPOINT="${ROOT_DIR}/Model/LP_PDBBind_HybridUpperBound_v2/cached_seed87/Model_cached_best.pth"
BOLTZ_CHECKPOINT="${ROOT_DIR}/external_benchmarks/results/boltz_pairwise_rank010_pairDelta_save/model.pt"
THREED_CHECKPOINT="${ROOT_DIR}/external_benchmarks/results/3dprot_dta_lp_official_ckpt/best_model.pt"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/data/LP_PDBBind_RankFusionCache_v4}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/Model/LP_PDBBind_RankFusion_v4}"
ARTIFACT_ROOT="${CACHE_ROOT}/experts"
JOIN_ROOT="${CACHE_ROOT}/joined"
NORMALIZATION_PATH="${CACHE_ROOT}/normalization.json"
SELECTION_ROOT="${MODEL_ROOT}/selection"
TEST_RESULT_ROOT="${MODEL_ROOT}/test_result"

BOLTZ_LMDB_DIR="${BOLTZ_LMDB_DIR:-$("${PYTHON}" -c 'import torch; p=torch.load("'"${BOLTZ_CHECKPOINT}"'",map_location="cpu"); print(p["config"]["merged_lmdb_dir"])')}"
BOLTZ_INDEX_PATH="${BOLTZ_INDEX_PATH:-$("${PYTHON}" -c 'import torch; p=torch.load("'"${BOLTZ_CHECKPOINT}"'",map_location="cpu"); print(p["config"]["index_path"])')}"
THREED_DATA_DIR="${THREED_DATA_DIR:-$("${PYTHON}" -c 'import pathlib,torch; p=torch.load("'"${THREED_CHECKPOINT}"'",map_location="cpu"); value=pathlib.Path(p["args"]["data_dir"]); print(value if value.is_absolute() else pathlib.Path("'"${ROOT_DIR}"'")/value)')}"

manifest_for_split() {
  local split="$1"
  printf '%s/%s_manifest.json\n' "${MANIFEST_ROOT}" "${split}"
}

preflight_paths() {
  local required_files=(
    "${UPPER_CHECKPOINT}"
    "${BOLTZ_CHECKPOINT}"
    "${THREED_CHECKPOINT}"
    "${BOLTZ_INDEX_PATH}"
    "${MANIFEST_ROOT}/train_manifest.json"
    "${MANIFEST_ROOT}/valid_manifest.json"
    "${MANIFEST_ROOT}/test_manifest.json"
    "${THREED_DATA_DIR}/full.csv"
    "${THREED_DATA_DIR}/folds/train_fold_setting1.txt"
    "${THREED_DATA_DIR}/folds/test_fold_setting1.txt"
    "${THREED_DATA_DIR}/ligand_to_graph.pkl"
    "${THREED_DATA_DIR}/protein_to_graph.pkl"
    "${THREED_DATA_DIR}/ligand_to_ecfp.pkl"
  )
  local required_dirs=(
    "${BOLTZ_LMDB_DIR}"
    "${UPPER_CACHE_ROOT}/train"
    "${UPPER_CACHE_ROOT}/valid"
    "${UPPER_CACHE_ROOT}/test"
  )
  local path
  for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
      printf 'preflight_missing_file: %s\n' "${path}" >&2
      return 1
    fi
  done
  for path in "${required_dirs[@]}"; do
    if [[ ! -d "${path}" ]]; then
      printf 'preflight_missing_directory: %s\n' "${path}" >&2
      return 1
    fi
  done
  printf 'physical_gpu=%s logical_device=%s memory_budget_gb=%s\n' \
    "${GPU_ID}" "${DEVICE}" "${MEMORY_BUDGET_GB}"
  printf 'three_dprot_data=%s\n' "${THREED_DATA_DIR}"
}

export_split() {
  local split="$1"
  local limit="${2:-}"
  local limit_args=()
  if [[ -n "${limit}" ]]; then
    limit_args=(--limit "${limit}")
  fi
  "${PYTHON}" -m dta.rank_fusion_v4.export_expert \
    --expert upper_bound \
    --manifest "$(manifest_for_split "${split}")" \
    --checkpoint "${UPPER_CHECKPOINT}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --split "${split}" --device "${DEVICE}" --batch-size 64 \
    --upper-bound-cache "${UPPER_CACHE_ROOT}/${split}" \
    ${limit_args[@]+"${limit_args[@]}"}
  "${PYTHON}" -m dta.rank_fusion_v4.export_expert \
    --expert boltz_pairwise \
    --manifest "$(manifest_for_split "${split}")" \
    --checkpoint "${BOLTZ_CHECKPOINT}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --split "${split}" --device "${DEVICE}" --batch-size 64 \
    --boltz-lmdb-dir "${BOLTZ_LMDB_DIR}" \
    --boltz-index-path "${BOLTZ_INDEX_PATH}" \
    ${limit_args[@]+"${limit_args[@]}"}
  "${PYTHON}" -m dta.rank_fusion_v4.export_expert \
    --expert three_dprot \
    --manifest "$(manifest_for_split "${split}")" \
    --checkpoint "${THREED_CHECKPOINT}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --split "${split}" --device "${DEVICE}" --batch-size 32 \
    --three-dprot-data-dir "${THREED_DATA_DIR}" \
    ${limit_args[@]+"${limit_args[@]}"}
}

build_join() {
  local split="$1"
  local limit="${2:-}"
  local limit_args=()
  local normalization_args=()
  if [[ -n "${limit}" ]]; then
    limit_args=(--limit "${limit}")
  fi
  if [[ "${split}" == "train" ]]; then
    normalization_args=(--normalization-output "${NORMALIZATION_PATH}")
  else
    normalization_args=(--normalization-input "${NORMALIZATION_PATH}")
  fi
  "${PYTHON}" -m dta.rank_fusion_v4.build_cache \
    --manifest "$(manifest_for_split "${split}")" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --output "${JOIN_ROOT}/${split}" \
    --split "${split}" \
    "${normalization_args[@]}" \
    ${limit_args[@]+"${limit_args[@]}"}
}

train_seeds() {
  local epochs="$1"
  local patience="$2"
  shift 2
  local seeds=("$@")
  local seed resume_arg
  for seed in "${seeds[@]}"; do
    resume_arg=""
    if [[ -f "${MODEL_ROOT}/seed${seed}/checkpoint_last.pt" || -f "${MODEL_ROOT}/seed${seed}/summary.json" ]]; then
      resume_arg="--resume"
    fi
    "${PYTHON}" -m dta.rank_fusion_v4.train \
      --train-cache "${JOIN_ROOT}/train" \
      --valid-cache "${JOIN_ROOT}/valid" \
      --output "${MODEL_ROOT}/seed${seed}" \
      --seed "${seed}" --device "${DEVICE}" \
      --epochs "${epochs}" --patience "${patience}" --batch-size 64 \
      --learning-rate 2e-4 --weight-decay 1e-3 \
      ${resume_arg:+"${resume_arg}"}
  done
}

select_validation() {
  local expected_count="$1"
  shift
  local seeds=("$@")
  local arguments=()
  local seed
  for seed in "${seeds[@]}"; do
    arguments+=(--run "${seed}=${MODEL_ROOT}/seed${seed}")
    arguments+=(--expected-seed "${seed}")
  done
  "${PYTHON}" -m dta.rank_fusion_v4.select \
    --valid-cache "${JOIN_ROOT}/valid" \
    "${arguments[@]}" \
    --output "${SELECTION_ROOT}" \
    --expected-count "${expected_count}"
}

evaluate_test() {
  local expected_count="$1"
  "${PYTHON}" -m dta.rank_fusion_v4.evaluate_test \
    --selection "${SELECTION_ROOT}" \
    --test-cache "${JOIN_ROOT}/test" \
    --output "${TEST_RESULT_ROOT}" \
    --device "${DEVICE}" --batch-size 256 \
    --expected-count "${expected_count}"
}

write_pipeline_summary() {
  "${PYTHON}" -c 'import json,sys; from pathlib import Path; model=Path(sys.argv[1]); selection=json.loads((model/"selection"/"selection.json").read_text()); result=json.loads((model/"test_result"/"summary.json").read_text()); seeds={path.parent.name:json.loads(path.read_text())["best_metrics"] for path in sorted(model.glob("seed*/summary.json"))}; payload={"status":"complete","seed_validation_metrics":seeds,"selection":selection,"test":result}; temporary=model/"pipeline_summary.json.tmp"; temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)); temporary.replace(model/"pipeline_summary.json")' "${MODEL_ROOT}"
}
