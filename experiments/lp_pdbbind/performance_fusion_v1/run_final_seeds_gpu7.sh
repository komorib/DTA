#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
VALID_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/valid_manifest.json"
TRAIN_NATIVE="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_train"
VALID_NATIVE="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_valid"
GERNA_CHECKPOINT="${ROOT}/Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth"
INTERFACE_CHECKPOINT="${ROOT}/Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth"
PREFLIGHT_ROOT="${ROOT}/data/LP_PDBBind_HybridAffinityPreflight"
DEV_ROOT="${ROOT}/Model/LP_PDBBind_HybridPerformance_v1/dev_seed42"
FINAL_ROOT="${ROOT}/Model/LP_PDBBind_HybridPerformance_v1/final_seeds"
SELECTION_JSON="${SELECTION_JSON:-${ROOT}/Model/LP_PDBBind_HybridPerformance_v1/winner.json}"
if [[ "${SELECTION_JSON}" != /* ]]; then
  SELECTION_JSON="${ROOT}/${SELECTION_JSON}"
fi

fixed_inputs=(
  --train-manifest "${TRAIN_MANIFEST}"
  --valid-manifest "${VALID_MANIFEST}"
  --train-native-cache "${TRAIN_NATIVE}"
  --valid-native-cache "${VALID_NATIVE}"
  --gerna-checkpoint "${GERNA_CHECKPOINT}"
  --interface-checkpoint "${INTERFACE_CHECKPOINT}"
  --train-preflight-report "${PREFLIGHT_ROOT}/train.json"
  --valid-preflight-report "${PREFLIGHT_ROOT}/valid.json"
  --device cuda:0
  --num-workers 0
  --memory-budget-gb 40
)

for path in \
  "${TRAIN_MANIFEST}" \
  "${VALID_MANIFEST}" \
  "${TRAIN_NATIVE}" \
  "${VALID_NATIVE}" \
  "${GERNA_CHECKPOINT}" \
  "${INTERFACE_CHECKPOINT}" \
  "${SELECTION_JSON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required final-seed input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${PREFLIGHT_ROOT}" "${FINAL_ROOT}"
WINNER_ROW="$(mktemp)"
trap 'rm -f "${WINNER_ROW}"' EXIT

"${PY}" - "${SELECTION_JSON}" "${ROOT}" "${DEV_ROOT}" > "${WINNER_ROW}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def resolve_from_root(value, root):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


selection_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
dev_root = Path(sys.argv[3]).resolve()
payload = json.loads(selection_path.read_text(encoding="utf-8"))
entries = payload.get("entries")
if len(entries) != 1:
    raise ValueError("winner must contain exactly one entry")

entry = entries[0]
fusion = str(entry["fusion_type"])
rank = float(entry["rank_weight"])
if fusion not in {"mlp", "transformer"}:
    raise ValueError(f"invalid fusion type: {fusion}")
if rank not in {0.1, 0.2, 0.4}:
    raise ValueError(f"invalid rank weight: {rank}")
rank_text = f"{rank:.1f}"
expected_run = (dev_root / f"{fusion}_rank{rank_text}").resolve()
run_dir = resolve_from_root(entry["run_dir"], root)
if run_dir != expected_run:
    raise ValueError(
        f"winner run path mismatch: {run_dir} != {expected_run}"
    )
checkpoint = resolve_from_root(entry["training_checkpoint"], root)
expected_checkpoint = run_dir / "training_checkpoint.pth"
if checkpoint != expected_checkpoint:
    raise ValueError(
        "winner checkpoint path does not match its run directory"
    )
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
expected_sha = str(entry["training_checkpoint_sha256"]).lower()
actual_sha = sha256_file(checkpoint)
if expected_sha != actual_sha:
    raise ValueError(f"checkpoint SHA256 mismatch: {checkpoint}")

config = json.loads(
    (run_dir / "run_config.json").read_text(encoding="utf-8")
)
if (
    config.get("fusion_type") != fusion
    or float(config.get("rank_weight")) != rank
    or int(config.get("seed")) != 42
):
    raise ValueError(f"winner run configuration mismatch: {run_dir}")
print(fusion, rank_text, sep="\t")
PY

IFS=$'\t' read -r fusion rank_weight < "${WINNER_ROW}"
if [[ -z "${fusion}" || -z "${rank_weight}" ]]; then
  echo "invalid empty winner row" >&2
  exit 1
fi

cd "${ROOT}"
shopt -s nullglob dotglob
for seed in 1 7 21 42 87; do
  output="${FINAL_ROOT}/${fusion}_rank${rank_weight}_seed${seed}"
  checkpoint="${output}/training_checkpoint.pth"
  resume_args=()
  if [[ -e "${output}" && ! -d "${output}" ]]; then
    echo "final-seed output is not a directory: ${output}" >&2
    exit 1
  fi
  if [[ -f "${checkpoint}" ]]; then
    resume_args=(--resume)
  elif [[ -d "${output}" ]]; then
    existing=("${output}"/*)
    if (( ${#existing[@]} != 0 )); then
      echo "refusing nonempty output without checkpoint: ${output}" >&2
      exit 1
    fi
  fi

  echo "starting fusion=${fusion} rank_weight=${rank_weight} seed=${seed}"
  echo "physical_gpu=${GPU_ID} logical_device=cuda:0"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  "${PY}" -u dta/train_hybrid_affinity.py \
    --fusion-type "${fusion}" \
    --rank-weight "${rank_weight}" \
    --seed "${seed}" \
    --head-epochs 2 \
    --high-epochs 15 \
    --full-epochs 40 \
    --output-dir "${output}" \
    "${resume_args[@]}" \
    "${fixed_inputs[@]}"
done
