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
SELECTION_JSON="${SELECTION_JSON:-${ROOT}/Model/LP_PDBBind_HybridPerformance_v1/dev_selection_top2.json}"
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
    echo "required continuation input is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${PREFLIGHT_ROOT}"
cd "${ROOT}"
SELECTION_ROWS="$(mktemp)"
trap 'rm -f "${SELECTION_ROWS}"' EXIT

"${PY}" - "${SELECTION_JSON}" "${ROOT}" "${DEV_ROOT}" > "${SELECTION_ROWS}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from utils.experiment_integrity import atomic_write_json


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
if len(entries) != 2:
    raise ValueError("selection must contain exactly two entries")

seen = set()
validated = []
for entry in entries:
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
            f"selection run path mismatch: {run_dir} != {expected_run}"
        )
    checkpoint = resolve_from_root(
        entry["training_checkpoint"], root
    )
    expected_checkpoint = run_dir / "training_checkpoint.pth"
    if checkpoint != expected_checkpoint:
        raise ValueError(
            "selection checkpoint path does not match its run directory"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_sha = str(entry["training_checkpoint_sha256"]).lower()
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError("training checkpoint SHA256 is invalid")

    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("fusion_type") != fusion
        or float(config.get("rank_weight")) != rank
        or int(config.get("seed")) != 42
    ):
        raise ValueError(f"run configuration mismatch: {run_dir}")
    identity = (fusion, rank_text, str(run_dir))
    if identity in seen:
        raise ValueError("selection contains duplicate entries")
    seen.add(identity)

    marker_path = run_dir / ".hybrid_continuation_selection.json"
    marker = {
        "schema_version": 1,
        "fusion_type": fusion,
        "rank_weight": rank,
        "run_dir": str(run_dir),
        "training_checkpoint": str(checkpoint),
        "selected_training_checkpoint_sha256": expected_sha,
    }
    if marker_path.is_file():
        actual_marker = json.loads(
            marker_path.read_text(encoding="utf-8")
        )
        if actual_marker != marker:
            raise ValueError(
                f"continuation selection marker mismatch: {marker_path}"
            )
        marker_required = False
    else:
        actual_sha = sha256_file(checkpoint)
        if expected_sha != actual_sha:
            raise ValueError(
                f"checkpoint SHA256 mismatch before continuation: {checkpoint}"
            )
        marker_required = True
    validated.append(
        (
            fusion,
            rank_text,
            run_dir,
            checkpoint,
            marker_path,
            marker,
            marker_required,
        )
    )

for (
    fusion,
    rank_text,
    run_dir,
    checkpoint,
    marker_path,
    marker,
    marker_required,
) in validated:
    if marker_required:
        atomic_write_json(marker_path, marker)
    print(fusion, rank_text, run_dir, checkpoint, sep="\t")
PY

while IFS=$'\t' read -r fusion rank_weight run_dir checkpoint; do
  if [[ -z "${fusion}" || -z "${checkpoint}" ]]; then
    echo "invalid empty continuation selection row" >&2
    exit 1
  fi
  echo "continuing fusion=${fusion} rank_weight=${rank_weight}"
  echo "checkpoint=${checkpoint}"
  echo "physical_gpu=${GPU_ID} logical_device=cuda:0"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  "${PY}" -u dta/train_hybrid_affinity.py \
    --fusion-type "${fusion}" \
    --rank-weight "${rank_weight}" \
    --seed 42 \
    --head-epochs 2 \
    --high-epochs 15 \
    --full-epochs 40 \
    --output-dir "${run_dir}" \
    --resume \
    "${fixed_inputs[@]}"
done < "${SELECTION_ROWS}"
