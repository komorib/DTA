#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
SMOKE_SAMPLE_COUNT=16
SMOKE_SEED=42
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/LP_PDBBind_HybridPerformance_v1_smoke_${RUN_ID}}"
INPUT_ROOT="${SMOKE_ROOT}/inputs"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${SMOKE_ROOT}/run}"

SOURCE_TRAIN_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/train_manifest.json"
SOURCE_VALID_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/valid_manifest.json"
SOURCE_TRAIN_NATIVE="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_train"
SOURCE_VALID_NATIVE="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_valid"
GERNA_CHECKPOINT="${ROOT}/Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth"
INTERFACE_CHECKPOINT="${ROOT}/Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth"

TRAIN_MANIFEST="${INPUT_ROOT}/train_manifest.json"
VALID_MANIFEST="${INPUT_ROOT}/valid_manifest.json"
TRAIN_NATIVE="${INPUT_ROOT}/native_train"
VALID_NATIVE="${INPUT_ROOT}/native_valid"

fixed_inputs=(
  --train-manifest "${TRAIN_MANIFEST}"
  --valid-manifest "${VALID_MANIFEST}"
  --train-native-cache "${TRAIN_NATIVE}"
  --valid-native-cache "${VALID_NATIVE}"
  --gerna-checkpoint "${GERNA_CHECKPOINT}"
  --interface-checkpoint "${INTERFACE_CHECKPOINT}"
  --device cuda:0
  --num-workers 0
  --memory-budget-gb 40
)

for path in \
  "${SOURCE_TRAIN_MANIFEST}" \
  "${SOURCE_VALID_MANIFEST}" \
  "${SOURCE_TRAIN_NATIVE}/data.mdb" \
  "${SOURCE_TRAIN_NATIVE}/lock.mdb" \
  "${SOURCE_TRAIN_NATIVE}/metadata.json" \
  "${SOURCE_VALID_NATIVE}/data.mdb" \
  "${SOURCE_VALID_NATIVE}/lock.mdb" \
  "${SOURCE_VALID_NATIVE}/metadata.json" \
  "${GERNA_CHECKPOINT}" \
  "${INTERFACE_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required smoke input is missing: ${path}" >&2
    exit 1
  fi
done

if [[ -e "${SMOKE_ROOT}" ]]; then
  if [[ ! -d "${SMOKE_ROOT}" ]]; then
    echo "smoke root is not a directory: ${SMOKE_ROOT}" >&2
    exit 1
  fi
  shopt -s nullglob dotglob
  existing=("${SMOKE_ROOT}"/*)
  if (( ${#existing[@]} != 0 )); then
    echo "refusing nonempty smoke root: ${SMOKE_ROOT}" >&2
    exit 1
  fi
fi
mkdir -p "${INPUT_ROOT}"

cd "${ROOT}"
"${PY}" - \
  "${SOURCE_TRAIN_MANIFEST}" "${SOURCE_TRAIN_NATIVE}" \
  "${SOURCE_VALID_MANIFEST}" "${SOURCE_VALID_NATIVE}" \
  "${TRAIN_MANIFEST}" "${TRAIN_NATIVE}" \
  "${VALID_MANIFEST}" "${VALID_NATIVE}" \
  "${SMOKE_SAMPLE_COUNT}" "${SMOKE_SEED}" <<'PY'
import json
from pathlib import Path
import random
import sys

from utils.experiment_integrity import (
    manifest_fingerprint,
    sha256_file,
)


def build_view(
    source_manifest,
    source_cache,
    output_manifest,
    output_cache,
    expected_split,
    count,
    seed,
):
    source_manifest = Path(source_manifest).resolve()
    source_cache = Path(source_cache).resolve()
    output_manifest = Path(output_manifest).resolve()
    output_cache = Path(output_cache).resolve()
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if payload.get("split") != expected_split:
        raise ValueError(
            f"{source_manifest} is not split {expected_split}"
        )
    source_fingerprint = manifest_fingerprint(source_manifest)
    source_metadata_path = source_cache / "metadata.json"
    source_metadata = json.loads(
        source_metadata_path.read_text(encoding="utf-8")
    )
    if source_metadata.get("status") != "complete":
        raise RuntimeError(
            f"source native cache is not complete: {source_cache}"
        )
    if source_metadata.get("manifest") != source_fingerprint:
        raise RuntimeError(
            "source native cache manifest provenance does not match "
            + str(source_manifest)
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < count:
        raise ValueError(
            f"{source_manifest} has fewer than {count} samples"
        )

    indices = sorted(
        random.Random(f"{seed}:{expected_split}").sample(
            range(len(samples)), count
        )
    )
    subset = dict(payload)
    subset["samples"] = [samples[index] for index in indices]
    for field in ("structure_lmdb_path", "lmdb_path"):
        value = subset.get(field)
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = source_manifest.parent / path
            subset[field] = str(path.resolve())

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fingerprint = manifest_fingerprint(output_manifest)

    output_cache.mkdir(parents=True, exist_ok=False)
    for name in ("data.mdb", "lock.mdb"):
        (output_cache / name).symlink_to(
            (source_cache / name).resolve()
        )
    metadata = {
        "schema_version": 1,
        "builder": "hybrid_affinity_smoke_native_view",
        "status": "complete",
        "manifest": fingerprint,
        "source_native_cache": str(source_cache),
        "source_metadata_sha256": sha256_file(
            source_cache / "metadata.json"
        ),
    }
    (output_cache / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


(
    source_train_manifest,
    source_train_cache,
    source_valid_manifest,
    source_valid_cache,
    output_train_manifest,
    output_train_cache,
    output_valid_manifest,
    output_valid_cache,
    count,
    seed,
) = sys.argv[1:]
build_view(
    source_train_manifest,
    source_train_cache,
    output_train_manifest,
    output_train_cache,
    "train",
    int(count),
    int(seed),
)
build_view(
    source_valid_manifest,
    source_valid_cache,
    output_valid_manifest,
    output_valid_cache,
    "valid",
    int(count),
    int(seed),
)
PY

echo "physical_gpu: ${GPU_ID}"
echo "logical_device: cuda:0"
echo "smoke_root: ${SMOKE_ROOT}"
echo "output_dir: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
"${PY}" -u dta/train_hybrid_affinity.py \
  --fusion-type transformer \
  --rank-weight 0.2 \
  --seed "${SMOKE_SEED}" \
  --head-epochs 1 \
  --high-epochs 1 \
  --full-epochs 1 \
  --output-dir "${OUTPUT_DIR}" \
  "${fixed_inputs[@]}"

cat "${OUTPUT_DIR}/memory_preflight.json"
cat "${OUTPUT_DIR}/best_metrics.json"
