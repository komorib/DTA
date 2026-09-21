#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
CONFIRM_FINAL_TEST="${CONFIRM_FINAL_TEST:-}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${ROOT}/Model/LP_PDBBind_HybridPerformance_v1/final_test}"
TEST_MANIFEST="${ROOT}/data/PDBbind_chain_esm_residue_DTA/lp_pdbbind/test_manifest.json"
TEST_NATIVE="${ROOT}/data/LP_PDBBind_InterfaceFirstGerNA_v1/native_test"

if [[ -z "${SELECTION_MANIFEST}" ]]; then
  echo "SELECTION_MANIFEST is required" >&2
  exit 1
fi
if [[ "${SELECTION_MANIFEST}" != /* ]]; then
  SELECTION_MANIFEST="${ROOT}/${SELECTION_MANIFEST}"
fi
if [[ "${CONFIRM_FINAL_TEST}" != "YES" ]]; then
  echo "set CONFIRM_FINAL_TEST=YES for the one final test evaluation" >&2
  exit 1
fi
for path in \
  "${PY}" \
  "${SELECTION_MANIFEST}" \
  "${TEST_MANIFEST}" \
  "${TEST_NATIVE}/data.mdb" \
  "${TEST_NATIVE}/lock.mdb" \
  "${TEST_NATIVE}/metadata.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "required final-test input is missing: ${path}" >&2
    exit 1
  fi
done
if ! nvidia-smi -i "${GPU_ID}" --query-gpu=index,memory.total \
  --format=csv,noheader >/dev/null 2>&1; then
  echo "physical GPU ${GPU_ID} is unavailable" >&2
  exit 1
fi

cd "${ROOT}"
"${PY}" - "${SELECTION_MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
metrics = payload.get("validation_metrics")
if not isinstance(metrics, dict):
    raise ValueError("selection manifest has no validation metrics")
ci = float(metrics["CI"])
print("frozen_selection=" + str(path))
print("frozen_validation_metrics=" + json.dumps(
    metrics, sort_keys=True
))
if ci <= 0.697287:
    raise ValueError(
        f"validation CI {ci:.6f} does not exceed 0.697287"
    )
PY

echo "physical_gpu=${GPU_ID} logical_device=cuda:0"
echo "final_output=${FINAL_OUTPUT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
"${PY}" -u dta/eval_hybrid_affinity.py \
  --selection-manifest "${SELECTION_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --test-native-cache "${TEST_NATIVE}" \
  --output-dir "${FINAL_OUTPUT}" \
  --device cuda:0 \
  --confirm-final-test
