#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT}/data/OptionalStructureMultitask_v21/manifests_v2}"
LIGAND_CACHE="${LIGAND_CACHE:-${ROOT}/data/OptionalStructureMultitask_v21/ligands_v2.lmdb}"

cd "${ROOT}"
if [[ ! -f "${MANIFEST_ROOT}/metadata.json" ]]; then
  "${PYTHON_BIN}" -m dta.optional_structure_multitask_v21.build_manifests \
    --output "${MANIFEST_ROOT}"
fi
if [[ ! -f "${LIGAND_CACHE}/metadata.json" ]]; then
  "${PYTHON_BIN}" -m dta.optional_structure_multitask_v21.build_ligand_cache \
    --manifests \
      "${MANIFEST_ROOT}/train.jsonl" \
      "${MANIFEST_ROOT}/valid.jsonl" \
      "${MANIFEST_ROOT}/test.jsonl" \
    --output "${LIGAND_CACHE}"
fi

echo "manifest_root=${MANIFEST_ROOT}"
echo "ligand_cache=${LIGAND_CACHE}"
