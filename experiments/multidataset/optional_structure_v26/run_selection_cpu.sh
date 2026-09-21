#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
RUN_ID="${RUN_ID:-optional_structure_simplex_router_v26_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-$REPO_ROOT/Model/OptionalStructureSimplexRouter_v26/$RUN_ID}"

"$PYTHON_BIN" -m dta.optional_structure_simplex_router_v26.run_selection \
  --train-manifest "$REPO_ROOT/data/OptionalStructureMultitask_v21/manifests_v2/train.jsonl" \
  --valid-manifest "$REPO_ROOT/data/OptionalStructureMultitask_v21/manifests_v2/valid.jsonl" \
  --v22-valid "$REPO_ROOT/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/valid_predictions.npz" \
  --v20-valid "$REPO_ROOT/Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20/msa_interface_pair_latent_stability_fusion_v20_full_20260902_140000/valid_predictions.npz" \
  --v25-valid "$REPO_ROOT/Model/OptionalStructureCalibratedRouter_v25/optional_structure_calibrated_router_v25_full_20260910_143300/valid_predictions.npz" \
  --output "$OUTPUT"

echo "v26_selection_output=$OUTPUT"
echo "v26_selection_result=$OUTPUT/selection.json"
