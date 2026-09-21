#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python}"
SELECTION_OUTPUT="${SELECTION_OUTPUT:?set SELECTION_OUTPUT to a passed V23 validation selection directory}"

"$PYTHON_BIN" -m dta.optional_structure_collaborative_router_v23.run_test \
  --selection-output "$SELECTION_OUTPUT" \
  --test-manifest "$REPO_ROOT/data/OptionalStructureMultitask_v21/manifests_v2/test.jsonl" \
  --v22-test "$REPO_ROOT/Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/test_predictions.npz" \
  --v20-test "$REPO_ROOT/Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20/msa_interface_pair_latent_stability_fusion_v20_full_20260902_140000/test_predictions.npz"

echo "v23_result=$SELECTION_OUTPUT/result.json"
