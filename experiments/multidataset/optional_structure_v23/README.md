# Optional-Structure Collaborative Router V23

V23 is a CPU post-training router over frozen V20/V22 predictions.

- LP-PDBBind with 3D: validation-selected V20/V22 blend.
- LP-PDBBind without 3D: V22 sequence/ligand base fallback.
- Davis/KIBA: validation-selected train-only Bias-SVD blend over the V22 base.
- Unknown protein or ligand IDs hard-fallback to V22.
- Exact evaluation pairs are excluded before Bias-SVD fitting.
- Affinity blend, binary score blend, and binary decision threshold are selected only on validation.
- Test input is not accepted by the selection CLI. The test CLI verifies SHA-256 seals before reading test.

Run validation selection:

```bash
RUN_ID=optional_structure_collaborative_router_v23_full_$(date +%Y%m%d_%H%M%S) \
bash experiments/multidataset/optional_structure_v23/run_selection_cpu.sh
```

Only if `status=validation_passed`, report test once:

```bash
SELECTION_OUTPUT=/absolute/path/from/the/selection/command \
bash experiments/multidataset/optional_structure_v23/run_test_cpu.sh
```
