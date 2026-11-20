# DTA Experiment Results Archive

This archive preserves previously completed DTA experiment results and provides
a branch for future result reports. The model and training code remain on the
`DTA` branch; this branch adds result records only.

## Contents

- `reports/LP_PDBBind_experiment_record.md`: consolidated LP-PDBBind experiment
  ledger through the recorded 2026-11-08 update.
- `reports/LP_PDBBind_recent_experiments_20261108.md`: recent experiment status
  and result summary.
- `reports/LP_PDBBind_test_ablation_comparison_20261117.md`: official-split
  validation and test comparison for the final early-fusion model and ablation
  arms, including the sequence linear and embedding controls.
- `reports/LP_PDBBind_coldstart_complete_group_first_report.md`: random
  cold-start test results grouped by test cohort.
- `reports/LP_PDBBind_modality_ablation_official_v1_report.md`: original five
  official-split ablation protocol and validation results.
- `reports/pdbbind_protein_ligand_experiment_record.md`: earlier PDBbind
  protein-ligand setup and experiment record.
- `snapshots/`: aggregate JSON/CSV evidence and historical source summaries.

The snapshots contain aggregate metrics and protocol metadata only. Model
checkpoints, feature caches, datasets, logs, and per-example prediction arrays
are excluded. Some reports refer to original `.codex_runtime` locations as
provenance; the aggregate result files needed to inspect the reported metrics
are included here.

## Future results

Add new dated reports or aggregate metric files under this directory and update
`MANIFEST.sha256`. Keep model implementation changes on the `DTA` branch.
