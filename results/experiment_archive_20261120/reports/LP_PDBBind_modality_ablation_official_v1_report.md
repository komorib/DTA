# LP-PDBBind Official Modality Ablation Report

## Protocol

- Split: `lp_pdbbind_official`
- Counts: train/valid/test = `11037/2340/4751`
- Train manifest SHA256: `def47b0307750099bcd8accc17ac5c6960d4d92ffc789daac2cadb10faf55213`
- Valid manifest SHA256: `dc996a2665a07a3a708b7a5b7624f27f8ba3bde881bd18fe4569a2d65e3943d8`
- Training budget: 20,700 optimizer steps per arm
- Selection: validation CI lexicographic policy; EMA checkpoint
- The official test split was not read during training or model selection. The results below are validation results.

The five ablation arms use the strict input contracts implemented in
`dta/lp_pdbbind_modality_ablation_lp_v1/spec.py`:

| Arm | Retained inputs |
|---|---|
| `3d_only` | Pocket geometry, pocket coordinates/mask, ligand atom features/coordinates/mask, bond features |
| `1d_only` | Full-chain ESM, full-chain MSA, full-chain mask, global MoLFormer molecule token |
| `no3d_noesm` | Full-chain MSA, full-chain mask, global MoLFormer molecule token |
| `no3d_nomsa` | Full-chain ESM, full-chain mask, global MoLFormer molecule token |
| `no3d_nomolformer` | Full-chain ESM, full-chain MSA, full-chain mask |

## Validation Results

| Model | CI | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| **Full model: full-chain MolFormer branch** | **0.7167** | **1.2588** | **0.9810** | **0.6331** | **0.6102** |
| 1D only | 0.7000 | 1.3373 | 1.0328 | 0.5820 | 0.5689 |
| 3D only | 0.6778 | 1.4076 | 1.0895 | 0.5271 | 0.5094 |
| No 3D + no ESM | 0.6971 | 1.3503 | 1.0453 | 0.5767 | 0.5624 |
| No 3D + no MSA | 0.7006 | 1.3317 | 1.0282 | 0.5854 | 0.5714 |
| No 3D + no MolFormer | 0.6311 | 1.5345 | 1.1934 | 0.3749 | 0.3841 |

## Full Model Gain Relative To Each Ablation

Positive values mean the full model is better. For RMSE and MAE, the values
are reported as `ablation - full`, so positive values also mean improvement.

| Comparison | ΔCI | ΔRMSE | ΔMAE | ΔPearson | ΔSpearman |
|---|---:|---:|---:|---:|---:|
| Full - 1D only | +0.0167 | +0.0785 | +0.0517 | +0.0512 | +0.0413 |
| Full - 3D only | +0.0389 | +0.1488 | +0.1089 | +0.1060 | +0.1008 |
| Full - no3d_noesm | +0.0196 | +0.0915 | +0.0643 | +0.0564 | +0.0478 |
| Full - no3d_nomsa | +0.0161 | +0.0729 | +0.0472 | +0.0477 | +0.0388 |
| Full - no3d_nomolformer | +0.0856 | +0.2757 | +0.2124 | +0.2583 | +0.2261 |

The largest loss comes from removing MoLFormer: CI decreases from `0.7167` to
`0.6311`. The 3D-only arm is also substantially below the full model, while
the three non-3D 1D variants remain much closer to the full model.

## Run Evidence

### Full model

- Summary: `.codex_runtime/fullchain-molformer-gpu7-20261109_210714-16506/summary.json`
- Version: `lp-pdbbind-fullchain-molformer-molecule-branch-v1`
- Validation metrics: CI `0.7166669844`, RMSE `1.2588422773`, MAE `0.9810297717`, Pearson `0.6331461222`, Spearman `0.6102305164`
- Test manifest read: `false`

### Five official ablation arms

All five summaries report `status=training_complete`, 20,700 optimizer steps,
the same readiness SHA256
`69ab891bf1dc25aa52f29556272001de6b02abe68767e7f4d4a08020c0283058`, and
`test_manifest_read=false`:

- `.codex_runtime/modality-ablation-lp-v1-gpu7/1d_only/summary.json`
- `.codex_runtime/modality-ablation-lp-v1-gpu7/3d_only/summary.json`
- `.codex_runtime/modality-ablation-lp-v1-gpu7/no3d_noesm/summary.json`
- `.codex_runtime/modality-ablation-lp-v1-gpu7/no3d_nomsa/summary.json`
- `.codex_runtime/modality-ablation-lp-v1-gpu7/no3d_nomolformer/summary.json`

The reproducible code and GPU-7 launchers are in
`dta/lp_pdbbind_modality_ablation_lp_v1/` and
`experiments/lp_pdbbind/modality_ablation_lp_v1/`.
