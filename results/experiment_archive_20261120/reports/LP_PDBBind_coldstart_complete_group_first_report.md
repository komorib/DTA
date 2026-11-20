# LP-PDBBind Complete Cold-Start Report

## Protocol

- Random seed-42 split: train/valid/test = 14,502/1,813/1,813.
- Test groups: `cold_ligand` = 804, `cold_protein` = 253, `cold_both` = 565, `both_seen` = 191.
- The test set was not used for model selection or training.
- Every row reports CI, RMSE, MAE, Pearson, and Spearman.
- The five reported ablation arms are the original v3 arms: `1D-only`, `3D-only`, `No 3D + no ESM`, `No 3D + no MSA`, and `No 3D + no MolFormer`.

## Group: All

| Model | n | CI | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Full early fusion | 1813 | 0.7852 | 1.2369 | 0.9324 | 0.7588 | 0.7586 |
| 1D-only | 1813 | 0.7805 | 1.2493 | 0.9261 | 0.7502 | 0.7442 |
| 3D-only | 1813 | 0.7425 | 1.4085 | 1.0893 | 0.6644 | 0.6708 |
| No 3D + no ESM | 1813 | 0.7823 | 1.2438 | 0.9194 | 0.7527 | 0.7485 |
| No 3D + no MSA | 1813 | 0.7834 | 1.2238 | 0.9146 | 0.7605 | 0.7516 |
| No 3D + no MolFormer | 1813 | 0.7201 | 1.5014 | 1.1304 | 0.6066 | 0.6103 |

## Group: Cold Ligand

| Model | n | CI | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Full early fusion | 804 | 0.8066 | 1.1340 | 0.8494 | 0.8012 | 0.8023 |
| 1D-only | 804 | 0.7930 | 1.1962 | 0.8976 | 0.7747 | 0.7726 |
| 3D-only | 804 | 0.7567 | 1.3588 | 1.0493 | 0.6964 | 0.7029 |
| No 3D + no ESM | 804 | 0.7999 | 1.1711 | 0.8774 | 0.7843 | 0.7876 |
| No 3D + no MSA | 804 | 0.7956 | 1.1855 | 0.8845 | 0.7794 | 0.7769 |
| No 3D + no MolFormer | 804 | 0.7306 | 1.4636 | 1.0945 | 0.6304 | 0.6323 |

## Group: Cold Protein

| Model | n | CI | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Full early fusion | 253 | 0.7593 | 1.3976 | 1.0751 | 0.7159 | 0.7014 |
| 1D-only | 253 | 0.7697 | 1.3741 | 1.0004 | 0.7247 | 0.7097 |
| 3D-only | 253 | 0.7298 | 1.4989 | 1.1627 | 0.6457 | 0.6330 |
| No 3D + no ESM | 253 | 0.7678 | 1.3989 | 1.0209 | 0.7148 | 0.7063 |
| No 3D + no MSA | 253 | 0.7689 | 1.3687 | 1.0117 | 0.7267 | 0.7123 |
| No 3D + no MolFormer | 253 | 0.7107 | 1.6190 | 1.2497 | 0.5735 | 0.5971 |

## Group: Cold Both

| Model | n | CI | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Full early fusion | 565 | 0.7434 | 1.3606 | 1.0366 | 0.6596 | 0.6702 |
| 1D-only | 565 | 0.7423 | 1.3373 | 1.0042 | 0.6646 | 0.6630 |
| 3D-only | 565 | 0.7086 | 1.4516 | 1.1320 | 0.5742 | 0.5938 |
| No 3D + no ESM | 565 | 0.7391 | 1.3443 | 0.9936 | 0.6572 | 0.6576 |
| No 3D + no MSA | 565 | 0.7480 | 1.2840 | 0.9806 | 0.6882 | 0.6780 |
| No 3D + no MolFormer | 565 | 0.6917 | 1.5317 | 1.1507 | 0.5217 | 0.5431 |

## Group: Both Seen

| Model | n | CI | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Full early fusion | 191 | 0.8405 | 1.0228 | 0.7848 | 0.8659 | 0.8614 |
| 1D-only | 191 | 0.8492 | 0.9999 | 0.7163 | 0.8716 | 0.8712 |
| 3D-only | 191 | 0.7866 | 1.3608 | 1.0338 | 0.7471 | 0.7647 |
| No 3D + no ESM | 191 | 0.8456 | 0.9869 | 0.7425 | 0.8759 | 0.8691 |
| No 3D + no MSA | 191 | 0.8503 | 0.9709 | 0.7174 | 0.8792 | 0.8719 |
| No 3D + no MolFormer | 191 | 0.7621 | 1.4031 | 1.0639 | 0.7286 | 0.6962 |

## Sources and comparability

- Original modality-ablation results: `.codex_runtime/modality-ablation-coldstart-v3-coldstart-test-gpu7/coldstart_comparison.json`.
- Full early-fusion reference: `.codex_runtime/random-full-v2-ablations-coldstart-test-v1-20261114/coldstart_comparison.csv`.
- Full and ablation runs use different training-loss implementations, so numerical proximity is descriptive rather than a proof of identical optimization conditions.
