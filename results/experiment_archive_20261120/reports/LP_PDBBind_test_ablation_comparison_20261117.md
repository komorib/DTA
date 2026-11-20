# LP-PDBBind 消融实验 Validation / Test 结果

数据划分为官方 LP-PDBBind split：validation 2,340 条，test 4,751 条。test manifest SHA-256：`f8741d40837568c552ca0bd3725e0b079ba855d56d2a663c0bb95c4bc09c95b8`。原五个消融及新增序列线性层消融使用相同训练协议；test 仅用于固定 checkpoint 的最终评估，没有参与模型选择。

## Validation

| 模型 | 输入模态 | CI | RMSE | MAE | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 完整模型（仅前融合） | ESM + MSA + 3D + MoLFormer K/V | **0.716777** | **1.265875** | **0.981389** | **0.631061** | **0.609087** |
| 1D only | ESM + MSA + MoLFormer | 0.699989 | 1.337320 | 1.032761 | 0.581956 | 0.568941 |
| 3D only | 局部 3D 结构 | 0.677757 | 1.407632 | 1.089493 | 0.527145 | 0.509384 |
| 去 ESM（无 3D） | MSA + MoLFormer | 0.697095 | 1.350297 | 1.045314 | 0.576719 | 0.562400 |
| 去 MSA（无 3D） | ESM + MoLFormer | 0.700569 | 1.331696 | 1.028194 | 0.585401 | 0.571385 |
| 序列线性层（无 3D） | 原始序列 one-hot 线性层 + MoLFormer | 0.690377 | 1.388140 | 1.072265 | 0.559609 | 0.545530 |
| 序列 Embedding（无 3D） | 原始序列 `nn.Embedding` + MoLFormer | 0.695396 | 1.362192 | 1.053279 | 0.573527 | 0.559115 |
| 去 MoLFormer（无 3D） | ESM + MSA | 0.631110 | 1.534532 | 1.193417 | 0.374858 | 0.384096 |

## Test

| 模型 | 输入模态 | CI | RMSE | MAE | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 完整模型（仅前融合） | ESM + MSA + 3D + MoLFormer K/V | **0.691291** | **1.442721** | **1.142213** | **0.569881** | **0.545449** |
| 1D only | ESM + MSA + MoLFormer | 0.678617 | 1.496409 | 1.177822 | 0.537883 | 0.510595 |
| 3D only | 局部 3D 结构 | 0.673200 | 1.479384 | 1.161937 | 0.536123 | 0.495962 |
| 去 ESM（无 3D） | MSA + MoLFormer | 0.673513 | 1.508216 | 1.191373 | 0.525388 | 0.498823 |
| 去 MSA（无 3D） | ESM + MoLFormer | 0.686156 | 1.465870 | 1.154004 | 0.558282 | 0.530414 |
| 序列线性层（无 3D） | 原始序列 one-hot 线性层 + MoLFormer | 0.674580 | 1.497473 | 1.179179 | 0.531439 | 0.500088 |
| 序列 Embedding（无 3D） | 原始序列 `nn.Embedding` + MoLFormer | 0.677183 | 1.486793 | 1.172186 | 0.539721 | 0.507620 |
| 去 MoLFormer（无 3D） | ESM + MSA | 0.581518 | 1.747724 | 1.368110 | 0.226600 | 0.242685 |

序列线性层模型的 test CI 为 0.674580；`nn.Embedding` 版本提高到 0.677183，RMSE 从 1.497473 降至 1.486793，但仍低于 1D only（0.678617）、去 MSA（0.686156）及完整前融合模型（0.691291）。Embedding 版本仍采用 mean pooling，因此它与线性层版本一样不保留残基顺序；它仍使用预训练 MoLFormer 配体特征，所以不能据此判断预训练模型整体是否有用，也不能单独估计 ESM/MSA 预训练表示的贡献。

经典序列输入 DTA 方法 DeepDTA 使用类别索引和可学习 Embedding 后接多层 1D-CNN，以学习局部序列模式；DGDTA 使用可学习 Embedding、Bi-LSTM 和 CNN 建模上下文。仅将当前 one-hot 替换为 `nn.Embedding`、仍做均值池化，也不会保留顺序。要回答“去掉 ESM/MSA 预训练表示后，原始蛋白序列能否提供增益”，应另设随机初始化的可学习 residue embedding 加顺序感知编码器作为基线，并与预训练表示模型共享数据划分、配体输入、预测头和训练/选择协议。参考：[DeepDTA](https://doi.org/10.1093/bioinformatics/bty593)、[DGDTA](https://doi.org/10.1186/s12859-023-05497-5)。

## 结果文件

- 五个消融的 validation：`.codex_runtime/modality-ablation-lp-v1-gpu7/{arm}/summary.json`。
- 五个消融的 test：`.codex_runtime/modality-ablation-lp-v1-test-gpu7/comparison.json` 和 `comparison.csv`。
- 新增序列线性层消融的 validation：`.codex_runtime/modality-ablation-lp-v2-sequence-linear-gpu7/no3d_sequence_linear/summary.json`。
- 新增序列线性层消融的 test：`.codex_runtime/modality-ablation-lp-v2-sequence-linear-test-gpu7/comparison.json` 和 `comparison.csv`。
- 新增序列 Embedding 消融的 validation：`.codex_runtime/modality-ablation-lp-v3-sequence-embedding-gpu7/no3d_sequence_embedding/summary.json`。
- 新增序列 Embedding 消融的 test：`.codex_runtime/modality-ablation-lp-v3-sequence-embedding-test-gpu7/comparison.json` 和 `comparison.csv`。
- 完整模型（仅前融合）的 validation：`.codex_runtime/molformer-early-kv-gpu7-20261112_222241-63318/summary.json`。
- 完整模型（仅前融合）的 test：`.codex_runtime/early-kv-full-model-test-gpu7-20261117/summary.json`。

七个消融的 test prediction keys 均与官方 manifest 顺序一致，key SHA-256 为 `f3ee30f57dfb0d2198040e64a129235dce93b4880a5dc637bbd708c30c37a6b5`。新增两个模型 test 均覆盖 4,751/4,751 条，且 `test_used_for_selection=false`；完整前融合模型 test 覆盖 4,751/4,751 条，且 `test_used_for_selection=false`、`labels_passed_to_model=false`。完整模型表项仅列前融合版本。
