# LP-PDBBind 实验总记录

更新时间：2026-11-08（含KERMT正式完训及完整validation八排列诊断完成；10:43历史快照保留）
数据说明：`docs/LP_PDBBind_analysis_processing_manual.md`
原始标签表：`data/LP_PDBBind.csv`

## 1. 记录范围与使用规则

本文件是项目实验结果账本。原有结果使用 LP-PDBBind 官方 `new_split`；从 2026-11-04 起，另分区收录用户批准的“合并现有 LP 可用池、新建独立随机 8:1:1 划分”实验。两类划分与 validation／test 分开记账。每个条目给出做法、结果、状态及原始产物定位。

LP-PDBBind 官方原始划分为 train 11,513、val 2,422、test 4,860。不同模型因结构缺失、分子解析失败、序列长度限制或其官方数据处理约束而有不同的有效样本数；跨方法比较 CI 时必须同时检查 `n`，不能把不同有效样本集合视为严格一一可比。

本记录仍不收录其他旧 PDBbind 随机划分、`fixed_seed42_filtered` 或无关数据集的结果；本次新增的 LP 可用池随机协议并不属于这些旧实验，不在旧清理范围。

最新补录见 [近期实验总览](LP_PDBBind_recent_experiments_20261108.md) 和 5.79–5.87；[39份原始聚合产物／history／失败记录及哈希](../experiments/lp_pdbbind/recent_record_20261108/README.md)已归档。5.55–5.78.5 的既有分支记录同步回原工作区，保留当时的准备状态；最新运行状态以下述补录为准。此前未统一汇总，不等于这些实验未执行。

## 2. 当前结论摘要

下表为原 LP 历史结果，不含随机划分。近期随机完整版 validation CI `0.794675`（n=1,813）、固定 checkpoint test CI `0.785159`（n=1,813），见 5.80／5.82；两者不是同一集合。原 LP 的32行4层交互训练被中断；KERMT随后已完训，validation CI `0.706107`、五项均略差于四层基线，见5.86.1。

| 方法/版本 | 状态 | 选择依据 | Val CI | Test CI | Test RMSE | Test Pearson | Test Spearman | Test n | 最佳权重 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MSA Interface Pair Latent Stability Fusion v20 | 已完成 | validation raw-PCA stability grid | **0.737938** | **0.736092** | **1.297841** | **0.674478** | **0.650757** | 4,751 | `Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20/msa_interface_pair_latent_stability_fusion_v20_full_20260902_140000/model.joblib` |
| MSA Interface Pair Latent Fusion v19 | 已完成 | validation pair-only grouped-OOF residual | **0.736645** | **0.734503** | **1.300913** | **0.671592** | **0.647100** | 4,751 | `Model/LP_PDBBind_MSAInterfacePairLatentFusion_v19/msa_interface_pair_latent_fusion_v19_full_20260902_114500/model.joblib` |
| MSA Spatial Stability Fusion v16 | 已完成 | validation expanded grouped-OOF stability residual | **0.735954** | 0.731367 | 1.309480 | 0.665183 | 0.640083 | 4,751 | `Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824/model.joblib` |
| MSA Spatial Interface Feature Fusion v15 | 已完成 | validation grouped-OOF spatial interface residual | **0.735177** | **0.730115** | **1.309242** | **0.664159** | **0.637505** | 4,751 | `Model/LP_PDBBind_MSAInterfaceSpatialFusion_v15/msa_interface_spatial_fusion_v15_full_20260829_160048/model.joblib` |
| MSA Interface Feature Fusion v14 | 已完成 | validation grouped-OOF interface residual | **0.734170** | **0.726950** | **1.315046** | **0.658056** | **0.631365** | 4,751 | `Model/LP_PDBBind_MSAInterfaceFusion_v14/msa_interface_feature_fusion_v14_full_20260829_145451/model.joblib` |
| Pretrained MSA Pair Fusion v12 | 已完成 | validation blend + grouped OOF residual | **0.732831** | **0.724087** | **1.321893** | **0.652407** | **0.624838** | 4,751 | `Model/LP_PDBBind_MSAPairFusion_v12/msa_pair_pretrained_v12_full_20260826_144811/model.joblib` |
| MSA Pair Fusion v11 | 已完成 | validation blend + grouped OOF residual | **0.731556** | **0.722219** | **1.336469** | **0.648779** | **0.620379** | 4,751 | `Model/LP_PDBBind_MSAPairFusion_v11/msa_pair_v11_full_20260825_165726/model.joblib` |
| MSA Rank Fusion v10 | 已完成 | grouped OOF CI | **0.730892** | **0.722141** | **1.346158** | **0.648728** | **0.620507** | 4,751 | `Model/LP_PDBBind_MSARankFusion_v10/run/model.joblib` |
| Repeated Rank Fusion v8 | 已完成 | repeated grouped OOF CI | **0.728305** | **0.718266** | **1.374328** | **0.637626** | **0.611355** | 4,751 | `Model/LP_PDBBind_RepeatedRankFusion_v8/run/models/` |
| Rank-Unlocked Fusion v7 | 已完成 | grouped OOF CI | **0.723261** | **0.716345** | **1.381976** | **0.629327** | **0.605653** | 4,751 | `Model/LP_PDBBind_RankUnlockedFusion_v7/run/models/seed_7/` |
| Anchored Expert Fusion v5 | 已完成 | sealed validation CI | 0.713374 | 0.708700 | 1.511411 | 0.599770 | 0.586793 | 4,751 | `Model/LP_PDBBind_AnchoredExpertFusion_v5/validation/validation_package` |
| Boltz pairwise + rank loss + pair delta | 已完成 | max val CI | 0.695390 | **0.706858** | 1.448719 | 0.611562 | 0.580972 | 4,758 | `external_benchmarks/results/boltz_pairwise_rank010_pairDelta_save/model.pt` |
| Boltz raw ensemble | 已完成 | val ensemble CI | **0.698789** | 0.705772 | 1.454434 | 0.610139 | 0.579355 | 4,758 | 三个成员权重，见第 4 节 |
| Hybrid Upper-Bound v2 cached seed87 | 已完成 | max val CI | 0.700478 | 0.686741 | 1.497357 | 0.558346 | 0.534113 | 4,751 | `Model/LP_PDBBind_HybridUpperBound_v2/cached_seed87/Model_cached_best.pth` |
| MSA Pocket Expert v9 | 已完成 | max val CI | 0.690276 | 0.682576 | 1.468233 | 0.549479 | 0.523271 | 4,751 | `Model/LP_PDBBind_MSAPocketExpert_v9/msa_pocket_v9_full_retry_20260824_154027/best.pt` |
| GerNA ligand-centered pocket ESM 8 A | 已停止 | max val CI | 0.687689 | 0.682945 | 1.490508 | 0.550505 | 0.523357 | 4,751 | Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/Model_dta_best.pth |
| Interface Direct v1 seed42 | 已完成 | max val CI | 0.665540 | 0.633671 | 1.627943 | 0.405155 | 0.394480 | 4,751 | `Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/Model_dta_best.pth` |
| 3DProtDTA 官方架构 | 已完成 | max val CI | 0.693622 | 0.696331 | 1.707948 | 0.436531 | 0.413826 | 4,775 | `external_benchmarks/results/3dprot_dta_lp_official_ckpt/best_model.pt` |
| TankBind 官方架构，从零训练 | 训练中 | max val CI | 0.687428 | 0.680960 | 1.673226 | 0.544692 | 0.517133 | 4,775 | `external_benchmarks/results/tankbind_lp_native_from_scratch/best_model.pt` |
| CASTER-DTA 官方架构，从零训练 | 训练中 | max val CI | 0.659453 | 0.651060 | 1.575569 | 0.461938 | 0.441575 | 4,732 | `external_benchmarks/results/caster_dta_lp_official_from_scratch/best_model.pt` |
| GraphDTA GIN 官方架构 | 已记录 | max val CI | 0.649940 | 0.639861 | 1.654261 | 0.365233 | 0.410469 | 4,838 | `external_benchmarks/results/graphdta_lp_gin_noearlystop/best_model.pt` |
| DeepDTAGen 官方架构，去除超长序列 | 已完成 | max val CI | 0.618592 | 0.634574 | 1.730362 | 0.408597 | 0.392472 | 4,716 | `external_benchmarks/results/deepdtagen_lp_official_dropLong138/best_model.pt` |
| AttentionMGT-DTA 官方架构 | 训练中 | max val CI | 0.610910 | 0.604407 | 1.720493 | 0.288959 | 0.310719 | 4,741 | `external_benchmarks/results/attentionmgt_dta_lp_official/best_model.pt` |

当前同一套 4,751 条 joined cache 上按主目标 CI 的最佳可复现结果为 MSA Interface Pair Latent Stability Fusion v20：Validation CI `0.737938`；Test CI `0.736092`、RMSE `1.297841`、MAE `1.002948`、Pearson `0.674478`、Spearman `0.650757`。V20 复用 V19 已封存的 contact-aware pair latent，以 V16 prediction 为 residual 建模底座、只以 V19 OOF prediction 为 validation promotion benchmark；raw PCA-64、ExtraTrees minimum leaf `2`、max features `1.0` 和 residual budget `0.375` 全部由 validation 冻结。相对 V19，Test CI、RMSE、MAE、Pearson、Spearman 五项全部改善。当前最佳单 expert/单模型仍为 Boltz pairwise + rank loss + pair delta 的 Test CI `0.706858`，其有效 test 样本为 4,758 条，不能与 4,751 条融合结果视为完全相同样本集合。

GerNA 已有一组使用 LP-PDBBind 官方 new_split 的 ligand-centered pocket ESM 结果，Test CI 为 0.682945。该版本低于当前最佳 Boltz 单模型，作为 GerNA 结构优化的有效对照保留。

## 3. 官方/外部模型从零训练

### 3.1 3DProtDTA 官方架构

做了什么：使用官方 3DProtDTA 架构在 LP-PDBBind prepared split 从零训练 700 epoch，不早停；按 validation CI 保存最佳 checkpoint。最佳 checkpoint 为 epoch 90。

结果怎么样：Val CI `0.693622`，Val MSE `2.520049`，Val MAE `1.245632`，Val Pearson `0.449627`。使用该 checkpoint 单独评估 test：Test CI `0.696331`，RMSE `1.707948`，MAE `1.328884`，Pearson `0.436531`，Spearman `0.413826`，有效样本 `4,775`。

结果分析：后期训练明显过拟合；epoch 700 的 train MSE `0.175`、Pearson `0.976`，但未超过 epoch 90 的 validation CI。最佳 checkpoint 在 test 上保持接近的 CI，但误差和相关系数一般。权重及指标：`external_benchmarks/results/3dprot_dta_lp_official_ckpt/`。

### 3.2 GraphDTA GIN 官方架构

做了什么：以 LP-PDBBind 为数据入口运行 GraphDTA 的官方 GIN 架构，训练不使用早停，并按 validation CI 保存最优权重。

结果怎么样：最佳为 epoch 2，Val CI `0.649940`；Test CI `0.639861`，RMSE `1.654261`，MAE `1.303345`，Pearson `0.365233`，Spearman `0.410469`，有效样本 `4,838`。

结果分析：训练早期即达到最高 validation CI，之后没有得到更好的排序泛化。该结果可作为官方 2D 图模型的 LP-PDBBind 参考下限，但不能与使用不同有效样本数的方法做严格显著性比较。权重及指标：`external_benchmarks/results/graphdta_lp_gin_noearlystop/`。

### 3.3 DeepDTAGen 官方架构

做了什么：运行官方 DeepDTAGen 架构。由于模型的位置编码最大长度为 138，对超过该限制的蛋白样本在建数据集阶段剔除；其余设置保持官方结构与训练流程，按 validation CI 保存最佳权重。

结果怎么样：最佳为 epoch 30，Val CI `0.618592`。Test CI `0.634574`，RMSE `1.730362`，MAE `1.370849`，Pearson `0.408597`，Spearman `0.392472`，有效样本 `4,716`。

结果分析：超长序列过滤使有效 test 样本少于其他方法，因此本结果只能作为该模型在可处理子集上的参考。CI、相关系数均低于前述 3DProtDTA 与 Boltz 版本。权重及指标：`external_benchmarks/results/deepdtagen_lp_official_dropLong138/`。

### 3.4 TankBind 官方架构，从零训练

做了什么：在 LP-PDBBind 上从零训练 TankBind 官方架构，使用 native cache、20 A pocket radius、8 A contact cutoff、relative-k `0.01`，不早停，按 validation CI 保存最佳模型。

结果怎么样：当前最佳为 epoch 33，Val CI `0.687428`；对应 Test CI `0.680960`，RMSE `1.673226`，MAE `1.326387`，Pearson `0.544692`，Spearman `0.517133`，有效样本 `4,775`。

结果分析：当前日志已到 epoch 58，epoch 56--57 的 Val CI 为 `0.677047` 和 `0.676880`，低于最佳值，显示继续训练未改善排序能力。训练尚未正式结束，清理前应保留其目录、最佳权重和日志。结果目录：`external_benchmarks/results/tankbind_lp_native_from_scratch/`。

### 3.5 AttentionMGT-DTA 官方架构

做了什么：运行 AttentionMGT-DTA 官方架构，LP-PDBBind cache 为 `external_benchmarks/AttentionMGT-DTA/data/lp_pdbbind`，不早停，按 validation CI 保存最佳模型。

结果怎么样：当前最佳为 epoch 1，Val CI `0.610910`；对应 Test CI `0.604407`，RMSE `1.720493`，MAE `1.342944`，Pearson `0.288959`，Spearman `0.310719`，有效样本 `4,741`。

结果分析：当前日志已到 epoch 315，最近 epoch 313--314 的 Val CI 在 `0.580483`--`0.582769`，低于 epoch 1 最优值，存在明显过拟合或训练目标与本数据不匹配。训练尚未正式结束；只保留当前 best checkpoint 作为后续可复现对照。结果目录：`external_benchmarks/results/attentionmgt_dta_lp_official/`。

### 3.6 CASTER-DTA 官方架构，从零训练

做了什么：使用 CASTER-DTA 官方架构，输入完整 protein 结构，在 LP-PDBBind 上从零训练，按 validation CI 保存最佳模型，不早停。

结果怎么样：当前最佳为 epoch 20，Val CI `0.659453`；对应 Test CI `0.651060`，RMSE `1.575569`，MAE `1.245842`，Pearson `0.461938`，Spearman `0.441575`，有效样本 `4,732`。

结果分析：当前日志已经运行到 epoch 494，`bad_epochs` 达到 `473`；epoch 493 的 Val CI `0.637271`、Test CI `0.627302`，持续低于最佳结果，说明长时间继续训练没有收益。训练尚未正式结束，因此清理前必须保留结果目录和 best checkpoint；训练结束后可删除中间 checkpoint（若存在）。结果目录：`external_benchmarks/results/caster_dta_lp_official_from_scratch/`。

### 3.7 早期官方预训练权重推理（不进入最终比较）

做了什么：在 LP-PDBBind val/test 上直接推理官方提供的 TankBind 与 CASTER-DTA 预训练权重，没有使用 LP-PDBBind 重新训练。

结果怎么样：TankBind 的 Val/Test CI 分别为 `0.734858`/`0.741895`（test n=`4,775`）；CASTER-DTA 的 Val/Test CI 分别为 `0.611088`/`0.637223`（test n=`4,092`）。原始汇总位于 `external_benchmarks/results/lp_external_pretrained_val_test_summary.md`。

结果分析：这些权重可能在与 LP-PDBBind 存在来源重叠的 PDBbind 数据上预训练，不能与本文件中从零训练的模型或后续 GerNA LP-PDBBind 训练做公平比较。因此只保留其结果摘要作为排查和历史对照，不保留为最终 benchmark 结论，也不需要保留对应的大型预训练缓存或推理临时文件。

## 4. Boltz pairwise 实验与集成

### 4.1 单模型消融

做了什么：基于 Boltz pairwise 模型，在同一 LP-PDBBind 数据索引上比较 CI 选择、rank loss 和 pair-delta。三组实验均使用 3 blocks、dim/hidden `128`、pair dim `64`、batch size `64`，按 validation CI 保存 `model.pt`。

| 版本 | 关键差异 | Val CI | Test CI | Test RMSE | Test Pearson | Test Spearman | 权重 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ciSelect_noDelta` | 无 rank loss，无 pair delta | 0.694791 | 0.696812 | 1.512275 | 0.589105 | 0.558247 | `external_benchmarks/results/boltz_pairwise_ciSelect_noDelta_save/model.pt` |
| `rank010_noDelta` | rank loss `0.1`，无 pair delta | 0.694643 | 0.696696 | 1.520812 | 0.587443 | 0.557824 | `external_benchmarks/results/boltz_pairwise_rank010_noDelta_save/model.pt` |
| `rank010_pairDelta` | rank loss `0.1`，启用 pair delta | **0.695390** | **0.706858** | **1.448719** | **0.611562** | **0.580972** | `external_benchmarks/results/boltz_pairwise_rank010_pairDelta_save/model.pt` |

结果分析：仅加入 rank loss 而不使用 pair delta 没有提升；rank loss 与 pair delta 同时使用后，test CI 相比无 rank loss/no-delta 基线提高约 `0.0100`，同时 RMSE 和相关系数也改善。因此保留 `rank010_pairDelta` 作为当前 Boltz 最佳单模型。

### 4.2 Ensemble

做了什么：对上述三个 Boltz 成员的已保存预测进行网格权重搜索。raw ensemble 最终权重为 `[0.3, 0.2, 0.5]`，对应成员顺序为 `ciSelect_noDelta`、`rank010_noDelta`、`rank010_pairDelta`。

结果怎么样：raw ensemble 的 Val CI `0.698789`，Test CI `0.705772`，Test RMSE `1.454434`，Pearson `0.610139`，Spearman `0.579355`，有效样本 `4,758`。rank ensemble 的 Val CI `0.699248`，Test CI `0.706205`，但 rank-only 输出不保留亲和力数值尺度，RMSE/MAE 不可用于与回归模型比较。

结果分析：ensemble 在 validation CI 略高，但 raw ensemble 的 test CI 低于最佳单模型 `0.706858`。因此最终保留最佳单模型作为 test 排序最优版本，同时保留 ensemble 指标 JSON 用于完整复现。结果目录：`external_benchmarks/results/boltz_ensemble_pairwise_raw/`、`external_benchmarks/results/boltz_ensemble_pairwise_rank/`、`external_benchmarks/results/boltz_ensemble_pairwise_standardized/`。

## 5. GerNA 状态

### 5.1 Ligand-centered pocket ESM 8 A

做了什么：在 LP-PDBBind 官方 new_split 上训练 GerNA。蛋白 3D 分支由完整链改为 ligand atom 周围 8 A 内的 protein CA 残基，并使用对应的 640 维 ESM embedding；完整链 ESM 1D 分支与 ligand 2D 图分支仍保留。训练使用 FP32、无 AMP、无 rank loss、按 validation CI 保存最佳 checkpoint，未启用早停。

结果怎么样：训练在 epoch 17 的 batch 3000 后中断，最佳 checkpoint 为 epoch 15。Val CI 0.687689，RMSE 1.392772，Pearson 0.550881，Spearman 0.538180。对应 Test CI 0.682945，RMSE 1.490508，MAE 1.170277，Pearson 0.550505，Spearman 0.523357，有效样本 4,751。权重与预测：Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/。

结果分析：配体中心口袋化没有带来所需的 CI 提升，当前 test CI 低于最优 Boltz 单模型 0.706858。epoch 15 在连续五轮无 CI 改善后出现小幅回升，但 epoch 16 又下降，且中断发生在首次学习率衰减前；从绝对指标看，不应恢复这条训练。保留 epoch 15 最佳权重，作为后续引入更强预训练交互表征时的 GerNA 对照。

## 5.2 GerNA-Boltz frozen interface fusion (prepared)

做了什么：在 GerNA 原生 1D ESM、ligand 2D 图和 ligand-centered 8 A protein 3D 分支之外，新增独立的 Boltz interface 分支。该分支直接读取 Boltz 预训练后冻结的 384 维 pocket residue embedding 和 ligand atom embedding，使用 Boltz 原生 10 A heavy-atom pocket 规则（最少 16、最多 64 个残基），在 GerNA 外部完成 local pair attention/mean-delta pooling 与 global feature 投影；得到的可训练 latent 在最终 affinity head 前与 GerNA latent 拼接。两条分支不做原子顺序对齐，避免 GerNA 与 Boltz 配体预处理差异造成静默错配。

结果怎么样：尚未训练。已完成预检：原始 LP-PDBBind manifests 共 18,128 条，Boltz merged LMDB 缺 103 条（train 66、valid 11、test 26）。因此创建了独立交集 split data/PDBbind_chain_esm_residue_DTA/lp_pdbbind_boltz_interface/，其 train/valid/test 样本数分别为 10,971/2,329/4,725，并经 coverage checker 复核为 0 missing。实际样本读取得到 Boltz pocket [64,384]、ligand [33,384]、global [768]；collate 后维度正确。FP32 CPU 单样本端到端前向输出 [1,1] 且为有限值。

结果分析：Boltz 最佳单模型的优势来自冻结预训练交互表征和显式 pocket-ligand pair 建模，而不是其表面上的小参数量。本融合保留 GerNA 的原生结构分支，同时独立引入该局部交互信号，属于高信息增益的结构性尝试。训练与评估必须固定使用上述交集 split；若要同原始 LP-PDBBind 基线严格比较，应在同一交集上重新评估基线。相关实现：net/boltz_interface.py、scripts/check_boltz_interface_coverage.py。

### 5.3 Interface-first GerNA v1（实现完成，正式运行待开始）

做了什么：按冻结的 LP-PDBBind-only 方案新增独立 `interface_first` 架构。局部路径使用 ligand 2D atom states、protein all-atom chemical states、gated residue ESM-to-atom adapter、四层 typed scalar-vector interface blocks、ligand-centric interface pooling 和 conditioned global readout；Stage 1 仅用 official train 的对称性修正 hard-pose decoy，Stage 2 仅用 SmoothL1，前三轮 warm-up 后全量联合微调。训练固定为 FP32、无 AMP、无 affinity rank loss、无早停，validation CI 只负责 checkpoint 选择，test 只在加载最佳权重后评估一次。相关代码、缓存构建、train-only smoke、五 seed 训练和 clustered-bootstrap 汇总入口位于 `experiments/lp_pdbbind/interface_first_v1/`。

结果怎么样：实现级 CPU 单元与集成回归已经通过，真实 LP 样本 CPU forward、Stage-1/Stage-2 checkpoint、精确断点恢复、keyed prediction 和 complex/exact-target/related-target bootstrap 契约均已验证。GPU train-only diagnostic smoke 尚未运行，正式五 seed 训练尚未开始，因此尚未产生 validation/test CI，也没有可报告的 RMSE、MAE、Pearson 或 Spearman。

结果分析：当前只能确认实现和数据隔离协议闭合，不能据此声称模型效果提升。下一步必须先在指定 GPU 上运行 `smoke_train_only.sh` 并通过 shortcut diagnostic gate；只有该 gate 通过后，才允许启动 seeds 42--46 的 Stage 1 和 Stage 2。最终结论必须与同样 seeds、相同 ordered test keys 的 GerNA baseline 做 paired comparison，并同时报告 complex-level、exact-target 和 90% identity target-cluster 的 Delta CI 置信区间。

#### 5.3.1 GPU 7 train-only smoke 首次运行

做了什么：完成 Interface-first v1 的 `native_train`、`native_valid`、`native_test` 和 `decoys_train` 缓存后，在物理 GPU 7 上运行 seed 42、单 epoch、只接触 official train 的 diagnostic smoke。输出目录为 `Model/LP_PDBBind_InterfaceFirstGerNA_v1_smoke_seed42/`，日志为 `experiments/lp_pdbbind/interface_first_v1/logs/smoke_train_only_seed42_20260727_071234.log`。

结果怎么样：训练正常运行到 batch 100，累计 mean loss 从 batch 25 的 `1.307748` 降至 batch 100 的 `0.865307`，随后因一个 native pose 被 vdW 硬阈值判定为 clash、64 次 jitter 均不合法而中断。该次运行没有完成 epoch、没有生成 checkpoint 或 shortcut diagnostic gate 报告，因此不能报告任何 validation/test 指标，也不能据此启动正式五 seed 训练。

结果分析：在 3,799 个具有完整 decoy set 的 Stage-1 样本中，有 212 个 native pose（5.58%）触发相同 vdW 阈值。直接删除这些真实结构会进一步扩大 Stage-1 选择偏差，因此保留其 native-vs-decoy pose loss；只有在所有 jitter 尝试均失败时才使用原生坐标作为可审计 identity fallback，使该样本不引入伪造几何扰动，并新增 `jitter_native_clash_samples` 与 `jitter_native_clash_identity_fallbacks` 计数。decoy 的 vdW 硬拒绝规则保持不变。修复后 19 个 Stage-1 回归测试通过，smoke 待在 GPU 7 上从头重跑。


### Rotation-dominant matched decoy v2 implementation

**做了什么**

2026-07-29 完成并冻结了 LP-PDBBind Interface-first GerNA 的 rotation-dominant matched decoy v2 实现。v2 将随机物理 decoy 与 matched decoy 存入独立 bank；生成、数据诊断、cache provenance、Stage-1 bank 选择和 Stage-1 completion artifact 分别由提交 6ac0e3e、5eb7950、33607c4、a0377b8、2f6e879、ce2d3f3 完成。随后提交 4e406e3 使 Stage 2 gate 能正确消费 v2 的嵌套 checkpoint/model-diagnostic completion marker，提交 032508d 固定了 diagnostic cache、train cache、paired smoke、matched-only Stage 1、Stage 2、五 seed 和 bootstrap summary 脚本，并增加 paired-smoke comparator。

Task 8 已执行以下实现验证：完整 focused unittest suite（test_local_interface_cache、test_local_interface_decoys、test_local_interface_decoys_v2、test_interface_decoy_audit、test_interface_pretraining、test_interface_evaluation）退出码为 0；Python compile、v2 shell syntax、pretrain_local_interface.py --help 和 compare_interface_v2_smokes.py --help 均通过。此前 Task 6 的 test_interface_pretraining 与 test_interface_decoy_audit 回归共 61 项通过。paired-smoke comparator 额外覆盖了成功写出报告和拒绝不同 initial-state SHA256 两种情况。

**结果怎么样**

执行了 20 个固定 train-internal diagnostic key 的 CPU cache smoke，路径为 /tmp/interface_v2_plan_smoke/decoys。cache 完整结束，requested=20，written/included=2，excluded=18，invocation_counts 为 processed=20, written=2, errors=18。随后 evaluator 按设计拒绝继续，报错为 paired diagnostic requires at least five target groups，因此没有产生可用于 scientific data gate 的 decoy_v2_data_diagnostic.json。这是预期的最小样本保护，不是 validation/test 结果。

完整 diagnostic cache、full data gate、paired GPU smoke 和正式 matched-only Stage 1/Stage 2 仍未运行。v2 实现期间没有查看 official validation 或 test 性能，也没有启动 GPU 训练。

**结果分析**

20-key smoke 的 paired-complete 保留率只有 10%，且绝大部分排除发生在 medium/high stratum 的 hard-vdW-clash 或不足四个合法 candidate。这表明当前冻结 v2 规则的可接受 pose 覆盖率可能是后续 data gate 的主要风险。不能根据这个小样本临时放宽阈值或调整生成器；下一步应先用 build_diagnostic_cache.sh 生成完整 1,104-key diagnostic cache，并让冻结 gate 给出真实的 matched coarse shortcut、paired reduction 和 complete-acceptance 结论。若 full gate 不通过，流程必须停止在 GPU 训练之前并记录失败原因。


## 5.4 2026-07-30 至 2026-08-01：decoy feasibility 路线结果

做了什么：依次运行 rotation-dominant v2 完整 diagnostic gate、physics-guided v3 generator smoke、v4 high-stratum translation feasibility probe 和 v4b physics-guided rotation probe。这些实验只使用 official train 内部固定 cohort，没有用 validation/test 进行生成器选择。

结果怎么样：

- v2 完整 diagnostic 共请求 1,104 条，matched complete `10/1104`，random-physical complete `58/1104`；matched coarse nonlinear CV AUROC `0.408333`，random-minus-matched `0.201667`，但 matched requested fraction 只有 `0.009058`，gate 因 `matched_requested_fraction_too_low` 失败。
- v3 generator smoke 请求 64 条，写入 15，错误 49；random-physical complete `15/64`，matched complete `3/64`。
- v4 translation probe 的 56 个 eligible 样本中，physical complete 21，matched complete 10，`expanded_no_vdw_safe_fraction=0.924909`，未达到 matched complete 至少 20 且 no-vdW-safe fraction 不高于 0.90 的条件。
- v4b rotation probe 的 56 个 eligible 样本中，physical complete 18，matched complete 8，`no_vdw_safe_family_fraction=0.955741`，仍未达 gate。

结果分析：增加 translation 方向、距离网格和 physics-guided rotation 均没有解决大部分 rotation family 的 vdW feasibility 问题，而且完成样本对小 ligand 有明显选择偏差。该 decoy 路线已停在 probe 阶段，没有进入正式 affinity 训练。报告路径分别为 `data/LP_PDBBind_InterfaceFirstGerNA_v2/diagnostic/decoy_v2_data_diagnostic.json`、`data/LP_PDBBind_InterfaceFirstGerNA_v3/decoys_generator_smoke_v3/decoy_report.json`、`data/LP_PDBBind_InterfaceFirstGerNA_v4/translation_probe_high_seed42/translation_probe_report.json` 和 `data/LP_PDBBind_InterfaceFirstGerNA_v4b/rotation_probe_high_seed42/rotation_probe_report.json`。

## 5.5 Interface Direct v1 seed42

做了什么：放弃未通过 feasibility gate 的 decoy pretraining，使用 official train 上的 native complex 从零训练 Interface-first direct affinity model，按 validation CI 保存最佳 checkpoint，test 不参与选择。

结果怎么样：最佳为 epoch 20，Val CI `0.665540`、RMSE `1.467229`、Pearson `0.476204`、Spearman `0.479304`。对应全量 official test `n=4751`：CI `0.633671`、RMSE `1.627943`、MAE `1.288285`、Pearson `0.405155`、Spearman `0.394480`。

结果分析：该局部界面架构单独做 affinity 回归的上限明显不足，但 checkpoint 可作为后续 hybrid 的 Interface expert 初始化。结果目录：`Model/LP_PDBBind_InterfaceFirstDirect_v1_seed42/`。

## 5.6 Hybrid Performance v1

做了什么：将 GerNA 与 Interface expert 进行端到端 hybrid fusion，比较 MLP fusion 和 rank weight `0.1/0.2`，按 validation CI 选择。

结果怎么样：MLP rank `0.1` 最佳 Val CI `0.676095`，MLP rank `0.2` 最佳 Val CI `0.677824`。该路线没有生成可核对的正式 test artifact。

结果分析：端到端重复提取两个大 expert 的特征，训练速度慢；且 validation CI 没有超过原 GerNA 和当前外部基线，因此转向可预处理 feature cache 的 Upper-Bound v2。结果目录：`Model/LP_PDBBind_HybridPerformance_v1/dev_seed42/`。

## 5.7 Hybrid Upper-Bound v2

做了什么：预计算 GerNA 与 Interface multiscale features，训练 Perceiver/MoE cached fusion；使用 seeds `1,7,21,42,87` 搜索 cached model，再对最佳 seeds 运行 selective E2E。所有选择只使用 validation。

结果怎么样：cached seeds 的 Val CI 为 `0.696067/0.698425/0.698280/0.698953/0.700478`，其中 seed87 最佳。E2E EMA 的最好 Val CI 为 seed42 的 `0.694099`，seed7 为 `0.693623`，seed87 为 `0.694019`，均低于各自 cached teacher。2026-08-12 使用 v3 test-only evaluator 首次计算 cached seed87 的全量 official test `n=4751`：CI `0.686741`、RMSE `1.497357`、MAE `1.199309`、Pearson `0.558346`、Spearman `0.534113`。

结果分析：cached seed87 是当前 GerNA/Interface/Hybrid 自研路线中最好的 official test 结果，比 GerNA-only Test CI `0.682945` 高 `0.003796`，但仍低于 3DProtDTA `0.696331` 和 Boltz pairwise `0.706858`。validation 到 test 的 CI 下降为 `0.013737`，说明 validation 改善没有完全转化到 test。权重：`Model/LP_PDBBind_HybridUpperBound_v2/cached_seed87/Model_cached_best.pth`；正式 test artifact：`Model/LP_PDBBind_HybridResidual_v3/baseline_test/test_metrics_cached_teacher.json`。

## 5.8 Anchored Residual v3

做了什么：新建隔离的 `dta/anchored_residual_v3/` 和 `experiments/lp_pdbbind/performance_fusion_v3/`，固定 `prediction = cached_seed87_teacher + residual_delta`，residual output 零初始化，依次训练 residual head、fusion 和 selective top-layer E2E；只有 validation 改善才接受新状态，否则回滚到 cached teacher。

结果怎么样：2026-08-12 的 GPU 7 smoke 在 train/valid/test 各 16 条上完整走通。16 条 validation teacher CI 为 `0.700000`，三阶段训练未超过 teacher，因此 `selected_variant=cached_teacher`。16 条 test CI `0.361345` 只是固定前缀小切片的功能性结果，不进入模型排名。随后 full run 完整结束：residual 第 1 轮 Val CI `0.696589`（RMSE 改善至 `1.326461`），fusion 最好 Val CI `0.690601`，selective E2E 最好 Val CI `0.690731`，全部低于 teacher Val CI `0.700478`。最终 `improved_over_teacher=false`、`selected_variant=cached_teacher`，正式 test 回退结果与 teacher 相同：CI `0.686741`、RMSE `1.497357`、MAE `1.199309`、Pearson `0.558346`、Spearman `0.534113`，`n=4751`。

结果分析：v3 的 rollback 保护按设计生效，没有覆盖 v2 好结果。但其训练目标优先改善数值误差：residual 第 1 轮 RMSE 比 teacher 低 `0.044050`，CI 却下降 `0.003889`；后续解冻 fusion/E2E 后排序继续退化。因此不再延长 v3 epoch 或微调同一 loss，下一版应将排序改善作为主目标，并继续以 cached seed87 为可回退基线。完整 artifact：`Model/LP_PDBBind_HybridResidual_v3/run/summary.json`。

## 5.9 Multi-Teacher Rank Fusion v4（正式运行完成）

做了什么：新建隔离的 `dta/rank_fusion_v4/` 和 `experiments/lp_pdbbind/performance_fusion_v4/`，以 Hybrid Upper-Bound v2 cached seed87、Boltz pairwise rank010 pair-delta 和 3DProtDTA 为三个冻结 teacher，离线导出 keyed prediction/latent，再训练以 CI/Spearman 为主的 rank fusion。官方 split 固定为 train `11,037`、valid `2,340`、test `4,751`；validation 负责模型选择，test 只允许在 selection seal 后运行一次。3DProt 适配数据已完整生成到 `external_benchmarks/3d-prot-dta/data/lp_pdbbind/`，共 18,128 行和 18,128 个 protein graph，299 条缺失 SMILES 已从 ligand SDF 恢复。

结果怎么样：首次 GPU 7 smoke 中 Upper-Bound teacher 的 train 前 64 条导出成功，Boltz teacher 在首条 `10gs_protein` 中断。根因不是 key/split 错位，而是 Boltz 旧索引从 `Ki=0.4uM` 重新计算得到 `6.397940...`，官方 manifest 使用 PDBbind 标签 `6.4`。全量对比显示 18,128 条 key 全部匹配，其中 18,120 条源标签与 manifest 的差值不超过 `0.0050001`；其余 8 条为历史注释差异（train 7、valid 1、test 0）。提交 `e5b7dcd` 将适配器升级为 `boltz-pairwise-v2`：官方 manifest 始终是 fusion artifact 的唯一标签源，Boltz 源标签只用于审计；超过四舍五入容差的条目写入 `adapter_metadata.json`，不再阻断与标签无关的冻结特征导出，非有限源标签仍立即失败。

结果分析：真实 `10gs_protein` 单样本导出已完成，`count=1`、`missing_count=0`、latent dimension `640`，审计到的绝对标签差为 `0.0020599365`。Rank Fusion v4 全套回归测试 `82/82` 通过，Python compileall 和 v4 GPU shell syntax 均通过。2026-08-14 在物理 GPU 7 完成全新 64 条、seeds 42/87、单 epoch smoke；三个 teacher 的 train/valid/test export、共享 normalization、keyed join、训练、validation selection、selection seal 和受限 test evaluator 全链路均完成。64 条 validation 前缀上 baseline guard 选择 `boltz_pairwise`，CI `0.631816`；对应 64 条 test 前缀 CI `0.627304`、RMSE `2.127970`。这些小前缀指标只证明链路可运行，不进入 2,340/4,751 条正式性能比较。首次全量启动在任何 expert 导出前因 Bash 4.2 `set -u` 展开空 `limit_args[@]` 而退出；提交 `203b192` 将四处可选 limit 参数改为 nounset-safe 展开，并增加真实调用 `export_split train`/`build_join train` 的回归测试。修复后 v4 全套测试为 `83/83`。

2026-08-14 全量五 seed 正式运行完成。validation selection 最终回退到 `upper_bound` 单 expert：Val CI `0.700478`、RMSE `1.370511`、MAE `1.088446`、Pearson `0.592084`、Spearman `0.567070`。全量 official test `n=4751` 与 Upper-Bound v2 相同：CI `0.686741`、RMSE `1.497357`、MAE `1.199309`、Pearson `0.558346`、Spearman `0.534113`。因此 v4 没有带来可部署提升，后续转向以 validation-only static fusion 为锚点、受限 residual 为修正项的 v5。

## 5.10 Anchored Expert Fusion v5

做了什么：新建隔离的 `dta/anchored_expert_fusion_v5/` 和 `experiments/lp_pdbbind/performance_fusion_v5/`。以 Upper-Bound、Boltz pairwise 和 3DProtDTA 三个冻结 expert 为输入，先在 official train 内做五折 raw/rank static fusion，再训练受预算约束的 anchored residual。正式 inventory 固定为 47 个 validation candidate：3 个 expert、2 个 static、30 个 residual（family `raw/rank`，budget `0.10/0.25/0.50`，seed `1/7/21/42/87`）和 12 个 ensemble。模型选择只使用 2,340 条 validation；package 封存后只运行一次 4,751 条 official test。

结果怎么样：重封存后的最终选择为 `residual_rank_10_1`，即 rank static anchor、residual budget `0.10`、seed `1`。Validation 指标为 CI `0.713374`、RMSE `1.552813`、MAE `1.217081`、Pearson `0.600760`、Spearman `0.600070`。全量 official test `n=4751`、coverage `1.0`，状态 `passed`：CI `0.708700`、RMSE `1.511411`、MAE `1.174794`、Pearson `0.599770`、Spearman `0.586793`。Test CI 比冻结目标 `0.706858` 高 `0.001842`，比 Upper-Bound v2/v4 的 `0.686741` 高 `0.021959`，比已记录的 3DProtDTA `0.696331` 高约 `0.012369`。Validation 到 test 的 CI 下降为 `0.004675`。

结果分析：v5 首次把自研 LP-PDBbind 路线的 official test CI 提升到 `0.70` 以上，并严格超过预设目标。提升主要体现在排序：相对 Upper-Bound，Test Spearman 从 `0.534113` 提升到 `0.586793`、Pearson 从 `0.558346` 提升到 `0.599770`；RMSE 从 `1.497357` 小幅变差到 `1.511411`，MAE 从 `1.199309` 改善到 `1.174794`。这说明 anchored rank residual 有效改善全局排序，但下一步若继续优化，应在保留 CI/Spearman 的同时控制平方误差尾部。

工程记录：正式运行期间依次修复 CUDA bool `argsort`、partial-grid `RESUME=1` 调度，以及 ensemble 生成端使用内存 `float64`、evaluator 使用落盘 `float32` 导致的 1 ULP 封存不一致；对应提交为 `a041e62`、`e134624`、`92165cb`。最终 selection seal SHA256 为 `377811ff29aeeeebf8f30173970a32ef86c95cf5f9625179ed25e5ac48230cc2`，prediction SHA256 为 `4a11ec4ebbaa32d03c1341414892fe8de8ff69e324ec7635b93df06df7769e37`。正式结果位于 `Model/LP_PDBBind_AnchoredExpertFusion_v5/official_test/result.json`；`evaluation.complete` 已存在，不再重复运行 official test。

## 5.11 Rank-Locked Calibration v6

做了什么：在独立 `dta/rank_locked_calibration_v6/` 中固定 v5 的样本排序，只拟合 identity、affine、单调 spline 和条件校准器，目标是在不改变 CI/Spearman 的前提下降低 RMSE/MAE。

结果怎么样：正式 validation 上所有候选的 CI 均保持 `0.713374`、Spearman 均为 `0.600070`。最佳数值误差候选 `spline_8_knots` 的 Val RMSE 为 `1.526582`、MAE 为 `1.197877`，改善幅度未达到 promotion gate，因此状态为 `validation_failed`，没有运行新的 official test，v5 正式结果保持不变。

结果分析：纯单调校准只能改变数值尺度，不能继续提升当前核心目标 CI。该路线按预设回退，没有覆盖 v5 artifact，后续转向允许改变排序的融合模型。

## 5.12 Rank-Unlocked Fusion v7

做了什么：新建隔离的 `dta/rank_unlocked_fusion_v7/` 和 `experiments/lp_pdbbind/performance_fusion_v7/`。模型将 Upper-Bound、Boltz pairwise、3DProt 的 `1024/640/1024` 维 latent 分别适配到 256 维，通过 4-token、3-layer、8-head Transformer、动态 expert gate 和 v5 anchor residual 联合建模；损失组合 SmoothL1、MSE、pairwise ranking、hard-pair 和 residual 约束。2,340 条 validation 只用于 grouped five-fold OOF 训练与选择，正式 test 在 selection 写出后评估。

结果怎么样：正式 OOF CI `0.723261`、RMSE `1.251133`、MAE `0.963312`、Pearson `0.630514`、Spearman `0.625432`。最终 OOF 选择 seed 7、residual scale `0.5`。全量 official test `n=4751`：CI `0.716345`、RMSE `1.381976`、MAE `1.085512`、Pearson `0.629327`、Spearman `0.605653`；相对 v5 Test CI 提升 `0.007646`，状态 `passed`。

结果分析：v7 同时改善排序、线性相关和回归误差，在 v8 正式运行前是 LP-PDBBind joined cache 上最好的正式结果。train/valid/test key 交集均为 0；剔除 test 中与 valid 存在 exact protein sequence 或 exact SMILES 重叠的样本后剩 4,408 条，v5/v7 CI 分别为 `0.706781`/`0.714161`，提升仍为 `0.007380`，没有证据表明提升由 exact overlap 驱动。但 test 已被历史多个版本重复评估，不能把该结果描述为全新盲测。正式结果：`Model/LP_PDBBind_RankUnlockedFusion_v7/run/result.json`。

## 5.13 Repeated Rank Fusion v8（正式运行完成）

做了什么：新建隔离的 `dta/repeated_rank_fusion_v8/` 和 `experiments/lp_pdbbind/performance_fusion_v8/`，不修改 v7。V8 保留 v5 anchor 和 v7 seeds `1/7/21` 的 OOF 预测，新增 fold seeds `137/271/389` 的三组 grouped five-fold repeat，共新增 15 个模型；七列 OOF 特征由非负 pairwise rank stacker 组合，正则候选为 `0.01/0.05/0.10/0.50`。正式模式只有在 meta-OOF CI 严格超过 v7 OOF 后才读取 test，最终 test CI 还必须严格超过 v7 `0.716345`。

结果怎么样：V8 单元测试 `12/12`、V7 回归 `9/9`、Python 编译和 shell syntax 均通过。正式 OOF 选择 `positive_pairwise_0.01`，OOF CI `0.728305`、RMSE `1.224576`、MAE `0.941856`、Pearson `0.650294`、Spearman `0.636429`，相对 v7 OOF CI 提升 `0.005044`。全量 official test `n=4751` 状态 `passed`：CI `0.718266`、RMSE `1.374328`、MAE `1.072943`、Pearson `0.637626`、Spearman `0.611355`。相对 v7 Test CI 提升 `0.001920`，其余四项 test 指标也全部改善。

结果分析：V8 的目的不是增加单个网络容量，而是降低 v7 对单一 grouped-fold 划分和 seed 7 的依赖，并利用严格 OOF 的多次重复估计改善排序泛化。最终七路非负权重按 `v5_anchor/v7_seed_1/v7_seed_7/v7_seed_21/repeat_137/repeat_271/repeat_389` 顺序为 `0.188222/0/0.134406/0/0.260032/0.102889/0.141023`，说明三个新增 repeat 均被保留，而 v7 seed 1 和 21 被淘汰。OOF 增益只有一部分转移到 test，但方向一致且 test 回归与相关指标没有被牺牲。按官方 CSV exact protein sequence 或 exact SMILES 剔除 322 条重叠 test 样本后剩 4,429 条，v7/v8 CI 分别为 `0.714242`/`0.715721`，增益仍为 `0.001478`。正式结果为 `Model/LP_PDBBind_RepeatedRankFusion_v8/run/result.json`，selection SHA256 为 `542727dce7dcfd95396530ad76e34cd70f79e3d035a5efd6b0786a1ba5347f78`。

## 5.14 MSA Pocket Expert v9：mini 数据库管线验证

做了什么：新建独立的 `dta/msa_pocket_expert_v9/`，不修改 v8。首先实现 LP-PDBbind chain sequence 导出器，按规范化序列 SHA256 跨 train/valid/test 去重，并保留 split、LMDB key、chain ID 到唯一序列的完整映射。使用官方 `mini_swissprot2503` 数据库验证 MMseqs2 搜索与 A3M 生成。

结果怎么样：mini 压缩包大小 `275651964` 字节并通过 gzip 完整性检查；MMseqs2 版本为 `18.8cc5c`。从 train/valid/test 各取 4 个真实样本，得到 15 个 chain occurrence 和 12 条唯一序列。直接把 profile search 结果交给 `_seq` 库会产生 target ID 错配和非法 A3M，该输出已判为失败且未进入后续缓存。改为官方三迭代流程 `search -> profile -> expandaln -> align -> filterresult -> result2msa` 后，12/12 查询均生成合法 A3M；深度最小 `71`、中位数 `262`、均值 `297.42`、最大 `614`。逐文件验证 query 与 SHA manifest 完全一致，所有 alignment 去除小写 insertion 后均与 query 等长。序列导出单元测试 `2/2` 通过，Python compile 通过。

结果分析：mini smoke 已证明 LP manifest/LMDB 取序列、跨 split 去重、MMseqs profile 展开和 A3M 对齐链可运行。mini 的 MSA 深度不用于模型性能判断。正式 UniRef30 归档已上传到服务器且大小与远端 `106729528496` 字节一致；下一步在独立正式 cache 中运行全量唯一序列搜索，再提取 residue profile、entropy、coverage 和 Neff。临时 smoke 输出位于 `/tmp/LP_PDBBind_MSA_v9_mini_queries/`。

2026-08-23 继续完成正式搜索准备。UniRef30 已解压到 `/home/user/wangzhiwen/databases/colabfold_v9_uniref30/`，主库及 `_seq/_aln` 数据和索引齐全。全量 LP-PDBbind 共导出 `22,077` 个 chain occurrence，按规范化序列 SHA256 去重后为 `10,688` 条唯一查询；清单位于 `data/LP_PDBBind_MSA_v9/sequences/`。新增 `dta/msa_pocket_expert_v9/mmseqs_search.py`，固定执行 ColabFold 的 `createdb -> search -> profile -> expandaln -> align -> filterresult -> result2msa -> unpackdb` 链，并为各阶段写入恢复 marker、校验调用谱系以及逐条检查 A3M query/列数。新执行器在 mini 库上复现 `12/12` 合法 A3M，深度最小 `71`、均值 `297.42`、最大 `614`；中断后重试会跳过所有已完成阶段。V9 focused tests 当前为 `7/7`。正式 UniRef30 smoke 脚本为 `experiments/lp_pdbbind/performance_fusion_v9/smoke_uniref30.sh`，全量脚本为同目录 `run_uniref30_full.sh`；两者会拒绝在 `ln*` 登录节点执行，正式 smoke 已在 `gpu10` 使用 32 线程完成：12/12 查询通过 A3M 校验，深度最小 `1,547`、均值 `6,671.42`、最大 `11,708`；八个搜索阶段合计约 `21.8` 分钟，A3M 共约 `25 MB`，含中间库共约 `244 MB`。随后新增 `msa_profile.py`，使用 Henikoff position-based 权重生成每残基 50 维频率、log-odds、entropy、conservation、coverage、insertion 和 Neff 特征；正式 smoke 的 profile LMDB 为 `662 KB`，Neff 范围 `704.73–4,032.16`，频率归一化最大误差 `1.19e-7`。全量 `unpackdb` 在写出 4,468/10,688 个松散 A3M 后因 BeeGFS 元数据 I/O 停止推进；保留完整 20GB `uniref.a3m` 数据库，并新增 `mmseqs_profile_db.py` 按 `.index` offset 顺序直接读取、使用节点本地单一临时文件解析并增量写 LMDB。全量脚本已固定 `--unpack 0`，不再重复该高元数据压力步骤。


2026-08-24 完成全量 profile 与 V9 训练链路。直接数据库读取共处理 `10,688/10,688` 条唯一序列，覆盖全部 `22,077/22,077` chain occurrence；MSA depth 最小 `1`、均值 `5,108.36`、最大 `20,660`，Henikoff Neff 最小 `1.0`、均值 `2,584.62`、最大 `8,697.83`。50 维 residue profile LMDB 位于 `data/LP_PDBBind_MSA_v9/uniref30_profiles/`，约 `636 MB`，覆盖 train/valid/test 的 `11,037/2,340/4,751` 个样本。新增 pocket complex cache，将 profile 严格映射到 chain residue 顺序，并保存最多 160 个口袋残基、ESM2 residue embedding、ligand graph、全原子 residue-ligand 最短距离和 contact label。独立 V9 模型采用 gated MSA/ESM 双分支、geometry-biased joint protein-ligand Transformer、contact 辅助头、affinity head 和 1024 维 latent；正式配置为 width 512、8 layers、8 heads、BF16、label-spread batch 和 CI 优先的 raw/EMA checkpoint 选择。64 条各 split cache 已完成；真实 cache 的 CPU 端到端训练、valid/test 预测与 latent 导出通过，V9 focused tests 更新为 `11/11`。GPU 7 smoke 与正式入口分别为 `smoke_training_gpu7.sh` 和 `run_training_gpu7.sh`，输出均与 V8 隔离。

GPU 7 smoke 已完成，状态 `complete`，64 条 train/valid/test、3 epochs 全链路无 CUDA、BF16、非有限梯度或 LMDB worker 错误。训练 loss 从 `1.038130` 降至 `0.863380`；由于样本极少，best epoch 0 的 validation CI `0.529015` 和 test CI `0.498505` 仅用于运行验证，不作为性能结论。输出为 `/tmp/LP_PDBBind_MSAPocketExpert_v9_msa_pocket_v9_smoke_20260824_143240`。

首次 full cache 在 train 第 501--600 条之间遇到 `1idg_protein` 的 166 原子配体并因原 160 上限停止。随后顺序扫描源 LMDB 的 18,862 个 complex：配体原子数最大 224，P99/P99.5/P99.9 分别为 123/135.7/156.1；13 个超过 160、1 个超过 192、0 个超过 256。正式 cache 上限据此提高到 256，已写入的 500 条全部满足旧上限并保留复用；smoke cache 仍保持 160 谱系。


正式 V9 使用 width 512、8 layers、8 heads、BF16 在全部 `11,037/2,340/4,751` 样本上完成，best epoch 4、EMA checkpoint。Validation 为 CI `0.690276`、RMSE `1.365131`、MAE `1.070981`、Pearson `0.560616`、Spearman `0.542523`；Test 为 CI `0.682576`、RMSE `1.468233`、MAE `1.179861`、Pearson `0.549479`、Spearman `0.523271`。V9 单专家未超过 V8，但其 prediction 与 1024 维 latent 保留用于后续互补融合。正式目录为 `Model/LP_PDBBind_MSAPocketExpert_v9/msa_pocket_v9_full_retry_20260824_154027/`。

## 5.15 MSA Rank Fusion v10：V8/V9 非线性 latent 融合

做了什么：新建隔离的 `dta/msa_rank_fusion_v10/` 和 `experiments/lp_pdbbind/performance_fusion_v10/`，不修改 V8/V9。首先仅在 validation 上按 0.01 网格选择标准化 V8/V9 prediction blend，得到 V8/V9 权重 `0.85/0.15`；随后将冻结 V9 的 1024 维 latent 用 validation-only PCA 压缩到 64 维，与 V8 七路 expert prediction 和 V9 prediction 拼接。使用 protein/ligand grouped five-fold OOF 选择 ExtraTrees residual 的 leaf size、feature fraction 和 anchor budget，最终为 minimum leaf `10`、max features `1.0`、residual budget `0.30`。Test 在所有 validation 选择完成后才进行最终评估。

结果怎么样：blend validation CI 为 `0.729572`；非线性 residual grouped-OOF validation CI 提升到 `0.730892`，RMSE `1.208193`、MAE `0.932012`、Pearson `0.662004`、Spearman `0.642909`。全量 Test `n=4,751` 状态 `passed`：CI `0.722141`、RMSE `1.346158`、MAE `1.049516`、Pearson `0.648728`、Spearman `0.620507`。相对 V8，Test CI 提升 `0.003876`，RMSE 改善 `0.028170`，并且 MAE、Pearson、Spearman 全部改善。

结果分析：V9 单专家的总体排序弱于 V8，但 MSA latent 中存在 V8 未覆盖的局部残差信号。较小的 15% prediction blend 加上受限 30% 非线性残差能够把该信号转化为 test 增益；validation 与 test 同向提升，且不是以回归误差退化换取 CI。正式产物位于 `Model/LP_PDBBind_MSARankFusion_v10/run/`，其中 `model.joblib`、`valid_predictions.npz`、`test_predictions.npz` 和 `result.json` 已完成加载与指标重算校验。

## 5.16 MSA Pair Expert v11：原始 MSA 共进化建模（正式运行完成）

做了什么：新建隔离的 `dta/msa_pair_expert_v11/` 和 `experiments/lp_pdbbind/performance_fusion_v11/`，不修改 V9/V10。V11 从完整 A3M 为每条唯一蛋白确定性保留最多 128 条多样化 MSA row；训练时每条链独立随机抽取 64 行，跨链位置不构造伪共进化关系。模型使用 row/column axial attention 和低秩 Outer Product Mean 生成 pocket residue-pair representation，并将其注入 residue-residue bias、residue-ligand cross bias 和 residue token。训练目标包含 affinity regression、pairwise rank、protein-ligand contact、同链 residue contact 和高低亲和力辅助分类。最终导出 prediction 与 1024 维 latent，使用 grouped-fold OOF 模型与冻结 V10 融合；V10 独立 artifact 始终保留作为回退。

当前验证：V11 focused tests `5/5` 通过。真实 A3M probe 成功读取 6 条查询，原始深度最小 `1,547`、均值 `7,462.33`、最大 `11,708`；两个真实 validation complex 已拼接为 `(2,8,82)` pocket-MSA 并完成有限值前向。正式配置约 `33.52M` 参数，128 行 token 与 insertion 主体估计约 `0.746 GiB`。2026-08-25 在物理 GPU 7 完成 train/valid/test 各 24 条、3 epochs 的 smoke。63 条所需唯一序列全部写入 32-row cache，原始 MSA depth 最小 `706`、均值 `6,819.92`、最大 `12,775`；CUDA、BF16、LMDB worker、反向传播、EMA checkpoint 和预测导出全链路完成，无 OOM、dtype、非有限 loss 或非有限梯度错误。best epoch 0、EMA validation CI `0.509091`，test CI `0.472727`；该固定小前缀只用于运行验证，不参与模型排名或性能判断。正式 V11 在全部 `11,037/2,340/4,751` 样本上运行 34 epochs，best epoch 19、EMA checkpoint。单专家 Validation CI `0.689116`、RMSE `1.356328`、MAE `1.066883`、Pearson `0.567243`、Spearman `0.537211`；Test CI `0.672090`、RMSE `1.518105`、MAE `1.219500`、Pearson `0.517828`、Spearman `0.494342`，单专家弱于 V9。V10+V11 validation 选择为标准化 prediction blend，权重 V10/V11=`0.91/0.09`；ExtraTrees latent residual budget 为 `0.0`，说明 V11 latent 没有获得额外权重。融合 Validation CI `0.731556`、RMSE `1.204330`；全量 Test CI `0.722219`、RMSE `1.336469`、MAE `1.042882`、Pearson `0.648779`、Spearman `0.620379`。相对 V10，Test CI 仅增加 `0.000078`，RMSE 改善 `0.009689`、MAE 改善 `0.006634`，Spearman 下降 `0.000128`。状态虽为 `passed`，但排序增益接近零，主要收益是数值误差改善；当前 MSA pair encoder 尚未把共进化表示转化为强互补 latent。正式产物位于 `Model/LP_PDBBind_MSAPairExpert_v11/msa_pair_v11_full_20260825_165726/` 和 `Model/LP_PDBBind_MSAPairFusion_v11/msa_pair_v11_full_20260825_165726/`。

## 5.17 Pretrained MSA Pair Fusion v12（正式运行完成）

做了什么：新建隔离的 `dta/msa_pair_pretrained_v12/` 和 `experiments/lp_pdbbind/performance_fusion_v12/`，不修改 V11 代码、checkpoint 或预测。V12 在 V11 MSA-to-pair encoder 上增加两项结构预训练：15% masked-MSA token reconstruction，以及仅同链、排除近邻残基的 10-bin residue distogram；同时保留 balanced residue-contact 辅助目标。亲和力微调严格加载预训练 `encoder_state`，主网络使用基础学习率，MSA-pair encoder 使用 0.30 倍学习率，并继续施加 distogram loss，避免预训练表征在微调早期被覆盖。最终融合锚点固定为 V11 fusion，而不是 V10；prediction blend、PCA 和 ExtraTrees residual 的全部选择只读取 validation，test 仅报告是否真正超过 V11 fusion。

当前验证：V12 模型、预训练器、微调器和融合器已通过 Python compile；full/smoke GPU7 脚本均通过 `bash -n`。专属 CPU 测试 `8/8` 通过，V11 回归与 V12 合并测试 `13/13` 通过，覆盖 MLM/distogram 输出与有限梯度、预训练 checkpoint 严格加载、刚体运动不变性、V11 fusion 字段对齐以及 synthetic 端到端融合。另用两个官方 validation complex 和正式 row cache 做过真实 CPU pretraining probe：masked token 数 172，distogram shape `(2,82,82,10)`，总 loss `5.772339`，所有梯度有限。正式锚点已核对为 validation `2,340`、test `4,751`，字段包含 V10/V11/final predictions 和 validation fold IDs。

2026-08-26 首次 GPU 7 smoke 的 masked-MSA/distogram 预训练和 3-epoch affinity 微调均完整结束，best epoch 1、EMA checkpoint；24 条 validation CI `0.532727`，24 条 test CI `0.465455`，这些固定小前缀指标只用于功能验证。最后的 CPU 融合阶段首次因截断后的原始 fold IDs 不连续而被底层 OOF 校验拒绝。修复将 smoke 子集实际出现的 fold IDs 稳定重映射到连续 `0..K-1`，完整正式 fold 的语义不变；非连续 fold 端到端回归及全部 V12 focused tests `8/8` 通过。复用已完成的预训练和微调输出后，融合重试完整结束：24 条 validation 上 V11 fusion anchor CI `0.756364`，选择 V11/V12 权重 `0.71/0.29` 后 CI `0.789091`，ExtraTrees residual budget 为 `0.0`；对应 24 条 test CI 从 anchor `0.472727` 降到 `0.461818`，因此状态按保护规则为 `not_improved`。Test RMSE 从 `2.396976` 小幅改善到 `2.388113`。

2026-08-26 正式全量运行完成。预训练运行 12 epochs，best epoch 9、raw checkpoint；预训练 validation loss `2.288492`，masked-MSA loss `0.731416`、token accuracy `0.786550`，distogram loss `1.409048`、distogram accuracy `0.434317`，balanced contact loss `0.740144`。Affinity 在全部 `11,037/2,340/4,751` 条 train/validation/test 样本上运行 27 epochs，best epoch 12、EMA checkpoint。Standalone Validation 为 CI `0.685643`、RMSE `1.368792`、MAE `1.073438`、Pearson `0.537229`、Spearman `0.533873`；Standalone Test 为 CI `0.675677`、RMSE `1.484315`、MAE `1.173943`、Pearson `0.521637`、Spearman `0.505782`。

正式融合的全部模型选择只读取 validation。标准化 prediction blend 选择 V11 fusion/V12 权重 `0.88/0.12`，Validation CI `0.732808`；对 V12 latent 做 64 维 PCA 后，grouped-OOF ExtraTrees 进一步选择 minimum leaf `10`、max features `0.7`、residual budget `0.10`。最终 Validation 为 CI `0.732831`、RMSE `1.199928`、MAE `0.928141`、Pearson `0.667722`、Spearman `0.647241`，相对冻结 V11 validation CI 提升 `0.001276`。随后只报告全量 Test `n=4,751`：CI `0.724087`、RMSE `1.321893`、MAE `1.030705`、Pearson `0.652407`、Spearman `0.624838`，状态 `passed`。相对 V11，Test CI 提升 `0.001868`、RMSE 改善 `0.014576`、MAE 改善 `0.012177`、Pearson 提升 `0.003628`、Spearman 提升 `0.004459`。

结果分析：V12 standalone 明显弱于 V11 fusion，但 validation 只给它 `12%` prediction 权重，说明预训练专家的价值主要来自与既有锚点互补，而不是独立预测能力。与 V11 的 `0.0` latent residual budget 不同，V12 的 `0.10` residual 被 grouped OOF validation 接受，且 validation 与 test 的 CI、回归误差和相关系数全部同向改善，表明 masked-MSA/distogram 预训练确实产生了可利用的互补 latent；绝对 CI 增益仍较小，后续版本应继续以冻结 V12 fusion 为新锚点。正式产物分别位于 `Model/LP_PDBBind_MSAPairPretraining_v12/msa_pair_pretrained_v12_full_20260826_144811/`、`Model/LP_PDBBind_MSAPairExpert_v12/msa_pair_pretrained_v12_full_20260826_144811/` 和 `Model/LP_PDBBind_MSAPairFusion_v12/msa_pair_pretrained_v12_full_20260826_144811/`。

## 5.18 MSA Interface Pretraining v13（正式运行完成，validation 未晋级）

做了什么：新建隔离的 `dta/msa_interface_pretrained_v13/` 和 `experiments/lp_pdbbind/performance_fusion_v13/`，不覆盖 V12 代码、checkpoint、预测或融合产物。V13 在可严格载入 affinity 的同一个 encoder 内加入 masked residue–ligand 9-bin distance reconstruction、balanced interface contact 和双向 in-batch protein–ligand matching，同时保留 V12 的 masked-MSA、同链 residue distogram 和 balanced residue-contact 目标。被选择重建的 residue–ligand 距离会同时从输入 RBF 和 visibility bit 中隐藏，避免接口头直接读取监督答案；接口 pair state、residue/ligand summary 和 cross-attention bias 再注入 affinity trunk。

初始化与训练策略：V13 预训练从 V12 正式 `best_pretrain.pt` 兼容加载已有 MSA/residue-pair 参数，新接口层独立初始化；affinity 微调从 V12 正式 `best.pt` 加载除旧 encoder 外的 task trunk，再严格加载 V13 `encoder_state`。基础学习率为 `1.5e-4`，encoder 使用 `0.20` 倍学习率；微调继续施加 masked interface distogram/contact、matching 和 residue distogram 等辅助目标。正式维度为 MSA width `192`、4 layers、6 heads，pair factor/width `8/96`，主干 width `512`、8 layers、8 heads，row depth `64`、batch size `4`、BF16。

模型选择隔离：冻结锚点固定为 V12 fusion `Model/LP_PDBBind_MSAPairFusion_v12/msa_pair_pretrained_v12_full_20260826_144811/`。正式 affinity 首轮使用 `--skip-test`，只生成 validation prediction；validation blend、PCA 和 grouped-OOF ExtraTrees residual 至少要比 V12 validation CI 提升 `0.003` 才写出 `validation_passed`。选择文件封存 V13 validation prediction SHA256，只有通过后 `--test-only --selection-result` 才能读取 test；最终 test CI 还必须严格超过 V12 才标记 `passed`。validation selection 与最终融合写入两个独立目录，避免选择阶段产物阻塞最终输出；脚本也会复用已经完整生成的 pretrain、validation、test 或 final fusion artifact，便于昂贵阶段后续跑。

当前验证：全部 V13 Python 模块、两个专属测试文件均通过 compile，三个 CLI help 和 full/smoke 两个 GPU7 Bash 脚本通过检查。V13 专属测试 `14/14` 通过；加入 V11/V12 回归后的合并套件为 `27/27`，覆盖接口输出与有限梯度、masked distance 不可见性、刚体不变性、V12 pretrain 参数兼容复用、V12 affinity trunk warm-start、V13 encoder 严格加载、非连续 fold ID、validation 失败时不读取 test、validation/test 两阶段执行和错误 selection SHA256 拒绝。正式 V12 checkpoint 已在 V13 正式维度上做 CPU 实载，分别成功复用 `103` 个 encoder 参数和 `185` 个 affinity trunk 参数。另从正式 validation cache 读取 `1a0q_protein`、`1a28_protein` 两个真实 complex，CPU probe 的 MSA shape 为 `(2,8,82)`、interface logits 为 `(2,82,23,9)`，遮蔽 `174` 个 MSA token 和 `1,163` 个 residue–atom pair，总 loss `6.998013`，`116` 组有梯度参数全部有限。

2026-08-27 正式全量运行完成。预训练共运行 16 epochs，best epoch 10、EMA checkpoint；validation 总 loss `2.497240`，masked-MSA loss `0.717879`、token accuracy `0.791459`，residue distogram loss `1.358792`、accuracy `0.463409`，residue contact loss `0.703736`，interface distogram loss `0.972571`、accuracy `0.583373`，interface contact loss `0.325670`，matching loss `1.631378`。这些结果说明新接口重建目标本身能够收敛。

Affinity 使用全部 `11,037` 条 train 和 `2,340` 条 validation，严格保持 test 未构造；训练因 early stopping 共运行 16 epochs，best epoch 1、EMA checkpoint。Standalone Validation 为 CI `0.687731`、RMSE `1.357051`、MAE `1.056715`、Pearson `0.546104`、Spearman `0.537675`。相对 V12 standalone，五项 validation 指标均改善：CI `+0.002088`、RMSE 改善 `0.011741`、MAE 改善 `0.016723`、Pearson `+0.008875`、Spearman `+0.003802`。

正式 validation fusion 仍将 V13 完全淘汰：标准化 blend 选择 V12 fusion/V13 权重 `1.0/0.0`，ExtraTrees residual budget 为 `0.0`；最终 Validation CI `0.732831`，相对冻结 V12 fusion 的 CI 增益严格为 `0.0`，低于预设 `+0.003` promotion gate，状态为 `validation_failed`。因此脚本没有读取 V12 anchor test、没有构造 V13 test dataset、没有生成 `test_predictions.npz`，也没有任何 V13 test 五项指标可报告。正式选择产物位于 `Model/LP_PDBBind_MSAInterfaceFusion_v13/msa_interface_pretrained_v13_full_20260827_132414_validation_selection/`。

结果分析：V13 与 V12 fusion 的 validation prediction Pearson 为 `0.840717`，二者误差 Pearson 高达 `0.902490`，说明接口预训练带来的 standalone 小幅增益主要沿用了既有错误模式，没有形成可供冻结锚点利用的新互补残差。Affinity 最优点过早出现在 epoch 1；后续 raw validation CI 多次明显坍塌，而 EMA 只能维持在约 `0.67–0.69`，同时训练 interface distogram loss 从约 `0.80` 上升到约 `0.92`，显示 warm-start 后仍存在早期遗忘和优化不稳定。失败原因不是 promotion gate 过严：即使取消 `+0.003` 门槛，validation 模型选择本身也给 V13 prediction 和 latent residual 零权重。V12 保持正式最佳，V13 的 pretrain、affinity 与 validation selection 独立产物全部保留用于后续诊断，但不运行 test。

## 5.19 MSA Interface Feature Fusion v14（正式运行完成）

做了什么：新建隔离的 `dta/msa_interface_feature_fusion_v14/`、`experiments/lp_pdbbind/performance_fusion_v14/`、`Model/LP_PDBBind_MSAInterfaceFeatures_v14/` 和 `Model/LP_PDBBind_MSAInterfaceFusion_v14/`，不修改或覆盖 V12/V13。V14 不重训 V13，而是严格加载 V13 正式 pretrain encoder 与 affinity encoder，分别导出不含最终 affinity latent 的纯接口多尺度统计：interface pair、residue-side summary、ligand-side summary 的 mean/std/max，9-bin interface distance probability 的 mean/std/max、distance entropy、contact probability，以及 residue/ligand 最大 contact 的统计。正式 pair width 为 `96`，所以每路特征固定为 `9 * 96 + 39 = 903` 维。两路表征构成 `pretrain`、`affinity`、`affinity-pretrain delta` 和三者拼接 `combined` 四个候选 family；每个 family 独立进行 validation-only StandardScaler、PCA 和 grouped-OOF ExtraTrees residual 选择，冻结 V12 fusion 始终作为 affinity 锚点。该路径保留论文主体所需的信息流：蛋白 1D MSA/序列表征、配体 2D atom/bond 表征及蛋白-配体 3D 距离共同进入接口编码器，最终与现有融合锚点一起输出 affinity。

模型选择与 test 隔离：正式 promotion gate 在运行前固定为 validation CI gain 至少 `0.001`、Spearman 不下降、RMSE 增量不超过 `0.005`、至少 `4/5` grouped folds 的 CI gain 为正，并且 residual budget 必须非零。第一阶段只生成 `valid_features.npz` 和 validation selection；若状态为 `validation_failed`，脚本直接结束且不构造 test dataset。只有 `validation_passed` 后，test-only 阶段才校验 validation feature SHA256、V13 两个 checkpoint SHA256 和 selection 状态，再生成 `test_features.npz`；test 只报告完整五项指标及是否严格超过 V12，不参与 family、PCA、树参数或 residual budget 选择。`--allow-zero-budget` 仅存在于 24 条 smoke，用于强制验证完整功能链，正式脚本不启用。

当前验证：V14 专属 synthetic CPU 测试 `9/9` 通过，覆盖 `9W+39` 特征宽度、有限值、padding 不变性、刚体运动不变性、V13 pretrain/affinity encoder 严格 checkpoint 加载、四种 feature family、非连续 fold ID 稳定重映射、正式 gate 的五个条件、validation 失败不读取 test、validation/test 两阶段执行和错误 SHA256 拒绝。V11/V12/V13/V14 合并相关回归为 `36/36` 通过；全部 V14 Python 文件 compile、两个 CLI help 和两个 GPU7 Bash 脚本的 `bash -n` 均通过。正式 V13 两份 checkpoint 已在 CPU 上严格实载：pretrain 为 epoch 10 EMA，affinity 为 epoch 1 EMA，encoder 配置一致为 MSA width `192`、4 layers、6 heads、pair factor/width `8/96`、2 个 interface blocks。另用 2 条真实 validation complex、row depth 4 做 CPU 前向，两路输出均为 `(2,903)` 且全部有限，平均绝对差为 `0.165219`，确认 affinity 微调后的接口表征与 pretrain 表征不是重复特征。

2026-08-29 正式全量运行完成。使用 row depth `64`、2 个 inference subsets，在全部 `2,340` 条 validation 和 `4,751` 条 test 上分别导出两路 903 维接口特征。Validation 的四个 family 均获得非零 residual：`affinity` CI `0.734170`、`combined` CI `0.734083`、`pretrain` CI `0.733899`、`delta` CI `0.733378`。按预先固定的 CI 主选择规则最终选中 affinity encoder 特征、64 维 PCA、minimum leaf `10`、max features `1.0`、residual budget `0.30`。其五个 grouped folds 的 CI gain 依次为 `+0.001946`、`+0.001845`、`+0.000386`、`+0.002114`、`+0.003500`，达到 `5/5` 正增益。最终 Validation CI `0.734170`、RMSE `1.195906`、MAE `0.925016`、Pearson `0.670507`、Spearman `0.650233`；相对冻结 V12 validation，CI `+0.001339`、RMSE 改善 `0.004022`、MAE 改善 `0.003125`、Pearson `+0.002785`、Spearman `+0.002991`，通过全部正式 promotion gate。

只有 validation 晋级后才生成正式 test 特征。全量 Test 状态为 `passed`：CI `0.726950`、RMSE `1.315046`、MAE `1.022932`、Pearson `0.658056`、Spearman `0.631365`。相对 V12 fusion，Test CI 提升 `0.002863`、RMSE 改善 `0.006846`、MAE 改善 `0.007773`、Pearson 提升 `0.005649`、Spearman 提升 `0.006527`，五项指标全部同向改善。保存预测重新计算的 valid/test 指标与 `result.json` 完全一致，V12/V14 key 完全对齐；valid/test feature SHA256 也与 validation selection 和 feature result 中封存的值一致。正式产物位于 `Model/LP_PDBBind_MSAInterfaceFeatures_v14/msa_interface_feature_fusion_v14_full_20260829_145451/`、`Model/LP_PDBBind_MSAInterfaceFusion_v14/msa_interface_feature_fusion_v14_full_20260829_145451_validation_selection/` 和 `Model/LP_PDBBind_MSAInterfaceFusion_v14/msa_interface_feature_fusion_v14_full_20260829_145451/`。

结果分析：V14 成功把 V13 中未被最终 affinity prediction/latent 融合利用的接口信号提取为互补 residual。所选 affinity family 在五个 grouped folds 全部改善，且 validation 与 test 的 CI、误差和相关系数一致变好，因此这次提升比 V11 到 V12 的小权重 prediction blend 更稳定。combined family 虽有更低的 validation RMSE 和更高的 Pearson，但 CI 略低且有一个 fold 下降，按预先固定规则不应改选；也不能在看到 test 后再比较其他 family。V14 在完成时成为正式最佳版本，同时保持蛋白 1D、配体 2D、蛋白-配体 3D 接口建模和最终 affinity 输出的论文主体架构；随后已被沿这一方向扩展空间分组统计的 V15 超过。

## 5.20 MSA Spatial Interface Feature Fusion v15（正式运行完成）

做了什么：新建完全隔离的 `dta/msa_interface_spatial_fusion_v15/`、`experiments/lp_pdbbind/performance_fusion_v15/`、`Model/LP_PDBBind_MSAInterfaceSpatialFeatures_v15/` 和 `Model/LP_PDBBind_MSAInterfaceSpatialFusion_v15/`，不修改 V12–V14 的代码或正式产物。V15 继续严格加载 V13 正式 pretrain/affinity encoder，但不再把接口压缩为单一全局 mean/std/max。每个 encoder 额外导出六个 3D 距离壳层（`<4.5`、`4.5–6`、`6–8`、`8–12`、`12–24`、`>=24 A`）的 pair state/contact/entropy 统计，六类蛋白残基化学组、六类配体原子元素组的 pair mean/std 与占比，以及 contact-probability weighted pair moments 和 top-`1/4/16` 高接触 pair pooling。连同 V14 基础统计，每路正式特征固定为 `56 * pair_width + 96 = 5,472` 维。该表示显式保留蛋白 1D residue/MSA 类型、配体 2D atom/bond 类型和蛋白–配体 3D 距离分层，最终仍输出 affinity。

融合与隔离策略：V15 使用冻结 V12 prediction 作为 residual 建模底座，使用冻结 V14 OOF prediction 只做 promotion benchmark，而不把 V14 OOF prediction 当作第二层训练 target 的 anchor，避免同一 validation folds 上的 stacking 泄漏。候选 family 固定为 `affinity`、`pretrain`、`affinity-pretrain delta` 和 `affinity+delta joint`；每个 family 做 96 维 validation-only PCA，再在 minimum leaf `5/10/20`、max features `0.7/1.0` 和 residual budget `0.0–0.60` 的预设网格上进行 grouped-OOF ExtraTrees 选择。正式晋级条件相对 V14 固定为 validation CI 至少 `+0.001`、Spearman 不下降、RMSE 增量不超过 `0.005`、至少 `4/5` folds 的 CI gain 为正且 budget 非零。若失败，脚本在构造 test dataset 前结束；只有 `validation_passed` 后，test-only 阶段才校验 feature/checkpoint SHA256 并生成 test 特征，test 不参与任何 family 或超参数选择。

显存策略：V14 已在相同 width/layers、row depth `64`、batch `4` 下完成。V15 新增部分只对已有 interface pair tensor 做顺序统计，encoder 激活仍是显存主体；正式脚本主动降为 BF16 batch `2`，smoke 为 batch `1`，因此相对已跑通 V14 留出超过一半的 batch 激活余量，目标是不超过物理 GPU 7 的 40GB。

当前验证：V15 专属 CPU 测试 `9/9` 通过，覆盖 `56W+96` 宽度、有限值、padding 不变性、刚体运动不变性、距离/残基/元素分组完整覆盖、四个 family、非连续 fold ID、V14-relative 正式 gate、validation 失败不读取 test、validation/test 两阶段执行和错误 SHA256 拒绝。V11–V15 合并相关回归为 `45/45`；全部 Python compile、两个 CLI help 和两个 Bash 脚本的 `bash -n` 均通过。正式 V13 两个 checkpoint 与 2 条真实 validation complex 的 CPU 前向输出均为 `(2,5472)` 且全部有限。正式 V12 base 与 V14 benchmark 已按字段实载核验：validation `2,340`、test `4,751`，两者 valid/test key 及 grouped fold IDs 完全一致。

2026-08-29 正式全量运行完成。全部 `2,340` 条 validation 和 `4,751` 条 test 均成功导出两路 5,472 维空间接口特征。Validation 四个 family 的 CI 为：`affinity` `0.735177`、`pretrain` `0.734917`、`joint` `0.733356`、`delta` `0.733322`。按预先固定的 CI 主选择规则，最终选中 affinity encoder 特征、96 维 PCA、minimum leaf `5`、max features `1.0`、residual budget `0.60`。五个 grouped folds 相对 V14 的 CI gain 依次为 `+0.003130`、`+0.000018`、`+0.002829`、`+0.000092`、`+0.002159`，达到 `5/5` 正增益。最终 Validation CI `0.735177`、RMSE `1.188280`、MAE `0.924626`、Pearson `0.675708`、Spearman `0.652789`；相对冻结 V14 validation，CI `+0.001007`、RMSE 改善 `0.007626`、MAE 改善 `0.000390`、Pearson `+0.005201`、Spearman `+0.002556`，通过全部正式 promotion gate。

只有 validation 晋级后才生成正式 test 特征。全量 Test 状态为 `passed`：CI `0.730115`、RMSE `1.309242`、MAE `1.014992`、Pearson `0.664159`、Spearman `0.637505`。相对 V14，Test CI 提升 `0.003165`、RMSE 改善 `0.005805`、MAE 改善 `0.007940`、Pearson 提升 `0.006103`、Spearman 提升 `0.006139`，五项指标全部同向改善。CPU 独立重算 valid/test 五项指标与 `result.json` 完全一致；valid/test key 均唯一，validation-selection 目录不含任何 test 产物，selection/final 的 family、PCA、树参数、预算和指标完全一致。两份 valid prediction 仅有 50 个值存在最大 `1.78e-15` 的末位浮点差，指标不变。valid/test feature SHA256 分别为 `571384858129a9d9790a52c14022b33e3e0cc5194cf1ffb3e87823044695985e` 和 `b26ffaff3185ac4e23d3d73f08549d7b9a12e965a2d1bb63b992162bcb79fc7b`，均与封存值一致。正式产物位于 `Model/LP_PDBBind_MSAInterfaceSpatialFeatures_v15/msa_interface_spatial_fusion_v15_full_20260829_160048/`、`Model/LP_PDBBind_MSAInterfaceSpatialFusion_v15/msa_interface_spatial_fusion_v15_full_20260829_160048_validation_selection/` 和 `Model/LP_PDBBind_MSAInterfaceSpatialFusion_v15/msa_interface_spatial_fusion_v15_full_20260829_160048/`。

结果分析：V15 在完成时成为正式最佳版本，证明按 3D 距离壳层、蛋白残基化学组、配体元素组和高接触 pair 分层保留接口信息，能够补充 V14 全局压缩统计丢失的排序信号；同时继续保持蛋白 1D、配体 2D、蛋白–配体 3D 接口共同进入模型并最终输出 affinity 的论文主体。V15 随后作为冻结特征与 promotion benchmark 被 V16 继续利用。

## 5.21 MSA Spatial Stability Fusion v16（正式运行完成）

做了什么：新建完全隔离的 `dta/msa_interface_spatial_stability_fusion_v16/`、`experiments/lp_pdbbind/performance_fusion_v16/` 和 `Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/`，不修改 V12–V15 代码或正式产物。V16 不重新运行 GPU encoder，而是复用 V15 已封存的 validation/test 两路 5,472 维空间接口特征。根据 V15 validation 边界做了预先诊断：保持 V15 OOF residual 不变时，把 budget 从 `0.60` 提高到 `0.65–1.20` 会降低 CI，因此 V16 不是盲目继续增大预算，而是固定 PCA `96`，将 minimum leaf 扩展为 `2/3/4/5/7`、max features 扩展为 `0.7/0.85/1.0`，并把 residual budget 改为 `0.000–1.000` 的 `0.025` 细网格。四个 family 仍为 `affinity/pretrain/delta/joint`，OOF/final trees 分别使用 `300/1200` 个 estimators，seed 固定为 `20260831`。

建模与隔离策略：V16 继续以冻结 V12 prediction 作为 residual target 底座，仅把冻结 V15 OOF prediction 用作 promotion benchmark，避免把已在同组 folds 上训练得到的 V15 OOF prediction 当成第二层 anchor 而造成 stacking 泄漏。正式 gate 在运行前固定为相对 V15 validation CI 至少 `+0.0005`、Spearman 不下降、RMSE 增量不超过 `0.003`、至少 `4/5` folds 正增益且 budget 非零。CPU-only 脚本第一阶段只检查并读取三份 valid 归档；只有写出 `validation_passed` 后才检查 test 归档。Final 阶段使用同一 seed 和完整网格重算 validation，并逐项校验 selection JSON 的版本、输入 SHA256、搜索空间、family、PCA、树参数、budget、五项指标和 gate criteria，一致后才读取 test。

当前验证：V16 专属 CPU 测试 `11/11` 通过，新增覆盖扩展网格、V15 benchmark 历史字段、validation-only/test 两阶段、错误 selection SHA 在 test 前拒绝；V11–V16 合并回归为 `50/50`。V16 Python compile、CLI 入口和 CPU Bash 的 `bash -n` 均通过。正式 validation selection 目录不含任何 test 字段或产物。

2026-08-30 正式 CPU 全量运行完成。Validation 四个 family 的最佳 CI 为：`pretrain` `0.735954`、`affinity` `0.735626`、`joint` `0.733796`、`delta` `0.733616`。最终选中 pretrain encoder 空间特征、96 维 PCA、minimum leaf `3`、max features `0.85`、residual budget `0.625`。五个 grouped folds 相对 V15 的 CI gain 为 `+0.000826`、`+0.001505`、`-0.000468`、`+0.000662`、`+0.001176`，达到 `4/5` 正增益。最终 Validation CI `0.735954`、RMSE `1.184579`、MAE `0.921047`、Pearson `0.678207`、Spearman `0.654356`；相对 V15，CI `+0.000777`、RMSE 改善 `0.003702`、MAE 改善 `0.003579`、Pearson `+0.002499`、Spearman `+0.001567`，通过全部预设 gate。

冻结 Test 状态为 `passed`：CI `0.731367`、RMSE `1.309480`、MAE `1.013752`、Pearson `0.665183`、Spearman `0.640083`。相对 V15，Test CI `+0.001251`、MAE 改善 `0.001240`、Pearson `+0.001025`、Spearman `+0.002578`，但 RMSE 回退 `0.000238`。因此 V16 成为当前 CI、MAE、Pearson 和 Spearman 最佳版本，V15 的 RMSE 仍略优；不能把 V16 描述为 test 五项全部改善。保存预测的 valid/test 五项指标已在 CPU 上独立重算并与 `result.json` 一致；selection/final valid 最大浮点差为 `1.78e-15`，键完全一致且均唯一。selection seal SHA256 为 `ac8b338c0c0afd46166ceb35789f33fefb3cd86813382dfbb692d6fda0775524`，valid/test prediction SHA256 分别为 `e328f32ae97234910430f93620f68b5c7d596f15580c2a1eb9e1f83e91dea6c0` 和 `819aaf8b361b3d0856efa4930cfd212fa8ab65a119b3fe7930b4b61b2ef3c827`；全部输入 SHA 与封存值一致。正式产物位于 `Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824_validation_selection/` 和 `Model/LP_PDBBind_MSAInterfaceSpatialStabilityFusion_v16/msa_interface_spatial_stability_fusion_v16_full_20260830_112824/`。

结果分析：V16 的提升来自 validation 已显示更优的 pretrain 空间 residual 与更细树正则/预算组合，而不是 test 后改选。它继续保留由 V13/V15 encoder 产生的蛋白 1D、配体 2D、蛋白–配体 3D 接口表征，并通过冻结融合输出 affinity，因此论文主体架构不变。V16 对 CI 和秩相关的增益成功转移到 test，但增益已较小且 RMSE 轻微回退；后续若继续，只能依据 validation 上 affinity/pretrain 两路互补性设计预先封存的组合，不能利用已经看到的 V16 test 选择融合权重。随后仅用 validation OOF 做了 V15/V16 标准化凸组合诊断：最佳 V15/V16 权重为 `0.14/0.86`，CI `0.735971`，相对 V16 仅 `+0.000017`，只有 `3/5` folds 正增益，同时 RMSE、MAE 和 Pearson 略退，远低于预设 `+0.0005` 的继续门槛；因此没有建立该 V15/V16 blend 版本，也没有读取任何 blend test。

## 5.22 MSA Interface Train-Supervised Fusion v17（正式 validation 未晋级）

做了什么：新建独立的 `dta/msa_interface_train_supervised_fusion_v17/`、`experiments/lp_pdbbind/performance_fusion_v17/`、`Model/LP_PDBBind_MSAInterfaceTrainFeatures_v17/` 和 `Model/LP_PDBBind_MSAInterfaceTrainSupervisedFusion_v17/` 命名空间，不修改或覆盖 V13–V16。V17 保持 V13/V15 的蛋白 1D MSA、配体 2D atom/bond graph 与蛋白–配体 3D interface encoder，新增全部 11,037 条 train 的双路 5,472 维空间特征导出。监督 head 直接预测 affinity，不使用当前不存在的 V16 train OOF prediction 构造伪 residual；正式候选为 train-fitted PCA `96/192` 后的两种 ExtraTrees、HistGradientBoosting 和 PLS-32，并只在 validation 上选择 feature family、head 和 `prediction=(1-w)*V16+w*direct_head` 的预注册融合权重。

隔离与恢复策略：train 特征先独立落盘并写出 `train_ready` 封存，valid 阶段中断时可以校验 checkpoint/feature SHA 后继续，避免重跑昂贵 train encoder。validation selection 只读取 train/valid feature 和 V16 validation OOF prediction；正式 gate 固定为 CI 至少 `+0.001`、Spearman 不下降、RMSE 增量不超过 `0.003`、至少 `4/5` folds 正增益且 direct-head 权重大于零。只有 gate 通过后才允许构造 test dataset；test-only feature extraction、final train+validation refit 和 test 推理均校验 selection、train/valid feature、checkpoint 与 test feature SHA。smoke 允许零 head 权重仅用于验证完整功能链，正式脚本不启用。

数据谱系核查：`data/LP_PDBBind.csv` 的原始 official assignments 为 train/valid/test `11,513/2,422/4,860`。项目 imported index 可用子集为 `11,135/2,367/4,785`，再与 GerNA chain ESM LMDB 取交集后得到当前 `11,037/2,340/4,751`；三段 split assignment 没有重新随机划分。论文应描述为 “LP-PDBBind official split assignments restricted to 18,128 complexes with complete GerNA-required inputs”，不能写成未过滤的完整 official split。

2026-08-31 正式全量 train/validation 已完成。四个 feature family、PCA `96/192`、两种 ExtraTrees、HistGradientBoosting 与 PLS-32 的预注册网格均只在 validation 上选择。按 CI 排序的候选为 pretrain feature、PCA `96`、ExtraTrees（minimum leaf `3`、max features `0.85`），但最优 direct-head 融合权重为 `0.0`，没有一个 direct head 能补充 V16。该 direct head 自身的 Validation CI `0.665312`、RMSE `1.515882`、MAE `1.200576`、Pearson `0.485020`、Spearman `0.473951`，显著弱于 V16 anchor。V17 valid feature SHA256 为 `571384858129a9d9790a52c14022b33e3e0cc5194cf1ffb3e87823044695985e`。

最终 selected validation prediction 因权重为零而与 V16 完全相同：CI `0.735954`、RMSE `1.184579`、MAE `0.921047`、Pearson `0.678207`、Spearman `0.654356`。五个 grouped folds 的 CI gain 均为 `0.0`，未达到预设的 CI 至少 `+0.001`、至少 `4/5` folds 正增益且 direct-head 权重大于零的 gate。状态为 `validation_failed`；test inputs 未被读取，未生成 test 特征或 prediction，因此不存在 V17 test 指标，也绝不能据此进行 test 调参。该负结果表明：冻结 V13/V15 encoder 的空间统计特征可用于 V12 residual 的局部校正，但在当前 direct-affinity 后处理 head 下不足以形成独立且互补的 predictor。V16 仍为当前正式最佳；不应继续扩大同类 post-hoc 网格，下一步应转向训练期端到端监督的独立 expert，并保持 V16 仅作冻结 validation-only anchor。

2026-09-01 CPU validation-only 互补性审计：将现有 Boltz pairwise + pair-delta expert 的 validation prediction 按 complex key 映射到 V16，共同覆盖 `2,329/2,340` 条，另有 `11` 条保持 V16 回退；Boltz 原始训练标签版本与当前 V16 标签在少量样本上不完全一致，故所有评估均以 V16 labels 为准。标准化凸组合的最优权重为 V16/Boltz `0.975/0.025`，Validation CI `0.736038`，仅比 V16 `+0.000084`，5 folds 只有 `3/5` 正增益，不达到继续 gate。进一步在 V15 pretrain PCA-96 特征上加入 V16、Boltz、二者差值与缺失指示符，按相同 grouped folds 做 OOF ExtraTrees residual；最轻的非零 budget `0.05` 已退化为 CI `0.735618`、RMSE `1.185238`、Spearman `0.653427`、`0/5` 正增益，增大 budget 继续恶化。该诊断的选择与指标计算只使用 validation，未构造或报告任何 V16+Boltz 组合 test 指标。因此不建立该跨模型 fusion 版本。


## 5.23 MSA Interface Pair Pooling v18（正式 validation 未晋级）

做了什么：新建完全隔离的 `dta/msa_interface_pair_pooling_v18/`、`experiments/lp_pdbbind/performance_fusion_v18/`、`Model/LP_PDBBind_MSAInterfacePairPooling_v18/` 和 `Model/LP_PDBBind_MSAInterfacePairPoolingFusion_v18/`，不修改或覆盖 V13–V17。V18 保留 V13 的蛋白 1D MSA/residue、配体 2D atom/bond graph、蛋白–配体 3D interface encoder，在 `interface_pair` 上按真实 residue–ligand 3D 距离选取 top-K pair tokens。每个 token 融合 interface pair state、distance RBF、query-row residue token 和 ligand atomic number，经 2 层 Transformer 与 learned attention pooling 得到 pair latent，再用零初始化 residual head 修正 warm-start affinity prediction。base affinity 与 pair pooling 共享同一次 MSA/interface encoder 前向，避免重复计算和额外显存。

初始化与训练策略：严格复用 V13 正式 interface pretrain encoder `Model/LP_PDBBind_MSAInterfacePretraining_v13/msa_interface_pretrained_v13_full_20260827_132414/best_pretrain.pt`，并使用 V12 affinity checkpoint `Model/LP_PDBBind_MSAPairExpert_v12/msa_pair_pretrained_v12_full_20260826_144811/best.pt` 初始化非 encoder trunk；pretrained encoder 以基础 LR 的 `0.10` 微调。正式配置为 row depth `64`、MSA width `192`、4 layers、6 heads、pair width `96`、main width `512`、8 layers、pair-token width `256`、2 layers、8 heads、top-K `128`、BF16 batch `2`、最多 60 epochs、patience `12`。V16 为冻结 anchor；validation 上预注册选择标准化 prediction blend 与 V18 full latent PCA-128 grouped-OOF residual，晋级条件为 CI 至少 `+0.001`、Spearman 不下降、RMSE 增量不超过 `0.003`、至少 `4/5` folds CI 正增益，且 V18 prediction 或 latent residual 至少一路非零。

验证与 smoke：V18 专属 CPU 测试 `18/18` 通过；与后续 V19 合并回归为 `24/24`。Python compile、两个 CLI、正式/smoke Bash 均通过。真实 validation complex CPU 小模型前后向的输出和梯度有限，pair readout 梯度非零；正式维度成功加载 V13 epoch 10 EMA encoder，并复用 V12 非 encoder 参数 185 组。24 条 GPU smoke 未发生 OOM，全链路完成；其 validation/test 前缀结果仅作功能验证，不进入模型判断。

正式结果：全量 train `11,037`、validation `2,340`，共训练 14 epochs，best epoch `1`、EMA。V18 standalone Validation CI `0.683750`、RMSE `1.435430`、MAE `1.141195`、Pearson `0.537788`、Spearman `0.529274`，低于 V13 standalone；后期 raw CI 多次坍塌，说明联合微调仍存在 warm-start 遗忘和优化不稳定。冻结 V16/V18 prediction blend 选择 V16/V18 权重 `1.0/0.0`。full latent residual 选择 PCA-128、minimum leaf `10`、max features `1.0`、budget `0.30`，得到 Validation CI `0.736464`、RMSE `1.179603`、MAE `0.918439`、Pearson `0.681518`、Spearman `0.655890`。相对 V16，CI `+0.000510`、RMSE 改善 `0.004976`、MAE 改善 `0.002608`、Pearson `+0.003311`、Spearman `+0.001535`；fold CI gain 为 `+0.002533/+0.001643/-0.000625/+0.001176/+0.001461`，达到 `4/5` 正增益，但 CI 未达到预注册的 `+0.001`，状态为 `validation_failed`。脚本没有读取 test，也没有生成 V18 正式 test 指标。产物位于 `Model/LP_PDBBind_MSAInterfacePairPooling_v18/msa_interface_pair_pooling_v18_full_20260901_182434/` 和对应 fusion validation-selection 目录。

结果分析：V18 不是完全无贡献，而是 standalone 退化但 latent 对 V16 存在较弱互补。2026-09-02 追加纯 validation 审计：只用 V12/V15/V16 历史 prediction 标量，或再加入 V18 prediction，最优 budget 均为 `0.0`；V18 base latent 单独使用也为 `0.0`。只有新增 pair latent 单独使用时出现更强非零 residual：Validation CI `0.736645`、RMSE `1.178634`、MAE `0.918605`、Pearson `0.682164`、Spearman `0.656266`，相对 V16 CI `+0.000691`、RMSE 改善 `0.005945`、Spearman `+0.001911`，仍为 `4/5` folds 正增益。这证明有效信号来自 contact-aware pair branch，并被 full latent 中无效的 base latent 稀释；该结论仅使用 validation，未读取 test，据此建立独立 V19。

## 5.24 MSA Interface Pair Latent Fusion v19（正式运行完成，V20 前最佳）

做了什么：新建独立 `dta/msa_interface_pair_latent_fusion_v19/`、`experiments/lp_pdbbind/performance_fusion_v19/`、`Model/LP_PDBBind_MSAInterfacePairLatent_v19/` 和 `Model/LP_PDBBind_MSAInterfacePairLatentFusion_v19/`。V19 不重训、不修改 V18 checkpoint，只从 V18 2,048 维 latent 中按 checkpoint 配置严格提取后半段 1,024 维 pair latent；前半段 base latent 被验证为 validation 无贡献后排除。V16 prediction 固定为 anchor，pair latent 做 validation-only PCA-128，并与封存的 V12/V15/V16/V18 标量一同进入 grouped-OOF ExtraTrees residual。历史标量和 V18 prediction 的 scalar-only 审计均为零 budget，因此正式非零改善可归因于 pair latent。

隔离策略：V19 使用与 V16 相同量级、且不是针对结果临时精确贴线的 promotion gate：Validation CI 至少 `+0.0005`、Spearman 不下降、RMSE 增量不超过 `0.003`、至少 `4/5` folds 正增益且 residual budget 非零。validation feature artifact 封存 V18 checkpoint、V18 result 与 V18 valid prediction SHA256；selection seal 进一步封存 V16/V19 valid SHA、PCA、树参数、budget、五项指标和 gate criteria。只有 seal 为 `validation_passed` 后，test-only 阶段才允许加载 V18 checkpoint 并构造 test dataset；final fusion 会重算 validation 并逐项核对 selection 后才读取 test。

当前验证：V19 专属测试 `6/6`、V18+V19 合并回归 `24/24` 通过；Python compile、两个 CLI help 和 GPU7 Bash `bash -n` 均通过。正式 validation pair feature 共 `2,340 x 1,024`，valid SHA256 为 `30ae9fa442cbf8d87d2f7b9538e9d313015c1f0c4853c79b51d48afb735df32e`。正式 selection 选择 minimum leaf `10`、max features `1.0`、budget `0.30`，Validation CI `0.736645`、RMSE `1.178634`、MAE `0.918605`、Pearson `0.682164`、Spearman `0.656266`；相对 V16 的 CI gain `+0.000691`、RMSE change `-0.005945`、Spearman gain `+0.001911`，fold gain 为 `+0.002414/+0.002102/-0.000579/+0.001167/+0.001029`，通过全部 gate。selection seal SHA256 为 `f8f6a0f2fa42dd07e00cc21fc0a78232f940fcbf0b70f8baaf8f7f6a099b6dce`。

结果怎么样：selection seal 通过后，物理 GPU 7 完成全部 `4,751` 条 test 的两次 MSA subset 推理，并生成 1,024 维 pair latent；final 阶段重算 validation 后逐项通过 seal 校验。正式 Test 状态为 `passed`：CI `0.734503`、RMSE `1.300913`、MAE `1.005961`、Pearson `0.671592`、Spearman `0.647100`。相对 V16，Test CI 提升 `0.003136`、RMSE 改善 `0.008567`、MAE 改善 `0.007791`、Pearson 提升 `0.006409`、Spearman 提升 `0.007017`，五项指标全部同向改善。因此 V19 在完成时成为 LP-PDBBind 正式最佳版本，随后被 V20 超过。正式产物位于 `Model/LP_PDBBind_MSAInterfacePairLatent_v19/msa_interface_pair_latent_fusion_v19_full_20260902_114500/` 和 `Model/LP_PDBBind_MSAInterfacePairLatentFusion_v19/msa_interface_pair_latent_fusion_v19_full_20260902_114500/`。

结果分析：V19 是对 V18 validation 结果的独立后续模型选择，而不是降低 V18 已失败的 `+0.001` gate 或追认 V18。它保留 V18 encoder 中蛋白 1D、配体 2D、蛋白/配体 3D contact-aware pair 建模，并最终通过冻结 V16 anchor 输出 affinity；论文主体架构保持不变。较小的 validation CI 增益在 test 上放大到 `+0.003136`，同时误差与两个相关系数全部改善，证明 pair-only 选择有效去除了 full latent 中 base 分量的噪声。独立 CPU 复算的 valid/test 五项指标与 `result.json` 完全一致；valid/test keys 分别为 `2,340/4,751` 个且全部唯一，selection/final valid prediction 最大差仅 `1.78e-15`。final result、valid prediction、test prediction SHA256 分别为 `c3585f45534a2195c9f989bb54535f6d9e3000e65a67ca12a7df2226c13d78ec`、`6bd0ef6b79e22eec5085e0d984d9d49fb962c62c9bbea09de589a0793aa7c7cc` 和 `16736e32947ecc0d6f2f036f9573886747f4970f9d155729bbfc87d6662d77b7`。test 只用于该冻结配置的一次报告，没有参与 family、PCA、树参数或 budget 选择。


## 5.25 MSA Interface Pair Latent Stability Fusion v20（正式运行完成，当前最佳）

做了什么：新建完全独立的 `dta/msa_interface_pair_latent_stability_fusion_v20/`、`experiments/lp_pdbbind/performance_fusion_v20/` 和 `Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20/`，不修改或覆盖 V16–V19 代码与产物。V20 不重新运行 GPU encoder，直接复用 V19 封存的 `2,340/4,751 x 1,024` contact-aware pair latent。训练 target 仍严格定义为 `label - V16 prediction`，V19 OOF prediction 只作为 promotion benchmark，不能作为二层 residual anchor。这样既利用 V19 已证明有效的蛋白 1D、配体 2D 与蛋白/配体 3D contact-aware pair 表征，又避免把在相同 grouped folds 上形成的 V19 OOF prediction 再次用于残差训练而产生 stacking 泄漏。

validation-only 选择：先审计“只增大 V19 budget”，最优约为 `0.375`，但 CI 相对 V19 几乎不变，故未建立单纯 budget 版本。随后只用 validation 比较 raw/scaled PCA 与多个宽度；raw PCA-64 显示互补后，正式搜索空间固定为 minimum leaf `2/3/5/7/10/15/20/30/40`、max features `0.5/0.7/0.85/1.0`、budget `0.000–1.000` 的 `0.025` 网格，OOF/final trees 为 `300/1200`，seed 为 `20260902`。PCA 直接拟合未预先标准化的 1,024 维 pair latent 并 whiten；V12/V15/V16/V18 四个 prediction 标量单独标准化后拼接。晋级 gate 在 test 前固定为相对 V19 Validation CI 至少 `+0.0005`、Spearman 不下降、RMSE 增量不超过 `0.003`、至少 `4/5` folds 正增益且 budget 非零。

验证与隔离：V20 专属 CPU 测试 `6/6`、V18–V20 合并回归 `30/30` 通过；Python compile、CLI help 和 CPU Bash `bash -n` 均通过。新增测试覆盖固定搜索空间、raw PCA 与标量拼接、非连续 fold ID 稳定映射、相对 V19 benchmark 的全部 gate、validation 失败不读取 test、selection/final 两阶段，以及错误 validation SHA 在 test 前拒绝。正式 selection 阶段只读取 V16/V19 validation 归档，selection 目录仅含 `result.json` 与 `valid_predictions.npz`，没有 test 字段或产物。final 阶段使用相同 seed 和完整网格重算 validation，并核对版本、三份 valid SHA、搜索空间、PCA、树参数、budget、五项指标和 gate 后，才读取已有 test 归档做一次冻结报告。

结果怎么样：正式 selection 选择 raw PCA-64、minimum leaf `2`、max features `1.0`、budget `0.375`。Validation CI `0.737938`、RMSE `1.172255`、MAE `0.914334`、Pearson `0.686389`、Spearman `0.659726`；相对 V19，CI `+0.001293`、RMSE 改善 `0.006379`、MAE 改善 `0.004271`、Pearson `+0.004225`、Spearman `+0.003460`。五个 fold CI gain 为 `+0.003093/+0.001881/+0.000937/-0.000579/+0.001332`，达到 `4/5` 正增益并通过全部预设 gate。

正式 Test 状态为 `passed`：CI `0.736092`、RMSE `1.297841`、MAE `1.002948`、Pearson `0.674478`、Spearman `0.650757`。相对 V19，Test CI `+0.001589`、RMSE 改善 `0.003072`、MAE 改善 `0.003014`、Pearson `+0.002886`、Spearman `+0.003657`，五项指标全部改善；相对 V16，Test CI 累计提升 `0.004725`。因此 V20 成为当前 LP-PDBBind 正式最佳。由于 V20 只是对已有 contact-aware pair latent 的 validation-selected 稳定性提炼，它没有削减论文要求的蛋白 1D、配体 2D、蛋白与配体 3D 经模型交互并输出 affinity 的主体架构。

完整性复算：valid/test key 分别为 `2,340/4,751` 且全部唯一，独立重算五项指标与 `result.json` 完全一致；selection/final 的搜索空间和 selected residual 完全一致，valid prediction 最大差 `1.78e-15`。selection result、final result、model、valid prediction、test prediction SHA256 分别为 `832d64ed1f3d90f6c4337f1ceae070d660e0cf81fcbacf68f11f2728b1c7c7bc`、`5627cd72e88e56fc03340e082bffe823fe3e8e938b65d88f461e67a47a9d1826`、`c2caa511084631b0e658ceb43d6d332c5bda63b1818cefe26a0dd00c0a2b8085`、`df4f1986f678a8a09a7adea0faf05ab909806c5eecc0167884a55d7d6d4fdc1f` 和 `50fec979b7ee8b5530fb78c577c76845637f9d0c5a647d7c9f8ef260fa4b2fce`。正式产物位于 `Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20/msa_interface_pair_latent_stability_fusion_v20_full_20260902_140000_validation_selection/` 与 `Model/LP_PDBBind_MSAInterfacePairLatentStabilityFusion_v20/msa_interface_pair_latent_stability_fusion_v20_full_20260902_140000/`；test 只用于 validation seal 通过后的该次冻结报告，没有参与任何配置选择。

## 5.26 Davis/KIBA Sequence+SMILES Binary Probe v1（正式 CPU 评估完成）

目的与边界：本实验用于回答“仅给蛋白序列和小分子 SMILES 时，当前工程在 Davis/KIBA 二分类划分上能达到什么推理性能”。V20 依赖 LP-PDBBind 的 3D interface/pair latent，不能在没有 3D 输入时直接迁移，因此新建完全独立的 `dta/sequence_smiles_binary_probe_v1/`、`experiments/davis_kiba/sequence_smiles_binary_probe_v1/` 和 `Model/DavisKIBA_SequenceSmilesBinaryProbe_v1/`；没有修改或覆盖 V1–V20。该结果是 Davis/KIBA train 上重新拟合的 sequence+SMILES 性能探针，不是 V20 的跨数据集 zero-shot 结果。

数据与二分类定义：直接解码 `data/DAVIS/` 和 `data/KIBA/` 的 Fairseq mmap 数据。Davis 固定 split 为 train/valid/test `21,039/3,006/6,011`，正类定义为 `pKd >= 7.522878745280337`（Kd 不高于 30 nM），正类数为 `1,056/161/310`；KIBA split 为 `82,778/11,825/23,651`，正类定义为 `KIBA score >= 12.1`，正类数为 `17,388/2,519/4,921`。Davis test 正类率只有 `5.157%`，因此 ACC 会明显受类别不平衡影响，模型比较应优先同时查看 MCC、ROC-AUC 和 PR-AUC。

方法：蛋白分支使用字符 1–3-mer hashing、TruncatedSVD 和氨基酸组成描述符；小分子分支使用 RDKit Morgan fingerprint、分子描述符和 TruncatedSVD；二者加入低维交互特征。由于两个官方随机 test 的药物和蛋白实体均已在 train 出现，另加入严格只由 train 标签形成、训练样本使用 leave-one-out 的 drug/protein/exact-pair 统计，以及 train-only bias-SVD 协同信号。HistGradientBoosting regression/classification、统计 head 和 rank-8/16/32 bias-SVD 的组合只按 validation PR-AUC（其次 ROC-AUC）选择；MCC/F1 decision threshold 也只由 validation 确定。`select` 阶段只读取 train/valid 并封存 bundle 与输入 SHA256，`test` 阶段先核对 seal 和 SHA256，之后才读取 test。原始 test 之外同时报告排除所有 train 已见 exact pair 的 novel-pair test。

Validation-only 选择结果如下，指标列与后续论文表保持一致：

| Dataset | ACC | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Davis validation | 0.9617 | 0.6742 | 0.5528 | 0.6075 | 0.5908 | 0.9464 | 0.6245 |
| KIBA validation | 0.8641 | 0.6883 | 0.6618 | 0.6748 | 0.5891 | 0.8870 | 0.7400 |

Davis 选择 `0.75 * bias-SVD rank-8 + 0.25 * sequence/SMILES regressor`，validation decision threshold 为 `0.0980108875`。KIBA 选择 `0.75 * bias-SVD rank-16 + 0.25 * sequence/SMILES classifier`，validation decision threshold 为 `0.4180915025`。

封存后正式 test 结果：

| Dataset | ACC | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Davis test | 0.9567 | 0.5850 | 0.5548 | 0.5695 | 0.5470 | 0.9465 | 0.5983 |
| KIBA test | 0.8651 | 0.6890 | 0.6407 | 0.6640 | 0.5803 | 0.8850 | 0.7444 |

去除 train 中已见 exact pair 后的 test 结果：Davis 保留 `4,979/6,011` 条，KIBA 保留 `23,522/23,651` 条。

| Dataset | ACC | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Davis novel-pair test | 0.9669 | 0.5878 | 0.4555 | 0.5133 | 0.5007 | 0.9382 | 0.5480 |
| KIBA novel-pair test | 0.8648 | 0.6881 | 0.6395 | 0.6629 | 0.5791 | 0.8843 | 0.7428 |

结果分析：KIBA 原始与 novel-pair 几乎一致，因为 test 只有 129 条样本的 exact pair 在 train 出现；Davis 去重后 PR-AUC 从 `0.5983` 降至 `0.5480`，说明其原始随机 split 的重复 pair 会带来可见增益，但去重后的 ROC-AUC 仍为 `0.9382`。这些数值只能与采用相同 split、相同正类阈值的工作直接比较；若外部表格使用更高正类比例或不同标签构造，ACC、Precision、Recall、F1、MCC 和 PR-AUC 均不可横向直接比较。Davis test 在指标字段统一后使用同一个已封存 bundle、融合权重和 validation threshold 重新生成报告，没有重新训练或依据 test 改动任何选择。

验证与产物：Python compile、CLI help、Bash `bash -n` 均通过，新增解码、leave-one-out 统计、validation score/threshold 选择测试为 `3/3` 通过。正式目录为 `Model/DavisKIBA_SequenceSmilesBinaryProbe_v1/sequence_smiles_binary_probe_v1_full_20260902_164000/`；两个数据集子目录各含 `selection.json`、`selection_bundle.joblib`、`valid_predictions.npz`、`test_predictions.npz` 和 `result.json`。该 CPU probe 不调用 GPU。

2026-09-02 追加反推协议诊断：用户提供的 Davis 对照表中 DeepDTA 的 ACC/Precision/Recall 为 `0.8569/0.7429/0.7526`，由三者可反推出该 test 的正类率约为 `28.177%`，与上述 30 nM 协议的 `5.157%` 明显不同。为判断差距是否主要来自协议，建立独立诊断输出 `Model/DavisKIBA_SequenceSmilesBinaryProtocolDiagnostic_v1/inferred_prevalence_full_20260902_180000/DAVIS/`。本诊断不读取本地 test 来确定标签：只在 train 标签上取目标正类率对应的分位数，得到固定 cutoff `affinity >= 5.292429823902063`，实际 train/valid/test 正类率为 `28.195%/27.112%/27.766%`。模型与融合仍只按 validation PR-AUC/ROC-AUC 选择，decision threshold 改为只在 validation 最大化 F1；选中 `0.50 * bias-SVD rank-16 + 0.50 * bias-SVD rank-8`，validation F1 threshold 为 `0.7184902371`。

完整重训后的 Validation 为 ACC `0.8659`、Precision `0.7537`、Recall `0.7509`、F1 `0.7523`、MCC `0.6604`、ROC-AUC `0.9102`、PR-AUC `0.8303`。selection seal 通过后正式 Test 为 ACC `0.8548`、Precision `0.7532`、Recall `0.7094`、F1 `0.7306`、MCC `0.6319`、ROC-AUC `0.9031`、PR-AUC `0.8183`。相对用户图片中的 “our” 行 `0.8643/0.7741/0.7318/0.7523/0.6595/0.9052/0.7994`，差值依次为 `-0.0095/-0.0209/-0.0224/-0.0217/-0.0276/-0.0021/+0.0189`。因此对齐近似正类率后，ROC-AUC 基本持平、PR-AUC 更高，其余 threshold-dependent 指标仅低约 `0.01–0.03`，此前看似巨大的差距主要由二分类协议不一致造成。

去除 train 已见 exact pair 后，诊断 Test 保留 `4,979` 条：ACC `0.8542`、Precision `0.7349`、Recall `0.6750`、F1 `0.7037`、MCC `0.6082`、ROC-AUC `0.8972`、PR-AUC `0.7923`。该实验仍只能标记为 post-hoc protocol diagnostic：正类率来自对图片四舍五入指标的反推，并不等价于获得了原论文真实 cutoff、负采样与 split 规则；不得作为论文正式复现结果。可复现入口为 `experiments/davis_kiba/sequence_smiles_binary_probe_v1/run_inferred_davis_cpu.sh`。

## 5.27 Optional-Structure Multitask v21（正式运行完成，未晋级）

目标与架构：V21 是与 V1–V20 完全隔离的新路线，代码位于 `dta/optional_structure_multitask_v21/`，实验入口位于 `experiments/multidataset/optional_structure_v21/`，模型输出保留在 `Model/OptionalStructureMultitask_v21/`。V20 正式产物不修改，仍作为 LP-PDBBind benchmark/fallback。V21 以完整蛋白序列 ESM2-650M residue embedding 和无坐标 SMILES atom/bond graph 经双向 cross-attention 产生 dataset-specific base affinity；LP-PDBBind 有结构时，再用 pocket residue、ligand atom 与 residue–ligand distance 的刚体不变 3D pair encoder，经硬 `has_3d` mask、learned gate 和零初始化 delta head 修正。无结构样本不访问结构字段，输出严格等于 base。LP、Davis、KIBA 共享 1D/2D trunk，但使用独立 affinity head 和 train-only label normalization。

训练与选择隔离：全部样本监督 base SmoothL1/MSE 和 dataset-local rank；有结构样本额外监督 full affinity、balanced contact、16-bin distogram 和 full-to-base distillation。LP 采用 `0.30` structure dropout；三数据集 homogeneous batch 采样比例为 `0.40/0.25/0.35`。raw/EMA 每轮只按 LP validation CI 选择，其次比较 RMSE 和 macro CI；选定后再完整报告三套 validation。`run_training.py` 不存在 test manifest 参数，selection seal 封存 train/valid manifest、checkpoint 与 prediction SHA256；独立 `run_test.py` 先验 seal 后才读取 test，并报告 LP full/base 五项指标及 Davis/KIBA 回归与固定 cutoff 二分类指标。

数据审计：active manifest 为 `data/OptionalStructureMultitask_v21/manifests_v2/`。以 sequence+canonical SMILES identity 将任一 validation/test exact pair 从训练移除，剔除 Davis `3,135`、KIBA `505` 条；最终 train 为 LP/Davis/KIBA `11,037/17,904/82,273`，validation 为 `2,340/3,006/11,825`，test 为 `4,751/6,011/23,651`，独立复扫 train-protected overlap 为 `0`。共有 `12,254` 条唯一蛋白，长度中位数 `319`、P95 `1,007`、最大 `4,735`，其中 `577` 条超过单窗口；提取器对全部残基做 1,022-residue overlap stitching，再池化到最多 512 tokens。train/valid/test manifest SHA256 为 `1d54d27fe7af694ffbc3a8641da835d87b1f85133a3bc42237c29dd6d67dc692`、`42fded13c2afb3681f41ef0056c8da4c477e7f559c786b677929dd0c376d3133`、`b5d8eba8f6837cd81cf82767225f62152b291f537c3b6ebdbf835cb6f0348c2d`。

2D cache：`data/OptionalStructureMultitask_v21/ligands_v2.lmdb/` 覆盖全部 `17,281` 个唯一 ligand，缺失 key 为 `0`，原子数最小/平均/最大 `4/33.64/268`。其中 `16,982` 个来自 RDKit SMILES；LP CSV 的 `299` 条空 SMILES 按 PDB structure key 使用唯一 ligand ID，只从 complex cache 的 atom/bond 字段生成无坐标 2D graph，不删除样本且 base encoder 不读取坐标；另有 2 条特殊无机 SMILES 使用 connectivity-only 回退。

验证状态：Python compile、5 个 CLI help 和 5 个 Bash 的 `bash -n` 均通过。focused CPU tests `7/7`，覆盖纯 sequence-only、混合 hard routing、零初始化回退、结构辅助 loss 反向、刚体变换不变性、长序列全覆盖、两类 2D graph schema，以及真实 LP structure LMDB + Davis/KIBA 混合前后向。`gernabind` 无 `fair-esm`，故默认采用已验证可导入的 Transformers ESM backend，同时保留 `fair_esm` 选项。本机尚无 ESM2-650M 权重缓存，首次提取会下载 `facebook/esm2_t33_650M_UR50D`，后续按 lineage 可恢复。尚未运行任何 V21 GPU 命令，无 V21 性能结果，V20 仍为正式最佳。

2026-09-04 运行保障补充：正式入口 `run_full_gpu7.sh` 新增 Bash 4.2/`set -u` 安全的 `RESUME` 开关。默认 `RESUME=0` 保持全新运行；仅 `RESUME=1/true/yes` 会向同一 `RUN_ID` 追加 `--resume`，非法值在 Python、CUDA 或 GPU 启动前拒绝。恢复时 `run_training.py` 校验 `last.pt` 中完整配置，并恢复 raw/EMA、optimizer、scheduler、AMP scaler、history、best 和 patience 状态，避免中断后重跑已完成 epoch。参数展开检查、Python compile、5 个 Bash `bash -n` 与 focused CPU tests `7/7` 再次通过；该改动没有启动 GPU，也没有产生新的 validation/test 指标。

2026-09-04 ESM 特征提取完成：物理 GPU 7 已生成 `data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb/`，覆盖 `12,254/12,254` 个唯一蛋白，embedding width `1,280`、FP16 存储、每条最多 `512` tokens；模型为 `facebook/esm2_t33_650M_UR50D`，所有长序列使用 1,022-residue overlap stitching 后自适应池化。metadata 中 train/valid/test manifest SHA256 与封存值完全一致。随后 CPU 只读审计确认 LMDB 恰有 `12,254` 个键、与三份 manifest 的 sequence ID 并集完全相等且无缺失/多余键，抽样 embedding 的 shape、dtype 与有限值检查通过。ESM 预处理已完成，但尚未执行 V21 smoke、正式训练或 test，因此仍无 V21 性能指标。

首次 GPU smoke `optional_structure_v21_smoke_20260904_151145` 在模型训练开始前停止，未生成 checkpoint 或 prediction，已完成的 ESM cache 不受影响。根因是 smoke 原配置按每数据集读取 train manifest 的前 8 条，而 Davis 前 8 条标签均为 `5.0`，标准差为零，`label_statistics` 按设计拒绝无效归一化。修复仅作用于独立 V21 smoke 脚本：将每数据集训练前缀增至 64 条，Davis 子集标准差为 `0.453659`，保持 6 个训练 step 不变。新增覆盖三数据集 smoke 标签方差的回归测试；V21 focused CPU tests 更新为 `8/8` 通过，Python compile 和 Bash `bash -n` 通过。失败目录为空但未删除；下一次 smoke 使用新 `RUN_ID`，直接复用完整 ESM/ligand cache。

第二次 GPU smoke `optional_structure_v21_smoke_20260904_162926` 进入首个 BF16 模型前向后停止，同样未生成 checkpoint 或 prediction。根因是 CUDA autocast 将 dataset-specific head 输出为 BF16，而 `_dataset_head` 按 latent 预分配的聚合张量为 FP32；`masked_scatter` 要求 source/self dtype 完全一致。V21 模型现将 base/delta head prediction、3D latent/quality、contact 与 distogram logits 在写入聚合张量前显式转换为目标 dtype，保持聚合和 loss 的 dtype 一致，并保留梯度。新增 CPU BF16 定向测试，直接复现 FP32 latent 与 BF16 head 的组合，同时覆盖 mixed dataset、可选 3D 路由和有限反向；V21 focused tests 更新为 `9/9` 通过，Python compile 通过。两次失败 smoke 目录均为空且未删除，ESM/ligand cache 继续直接复用；仍无 V21 validation/test 性能结果。

第三次 GPU smoke `optional_structure_v21_smoke_20260904_165650` 完整成功，证明修复后的 CUDA BF16 前后向、三数据集采样、可选 3D 辅助 loss、raw/EMA 选择和 validation seal 全链路可运行。smoke 使用 train/valid `192/24` 条（每数据集 `64/8`）、6 个训练 batch、1 epoch，选中 epoch 0 EMA；总 loss `1.244505`，base/full affinity、rank、contact、distogram 与 distillation 分量均为有限值。小前缀 Validation CI 为 LP-PDBBind `0.892857`、Davis `0.923077`、KIBA `0.407407`；LP RMSE `2.783226`、MAE `2.428889`、Pearson `0.876997`、Spearman `0.904762`。这些极小样本指标仅用于功能验证，尤其 KIBA binary 仅 8 条，不用于模型选择或性能判断。产物审计确认 checkpoint/prediction SHA 与 selection seal 一致，24 个 sample key 唯一且每数据集各 8 条，预测全部有限，`test_manifest_read=false`；checkpoint SHA256 为 `e8d776b7af3750209bba1b693789bf9fdf7e37da1abdc7745a8f252f1fe59e9f`，valid prediction SHA256 为 `3045ec1f35d605ac6599a5649472d0a5cd15a9ebbbc94107fa6b060c9e55324a`。V21 已满足启动正式全量训练的条件，但尚无正式 validation/test 指标，V20 仍为正式最佳。

2026-09-05 正式 train/validation 运行 `optional_structure_v21_full_20260905_143425` 完成。训练集 `111,214`、validation `17,171`，在 epoch 4 选择 EMA，随后 12 轮无 LP CI 新高并于 epoch 16 早停，共 17 epochs；best checkpoint SHA256 为 `edd03ec0105035905c9b4fa4973b21502ce239418b2dd63ed9989b54f23ed882`，valid prediction SHA256 为 `60e99b78051f12e4b6db0581e82da218f444aed76826fb506c5cdd73ca44e9d2`，test 尚未读取。LP-PDBBind base Validation 为 CI `0.653453`、RMSE `1.495641`、MAE `1.179866`、Pearson `0.436986`、Spearman `0.445640`；加入 3D 后为 CI `0.684928`、RMSE `1.382387`、MAE `1.076283`、Pearson `0.549295`、Spearman `0.527153`，即五项均明显改善，但相对 V20 full validation 仍分别落后 CI `0.053010`、RMSE `0.210133`、MAE `0.161949`、Pearson `0.137094`、Spearman `0.132573`，不能晋升为 LP 最佳。Davis/KIBA 无结构，因此 full 与 base 严格相等；Validation CI 分别为 `0.683253/0.639014`，ROC-AUC 为 `0.763785/0.602310`，PR-AUC 为 `0.163970/0.279547`。两者预测范围严重压缩，固定 affinity cutoff 下均未预测正类，F1/MCC 为 0。

路由和互补性诊断仅使用 validation：Davis/KIBA 的 full/base 最大绝对差均为 `0` 且 gate 全为 `0`；LP gate 在 BF16 中饱和为 `1`，3D delta 最大绝对修正 `2.273444`。训练 loss 从 `1.166720` 下降到约 `0.72`，但 LP validation 在 epoch 4 后回落，说明主要瓶颈是共享多任务 trunk 的欠拟合/任务冲突与后期 LP 泛化下降，不是 3D 分支完全无效。冻结 V20/V21 raw prediction validation-only 网格的最优 V21 权重仅 `0.08`，CI `0.738102`，相对 V20 只增加 `0.000164`；标准化版本 CI `0.738076` 且只有 `3/5` folds 正增益，远不足以建立新 fusion。当前 V21 应作为首个可选结构多数据集基线保留，不替代 V20；下一步只允许对已经封存的 checkpoint 做一次 test 报告，不根据 test 修改权重或训练配置。

封存后正式 Test 已一次性完成，`test_used_for_selection=false`，共 `34,413` 条：LP-PDBBind/Davis/KIBA 为 `4,751/6,011/23,651`，sample key 全部唯一、预测全部有限，test prediction SHA256 为 `9f446f2cd597b85659189432b41bdbef6841fb53f4ca228c3407b8781f04909e`。LP base Test 为 CI `0.603024`、RMSE `1.688885`、MAE `1.336391`、Pearson `0.296263`、Spearman `0.306478`；加入 3D 后为 CI `0.647242`、RMSE `1.608049`、MAE `1.265907`、Pearson `0.442781`、Spearman `0.425780`。因此 3D 对 base 的 Test 改善为 CI `+0.044218`、RMSE `-0.080836`、MAE `-0.070484`、Pearson `+0.146518`、Spearman `+0.119302`，五项同向，确认 optional 3D branch 的信号真实泛化；但相对 V20，CI `-0.088850`、RMSE `+0.310208`、MAE `+0.262959`、Pearson `-0.231697`、Spearman `-0.224977`，正式状态为不晋级。

Davis Test 为 CI `0.670660`、RMSE `0.900675`、MAE `0.502249`、Pearson `0.358952`、Spearman `0.250365`、ROC-AUC `0.735555`、PR-AUC `0.140880`；KIBA Test 为 CI `0.643593`、RMSE `0.810552`、MAE `0.529886`、Pearson `0.308925`、Spearman `0.393048`、ROC-AUC `0.612814`、PR-AUC `0.286788`。两者固定 affinity cutoff 下仍无正类预测，F1/MCC 为 0。相较同 split 的 sequence+SMILES binary probe，即使参考其 novel-pair 结果，V21 的阈值无关 ROC/PR-AUC 也显著更低，故不能只归因于去重训练协议；核心失败是联合训练尚未学到足够强的 1D/2D base。V21 的论文结论应限定为“可选结构机制成立但首版多任务训练失败”，不得称为性能提升模型。后续架构与超参数只能依据已经观察到的 train/validation 动态设计，不利用本次 test 选择候选。


## 5.28 Optional-Structure Staged v22（正式运行完成，未晋级 LP 最佳）

目标与隔离：V22 针对 V21 “3D 分支有效、但共享多任务 1D/2D base 明显欠拟合”这一 validation 诊断建立，使用完全独立的 `dta/optional_structure_staged_v22/`、`experiments/multidataset/optional_structure_v22/` 与 `Model/OptionalStructureStaged_v22/`，不覆盖 V20/V21 代码或产物。论文主体仍是一套模型接收蛋白 1D、配体 2D，并在结构可用时加入蛋白/配体 3D 后输出 affinity；无结构样本只走 1D/2D base，硬路由保证 prediction 严格等于 base prediction。

Stage A：完全不读取结构字段，只训练 full-sequence ESM2 residue embedding、无坐标 ligand 2D graph、双向 cross-attention、共享 trunk 与 dataset-specific head。由 V21 正式 best checkpoint warm-start 共同的 base 参数，新加 LP/Davis/KIBA 三个零初始化 residual adapter；实测 V21 checkpoint 的 391 个 shape-compatible state groups 中，Stage A 按排除 3D 前缀规则加载 237 个 base groups。目标为 normalized SmoothL1 + 0.50 MSE + 0.25 dataset-local rank + 0.20 fixed-affinity-cutoff balanced BCE，后者只作用于 Davis 7.5228787453 与 KIBA 12.1 协议，旨在纠正 V21 的预测范围压缩。正式配置为 batch 4、gradient accumulation 4、BF16、30 epochs、每轮 7,000 batches、patience 8、LP/Davis/KIBA sampling 0.15/0.25/0.60。raw/EMA 只按 validation macro CI 选择，再依次比较 macro Spearman、macro Pearson 与 LP RMSE；Stage A CLI 不存在 test manifest 参数。

Stage B：从封存的 Stage A checkpoint 初始化，冻结 protein/ligand/cross shared trunk 和 Davis/KIBA adapter/head；只训练 invariant 3D encoder、FP32-logit `2 * sigmoid` gate、delta heads，并允许 LP adapter/head 以 structure LR 的 0.05 微调。gate 最后一层与 delta 最后一层均零初始化，所以初始 structure gate 严格为 1、delta 为 0，加入 3D 前后的初始 affinity 完全一致，避免 V21 BF16 gate 饱和问题。V21 仅作为 `structure_encoder.*` warm-start，实测可加载 130 个 state groups，不会污染封存 base。正式配置为 LR 1e-4、40 epochs、每轮 3,000 个 LP structured batches、patience 10；目标保留 affinity SmoothL1/MSE/rank、balanced contact、16-bin distogram、delta L2 与 gate-to-one regularization。正式 validation gate 要求 LP full 相对同一 checkpoint base 的 CI 至少 +0.005、Spearman 不下降、RMSE 增量不超过 0.01。

选择与 test 隔离：Stage A seal 封存 train/valid manifest、checkpoint、valid prediction 和 cache lineage；Stage B 在训练前核对 Stage A checkpoint 的 version/phase、model config、train/valid SHA256 与 protein/ligand cache metadata lineage。Stage B selection 继续封存 Stage A selection/checkpoint、Stage B checkpoint、valid prediction 与全部 validation gate 信息。独立 test CLI 先验证 validation_passed、所有 SHA256、checkpoint 内部 phase、train/valid manifest、protein/ligand cache 和 Stage A source lineage，之后才解析并读取 test manifest；已有 `result.json` 或 `test_predictions.npz` 时拒绝覆盖。test 不参与任何 checkpoint、EMA 或 gate 选择。

CPU 验证：V22 专属回归 13/13、V21+V22 合并回归 22/22 通过；覆盖无结构字段缺失时的硬回退、adapter 零初始化与按数据集路由、FP32 latent/BF16 head 聚合、balanced BCE 将正负预测推过固定 cutoff、Stage B 只允许 3D 与 LP 专属参数产生梯度、一次 optimizer step 后 Davis/KIBA prediction bitwise 不变、V21 base warm-start prediction 严格一致、structure-only warm-start 不修改 base、validation gate，以及 validation 失败或 checkpoint SHA 错误时 test 不可访问。全部 V22 Python compile、3 个 CLI help 与 4 个 Bash `bash -n` 通过。真实 cache 的 LP/Davis/KIBA 混合 CPU 前后向使用 `3 x 512 x 1280` protein、`3 x 32` ligand 与 `3 x 16` pocket 张量，loss 和 119 个 gradient tensors 全部有限。smoke train 前缀中 Davis 正/负为 1/127、KIBA 为 24/104，满足 BCE 两类审计。

纯 CPU 端到端微型链路也已成功串联 Stage A、Stage B 和 test seal：train/valid 分别为每数据集 128/8 条、每阶段仅 1 个 batch，test 每数据集 1 条；Stage B 使用放宽 gate，仅验证工件与读取顺序，输出位于 `/tmp/optional_structure_staged_v22_cpu_smoke_20260905/`，其中任何指标都不得用于模型性能判断。2026-09-05 GPU smoke `optional_structure_staged_v22_smoke_20260905_222249` 随后完整成功：物理 GPU 7 完成 Stage A BF16 前后向、Stage B 3D 辅助目标与冻结参数训练、raw/EMA validation 选择、两层 validation seal 和 test-only 推理。smoke 使用 train/valid `384/192` 条（每数据集 `128/64`）、每阶段 6 个 batches、1 epoch，test 为每数据集 8 条；Stage A 与 Stage B 均选择 epoch 0 EMA。Stage B 放宽 gate 下 full/base LP Validation CI 均为 `0.543272`，Spearman gain `+0.000412`、RMSE change `+0.000067`，仅用于功能验证。sequence/structure checkpoint SHA256 分别为 `26a0a0621ea4b27a87beac831e92574b534fdd07fa4ceda3b4ba337f92fb3d20` 与 `2a7ee7bce48329798036d7b995f07a375b15e724288bd63fccc94bc7fd08db18`；valid prediction SHA256 分别为 `406c75de1e9b5614016c9d52d46ac70f8d0f8da319a5ed676885b160164c25ad` 与 `ba49c42f47e860834da11adf95be781aeb93a7584c00edf6bccbd974539bdff4`，test prediction SHA256 为 `0225941da764536ff005f2429853eeb91256dc44bd191e8d82fb36d945103b78`。独立 CPU 审计确认 valid/test sample keys 为 `192/24` 且全部唯一，预测全部有限，所有工件 SHA 复算一致，`test_used_for_selection=false`。smoke 指标不得用于性能判断。2026-09-06 至 09-07 正式 Stage A `optional_structure_staged_v22_full_20260906_132450` 完成，训练/validation 为 `111,214/17,171` 条，完整运行 30 epochs；最优点位于最后的 epoch 29 raw，末轮 LR 已降至 `7.5e-6`，test manifest 未读取。Validation sequence-only 指标：LP-PDBBind CI `0.619422`、RMSE `1.678713`、MAE `1.326317`、Pearson `0.361117`、Spearman `0.349512`；Davis CI `0.822120`、RMSE `0.688529`、MAE `0.350980`、Pearson `0.667138`、Spearman `0.478320`、ROC-AUC `0.891831`、PR-AUC `0.500738`、F1 `0.481203`、MCC `0.468878`；KIBA CI `0.789814`、RMSE `0.586559`、MAE `0.363884`、Pearson `0.719193`、Spearman `0.720982`、ROC-AUC `0.833804`、PR-AUC `0.661068`、F1 `0.610199`、MCC `0.522299`。相对 V21 base，Davis/KIBA CI 分别提升约 `+0.138867/+0.150800`，且两者 F1/MCC 从 0 恢复为非零，说明 fixed-cutoff BCE、分阶段训练和 task adapter 已显著解决 sequence prediction range compression；Davis/KIBA prediction range 分别为 `4.980–8.981/9.716–14.869`。代价是 LP base CI `-0.034031`、RMSE `+0.183071`、Spearman `-0.096128`，因此 Stage B 必须依靠冻结的 3D residual 修复 LP，不能把 Stage A 单独视为 LP 晋级模型。三数据集 macro CI 从 V21 base 约 `0.658573` 提高至 `0.743785`。正式 prediction 共 17,171 条且 sample key 全部唯一、标签和预测全部有限；V21 base warm-start 实际加载 237 groups。selection/checkpoint/valid prediction SHA256 分别为 `ea6972af2c6ff1f3a30cdd0ff53f1b60017fc13bb0ea3641d518a1d0db243a1e`、`a202db6a91f0ffc6ac60d3c9e4ef06707e6000b93c5523058772f0f84656d8e6`、`81e5809f93c74f8607aeb0d7ba0d85f59d323ed89ef3e8f4fc30483dab98ea20`，复算一致。末段 macro CI 改善已变小、LP 约稳定在 0.62 且 raw/EMA 接近，故依据 validation 决定不额外延长 Stage A，进入既定 Stage B；2026-09-08 正式 Stage B 完成并通过预设 validation gate。训练在 epoch 23 raw 达到最佳，至 epoch 33 早停，共 34 epochs；V21 structure-only warm-start 加载 130 groups，test manifest 尚未读取。LP-PDBBind base Validation 为 CI `0.618640`、RMSE `1.684070`、MAE `1.331449`、Pearson `0.357701`、Spearman `0.347134`；加入 3D 后为 CI `0.663413`、RMSE `1.544957`、MAE `1.215387`、Pearson `0.493385`、Spearman `0.471897`。因此 3D 相对同 checkpoint base 的 CI gain `+0.044773`、RMSE change `-0.139113`、MAE change `-0.116063`、Pearson gain `+0.135683`、Spearman gain `+0.124763`，远超 CI 至少 +0.005、Spearman 不下降和 RMSE 增量不超过 0.01 的预设 gate。LP gate 范围 `0.531761–1.261370`、均值 `1.002059`，未发生 BF16 饱和；full/base 最大修正绝对值为 `3.689161`。Davis/KIBA 的 full-base 差为 0，且其 Stage B base prediction 与 Stage A prediction 最大差均严格为 0，证明冻结路径保持了 Stage A 的全部 sequence-only 提升。Stage B full LP 仍比 V21 full CI 低约 `0.021515`、比 V20 Validation CI 低约 `0.074525`，因此不能晋级为 LP 最佳，但 optional 3D 的增益在 validation 上再次成立。17,171 条 validation key 全部唯一且预测有限；checkpoint、valid prediction、selection SHA256 分别为 `edff376f25af7e789197bdcb7e4664c22446f0fb170c83cd01ad8aa0b1268d08`、`e1430f66590139039e2e7d680aca3ebc0495794ba8ce2d8937be023967e38666`、`3d1a819ac50415d81b53c8c0c0c2ce1c1b8be5b297c05e79e503d7e75bb594e4`，复算一致。正式 test seal 随后一次性解封并完成，`test_used_for_selection=false`，共 34,413 条：LP-PDBBind/Davis/KIBA 为 `4,751/6,011/23,651`，sample key 全部唯一，标签、full prediction 与 base prediction 全部有限。LP base Test 为 CI `0.617769`、RMSE `1.765341`、MAE `1.384919`、Pearson `0.344072`、Spearman `0.345456`；加入 3D 后为 CI `0.663464`、RMSE `1.630012`、MAE `1.278612`、Pearson `0.495324`、Spearman `0.471381`。3D 的 Test 增益为 CI `+0.045694`、RMSE `-0.135329`、MAE `-0.106307`、Pearson `+0.151252`、Spearman `+0.125925`，五项全部改善，并与 validation 的增益幅度高度一致，证明 optional 3D 信号真实泛化。相对 V21 full，V22 full Test CI `+0.016221`、Pearson `+0.052543`、Spearman `+0.045601`，但 RMSE `+0.021963`、MAE `+0.012705` 略差；相对 V20，CI `-0.072628`、RMSE `+0.332171`、MAE `+0.275664`、Pearson `-0.179154`、Spearman `-0.179376`，因此 V22 不晋级为 LP 最佳。

Davis Test 为 CI `0.832707`、RMSE `0.667753`、MAE `0.345257`、Pearson `0.692159`、Spearman `0.479864`、ACC `0.953751`、Precision `0.552743`、Recall `0.432343`、F1 `0.485185`、MCC `0.465171`、ROC-AUC `0.906882`、PR-AUC `0.495135`。KIBA Test 为 CI `0.785914`、RMSE `0.594837`、MAE `0.364332`、Pearson `0.712170`、Spearman `0.709449`、ACC `0.849605`、Precision `0.673009`、Recall `0.539118`、F1 `0.598669`、MCC `0.512245`、ROC-AUC `0.825771`、PR-AUC `0.665786`。Davis/KIBA full 与 base 严格相等，且 validation→test 指标稳定；三数据集 Test macro CI 为 `0.760695`，相对 V21 的约 `0.653832` 明显提高。独立 CPU 复算三数据集五项指标与 `result.json` 一致；selection/test prediction/result SHA256 分别为 `3d1a819ac50415d81b53c8c0c0c2ce1c1b8be5b297c05e79e503d7e75bb594e4`、`f16112c9ddd251497fee9fbb5f1134585ab26689a5ccf94a50c6c388ebb70888`、`bd9c8435e93fd21285d6c96adafab1ce9b872764f087c0b6f63a9f8171fc9218`。最终结论：V22 成功建立了同一模型的 sequence+2D 与 optional 3D 两种运行模式，并显著提升 Davis/KIBA 和跨任务 macro performance；但当前 LP-PDBBind 最佳仍为 V20。


## 5.29 Optional-Structure Collaborative Router v23（正式 validation 未晋级）

V23 新建独立的 dta/optional_structure_collaborative_router_v23/、experiments/multidataset/optional_structure_v23/ 和 Model/OptionalStructureCollaborativeRouter_v23/，不修改 V20/V22。LP-PDBBind 仅用 validation 选择 V20 structured 与 V22 optional-structure 权重；Davis/KIBA 使用剔除 evaluation exact pair 的 train-only entity bias/Bias-SVD，未见实体硬回退 V22。正式目录为 Model/OptionalStructureCollaborativeRouter_v23/optional_structure_collaborative_router_v23_full_20260910_125651/。

Validation 五项 CI/RMSE/MAE/Pearson/Spearman：LP 为 0.739125/1.169448/0.912057/0.688446/0.661846，选择 V20/V22 0.90/0.10；Davis 为 0.836694/0.645788/0.365610/0.700349/0.492313，选择 rank 8、collaborative weight 0.60；KIBA 为 0.825377/0.524892/0.337735/0.793056/0.795572，选择 rank 16、weight 0.80。LP 和 KIBA gate 通过；Davis affinity CI 增益 +0.014574 且 5/5 分片为正，但 binary 最优仍是 V22，PR-AUC 增益 0、0/5 分片为正。因此 status=validation_failed、test_inputs_read=false，没有读取 test。

## 5.30 Optional-Structure Sequence Router v24（正式 validation 未晋级）

V24 新建独立的 dta/optional_structure_sequence_router_v24/、experiments/multidataset/optional_structure_v24/ 和 Model/OptionalStructureSequenceRouter_v24/。新增 train-only sequence/SMILES 专家：蛋白字符 1--3-mer、配体 Morgan/SVD 与描述符、训练标签构造且训练行 leave-one-out 的 entity statistics，以及回归/分类候选；所有选择仍只使用 validation。正式目录为 Model/OptionalStructureSequenceRouter_v24/optional_structure_sequence_router_v24_full_20260910_135250/。

LP 五项仍为 0.739125/1.169448/0.912057/0.688446/0.661846。Davis 选择 0.30 * reg_all + 0.70 * V22，五项为 0.837733/0.665682/0.360933/0.677633/0.493864；KIBA 选择 0.60 * Bias-SVD(rank 16) + 0.40 * reg_all，五项为 0.827052/0.525579/0.337267/0.792766/0.797388。KIBA affinity CI/PR-AUC 增益 +0.037285/+0.074686，均为 5/5 正增益。Davis affinity CI 增益 +0.015613、5/5 为正；binary PR-AUC 0.519979，增益 +0.018114、4/5 为正，比预设 +0.02 门槛少 0.001886。故 status=validation_failed、test_inputs_read=false，没有读取 test。

## 5.31 Optional-Structure Calibrated Router v25（正式运行完成，当前综合最佳）

V25 新建独立的 dta/optional_structure_calibrated_router_v25/、experiments/multidataset/optional_structure_v25/ 与 Model/OptionalStructureCalibratedRouter_v25/，不覆盖 V23/V24。它保留 V24 affinity route，并为 Davis 增加 balanced ExtraTrees binary expert（500 trees、minimum leaf 5、max features 0.7、class_weight=balanced）和固定 0.05--0.95 blend 网格。输入语义仍是蛋白 1D + 配体 2D，结构可用时加入蛋白/配体 3D：LP 使用 V20/V22 structured route，无结构时可回退 V22 base；Davis/KIBA 使用 sequence/SMILES 与 train-only expert。V25 是 validation-selected router/ensemble，不是单一端到端 checkpoint。正式目录为 Model/OptionalStructureCalibratedRouter_v25/optional_structure_calibrated_router_v25_full_20260910_143300/。

Validation 五项：LP 为 0.739125/1.169448/0.912057/0.688446/0.661846；Davis 为 0.837733/0.665682/0.360933/0.677633/0.493864；KIBA 为 0.827052/0.525579/0.337267/0.792766/0.797388。LP 选择 V20/V22 0.90/0.10。Davis binary 选择 0.35 * balanced ExtraTrees + 0.65 * V22，PR-AUC 0.530834、ROC-AUC 0.908852、MCC 0.498801、F1 0.525994、阈值 0.1747288842；PR-AUC 增益 +0.028968、4/5 为正，affinity CI 增益 +0.015613、5/5 为正。KIBA binary 使用 0.75 * Bias-SVD(rank 16) + 0.25 * V22，PR-AUC 0.735754、ROC-AUC 0.881315、阈值 0.4762133944；CI/PR-AUC 增益 +0.037285/+0.074686，均为 5/5 正增益。三套 gate 全通过，selection 为 validation_selection_sealed，test_inputs_read=false。

封存后独立 CPU test 才读取 test，且未重选配置。Test 共 34,413 条，LP/Davis/KIBA 为 4,751/6,011/23,651。LP 五项为 0.736892/1.284445/0.992416/0.676699/0.652550；相对正式 V20 test，CI +0.000800、RMSE -0.013397、MAE -0.010531、Pearson +0.002221、Spearman +0.001793，五项全改善，成为当前 LP 最佳。

Davis Test 五项为 0.848047/0.638305/0.350525/0.707711/0.503501；binary 为 ACC 0.940942、Precision 0.444444、Recall 0.580645、F1 0.503497、MCC 0.477466、ROC-AUC 0.912252、PR-AUC 0.508332。相对 V22，CI +0.015340、RMSE -0.029448、Pearson +0.015552、Spearman +0.023638，但 MAE +0.005268；PR-AUC、ROC-AUC、F1、MCC、Recall 改善，ACC 与 Precision 下降。KIBA Test 五项为 0.824466/0.525257/0.334313/0.794901/0.790310；binary 为 ACC 0.864699、Precision 0.692205、Recall 0.629750、F1 0.659502、MCC 0.576315、ROC-AUC 0.878746、PR-AUC 0.741669。相对 V22，CI 约 +0.038552、RMSE -0.069580、MAE -0.030020、Pearson +0.082731、Spearman +0.080862，PR-AUC/ROC-AUC 约 +0.0759/+0.0530。

审计确认 test_used_for_selection=false、test_configuration_reselected=false、exact_pair_decontamination_uses_test_labels=false。Davis/KIBA 仅拟合训练集 17,904/82,273 条，train--test exact pair 交集均为 0；Davis 双实体覆盖 5,657/6,011，另 354 条回退，KIBA 为 23,651/23,651。独立 CPU 复算全部指标一致；sample ID 唯一且与 manifest 顺序/标签一致，预测与有效 score 有限，binary class 与封存阈值一致；V23--V25 回归 18/18 passed。selection、bundle、valid/test prediction、result SHA256 为 45232d2e4e73fc47b93d4cd8e407921acd8abeeb7623abb5af65f0880c308414、bfe18883f01752a8fc4a30ecd08ab547e2959fbe3e41f69c01952ef375f30ba7、6f7e24a854d5741efedc1732739fd2e19366c0ef3f17f924a3498a704a1e7fcb、25d7cb3c8f44fd3979d2955ecffa77d90a2cc2ea52e0d62c9baa9778254a02a2、8dcd6e935d8750fcc088baac6b5abbce2db7691207357745902c16aa46f4e94b。与外部论文比较仍须对齐 label 变换、split、去重和二分类阈值。

## 5.32 Optional-Structure Simplex Router v26（正式运行完成，当前综合最佳）

目标与隔离：V26 在 V25 之外新建 dta/optional_structure_simplex_router_v26/、experiments/multidataset/optional_structure_v26/ 和 Model/OptionalStructureSimplexRouter_v26/，不覆盖旧版本。设计完全来自 V25 validation 诊断：V25 只允许两路 blend，但 Davis 的 V22、sequence regressor 与两个 Bias-SVD rank，以及 KIBA 的 V22、regressor、Bias-SVD/classifier 仍有互补。V26 对每项任务预注册 4 个候选分量，用 0.05 步长枚举 1,771 个非负、和为 1 的稀疏凸组合。LP-PDBBind 冻结 V25 的 V20/V22 route。晋级门槛相对 V25 validation 固定为 CI 至少 +0.001、PR-AUC 至少 +0.003，并且两者至少 3/5 哈希分片正增益；test 不参与设计或选择。

正式目录为 Model/OptionalStructureSimplexRouter_v26/optional_structure_simplex_router_v26_20260910_161346/。Validation：LP 五项 CI/RMSE/MAE/Pearson/Spearman 与 V25 相同，为 0.739125/1.169448/0.912057/0.688446/0.661846。Davis affinity 选择 V22/reg_all/Bias-SVD rank8/rank16 权重 0.55/0.30/0.05/0.10，五项为 0.841490/0.656543/0.362763/0.687566/0.498845，相对 V25 CI +0.003758、5/5 分片正；binary 选择 V22/balanced ExtraTrees/reg_all/Bias-SVD rank8 权重 0.45/0.20/0.25/0.10，PR-AUC 0.538513、ROC-AUC 0.911373、F1 0.543750、MCC 0.518112，PR-AUC +0.007680、3/5 分片正。KIBA affinity 选择 V22/reg_all/Bias-SVD rank16 权重 0.15/0.30/0.55，五项为 0.828822/0.520582/0.332582/0.797402/0.800994，相对 V25 CI +0.001771、5/5 正；binary 选择 V22/Bias-SVD rank16/cls_all 权重 0.20/0.65/0.15，PR-AUC 0.739072、ROC-AUC 0.884793、F1 0.661010、MCC 0.584338，PR-AUC +0.003318、5/5 正。三套 gate 全通过，selection 为 validation_selection_sealed、test_inputs_read=false。

validation 封存后，test-only CPU CLI 才读取 test，并仅按固定组合重拟合 train-only expert；V25 test 仅作为报告锚点，未用于重选。LP Test 与 V25 bitwise 同路线，五项为 0.736892/1.284445/0.992416/0.676699/0.652550。Davis Test 五项为 0.853036/0.629748/0.352016/0.716937/0.511165；相对 V25，CI +0.004990、RMSE -0.008557、Pearson +0.009226、Spearman +0.007664，但 MAE +0.001490。Davis binary 为 ACC 0.944602、Precision 0.468665、Recall 0.554839、F1 0.508124、MCC 0.480909、ROC-AUC 0.915142、PR-AUC 0.514434；相对 V25 PR-AUC/ROC-AUC 为 +0.006103/+0.002890。

KIBA Test 五项为 0.825959/0.521878/0.329735/0.797914/0.792905；相对 V25，CI +0.001493、RMSE -0.003379、MAE -0.004577、Pearson +0.003013、Spearman +0.002594，五项全部改善。KIBA binary 为 ACC 0.874170、Precision 0.749040、Recall 0.594391、F1 0.662814、MCC 0.592652、ROC-AUC 0.883197、PR-AUC 0.747483；相对 V25 PR-AUC/ROC-AUC 为 +0.005814/+0.004451。V26 因此在不损失 LP 的情况下同时提高 Davis 和 KIBA，成为新的综合锚点。

完整性审计：test_used_for_selection=false、test_configuration_reselected=false、exact_pair_decontamination_uses_test_labels=false；test 为 34,413 条，sample ID 全部唯一且与 manifest 顺序/标签一致，预测和有效 binary score 全部有限。独立 CPU 复算三个数据集五项 affinity 与 Davis/KIBA 七项 binary 指标均与 result.json 精确一致，保存类别也与封存阈值一致。V23--V26 combined tests 为 23/23 passed，Python compile 和两个 Bash 语法检查通过。selection、bundle、valid prediction、test prediction、result SHA256 分别为 e73db384fd3b1c52473a4cefc08d285b3735bf508a848dc60220c10d12b7f9d0、f0dde15cbca07323afd30431c4601ffb1e763c0a3835f4362fe2092680bcc5cc、eca88b096f98af527cf6f61badddd12febae12e3ec08d03bd10a5dd4e5a4d71a、6b26fefdd690756525cfa4ee45765f6a46cd648ce63d032dd42687435c0d33d5、882177f0f5c9d2f32dba494f62bf75f1784b22f261b189e3f555c17a09bf3777。V26 仍是 optional-3D neural route 加 train-only experts 的 router/ensemble，不应描述为单一端到端 checkpoint。

## 5.33 Optional-Structure Privileged Pair Co-Training v27（正式 validation 未晋级）

目标与隔离：V27 在 V26 之外新建 `dta/optional_structure_privileged_pair_v27/`、`experiments/multidataset/optional_structure_v27/` 和 `Model/OptionalStructurePrivilegedPair_v27/`，不修改或覆盖 V22/V26 的代码、checkpoint 与正式预测。它不再增加最终标量专家，而是实现一个真正统一的 optional-3D 神经架构：完整序列的 ESM residue token（蛋白 1D）与 ligand atom/bond graph（配体 2D）先构造 sequence–ligand pair token；结构存在时，pocket–ligand geometry pair token 通过 cross-attention 更新同一个 pair latent，再由共享 dataset head 输出 affinity；结构不存在时同一 checkpoint 硬回退到 base path，完全不访问 3D 字段。V22 正式结构 checkpoint 作为 warm-start，V26 仅作为冻结的 validation/test 安全 anchor。

架构与训练：正式模型宽度为 `384`，8 heads，protein/ligand/cross layers 为 `2/4/3`，latent/pair width 为 `768/192`；sequence 与 structure 各保留 top-128 pair tokens。sequence pair selector 使用 hard top-K sparse Transformer pooling 与可微 masked soft-attention pooling 各 0.5，避免 contact selector 仅经过离散 top-K 而无梯度。可用 3D 时进行 25% masked cross-distance reconstruction，visibility embedding 明确区分可见/遮蔽 pair；geometry adapter 在中层 latent 注入结构，而非额外标量 structure expert。base/full 同时接受 SmoothL1、MSE 与同靶点 rank 监督，并加入反对称 relative head、relative consistency、full→base affinity/pair/geometry-quality 蒸馏、balanced contact、16-bin masked distogram、delta L2 和 gate regularization。训练 sampler 以 LP/Davis/KIBA `0.30/0.25/0.45` 平衡采样，70% batch 主动构造同靶点不同配体对，结构 dropout 为 0.15。正式脚本配置 batch 4、gradient accumulation 4、BF16、最多 45 epochs、每轮 7,000 batches、patience 10；40GB GPU 首先必须经过独立 smoke。

初始化审计：正式 V22 checkpoint `Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt` 实际兼容加载 391 组参数，SHA256 为 `edff376f25af7e789197bdcb7e4664c22446f0fb170c83cd01ad8aa0b1268d08`。V27 正式模型共 39,988,445 个参数；new-pair/structure/pretrained-shared 三个优化器组分别为 2,950,479/11,120,353/25,917,613 个参数，均非空且互不重叠。用正式 validation 的 LP/Davis/KIBA 各一条真实样本比较 V22 与 warm-start 后 V27 的 sequence-only prediction，最大绝对差为 0 且 bitwise equal；新增 base pair adapter 与 geometry adapter 的末层零初始化，因此新路径在开始训练前不会破坏 V22 prediction。

Validation-only 选择与 test 隔离：checkpoint 只按 validation 的 `0.50 * LP CI + 0.25 * Davis CI + 0.25 * KIBA CI` 选择 raw/EMA。随后对 LP/Davis/KIBA 分别在 `0.00–1.00`、步长 0.05 的网格中选择 V26/V27 affinity 权重；正式晋级要求 CI 至少 `+0.0005`、Spearman 不下降、RMSE 增量不超过 `0.003`，且至少 3/5 hash folds 的 CI 为正。LP full 还必须相对同 checkpoint base 达到 CI `+0.005`、Spearman 不下降、RMSE 增量不超过 `0.01`。任一数据集未通过 anchor gate 即将其 V27 有效权重精确设为 0；只有 LP structure gate 通过且至少一个数据集晋级，才写 `validation_passed` 并允许独立 test CLI 读取 test。test CLI 先验证 checkpoint、valid/selected-valid prediction、V26 anchor、train/valid manifest 与两个 LMDB lineage 的 SHA256，之后才解析 test manifest；固定 validation 权重，不允许 test 重选。

二分类路线保持冻结：V27 当前只选择 affinity route；Davis/KIBA 的 V26 `binary_scores` 与 `binary_predictions` 在 validation/test prediction archive 中逐元素原样复制，并独立报告 frozen V26 binary metrics，不能用 V27 affinity 阈值结果冒充最终二分类路线。全量 V26 test archive 复算得到的 Davis/KIBA ACC、Precision、Recall、F1、MCC、ROC-AUC、PR-AUC 与 V26 `result.json` 全部差值严格为 0。对齐审计还修复了一处正式阻断：V26 archive 保留 manifest float64 label，而 V27 collator 导出 float32 label，真实最大往返差为 `4.7003e-7`；label 容差由不合理的 `1e-8` 收紧语义调整为仅容纳 float32 往返的 `1e-6`，sample ID 与 dataset index 仍严格相等，明显错误标签仍拒绝。

CPU 验证：V27 focused tests 为 `14/14`，加入 V22 回归后为 `27/27`；全部 V27 Python 文件在正式 `gernabind` Python 3.8 下 compile，通过两个 CLI help 与三个 Bash 的 `bash -n`。测试覆盖无结构硬回退、中层 geometry 注入、零初始化 prediction safety、反对称 relative head、masked contact/distogram 与蒸馏有限反向、sequence selector/structure distance head 梯度、刚体变换不变性、同靶点 sampler、validation-only blend、失败精确回退 V26、float32 label lineage、冻结 binary route，以及 validation 失败时 test 不可访问。真实 cache probe 使用 2 条 LP structured 与 2 条 Davis sequence-only 样本，输入蛋白 shape 为 `(4,512,1280)`、ligand 为 `(4,32,8)`、pocket 为 `(4,16,640)`；遮蔽 277 个 geometry pairs，形成 1 个同靶点 relative pair，loss `2.821419`，287 组 gradient tensors 全部有限。

另完成一次纯 CPU 微型工件链路：train/valid 为每数据集 `128/12`，只训练 1 batch/1 epoch；训练入口明确记录不存在 test manifest 参数，validation seal 完成后独立 test CLI 才打印 `validation seal verified; reading test inputs now` 并读取每数据集 2 条 test。为强制覆盖成功解封分支，该链路使用人为劣化的临时 validation anchor 与放宽 gate，所有指标仅用于功能验证，不能用于性能判断；已验证 selected-valid/test 中冻结 binary score 与 decision 均和输入 anchor 逐元素完全相同。临时工件位于 `/tmp/optional_structure_privileged_pair_v27_cpu_e2e_q3Cjyn/`，不属于正式结果。

2026-09-11 正式等宽 GPU smoke `optional_structure_privileged_pair_v27_smoke_20260911_162802` 已完整完成，生成 best/last checkpoint、validation selection 和 test report。smoke 使用每数据集 train/valid `128/64`、6 个 batches、1 epoch，best 为 epoch 0 EMA。LP full/base validation CI 为 `0.552776/0.552526`，差值 `+0.000250`；放宽 gate 下 Davis/KIBA 分别选择 V27 affinity 权重 `0.80/0.15`，LP 权重为 `0.0`。该 64 条前缀选择与每数据集 8 条 test 仅证明正式宽度、batch 4、BF16、两阶段 seal 和 40GB 显存可运行，不能用于性能判断。

随后启动正式全量运行 `optional_structure_privileged_pair_v27_full_20260911_164124`。服务器外部中断前完整完成 epoch 0 的 7,000 个 training batches 与全量 validation，best 为 epoch 0 EMA，`stale=0`。LP-PDBBind full validation 为 CI `0.618540`、RMSE `1.688721`、MAE `1.334909`、Pearson `0.357295`、Spearman `0.346842`；同 checkpoint base 为 CI `0.617233`、RMSE `1.699790`、MAE `1.345351`、Pearson `0.354267`、Spearman `0.343113`。因此首轮 3D 增益为 CI `+0.001307`、RMSE `-0.011069`、MAE `-0.010443`、Pearson `+0.003028`、Spearman `+0.003729`：方向全部正确，但 CI 尚未达到正式 `+0.005` structure gate。Davis validation 为 CI `0.820006`、RMSE `0.687405`、MAE `0.349023`、Pearson `0.669857`、Spearman `0.475481`；KIBA 为 CI `0.789803`、RMSE `0.586158`、MAE `0.363644`、Pearson `0.719757`、Spearman `0.721568`。这些只是 epoch 0 中间结果，不得进行 test 或宣称 V27 晋级。

中断审计：正式目录只包含原子写入且已成功 CPU 实载的 `best.pt`（153MB）和 `last.pt`（611MB），没有 selection/test 文件；相关训练进程已不存在，也未发现模型 traceback 产物，符合服务器在 epoch 1 开始前后外部中断。通过原 `RUN_ID` 与 `RESUME=1` 可从 epoch 1 恢复 optimizer、scheduler、GradScaler、EMA、history 和 stale 状态，无需重跑 epoch 0。首轮总训练 loss 为 `7.044331`，其中未加权 pair distillation 为 `40.020409`，乘 0.15 后约占总 loss 85.2%，后续恢复时必须观察它是否快速下降以及 base CI 是否保持；单个 epoch 还不足以据此修改正式路线。V26 当前仍是综合最佳；只有 V27 正式 validation seal 通过后才允许运行 test。


2026-09-13 至 2026-09-14，正式运行通过原 `RUN_ID` 和 `RESUME=1` 从 epoch 1 成功恢复，并在 epoch 12 正常达到 patience 10 后早停；不是再次发生服务器中断。完整 history 共 13 个 epoch，validation 最优为 epoch 2 EMA。正式输出为 `Model/OptionalStructurePrivilegedPair_v27/optional_structure_privileged_pair_v27_full_20260911_164124/`，best/last checkpoint、valid/selected-valid prediction 与 selection seal 均已生成。

V27 standalone validation 五项 CI/RMSE/MAE/Pearson/Spearman：LP-PDBBind 为 `0.636069/1.627647/1.277605/0.400905/0.397358`，Davis 为 `0.814805/0.693352/0.360513/0.662954/0.464376`，KIBA 为 `0.780851/0.605163/0.375444/0.698003/0.704880`。LP 同 checkpoint base 为 `0.615764/1.727846/1.371115/0.348602/0.339900`，因此 3D geometry route 带来 CI `+0.020305`、RMSE `-0.100199`、Spearman `+0.057458`，structure gate 明确通过；说明可选 3D 中层注入本身有效。

但相对冻结 V26 anchor，三项 affinity gate 均未晋级。LP 搜索最优 V27 权重为 `0.0`；Davis/KIBA 都只选择 `0.05` 的 searched weight，CI 增益分别仅 `+0.00000462/+0.00003692`，均低于 `+0.0005` 门槛且仅 2/5 folds 为正。故三个 effective V27 weight 全部精确回退 `0.0`，`any_promoted=false`、`status=validation_failed`、`validation_selected=false`，selected validation metrics 等于 V26 anchor；test manifest 未读取，禁止运行 V27 test。V26 继续作为综合最佳。

训练趋势显示明显表示失稳：每 epoch 的 `sums` 在 fit 循环内确实重新初始化，因此日志不是累计显示错误。未加权 pair-distillation 从 epoch 0 的约 `40.02` 上升到 epoch 1/2 的 `453.96/957.24`，至 epoch 12 达 `4534.25`；按固定权重 0.15 后主导总 loss，同时 Davis/KIBA 与后期 LP validation 持续退化。这说明当前“移动 full-pair teacher + 无尺度约束的 latent SmoothL1 蒸馏”破坏了 warm-start 表示。若继续架构路线，应建立新版本并优先冻结 teacher、对 pair latent 做 LayerNorm/余弦蒸馏或显著降低并 warm-up 蒸馏权重，而不是恢复已正常早停的本次 V27。

正式 SHA256：best checkpoint `1e725e8313dcfb9d5f937a89b9daad3595a79e086cd10bfbb7b6a56c8da8dc04`，raw valid prediction `35e001f80e95804da9ea0a344f07b21071595d0a698afb5805c82e3556087fe9`，selected-valid prediction `f874a7d3d838ebf6d0e2d84b76d4c01ddfa2da0b88349453e8c36b02624aefdb`，V26 anchor valid prediction `eca88b096f98af527cf6f61badddd12febae12e3ec08d03bd10a5dd4e5a4d71a`.


## 5.34 Optional-Structure Stable Pair Co-Training v28（正式 validation 未晋级）

目标与隔离：V28 在 V27 正式 validation 未晋级后新建独立的 `dta/optional_structure_stable_pair_v28/`、`experiments/multidataset/optional_structure_v28/` 和 `Model/OptionalStructureStablePair_v28/`，不覆盖 V22/V26/V27。模型继续保持单一可选结构主体：蛋白完整序列 ESM residue token（1D）与 ligand atom/bond graph（2D）形成 base pair 表征；存在结构时再用 pocket--ligand geometry pair（蛋白/配体 3D）更新共享 latent 并输出 affinity；没有结构时硬回退到同一 checkpoint 的 1D+2D path。该版本不使用 MSA，V26 仍只作为冻结 validation/test 安全 anchor。

针对 V27 失稳的改动：V28 在载入正式 V22 warm checkpoint 后立即复制并永久冻结一个 sequence-only base teacher，teacher 始终以 `force_no_3d=True`、`eval()` 和 stop-gradient 前向；student base 增加权重 `0.50` 的固定 affinity anchor。前 2 epochs 冻结 warm-start pretrained 主体，第 3 epoch 解冻；主 LR 调至 `1.0e-4`，pretrained/structure LR scale 分别为 `0.05/0.35`。pair 蒸馏不再对未归一化 raw latent 使用移动 full-pair teacher，而改为 LayerNorm 后 cosine 与 normalized SmoothL1 各占 0.5，权重由 `0.15` 降至 `0.03`；冻结期 scale 为 0，解冻后 5 epochs 按 `0.2/0.4/0.6/0.8/1.0` warm-up。normalized pair loss 大于 `4.0` 时 fail-fast，并记录 pair norm、cosine、regression、effective weight。privileged affinity 蒸馏权重降为 `0.02`，quality/contact/distogram/rank 等监督保留。

选择协议未改变：checkpoint 仍只按 validation 选择，随后以冻结 V26 为 anchor 完成三数据集 validation gate；LP 还必须先通过同 checkpoint full 对 base 的 structure gate。任一数据集不晋级即将其 V28 effective weight 精确回退为 0，只有 validation seal 通过后独立 test CLI 才允许解析 test manifest。Davis/KIBA 的 V26 binary route 原样冻结，test 不参与调参或模型选择。

2026-09-14 CPU 验证完成：V28 专属测试 `18/18` 通过，V22 回归 `13/13` 通过；V28 Python 3.8 compile、两个 CLI help 与三个 Bash `bash -n` 均通过。真实 train cache CPU probe 读取 1 条 LP-PDBBind structured 与 1 条 Davis sequence-only 样本，V22 warm-start 兼容加载 391 组参数；双方 `eval()`、`force_no_3d=True` 时固定 teacher 与 student 初始化 prediction 一致。训练态 3D forward/backward 的 normalized pair loss 为 `0.121377`（cosine `0.086436`、regression `0.156318`），低于 fail-fast 阈值 4.0；base anchor distillation 为 `0.003835`，215 个 student gradient tensors 全部有限，teacher 无梯度，base pair adapter 和 structure distance head 均获得梯度。

2026-09-15 等宽 GPU smoke `optional_structure_stable_pair_v28_smoke_20260915_141930` 已完整生成 best/last checkpoint、validation seal 与 test report。smoke 使用三数据集 train/valid 各 `128/64`（合计 `384/192`），3 epochs、每轮 4 steps，best 为 epoch 0 raw。调度与预注册设计一致：epoch 0/1 的 `pretrained_frozen=true` 且 pair scale 为 0；epoch 2 解冻且 pair scale 为 `0.2`、effective pair weight 为 `0.006`。三轮未加权 normalized pair loss 为 `0.040903/0.034418/0.080447`，base anchor distillation 为 `0.016935/0.021175/0.028994`，均有限且没有复现 V27 的 latent loss 爆炸。未出现 OOM、non-finite 或 fail-fast。

smoke 的 gate 被有意放宽，只用于覆盖完整 GPU 链路：64 条/数据集 validation 前缀上 LP 的 full/base CI 为 `0.553027/0.552026`，结构 CI 增益 `+0.001001`；LP/Davis/KIBA 的有效 V28 融合权重为 `0/0.80/0.15`。validation seal 后独立读取每数据集 8 条 test，`test_used_for_selection=false`、`test_configuration_reselected=false`。这些前缀权重和 test 指标样本过小，不用于正式泛化判断，也不改变 V26 锚点。

2026-09-15 至 2026-09-16 正式全量运行 `optional_structure_stable_pair_v28_full_20260915_150049` 正常完成 17 个 epoch，并在 best epoch 6 raw 后达到 patience 10 而早停。正式 train/valid manifest 总计为 `111,214/17,171` 条，best checkpoint SHA256 为 `63e0ed16c87adff9b959b3b3a2c6cf6d3ff4910052e8a157d63bb6237e0ddc99`。V28 standalone validation 五项 CI/RMSE/MAE/Pearson/Spearman 为：LP-PDBBind `0.653876/1.584468/1.240590/0.468463/0.445402`，Davis `0.821808/0.685340/0.349716/0.669526/0.476573`，KIBA `0.791299/0.587652/0.360226/0.719829/0.723758`。LP 同 checkpoint base 为 `0.615855/1.715465/1.356121/0.349301/0.339506`，故 3D route 的 CI `+0.038021`、RMSE `-0.130997`、Spearman `+0.105896`，structure gate 明确通过。

相对 V27，V28 三个 standalone 数据集均改善，且 V27 的 pair-loss 爆炸已经被消除：未加权 normalized pair loss 从 epoch 0 的 `0.039609` 降到 best epoch 的 `0.004434`、最后 epoch 的 `0.002249`，有效 pair 权重最高仅 `0.03`。不过 pair latent norm 仍由约 `4.54` 增至 `62.39`，而 fixed base-anchor distillation 由 `0.0259` 升至 `0.0487`；这表明表示尺度仍在漂移，但没有再主导训练 loss。V28 的主要问题不是数值失稳，而是其独立 affinity 仍远弱于冻结 V26 router anchor。

正式 validation-only anchor selection 未晋级。LP 和 KIBA 最优 searched V28 权重均为 `0.0`，没有独立互补增益。Davis 的最优 searched weight 为 `0.20`，CI 从 anchor `0.841490` 到 `0.841803`，增益仅 `+0.000312`，虽有 `3/5` folds 为正且 RMSE 仅增加 `0.000283`，仍未达到预注册 CI 至少 `+0.0005` 的门槛。因此 LP/Davis/KIBA 的 effective V28 weight 均精确回退为 `0.0`，`any_promoted=false`、`status=validation_failed`、`validation_selected=false`。test manifest 未读取，`test_metrics=null`，不得运行或报告 V28 正式 test；V26 继续是当前综合正式最佳。raw validation prediction SHA256 为 `6dfb37d8e4613c6a3c276a3b3f85a3c2e8483333e526cc705c05f68ea85fcf03`。


## 5.35 Optional-Structure Sealed V22 Bounded Residual v29（正式 validation 未晋级）

目标与隔离：V29 在 V28 正式 validation 未晋级后新建独立的 `dta/optional_structure_sealed_residual_v29/`、`experiments/multidataset/optional_structure_v29/`、`Model/OptionalStructureSealedResidual_v29/` 与专属测试文件，不修改 V22/V26/V27/V28。V29 不使用 MSA；主体仍为蛋白 1D ESM、配体 2D atom/bond graph、可选 pocket--ligand 3D 到 affinity。无结构时严格输出 V22 base；有结构时以 V22 full 预测为起点，再加一个新训练的有界 3D residual：`prediction = V22_full + sigmoid(gate) * 0.75 * tanh(delta)`。residual head 最后一层零初始化，因此训练前有结构样本也严格等于 V22 full。

设计依据：V28 已消除 V27 的 loss 爆炸，却只达到 LP validation CI `0.653876`，低于 V22 full `0.663413` 和 V26 LP anchor `0.739125`。根因是 V28 只兼容加载 V22 的部分参数，新的 geometry route 从零开始，并且 moving pair teacher 仍导致 latent norm 漂移。V29 因此 strict-load 正式 V22 structure checkpoint `Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt`（SHA256 `edff376f25af7e789197bdcb7e4664c22446f0fb170c83cd01ad8aa0b1268d08`），将整个 V22 backbone 固定为 `eval()`、`requires_grad=false`、`no_grad`；只训练 932,554 个新增 residual 参数，取消 moving pair distillation。

训练与选择：训练 sampler 只抽 LP-PDBBind structured rows，并保留 70% 同 target 不同 ligand pair 来提供 rank 监督；监督包含 full affinity SmoothL1/MSE/rank、对冻结 V22 full prediction 的 anchor loss、residual L2 与 gate-center 正则。Davis/KIBA 不让 V29 模型参与 affinity 选择，仍完整冻结 V26 route 与 binary score/decision。正式 full validation（未截断且 evaluation batch size 与 V22 archive 同为 4）同时要求 V29 内嵌 V22 backbone prediction 与 V22 full valid archive 一致（容差 `1e-4`），再只在 LP 上网格选择 V26/V29 blend；limited smoke 则以独立 strict-load 的 V22 reference 在完全相同的 padded batch、BF16 autocast 下作 `1e-4` 硬审计，archive 差异仅记录为动态重分批诊断，不能放宽正式审计。晋级仍要求 CI 至少 `+0.0005`、Spearman 不下降、RMSE 增量不超过 `0.003`、至少 `3/5` hash folds 为正。test CLI 必须先验证 selection、V22 checkpoint/valid archive、V26 anchor、manifest/cache SHA；只有 validation seal 通过后才读取 test，并再次与 V22 test archive 做 backbone 等价性审计。

2026-09-16 首轮 GPU smoke `optional_structure_sealed_residual_v29_smoke_20260916_145718` 已完成训练与 validation prediction 写出，但在 validation seal 的 V22 equivalence audit 处中止：最大差异为 `0.01271248`，故没有生成 selection 或读取 test。审计确认 raw V29 backbone 与 V22 `409/409` 个 state tensor 逐位相同；问题仅在首轮选择的 EMA checkpoint：通用 EMA 即使面对 `requires_grad=false` 参数仍执行浮点乘加，导致 `387/409` 个 V22 tensor 产生最大 `2.861e-6` 的舍入漂移，并在 BF16 深层推理中放大。该 smoke 产物保留以供审计，但不得用于性能结论或恢复。

随后已在 V29 独立代码中修复：EMA 只更新新增 residual 分支，V22 backbone 的 parameter/buffer 完全跳过 EMA；恢复训练、最终 checkpoint 载入及 test 载入均再次 strict-load 同一 sealed V22 checkpoint，旧 EMA checkpoint 也会被重新封印。新增位级 EMA-backbone 不变性和 legacy checkpoint re-seal 回归测试后，CPU 测试为 `19/19`、V22 回归为 `13/13`；正式 V22 checkpoint CPU probe 得到 `409` 个 backbone tensor、EMA mismatch `0`。正式 V22 checkpoint strict load 成功，backbone 可训练参数为 0，新增 residual 参数为 932,554。真实 train cache probe 在两条 LP structured 样本上确认 V29 zero-init base/full prediction 与独立 V22 前向严格一致；蛋白/3D pocket shape 为 `(2,32,1280)/(2,8,640)`，loss `1.141321`，初始 mean absolute residual 为 0，residual gradient 有限且全部 V22 参数无梯度。

第二次 GPU smoke `optional_structure_sealed_residual_v29_smoke_fixema_20260916_153242` 已跑满 3 个训练 epoch，best 为 epoch 2 EMA，且其 best/last raw/last EMA 与 V22 的 `409/409` backbone tensor 均逐位一致；但在旧的 archive-only audit 处再次停止，最大预测差 `0.00726604`，未生成 selection/test。该差异不是冻结破坏或样本错位：192 条 truncated validation 中 DAVIS `64/64` 精确一致，KIBA 仅 6 条、LP 仅 25 条非零。最差 LP `1a0q` 在 full V22 archive 中与 KIBA 同 batch（protein/ligand/pocket padded shape `512/32/59`），而 smoke 中为 LP-only batch（`480/40/82`）；截断后的动态 padding 改变 BF16 Transformer/einsum 的数值归约路径。故 archive 不可作为 truncated smoke 的 `1e-4` 绝对参考。

现已新增同 batch 独立 V22 audit 和 archive applicability policy：full validation 仍严格比较 archive，`limit_valid` smoke 则只在相同 padded batch 比较并保持 `1e-4` 硬门。真实 cache CPU probe 在 smoke 的最差 batch `1a0q/1a28/1a4h/1a52` 上得到最大差异 `0`；新增两项测试后 V29 专属 CPU 测试为 `21/21`，V22 回归为 `13/13`，Python 3.8 compile、训练 CLI help 与三个 Bash `bash -n` 均通过。smoke 脚本新增 `RESUME=1` 透传；首轮 EMA 污染 smoke 仍不得恢复。

使用第二次 smoke 的原 `RUN_ID` 以 `RESUME=1` 恢复后，因 last 为 epoch `2` / maximum epoch `3`，没有重跑训练，只重封 checkpoint、重做 validation prediction、same-batch audit 与 selection。该 smoke 的独立 V22 audit 覆盖 `192` 条 validation（其中 `64` 条 structured），maximum/mean difference 均为 `0`，通过 `1e-4` 硬门；旧 archive 的 `0.00726604` 差异保留为明确的 `limited_validation_rebatches_bfloat16_dynamic_padding` 诊断、未参与 pass/fail。smoke selection 正常写出，且 `test_manifest_read=false`。在该 64 条/数据集前缀上，LP V29 standalone CI 为 `0.568784`，V22 backbone 同为 `0.568784`，structure 相对 base CI 增益 `+0.017009`、RMSE 改善 `-0.031497`；但 V26 anchor 的 LP CI 为 `0.694347`，validation 搜索的 V29 权重为 `0.0`，`any_promoted=false`，故 `status=validation_failed`、`validation_selected=false`。这是功能 smoke 的正确结果，不可用于判断正式泛化，test 仍严格禁止读取；下一步是全量 V29 validation。

2026-09-16 至 2026-09-17，正式全量运行 `optional_structure_sealed_residual_v29_full_20260916_172847` 完成 11 个 epoch，并因连续 10 轮未刷新 best 而停止；best 为 epoch 0 的 raw residual。正式 V22 封印审计完全通过：全 validation archive 与同 padded batch 独立 V22 reference 的最大/平均 prediction difference 均为 `0`，覆盖全部 `17,171` 条 validation（其中 `2,340` 条 LP structured），`backbone_trainable_parameter_count=0`。因此本次结论不是 checkpoint 漂移、BF16 重分批或 EMA 污染造成的假失败。

V29 standalone validation 五项（CI/RMSE/MAE/Pearson/Spearman）为：LP-PDBBind `0.666156/1.487737/1.167787/0.502315/0.479698`，Davis `0.822120/0.688529/0.350980/0.667138/0.478320`，KIBA `0.789814/0.586559/0.363884/0.719193/0.720982`。LP 的 V22 frozen full backbone 为 `0.663413/1.544957/1.215387/0.493385/0.471897`，而 V22 无结构 base 为 `0.618640/1.684070/1.331449/0.357701/0.347134`。故 V29 相对 V22 full 的 bounded residual gate 通过（CI `+0.002743`、RMSE `-0.057219`、Spearman `+0.007801`），结构相对 base 的 gate 也通过（CI `+0.047516`、RMSE `-0.196333`、Spearman `+0.132565`）。这确认冻结 1D+2D backbone 上的可选 3D residual 确有有效信号。

但正式最终选择的 anchor 是 V26 的 LP route（冻结 V20/V22 `0.90/0.10`），其 validation 为 `0.739125/1.169448/0.912057/0.688446/0.661846`，仍显著强于 V29。validation-only V26/V29 blend 网格的最佳 V29 权重为 `0.0`；最好的非零 `0.05` 权重已经将 LP CI 降至 `0.739013`（`-0.000112`），RMSE 增加 `0.001315`、Spearman 降低 `0.000456`，且仅 `2/5` folds 为正。因此未达到 CI 至少 `+0.0005`、至少 `3/5` folds 正增益的预注册 gate，最终 `status=validation_failed`、`validation_selected=false`，selected validation prediction 精确回退 V26。Davis/KIBA 按设计不在 V29 的 LP-structure 优化范围内，保持 V26 route 与 binary 输出不变。

test manifest 严格未读取（`test_manifest_read=false`、`test_metrics=null`），不得生成或报告 V29 test 指标。训练 loss 从 `0.279632` 降至 `0.228077`，但 LP validation 的最优点在 epoch 0；mean/max absolute residual 则从 `0.053776/0.093864` 增至 `0.152469/0.313581`，说明当前 residual 训练后期是过拟合而非数值失稳。后续不应通过降低该 gate 或按本次结果细调 residual 预算来晋级；若继续无 MSA 的单模型路线，应新建独立版本，将训练期目标提升到冻结 V26/V20 route 的 train-only teacher 或重做更强的全模型交互，而不是继续以明显较弱的 V22 full 输出作为最终性能锚点。正式产物为 `Model/OptionalStructureSealedResidual_v29/optional_structure_sealed_residual_v29_full_20260916_172847/`；best checkpoint SHA256 为 `e55377c8e76e4f74043856eee66934f68d238b296711643e81973dc664114b0a`，raw/selected-valid prediction SHA256 分别为 `cc07a5d4560a485b78e51763aaa4e6816d63eb2503c8bdf5c721e7b9ff26cf76`、`53ab013c754bee70cb3f64c0e780ccb9ea23f5653a97d52ab827f785f058c026`。

## 5.36 Optional-Structure Privileged-MSA Train-only Distillation v30（正式 full 已完成，validation 未晋级，test 未读取）

目标与隔离：V30 在 V29 正式 validation 未晋级后新建 `dta/optional_structure_privileged_msa_v30/`、`experiments/multidataset/optional_structure_v30/`、`Model/OptionalStructurePrivilegedMSA_v30/`、`data/OptionalStructurePrivilegedMSA_v30/` 与专属 CPU 测试，不修改或覆盖 V22/V26/V29。2026-09-17 已完成一次 CPU-only train teacher cache：`data/OptionalStructurePrivilegedMSA_v30/optional_structure_privileged_msa_v30_teacher_20260917_170913/`，archive SHA256 为 `a962c51fcba50352384bd576afb611bd741a1e8719240e39e113dab6495798da`。它覆盖全部 `11,037` 条 LP train，5-fold OOF train diagnostic 为 CI `0.735219`、RMSE `1.422639`、MAE `1.120986`、Pearson `0.662528`、Spearman `0.650737`；这只是训练期 soft-target 质量诊断，不是 validation/test 模型结果，也不参与 V30 checkpoint 选择。随后 GPU smoke 与正式 full 均已完成；正式 test 因 validation 未晋级而未读取，完整 validation 结论见后文。

主体模型：V30 保留单一可选结构神经路径。正式 V22 structure checkpoint 被 strict-load 为冻结 backbone；蛋白完整序列 ESM 为 1D 输入，ligand atom/bond graph 为 2D 输入，有结构时 pocket--ligand geometry 为 3D 输入。V30 只在存在 3D 的样本上从冻结 base/structure latent 学习 label head、teacher head 与 learned mixture 的双头有界 residual；两个 head 的末层均零初始化，因此初始 V30 prediction 与 V22 full 精确相同。backbone 始终 `eval()`、`requires_grad=false`、`no_grad`，EMA 也只更新 V30 新分支。无 3D 时 V30 严格输出冻结 V22 path。

训练期 privileged teacher：只使用 `Model/LP_PDBBind_MSAInterfaceTrainFeatures_v17/msa_interface_train_supervised_fusion_v17_full_20260831_170712/train_features.npz` 的 `pretrain_features`（`11037 x 5472` float32，SHA256 `9d5741709ed5156b0ea4cf91281d3b5371032fb0f1007bc1dc53434aab23ef1a`），明确不使用同归档的 supervised `affinity_features`，也不读取 V20/V26。实际核验该 archive 与 V21 LP train 的 `11,037` 个 `structure_key` 集合完全相同，所有 LP 行 `split=train`，按 key 对齐后的标签最大绝对误差为 float32 round-trip 的 `4.58e-7`。teacher builder 接口没有 valid/test manifest 参数；它只取 LP train，按精确 `sequence_id` 做 5-fold hash grouped OOF，每 fold 仅用其余 train rows 拟合 whitened PCA-96 + ExtraTrees（400 trees、minimum leaf 3、max features 0.85），再预测 heldout fold。缓存会校验 split、key、sample ID、标签、fold、manifest/archive checksum，先在 sibling 临时目录完整写入，再原子发布最终目录；metadata 固定为 `labels_scope=train_only`、`valid_manifest_read=false`、`test_manifest_read=false`。专属反泄漏回归中，只改某 heldout sequence/fold 的全部标签并重建 cache，该 fold 的 OOF teacher prediction 逐元素不变，确认 heldout 标签未进入该 fold 的 PCA/tree 拟合。该 grouping 仅隔离完全相同的 sequence ID，不应夸大为完整 homolog 或 ligand decontamination。

选择与部署边界：V30 standalone 的 `forward`、validation/test 推理均不读取 MSA feature、teacher cache、V20 或 V26；teacher 只出现在 LP train loss。正式 validation 仅在 LP 上选择冻结 V26/V30 blend，Davis/KIBA 的 V26 binary score/decision 保持冻结。必须分开报告 V30 standalone 与 V26/V30 fusion：前者是无 MSA 运行时输入的 1D+2D+optional-3D 单模型；后者仍依赖预先生成的 V26 anchor prediction，而 V26 LP route 含 V20 MSA 成分，不能称为端到端 no-MSA deployment result。test CLI 现在要求完整且未放宽的 formal validation selection；smoke、截断 valid/train、放宽 gate、test subset、改变 BF16 evaluation batch geometry、替换 V26/V22 test archive 或替换 test manifest 均会在读取 V30 test dataset 前被拒绝。test 阶段还校验两个 anchor 同目录 `result.json` 中的预测 SHA、test manifest SHA 与 `test_used_for_selection=false`。

CPU 验证：V30 专属 pytest `11/11` 通过，覆盖 train-only OOF cache、heldout-label isolation、split/key/ID/标签谱系、LP-only sampler、zero-init identity、冻结 V22 无梯度、teacher mask、nonzero blend gate、smoke/relaxed selection、test subset 和 test-path substitution 的预读取拒绝。V22/V21 兼容回归 `22/22` 通过；V30 Python 3.8 compile、三个 CLI help、四个 Bash `bash -n` 均通过。真实 cache 的 CPU probe 严格加载正式 V22 checkpoint，并对 `LP_PDBBind:10gs`、`LP_PDBBind:11gs` 两条 3D complex 完成 V30 forward/backward：loss `0.690194`、prediction 和 22 个新增分支 gradient 均有限、全部 V22 backbone gradient 缺失，且零初始化 V30 prediction 与 V22 full 逐位相等。上述验证阶段未启动 GPU。

2026-09-17 GPU smoke：`optional_structure_privileged_msa_v30_smoke_20260917_180439` 完成 3 epochs（每 epoch 4 steps），训练/validation 分别为每数据集 `128/64` 条、合计 `384/192` 条，best 为 epoch 2 EMA；没有 OOM、non-finite、teacher cache 谱系错误或冻结 V22 漂移。新增分支可训练参数为 `3,858,850`，V22 backbone 可训练参数恒为 `0`。三个 epoch 的 mean absolute label residual 为 `0.004282/0.025035/0.041512`，teacher residual 为 `0.005086/0.025368/0.045105`，loss 与各 teacher/rank component 均有限，说明双头和 mixture 路径确实参与优化。limited validation 的独立同 padded batch V22 reference audit 覆盖 `192` 条（含 `64` 条 LP 3D），最大/平均差均为 `0`；与 full V22 archive 的最大差 `0.007266` 仅来自 BF16 动态 padding 重分批，按预注册 policy 只作诊断、不参与 smoke pass/fail。

该 smoke 的 LP standalone CI 为 `0.568784`，与冻结 V22 full backbone 的 CI 相同；V30 相对 V22 full 的 CI gain 为 `0.000000`、RMSE change `-0.000425`，而相对无结构 base 的 CI gain 为 `+0.017009`。在 64 条 LP 前缀上，V26 anchor CI 为 `0.694347`，V26/V30 searched/effective V30 weight 均为 `0.0`，故 `status=validation_failed`、`validation_selected=false`、`formal_validation_selection=false`。这是经过极少 updates 的功能链路验证，不能用来判断正式泛化，更不能降低正式 gate；test manifest 未读取，test 仍被 CLI 硬性禁止。

2026-09-17 至 2026-09-18 正式 full：`optional_structure_privileged_msa_v30_full_20260917_213334` 使用上述完整 train-only cache 完成 `12` 个 epoch，best 为 epoch `1` raw，之后按 patience 正常停止。完整 train/validation 为 `111,214/17,171`，formal validation policy 全部通过，且 V30 内嵌 V22 与独立 V22 reference、全量 V22 valid archive 均逐位一致（最大/平均 prediction difference 都为 `0`，覆盖全部 `17,171` 条、其中 `2,340` 条 LP 3D）。因此本次失败不是 EMA、BF16 padding、冻结破坏或 teacher-cache 谱系错误。

V30 standalone 的 LP-PDBBind validation 五项 CI/RMSE/MAE/Pearson/Spearman 为 `0.666076/1.490985/1.173289/0.502653/0.479964`；冻结 V22 full 为 `0.663413/1.544957/1.215387/0.493385/0.471897`，即 V30 student gate 通过：CI `+0.002662`、RMSE `-0.053972`、Spearman `+0.008067`。相对无结构 V22 base，3D route 仍有 CI `+0.047436`、RMSE `-0.193085`、Spearman `+0.132830`。但 V30 teacher head 的 LP CI 仅 `0.659599`、RMSE `1.471451`、Spearman `0.460316`，没有形成比 label head 更强的 validation rank signal。Davis/KIBA 按设计保持 V22 standalone；V26 binary route 没有被改变。

正式融合仍未晋级：冻结 V26 LP anchor 为 `0.739125/1.169448/0.912057/0.688446/0.661846`，validation blend 网格的最优 V30 weight 为 `0.0`。最好的非零 `0.05` 已退化为 CI `0.738951`、RMSE `1.171111`、MAE `0.913565`、Pearson `0.687567`、Spearman `0.661229`，五项均不优于 anchor；更高权重持续下降。故 `any_promoted=false`、`status=validation_failed`、`validation_selected=false`，selected validation 逐项精确回退 V26。`test_manifest_read=false`、`test_metrics=null`，不得运行 V30 test，也不能用此 validation 负结果反向调节 V30 loss/权重。训练曲线同时显示 teacher value loss 从 `0.191440` 降至 `0.059872`，但 LP CI 在 epoch 1 达 `0.666076` 后回落至约 `0.659`，而 label residual 由 `0.067038` 增至 `0.225337`；这更符合弱 OOF teacher 被拟合、但没有跨 validation 形成 V26 互补性的解释。

后续不继续重跑或微调 V30。若继续可选结构单模型路线，应新建独立版本，使用更强的严格 train-only contact-aware MSA pair-latent teacher，并让学生在 base/cross-interaction 表征层获得受控训练期蒸馏，而不是只在冻结 V22 输出后继续堆叠 residual。该下一版本仍须保持 runtime 不读取 MSA、validation-only 选择、test 禁止调参的边界。

## 5.37 Optional-Structure Privileged-MSA Pair-Latent Distillation v31（实现与 CPU 验证完成，尚无 GPU 结果）

目标与隔离：在 V30 的正式 validation 未晋级后，V31 新建独立的 `dta/optional_structure_privileged_msa_pair_v31/`、`experiments/multidataset/optional_structure_v31/`、`Model/OptionalStructurePrivilegedMSAPair_v31/`、`data/OptionalStructurePrivilegedMSAPair_v31/` 与专属测试，不修改或覆盖 V22/V26/V30。当前只完成代码、CPU 单元/回归验证和真实 structured sample 的 forward/backward probe；尚未生成 outer-fold teacher、V22 LP-train export、teacher cache、GPU smoke、完整 validation 或 test 产物。因此没有任何可报告的 V31 validation/test 性能指标，test 也尚未读取。

部署学生：V31 strict-load 并冻结正式 V22 structure backbone，运行时输入仍严格只有蛋白 1D ESM、ligand 2D atom/bond graph 与可选 pocket--ligand 3D。MSA、teacher cache、V20/V26 prediction 均不进入 `forward` 或 standalone validation/test 推理。新增 cross-gated residual adapter 分别投影 V22 base/structure latent，并显式组合 base×structure 低秩乘积、structure−base 差分和 scalar route；label/teacher 两个有界 residual head 均零初始化，因此训练前 prediction 与冻结 V22 full 逐位一致。无 3D 输入时仍严格回退冻结 V22 path。

训练期 privileged teacher：LP 的 `11,037` 条 train row 按精确 `sequence_id` 做确定性 size-balanced 5-fold outer split（目标行数 `2208/2208/2207/2207/2207`）。每 fold 只用 outer-fit 的 inner train/valid 选择 epoch/EMA；checkpoint 固化后可固定该选择 refit outer-fit，随后才导出 blind outer holdout 的 V18-style contact-aware MSA pair latent。每 fold 的 CPU reader 只在该 fold outer-fit latent/label 上拟合 PCA+ExtraTrees，并对 blind holdout 预测 `label − V22_full_train_prediction` residual；最终 soft target 为 `V22_full_train_prediction + OOF_predicted_residual`。这仅隔离 exact sequence ID，不可表述为 homolog 或 ligand 去污染。

谱系与选择边界：teacher fold CLI 没有 official validation/test manifest 参数，且明确拒绝 V18/V12 affinity checkpoint 或任何含 `model_state` 的 affinity checkpoint 作为初始化。teacher cache 中 V22 LP-train export 的 checkpoint SHA256 必须与学生 strict-load 的 V22 checkpoint 完全相同；V22/V26 validation archive 也必须与各自 sibling sealed `selection.json` 的路径、版本、状态和 SHA256 一致，不能用同形状 prediction 文件替换 anchor。可选 `fold_local_pretraining` 才可声明完整 teacher ancestry train-only；性能优先的 `historical_self_supervised` 可使用历史 V13 self-supervised encoder，但其历史辅助任务选择曾接触 official validation，必须在 metadata 中标为 `full_ancestry_train_only=false`。V31 正式 validation 只会选择冻结 V26/V31 的 LP affinity blend；Davis/KIBA binary route 仍复制冻结 V26 score/decision。独立 test CLI 只有在未截断、未放宽的 validation selection 通过后，才会读取预注册 test manifest/V26/V22 test archive，并拒绝 test-time 重选、子集、batch geometry 或输入替换。

CPU 验证：专属 V31 测试与 V22 回归为 `24/24` 通过；覆盖 outer-fold sequence 隔离、balanced fold、heldout-label isolation、fold/cache/manifest checksum、禁止 official validation/test 输入、禁止 affinity checkpoint 初始化、teacher V22 checkpoint/validation anchor SHA 谱系、zero-init identity、V22 冻结无梯度和 validation-only/test seal。V31 全部 Python 3.8 compile、6 个 CLI `--help` 与 7 个 Bash `bash -n` 均通过。正式 V22 checkpoint 的真实 cache probe 已 strict-load，并在两条 LP structured sample 上得到有限 loss/新增分支梯度、V22 backbone gradient tensor 数为 `0`，zero-init identity 最大差为 `0.0`。这些仅证明实现与隔离链路可运行，不构成任何性能结果；GPU 执行顺序必须为 CPU folds → V22 train export → 5 个 outer teacher → CPU OOF cache → smoke → full →（仅 validation promotion 后）test。

## 5.38 External KIBA 二分类模型对比适配器（2026-09-19，尚无正式性能结果）

为后续在同一随机固定划分上比较外部模型，新建独立数据包 `data/ExternalBenchmarkKIBA_v1/`。任务固定为 `label = affinity >= 12.1`，清理 train 与 valid/test 的重复 sequence--SMILES pair 后，行数为 train/valid/test `82,506/11,825/23,651`，正样本数为 `17,329/2,519/4,921`。模型选择只读 validation，test 只在最终 checkpoint 确定后报告；统一指标为 ACC、F1、MCC、Precision、Recall、ROC-AUC、PR-AUC。该数据包和输入生成脚本不修改任何既有 GerNA-Bind 版本。

已完成两个可正式训练的独立适配器：

1. `dta/external_benchmark_graphdta_kiba_v1/`：GraphDTA GCN 分子图 + 蛋白序列双分支，affinity 头改为 `BCEWithLogitsLoss` 二分类；独立实验脚本为 `experiments/external_benchmarks/graphdta_kiba_binary_v1/`，输出目录为 `Model/ExternalBenchmarkGraphDTA_KIBA_v1/`。初版 GAT/Conv1d 在目标 GPU 的大 batch 下连续触发 cuDNN/illegal-memory/cuBLAS 错误；为保证可运行性，现切换到 GraphDTA 原始 GCN 家族的纯 PyTorch `index_add_` 消息传递，序列分支为 embedding mean/max pooling + projection，完全移除 PyG CUDA scatter、GAT、Conv1d 和 unfold，默认 batch size 降为 64。CPU smoke 已重新通过，覆盖 RDKit 图构造、PyG batch、前向、BCE 反向和有限梯度。
2. `dta/external_benchmark_deepdta_kiba_v1/`：DeepDTA categorical 双路 CNN（SMILES 长度 100、蛋白长度 1000、三层卷积和 GlobalMaxPool），affinity 头改为二分类 BCE；独立实验脚本为 `experiments/external_benchmarks/deepdta_kiba_binary_v1/`，输出目录为 `Model/ExternalBenchmarkDeepDTA_KIBA_v1/`。CPU smoke 已通过，覆盖编码形状、前向、BCE 反向和有限梯度。由于当前环境无 TensorFlow，使用无 TensorFlow 依赖的 PyTorch faithful implementation，模型结构与原始 categorical CNN 保持一致。

PSICHIC 的原始 contact-aware 输入仍受限于逐 residue contact cache；现有 V21 `proteins_esm2_t33_v21.lmdb` 只保存最多 512 residue 的池化 embedding，不含 contact map。为完成同一 KIBA 划分上的公平可运行对比，已新增明确标注的 `PSICHIC-sequence-edge` 适配器：`dta/external_benchmark_psichic_kiba_v1/`、`experiments/external_benchmarks/psichic_kiba_binary_v1/` 和 `Model/ExternalBenchmarkPSICHIC_KIBA_v1/`。它在 GPU 7 上用同一 Transformers ESM2 权重生成逐 residue embedding，使用固定 sequence +/-1、+/-2 edges 替代 contact edges，并保留蛋白图消息传递、配体图消息传递和蛋白-配体交互注意力；不声称复刻原始 contact-aware PSICHIC。CPU synthetic forward/backward smoke 已通过。原始输入审计仍保留在 `Model/ExternalBenchmarkPSICHIC_KIBA_v1/preflight/result.json`，其 `blocked_missing_per_residue_contact_cache` 只针对原始 contact-aware 版本，不影响 sequence-edge 版本。

当前尚无 GraphDTA、DeepDTA 或 PSICHIC 的正式 validation/test 指标，不应在论文或组会中引用性能数字。GPU 训练命令由用户在物理 GPU 7 上启动：

```bash
GPU_ID=7 RUN_ID=graphdta_kiba_binary_v1_full_$(date +%Y%m%d_%H%M%S) \
bash experiments/external_benchmarks/graphdta_kiba_binary_v1/run_full_gpu7.sh 2>&1 | tee /tmp/graphdta_kiba_binary_v1_full_gpu7.log

GPU_ID=7 RUN_ID=deepdta_kiba_binary_v1_full_$(date +%Y%m%d_%H%M%S) \
bash experiments/external_benchmarks/deepdta_kiba_binary_v1/run_full_gpu7.sh 2>&1 | tee /tmp/deepdta_kiba_binary_v1_full_gpu7.log

# PSICHIC-sequence-edge：先生成逐 residue ESM2/sequence-edge cache，再训练
GPU_ID=7 FEATURE_ROOT=/home/user/wangzhiwen/GerNA-Bind/data/ExternalBenchmarkPSICHIC_KIBA_v1/psichic_sequence_edge_features_v1 \
bash experiments/external_benchmarks/psichic_kiba_binary_v1/build_features_gpu7.sh \
2>&1 | tee /tmp/psichic_sequence_edge_feature_build_gpu7.log

GPU_ID=7 FEATURE_ROOT=/home/user/wangzhiwen/GerNA-Bind/data/ExternalBenchmarkPSICHIC_KIBA_v1/psichic_sequence_edge_features_v1 \
RUN_ID=psichic_sequence_edge_kiba_binary_v1_full_$(date +%Y%m%d_%H%M%S) \
bash experiments/external_benchmarks/psichic_kiba_binary_v1/run_full_gpu7.sh \
2>&1 | tee /tmp/psichic_sequence_edge_kiba_binary_v1_full_gpu7.log
```

## 5.39 GraphDTA-GCN KIBA 二分类正式结果（2026-09-20）

正式运行 `graphdta_gcn_kiba_binary_v1_full_fix3_20260920_130130` 已完成，输出目录为 `Model/ExternalBenchmarkGraphDTA_KIBA_v1/graphdta_gcn_kiba_binary_v1_full_fix3_20260920_130130/`。本次使用 GraphDTA GCN 家族的纯 PyTorch 消息传递、蛋白序列 embedding mean/max pooling + projection 和二分类 BCE；训练/validation/test 行数为 `82,506/11,825/23,651`。仅依据 validation ROC-AUC 选择 checkpoint，best epoch 为 `77`，test 未参与选择。

Validation 指标为：ACC `0.751205`、F1 `0.569001`、MCC `0.439727`、Precision `0.450894`、Recall `0.770941`、ROC-AUC `0.839329`、PR-AUC `0.632946`。

最终 test 指标为：ACC `0.751765`、F1 `0.561702`、MCC `0.434204`、Precision `0.443946`、Recall `0.764479`、ROC-AUC `0.839708`、PR-AUC `0.642637`。结果显示该实现已稳定完成全量推理，test ROC-AUC 与 PR-AUC 略高于 validation，但不涉及 test 调参。由于为避免目标 GPU 的 PyG/cuDNN kernel 错误而使用稳定 GCN/sequence-pooling 变体，后续与 DeepDTA 的比较应明确标注模型变体，不能把它描述为原始 GAT-GCN 版本。

## 5.40 DeepDTA KIBA 二分类正式结果（2026-09-20）

正式运行 `deepdta_kiba_binary_v1_full_20260920_140523` 已完成，输出目录为 `Model/ExternalBenchmarkDeepDTA_KIBA_v1/deepdta_kiba_binary_v1_full_20260920_140523/`。模型为无 TensorFlow 依赖的 PyTorch faithful categorical-CNN：SMILES 长度 100、蛋白序列长度 1000、三层卷积、GlobalMaxPool 和全连接二分类头。训练/validation/test 行数为 `82,506/11,825/23,651`，仅依据 validation ROC-AUC 选择 checkpoint，best epoch 为 `34`，test 未参与选择。

Validation 指标为：ACC `0.874503`、F1 `0.735661`、MCC `0.660478`、Precision `0.667205`、Recall `0.819770`、ROC-AUC `0.929533`、PR-AUC `0.810822`。

最终 test 指标为：ACC `0.869139`、F1 `0.716341`、MCC `0.637512`、Precision `0.652421`、Recall `0.794148`、ROC-AUC `0.922639`、PR-AUC `0.799517`。相较 GraphDTA-GCN，DeepDTA 在同一 KIBA 划分上的 test ROC-AUC `0.922639` vs `0.839708`、PR-AUC `0.799517` vs `0.642637`、F1 `0.716341` vs `0.561702`，当前二分类对比中 DeepDTA 明显更强。

## 5.41 MSA 与 sequence-only 的可比性审计（2026-09-21）

本节专门检查“只改变 MSA”与“只增加 sequence-only 数据”是否已有严格成对结果。结论是：当前历史产物中没有满足全部单变量条件的正式 pair，因此以下近似结果只能用于路线判断，不能写成严格消融结论。

### 5.41.1 MSA vs no-MSA

严格成对实验：**尚未完成**。现有 MSA 版本同时改变了 encoder、pair/interface 模块、辅助目标或融合方式；现有 no-MSA GerNA 版本也使用不同的主干配置。因此不能声称下面的差值是 MSA 的纯因果增益。

最接近的单模型对照均使用 LP-PDBBind 官方 new_split、`n=4,751`：

| 版本 | MSA | 主要输入/主体 | Test CI | RMSE | MAE | Pearson | Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| GerNA ligand-pocket ESM 8 Å | 否 | full-sequence ESM + ligand 2D + ligand-pocket residue 3D；`disable_protein_2d=true` | 0.682945 | 1.490508 | 1.170277 | 0.550505 | 0.523357 |
| MSA Pocket Expert v9 | 是 | ESM + 50-dim MSA profile gate + ligand 2D + pocket 3D + geometry Transformer/contact head | 0.682576 | 1.468233 | 1.179861 | 0.549479 | 0.523271 |

V9 相对 no-MSA GerNA 的差值为 CI `-0.000370`、RMSE `-0.022274`、MAE `+0.009584`、Pearson `-0.001027`、Spearman `-0.000086`。因此在这组“最接近”对照中，MSA 没有带来排序提升，只带来轻微 RMSE 改善；但由于 V9 的主干和辅助头同时改变，该结果不能归因于 MSA 本身。产物分别为 `Model/LP_PDBBind_GerNA_ligandPocketESM8A_ciSelect_fp32/best_metrics.json` 和 `Model/LP_PDBBind_MSAPocketExpert_v9/msa_pocket_v9_full_retry_20260824_154027/result.json`。

另一个路线级参考是 V8→V10：V8 是 no-MSA repeated rank fusion，V10 在其上加入 V9 MSA expert 并进行 validation-only blend/residual。Test 从 `0.718266/1.374328/1.072943/0.637626/0.611355` 变为 `0.722141/1.346158/1.049516/0.648728/0.620507`（CI/RMSE/MAE/Pearson/Spearman），差值为 `+0.003876/-0.028170/-0.023427/+0.011102/+0.009152`。这说明 MSA latent 作为互补 expert 有价值，但仍不是“只开关 MSA”的严格 ablation。

当前最佳 MSA 路线 V20 与 no-MSA optional-structure V22 的整体参考为：V20 Test `0.736092/1.297841/1.002948/0.674478/0.650757`，V22 full Test `0.663464/1.630012/1.278612/0.495324/0.471381`。V22 相对 V20 的差值为 `-0.072629/+0.332171/+0.275665/-0.179154/-0.179375`，但该比较同时混入 MSA、主干、训练数据和融合策略差异，只能作为路线级背景，不能作为 MSA 消融。

### 5.41.2 sequence-only vs no-sequence-only

严格成对实验：**尚未完成**。V21、V22、V23–V30 都已经使用或继承了 sequence-only 数据；项目中没有同一 optional-structure backbone、相同训练预算、仅删除 DAVIS/KIBA sequence-only rows 的正式 checkpoint。因此不能把 V20/V21 或 GerNA/V21 的差异写成“增加 sequence-only 数据带来的提升”。

最接近的历史参考（同为 no-MSA 运行时，但主干和训练协议仍不同）如下：

| 版本 | sequence-only 训练数据 | LP Test CI | RMSE | MAE | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GerNA ligand-pocket baseline | 无；仅 LP-PDBBind | 0.682945 | 1.490508 | 1.170277 | 0.550505 | 0.523357 |
| V21 optional-structure multitask | 有；LP + DAVIS + KIBA，无结构样本共同训练 1D/2D trunk | 0.647242 | 1.608049 | 1.265907 | 0.442781 | 0.425780 |

V21 相对该 GerNA baseline 的 LP 差值为 CI `-0.035703`、RMSE `+0.117542`、MAE `+0.095630`、Pearson `-0.107724`、Spearman `-0.097577`。这不是 sequence-only 的纯负贡献证据，因为 V21 同时改用了完整序列 ESM、双向 cross-attention、可选结构路由、dataset-specific head、多任务 loss 和新的训练采样。

V21→V22 说明的是训练方案改进，而不是 sequence-only 数据开关：两者都包含 DAVIS/KIBA sequence-only rows；V22 采用 Stage A 无结构训练、固定 cutoff balanced BCE 和 Stage B 结构微调。LP full Test 从 V21 的 `0.647242/1.608049/1.265907/0.442781/0.425780` 提升到 V22 的 `0.663464/1.630012/1.278612/0.495324/0.471381`，CI/Pearson/Spearman 分别 `+0.016221/+0.052543/+0.045601`，但 RMSE/MAE 分别 `+0.021963/+0.012705`。更明显的收益出现在 sequence-only 数据集：Davis ROC-AUC/PR-AUC 从 V21 的 `0.735555/0.140880` 提升到 V22 的 `0.906882/0.495135`，KIBA 从 `0.612814/0.286788` 提升到 `0.825771/0.665786`；这主要来自 staged training 与 balanced BCE，不能归因于新增 sequence-only 数据量。

### 5.41.3 可用于汇报的结论

1. 当前没有严格的 MSA-only 或 sequence-only-only 消融，不能把 V9/V20 或 V21/V22 的差值写成单变量因果结论。
2. 最近似 MSA 结果显示：单个 MSA expert V9 的 CI 几乎不变，但加入 V8 的融合 V10 后 Test CI 提升 `0.003876`，说明 MSA 更像互补表征而非独立强 predictor。
3. sequence-only 路线的主要已证实收益是 DAVIS/KIBA 的任务适配；V22 的 staged training 显著改善二分类排序，但尚未证明“仅增加 sequence-only 样本”本身提升 LP-PDBBind。
4. 若论文需要严格消融，下一步应以同一个 V22-style backbone 固定 seed、optimizer、epoch、split 和结构输入，分别建立：(a) `no-MSA`/`MSA-only` 两个版本；(b) `LP-only`/`LP+DAVIS+KIBA sequence-only` 两个版本。两组都需只用 validation 选 checkpoint，最后一次性报告 test 五项指标。

## 5.42 LP-PDBbind 单模型两阶段与窗口化全链 MSA 进展（2026-09-24—29）

本节补录此前未进入总记录的 v1–v3、低学习率 Stage 1、Stage 2 配对消融及窗口化全链 MSA。下表均为同一 LP-PDBbind validation（`n=2,340`）上的结果，不是 test；历史 V18/V20 是参照而非严格单变量对照。train/valid manifest SHA-256 分别为 `def47b0307750099bcd8accc17ac5c6960d4d92ffc789daac2cadb10faf55213` / `dc996a2665a07a3a708b7a5b7624f27f8ba3bde881bd18fe4569a2d65e3943d8`。

| 版本或对照 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 历史 V18 单模型直接预测 | 0.683750 | 1.435430 | 1.141195 | 0.537788 | 0.529274 | 历史单模型参照 |
| v1 full | 0.687801 | 1.389118 | 1.102644 | 0.555082 | 0.538595 | validation failed |
| v1 固定两子集复算 | 0.687739 | 1.389089 | 1.102630 | 0.555032 | 0.538431 | 仅诊断，不更新选模 |
| v2 full | 0.688197 | 1.400755 | 1.084636 | 0.550557 | 0.539322 | validation failed |
| v3 初版 Stage 1 | 0.687561 | 1.374161 | 1.077600 | 0.538676 | 0.536793 | epoch 2 EMA，Stage 1 完成 |
| v3 初版 Stage 2 联合训练 | 0.693271 | 1.376226 | 1.061887 | 0.561871 | 0.550179 | validation failed |
| v3 低学习率 Stage 1 | 0.691506 | 1.330366 | 1.037810 | 0.565915 | 0.546799 | epoch 6 EMA，当前保留起点 |
| 配对 A：零融合头冻结，继续微调 | 0.690708 | 1.375131 | 1.067983 | 0.567466 | 0.544537 | validation failed |
| 配对 B：融合头参与联合训练 | 0.690279 | 1.353207 | 1.053349 | 0.566627 | 0.543840 | validation failed |
| 历史 V20 多源残差融合 | 0.737938 | 1.172255 | 0.914334 | 0.686389 | 0.659726 | 既定五项性能门槛 |

### 5.42.1 v1–v3 与历史单模型参照

v1 的正式 `selection.json` 位于 `Model/LP_PDBBind_EndToEndMSAStructure_v1/lp_pdbbind_e2e_msa_structure_v1_full_20260924_170127/`；固定 `inference_subsets=2` 的 `revalidation.json` 位于同名前缀的 `_subsets2_revalidation_20260924_233100_audit/`，其 `diagnostic_only=true`，不能代替正式选模。v2 正式记录位于 `Model/LP_PDBBind_EndToEndMSAStructure_v2/lp_pdbbind_e2e_msa_structure_v2_full_20260925_104659/`。两版五项均未过 V20 no-regression gate，未读取 test。

v3 的两阶段、同一主干在线联合训练 GPU smoke 在 `Model/LP_PDBBind_EndToEndMSAStructure_v3/lp_pdbbind_e2e_msa_structure_v3_gpu_smoke_fix2_20260926_145526/result.json` 记录为 `smoke_passed`；Stage 2 融合头、主干和 MSA encoder 均有参数更新，存取重载及两子集完整前向取均值检查通过。这只证明小样本链路与显存可运行，不是正式 validation 性能。随后初版全量结果位于同版本 `lp_pdbbind_e2e_msa_structure_v3_full_20260926_153903/{stage1,stage2_joint}/selection.json`。Stage 2 比 Stage 1 改善 CI、MAE、Pearson、Spearman，但 RMSE 略升；Stage 2 主干也被更新，不能把预测整体上移归因于融合头。其 V20 五项 gate 未过，未运行 test。

历史 V18 直接预测、v3 初版两阶段及 V20 在 `handoff/v3_v18_validation_diagnostic_20260926/README.md` 中按相同 2,340 个 key 和标签复算。v3 Stage 1 五项略优于 V18，故“未达到 V20”不能直接解释为 Stage 1 训练故障。V20 使用历史预测、latent 特征和残差融合，并在 validation 内部分折拟合残差；它是保留的性能目标，不是与 v3 单模型完全相同的标签使用协议。

### 5.42.2 低学习率 Stage 1 与配对 Stage 2

原 v3 Stage 1 的主干/MSA encoder 学习率为 `1.5e-4/3e-5`；单次低学习率对照改为 `5e-5/1e-5`，选中 epoch 6 EMA，记录位于 `Model/LP_PDBBind_EndToEndMSAStructure_v3/lp_pdbbind_e2e_msa_structure_v3_stage1_lr3_20260927_144630/selection.json`。它的总体五项优于原 Stage 1，但真实标签最低四分位的 MAE 由约 `1.311` 升至 `1.458`。已有按真实标签和按预测值分组的诊断表明，这不是一个可通过全局向下平移修复的问题；不据此改输出偏置、低标签权重或 validation 校准器。该低学习率 checkpoint 仅作为本组实验保留起点，未晋级 V20。

配对 A/B 从**同一个**低学习率 Stage 1 完整 checkpoint 出发，均训练 12 epoch、`33,120` optimizer steps；主干/MSA encoder 学习率均为 `1.5e-5/3e-6`，B 的融合头学习率为 `5e-5`，A 则将零融合头保持冻结。两臂结果分别在 `Model/LP_PDBBind_EndToEndMSAStructure_v3/lp_pdbbind_e2e_msa_structure_v3_stage2_pair_20260928_134340_A_frozen_zero/selection.json` 和相同前缀的 `_B_joint/selection.json`；两臂均选择首轮训练后的 epoch 0 EMA（不是未更新的起点），checkpoint reload 一致，A/B 融合头变更张量数分别为 `0/6`。B 相对 A 的 RMSE、MAE 较好，但 CI、Pearson、Spearman 略低；A、B 相对共同 Stage 1 起点的 CI、RMSE、MAE、Spearman 都变差。结论限于这次单种子、固定配方：暂停当前 Stage 2 的重复训练，保留低学习率 Stage 1；不能声称融合头完全无作用，也不启动 test。

### 5.42.3 窗口化全链 MSA：缓存与审计进展

新的 `dta/lp_pdbbind_windowed_msa_v1/` 实现把全链 A3M 按固定重叠窗口编码、等权合并到完整链，再将对应残基特征经零初始化投影注入口袋 token；冻结旧主干，仅训练新增投影。它提供全链**窗口覆盖**，不是整条长链的一次全局注意力。独立来源清单绑定了预期 A3M 哈希和各模式的选行身份；CPU 定点复审见 `DTA_Windowed_MSA_Source_Manifest_Review_20260929.md`。CPU 接线和来源绑定通过不等于真实全量缓存或亲和力收益已验证。

两链真实 GPU6 smoke：`.codex_runtime/windowed-msa-real-smoke-preflight-20260929/extraction.json` 为 `real_windowed_extraction_passed`，query-only 和 aligned-MSA 各两链，官方权重 SHA-256 `b8f276f513fd8bdfcb0c8ae38371075e35f4af559449e5212374a08880a20429`；同目录 `consumption.json` 为 `real_cache_consumption_passed`，train `5iq6_protein` 与 valid `1exv_protein` 口袋特征分别为 `[73,768]`、`[83,768]`。两模式特征存在差异，但 smoke 未运行亲和力模型前向、训练或指标计算。

全量 CPU 来源预检 `.codex_runtime/windowed-msa-full-preflight-20260929/preflight.json` 为 `source_preflight_passed`：train `11,037` 样本/`7,193` 链，valid `2,340` 样本/`1,727` 链，合并去重 `8,863` 链；来源清单 SHA-256 为 `869e531e2968172545c73a95f5c6af31cc2040cd06f762c48ad02a9da8cea19d`，原始 A3M 数据库 SHA-256 为 `3ed43738229659206d322e456b9d5139851046e70cbd32129d639110a6a6fd26`。预检本身只确认输入来源和目标清单，不生成全量特征。

GPU6 wrapper 初次因 Bash 空数组与 `set -u` 在 Python 启动前报错，后已修复入口。两臂全量提取现均完成：`.codex_runtime/windowed-msa-full-preflight-20260929/{query_only,aligned_msa}/metadata.json` 分别记录 `status=built`、`target_count=8863`、`newly_written=8863`、`reused=0`；来源清单 SHA-256 均为 `869e531e2968172545c73a95f5c6af31cc2040cd06f762c48ad02a9da8cea19d`，权重 SHA-256 均为 `b8f276f513fd8bdfcb0c8ae38371075e35f4af559449e5212374a08880a20429`，窗口协议均为 `aligned_columns_512_stride_256_right_aligned_tail_equal_mean_v1`。CPU 全样本消费审计 `.codex_runtime/windowed-msa-full-preflight-20260929/coverage_audit.json` 已写出 `status=full_cache_coverage_passed`：每臂通过 train `11,037`、valid `2,340` 样本读取及特征形状/有限性检查，`test_manifest_read=false`、`training_run=false`。因此可以进入锁定的冻结主干投影 A/B 训练；目前**尚无该 A/B 的训练或 validation 性能结果**，也未读取 test。

后续路线已明确：先按锁定协议完成本次 A/B，不在实验中途更换输入。随后另建版本，**用全链对齐 MSA 编码替换旧的口袋列 MSA 编码**，而不是继续同时保留两套 MSA 输入并把结果称为“全序列替换”。这项后续修改尚未实施；全链特征最终如何与口袋–配体交互衔接及相应无口袋 MSA 对照，需在新版本中明确，不能把本次增量注入实验当作替换实验。

### 5.42.4 窗口化全链特征的投影 A/B 正式结果（2026-09-29）

`Model/LP_PDBBind_WindowedMSAProjection_v1/pair_20260929_202011/summary.json` 记录两臂 `paired_training_complete`：共同的低学习率 Stage 1 checkpoint SHA-256 为 `37d00e82039d74ce0524f6d17249b83130091686f41d851ba16ad7ce00d8c2a1`，来源清单与冻结编码器权重 SHA-256 与上节一致。两臂各训练 12 epoch、`33,120` optimizer steps，batch size 4、投影学习率 `1e-4`、相同 seed `20260929`；均选中完成第一轮训练后的 epoch 0 raw。训练前零投影与 Stage 1 的预测最大差为 `0.0`，选中 checkpoint 重载的预测最大差也为 `0.0`。本次只训练新增投影，**旧口袋 MSA 路径仍在**，所以不是“全链 MSA 替换口袋 MSA”的实验。

| 同一 validation，n=2,340 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 保留的低学习率 Stage 1 | 0.691506 | 1.330366 | 1.037810 | 0.565915 | 0.546799 |
| 新投影 A：全链 query-only | 0.690544 | 1.335697 | 1.045159 | 0.567741 | 0.543645 |
| 新投影 B：全链 aligned-MSA | 0.693494 | 1.334622 | 1.052332 | 0.571871 | 0.551197 |
| 历史 V20 性能门槛 | 0.737938 | 1.172255 | 0.914334 | 0.686389 | 0.659726 |

B 相比 A 改善 CI、RMSE、Pearson、Spearman，但 MAE 变差；相比 Stage 1，B 的 CI/Pearson/Spearman 较好，而 RMSE/MAE 较差。两臂五项 V20 gate 均未通过，`validation_selected=false`，不读 test，也不从单种子结果宣称全链同源上下文无效或已显著获益。Stage 1 保持回退候选。下一步按用户要求另建**全链 MSA 替换版**：在完整链对齐上完成窗口化编码，然后映射到结构口袋用于配体交互；模型不得再读取旧的口袋 MSA tokens/profile。此处“全链”指 MSA 编码输入覆盖整条链，不等于要求每个非口袋残基都进入下游口袋–配体结构交互。新版本不能使用本次 validation 指标放宽 V20 门槛。

### 5.42.5 全链 MSA 替换版的实现与验收（2026-09-30）

独立目录 `dta/lp_pdbbind_fullchain_msa_replacement_v1/` 已实现新数据读取、模型、训练入口和诊断入口；`experiments/lp_pdbbind/fullchain_msa_replacement_v1/` 提供 GPU6 smoke 与正式配对运行脚本。与上节的“新增投影”不同，替换版在模型入口拒绝旧 `pocket_msa`、多行 pocket tokens/insertions/row features；数据集不打开旧 MSA row LMDB，旧复合物缓存里的 50 维 profile 在组 batch 前丢弃。旧口袋列 axial blocks、token/row embedding、profile adapter/gate 不再留在新模型状态中。完整链窗口特征在上游已编码，按原链位置映射到有坐标的口袋残基，再进入保留的结构交互主干。口袋外残基没有直接进入下游几何 attention，故仍不能称为长链一次全局注意力。

两臂将使用相同初始化和训练协议，分别消费 query-only 与 aligned-MSA 全链缓存；Stage 1 的兼容几何/配体权重被转移，新全链输入层从同一种子初始化，旧零残差头冻结。正式 V20 五项 gate 不变，test 没有入口。CPU 验收中发现复用的 V18 pair-token 残基嵌入使用 0–19 的氨基酸编号，而复合物缓存使用 1–20；现已在新模型内部显式映射并增加回归测试，不读取旧 MSA query row。相关 CPU 回归为 `38 passed`；修正后用两条真实 train 样本及 Stage 1 权重再次完成完整损失、非零全链输入梯度和一步更新，batch 不含旧口袋 MSA 字段；此前另有 4 train/4 valid 的诊断性一轮 CPU fit，完成 2 次更新及 checkpoint 重载（预测最大差 `0.0`），但该小样本 fit 尚未在编号映射修正后重跑。**这些不是可报告的 validation 指标。**

GPU6 单步 smoke 已由用户执行，报告位于 `Model/LP_PDBBind_FullChainMSAReplacement_v1/fullchain_replacement_smoke_20260930_133711/result.json`，状态 `real_gpu_smoke_passed`。batch size 4 的新增全链输入层梯度绝对和为 `4573.8525`、参数已更新、旧口袋 MSA batch 字段为空、checkpoint 重载预测最大差 `0.0`；峰值 allocated/reserved 分别为 `742,109,184/767,557,632` bytes。Stage 1 checkpoint 与来源清单哈希均匹配，`test_manifest_read=false`，本次没有计算 validation 指标。该显存仅是这四条 train 样本的一步峰值，不保证全量训练中最长样本的峰值。正式两臂结果见下一节；本次固定 batch 为 4，未因 40GB 冗余在训练中途修改。具体输入/训练边界见 `dta/lp_pdbbind_fullchain_msa_replacement_v1/README.md`。

### 5.42.6 全链 MSA 替换版的配对 validation 结果（2026-09-30）

`Model/LP_PDBBind_FullChainMSAReplacement_v1/fullchain_replacement_pair_20260930_140728/summary.json` 记录 `paired_training_complete`。两臂共享 Stage 1 checkpoint SHA-256 `37d00e82039d74ce0524f6d17249b83130091686f41d851ba16ad7ce00d8c2a1`、完整链来源清单 SHA-256 `869e531e2968172545c73a95f5c6af31cc2040cd06f762c48ad02a9da8cea19d`、上游编码器权重 SHA-256 `b8f276f513fd8bdfcb0c8ae38371075e35f4af559449e5212374a08880a20429`，且 batch size 4、seed `20260930`、主干/新下游 encoder 学习率 `5e-5/1e-5`、每臂 20 epoch 和 `55,200` 次更新均相同。两臂均选中完成第 5 轮训练后的 epoch 4 EMA，checkpoint 重载预测最大差为 `0.0`。两份验证预测与 Stage 1 的 `2,340` 个 key、标签逐项一致；`test_manifest_read=false`。

| 同一 validation，n=2,340 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 保留的低学习率 Stage 1 | 0.691506 | 1.330366 | 1.037810 | 0.565915 | 0.546799 |
| 替换 A：全链 query-only | 0.692456 | 1.342815 | 1.048839 | 0.568455 | 0.549604 |
| 替换 B：全链 aligned-MSA | 0.690234 | 1.340052 | 1.049621 | 0.566120 | 0.541680 |
| 历史 V20 性能门槛 | 0.737938 | 1.172255 | 0.914334 | 0.686389 | 0.659726 |

aligned-MSA 相对 query-only 的 RMSE 低约 `0.002763`，但 CI、MAE、Pearson、Spearman 分别差约 `0.002222/0.000783/0.002335/0.007924`；在这次固定配方的单种子实验中，未见稳定的同源上下文增量。两臂 RMSE、MAE 都差于共同的 Stage 1 起点，且均未通过 V20 五项 no-regression gate，`validation_selected=false`；保留 Stage 1，不启动 test。query-only 仍使用预训练 MSA Transformer 的 query 表征，并非“无 MSA 模型”。

本实验是**旧口袋 MSA 训练过的 Stage 1 权重热启动**：运行时旧口袋 MSA 输入已被移除，但转移的几何/交互/预测权重可能保留其训练先验，因此不能称为从零训练的纯全链替换。完整链特征先窗口化编码，再只读取口袋残基送入下游；口袋外 token 没有直接参与配体交互。上述结果仅约束这一热启动、口袋读出结构，不能据此否定完整链 ESM＋MSA 融合或非口袋残基直接参与交互的方案。若以后增加 batch，须作为新的成对训练协议验证，不能与本次结果混称单变量对照。

### 5.42.7 完整链 ESM＋MSA 直接上下文版的准备（2026-09-30；未运行 GPU）

`dta/lp_pdbbind_fullchain_esm_msa_cross_v1/` 是针对上节两个限制另建的版本：不加载任何旧口袋 MSA 训练后的 Stage 1 权重；对 train/valid 原始 LMDB 中完整链 ESM 与已审计的窗口化完整链 MSA 特征逐残基融合，再让结构口袋/配体 token 通过交叉注意力**直接读取非口袋残基**。局部结构分支仍使用从完整链特征按原位置取出的口袋向量，而不是旧 pocket-MSA 行或 50 维 profile。query-only 与 aligned-MSA 使用相同新初始化、完整链 ESM、数据、损失和预算；只改变缓存的 MSA 表征模式。这是新的结构和新的训练协议，不是上节结果的同变量重复。完整链特征仍来自 512 列重叠窗口，不应称为整条长链一次全局注意力。

GPU 前的本地 CPU 接线与诊断表明：非口袋特征扰动改变预测，完整链 ESM/MSA 与非口袋交叉注意力均有有限非零梯度；两条真实 train 样本的完整链长度为 416（双链），非口袋位置 681 个；4 train/4 valid 小样本一轮诊断完成 2 次更新和 checkpoint 重载预测零差。最长 8 条 train 复合物的 stress batch 为 `[8,2701,640]` 完整链 ESM、19,773 个非口袋位置，口袋 ESM 与完整链按索引提取结果一致；同一 train key 的 query-only/aligned-MSA 使用完全相同的 ESM 和位置掩码，MSA 特征不同。小样本指标没有性能解释。此时尚未执行 GPU smoke、正式两臂训练或完整 validation；正式两臂必须固定同一已通过的 batch。具体协议与命令见 `dta/lp_pdbbind_fullchain_esm_msa_cross_v1/README.md`。

GPU6 长样本 batch-8 smoke 随后已完成，报告为 `Model/LP_PDBBind_FullChainESMMSACross_v1/fullchain_cross_smoke_b8_20260930_235028/result.json`，`status=real_gpu_smoke_passed`。8 条 train 样本中最大拼接全链长 2,701、最大口袋长 108、非口袋位置总数 19,773；完整链 ESM 投影、MSA 投影和非口袋交叉注意力的梯度绝对和分别为 `115.307/230.305/334.747`，ESM 投影更新、重载预测差 `0.0`，无旧 pocket-MSA batch 字段或 Stage 1 权重。峰值 allocated/reserved 为 `1,720,645,632/1,948,254,208` bytes。首步计时 `11.33` 秒包含诊断同步，不是稳态训练吞吐；batch 数据载入 `3.40` 秒，扫描长 train 样本 `25.69` 秒只属于 smoke 选样开销。本次未计算 validation 指标、未读取 test。下一步可在相同策略下试 batch 16，确定正式两臂共同 batch；不能因 batch 8 显存低就直接假定 batch 32 更快。

GPU6 batch-16 长样本 smoke 也通过，报告为 `Model/LP_PDBBind_FullChainESMMSACross_v1/fullchain_cross_smoke_b16_20261001_171304/result.json`。最大拼接链长仍为 2,701，最大口袋长 110；三条新增路径梯度有限且非零，完整链 ESM 投影已更新，重载预测差 `0.0`，初始化指纹与 batch 8 相同。峰值 allocated/reserved 为 `3,274,001,920/3,449,815,040` bytes。首步诊断时间 `15.00` 秒、约 `1.067` 样本/秒，相比 batch 8 的 `0.706` 样本/秒较高，但包含诊断同步，不能代表整轮稳态速度；长样本 batch 载入 `27.57` 秒也显示 I/O 波动。另用正式 4-worker DataLoader 连续读取两批普通 train 样本，第一批含 worker 启动约 `18.80` 秒，下一批预取后约 `0.03` 秒；每轮 batch 16 为 `690` 次更新。决定本次正式两臂固定 batch 16、每臂 30 轮，不再仅因剩余显存试 batch 32（同轮数会把更新次数减半）。正式训练和 validation 尚未执行，不读 test。

### 5.42.8 完整链 ESM＋MSA 直接上下文版的配对 validation 结果（2026-10-02）

`Model/LP_PDBBind_FullChainESMMSACross_v1/fullchain_cross_pair_b16_20261001_210728/summary.json` 记录 `paired_training_complete`。query-only 和 aligned-MSA 均为无旧 Stage 1 权重的相同随机初始化（指纹 `e43ed446779d4b39bab2bf6e68ec2effaede935780a51761689b21d9a811820e`），batch 16、seed `20261001`、学习率 `1e-4`，各完成 30 轮、`20,700` 次更新。两臂均选中完成第 3 轮后的 epoch 2 EMA；checkpoint 重载预测最大差 `0.0`。两份预测与低学习率 Stage 1 的 2,340 个 validation key/标签逐项一致，按项目指标函数独立重算五项与归档完全一致；`test_manifest_read=false`。

| 同一 validation，n=2,340 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 保留的低学习率 Stage 1 | 0.691506 | 1.330366 | 1.037810 | 0.565915 | 0.546799 |
| 直接上下文 A：全链 query-only | 0.700315 | 1.313526 | 1.020625 | 0.583405 | 0.565655 |
| 直接上下文 B：全链 aligned-MSA | **0.700513** | **1.308736** | **1.017176** | **0.589212** | **0.565923** |
| 历史 V20 性能门槛 | 0.737938 | 1.172255 | 0.914334 | 0.686389 | 0.659726 |

在这次固定初始化、单种子配对实验中，aligned-MSA 相比 query-only 五项都向好，但增量较小：CI `+0.000198`、RMSE `-0.004790`、MAE `-0.003449`、Pearson `+0.005807`、Spearman `+0.000268`。aligned-MSA 相比旧低学习率 Stage 1 五项也都更好；不过两者同时改变了架构、初始化、batch 和训练协议，不能把这组跨版本差值单独归因于非口袋上下文或 MSA。与 V20 相比，aligned-MSA 仍差 CI `0.037425`、RMSE `0.136481`、MAE `0.102842`、Pearson `0.097178`、Spearman `0.093803`，两臂五项 gate 均未通过，`validation_selected=false`，不启动 test。保留 aligned-MSA 为本组更强的单模型研究候选，同时保留 Stage 1 参照；不作正式 V20 晋级。

两臂 `history.json` 中，epoch 2 都是 CI、RMSE、MAE、Pearson、Spearman 的全程最佳；之后 train loss 持续下降（query-only `0.9857→0.4455`，aligned-MSA `1.0007→0.4540`，从 epoch 2 到 29），最终 validation 五项均明显变差。这是该配方的过拟合迹象，不支持简单延长相同训练。现有 A/B 只比较 query-only 与同源 MSA 模式，尚未通过关闭非口袋交叉注意力的匹配对照单独测出该分支的贡献；也不能由单种子的小幅 B−A 差异宣称稳定增益。

### 5.42.9 完整链 MSA 仅经全链上下文进入的独立版本（2026-10-02；GPU smoke 已通过）

用户要求进一步排除“口袋 MSA”的局部读出。上一版 `fullchain_esm_msa_cross_v1` 虽没有旧口袋 MSA 缓存或 Stage 1 权重，但仍把完整链 MSA 表征按口袋位置取出，送入局部 pair/interface 分支。本版 `dta/lp_pdbbind_fullchain_esm_msa_all_v2/` 改为：局部有坐标的口袋–配体分支只读取 ESM/结构；完整链 MSA 只与完整链 ESM 在全链残基上下文中融合，再由口袋/配体 token 对**全部完整链残基**做交叉注意力，不再生成或消费 `fullchain_msa_pocket`。旧口袋 MSA row/profile 仍被拒绝，模型随机初始化，不加载 v3 Stage 1 或旧口袋 MSA 权重。这里仍保留结构口袋用于有坐标的几何/接触计算；“无口袋 MSA”不等于取消结构口袋。上游 MSA 编码仍为 512 列重叠窗口覆盖，不是整条长链一次全局 attention。

计划在相同新初始化和训练协议下配对比较 query-only 与 aligned-MSA，固定 batch 16、seed `20261001`、学习率 `1e-4`、30 epoch，并沿用 V20 五项 validation gate；不读 test。由于相对上一版同时改变了局部输入和全链注意力范围，跨版本差值不能单独归因其中某一项。新版本及两条复用路径的 focused CPU 测试 `12/12` 通过；两条真实 train 样本的 CPU 探针中，batch 无局部 MSA 字段，完整链 ESM `[2,416,640]`、结构口袋位置 151、非口袋位置 681，完整链 ESM/MSA、全链注意力及局部 ESM 路径均有有限非零梯度。另有 4 train/4 valid 一轮 CPU 诊断 fit，完成 2 步更新且 checkpoint 重载预测最大差为 `0.0`。**这些不是正式 validation 性能指标。**

用户已在 GPU6 执行 batch-16 长样本 smoke；报告位于 `Model/LP_PDBBind_FullChainESMMSAAll_v2/fullchain_all_smoke_b16_20261002_150510/result.json`，本地文件核对状态为 `real_gpu_smoke_passed`、版本 `lp-pdbbind-fullchain-esm-msa-all-v2`。16 条 train 样本的最大拼接全链长度为 2,701、最大结构口袋长度 110；`local_msa_input=false`、`local_protein_source=esm_only`、`msa_context_scope=all_fullchain_residues`，旧口袋 MSA batch 字段为空且无旧权重转移。完整链 ESM 投影、MSA 投影与上下文注意力梯度绝对和分别为 `65.260/140.923/194.265`，ESM 投影已更新，重载预测最大差 `0.0`。峰值 allocated/reserved 为 `3,258,368,512/3,393,191,936` bytes；首步 `12.01` 秒是含诊断同步的时间，不代表稳态吞吐。`test_manifest_read=false`、未计算 validation。正式配对结果见下节。命令及边界见新版本 README。

### 5.42.10 完整链 MSA 全残基上下文版的配对 validation 结果（2026-10-02）

`Model/LP_PDBBind_FullChainESMMSAAll_v2/fullchain_all_pair_b16_20261002_152353/summary.json` 记录 `paired_training_complete`。两臂共享随机初始化指纹 `21d4a28bc5277b8e688fb9d4f6d56e5df585b9860512a99a05ebeee58707c790`、完整链来源清单和上游编码器权重；batch 16、seed `20261001`、学习率 `1e-4`，各完成 30 轮、20,700 步。局部分支为 ESM-only，MSA 仅经全部完整链残基的上下文注意力进入，不加载旧 Stage 1／口袋 MSA 权重。两臂均选中 epoch 4 EMA，checkpoint 重载预测最大差 `0.0`，`test_manifest_read=false`。两份预测与上一版两臂和低学习率 Stage 1 的 2,340 个 validation key、标签逐项一致；用项目指标函数独立重算五项与归档一致。

| 同一 validation，n=2,340 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 低学习率 Stage 1（含旧口袋 MSA 训练先验） | 0.691506 | 1.330366 | 1.037810 | 0.565915 | 0.546799 |
| 上一版直接上下文：query-only | 0.700315 | 1.313526 | 1.020625 | 0.583405 | 0.565655 |
| 上一版直接上下文：aligned-MSA | 0.700513 | 1.308736 | 1.017176 | 0.589212 | 0.565923 |
| 本版无局部 MSA：query-only | 0.699828 | 1.310722 | 1.015146 | 0.589724 | 0.564688 |
| 本版无局部 MSA：aligned-MSA | **0.703450** | **1.298584** | **1.010086** | **0.602351** | **0.573856** |
| 历史 V20 正式晋级门槛 | 0.737938 | 1.172255 | 0.914334 | 0.686389 | 0.659726 |

本版 aligned-MSA 对 query-only 五项都改善，差值依次为 CI `+0.003622`、RMSE `−0.012138`、MAE `−0.005060`、Pearson `+0.012626`、Spearman `+0.009169`。这支持**在本次固定初始化和训练配方下**，真实同源对齐相对 query-only 提供了增量；单种子不能证明跨种子稳定性。相比上一版 aligned-MSA，本版五项也都更好，但本版同时把局部输入改为 ESM-only、把注意力范围扩大到全部链残基，不能把跨版本收益单独归因于“去掉口袋位置的 MSA”或“新增口袋位置的全链注意力”。query-only 仍使用预训练 MSA Transformer 的 query 表征，并非完全无 MSA。

两臂仍未通过 V20 五项 gate，`validation_selected=false`，不读取 test；本版 aligned-MSA 保留为这组完整链单模型中的较强研究候选，不替代正式 V20。历史最优出现在训练前期：query-only 的五项全程最佳都在 epoch 4；aligned-MSA 的 CI、MAE、Spearman 最佳在 epoch 4，RMSE、Pearson 的单项最佳在 epoch 5，但按既定选模规则选中 epoch 4。之后 train loss 持续下降而最后一轮 validation 明显变差（aligned-MSA CI `0.703450→0.683067`、RMSE `1.298584→1.470818`），不支持简单延长相同配方。

### 5.43 LP-PDBBind 六外部架构公平重训准备（2026-10-03；未开训）

用户确定使用官方 `new_split` 的 LP-PDBBind train/val/test 协议、六个外部架构各单种子从随机可训练权重重训，要求上游网络完整和真实输入接线，不以历史不同样本数的 test 汇总直接排名。准备阶段封存的**物理 train/val-only** 数据源为 `.codex_runtime/lp_external_fair_source_v1/train_val.csv`，train `11,513`、val `2,422`，SHA-256 `d57809d39cb68aa7ee84798ebe17b1366db44d6aa9dfab37b6e0bf5e9700f842`；不含 test 行。六个公平训练入口只允许该源、新缓存/输出、单种子 `42` 和 validation CI 选模；历史默认入口保持原状。预算、源码 commit、执行/导出门禁及局限见 `docs/LP_PDBBind_external_fair_test_plan.md` 与 `experiments/external_benchmarks/lp_pdbbind_fair_v1/protocol.json`。

上游六个源码快照哈希核验均通过，这只证明来源。3DProtDTA、GraphDTA GIN、DeepDTAGen、TankBind 和 CASTER-DTA 又分别用真实 LP train 样本完成上游完整网络的有限前后向，必需输入和主要分支梯度非零；证据位于 `docs/LP_PDBBind_external_architecture_evidence_4.json` 及 DeepDTAGen 的 `architecture_acceptance/deepdtagen_real_train_probe_v3.json`。DeepDTAGen 的 affinity-only 无标签路径与完整前向、改变/删除 `y` 的 affinity 输出最大差均为 `0.0`，避免未来推理将真实标签输入网络。AttentionMGT-DTA 完整网络目前只通过**合成** 320 维输入前向；官方 ESM2 8M 权重已从 fair-esm 来源取得，SHA-256 `46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928`，CPU 核对为 6 层/320 维。它仍缺真实 320 维 train/val 缓存与真实 LP 样本前后向，公平入口禁止截旧 640 维或补零，因此整组六模型启动器仍正确阻断。Attention 的固定窗口全链提取未执行。

提取前的只读覆盖核查发现：官方 train/val 的 `13,935` 个唯一 key 中，源链 LMDB 缺 `460` 个（train `389`、val `71`）；其中 `332` 个同时缺原生口袋 PDB（train `288`、val `44`），其余 `128` 个仅缺源链记录。随后完成的**逐 key 原生输入资格账本**为 `.codex_runtime/attentionmgt_eligibility_trainval_v1.json`，SHA-256 `5f1ee2ef882e036733edd88a95424a5baa435707f527b1610154398fb344d878`：合格 train/val `9,323/2,125`，明确排除 `2,190/297`，合计严格分割全部 `13,935` 条。排除原因组合为口袋残基无法映射 `1,729`、同时缺口袋和链 `332`、无效原生配体图 `250`、仅缺链 `128`、无效配体图加映射失败 `48`。这是 CPU 资格审计，不是原生图/tokenizer 全量预检通过，更不是 GPU 320 维提取或训练完成。旧的“全部 CSV 行必须有链 LMDB/口袋”提取会确定性失败；新流程只对合格 key 严格映射与提取，并如实报告差异覆盖，不能称“完整官方 13,935 行全部训练”。

全量 train/val 的**静态**字段/文件扫描见 `docs/LP_PDBBind_external_nominal_input_coverage_4.json`：可解析 SMILES `11,229/2,404`，蛋白 PDB 与原生 ligand 文件均存在 `11,225/2,378`；GraphDTA 超过 1,000 残基的行有 `556/23`，多链分隔符行有 `2,676/626`。这不是实际全量图构建成功率；正式运行须报告各模型原生覆盖及共同 key 交集。TankBind 使用原生配体位姿/口袋，Attention 需预训练 ESM 输入，不能与纯原始输入模型称为“等信息量的完全从零架构排名”。

六个带 key 导出器默认只处理 validation，并将每条源 key 明确分为预测或带原因排除；test 必须在六份 checkpoint 和带 key validation 预测均封存后另行显式释放，从独立无标签 test-only 输入生成预测。共享消费边界要求正式释放的 `.source.json` 与六臂当前封存哈希一致，独立模型导出脚本也不能仅凭 `--execute-sealed-test` 提前读取 test 输入。正式六臂评估新增 `--require-sealed-six --released-input-csv`，在打开 canonical test 标签 manifest 前核对六个固定候选、当前 run/validation seal、释放输入与各份 test 预测 provenance；报告单列 Attention 的预训练 ESM2、TankBind 的原生配体位姿/位点和其他输入层级，不能包装成等信息量的架构排名。3DProtDTA 的 fair 路径另修复上游 CI 随验证样本顺序变化的问题，改用与统一评估相同的精确可比较对 CI，乱序标签反例测试通过，历史入口不变。统一评估器沿用 V20 的 CI 预测并列半分及 stable-rank Spearman，输出原生覆盖与共同样本指标。当前 `ln02` 会话无 CUDA，未正式重训、未产生新的六模型 validation/test 预测，也未读取 test manifest 或运行 test；旧 test 已被历史实验查看，未来共同样本结果只能是**回顾性**比较。源/架构/导出/启动器的 CPU 检查通过，但不得将其写成 GPU 训练完成或性能结论。

本轮合并的 11 个相关 CPU 测试文件共 `63/63` 通过；只读六臂启动器预检在前五臂均通过后，如预期被 AttentionMGT 尚未完成的原生图/tokenizer、GPU 特征及真实样本反传证据阻断。Attention 的全量 CPU 原生图/tokenizer 预检仍在进行；验收报告未生成前不把资格账本误称为训练放行。

### 5.44 AttentionMGT 原生 320 维提取与 CPU 续验修复（2026-10-04；未开训）

GPU6 的 AttentionMGT ESM2 8M 固定窗口提取已生成 `.codex_runtime/attentionmgt_native_esm320_trainval_v1/attentionmgt_source.json` 及约 5.1 GB 的 `data.mdb`；提取之后的 CPU 输入预检在 `load_lp_splits()` 报 `missing required fields`，**不是 GPU 提取或模型训练失败**。封存 CSV 的 train/val 共有 `282/18` 条空 SMILES，全部已在资格账本中明确排除，合格 `9,323/2,125` 条中没有空 SMILES。旧加载器过早对全部源行要求 SMILES 并执行全表 `dropna`，与“先核对完整资格分区，再只构建合格样本”的协议不一致。

修复保留全部 `11,513/2,422` 条源行用于严格分区检查，只在分区确认后拒绝合格样本的空 SMILES；非公平历史入口的缺失值行为不变。旧原生图/tokenizer 报告和生产脚本均未改写，原报告 SHA-256 `27af9570dea2de21b1c28e33721c907bd7eb9a7e6d9c7c787ff1835771995dd1`、旧适配器 SHA-256 `45918fb496d19dd4ac6fe1e025ef85295249c50047e9979489bcb3fee0a52d2e`。新兼容证明将修复后适配器字节级反向重建为旧版，限定仅加载器、资格检查与报告校验函数可变；真实来源复核确认原图报告仍覆盖 `11,448/11,448`、失败 `0`。相关 19 项 CPU 单元测试通过。既有 320 维缓存保持原样，不重启一次性 GPU 提取。

`run_postextract_cpu.sh` 的全量 CPU 输入预检已通过：合格 `11,448/11,448`，特征 `data.mdb` SHA-256 `b61e11129ed0382171c36ca2f0f5e373d3dcd3317fd30b3f26b72c2bbcc85181`、原图报告和权重均与来源绑定；预检报告 SHA-256 `f7495e5c098fe729567ea85b99309aff6140d2fa1e1efc350bf79fb78ebb5d8a`。真实 train key `10gs_protein` 独立 CPU ESM2 重算与既有 GPU 缓存最大差 `1.0431e-5`，完整 AttentionMGT 七个检查分支均有有限、非零梯度，`optimizer_step=false`；审计报告 SHA-256 `25efa0a20d003eb44b32c8fe474e2a78ced111bf645ddf49d7843cee238225a1`。最终 `architecture_acceptance.json` 已记录 AttentionMGT `approved_for_fair_training=true`。这只放行公平重训的真实输入和完整网络接线；尚无本次六臂训练、validation 性能或 test 结果，GPU 命令由用户执行，test 继续封存。

六臂启动器随后以**只读模式**完成最终预检，返回 `status=preflight_passed`、`execute=false`、物理 GPU 配置 `6`，六个候选均列出封存命令，且 `test_manifest_read=false`；没有建立模型输出或开始训练。用户可在 GPU6 主机执行 `bash experiments/external_benchmarks/lp_pdbbind_fair_v1/run_gpu6.sh`，该脚本顺序训练六臂并在训练结束后仅导出 validation 带 key 预测；正式 test 仍须单独显式释放。

### 5.45 六外部架构公平重训的当前结果整理（2026-10-06；test 封存）

本节按本地可见的运行产物整理，不代表我在 GPU 主机上启动或控制训练。六臂协议仍是 seed `42`、官方 train/val-only 源（`11,513/2,422` 行，SHA-256 `d57809d39cb68aa7ee84798ebe17b1366db44d6aa9dfab37b6e0bf5e9700f842`），各臂按其 validation CI 选 best checkpoint；未读取 test。

| 模型 | 当前状态 | 实际 train/val 数 | best epoch | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DProtDTA | 完成 700/700 epoch | 10,950 / 2,364 | 390 | 0.654915 | 1.611626* | 1.263095 | 0.453741 | 0.450346 |
| GraphDTA (GIN) | 完成；epoch 33 早停 | 11,229 / 2,404 | 3 | 0.661718 | 1.541497 | 1.221757 | 0.440004 | 0.472199 |
| DeepDTAGen | 完成 500/500 epoch | 11,225 / 2,404 | 44 | 0.622440 | 1.637100 | 1.290453 | 0.372136 | 0.357488 |
| TankBind | 进行中记录；当前仅 interim | 10,949 / 2,362 | 21† | 0.686344 | 1.406238 | 1.103690 | 0.548542 | 0.533171 |
| CASTER-DTA | 尚未开始 | — | — | — | — | — | — | — |
| AttentionMGT-DTA | 尚未开始；输入验收已通过 | 资格账本 9,323 / 2,125‡ | — | — | — | — | — | — |

\* 3DProtDTA 的 `best_metrics.json` 保存 validation MSE；表中 RMSE 为其平方根。† TankBind 的 epoch 21 是当前 `best_metrics.json` 中的 best-so-far，不是最终选模。最近可见的 `train.log` 时间为 2026-10-06 16:36，停在 epoch 25、batch 2700；`history.csv` 只到完整 epoch 24，尚无最终 `fair_run_manifest.json`。因此不能据这份快照断言训练已完成，也不能把 epoch 21 当成最终结果。‡ AttentionMGT 的数量来自已封存原生输入资格账本，表示允许纳入的 train/val key，不是已完成训练的实际样本数。

#### 自研参照：单次训练的最佳独立模型

按用户定义，“独立”指一个亲和力预测网络经一次训练流程完成，不做事后多模型 stacking/ensemble。依当前已完成的自研单网络实验，主比较加入 **FullChainESMMSAAll v2 aligned-MSA**：它从干净随机初始化训练完整预测网络，seed `20261001`，30 epoch/20,700 steps，按 validation CI 选择 epoch 4 EMA；没有转入旧 Stage 1 权重，也没有输出级模型融合。

| 自研单模型参考 | 输入/训练边界 | validation n | best epoch | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullChainESMMSAAll v2 aligned-MSA | 单一训练网络；使用预计算的 ESM/MSA 特征，trainable predictor 随机初始化 | 2,340 | 4 EMA | 0.703450 | 1.298584 | 1.010086 | 0.602351 | 0.573856 |

因此最终 validation 比较应包含该自研单模型参考行，并在共同 key 上统一复算；其预计算 ESM/MSA 输入仍需与各外部模型的输入层级并列披露。Boltz pairwise 是冻结 Boltz 表征上的单独读出，V20 是多阶段融合系统；二者都不作为本次“单个预测网络一次训练”的主参照，V20 尤其不能替代独立模型行。更多来源边界见 `docs/LP_PDBBind_external_fair_test_plan.md`。

三条完成训练的记录分别达到预设训练预算或早停条件；各自的 train/validation 数量并不完全相同。DeepDTAGen 的准备摘要记录 6 个 token/预处理问题，但 validation 仍有 2,404 个样本进入其 best metrics；GraphDTA 和 TankBind 的有效行数可由各自输出的 keyed rows 与缓存 metadata 核对。模型自身 `best_metrics.json` 是当前验证分数的来源，不应直接视为公平横向比较的最终指标：截至本记录，六臂带 key validation 预测归档尚未齐备，统一评估器也尚未生成共同覆盖结果；不同训练入口的指标实现及并列值处理还可能不同。待六臂完成、validation 预测与 provenance 封存后，才可用统一评估器比较原生覆盖和共同 key 子集。

当前能说的是：已完成的三条从头重训均已按各自 validation CI 留存 checkpoint；TankBind 的 best-so-far 记录中 CI/相关性数值较高、RMSE/MAE 较低，但它仍未完训且验证覆盖为 2,362，不能据此宣称六模型胜出。CASTER-DTA 与 AttentionMGT-DTA 尚无本轮 validation 结果。AttentionMGT 使用预训练 ESM2 输入，TankBind 使用原生配体位姿/口袋；加上有效样本覆盖差异，这组实验是统一数据划分下的**完整模型流程比较**，不是输入信息完全相同的纯架构消融。六臂单种子结果也不估计随机种子方差。

尚无本轮六臂 validation 汇总导出、共同 key 评估或 test 预测；test manifest、标签和模型选择均未读取。旧 LP-PDBind test 已被历史实验看过，因此未来 test 共同样本结果只能标为回顾性比较，而非盲测。

### 5.46 TankBind 事后提前停止与六臂续接准备（2026-10-08；尚未由助手执行 GPU）

原封存 `protocol.json` 给 TankBind 的预算为 200 epoch，`early_stop_patience=0`，即关闭早停。用户决定不再为已连续多轮未刷新的 validation CI 跑满余下预算，并由用户自己在 GPU6 终端停止原流程。此决定是**看过训练中 validation 后作出的事后协议变更**，不是预注册早停，也不能将未来的 TankBind 结果称为原协议下的 200/200 epoch 完训。原协议文件、前三臂结果和 test 门禁保持不变。

本地只读核对时，TankBind 的 `history.csv` 连续完整到 epoch 97，最佳为 epoch 65、validation CI `0.693896928348033`，其后连续 32 个完整 epoch 未提升；`best_model.pt`、`last_model.pt`、`training_checkpoint.pt` 均存在，但还没有 `fair_run_manifest.json`。CPU 只读封存预检核对了 history、checkpoint、最佳指标、来源和适配器，返回 `early_stop_ready`；这不是已经写入封存，也不能证明远端 GPU 进程已经停止。若用户实际停止时文件推进到更晚的完整 epoch，正式封存必须重新按当时产物核对；若发生 epoch 边界不一致则拒绝，不会伪造完成轮次。

新增 `scripts/seal_tankbind_fair_early_stop_v1.py` 和 `scripts.continue_lp_external_fair_v1`：前者只在显式 `--execute` 且检查通过后写入带原计划预算、实际完成轮次、最佳 epoch、来源/产物哈希及 `preregistered=false` 的事后提前停止 manifest；后者验证前三臂已有封存，只启动尚无产物的 CASTER-DTA、AttentionMGT-DTA，随后仅导出缺失的 validation keyed 预测。发现未封存的半成品会停止，而不是覆盖或暗中重训。用户先在原终端停止 GPU 任务，再自行执行 `bash experiments/external_benchmarks/lp_pdbbind_fair_v1/continue_gpu6.sh`；该入口在封存和训练前自行执行完整预检，另可用 `python -m scripts.continue_lp_external_fair_v1` 做只读预检。助手不启动、不中断 GPU 任务，不读取 test。

### 5.47 未开训两臂的前瞻性 validation CI 早停（2026-10-08；GPU 未由助手启动）

用户确认前三条已完成模型的结果维持原样，只为尚未开训的 CASTER-DTA 与 AttentionMGT-DTA 增加自动早停。两臂在本次修订时均无输出目录；其已验收训练器 `train_caster_lp.py`、`train_attentionmgt_lp.py` 与 `architecture_acceptance.json` 的哈希保持不变，前三臂及 TankBind 的原有验收/封存不因修改适配器而失效。原 `protocol.json` 仍保留 2000/1000 epoch 上限，早停作为**在这两臂训练前确定、但不属于最初六臂协议的修订**单独记入新 run manifest，不能称最初协议已预注册。

固定规则：按原有 validation CI 选模，严格提高才重置计数、并列不重置；连续 100 次已完成的 validation 检查不提高，即在确认同一完整 epoch 的 history、`training_checkpoint.pt`、`last_model.pt` 和最佳 checkpoint/metrics 一致后，由外层监督器终止它自己启动的训练进程组。若先达到原最大 epoch 则正常完成。原训练器的 CASTER 50 次和 AttentionMGT 80 次 `ReduceLROnPlateau` 耐心不变；100 次 CI 耐心给平台期调整留出时间，但不保证调度器一定触发。中断、未封存半成品、checkpoint 不一致不会伪装成成功完训。早停和停止原因、实际/最佳 epoch、产物哈希均写入该臂 `protocol_amendment`；test 不参与计数或选模。GPU 命令继续由用户执行，仍为 `bash experiments/external_benchmarks/lp_pdbbind_fair_v1/continue_gpu6.sh`。

### 5.48 六臂训练封存与 DeepDTAGen validation 导出修复（2026-10-15；test 仍封存）

六条训练 run manifest 已存在。TankBind 以事后提前停止封存于完整 epoch 98，最佳 epoch 65、validation CI `0.693896928348033`；CASTER-DTA 按训练前固定的 100 次 validation-CI 无提升规则停于 epoch 121，最佳 epoch 21、CI `0.664241550227596`；AttentionMGT-DTA 同规则停于 epoch 103，最佳 epoch 3、CI `0.6065266284895251`。这些是各训练器原生 validation 数值，不是六模型共同 key 的统一复算或 test 结果。

续接导出时，3DProtDTA 与 GraphDTA 的 keyed validation 预测及 provenance 已写出；DeepDTAGen 导出在读取训练时保存的 tokenizer pickle 时失败，报 `AttributeError: Can't get attribute 'Tokenizer' on <module 'utils' (namespace)>`。原因是仓库根目录同名 `utils/` 命名空间与上游 `external_benchmarks/DeepDTAGen/utils.py` 冲突，而导出器此前在 `pickle.load()` **之后**才安装上游路径。仅修改导出器：在解 pickle 前安装并核对上游 `utils.py` 来源，必要时替换已经导入的空 namespace；若是另一份真实 `utils.py` 则拒绝，不静默替换。训练器、权重、已完成两份预测、协议和 test 门禁均不修改。DeepDTAGen 失败时尚未写出 `valid_predictions.npz` 或 provenance，重跑续接入口会验证并跳过前两份已有归档，再从 DeepDTAGen 继续；在六份 validation 输出全部封存前，不做共同结果或 test。

### 5.49 六外部模型与独立单模型的共同 validation 核对、test 输入释放（2026-10-16）

六份带 key validation 预测及 provenance 均已封存。将六个外部候选与预先指定的独立参照 FullChainESMMSAAll v2 aligned-MSA 按规范 validation manifest 对齐，七者共同覆盖 **2,105** 条；各归档中存在的标签与规范 manifest 相符。以下五项均由同一项目指标函数在**同一 2,105 条**上复算，不是训练日志的原生覆盖分数：

| 候选 | 原生预测 n | 共同 CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DProtDTA | 2,326 | 0.651811 | 1.616199 | 1.266197 | 0.446047 | 0.441942 |
| GraphDTA GIN | 2,326 | 0.660981 | 1.520009 | 1.203856 | 0.440324 | 0.469419 |
| DeepDTAGen | 2,326 | 0.622455 | 1.618936 | 1.271610 | 0.375340 | 0.357297 |
| TankBind | 2,324 | 0.691345 | 1.569259 | 1.208997 | 0.567242 | 0.543425 |
| CASTER-DTA | 2,316 | 0.660639 | 1.472473 | 1.156126 | 0.478616 | 0.463716 |
| AttentionMGT-DTA | 2,115 | 0.605751 | 1.529426 | 1.207391 | 0.366469 | 0.310445 |
| FullChainESMMSAAll v2 aligned-MSA（单模型参照） | 2,340 | 0.703112 | 1.305138 | 1.011539 | 0.604401 | 0.572202 |

这说明在固定的共同 validation 子集上，该独立参照优于六个外部候选；但它使用预计算的全链 ESM/MSA，模型的训练池为规范 LMDB 2,340 条 validation 对应的划分，外部六臂则从官方名义 train/val CSV 各自可用样本训练，不能据此声称等输入、等训练样本的纯架构优越性。单种子也不表征训练稳定性。其 checkpoint 仍为 validation gate 未通过的研究候选，不能写成 V20 晋级。

在既有六臂 checkpoint/validation seal 后，已按事先固定的规范 test manifest SHA `f8741d40837568c552ca0bd3725e0b079ba855d56d2a663c0bb95c4bc09c95b8` 显式释放无标签 `test_input.csv`：4,751 key、SHA-256 `86abb653714b3accdde20294129adcbd32685329345a1ed76151b1f16bc91a7e`，只含 `pdb_id,new_split,date,seq,smiles`；原始 CSV 的 21 条空 SMILES 保留并在 `.source.json` 列出，不补造或删行。4,751 个 key 均在源链 LMDB；这不等于每个模型都能构造有效输入。释放器打开规范 test manifest 核 key，但没有把标签写入模型输入。

用户已在 GPU6 完成 AttentionMGT 原生 ESM2 320-D test 特征提取；`.codex_runtime/attentionmgt_native_esm320_test_v1/attentionmgt_source.json` SHA-256 为 `775d41100f63245e67173424b11e672a3a644b5cbdf2efb00edce044242b1791`，记录 `test_count=protein_key_count=4,751`、5,253 条链、零个源链缺失、`test_labels_read=false`、`test_manifest_read=false`。它与释放 CSV、训练侧提取器/权重/窗口协议的身份字段相符；用户交接片段和本地清单支持“提取产物已写出”，但本轮助手为避免与后续 GPU 推理争抢源 LMDB I/O，未独立跑完逐 key 源 payload 全量哈希复核；AttentionMGT 导出器仍会执行该校验。**尚未执行**六臂 test 推理、正式 test 指标或独立单模型 test 推理。GPU6 命令仍由用户运行；评估须等七者预测来源封存后统一进行，不用 test 分数改候选或 checkpoint。独立单模型的 test 完整链 MSA 需要 2,012 个序列 ID，其中 1,825 个不在既有 train/valid MSA 特征缓存，不能用六臂已就绪代替它的 test-only 链路。

### 5.50 六外部模型无标签 test 导出与独立单模型 test-only 预检（2026-10-17；尚无 test 指标）

用户已在 GPU6 执行六臂 test 导出；六份 `test_predictions.npz` 及 provenance 均存在。来源门禁 `verify_sealed_six_test_inputs()` 已通过，且未读取 test 标签。按 4,751 个已释放的无标签 key，各模型实际预测覆盖为：3DProtDTA 4,730、GraphDTA 4,730、DeepDTAGen 4,730、TankBind 4,751、CASTER-DTA 4,687、AttentionMGT-DTA 4,310；六臂共同 key 为 4,291。AttentionMGT 排除 441 条，其中 420 条因残基映射无法满足其禁止 unmatched-residue 零嵌入的 fair 输入规则，21 条为原始空 SMILES；这些不能被补零后伪称完整覆盖。此处只有无标签预测与覆盖统计，**没有 test 性能分数**。

预先固定的独立单模型参照仍是 `FullChainESMMSAAll v2 aligned-MSA` 的 epoch 4 EMA checkpoint，SHA-256 `da4f99714113525be239811e7068bf73219d0e474e580a51e83db0e3056852f7`，不因 test 预测更换。其 test-only CPU `prepare` 已成功：释放输入 4,751 key 对应完整链 2,012 个序列 ID，来源 manifest SHA-256 `4f212b050eeb4e658d8438af9f6d09d516ff5c763832a616ff1b27c1537618a4`，与训练侧 A3M/编码器来源核对通过，`test_labels_read=false`、`test_manifest_read=false`、`training_run=false`。下一步由用户运行 GPU6 完整链 MSA 提取，再运行该固定 checkpoint 的 GPU6 无标签预测；七份预测全部封存后，才运行 CPU 七模型正式 test 评估。评估将同时报告原生覆盖和共同 key 指标，并标明这是历史 test 已被查看过、不同输入条件与训练池的**回顾性比较**，不是盲测或等输入的纯架构排名。历史链 LMDB 的 pickle 记录物理上含 `affinity` 字段；独立模型 test-only 预测只取结构/ESM 输入，使用字面量零 dummy 标签，不读取或传递该字段，不能把这一点表述成源记录字节级零接触标签。

2026-10-19 进度：用户已在 GPU6 完成上述独立单模型的 `aligned_msa` test 特征提取。落盘 metadata 为 `status=built`、目标/新写入 `2012/2012`、复用 0，`data.mdb` SHA-256 `4f8e10a0b5434d481774d9b5a2233297104f3d8053d853f840e4b7ff0b65eb54`；本地只读复核确认 metadata 的来源清单、释放 CSV 和编码器权重与 preflight 一致，`test_labels_read=false`、`test_manifest_read=false`。此时仍未生成独立单模型 test 预测，也未运行七模型 test 评分；下一条 GPU6 操作仅为固定 checkpoint 的无标签预测。

### 5.51 七候选共同 test 评估（2026-10-19；回顾性）

用户随后在 GPU6 完成固定独立单模型的无标签预测：4,751/4,751 key，预测归档 SHA-256 `b5e21d1a5e174d9d1566b57dd749daebfd66ac5058d205713dc907e8b775b37b`，仍使用上述 epoch 4 EMA checkpoint，来源侧记录 `test_labels_read=false`、`test_manifest_read=false`、`test_used_for_selection=false`。在七份预测均齐备后，CPU 评估器先通过六外部模型与独立模型的归档/来源门禁，再读取固定规范 test manifest 计算指标；没有依据 test 修改候选、权重或协议。结果文件为 `Model/LP_PDBBind_ExternalFair_v1/sealed_seven_test_20261017.json`，SHA-256 `3564b2b01e37ffa7ecceb5231000f3551c97d73827541fcb3a4750cf8f4388fc`，状态 `retrospective_common_key_evaluation_complete`。规范 test 为 4,751 条；七者共同 key 为 4,291，key 集合 SHA-256 `c4a475c98d08b6c347c8fc0c4298c7d2c4197e7400bc47733b7c98087df0517f`。下表五项指标均在**同一 4,291 条**上统一复算，原生预测 n 只表示各模型覆盖范围，不用于跨行混合比较。

| 候选 | 原生预测 n | 共同 CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DProtDTA | 4,730 | 0.645769 | 1.777866 | 1.353088 | 0.419061 | 0.416135 |
| GraphDTA GIN | 4,730 | 0.648779 | 1.611508 | 1.270908 | 0.396198 | 0.438160 |
| DeepDTAGen | 4,730 | 0.613192 | 1.725266 | 1.362701 | 0.354819 | 0.329803 |
| TankBind | 4,751 | 0.683456 | 1.594229 | 1.255389 | 0.559531 | 0.524701 |
| CASTER-DTA | 4,687 | 0.660613 | 1.524568 | 1.219446 | 0.482481 | 0.466094 |
| AttentionMGT-DTA | 4,310 | 0.578046 | 1.770995 | 1.372703 | 0.239911 | 0.233039 |
| FullChainESMMSAAll v2 aligned-MSA（独立单模型） | 4,751 | **0.693874** | **1.411701** | **1.109051** | **0.584885** | **0.550258** |

独立单模型在固定共同子集的五项指标均优于本次六个外部候选。该结论仅限这些已封存的单种子训练和输入条件；各模型原生覆盖、训练池和输入表征不同，自研参照使用预训练全链 ESM/MSA，不能宣称等输入、纯架构或统计稳定性上的胜出。历史 LP test 已在更早实验中被查看，故报告明确为**非盲的回顾性比较**。独立模型在 validation 阶段未通过 V20 晋级门槛，本次 test 表格也不改变其候选身份或触发新的 test 驱动选模。

### 5.52 完整链 v3、ESM-only 与 MSA pair-bias 的同 key validation 对照（2026-10-08—19）

本节补录 5.42.10 之后完成的完整链单网络对照，统一使用原 train `11,037` / validation `2,340`、seed `20261001` 和预先固定的 validation 选模口径；train/valid manifest SHA-256 分别为 `def47b0307750099bcd8accc17ac5c6960d4d92ffc789daac2cadb10faf55213` / `dc996a2665a07a3a708b7a5b7624f27f8ba3bde881bd18fe4569a2d65e3943d8`。四个新训练 run 均完成 30 epoch、`20,700` optimizer steps，从干净随机预测网络初始化；下表是各自已选中的 EMA checkpoint 在**相同 2,340 个 key 和标签**上的归档预测复算。v2 和 V20 只作参照，V20 为多阶段融合系统，并非同一单模型控制变量。

| 候选 | 本组相对 v3 的主要变量 | 选中 epoch | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FullChainESMMSAAll v2 aligned-MSA（既有参照） | v2 原配方 | 4 EMA | 0.703450 | 1.298584 | 1.010086 | 0.602351 | 0.573856 |
| FullChainESMMSAAll v3 aligned-MSA | v3 dropout `0.20` | 3 EMA | 0.705546 | 1.289022 | 1.001342 | 0.605028 | 0.580166 |
| FullChainESMOnly v1 | 不输入完整链 MSA | 2 EMA | 0.701382 | 1.295277 | 1.005807 | 0.597305 | 0.569301 |
| FullChainMSAPairBias v1 | v3 外加预计算 MSA 行注意力 PP bias | 4 EMA | 0.705909 | 1.289025 | 0.999031 | 0.606674 | 0.580233 |
| FullChain v3 affinity-only v1 | v3 七项辅助损失整组关闭；见 5.54 | 2 EMA | 0.706457 | 1.288747 | 1.001928 | 0.608739 | 0.583349 |
| 历史 V20 validation 目标（非单网络对照） | 多阶段融合 | — | 0.737938 | 1.172255 | 0.914334 | 0.686389 | 0.659726 |

v3 对 v2 的五项均有改善（CI `+0.002096`、RMSE `−0.009562`）；v3 对 ESM-only 的五项也均较好（CI `+0.004164`、RMSE `−0.006255`、MAE `−0.004465`、Pearson `+0.007723`、Spearman `+0.010865`）。这说明本次固定配方的 aligned-MSA 提供了小幅增量，不等于证明该收益跨种子稳定，也不等于 MSA 对 V20 融合系统没有作用。v3/ESM-only 的正式结果分别在 `Model/LP_PDBBind_FullChainESMMSAAll_v3/fullchain_all_v3_aligned_b16_20261008_191746/summary.json`（SHA-256 `08a340d16b4dd5502d5e11ab0b7d0aa79e079d12c1720ea135537707a1e334cd`）与 `Model/LP_PDBBind_FullChainESMOnly_v1/fullchain_esm_only_train_b16_20261009_144435/summary.json`（SHA-256 `ffaead573c5c2dd8b27139b7018b2538dc254d760682e359315ef013c72f7e50`）。各自真实 GPU smoke 状态均为 `real_gpu_smoke_passed`，但 smoke 不提供 validation 性能。

### 5.53 全链 MSA pair-bias P2/P3 缓存与正式训练（2026-10-10—19）

P2 的三个 train key 有界真实 GPU smoke 返回 `p2_bounded_real_gpu_smoke_passed`，未存训练 checkpoint、未算 validation。P3 先按 train/validation 结构请求准备，再由用户在 GPU6 提取完整链 MSA Transformer 行注意力，经对称化和 APC 形成 144 通道的 pocket×pocket 关系特征；旧完整链节点缓存不被替换，旧口袋 MSA 不重新进入局部分支。P3 审计为 `p3_train_valid_full_cache_coverage_passed`：`8,863` 条序列、train/validation `11,037/2,340` 样本、请求去重 pair `18,208,609`、有共同窗口可观测 pair `18,112,598`；差额 `96,011` 是按窗口定义不可共观测的请求，不是提取失败。审计文件 `.codex_runtime/fullchain-msa-pair-bias-p3-gpu6-20261014/audit.json` SHA-256 为 `5ec7c57a7fa28c56a842dca3c36069e534e283b88e3add5ac8af7a240f554692`，`test_manifest_read=false`。这是缓存/索引覆盖，不是亲和力效果。

随后真实训练 smoke 返回 `real_gpu_smoke_passed`，新增 `msa_pair_projection.weight` 梯度非零，checkpoint 重载预测差为零。正式 pair-bias 单网络训练完成 `20,700` 步，选择 epoch 4 EMA；结果 `Model/LP_PDBBind_FullChainMSAPairBias_v1/pair_bias_full_b16_20261019_113922/summary.json` SHA-256 为 `f3caed693b50547e65d922d0e3e9d0057d64cd7d7e21c997ba7f6774c7bb7872`。与 v3 同 key 复算，CI `+0.000363`、MAE `−0.002312`、Pearson `+0.001646`、Spearman `+0.000067`，RMSE `+0.000003`（越低越好，数值上略差）。这些差异很小，不能宣称 pair-bias 恢复 V20 或证明 AF3 类 pair 表征的效力；本实现也不是动态 Pairformer。

### 5.54 v3 七项辅助损失整组关闭对照（2026-10-20—21）

唯一干预是把 contact、residue-contact、affinity-class、distogram、interface-distogram、interface-contact、match 七项辅助损失的有效训练系数统一设为零；保留 SmoothL1＋`0.15` MSE＋`0.40` 原 rank loss。模型图、完整链 ESM/aligned-MSA 输入、初始化指纹、seed、dropout、遮蔽、优化器、EMA、30 epoch/`20,700` 步及选模规则沿用 v3；不接入 pair-bias，不加载 v3 最佳权重。用户在物理 GPU6 完成训练，选中 epoch 2 EMA，checkpoint 重载预测最大差 `0.0`。结果 `Model/LP_PDBBind_FullChainV3AffinityOnly_v1/fullchain_v3_affinity_only_b16_20261020_191527/summary.json` SHA-256 为 `1e4b50f461ed7373bf120dcff3ac8b7b4f61ead26885145ffac321d5dbc62b84`；同 run 的 `aligned_msa/history.json` SHA-256 为 `79576c6797b1863071f900cc6b46ad37afb2cd655597ea1624f0ff0a961cdab8`。两份小文件及说明已单独置于审阅分支 `review/fullchain-v3-aux-loss-off-20261020` 的 `experiments/lp_pdbbind/fullchain_v3_affinity_only_v1/review_evidence/fullchain_v3_affinity_only_b16_20261020_191527/`；权重和预测归档未上传。

与 v3 相同 2,340 validation key/标签及归档哈希核对后，候选 CI `+0.000911`、RMSE `−0.000275`、Pearson `+0.003711`、Spearman `+0.003183`，但 MAE `+0.000586`（变差）。训练已完成；臂内 `validation_failed` 表示未通过 V20 五项晋级 gate，不是运行异常。单种子和这样小的差异不能证明辅助监督是瓶颈，更不能宣称恢复历史最佳。上述四个新候选均未通过 V20 gate；它们没有读取 test manifest，也没有用 test 选模。5.51 的回顾性 test 仅对应预先固定的 **v2** 参照及六外部模型，不能移植给 v3、ESM-only、pair-bias 或本次 affinity-only 候选。上述 `Model/` 和 `.codex_runtime/` 路径是本地原始产物定位，除注明的少量审阅证据外，不表示权重、缓存或预测归档已上传 GitHub。

### 5.54.1 补录四层 affinity-only、固定 checkpoint 诊断与 SAM（2026-10-21—24）

四层 affinity-only AdamW 与 SAM 均已完成30轮／20,700次更新，使用原LP validation2,340条，重载预测差0。AdamW选epoch2 EMA：CI `0.707077`／RMSE `1.281549`／MAE `0.993971`／Pearson `0.608429`／Spearman `0.585471`。SAM选epoch4 EMA：`0.698825`／`1.309870`／`1.017246`／`0.584349`／`0.561337`，五项均差于AdamW；SAM有41,400次backward而非41,400次参数更新。两者臂内`validation_failed`为V20 gate未通过，非运行失败。

固定AdamW checkpoint全train eval n11,037已完成，CI `0.772790`／RMSE `1.285558`／MAE `0.987886`／Pearson `0.731496`／Spearman `0.735415`；状态指纹不变、12条validation预检差0、梯度／优化器步0。Train排序好而绝对误差接近validation，不当作独立泛化成绩。原运行与summary/history、SAM比较、GPU固定checkpoint诊断摘要均收于[补录第1节](LP_PDBBind_recent_experiments_20261108.md#1-原-lp四层sam-和固定-checkpoint-诊断)及本次小文件归档。

### 5.55 Selector-only 完整链 v2 四臂完训与 query-only 锚点阻断（2026-10-27复核）

本次实验分支 `review/fullchain-msa-selector-diversity-20261024` 从已核实的 `66fdb1fb6a8de223badd0772f6e3a02b2a11cc80` 创建；成功 GPU 阶段运行 `f4cc212be12c20e7dc6d5fffd33f106bcf7a6cd8`。仅改变7条homolog的 deterministic selector（旧source-order vs 前256候选greedy max-min），不改v2模型/原多任务loss/seed/split/30epoch/EMA/选模。用户已于10月26日在同一GPU6完成两臂各query-only/aligned-MSA，共四个arm、82,800步；两份完整cache audit和smoke通过，训练前逐链query-only embedding在8,863条链上完全相等。以下是相同2,340 validation key/标签上的项目指标复算，**只作描述，不是可归因selector效应**：

| Arm | EMA epoch | CI | RMSE | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old selector query-only | 4 | 0.702783 | 1.304562 | 1.012524 | 0.597555 | 0.572661 |
| New selector query-only | 2 | 0.702239 | 1.297418 | 1.008599 | 0.595592 | 0.571937 |
| Old selector aligned-MSA | 3 | 0.704539 | 1.289208 | 1.004674 | 0.604358 | 0.577144 |
| New selector aligned-MSA | 4 | 0.700401 | 1.307749 | 1.013643 | 0.592960 | 0.566179 |

比较器返回 `blocked_query_only_anchor` / exit2：两个query-only归档有2,324/2,340条预测差超过预注册1e-6，平均绝对差0.218943、最大1.185078；初始网络指纹相同，但history在epoch0的train loss已分叉（1.266780 vs1.274039）。各自checkpoint重载delta0不能替代两次独立训练一致。当前runner重置随机种子，但未建立/记录完整CUDA确定性及逐step输入/RNG/梯度轨迹；没有发现明确的split/config/key/label污染证据，也尚不能确定具体算子、BF16或共卡为根因。

旧aligned减新aligned的描述性MAE gain为-0.008970，500次/seed20261024、1,461components的95%CI为[-0.026219,0.007315]；RMSE gain为-0.018541、CI[-0.036349,-0.002168]。即使RMSE区间不跨0，component重采样也没有估计独立训练变异，不能在锚点失败后宣布selector变差或有效。质量/fallback分层按门禁跳过。**结论：证据不足**；不放宽阈值、不覆盖结果、不自动重训或进入geometry实验。下一步只建议另行确认有界query-only首次分叉step诊断；GPU仍由用户运行。

完整报告、两份summary、四份history/selection、15份stage manifest和CPU审计已放入 [selector完成报告](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/completed_run_review_20261027/validation_selector_report.md)；45项原始日志checksum和4份生成predictionNPZ/2份smoke checksum通过。只发布小型生成证据，不上传数据/预测NPZ/特征/权重。本轮未读取test manifest/LMDB或A3M数据库、未全库哈希、未启动GPU，未修改外部六模型目录或主工作区既有修改。

### 5.56 Query-only 首次分叉有界诊断准备（2026-10-27）

用户已批准继续定位5.55的query-only首次分叉。新增独立诊断入口，不改原训练函数/模型/数据或seal，直接用AST定位的Python行追踪观察原v2 `_fit_arm`。baseline/treatment分别在全新进程执行前16步，沿用seed20261001、b16/workers4/BF16和原完整20,700步scheduler；逐步记录完整batch、RNG、mask、所有输出/loss、clip前后梯度及AdamW/EMA状态指纹。首次精确字节差异与affinity单位预测差首次超过1e-6分别报告，不执行validation推理或保存checkpoint。

实际CPU首批接线已通过：原train/validation元数据11,037/2,340、两份selector source seal、smoke/summary lineage均通过；两臂首批16条完整输入逐字段一致。定向CPU回归289 passed、5条既有Transformer warnings，compileall和bash -n通过。CPU数值/RNG等价性和失败路径测试见[诊断操作说明](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/query_only_first_divergence_20261027/README.md)及同目录`cpu_checks.json`。GPU6命令已准备，每臂最多16步/600秒追踪、780秒硬上限；由用户执行。当前没有GPU首次分叉结果，不改变5.55的“证据不足”，也没有重跑四臂正式训练。

启动后发现计算节点Git过旧、不支持`git -C`，失败发生在输出目录创建/GPU启动之前。现仅将脚本定位代码根目录的方式改为子shell内`cd`后执行`git rev-parse`；训练/追踪代码和预算不变。新增旧Git/异目录启动派发测试，相关CPU回归32 passed，bash -n和diff检查通过；同一GPU6命令可重试。

### 5.57 Query-only 有界GPU诊断：首步梯度分叉（2026-10-27）

用户使用`ac6a55c1f684869814bd92edea28976d9e551692`在gpu10/GPU6 A100-40GB完成两臂各16步，stage exit0、195.446秒；未执行validation或保存checkpoint。两臂各98个事件，全部16批完整输入及mask后输入相同，全部98边界的Python/NumPy/Torch CPU/CUDA RNG相同，初始模型/配置/encoder/backend一致。首次可观测差异在第1步backward后、clip前：46个有限float32梯度张量不同；clip后290个梯度及1个norm字段不同；第一次AdamW/EMA更新后104个模型、56个EMA、580个optimizer状态张量不同。第1–5步所有输出和loss仍完全相同；第6步首次出现affinity预测差，2/16超过1e-6，最大0.002520955821976756；第7步最大0.0036668448319661096。

CPU完整复算与原aggregate完全相同，2份trace全文hash、196个事件描述符和3项原日志checksum通过，11份原始小文件逐字节归档。结果支持当前训练在相同已记录输入/RNG下出现不可逐位复现的梯度；不能归因为selector改变了query-only输入，也未定位具体CUDA算子、BF16或共卡原因，不能据此判断MSA数据库/融合效果。原selector“证据不足”和blocked anchor保留。唯一建议下一步是另立有界严格确定性首批复测，fail-fast定位不支持确定性计算的算子；本轮未实施、未自动重训。详细[诊断完成报告](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/query_only_first_divergence_20261027/completed_run_review_20261027/README.md)及同目录aggregate/provenance/日志已归档；本地逐样本trace不上传。

### 5.58 Query-only 严格确定性有界复测准备（2026-10-27；GPU未运行）

用户批准继续5.57后的下一步。新增独立strict入口：Python/Torch导入前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`；两个全新worker各自启用`torch.use_deterministic_algorithms(True, warn_only=False)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`，并核对实际记录。保持v2原模型/loss/seed/b16/workers4/BF16、遮蔽、优化器/EMA、20,700步scheduler、source seal和比较阈值；不手动切换TF32/matmul precision/SDPA。普通诊断仅新增worker模块派发和失败上下文记录，默认数值路径不变。

CPU回归`333 passed, 5 warnings`（既有Transformer提示），compileall、bash -n通过。实际两臂首批16条train接线及原seal/smoke/summary来源检查通过，train/valid元数据仍为11,037/2,340；两臂CPU完整batch SHA均为`8c83456382e4635aa7175e565a00df424f36a48e93ef78d6765d5f40f7b0245a`，与上一轮CPU一致。和上一轮GPU首批比较，所有张量描述符及逐key值哈希相同；仅`keys`容器因pin-memory从tuple变成list，未改输入或GPU两臂比较规则。

GPU6操作脚本已准备，每次创建全新结果目录；顺序运行两臂各最多16步/600秒追踪、780秒worker硬上限，失败立即停止并保留完整traceback/最后事件，不回退、不重试，不执行validation或保存checkpoint。严格模式未实际运行，具体CUDA算子仍未定位，原selector“证据不足”不变。唯一下一行动是用户执行[严格确定性操作说明](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/query_only_strict_determinism_20261027/README.md)中的GPU6命令，返回`strict_result.json`；确定性算子报错属于诊断发现，不是训练成功。本轮未读test/A3M数据库、未全库哈希、未修改外部模型或原结果。

### 5.59 Query-only 严格确定性 GPU 复测：16 步完全一致（2026-10-27）

用户在 gpu10/物理 GPU6 A100-40GB 完成两臂各16步，运行代码`87003e0ba619959c697bd657b1a45665768a2350`，stage及两worker均exit0，stage耗时312.818531秒。结果`strict_determinism_reproduced_in_bound`：两臂各98个事件全部匹配，16步预测/loss差均0，clip前后梯度及model/AdamW/scheduler/EMA状态均无已记录差异，两份完整trace SHA均为`772b0d12e0ad008de4e4b5c495b8f51b892479344997481cfd73fbc89b682f60`。没有触发不支持确定性实现的算子报错，没有validation推理或保存checkpoint。

跨5.57普通模式与本次strict同名arm，16批完整输入、16批mask后输入、98个主进程/当前CUDA RNG边界及初始化payload相同；provenance仅backend/git_commit不同，原训练文件/配置/encoder身份/小输入哈希均相同。实际后端变化是deterministic_algorithms与cudnn_deterministic从false到true，CUBLAS_WORKSPACE_CONFIG从null到`:4096:8`；BF16、TF32与SDPA开关未改。两次Git提交不同，不能声称全部源码逐字相同。GPU仍为获准共卡而非独占，普通195秒/严格313秒不是工程提速或降速的受控基准。

CPU完整复算与原aggregate、strict_result一致，196个事件描述符和3项原日志checksum通过；12份原始小文件逐字节归档，回归`52 passed`（15.05秒）。[严格复测完成报告](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/query_only_strict_determinism_20261027/completed_run_review_20261027/README.md)附原strict_result、provenance、summary及stdout/stderr；约15MB×2的逐样本trace留本地不上传。支持严格配置组合在本次16步界内消除分叉，不定位具体CUDA算子、不证明完整20,700步一致，也不提供新的validation或MSA收益。原selector blocked anchor和“证据不足”不追认改变。唯一建议下一行动是准备同一strict配置的完整query-only双臂锚点复测；本轮未实施/启动长训练，未读test/A3M/原数据权重或触碰外部模型。

### 5.60 严格配置完整 query-only 双臂锚点复测准备（2026-10-27；GPU未运行）

用户批准继续5.59后的完整query-only锚点准备。新独立入口在全新目录顺序执行baseline/treatment两次query-only，每臂30epoch、20,700步（合计41,400），仍从干净随机初始化，不训练aligned-MSA、不混入旧aligned预测。保留原v2完整链输入、seed20261001、b16/eval6/workers4/BF16、dropout0.10、原loss/AdamW/EMA、V20 gate/选模和reload校验；不改冻结训练源或缓存。两个fresh worker均在strict context内直接运行原`_fit_arm`，不装tracer；CUBLAS/确定性配置沿用成功16步proof，版本/TF32/precision/SDPA还与proof绑定，实际backend前后/跨臂记录并校验。

真实CPU前置检查通过：重算strict proof完整两臂trace，原source seals/smoke v2协议/V20 validation参照相符，train/valid元数据11,037/2,340、两臂各16条首批输入相同。CPU回归`528 passed, 5 warnings`，41.81秒；compileall、bash -n通过。开发中一轮旧测试错误文案正则失败已保留于cpu_checks，最终完整回归通过，未放宽任何阈值。

新比较器只消费两份新query-only完整结果，在2,340个相同valid key/label上保留原逐条绝对差<=1e-6预测锚点；history30epoch/选中epoch与variant一致性单列，完整诊断成功要求两者都过。V20 gate的validation_failed为性能结果，不是复测运行异常；不因旧非strict指标高低改选模型。GPU6脚本已备，每臂24小时硬限、失败不降级/不重试，原smoke+strict16步proof作为前置；完整strict validation/reload尚待本次GPU覆盖。详见[完整锚点操作说明](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/query_only_full_anchor_20261027/README.md)和同目录CPU证据。GPU尚未执行，无新validation或完整可复现结论；旧selector blocked与“证据不足”仍保留。本轮未读test/A3M、未全库hash、未修改外部训练或主工作区既有改动。唯一下一行动是用户运行该GPU6命令并返回anchor_result路径。

### 5.61 严格配置完整 query-only 锚点通过（2026-10-28复核）

用户在gpu10/物理GPU6 A100-40GB完成5.60准备的两次独立完整query-only训练，运行提交`4ff08e5651cf746666bf37b82ef0ceb190d77d07`，原目录`.codex_runtime/query-only-full-anchor-20261027-gpu6.VuBkzc`。每臂30epoch、20,700步，总41,400步；parent及两个fresh worker均exit0。沿用v2模型/原多任务loss、完整链ESM/query-only MSA、干净随机初始化、seed20261001、b16/eval6/workers4/BF16/dropout0.10及原V20参照选模，不训练aligned-MSA。

CPU原项目指标及比较器复算与原`anchor_result.json`一致：`query_only_full_anchor_passed`；同2,340个validation key、跨归档标签严格相同，预测最大/平均绝对差均0，超原1e-6阈值数量0。两份30epoch history逐字段相同，stdout完整60条epoch事件与两份history对应；两臂均选epoch4 EMA，reload预测差0。两臂相同validation结果为CI`0.7056825864240892`、RMSE`1.2876625445154002`、MAE`1.0005012356829803`、Pearson`0.6079186525447589`、Spearman`0.5798868819227323`。原逐条预测anchor和额外完整history/选模一致性分别通过；两臂的`validation_failed`表示未达V20五项gate，不是训练失败。

两臂训练前后四份backend完全一致，strict配置不变，BF16/TF32/SDPA未改。原stage的stdout/stderr/manifest三项checksum通过；Asia/Shanghai运行时间为10月27日21:37:54.775723至10月28日07:11:53.276904，总34,438.501181秒（约9小时34分钟），两臂summary分别17,284.385774秒和17,042.318124秒。获准共卡，不构成与普通模式的速度基准。checkpoint记录哈希不同，本次未读取/重哈希权重或记录逐步状态，不宣称完整模型/优化器状态逐位一致。原summary/provenance的`validation_inference:false`是输入前置元数据遗留；实际已执行epoch validation及checkpoint重载推理。保留原文件，审计单列说明。

[完整锚点完成报告](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/query_only_full_anchor_20261027/completed_run_review_20261028/README.md)归档15份原始小文件、CPU审计、清单及checksum，原件共92,452字节且逐字节核对一致；本轮定向CPU回归195 passed（59.03秒），compileall/bash -n通过。权重/预测NPZ/数据缓存未上传。结果将严格配置可复现证据从16步扩展到本次完整history/选模/validation输出，但不追认5.55旧四臂比较，也没有新的selector/MSA收益结论。当前selector效果仍为“证据不足”。唯一建议下一步是另行获准后准备同strict配置的新baseline/treatment aligned-MSA对照；本轮未实施或启动，未读test/A3M、全库hash或修改外部六模型目录。

### 5.62 同strict配置的新旧selector aligned-MSA对照准备（2026-10-28；GPU未运行）

用户确认继续5.61后的受控selector比较。起点`65a5065ddc20f72d7c7e5b3b5fa63d3cd7e9bf5f`，仍用`review/fullchain-msa-selector-diversity-20261024`。新入口只补跑baseline/treatment两个aligned-MSA臂，不再重复query-only；两臂各30epoch/20,700步，分别从干净随机初始化调用原v2 `_fit_arm`。模型/完整链输入、dropout0.10、原多任务loss、seed20261001、b16/eval6/workers4/BF16、AdamW/EMA/scheduler、划分和V20参照选模不变；不加载旧checkpoint，不读旧非strict aligned预测。复用并重验原独立selector sealed缓存及各自实际mode=aligned_msa的smoke；不改变selector或重新提取缓存。新worker的strict backend须与5.61完整锚点相同且全过程不变。GPU仍由用户在物理6号卡执行。

真实CPU preflight通过：重算并绑定5.61完整query-only锚点，原来源/encoder/seal及train/valid11,037/2,340一致；两臂各16条train首批仅`$/fullchain_msa`值不同，其余完整字段相同，MSA为有限float32 `[16,1011,768]`。定向CPU回归`645 passed, 5 warnings`（48.91秒，既有Transformer提示），包括原模型/原loss的小型forward/backward、117项新增入口/比较器/launcher检查；compileall、bash -n通过。原训练源SHA仍为`383f40d7c910ed41a746bb3c639e02aee60421ea8582a57c8ee0559522c807f3`。详见[操作说明与CPU证据](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/strict_aligned_pair_20261028/README.md)。

新比较器允许aligned臂的预测/history/选中epoch不同，先验证各自完整训练及相同控制项，再原样调用旧selector比较器：相同2,340 validation key/标签、原逐条query锚点<=1e-6、500次/seed20261024 component bootstrap、原质量分层/fallback/changed分组。额外描述每个新aligned相对同strict query-only的指标增量；比较完成不等于性能提升，V20 `validation_failed`不等于运行失败。用户执行脚本后生成新`.codex_runtime/selector-strict-aligned-20261028-gpu6.XXXXXX`和完整日志；每臂24小时上限，失败不降级/不重试。最近strict双query约9小时34分钟仅作耗时参考。旧aligned smoke为普通后端，新strict aligned完整validation/reload仍待本次GPU覆盖，当前无新性能结果或selector收益结论。本轮未运行GPU、未读取test/A3M、未全库hash或修改外部模型目录；仅推送实现和小型CPU证据。

### 5.63 Strict aligned-MSA selector对照完成：未获得稳定提升（2026-10-29复核）

用户在gpu10/物理GPU6 A100-40GB完成5.62准备的旧selector baseline和新selector treatment，运行提交`5e2a819026b327c800888767f45375d11391b39e`，原目录`.codex_runtime/selector-strict-aligned-20261028-gpu6.q3y1XH`。每臂30epoch、20,700步、干净随机初始化；不重训query-only、不读旧非strict aligned预测。parent与两个fresh worker均exit0，实际strict aligned训练/validation/reload已覆盖；两臂backend前后及相对query-only锚点均相同，原模型/loss/配置/split/选模不变。旧selector选epoch4 EMA，新selector选epoch5 EMA，reload预测差均0。两臂`validation_failed`表示未达V20 gate，不是训练失败。

| 相同2,340 validation key/标签上的结果 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict query-only（两控制臂相同，epoch4 EMA） | 0.705683 | 1.287663 | 1.000501 | 0.607919 | 0.579887 |
| strict旧selector aligned（epoch4 EMA） | 0.703442 | 1.298831 | 1.005282 | 0.600499 | 0.572085 |
| strict新selector aligned（epoch5 EMA） | 0.704160 | 1.298859 | 1.013556 | 0.601838 | 0.576958 |

新selector相对旧selector的CI/Pearson/Spearman虽略好，MAE却高0.0082743695、RMSE高0.0000283222。原500次/seed20261024、1,461个component bootstrap的MAE gain（旧减新）为-0.0082743695，95%CI[-0.0231009609,0.0076254430]；RMSE gain为-0.0000283222，95%CI[-0.0164946317,0.0182017224]。两区间均跨0，故结论是**selector-only证据不足，选行改动没有稳定改善validation affinity**；不能宣布稳定提升或已经证明下降。两种aligned五项也均未优于query-only。这里query-only仍有单query行MSA Transformer特征，并非ESM-only/完全删除MSA。

选行实际改变了输入：8,863条完整链中8,829条selected set变化，完整链identity/diversity中位数为90.1515%→56.2396%／12.8137%→48.0853%，平均row coverage为98.4572%→95.1822%。原validation质量诊断另在同complex pocket位置列计算并按残基数加权，其identity/diversity中位数96.7637%→77.6448%／3.7284%→28.3584%，平均pocket coverage99.5967%→98.9386%；不能把两种汇总范围混为一谈，也不表示模型重新输入旧口袋MSA。depth/Neff/窗口/profile IDs不变；33条链fallback但0条额外补行，validation4/2,340样本带fallback标志。增加diversity并未转化为本次总体收益，不能由此认定数据库或融合方式就是根因。

60条epoch日志逐臂匹配完整history，summary/provenance/selection一致；原stdout/stderr/manifest三项checksum通过，无运行降级/重试或失败标记（不同于selector的fallback标志）。Asia/Shanghai运行时间10月28日11:19:15.435156—21:27:02.083994，stage36,466.648838秒（约10小时7分47秒），两臂17,568.952423／18,718.629492秒；获准共卡，未记录全程峰值显存，不作提速基准。[完成报告](../experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/selector_only_diversity_20261024/strict_aligned_pair_20261028/completed_run_review_20261029/README.md)附两臂summary/history、selector结果、紧凑统计、来源审计及原日志；含逐样本误差的完整原统计留本地，未上传checkpoint/预测NPZ/缓存。CPU重算及检查详见同目录`evidence_audit.json`和`cpu_checks.json`。旧非strict四臂blocked结论不追认改变。本轮未读test/A3M、全库hash、运行新GPU任务或修改外部训练目录。

本轮CPU重算的新selector汇总及完整原统计（含500次bootstrap、全部分层和逐样本误差）与原结果完全相同，统计字节哈希一致；定向测试117 passed（5.28秒），compileall/bash -n通过。17份原始小文件共106,625字节，逐字节归档一致；上传统计为明确移除逐样本预测/误差的派生汇总，不冒充完整原统计。

唯一建议下一行动：停止继续selector调参，将本报告交给网页GPT，对MSA表征进入交互层并被affinity目标使用的路径做定向代码审查，再决定是否批准新实现或GPU实验。本轮不自动启动新模型、训练或geometry/pair实验。

### 5.64 FullChain v2 MSA→interaction→affinity 路由审查及固定 checkpoint 诊断准备（2026-10-29；GPU未运行）

按网页审查意见冻结5.63的selector结果，本轮不继续selector/depth调参，不改融合或启动训练。以已发布`0f22aae3ac9db5dcfb06c91071a42a06fccf777d`为基线，在独立`review/fullchain-msa-routing-audit-20261029`实现观察工具。静态确认：MSA缓存是冻结MSA Transformer第12层query-row逐残基表征、重叠窗口均值，形状`[B,L,768]`，无原始同源行轴；不是跨同源行简单平均。当前v2不是AF2/AF3 Evoformer/Pairformer直接迁移。完整链MSA经投影、与ESM相加/LN、全链cross-attention进入global/pocket/ligand tokens，再经主干进入base affinity与post-trunk contact。名为`msa_pair_encoder`的局部模块实际读取`pocket_esm`；local PP pair/interface、pair-token latent/residual在该MSA注入前产生，故替换MSA时应不变，不表示整个ligand交互或affinity完全没接入MSA。

新增固定权重诊断，只用strict旧selector aligned的epoch4 EMA，checkpoint SHA `f9e304210f24273665cadfdf850fe17dc1cbfdc1f8350c27b17d26fd0709ce9c`。固定原validation首批6条（原eval batch6），四种输入为query-only/旧aligned/新aligned/zero_cached_input；只替换`fullchain_msa`，不比较分别训练的三个模型。CPU已完成原metadata/来源/encoder identity、逐记录既有校验、checkpoint哈希与strict load、6条key/label对原预测归档、其他batch字段相同的接线检查。实际完整链张量`[6,427,768]`、pocket ESM`[6,91,640]`；旧/新输入相对L2差`0.508803`，query-only/旧`0.919491`，证明本批缓存输入有变化，尚未测得真实GPU输出响应。

GPU命令仅准备，由用户物理GPU6执行：先以原BF16/no_grad路径和float64反归一化重现保存首批预测（阈值1e-6）；四条件统一autograd-enabled eval做逐层前向差异，再对旧aligned固定batch、固定seed20261029/.15界面隐藏mask做逐项loss梯度，dropout仍关闭。可微loss逐项与canonical值/总值核对，分开测3项affinity、7项aux与3组总损失；encoder梯度N/A，缓存/投影/局部pair/interface/affinity代表张量的None与connected-zero分别报告。无optimizer、更新或新checkpoint，前后state fingerprint须相同。这是固定eval局部路由诊断，不冒充原训练步梯度、全程任务冲突或泛化效果。

CPU回归81 passed、28条既有Transformer warnings（31.97秒），compileall/bash -n通过；实际首批接线/严格加载8.07秒，最终prepare SHA为`030ae849c212145902edfa666656143424bdde3f205f836a482429c59b324c1b`。精确源码路由、CPU证据和GPU6完整命令见[诊断说明](../experiments/lp_pdbbind/msa_routing_audit_v1/README.md)、[静态审查](../experiments/lp_pdbbind/msa_routing_audit_v1/static_review.md)、同目录`cpu_checks.json`。单batch预计2–8GiB/1–5分钟（未测估算），allocator上限40%卡容量，worker硬限900秒、终止宽限30秒；失败不降精度/缩batch/重试。沿用strict确定性与原共卡显存检查，输出全新目录。未运行GPU、读test/raw A3M/全库哈希、改外部训练或主工作区既有文件；不上传权重/缓存/预测NPZ。下一步仅为用户执行固定checkpoint诊断，取得真实张量与梯度数据后再判断机制，不自动增加融合实现或新训练。

### 5.65 固定 checkpoint GPU 路由诊断完成：MSA影响affinity，但不直接更新local pair/interface（2026-10-29）

用户在gpu10/物理GPU6 A100-40GB完成5.64的单batch诊断，运行提交`96e8801cbc3038a91dad0358e4e3b3eaf470368c`，原目录`.codex_runtime/msa-routing-audit-20261029-gpu6.h0vC3h`。固定strict旧selector aligned epoch4 EMA、原validation首6条、只换MSA缓存。stage/worker exit0，耗时43.436949/29.822335秒，峰值allocated/reserved为0.915775/1.0078125GiB；北京时间16:58:02.881602—16:58:46.318551。未训练/更新参数。原no_grad首批预测重现差、no_grad与autograd eval差、canonical loss parity差均0；state指纹前后及prepare相同，16项输入记录/17项代码哈希与原plan一致，原日志3项checksum通过。

| 相对同一旧aligned权重/旧aligned输入的干预 | affinity平均绝对预测变化 | 最大绝对预测变化 |
| --- | ---: | ---: |
| query-only | 0.122839302 | 0.198009621 |
| 新aligned | 0.201676466 | 0.396019242 |
| zero_cached_input | 0.218788408 | 0.366684483 |

上述是敏感性，不是MAE/RMSE改善；query-only仍含单query行MSA Transformer表征，零输入也不等于彻底关闭投影支路。新aligned输入的相对L2变化为缓存0.508803→MSA投影0.535263→融合LN0.410191→context0.619883→注入后tokens0.175601→主干final0.214767→affinity输入0.131561，差异持续传到预测。MSA/ESM投影RMS为0.576135/0.591027。三干预的local single/PP pair/interface、pair-token pool/latent/residual、主干pair bias及local结构logits严格不变；context K/V、主干tokens、affinity和post-trunk contact有变化，符合ESM-only局部路径在MSA注入前的源码顺序，不表示所有ligand交互不受MSA影响。

另在eval/dropout关闭、固定seed20261029/.15界面距离隐藏（2165对）条件下测13项/组loss梯度。affinity对缓存MSA与投影矩阵的L2为0.094740614/1.919047117，加权aux总梯度仅其6.8277%/5.1253%，两者cosine +0.930579/+0.903573；total与affinity的cosine 0.999714/0.999757。本批不支持辅助梯度在MSA支路上主导对抗。contact在缓存空间cosine虽为-0.594361，加权梯度却仅affinity的约0.02323%，且投影空间cosine为+0.035498，不能只凭一个负cosine判定主因。residue_contact/distogram/interface_distogram/interface_contact/match对缓存MSA/投影均disconnected，而非connected-zero；它们未给该完整链MSA支路结构监督。encoder梯度N/A。pocket占有效残基27.93%、占缓存affinity梯度平方范数约52.27%，pocket RMS约为nonpocket的1.68113倍，padding梯度0；未见本批pocket信号被完全淹没，但未记录attention mass，不能全面排除长链/其他样本的稀释问题。

结论：MSA-affinity接线真实有效，当前明确的结构限制是缺少MSA→local pair/interface直接更新，尚不能证明它就是性能差的根因或补上后能提升validation；数据库质量也未被本诊断判定。**唯一建议下一步是网页审查一个显式MSA→ligand-conditioned pair/interface更新的单一机制假设，并明确区别于5.53已做过、收益很小的静态pair-bias实验。** 本轮不实现新架构/selector/loss，不启动新训练，也无新完整validation性能。

[完成报告及可复核原证据](../experiments/lp_pdbbind/msa_routing_audit_v1/completed_run_review_20261029/README.md)包含原224750-byte result（SHA `2e90a8706c26c4ac2e3ccc526118a030bfafae81d37830617ec3d360e8a0182f`）、原日志、CPU复核脚本与审计。五份原始小文件合计229956 bytes、逐字节归档一致，原/归档两次审计结果相同；CPU回归97 passed、28条既有Transformer warnings（91.34秒），含16项新增离线复核/篡改失败路径测试，compileall通过。梯度cosine沿用原GPU记录，未声称CPU从不存在的原始梯度向量重算。未上传权重/特征缓存/预测NPZ；本轮CPU只读小型生成产物、prepare与源码，不读test/LMDB/raw A3M或重哈希大资产，不改外部训练和主工作区既有修改。原5.64 GPU命令标为已执行，无需重跑。

### 5.66 下一步单机制方案审查：MSA-conditioned interface readout residual（2026-10-29；未实施）

承接5.65，仅审查一个最小改动：从已编码完整链MSA按精确pocket→fullchain整数索引读取残基向量，与原ligand-conditioned最终interface pair做乘性交互，经零初始化输出投影得到一次readout residual；随后共享原interface heads/summaries和下游pair-token/主干读取。固定hidden64、无新增dropout，静态估算新增63,168参数。原完整链attention、PP pair/match、loss、selector保留；不是重做不连续pocket MSA，也不是AF2/AF3模块迁移或显式多行共变。区别于5.53仅给主干PP attention加静态144通道bias，本方案直接更新protein–ligand interface表征，但不反馈原interface迭代blocks。

拟议以当前strict旧selector aligned v2/dropout0.10为基座做route-off/on两臂，同一fresh初始state/RNG、train/valid11,037/2,340、seed20261001和原30epoch/20,700步配方，不能混用v3/.20参照。原选模tuple `(V20_gate_passed, CI, -RMSE, -MAE, Pearson, Spearman)`及五项gate不变；off须复核canonical strict锚点，比较同2,340key/标签。该对照只识别新增架构路径效应，不能单独分辨同源信息与容量贡献。原下游adapter/head也是零初始化，故首步affinity梯度可能connected-zero；应分开验收初始等价、total loss更新新输出投影及之后的新路径affinity连通，不能拿旧globalcontext已有的MSA总梯度冒充新接线通过。

[完整方案、源码依据及实施/验收边界](../experiments/lp_pdbbind/msa_routing_audit_v1/next_interface_update_plan_20261029.md)。本轮仅文档与只读复核，无模型实现、新CPU模型测试、GPU或新validation性能；后续实现/CPU验证、用户有界GPU smoke和正式paired训练分阶段开展。唯一下一行动为审定本方案后做隔离实现，不自动启动新训练。

### 5.67 MSA-conditioned interface residual隔离实现与CPU验收（2026-10-29；GPU未运行）

用户批准继续5.66。以已推送`77384aa2b4b324bff354cdc434577a16fa9f859b`为基线，在`feature/fullchain-msa-interface-residual-20261029`新增独立package、精确索引dataset/collator及有界smoke入口，不改原v2/封存训练代码、缓存或外部模型。新encoder wrapper保留原对象，只有MSA→ligand-conditioned最终interface readout残差；hidden64/无新dropout/单零输出投影，实际新增63,168参数（37,719,863→37,783,031）。原完整链context、PP pair/match、loss、selector和选模均保留；不是AF2/AF3迁移或历史静态PP bias，也不声称完整链MSA新增了同源行轴/显式共变。

CPU最终回归170 passed、44条既有Transformer warnings（20.01秒），其中85项新增；compileall和bash-n通过。覆盖原共享state与RNG、零初始化输出/loss、off三步dropout/mask/AdamW/EMA等价、total loss更新Wo、独立内部gather的affinity连通探针、精确多链/乱序重复位置与padding、严格加载/协议篡改/超时失败路径。探针仅是明确标注的合成单测，不改真实模型下游零初始化。

真实CPU prepare只读首16 train/首6 valid缓存及必要小元数据，17.08秒；fullchain MSA分别为[16,826,768]/[6,427,768]，新增整数索引[16,93]/[6,91]，其余batch字段与canonical完全相同。shared初始指纹仍为`21d4a28bc5277b8e688fb9d4f6d56e5df585b9860512a99a05ebeee58707c790`；on/off完整state与主CPU RNG一致。原prepare SHA `5673492e127ffdba1d0615b1d214b7fdc48070d14e8e814d5234119498677cb9`，绑定23项源码/脚本/参照文件与8项输入元数据哈希。另对真实train/valid各前2条、完整生产模型配置作CPU FP32前向（15.20秒），canonical/off/零on全部输出逐张量相同、有限，affinity差0，无更新。未读checkpoint/encoder权重/raw A3M/test，无全库hash。原5.65的17项源码和3项日志checksum复核仍通过。

用户GPU6 smoke已准备但未执行：原batch16/eval6/BF16/strict profile，canonical/off/on各最多3步，调度仍按原20,700步；off与canonical需比较prediction/loss/mask、共享梯度/model/AdamW/EMA、scheduler及RNG，on须验证Wo实更新及独立内部m_p affinity梯度非零，初值与严格重载按affinity单位1e-6验收。前3步原warmup可能使BF16前向差量化为0，如实记录，不改读头/调度制造响应。单worker900秒+30秒终止宽限、allocator40%上限，允许既有获准的显存足够时共卡；无正式paired训练命令或新validation性能。

[实现说明、CPU摘要、原prepare与完整GPU6命令](../experiments/lp_pdbbind/fullchain_msa_interface_residual_v1/README.md)。下一步仅由用户运行有界GPU smoke并返回`interface_smoke_result`，复核后再准备正式对照；本轮不自动训练。

### 5.68 首次interface-residual GPU smoke失败与冷/热AdamW运行时修复（2026-10-29）

用户在GPU6执行5.67的smoke，源码`ce4c896c74d0fe5a30d29da03a38b0a4ea7217ff`，目录`.codex_runtime/msa-interface-residual-20261029-gpu6.pJry9o`，stage140.767198秒/exit1。canonical完成3步，off在首步optimizer/EMA更新后的总record比较失败，on尚未进入；没有result.json或训练checkpoint。原实现未将双方record/字段差异落盘，故不能从该GPU证据断言数值分叉或只有RNG不同。五个原始小文件8,428bytes逐字节归档、3项日志checksum通过。

在干净CPU进程中完整复现该报错，定位到足以触发失败的harness缺陷：第一次AdamW构造惰性导入torch._dynamo并推进Python random状态，而同worker中后续臂复用已导入模块；NumPy/Torch CPU RNG不变。cold完整循环首步只有rng_fingerprint不同，预测/loss/components/mask、clip前后梯度、model/AdamW/EMA及scheduler均相同。修复仅将已有依赖初始化移到所有臂受控seed之前，不启用torch.compile、不改模型/data/loss/精度/调度，不删除或放宽任何RNG比较。另增加比较前create-only逐步快照和first_divergence字段明细，失败保留证据。

新fresh-subprocess回归覆盖原cold完整loop仅RNG分叉与修复后三臂各3步CPU模拟通过（保留真实数值/优化器/梯度及Python/NumPy/Torch CPU RNG比较，CUDA transport/autocast/device/backend/RNG明确mock，不冒充GPU）。新增证据测试覆盖1e-12差仍失败、缺字段、快照alias及不可覆盖；最终187 passed、44条既有Transformer warnings（66.36秒），compileall/bash-n通过，见[修复报告](../experiments/lp_pdbbind/fullchain_msa_interface_residual_v1/failed_smoke_review_20261029/README.md)。新CPU prepare17.46秒通过，shared/full初始指纹不变，SHA `3d46b31e26738c794255df442b006e28d719af310b7693e96fbc771e9ca45fe1`；旧prepare和失败运行均保留。

本轮未运行GPU、未读旧checkpoint/encoder权重/raw A3M/test或全库hash，亦无新validation性能。原GPU具体差异没有历史记录，不能追认为已查明或通过；唯一下一行动是用户使用新rngfix prepare在GPU6重测同一有界smoke，若再次失败由新增字段证据定位，不进入正式训练。实现/报告将推送独立`fix/fullchain-msa-interface-smoke-trace-20261029`，基线ce4c896完整SHA如上。

### 5.69 修复版interface-residual GPU smoke通过：接线验收，无新性能结果（2026-10-29）

用户在gpu10/物理GPU6 A100-40GB完成修复版smoke，运行提交`093de4eb6163103ae35996da5dcb3f4fa962fc41`，原目录`.codex_runtime/msa-interface-residual-20261029-gpu6.HI32mI`。canonical/off/on各3步、重复同一固定train batch16，原strict v2旧selector/dropout0.10/BF16、eval6及20,700步scheduler保持不变。stage/worker exit0，分别52.459882/29.489554秒，北京时间19:57:43.266100—19:58:35.725982；峰值allocated/reserved为2.297583/2.517578GiB，allocator上限40%，获准共卡，不作为提速基准。

三臂初始validation预测差均0；off与canonical三条完整record精确相同，包含prediction/loss/components/mask、clip前后共享梯度、model/AdamW/EMA、scheduler和含Python/NumPy/Torch CPU/CUDA的RNG指纹。on的Wo实际更新，三步raw梯度L2为0.0128989764/0.0119461486/0.0131514282。首步clip前共享梯度相同，新增Wo梯度改变全局clip范数后共享state开始分叉，符合机制，不是off失败。内部独立m_p对standardized affinity predictions之和的梯度从connected-zero到`1.1717264496923064e-12`，只证明接通，量级很小，不能证明贡献已充分。原nonpocket梯度非零、padding梯度0。训练后三步MSA置零干预的interface pair/contact变化仅3.8147e-6/4.7684e-7，原PP/match等输出不变；三步on预测hash/loss恰仍与canonical相同。真实内存torch.save/load及strict state重载后affinity预测差0，未落盘checkpoint。

[完成复核报告与原始证据](../experiments/lp_pdbbind/fullchain_msa_interface_residual_v1/completed_smoke_review_20261029/README.md)归档14份小原文件45,126bytes，逐字节相同、3项原日志checksum通过；result SHA为`dcc96189960ac198ab124a26485a44cf83267c3a11e4f168e595f5c5b9ca5621`。9份单独step record与result一致，prepare/23项源码和8项输入identity绑定相同。原/归档两次标准库审计一致；CPU回归241 passed、44条既有Transformer warnings（146.24秒），含54项新增离线证据/篡改失败路径测试，compileall/bash-n通过。命令及审计见同目录；本轮未重读8项实际输入资产、LMDB/raw A3M/test或权重。5.68首次失败证据仍保留，不追补原GPU没有记录的字段差异。原GPU命令已标完成、无需重跑。

本次只支持真实GPU接线、三步off等价、更新与重载；没有完整validation指标或完整off训练锚点，不支持性能提升结论。下一步才是准备既定route-off/on受控正式对照，本轮未实施或启动；未修改模型、缓存、外部训练或旧结果。

### 5.70 Interface-residual正式route-off/on对照准备（2026-10-29；GPU未运行）

用户批准继续5.69后的正式对照准备。基线`6834a6cca7eb57a0ac497282b65053ee7129abc4`，独立分支`feature/fullchain-msa-interface-paired-training-20261029`；新增training/比较器/GPU6脚本，直接调用封存原`cross_v1._fit_arm`的model_type/collator_type/protocol hooks，不复制循环、不传custom loss/step或tracer。模型/dataset/原训练源码、缓存、旧运行和外部六模型均不变，成功smoke无需重跑。

固定strict v2旧selector aligned、dropout0.10、train/valid11,037/2,340、seed20261001、b16/eval6/workers4/BF16、原loss/AdamW/EMA/mask/sampler/20,700步调度和V20 gate选模。两臂从同一完整fresh state开始，每臂独立fresh进程30epoch/20,700步；仅route false/true不同，原完整链context不改。off先复核固定strict旧aligned完整history精确、epoch/variant和2,340key/标签逐条原1e-6预测锚点，通过才启动on；direct-on worker也重验off及拒绝已失败/blocked根目录。原首次AdamW冷导入保持历史行为，不将同worker三臂smoke的预加载修复机械移植。固定factory保障训练与重载都使用正确route，checkpoint明确写入route/hidden/来源/完整及共享初始指纹。

最终真实CPU prepare21.13秒通过，33项源码/脚本/参照、9项必要小输入identity及原smoke绑定；SHA `784cceaa6368e8608a93728653413db3df3b3c8f5e8c3abbd8ce7b8c493adbc2`。只读固定首16train/6valid及额外实际epoch0 sampler16train缓存；sampler完整链MSA为[16,1011,768]、映射[16,85]，除新增索引外与canonical所有字段相同。原完整/共享初始指纹仍为9a5b3847…/21d4a28b…；原source、配置和5.69smoke不变。prepare原件逐字节归档见[正式说明与CPU证据](../experiments/lp_pdbbind/fullchain_msa_interface_residual_v1/paired_training_20261029/README.md)。CPU合成fresh-cold子进程真实运行原fit循环、forward/backward/保存重载，off与canonical完整history/选模/预测/selected共享state一致；on投影更新，严格reload差0，不冒充生产GPU锚点。

新比较器用项目函数复算五项和on−off，预定主终点为off−on RMSE gain及MAE是否退化，同时报告V20 gate；未提升和validation_failed不等于运行失败。无新bootstrap、不把单seed架构对照当纯MSA消融或数据库结论。最终CPU回归377 passed、47条既有Transformer warnings（127.80秒），含136项新增训练/比较/launcher测试，compileall/bash-n通过，最终源码freeze后validate_plan再验通过。完整命令见同目录cpu_checks.json；尚无正式训练history/checkpoint、完整off锚点或新validation性能。

用户GPU6一条脚本顺序跑off/锚点/on/比较，每次新目录，每臂24小时硬限/30秒终止宽限、allocator40%、沿用获准共卡，失败不降级或重试。预计总10–14小时只是旧运行参考，正式峰值未知；脚本不会触碰外部模型目录、旧缓存或结果。下一步仅为用户执行上述说明中的GPU6命令；本轮未运行GPU、读取test/raw A3M/encoder或旧checkpoint、全库hash或上传原始数据/权重。

### 5.71 Interface-residual正式配对完成：五项均退化，off全程锚点精确通过（2026-10-30）

用户在gpu10/GPU6/A100-40GB执行5.70命令，源码`67223c04fa4a5dd6e5f1636ab21195d619bb5db8`，新目录`.codex_runtime/msa-interface-paired-training-20261029-gpu6.3HZpQy`；两臂均30epoch/20,700实际optimizer steps，batch仍16。先off后on、无降级或重试，stage及两个worker exit0。两臂provenance唯一不同字段为arm/route_enabled，完整/shared初始化、原strict v2/.10/旧selector aligned、loss、数据、backend和选模保持相同。off完整30epoch history及validation NPZ与冻结strict旧aligned逐字节相同，最大预测差0，锚点通过。

同2,340个unique validation key和完全相同归档标签，双方均选epoch4 EMA，checkpoint reload差均0。项目指标函数从本地NPZ复算及生产比较器CPU只读重执行都与保存JSON精确相同：

| 指标 | route-off | route-on | on−off |
| --- | ---: | ---: | ---: |
| CI ↑ | 0.703442 | 0.701161 | −0.002281 |
| RMSE ↓ | 1.298831 | 1.305484 | +0.006654 |
| MAE ↓ | 1.005282 | 1.010118 | +0.004836 |
| Pearson ↑ | 0.600499 | 0.593280 | −0.007219 |
| Spearman ↑ | 0.572085 | 0.567953 | −0.004131 |

预定RMSE gain(off−on)为−0.0066539323，MAE退化，两臂V20五项gate全未过。末epoch train loss off/on为0.450103/0.442882；更低训练loss没有转化为选中模型validation收益。结论：**本次新增融合路径没有提升，不替换route-off模型**。这是两臂都有完整链MSA的单seed架构比较，不是有/无MSA消融，不据此归因数据库或声称多seed稳定劣势；未做bootstrap。

北京时间10月29日20:44:54至30日05:27:41，总stage8小时42分46.8秒；off/on worker15,359.574/15,897.321秒，峰值allocated6.275/6.644GiB、reserved9.102/9.303GiB，获准共卡，不作为独立提速基准。33项绑定源码、9项actual小输入、8项历史small input、prepare与原日志checksum复核匹配。16份原始小文件154,442bytes逐字节归档，含两臂summary/provenance/selection/完整history、比较、anchor及原stage日志，见[完成报告](../experiments/lp_pdbbind/fullchain_msa_interface_residual_v1/completed_training_review_20261030/README.md)。本轮不改模型/训练源码，不运行GPU，不读取checkpoint/test/raw A3M或hash全库，不上传原始数据、NPZ、特征或权重；仅做CPU复核与结果报告交接，不自动开始新训练。

### 5.72 MSA配体条件块内消息传播v1准备：新Q/A，无V20 gate（2026-10-30）

固定起点`cda63f6f4c89e08ea888dd0d31476f03510ad5cf`，隔离checkout `.codex_runtime/msa-interface-message-20261030`，独立分支`review/fullchain-msa-interface-message-20261030`。保留主工作区既有修改、节点缓存及历史结果，不叠加5.71末端residual。新模块在两个原interface block的transition/mask之后、residue/ligand更新之前显式调用；两层共享一份Cz96/Ce144/d32/H4/K16模块，关系进入key及sigmoid-gated value，由当前配体条件pair状态传播。只Wo零初始化，无新增dropout/零gate；FP32安全softmax/归约，delta转回pair dtype后只相加一次。实际新总参数37,739,287、可训练37,202,392，相对canonical37,719,863新增19,424；共享初始指纹仍21d4a28b…，新完整指纹0e74fc87…。

原strict v2/dropout0.10、局部ESM、完整链context、几何/配体、原多任务loss、单组AdamW、sampler/mask/EMA和20,700步调度保持。train/valid为11,037/2,340；仅新Q/A各30epoch，b16/eval6/workers4/seed20261001/BF16，复用既有strict old Q/A，不完整重训旧锚点。新入口选择`(CI,-RMSE,-MAE,Pearson,Spearman)`，保留原tie处理，不加载V20 selection、不做V20 gate；旧入口默认逻辑不变。43项完整training_config与封存strict old-A及route-off逐字段相同，不激活原不生效的分组学习率字段。

Q/A原[L,768]节点缓存不改；新增`msa_row_attention_sparse_pair_v1`：冻结同一官方MSAT、FP32、512列/stride256/原尾窗，两轴去BOS/padding/query-gap后，在窗口全部真实query残基上sym+APC，保留负值，layer-major/head-minor144通道，仅存同链实际稀疏边、平均共同窗口。共用邻域先按真实残基去重/去自身，依PP几何与稳定tie取K16，不按attention或标签挑边。Q使用真实单query关系，不置零或旁路；A保留旧source-order selector/depth≤8。旧静态pair数值不复用，只复用已逐记录核实身份的query-window映射；869e531e…旧manifest与850e6c7d…strict manifest之间的兼容由8,863条query/A3M/选行身份相同证明，不跳过哈希。

定向CPU新增127 passed/12条既有Transformer warnings/18.61秒；受影响旧v2/replacement/cross及residual模型/训练/trace回归99 passed/31条既有warnings/110.67秒，均非整库测试。覆盖形状/空邻域/mask/跨链/自身/共同窗口/APC、显式off及zero-Wo原输出与RNG、非零探针发送端/关系/配体依赖、第一层更新及第二层输入、原真实loss合成CPU训练/严格重载、无V20选模隔离及比较/失败路径。CPU真实节点接线使用明确合成关系；真实attention缓存、真实Q关系及CUDA strict算子验证由用户GPU入口完成，不把准备完成写成实验完成。完整命令、配置与准备身份见[本轮说明](../experiments/lp_pdbbind/fullchain_msa_interface_message_v1/README.md)及[cpu_checks.json](../experiments/lp_pdbbind/fullchain_msa_interface_message_v1/cpu_checks.json)。

最终全量CPU prepare923.87秒通过并再验封存：40项源码/配置/入口及43项小输入身份，SHA `ad3d494c1358af73678420f01d9dc0e194e242a7dae98bf101f308aa0ea6bd27`，25,488bytes原prepare逐字节归档。生成8,855条有边链/8,914,234条去重有向边，train邻域覆盖760,845/760,857、valid162,593/162,598；其余12/5个残基不补造有效邻居，全部complex至少有一条边。两模式各2train/2valid真实节点＋明确合成关系，生产尺寸forward的11项原有效输出形状/dtype及数值均与canonical精确相同，未做优化器更新。开发失败/非最终prepare目录保留；真实关系值仍待用户GPU提取。

单入口顺序为固定少量链真实probe→全量新增关系/audit→新Q/A各3步smoke→独立fresh Q→独立fresh A→CPU同2,340key/标签比较和500次protein-component bootstrap(seed20261024)。Q指标差不会阻断A，技术/资源失败保留原日志后停止，不自动降级或搜索。报告RMSE/MAE的G_homolog、G_absolute、G_arch_aligned及交互I，主要区间覆盖前两者；旧预测缺失时不伪造旧臂区间。新A必须相对new Q和更强old Q均改善RMSE、MAE均不恶化，才是待确认初步信号。当前状态ready_for_user_gpu，无本轮新validation指标；不支持收益结论，也不称AF2/AF3模块迁移或完整recycling。

### 5.73 块内MSA消息真实提取/smoke通过，Q正式训练OOM停止（2026-10-30）

用户执行5.72入口，代码`cd1dd02607cc6487fdd6f3f7cf10b72cf925967f`，目录`.codex_runtime/msa-interface-message-20261030-gpu6-20261030_124121`。GPU6/A100-40GB的extract、Q/A smoke原stage exit0；正式Q在原fit loop backward抛出OutOfMemoryError/exit1，A正式训练及compare未启动，没有完整history、选中checkpoint/预测或新validation指标。错误时额外申请1.72GiB，本进程占15.49GiB，PyTorch allocated11.45/reserved未allocated3.49GiB，驱动free2.03GiB；共卡PID6003/30610占16.95/4.89GiB。另有原40% allocator上限（按错误容量39.38GiB约15.75GiB）；reserved14.94+新申请1.72约16.66GiB超过上限，碎片回收约束也可能参与，不能单独归因共卡或保证换空闲卡就成功。

真实关系缓存8,855链/8,914,234有向边，Q/A各9,853窗，压缩合计8.152GiB，原节点未改。A深度1/3/4/8窗口26/2/2/9,823，fallback26窗。Q/A各3步真实BF16/strict/原loss与调度smoke均更新Wo，初始原输出及严格重载差均0，峰值allocated/reserved4.566/6.209GiB。完整extract/smoke/Q原stage耗时9,161.970/457.743/362.960秒，worker记录8,493.066/85.936/25.302秒，计时口径不同不混用。

有界CPU读取train原结构，smoke首16的Pmax/Amax为93/41，原epoch0 sampler首批为85/96；单个[B,P,16,A,32] FP32张量从124,944,384增至267,386,880bytes，说明固定首16接线smoke不覆盖正式padding形状。实际失败batch/step未记录，不把首批推断成失败批，也不推断具体已完成optimizer步数。源码会保留两层多个大6D激活，新增参数小不意味着激活小。

本轮仅CPU只读诊断/归档，27份原始小文件108,982bytes逐字节一致，9项原日志checksum及40项源码/配置身份通过，6份smoke step sidecar与result一致，见[报告与原始证据](../experiments/lp_pdbbind/fullchain_msa_interface_message_v1/failed_training_review_20261030/README.md)。不改模型、batch、精度、allocator或strict，不重跑GPU、不读test/raw A3M/权重，不重扫完整关系数组或全库hash。原成功缓存和失败运行均保留；下一步建议是获准后做消息激活等价省显存及验证，并通过明确身份绑定复用已完成关系，不重复提取或移除失败标记。

### 5.74 块内MSA消息GPU3重跑仍OOM：换卡未解决，无新validation结果（2026-10-31）

用户将5.73的原pipeline设备改为物理GPU3重跑，运行提交`ed0f54f1876915afb152007b84ce08f1fe877c2d`，新目录`.codex_runtime/msa-interface-message-20261030-gpu3-20261031_100941`。extract/audit及Q/A各3步smoke再度exit0，正式Q仍在`scaler.scale(loss).backward()`抛出CUDA OutOfMemoryError/exit1；A及compare没有启动，没有history、checkpoint、summary或新validation预测。原stdout仅有arm_start，无已完成step/epoch证据，不推断具体失败batch或完成更新数。

OOM额外申请1.72GiB，Torch allocated11.45、reserved未allocated3.46GiB，本进程15.46GiB，驱动free3.23GiB，另PID54125占20.67GiB。代码仍固定40% allocator上限，按错误总容量39.38GiB约15.75GiB；reserved约14.91+新扩1.72约16.63GiB超过cap，碎片也可能参与，不能仅归因共卡或保证换卡成功。GPU3命令只改卡号，没有修复显存瓶颈，不再原样重复提取/训练。

三stage耗时extract/smoke/Q分别9,972.056733/1,102.301398/1,111.526552秒，worker记录9,090.120408/184.691281/108.909771秒（至失败），不可混用。成功关系仍为8,855链/8,914,234边，Q/A各9,853窗，保留原地；Q/A smoke峰值allocated/reserved为4,903,088,128/6,666,846,208bytes，Wo更新和strict重载差0，只证明有界接线。

本轮只读CPU核对和结果归档，27份原始小文件108,806bytes逐字节保存，9项原日志checksum、40项源码/配置及prepare/request/初始化绑定通过。通用化原小证据审计工具的目录/提交参数，默认GPU6审计保留；原件/归档复验、错误提交拒绝、compileall及任务diff检查通过。见[GPU3完整诊断与原始日志](../experiments/lp_pdbbind/fullchain_msa_interface_message_v1/failed_training_review_20261031_gpu3/README.md)。未改模型/训练/资源配置，未运行GPU、读取test/A3M/权重/完整关系数组或全库hash。下一步建议仍是获准后做等价激活省显存、正式sampler形状smoke和成功缓存身份绑定复用入口；本轮未实施。

### 5.75 用户批准取消块内MSA消息实验的40%进程显存配额（2026-10-31；新GPU尚未运行）

用户明确要求“取消限制，有多少显存就用多少”。以`8358e9db6783c548d4b577754038edd15083668f`建立独立`fix/fullchain-msa-interface-memory-unlimited-20261031`与`.codex_runtime/msa-interface-message-unlimited-20261031`，保留原实现/prepare/GPU6、GPU3失败及成功缓存。删除`run.worker`和`training.fit_mode`两处`set_per_process_memory_fraction(.40,0)`，不是改成另一个固定比例；默认allocator按实际可用显存按需分配。fixed_config/plan显存额度字段为null，smoke及formal resource为null，provenance记录不限额策略。新plan和正式训练明确拒绝旧40%或缺策略的smoke。

只有上述资源策略及其封存/记录变化。原模型/参数、完整链MSA与原selector、batch16/eval6、BF16/strict、原loss/AdamW/EMA/sampler/30epoch/20,700步及无V20选模保持不变；不做分块/activation checkpoint/自动batch调整，也不修改其他实验或正在运行的进程。

最终CPU定向回归161 passed/12条既有Transformer warnings/22.34秒，含17项poison setter与资源策略测试、16项资源刷新测试及扩展原篡改覆盖。真实合成CPU模型forward/backward/reload仍运行；模拟GPU控制测试明确mock CUDA/backend/dataset/数值fit，不称为真实GPU通过。compileall/bash-n/任务diff-check通过。

新的CPU资源封存49.07秒完成，先以原checkout和原prepare SHA完整validate旧40项源码/43项输入，再限定run/training/smoke/fixed_config四文件差异和配置仅0.4→null，重建新ROOT相关身份/参照并再次验证全部小输入、环境、原请求SHA。只读复用原邻接请求，没有重扫全量几何；重跑fresh初始化和Q/A各2train/2valid真实节点＋合成关系接线，四组11输出相对canonical最大差均0、optimizer steps0。完整/shared初始指纹仍0e74fc87…/21d4a28b…，总参数37,739,287不变。新prepare28,349bytes/SHA`e09868cde911ae2046a0c3987ea652d60b1c09e012ab609408860c3e0b3c9ef5`，归档与原件相同；原prepare和请求SHA均未改变。

[资源变更说明、CPU原始配置与GPU4完整命令](../experiments/lp_pdbbind/fullchain_msa_interface_message_v1/memory_unlimited_20261031/README.md)。必须使用新checkout及新prepare，旧命令/已启动旧进程不会自动取消配额。原顺序pipeline仍会在新输出目录生成关系并重测smoke/正式Q/A；本轮未增加关系数组复用入口，未运行GPU或得到新validation成绩，不能将取消配额当作全训练成功。未读取test、原始A3M或encoder/训练权重、未全库hash；下一步由用户执行新的GPU4入口。

### 5.76 不限额GPU4入口在extract前初始化失败：显式CUDA init修复（2026-10-31；新GPU未复测）

用户执行5.75的GPU4入口，代码`c62afb23405cd9b31749c78563553505c90828f0`，目录`.codex_runtime/msa-interface-message-unlimited-gpu4-20261031_220211`。extract stage25.361571秒/exit1，原stdout为空，首错位于`run.worker`的`torch.cuda.reset_peak_memory_stats(0)`：`Invalid device argument 0: did you call init?`。原manifest记录当时物理GPU4/A100-40GB空闲40,326MiB、无compute PID，逻辑device0映射物理GPU4；这不是OOM或卡号错误。在实际提取前退出，无relations/probe、smoke、训练或validation结果。

本机Torch2.1源码确认原fraction setter先执行`_lazy_init()`，本次显式device0的统计重置不会完成该初始化；删除setter时连带失去该初始化，上次CPU控制测试无条件mock reset而漏检。本轮以c62afb2为工程基线建立独立`fix/fullchain-msa-interface-cuda-init-20261031`与`.codex_runtime/msa-interface-message-cuda-init-20261031`，只在worker和direct formal fit的reset前添加`torch.cuda.init()`，保持不限额字段None、不恢复setter、不改模型/batch/原loss/strict或训练配方。formal原无参reset可能经默认设备查询间接初始化，未实际执行到该处，不将统一显式init写成另一项已发生的GPU失败。原实现、prepare、失败日志和成功缓存均保留。

最终定向CPU173 passed/12条既有Transformer warnings/19.84秒；新增12项初始化回归，保留真实init/reset Python wrappers而仅模拟lazy-init/C统计端点：冷reset复现同款错误，init→reset通过，worker四phase和formal冷状态均检查调用顺序，初始化失败不执行后续工作并保留formal failure，CPU compare不初始化CUDA。原CPU模型forward/backward/strict reload仍实际执行；这些mock不称为GPU复测。compileall/bash-n/任务diff-check通过。

新资源封存49.66秒，40项源码/43项小输入及初始化重新核对，原请求只读复用，重跑Q/A各2train/2valid真实节点＋合成关系接线；四组11输出相对canonical最大差0，optimizer steps0。新prepare28,348bytes/SHA`1c4d1a58005f6d2260dae4cd4ee71fab6b5b02973849b98cf5a867d2dfabf863`，原件与归档一致，完整/shared指纹仍0e74fc87…/21d4a28b…。7份原始小文件共7,604bytes（包括0字节stdout）逐字节归档，3项原日志checksum及40项原源码身份通过；见[失败证据、修复说明、CPU结果和新GPU4命令](../experiments/lp_pdbbind/fullchain_msa_interface_message_v1/cuda_init_fix_20261031/README.md)。

本轮未运行GPU、读取test/A3M/encoder或训练权重，未全库hash。初始化可能处理待执行seed回调，不声称完全不触及CUDA RNG；仍保留原fresh seed/初始化流程。新入口的GPU smoke、显存峰值、完整训练及validation尚待用户运行，不将CPU准备当作全流程成功。唯一下一步为用户使用新checkout/prepare和新输出目录执行GPU4命令。

### 5.77 块内MSA界面消息实验结束：Q完整无提升，A四轮中断后不再恢复（2026-11-01）

用户执行5.76修复后的实际代码`32956f967b65a3bfc3cc49f6bf2fad6c0f3cacdf`，物理GPU4/A100-40GB，不设置进程显存配额；运行目录`.codex_runtime/msa-interface-message-cuda-init-gpu4-20261031_223459`。extract/smoke均exit0，8,855链、8,914,234条有向边的真实关系审计通过，Q/A各3步真实smoke通过。stage耗时extract8,926.75秒、smoke772.57秒；函数耗时分别8,169.95/73.76秒，不混作训练吞吐。

Q已完整30轮/20,700步，选中epoch2 EMA（第3轮），validation n=2,340：CI **0.7022865556**、RMSE **1.2940411754**、MAE1.0059204875、Pearson0.5980776353、Spearman0.5728435518。相对旧strict Q，CI−0.003396、RMSE+0.006379、MAE+0.005419，五项均略差；后期train loss下降而validation退化。原summary/history/selection一致，原预测SHA`93f2f0002da6c7798305868dde2f519ba0f24986ce4e217931d53c38ba7af295`匹配，checkpoint重载最大预测差0。CPU用项目指标函数重算新Q/旧strict Q/A，五项差均0；三者2,340个唯一validation key及逐key标签一致。Q worker耗时37,079.42秒（10小时18分），含前置校验stage37,456.47秒；峰值allocated24.04GiB、reserved38.66GiB。

A已完整记录epochs0–3（4/30轮），当前最好epoch2 EMA：CI0.7016233342、RMSE1.3036250421、MAE1.0118496101、Pearson0.5978876931、Spearman0.5699151566。北京时间13:13左右记录中断，supervisor收到SIGTERM15、worker exit−15，发送者/精确中断step未记录；原stderr无OOM，也不是24小时超时异常。A manifest仍为running/end_time=null是未收尾原件，不表示任务仍运行。A无最终summary/history/预测导出或重载校验，compare未执行，不能将前4轮最好记录当作完整A结果。

Q是单query MSAT节点/关系，A是同encoder和旧selector的query加最多7同源；同一新架构和完整初始state，A独立fresh训练，不接Q。既定30轮预算未因Q差而自动gate A；本次由用户在A中断后决定**结束本轮，不恢复、不新增训练**。目前证据为Q没有改善、A暂未显示收益但不完整，不声称A后续一定不提升或数据库/MSA整体无效。

[停止结果及可复核审计](../experiments/lp_pdbbind/fullchain_msa_interface_message_v1/stopped_review_20261101/README.md)归档26份小原件160,928bytes，逐字节核对、9项已完成stage原checksum通过；保留A未收尾manifest和原始四轮stdout，不补造完整A history/结束时间。原始validation预测、模型权重和关系数组仅保留服务器，不上传。Q和关系缓存无需删除；A best.pt保留但没有完整optimizer/scheduler/RNG续训状态。本轮仅CPU审计/归档和报告交接，未运行GPU、未改模型/配方、未读test/A3M/权重或全库hash；本候选到此结束。

### 5.78 Persistent joint-pair v1：持续联合PP/PL/LP/LL精炼，CPU-ready（2026-11-02）

以参考提交`e5e57d2f0dc67c5c8ecf6ab54828621dcf20f553`隔离实现AF3-inspired局部主干：只初始化一次S192/Z96，两个独立block持续传递有向联合pair，按outgoing/incoming triangle multiplication、starting/ending triangle attention、pair transition、pair-biased single attention、single transition顺序更新。真实144通道MSA关系只初始化已观测PP；观测mask不限制dense pair更新。PL/LP不重新读取隐藏距离，LL使用已有bond及内部距离，最终原heads/summaries读取精炼S/Z。外层8层几何网络、完整链context、pair-token、affinity head及原loss/训练配方不变，不执行旧末端residual或固定K16消息旁路。

真实参数：canonical37,719,863，保留37,325,111、删除394,752、新增1,718,272；候选39,043,383、可训练38,506,488，净增1,323,520。保留参数显式key逐值复制canonical干净随机初值，Q/A完整初值相同；不加载旧训练权重。复用5.77已完成的8,855链/8,914,234有向边节点/关系缓存，不重新提取。上一候选仍结束、A不恢复。

本轮CPU定向测试202 passed/40个既有Transformer warning，数学loop/梯度/chunk/checkpoint/mask、跨PP↔PL/LP与持续Z消费、训练数值/增量history/恢复保存及旧入口回归通过。Q/A×train/valid各2条真实缓存forward全部输出有限。实际扫描train11,037/valid2,340，最大局部R160/A177（valid A162），lineage合法A上限256，完整链最大2701；用户资源probe固定B16/R160/A256/fullchain2701，不缩样本。BF16持久Z、重要归约FP32、端点8分块与activation checkpoint控制激活，不声称总显存已通过。

用户入口为cache_check→Q/A各3步真实smoke及合成最大尺寸0更新probe→新Q30轮/20,700步→新A相同预算→CPU比较。两臂fresh、不因Q差gate A；batch16/eval6/workers4、strict/BF16、单组AdamW/原warmup与EMA/全部loss/CI优先选模不变，不读V20/test，无显存比例cap，无自动降配或resume。每轮增量history、best及model/optimizer/scheduler/EMA/scaler/RNG/sampler的latest恢复状态；不宣称恢复worker生命周期。

当前仅**CPU-ready**，没有GPUsmoke/正式训练、新Q/A summary或性能结果。与旧strict Q/A的同2340key/标签五指标及500次protein-component配对区间由已实现compare输出；目标是New A同时改善New Q及Old Q的RMSE、MAE不恶化。总显存/耗时未实测，用户执行入口后验证；原10小时不能直接外推。实现、完整测试/prepare/参数清单与准确命令见[本轮交付说明](../experiments/lp_pdbbind/persistent_joint_pair_v1/README.md)。

### 5.78.1 Persistent joint-pair输入深度8→32，GPU4准备（2026-11-06）

用户同意扩大MSA深度并指定物理GPU4，GPU仍由用户执行。新分支`review/persistent-joint-pair-depth32-20261106`隔离实现仅A臂的depth32对照，原2个persistent joint-pair blocks和外层8层不变；保留旧selector及每链原8行前缀，完整链node和144维relation同时重提取。窗口512/stride256、邻居/共同窗口/ESM/局部结构/全部loss/随机初始化/seed/batch16/eval6/strict BF16/30轮20,700步/CI优先选模均不变，未增加早停或模型层数。token预算支持32×513=16,416，旧入口默认8上限不变，无显存比例配额。

对照为`.codex_runtime/persistent-joint-pair-aligned-gpu4-20261106_141117-83204/aligned_msa`，不是随机划分或MSA-only版。本轮读取时13轮已完成，最好epoch4 EMA、CI0.704559253/RMSE1.303995321，尚无最终summary；不动原运行目录。新训练直接调用原fit_arm，最终按同2,340个validation key和保存标签重算五指标、输出32−8，不读test、不训练Q、不融合多个预测模型。

CPU回归120项通过；3条真实indexed A3M深度32/旧8前缀/hash/长链窗口通过，同完整初始化`5dbfc734…`，总参数39,043,383/可训练38,506,488；原8行真实缓存train/valid各1条CPU forward通过。新32行GPU特征、smoke及性能尚未运行。全量CPU prepare预扫描约200条后主动中止，命令在新目录重做完整prepare；不将残留目录当成成功。未hash20GB数据库、未运行GPU或读取test。

详见[完整改动、CPU证据、资源与GPU4命令](../experiments/lp_pdbbind/joint_pair_depth32_v1/README.md)。本次仅CPU-ready，不声称32行已提升；离线提取资源由用户首个GPU probe实测，训练特征维度/参数不变。

### 5.78.2 固定32行MSA，将persistent joint-pair从2层加至4层（2026-11-06，CPU-ready）

用户授权按建议推进下一实验、优先结果并接受更慢训练。以`0bd2734`新建`review/joint-pair-depth32-blocks4-20261106`，只新增独立四层包和入口，原2层模型/训练循环/32行提取代码不变。两个新增block独立随机初始化、末端残差投影置零，共享全部原2层干净随机初值；不是旧checkpoint续训。总参数39,043,383→40,696,759，可训练40,159,864，新增1,653,376；共享初始指纹仍`5dbfc734…`，4层完整指纹`3e7a7953…`。

同一套32行node/relation缓存用于2层对照和4层候选，各自独立A臂、30轮/20,700步，不训练Q、不集成。固定train11,037/valid2,340、seed、batch16/eval6、原loss/AdamW/EMA/strict BF16/CI选模，暂不同时改变早停。正式loop直接复用原`_fit_arm`并显式传新model class；checkpoint明确`joint_pair_blocks=4`。最终比较同2,340key/标签五指标，不把8行→32行和2层→4层混为单一加层收益。

147项CPU测试通过；原真实8行缓存train/valid各1条的初始11项输出与2层最大差均0，四层合成CPU三步更新及重载通过。完整几何扫描及正式30轮sampler扫描选出三组大batch，另以当前数据R160/A177/fullchain2701/B16包络执行两步完整GPU训练资源探针（AdamW+EMA常驻），GPU尚未运行。无显存比例cap、无自动降配；4层资源验证安排在两次正式训练之前。完整CPU证据与GPU4命令见[本轮说明](../experiments/lp_pdbbind/joint_pair_blocks4_v1/README.md)。

准备时旧14:11目录13轮已中断/exit130，20:14新目录5轮且manifest仍running，两处都为8行2层，本轮没有干预。尚无完成的32行缓存，默认命令在新目录prepare/extract/audit一次后共用；可显式复用已审计32行输入及已完成同输入2层对照。本轮未运行GPU、读test或hash原始数据库，未得到新性能结果。

### 5.78.3 MSA同源选行代码审查：未发现当前8→32错位，记录边界问题及实际冗余（2026-11-06）

仅审查`b285a66`的现行legacy selector，不改模型/选行/训练配置，不启动GPU。由原train/valid的8,863链清单有界抽查59条indexed A3M（固定hash样本加长度/文件大小边界），独立解析前32行、原8行前缀、逐条hash/index和query/window映射全部一致。54条选满32行，5条源record仅query。node/pair共用alignment及window函数，新增CPU入口拦截测试验证相同行集合；并非分别选同源。

复现旧代码的两个边界问题：非法行先占用ID会跳过同ID的后续合法行；空FASTA header抛IndexError而非预期ValueError。59条真实样本均未触发。样本中6,612条同ID不同alignment前后均字符/长度合法，不能把该计数直接解释为源损坏；当前仍按既定规则保留首次出现。未擅自改冲突策略。

选择仍为query+前31个有效且完整aligned字符串唯一的同源，没有新增coverage/代表性选样。93窗中12窗有裁切后完全重复行；1,674条被选同源有38条coverage<80%、0条<50%。全query-80%冗余代理均值8行2.3614→32行7.4939，中位数1.0000→2.0022（含5条仅query），不是AF标准Neff或性能指标。说明行数增加不保证同等比例的信息增加，不能据此认定性能瓶颈已定位。

最终57项CPU测试通过，完整[审查说明、CPU测试与有界原始统计](../experiments/lp_pdbbind/msa_selection_review_20261106/README.md)随代码推送；保留中间证据和测试fixture笔误的失败记录。本轮未训练/推理、未读test、未全库hash，未改变现行两层/四层实验。

### 5.78.4 下一次MSA输入异常修补与特征算法审查（2026-11-06，CPU-ready）

用户要求修补5.78.3发现的异常、仅下次实验生效，并审查MSA特征算法。新建`fix/msa-input-next-20261106`及独立checkout，不动旧运行目录。新增显式`first_valid_identifier_occurrence_v2`：非法行不再占用ID，合法重复/冲突仍保留首次合法记录；空FASTA/indexed query标题明确ValueError，不跳过/猜测身份。默认legacy调用不变，下一次入口显式启用修补。

修补进入selector/source/node身份、逐链恢复报告与prepare/probe/extract一致策略。仍先核对原record hash及legacy8前缀，再允许可解释的恢复变化；query映射不变。不再把修补可能改变的选择误称为纯8→32前缀实验。新入口只接受新prepare/cache，完整源码身份和CPU证据重新封存。

184项CPU测试通过；59条正常真实A3M修补前后选行/窗口输入全部一致，无恢复；未重新执行GPU encoder。原真实缓存train/valid各1条两层/四层初始11输出最大差0，初始fingerprint不变。全量是否存在恢复留给下次prepare逐链记录，本轮不声称全库无异常。

MSA特征算法审查：symmetrize/APC公式及全真实query窗口→再取边顺序未发现实现错误。32个train/valid复合物抽样中，空间K16保留同链同窗口候选pair的25.06%，序列间隔≥24候选的14.71%；建议后续单独验证固定K预算的空间/MSA高分混合选边。另抽查旧8行16链/13,628条关系，初始LN输出RMS中位数0.97798，10.34%关系通道方差低于eps；支持审视幅度/局部depth/coverage可靠性输入，但不证明LN导致性能问题。不把稀疏覆盖率当作有效信息损失率。

算法建议和单forward提取工程方案均未混入本次修补；仍保留32行、两层/四层单模型对照原训练协议。详见[完整修补、原始证据、优化优先级和下一次GPU4命令](../experiments/lp_pdbbind/msa_input_repair_next_v1/README.md)。无新亲和力性能结果，未运行GPU/读test/全库hash；代码与报告按约定推送。

### 5.78.5 改为32行MSA＋4层joint-pair单A候选优先（2026-11-07，CPU-ready）

用户明确当前优先尽快获得更好结果，不再为每个候选重训归因对照。基于4ee443a建立独立`feature/joint-pair-single-a-20261107`，只更改四层实验编排：prepare封存candidate-only，正式train不调用两层verify_control，新shell无两层smoke/train或paired比较，CPU report仅重算本候选2,340条validation预测。旧双臂代码目录和已有运行不动。

仍为完整ESM＋32行aligned-MSA＋四层交互单预测模型，输入异常修补保留；模型/初始化/数据/seed/loss/batch/EMA/选模及30轮20,700步不变。省去一轮完整两层对照训练，不自动改变早停。特征提取代码未改，支持显式复用已完成、已审计且同身份的修补版32行输入；未完成prepare不能冒充缓存。

99项CPU测试、真实缓存CPU接线及新源码封存通过。新入口、CPU验证及GPU4完整命令见[单A运行说明](../experiments/lp_pdbbind/joint_pair_single_a_v1/README.md)。新summary明确control_summary=null、newly_trained_arm_count=1；主结果为candidate4/training/summary.json及candidate4/candidate_result.json。本轮未运行GPU/读取test，尚无候选新成绩；相关代码和报告推送。

### 5.79 完整链 MSA 替换 ESM 已完训，补齐真正 ESM-only 比较（2026-11-04—07）

原LP GPU6单aligned候选完成30轮／20,700步，epoch4 EMA，validation n2,340：CI `0.706028`、RMSE `1.298527`、MAE `1.010179`、Pearson `0.604664`、Spearman `0.583486`。对strict完整A，CI增加`0.002586`，RMSE降低`0.000304`但MAE增加`0.004898`，误差收益bootstrap区间跨0，没有明确整体收益。这里只删除完整链ESM，局部pocket ESM保留。

原LP真正ESM-only历史CI `0.701382`／RMSE `1.295277`；同dropout0.2的完整v3五项更好，但增益很小。MSA-only context沿strict v2 dropout0.1，不能把三者写成完全同配方三路消融；query-only MSAT不等于ESM-only。原summary/history/同key比较及完整五指标见[补录第2节](LP_PDBBind_recent_experiments_20261108.md#2-原-lp完整链-msa-替换-esm以及真正-esm-only)。

### 5.80 独立随机8:1:1协议及完整版／MSA-only已完成结果（2026-11-04—07）

用户批准合并原LP可用池18,128个唯一key，排序后`random.Random(42).shuffle`，round的8:1:1为train14,502／validation1,813／test1,813。新建`data/LP_PDBBind_random_v1_seed42_20261104`，不改源数据、源划分或缓存；协议SHA `ab40709e0fd67c1d3cf8cffb015a6d0b5e1e15bc0d21a0101284851f57696c24`。数据层保留所有key，缺输入的模型覆盖另报。

用户确认主模型为完整FullChainESMMSAAll v2，而非MSA-only或persistent joint-pair。完整／MSA-only均fresh seed42，30轮27,210步、epoch21 EMA、同1,813 validation key与标签。完整模型五指标为CI `0.794675`／RMSE `1.149747`／MAE `0.884875`／Pearson `0.786684`／Spearman `0.779709`；MSA-only为`0.788675`／`1.182766`／`0.896897`／`0.773906`／`0.766177`。完整五项更好，CI +0.006000、RMSE −0.033019。不是原LP分数，也没有已完成的随机真正ESM-only臂。

完整模型与MSA-only的summary/history/selection、统一指标报告和protocol已归档；准确原路径见[补录第4节](LP_PDBBind_recent_experiments_20261108.md#4-随机划分完整版msa-only-context-和外部模型)。内部固定30轮，没有新增有效早停。

### 5.81 随机外部模型修补、早停和真实完成范围（2026-11-05—08快照）

外部适配修复Attention masked_v2 padding、TankBind原子顺序、CASTER动态sampler及严格加载；新增opt-in validation-CI早停，旧进程不热更新。Graph／3D已完成模型可保留，不一律重训；Deep用户已停，缺完成summary／预测／seal。TankBind GPU6失败，GPU4补跑记录为signal2中断。CASTER／Attention未见本批完训产物。

当前主比较可核实完成3/7，在同1,783 validation key上：完整v2 CI `0.795792`／RMSE `1.146102`／MAE `0.882169`；3DProtDTA `0.790649`／`1.158522`／`0.865078`；GraphDTA `0.735823`／`1.396302`／`1.068271`。自研对3D四项好、MAE差，不包装五项全胜或七模型完整排名。完整五指标、外部早停说明及失败原件见[补录第4节](LP_PDBBind_recent_experiments_20261108.md#4-随机划分完整版msa-only-context-和外部模型)。

### 5.82 冷启动身份修复、轻量迁移包及随机test固定checkpoint结果（2026-11-06—07）

4条train配体身份异常已用同一native-SDF规则修复，完整train索引14,502不删／移样本、不重划分。原1,813条test全部保留：cold_ligand804／cold_protein253／cold_both565／双已见未标191／unknown0。轻量迁移包`data/LP_PDBBind_random_seed42_coldstart_transfer_20261106.tar.gz`约5.6MB，无模型特征／原始结构，供已有官方数据的服务器按pdb_id关联。

validation选出的完整v2 epoch21 EMA固定checkpoint分组推理已完成：全部test CI `0.785159`／RMSE `1.236915`／MAE `0.932426`／Pearson `0.758796`／Spearman `0.758635`。cold_ligand CI `0.806561`，cold_protein `0.759294`，cold_both `0.743364`，双已见`0.840531`。全体不是子组指标平均，test `0.785159`不是validation `0.794675`。

原聚合报告和既有CPU五指标零差核对记录已收录，本轮不重读逐条test标签或预测。用户另服务器外部截图与自研冷启动三组结果的合图在results-comparison分支；截图不能替代同key独立复算。全部分组五指标、人数和标注修复summary见[补录第5节](LP_PDBBind_recent_experiments_20261108.md#5-冷启动标记与随机-test-固定-checkpoint-分组结果)。

### 5.83 随机完整模型三组消融：no_3d已完成，其余未完成（2026-11-08）

独立`random-full-v2-ablations`实现，GPU7运行根`.codex_runtime/random-full-v2-ablations-gpu7-20261108_074857-83204`。同协议fresh seed42、固定30轮／27,210步，不重训full。no_3d已完成，epoch25 EMA，重载差0，统一平均并列秩五指标CI `0.790392`／RMSE `1.184064`／MAE `0.896073`／Pearson `0.778938`／Spearman `0.768027`，五项低于完整模型。selection旧Spearman `0.768033`与统一版仅差约6.1e-6，不改checkpoint。

no_3d删除显式几何／距离路径及结构辅助监督，仍使用相同结构选定口袋，不是完全无结构。no_1d移除全部ESM/MSAT与全链上下文；归档仅有增量history，无最终summary。no_pretrained以随机AA embedding替换ESM/MSAT、保留全链上下文与几何，本批尚无产物。不以未完成臂早期分数当最终对照。定义、CPU准备来源与本次原件见[补录第6节](LP_PDBBind_recent_experiments_20261108.md#6-随机划分三个消融最新运行状态)。

### 5.84 Persistent joint-pair：8行2层单A正式训练已完成（2026-11-06—07）

GPU4原目录`.codex_runtime/persistent-joint-pair-aligned-gpu4-20261106_201437-83204`完成30轮／20,700更新，选epoch4 EMA，重载差0。原LP validation n2,340：CI `0.704559`／RMSE `1.303995`／MAE `1.009105`／Pearson `0.598426`／Spearman `0.575073`。对strict完整A仅部分排序指标升、误差和Pearson差，没有整体收益。不是随机划分，不将前几轮打印值误写成未完训。summary/history/selection见本次归档。

### 5.85 修补版32行4层单A：smoke通过，正式训练16轮后中断（2026-11-07—08）

GPU4输入提取中断后转GPU7续提取，后修补`sampler probe keys changed`，keyfix运行真实四层smoke通过／严格重载差0。正式运行`.codex_runtime/joint-pair-d32-blocks4-single-keyfix-gpu7-20261107_181921-83204`的`failure.json`为`KeyboardInterrupt`，history完成epoch0–15共16轮，无最终summary。已完成history最佳CI为epoch6 EMA：`0.702261`／RMSE `1.302930`／MAE `1.012238`／Pearson `0.592912`／Spearman `0.569512`；最后epoch15 CI `0.684390`。

当前未体现优于8行2层的整体效果，但这是中断记录，不是正式选定／重载后的完训成绩。修补版输入、32行和四层同变，不称纯层数实验；旧smoke失败与本次人为中断分开保留。见[补录第3节](LP_PDBBind_recent_experiments_20261108.md#3-原-lppersistent-joint-pair-最新状态)。

### 5.86 KERMT v2 配体二维分支：全量缓存完成，GPU5修复版smoke已通过（2026-11-07—08）

原LP四层affinity-only基线上新增LN1600＋零初始化Linear1600→512配体原子残差，新增822,912参数；独立checkout／新环境，不修改根原模型或官方KERMT源码。官方encoder 104/104张量严格加载、singleton／按最大入度分组batch特征差0，13,377条原子缓存全部提取并audit。

首次GPU5 DTA smoke因allocator未初始化先reset统计失败；修复显式CUDA初始化顺序，CPU20项通过，旧现场保留并新建plan。新`_cuda_init_fix`目录**真实GPU smoke已通过**：batch16、初始零残差前向一致、投影梯度非零且更新、重载差0；峰值allocated2,171,906,560／reserved2,428,502,016 bytes，22.95秒。归档时尚无正式训练summary／validation比较，不报告收益，不重提已有缓存。probe／audit／失败／修复计划及新smoke原件与实现说明见[补录第7节](LP_PDBBind_recent_experiments_20261108.md#7-kermt-v2-配体二维原子分支)。

### 5.86.1 KERMT GPU5正式完训：未改善四层基线（2026-11-08）

`fullchain_kermt2d_b16_gpu5_20261108_085657-16506_cuda_init_fix`已完成30轮／20,700次参数更新，取代5.86的smoke-only状态。选中epoch5 EMA（第6轮），checkpoint重载差0，训练耗时7,366.24秒。原LP同2,340条validation：CI `0.706107`／RMSE `1.295137`／MAE `1.009735`／Pearson `0.605545`／Spearman `0.582452`。对四层affinity-only基线，五项差值分别为`−0.000970`／`+0.013588`／`+0.015764`／`−0.002884`／`−0.003019`，均变差，本次不替换基线。

epoch5的train loss0.559140，末轮降至0.190392，但validation CI降至0.680264／RMSE升至1.485102，后期出现过拟合；末轮不是最终最佳成绩。全部30轮有效辅助loss为0，其他辅助项仍仅诊断。臂内`validation_failed`是V20 gate未过，不是训练崩溃。

CPU只读复核2,340个唯一key／顺序／标签完全一致，原指标函数五项复算差0；30轮／20,700步、epoch5选模、权重／预测／基线摘要及15份生产源码哈希通过，未读test或运行模型。summary SHA `09a67fb194f4ca6df4f8e0c542a8cf1a56872906a904e11d6ce4a44aada0b2b8`。5份原summary/history/selection/plan/比较和核对原件归档于[正式结果报告](../experiments/lp_pdbbind/fullchain_kermt2d_v1/completed_run_review_20261108/README.md)，随记录提交推送；保留此前失败现场和10:43静态快照，不改模型源码或数据。

### 5.86.2 KERMT未提升的网页审查交接与训练后adapter核对（2026-11-08）

用户要求交网页GPT检查原因。CPU只读选中epoch5 EMA checkpoint：`kermt_proj.weight`的819,200个权重均已离开零初值，L2=5.801985；LayerNorm参数亦变化，全部有限，checkpoint SHA仍为`8113cce5f17a42bf0324eaac26a9c9844ae4a569c516125cb1e03c7f4fa76bd0`。可排除投影始终为零，但未测量实际残差幅度或预测贡献，不把非零权重当作有效特征证据。

静态源码核实：KERMT仅进入主干ligand token、在完整链cross-attention之前；另一套local interface／pair-token路径不直接读取KERMT，符合首轮唯一hook和保留另一配体编码的要求。冻结通用2D特征、路径利用受限、信息冗余及泛化不足是待判别解释，尚不能唯一归因。新增[网页审查入口](LP_PDBBind_KERMT2D_Web_Review_20261108.md)，列出代码导航、已证实事实／未做诊断和审查请求，连同CPU权重统计及复现脚本推送。未运行新模型前向／GPU／训练或test读取，未改生产模型。

### 5.86.3 KERMT固定EMA功能贡献诊断：准备记录，后续已完成见5.86.4（2026-11-08）

按网页审查建议新增独立诊断包，不改原模型／训练源码。固定已选epoch5 EMA，key哈希预选train／valid各32条，三条件为原输入、`kermt_enabled=False`完整旁路、有效原子内两路特征同步固定置换。只读hooks记录有效原子RMS残差比、base／pair及总预测变化；BF16加法舍入单独列项，转换使用checkpoint的label_mean/std。eval／no_grad、无optimizer，前后核对状态与输入hash。不重训、不读test、不搜索缩放或重新选模；小样本指标不充当正式validation分数。

新诊断＋KERMT＋AllV2回归23 passed，旧四层回归5 passed，共28；CPU已实际读取64个预选缓存记录，保持原逐条校验，64/64置换确实改变特征，模型forward=0。GPU尚未执行，不能宣布分支有贡献、没有贡献或定位未提升根因。代码、运行命令、封存plan和CPU原件见[功能贡献诊断](../.codex_runtime/fullchain-kermt2d-v1-20261107/experiments/lp_pdbbind/kermt_contribution_v1/README.md)。相关实现与证据在独立实验分支提交推送，根工作区原模型不改。

### 5.86.4 KERMT固定EMA功能诊断已完成：分支被使用，原子对应未显稳定验证收益（2026-11-08）

GPU5固定epoch5 EMA，预选train／valid各32条，37次batch前向／30.58秒，无训练。正常／关闭／打乱的train RMSE分别1.766874／1.892849／1.783325；valid RMSE分别1.121770／1.167479／1.085307，MAE分别0.837215／0.885511／0.816822。正常相对关闭两组五指标均更好；打乱对误差的影响在train／valid相反，不写成稳定原子对应收益。

有效原子RMS残差／原token的样本均值为train0.770258、valid0.733532；关闭后预测平均绝对变化0.327667／0.354567，打乱后0.096011／0.112483。pair两组全条件变化0，历史fusion为0；变化来自base及单列的BF16舍入。分支并非未接通、没用或残差接近零，但这不证明超出独立基线的泛化价值，也不能唯一归因信息冗余或接入位置。

CPU逐条重算全部小样本指标与组件汇总差0；64个原key／标签／顺序／置换与CPU/GPU输入hash一致，固定checkpoint与GPU前后状态指纹一致，模型权重不变。原全验证2,340条的正式负结果不变，本次不是新全验证成绩。原result／GPU日志、CPU复核及完整解释见[已完成诊断](../.codex_runtime/fullchain-kermt2d-v1-20261107/experiments/lp_pdbbind/kermt_contribution_v1/completed_gpu5_20261108/README.md)，不上传逐条标签／预测、权重或特征；本轮不改模型或启动新实验。相关代码和报告在独立实验分支提交推送。

### 5.86.5 KERMT完整validation八排列诊断：准备记录，后续完成见5.86.6（2026-11-08）

新增独立诊断包，固定同一epoch5 EMA，不修改模型／训练／损失／特征生成。完整原validation 2,340条，normal、disabled和8次同步有效原子排列共10条件；先完整normal复现原archive，预测最大差≤1e-5且旧指标差≤1e-10才允许干预。后续按exact-sequence蛋白组配对bootstrap 2,000次，同时报告legacy与标准average-rank Spearman、全部8排列分布、期望误差和原子数／标签／蛋白组分层，不挑最好排列，不把8排列当8个训练seed。

CPU实际prepare通过：所有2,340条均有8个不同非恒等排列，无无法打乱或必须重复；exact-sequence组1,551个，singleton 1,208个，最大16条；首尾12条真实缓存join及8排列输入不变性通过，模型forward=0。新旧五套回归49 passed，GPU脚本语法通过。实现commit `0aea25bd52e7c5235df313e2a9260a52fd6746a8`，新sealed plan SHA `3c25c6d02928803b8bc8031bfa15f58a11245e17e979b8e1d583748d657157de`。本轮未读train manifest／train标签或test，未运行GPU，**尚无全量干预指标或bootstrap结果**；5.86.4的小样本结论不升级为全验证结论。

完整方法、CPU原件、固定输入身份和仅GPU5的用户命令见[完整validation八排列入口](../.codex_runtime/fullchain-kermt2d-v1-20261107/experiments/lp_pdbbind/kermt_contribution_v1/full_validation_8permutation_20261108/README.md)。完成后仅导出聚合报告和验证记录；逐样本排列／标签／预测留本地。此任务结束后停止，不自动训练后续融合模型。

### 5.86.6 KERMT完整validation八排列诊断已完成：正确对应有收益，候选仍未超基线（2026-11-08）

固定epoch5 EMA、物理GPU5，在同2,340条validation完成10条件／3,900次batch forward，耗时580.41秒，无OOM／跳样本。normal与原archive逐条预测最大差0、旧指标重算差0；关闭分支CI从0.706107降至0.670338、RMSE从1.295137升至1.418075。8次shuffle的CI为0.702340–0.703937、RMSE1.307510–1.311496、MAE1.018589–1.023060；normal五项legacy及标准Spearman均优于所有排列。

按1,551个exact-sequence组配对bootstrap 2,000次，disabled及8个shuffle的RMSE／MAE／CI／Pearson／标准Spearman共45个逐项95%区间下界均大于0。8排列的normal优势中位数：CI+0.002956、RMSE少0.013216、MAE少0.011538；随机排列期望RMSE1.309222、MAE1.020970。主要收益在>10原子及标签Q3／Q4；Q1各排列误差反而略好，Q2的MAE也全部更好，6–10原子层不稳定。正收益最高5蛋白组占正向绝对误差减少4.49%–6.32%，不是少数5组主导，也不声称每组均受益。

CPU再次独立重算全部analysis、逐样本汇总及2,000次bootstrap完全一致，12份原产物哈希及公开白名单通过；checkpoint与GPU前后state相同。全量结果支持当前模型利用正确原子对应，不能沿用此前32条valid的打乱略好推断对应有害。但候选对独立四层基线仍五项略差（基线CI0.707077／RMSE1.281549），本轮不是独立重训消融或新模型提升。

诊断运行commit `2b07c8549b1413a4d9f35b13ccc96d40aa238ec9`；result SHA `8259665cecb3d02738de5f17c509e38fbc72d080bcd23f27987f9678331cd155`。完整10条件、45个配对区间、8排列分布、分层及原件见[完成报告](../.codex_runtime/fullchain-kermt2d-v1-20261107/experiments/lp_pdbbind/kermt_contribution_v1/full_validation_8permutation_20261108/completed_gpu5_20261108/README.md)。未训练、未读train manifest／train标签或test、未改模型；本轮结束，不自动开始新融合实验。

### 5.87 本次实验记录统一与推送范围（2026-11-08）

此前最近实验确实未全部合并：原工作区记录只到5.54，独立分支分别有selector／交互、随机数据／消融、MSA替换／截图及KERMT材料。现保留历史准备条目，补齐四层／SAM／固定checkpoint诊断、MSA-only和真正ESM-only比较、随机完整与外部状态、冷启动标注／test、joint-pair完训及中断、三组消融进度、KERMT新smoke。所有具体状态以本次已存小文件快照为准，不保证仍在写入的history此后不变。

[近期总览](LP_PDBBind_recent_experiments_20261108.md)和[39份summary/history/selection/聚合指标/失败原件及SHA清单](../experiments/lp_pdbbind/recent_record_20261108/README.md)与本记录一并提交推送。未上传权重、特征、原始数据或逐样本test预测；本轮没有训练／推理，没有改原实验协议或模型源码。

## 6. 清理前保留规则

1. 保留本文件、`docs/LP_PDBBind_analysis_processing_manual.md`、每个表中列出的最佳权重、`best_metrics.json`、最终 test 指标和训练参数。
2. 六臂训练及 validation 导出已封存，test 输入也已无标签释放；清理前不删除 checkpoint、运行 manifest、历史、日志、验收材料、预测归档、释放来源报告或必要输入缓存。
3. 当训练结束后，结果目录内可删除未被本文件引用的中间 checkpoint、逐 batch 日志和可重建 cache；先保留 `best_model.pt`/`model.pt`、`best_metrics.json`、最终 metrics、`run_args.json` 与本文件中引用的预测或 ensemble JSON。
4. 所有非 LP-PDBBind 数据、其他旧划分模型目录和历史训练权重，都须先生成逐路径清单并经确认后处理；本轮新随机协议、近期运行产物与本记录归档不在旧清理范围。本轮未删除任何产物。
