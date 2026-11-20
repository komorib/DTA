# LP-PDBBind 近期实验补录（2026-11-08）

后续更新：KERMT已完成正式训练，五项均略差于四层基线；见[第8节](#8-kermt正式完训后续更新)。第1–7节及39份原件保留10:43静态快照，不将过去的smoke-only状态继续当作当前状态。

最新补充：KERMT固定EMA三条件功能诊断已完成，分支有明显预测影响，但未显示正确原子对应的稳定小样本验证收益；见[第9节](#9-kermt固定ema功能诊断已完成)。

最新全量结果：第10节已完成原2,340条validation八排列诊断，正常稳定优于所有shuffle且配对区间支持收益；第9节小样本观察不代表全量。KERMT候选仍未超过独立四层基线。

此前没有全部汇入同一份总记录：原工作区停在 5.54，各实验分支分别追加了后续内容；部分准备条目也没有更新成真实运行状态。本次回收既有历史记录，并补齐下列结果。本文是静态归档快照；未完训项目不按最终结果记账。epoch 均沿用产物中的 **0 起算**。

所有指标来自已保存的 summary／selection／聚合比较报告。本轮只核对小文件及归档 SHA，不重新训练、推理或读取逐条 test 标签／预测。[原件与哈希索引](../experiments/lp_pdbbind/recent_record_20261108/README.md)随报告推送。原 LP validation、随机 validation、共同子集和 test 分开记录，不混用。

## 1. 原 LP：四层、SAM 和固定 checkpoint 诊断

同一原 LP validation 2,340 条；四层 AdamW 与 SAM 均已完成 30 轮／20,700 次参数更新，checkpoint 重载预测差为 0。`validation_failed` 是未通过历史 V20 gate，不是训练崩溃。

| 候选 | 选中 epoch／版本 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 八层 affinity-only | 2 EMA | 0.706457 | 1.288747 | 1.001928 | 0.608739 | 0.583349 |
| 四层 affinity-only AdamW | 2 EMA | 0.707077 | 1.281549 | 0.993971 | 0.608429 | 0.585471 |
| 四层 SAM-AdamW | 4 EMA | 0.698825 | 1.309870 | 1.017246 | 0.584349 | 0.561337 |

四层略优于八层的 CI／误差，但 Pearson 略低；SAM 相比四层 AdamW 五项均退化（CI −0.008252、RMSE +0.028320），没有提升。SAM 记录 41,400 次 backward，不写成 41,400 次优化器更新。

固定四层 epoch2 EMA 的全 train eval 已完成：n=11,037，CI 0.772790、RMSE 1.285558、MAE 0.987886、Pearson 0.731496、Spearman 0.735415。12 条 validation 预检差为 0，模型指纹不变，优化器／梯度步骤为 0。Train 排序相关性更好，绝对误差与 validation 接近；这是已见数据诊断，不是新的独立泛化分数。

原件：`sources/depth4_*`、`sources/sam_*`、`sources/fixed_checkpoint_train_eval.json`。原运行根分别为 `Model/LP_PDBBind_FullChainAffinityDepth4_v1/fullchain_affinity_depth4_b16_20261021_215826`、`Model/LP_PDBBind_FullChainAffinityDepth4SAM_v1/fullchain_affinity_depth4_sam_b16_20261024_143212`。

## 2. 原 LP：完整链 MSA 替换 ESM，以及真正 ESM-only

这里的 MSA-only context **只删除完整链 ESM，保留局部 pocket ESM**。不是全模型只用 MSA，也不是下面随机划分的结果。

| 原 LP validation，n=2,340 | 配方 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 完整 ESM＋MSA，strict v2 | dropout 0.1 | 0.703442 | 1.298831 | 1.005282 | 0.600499 | 0.572085 |
| MSA-only context | strict v2，去完整链 ESM | 0.706028 | 1.298527 | 1.010179 | 0.604664 | 0.583486 |
| 完整 ESM＋MSA，v3 | dropout 0.2 | 0.705546 | 1.289022 | 1.001342 | 0.605028 | 0.580166 |
| 真正 ESM-only v1 | v3 配方，去完整链 MSA | 0.701382 | 1.295277 | 1.005807 | 0.597305 | 0.569301 |

MSA-only context 已在 GPU6 完训，epoch4 EMA；对 strict 完整 A，CI +0.002586、RMSE −0.000304、MAE +0.004898，没有明确整体收益。500 次 protein-component bootstrap 的误差收益区间均跨 0。真正 ESM-only 的同配方对照是 v3：加入 MSA 后五项更好，但 CI 仅 +0.004164、RMSE −0.006255。两种 dropout／配方不能合称完全同条件三路消融；query-only MSAT 也不是 ESM-only。

来源：`sources/msa_context_comparison.json`、`sources/esm_only_comparison.json`。原运行 `.codex_runtime/fullchain-msa-only-context-gpu6-20261104_151052`；此前完整图表与原件见 [results-comparison 分支](https://github.com/komorib/DTA/tree/review/results-comparison-20261107/experiments/lp_pdbbind/results_comparison_20261107)。

## 3. 原 LP：persistent joint-pair 最新状态

| 版本 | 实际状态 | epoch／版本 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8行 MSA＋2层交互，单 A | 完成30轮／20,700步 | 4 EMA | 0.704559 | 1.303995 | 1.009105 | 0.598426 | 0.575073 |
| 32行 MSA＋4层交互，修补版单 A | 完成16轮后中断，非最终 | 已完成 history 中最佳为6 EMA | 0.702261 | 1.302930 | 1.012238 | 0.592912 | 0.569512 |

两者均使用原 LP validation 2,340 条，不是随机划分。8行2层相对 strict 完整 A，CI 小升、RMSE／MAE／Pearson 变差，无整体提升。32行4层 `failure.json` 为 `KeyboardInterrupt`；最后完成 epoch15，CI 0.684390／RMSE 1.412763。表中 epoch6 仅为已写入 history 的 CI 最佳记录；尚无最终 summary、正式重载或完整报告，不能宣称完训或将它当最终 checkpoint 结果。

32行输入曾在 GPU4 中断，后在 GPU7 复用已验条目续提取；旧 smoke 的 `sampler probe keys changed` 已修补，keyfix 运行的真实四层 smoke 通过、严格重载差为 0。训练中断不是该旧 smoke 失败。输入修补、32行及层数等变化并存，也不是纯层数单变量实验。

8行2层原目录 `.codex_runtime/persistent-joint-pair-aligned-gpu4-20261106_201437-83204`；32行4层原目录 `.codex_runtime/joint-pair-d32-blocks4-single-keyfix-gpu7-20261107_181921-83204`。原件：`sources/joint8_*`、`sources/joint32_*`。此前 MSA 选行、异常修补、算法审查及单 A 编排已保留在总记录 5.78.1–5.78.5。

## 4. 随机划分：完整版、MSA-only context 和外部模型

合并已有 LP 可用池 18,128 个唯一 key，新建独立数据目录，排序后 `random.Random(42).shuffle`，按 round 的 8:1:1 得到 **14,502 train／1,813 validation／1,813 test**；源数据和原划分未修改。协议 SHA-256 `ab40709e0fd67c1d3cf8cffb015a6d0b5e1e15bc0d21a0101284851f57696c24`。

同 1,813 validation key／标签、同 seed42 和共享训练配置，两模型均完成30轮／27,210步，均 epoch21 EMA：

| 随机 validation | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整 FullChainESMMSAAll v2 | 0.794675 | 1.149747 | 0.884875 | 0.786684 | 0.779709 |
| MSA-only context | 0.788675 | 1.182766 | 0.896897 | 0.773906 | 0.766177 |

完整版五项更好：CI +0.006000、RMSE −0.033019、MAE −0.012023。随机划分未归档到已完成的真正 ESM-only 臂。完整版不是 persistent joint-pair，也不是四层 KERMT 基线。

可核实完成的主比较仍为 **3/7**，共同 validation n=1,783（不是把自研原生 n=1,813 分数混入外部表）：

| 同1,783条 validation | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整自研 v2 | 0.795792 | 1.146102 | 0.882169 | 0.789294 | 0.781719 |
| 3DProtDTA | 0.790649 | 1.158522 | 0.865078 | 0.778571 | 0.766232 |
| GraphDTA-GIN | 0.735823 | 1.396302 | 1.068271 | 0.672067 | 0.648574 |

自研对 3DProtDTA 四项更好，但 MAE 更差；不是七模型完整排名。GraphDTA 已228轮／best198，3DProtDTA 已700轮／best670，可保留；DeepDTAGen 被用户停止，缺最终 summary／keyed预测／seal；TankBind GPU6 失败，GPU4 补跑阶段被 signal2 中断；CASTER／Attention 未见本批完成产物。未把 best_metrics 单文件当成完训。

外部修补已包括 Attention masked_v2 padding、TankBind 原子顺序、CASTER 动态 sampler／严格加载及 opt-in validation-CI 早停。旧运行不会热更新。自研本批完整 v2／三个消融仍固定30轮，不把配置里未生效的 patience 当成早停。

来源：`sources/random_validation_ablation.json`、`sources/common_validation_three.json`、`sources/random_full_*`、`sources/random_msa_*`、`sources/tankbind_*`。详细修补与已完成外部原件见 [外部适配分支](https://github.com/komorib/DTA/tree/review/external-baselines-pl-comparison-v2-20261105/experiments/external_benchmarks/pl_comparison_v2)和[完整随机主比较分支](https://github.com/komorib/DTA/tree/review/lp-pdbbind-random-full-v2-20261105/experiments/lp_pdbbind/random_full_v2)。

## 5. 冷启动标记与随机 test 固定 checkpoint 分组结果

4条 train 配体异常已按同一 native-SDF 身份规则恢复，train 身份索引仍为完整14,502条，没有丢弃／移动／改划分。1813条 test 全保留：cold_ligand 804、cold_protein 253、cold_both 565、双已见未标类别191、unknown 0。此前因 train 异常连带 unknown 的旧输出不能作为最终类别计数。

完整版使用 validation 选出的 epoch21 EMA 固定 checkpoint，推理不训练、不用 test 选模。现有聚合报告如下：

| 随机 test 分组 | n | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 | 1,813 | 0.785159 | 1.236915 | 0.932426 | 0.758796 | 0.758635 |
| cold_ligand | 804 | 0.806561 | 1.133979 | 0.849388 | 0.801236 | 0.802251 |
| cold_protein | 253 | 0.759294 | 1.397579 | 1.075057 | 0.715941 | 0.701388 |
| cold_both | 565 | 0.743364 | 1.360609 | 1.036610 | 0.659599 | 0.670169 |
| 双已见未标类别 | 191 | 0.840531 | 1.022825 | 0.784848 | 0.865878 | 0.861398 |

Validation CI **0.794675** 与 test CI **0.785159** 是不同集合，不是同一分数被修改。全体指标由全部样本计算，不是子组指标平均。既有 CPU 核对记录五指标复算差为0；本轮仅读取该聚合报告，不重新触碰逐条 test 标签／预测。

已有轻量迁移包 `data/LP_PDBBind_random_seed42_coldstart_transfer_20261106.tar.gz`（约5.6MB），只含划分、通用输入和标记，不含预计算特征／原始结构。对方已有官方原始数据时按 pdb_id 关联即可。用户另服务器六模型 test 截图的合图保留在 results-comparison 分支；截图四位精度且无逐key核对，不能当作此处同key外部复算表。

来源：`sources/annotation_summary.json`、`sources/coldstart_group_metrics.json`、`sources/coldstart_cpu_verification.json`；原报告 `.codex_runtime/random-full-v2-coldstart-eval-20261106/group_metrics.json`。

## 6. 随机划分三个消融：最新运行状态

运行根 `.codex_runtime/random-full-v2-ablations-gpu7-20261108_074857-83204`。同随机协议、fresh seed42、30轮／27,210步，完整基线不重训。

| 候选 | 归档快照状态 | CI ↑ | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 完整模型参照 | 已完训 | 0.794675 | 1.149747 | 0.884875 | 0.786684 | 0.779709 |
| no_3d | 已完训，epoch25 EMA | 0.790392 | 1.184064 | 0.896073 | 0.778938 | 0.768027 |
| no_1d | 仅有增量 history，未完成 | — | — | — | — | — |
| no_pretrained | 尚无本批运行产物 | — | — | — | — | — |

no_3d 相比完整模型五项变差：CI −0.004283、RMSE +0.034317、MAE +0.011198。使用 summary 的 `reported_validation_metrics`（平均并列秩版）；其历史 selection Spearman 为0.768033，统一报告为0.768027，差约6.1e-6，不重选 checkpoint。

no_3d 去掉显式坐标、几何量／距离路径和结构辅助监督，**仍用相同的结构选定口袋**，不是完全无结构。no_1d 删除所有 ESM／MSAT 和完整链上下文，保留局部随机AA身份和结构；no_pretrained 用随机AA embedding 替代局部／完整链 ESM／MSAT，保留上下文与几何。未完训臂的早期分数不写进最终比较。

来源：`sources/no3d_summary.json`、`sources/no3d_selection.json`、`sources/no3d_history.json`、`sources/no1d_history_snapshot.json`。原准备／CPU验收见 [消融分支](https://github.com/komorib/DTA/tree/review/random-full-v2-ablations-20261107/experiments/lp_pdbbind/random_full_v2_ablations_v1)。

## 7. KERMT v2 配体二维原子分支

本节为10:43快照；随后正式完训结果见第8节。

这是**原 LP 四层 affinity-only**单候选，不是随机数据集或 joint-pair。新代码与环境在独立 checkout／环境，未修改根工作区原模型或官方 KERMT 源码。

官方冻结 encoder 104/104 张量严格加载；真实 singleton／分组batch／反序batch提取差0；13,377条原子特征全部提取、audit覆盖 train11,037／valid2,340。新增 LN1600＋zeroLinear1600→512，残差接在 ligand_norm 后、全链 cross-attention 前，新增822,912参数。

首次 GPU5 DTA smoke 在首次前向前因 CUDA allocator 未初始化就 reset 显存统计失败。已修复显式初始化顺序、保留失败目录，用新 `_cuda_init_fix` plan；20项CPU回归通过。**修复版 GPU5 模型 smoke 现已通过**：batch16、零初始化前向与基线一致、投影梯度绝对和156.825653且参数已更新、重载预测差0，峰值allocated2,171,906,560／reserved2,428,502,016 bytes，耗时22.95秒。smoke 不产生亲和力 validation 成绩。

归档时新目录尚无训练 summary／validation_comparison，未报告候选收益；不把 smoke 更新当成正式训练。原完整缓存可复用，不重新提取。入口、固定30轮／20,700步配置、环境、官方来源及实现细节见 [KERMT准备说明](https://github.com/komorib/DTA/blob/feature/fullchain-kermt2d-v1-20261107/docs/LP_PDBBind_KERMT2D_Preparation_20261108.md)和[实验README](https://github.com/komorib/DTA/blob/feature/fullchain-kermt2d-v1-20261107/experiments/lp_pdbbind/fullchain_kermt2d_v1/README.md)。smoke原件 `sources/kermt_gpu5_smoke_passed.json`，其他 probe／audit／失败原件保留在该实验的 `review_evidence/`。

## 8. KERMT正式完训后续更新

新增[网页GPT审查入口](LP_PDBBind_KERMT2D_Web_Review_20261108.md)：包括相关源码、训练原件、最佳checkpoint投影已离开零初值的CPU统计，以及“只进主干token、不直接进入另一套local interface／pair-token”的接入边界。原因仍待判别，本轮不改模型或启动新实验。

GPU5已完成30轮／20,700步，选中epoch5 EMA（第6轮），checkpoint重载差0。原LP同2,340条validation为CI **0.706107**、RMSE **1.295137**、MAE **1.009735**、Pearson **0.605545**、Spearman **0.582452**。相对四层基线分别为−0.000970、+0.013588、+0.015764、−0.002884、−0.003019，五项均变差，本次无提升，不替换基线。

末轮train loss继续下降至0.190392，但validation CI降至0.680264，表现为后期过拟合；最后一轮不是最佳checkpoint成绩。CPU同key／标签及原指标函数五项复算通过、最大差0，未读test或执行新训练／推理。[完训报告与5份原始summary/history/selection/plan/比较](../experiments/lp_pdbbind/fullchain_kermt2d_v1/completed_run_review_20261108/README.md)另行归档，不覆盖原39份历史快照。

## 9. KERMT固定EMA功能诊断已完成

GPU5固定epoch5 EMA、train／valid预选各32条，正常／完整旁路／有效原子内同步置换，37次batch前向，30.58秒，无训练／test读取。RMS残差／原token均值0.770258／0.733532，关闭后预测平均绝对变化0.327667／0.354567，说明分支功能影响明确，不是残差几乎为零。pair变化0、历史fusion0，权重前后不变；CPU从逐样本预测复算汇总差0，原key／标签／置换与输入hash一致。

正常／关闭／打乱的train RMSE分别1.766874／1.892849／1.783325；valid RMSE1.121770／1.167479／1.085307，MAE0.837215／0.885511／0.816822。正常优于关闭，但正确原子对应的误差收益在两组间不稳定；不把小样本分数当新正式成绩，不改变2,340条全验证的负结果。不改融合或重训，原件及解释见[已完成诊断](../.codex_runtime/fullchain-kermt2d-v1-20261107/experiments/lp_pdbbind/kermt_contribution_v1/completed_gpu5_20261108/README.md)。

## 10. KERMT完整validation八排列诊断已完成

固定同一epoch5 EMA／GPU5，normal、disabled、shuffle_0…7全部覆盖原2,340条；normal历史复现差0。normal的CI0.706107／RMSE1.295137，关闭后为0.670338／1.418075，8次打乱为CI0.702340–0.703937／RMSE1.307510–1.311496。按1,551个exact-sequence组配对bootstrap 2,000次，9干预×5项指标的45个逐项95%区间均支持normal更好。支持固定模型使用正确原子对应；此前32条valid打乱略好未在全量重现。

收益不均匀：>10原子及标签Q3／Q4稳定正向，Q1反而打乱更好、6–10原子层不稳定。CPU再次独立重算全部指标与2,000次bootstrap一致；无OOM／跳样本、无训练／test读取、未改模型。独立四层基线仍为CI0.707077／RMSE1.281549，本轮不是新模型提升。完整10条件、全部区间、分层和公开原件见[全量完成报告](../.codex_runtime/fullchain-kermt2d-v1-20261107/experiments/lp_pdbbind/kermt_contribution_v1/full_validation_8permutation_20261108/completed_gpu5_20261108/README.md)。任务结束，不自动开始下一项训练。
