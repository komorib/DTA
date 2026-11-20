# LP-PDBBind 外部模型公平重训与共同样本评估协议

状态（2026-10-03）：协议、五个候选的真实 train 样本架构/输入门槛、六模型的验证预测导出与隔离式启动器已准备；AttentionMGT-DTA 所需官方 ESM2 8M 权重已取得并验证为原生 320 维。逐 key CPU 资格账本已完成，13,935 条名义 train/val 中 11,448 条合格、2,487 条明确排除；原生图/tokenizer 全量预检、真实 320 维 train/val 特征和真实样本前后向尚未完成，因此六模型整组训练仍被门禁阻断。当前 `ln02` 会话未暴露 CUDA，未做正式 GPU 重训或 test 推理。用户已明确限定为 **LP-PDBBind 官方 `new_split`**、每模型**单种子**，可训练网络均从随机权重开始。本计划不改变现有模型晋级门槛。由于旧 LP-PDBBind test 已被历史实验多次查看，最终只能称“回顾性共同样本比较”，不是全新盲测。

## 1. 必须区分的两种证据

1. **第一层：相同 LP test 样本的回顾性比较**。冻结已有 checkpoint，在 LP-PDBBind `new_split` 的共同 test key 交集上重新推理/复算。它解决当前总表有效样本数不同的问题，但训练输入、预训练资源和选模规则仍可能不同。
2. **本轮主实验：统一 LP train/valid 协议后的重训比较**。用户已选定**每模型单种子**；执行种子拟统一为 `42`，须与完整配置一起在首次正式训练前封存。六个外部架构都须先固定上游源码 revision、完整架构配置、输入适配与预声明预算，再在同一个 LP test key 交集上评估。它改善训练协议的可比性，但因旧 LP test 已被查看，仍不是新的盲测；单种子不能证明稳定性。

## 2. 候选分层，不混称同一种模型

| 层级 | 候选 | 当前证据与边界 |
| --- | --- | --- |
| **本轮候选：LP-PDBBind 上可训练网络随机初始化的外部架构** | 3DProtDTA、GraphDTA **GIN 变体**、DeepDTAGen、TankBind、CASTER-DTA、AttentionMGT-DTA | 仓库有各自 upstream 模型代码及本地 LP 数据适配器、历史 best checkpoint；有效 test 样本数不同。AttentionMGT-DTA 的上游输入含预训练 ESM embedding，必须显式标注并与纯原始输入组分层。重新训练前仍须逐一核对输入/架构/选模配置，不能仅凭“官方架构”四字认定论文协议完整复现。 |
| 预训练表征＋LP 读出 | Boltz pairwise | 冻结 Boltz-2 embedding 上训练自定义 `PairwiseTwoBranch`，**不是完整 Boltz-2 端到端模型**；预训练来源和与 LP-PDBBind 的潜在重叠必须单列。 |
| 官方预训练权重直接推理 | 历史 TankBind、CASTER-DTA 预训练权重 | 没有在 LP-PDBBind 从零训练，且可能含 PDBbind 来源重叠；只列探索性参照，不混入从零训练排名。 |
| 项目内模型 | GerNA 单模型、完整链模型、V20 融合系统 | 单模型与多级融合分栏；现有 validation gate 未通过的版本不得因本计划自动读取旧 test。 |

## 3. 架构真实性硬门槛

每个候选训练前必须生成一份**通过/不通过**的架构验收记录，至少包含：上游公开仓库 URL 与不可变 commit、模型入口类、完整构造参数、模型源文件及本地适配器 SHA-256、参数量/模块树、随机初始化证明、一次真实 LP train 样本的全部必需输入与有限前向/反向，以及任何上游源码改动和缺失/替代特征。来源仓库代码被本地修改、模型分支被删、使用虚构零特征占位、或无法说明某个必需输入的来源时，不得以“完整官方架构”进入主表。构造方式应调用上游模型类；本地适配器只负责 LP 数据转换、训练/评估封存，不得悄悄替换核心网络。

“从零训练”明确指本轮**可训练权重**从随机初始化开始、无旧 LP 或官方任务 checkpoint 转移。若上游架构规定使用冻结的预训练输入编码器，该编码器必须单列来源/权重/可能的数据重叠，不得暗称整个系统无预训练；涉及预训练输入的模型应在结果内显式分层，不能与只用原始输入的模型作无条件架构排名。现有 Boltz pairwise 属于自定义读出，留在参考栏，不进入六模型主表。

现在五个候选通过了“固定上游网络＋真实 LP train 输入前后向”门槛；AttentionMGT-DTA 只通过完整上游网络的合成 CPU 前向及逐 key 输入资格审计，真实原生 320 维输入仍被门禁阻止。门槛通过不等于论文数据预处理逐字节复现，也不等于全量样本构图成功。六个历史运行不能追认为满足新协议。

已定位的上游代码快照（只证明来源可追踪，不代表架构验收通过）：

| 候选 | 上游仓库 commit | 当前适配器实例化的上游入口 |
| --- | --- | --- |
| 3DProtDTA | `800bdeabf5bc4c12f02f6d323050ac29644dd315` | `DTIProtGraphChemGraphECFP` |
| GraphDTA GIN | `6162245c6776684b391d30610850fe7077478435` | `GINConvNet` |
| DeepDTAGen | `de7fdd0a20c3331ea6de8f1e3e70209f85736910` | `DeepDTAGen` |
| TankBind | `ff85f511db11d7a3e648d2e01cd6fdb4f9823483` | `get_model(mode, ...)` |
| CASTER-DTA | `2fbe7a46b968184589984a34cf7966a75e9d53b3` | `JointGNN` |
| AttentionMGT-DTA | `e94a28dad3642abab82f353f799aa1246e7ab0dc` | `MDGTDTInet` |

准备阶段的 `scripts/audit_lp_external_source_snapshots.py` 已在当前工作区执行：六个上游 origin、commit、主要模型源文件 SHA-256 均与固定清单一致，未见这些上游仓库的 tracked Python 源码改动；结果为 `source_snapshot_passed`。该检查**只验来源，不验是否完整实例化、输入是否真实或是否复现论文训练协议**。这些更强的门槛仍待逐模型真实样本 smoke 和配置审计。

当前架构/真实 train 输入审计状态（证据：`docs/LP_PDBBind_external_architecture_evidence_4.json` 与 DeepDTAGen 单样本报告）：

| 候选 | 已核对 | 尚不能声称 |
| --- | --- | --- |
| 3DProtDTA | 上游完整三分支网络、真实 train 蛋白图／配体图／ECFP，CPU 前后向有限且各分支非零梯度 | 本地 LP 构图与上游 Davis/KIBA 预计算图完全等价；全量构图成功率 |
| GraphDTA GIN | 上游完整 GIN、真实 train 图／序列，CPU 前后向通过；公平入口按 validation CI 而非历史 RMSE 选模 | >1000 残基截断与多链分隔处理无影响；全量转图成功率 |
| DeepDTAGen | 上游完整 affinity＋生成分支、真实 train 样本前后向通过；无标签预测与完整 forward 及改变/删除 `y` 的 affinity 输出相同；train-only tokenizer | 全量覆盖、性能提升或 `y` 在辅助分支中的影响可外推为 test 标签可输入推理 |
| TankBind | 上游 mode 0 网络、真实 train GVP／GIN／三角模块、接触＋亲和力目标前后向通过，含实际口袋中心噪声 | 原生配体位姿／COM 口袋与序列模型输入信息量相同；全量构图成功率 |
| CASTER-DTA | 上游 JointGNN、真实 train 蛋白／配体图及交叉注意力前后向通过；官方配置关闭 ESM2，train-only 标签标准化 | 全量图构建成功率或论文所有预处理步骤逐字节一致 |
| AttentionMGT-DTA | 上游完整 MDGTDTInet 的合成 320 维输入 CPU 前向通过；官方 ESM2 8M 权重已验证 6 层/320 维；公平入口拒绝现有 640 维截取或零填充 | 真实原生 320 维 train/val 提取、合格样本全量映射与真实样本前后向；目前 **不准启动** |

前五行的放行范围仅是“完整上游网络＋真实 LP train 输入接线”；第六行仍阻断。通过者的参数量、所用 train key 和各分支梯度见机器可读证据。所有模型的全量输入过滤数要以未来新鲜 train/val cache 的实际日志为准。

## 4. 已发现的协议差异，执行前必须处理

- LP-PDBBind 原始 `new_split` 为 train/val/test `11,513/2,422/4,860`；各模型过滤后 test 有效样本约 `4,716–4,838`，V20 joined cache 为 `4,751`。先报告每模型原生覆盖率，再在**同一组 key 与同一 canonical label** 上做主要横向比较。不得将不同 `n` 的原表 CI 当作成对胜负。
- 3DProtDTA 新增的 `--fair-train` 入口只接受物理上仅有 train/val 的新 CSV、新数据目录与新输出目录；其 test fold 为空，训练不使用 test，并严格按 validation CI 选模，不在 CI 无效时回退到 validation loss。历史默认入口保持不变。真实 LP train 样本前后向现已通过；另已修正上游 CI 对验证行顺序的依赖，fair 路径改用与统一评估一致的精确 CI，乱序标签反例已通过，旧入口不变。
- GraphDTA **历史默认入口**按 validation **RMSE** 保存 best，而总实验记录写“max validation CI”；须先核对归档运行的真实选模依据并修正说明。新增公平入口才按 validation CI 选模，不能直接把旧 GraphDTA best checkpoint 称为 CI 选出的模型。
- 新增的 GraphDTA `--fair-train` 入口只接受**物理上仅含 train/val 行**的独立 CSV，拒绝已有 processed cache、已有输出目录、样本数裁剪和 cache 删除；训练时不构造 test dataset/loader、不计算或记录 test，并以最高 validation CI 保存 best。旧入口保持原行为以便历史运行可复现。真实 LP train 样本前后向已通过，尚未正式 GPU 重训；这也不能替代其他模型各自的 test 隔离证据。
- GraphDTA、DeepDTAGen、TankBind、CASTER-DTA、AttentionMGT-DTA 的**历史默认入口**在训练/定期验证时计算 test。新增公平入口避免读取/构造/评估 test，但这些入口仍需真实样本验收。旧结果仍可作回顾性运行记录，不能追认已满足新一轮 test 封存。3DProtDTA 历史入口将 validation 训练与 checkpoint test 评估分开，但准备数据时仍纳入 test；新增公平入口避免此事。
- TankBind 新增 `--fair-train` 入口使用独立 train/val CSV、新 cache/output，只为 train/val 构造特征与 loader，严格按 validation CI 选模；完整上游网络和真实 LP train 输入已完成有限验收。其 native ligand COM 口袋定义提供了与序列模型不同的结构/位点信息，不能据此作等信息量架构排名。
- CASTER-DTA 历史适配器把 train/val/test 同时交给上游 dataset 的 `scale_output=["standardize"]`，其均值/标准差会从**所有标签**计算。新增公平入口先禁用该自动变换，再仅用构图成功的 train 标签拟合相同标准化形式并施加到 train/val；这一点有合成单测，但仍需真实样本检查与上游配置对照。
- DeepDTAGen 历史默认入口在 `prepare_lp_dataset()` 中用 train、val、test 全部 SMILES 建 tokenizer，属于测试输入参与训练侧预处理。新增 `--fair-train` 入口要求独立 train/val CSV 与全新 cache/output，只用 **train SMILES** 建词表，不构造 test cache/loader、不计算 test；未知 token 沿用上游 tokenizer 的 `<mask>` 行为并报告计数，模型仅按有限 validation CI 选 best。真实 LP train 样本前后向已通过，长 SMILES 截断策略已在 `protocol.json` 封存。
- AttentionMGT-DTA 当前 LP 适配器从 `esm2_embedding` 构造 320 维残基输入；匹配不到残基或 ESM 时使用零向量。抽查一个 **train** key `10gs_protein`，缓存中两条链的 ESM2 特征形状均为 `[208, 640]`，而 `resize_embedding()` 直接取前 320 维；这不是经证明与上游 320 维 ESM 特征等价的转换。上游模型也要求 ESM 表征，因此“网络随机初始化”不等于“完全不使用预训练”；缺失 ESM 的零占位与维度裁剪都必须解决或单列为有意的输入适配，不能静默通过完整架构验收。此抽查没有访问 test。
- AttentionMGT-DTA 原公平入口在构图/训练前要求**全部 train/val 行**具备源链 LMDB、口袋 PDB 和原生 320 维特征。只读静态交集核查发现官方 `13,935` 行里源链 LMDB 缺 `460` 行，其中 `332` 行同时缺口袋 PDB；原全量提取必定失败，不能直接在 GPU6 启动。应先封存逐 key 的原生输入合格/排除账本，严格验证合格样本的残基映射，不允许缺失输入以 640 维截取或零向量替代。官方 ESM2 8M 权重已从 fair-esm 官方地址取得，SHA-256 `46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928`，CPU 检查为 6 层、320 维；这还不是全量特征或真实 LP train smoke。具体来源和提取入口见 `experiments/lp_pdbbind/attentionmgt_fair_v1/README.md`。
- DeepDTAGen 的位置编码长度为 138，历史默认对超长 SMILES **截断**，也支持丢弃；正式策略必须预声明。TankBind 使用 native ligand COM 定义口袋；AttentionMGT-DTA 使用本地构造的图/残基特征。神经网络类来自上游仓库，不等于输入构建和论文全部训练协议已经证明等价。
- LP-PDBBind `new_split` 不是未见蛋白、未见配体或全新时间外推的同义词。若要测这些能力，应另立分组/去重协议；不得在看过新测试结果后更改分组。

## 5. 执行器的最低验收条件

1. 在**不读取 test 标签**的准备阶段，列出每个候选的源码 revision、配置、checkpoint SHA-256、数据缓存/过滤规则、原生 train/valid/test key 覆盖清单；确认 key 唯一和 split 隔离。未确认来源的权重不进入主表。
2. 预先锁定候选和主指标（CI；RMSE、MAE、Pearson、Spearman 同报）、validation 选模规则（最高 CI；严格并列取较早 epoch）、单种子/训练预算、失败样本处理、共同交集定义和统计区间方法。不得因 test 结果追加模型或修改阈值。各架构可以使用其预先声明的训练预算，但不得看 test 后再调；统计区间只能描述这一次固定模型在样本上的不确定性，不能伪装成跨种子稳定性。
3. 每模型输出 `key, prediction, input_status`；标签只由统一评估器从 canonical manifest 读取。评估器按 key join，拒绝重复、缺失、非有限值和标签冲突，记录每模型原生覆盖与全体共同交集，分别计算两张表。共同交集的所有模型必须严格使用同一组样本。
4. LP test 路径仅在 checkpoint/配置/代码和预测生成器封存后由独立评估器统一执行；封存 SHA-256 与评估日志一同存档。本轮即使实现一次性统一重评，也只能标记“回顾性共同样本比较”，不能追认旧 LP test 为未见盲测。
5. 对预训练 Boltz 及官方预训练权重，单列预训练数据来源、可能的 PDBbind/LP 重叠和输入信息优势；不得与从零训练方法合并成无条件的“架构优劣”结论。

## 6. 当前产物盘点与执行顺序

六个从零训练的外部运行目录及 Boltz pairwise 均有 best checkpoint。Boltz pairwise 保存 `valid_pred.npy`/`test_pred.npy`，但不是自描述的 keyed prediction；其他外部运行目录主要是 best checkpoint/汇总指标。不能直接从这些汇总文件构造逐样本共同交集，需要用封存的模型和输入适配器重新导出带 key 的预测，或证明数组顺序与版本化 key 清单完全一致。V20 与较新项目内模型已有 keyed 预测归档，但是否纳入比较及何时允许读取 test，仍取决于选择的测试层级。

统一评估器 `dta/lp_pdbbind_fair_eval_v1/evaluate.py` 强制核对 canonical manifest SHA-256、预测 key 唯一/有限/属于 cohort，并可要求 sidecar 来源校验；按 manifest 顺序分别计算模型原生覆盖与所有候选共同 key 指标，标记 `blind_test=false`。没有显式 `--execute-sealed-test` 时拒绝读取 test manifest。统一指标沿用 V20 的 `affinity_metrics`：预测并列的 CI 半分，Spearman 用 stable-rank；外部训练日志里的 scipy Spearman 不能无条件与其混比。六个适配器都有独立 `--fair-train` 代码路径和带 key 导出器，默认仅导出 validation。真实 test 尚未读取或执行。

**已封存**：官方 train/val 独立源 SHA、单种子 `42`、六个固定预算/超参数、validation CI 精确择优规则、六个上游源码 revision、原生/共同 key 报告方式；见 `experiments/external_benchmarks/lp_pdbbind_fair_v1/protocol.json`。预训练 ESM 输入、TankBind 原生结构/位姿、Boltz 自定义读出与 V20 融合系统必须分层标注，不并入无条件“纯从零且等信息量架构”排名。**尚未完成**：AttentionMGT 原生 320 维特征，以及正式 GPU 重训、验证预测和统一 test 释放。

## 7. 当前执行封存与未越过的门禁（2026-10-03）

独立 `train_val.csv` 物理上只有官方 `new_split` 的 train `11,513` 和 val `2,422` 行，SHA-256 为 `d57809d39cb68aa7ee84798ebe17b1366db44d6aa9dfab37b6e0bf5e9700f842`。六个 `--fair-train` 入口都强制核对该哈希与条数，拒绝旧缓存、旧输出、人为设定的样本上限、恢复旧权重；原生输入不可用时仍须逐 key 记录排除，不能把“禁止人为裁剪”误写成“所有模型都能构图 13,935 行”。项目旧训练入口未被追认。已有 canonical LMDB manifest 只覆盖其中 train `11,037`、valid `2,340` 行；额外 `476/82` 行属于官方名义划分，不应误称不在 LP-PDBBind 数据集。最终 external-to-external 可按官方名义源比较，与 V20 的既有训练池差异必须单独披露。

静态 train/val 输入扫描见 `docs/LP_PDBBind_external_nominal_input_coverage_4.json`：RDKit 可解析 SMILES `11,229/2,404`，非空蛋白序列 `11,513/2,422`，蛋白 PDB 与原生 ligand 文件均存在 `11,225/2,378`；GraphDTA 超过 1,000 残基需截断 `556/23`，含多链分隔符 `2,676/626`。这些仅是字段/文件存在和基本解析数，**不是实际 graph 缓存成功数**；最终每模型过滤数要随新训练缓存报告。

上述 Attention 源 LMDB 静态 key 覆盖审计曾枚举混合 LMDB 的 key 元数据，未解码 test 链 payload、test 标签或读取 test manifest，也未据此选模；因此不能把准备阶段写成“混合 LMDB 的任何 test 页字节都从未被碰过”。正式特征生产和来源摘要改为只按封存 train/val key 定址读取，test 特征只能在显式释放后另行生成。此限制不改变旧 test 已被查看、最终只能做回顾性比较的结论。

`scripts/run_lp_external_fair_v1.py` 默认只做只读预检；物理 GPU 6 启动脚本为 `experiments/external_benchmarks/lp_pdbbind_fair_v1/run_gpu6.sh`。启动器会在任一臂架构/来源门槛未通过时停止整组，不会默默变成五模型比较。当前五臂逐个只读预检可通过，整组六臂仍按预期被 AttentionMGT 真实输入验收阻断；不能手工把 640 维裁剪或写一个 `approved=true` 绕过它。官方权重与全量 CPU 资格账本现已取得：train 合格/排除 `9,323/2,190`，val `2,125/297`，清单 SHA-256 `5f1ee2ef882e036733edd88a95424a5baa435707f527b1610154398fb344d878`。其中 `1,729` 条因口袋残基无法严格映射而排除；有效训练覆盖并不与其他架构相同。下一门是合格样本的原生图/官方 tokenizer 全量 CPU 预检；通过后再按 `experiments/lp_pdbbind/attentionmgt_fair_v1/run_extract_gpu6.sh` 在物理 GPU 6 提取，仅对 train/val 做全链编码，运行合格样本全量特征预检及真实 train 样本前后向，然后自动更新带证据和源码哈希的验收记录，才可开始六臂训练。

`scripts/export_lp_external_fair_v1.py` 默认只规划 validation 带 key 导出；正式训练完成后才能 `--execute`。六个单模型导出器默认只读取 train/val CSV 与 canonical valid manifest，核 checkpoint、训练源/输入缓存和 key 顺序，并从 checkpoint 严格加载模型预测；provenance 需将全部源 key 精确分成预测与有理由排除。test 释放时可用 `scripts/release_lp_external_test_inputs_v1.py` 从官方原表及 canonical test manifest **仅按 key 筛选并输出** `pdb_id,new_split,date,seq,smiles`；它要求 `--execute-sealed-test`，且在打开 manifest 前先核对六臂 checkpoint、运行清单及完整 validation 预测归档。释放时会打开 manifest，但不把标签值用于模型输入或写入输出；当前**没有执行**。后续不论走统一还是独立导出器，读取 test 输入前均需核对六臂仍封存且输入为该释放器生成的 `.source.json` 配对文件。正式六臂统一 test 评估须使用 `--require-sealed-six --released-input-csv`、六个固定别名与独立封存的 canonical test manifest SHA；该模式在读取 test 标签 manifest 前复核六臂运行、validation、共同释放输入和逐模型 test 来源，并将 Attention 预训练 ESM2、TankBind 原生位姿/位点分层标记。通用评估函数仍可用于合成诊断，但不得代替正式六臂报告。不能从准备代码推断任何新 test 分数。当前 `ln02` 环境 `torch.cuda.is_available()` 为 false，也没有可用 `nvidia-smi`；因此真实 GPU6 执行状态仍是“未运行”，不是“已完成重训”。
