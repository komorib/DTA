# GerNA-Bind


## 模型主体

同一个 checkpoint 支持两种输入模式：

```text
蛋白序列 -> ESM2 residue embedding -> Protein1DEncoder ----+
                                                        |-> 双向 cross-attention -> base affinity
SMILES -> atom/bond graph -> Ligand2DEncoder ------------+

可选 pocket/ligand 3D -> SE(3)-invariant distance encoder
                      -> quality-aware gate * residual delta

最终 affinity = base affinity + gate * structure delta
```

- 1D：ESM2 `facebook/esm2_t33_650M_UR50D` 的逐残基 1280 维表示，经 adapter、位置编码和 Transformer 编码。
- 2D：由 SMILES 构造原子类别/连续特征及 bond pair bias，不读取 conformer 坐标。
- 跨模态：蛋白与配体进行多层双向 cross-attention，再 attention pooling。
- 3D：口袋残基与配体原子通过距离 RBF、pair token、contact/distogram 辅助目标编码；坐标只通过距离进入，因此结构分支保持 SE(3) 不变性。
- 缺少 3D：`has_3d=false` 时结构分支不执行，`gate=0`，严格回退到 1D+2D base。
- 多数据集：LP-PDBBind、Davis、KIBA 共用主体，并使用 dataset adapter/head 处理标签尺度差异。


## 目录

```text
dta/optional_structure_staged_v22/       V22 模型、两阶段训练、严格 test
dta/optional_structure_multitask_v21/    manifest/LMDB/ESM 预处理及 warm-start 兼容
dta/rank_fusion_v4/metrics.py            affinity 指标依赖
dta/sequence_smiles_binary_probe_v1/     Davis/KIBA 读取与二分类指标依赖
experiments/multidataset/                可复现实验脚本
tests/                                   V21/V22 CPU 测试
ARTIFACTS_TO_MIGRATE.md                  权重和数据缓存清单
```

## 环境

建议使用 Linux、Python 3.8–3.10，并先安装与目标 CUDA 匹配的 PyTorch。随后在本目录执行：

```bash
python -m pip install -r requirements.txt
```

若 `rdkit` 的 pip wheel 不适配目标系统，可使用 conda-forge 安装 RDKit。抽取 ESM2 特征需要 `transformers`；也支持可选 `fair-esm` backend。

## 数据准备

默认脚本期望 GerNA-Bind 原有相对布局。已有缓存时，按 [ARTIFACTS_TO_MIGRATE.md](ARTIFACTS_TO_MIGRATE.md) 复制即可。重新构建时：

```bash
PYTHON_BIN=python \
bash experiments/multidataset/optional_structure_v21/prepare_cpu.sh

GPU_ID=7 PYTHON_BIN=python \
bash experiments/multidataset/optional_structure_v21/extract_esm_gpu7.sh
```

可以用环境变量覆盖默认位置：

- `MANIFEST_ROOT`
- `PROTEIN_CACHE`
- `LIGAND_CACHE`
- `OUTPUT_ROOT` / `OUTPUT`
- `PYTHON_BIN`
- `GPU_ID`

`build_manifests.py` 默认还需要仓库格式的 `data/LP_PDBBind.csv`、LP-PDBBind split manifest、Davis 与 KIBA 解包目录。若迁移现成 `manifests_v2`，无需重新构建。

## 两阶段训练

正式 V22 先训练并密封 1D+2D base，再只用 validation 选择 3D residual。脚本默认复用 V21 权重以复现原实验；迁移后应显式指定本机路径。

Stage A：

```bash
GPU_ID=7 \
RUN_ID=optional_structure_staged_v22_full_$(date +%Y%m%d_%H%M%S) \
INITIAL_CHECKPOINT=/path/to/v21/best.pt \
bash experiments/multidataset/optional_structure_v22/run_sequence_gpu7.sh
```

Stage B（使用与 Stage A 相同的 `RUN_ID`）：

```bash
GPU_ID=7 \
RUN_ID=<STAGE_A_RUN_ID> \
WARM_STRUCTURE_CHECKPOINT=/path/to/v21/best.pt \
bash experiments/multidataset/optional_structure_v22/run_structure_gpu7.sh
```

只有 Stage B 的 validation gate 通过后，才运行正式 test：

```bash
GPU_ID=7 RUN_ID=<VALIDATION_PASSED_RUN_ID> \
bash experiments/multidataset/optional_structure_v22/run_test_gpu7.sh
```

这些是交给有 GPU 权限的用户运行的命令。本代码包的整理和验收不启动 GPU。

## 迁移现有 checkpoint 后推理

原始 `run_test.py` 为正式实验保留严格 validation seal，并校验原路径。换机器后对现有 `best.pt` 做普通评估，可用便携入口：

```bash
python -m dta.optional_structure_staged_v22.infer_portable \
  --checkpoint /path/to/structure/best.pt \
  --manifest /path/to/test.jsonl \
  --protein-cache /path/to/proteins_esm2_t33_v21.lmdb \
  --ligand-cache /path/to/ligands_v2.lmdb \
  --structure-root /path/to/complex_cache_full \
  --output /path/to/inference_output \
  --device cuda:0
```

只测试 sequence+2D 路径：追加 `--force-no-3d`。便携入口不会参与正式模型选择；正式论文结果仍以 validation seal + `run_test.py` 为准。

## 已有正式结果

以下来自项目原 split，test 未用于选择：

| 数据集 | CI | RMSE | MAE | Pearson | Spearman | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| LP-PDBBind | 0.663464 | 1.630012 | 1.278612 | 0.495324 | 0.471381 | — | — |
| Davis | 0.832707 | 0.667753 | 0.345257 | 0.692159 | 0.479864 | 0.906882 | 0.495135 |
| KIBA | 0.785914 | 0.594837 | 0.364332 | 0.712170 | 0.709449 | 0.825771 | 0.665786 |

跨论文比较前必须对齐 split、标签变换、二分类阈值和泄漏过滤规则。

## CPU 验收

```bash
python -m compileall -q dta
bash -n experiments/multidataset/optional_structure_v21/*.sh
bash -n experiments/multidataset/optional_structure_v22/*.sh
pytest -q \
  tests/test_optional_structure_multitask_v21.py \
  tests/test_optional_structure_multitask_v21_data.py \
  tests/test_optional_structure_staged_v22.py
```

`MANIFEST.sha256` 记录本代码包文件哈希。迁移后可用 `sha256sum -c MANIFEST.sha256` 验证。

