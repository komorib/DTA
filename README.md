# DTA

这是一个代码与配置仓库，用于药物–靶标亲和力（Drug–Target Affinity，DTA）模型的实现、测试和受控实验。

## 当前代码范围

- GerNA 衍生的 DTA 回归、界面建模与融合实现
- sequence-only、可选 3D 结构、多数据集及 MSA 实验分支
- Davis、KIBA、LP-PDBBind 等数据接口和评估指标
- 外部基准模型的适配代码与启动脚本
- CPU 单元测试、实验脚本和数据/权重迁移说明

本仓库不包含数据集、模型权重、LMDB、缓存、实验输出、日志、论文附件或密钥。所需资产见
[ARTIFACTS_TO_MIGRATE.md](ARTIFACTS_TO_MIGRATE.md)。

## 主要目录

```text
dta/                    DTA 模型、训练、推理与实验版本
data_utils/             数据读取、特征和缓存接口
net/                    GerNA 共享网络组件
utils/                  指标与实验完整性工具
experiments/            可复现实验启动脚本（不含运行产物）
scripts/                数据准备、适配和结果处理脚本
tests/                  CPU 测试
gerna-kdbnet/           KDBNet/Pairformer 相关实现
external*/              第三方基准源码与适配所需说明
```

## 环境

建议使用 Linux 与 Python 3.8–3.10。先安装与实际运行设备匹配的 PyTorch，再安装：

```bash
python -m pip install -r requirements.txt
```

`environment-full.yml` 是来源工作区的完整历史环境，仅供依赖定位，不保证可以跨服务器原样复现。
GPU 编号、Python 路径、数据路径和输出路径均应由运行者根据实际环境显式设置；仓库不规定默认 GPU。

## 验证

```bash
python -m compileall -q dta data_utils net utils scripts gerna-kdbnet/kdbnet

python -m pytest -q \
  tests/test_optional_structure_multitask_v21.py \
  tests/test_optional_structure_staged_v22.py \
  tests/test_optional_structure_privileged_msa_v30.py \
  tests/test_optional_structure_privileged_msa_pair_v31.py \
  tests/test_sequence_smiles_binary_probe_v1.py
```

本次代码导入与验证结果见 [VALIDATION.md](VALIDATION.md)，来源说明见 [PROVENANCE.md](PROVENANCE.md)。

