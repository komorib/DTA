# Validation record

日期：2026-09-21（Asia/Shanghai）

本记录针对新 DTA 仓库的首次代码导入，不代表重新训练或复现任何模型指标。

## 已执行

- Python：`3.8.20`，现有 `gernabind` 环境。
- 核心 Python 语法编译：通过。
- 仓库内 Shell 脚本 `bash -n`：通过。
- 无外部数据的定向 CPU 测试：`43 passed, 17 warnings`。
- 包含真实数据依赖测试的同一测试集合：`45 passed, 2 failed, 17 warnings`。
  两个失败均为代码包未携带 `manifests_v2`/LMDB 所致的 `FileNotFoundError`，不是模型计算失败。
- 大文件类型检查：未发现权重、LMDB、NumPy 结果、压缩包、PDF 或日志。
- 高置信度密钥模式扫描：无命中。

PyTorch warnings 为 `TransformerEncoderLayer(norm_first=True)` 禁用 nested-tensor fast path 的提示。

## 未执行

- 未使用 GPU。
- 未运行训练、完整推理或高成本工作流。
- 未迁移数据集、缓存或 checkpoint。
- 未验证外部第三方模型的完整环境与运行结果。

