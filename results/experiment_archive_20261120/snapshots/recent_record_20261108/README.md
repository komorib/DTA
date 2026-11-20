# 近期实验归档（2026-11-08）

本目录补齐总实验记录中分散或过时的近期结果。只整理既有 summary、history、selection、失败记录和聚合指标；本轮不训练、不推理，不读取逐条 test 标签或预测，不改原模型、数据和运行目录。

`sources/` 为原始小文件的逐字节副本，`manifest.json` 记录原路径、大小、SHA-256 和采集时间。未完成训练的 history 是采集时快照，不是最终选模结果。权重、特征、原始结构和逐样本预测不上传。

阅读入口：[实验总记录](../../reports/LP_PDBBind_experiment_record.md)、[近期结果总览](../../reports/LP_PDBBind_recent_experiments_20261108.md)。

采集快照：2026-11-08 10:43 北京时间。39份文件共526,832 bytes，采集时均与原件逐字节一致；归档内全部SHA与报告表格核对通过，结果见 `verification.json`。未完成的 no_1d 在此快照已有4轮完整history，仍无最终summary。后续训练进度不回写这个静态快照。

从本目录运行 `python verify.py` 可复核快照和归档的近期结果表；仅需标准库，不运行模型或读取原数据。
