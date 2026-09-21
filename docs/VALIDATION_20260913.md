# 迁移包验收记录

日期：2026-09-13（Asia/Shanghai）

- Python compile：通过。
- V22 `run_sequence --help`：通过。
- V22 `run_structure --help`：通过。
- V22 `run_test --help`：通过。
- V22 `infer_portable --help`：通过。
- 9 个实验 Bash 脚本 `bash -n`：通过。
- CPU 回归：`20 passed, 2 skipped, 14 warnings`。
  - 两个 skip 仅因代码包不携带外部 `manifests_v2` 与 LMDB；复制 `ARTIFACTS_TO_MIGRATE.md` 所列数据后可运行。
  - warning 为 PyTorch 2.1 Transformer `norm_first=True` 的 nested-tensor 提示，不影响测试结果。
- 正式 V22 checkpoint、真实 test manifest、每数据集 1 条、CPU、强制 1D+2D：通过，共 3 条。
- 同一 checkpoint、CPU、可选 3D，并用 `--structure-root` 重写 1 条 LP-PDBBind 结构路径：通过，共 3 条。
- 两次 smoke 均校验正式 Stage B checkpoint SHA256：`edff376f25af7e789197bdcb7e4664c22446f0fb170c83cd01ad8aa0b1268d08`。

验收过程中未运行任何 GPU 命令。

