# 外部工件迁移清单

本目录默认只包含代码、实验脚本和 CPU 测试，不包含大型数据、LMDB 或模型权重。以下路径均相对于原 GerNA-Bind 仓库根目录。

## 最小推理工件

若要复现已经训练好的 V22，至少迁移：

| 工件 | 原路径 | SHA256 / 说明 |
|---|---|---|
| V22 Stage B 权重 | `Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/structure/best.pt` | `edff376f25af7e789197bdcb7e4664c22446f0fb170c83cd01ad8aa0b1268d08` |
| 蛋白缓存 | `data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb/` | ESM2 residue embedding，1280 维 |
| 配体缓存 | `data/OptionalStructureMultitask_v21/ligands_v2.lmdb/` | SMILES atom/bond graph |
| 数据清单 | `data/OptionalStructureMultitask_v21/manifests_v2/` | train/valid/test JSONL |
| LP-PDBBind 3D 缓存 | `data/LP_PDBBind_MSA_v9/complex_cache_full/{train,valid,test}/complexes.lmdb/` | 仅有 3D 的样本需要；目录名历史上含 `MSA_v9`，V22 不读取 MSA 字段 |

只做 1D+2D 推理时，可不迁移 LP-PDBBind 3D 缓存，并使用 `infer_portable.py --force-no-3d`。清单仍可包含 `has_3d=true`；该参数会强制禁用结构读取。

## 完整训练复现工件

除上述数据外，建议迁移：

| 工件 | 原路径 | SHA256 / 用途 |
|---|---|---|
| V21 warm start | `Model/OptionalStructureMultitask_v21/optional_structure_v21_full_20260905_143425/best.pt` | `edd03ec0105035905c9b4fa4973b21502ce239418b2dd63ed9989b54f23ed882`；Stage A 与 Stage B 的初始化来源 |
| V22 Stage A 权重 | `Model/OptionalStructureStaged_v22/optional_structure_staged_v22_full_20260906_132450/sequence/best.pt` | `a202db6a91f0ffc6ac60d3c9e4ef06707e6000b93c5523058772f0f84656d8e6` |
| V22 Stage A seal | 同目录下 `selection.json` 与 `valid_predictions.npz` | Stage B 会校验训练/validation lineage |
| V22 Stage B seal | `.../structure/selection.json` 与 `valid_predictions.npz` | 原始 `run_test.py` 会校验 validation seal |

原始 `selection.json` 保存的是绝对路径。换机器后，若要继续走严格 seal 流程，应重新运行 Stage A/Stage B 生成本机路径的 seal；若只需要对现有 checkpoint 做评估，使用随包提供的 `infer_portable.py` 并显式传入本机缓存路径。

## 建议复制布局

```text
<TARGET_ROOT>/
├── Model/OptionalStructureStaged_v22/<RUN_ID>/
│   ├── sequence/
│   └── structure/
└── data/
    ├── OptionalStructureMultitask_v21/
    │   ├── manifests_v2/
    │   ├── proteins_esm2_t33_v21.lmdb/
    │   └── ligands_v2.lmdb/
    └── LP_PDBBind_MSA_v9/complex_cache_full/
```

迁移后可用 `sha256sum <checkpoint>` 核对上表权重哈希。

