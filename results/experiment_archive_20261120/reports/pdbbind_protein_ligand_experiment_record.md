# GerNA-Bind 迁移到 Protein-Ligand 的实验记录

记录时间：2026-05-19  
项目目录：`/home/user/wangzhiwen/GerNA-Bind`

## 目标

把原 GerNA-Bind 的 RNA-ligand 框架迁移到 protein-ligand 场景。当前阶段的目标不是严格复现原 RNA-ligand 论文结果，而是在尽量保留模型结构的前提下，用 PDBbind protein-ligand 数据跑通训练，并逐步补齐 protein 表征、pairwise/contact 监督和更严格的数据划分。

## 数据来源

已下载并放入：

```text
PDBbind/index.tar.gz
PDBbind/P-L.tar.gz
PDBbind/P-P (1).tar.gz
PDBbind/P-NA.tar.gz
PDBbind/NA-L.tar.gz
```

当前 protein-ligand 任务实际使用：

```text
PDBbind/index.tar.gz
PDBbind/P-L.tar.gz
```

其中：

```text
index/INDEX_general_PL.2020R1.lst
P-L/**/{pdbid}_pocket.pdb
P-L/**/{pdbid}_ligand.sdf
P-L/**/{pdbid}_ligand.mol2
```

## 数据转换

新增脚本：

```text
data_utils/process_pdbbind.py
```

转换逻辑：

```text
protein pocket.pdb  -> residue feature / residue graph / CA coordinates
ligand sdf/mol2     -> molecular graph / atom type / atom coordinates / LAS mask
index binding value -> binary label
```

二分类标签规则：

```text
Ki/Kd/IC50 <= 1 uM   -> positive, label = 1
Ki/Kd/IC50 >= 10 uM  -> negative, label = 0
中间区间             -> 跳过
```

最终转换结果：

```json
{
  "candidate_records": 15830,
  "kept": 15827,
  "missing_structure": 0,
  "parse_errors": 3,
  "positive": 11340,
  "negative": 4487
}
```

输出文件：

```text
data/pdbbind_protein_ligand.pkl
data/pdbbind_protein_ligand_metadata.json
data/pdbbind_protein_ligand.pkl.checkpoint/
```

完整性检查结果：

```text
top_level_fields = 10
每个字段长度 = 15827
label 统计 = positive 11340, negative 4487
```

首个样本维度：

```text
protein_repre    [83, 640]
protein_graph.x  [83, 640]
protein_feats    [83]
protein_ca       [83, 3]
mol_graph.x      [33, 55]
mol_feats        [33]
mol_coors        [33, 3]
LAS              [33, 33]
label            1
```

## 数据切分

由于总 pkl 为 6.8G，直接切成 `train_data.pkl / valid_data.pkl / test_data.pkl` 会导致内存峰值过高。因此改为基于 checkpoint chunk 生成轻量 manifest。

新增脚本：

```text
data_utils/split_pdbbind_dataset.py
```

生成文件：

```text
data/PDBbind/random/train_manifest.json
data/PDBbind/random/valid_manifest.json
data/PDBbind/random/test_manifest.json
data/PDBbind/random/split_summary.json
```

切分比例：

```text
train: 0.8
valid: 0.1
test:  0.1
seed:  99
```

切分结果：

```text
train: 12662, positive 9072, negative 3590
valid: 1583,  positive 1134, negative 449
test:  1582,  positive 1134, negative 448
```

## 当前代码改动

### `data_utils/dataset.py`

新增：

```text
GerNAChunkedDataset
```

作用：

```text
根据 train_manifest.json / valid_manifest.json / test_manifest.json
按需懒加载 data/pdbbind_protein_ligand.pkl.checkpoint/chunk_*.pkl
```

这样训练时不需要一次性把 6.8G 总 pkl 全部读进内存。

### `net/model.py`

新增参数：

```python
rna_num_tokens=None
```

用途：

```text
RNA 原版默认仍使用原来的 token 数；
PDBbind protein 版使用 rna_num_tokens=21，对应 20 种氨基酸 + padding/unknown。
```

### `train_model.py`

已支持：

```text
本地 data_root
manifest 懒加载数据
PDBbind 自动 rna_num_tokens=21
训练集 minority class oversampling
可选 pairwise/contact loss
```

新增参数：

```text
--data_root
--train_path
--valid_path
--test_path
--balance_strategy
--rna_num_tokens
--pairwise_loss_weight
--contact_cutoff
--contact_positive_weight
```

### `tests/test_pdbbind_setup.py`

用于检查：

```text
split index 是否覆盖所有样本且不重叠
GerNA 是否支持 rna_num_tokens=21
balanced index 是否能重复 minority class
GerNAChunkedDataset 是否能按 manifest 读样本
contact target 是否能从 CA-ligand 距离生成
masked pairwise BCE 是否尊重 mask
```

## 已通过的验证

```text
python tests/test_pdbbind_setup.py -> ok
python3 -m py_compile data_utils/dataset.py data_utils/split_pdbbind_dataset.py train_model.py net/model.py -> ok
```

PDBbind manifest batch 验证：

```text
len 12662
collate_items 13
RNA_repre (2, 70, 640)
Mol_graph_nodes torch.Size([60, 55])
RNA_graph_nodes torch.Size([113, 640])
RNA_feats (2, 70)
RNA_C4 (2, 70, 3)
Mol_feats (2, 33)
Mol_coors (2, 33, 3)
labels [1.0, 1.0]
```

模型前向 smoke test：

```text
affinity_pred (1, 2)
pairwise (1, 70, 27)
ok
```

真实 batch contact target 验证：

```text
contact_targets (2, 70, 33)
valid_pairs 3309
positives 411
loss 0.7792407870292664
```

模型实际 pairwise + contact loss 验证：

```text
affinity_pred (1, 2)
pairwise (1, 70, 27)
contact_loss 0.8649351000785828
ok
```

## Baseline 训练

运行设备：

```text
物理 GPU 7
A100 40G
```

运行命令：

```bash
python train_model.py \
  --dataset PDBbind \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_random/ \
  --cuda 7 \
  --batch_size 4 \
  --epoch 12
```

说明：

```text
CUDA_VISIBLE_DEVICES=7
程序内部使用 cuda:0
实际使用物理 7 号卡
```

当前显存观察：

```text
约 10476 MiB / 40G
```

结论：

```text
显存没有占满不代表结果差。
提高显存占用本身不会提高结果，只有通过更大 batch、更大模型或更强特征才可能间接影响结果。
当前建议先跑完 batch_size=4 baseline，再做 batch_size=8 对照实验。
```

## Baseline 已观察结果

### Epoch 0

```text
threshold: 0.4975803060939241

train 12662
TN 3050 FN 2636 FP 540 TP 6436
Pre 0.922592 Sen 0.709436 Spe 0.849582 Acc 0.749171
F1 0.802094 Mcc 0.506545 AUC 0.846416 AUPRC 0.929511

valid 1583
TN 369 FN 334 FP 80 TP 800
Pre 0.909091 Sen 0.705467 Spe 0.821826 Acc 0.738471
F1 0.794439 Mcc 0.478369 AUC 0.840386 AUPRC 0.927656

test 1582
TN 383 FN 309 FP 65 TP 825
Pre 0.926966 Sen 0.727513 Spe 0.854911 Acc 0.763590
F1 0.815217 Mcc 0.528977 AUC 0.857968 AUPRC 0.930635
```

### Epoch 1

用户提供了 valid/test：

```text
valid 1583
TN 347 FN 247 FP 102 TP 887
Pre 0.896866 Sen 0.782187 Spe 0.772829 Acc 0.779533
F1 0.835610 Mcc 0.516706 AUC 0.855554 AUPRC 0.930907

test 1582
TN 355 FN 244 FP 93 TP 890
Pre 0.905392 Sen 0.784832 Spe 0.792411 Acc 0.786979
F1 0.840812 Mcc 0.536184 AUC 0.868209 AUPRC 0.935764
```

### Epoch 2

```text
threshold: 0.40056162442766946

train 12662
TN 2985 FN 2098 FP 605 TP 6974
Pre 0.920174 Sen 0.768739 Spe 0.831476 Acc 0.786527
F1 0.837667 Mcc 0.551874 AUC 0.871307 AUPRC 0.942175

valid 1583
TN 359 FN 267 FP 90 TP 867
Pre 0.905956 Sen 0.764550 Spe 0.799555 Acc 0.774479
F1 0.829268 Mcc 0.520052 AUC 0.860154 AUPRC 0.936754

test 1582
TN 371 FN 247 FP 77 TP 887
Pre 0.920124 Sen 0.782187 Spe 0.828125 Acc 0.795196
F1 0.845567 Mcc 0.563592 AUC 0.875638 AUPRC 0.938338
```

### Epoch 3

```text
threshold: 0.38384658792846377

train 12662
TN 3114 FN 2331 FP 476 TP 6741
Pre 0.934045 Sen 0.743056 Spe 0.867409 Acc 0.778313
F1 0.827675 Mcc 0.555754 AUC 0.873962 AUPRC 0.942874

valid 1583
TN 375 FN 303 FP 74 TP 831
Pre 0.918232 Sen 0.732804 Spe 0.835189 Acc 0.761845
F1 0.815105 Mcc 0.517409 AUC 0.861069 AUPRC 0.936919

test 1582
TN 382 FN 265 FP 66 TP 869
Pre 0.929412 Sen 0.766314 Spe 0.852679 Acc 0.790771
F1 0.840019 Mcc 0.567248 AUC 0.879937 AUPRC 0.937014
```

截至目前的趋势：

```text
test AUC:
epoch 0: 0.857968
epoch 1: 0.868209
epoch 2: 0.875638
epoch 3: 0.879937

valid AUC:
epoch 0: 0.840386
epoch 1: 0.855554
epoch 2: 0.860154
epoch 3: 0.861069
```

判断：

```text
baseline 正常运行；
模型仍在学习；
截至 epoch 3 未见明显过拟合；
当前训练脚本按 valid AUC 保存最优模型。
```

模型保存位置：

```text
Model/PDBbind_random/Model_baseline.pth
```

## 是否算完整迁移复现

当前不算完整迁移复现，只能算：

```text
GerNA-Bind protein-ligand adaptation v1 baseline
```

已经完成：

```text
RNA token       -> protein residue
RNA C4' 坐标    -> protein CA 坐标
RNA 图          -> protein residue graph
ligand 图/坐标  -> 保留
GerNA-Bind 模型 -> 基本保留
二分类训练       -> 跑通
```

仍未完成：

```text
RNA-FM 对应的 protein 预训练表示：ESM/ProtT5
更完整的 protein 3D 原子表示：N, CA, C, O, CB 等
严格的 residue-ligand 全原子 contact label
protein cluster split / ligand scaffold split / both cold split
系统消融实验
affinity regression 指标
```

## 下一步实验建议

### 1. 先等当前 baseline 跑完

当前实验：

```text
batch_size=4
pairwise_loss_weight=0
```

先记录完整 12 epoch 的最优 valid/test 结果，作为 baseline。

### 2. batch size 对照

```bash
python train_model.py \
  --dataset PDBbind \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_random_bs8/ \
  --cuda 7 \
  --batch_size 8 \
  --epoch 12
```

### 3. contact/pairwise loss 实验

新增代码已经支持，但默认关闭。新实验命令：

```bash
python train_model.py \
  --dataset PDBbind \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_random_pairwise01/ \
  --cuda 7 \
  --batch_size 4 \
  --epoch 12 \
  --pairwise_loss_weight 0.1 \
  --contact_cutoff 8.0 \
  --contact_positive_weight 2.0
```

说明：

```text
当前 contact label 是 CA-ligand atom 距离 <= 8.0 A 的粗粒度标签。
这只是第一版 pairwise supervision。
后续更严谨版本应改成 residue 任意重原子到 ligand 重原子 <= 4.5 A。
```

## 当前文件状态摘要

核心新增/修改：

```text
data_utils/process_pdbbind.py
data_utils/split_pdbbind_dataset.py
data_utils/dataset.py
net/model.py
train_model.py
tests/test_pdbbind_setup.py
docs/pdbbind_protein_ligand_experiment_record.md
```

重要数据文件：

```text
data/pdbbind_protein_ligand.pkl
data/pdbbind_protein_ligand_metadata.json
data/pdbbind_protein_ligand.pkl.checkpoint/
data/PDBbind/random/train_manifest.json
data/PDBbind/random/valid_manifest.json
data/PDBbind/random/test_manifest.json
data/PDBbind/random/split_summary.json
```


## 2026-05-20 ESM2 预提取版准备

目标：在 RNA->protein 改造的基础上，把原先的手工 640 维占位蛋白特征替换为本地 ESM2 表征，先跑通完整版本，后续再做消融。

### 本次新增/修改代码

```text
data_utils/extract_pdbbind_esm.py
train_model.py
tests/test_pdbbind_setup.py
```

关键逻辑：

```text
protein_feats residue id 1..20 -> amino-acid sequence -> ESM2 -> [L, 640]
```

写回位置：

```text
data[0] protein_repre
data[2][i].x protein_graph.x
```

这样 1D protein 分支和 protein graph 分支都使用同一份 ESM2 per-residue embedding，原来的 10 字段 chunk 格式保持不变，训练入口基本不用改。

### 本地 ESM2 权重

```text
/home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D
```

已确认：

```text
hidden_size = 640
transformers = 4.38.2
safetensors 可导入
```

这个 hidden size 与当前 `net/model.py` 的 `input_dim_rna=640` 一致，所以不需要改模型输入维度。

### 已完成验证

新增测试：

```text
residue id -> ESM sequence
ESM embedding 同步替换 protein_repre 和 protein_graph.x
PDBbind_esm 自动使用 rna_num_tokens=21
```

验证命令：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python tests/test_pdbbind_setup.py
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m py_compile data_utils/extract_pdbbind_esm.py train_model.py tests/test_pdbbind_setup.py
```

结果：

```text
tests/test_pdbbind_setup.py -> ok
py_compile -> exit 0
```

另外用真实 ESM2 权重做了 2 个样本 smoke check，输出到 `/tmp/gernabind_esm_smoke.checkpoint`：

```text
n_samples 2
protein_repre_0 (83, 640) float32
protein_graph_x_0 (83, 640) torch.float32
protein_feats_0 (83,)
label_list [1, 0]
collated_RNA_repre (2, 83, 640)
collated_seq_mask (2, 83) sum=142
collated_RNA_graph_x (142, 640)
collated_Mol_coors (2, 33, 3)
```

说明：ESM2 特征维度和当前 collate/model 输入契约是对齐的。

### 完整预提取命令

建议等当前 baseline 训练结束后再跑，避免和训练抢 7 号卡显存。

```bash
CUDA_VISIBLE_DEVICES=7 /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/extract_pdbbind_esm.py \
  --input_checkpoint_dir data/pdbbind_protein_ligand.pkl.checkpoint \
  --output_checkpoint_dir data/pdbbind_protein_ligand_esm.pkl.checkpoint \
  --esm_model_path /home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D \
  --device cuda:0 \
  --batch_size 16 \
  --resume
```

断点续跑：重复同一命令即可，已写出的 `chunk_*.pkl` 会跳过。

### 生成 ESM manifest

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/split_pdbbind_dataset.py \
  --checkpoint_dir data/pdbbind_protein_ligand_esm.pkl.checkpoint \
  --output_dir data/PDBbind_esm/random \
  --output_format manifest
```

### ESM 完整版训练命令

先不加 pairwise contact supervision，作为 ESM 完整版 baseline：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python train_model.py \
  --dataset PDBbind_esm \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_esm_random/ \
  --cuda 7 \
  --batch_size 4 \
  --epoch 12 \
  --port 10087
```

`train_model.py` 已经支持 `PDBbind_esm` 自动使用 `rna_num_tokens=21`，不需要额外传这个参数。


## 2026-05-20 PDBbind Atom5 细粒度 3D 输入

目标：PDBbind 的 `*_pocket.pdb` 原本包含蛋白口袋原子坐标。之前第一版预处理只用了每个 residue 的 CA 坐标，导致 protein 3D 是粗粒度。现在把 N/CA/C/O/CB 这些原子点接入 3D 分支。

### 当前数据字段含义

仍保持 GerNA-Bind 原来的 10 字段格式，但 protein 版字段语义更新如下：

```text
data[0] protein_repre      residue-level [L, 640]
data[2] protein_graph.x    residue-level [L, 640]
data[3] protein_3d_tokens  atom5-level [P]，N/CA/C/O/CB -> 1..5
data[4] protein_ca_coors   residue-level [L, 3]，给 2D pair 距离/CA 几何用
data[5] protein_3d_coors   atom5-level [P, 3]，给 3D Equiformer 用
```

其中：

```text
L = residue 数
P = 真实存在的 N/CA/C/O/CB 原子点数，通常约 4-5 倍 L
```

这样做以后：

```text
1D 分支：仍然 residue-level，可以继续接 ESM2 [L,640]
2D graph 分支：仍然 residue-level protein graph
2D triangle pair：仍然用 CA-CA residue 距离，维度 [L,L]
3D Equiformer 分支：改用 atom5-level protein 原子点，维度 [P,3]
```

### 已完成代码改动

```text
data_utils/process_pdbbind.py
  - 新增 PROTEIN_ATOM5_TOKEN
  - read_pocket_from_pdb_text 同时返回 residue-level CA 和 atom5-level 坐标
  - metadata 记录 residue_sequence/residue_ids/protein_3d_mode/protein_3d_num_tokens

data_utils/extract_pdbbind_esm.py
  - atom5 chunk 不再从 data[3] 推断序列
  - 优先使用 metadata.residue_sequence / metadata.residue_ids
  - ESM embedding 长度按 data[0] 的 residue 数校验

train_model.py
  - PDBbind_atom5 / PDBbind_atom5_esm 自动使用 rna_num_tokens=6
```

### Smoke check

2 个真实 PDBbind 样本转换到 `/tmp/pdbbind_atom5_smoke.pkl.checkpoint`，结果：

```text
sample 0:
  residue_repre (83, 640)
  graph_x (83, 640)
  atom5_tokens (409,) max=5
  ca_coords (83, 3)
  atom5_coords (409, 3)

sample 1:
  residue_repre (59, 640)
  graph_x (59, 640)
  atom5_tokens (292,) max=5
  ca_coords (59, 3)
  atom5_coords (292, 3)

collated_repre (2, 83, 640)
collated_atom5_feats (2, 409) max=5
collated_ca (2, 83, 3)
collated_atom5_coords (2, 409, 3)
collated_atom5_mask (2, 409) sum=701
```

另用 atom5 小样本跑过 ESM2 预提取，确认 ESM 读取的是 metadata 中的 residue 序列，而不是 atom5 token：

```text
repre (83, 640) float32
graph_x (83, 640) torch.float32
atom_tokens (409,) max=5
atom_coords (409, 3)
sequence_len 83 metadata_atom_tokens 409
```

验证命令：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python tests/test_pdbbind_setup.py
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m py_compile data_utils/process_pdbbind.py data_utils/extract_pdbbind_esm.py train_model.py tests/test_pdbbind_setup.py
```

结果：

```text
tests/test_pdbbind_setup.py -> ok
py_compile -> exit 0
```

### 完整 Atom5 数据转换命令

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/process_pdbbind.py \
  --pdbbind_dir PDBbind \
  --output data/pdbbind_protein_ligand_atom5.pkl \
  --metadata_output data/pdbbind_protein_ligand_atom5_metadata.json \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5.pkl.checkpoint \
  --checkpoint_every 500 \
  --progress_every 500 \
  --resume
```

### Atom5 split

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/split_pdbbind_dataset.py \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5.pkl.checkpoint \
  --output_dir data/PDBbind_atom5/random \
  --output_format manifest
```

### Atom5 baseline 训练

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python train_model.py \
  --dataset PDBbind_atom5 \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_atom5_random/ \
  --cuda 7 \
  --batch_size 2 \
  --epoch 12 \
  --port 10088
```

说明：Atom5 的 3D token 数是 6，`train_model.py` 会自动推断。因为 3D 点数约变成 4-5 倍，建议先用 `batch_size=2`，确认显存后再加到 4。

### Atom5 + ESM2 预提取

```bash
CUDA_VISIBLE_DEVICES=7 /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/extract_pdbbind_esm.py \
  --input_checkpoint_dir data/pdbbind_protein_ligand_atom5.pkl.checkpoint \
  --output_checkpoint_dir data/pdbbind_protein_ligand_atom5_esm.pkl.checkpoint \
  --esm_model_path /home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D \
  --device cuda:0 \
  --batch_size 16 \
  --resume
```

### Atom5 + ESM2 split

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/split_pdbbind_dataset.py \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5_esm.pkl.checkpoint \
  --output_dir data/PDBbind_atom5_esm/random \
  --output_format manifest
```

### Atom5 + ESM2 完整版训练

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python train_model.py \
  --dataset PDBbind_atom5_esm \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_atom5_esm_random/ \
  --cuda 7 \
  --batch_size 2 \
  --epoch 12 \
  --port 10089
```

### 当前限制

当前 `pairwise_loss_weight` 监督的仍然是返回的 2D residue-ligand pairwise matrix，因此如果打开 contact supervision，默认仍是 CA-ligand 粗标签。Atom5 细粒度坐标已经进入 3D Equiformer 和 3D triangle/DMA 分支，但 residue heavy-atom contact supervision 还需要下一步单独加 residue-atom 映射或从 atom5 坐标聚合成 residue-level contact label。


## 2026-05-20 早停 Early Stopping

训练入口已加入可选早停，默认关闭，不影响之前命令的行为。

新增参数：

```text
--early_stop_patience 0      默认 0，表示关闭早停
--early_stop_min_delta 0.0   valid AUC 至少提升多少才算 improvement
```

判断指标：

```text
valid AUC
```

逻辑：

```text
valid AUC 提升 -> 保存 best checkpoint，bad_epochs 清零
valid AUC 没提升 -> bad_epochs + 1
bad_epochs >= patience -> rank 0 广播 early stop signal，所有 DDP rank 一起停止
```

代码位置：

```text
train_model.py:update_early_stop_state
train_model.py:train_and_eval epoch 评估后广播 early_stop_signal
```

验证命令：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python tests/test_pdbbind_setup.py
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m py_compile train_model.py tests/test_pdbbind_setup.py
```

结果：

```text
tests/test_pdbbind_setup.py -> ok
py_compile -> exit 0
```

推荐训练时先用 patience=3：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python train_model.py \
  --dataset PDBbind_atom5_esm \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_atom5_esm_random/ \
  --cuda 7 \
  --batch_size 2 \
  --epoch 30 \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.0005 \
  --port 10089
```

说明：加早停后可以把 `--epoch` 设大一点，例如 30；实际会在 valid AUC 连续 3 个 epoch 没有超过 `best + min_delta` 时停止。最终保存的仍然是 valid AUC 最好的 `Model_baseline.pth`。


## 2026-05-20 PDBbind_atom5_esm 四种划分补齐

目标：把论文里的 split 思路迁移到 PDBbind protein-ligand 版本。原论文是 `random / RNA / mol / both`，这里对应为：

```text
random  : 普通按标签分层随机划分，只保证正负样本比例接近。
protein : protein cold split，同一 protein_group 不跨 train/valid/test。
mol     : ligand cold split，同一 ligand scaffold 不跨 train/valid/test。
both    : protein_group 和 ligand scaffold 构成二部图，按连通分量划分，保证两边都不泄漏。
```

代码改动：

```text
data_utils/split_pdbbind_dataset.py
- 新增 --split_method random|protein|mol|both
- protein 使用 residue_sequence/protein_group 分组
- mol 使用 ligand_scaffold_smiles/ligand_group 分组
- both 使用 protein_group 与 ligand_group 的二部图连通分量分组
- manifest summary 会输出 group overlap 检查

train_model.py
- --split_method choices 增加 protein

tests/test_pdbbind_setup.py
- 增加 group split、both component 和 train_model CLI protein choice 测试
```

已生成目录：

```text
data/PDBbind_atom5_esm/random
data/PDBbind_atom5_esm/protein
data/PDBbind_atom5_esm/mol
data/PDBbind_atom5_esm/both
```

每个目录都有：

```text
train_manifest.json
valid_manifest.json
test_manifest.json
split_summary.json
```

### random split 结果

```text
total 15827
train 12662: positive 9072, negative 3590
valid 1583 : positive 1134, negative 449
test  1582 : positive 1134, negative 448
```

### protein cold split 结果

```text
total 15827
train 12661: positive 9072, negative 3589, group_count 12661
valid 1583 : positive 1134, negative 449,  group_count 1360
test  1583 : positive 1134, negative 449,  group_count 1359
protein_group_overlap: train_vs_valid 0, train_vs_test 0, valid_vs_test 0
```

### mol cold split 结果

```text
total 15827
train 12566: positive 9215, negative 3351, group_count 8330
valid 1630 : positive 1063, negative 567,  group_count 107
test  1631 : positive 1062, negative 569,  group_count 102
ligand_group_overlap: train_vs_valid 0, train_vs_test 0, valid_vs_test 0
```

### both cold split 结果

```text
total 15827
train 12567: positive 9214, negative 3353, group_count 8114
valid 1628 : positive 1066, negative 562,  group_count 94
test  1632 : positive 1060, negative 572,  group_count 96
protein_group_overlap: train_vs_valid 0, train_vs_test 0, valid_vs_test 0
ligand_group_overlap : train_vs_valid 0, train_vs_test 0, valid_vs_test 0
```

说明：cold split 是按组划分，不能像 random 一样严格保持 8:1:1 和正负比例；这里优先保证 group 不泄漏。

### 生成命令

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/split_pdbbind_dataset.py \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5_esm.pkl.checkpoint \
  --output_dir data/PDBbind_atom5_esm/random \
  --output_format manifest \
  --split_method random \
  --seed 99

/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/split_pdbbind_dataset.py \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5_esm.pkl.checkpoint \
  --output_dir data/PDBbind_atom5_esm/protein \
  --output_format manifest \
  --split_method protein \
  --seed 99

/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/split_pdbbind_dataset.py \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5_esm.pkl.checkpoint \
  --output_dir data/PDBbind_atom5_esm/mol \
  --output_format manifest \
  --split_method mol \
  --pdbbind_dir PDBbind \
  --seed 99

/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/split_pdbbind_dataset.py \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5_esm.pkl.checkpoint \
  --output_dir data/PDBbind_atom5_esm/both \
  --output_format manifest \
  --split_method both \
  --pdbbind_dir PDBbind \
  --seed 99
```

### 训练命令模板

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python train_model.py \
  --dataset PDBbind_atom5_esm \
  --split_method protein \
  --data_root data \
  --model_output_path Model/PDBbind_atom5_esm_protein/ \
  --cuda 7 \
  --batch_size 2 \
  --epoch 30 \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.0005 \
  --port 10090

/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python train_model.py \
  --dataset PDBbind_atom5_esm \
  --split_method mol \
  --data_root data \
  --model_output_path Model/PDBbind_atom5_esm_mol/ \
  --cuda 7 \
  --batch_size 2 \
  --epoch 30 \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.0005 \
  --port 10091

/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python train_model.py \
  --dataset PDBbind_atom5_esm \
  --split_method both \
  --data_root data \
  --model_output_path Model/PDBbind_atom5_esm_both/ \
  --cuda 7 \
  --batch_size 2 \
  --epoch 30 \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.0005 \
  --port 10092
```

验证：

```text
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python tests/test_pdbbind_setup.py -> ok
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m py_compile data_utils/split_pdbbind_dataset.py data_utils/process_pdbbind.py data_utils/extract_pdbbind_esm.py data_utils/dataset.py train_model.py tests/test_pdbbind_setup.py net/model.py -> exit 0
```


## 2026-05-20 追加连续分数评价指标

按要求追加以下指标：

```text
RMSE
MAE
Pearson
Spearman
CI
```

注意：这里没有改变论文架构，也没有把任务改成 affinity regression。模型仍然输出两类 Dirichlet evidence，loss 仍然是 `evidential_classification`。新增指标只是把正类概率 `p(binding)` 当作连续分数，与 0/1 标签计算附加评价。

代码位置：

```text
utils/metrics.py:get_score_metrics
utils/metrics.py:concordance_index
train_model.py:perf_name 增加 RMSE/MAE/Pearson/Spearman/CI
test_model.py:perf_name 增加 RMSE/MAE/Pearson/Spearman/CI
```

早停逻辑仍然使用 valid AUC，不受新增指标位置影响。

验证：

```text
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python tests/test_pdbbind_setup.py -> ok
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m py_compile utils/metrics.py train_model.py test_model.py tests/test_pdbbind_setup.py -> exit 0
```


## 2026-05-20 新增 DTA / affinity regression 独立代码目录

新增目录：

```text
dta/
```

目的：单独做 protein-ligand DTA，不覆盖原论文二分类复现代码。

任务定义：

```text
输入：protein pocket + ligand
标签：pAffinity = -log10(Kd/Ki/IC50 in M)
输出：连续 affinity 分数
loss：MSE 或 SmoothL1
指标：RMSE / MAE / Pearson / Spearman / CI
```

与论文复现路线的区别：

```text
原 train_model.py：binding specificity 二分类，evidential_classification，输出两类 Dirichlet evidence
DTA train_dta.py：affinity regression，SmoothL1/MSE，输出一个连续 pAffinity
```

新增文件：

```text
dta/process_pdbbind_dta.py   从 PDBbind exact Kd/Ki/IC50 生成 pAffinity 数据
dta/split_pdbbind_dta.py     生成 random/protein/mol/both manifest
dta/model_dta.py             复用 GerNA 主体，去掉回归头 sigmoid，输出原始连续值
dta/train_dta.py             单 GPU DTA 训练入口
dta/tests/test_dta_setup.py  DTA 基础测试
dta/README.md                使用说明和命令
```

数据转换命令：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/process_pdbbind_dta.py \
  --pdbbind_dir PDBbind \
  --output data/pdbbind_protein_ligand_atom5_dta.pkl \
  --metadata_output data/pdbbind_protein_ligand_atom5_dta_metadata.json \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5_dta.pkl.checkpoint \
  --checkpoint_every 500 \
  --progress_every 500 \
  --resume
```

可选 ESM2 预提取：

```bash
CUDA_VISIBLE_DEVICES=7 /home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/extract_pdbbind_esm.py \
  --input_checkpoint_dir data/pdbbind_protein_ligand_atom5_dta.pkl.checkpoint \
  --output_checkpoint_dir data/pdbbind_protein_ligand_atom5_esm_dta.pkl.checkpoint \
  --esm_model_path /home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D \
  --device cuda:0 \
  --batch_size 16 \
  --resume
```

划分命令：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/split_pdbbind_dta.py \
  --checkpoint_dir data/pdbbind_protein_ligand_atom5_esm_dta.pkl.checkpoint \
  --output_dir data/PDBbind_atom5_esm_DTA/random \
  --split_method random \
  --seed 99
```

训练命令：

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
  --dataset PDBbind_atom5_esm_DTA \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_atom5_esm_DTA_random/ \
  --cuda 7 \
  --batch_size 2 \
  --epoch 30 \
  --loss smooth_l1 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001
```

验证：

```text
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/tests/test_dta_setup.py -> ok
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python -m py_compile dta/process_pdbbind_dta.py dta/split_pdbbind_dta.py dta/model_dta.py dta/train_dta.py dta/tests/test_dta_setup.py -> exit 0
process_pdbbind_dta.py --help -> exit 0
split_pdbbind_dta.py --help -> exit 0
train_dta.py --help -> exit 0
```

真实小样本 smoke test：

```text
命令：dta/process_pdbbind_dta.py --max_samples 2 写入 /tmp
结果：candidate_records 18881, kept 2, parse_errors 0
命令：dta/split_pdbbind_dta.py 对 /tmp checkpoint 划分
结果：manifest 和 split_summary 正常写出
```




## 2026-05-22 更新：DTA 默认改为 residue/CA

为了先跑通更省显存的 DTA 完整流程，`dta/process_pdbbind_dta.py` 默认不再使用 atom5 作为 protein 3D token，而是使用氨基酸 residue id + CA 坐标。

```text
protein_3d_mode: residue
RNA_feats 位置: residue/amino-acid id
RNA_coors 位置: CA 坐标
protein_3d_num_tokens: 21
```

PDBbind 原始目录当前按压缩包读取，不要求提前解压。

```text
PDBbind/index.tar.gz                 affinity 标签索引
PDBbind/P-L.tar.gz                   protein-ligand 结构数据
P-L/<year-range>/<pdb_id>/           每个复合物一个目录
<pdb_id>_pocket.pdb                  当前脚本使用的 pocket 结构
<pdb_id>_ligand.sdf                  当前脚本优先使用的 ligand 结构
<pdb_id>_ligand.mol2                 sdf 不可用时作为后备
<pdb_id>_protein.pdb                 当前脚本暂不使用
```

如果显式需要旧 atom5 版本，可以在转换命令中加 `--protein_3d_mode atom5`。当前推荐先跑 residue/CA + ESM：数据路径用 `data/pdbbind_protein_ligand_dta.pkl.checkpoint`，ESM 输出用 `data/pdbbind_protein_ligand_esm_dta.pkl.checkpoint`，训练数据集名用 `PDBbind_esm_DTA`。

## 2026-06-01 起：后续实验统一记录规则

从这一节开始，之后所有 GerNA-DTA / DTI 融合相关实验都持续追加到本文件，避免结果分散在终端日志里。

每个实验至少记录：

```text
实验编号：
实验目的：
数据版本：
模型路线：
关键参数：
运行命令：
显存/速度：
best valid 指标：
对应 test 指标：
模型路径：
结论：
下一步：
```

如果实验失败，也记录：

```text
失败阶段：
报错信息：
初步原因：
处理方式：
是否需要重跑：
```

后续默认优先比较：

```text
valid RMSE 越低越好
valid Pearson / Spearman / CI 越高越好
test 只作为最终泛化参考，不用 test 选模型
```

## 2026-06-01 当前 DTA 主线结果汇总

当前主线只保留两个 GerNA-DTA 路线：

```text
1. residue/CA 3D + chain ESM
2. pocket all-atom 3D + chain ESM
```

已不作为主线继续推进：

```text
pocket pharmacophore ESM
早期 atom5 口袋原子版本
口袋 ESM，不连续 pocket 序列不适合作为主 ESM 来源
```

### residue/CA + chain ESM

数据集：

```text
data/PDBbind_chain_esm_residue_DTA/random
```

训练设置：

```text
protein_3d_mode: residue
protein_embedding_mode: esm
protein_edge_mode: radius
protein_contact_cutoff: 6.0
batch_size: 2
loss: smooth_l1
early_stop_patience: 5
```

当前确认最好结果：

```text
valid 1554
RMSE 1.314111
MAE 1.018679
Pearson 0.694022
Spearman 0.698173
CI 0.755570

test 1553
RMSE 1.341111
MAE 1.035488
Pearson 0.696471
Spearman 0.708099
CI 0.757421
```

结论：

```text
这是目前 GerNA-DTA 最强 baseline。
虽然 3D 只用 CA 点，但训练稳定、显存可控、验证和测试一致性较好。
```

### pocket all-atom + chain ESM

数据集：

```text
data/PDBbind_chainESM_pocketAllAtom_DTA_atom800/random
```

训练设置：

```text
protein_3d_mode: all_atom
structure_lmdb: pocket LMDB
ESM 来源: ligand 接触链 / 作用链 chain ESM
max_protein_atom: 800
batch_size: 1
loss: smooth_l1
early_stop_patience: 5
```

当前确认最好结果：

```text
valid 1354
RMSE 1.572316
MAE 1.240450
Pearson 0.570830
Spearman 0.581946
CI 0.705642

test 1354
RMSE 1.593689
MAE 1.263059
Pearson 0.559095
Spearman 0.561738
CI 0.698122
```

结论：

```text
当前 all-atom 版本没有超过 residue/CA。
主要原因不是 all-atom 理论上无效，而是当前 all-atom 输入只把原子元素 token + 坐标送入 Equiformer，
缺少 residue identity、侧链功能类型、原子名、原子所属 residue、与 chain ESM 的细粒度对齐等信息。
同时 all-atom token 数多，batch size 小，训练更慢且更不稳定。
```

### pocket pharmacophore ESM

当前结果：

```text
valid 1205
RMSE 1.608735
MAE 1.224561
Pearson 0.529691
Spearman 0.590028
CI 0.708941

test 1205
RMSE 1.684285
MAE 1.278204
Pearson 0.473898
Spearman 0.522403
CI 0.682456
```

结论：

```text
不作为主线。
它压缩了 all-atom 信息，稳定性和效果都没有超过 residue/CA。
```

## 2026-06-01 DTI 融合来源确定

用户指定另一个项目中当前最好的 DTI 代码：

```text
/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen/train_dta_seq_interaction_tokenonly_unimol_v12_pooledfusion.py
```

对应 run：

```text
/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen/runs/seq_interaction_tokenonly_unimol_v12_pooledfusion/20260517_224200
```

选用 checkpoint：

```text
/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen/runs/seq_interaction_tokenonly_unimol_v12_pooledfusion/20260517_224200/best_pearson.pt
```

注意：

```text
虽然文件名叫 best_pearson.pt，但该 run 的 config 中 early_stop_metric = "auc"。
因此这个 checkpoint 实际是按 valid AUC 最优保存。
```

该 DTI 模型验证集最佳点：

```text
epoch 3
global_step 13000
trigger val@step13000

AUC      0.735095
AUPR     0.775979
Accuracy 0.687772
F1       0.741916
RMSE     0.456545
MAE      0.431347
Pearson  0.398439
Spearman 0.403033
C-index  0.733836
```

最终 final_val：

```text
num_samples 28018
AUC      0.706031
AUPR     0.745449
Accuracy 0.638125
F1       0.649788
RMSE     0.492183
MAE      0.404524
Pearson  0.346626
Spearman 0.353198
C-index  0.705532
```

外部 benchmark 结果仅作参考，不直接作为 GerNA-DTA 融合实验的选择依据：

```text
DUD-E:
global AUC 0.739159
global AUPR 0.134327

LIT-PCBA:
global AUC 0.191694
global AUPR 0.069993
```

## 2026-06-01 GerNA-DTA 已加入 DTI 融合接口

新增目的：

```text
不在 GerNA 项目内重新处理 DrugCLIP 数据。
只要求外部 DTI 项目把每个 PDBbind 样本的中间 embedding 写成 GerNA 能读取的 LMDB。
```

外部 LMDB 格式：

```python
{
    "protein_tokens": np.ndarray((L, D_front), dtype=np.float32),
    "protein_mask": np.ndarray((L,), dtype=bool),  # 可选
    "global": np.ndarray((D_late,), dtype=np.float32),
}
```

key：

```text
使用 GerNA manifest 中的 lmdb_key，例如 1m13_protein。
```

硬性约束：

```text
protein_tokens 的 L 必须等于 GerNA 当前样本的 residue 数。
否则前融合不成立，代码会直接报错。
```

建议从 DTI v12 模型提取的特征：

```text
front fusion:
protein_tokens = DTI backbone.prot_proj 后、TokenLengthCompressor 前的 protein token
shape [L, 256]
GerNA 参数 --dti_front_input_dim 256

late fusion:
global = concat(DTI protein_vec, DTI ligand_vec)
shape [512]
GerNA 参数 --dti_late_input_dim 512 --dti_late_hidden_dim 128
```

GerNA-DTA 新增参数：

```text
--dti_fusion_lmdb_path
--dti_front_fusion none|add|concat
--dti_front_input_dim
--dti_late_fusion none|concat
--dti_late_input_dim
--dti_late_hidden_dim
```

已通过验证：

```text
python3 -m py_compile data_utils/lmdb_dataset.py data_utils/dataset.py net/model.py dta/model_dta.py dta/train_dta.py dta/eval_dta.py dta/tests/test_dta_setup.py -> exit 0

/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/tests/test_dta_setup.py -> ok
```

下一步实验计划：

```text
1. 编写或使用外部脚本，从 DTI v12 best_pearson.pt 提取 protein_tokens/global 到 LMDB。
2. 先跑 residue/CA + chain ESM + DTI front add。
3. 再跑 residue/CA + chain ESM + DTI late concat。
4. 如果两个单独方向有效，再跑 front add + late concat 联合。
5. all-atom 路线只在 residue 路线看到收益后再接 DTI 融合，避免训练成本过高。
```

## 后续实验追加模板

```text
### 实验 YYYY-MM-DD-N：实验名

实验目的：

数据版本：

模型路线：

关键参数：

运行命令：

运行状态：

结果：
valid ...
test  ...

模型路径：

结论：

下一步：
```

## 2026-06-02 实验对比：去掉 protein 2D vs 保留 protein 2D

实验目的：

```text
比较 residue/CA 路线和 pocket all-atom 路线在去掉 protein 2D 后的效果，
判断 DTI 融合阶段应该基于哪个 GerNA-DTA 架构继续。
```

### 路线一：residue/CA + chain ESM

保留 protein 2D 的当前最好结果，来源于训练输出记录：

```text
dataset: data/PDBbind_chain_esm_residue_DTA/random
model: Model/PDBbind_chain_esm_residue_DTA_random/Model_dta_best.pth
protein_3d_mode: residue
protein_embedding_mode: esm
disable_protein_2d: false

valid 1554
RMSE     1.314111
MAE      1.018679
Pearson  0.694022
Spearman 0.698173
CI       0.755570

test 1553
RMSE     1.341111
MAE      1.035488
Pearson  0.696471
Spearman 0.708099
CI       0.757421
```

去掉 protein 2D 的结果，来源于 `Model/PDBbind_chain_esm_residue_DTA_noProtein2D_random/best_metrics.json`：

```text
dataset: data/PDBbind_chain_esm_residue_DTA/random
model: Model/PDBbind_chain_esm_residue_DTA_noProtein2D_random/Model_dta_best.pth
protein_3d_mode: residue
protein_embedding_mode: esm
disable_protein_2d: true
best epoch: 4

valid
RMSE     1.475106
MAE      1.145961
Pearson  0.642228
Spearman 0.644995
CI       0.730768

test
RMSE     1.524103
MAE      1.185277
Pearson  0.632400
Spearman 0.644878
CI       0.729018
```

结论：

```text
residue/CA 路线中，去掉 protein 2D 明显变差。
protein 2D 对 residue/CA 版本仍然有用，可能补充了 CA 级 3D 图没有显式表达的 residue-residue 几何/配对先验。
```

### 路线二：pocket all-atom + chain ESM

保留 protein 2D 的当前最好结果，来源于 `Model/PDBbind_chainESM_pocketAllAtom_DTA_atom800_random/eval_best_metrics.json`：

```text
dataset: data/PDBbind_chainESM_pocketAllAtom_DTA_atom800/random
model: Model/PDBbind_chainESM_pocketAllAtom_DTA_atom800_random/Model_dta_best.pth
protein_3d_mode: all_atom
protein_embedding_mode: esm
disable_protein_2d: false

valid 1354
RMSE     1.572316
MAE      1.240450
Pearson  0.570830
Spearman 0.581946
CI       0.705642

test 1354
RMSE     1.593689
MAE      1.263059
Pearson  0.559095
Spearman 0.561738
CI       0.698122
```

去掉 protein 2D 的结果，来源于 `Model/PDBbind_chainESM_pocketAllAtom_DTA_atom800_noProtein2D_random/best_metrics.json`：

```text
dataset: data/PDBbind_chainESM_pocketAllAtom_DTA_atom800/random
model: Model/PDBbind_chainESM_pocketAllAtom_DTA_atom800_noProtein2D_random/Model_dta_best.pth
protein_3d_mode: all_atom
protein_embedding_mode: esm
disable_protein_2d: true
best epoch: 2

valid
RMSE     1.468273
MAE      1.144503
Pearson  0.615609
Spearman 0.618029
CI       0.721748

test
RMSE     1.518537
MAE      1.188038
Pearson  0.592121
Spearman 0.602971
CI       0.713832
```

结论：

```text
pocket all-atom 路线中，去掉 protein 2D 反而更好。
原因可能是 all-atom 3D 已经提供了更直接的局部空间信息，而额外的 protein 2D pair 特征带来噪声或优化负担。
但即便去掉 protein 2D，all-atom 当前仍然没有超过 residue/CA + protein 2D 主线。
```

### DTI 融合选择

```text
主线 DTI 融合选择：
residue/CA + chain ESM + 保留 protein 2D

理由：
1. 当前所有 GerNA-DTA 结果里 test RMSE/MAE/Pearson/Spearman/CI 综合最好。
2. valid 和 test 一致性最好，泛化更稳定。
3. DTI 前融合的 protein_tokens 是 residue/token 级特征，和 residue/CA 路线天然对齐。
4. all-atom noProtein2D 虽然优于 all-atom withProtein2D，但整体仍弱于 residue/CA 主线。

后续如果专门做 all-atom + DTI 融合，应该使用：
pocket all-atom + chain ESM + 去掉 protein 2D
```

## 2026-06-02 备注：noProtein2D 结果的可比性限制

用户补充：

```text
去掉 protein 2D 的实验并未完整跑完，训练中途因服务器原因中断。
因此当前 noProtein2D 的 best_metrics 只能作为趋势观察，不能视为严格最终结果。
```

混杂变量：

```text
all-atom 路线中，去掉 protein 2D 后显存下降，batch size 可能从 1 提高到 2。
这会改变优化过程：梯度噪声、每个 epoch 的 update 次数、Adam 的有效更新轨迹都会变化。
因此 all-atom withProtein2D vs noProtein2D 的提升，不能完全归因于去掉 protein 2D。
```

当前更稳妥判断：

```text
1. residue/CA 路线：如果保留 2D 和去掉 2D 都是 batch_size=2，则比较相对更干净；但 noProtein2D 仍需完整跑完再定论。
2. all-atom 路线：noProtein2D 当前看起来更好，但它同时改变了 protein 2D 开关和 batch size，结论不严格。
3. DTI 融合主线暂时仍建议优先使用最完整、最稳定、已完整跑完的 residue/CA + chain ESM + 保留 protein 2D。
4. 若要验证去掉 protein 2D 是否真的更好，需要做同 batch size 对照实验。
```

建议对照实验：

```text
A. residue/CA + chain ESM + 保留 protein 2D，batch_size=2
B. residue/CA + chain ESM + 去掉 protein 2D，batch_size=2

C. all-atom + chain ESM + 保留 protein 2D，batch_size=1
D. all-atom + chain ESM + 去掉 protein 2D，batch_size=1
E. all-atom + chain ESM + 去掉 protein 2D，batch_size=2

其中 C vs D 判断去掉 2D 的真实影响；D vs E 判断 batch size 的影响。
```

## 2026-06-03 运行记录：DTI front add 训练前置 LMDB 缺失

用户运行：

```text
PDBbind_chain_esm_residue_DTA + dti_front_fusion add
--dti_fusion_lmdb_path data/PDBbind_dti_v12_fusion_lmdb
```

报错：

```text
lmdb.Error: /home/user/wangzhiwen/GerNA-Bind/data/PDBbind_dti_v12_fusion_lmdb: No such file or directory
```

根因：

```text
训练命令本身已经进入 train_path/valid_path/test_path 解析阶段，说明参数解析正常。
失败发生在 DataLoader worker 读取 DTI fusion LMDB 时。
data/PDBbind_dti_v12_fusion_lmdb 尚未生成，因此不是模型训练错误。
```

修复：

```text
新增 dta/extract_dti_v12_fusion_lmdb.py
新增 dta/tests/test_dti_fusion_extract.py

提取策略：
从 GerNA 的 chain ESM [L,640] 读取 protein residue embedding，
使用 DrugCLIP v12 checkpoint 中 backbone.prot_proj.weight/bias 投影为 protein_tokens [L,256]，
写入 GerNA 可读取的 DTI fusion LMDB。

LMDB key: GerNA manifest 中的 lmdb_key，例如 1m13_protein
LMDB value: pickle dict，包含 protein_tokens [L,256] 和 protein_mask [L]
```

已验证：

```text
python3 -m py_compile dta/extract_dti_v12_fusion_lmdb.py dta/tests/test_dti_fusion_extract.py -> exit 0
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/tests/test_dti_fusion_extract.py -> ok

3 样本 smoke extraction:
written 3
feature_missing 0
shape_errors 0
prot_proj input_dim 640
prot_proj output_dim 256
读回样本 1m13_protein: protein_tokens shape (273, 256), protein_mask shape (273,)
```

下一步：

```text
先运行完整 extraction 生成 data/PDBbind_dti_v12_fusion_lmdb，
然后重新运行 DTI front add 训练命令。
```

## 2026-06-04 更新：residue/CA 去掉 protein 2D 完整重跑趋势已超过保留 2D

实验路线：

```text
dataset: data/PDBbind_chain_esm_residue_DTA/random
model: Model/PDBbind_chain_esm_residue_DTA_noProtein2D_random/Model_dta_best.pth
protein_3d_mode: residue
protein_embedding_mode: esm
disable_protein_2d: true
batch_size: 2
loss: smooth_l1
```

当前 best_metrics.json 已更新到 epoch 11：

```text
epoch 11
train_loss 0.637786
learning_rate 0.00015

valid 1554
RMSE     1.264559
MAE      0.980991
Pearson  0.726325
Spearman 0.719164
CI       0.766536

test 1553
RMSE     1.309389
MAE      1.004716
Pearson  0.719031
Spearman 0.724378
CI       0.765482
```

与此前 residue/CA + chain ESM + 保留 protein 2D 最好结果对比：

```text
with protein 2D:
valid RMSE 1.314111, MAE 1.018679, Pearson 0.694022, Spearman 0.698173, CI 0.755570
test  RMSE 1.341111, MAE 1.035488, Pearson 0.696471, Spearman 0.708099, CI 0.757421

no protein 2D, epoch 11 current best:
valid RMSE 1.264559, MAE 0.980991, Pearson 0.726325, Spearman 0.719164, CI 0.766536
test  RMSE 1.309389, MAE 1.004716, Pearson 0.719031, Spearman 0.724378, CI 0.765482
```

结论更新：

```text
residue/CA 路线去掉 protein 2D 后已经超过保留 protein 2D 的完整基线。
由于二者使用同一 residue 数据集且 batch_size 均为 2，这个比较比之前 all-atom 的 batch size 混杂更可信。
DTI 前融合主线应切换为：residue/CA + chain ESM + disable_protein_2d + DTI front add。
```

注意：

```text
当前训练仍在继续，最终数值以 early stopping 后 best_metrics.json 为准。
但架构选择可以先更新为 noProtein2D residue 主线。
```

## DTA 实验结果统计（更新：2026-06-08）

说明：`random` 与 `fixed_seed42_filtered` 不是同一划分，不能严格横向比较。后续主对比以 `fixed_seed42_filtered` 为准。

### Fixed Seed42 Filtered 主结果

| 实验路线 | 结果来源 | Best epoch | Valid RMSE | Valid MAE | Valid Pearson | Valid Spearman | Valid CI | Test RMSE | Test MAE | Test Pearson | Test Spearman | Test CI | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chain ESM + residue 3D + noProtein2D + ligand Uni-Mol 前融合 concat | `best_metrics.json` | 20 | 1.257870 | 0.957563 | 0.728742 | 0.725549 | 0.767864 | 1.280396 | 0.976407 | 0.715201 | 0.714782 | 0.763105 | 当前 fixed split 最好，有 checkpoint |
| chain ESM + residue 3D + noProtein2D + DTI late concat h128 | terminal recovered | 10 | 1.277171 | 0.975824 | 0.701343 | 0.703007 | 0.756206 | 1.300115 | 0.993446 | 0.685584 | 0.686943 | 0.750703 | 指标从终端找回；对应 best checkpoint 可能已被覆盖 |
| chain ESM + residue 3D + noProtein2D + DTI late concat h128 | overwritten `best_metrics.json` | 4 | 1.408631 | 1.060398 | 0.632683 | 0.679252 | 0.744251 | 1.449235 | 1.088403 | 0.604359 | 0.638514 | 0.728935 | 后续重跑覆盖后的记录，不作为该路线最好结果 |

### Random Split 历史结果（仅作参考）

| 实验路线 | Split | Best epoch | Valid RMSE | Valid MAE | Valid Pearson | Valid Spearman | Valid CI | Test RMSE | Test MAE | Test Pearson | Test Spearman | Test CI | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DTI front add | random | 14 | 1.306299 | 1.003504 | 0.715363 | 0.712131 | 0.761535 | 1.326820 | 1.022276 | 0.716560 | 0.727985 | 0.765698 | 旧 random 结果，不能直接与 fixed split 比 |
| residue + noProtein2D | random | 2 | 1.491546 | 1.157390 | 0.619450 | 0.628409 | 0.724112 | 1.544204 | 1.197383 | 0.609895 | 0.632736 | 0.724174 | random baseline |
| pocket all-atom atom800 + noProtein2D | random | 2 | 1.524702 | 1.199464 | 0.592852 | 0.607059 | 0.715189 | 1.567626 | 1.224702 | 0.576002 | 0.590396 | 0.708666 | all-atom 当前不理想 |
| pocketPharmESM | random | 3 | 1.608735 | 1.224561 | 0.529691 | 0.590028 | 0.708941 | 1.684285 | 1.278204 | 0.473898 | 0.522403 | 0.682456 | 淘汰 |

### 当前结论

1. `fixed_seed42_filtered` 上，ligand Uni-Mol 前融合 concat 仍是当前最好路线：test RMSE 1.280396, Pearson 0.715201。
2. DTI late concat 的终端恢复最好结果接近前融合，但略差：test RMSE 1.300115, Pearson 0.685584。
3. 因为 DTI late concat 的 checkpoint 可能已经被覆盖，当前只能把终端恢复结果用于实验统计，不能用于模型加载。
4. 训练脚本已改为默认拒绝覆盖已有 `Model_dta_best.pth` / `Model_dta_last.pth` / `best_metrics.json`，并自动写 `train.log`。

