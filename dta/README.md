# DTA / Affinity Regression

这个目录是单独的 DTA 路线，不覆盖原论文二分类复现代码。

## 任务定义

```text
输入：protein pocket + ligand
标签：pAffinity = -log10(Kd/Ki/IC50 in M)
输出：一个连续值
loss：MSE 或 SmoothL1
指标：RMSE / MAE / Pearson / Spearman / CI
```

默认只保留精确 affinity 记录：`Kd=...`、`Ki=...`、`IC50=...`。`<`、`>` 这类上下界默认不作为精确回归标签，除非显式加 `--include_bounds`。

## 1. 构建 DTA 数据

默认构建 residue/CA 级 protein 3D：`RNA_feats` 位置使用氨基酸 id，`RNA_coors` 位置使用 CA 坐标。这个版本比 atom5 显存低很多。

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/process_pdbbind_dta.py \
  --pdbbind_dir PDBbind \
  --output data/pdbbind_protein_ligand_dta.pkl \
  --metadata_output data/pdbbind_protein_ligand_dta_metadata.json \
  --checkpoint_dir data/pdbbind_protein_ligand_dta.pkl.checkpoint \
  --checkpoint_every 500 \
  --progress_every 500 \
  --resume
```

如果还要旧的 atom5 版本，额外加：

```bash
--protein_3d_mode atom5
```

## 2. 可选：预提取 ESM2

现有 `data_utils/extract_pdbbind_esm.py` 可以复用，因为 DTA checkpoint 的字段顺序不变，只是最后一项 label 从 0/1 变成 pAffinity。

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python data_utils/extract_pdbbind_esm.py \
  --input_checkpoint_dir data/pdbbind_protein_ligand_dta.pkl.checkpoint \
  --output_checkpoint_dir data/pdbbind_protein_ligand_esm_dta.pkl.checkpoint \
  --esm_model_path /home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D \
  --device cuda:0 \
  --batch_size 2 \
  --resume
```

## 3. 划分数据

```bash
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/split_pdbbind_dta.py \
  --checkpoint_dir data/pdbbind_protein_ligand_esm_dta.pkl.checkpoint \
  --output_dir data/PDBbind_esm_DTA/random \
  --split_method random \
  --seed 99
```

也可以把 `--split_method` 换成 `protein`、`mol`、`both`。

## 4. 训练 DTA

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
  --dataset PDBbind_esm_DTA \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_esm_DTA_random/ \
  --cuda 0 \
  --batch_size 2 \
  --num_workers 4 \
  --epoch 30 \
  --loss smooth_l1 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001 \
  --eval_train_every 0
```

这里的 `--cuda 0` 指的是 `CUDA_VISIBLE_DEVICES=7` 暴露出来的第 0 张卡，也就是物理 7 号卡。当前训练脚本只走 FP32，已移除 AMP 入口，避免 Equiformer 和融合特征出现 Half/Float dtype 不一致。

输出文件：

```text
Model_dta_best.pth
Model_dta_last.pth
best_metrics.json
```

`best_metrics.json` 会记录 best epoch 对应的 valid/test 完整指标：RMSE / MAE / Pearson / Spearman / CI。

## 5. 去掉 protein 2D 的消融实验

这个消融保留 chain ESM、ligand 2D graph、protein/ligand 3D 分支，但关闭 protein residue graph 和 2D pairwise triangle 中的 protein CA-CA 几何。训练时加：

```bash
--disable_protein_2d
```

residue/CA no-protein-2D：

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
  --dataset PDBbind_chain_esm_residue_DTA \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_chain_esm_residue_DTA_noProtein2D_random/ \
  --cuda 0 \
  --protein_embedding_mode esm \
  --protein_edge_mode radius \
  --protein_contact_cutoff 6.0 \
  --batch_size 2 \
  --num_workers 4 \
  --epoch 30 \
  --loss smooth_l1 \
  --early_stop_patience 5 \
  --disable_protein_2d
```

all-atom atom800 no-protein-2D：

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
  --dataset PDBbind_chainESM_pocketAllAtom_DTA_atom800 \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_chainESM_pocketAllAtom_DTA_atom800_noProtein2D_random/ \
  --cuda 0 \
  --protein_embedding_mode esm \
  --protein_edge_mode radius \
  --protein_contact_cutoff 6.0 \
  --batch_size 1 \
  --num_workers 4 \
  --epoch 30 \
  --loss smooth_l1 \
  --early_stop_patience 5 \
  --disable_protein_2d
```


## 6. 外部 DTI 特征融合接口

GerNA-DTA 现在可以读取一个额外 LMDB，把另一个 DTI 模型的中间特征接入本模型。这个 LMDB 不要求由本仓库生成，你后面自己处理 DrugCLIP/其他模型输出时，只要按下面格式写入即可。

LMDB key 使用当前样本的 `lmdb_key`，例如 manifest 里的 `1m13_protein`。value 是 `pickle.dumps(dict)`，支持字段：

```python
{
    # 前融合用：未 pooling 的蛋白 token，必须和 GerNA residue 长度 L 对齐
    "protein_tokens": np.ndarray(shape=(L, D_front), dtype=np.float32),

    # 可选；不写则默认全部为 True
    "protein_mask": np.ndarray(shape=(L,), dtype=bool),

    # 后融合用：已经 pooling 的 DTI 全局向量
    "global": np.ndarray(shape=(D_late,), dtype=np.float32),
}
```

别名也能识别：`prot_tokens` / `protein_token_embeddings` / `dti_protein_tokens`，以及 `pooled` / `pooled_embedding` / `global_embedding` / `dti_global` / `pair_embedding`。前融合要求 `protein_tokens.shape[0] == n_residue`，否则训练会直接报错，避免把不对齐的 token 硬拼进去。

只做前融合，推荐先用 `add`：

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
  --dataset PDBbind_chain_esm_residue_DTA \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_chain_esm_residue_DTA_dtiFrontAdd_random/ \
  --cuda 0 \
  --protein_embedding_mode esm \
  --protein_edge_mode radius \
  --protein_contact_cutoff 6.0 \
  --batch_size 2 \
  --num_workers 4 \
  --epoch 30 \
  --loss smooth_l1 \
  --early_stop_patience 5 \
  --dti_fusion_lmdb_path data/your_dti_fusion_lmdb \
  --dti_front_fusion add \
  --dti_front_input_dim 256
```

只做后融合：

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/train_dta.py \
  --dataset PDBbind_chain_esm_residue_DTA \
  --split_method random \
  --data_root data \
  --model_output_path Model/PDBbind_chain_esm_residue_DTA_dtiLate_random/ \
  --cuda 0 \
  --protein_embedding_mode esm \
  --protein_edge_mode radius \
  --protein_contact_cutoff 6.0 \
  --batch_size 2 \
  --num_workers 4 \
  --epoch 30 \
  --loss smooth_l1 \
  --early_stop_patience 5 \
  --dti_fusion_lmdb_path data/your_dti_fusion_lmdb \
  --dti_late_fusion concat \
  --dti_late_input_dim 512 \
  --dti_late_hidden_dim 128
```

前融合和后融合可以同时打开。all-atom 路线也一样，只需要把 `--dataset`、`--model_output_path`、`--batch_size` 换成当前 all-atom 设置。

## 7. 重新评估 best checkpoint

如果某次训练终端日志没有保存完整指标，可以用 `eval_dta.py` 重新加载 best checkpoint 评估 valid/test。例子：

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
/home/user/wangzhiwen/miniconda3/envs/gernabind/bin/python dta/eval_dta.py \
  --dataset PDBbind_chainESM_pocketAllAtom_DTA_atom800 \
  --split_method random \
  --data_root data \
  --checkpoint_path Model/PDBbind_chainESM_pocketAllAtom_DTA_atom800_random/Model_dta_best.pth \
  --output_json Model/PDBbind_chainESM_pocketAllAtom_DTA_atom800_random/eval_best_metrics.json \
  --cuda 0 \
  --protein_embedding_mode esm \
  --protein_edge_mode radius \
  --protein_contact_cutoff 6.0 \
  --batch_size 1 \
  --num_workers 4 \
  --eval_splits valid,test
```

这里的 `--cuda 0` 仍然表示 `CUDA_VISIBLE_DEVICES=7` 暴露出来的第 0 张卡，也就是物理 7 号卡。

## 代码边界

- `model_dta.py` 复用 GerNA 主体，只去掉回归头上的 sigmoid。
- `train_dta.py` 是单 GPU 回归训练入口。
- 原始 `train_model.py` 仍然是论文二分类路线。
