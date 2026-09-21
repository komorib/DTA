# PBCNet2.0 per-atom 配体特征抽取（便携包）

把这个目录整个拷到另一台**有 dgl 的服务器**上跑，产出 per-atom LMDB，再把 LMDB 拷回来即可。

## 这个包里有什么
```
extract_pbcnet2_atom_lmdb.py   # 抽取脚本（自包含，不依赖 GerNA 仓库其它代码）
merge_lmdb.py                  # 分片并行后合并各分片 LMDB
PBCNet2.0/                     # PBCNet2.0 模型代码 + 权重
  Graph2pickle.py              #   建图
  model_code/                  #   TensorNet / readout 等
  PBCNet2.pth                  #   官方权重 (hidden_channels=96)
manifests/                     # train/valid/test manifest（列出要抽哪些复合物，共 ~15267 个）
README.md
```

## 还需要单独传过去的大文件（不在包里）
**特征 LMDB（约 23G）**：每个复合物的蛋白/配体特征，抽取的输入。
本服务器路径：`/data/home/scxk284/run/gerna/data/PDBbind_chain_lmdb_esm_gernabind_v2/`
（是个目录，含 `data.mdb` + `lock.mdb`）。整目录 rsync/scp 过去，记下它在对方服务器的路径，下面用 `--feature-lmdb-path` 指过去。
> manifest 里自带的 `lmdb_path` 是旧机器的死路径，已被 `--feature-lmdb-path` 覆盖，无需管它。

## 依赖（对方服务器的 python 环境）
python 3.8–3.11 + 这些包：
```
torch==2.1.*        # 任意能被 dgl 匹配的版本即可
dgl==2.1.0          # CPU 版即可（PyPI: pip install dgl==2.1.0）；模型前向走 dgl
rdkit  biopython  lmdb  numpy<2  pandas
```
> 注：dgl 的 PyPI 轮子是 CPU 版，模型前向在 CPU；瓶颈是建图（CPU），GPU 帮不上，**用 CPU 跑即可**。
> 如果对方有现成的 torch2.1 环境，直接 `pip install dgl==2.1.0 pandas` 补两个包最省事。

## 跑抽取
```bash
cd <这个包目录>
FEAT=/对方服务器/上/PDBbind_chain_lmdb_esm_gernabind_v2   # 你拷过去的 23G 特征库

python extract_pbcnet2_atom_lmdb.py \
  --manifest manifests/train_manifest.json \
  --manifest manifests/valid_manifest.json \
  --manifest manifests/test_manifest.json \
  --feature-lmdb-path "$FEAT" \
  --output-lmdb PDBbind_pbcnet2_atom_fixed_seed42_filtered \
  --map-size-gb 8 --progress-every 200 --resume
```
- `--pbcnet-root` / `--checkpoint` 默认就指向包内 `PBCNet2.0/`，不用传。
- `--resume` 可断点续跑（重复同命令跳过已写 key）。
- 输出：`PDBbind_pbcnet2_atom_fixed_seed42_filtered/`（每个 key 存 `ligand_tokens [A,96]` + `ligand_mask` + `atoms`）。

## 加速：CPU 多进程分片 + 合并（~15267 个，单进程几小时，分片几十分钟）
开 N 个进程各跑一个分片，各写各的 LMDB，最后合并：
```bash
FEAT=/对方服务器/.../PDBbind_chain_lmdb_esm_gernabind_v2
N=16
for i in $(seq 0 $((N-1))); do
  python extract_pbcnet2_atom_lmdb.py \
    --manifest manifests/train_manifest.json \
    --manifest manifests/valid_manifest.json \
    --manifest manifests/test_manifest.json \
    --feature-lmdb-path "$FEAT" \
    --num-shards $N --shard $i \
    --output-lmdb shards/shard_$i --map-size-gb 4 --resume &
done
wait

# 合并所有分片到最终 LMDB
python merge_lmdb.py $(for i in $(seq 0 $((N-1))); do echo --input shards/shard_$i; done) \
  --output PDBbind_pbcnet2_atom_fixed_seed42_filtered --map-size-gb 8
```
> 若特征库放在网络盘且报 `Bus error`（LMDB mmap over 网络FS），先把 `$FEAT` 放到本地盘（如 /dev/shm 或本地 NVMe）再抽。

## 抽完拿回来
把 `PDBbind_pbcnet2_atom_fixed_seed42_filtered/`（应该不大，~几百 MB）拷回本服务器：
```
放到: /data/home/scxk284/run/gerna/data/PDBbind_pbcnet2_atom_fixed_seed42_filtered/
```
然后这边直接 `sbatch run_gernabind_dta_pbcnet_atom.sh`（已配 `--ligand_pbcnet_atom_input_dim 96`）。

## 自检（对方环境装好后，先验证能加载）
```bash
python -c "import torch,dgl,rdkit,lmdb,Bio,pandas;print('torch',torch.__version__,'dgl',dgl.__version__)"
python -c "import sys;sys.path[:0]=['PBCNet2.0','PBCNet2.0/model_code'];import torch;m=torch.load('PBCNet2.0/PBCNet2.pth',map_location='cpu',weights_only=False);print('hidden',m.hidden_channels)"
# 小样本 smoke（先抽 5 个确认跑通）
python extract_pbcnet2_atom_lmdb.py --manifest manifests/valid_manifest.json \
  --feature-lmdb-path "$FEAT" --output-lmdb /tmp/_smoke --max-samples 5 --progress-every 1
```
