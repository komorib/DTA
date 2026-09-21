#!/usr/bin/env bash
set -euo pipefail

# 强制切换到【你自己的】工作根目录
cd /home/user/yangyuhang/KDBNet

GPU_ID=${1:-1}
SEEDS="1 2 3 4 5"
STAMP=$(date +%Y%m%d_%H%M%S)
N_EPOCHS=100
PATIENCE=20
BATCH_SIZE=32
MANIFEST="data/PDBbind_full/processed/pdbbind_pl_manifest.tsv" # 现在你也有这个路径了

# 用你自己的 Python 环境
PYTHON="/home/user/yangyuhang/miniconda3/envs/kdb_env/bin/python"

for SEED in $SEEDS; do
  RUN_TAG="pdbbind_random_seed${SEED}_gpu${GPU_ID}_${STAMP}"
  OUT_DIR="./output/PDBbind_runs/${RUN_TAG}"
  mkdir -p "$OUT_DIR"
  
  echo " [$(date +%T)] 开始狂奔 -> Seed: $SEED | 输出将被保存在: $OUT_DIR"
  
  # 调用现在属于你的 run_example.py
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/run_example.py \
    --task pdbbind_pl \
    --pdbbind_manifest "$MANIFEST" \
    --split_method random \
    --seed "$SEED" \
    --n_epochs "$N_EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --monitor_metric mse \
    --output_dir "$OUT_DIR" \
    --save_prediction \
    --save_log \
    --save_checkpoint
    
  echo "[$(date +%T)] Seed $SEED 训练圆满结束！"
done
