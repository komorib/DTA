#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Haru/Desktop/dta/KDBNet

GPU_ID=${KDBNET_GPU_ID:-0}
SEED=${KDBNET_SEED:-42}
STAMP=${KDBNET_STAMP:-$(date +%Y%m%d_%H%M%S)}
N_EPOCHS=${KDBNET_N_EPOCHS:-100}
PATIENCE=${KDBNET_PATIENCE:-20}
BATCH_SIZE=${KDBNET_BATCH_SIZE:-32}

PYTHON=${KDBNET_PYTHON:-/home/haru/miniconda3/envs/kdbnet/bin/python}
MANIFEST=${KDBNET_MANIFEST:-data/PDBbind_full/processed/pdbbind_pl_manifest.tsv}
SPLIT_FILE=${KDBNET_SPLIT_FILE:-data/PDBbind_full/processed/pdbbind_pl_fixed_split_seed42.tsv}
OUT_DIR=${KDBNET_OUT_DIR:-./output/PDBbind_runs/pdbbind_fixed_wsl_seed${SEED}_${STAMP}}

if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi
if [ ! -f "$SPLIT_FILE" ]; then
  echo "Missing fixed split file: $SPLIT_FILE" >&2
  echo "Create it first with: $PYTHON scripts/create_pdbbind_fixed_split.py --output $SPLIT_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Starting one fixed-split PDBbind run in WSL"
echo "Python: $PYTHON"
echo "Manifest: $MANIFEST"
echo "Split file: $SPLIT_FILE"
echo "Output: $OUT_DIR"
echo "Epochs: $N_EPOCHS"
echo "Patience: $PATIENCE"
echo "Batch size: $BATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES=$GPU_ID"
if [ "$#" -gt 0 ]; then
  echo "Extra run_example.py args: $*"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/run_example.py \
  --task pdbbind_pl \
  --pdbbind_manifest "$MANIFEST" \
  --split_method fixed \
  --split_file "$SPLIT_FILE" \
  --seed "$SEED" \
  --n_epochs "$N_EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --monitor_metric mse \
  --output_dir "$OUT_DIR" \
  --save_prediction \
  --save_log \
  --save_checkpoint \
  "$@"

echo "Finished fixed-split PDBbind run: $OUT_DIR"
