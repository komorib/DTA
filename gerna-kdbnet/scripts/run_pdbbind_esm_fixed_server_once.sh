#!/usr/bin/env bash
set -euo pipefail

cd /home/user/yangyuhang/KDBNet

GPU_ID=${1:-${KDBNET_GPU_ID:-1}}
if [ "$#" -gt 0 ]; then
  shift
fi

SEED=${KDBNET_SEED:-42}
STAMP=${KDBNET_STAMP:-$(date +%Y%m%d_%H%M%S)}
N_EPOCHS=${KDBNET_N_EPOCHS:-100}
PATIENCE=${KDBNET_PATIENCE:-20}
BATCH_SIZE=${KDBNET_BATCH_SIZE:-32}

PYTHON=${KDBNET_PYTHON:-/home/user/yangyuhang/miniconda3/envs/kdb_env/bin/python}
MANIFEST=${KDBNET_MANIFEST:-data/PDBbind_full/processed/pdbbind_pl_manifest_esm.tsv}
SPLIT_FILE=${KDBNET_SPLIT_FILE:-data/PDBbind_full/processed/pdbbind_pl_fixed_split_seed42.tsv}
RUN_TAG=${KDBNET_RUN_TAG:-pdbbind_esm_fixed_seed${SEED}_gpu${GPU_ID}_${STAMP}}
OUT_DIR=${KDBNET_OUT_DIR:-./output/PDBbind_runs/${RUN_TAG}}

if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "Missing ESM manifest: $MANIFEST" >&2
  exit 1
fi
if [ ! -f "$SPLIT_FILE" ]; then
  echo "Missing fixed split file: $SPLIT_FILE" >&2
  echo "Create it first with:" >&2
  echo "  $PYTHON scripts/create_pdbbind_fixed_split.py --manifest $MANIFEST --output $SPLIT_FILE --seed 42" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Starting one fixed-split PDBbind ESM1b run on server"
echo "GPU: $GPU_ID"
echo "Python: $PYTHON"
echo "Manifest: $MANIFEST"
echo "Split file: $SPLIT_FILE"
echo "Output: $OUT_DIR"
echo "Epochs: $N_EPOCHS"
echo "Patience: $PATIENCE"
echo "Batch size: $BATCH_SIZE"
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

echo "Finished fixed-split PDBbind ESM1b run: $OUT_DIR"
