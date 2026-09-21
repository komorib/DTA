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
LMDB_DIR=${KDBNET_LMDB_DIR:-data/PDBbind_chain_lmdb_esm2}
INDEX_PATH=${KDBNET_LMDB_INDEX:-${LMDB_DIR}/index.pkl}
STRUCTURE_ROOT=${KDBNET_STRUCTURE_ROOT:-data/PDBbind_full/extracted}
SPLIT_FILE=${KDBNET_SPLIT_FILE:-pdbbind_pl_temporal_split_wangzhiwen_15267.tsv}
ESM_KEY=${KDBNET_LMDB_ESM_KEY:-esm2_embedding}
PROT_EMB_DIM=${KDBNET_PROT_EMB_DIM:-1280}
LIGAND_SOURCE=${KDBNET_LIGAND_SOURCE:-sdf}
LIGAND_UNIMOL_LMDB_DIR=${KDBNET_LIGAND_UNIMOL_LMDB_DIR:-}
LIGAND_UNIMOL_INDEX=${KDBNET_LIGAND_UNIMOL_INDEX:-}
LIGAND_EMB_KEY=${KDBNET_LIGAND_EMB_KEY:-atom_emb}
LIGAND_EMB_DIM=${KDBNET_LIGAND_EMB_DIM:-768}
LIGAND_ATOM_SYMBOL_CHECK=${KDBNET_LIGAND_ATOM_SYMBOL_CHECK:-1}
LIGAND_VALIDATION_CACHE=${KDBNET_LIGAND_VALIDATION_CACHE:-}
REBUILD_LIGAND_VALIDATION_CACHE=${KDBNET_REBUILD_LIGAND_VALIDATION_CACHE:-0}
USE_POCKET_ATOM=${KDBNET_USE_POCKET_ATOM:-0}
POCKET_ATOM_CUTOFF=${KDBNET_POCKET_ATOM_CUTOFF:-5.0}
POCKET_ATOM_EDGE_CUTOFF=${KDBNET_POCKET_ATOM_EDGE_CUTOFF:-4.5}
POCKET_ATOM_INCLUDE_HYDROGEN=${KDBNET_POCKET_ATOM_INCLUDE_HYDROGEN:-0}
POCKET_ATOM_INCLUDE_HETATM=${KDBNET_POCKET_ATOM_INCLUDE_HETATM:-0}
USE_ONE_D_TWO_D=${KDBNET_USE_ONE_D_TWO_D:-0}
ONE_D_TWO_D_PROTEIN_EDGE_CUTOFF=${KDBNET_ONE_D_TWO_D_PROTEIN_EDGE_CUTOFF:-8.0}
ONE_D_TWO_D_HIDDEN_DIM=${KDBNET_ONE_D_TWO_D_HIDDEN_DIM:-64}
ONE_D_TWO_D_OUTPUT_DIM=${KDBNET_ONE_D_TWO_D_OUTPUT_DIM:-128}
ONE_D_TWO_D_GNN_LAYERS=${KDBNET_ONE_D_TWO_D_GNN_LAYERS:-3}
ONE_D_TWO_D_TRIANGLE_LAYERS=${KDBNET_ONE_D_TWO_D_TRIANGLE_LAYERS:-2}
ONE_D_TWO_D_DROPOUT=${KDBNET_ONE_D_TWO_D_DROPOUT:-0.25}
ONE_D_TWO_D_USE_PROTEIN_2D=${KDBNET_ONE_D_TWO_D_USE_PROTEIN_2D:-1}
ONE_D_TWO_D_MAX_PROTEIN_RESIDUES=${KDBNET_ONE_D_TWO_D_MAX_PROTEIN_RESIDUES:-128}
USE_INTERFACE_PAIRFORMER=${KDBNET_USE_INTERFACE_PAIRFORMER:-1}
INTERFACE_PAIRFORMER_SINGLE_DIM=${KDBNET_INTERFACE_PAIRFORMER_SINGLE_DIM:-128}
INTERFACE_PAIRFORMER_PAIR_DIM=${KDBNET_INTERFACE_PAIRFORMER_PAIR_DIM:-64}
INTERFACE_PAIRFORMER_OUTPUT_DIM=${KDBNET_INTERFACE_PAIRFORMER_OUTPUT_DIM:-128}
INTERFACE_PAIRFORMER_BLOCKS=${KDBNET_INTERFACE_PAIRFORMER_BLOCKS:-4}
INTERFACE_PAIRFORMER_HEADS=${KDBNET_INTERFACE_PAIRFORMER_HEADS:-4}
INTERFACE_PAIRFORMER_NUM_RBF=${KDBNET_INTERFACE_PAIRFORMER_NUM_RBF:-32}
INTERFACE_PAIRFORMER_FINE_NUM_RBF=${KDBNET_INTERFACE_PAIRFORMER_FINE_NUM_RBF:-32}
INTERFACE_PAIRFORMER_MAX_POCKET=${KDBNET_INTERFACE_PAIRFORMER_MAX_POCKET:-64}
INTERFACE_PAIRFORMER_MAX_LIGAND=${KDBNET_INTERFACE_PAIRFORMER_MAX_LIGAND:-64}
INTERFACE_PAIRFORMER_MAX_RESIDUE_ATOMS=${KDBNET_INTERFACE_PAIRFORMER_MAX_RESIDUE_ATOMS:-14}
INTERFACE_PAIRFORMER_POCKET_RADIUS=${KDBNET_INTERFACE_PAIRFORMER_POCKET_RADIUS:-10.0}
INTERFACE_PAIRFORMER_MIN_POCKET=${KDBNET_INTERFACE_PAIRFORMER_MIN_POCKET:-16}
INTERFACE_PAIRFORMER_DISTANCE_MAX=${KDBNET_INTERFACE_PAIRFORMER_DISTANCE_MAX:-20.0}
INTERFACE_PAIRFORMER_FINE_DISTANCE_MAX=${KDBNET_INTERFACE_PAIRFORMER_FINE_DISTANCE_MAX:-8.0}
INTERFACE_PAIRFORMER_DROPOUT=${KDBNET_INTERFACE_PAIRFORMER_DROPOUT:-0.15}
INTERFACE_PAIRFORMER_GATE_HIDDEN=${KDBNET_INTERFACE_PAIRFORMER_GATE_HIDDEN:-128}
SPLIT_MISSING_POLICY=${KDBNET_SPLIT_MISSING_POLICY:-intersection}
RUN_TAG=${KDBNET_RUN_TAG:-pdbbind_lmdb_temporal15267_seed${SEED}_gpu${GPU_ID}_${STAMP}}
OUT_DIR=${KDBNET_OUT_DIR:-./output/PDBbind_runs/${RUN_TAG}}

if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 1
fi
if [ ! -d "$LMDB_DIR" ]; then
  echo "Missing LMDB directory: $LMDB_DIR" >&2
  exit 1
fi
if [ ! -f "$INDEX_PATH" ]; then
  echo "Missing LMDB index: $INDEX_PATH" >&2
  exit 1
fi
if [ ! -f "$SPLIT_FILE" ]; then
  echo "Missing fixed split file: $SPLIT_FILE" >&2
  exit 1
fi
if [ -n "$LIGAND_UNIMOL_LMDB_DIR" ] && [ ! -d "$LIGAND_UNIMOL_LMDB_DIR" ]; then
  echo "Missing ligand Uni-Mol2 LMDB directory: $LIGAND_UNIMOL_LMDB_DIR" >&2
  exit 1
fi
if [ -n "$LIGAND_UNIMOL_LMDB_DIR" ]; then
  if [ -z "$LIGAND_UNIMOL_INDEX" ]; then
    LIGAND_UNIMOL_INDEX=${LIGAND_UNIMOL_LMDB_DIR}/index.pkl
  fi
  if [ ! -f "$LIGAND_UNIMOL_INDEX" ]; then
    echo "Missing ligand Uni-Mol2 index: $LIGAND_UNIMOL_INDEX" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"

LIGAND_ARGS=()
if [ -n "$LIGAND_UNIMOL_LMDB_DIR" ]; then
  LIGAND_ARGS+=(
    --ligand_unimol_lmdb_dir "$LIGAND_UNIMOL_LMDB_DIR"
    --ligand_unimol_index "$LIGAND_UNIMOL_INDEX"
    --ligand_emb_key "$LIGAND_EMB_KEY"
    --ligand_emb_dim "$LIGAND_EMB_DIM"
  )
  if [ -n "$LIGAND_VALIDATION_CACHE" ]; then
    LIGAND_ARGS+=(--ligand_validation_cache "$LIGAND_VALIDATION_CACHE")
  fi
  if [ "$REBUILD_LIGAND_VALIDATION_CACHE" = "1" ]; then
    LIGAND_ARGS+=(--rebuild_ligand_validation_cache)
  fi
  if [ "$LIGAND_ATOM_SYMBOL_CHECK" = "0" ]; then
    LIGAND_ARGS+=(--no_ligand_atom_symbol_check)
  fi
fi

POCKET_ATOM_ARGS=()
if [ "$USE_POCKET_ATOM" = "1" ]; then
  POCKET_ATOM_ARGS+=(
    --use_pocket_atom
    --pocket_atom_cutoff "$POCKET_ATOM_CUTOFF"
    --pocket_atom_edge_cutoff "$POCKET_ATOM_EDGE_CUTOFF"
  )
  if [ "$POCKET_ATOM_INCLUDE_HYDROGEN" = "1" ]; then
    POCKET_ATOM_ARGS+=(--pocket_atom_include_hydrogen)
  fi
  if [ "$POCKET_ATOM_INCLUDE_HETATM" = "1" ]; then
    POCKET_ATOM_ARGS+=(--pocket_atom_include_hetatm)
  fi
fi

ONE_D_TWO_D_ARGS=()
if [ "$USE_ONE_D_TWO_D" = "1" ]; then
  ONE_D_TWO_D_ARGS+=(
    --use_one_d_two_d
    --one_d_two_d_protein_edge_cutoff "$ONE_D_TWO_D_PROTEIN_EDGE_CUTOFF"
    --one_d_two_d_max_protein_residues "$ONE_D_TWO_D_MAX_PROTEIN_RESIDUES"
    --one_d_two_d_hidden_dim "$ONE_D_TWO_D_HIDDEN_DIM"
    --one_d_two_d_output_dim "$ONE_D_TWO_D_OUTPUT_DIM"
    --one_d_two_d_gnn_layers "$ONE_D_TWO_D_GNN_LAYERS"
    --one_d_two_d_triangle_layers "$ONE_D_TWO_D_TRIANGLE_LAYERS"
    --one_d_two_d_dropout "$ONE_D_TWO_D_DROPOUT"
  )
  if [ "$ONE_D_TWO_D_USE_PROTEIN_2D" = "0" ]; then
    ONE_D_TWO_D_ARGS+=(--no_one_d_two_d_protein_2d)
  fi
fi

INTERFACE_PAIRFORMER_ARGS=()
if [ "$USE_INTERFACE_PAIRFORMER" = "1" ]; then
  INTERFACE_PAIRFORMER_ARGS+=(
    --use_interface_pairformer
    --interface_pairformer_single_dim "$INTERFACE_PAIRFORMER_SINGLE_DIM"
    --interface_pairformer_pair_dim "$INTERFACE_PAIRFORMER_PAIR_DIM"
    --interface_pairformer_output_dim "$INTERFACE_PAIRFORMER_OUTPUT_DIM"
    --interface_pairformer_blocks "$INTERFACE_PAIRFORMER_BLOCKS"
    --interface_pairformer_heads "$INTERFACE_PAIRFORMER_HEADS"
    --interface_pairformer_num_rbf "$INTERFACE_PAIRFORMER_NUM_RBF"
    --interface_pairformer_fine_num_rbf "$INTERFACE_PAIRFORMER_FINE_NUM_RBF"
    --interface_pairformer_max_pocket_residues "$INTERFACE_PAIRFORMER_MAX_POCKET"
    --interface_pairformer_max_ligand_atoms "$INTERFACE_PAIRFORMER_MAX_LIGAND"
    --interface_pairformer_max_residue_atoms "$INTERFACE_PAIRFORMER_MAX_RESIDUE_ATOMS"
    --interface_pairformer_pocket_radius "$INTERFACE_PAIRFORMER_POCKET_RADIUS"
    --interface_pairformer_min_pocket_residues "$INTERFACE_PAIRFORMER_MIN_POCKET"
    --interface_pairformer_distance_max "$INTERFACE_PAIRFORMER_DISTANCE_MAX"
    --interface_pairformer_fine_distance_max "$INTERFACE_PAIRFORMER_FINE_DISTANCE_MAX"
    --interface_pairformer_dropout "$INTERFACE_PAIRFORMER_DROPOUT"
    --interface_pairformer_gate_hidden_dim "$INTERFACE_PAIRFORMER_GATE_HIDDEN"
  )
fi

echo "Starting one fixed-split PDBbind LMDB run on server"
echo "GPU: $GPU_ID"
echo "Python: $PYTHON"
echo "LMDB: $LMDB_DIR"
echo "Index: $INDEX_PATH"
echo "Structure root: $STRUCTURE_ROOT"
echo "Split file: $SPLIT_FILE"
echo "ESM key: $ESM_KEY"
echo "Protein embedding dim: $PROT_EMB_DIM"
echo "Ligand source: $LIGAND_SOURCE"
if [ -n "$LIGAND_UNIMOL_LMDB_DIR" ]; then
  echo "Ligand Uni-Mol2 LMDB: $LIGAND_UNIMOL_LMDB_DIR"
  echo "Ligand Uni-Mol2 index: $LIGAND_UNIMOL_INDEX"
  echo "Ligand embedding key: $LIGAND_EMB_KEY"
  echo "Ligand embedding dim: $LIGAND_EMB_DIM"
  echo "Ligand fusion: hidden_scalar_add"
  echo "Ligand atom symbol check: $LIGAND_ATOM_SYMBOL_CHECK"
  echo "Ligand validation cache: ${LIGAND_VALIDATION_CACHE:-auto}"
  echo "Rebuild ligand validation cache: $REBUILD_LIGAND_VALIDATION_CACHE"
else
  echo "Ligand Uni-Mol2 LMDB: disabled"
fi
echo "Split missing policy: $SPLIT_MISSING_POLICY"
echo "Use pocket atom branch: $USE_POCKET_ATOM"
if [ "$USE_POCKET_ATOM" = "1" ]; then
  echo "Pocket atom cutoff: $POCKET_ATOM_CUTOFF"
  echo "Pocket atom edge cutoff: $POCKET_ATOM_EDGE_CUTOFF"
  echo "Pocket atom include hydrogen: $POCKET_ATOM_INCLUDE_HYDROGEN"
  echo "Pocket atom include HETATM: $POCKET_ATOM_INCLUDE_HETATM"
fi
echo "Use 1D2D branch: $USE_ONE_D_TWO_D"
if [ "$USE_ONE_D_TWO_D" = "1" ]; then
  echo "1D2D protein edge cutoff: $ONE_D_TWO_D_PROTEIN_EDGE_CUTOFF"
  echo "1D2D max protein residues: $ONE_D_TWO_D_MAX_PROTEIN_RESIDUES"
  echo "1D2D hidden dim: $ONE_D_TWO_D_HIDDEN_DIM"
  echo "1D2D output dim: $ONE_D_TWO_D_OUTPUT_DIM"
  echo "1D2D GNN layers: $ONE_D_TWO_D_GNN_LAYERS"
  echo "1D2D triangle layers: $ONE_D_TWO_D_TRIANGLE_LAYERS"
  echo "1D2D dropout: $ONE_D_TWO_D_DROPOUT"
  echo "1D2D use protein 2D: $ONE_D_TWO_D_USE_PROTEIN_2D"
fi
echo "Use interface Pairformer: $USE_INTERFACE_PAIRFORMER"
if [ "$USE_INTERFACE_PAIRFORMER" = "1" ]; then
  echo "Interface Pairformer dims: single=$INTERFACE_PAIRFORMER_SINGLE_DIM pair=$INTERFACE_PAIRFORMER_PAIR_DIM output=$INTERFACE_PAIRFORMER_OUTPUT_DIM"
  echo "Interface Pairformer blocks/heads: $INTERFACE_PAIRFORMER_BLOCKS/$INTERFACE_PAIRFORMER_HEADS"
  echo "Interface Pairformer caps: pocket=$INTERFACE_PAIRFORMER_MAX_POCKET ligand=$INTERFACE_PAIRFORMER_MAX_LIGAND residue_atoms=$INTERFACE_PAIRFORMER_MAX_RESIDUE_ATOMS"
  echo "Interface Pairformer RBF: broad=$INTERFACE_PAIRFORMER_NUM_RBF/$INTERFACE_PAIRFORMER_DISTANCE_MAX fine=$INTERFACE_PAIRFORMER_FINE_NUM_RBF/$INTERFACE_PAIRFORMER_FINE_DISTANCE_MAX"
  echo "Interface Pairformer radius/min pocket: $INTERFACE_PAIRFORMER_POCKET_RADIUS/$INTERFACE_PAIRFORMER_MIN_POCKET"
fi
echo "Output: $OUT_DIR"
echo "Epochs: $N_EPOCHS"
echo "Patience: $PATIENCE"
echo "Batch size: $BATCH_SIZE"
if [ "$#" -gt 0 ]; then
  echo "Extra run_pdbbind_lmdb.py args: $*"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/run_pdbbind_lmdb.py \
  --lmdb_dir "$LMDB_DIR" \
  --index_path "$INDEX_PATH" \
  --structure_root "$STRUCTURE_ROOT" \
  --split_file "$SPLIT_FILE" \
  --split_method fixed \
  --split_missing_policy "$SPLIT_MISSING_POLICY" \
  --seed "$SEED" \
  --esm_key "$ESM_KEY" \
  --prot_emb_dim "$PROT_EMB_DIM" \
  --ligand_source "$LIGAND_SOURCE" \
  ${LIGAND_ARGS[@]+"${LIGAND_ARGS[@]}"} \
  ${POCKET_ATOM_ARGS[@]+"${POCKET_ATOM_ARGS[@]}"} \
  ${ONE_D_TWO_D_ARGS[@]+"${ONE_D_TWO_D_ARGS[@]}"} \
  ${INTERFACE_PAIRFORMER_ARGS[@]+"${INTERFACE_PAIRFORMER_ARGS[@]}"} \
  --n_epochs "$N_EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --monitor_metric mse \
  --output_dir "$OUT_DIR" \
  --save_prediction \
  --save_log \
  --save_checkpoint \
  "$@"

echo "Finished fixed-split PDBbind LMDB run: $OUT_DIR"
