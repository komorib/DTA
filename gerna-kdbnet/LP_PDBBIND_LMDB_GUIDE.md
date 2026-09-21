# LP-PDBbind → KDBNet LMDB

This pipeline creates the LMDB schema consumed by `scripts/run_pdbbind_lmdb.py`.
It preserves the supplied LP-PDBbind split exactly and rejects any base complex
that appears in more than one canonical partition.

## 1. Supported LP split inputs

`--split_source` accepts:

- TSV/CSV with ID aliases such as `complex_id`, `pdbid`, `pdb_id` and a
  `split`/`set` column;
- a directory containing `train*.txt`, `valid*.txt`/`val*.txt`, and
  `test*.txt` files, one PDB ID per line;
- pickle/JSON objects such as `{"train": [...], "valid": [...], "test": [...]}`.

Tiered labels such as `test_cl1` remain in `split_detail` while the runtime
canonical split is `test`.

## 2. Expected structures

The script recursively discovers:

```text
<PDBID>_protein.pdb   # preferred for the global protein branch
<PDBID>_pocket.pdb    # required for full-atom pocket enrichment
<PDBID>_ligand.sdf    # preferred
<PDBID>_ligand.mol2   # fallback; converted to SDF with --normalise_ligands
```

`--protein_source auto` uses the full protein PDB and falls back to pocket PDB.
The LMDB stores chain sequence, N/CA/C/O coordinates, residue identifiers and
residue-level ESM embeddings. The runtime Pairformer reads real side-chain
heavy atoms on demand from `_pocket.pdb`.

## 3. First validate without writing LMDB

This does not require `py-lmdb`:

```bash
python scripts/build_lp_pdbbind_lmdb.py \
  --structure_root /path/to/LP-PDBbind/structures \
  --split_source /path/to/LP-PDBbind/splits \
  --affinity_source /path/to/INDEX_general_PL.2020R1.lst \
  --output_lmdb data/lp_pdbbind_dryrun \
  --esm_mode zero \
  --limit 100 \
  --dry_run --overwrite
```

Inspect `build_report.json` before the full run.

## 4. Final build with ESM-2

Install dependencies:

```bash
pip install lmdb rdkit biopython pandas tqdm fair-esm
```

Using a local ESM-2 checkpoint avoids compute-node downloads:

```bash
python scripts/build_lp_pdbbind_lmdb.py \
  --structure_root /path/to/LP-PDBbind/structures \
  --split_source /path/to/LP-PDBbind/splits \
  --affinity_source /path/to/INDEX_general_PL.2020R1.lst \
  --output_lmdb data/lp_pdbbind_lmdb \
  --protein_source auto \
  --esm_mode generate \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt \
  --esm_repr_layer 33 \
  --esm_device cuda \
  --embedding_key esm2_embedding \
  --embedding_dim 1280 \
  --embedding_dtype float16 \
  --normalise_ligands \
  --map_size_gb 64 \
  --overwrite
```

For precomputed per-complex embeddings:

```bash
--esm_mode precomputed --embedding_dir /path/to/embeddings
```

Supported artifacts include chain dictionaries or a concatenated `[L,D]`
tensor saved as `<complex_id>.pt/.npy/.pkl`.

## 5. Audit

```bash
python scripts/audit_lp_pdbbind_lmdb.py \
  --lmdb_dir data/lp_pdbbind_lmdb \
  --embedding_key esm2_embedding \
  --embedding_dim 1280 \
  --check_paths
```

A non-zero exit code means the LMDB should not be used for training.

## 6. Train the Pairformer model

```bash
python scripts/run_pdbbind_lmdb.py \
  --lmdb_dir data/lp_pdbbind_lmdb \
  --index_path data/lp_pdbbind_lmdb/index.pkl \
  --split_file data/lp_pdbbind_lmdb/split.tsv \
  --structure_root /path/to/LP-PDBbind/structures \
  --ligand_source lmdb_mol \
  --esm_key esm2_embedding \
  --prot_emb_dim 1280 \
  --use_interface_pairformer \
  --heteroscedastic \
  --uncertainty \
  --n_ensembles 5 \
  --nll_warmup_epochs 10 \
  --output_dir output/lp_pairformer \
  --save_log --save_checkpoint --save_prediction
```

## Output schema

Each LMDB key is `<complex_id>_protein`. `index.pkl` contains the regression
label under `affinity` and the exact accepted split. Additional metadata is
written to `index.tsv`, `schema.json` and `build_report.json`.
