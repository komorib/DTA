# Pocket–Ligand Interface Pairformer v2

## Purpose

This branch augments the original global KDBNet representation with a local,
structure-sensitive pocket–ligand interface model.  It is designed for affinity
prediction under crystal, perturbed, docking, and predicted-receptor inputs.

The final mean prediction uses gated local correction:

```text
mu = mu_global + sigmoid(gate(global, local)) * delta_mu_local
```

When heteroscedastic mode is enabled, the log-variance head reads the global
representation, local interface representation, and gate value.

## Coordinate representation

The v2 module does **not** concatenate absolute xyz coordinates to token
features.  That would make predictions depend on the arbitrary global frame.
Instead it uses invariant/equivariant-derived geometry:

1. broad C-alpha–ligand distance RBF (default 0–20 A, 32 bases);
2. fine C-alpha–ligand distance RBF (default 0–8 A, 32 bases);
3. N-, C-, O-, C-beta-, and side-chain-centre-to-ligand distances;
4. exact minimum distance between each ligand atom and all available heavy
   atoms of a residue;
5. ligand direction and displacement expressed in the residue-local N–CA–C
   orthonormal frame; and
6. side-chain-direction/ligand-direction alignment.

This makes the representation invariant to a joint rigid translation or
rotation, while retaining sensitivity to sub-angstrom relative changes.

## Protein atom availability

The original protein LMDB stores only N, CA, C, and O.  The revised pipeline:

- always stores backbone coordinates, a local frame, and a pseudo-CB fallback
  on the protein graph;
- when a PDBbind `*_pocket.pdb` path is available, parses its real heavy atoms;
- matches pocket residues to full-protein graph nodes by C-alpha coordinates;
- stores up to 14 heavy atoms per residue, the real C-beta when available, and
  the real side-chain centroid; and
- falls back safely to backbone + pseudo-CB if a residue cannot be matched.

The graph attribute `interface_full_atom_residue_count` records how many pocket
residues were matched for each complex.  It is useful for a preprocessing audit.

## Main files

- `kdbnet/interface_pairformer.py`: rectangular pocket-residue x ligand-atom
  Pairformer and multi-atom geometry encoder.
- `kdbnet/protein_geometry.py`: pocket PDB parsing and full-atom residue mapping.
- `kdbnet/pdb_graph.py`: backbone/local-frame/pseudo-CB node attributes.
- `kdbnet/pdbbind_lmdb.py`: attaches real pocket atoms and fixes the crystal
  pocket residue mask.
- `kdbnet/model.py`: gated local correction and uncertainty fusion.
- `scripts/smoke_test_interface_pairformer.py`: geometry invariance and
  sensitivity checks.

## New command-line settings

```bash
--interface_pairformer_fine_num_rbf 32
--interface_pairformer_fine_distance_max 8.0
--interface_pairformer_max_residue_atoms 14
```

The existing broad settings remain:

```bash
--interface_pairformer_num_rbf 32
--interface_pairformer_distance_max 20.0
```

Recommended initial model:

```bash
python scripts/run_pdbbind_lmdb.py \
  --lmdb_dir <LMDB_DIR> \
  --index_path <INDEX_PKL> \
  --structure_root <PDBBIND_ROOT> \
  --split_file <LP_PDBBIND_SPLIT> \
  --split_method fixed \
  --use_interface_pairformer \
  --interface_pairformer_blocks 4 \
  --interface_pairformer_single_dim 128 \
  --interface_pairformer_pair_dim 64 \
  --interface_pairformer_output_dim 128 \
  --interface_pairformer_max_pocket_residues 64 \
  --interface_pairformer_max_ligand_atoms 64 \
  --interface_pairformer_max_residue_atoms 14 \
  --interface_pairformer_num_rbf 32 \
  --interface_pairformer_distance_max 20.0 \
  --interface_pairformer_fine_num_rbf 32 \
  --interface_pairformer_fine_distance_max 8.0 \
  --heteroscedastic \
  --uncertainty \
  --n_ensembles 5
```

## Validation performed

The dependency-light smoke test verifies:

- forward and backward propagation;
- variable-length pocket/ligand batching;
- joint rotation and translation invariance;
- response to a 0.05 A ligand shift; and
- response to a 0.10 A side-chain-only shift.

Run:

```bash
python scripts/smoke_test_interface_pairformer.py
```

## Important boundary

This upgrade makes the model capable of reading subtle coordinate changes.  It
does not yet generate noisy/docked/AF structural views.  Those views must still
be added to the training/evaluation data pipeline, with all views of one
protein–ligand pair sharing the same experimental affinity label.
