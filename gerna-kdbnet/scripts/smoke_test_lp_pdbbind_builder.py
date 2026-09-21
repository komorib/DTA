#!/usr/bin/env python
"""Dependency-light smoke test for LP-PDBbind preprocessing.

It validates split parsing, affinity conversion, structure discovery, PDB
backbone parsing, ligand parsing and in-memory record construction. It does not
require py-lmdb; the full writer is covered by the same record-building path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem

from kdbnet.lp_pdbbind_lmdb import (
    build_complex_record,
    discover_structure_files,
    load_affinity_table,
    load_lp_split,
    merge_split_and_affinity,
)


def write_pdb(path: Path):
    lines = []
    serial = 1
    for residue_index in (1, 2):
        base = (residue_index - 1) * 3.8
        atoms = [
            ("N", (base, 0.0, 0.0)),
            ("CA", (base + 1.45, 0.2, 0.0)),
            ("C", (base + 2.8, 0.0, 0.1)),
            ("O", (base + 3.3, -0.8, 0.1)),
            ("CB", (base + 1.45, 1.5, 0.2)),
        ]
        for atom, (x, y, z) in atoms:
            lines.append(
                "ATOM  {serial:5d} {atom:^4s} ALA A{resi:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {elem:>2s}\n".format(
                    serial=serial, atom=atom, resi=residue_index,
                    x=x, y=y, z=z, elem=atom[0]
                )
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        structure_root = root / "structures"
        split_root = root / "splits"
        structure_root.mkdir()
        split_root.mkdir()
        ids = ("1aaa", "2bbb", "3ccc")
        for seed, cid in enumerate(ids):
            complex_dir = structure_root / cid
            complex_dir.mkdir()
            write_pdb(complex_dir / (cid + "_protein.pdb"))
            write_pdb(complex_dir / (cid + "_pocket.pdb"))
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=seed + 1)
            mol = Chem.RemoveHs(mol)
            writer = Chem.SDWriter(str(complex_dir / (cid + "_ligand.sdf")))
            writer.write(mol)
            writer.close()
        (split_root / "train.txt").write_text("1aaa\n")
        (split_root / "valid.txt").write_text("2bbb\n")
        (split_root / "test.txt").write_text("3ccc\n")
        affinity_path = root / "INDEX_general_PL.2020R1.lst"
        affinity_path.write_text(
            "1aaa 2.0 2020 Kd=10nM //\n"
            "2bbb 2.0 2020 Ki=100nM //\n"
            "3ccc 2.0 2020 IC50=1uM //\n"
        )

        split = load_lp_split(split_root)
        affinity = load_affinity_table(affinity_path)
        merged = merge_split_and_affinity(split, affinity)
        structures = discover_structure_files(structure_root)
        assert len(merged) == 3
        assert set(merged["split"]) == {"train", "valid", "test"}
        assert set(round(value, 5) for value in merged["affinity"]) == {6.0, 7.0, 8.0}

        for row in merged.to_dict("records"):
            cid = row["complex_id"]
            entry, index_row = build_complex_record(
                row=row,
                files=structures[cid],
                structure_root=structure_root,
                protein_source="auto",
                esm_mode="zero",
                embedding_key="esm2_embedding",
                embedding_dim=1280,
                embedding_dtype="float16",
                embedder=None,
                embedding_dir=None,
                path_mode="absolute",
                normalised_ligand_dir=None,
                store_ligand_mol=True,
            )
            assert index_row["lmdb_key"] == cid + "_protein"
            assert entry["ligand_features"]["mol"].GetNumConformers() == 1
            chain = next(iter(entry["protein_features"].values()))
            assert chain["backbone_coords"].shape == (2, 4, 3)
            assert chain["esm2_embedding"].shape == (2, 1280)

        # Verify split overlap fails loudly.
        (split_root / "test.txt").write_text("1aaa\n3ccc\n")
        try:
            load_lp_split(split_root)
        except ValueError as exc:
            assert "overlap" in str(exc).lower()
        else:
            raise AssertionError("overlapping split was not rejected")

    print("LP-PDBbind builder smoke test: PASS")


if __name__ == "__main__":
    main()
