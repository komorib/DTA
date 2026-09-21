#!/usr/bin/env python
"""Audit ligand-proximal protein atoms in PDBbind pocket PDB files.

This is a read-only diagnostic script. It parses each ligand SDF and pocket PDB,
then counts protein atoms whose distance to any ligand atom is within a cutoff.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def resolve_path(path_value, structure_root=None):
    path = Path(str(path_value))
    if path.exists():
        return path

    roots = []
    if structure_root is not None:
        roots.append(Path(structure_root))
    roots.append(Path("."))

    for root in roots:
        candidates = [root / path]
        if len(path.parts) >= 2:
            complex_dir = path.parts[-2]
            filename = path.name
            candidates.extend([
                root / complex_dir / filename,
                root / "P-L" / complex_dir / filename,
            ])
        for candidate in candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError("Could not resolve path: {}".format(path_value))


def infer_pdb_element(atom_name, line):
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    for char in atom_name.strip():
        if char.isalpha():
            return char.upper()
    return ""


def parse_pocket_atoms(pdb_path, include_hydrogen=False, include_hetatm=False):
    atoms = []
    with open(pdb_path, "r") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "ATOM":
                pass
            elif record == "HETATM" and include_hetatm:
                pass
            else:
                continue

            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue

            atom_name = line[12:16].strip()
            element = infer_pdb_element(atom_name, line)
            if not include_hydrogen and element == "H":
                continue

            try:
                coord = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue

            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_key = "{}:{}:{}:{}".format(chain, resseq, icode, resname)
            atoms.append({
                "atom_name": atom_name,
                "element": element,
                "coord": coord,
                "residue_key": residue_key,
                "is_backbone": atom_name in BACKBONE_ATOMS,
            })
    return atoms


def load_ligand_coords(sdf_path, include_hydrogen=False):
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=not include_hydrogen)
    if mol is None:
        raise ValueError("RDKit failed to parse ligand SDF: {}".format(sdf_path))
    conf = mol.GetConformer()
    coords = []
    elements = []
    for atom in mol.GetAtoms():
        if not include_hydrogen and atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        elements.append(atom.GetSymbol())
    if not coords:
        raise ValueError("Ligand has no usable atoms: {}".format(sdf_path))
    return np.asarray(coords, dtype=np.float32), elements


def summarise_numeric(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {"count": 0}
    quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "max": float(values.max()),
    }


def audit_row(row, args):
    complex_id = str(row["complex_id"]).lower()
    ligand_path = resolve_path(row["ligand_sdf"], args.structure_root)
    pocket_path = resolve_path(row["pocket_pdb"], args.structure_root)

    ligand_coords, ligand_elements = load_ligand_coords(
        ligand_path,
        include_hydrogen=args.include_hydrogen,
    )
    pocket_atoms = parse_pocket_atoms(
        pocket_path,
        include_hydrogen=args.include_hydrogen,
        include_hetatm=args.include_hetatm,
    )
    if not pocket_atoms:
        raise ValueError("Pocket PDB has no usable atoms: {}".format(pocket_path))

    protein_coords = np.stack([atom["coord"] for atom in pocket_atoms], axis=0)
    dists = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :],
        axis=-1,
    )
    min_dists = dists.min(axis=1)
    contact_mask = min_dists <= args.cutoff
    contact_atoms = [
        (atom, float(min_dist))
        for atom, min_dist, is_contact in zip(pocket_atoms, min_dists, contact_mask)
        if is_contact
    ]

    pocket_residues = {atom["residue_key"] for atom in pocket_atoms}
    contact_residues = {atom["residue_key"] for atom, _ in contact_atoms}
    contact_elements = Counter(atom["element"] for atom, _ in contact_atoms)
    contact_atom_names = Counter(atom["atom_name"] for atom, _ in contact_atoms)
    contact_backbone = sum(1 for atom, _ in contact_atoms if atom["is_backbone"])
    contact_sidechain = len(contact_atoms) - contact_backbone
    contact_distances = [dist for _, dist in contact_atoms]

    return {
        "complex_id": complex_id,
        "split": row.get("split", ""),
        "ligand_sdf": str(ligand_path),
        "pocket_pdb": str(pocket_path),
        "ligand_atoms": int(len(ligand_coords)),
        "pocket_atoms": int(len(pocket_atoms)),
        "pocket_residues": int(len(pocket_residues)),
        "contact_atoms": int(len(contact_atoms)),
        "contact_residues": int(len(contact_residues)),
        "contact_backbone_atoms": int(contact_backbone),
        "contact_sidechain_atoms": int(contact_sidechain),
        "contact_atom_fraction": (
            float(len(contact_atoms) / len(pocket_atoms)) if pocket_atoms else 0.0
        ),
        "contact_residue_fraction": (
            float(len(contact_residues) / len(pocket_residues)) if pocket_residues else 0.0
        ),
        "min_protein_ligand_distance": float(min_dists.min()),
        "mean_contact_distance": (
            float(np.mean(contact_distances)) if contact_distances else np.nan
        ),
        "max_contact_distance": (
            float(np.max(contact_distances)) if contact_distances else np.nan
        ),
        "contact_elements_json": json.dumps(dict(contact_elements), sort_keys=True),
        "contact_atom_names_json": json.dumps(dict(contact_atom_names), sort_keys=True),
        "ligand_elements_json": json.dumps(dict(Counter(ligand_elements)), sort_keys=True),
        "ok": True,
        "error": "",
    }


def build_summary(result_df, args):
    ok_df = result_df[result_df["ok"]].copy()
    fail_df = result_df[~result_df["ok"]].copy()
    element_counter = Counter()
    atom_name_counter = Counter()
    for value in ok_df.get("contact_elements_json", []):
        element_counter.update(json.loads(value))
    for value in ok_df.get("contact_atom_names_json", []):
        atom_name_counter.update(json.loads(value))

    summary = {
        "cutoff": args.cutoff,
        "include_hydrogen": args.include_hydrogen,
        "include_hetatm": args.include_hetatm,
        "rows_total": int(len(result_df)),
        "rows_ok": int(len(ok_df)),
        "rows_failed": int(len(fail_df)),
        "rows_zero_contact_atoms": int((ok_df["contact_atoms"] == 0).sum()) if len(ok_df) else 0,
        "contact_atoms": summarise_numeric(ok_df["contact_atoms"]) if len(ok_df) else {"count": 0},
        "contact_residues": summarise_numeric(ok_df["contact_residues"]) if len(ok_df) else {"count": 0},
        "pocket_atoms": summarise_numeric(ok_df["pocket_atoms"]) if len(ok_df) else {"count": 0},
        "pocket_residues": summarise_numeric(ok_df["pocket_residues"]) if len(ok_df) else {"count": 0},
        "min_protein_ligand_distance": (
            summarise_numeric(ok_df["min_protein_ligand_distance"]) if len(ok_df) else {"count": 0}
        ),
        "contact_backbone_atoms": (
            summarise_numeric(ok_df["contact_backbone_atoms"]) if len(ok_df) else {"count": 0}
        ),
        "contact_sidechain_atoms": (
            summarise_numeric(ok_df["contact_sidechain_atoms"]) if len(ok_df) else {"count": 0}
        ),
        "contact_elements_total": dict(element_counter.most_common()),
        "contact_atom_names_total": dict(atom_name_counter.most_common(30)),
        "top_errors": fail_df["error"].value_counts().head(20).to_dict() if len(fail_df) else {},
    }
    if "split" in ok_df.columns and ok_df["split"].astype(str).str.len().gt(0).any():
        split_summary = {}
        for split, frame in ok_df.groupby("split"):
            split_summary[str(split)] = {
                "rows": int(len(frame)),
                "contact_atoms": summarise_numeric(frame["contact_atoms"]),
                "contact_residues": summarise_numeric(frame["contact_residues"]),
                "zero_contact_atoms": int((frame["contact_atoms"] == 0).sum()),
            }
        summary["by_split"] = split_summary
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit protein pocket atoms within a cutoff of ligand atoms."
    )
    parser.add_argument(
        "--manifest",
        default="data/PDBbind_full/processed/pdbbind_pl_manifest.tsv",
        help="PDBbind manifest TSV with complex_id, ligand_sdf, and pocket_pdb columns.",
    )
    parser.add_argument(
        "--split_file",
        default="data/PDBbind_full/processed/pdbbind_pl_fixed_split_seed42.tsv",
        help="Optional fixed split TSV. Use '' to audit the full manifest without split labels.",
    )
    parser.add_argument(
        "--structure_root",
        default="data/PDBbind_full/extracted",
        help="Fallback root for resolving relative SDF/PDB paths.",
    )
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--limit", type=int, help="Limit rows for smoke tests.")
    parser.add_argument("--include_hydrogen", action="store_true")
    parser.add_argument("--include_hetatm", action="store_true")
    parser.add_argument(
        "--output_tsv",
        help="Output per-complex TSV. Defaults to a cutoff-specific file under processed/.",
    )
    parser.add_argument(
        "--summary_json",
        help="Output JSON summary. Defaults to a cutoff-specific file under processed/.",
    )
    parser.add_argument("--log_every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    cutoff_label = str(args.cutoff).replace(".", "p")
    if args.output_tsv is None:
        args.output_tsv = (
            "data/PDBbind_full/processed/"
            "pdbbind_pocket_atom_contacts_cutoff{}.tsv".format(cutoff_label)
        )
    if args.summary_json is None:
        args.summary_json = (
            "data/PDBbind_full/processed/"
            "pdbbind_pocket_atom_contacts_cutoff{}_summary.json".format(cutoff_label)
        )

    manifest = pd.read_table(args.manifest)
    required = {"complex_id", "ligand_sdf", "pocket_pdb"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError("Manifest missing required columns: {}".format(sorted(missing)))

    manifest = manifest.copy()
    manifest["complex_id"] = manifest["complex_id"].astype(str).str.lower()

    if args.split_file:
        split_df = pd.read_table(args.split_file)
        if not {"complex_id", "split"}.issubset(split_df.columns):
            raise ValueError("Split file must contain complex_id and split columns")
        split_df = split_df[["complex_id", "split"]].copy()
        split_df["complex_id"] = split_df["complex_id"].astype(str).str.lower()
        rows = split_df.merge(manifest, on="complex_id", how="inner")
    else:
        rows = manifest

    if args.limit is not None:
        rows = rows.head(args.limit).copy()

    results = []
    for i, row in enumerate(rows.to_dict("records"), start=1):
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print("auditing {}/{}".format(i, len(rows)), flush=True)
        try:
            results.append(audit_row(row, args))
        except Exception as exc:
            results.append({
                "complex_id": str(row.get("complex_id", "")).lower(),
                "split": row.get("split", ""),
                "ligand_sdf": row.get("ligand_sdf", ""),
                "pocket_pdb": row.get("pocket_pdb", ""),
                "ligand_atoms": np.nan,
                "pocket_atoms": np.nan,
                "pocket_residues": np.nan,
                "contact_atoms": np.nan,
                "contact_residues": np.nan,
                "contact_backbone_atoms": np.nan,
                "contact_sidechain_atoms": np.nan,
                "contact_atom_fraction": np.nan,
                "contact_residue_fraction": np.nan,
                "min_protein_ligand_distance": np.nan,
                "mean_contact_distance": np.nan,
                "max_contact_distance": np.nan,
                "contact_elements_json": "{}",
                "contact_atom_names_json": "{}",
                "ligand_elements_json": "{}",
                "ok": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
            })

    result_df = pd.DataFrame(results)
    output_tsv = Path(args.output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_tsv, sep="\t", index=False)

    summary = build_summary(result_df, args)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("wrote_tsv={}".format(output_tsv))
    print("wrote_summary={}".format(summary_json))
    print("rows_total={}".format(summary["rows_total"]))
    print("rows_ok={}".format(summary["rows_ok"]))
    print("rows_failed={}".format(summary["rows_failed"]))
    print("rows_zero_contact_atoms={}".format(summary["rows_zero_contact_atoms"]))
    print("contact_atoms_median={}".format(summary["contact_atoms"].get("median")))
    print("contact_atoms_p95={}".format(summary["contact_atoms"].get("p95")))
    print("contact_residues_median={}".format(summary["contact_residues"].get("median")))
    print("contact_residues_p95={}".format(summary["contact_residues"].get("p95")))


if __name__ == "__main__":
    main()
