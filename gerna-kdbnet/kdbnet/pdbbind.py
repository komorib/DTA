"""Utilities for preparing PDBbind protein-ligand data for KDBNet."""
import csv
import json
import math
import os
import re
import shutil
import tarfile
from pathlib import Path


AFFINITY_RE = re.compile(
    r"^(?P<kind>Kd|Ki|IC50|EC50)(?P<op><=|>=|<|>|=)?"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
    r"(?P<unit>fM|pM|nM|uM|mM|M)$"
)

UNIT_TO_MOLAR = {
    "fM": 1e-15,
    "pM": 1e-12,
    "nM": 1e-9,
    "uM": 1e-6,
    "mM": 1e-3,
    "M": 1.0,
}

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def pdbbind_y_from_molar(affinity_molar):
    if affinity_molar <= 0:
        raise ValueError("affinity_molar must be positive")
    return -math.log10(affinity_molar)


def parse_affinity_expression(expr):
    match = AFFINITY_RE.match(expr.strip())
    if match is None:
        return None
    value = float(match.group("value"))
    unit = match.group("unit")
    affinity_molar = value * UNIT_TO_MOLAR[unit]
    return {
        "affinity_type": match.group("kind"),
        "affinity_op": match.group("op") or "=",
        "affinity_value": value,
        "affinity_unit": unit,
        "affinity_molar": affinity_molar,
        "y": pdbbind_y_from_molar(affinity_molar),
    }


def parse_pdbbind_index_lines(lines):
    rows, skipped = [], []
    for line_no, line in enumerate(lines, start=1):
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        main = raw.split("//", 1)[0].strip()
        fields = main.split()
        if len(fields) < 4:
            skipped.append({"line_no": line_no, "line": raw, "reason": "malformed_index_line"})
            continue
        parsed = parse_affinity_expression(fields[3])
        if parsed is None:
            skipped.append({"line_no": line_no, "line": raw, "reason": "unsupported_affinity"})
            continue
        row = {
            "complex_id": fields[0].lower(),
            "resolution": fields[1],
            "release_year": fields[2],
            "affinity_raw": fields[3],
        }
        row.update(parsed)
        rows.append(row)
    return rows, skipped


def _read_pdb_source(pdb_source):
    if os.path.exists(str(pdb_source)):
        with open(pdb_source, "r") as handle:
            return handle.read()
    return str(pdb_source)


def parse_pdbbind_pocket_pdb(
    pdb_source,
    name=None,
    pdb_id=None,
    zero_embed_dim=None,
    embed_path=None,
):
    text = _read_pdb_source(pdb_source)
    residues = []
    residue_index = {}

    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_ATOMS:
            continue
        resname = line[17:20].strip().upper()
        aa = THREE_TO_ONE.get(resname)
        if aa is None:
            continue
        chain = line[21].strip() or "_"
        resseq = line[22:26].strip()
        icode = line[26].strip()
        key = (chain, resseq, icode)
        if key not in residue_index:
            residue_index[key] = len(residues)
            residues.append({"chain": chain, "aa": aa, "atoms": {}})
        try:
            coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        residues[residue_index[key]]["atoms"][atom] = coord

    valid = [res for res in residues if all(atom in res["atoms"] for atom in BACKBONE_ATOMS)]
    if not valid:
        raise ValueError("pocket PDB contains no residues with complete N/CA/C/O backbone")

    coords = {atom: [] for atom in BACKBONE_ATOMS}
    seq = []
    for residue in valid:
        seq.append(residue["aa"])
        for atom in BACKBONE_ATOMS:
            coords[atom].append(residue["atoms"][atom])

    first_chain = valid[0]["chain"]
    entry = {
        "name": name or pdb_id,
        "PDB_id": pdb_id or name,
        "chain": first_chain,
        "seq": "".join(seq),
        "coords": coords,
    }
    if embed_path:
        entry["embed"] = str(embed_path)
    elif zero_embed_dim is not None:
        entry["seq_emb"] = [[0.0] * int(zero_embed_dim) for _ in seq]
    return entry


MANIFEST_FIELDS = [
    "complex_id",
    "drug",
    "protein",
    "ligand_sdf",
    "pocket_pdb",
    "affinity_type",
    "affinity_op",
    "affinity_value",
    "affinity_unit",
    "affinity_molar",
    "y",
    "resolution",
    "release_year",
    "affinity_raw",
]


def build_manifest_rows(index_rows, file_records):
    manifest_rows, missing = [], []
    for row in index_rows:
        complex_id = row["complex_id"].lower()
        files = file_records.get(complex_id)
        if not files or not files.get("ligand_sdf") or not files.get("pocket_pdb"):
            missing.append({
                "complex_id": complex_id,
                "reason": "missing_structure_files",
            })
            continue
        manifest_row = {
            "complex_id": complex_id,
            "drug": complex_id,
            "protein": complex_id,
            "ligand_sdf": files["ligand_sdf"],
            "pocket_pdb": files["pocket_pdb"],
        }
        for key in MANIFEST_FIELDS:
            if key in row and key not in manifest_row:
                manifest_row[key] = row[key]
        manifest_rows.append(manifest_row)
    return manifest_rows, missing


def _ligand_sdf_skip_reason(ligand_sdf):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromMolFile(str(ligand_sdf))
    except Exception:
        return "invalid_ligand_sdf"
    if mol is None:
        return "invalid_ligand_sdf"
    if mol.GetNumConformers() == 0:
        return "ligand_sdf_missing_conformer"
    return None


def filter_manifest_rows_by_ligand_sdf(manifest_rows):
    filtered, skipped = [], []
    for row in manifest_rows:
        reason = _ligand_sdf_skip_reason(row["ligand_sdf"])
        if reason is None:
            filtered.append(row)
        else:
            skipped.append({
                "complex_id": row.get("complex_id"),
                "ligand_sdf": row.get("ligand_sdf"),
                "reason": reason,
            })
    return filtered, skipped


def find_pdbbind_members(p_l_tar_path):
    records = {}
    with tarfile.open(str(p_l_tar_path), "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            basename = os.path.basename(member.name)
            if basename.endswith("_ligand.sdf"):
                complex_id = basename[:-len("_ligand.sdf")].lower()
                records.setdefault(complex_id, {})["ligand_sdf_member"] = member.name
            elif basename.endswith("_pocket.pdb"):
                complex_id = basename[:-len("_pocket.pdb")].lower()
                records.setdefault(complex_id, {})["pocket_pdb_member"] = member.name
    return records


def _copy_tar_member(tar, member_name, target_path, overwrite=False):
    target_path = Path(target_path)
    if target_path.exists() and not overwrite:
        return str(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member_name)
    if src is None:
        raise ValueError("unable to extract tar member: {}".format(member_name))
    with src, open(str(target_path), "wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(target_path)


def extract_pdbbind_pl_subset(p_l_tar_path, complex_ids, output_dir, overwrite=False, member_records=None):
    output_dir = Path(output_dir)
    extracted_dir = output_dir / "extracted" / "P-L"
    targets = set(cid.lower() for cid in complex_ids)
    file_records, skipped = {}, []

    if member_records is not None:
        with tarfile.open(str(p_l_tar_path), "r:gz") as tar:
            for cid in targets:
                members = member_records.get(cid, {})
                ligand_member = members.get("ligand_sdf_member")
                pocket_member = members.get("pocket_pdb_member")
                if not ligand_member or not pocket_member:
                    skipped.append({"complex_id": cid, "reason": "missing_tar_members"})
                    continue
                target_dir = extracted_dir / cid
                ligand_path = target_dir / os.path.basename(ligand_member)
                pocket_path = target_dir / os.path.basename(pocket_member)
                file_records[cid] = {
                    "ligand_sdf": _copy_tar_member(tar, ligand_member, ligand_path, overwrite=overwrite),
                    "pocket_pdb": _copy_tar_member(tar, pocket_member, pocket_path, overwrite=overwrite),
                }
        return file_records, skipped

    seen = {cid: {} for cid in targets}
    with tarfile.open(str(p_l_tar_path), "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            basename = os.path.basename(member.name)
            if basename.endswith("_ligand.sdf"):
                cid = basename[:-len("_ligand.sdf")].lower()
                key = "ligand_sdf"
            elif basename.endswith("_pocket.pdb"):
                cid = basename[:-len("_pocket.pdb")].lower()
                key = "pocket_pdb"
            else:
                continue
            if cid not in targets:
                continue
            target_path = extracted_dir / cid / basename
            seen[cid][key] = _copy_tar_member(tar, member, target_path, overwrite=overwrite)
            if "ligand_sdf" in seen[cid] and "pocket_pdb" in seen[cid]:
                file_records[cid] = seen[cid]
                if len(file_records) == len(targets):
                    break

    for cid in sorted(targets):
        if cid not in file_records:
            skipped.append({"complex_id": cid, "reason": "missing_tar_members"})
    return file_records, skipped


def write_manifest_and_report(manifest_rows, skipped, output_dir):
    processed_dir = Path(output_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "pdbbind_pl_manifest.tsv"
    report_path = processed_dir / "pdbbind_pl_preprocess_report.json"

    with open(str(manifest_path), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    report = {
        "accepted": len(manifest_rows),
        "skipped": len(skipped),
        "skip_reasons": {},
        "skipped_examples": skipped[:50],
        "skipped_examples_by_reason": {},
    }
    for item in skipped:
        reason = item.get("reason", "unknown")
        report["skip_reasons"][reason] = report["skip_reasons"].get(reason, 0) + 1
        examples = report["skipped_examples_by_reason"].setdefault(reason, [])
        if len(examples) < 10:
            examples.append(item)
    with open(str(report_path), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return str(manifest_path), str(report_path), report


def read_pdbbind_pl_index(index_tar_path):
    with tarfile.open(str(index_tar_path), "r:gz") as tar:
        member = tar.extractfile("index/INDEX_general_PL.2020R1.lst")
        if member is None:
            raise ValueError("index/INDEX_general_PL.2020R1.lst not found in {}".format(index_tar_path))
        return member.read().decode("utf-8", errors="replace").splitlines()
