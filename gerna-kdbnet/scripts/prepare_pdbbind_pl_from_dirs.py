"""Build the KDBNet PDBbind protein-ligand manifest directly from an
ALREADY-EXTRACTED PDBbind data tree (no tarballs required).

Expected layout (as provided by the user under /mnt/d/KDBNet/data):

    data/
      index/INDEX_general_PL.2020R1.lst      # standard PDBbind index
      P-L/<year-group>/<pdbid>/<pdbid>_ligand.sdf
      P-L/<year-group>/<pdbid>/<pdbid>_pocket.pdb
      P-L/<year-group>/<pdbid>/<pdbid>_protein.pdb
      P-L/<year-group>/<pdbid>/<pdbid>_ligand.mol2

This produces the same manifest TSV + report JSON that
scripts/prepare_pdbbind_pl.py would, but reads the extracted directory
tree instead of index.tar.gz / P-L.tar.gz.

Label handling is inherited unchanged from kdbnet.pdbbind:
  affinity string (Kd/Ki/IC50/EC50 + unit) -> molar -> y = -log10(molar).

Ligand-SDF RDKit validity filtering is OPTIONAL here (default OFF) so the
manifest can be built in an environment without RDKit. Pass --validate_ligands
to enable it (requires rdkit); invalid/conformer-less ligands are then dropped
and reported, matching the tarball pipeline.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kdbnet.pdbbind import (
    build_manifest_rows,
    filter_manifest_rows_by_ligand_sdf,
    parse_pdbbind_index_lines,
    write_manifest_and_report,
)


def read_index_file(index_path):
    """Read a plain (already-extracted) PDBbind PL index .lst file."""
    with open(index_path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()


def scan_extracted_pl(pl_root):
    """Scan data/P-L/<group>/<pdbid>/ for ligand.sdf + pocket.pdb.

    Returns file_records: {complex_id: {"ligand_sdf": path, "pocket_pdb": path}}
    matching the structure kdbnet.pdbbind.build_manifest_rows expects.
    """
    pl_root = Path(pl_root)
    file_records = {}
    for group in sorted(os.listdir(pl_root)):
        group_dir = pl_root / group
        if not group_dir.is_dir():
            continue
        for cid in sorted(os.listdir(group_dir)):
            cx_dir = group_dir / cid
            if not cx_dir.is_dir():
                continue
            cid_l = cid.lower()
            ligand_sdf = cx_dir / "{}_ligand.sdf".format(cid)
            pocket_pdb = cx_dir / "{}_pocket.pdb".format(cid)
            rec = {}
            if ligand_sdf.exists():
                rec["ligand_sdf"] = str(ligand_sdf)
            if pocket_pdb.exists():
                rec["pocket_pdb"] = str(pocket_pdb)
            if rec:
                file_records[cid_l] = rec
    return file_records


def main():
    parser = argparse.ArgumentParser(
        description="Build KDBNet PDBbind PL manifest from an extracted data tree"
    )
    parser.add_argument("--data_dir", default="data",
                        help="Directory containing index/ and P-L/ (default: data)")
    parser.add_argument("--index_file", default=None,
                        help="Path to INDEX_general_PL.2020R1.lst "
                             "(default: <data_dir>/index/INDEX_general_PL.2020R1.lst)")
    parser.add_argument("--pl_root", default=None,
                        help="Path to the P-L root (default: <data_dir>/P-L)")
    parser.add_argument("--output_dir", default="data/PDBbind",
                        help="Output dir for processed/ manifest + report")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit accepted index rows before matching (debug)")
    parser.add_argument("--validate_ligands", action="store_true",
                        help="Drop ligands RDKit cannot parse / lacking a conformer "
                             "(requires rdkit; OFF by default)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    index_file = Path(args.index_file) if args.index_file else data_dir / "index" / "INDEX_general_PL.2020R1.lst"
    pl_root = Path(args.pl_root) if args.pl_root else data_dir / "P-L"
    output_dir = Path(args.output_dir)

    if not index_file.exists():
        raise FileNotFoundError(str(index_file))
    if not pl_root.exists():
        raise FileNotFoundError(str(pl_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines = read_index_file(index_file)
    index_rows, skipped = parse_pdbbind_index_lines(index_lines)
    if args.limit is not None:
        index_rows = index_rows[:args.limit]

    file_records = scan_extracted_pl(pl_root)
    manifest_rows, missing = build_manifest_rows(index_rows, file_records)

    ligand_skips = []
    if args.validate_ligands:
        manifest_rows, ligand_skips = filter_manifest_rows_by_ligand_sdf(manifest_rows)

    all_skips = skipped + missing + ligand_skips
    manifest_path, report_path, report = write_manifest_and_report(
        manifest_rows, all_skips, output_dir)

    print("index_rows={} matched_complexes={}".format(len(index_rows), len(file_records)))
    print("accepted={} skipped={}".format(report["accepted"], report["skipped"]))
    print("skip_reasons={}".format(report["skip_reasons"]))
    print("manifest={}".format(manifest_path))
    print("report={}".format(report_path))


if __name__ == "__main__":
    main()
