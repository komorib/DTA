import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kdbnet.pdbbind import (
    build_manifest_rows,
    extract_pdbbind_pl_subset,
    filter_manifest_rows_by_ligand_sdf,
    parse_pdbbind_index_lines,
    read_pdbbind_pl_index,
    write_manifest_and_report,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare PDBbind protein-ligand data for KDBNet")
    parser.add_argument("--pdbbind_dir", default="PDBbind", help="Directory containing index.tar.gz and P-L.tar.gz")
    parser.add_argument("--output_dir", default="data/PDBbind", help="Output directory for extracted subset and manifest")
    parser.add_argument("--limit", type=int, help="Limit accepted index rows before extraction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory and extracted files")
    args = parser.parse_args()

    pdbbind_dir = Path(args.pdbbind_dir)
    output_dir = Path(args.output_dir)
    index_tar = pdbbind_dir / "index.tar.gz"
    pl_tar = pdbbind_dir / "P-L.tar.gz"
    if not index_tar.exists():
        raise FileNotFoundError(str(index_tar))
    if not pl_tar.exists():
        raise FileNotFoundError(str(pl_tar))
    if args.overwrite and output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines = read_pdbbind_pl_index(index_tar)
    index_rows, skipped = parse_pdbbind_index_lines(index_lines)
    selected_rows = index_rows[:args.limit] if args.limit is not None else index_rows
    file_records, extract_skips = extract_pdbbind_pl_subset(
        pl_tar,
        [row["complex_id"] for row in selected_rows],
        output_dir,
        overwrite=args.overwrite,
    )
    manifest_rows, missing = build_manifest_rows(selected_rows, file_records)
    manifest_rows, ligand_skips = filter_manifest_rows_by_ligand_sdf(manifest_rows)
    all_skips = skipped + extract_skips + missing + ligand_skips
    manifest_path, report_path, report = write_manifest_and_report(manifest_rows, all_skips, output_dir)

    print("accepted={} skipped={}".format(report["accepted"], report["skipped"]))
    print("manifest={}".format(manifest_path))
    print("report={}".format(report_path))


if __name__ == "__main__":
    main()
