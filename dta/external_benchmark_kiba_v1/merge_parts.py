"""Merge independently generated KIBA split bundles without re-decoding data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SPLITS = ("train", "valid", "test")
SUBDIRS = ("common", "graphdta", "transformercpi", "graphormer", "psichic", "fmca")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(parts_root: Path, output: Path) -> dict:
    parts_root = parts_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for split in SPLITS:
        part = parts_root / split
        part_metadata = json.loads((part / "metadata.json").read_text(encoding="utf-8"))
        metadata.append(part_metadata)
        for subdir in SUBDIRS:
            destination = output / subdir
            destination.mkdir(parents=True, exist_ok=True)
            for path in (part / subdir).glob("*"):
                if path.is_file():
                    shutil.copy2(str(path), str(destination / path.name))
        flat = part / ("flat_%s.csv" % split)
        if flat.exists():
            shutil.copy2(str(flat), str(output / flat.name))

    first = metadata[0]
    files = {}
    for split in SPLITS:
        for subdir in SUBDIRS:
            for path in (output / subdir).glob("*"):
                if path.is_file():
                    relative = str(path.relative_to(output))
                    files[relative] = _sha256(path)
        flat = output / ("flat_%s.csv" % split)
        files["flat/%s.csv" % split] = _sha256(flat)
    audit = {
        "counts": {},
        "source_counts": {},
        "removed_training_pair_overlap_counts": {},
        "positive_counts": {},
        "positive_rates": {},
        "unique_smiles": {},
        "unique_sequences": {},
        "duplicate_pair_counts": {},
    }
    for item in metadata:
        audit_item = item["audit"]
        for key in audit:
            audit[key].update(audit_item.get(key, {}))
    overlaps = {}
    for item in metadata:
        overlaps.update(item["audit"].get("cross_split_pair_overlap", {}))
    pair_sets = {}
    for split in SPLITS:
        rows = []
        with (output / "common" / (split + ".jsonl")).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line)["pair_id"])
        pair_sets[split] = set(rows)
    overlaps = {
        "%s_%s" % (left, right): len(pair_sets[left] & pair_sets[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    audit["cross_split_pair_overlap"] = overlaps
    audit["train_pair_overlap_free"] = all(
        overlaps.get(key, 0) == 0 for key in ("train_valid", "train_test")
    )
    audit["cross_split_overlap_free"] = all(value == 0 for value in overlaps.values())
    merged = {
        "version": first["version"],
        "task": first["task"],
        "source_root": first["source_root"],
        "source_split_fingerprints": first["source_split_fingerprints"],
        "splits": list(SPLITS),
        "affinity_threshold": first["affinity_threshold"],
        "limit": first.get("limit"),
        "selection_policy": "validation only; training duplicates against valid/test removed; valid/test duplicate pairs reported",
        "audit": audit,
        "files_sha256": files,
        "parts": [str((parts_root / split).resolve()) for split in SPLITS],
        "status": "built",
    }
    (output / "metadata.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.parts_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
