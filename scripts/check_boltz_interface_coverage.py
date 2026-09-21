#!/usr/bin/env python3
"""Verify that every manifest sample has a Boltz interface LMDB record."""

import argparse
import json
import os

import lmdb


def _sample_key(sample):
    key = sample.get("boltz_interface_key") or sample.get("lmdb_key") or sample.get("key")
    if not key:
        raise ValueError("Manifest sample has no boltz_interface_key, lmdb_key, or key")
    return str(key)


def _missing_keys(lmdb_path, keys):
    env = lmdb.open(
        os.path.abspath(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=os.path.isdir(lmdb_path),
    )
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            return [key for key in keys if not cursor.set_key(key.encode("utf-8"))]
    finally:
        env.close()


def inspect_coverage(manifest_paths, boltz_lmdb_path, preview_limit=20):
    reports = []
    total = present = 0
    for manifest_path in manifest_paths:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        samples = manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"Manifest has no samples list: {manifest_path}")
        keys = [_sample_key(sample) for sample in samples]
        missing = _missing_keys(boltz_lmdb_path, keys)
        count_present = len(keys) - len(missing)
        total += len(keys)
        present += count_present
        reports.append({
            "manifest": os.path.abspath(manifest_path),
            "total": len(keys),
            "present": count_present,
            "missing": len(missing),
            "missing_preview": missing[:int(preview_limit)],
        })
    return {
        "boltz_lmdb": os.path.abspath(boltz_lmdb_path),
        "total": total,
        "present": present,
        "missing": total - present,
        "manifests": reports,
    }


def write_intersected_manifests(manifest_paths, boltz_lmdb_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    written_paths = []
    summaries = []
    for manifest_path in manifest_paths:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        samples = manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"Manifest has no samples list: {manifest_path}")
        missing = set(_missing_keys(boltz_lmdb_path, [_sample_key(sample) for sample in samples]))
        filtered = dict(manifest)
        filtered["samples"] = [sample for sample in samples if _sample_key(sample) not in missing]
        output_path = os.path.join(output_dir, os.path.basename(manifest_path))
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(filtered, handle, indent=2)
            handle.write("\n")
        written_paths.append(output_path)
        summaries.append({
            "source_manifest": os.path.abspath(manifest_path),
            "output_manifest": os.path.abspath(output_path),
            "original_samples": len(samples),
            "retained_samples": len(filtered["samples"]),
            "removed_samples": len(missing),
        })
    summary_path = os.path.join(output_dir, "boltz_interface_coverage.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "boltz_lmdb": os.path.abspath(boltz_lmdb_path),
            "manifests": summaries,
        }, handle, indent=2)
        handle.write("\n")
    return written_paths


def main():
    parser = argparse.ArgumentParser(description="Check LP-PDBBind manifest coverage in a Boltz interface LMDB.")
    parser.add_argument("--manifest", action="append", required=True, help="Manifest JSON. Repeat for train/valid/test.")
    parser.add_argument("--boltz-lmdb", required=True)
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--write-intersection-dir", default=None, help="Write separate manifests retaining only samples present in the Boltz LMDB.")
    args = parser.parse_args()
    report = inspect_coverage(args.manifest, args.boltz_lmdb, preview_limit=args.preview_limit)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write_intersection_dir:
        written = write_intersected_manifests(args.manifest, args.boltz_lmdb, args.write_intersection_dir)
        print(json.dumps({"written_intersection_manifests": [os.path.abspath(path) for path in written]}, indent=2))
    if report["missing"] and not args.write_intersection_dir:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
