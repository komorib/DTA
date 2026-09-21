"""Build compact MSA profiles directly from an MMseqs result database."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

import lmdb

from .msa_profile import (
    PROFILE_FEATURE_NAMES,
    _atomic_json,
    _sha256,
    build_msa_profile,
)


def _read_index(path):
    entries = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"invalid MMseqs index line {line_number}")
            entries.append((int(fields[0]), int(fields[1]), int(fields[2])))
    if not entries:
        raise ValueError("MMseqs index is empty")
    keys = [entry[0] for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("MMseqs index contains duplicate keys")
    return entries


def _lineage(a3m_db, index_path, sequence_manifest, manifest, map_size_gb):
    stat = a3m_db.stat()
    return {
        "a3m_database": str(a3m_db),
        "a3m_database_size": stat.st_size,
        "a3m_index": str(index_path),
        "a3m_index_sha256": _sha256(index_path),
        "map_size_gb": float(map_size_gb),
        "profile_feature_names": list(PROFILE_FEATURE_NAMES),
        "sequence_count": len(manifest["sequences"]),
        "sequence_manifest": str(sequence_manifest),
        "sequence_manifest_sha256": _sha256(sequence_manifest),
        "version": "mmseqs-profile-db-v1",
    }


def build_mmseqs_profile_cache(
    a3m_db,
    sequence_manifest,
    output,
    map_size_gb=16,
    temp_dir="/tmp",
):
    a3m_db = Path(a3m_db).expanduser().resolve()
    index_path = Path(str(a3m_db) + ".index")
    sequence_manifest = Path(sequence_manifest).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    temp_dir = Path(temp_dir).expanduser().resolve()
    if not a3m_db.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"missing MMseqs database or index: {a3m_db}")
    with sequence_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("sequence manifest contains no sequences")
    entries = _read_index(index_path)
    if len(entries) != len(sequences):
        raise ValueError(
            f"MMseqs index has {len(entries)} records, manifest has {len(sequences)}"
        )
    expected_keys = set(range(len(sequences)))
    if {entry[0] for entry in entries} != expected_keys:
        raise ValueError("MMseqs index keys do not match manifest positions")

    output.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    lineage = _lineage(a3m_db, index_path, sequence_manifest, manifest, map_size_gb)
    lineage_path = output / "lineage.json"
    if lineage_path.is_file():
        existing_lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if existing_lineage != lineage:
            raise ValueError("profile cache lineage does not match this invocation")
    else:
        _atomic_json(lineage_path, lineage)

    env = lmdb.open(
        str(output / "profiles.lmdb"),
        map_size=int(float(map_size_gb) * (1 << 30)),
        subdir=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(temp_dir), prefix="msa-v9-record-", suffix=".a3m"
    )
    os.close(descriptor)
    depths = []
    neff = []
    reused = 0
    written = 0
    processed = 0
    try:
        with a3m_db.open("rb", buffering=8 * 1024 * 1024) as source:
            for key, offset, size in sorted(entries, key=lambda entry: entry[1]):
                record = sequences[key]
                sequence_id = str(record["sequence_id"])
                source.seek(offset)
                raw = source.read(size)
                if len(raw) != size:
                    raise IOError(f"short read for MMseqs record {key}")
                raw = raw.rstrip(b"\x00")
                first_line = raw.split(b"\n", 1)[0]
                observed_id = first_line[1:].split(None, 1)[0].decode("utf-8") if first_line.startswith(b">") else ""
                if observed_id != sequence_id:
                    raise ValueError(
                        f"MMseqs record {key} header {observed_id!r} does not match {sequence_id!r}"
                    )
                a3m_sha256 = hashlib.sha256(raw).hexdigest()
                with env.begin(write=False) as transaction:
                    cached_raw = transaction.get(sequence_id.encode("ascii"))
                cached = pickle.loads(bytes(cached_raw)) if cached_raw is not None else None
                if cached is not None and cached.get("a3m_sha256") == a3m_sha256:
                    profile = cached
                    reused += 1
                else:
                    with open(temporary_name, "wb") as temporary:
                        temporary.write(raw)
                        temporary.write(b"\n")
                    profile = build_msa_profile(temporary_name, record["sequence"])
                    profile["a3m_sha256"] = a3m_sha256
                    profile["sequence_id"] = sequence_id
                    profile["sequence_sha256"] = record["sequence_sha256"]
                    profile["mmseqs_key"] = key
                    with env.begin(write=True) as transaction:
                        transaction.put(
                            sequence_id.encode("ascii"),
                            pickle.dumps(profile, protocol=4),
                            overwrite=True,
                        )
                    written += 1
                depths.append(int(profile["depth"]))
                neff.append(float(profile["henikoff_neff"]))
                processed += 1
                if processed % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "mmseqs_profile_progress",
                                "processed": processed,
                                "reused": reused,
                                "written": written,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        env.sync()
        env.close()

    report = {
        "a3m_database": str(a3m_db),
        "depth": {
            "maximum": max(depths),
            "mean": sum(depths) / len(depths),
            "minimum": min(depths),
        },
        "feature_count": len(PROFILE_FEATURE_NAMES),
        "henikoff_neff": {
            "maximum": max(neff),
            "mean": sum(neff) / len(neff),
            "minimum": min(neff),
        },
        "output": str(output),
        "processed": processed,
        "reused": reused,
        "status": "complete",
        "written": written,
    }
    _atomic_json(output / "profile_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3m-db", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-size-gb", type=float, default=16.0)
    parser.add_argument("--temp-dir", default="/tmp")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    build_mmseqs_profile_cache(
        args.a3m_db,
        args.sequence_manifest,
        args.output,
        map_size_gb=args.map_size_gb,
        temp_dir=args.temp_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
