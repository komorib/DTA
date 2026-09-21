"""Build a compact, deterministic multi-row MSA cache from an MMseqs A3M DB."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import lmdb
import numpy as np

from dta.msa_pocket_expert_v9.msa_profile import (
    AA_TO_INDEX,
    GAP_INDEX,
    UNKNOWN_INDEX,
    _atomic_json,
    _parse_a3m_row,
    _sha256,
)

from . import VERSION


ROW_CACHE_SCHEMA_VERSION = 1


def _read_index(path):
    entries = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError("invalid MMseqs index line %d" % line_number)
            entries.append((int(fields[0]), int(fields[1]), int(fields[2])))
    if not entries:
        raise ValueError("MMseqs index is empty")
    if len({entry[0] for entry in entries}) != len(entries):
        raise ValueError("MMseqs index contains duplicate keys")
    return entries


def _parse_fasta_rows(raw):
    rows = []
    names = []
    current = []
    for raw_line in raw.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"): 
            if current:
                rows.append("".join(current))
                current = []
            names.append(line[1:].split(None, 1)[0])
        else:
            if not names:
                raise ValueError("A3M sequence appears before its header")
            current.append(line)
    if current:
        rows.append("".join(current))
    if not rows or len(rows) != len(names):
        raise ValueError("A3M record has mismatched headers and sequences")
    return names, rows


def _encode_expected(sequence):
    return np.asarray(
        [AA_TO_INDEX.get(residue, UNKNOWN_INDEX) for residue in str(sequence).upper()],
        dtype=np.uint8,
    )


def _select_diverse_rows(alignment, coverage, identity, maximum_rows):
    depth = int(alignment.shape[0])
    maximum_rows = int(maximum_rows)
    if maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    if depth <= maximum_rows:
        return np.arange(depth, dtype=np.int64)

    bin_ids = np.digitize(identity[1:], [0.20, 0.40, 0.60, 0.80, 0.95])
    queues = []
    for bin_index in range(6):
        candidates = np.flatnonzero(bin_ids == bin_index) + 1
        candidates = sorted(candidates.tolist(), key=lambda index: (-coverage[index], index))
        queues.append(candidates)
    selected = [0]
    offsets = [0] * len(queues)
    # Alternating low/high identity prevents the cache from collapsing to near-query rows.
    order = (0, 5, 1, 4, 2, 3)
    while len(selected) < maximum_rows:
        changed = False
        for bin_index in order:
            offset = offsets[bin_index]
            if offset < len(queues[bin_index]):
                selected.append(queues[bin_index][offset])
                offsets[bin_index] += 1
                changed = True
                if len(selected) == maximum_rows:
                    break
        if not changed:
            break
    return np.asarray(selected, dtype=np.int64)


def parse_and_select_a3m(raw, expected_sequence, maximum_rows):
    names, rows = _parse_fasta_rows(raw)
    length = len(expected_sequence)
    alignment = np.empty((len(rows), length), dtype=np.uint8)
    insertions = np.empty((len(rows), length), dtype=np.uint16)
    for row_index, row in enumerate(rows):
        alignment[row_index], insertions[row_index] = _parse_a3m_row(row, length)
    expected = _encode_expected(expected_sequence)
    if not np.array_equal(alignment[0], expected):
        raise ValueError("A3M query does not match expected sequence")
    coverage = np.mean(alignment != GAP_INDEX, axis=1).astype(np.float32)
    identity = np.mean(alignment == expected[None, :], axis=1).astype(np.float32)
    selected = _select_diverse_rows(alignment, coverage, identity, maximum_rows)
    return {
        "alignment": alignment[selected],
        "coverage": coverage[selected],
        "headers": tuple(names[index] for index in selected),
        "identity": identity[selected],
        "insertions": np.minimum(insertions[selected], 255).astype(np.uint8),
        "original_depth": len(rows),
    }


def _required_sequence_ids(sequence_manifest, manifest_paths, limit_per_split):
    sequences = sequence_manifest["sequences"]
    if not manifest_paths:
        return {str(record["sequence_id"]) for record in sequences}
    selected_occurrences = set()
    for value in manifest_paths:
        path = Path(value).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        samples = list(manifest["samples"])
        if limit_per_split is not None:
            samples = samples[: int(limit_per_split)]
        split = str(manifest["split"])
        selected_occurrences.update((split, str(sample["lmdb_key"])) for sample in samples)
    result = set()
    for record in sequences:
        for occurrence in record.get("occurrences", []):
            occurrence_key = (str(occurrence["split"]), str(occurrence["lmdb_key"]))
            if occurrence_key in selected_occurrences:
                result.add(str(record["sequence_id"]))
                break
    if not result:
        raise ValueError("manifest filters selected no MSA sequences")
    return result


def build_row_cache(
    a3m_db,
    sequence_manifest,
    output,
    *,
    maximum_rows=128,
    map_size_gb=8.0,
    manifests=(),
    limit_per_split=None,
):
    a3m_db = Path(a3m_db).expanduser().resolve()
    index_path = Path(str(a3m_db) + ".index")
    sequence_manifest = Path(sequence_manifest).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not a3m_db.is_file() or not index_path.is_file():
        raise FileNotFoundError("missing MMseqs A3M database or index")
    with sequence_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    sequences = list(manifest.get("sequences", []))
    entries = _read_index(index_path)
    if len(entries) != len(sequences):
        raise ValueError("MMseqs index and sequence manifest counts differ")
    if {entry[0] for entry in entries} != set(range(len(sequences))):
        raise ValueError("MMseqs index keys do not match sequence manifest positions")
    required = _required_sequence_ids(manifest, manifests, limit_per_split)

    output.mkdir(parents=True, exist_ok=True)
    lineage = {
        "a3m_database": str(a3m_db),
        "a3m_database_size": a3m_db.stat().st_size,
        "a3m_index_sha256": _sha256(index_path),
        "limit_per_split": None if limit_per_split is None else int(limit_per_split),
        "manifests": [str(Path(value).expanduser().resolve()) for value in manifests],
        "map_size_gb": float(map_size_gb),
        "maximum_rows": int(maximum_rows),
        "required_count": len(required),
        "schema_version": ROW_CACHE_SCHEMA_VERSION,
        "sequence_manifest": str(sequence_manifest),
        "sequence_manifest_sha256": _sha256(sequence_manifest),
        "version": VERSION,
    }
    lineage_path = output / "lineage.json"
    if lineage_path.is_file():
        existing = json.loads(lineage_path.read_text(encoding="utf-8"))
        if existing != lineage:
            raise ValueError("MSA row cache lineage does not match")
    else:
        _atomic_json(lineage_path, lineage)

    environment = lmdb.open(
        str(output / "rows.lmdb"),
        map_size=int(float(map_size_gb) * (1 << 30)),
        subdir=True,
    )
    depths = []
    retained = []
    reused = 0
    written = 0
    processed = 0
    try:
        with a3m_db.open("rb", buffering=8 * 1024 * 1024) as source:
            for key, offset, size in sorted(entries, key=lambda entry: entry[1]):
                record = sequences[key]
                sequence_id = str(record["sequence_id"])
                if sequence_id not in required:
                    continue
                source.seek(offset)
                raw = source.read(size).rstrip(b"\x00")
                if not raw:
                    raise IOError("empty MMseqs A3M record for %s" % sequence_id)
                a3m_sha256 = hashlib.sha256(raw).hexdigest()
                with environment.begin(write=False) as transaction:
                    cached_raw = transaction.get(sequence_id.encode("ascii"))
                cached = pickle.loads(bytes(cached_raw)) if cached_raw is not None else None
                if (
                    cached is not None
                    and cached.get("a3m_sha256") == a3m_sha256
                    and int(cached.get("maximum_rows", -1)) == int(maximum_rows)
                ):
                    item = cached
                    reused += 1
                else:
                    parsed = parse_and_select_a3m(raw, record["sequence"], maximum_rows)
                    item = {
                        "a3m_sha256": a3m_sha256,
                        "alignment": parsed["alignment"].astype(np.uint8, copy=False),
                        "coverage": parsed["coverage"].astype(np.float16),
                        "identity": parsed["identity"].astype(np.float16),
                        "insertions": parsed["insertions"],
                        "maximum_rows": int(maximum_rows),
                        "original_depth": int(parsed["original_depth"]),
                        "query_sequence": str(record["sequence"]),
                        "schema_version": ROW_CACHE_SCHEMA_VERSION,
                        "sequence_id": sequence_id,
                        "version": VERSION,
                    }
                    with environment.begin(write=True) as transaction:
                        transaction.put(
                            sequence_id.encode("ascii"),
                            pickle.dumps(item, protocol=4),
                            overwrite=True,
                        )
                    written += 1
                depths.append(int(item["original_depth"]))
                retained.append(int(item["alignment"].shape[0]))
                processed += 1
                if processed % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "msa_v11_row_cache_progress",
                                "processed": processed,
                                "reused": reused,
                                "written": written,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        environment.sync()
        environment.close()
    if processed != len(required):
        raise RuntimeError("MSA row cache wrote %d of %d required sequences" % (processed, len(required)))
    report = {
        "depth": {"minimum": min(depths), "mean": float(np.mean(depths)), "maximum": max(depths)},
        "output": str(output),
        "processed": processed,
        "retained": {
            "minimum": min(retained),
            "mean": float(np.mean(retained)),
            "maximum": max(retained),
        },
        "reused": reused,
        "status": "complete",
        "written": written,
    }
    _atomic_json(output / "row_cache_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3m-db", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-rows", type=int, default=128)
    parser.add_argument("--map-size-gb", type=float, default=8.0)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--limit-per-split", type=int)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    build_row_cache(
        args.a3m_db,
        args.sequence_manifest,
        args.output,
        maximum_rows=args.maximum_rows,
        map_size_gb=args.map_size_gb,
        manifests=args.manifest,
        limit_per_split=args.limit_per_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
