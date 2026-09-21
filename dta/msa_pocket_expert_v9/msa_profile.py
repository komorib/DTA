"""Convert validated A3M files into compact residue-level evolutionary profiles."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile

import lmdb
import numpy as np

from . import VERSION
from .mmseqs_search import read_fasta


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
UNKNOWN_INDEX = len(AMINO_ACIDS)
GAP_INDEX = UNKNOWN_INDEX + 1
PROFILE_SCHEMA_VERSION = 1
PROFILE_FEATURE_NAMES = tuple(
    [f"frequency_{residue}" for residue in AMINO_ACIDS]
    + ["frequency_unknown", "frequency_gap"]
    + [f"log_odds_{residue}" for residue in AMINO_ACIDS]
    + [
        "entropy",
        "conservation",
        "coverage",
        "unknown_frequency",
        "insertion_fraction",
        "mean_insertion_length",
        "log_depth",
        "log_henikoff_neff",
    ]
)
BACKGROUND = np.asarray(
    [
        0.078, 0.051, 0.043, 0.053, 0.019, 0.040, 0.063, 0.074, 0.022, 0.052,
        0.091, 0.059, 0.024, 0.040, 0.051, 0.069, 0.059, 0.014, 0.032, 0.066,
    ],
    dtype=np.float64,
)
BACKGROUND /= BACKGROUND.sum()


def _parse_a3m_row(row, expected_length):
    aligned = []
    insertions = np.zeros(expected_length, dtype=np.uint16)
    column = -1
    for residue in row:
        if residue.islower():
            insertion_column = max(column, 0)
            if insertion_column >= expected_length:
                insertion_column = expected_length - 1
            insertions[insertion_column] += 1
            continue
        column += 1
        aligned.append(residue.upper())
    if len(aligned) != expected_length:
        raise ValueError(f"A3M row has {len(aligned)} aligned columns, expected {expected_length}")
    encoded = np.empty(expected_length, dtype=np.uint8)
    for index, residue in enumerate(aligned):
        if residue == "-":
            encoded[index] = GAP_INDEX
        else:
            encoded[index] = AA_TO_INDEX.get(residue, UNKNOWN_INDEX)
    return encoded, insertions


def _henikoff_weights(alignment):
    depth, length = alignment.shape
    weights = np.zeros(depth, dtype=np.float64)
    for column in range(length):
        values = alignment[:, column]
        nongap = values != GAP_INDEX
        if not np.any(nongap):
            continue
        counts = np.bincount(values[nongap], minlength=GAP_INDEX + 1).astype(np.float64)
        residue_types = np.count_nonzero(counts[:GAP_INDEX])
        if residue_types == 0:
            continue
        weights[nongap] += 1.0 / (residue_types * counts[values[nongap]])
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(depth, 1.0 / depth, dtype=np.float64)
    return weights / total


def build_msa_profile(a3m_path, expected_sequence):
    expected_sequence = str(expected_sequence).upper()
    records = read_fasta(a3m_path, uppercase=False, require_unique=False)
    depth = len(records)
    length = len(expected_sequence)
    alignment = np.empty((depth, length), dtype=np.uint8)
    insertion_counts = np.empty((depth, length), dtype=np.uint16)
    for row_index, (_name, row) in enumerate(records):
        alignment[row_index], insertion_counts[row_index] = _parse_a3m_row(row, length)
    query = "".join(
        "-" if value == GAP_INDEX else AMINO_ACIDS[value] if value < UNKNOWN_INDEX else "X"
        for value in alignment[0]
    )
    if query != expected_sequence:
        raise ValueError("A3M query does not match expected sequence")

    weights = _henikoff_weights(alignment)
    henikoff_neff = float(1.0 / np.square(weights).sum())
    frequencies = np.zeros((length, GAP_INDEX + 1), dtype=np.float64)
    insertion_fraction = np.zeros(length, dtype=np.float64)
    mean_insertion_length = np.zeros(length, dtype=np.float64)
    for column in range(length):
        frequencies[column] = np.bincount(
            alignment[:, column], weights=weights, minlength=GAP_INDEX + 1
        )
        inserted = insertion_counts[:, column] > 0
        insertion_fraction[column] = float(weights[inserted].sum())
        mean_insertion_length[column] = float(np.dot(weights, insertion_counts[:, column]))

    aa_frequency = frequencies[:, :UNKNOWN_INDEX]
    aa_mass = aa_frequency.sum(axis=1, keepdims=True)
    conditional = np.divide(
        aa_frequency,
        aa_mass,
        out=np.zeros_like(aa_frequency),
        where=aa_mass > 0.0,
    )
    safe_conditional = np.where(conditional > 0.0, conditional, 1.0)
    entropy = -np.sum(conditional * np.log(safe_conditional), axis=1)
    conservation = 1.0 - entropy / math.log(len(AMINO_ACIDS))
    coverage = 1.0 - frequencies[:, GAP_INDEX]
    log_odds = np.log((conditional + 1e-4) / (BACKGROUND[None, :] + 1e-4))
    log_odds = np.clip(log_odds, -8.0, 8.0)
    global_features = np.column_stack(
        [
            entropy,
            conservation,
            coverage,
            frequencies[:, UNKNOWN_INDEX],
            insertion_fraction,
            np.log1p(mean_insertion_length),
            np.full(length, math.log1p(depth)),
            np.full(length, math.log1p(henikoff_neff)),
        ]
    )
    residue_features = np.concatenate([frequencies, log_odds, global_features], axis=1).astype(np.float32)
    if residue_features.shape != (length, len(PROFILE_FEATURE_NAMES)):
        raise RuntimeError("unexpected MSA profile shape")
    if not np.isfinite(residue_features).all():
        raise RuntimeError("MSA profile contains non-finite values")
    return {
        "depth": depth,
        "feature_names": PROFILE_FEATURE_NAMES,
        "henikoff_neff": henikoff_neff,
        "query_sequence": expected_sequence,
        "residue_features": residue_features,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "version": VERSION,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + "-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_profile_cache(a3m_dir, sequence_manifest, output, map_size_gb=16):
    a3m_dir = Path(a3m_dir).expanduser().resolve()
    sequence_manifest = Path(sequence_manifest).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    with sequence_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("sequence manifest contains no sequences")
    output.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output / "profiles.lmdb"), map_size=int(map_size_gb * (1 << 30)), subdir=True)
    depths = []
    neff = []
    written = 0
    try:
        for record in sequences:
            sequence_id = str(record["sequence_id"])
            path = a3m_dir / f"{sequence_id}.a3m"
            if not path.is_file():
                continue
            profile = build_msa_profile(path, record["sequence"])
            profile["a3m_sha256"] = _sha256(path)
            profile["sequence_id"] = sequence_id
            profile["sequence_sha256"] = record["sequence_sha256"]
            with env.begin(write=True) as transaction:
                transaction.put(sequence_id.encode("ascii"), pickle.dumps(profile, protocol=4), overwrite=True)
            depths.append(profile["depth"])
            neff.append(profile["henikoff_neff"])
            written += 1
            if written % 100 == 0:
                print(json.dumps({"event": "msa_profile_progress", "written": written}, sort_keys=True), flush=True)
    finally:
        env.sync()
        env.close()
    if written == 0:
        raise ValueError("no A3M files matched the sequence manifest")
    report = {
        "a3m_dir": str(a3m_dir),
        "depth": {"maximum": max(depths), "mean": sum(depths) / written, "minimum": min(depths)},
        "feature_count": len(PROFILE_FEATURE_NAMES),
        "henikoff_neff": {"maximum": max(neff), "mean": sum(neff) / written, "minimum": min(neff)},
        "output": str(output),
        "sequence_manifest": str(sequence_manifest),
        "sequence_manifest_sha256": _sha256(sequence_manifest),
        "status": "complete",
        "written": written,
    }
    _atomic_json(output / "profile_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3m-dir", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-size-gb", type=float, default=16.0)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    build_profile_cache(args.a3m_dir, args.sequence_manifest, args.output, map_size_gb=args.map_size_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
