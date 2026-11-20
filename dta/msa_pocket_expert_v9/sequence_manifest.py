"""Build a deduplicated chain-sequence manifest for MMseqs searches."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile

import lmdb

from data_utils.lmdb_dataset import ordered_chain_items, sequence_to_string
from utils.experiment_integrity import sha256_file

from . import SEQUENCE_MANIFEST_SCHEMA_VERSION, VERSION


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def normalize_query_sequence(value):
    sequence = "".join(sequence_to_string(value).split()).upper()
    if not sequence:
        raise ValueError("protein chain sequence is empty")
    return "".join(residue if residue in CANONICAL_AA else "X" for residue in sequence)


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + "-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_manifest(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"invalid LP-PDBbind manifest: {path}")
    lmdb_path = payload.get("lmdb_path")
    if not lmdb_path:
        raise ValueError(f"manifest does not define lmdb_path: {path}")
    lmdb_path = Path(lmdb_path)
    if not lmdb_path.is_absolute():
        lmdb_path = (path.parent / lmdb_path).resolve()
    return path, payload, lmdb_path


def _open_lmdb(path):
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
        max_readers=8,
    )


def _fasta_text(records):
    lines = []
    for record in records:
        lines.append(">" + record["sequence_id"])
        sequence = record["sequence"]
        lines.extend(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    return "\n".join(lines) + "\n"


def build_sequence_manifest(manifest_specs, output_dir, limit_per_split=None):
    if not manifest_specs:
        raise ValueError("at least one split manifest is required")
    if limit_per_split is not None and int(limit_per_split) <= 0:
        raise ValueError("limit_per_split must be positive")

    output_dir = Path(output_dir).expanduser().resolve()
    records_by_sha = {}
    source_manifests = []
    envs = {}
    occurrence_count = 0

    try:
        for split, manifest_path in manifest_specs:
            path, manifest, lmdb_path = _load_manifest(manifest_path)
            declared_split = str(manifest.get("split", split))
            if declared_split != split:
                raise ValueError(f"manifest split mismatch: expected {split}, got {declared_split}")
            source_manifests.append(
                {
                    "split": split,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "sample_count": len(manifest["samples"]),
                    "lmdb_path": str(lmdb_path),
                }
            )
            env_key = str(lmdb_path)
            if env_key not in envs:
                envs[env_key] = _open_lmdb(lmdb_path)
            env = envs[env_key]
            samples = manifest["samples"]
            if limit_per_split is not None:
                samples = samples[: int(limit_per_split)]
            with env.begin(write=False) as transaction:
                for sample in samples:
                    key = str(sample["lmdb_key"])
                    raw = transaction.get(key.encode("utf-8"))
                    if raw is None:
                        raise KeyError(f"feature LMDB key not found: {key}")
                    item = pickle.loads(bytes(raw))
                    protein_features = item.get("protein_features")
                    if protein_features is None:
                        raise ValueError(f"protein_features missing for {key}")
                    for chain_id, chain in ordered_chain_items(protein_features):
                        sequence = normalize_query_sequence(chain.get("sequence", ""))
                        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                        record = records_by_sha.get(digest)
                        if record is None:
                            record = {
                                "sequence_id": "seq_" + digest,
                                "sequence_sha256": digest,
                                "sequence": sequence,
                                "length": len(sequence),
                                "occurrences": [],
                            }
                            records_by_sha[digest] = record
                        elif record["sequence"] != sequence:
                            raise RuntimeError("SHA256 collision between protein sequences")
                        record["occurrences"].append(
                            {"split": split, "lmdb_key": key, "chain_id": str(chain_id)}
                        )
                        occurrence_count += 1
    finally:
        for env in envs.values():
            env.close()

    records = sorted(records_by_sha.values(), key=lambda item: item["sequence_sha256"])
    payload = {
        "status": "complete",
        "schema_version": SEQUENCE_MANIFEST_SCHEMA_VERSION,
        "version": VERSION,
        "source_manifests": source_manifests,
        "sequence_count": len(records),
        "occurrence_count": occurrence_count,
        "sequences": records,
    }
    _atomic_write_text(output_dir / "sequence_manifest.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(output_dir / "queries.fasta", _fasta_text(records))
    result = {
        "status": "complete",
        "output": str(output_dir),
        "sequence_count": len(records),
        "occurrence_count": occurrence_count,
        "manifest_sha256": sha256_file(output_dir / "sequence_manifest.json"),
        "fasta_sha256": sha256_file(output_dir / "queries.fasta"),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _manifest_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("manifest must use SPLIT=PATH")
    split, path = value.split("=", 1)
    if not split or not path:
        raise argparse.ArgumentTypeError("manifest must use SPLIT=PATH")
    return split, path


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=_manifest_spec, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-split", type=int)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    build_sequence_manifest(args.manifest, args.output, limit_per_split=args.limit_per_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
