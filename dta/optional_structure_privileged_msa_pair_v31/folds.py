"""Build deterministic, exact-sequence-grouped V31 teacher folds.

This module deliberately has no validation/test-manifest argument.  It is the
first guardrail of the privileged-MSA teacher: each outer holdout sequence is
absent from the corresponding teacher fit, its inner checkpoint selection,
and its PCA/ExtraTrees scalar readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from dta.optional_structure_staged_v22.data import (
    manifest_fingerprint,
    read_jsonl,
    sha256_file,
)


VERSION = "optional-structure-privileged-msa-pair-folds-v31"
SCHEMA_VERSION = 1


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value):
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _rows_sha256(rows):
    # sample IDs and keys make a compact, stable lineage record without
    # pretending that JSON whitespace is semantically meaningful.
    value = [
        (str(row["sample_id"]), str(row["structure_key"]), str(row["sequence_id"]))
        for row in rows
    ]
    return _json_sha256(value)


def lp_train_rows(path, limit=None):
    """Return LP-PDBBind train rows only, preserving manifest order."""

    rows = [row for row in read_jsonl(path) if str(row.get("dataset")) == "LP_PDBBind"]
    if any(str(row.get("split")) != "train" for row in rows):
        raise ValueError("V31 fold source must contain LP_PDBBind train rows only")
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("V31 fold limit must be positive")
        rows = rows[: int(limit)]
    if len(rows) < 3:
        raise ValueError("V31 needs at least three LP train rows")
    required = ("sample_id", "structure_key", "sequence_id", "label")
    for row in rows:
        missing = [name for name in required if not str(row.get(name, ""))]
        if missing:
            raise ValueError("V31 fold row misses %s" % ", ".join(missing))
    for field in ("sample_id", "structure_key"):
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError("V31 fold source %s values are not unique" % field)
    return rows


def validate_msa_train_manifest(path, rows):
    """Check that V21 LP rows can be represented by the V18 cache manifest."""

    path = Path(path).expanduser().resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    if str(source.get("split")) != "train" or not isinstance(source.get("samples"), list):
        raise ValueError("V31 MSA source must be a train manifest with samples")
    samples = source["samples"]
    keys = [str(sample.get("lmdb_key", "")) for sample in samples]
    labels = np.asarray([float(sample["label"]) for sample in samples], dtype=np.float64)
    if not all(keys) or len(keys) != len(set(keys)) or not np.isfinite(labels).all():
        raise ValueError("V31 MSA source keys/labels are invalid")
    requested = [str(row["structure_key"]) for row in rows]
    positions = {key: index for index, key in enumerate(keys)}
    if len(requested) > len(keys) or not set(requested).issubset(set(keys)):
        raise ValueError("V31 LP rows do not map exactly into the MSA train manifest")
    order = np.asarray([positions[key] for key in requested], dtype=np.int64)
    expected = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    if not np.allclose(labels[order], expected, rtol=0.0, atol=1.0e-6):
        raise ValueError("V31 MSA source labels differ from LP train manifest")
    return path


def _tie_break(seed, value):
    return hashlib.sha256((str(seed) + "|" + str(value)).encode("utf-8")).hexdigest()


def balanced_sequence_folds(rows, folds, seed):
    """Assign whole sequence groups using deterministic size-balanced greedy bins."""

    folds = int(folds)
    if folds < 2:
        raise ValueError("V31 outer fold count must be at least two")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["sequence_id"])].append(index)
    if len(grouped) < folds:
        raise ValueError("V31 needs at least one unique sequence per outer fold")
    groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), _tie_break(seed, item[0]), item[0]),
    )
    loads = [0 for _ in range(folds)]
    group_counts = [0 for _ in range(folds)]
    assignment = {}
    for sequence_id, indices in groups:
        target = min(range(folds), key=lambda fold: (loads[fold], group_counts[fold], fold))
        assignment[sequence_id] = int(target)
        loads[target] += len(indices)
        group_counts[target] += 1
    if min(group_counts) <= 0:
        raise ValueError("V31 greedy sequence fold assignment produced an empty fold")
    row_folds = np.asarray([assignment[str(row["sequence_id"])] for row in rows], dtype=np.int64)
    return row_folds, assignment, loads, group_counts


def inner_validation_sequences(rows, outer_fold_ids, fold, fraction, seed):
    """Make a deterministic inner-valid split entirely inside an outer fit."""

    fraction = float(fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("V31 inner validation fraction must be in (0, 0.5)")
    allowed = [row for index, row in enumerate(rows) if int(outer_fold_ids[index]) != int(fold)]
    grouped = defaultdict(list)
    for row in allowed:
        grouped[str(row["sequence_id"])].append(row)
    if len(grouped) < 2:
        raise ValueError("V31 outer fit needs at least two sequence groups for inner selection")
    target = max(1, int(round(sum(len(value) for value in grouped.values()) * fraction)))
    ordered = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), _tie_break("%s|%s" % (seed, fold), item[0]), item[0]),
    )
    selected, current = set(), 0
    for sequence_id, members in ordered:
        without = abs(current - target)
        with_member = abs(current + len(members) - target)
        if current == 0 or with_member <= without:
            selected.add(sequence_id)
            current += len(members)
    if not selected or len(selected) == len(grouped):
        raise ValueError("V31 inner sequence split is degenerate")
    return selected


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_manifest(path, rows):
    """Write the compact JSON schema consumed by ``MSAPairComplexDataset``."""

    value = {
        "dataset_type": "PDBbind_chain_esm_residue_DTA",
        "task": "LP_PDBBind",
        "split": "train",
        "split_method": "v31_sequence_grouped_outer_inner",
        "samples": [
            {
                "lmdb_key": str(row["structure_key"]),
                "label": float(row["label"]),
                "pdb_id": str(row.get("pdb_id", "")),
            }
            for row in rows
        ],
    }
    _atomic_json(path, value)


def build_fold_manifests(args):
    source = Path(args.train_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("V31 fold output exists; choose a new FOLD_ID")
    rows = lp_train_rows(source, args.limit)
    msa_train_manifest = validate_msa_train_manifest(args.msa_train_manifest, rows)
    row_folds, assignment, loads, group_counts = balanced_sequence_folds(rows, args.folds, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s.building-" % output.name, dir=str(output.parent)))
    try:
        folds = []
        for fold in range(int(args.folds)):
            directory = temporary / ("fold_%d" % fold)
            directory.mkdir(parents=True, exist_ok=False)
            holdout = [row for index, row in enumerate(rows) if int(row_folds[index]) == fold]
            outer_fit = [row for index, row in enumerate(rows) if int(row_folds[index]) != fold]
            inner_sequences = inner_validation_sequences(
                rows, row_folds, fold, args.inner_valid_fraction, args.seed
            )
            inner_valid = [row for row in outer_fit if str(row["sequence_id"]) in inner_sequences]
            inner_train = [row for row in outer_fit if str(row["sequence_id"]) not in inner_sequences]
            if not holdout or not inner_train or not inner_valid:
                raise RuntimeError("V31 generated an empty outer/inner partition")
            holdout_sequences = {str(row["sequence_id"]) for row in holdout}
            fit_sequences = {str(row["sequence_id"]) for row in outer_fit}
            if holdout_sequences & fit_sequences:
                raise RuntimeError("V31 outer sequence split overlaps")
            if {str(row["sequence_id"]) for row in inner_train} & {
                str(row["sequence_id"]) for row in inner_valid
            }:
                raise RuntimeError("V31 inner sequence split overlaps")
            paths = {
                "outer_fit": directory / "outer_fit.json",
                "inner_train": directory / "inner_train.json",
                "inner_valid": directory / "inner_valid.json",
                "outer_holdout": directory / "outer_holdout.json",
            }
            for name, path in paths.items():
                _write_manifest(path, {"outer_fit": outer_fit, "inner_train": inner_train, "inner_valid": inner_valid, "outer_holdout": holdout}[name])
            folds.append(
                {
                    "fold": fold,
                    "directory": str(output / directory.name),
                    "counts": {
                        "outer_fit": len(outer_fit),
                        "inner_train": len(inner_train),
                        "inner_valid": len(inner_valid),
                        "outer_holdout": len(holdout),
                    },
                    "sequence_group_counts": {
                        "outer_fit": len(fit_sequences),
                        "inner_train": len({str(row["sequence_id"]) for row in inner_train}),
                        "inner_valid": len(inner_sequences),
                        "outer_holdout": len(holdout_sequences),
                    },
                    "manifests": {name: str(output / directory.name / path.name) for name, path in paths.items()},
                    "manifest_sha256": {name: sha256_file(path) for name, path in paths.items()},
                    "row_sha256": {
                        "outer_fit": _rows_sha256(outer_fit),
                        "inner_train": _rows_sha256(inner_train),
                        "inner_valid": _rows_sha256(inner_valid),
                        "outer_holdout": _rows_sha256(holdout),
                    },
                }
            )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "version": VERSION,
            "status": "built",
            "output": str(output),
            "train_manifest": str(source),
            "train_manifest_sha256": manifest_fingerprint(source),
            "msa_train_manifest": str(msa_train_manifest),
            "msa_train_manifest_sha256": sha256_file(msa_train_manifest),
            "labels_scope": "train_only",
            "official_valid_manifest_read": False,
            "test_manifest_read": False,
            "limit": None if args.limit is None else int(args.limit),
            "fold_group": "sequence_id",
            "fold_allocation": "size_desc_sha256_tiebreak_minimum_load_greedy",
            "folds": int(args.folds),
            "seed": int(args.seed),
            "inner_valid_fraction": float(args.inner_valid_fraction),
            "sample_count": len(rows),
            "sequence_group_count": len(assignment),
            "outer_row_loads": [int(value) for value in loads],
            "outer_group_loads": [int(value) for value in group_counts],
            "source_rows_sha256": _rows_sha256(rows),
            "sequence_to_outer_fold_sha256": _json_sha256(assignment),
            "sequence_to_outer_fold": assignment,
            "fold_artifacts": folds,
        }
        _atomic_json(temporary / "folds.json", metadata)
        os.replace(str(temporary), str(output))
    except BaseException:
        # Leave the interrupted temporary directory for inspection; it cannot
        # be mistaken for a completed FOLD_ID because it is never published.
        raise
    # The full sequence-to-fold mapping is sealed in folds.json; do not flood
    # a terminal/log with thousands of hashes.
    print(
        json.dumps(
            {
                "version": metadata["version"],
                "status": metadata["status"],
                "output": metadata["output"],
                "sample_count": metadata["sample_count"],
                "sequence_group_count": metadata["sequence_group_count"],
                "outer_row_loads": metadata["outer_row_loads"],
                "outer_group_loads": metadata["outer_group_loads"],
                "folds_sha256": sha256_file(output / "folds.json"),
                "official_valid_manifest_read": False,
                "test_manifest_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return metadata


def load_fold_metadata(fold_root, train_manifest, require_full=False):
    fold_root = Path(fold_root).expanduser().resolve()
    path = fold_root / "folds.json"
    if not path.is_file():
        raise FileNotFoundError("V31 fold root lacks folds.json")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("version") != VERSION:
        raise ValueError("V31 fold schema/version mismatch")
    if metadata.get("status") != "built" or metadata.get("labels_scope") != "train_only":
        raise ValueError("V31 fold root is not a completed train-only artifact")
    if metadata.get("official_valid_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
        raise ValueError("V31 fold root accessed official validation/test")
    if metadata.get("train_manifest_sha256") != manifest_fingerprint(train_manifest):
        raise ValueError("V31 fold root train-manifest lineage mismatch")
    if require_full and metadata.get("limit") is not None:
        raise ValueError("V31 formal teacher cache cannot use limited fold manifests")
    folds = int(metadata.get("folds", 0))
    if folds < 2 or len(metadata.get("fold_artifacts", [])) != folds:
        raise ValueError("V31 fold root has invalid fold metadata")
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--msa-train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-valid-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv=None):
    return 0 if build_fold_manifests(build_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    main()
