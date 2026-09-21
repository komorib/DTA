"""Build a leak-free, fixed-split KIBA binary benchmark input bundle.

The source is the repository's decoded ``data/KIBA`` dataset.  This module
does not train a model and never reads labels from the test split for model
selection; it only materializes the already-defined train/valid/test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping

import numpy as np

from dta.sequence_smiles_binary_probe_v1.core import sha256_file, split_fingerprints


VERSION = "external-benchmark-kiba-binary-v1"
DEFAULT_THRESHOLD = 12.1
SPLITS = ("train", "valid", "test")
_INDEX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_dictionary(path: Path) -> List[str]:
    tokens = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                tokens.append(line.rsplit(" ", 1)[0])
    return tokens


def _iter_mmap_sequences(prefix: Path, dictionary: List[str]) -> Iterator[str]:
    """Decode one mmap split lazily to avoid materialising all KIBA strings."""
    raw_index = (prefix.with_suffix(".idx")).read_bytes()
    if not raw_index.startswith(_INDEX_MAGIC) or len(raw_index) < 26:
        raise ValueError("invalid mmap index: %s" % prefix)
    dtype_code = raw_index[17]
    if dtype_code not in _DTYPES:
        raise ValueError("unsupported mmap dtype code %s" % dtype_code)
    dtype = np.dtype(_DTYPES[dtype_code]).newbyteorder("<")
    count = struct.unpack_from("<Q", raw_index, 18)[0]
    sizes_offset = 26
    pointers_offset = sizes_offset + 4 * count
    sizes = np.frombuffer(raw_index, dtype="<i4", count=count, offset=sizes_offset)
    pointers = np.frombuffer(raw_index, dtype="<i8", count=count, offset=pointers_offset)
    binary = np.memmap(prefix.with_suffix(".bin"), mode="r", dtype=np.uint8)
    for size, pointer in zip(sizes.tolist(), pointers.tolist()):
        token_ids = np.frombuffer(binary, dtype=dtype, count=int(size), offset=int(pointer))
        pieces = []
        for token_id in token_ids.tolist():
            if token_id in (0, 1, 2):
                continue
            dictionary_index = int(token_id) - 4
            pieces.append(dictionary[dictionary_index] if 0 <= dictionary_index < len(dictionary) else "?")
        yield "".join(pieces)


def _iter_split(kiba_root: Path, split: str) -> Iterator[tuple]:
    mol_dictionary = _read_dictionary(kiba_root / "input0" / "dict.txt")
    protein_dictionary = _read_dictionary(kiba_root / "input1" / "dict.txt")
    smiles_iter = _iter_mmap_sequences(kiba_root / "input0" / split, mol_dictionary)
    protein_iter = _iter_mmap_sequences(kiba_root / "input1" / split, protein_dictionary)
    labels = np.loadtxt(kiba_root / "label" / (split + ".label"), dtype=np.float64, ndmin=1)
    for index, (smiles, sequence) in enumerate(zip(smiles_iter, protein_iter)):
        if index >= labels.shape[0]:
            raise ValueError("more encoded pairs than labels in %s" % split)
        yield smiles, sequence, float(labels[index])
    if index + 1 != labels.shape[0]:
        raise ValueError("encoded pair/label count mismatch in %s" % split)


def _pair_ids_for_split(kiba_root: Path, split: str) -> set:
    return {_stable_id(sequence + "\0" + smiles) for smiles, sequence, _ in _iter_split(kiba_root, split)}


def _row(dataset_split: str, index: int, smiles: str, sequence: str, affinity: float, threshold: float) -> dict:
    pair_key = sequence + "\0" + smiles
    return {
        "dataset": "KIBA",
        "split": dataset_split,
        "index": int(index),
        "sample_id": "KIBA:%s:%d" % (dataset_split, index),
        "pair_id": _stable_id(pair_key),
        "ligand_id": _stable_id(smiles),
        "sequence_id": _stable_id(sequence),
        "smiles": str(smiles),
        "sequence": str(sequence),
        "affinity": float(affinity),
        "label": int(float(affinity) >= float(threshold)),
        "affinity_threshold": float(threshold),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return sha256_file(path)


def _write_csv(path: Path, rows: List[Mapping]) -> str:
    fields = ["sample_id", "pair_id", "smiles", "sequence", "affinity", "label", "affinity_threshold"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return sha256_file(path)


def _write_graphdta_csv(path: Path, rows: List[Mapping]) -> str:
    fields = ["compound_iso_smiles", "target_sequence", "affinity", "label", "sample_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "compound_iso_smiles": row["smiles"],
                    "target_sequence": row["sequence"],
                    "affinity": row["affinity"],
                    "label": row["label"],
                    "sample_id": row["sample_id"],
                }
            )
    return sha256_file(path)


def _write_psichic_csv(path: Path, rows: List[Mapping]) -> str:
    fields = ["Protein", "Ligand", "regression_label", "classification_label", "sample_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Protein": row["sequence"],
                    "Ligand": row["smiles"],
                    "regression_label": row["affinity"],
                    "classification_label": row["label"],
                    "sample_id": row["sample_id"],
                }
            )
    return sha256_file(path)


def _write_text(path: Path, rows: List[Mapping], mode: str) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if mode == "transformercpi":
                handle.write("%s %s %d\n" % (row["smiles"], row["sequence"], row["label"]))
            elif mode == "fmca":
                handle.write("%s %s %d\n" % (row["smiles"], row["sequence"], row["label"]))
            elif mode == "graphormer":
                handle.write(
                    "%s %s %s %d\n"
                    % (row["ligand_id"], row["sequence_id"], row["sequence"], row["label"])
                )
            else:
                raise ValueError("unknown text mode: %s" % mode)
    return sha256_file(path)


def _audit(rows_by_split: Mapping[str, List[Mapping]]) -> dict:
    pair_sets = {split: {row["pair_id"] for row in rows} for split, rows in rows_by_split.items()}
    cross_overlap = {
        "%s_%s" % (left, right): len(pair_sets[left] & pair_sets[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    return {
        "counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "positive_counts": {
            split: int(sum(int(row["label"]) for row in rows)) for split, rows in rows_by_split.items()
        },
        "positive_rates": {
            split: float(sum(int(row["label"]) for row in rows) / max(len(rows), 1))
            for split, rows in rows_by_split.items()
        },
        "unique_smiles": {split: len({row["smiles"] for row in rows}) for split, rows in rows_by_split.items()},
        "unique_sequences": {
            split: len({row["sequence"] for row in rows}) for split, rows in rows_by_split.items()
        },
        "duplicate_pair_counts": {
            split: len(rows) - len(pair_sets[split]) for split, rows in rows_by_split.items()
        },
        "cross_split_pair_overlap": cross_overlap,
        "cross_split_overlap_free": all(value == 0 for value in cross_overlap.values()),
    }


def build(
    kiba_root: Path,
    output: Path,
    threshold: float,
    limit: int | None = None,
    splits: Iterable[str] | None = None,
) -> dict:
    kiba_root = kiba_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output: %s" % output)
    output.mkdir(parents=True, exist_ok=True)

    selected_splits = tuple(splits or SPLITS)
    unknown = set(selected_splits) - set(SPLITS)
    if unknown:
        raise ValueError("unknown split(s): %s" % sorted(unknown))
    files = {}
    audit_by_split = {}
    pair_sets = {}
    protected_pairs = set()
    if "train" in selected_splits:
        for protected_split in ("valid", "test"):
            protected_pairs.update(_pair_ids_for_split(kiba_root, protected_split))
    for split in selected_splits:
        common_dir = output / "common"
        graphdta_dir = output / "graphdta"
        transformercpi_dir = output / "transformercpi"
        graphormer_dir = output / "graphormer"
        psichic_dir = output / "psichic"
        fmca_dir = output / "fmca"
        for directory in (common_dir, graphdta_dir, transformercpi_dir, graphormer_dir, psichic_dir, fmca_dir):
            directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "common/%s.jsonl" % split: common_dir / (split + ".jsonl"),
            "flat/%s.csv" % split: output / ("flat_%s.csv" % split),
            "graphdta/%s.csv" % split: graphdta_dir / (split + ".csv"),
            "transformercpi/%s.txt" % split: transformercpi_dir / (split + ".txt"),
            "graphormer/%s.txt" % split: graphormer_dir / (split + ".txt"),
            "psichic/%s.csv" % split: psichic_dir / (split + ".csv"),
            "fmca/%s/data" % split: fmca_dir / (split + ".data"),
        }
        common_handle = paths["common/%s.jsonl" % split].open("w", encoding="utf-8")
        flat_handle = paths["flat/%s.csv" % split].open("w", encoding="utf-8", newline="")
        graphdta_handle = paths["graphdta/%s.csv" % split].open("w", encoding="utf-8", newline="")
        psichic_handle = paths["psichic/%s.csv" % split].open("w", encoding="utf-8", newline="")
        transformercpi_handle = paths["transformercpi/%s.txt" % split].open("w", encoding="utf-8")
        graphormer_handle = paths["graphormer/%s.txt" % split].open("w", encoding="utf-8")
        fmca_handle = paths["fmca/%s/data" % split].open("w", encoding="utf-8")
        flat_writer = csv.DictWriter(
            flat_handle,
            fieldnames=["sample_id", "pair_id", "smiles", "sequence", "affinity", "label", "affinity_threshold"],
        )
        graphdta_writer = csv.DictWriter(
            graphdta_handle,
            fieldnames=["compound_iso_smiles", "target_sequence", "affinity", "label", "sample_id"],
        )
        psichic_writer = csv.DictWriter(
            psichic_handle,
            fieldnames=["Protein", "Ligand", "regression_label", "classification_label", "sample_id"],
        )
        flat_writer.writeheader()
        graphdta_writer.writeheader()
        psichic_writer.writeheader()
        pair_set = set()
        smiles_set = set()
        sequence_set = set()
        positive_count = 0
        count = 0
        source_count = 0
        try:
            for index, (smiles, sequence, affinity) in enumerate(_iter_split(kiba_root, split)):
                if limit is not None and index >= int(limit):
                    break
                row = _row(split, index, smiles, sequence, affinity, threshold)
                source_count += 1
                if split == "train" and row["pair_id"] in protected_pairs:
                    continue
                common_handle.write(json.dumps(row, sort_keys=True) + "\n")
                flat_writer.writerow({field: row[field] for field in flat_writer.fieldnames})
                graphdta_writer.writerow(
                    {
                        "compound_iso_smiles": row["smiles"],
                        "target_sequence": row["sequence"],
                        "affinity": row["affinity"],
                        "label": row["label"],
                        "sample_id": row["sample_id"],
                    }
                )
                psichic_writer.writerow(
                    {
                        "Protein": row["sequence"],
                        "Ligand": row["smiles"],
                        "regression_label": row["affinity"],
                        "classification_label": row["label"],
                        "sample_id": row["sample_id"],
                    }
                )
                transformercpi_handle.write("%s %s %d\n" % (row["smiles"], row["sequence"], row["label"]))
                graphormer_handle.write(
                    "%s %s %s %d\n"
                    % (row["ligand_id"], row["sequence_id"], row["sequence"], row["label"])
                )
                fmca_handle.write("%s %s %d\n" % (row["smiles"], row["sequence"], row["label"]))
                count += 1
                positive_count += int(row["label"])
                pair_set.add(row["pair_id"])
                smiles_set.add(row["smiles"])
                sequence_set.add(row["sequence"])
        finally:
            for handle in (
                common_handle,
                flat_handle,
                graphdta_handle,
                psichic_handle,
                transformercpi_handle,
                graphormer_handle,
                fmca_handle,
            ):
                handle.close()
        files.update({name: sha256_file(path) for name, path in paths.items()})
        pair_sets[split] = pair_set
        audit_by_split[split] = {
            "count": count,
            "source_count": source_count,
            "removed_training_pair_overlap_count": source_count - count if split == "train" else 0,
            "positive_count": positive_count,
            "positive_rate": float(positive_count / max(count, 1)),
            "unique_smiles": len(smiles_set),
            "unique_sequences": len(sequence_set),
            "duplicate_pair_count": count - len(pair_set),
        }

    cross_overlap = {
        "%s_%s" % (left, right): len(pair_sets[left] & pair_sets[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
        if left in pair_sets and right in pair_sets
    }
    audit = {
        "counts": {split: audit_by_split[split]["count"] for split in selected_splits},
        "source_counts": {split: audit_by_split[split]["source_count"] for split in selected_splits},
        "removed_training_pair_overlap_counts": {
            split: audit_by_split[split]["removed_training_pair_overlap_count"] for split in selected_splits
        },
        "positive_counts": {split: audit_by_split[split]["positive_count"] for split in selected_splits},
        "positive_rates": {split: audit_by_split[split]["positive_rate"] for split in selected_splits},
        "unique_smiles": {split: audit_by_split[split]["unique_smiles"] for split in selected_splits},
        "unique_sequences": {split: audit_by_split[split]["unique_sequences"] for split in selected_splits},
        "duplicate_pair_counts": {split: audit_by_split[split]["duplicate_pair_count"] for split in selected_splits},
        "cross_split_pair_overlap": cross_overlap,
        "train_pair_overlap_free": all(
            cross_overlap.get(key, 0) == 0 for key in ("train_valid", "train_test")
        ),
        "cross_split_overlap_free": all(value == 0 for value in cross_overlap.values()),
    }
    metadata = {
        "version": VERSION,
        "task": "KIBA binary interaction classification",
        "source_root": str(kiba_root),
        "splits": list(selected_splits),
        "source_split_fingerprints": split_fingerprints(kiba_root, SPLITS),
        "affinity_threshold": float(threshold),
        "limit": limit,
        "selection_policy": "validation only; training duplicates against valid/test removed; valid/test duplicate pairs reported",
        "audit": audit,
        "files_sha256": files,
        "status": "built",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kiba-root", type=Path, default=Path("data/KIBA"))
    parser.add_argument("--output", type=Path, default=Path("data/ExternalBenchmarkKIBA_v1"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", action="append", choices=SPLITS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(build(args.kiba_root, args.output, args.threshold, args.limit, args.split), sort_keys=True))


if __name__ == "__main__":
    main()
