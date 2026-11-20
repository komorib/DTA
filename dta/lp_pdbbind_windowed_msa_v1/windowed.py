"""Validate a complete A3M before slicing the same columns in every row."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dta.msa_transformer_smoke_v1.extract import encode_prepared_msa
from dta.msa_transformer_smoke_v1.msa import (
    MSARow,
    PreparedMSA,
    _ALIGNED_CHARACTERS,
    _RAW_A3M_CHARACTERS,
    _TARGET_CHARACTERS,
    _read_fasta_rows,
    normalize_a3m_sequence,
    sha256_file,
    sha256_text,
)

from . import WINDOW_COLUMNS, WINDOW_POLICY, WINDOW_STRIDE
from .selector import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DIVERSE_SELECTION_POLICY,
    LEGACY_SELECTION_POLICY,
    select_diverse_valid_rows,
)

LEGACY_ROW_VALIDATION = "first_identifier_occurrence_v1"
REPAIRED_ROW_VALIDATION = "first_valid_identifier_occurrence_v2"
ROW_VALIDATION_POLICIES = (LEGACY_ROW_VALIDATION, REPAIRED_ROW_VALIDATION)


@dataclass(frozen=True)
class FullAlignment:
    rows: tuple
    query_residue_columns: tuple
    query_sequence: str
    mode: str
    metadata: dict

    @property
    def width(self):
        return len(self.rows[0].aligned_sequence)


@dataclass(frozen=True)
class MSAWindow:
    start: int
    stop: int
    query_residue_indices: tuple
    prepared: PreparedMSA
    query_only_fallback: bool


def prepare_full_alignment(
    a3m_path, expected_query, mode, maximum_depth=8, *,
    selection_policy=LEGACY_SELECTION_POLICY,
    candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
    depth_limit=8,
    row_validation_policy=LEGACY_ROW_VALIDATION,
):
    """Select homologs once on the full alignment; never truncate the query."""

    if mode not in ("query_only", "aligned_msa"):
        raise ValueError("unknown MSA mode")
    # Existing callers retain their exact depth-8 contract. The isolated
    # depth-32 experiment must opt in explicitly; selection math is unchanged.
    if type(depth_limit) is not int or depth_limit not in (8, 32):
        raise ValueError("depth_limit must be 8 or 32")
    if not 1 <= int(maximum_depth) <= depth_limit:
        raise ValueError("maximum_depth must be between 1 and %d" % depth_limit)
    if selection_policy not in (LEGACY_SELECTION_POLICY, DIVERSE_SELECTION_POLICY):
        raise ValueError("unknown MSA selection policy")
    if row_validation_policy not in ROW_VALIDATION_POLICIES:
        raise ValueError("unknown MSA row validation policy")
    expected_query = str(expected_query).strip().upper()
    if not expected_query or not set(expected_query).issubset(_TARGET_CHARACTERS):
        raise ValueError("expected query is empty or contains unsupported residues")
    path = Path(a3m_path).expanduser().resolve()
    raw_rows = _read_fasta_rows(path)
    query_id, raw_query = raw_rows[0]
    if not raw_query or not set(raw_query).issubset(_RAW_A3M_CHARACTERS):
        raise ValueError("A3M query contains unsupported characters")
    query = normalize_a3m_sequence(raw_query)
    if not query or not set(query).issubset(_ALIGNED_CHARACTERS):
        raise ValueError("normalized A3M query is empty or invalid")
    if query.replace("-", "") != expected_query:
        raise ValueError("A3M query does not match the expected complete chain")

    valid = [MSARow(query_id, query)]
    source_indices = [0]
    seen_ids = {query_id: raw_query}
    seen_sequences = {query}
    invalid_ids, recovered_ids = set(), []
    rejected = {
        "empty": 0,
        "invalid": 0,
        "misaligned": 0,
        "duplicate_identifier": 0,
        "conflicting_identifier": 0,
        "duplicate_sequence": 0,
    }
    for source_index, (identifier, raw) in enumerate(raw_rows[1:], start=1):
        if identifier in seen_ids:
            category = "duplicate_identifier" if seen_ids[identifier] == raw else "conflicting_identifier"
            rejected[category] += 1
            continue
        if row_validation_policy == LEGACY_ROW_VALIDATION:
            seen_ids[identifier] = raw
        if not raw:
            invalid_ids.add(identifier)
            rejected["empty"] += 1
            continue
        if not set(raw).issubset(_RAW_A3M_CHARACTERS):
            invalid_ids.add(identifier)
            rejected["invalid"] += 1
            continue
        aligned = normalize_a3m_sequence(raw)
        if not aligned or not aligned.replace("-", ""):
            invalid_ids.add(identifier)
            rejected["empty"] += 1
            continue
        if not set(aligned).issubset(_ALIGNED_CHARACTERS):
            invalid_ids.add(identifier)
            rejected["invalid"] += 1
            continue
        if len(aligned) != len(query):
            invalid_ids.add(identifier)
            rejected["misaligned"] += 1
            continue
        if row_validation_policy == REPAIRED_ROW_VALIDATION:
            # Only malformed rows are recoverable. A syntactically valid row
            # reserves its ID even if it duplicates query/another sequence.
            seen_ids[identifier] = raw
            if identifier in invalid_ids:
                recovered_ids.append(identifier)
        if aligned in seen_sequences:
            rejected["duplicate_sequence"] += 1
            continue
        seen_sequences.add(aligned)
        valid.append(MSARow(identifier, aligned))
        source_indices.append(source_index)

    # A true absence of distinct homologs is a recorded query-only fallback.
    # Malformed source rows cannot masquerade as that absence.
    if len(valid) == 1 and any(rejected[key] for key in ("invalid", "misaligned", "conflicting_identifier")):
        raise ValueError("A3M has malformed homologs and no valid homolog")
    selection_metadata = {}
    if selection_policy == DIVERSE_SELECTION_POLICY:
        selected, selection_metadata = select_diverse_valid_rows(
            valid, source_indices,
            int(maximum_depth) - 1 if mode == "aligned_msa" else 0,
            candidate_pool_size,
        )
    else:
        selected = tuple(valid[: int(maximum_depth)] if mode == "aligned_msa" else valid[:1])
    columns = tuple(i for i, char in enumerate(query) if char != "-")
    if len(columns) != len(expected_query):
        raise ValueError("query residue mapping is incomplete")
    metadata = {
        "a3m_path": str(path),
        "a3m_sha256": sha256_file(path),
        "query_sha256": sha256_text(expected_query),
        "mode": mode,
        "alignment_width": len(query),
        "query_gap_count": query.count("-"),
        "raw_depth": len(raw_rows),
        "valid_homolog_count": len(valid) - 1,
        "selected_source_identifiers": [row.identifier for row in selected],
        "selection_policy": selection_policy,
        "normalization_policy": "remove_lowercase_and_dots_keep_match_column_gaps_v1",
        "rejected_rows": rejected,
        "no_valid_homolog": len(valid) == 1,
        "window_policy": WINDOW_POLICY,
        **selection_metadata,
    }
    if row_validation_policy == REPAIRED_ROW_VALIDATION:
        metadata.update(row_validation_policy=row_validation_policy,
                        recovered_invalid_identifiers=recovered_ids)
    return FullAlignment(selected, columns, expected_query, mode, metadata)


def window_bounds(width, columns=WINDOW_COLUMNS, stride=WINDOW_STRIDE):
    width, columns, stride = int(width), int(columns), int(stride)
    if width <= 0 or columns <= 0 or stride <= 0 or stride > columns:
        raise ValueError("invalid alignment width or window policy")
    if width <= columns:
        return ((0, width),)
    starts = list(range(0, width - columns + 1, stride))
    if starts[-1] != width - columns:
        starts.append(width - columns)
    return tuple((start, start + columns) for start in starts)


def prepare_windows(alignment):
    """Apply fixed boundaries to all selected rows, including query gaps."""

    residue_by_column = {
        column: index for index, column in enumerate(alignment.query_residue_columns)
    }
    windows = []
    for start, stop in window_bounds(alignment.width):
        columns = tuple(column for column in alignment.query_residue_columns if start <= column < stop)
        if not columns:
            continue  # A gap-only window cannot contribute a query representation.
        indices = tuple(residue_by_column[column] for column in columns)
        rows = [MSARow(row.identifier, row.aligned_sequence[start:stop]) for row in alignment.rows]
        if alignment.mode == "aligned_msa":
            rows = rows[:1] + [row for row in rows[1:] if row.aligned_sequence.replace("-", "")]
        fallback = alignment.mode == "aligned_msa" and len(rows) == 1
        query_sequence = "".join(alignment.query_sequence[index] for index in indices)
        prepared = PreparedMSA(
            rows=tuple(rows),
            query_residue_columns=tuple(column - start for column in columns),
            query_sequence=query_sequence,
            metadata={
                "window_start": start,
                "window_stop": stop,
                "mode": alignment.mode,
                "selected_source_identifiers": [row.identifier for row in rows],
                "query_only_fallback": fallback,
            },
        )
        if any(len(row.aligned_sequence) != stop - start for row in prepared.rows):
            raise ValueError("windowed MSA rows have unequal widths")
        windows.append(MSAWindow(start, stop, indices, prepared, fallback))
    if not windows:
        raise ValueError("no window contains a query residue")
    return tuple(windows)


def merge_window_embeddings(windows, embeddings, query_length):
    """Average every occurrence of a query residue across overlapping windows."""

    if len(windows) != len(embeddings) or not windows:
        raise ValueError("window embeddings do not match the window list")
    total = None
    counts = np.zeros(int(query_length), dtype=np.int64)
    for window, value in zip(windows, embeddings):
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[0] != len(window.query_residue_indices):
            raise ValueError("window embedding shape does not match query residues")
        if not np.isfinite(array).all():
            raise ValueError("window embedding contains non-finite values")
        if total is None:
            total = np.zeros((int(query_length), array.shape[1]), dtype=np.float64)
        if array.shape[1] != total.shape[1]:
            raise ValueError("window embedding widths differ")
        indices = np.asarray(window.query_residue_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= int(query_length)) or np.unique(indices).size != indices.size:
            raise ValueError("window query residue indices are invalid")
        total[indices] += array.astype(np.float64)
        counts[indices] += 1
    if np.any(counts == 0):
        raise ValueError("window embeddings do not cover every query residue")
    return (total / counts[:, None]).astype(np.float32), counts


def encode_windowed_alignment(model, alphabet, alignment, device, maximum_tensor_tokens=8192):
    """Use the restored official smoke encoder on each prevalidated window."""

    windows = prepare_windows(alignment)
    outputs = []
    runs = []
    for window in windows:
        embedding, runtime = encode_prepared_msa(
            model, alphabet, window.prepared, device, maximum_tensor_tokens
        )
        outputs.append(embedding)
        runs.append({
            "start": window.start,
            "stop": window.stop,
            "query_only_fallback": window.query_only_fallback,
            "selected_source_identifiers": window.prepared.metadata["selected_source_identifiers"],
            **runtime,
        })
    merged, counts = merge_window_embeddings(windows, outputs, len(alignment.query_sequence))
    return merged, {
        "window_policy": WINDOW_POLICY,
        "alignment": alignment.metadata,
        "windows": runs,
        "overlap_count_min": int(counts.min()),
        "overlap_count_max": int(counts.max()),
        "embedding_shape": list(merged.shape),
    }


__all__ = [
    "FullAlignment",
    "MSAWindow",
    "encode_windowed_alignment",
    "merge_window_embeddings",
    "prepare_full_alignment",
    "prepare_windows",
    "window_bounds",
]
