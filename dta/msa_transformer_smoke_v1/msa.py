"""Validate manifests and prepare deterministic MSA Transformer inputs."""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


ENCODER_NAME = "esm_msa1b_t12_100M_UR50S"
ESM_SOURCE_REVISION = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
NORMALIZATION_POLICY = "remove_lowercase_and_dots_keep_match_column_gaps_v1"
SELECTION_POLICY = "query_then_first_valid_unique_homologs_v1"
_RAW_A3M_CHARACTERS = frozenset(string.ascii_letters + "-.")
_ALIGNED_CHARACTERS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO-")
_TARGET_CHARACTERS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MSARow:
    identifier: str
    aligned_sequence: str


@dataclass(frozen=True)
class PreparedMSA:
    rows: Tuple[MSARow, ...]
    query_residue_columns: Tuple[int, ...]
    query_sequence: str
    metadata: Mapping[str, object]

    @property
    def depth(self):
        return len(self.rows)

    @property
    def width(self):
        return len(self.rows[0].aligned_sequence)

    def batch_converter_input(self):
        return [(row.identifier, row.aligned_sequence) for row in self.rows]


def _read_fasta_rows(path):
    rows = []
    identifier = None
    sequence_parts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    rows.append((identifier, "".join(sequence_parts)))
                header = line[1:].strip().split(None, 1)
                if not header:
                    raise ValueError("empty A3M identifier at line %d in %s" % (line_number, path))
                identifier = header[0]
                sequence_parts = []
            else:
                if identifier is None:
                    raise ValueError(
                        "A3M sequence precedes its header at line %d in %s"
                        % (line_number, path)
                    )
                sequence_parts.append(line)
    if identifier is not None:
        rows.append((identifier, "".join(sequence_parts)))
    if not rows:
        raise ValueError("A3M is empty: %s" % path)
    return rows


def normalize_a3m_sequence(sequence):
    """Remove A3M insertions/dots while preserving match-column gaps."""

    return "".join(
        character.upper()
        for character in str(sequence)
        if character != "." and not character.islower()
    )


def _positive_limit(value, name, hard_maximum):
    value = int(value)
    if value <= 0:
        raise ValueError("%s must be positive" % name)
    if value > int(hard_maximum):
        raise ValueError("%s=%d exceeds smoke maximum %d" % (name, value, hard_maximum))
    return value


def prepare_msa(
    a3m_path,
    expected_query,
    mode,
    maximum_depth=8,
    maximum_columns=512,
    maximum_tensor_tokens=8192,
    prepend_bos=True,
    append_eos=False,
):
    """Prepare one query-only or real aligned-MSA input without loading a model."""

    if mode not in {"query_only", "aligned_msa"}:
        raise ValueError("unknown MSA feature mode: %s" % mode)
    maximum_depth = _positive_limit(maximum_depth, "maximum_depth", 8)
    maximum_columns = _positive_limit(maximum_columns, "maximum_columns", 512)
    maximum_tensor_tokens = _positive_limit(
        maximum_tensor_tokens, "maximum_tensor_tokens", 8192
    )
    expected_query = str(expected_query).strip().upper()
    if not expected_query or not set(expected_query).issubset(_TARGET_CHARACTERS):
        raise ValueError("expected query is empty or contains unsupported residues")

    path = Path(a3m_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("MSA A3M does not exist: %s" % path)
    raw_rows = _read_fasta_rows(path)
    query_identifier, raw_query = raw_rows[0]
    if not raw_query or not set(raw_query).issubset(_RAW_A3M_CHARACTERS):
        raise ValueError("A3M query contains unsupported characters: %s" % path)
    query_aligned = normalize_a3m_sequence(raw_query)
    if not query_aligned or not set(query_aligned).issubset(_ALIGNED_CHARACTERS):
        raise ValueError("normalized A3M query is empty or invalid: %s" % path)
    if query_aligned.replace("-", "") != expected_query:
        raise ValueError("A3M query does not match the expected full target: %s" % path)
    if len(query_aligned) > maximum_columns:
        raise ValueError(
            "MSA match-column width %d exceeds maximum_columns=%d"
            % (len(query_aligned), maximum_columns)
        )

    valid_rows = [MSARow(query_identifier, query_aligned)]
    seen_identifiers = {query_identifier: raw_query}
    seen_sequences = {query_aligned}
    counts = {
        "raw_depth": len(raw_rows),
        "empty_homolog_count": 0,
        "invalid_homolog_count": 0,
        "misaligned_homolog_count": 0,
        "duplicate_identifier_count": 0,
        "conflicting_identifier_count": 0,
        "duplicate_query_count": 0,
        "duplicate_homolog_count": 0,
    }
    for identifier, raw_sequence in raw_rows[1:]:
        previous = seen_identifiers.get(identifier)
        if previous is not None:
            if previous != raw_sequence:
                counts["conflicting_identifier_count"] += 1
            else:
                counts["duplicate_identifier_count"] += 1
            continue
        seen_identifiers[identifier] = raw_sequence
        if not raw_sequence:
            counts["empty_homolog_count"] += 1
            continue
        if not set(raw_sequence).issubset(_RAW_A3M_CHARACTERS):
            counts["invalid_homolog_count"] += 1
            continue
        aligned = normalize_a3m_sequence(raw_sequence)
        if not aligned or not aligned.replace("-", ""):
            counts["empty_homolog_count"] += 1
            continue
        if not set(aligned).issubset(_ALIGNED_CHARACTERS):
            counts["invalid_homolog_count"] += 1
            continue
        if len(aligned) != len(query_aligned):
            counts["misaligned_homolog_count"] += 1
            continue
        if aligned == query_aligned:
            counts["duplicate_query_count"] += 1
            continue
        if aligned in seen_sequences:
            counts["duplicate_homolog_count"] += 1
            continue
        seen_sequences.add(aligned)
        valid_rows.append(MSARow(identifier, aligned))

    counts["validated_homolog_count"] = len(valid_rows) - 1
    if mode == "aligned_msa":
        selected_rows = valid_rows[:maximum_depth]
        if len(selected_rows) < 2:
            raise ValueError("aligned_msa requires at least one valid non-query homolog")
    else:
        selected_rows = valid_rows[:1]

    token_width = len(query_aligned) + int(bool(prepend_bos)) + int(bool(append_eos))
    tensor_tokens = len(selected_rows) * token_width
    if tensor_tokens > maximum_tensor_tokens:
        raise ValueError(
            "MSA tensor token count %d exceeds maximum_tensor_tokens=%d"
            % (tensor_tokens, maximum_tensor_tokens)
        )
    residue_columns = tuple(
        index for index, character in enumerate(query_aligned) if character != "-"
    )
    if len(residue_columns) != len(expected_query):
        raise ValueError("query residue mapping does not cover the complete target")

    metadata = {
        "a3m_path": str(path),
        "a3m_sha256": sha256_file(path),
        "normalization_policy": NORMALIZATION_POLICY,
        "selection_policy": SELECTION_POLICY,
        "mode": mode,
        "query_source_identifier": query_identifier,
        "selected_source_identifiers": [row.identifier for row in selected_rows],
        "selected_depth": len(selected_rows),
        "match_column_width": len(query_aligned),
        "query_residue_count": len(expected_query),
        "query_gap_count": query_aligned.count("-"),
        "query_residue_columns": list(residue_columns),
        "prepend_bos": bool(prepend_bos),
        "append_eos": bool(append_eos),
        "expected_tensor_tokens": tensor_tokens,
        "maximum_depth": maximum_depth,
        "maximum_columns": maximum_columns,
        "maximum_tensor_tokens": maximum_tensor_tokens,
        **counts,
    }
    return PreparedMSA(
        rows=tuple(selected_rows),
        query_residue_columns=residue_columns,
        query_sequence=expected_query,
        metadata=metadata,
    )


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError("invalid JSON at %s:%d" % (path, line_number)) from error
            rows.append((line_number, row))
    if not rows:
        raise ValueError("JSONL input is empty: %s" % path)
    return rows


def collect_targets(manifest_paths, maximum_targets=3):
    """Collect unique train/valid targets and reject test before model loading."""

    maximum_targets = _positive_limit(maximum_targets, "maximum_targets", 3)
    targets = {}
    sample_rows = {}
    source_manifests = []
    for manifest_path in manifest_paths:
        path = Path(manifest_path).expanduser().resolve()
        source_manifests.append({"path": str(path), "sha256": sha256_file(path)})
        for line_number, row in _read_jsonl(path):
            split = row.get("split")
            if split not in {"train", "valid"}:
                raise ValueError(
                    "manifest %s:%d sample %s has split %r; expected train or valid"
                    % (path, line_number, row.get("sample_id"), split)
                )
            missing = [
                key
                for key in ("sample_id", "dataset", "sequence_id", "sequence", "ligand_id")
                if not row.get(key)
            ]
            if missing:
                raise ValueError(
                    "manifest %s:%d is missing fields: %s"
                    % (path, line_number, ", ".join(missing))
                )
            sequence_id = str(row["sequence_id"])
            sequence = str(row["sequence"]).strip().upper()
            if not sequence or not set(sequence).issubset(_TARGET_CHARACTERS):
                raise ValueError("target %s contains unsupported residues" % sequence_id)
            previous = targets.get(sequence_id)
            if previous is not None and previous != sequence:
                raise ValueError("sequence_id maps to different target sequences: %s" % sequence_id)
            targets[sequence_id] = sequence
            sample_id = str(row["sample_id"])
            if sample_id in sample_rows:
                raise ValueError("duplicate sample_id in smoke manifests: %s" % sample_id)
            sample_rows[sample_id] = dict(row)
    if len(targets) > maximum_targets:
        raise ValueError(
            "smoke input contains %d targets; maximum_targets=%d"
            % (len(targets), maximum_targets)
        )
    return targets, sample_rows, source_manifests


def load_msa_manifest(path):
    """Load exact sequence_id-to-A3M mappings with unlabeled provenance."""

    path = Path(path).expanduser().resolve()
    mappings: Dict[str, Dict[str, object]] = {}
    for line_number, row in _read_jsonl(path):
        sequence_id = row.get("sequence_id")
        a3m_path = row.get("a3m_path")
        provenance = row.get("provenance")
        if not sequence_id or not a3m_path or not isinstance(provenance, dict):
            raise ValueError("invalid MSA manifest row at %s:%d" % (path, line_number))
        if provenance.get("uses_affinity_labels") is not False:
            raise ValueError(
                "MSA provenance must declare uses_affinity_labels=false at %s:%d"
                % (path, line_number)
            )
        sequence_id = str(sequence_id)
        specification = {
            "sequence_id": sequence_id,
            "a3m_path": str(Path(a3m_path).expanduser().resolve()),
            "provenance": dict(provenance),
        }
        previous = mappings.get(sequence_id)
        if previous is not None and previous != specification:
            raise ValueError("conflicting MSA mapping for sequence_id %s" % sequence_id)
        mappings[sequence_id] = specification
    return mappings, {"path": str(path), "sha256": sha256_file(path)}


def prepare_inputs(
    manifest_paths,
    msa_manifest_path,
    mode,
    maximum_targets=3,
    maximum_depth=8,
    maximum_columns=512,
    maximum_tensor_tokens=8192,
):
    targets, sample_rows, source_manifests = collect_targets(
        manifest_paths, maximum_targets=maximum_targets
    )
    mappings, msa_manifest = load_msa_manifest(msa_manifest_path)
    if set(mappings) != set(targets):
        missing = sorted(set(targets) - set(mappings))
        extra = sorted(set(mappings) - set(targets))
        raise ValueError("MSA mappings differ from targets; missing=%s extra=%s" % (missing, extra))
    prepared = {}
    for sequence_id, sequence in sorted(targets.items()):
        prepared[sequence_id] = prepare_msa(
            mappings[sequence_id]["a3m_path"],
            sequence,
            mode,
            maximum_depth=maximum_depth,
            maximum_columns=maximum_columns,
            maximum_tensor_tokens=maximum_tensor_tokens,
        )
    return {
        "targets": targets,
        "sample_rows": sample_rows,
        "source_manifests": source_manifests,
        "mappings": mappings,
        "msa_manifest": msa_manifest,
        "prepared": prepared,
    }
