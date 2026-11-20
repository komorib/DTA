"""Deterministic selection over rows already accepted by the legacy validator.

This module does not parse A3M or change row validity. Source indices refer to
the original FASTA row positions, including rejected rows (query is index 0).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


LEGACY_SELECTION_POLICY = "query_then_first_valid_unique_homologs_v1"
DIVERSE_SELECTION_POLICY = "query_then_diverse_maxmin_v1"
DEFAULT_CANDIDATE_POOL_SIZE = 256


def aligned_sequence_identity(left, right):
    """Return identity on jointly non-gap columns; no comparable columns is 0."""

    if len(left) != len(right):
        raise ValueError("aligned identity requires equal row widths")
    a = np.frombuffer(left.encode("ascii"), dtype=np.uint8)
    b = np.frombuffer(right.encode("ascii"), dtype=np.uint8)
    joint = (a != ord("-")) & (b != ord("-"))
    count = int(joint.sum())
    return float(((a == b) & joint).sum() / count) if count else 0.0


def select_diverse_valid_rows(valid_rows, source_indices, homolog_count, candidate_pool_size):
    """Select a bounded max-min pool, then explicitly record any source fallback.

    The first homolog maximizes query identity, with source order breaking ties.
    Subsequent homologs maximize minimum diversity to selected *homologs*, not
    to query. No reusable aligned-row coverage metric exists in this pipeline;
    pocket-restricted diagnostic coverage is not an interchangeable tiebreaker.

    With the experiment's pool size 256 and depth at most 8, a pool shortfall
    implies all valid homologs are exhausted. It records fallback_used=true,
    fallback_count=0 and the remaining shortfall; it never invents a row. Smaller
    explicitly configured pools can exercise the source-order fallback itself.
    """

    pool_size = int(candidate_pool_size)
    if not 1 <= pool_size <= DEFAULT_CANDIDATE_POOL_SIZE:
        raise ValueError("candidate_pool_size must be between 1 and 256")
    target = int(homolog_count)
    if not 0 <= target <= 7:
        raise ValueError("homolog_count must be between 0 and 7")
    if not valid_rows or len(valid_rows) != len(source_indices) or source_indices[0] != 0:
        raise ValueError("validated rows and source indices must start with query index 0")

    pool_count = min(pool_size, len(valid_rows) - 1)
    # Include at most seven possible fallback rows without encoding the full MSA.
    encoded_count = min(len(valid_rows), 1 + max(pool_count, target))
    encoded = np.stack([
        np.frombuffer(row.aligned_sequence.encode("ascii"), dtype=np.uint8)
        for row in valid_rows[:encoded_count]
    ])
    observed = encoded != ord("-")
    anomalies = set()
    identity_cache = {}

    def identities_to(index):
        if index not in identity_cache:
            joint = observed & observed[index]
            denominators = joint.sum(axis=1)
            matches = ((encoded == encoded[index]) & joint).sum(axis=1)
            values = np.divide(matches, denominators, out=np.zeros(encoded_count, dtype=np.float64),
                               where=denominators != 0)
            for other in np.flatnonzero(denominators == 0):
                anomalies.add(tuple(sorted((int(source_indices[index]), int(source_indices[other])))))
            identity_cache[index] = values
        return identity_cache[index]

    query_identity = identities_to(0)
    remaining = list(range(1, pool_count + 1))
    chosen = []
    minimum_diversity = np.ones(encoded_count, dtype=np.float64)
    if target and remaining:
        first = max(remaining, key=lambda index: (query_identity[index], -source_indices[index]))
        chosen.append(first)
        remaining.remove(first)
    while len(chosen) < target and remaining:
        # Updating only against the most recently selected homolog is equivalent
        # to min over the entire homolog set, without a quadratic pool matrix.
        minimum_diversity = np.minimum(minimum_diversity, 1.0 - identities_to(chosen[-1]))
        best = max(remaining, key=lambda index: (
            minimum_diversity[index], query_identity[index], -source_indices[index]
        ))
        chosen.append(best)
        remaining.remove(best)

    greedy_sources = [int(source_indices[index]) for index in chosen]
    fallback_used = len(chosen) < target
    fallback = []
    if fallback_used:
        chosen_set = set(chosen)
        for index in range(1, len(valid_rows)):
            if index not in chosen_set:
                fallback.append(index)
                if len(chosen) + len(fallback) == target:
                    break
    selected = [0] + sorted(chosen + fallback, key=lambda index: source_indices[index])
    diversities = [float(1.0 - identities_to(left)[right])
                   for left, right in combinations(selected[1:], 2)]
    summary = {
        "mean": float(np.mean(diversities)) if diversities else 0.0,
        "median": float(np.median(diversities)) if diversities else 0.0,
        "min": float(min(diversities)) if diversities else 0.0,
        "max": float(max(diversities)) if diversities else 0.0,
    }
    metadata = {
        "candidate_pool_size": pool_size,
        "candidate_count": pool_count,
        "coverage_tiebreak_available": False,
        "selected_depth": len(selected),
        "query_id": valid_rows[0].identifier,
        "selected_ids": [valid_rows[index].identifier for index in selected],
        "selected_source_indices": [int(source_indices[index]) for index in selected],
        "selected_query_identity": [float(query_identity[index]) for index in selected],
        "selected_pairwise_diversity_summary": summary,
        "selected_pairwise_diversity_scope": "selected_homologs_only",
        "greedy_selected_source_indices": greedy_sources,
        "fallback_used": fallback_used,
        "fallback_count": len(fallback),
        "fallback_reason": "candidate_pool_exhausted" if fallback_used else None,
        "selection_shortfall_count": target - len(selected) + 1,
        "identity_definition": "matches_on_jointly_non_gap_columns_v1",
        "identity_zero_denominator_count": len(anomalies),
        "identity_zero_denominator_pairs": [list(pair) for pair in sorted(anomalies)],
    }
    return tuple(valid_rows[index] for index in selected), metadata


__all__ = [
    "DEFAULT_CANDIDATE_POOL_SIZE",
    "DIVERSE_SELECTION_POLICY",
    "LEGACY_SELECTION_POLICY",
    "aligned_sequence_identity",
    "select_diverse_valid_rows",
]
