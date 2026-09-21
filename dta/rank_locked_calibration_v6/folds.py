"""Deterministic target-or-scaffold grouped folds for calibration v6."""

from collections import defaultdict
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


_CSV_COLUMNS = frozenset(("Unnamed: 0", "smiles", "seq", "new_split"))


def normalize_target(value):
    if value is None:
        raise ValueError("target sequence is empty")
    sequence = "".join(str(value).split()).upper()
    if not sequence:
        raise ValueError("target sequence is empty")
    return sequence


def murcko_scaffold(smiles):
    if smiles is None or not str(smiles).strip():
        raise ValueError("ligand SMILES is invalid")
    blocker = rdBase.BlockLogs()
    try:
        molecule = Chem.MolFromSmiles(str(smiles).strip())
    finally:
        del blocker
    if molecule is None:
        raise ValueError("ligand SMILES is invalid")
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=False
    )


def _scaffold_with_policy(
    smiles, key, allow_invalid_scaffold_fallback
):
    try:
        return murcko_scaffold(smiles)
    except ValueError:
        if not allow_invalid_scaffold_fallback:
            raise
        value = "" if smiles is None else str(smiles).strip()
        if not value:
            return "__missing_smiles__:" + key
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return "__invalid_smiles__:" + digest


@dataclass(frozen=True)
class FoldRow:
    key: str
    target: str
    scaffold: str

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("fold row key is empty")
        if not isinstance(self.scaffold, str):
            raise ValueError("fold row scaffold is invalid")
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "target", normalize_target(self.target))
        object.__setattr__(self, "scaffold", self.scaffold.strip())


@dataclass(frozen=True)
class FoldAssignment:
    ordered_keys: tuple
    fold_ids: np.ndarray
    component_ids: tuple
    by_key: Mapping
    component_by_key: Mapping
    fold_sizes: tuple
    fold_count: int
    seed: int
    sha256: str


def _row_key(identifier):
    if identifier is None or not str(identifier).strip():
        raise ValueError("metadata identifier is empty")
    value = str(identifier).strip()
    if value.endswith("_protein"):
        return value
    return value + "_protein"


def load_fold_rows(
    metadata_csv,
    cache_keys,
    *,
    expected_split="train",
    allow_invalid_scaffold_fallback=True,
):
    """Load selected cache rows from the authoritative LP-PDBbind CSV."""
    if not isinstance(expected_split, str) or not expected_split:
        raise ValueError("expected split is invalid")
    if not isinstance(allow_invalid_scaffold_fallback, bool):
        raise ValueError("invalid scaffold fallback policy must be boolean")
    if isinstance(cache_keys, str):
        raise ValueError("cache keys must be a sequence")
    keys = tuple(cache_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("cache keys must be nonempty strings")
    if len(keys) != len(set(keys)):
        raise ValueError("cache keys contain duplicates")
    requested = set(keys)

    path = Path(metadata_csv).expanduser().resolve()
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ValueError("metadata CSV is unavailable") from error
    selected = {}
    seen = set()
    try:
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not _CSV_COLUMNS.issubset(
                reader.fieldnames
            ):
                raise ValueError("metadata CSV columns do not match")
            for record in reader:
                key = _row_key(record.get("Unnamed: 0"))
                if key in seen:
                    raise ValueError("metadata CSV contains duplicate IDs")
                seen.add(key)
                if key not in requested:
                    continue
                split = record.get("new_split")
                if split != expected_split:
                    raise ValueError(
                        key + " metadata split does not match " + expected_split
                    )
                selected[key] = FoldRow(
                    key=key,
                    target=normalize_target(record.get("seq")),
                    scaffold=_scaffold_with_policy(
                        record.get("smiles"),
                        key,
                        allow_invalid_scaffold_fallback,
                    ),
                )
    except (csv.Error, UnicodeDecodeError) as error:
        raise ValueError("metadata CSV is malformed") from error

    missing = sorted(requested.difference(selected))
    if missing:
        raise ValueError(
            "metadata CSV is missing cache keys: " + ", ".join(missing[:5])
        )
    return tuple(selected[key] for key in keys)


class _UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _component_digest(keys):
    payload = json.dumps(
        list(keys), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assignment_hash(fold_count, seed, rows):
    payload = {
        "schema_version": 1,
        "fold_count": fold_count,
        "seed": seed,
        "rows": rows,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assign_grouped_folds(rows, *, fold_count=5, seed=42):
    """Assign connected target/scaffold components to balanced folds."""
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, Integral)
        or int(fold_count) < 2
    ):
        raise ValueError("fold_count must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    fold_count = int(fold_count)
    seed = int(seed)
    try:
        normalized_rows = tuple(rows)
    except TypeError as error:
        raise ValueError("rows must be an iterable of FoldRow values") from error
    if not normalized_rows or any(
        not isinstance(row, FoldRow) for row in normalized_rows
    ):
        raise ValueError("rows must contain FoldRow values")
    keys = tuple(row.key for row in normalized_rows)
    if len(keys) != len(set(keys)):
        raise ValueError("fold rows contain duplicate keys")

    ordered_rows = tuple(sorted(normalized_rows, key=lambda row: row.key))
    union_find = _UnionFind(row.key for row in ordered_rows)
    target_owner = {}
    scaffold_owner = {}
    for row in ordered_rows:
        target_owner.setdefault(row.target, row.key)
        union_find.union(row.key, target_owner[row.target])
        scaffold_key = row.scaffold
        if not scaffold_key:
            scaffold_key = "__acyclic__:" + row.key
        scaffold_owner.setdefault(scaffold_key, row.key)
        union_find.union(row.key, scaffold_owner[scaffold_key])

    members = defaultdict(list)
    for row in ordered_rows:
        members[union_find.find(row.key)].append(row.key)
    components = []
    for values in members.values():
        component_keys = tuple(sorted(values))
        component_id = _component_digest(component_keys)
        seed_order = hashlib.sha256(
            (str(seed) + ":" + component_id).encode("ascii")
        ).hexdigest()
        components.append((component_keys, component_id, seed_order))
    if len(components) < fold_count:
        raise ValueError("fewer connected components than folds")
    components.sort(key=lambda item: (-len(item[0]), item[2], item[0]))

    fold_sizes = [0] * fold_count
    by_key = {}
    component_by_key = {}
    for component_keys, component_id, _ in components:
        fold = min(range(fold_count), key=lambda index: (fold_sizes[index], index))
        fold_sizes[fold] += len(component_keys)
        for key in component_keys:
            by_key[key] = fold
            component_by_key[key] = component_id
    if any(size == 0 for size in fold_sizes):
        raise ValueError("grouped fold assignment produced an empty fold")
    if set(by_key) != set(keys):
        raise ValueError("grouped fold assignment coverage does not match")
    for component_keys, _, _ in components:
        if len({by_key[key] for key in component_keys}) != 1:
            raise ValueError("connected component crosses fold boundaries")

    ordered_keys = tuple(sorted(keys))
    fold_ids = np.asarray(
        [by_key[key] for key in ordered_keys], dtype=np.int64
    )
    fold_ids.setflags(write=False)
    component_ids = tuple(component_by_key[key] for key in ordered_keys)
    canonical_rows = [
        {
            "key": key,
            "fold": by_key[key],
            "component": component_by_key[key],
        }
        for key in ordered_keys
    ]
    return FoldAssignment(
        ordered_keys=ordered_keys,
        fold_ids=fold_ids,
        component_ids=component_ids,
        by_key=MappingProxyType(dict(by_key)),
        component_by_key=MappingProxyType(dict(component_by_key)),
        fold_sizes=tuple(fold_sizes),
        fold_count=fold_count,
        seed=seed,
        sha256=_assignment_hash(fold_count, seed, canonical_rows),
    )


__all__ = [
    "FoldAssignment",
    "FoldRow",
    "assign_grouped_folds",
    "load_fold_rows",
    "murcko_scaffold",
    "normalize_target",
]
