"""LP-PDBbind to LMDB preprocessing utilities.

The runtime loader in :mod:`kdbnet.pdbbind_lmdb` expects one pickled record per
complex with the following minimum schema::

    {
        "protein_features": {
            chain_id: {
                "sequence": str,
                "backbone_coords": np.ndarray[L, 4, 3],  # N, CA, C, O
                "esm2_embedding": np.ndarray[L, D],
            }
        },
        "ligand_features": {"mol": rdkit.Chem.Mol},
        "metadata": {...},
    }

This module deliberately keeps LP-PDBbind split parsing strict: every accepted
complex belongs to exactly one split and all derived artifacts inherit the
same base-complex split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kdbnet.pdbbind import parse_affinity_expression, parse_pdbbind_index_lines


ID_ALIASES = (
    "complex_id", "pdbid", "pdb_id", "pdb", "id", "name", "pdb_code",
)
SPLIT_ALIASES = ("split", "set", "subset", "partition", "fold")
TARGET_ALIASES = (
    "affinity", "y", "pk", "p_k", "pka", "plog", "label", "target",
    "-logkd/ki", "neg_log_affinity",
)
RAW_AFFINITY_ALIASES = (
    "affinity_raw", "measurement", "binding_affinity", "kd_ki_ic50",
)

SPLIT_NORMALISATION = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "dev": "valid",
    "development": "valid",
    "test": "test",
    "testing": "test",
    "holdout": "test",
    "hold-out": "test",
}

# Common PDB residue aliases. Modified residues are mapped conservatively to
# the parent amino acid when their backbone is standard enough for KDBNet.
THREE_TO_ONE_EXTENDED = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "HSD": "H", "HSE": "H",
    "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C",
    "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K", "SEP": "S",
    "TPO": "T", "PTR": "Y",
}
BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass(frozen=True)
class StructureFiles:
    complex_id: str
    protein_pdb: Optional[Path]
    pocket_pdb: Optional[Path]
    ligand_sdf: Optional[Path]
    ligand_mol2: Optional[Path]

    @property
    def ligand_path(self) -> Optional[Path]:
        return self.ligand_sdf or self.ligand_mol2


def normalise_complex_id(value: Any) -> str:
    value = str(value).strip().lower()
    # LP/PDBbind identifiers are normally four-character PDB IDs; preserve
    # longer IDs but remove accidental file suffixes.
    for suffix in (
        "_protein.pdb", "_pocket.pdb", "_ligand.sdf", "_ligand.mol2",
        ".pdb", ".sdf", ".mol2",
    ):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _normalise_split(value: Any) -> Tuple[str, str]:
    detail = str(value).strip().lower()
    canonical = SPLIT_NORMALISATION.get(detail)
    if canonical is None:
        # Keep tiered LP test labels (e.g. test_cl1/test-cl2) in split_detail
        # while exposing the canonical split expected by the runtime loader.
        if detail.startswith("test"):
            canonical = "test"
        elif detail.startswith("valid") or detail.startswith("val"):
            canonical = "valid"
        elif detail.startswith("train"):
            canonical = "train"
        else:
            raise ValueError("Unknown split label: {!r}".format(value))
    return canonical, detail


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _object_to_dataframe(obj: Any, source_name: str = "object") -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, Mapping):
        # Common split pickle/json: {"train": [...], "valid": [...], ...}
        split_keys = [key for key in obj if str(key).lower().startswith(("train", "val", "dev", "test"))]
        if split_keys:
            rows: List[Dict[str, Any]] = []
            for key in split_keys:
                value = obj[key]
                if isinstance(value, pd.DataFrame):
                    records = value.to_dict("records")
                    for record in records:
                        row = dict(record)
                        row.setdefault("split", key)
                        rows.append(row)
                elif isinstance(value, Mapping):
                    for cid, metadata in value.items():
                        row = {"complex_id": cid, "split": key}
                        if isinstance(metadata, Mapping):
                            row.update(metadata)
                        rows.append(row)
                else:
                    for cid in list(value):
                        if isinstance(cid, Mapping):
                            row = dict(cid)
                            row.setdefault("split", key)
                        else:
                            row = {"complex_id": cid, "split": key}
                        rows.append(row)
            return pd.DataFrame(rows)
        # Mapping of ID -> split or metadata.
        rows = []
        for key, value in obj.items():
            if isinstance(value, Mapping):
                row = {"complex_id": key}
                row.update(value)
            else:
                row = {"complex_id": key, "split": value}
            rows.append(row)
        return pd.DataFrame(rows)
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        values = list(obj)
        if not values:
            return pd.DataFrame()
        if isinstance(values[0], Mapping):
            return pd.DataFrame(values)
        return pd.DataFrame({"complex_id": values})
    raise TypeError("Unsupported {} type: {}".format(source_name, type(obj).__name__))


def _read_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle", ".pt"}:
        if suffix == ".pt":
            import torch
            obj = torch.load(path, map_location="cpu", weights_only=False)
        else:
            with open(path, "rb") as handle:
                obj = pickle.load(handle)
        return _object_to_dataframe(obj, source_name=str(path))
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return _object_to_dataframe(json.load(handle), source_name=str(path))
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix in {".csv", ".tsv", ".tab"}:
        separator = "," if suffix == ".csv" else "\t"
        frame = pd.read_csv(path, sep=separator)
        recognised = (
            _find_column(frame, ID_ALIASES) is not None
            or _find_column(frame, SPLIT_ALIASES) is not None
            or _find_column(frame, TARGET_ALIASES) is not None
        )
        if not recognised:
            raw = pd.read_csv(path, sep=separator, header=None, comment="#")
            columns = ["complex_id"] + ["value_{}".format(i) for i in range(1, raw.shape[1])]
            raw.columns = columns
            return raw
        return frame

    # Flexible plain-text reader: comments and whitespace are accepted.
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
    if not lines:
        return pd.DataFrame()
    # Header-aware if the first line names an ID/split column.
    first_tokens = re.split(r"[\t,\s]+", lines[0])
    lowered = {token.lower() for token in first_tokens}
    has_header = bool(lowered.intersection(ID_ALIASES + SPLIT_ALIASES + TARGET_ALIASES))
    rows = [re.split(r"[\t,\s]+", line) for line in lines]
    if has_header:
        return pd.DataFrame(rows[1:], columns=rows[0])
    width = max(len(row) for row in rows)
    if width == 1:
        return pd.DataFrame({"complex_id": [row[0] for row in rows]})
    if width >= 2:
        return pd.DataFrame({
            "complex_id": [row[0] for row in rows],
            "split": [row[1] for row in rows],
        })
    return pd.DataFrame()


def load_lp_split(split_source: os.PathLike[str] | str) -> pd.DataFrame:
    """Load and strictly validate an LP-PDBbind split.

    ``split_source`` may be a table/pickle/json file or a directory containing
    files whose names identify train/valid/test. The result always contains
    ``complex_id``, ``split`` and ``split_detail``.
    """
    source = Path(split_source)
    if not source.exists():
        raise FileNotFoundError(str(source))

    if source.is_dir():
        frames = []
        patterns = ("*.txt", "*.lst", "*.csv", "*.tsv", "*.pkl", "*.pickle", "*.json", "*.jsonl")
        files: List[Path] = []
        for pattern in patterns:
            files.extend(source.glob(pattern))
        for path in sorted(set(files)):
            stem = path.stem.lower()
            inferred = None
            for token in ("train", "training", "valid", "validation", "val", "dev", "test"):
                if token in stem:
                    inferred = token
                    break
            if inferred is None:
                continue
            frame = _read_table_file(path)
            if frame.empty:
                continue
            if _find_column(frame, SPLIT_ALIASES) is None:
                frame["split"] = inferred
            frames.append(frame)
        if not frames:
            raise ValueError("No train/valid/test split files found in {}".format(source))
        df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        df = _read_table_file(source)

    if df.empty:
        raise ValueError("Split source is empty: {}".format(source))
    id_col = _find_column(df, ID_ALIASES)
    split_col = _find_column(df, SPLIT_ALIASES)
    if id_col is None or split_col is None:
        raise ValueError(
            "Split source must provide ID and split columns; columns={}".format(list(df.columns))
        )

    out = pd.DataFrame()
    out["complex_id"] = df[id_col].map(normalise_complex_id)
    canonical, detail = [], []
    for value in df[split_col]:
        split, split_detail = _normalise_split(value)
        canonical.append(split)
        detail.append(split_detail)
    out["split"] = canonical
    out["split_detail"] = detail
    out = out[out["complex_id"] != ""].drop_duplicates().reset_index(drop=True)

    # A base complex may not cross canonical split boundaries. Multiple tier
    # labels within test are deduplicated to the first detail only.
    split_counts = out.groupby("complex_id")["split"].nunique()
    overlaps = split_counts[split_counts > 1]
    if len(overlaps):
        examples = overlaps.index[:20].tolist()
        raise ValueError("LP split overlap across train/valid/test: {}".format(examples))
    out = out.sort_values(["split", "complex_id", "split_detail"]).drop_duplicates(
        "complex_id", keep="first"
    ).reset_index(drop=True)
    missing = {"train", "valid", "test"}.difference(out["split"].unique())
    if missing:
        raise ValueError("LP split is missing canonical partitions: {}".format(sorted(missing)))
    return out


def _numeric_affinity_to_pk(value: Any, mode: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Non-finite affinity: {}".format(value))
    if mode == "pk":
        return numeric
    if mode == "molar":
        if numeric <= 0:
            raise ValueError("Molar affinity must be positive")
        return -math.log10(numeric)
    if mode == "auto":
        # PDBbind pK values are typically in roughly [0, 15]. Molar values are
        # usually <<1. Ambiguous values should be resolved explicitly.
        if 0 < numeric < 1e-2:
            return -math.log10(numeric)
        return numeric
    raise ValueError("Unknown numeric affinity mode: {}".format(mode))


def load_affinity_table(
    affinity_source: os.PathLike[str] | str,
    numeric_affinity_is: str = "pk",
) -> pd.DataFrame:
    """Load PDBbind/LP affinity labels and normalise the target to pK."""
    path = Path(affinity_source)
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Standard PDBbind INDEX_general_PL.2020R1.lst.
    if path.suffix.lower() in {".lst", ".index"} or path.name.upper().startswith("INDEX_"):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            rows, skipped = parse_pdbbind_index_lines(handle.readlines())
        if not rows:
            raise ValueError("No supported affinity rows in {} (skipped={})".format(path, len(skipped)))
        df = pd.DataFrame(rows)
        df["affinity"] = df["y"].astype(float)
        return df

    df = _read_table_file(path)
    if df.empty:
        raise ValueError("Affinity source is empty: {}".format(path))
    id_col = _find_column(df, ID_ALIASES)
    if id_col is None:
        raise ValueError("Affinity source has no complex ID column: {}".format(list(df.columns)))

    rows = []
    target_col = _find_column(df, TARGET_ALIASES)
    raw_col = _find_column(df, RAW_AFFINITY_ALIASES)
    type_col = _find_column(df, ("affinity_type", "measurement_type", "type"))
    op_col = _find_column(df, ("affinity_op", "operator", "op"))
    unit_col = _find_column(df, ("affinity_unit", "unit"))
    value_col = _find_column(df, ("affinity_value", "value"))

    for _, record in df.iterrows():
        cid = normalise_complex_id(record[id_col])
        parsed = None
        raw = None
        if raw_col is not None and pd.notna(record[raw_col]):
            raw = str(record[raw_col]).replace("μ", "u").replace(" ", "")
            parsed = parse_affinity_expression(raw)
        if parsed is None and type_col and value_col and unit_col:
            raw = "{}{}{}{}".format(
                str(record[type_col]),
                str(record[op_col]) if op_col and pd.notna(record[op_col]) else "=",
                str(record[value_col]),
                str(record[unit_col]).replace("μ", "u"),
            ).replace(" ", "")
            parsed = parse_affinity_expression(raw)
        if parsed is not None:
            row = {"complex_id": cid, "affinity_raw": raw}
            row.update(parsed)
            row["affinity"] = float(parsed["y"])
            rows.append(row)
            continue
        if target_col is None or pd.isna(record[target_col]):
            continue
        pk = _numeric_affinity_to_pk(record[target_col], numeric_affinity_is)
        rows.append({
            "complex_id": cid,
            "affinity": pk,
            "y": pk,
            "affinity_raw": str(record[target_col]),
            "affinity_type": str(record[type_col]) if type_col and pd.notna(record[type_col]) else "unknown",
            "affinity_op": str(record[op_col]) if op_col and pd.notna(record[op_col]) else "=",
            "affinity_value": np.nan,
            "affinity_unit": "pK",
            "affinity_molar": 10.0 ** (-pk),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Could not parse any affinities from {}".format(path))
    out = out.drop_duplicates("complex_id", keep="first").reset_index(drop=True)
    return out


def discover_structure_files(structure_root: os.PathLike[str] | str) -> Dict[str, StructureFiles]:
    """Recursively index PDBbind complex structure files."""
    root = Path(structure_root)
    if not root.exists():
        raise FileNotFoundError(str(root))
    mutable: Dict[str, Dict[str, Path]] = {}
    suffix_to_key = {
        "_protein.pdb": "protein_pdb",
        "_pocket.pdb": "pocket_pdb",
        "_ligand.sdf": "ligand_sdf",
        "_ligand.mol2": "ligand_mol2",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        for suffix, key in suffix_to_key.items():
            if lower.endswith(suffix):
                cid = normalise_complex_id(lower[: -len(suffix)])
                current = mutable.setdefault(cid, {})
                # Prefer a file whose parent directory is exactly the PDB ID,
                # then shorter paths. This resolves duplicate extracted trees.
                previous = current.get(key)
                if previous is None:
                    current[key] = path
                else:
                    prev_score = (previous.parent.name.lower() == cid, -len(previous.parts))
                    new_score = (path.parent.name.lower() == cid, -len(path.parts))
                    if new_score > prev_score:
                        current[key] = path
                break
    return {
        cid: StructureFiles(
            complex_id=cid,
            protein_pdb=files.get("protein_pdb"),
            pocket_pdb=files.get("pocket_pdb"),
            ligand_sdf=files.get("ligand_sdf"),
            ligand_mol2=files.get("ligand_mol2"),
        )
        for cid, files in mutable.items()
    }


def _parse_float_field(line: str, start: int, end: int, default: float = 0.0) -> float:
    try:
        return float(line[start:end])
    except (ValueError, TypeError):
        return default


def parse_protein_pdb(
    pdb_path: os.PathLike[str] | str,
    require_ca: bool = True,
) -> "OrderedDict[str, Dict[str, Any]]":
    """Parse chain sequences and N/CA/C/O coordinates from a PDB file.

    Missing backbone atoms are represented by NaN. Alternate locations are
    resolved by highest occupancy with blank/A preferred on ties. Only MODEL 1
    is used.
    """
    path = Path(pdb_path)
    residues: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
    in_first_model = True
    saw_model = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record == "MODEL":
                if saw_model:
                    in_first_model = False
                else:
                    saw_model = True
                    in_first_model = True
                continue
            if record == "ENDMDL" and saw_model:
                break
            if not in_first_model or record not in {"ATOM", "HETATM"}:
                continue
            atom_name = line[12:16].strip().upper()
            if atom_name not in BACKBONE_ATOMS:
                continue
            altloc = line[16:17].strip().upper()
            resname = line[17:20].strip().upper()
            aa = THREE_TO_ONE_EXTENDED.get(resname)
            if aa is None:
                continue
            chain = line[21:22].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            key = (chain, resseq, icode)
            x = _parse_float_field(line, 30, 38, np.nan)
            y = _parse_float_field(line, 38, 46, np.nan)
            z = _parse_float_field(line, 46, 54, np.nan)
            occupancy = _parse_float_field(line, 54, 60, 0.0)
            residue = residues.setdefault(key, {
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "resname": resname,
                "aa": aa,
                "atoms": {},
            })
            previous = residue["atoms"].get(atom_name)
            preference = (occupancy, altloc in {"", "A"})
            if previous is None or preference > previous[0]:
                residue["atoms"][atom_name] = (preference, [x, y, z])

    chains: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for residue in residues.values():
        atom_coords = residue["atoms"]
        if require_ca and "CA" not in atom_coords:
            continue
        chain = chains.setdefault(residue["chain"], {
            "sequence": [],
            "coords": [],
            "residue_ids": [],
            "residue_names": [],
        })
        coords = np.full((4, 3), np.nan, dtype=np.float32)
        for idx, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in atom_coords:
                coords[idx] = np.asarray(atom_coords[atom_name][1], dtype=np.float32)
        chain["sequence"].append(residue["aa"])
        chain["coords"].append(coords)
        chain["residue_ids"].append((residue["chain"], residue["resseq"], residue["icode"]))
        chain["residue_names"].append(residue["resname"])

    result: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for chain_id, chain in chains.items():
        if not chain["sequence"]:
            continue
        result[chain_id] = {
            "sequence": "".join(chain["sequence"]),
            "backbone_coords": np.stack(chain["coords"]).astype(np.float32),
            "residue_ids": chain["residue_ids"],
            "residue_names": chain["residue_names"],
        }
    if not result:
        raise ValueError("No usable protein residues in {}".format(path))
    return result


def load_ligand_mol(path: os.PathLike[str] | str, remove_hs: bool = False):
    from rdkit import Chem

    path = Path(path)
    suffix = path.suffix.lower()
    mol = None
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=remove_hs, sanitize=True)
        mol = next((candidate for candidate in supplier if candidate is not None), None)
        if mol is None:
            supplier = Chem.SDMolSupplier(str(path), removeHs=remove_hs, sanitize=False)
            mol = next((candidate for candidate in supplier if candidate is not None), None)
            if mol is not None:
                Chem.SanitizeMol(mol)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=remove_hs, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMol2File(str(path), removeHs=remove_hs, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol)
    else:
        raise ValueError("Unsupported ligand format: {}".format(path))
    if mol is None:
        raise ValueError("RDKit failed to parse ligand: {}".format(path))
    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand has no 3D conformer: {}".format(path))
    if mol.GetNumAtoms() == 0:
        raise ValueError("Ligand has no atoms: {}".format(path))
    return mol


def stable_sequence_hash(sequence: str, model_tag: str = "") -> str:
    return hashlib.sha1((model_tag + "\n" + sequence).encode("utf-8")).hexdigest()


class ESMEmbedder:
    """Optional fair-esm embedding generation with long-sequence chunking."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        model_path: Optional[os.PathLike[str] | str] = None,
        repr_layer: int = 33,
        device: str = "cuda",
        max_tokens: int = 1022,
        overlap: int = 128,
        cache_dir: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        try:
            import torch
            import esm
        except ImportError as exc:
            raise ImportError(
                "ESM generation requires fair-esm and torch. Install fair-esm "
                "or use --esm_mode precomputed/zero."
            ) from exc
        self.torch = torch
        self.esm = esm
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        if model_path:
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
            model_tag = str(Path(model_path).resolve())
        else:
            loader = getattr(esm.pretrained, model_name, None)
            if loader is None:
                raise ValueError("Unknown fair-esm model: {}".format(model_name))
            model, alphabet = loader()
            model_tag = model_name
        self.model = model.eval().to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layer = int(repr_layer)
        self.max_tokens = int(max_tokens)
        self.overlap = int(overlap)
        if self.max_tokens <= self.overlap:
            raise ValueError("max_tokens must be larger than overlap")
        self.model_tag = "{}:layer{}".format(model_tag, self.repr_layer)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _embed_window(self, sequence: str):
        _, _, tokens = self.batch_converter([("chain", sequence)])
        tokens = tokens.to(self.device)
        with self.torch.no_grad():
            output = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        return output["representations"][self.repr_layer][0, 1:1 + len(sequence)].detach().cpu().float()

    def embed(self, sequence: str) -> np.ndarray:
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / (stable_sequence_hash(sequence, self.model_tag) + ".npy")
            if cache_path.exists():
                array = np.load(cache_path)
                if array.shape[0] == len(sequence):
                    return array
        if len(sequence) <= self.max_tokens:
            result = self._embed_window(sequence)
        else:
            step = self.max_tokens - self.overlap
            result = None
            counts = self.torch.zeros(len(sequence), 1, dtype=self.torch.float32)
            for start in range(0, len(sequence), step):
                end = min(start + self.max_tokens, len(sequence))
                window = self._embed_window(sequence[start:end])
                if result is None:
                    result = self.torch.zeros(len(sequence), window.size(-1), dtype=self.torch.float32)
                result[start:end] += window
                counts[start:end] += 1.0
                if end == len(sequence):
                    break
            result = result / counts.clamp_min(1.0)
        array = result.numpy().astype(np.float32)
        if cache_path:
            np.save(cache_path, array)
        return array


def _tensor_to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def load_precomputed_embeddings(
    embedding_dir: os.PathLike[str] | str,
    complex_id: str,
    protein_features: Mapping[str, Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """Load a flexible per-complex .pt/.pkl/.npy embedding artifact."""
    root = Path(embedding_dir)
    candidates = [
        root / (complex_id + suffix)
        for suffix in (".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz")
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No embedding artifact for {} in {}".format(complex_id, root))
    if path.suffix in {".pt", ".pth"}:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
    elif path.suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
    elif path.suffix == ".npz":
        obj = dict(np.load(path))
    else:
        obj = np.load(path)

    chains = list(protein_features.keys())
    lengths = {chain: len(protein_features[chain]["sequence"]) for chain in chains}
    if isinstance(obj, Mapping):
        for wrapper in ("representations", "embeddings", "protein_features", "chains"):
            if wrapper in obj and isinstance(obj[wrapper], Mapping):
                obj = obj[wrapper]
                break
        result = {}
        for chain in chains:
            candidate = None
            for key in (chain, str(chain), "chain_{}".format(chain)):
                if key in obj:
                    candidate = obj[key]
                    break
            if isinstance(candidate, Mapping):
                for key in ("esm2_embedding", "esm_embedding", "seq_emb", "embedding", "representations"):
                    if key in candidate:
                        candidate = candidate[key]
                        break
            if candidate is not None:
                array = _tensor_to_numpy(candidate)
                if array.shape[0] != lengths[chain]:
                    raise ValueError("Embedding length mismatch for {} chain {}".format(complex_id, chain))
                result[chain] = np.asarray(array, dtype=np.float32)
        if len(result) == len(chains):
            return result

    array = _tensor_to_numpy(obj)
    if array.ndim != 2 or array.shape[0] != sum(lengths.values()):
        raise ValueError(
            "Precomputed embedding for {} must be chain mapping or concatenated [L,D]; got {}".format(
                complex_id, tuple(array.shape)
            )
        )
    result = {}
    offset = 0
    for chain in chains:
        length = lengths[chain]
        result[chain] = np.asarray(array[offset:offset + length], dtype=np.float32)
        offset += length
    return result


def attach_embeddings(
    protein_features: MutableMapping[str, MutableMapping[str, Any]],
    mode: str,
    embedding_key: str,
    embedding_dim: int,
    dtype: str,
    embedder: Optional[ESMEmbedder] = None,
    embedding_dir: Optional[os.PathLike[str] | str] = None,
    complex_id: Optional[str] = None,
) -> None:
    if mode == "zero":
        embeddings = {
            chain: np.zeros((len(data["sequence"]), embedding_dim), dtype=np.float32)
            for chain, data in protein_features.items()
        }
    elif mode == "generate":
        if embedder is None:
            raise ValueError("generate mode requires ESMEmbedder")
        embeddings = {
            chain: embedder.embed(data["sequence"])
            for chain, data in protein_features.items()
        }
    elif mode == "precomputed":
        if embedding_dir is None or complex_id is None:
            raise ValueError("precomputed mode requires embedding_dir and complex_id")
        embeddings = load_precomputed_embeddings(embedding_dir, complex_id, protein_features)
    else:
        raise ValueError("Unknown esm mode: {}".format(mode))

    target_dtype = np.float16 if dtype == "float16" else np.float32
    for chain, array in embeddings.items():
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[0] != len(protein_features[chain]["sequence"]):
            raise ValueError("Invalid embedding shape for chain {}: {}".format(chain, tuple(array.shape)))
        if array.shape[1] != int(embedding_dim):
            raise ValueError(
                "Embedding dimension mismatch for chain {}: expected {}, got {}".format(
                    chain, embedding_dim, array.shape[1]
                )
            )
        protein_features[chain][embedding_key] = array.astype(target_dtype)


def path_for_metadata(path: Optional[Path], root: Path, mode: str) -> Optional[str]:
    if path is None:
        return None
    if mode == "absolute":
        return str(path.resolve())
    if mode != "relative":
        raise ValueError("Unknown path mode: {}".format(mode))
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_complex_record(
    row: Mapping[str, Any],
    files: StructureFiles,
    structure_root: Path,
    protein_source: str,
    esm_mode: str,
    embedding_key: str,
    embedding_dim: int,
    embedding_dtype: str,
    embedder: Optional[ESMEmbedder],
    embedding_dir: Optional[Path],
    path_mode: str,
    normalised_ligand_dir: Optional[Path] = None,
    store_ligand_mol: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Construct one LMDB record and its index row."""
    cid = normalise_complex_id(row["complex_id"])
    if protein_source == "protein":
        protein_path = files.protein_pdb
    elif protein_source == "pocket":
        protein_path = files.pocket_pdb
    elif protein_source == "auto":
        protein_path = files.protein_pdb or files.pocket_pdb
    else:
        raise ValueError("Unknown protein_source: {}".format(protein_source))
    if protein_path is None:
        raise FileNotFoundError("No usable protein/pocket PDB for {}".format(cid))
    if files.pocket_pdb is None:
        raise FileNotFoundError("Missing pocket PDB for {}".format(cid))
    ligand_input = files.ligand_path
    if ligand_input is None:
        raise FileNotFoundError("Missing ligand SDF/MOL2 for {}".format(cid))

    protein_features = parse_protein_pdb(protein_path)
    attach_embeddings(
        protein_features,
        mode=esm_mode,
        embedding_key=embedding_key,
        embedding_dim=embedding_dim,
        dtype=embedding_dtype,
        embedder=embedder,
        embedding_dir=embedding_dir,
        complex_id=cid,
    )
    mol = load_ligand_mol(ligand_input)

    ligand_path = ligand_input
    if normalised_ligand_dir is not None:
        from rdkit import Chem
        output_dir = normalised_ligand_dir / cid
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (cid + "_ligand.sdf")
        writer = Chem.SDWriter(str(output_path))
        writer.write(mol)
        writer.close()
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise OSError("Failed to write normalised ligand {}".format(output_path))
        ligand_path = output_path

    metadata = {
        "schema_version": 2,
        "dataset": "LP-PDBbind",
        "complex_id": cid,
        "split": str(row["split"]),
        "split_detail": str(row.get("split_detail", row["split"])),
        "affinity": float(row["affinity"]),
        "affinity_raw": str(row.get("affinity_raw", row["affinity"])),
        "affinity_type": str(row.get("affinity_type", "unknown")),
        "affinity_op": str(row.get("affinity_op", "=")),
        "affinity_unit": str(row.get("affinity_unit", "pK")),
        "protein_path": path_for_metadata(protein_path, structure_root, path_mode),
        "pocket_path": path_for_metadata(files.pocket_pdb, structure_root, path_mode),
        "ligand_path": path_for_metadata(ligand_path, structure_root, path_mode),
        "protein_source": "protein" if protein_path == files.protein_pdb else "pocket",
    }
    ligand_features: Dict[str, Any] = {}
    if store_ligand_mol:
        ligand_features["mol"] = mol
    entry = {
        "protein_features": protein_features,
        "ligand_features": ligand_features,
        "metadata": metadata,
    }
    sequence_length = sum(len(data["sequence"]) for data in protein_features.values())
    index_row = {
        "lmdb_key": cid + "_protein",
        "complex_id": cid,
        "affinity": float(row["affinity"]),
        "y": float(row["affinity"]),
        "split": str(row["split"]),
        "split_detail": str(row.get("split_detail", row["split"])),
        "affinity_raw": metadata["affinity_raw"],
        "affinity_type": metadata["affinity_type"],
        "affinity_op": metadata["affinity_op"],
        "affinity_unit": metadata["affinity_unit"],
        "protein_path": metadata["protein_path"],
        "pocket_path": metadata["pocket_path"],
        "ligand_path": metadata["ligand_path"],
        "protein_source": metadata["protein_source"],
        "num_chains": len(protein_features),
        "protein_length": sequence_length,
        "ligand_atoms": int(mol.GetNumAtoms()),
    }
    return entry, index_row


def merge_split_and_affinity(split_df: pd.DataFrame, affinity_df: pd.DataFrame) -> pd.DataFrame:
    split_ids = set(split_df["complex_id"])
    affinity_ids = set(affinity_df["complex_id"])
    merged = split_df.merge(affinity_df, on="complex_id", how="inner", validate="one_to_one")
    merged.attrs["split_without_affinity"] = sorted(split_ids.difference(affinity_ids))
    merged.attrs["affinity_without_split"] = sorted(affinity_ids.difference(split_ids))
    return merged


def serialisable_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [convert(item) for item in value]
        return value
    return convert(report)
