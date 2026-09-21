"""CPU-only audit and lightweight feature/forward smoke for the KIBA bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_alignment(root: Path, split: str, rows: list[dict]) -> dict:
    graphdta = root / "graphdta" / (split + ".csv")
    psichic = root / "psichic" / (split + ".csv")
    transformercpi = root / "transformercpi" / (split + ".txt")
    graphormer = root / "graphormer" / (split + ".txt")
    fmca = root / "fmca" / (split + ".data")
    with graphdta.open(encoding="utf-8", newline="") as handle:
        graph_rows = list(csv.DictReader(handle))
    with psichic.open(encoding="utf-8", newline="") as handle:
        psichic_rows = list(csv.DictReader(handle))
    text_counts = {
        "transformercpi": sum(1 for _ in transformercpi.open(encoding="utf-8")),
        "graphormer": sum(1 for _ in graphormer.open(encoding="utf-8")),
        "fmca": sum(1 for _ in fmca.open(encoding="utf-8")),
    }
    if len(graph_rows) != len(rows) or len(psichic_rows) != len(rows) or any(
        count != len(rows) for count in text_counts.values()
    ):
        raise AssertionError("model-specific file count mismatch for %s" % split)
    for index in (0, len(rows) // 2, len(rows) - 1):
        row = rows[index]
        graph_row = graph_rows[index]
        psichic_row = psichic_rows[index]
        if graph_row["sample_id"] != row["sample_id"] or psichic_row["sample_id"] != row["sample_id"]:
            raise AssertionError("row order mismatch for %s index %d" % (split, index))
        if int(graph_row["label"]) != row["label"] or int(psichic_row["classification_label"]) != row["label"]:
            raise AssertionError("label mismatch for %s index %d" % (split, index))
    return {
        "rows": len(rows),
        "sha256": {
            "graphdta": _sha256(graphdta),
            "psichic": _sha256(psichic),
            "transformercpi": _sha256(transformercpi),
            "graphormer": _sha256(graphormer),
            "fmca": _sha256(fmca),
        },
    }


def _check_rdkit(rows: list[dict], sample_size: int) -> dict:
    from rdkit import Chem

    selected = rows[:sample_size]
    parsed = 0
    atom_counts = []
    for row in selected:
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            raise AssertionError("RDKit failed to parse %s" % row["sample_id"])
        parsed += 1
        atom_counts.append(molecule.GetNumAtoms())
    return {"sampled": len(selected), "parsed": parsed, "max_atoms": max(atom_counts or [0])}


def _check_cpu_forward(rows: list[dict], sample_size: int) -> dict:
    """Run a tiny CPU-only differentiable pair head as an input sanity check."""
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    selected = rows[:sample_size]
    features = []
    labels = []
    for row in selected:
        sequence = row["sequence"]
        sequence_features = [float(sequence.count(aa)) / max(len(sequence), 1) for aa in amino_acids]
        molecule = Chem.MolFromSmiles(row["smiles"])
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=128)
        ligand_features = np.zeros(128, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, ligand_features)
        features.append(sequence_features + ligand_features.tolist())
        labels.append(float(row["label"]))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    logits = model(x).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    if not torch.isfinite(loss) or any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
    ):
        raise AssertionError("non-finite CPU forward/backward result")
    return {"sampled": len(selected), "input_width": int(x.shape[1]), "loss": float(loss.item())}


def _check_esm_cache(root: Path, rows: list[dict], sample_size: int, cache_path: Path | None = None) -> dict:
    cache = cache_path or (Path(__file__).resolve().parents[2] / "data" / "OptionalStructureMultitask_v21" / "proteins_esm2_t33_v21.lmdb")
    if not cache.exists():
        return {"status": "not_found", "path": str(cache)}
    import lmdb
    import pickle

    selected = rows[:sample_size]
    environment = lmdb.open(str(cache), readonly=True, lock=False, readahead=False, max_readers=32)
    shapes = []
    with environment.begin(write=False) as transaction:
        for row in selected:
            payload = transaction.get(row["sequence_id"].encode("utf-8"))
            if payload is None:
                raise AssertionError("ESM cache missing %s" % row["sample_id"])
            embedding = np.asarray(pickle.loads(payload)["embedding"])
            if embedding.ndim != 2 or embedding.shape[1] != 1280:
                raise AssertionError("unexpected ESM shape %s" % (embedding.shape,))
            shapes.append(list(embedding.shape))
    return {
        "status": "checked",
        "path": str(cache),
        "sampled": len(shapes),
        "widths": sorted({shape[1] for shape in shapes}),
        "maximum_rows": max((shape[0] for shape in shapes), default=0),
    }


def run(root: Path, sample_size: int = 16, esm_cache: Path | None = None) -> dict:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    checks = {}
    for split in ("train", "valid", "test"):
        rows = _read_jsonl(root / "common" / (split + ".jsonl"))
        if not rows:
            raise AssertionError("empty split: %s" % split)
        checks[split] = {
            "alignment": _check_alignment(root, split, rows),
            "rdkit": _check_rdkit(rows, min(sample_size, len(rows))),
            "cpu_forward": _check_cpu_forward(rows, min(sample_size, len(rows))),
        }
        if split == "test":
            checks[split]["esm_cache"] = _check_esm_cache(
                root, rows, min(sample_size, len(rows)), esm_cache
            )
    counts = metadata["audit"]["counts"]
    for split, result in checks.items():
        if result["alignment"]["rows"] != counts[split]:
            raise AssertionError("metadata count mismatch for %s" % split)
    return {
        "version": "external-benchmark-kiba-binary-smoke-v1",
        "root": str(root.resolve()),
        "metadata_version": metadata.get("version"),
        "checks": checks,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ExternalBenchmarkKIBA_v1"))
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--esm-cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.sample_size, args.esm_cache), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
