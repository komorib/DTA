from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit(args):
    root = Path(args.data_root).expanduser().resolve()
    splits = {}
    proteins = set()
    ligands = set()
    for split in ("train", "valid", "test"):
        path = root / "psichic" / (split + ".csv")
        frame = pd.read_csv(path)
        required = {"Protein", "Ligand", "classification_label", "sample_id"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError("%s missing columns: %s" % (path, missing))
        splits[split] = {"rows": int(len(frame)), "positives": int(frame["classification_label"].sum())}
        proteins.update(frame["Protein"].astype(str))
        ligands.update(frame["Ligand"].astype(str))
    metadata_path = Path(args.protein_cache).expanduser().resolve() / "metadata.json"
    cache_metadata = None
    if metadata_path.is_file():
        cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pooled_cache = bool(cache_metadata and "maximum_tokens" in cache_metadata and cache_metadata.get("count"))
    payload = {
        "version": "external-benchmark-psichic-kiba-preflight-v1",
        "data_root": str(root),
        "splits": splits,
        "unique_proteins": len(proteins),
        "unique_ligands": len(ligands),
        "protein_cache": str(Path(args.protein_cache).expanduser().resolve()),
        "protein_cache_metadata": str(metadata_path),
        "protein_cache_found": metadata_path.is_file(),
        "protein_cache_sha256_key_example": _sha256(sorted(proteins)[0]) if proteins else None,
        "current_cache_is_pooled_v21": pooled_cache,
        "required_protein_artifact": {
            "per_sequence": "protein.pt or equivalent keyed by sequence",
            "fields": ["seq", "seq_feat[L,33]", "token_representation[L,1280]", "edge_index[2,E]", "edge_weight[E]"],
            "edge_source": "ESM contact probabilities plus sequence +/-1 and +/-2 fallback edges",
        },
        "status": "blocked_missing_per_residue_contact_cache" if pooled_cache else "preflight_only",
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser(description="Audit PSICHIC KIBA inputs without running a misleading fallback model.")
    parser.add_argument("--data-root", default="data/ExternalBenchmarkKIBA_v1")
    parser.add_argument("--protein-cache", default="data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb")
    parser.add_argument("--output")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()

