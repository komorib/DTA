#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DRUGCLIP_DTA_DIR = Path("/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen")
if str(DRUGCLIP_DTA_DIR) not in sys.path:
    sys.path.insert(0, str(DRUGCLIP_DTA_DIR))

from data_utils.lmdb_dataset import resolve_manifest_path  # noqa: E402
from lmdb_embedding_cache import (  # noqa: E402
    EmbeddingCacheWriter,
    export_cache_metadata_json,
    write_cache_metadata,
)


def load_manifest_paths(manifest_dir, splits):
    paths = [Path(manifest_dir) / f"{split}_manifest.json" for split in splits]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"manifest files not found: {missing}")
    return paths


def collect_samples(manifest_paths):
    samples = []
    seen = set()
    lmdb_paths = set()
    split_counts = {}
    for path in manifest_paths:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("dataset_type") != "pdbbind_lmdb":
            raise ValueError(f"not a PDBbind LMDB manifest: {path}")
        manifest_dir = os.path.dirname(os.path.abspath(path))
        lmdb_paths.add(os.path.abspath(resolve_manifest_path(manifest_dir, manifest["lmdb_path"])))
        split_name = manifest.get("split") or path.name.replace("_manifest.json", "")
        split_counts[split_name] = len(manifest.get("samples", []))
        for sample in manifest.get("samples", []):
            key = sample["lmdb_key"]
            if key in seen:
                continue
            seen.add(key)
            samples.append(sample)
    if len(lmdb_paths) != 1:
        raise ValueError(f"all manifests must use the same feature LMDB, got {sorted(lmdb_paths)}")
    return samples, sorted(lmdb_paths)[0], split_counts


def open_read_lmdb(path):
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True, max_readers=256)


def load_feature_item(txn, key):
    raw = txn.get(key.encode("utf-8"))
    if raw is None:
        raise KeyError(f"feature LMDB key not found: {key}")
    return pickle.loads(raw)


def ligand_atoms_and_coords(ligand_features, remove_hs=True):
    if "mol" not in ligand_features:
        raise ValueError("ligand_features is missing RDKit mol")
    mol = ligand_features["mol"]
    if mol.GetNumConformers() == 0:
        raise ValueError("ligand mol has no conformer")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if len(atoms) != len(coords):
        raise ValueError(f"atom/coord length mismatch: atoms={len(atoms)} coords={len(coords)}")
    if remove_hs:
        keep = [idx for idx, atom in enumerate(atoms) if atom != "H"]
        atoms = [atoms[idx] for idx in keep]
        coords = coords[keep]
    if not atoms:
        raise ValueError("ligand has no usable atoms after hydrogen filtering")
    return atoms, np.asarray(coords, dtype=np.float32)


def create_unimol_repr(args):
    weight_dir = Path(args.weight_dir).resolve()
    required = ["mol.dict.txt", "mol_pre_no_h_220816.pt" if args.remove_hs else "mol_pre_all_h_220816.pt"]
    missing = [name for name in required if not (weight_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing Uni-Mol files in {weight_dir}: {missing}")

    os.environ["UNIMOL_WEIGHT_DIR"] = str(weight_dir)
    tools_root = Path(args.unimol_tools_root).resolve()
    if not (tools_root / "unimol_tools" / "__init__.py").exists():
        raise FileNotFoundError(f"unimol_tools package not found under {tools_root}")
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))

    from unimol_tools import UniMolRepr

    return UniMolRepr(
        data_type="molecule",
        batch_size=int(args.batch_size),
        remove_hs=bool(args.remove_hs),
        model_name=args.model_name,
        model_size=args.model_size,
        use_cuda=bool(args.use_cuda),
        use_gpu=args.use_gpu,
    )


def build_payload_records(records, repr_output):
    cls_repr = repr_output.get("cls_repr")
    atomic_reprs = repr_output.get("atomic_reprs")
    if cls_repr is None or atomic_reprs is None:
        raise KeyError("Uni-Mol repr output must contain cls_repr and atomic_reprs")
    if len(records) != len(cls_repr) or len(records) != len(atomic_reprs):
        raise ValueError(f"repr output length mismatch: records={len(records)} cls={len(cls_repr)} atomic={len(atomic_reprs)}")

    payloads = []
    for idx, record in enumerate(records):
        token_embedding = np.asarray(atomic_reprs[idx], dtype=np.float32)
        pooled_embedding = np.asarray(cls_repr[idx], dtype=np.float32)
        if pooled_embedding.ndim == 1:
            pooled_embedding = pooled_embedding[None, :]
        if token_embedding.ndim != 2 or pooled_embedding.ndim != 2:
            raise ValueError(
                f"invalid Uni-Mol embedding shape for {record['lmdb_key']}: "
                f"token={token_embedding.shape} pooled={pooled_embedding.shape}"
            )
        if token_embedding.shape[0] != len(record["atoms"]):
            raise ValueError(
                f"Uni-Mol atom order length mismatch for {record['lmdb_key']}: "
                f"tokens={token_embedding.shape[0]} atoms={len(record['atoms'])}"
            )
        payloads.append(
            {
                "lmdb_key": record["lmdb_key"],
                "token_embedding": token_embedding,
                "token_attention_mask": np.ones((token_embedding.shape[0],), dtype=np.uint8),
                "pooled_embedding": pooled_embedding,
                "pooled_attention_mask": np.ones((pooled_embedding.shape[0],), dtype=np.uint8),
                "meta": {
                    "lmdb_key": record["lmdb_key"],
                    "pdb_id": record.get("pdb_id"),
                    "encoder_family": "unimol",
                    "source": "GerNA_ligand_mol_atoms_coordinates",
                    "key_policy": "lmdb_key",
                    "atomic_symbols": list(record["atoms"]),
                    "atom_order": "GerNA_LMDB_RDKit_mol_order_after_remove_hs",
                },
            }
        )
    return payloads


def chunked(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract(args):
    manifest_paths = load_manifest_paths(args.manifest_dir, args.splits)
    samples, feature_lmdb_path, split_counts = collect_samples(manifest_paths)
    if args.max_samples > 0:
        samples = samples[: int(args.max_samples)]

    feature_env = open_read_lmdb(feature_lmdb_path)
    token_writer = EmbeddingCacheWriter(args.output_token_lmdb, map_size_gb=args.token_map_size_gb)
    pooled_writer = EmbeddingCacheWriter(args.output_pooled_lmdb, map_size_gb=args.pooled_map_size_gb)
    repr_model = create_unimol_repr(args)

    stats = {
        "manifest_dir": os.path.abspath(args.manifest_dir),
        "feature_lmdb": os.path.abspath(feature_lmdb_path),
        "output_token_lmdb": os.path.abspath(args.output_token_lmdb),
        "output_pooled_lmdb": os.path.abspath(args.output_pooled_lmdb),
        "split_counts": split_counts,
        "candidate_samples": len(samples),
        "written": 0,
        "skipped_existing": 0,
        "feature_missing": 0,
        "shape_errors": 0,
        "remove_hs": bool(args.remove_hs),
        "output_key_policy": "lmdb_key",
        "token_hidden_size": None,
        "pooled_hidden_size": None,
    }
    errors = []

    try:
        with feature_env.begin(write=False) as txn:
            for batch in tqdm(list(chunked(samples, int(args.batch_size))), desc="Extracting GerNA-order Uni-Mol"):
                pending = []
                for sample in batch:
                    key = sample["lmdb_key"]
                    if args.resume and token_writer.contains(key) and pooled_writer.contains(key):
                        stats["skipped_existing"] += 1
                        continue
                    try:
                        item = load_feature_item(txn, key)
                        atoms, coords = ligand_atoms_and_coords(item["ligand_features"], remove_hs=bool(args.remove_hs))
                        pending.append({"lmdb_key": key, "pdb_id": sample.get("pdb_id"), "atoms": atoms, "coordinates": coords})
                    except KeyError as exc:
                        stats["feature_missing"] += 1
                        errors.append({"key": key, "error": str(exc)})
                    except Exception as exc:
                        stats["shape_errors"] += 1
                        errors.append({"key": key, "error": str(exc)})
                    if args.fail_on_error and errors:
                        raise RuntimeError(errors[-1])
                if not pending:
                    continue

                data = {
                    "atoms": [record["atoms"] for record in pending],
                    "coordinates": [record["coordinates"] for record in pending],
                }
                repr_output = repr_model.get_repr(data, return_atomic_reprs=True)
                payloads = build_payload_records(pending, repr_output)
                for payload in payloads:
                    token_writer.put(
                        payload["lmdb_key"],
                        payload["token_embedding"],
                        payload["token_attention_mask"],
                        extra_meta=payload["meta"],
                    )
                    pooled_writer.put(
                        payload["lmdb_key"],
                        payload["pooled_embedding"],
                        payload["pooled_attention_mask"],
                        extra_meta=payload["meta"],
                    )
                if payloads:
                    stats["written"] += len(payloads)
                    stats["token_hidden_size"] = int(payloads[0]["token_embedding"].shape[1])
                    stats["pooled_hidden_size"] = int(payloads[0]["pooled_embedding"].shape[1])
                if args.progress_every > 0 and stats["written"] % int(args.progress_every) < len(payloads):
                    print(json.dumps({k: stats[k] for k in ("candidate_samples", "written", "skipped_existing", "feature_missing", "shape_errors")}), flush=True)

        stats["errors_preview"] = errors[:20]
        token_meta = {**stats, "representation": "token", "cache_name": "gerna_order_ligand_unimol_token"}
        pooled_meta = {**stats, "representation": "pooled", "cache_name": "gerna_order_ligand_unimol_pooled"}
        write_cache_metadata(token_writer.env, token_meta)
        write_cache_metadata(pooled_writer.env, pooled_meta)
        export_cache_metadata_json(args.output_token_lmdb + ".meta.json", token_meta)
        export_cache_metadata_json(args.output_pooled_lmdb + ".meta.json", pooled_meta)
        if args.metadata_output:
            Path(args.metadata_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.metadata_output).write_text(json.dumps({"token_meta": token_meta, "pooled_meta": pooled_meta}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"token_meta": token_meta, "pooled_meta": pooled_meta}, indent=2), flush=True)
    finally:
        feature_env.close()
        token_writer.close()
        pooled_writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Uni-Mol ligand embeddings aligned to GerNA LMDB ligand atom order.")
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--output_token_lmdb", required=True)
    parser.add_argument("--output_pooled_lmdb", required=True)
    parser.add_argument("--metadata_output", default="")
    parser.add_argument("--unimol_tools_root", default="/home/user/wangzhiwen/DrugCLIP/DrugCLIP/Uni-Mol-Repo/unimol_tools")
    parser.add_argument("--weight_dir", default="/home/user/wangzhiwen/DrugCLIP/pretrained_models/unimol")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--token_map_size_gb", type=int, default=64)
    parser.add_argument("--pooled_map_size_gb", type=int, default=8)
    parser.add_argument("--model_name", choices=["unimolv1", "unimolv2"], default="unimolv1")
    parser.add_argument("--model_size", default="84m")
    parser.set_defaults(remove_hs=True)
    parser.add_argument("--remove_hs", dest="remove_hs", action="store_true")
    parser.add_argument("--no-remove_hs", dest="remove_hs", action="store_false")
    parser.set_defaults(use_cuda=False)
    parser.add_argument("--use_cuda", dest="use_cuda", action="store_true")
    parser.add_argument("--no-use_cuda", dest="use_cuda", action="store_false")
    parser.add_argument("--use_gpu", default="0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main():
    extract(parse_args())


if __name__ == "__main__":
    main()
