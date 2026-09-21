#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

DRUGCLIP_DTA_DIR = Path("/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen")
if str(DRUGCLIP_DTA_DIR) not in sys.path:
    sys.path.insert(0, str(DRUGCLIP_DTA_DIR))

from lmdb_embedding_cache import EmbeddingCacheWriter, export_cache_metadata_json, write_cache_metadata  # noqa: E402


def collect_records(jsonl_path, max_samples=0):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            atoms = list(item["atoms"])
            coords = np.asarray(item["coordinates"], dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f"invalid coordinates shape for {item.get('lmdb_key')}: {coords.shape}")
            if len(atoms) != len(coords):
                raise ValueError(f"atom/coord length mismatch for {item.get('lmdb_key')}: {len(atoms)} vs {len(coords)}")
            records.append({
                "lmdb_key": item["lmdb_key"],
                "pdb_id": item.get("pdb_id"),
                "atoms": atoms,
                "coordinates": coords,
            })
            if max_samples > 0 and len(records) >= int(max_samples):
                break
    return records


def chunked(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


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


def build_payloads(records, repr_output):
    cls_repr = repr_output.get("cls_repr")
    atomic_reprs = repr_output.get("atomic_reprs")
    if cls_repr is None or atomic_reprs is None:
        raise KeyError("Uni-Mol repr output must contain cls_repr and atomic_reprs")
    if len(records) != len(cls_repr) or len(records) != len(atomic_reprs):
        raise ValueError(f"repr output length mismatch: records={len(records)} cls={len(cls_repr)} atomic={len(atomic_reprs)}")

    payloads = []
    for idx, record in enumerate(records):
        token = np.asarray(atomic_reprs[idx], dtype=np.float32)
        pooled = np.asarray(cls_repr[idx], dtype=np.float32)
        if pooled.ndim == 1:
            pooled = pooled[None, :]
        if token.ndim != 2 or pooled.ndim != 2:
            raise ValueError(f"invalid embedding shape for {record['lmdb_key']}: token={token.shape} pooled={pooled.shape}")
        if token.shape[0] != len(record["atoms"]):
            raise ValueError(f"token/atom length mismatch for {record['lmdb_key']}: {token.shape[0]} vs {len(record['atoms'])}")
        meta = {
            "lmdb_key": record["lmdb_key"],
            "pdb_id": record.get("pdb_id"),
            "encoder_family": "unimol",
            "source": "GerNA_ligand_atoms_coordinates_jsonl",
            "key_policy": "lmdb_key",
            "atomic_symbols": list(record["atoms"]),
            "atom_order": "GerNA_LMDB_RDKit_mol_order_after_remove_hs",
        }
        payloads.append((record["lmdb_key"], token, np.ones(token.shape[0], dtype=np.uint8), pooled, np.ones(pooled.shape[0], dtype=np.uint8), meta))
    return payloads


def extract(args):
    records = collect_records(args.input_atoms_jsonl, max_samples=args.max_samples)
    token_writer = EmbeddingCacheWriter(args.output_token_lmdb, map_size_gb=args.token_map_size_gb)
    pooled_writer = EmbeddingCacheWriter(args.output_pooled_lmdb, map_size_gb=args.pooled_map_size_gb)
    model = create_unimol_repr(args)

    stats = {
        "input_atoms_jsonl": os.path.abspath(args.input_atoms_jsonl),
        "output_token_lmdb": os.path.abspath(args.output_token_lmdb),
        "output_pooled_lmdb": os.path.abspath(args.output_pooled_lmdb),
        "candidate_samples": len(records),
        "written": 0,
        "skipped_existing": 0,
        "shape_errors": 0,
        "remove_hs": bool(args.remove_hs),
        "output_key_policy": "lmdb_key",
        "token_hidden_size": None,
        "pooled_hidden_size": None,
        "errors_preview": [],
    }

    try:
        for batch in tqdm(chunked(records, int(args.batch_size)), desc="Extracting GerNA-order Uni-Mol"):
            pending = [
                record for record in batch
                if not (args.resume and token_writer.contains(record["lmdb_key"]) and pooled_writer.contains(record["lmdb_key"]))
            ]
            stats["skipped_existing"] += len(batch) - len(pending)
            if not pending:
                continue
            try:
                data = {
                    "atoms": [record["atoms"] for record in pending],
                    "coordinates": [record["coordinates"] for record in pending],
                }
                repr_output = model.get_repr(data, return_atomic_reprs=True)
                payloads = build_payloads(pending, repr_output)
                for key, token, token_mask, pooled, pooled_mask, meta in payloads:
                    token_writer.put(key, token, token_mask, extra_meta=meta)
                    pooled_writer.put(key, pooled, pooled_mask, extra_meta=meta)
                if payloads:
                    stats["written"] += len(payloads)
                    stats["token_hidden_size"] = int(payloads[0][1].shape[1])
                    stats["pooled_hidden_size"] = int(payloads[0][3].shape[1])
            except Exception as exc:
                stats["shape_errors"] += len(pending)
                if len(stats["errors_preview"]) < 20:
                    stats["errors_preview"].append({"keys": [r["lmdb_key"] for r in pending], "error": str(exc)})
                if args.fail_on_error:
                    raise
            if args.progress_every > 0 and stats["written"] % int(args.progress_every) < max(len(pending), 1):
                print(json.dumps({k: stats[k] for k in ("candidate_samples", "written", "skipped_existing", "shape_errors")}), flush=True)

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
        token_writer.close()
        pooled_writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Uni-Mol embeddings from GerNA-order ligand atoms/coordinates JSONL.")
    parser.add_argument("--input_atoms_jsonl", required=True)
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
