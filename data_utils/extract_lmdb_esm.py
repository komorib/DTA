import argparse
import json
import os
import pickle
import shutil
from typing import List, Sequence

import lmdb
import numpy as np


def sequence_to_string(sequence_value):
    if hasattr(sequence_value, "reshape") and hasattr(sequence_value, "tolist"):
        return "".join(str(item) for item in sequence_value.reshape(-1).tolist())
    if isinstance(sequence_value, (list, tuple)):
        return "".join(str(item) for item in sequence_value)
    if isinstance(sequence_value, bytes):
        return sequence_value.decode("utf-8")
    return str(sequence_value)


def load_esm(model_path: str, device: str):
    import torch
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = EsmModel.from_pretrained(model_path, local_files_only=True, add_pooling_layer=False)
    model = model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def embed_sequences(sequences: Sequence[str], tokenizer, model, device: str, batch_size: int) -> List[np.ndarray]:
    import torch

    embeddings = []
    hidden_size = int(model.config.hidden_size)
    torch_device = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = list(sequences[start:start + batch_size])
            encoded = tokenizer(batch_sequences, return_tensors="pt", padding=True, add_special_tokens=True)
            encoded = {key: value.to(torch_device) for key, value in encoded.items()}
            output = model(**encoded)
            hidden = output.last_hidden_state.detach().cpu()
            for row, sequence in enumerate(batch_sequences):
                length = len(sequence)
                embedding = hidden[row, 1:1 + length, :].numpy().astype(np.float32)
                if embedding.shape != (length, hidden_size):
                    raise ValueError(f"ESM embedding shape {embedding.shape} does not match sequence length {length}")
                embeddings.append(embedding)
    return embeddings


def checkpoint_state_path(output_lmdb):
    return os.path.join(output_lmdb, "esm_state.json")


def write_json_atomic(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def output_key_exists(env, key):
    with env.begin(write=False) as txn:
        return txn.get(key.encode("utf-8")) is not None


def process_pending(pending, output_env, tokenizer, model, device, esm_batch_size, state):
    if not pending:
        return
    sequences = []
    refs = []
    for item_index, payload in enumerate(pending):
        protein_features = payload["obj"]["protein_features"]
        for chain_id in sorted(protein_features):
            chain = protein_features[chain_id]
            sequence = sequence_to_string(chain.get("sequence", ""))
            if not sequence:
                raise ValueError(f"empty sequence for {payload['key']} chain {chain_id}")
            sequences.append(sequence)
            refs.append((item_index, chain_id))

    embeddings = embed_sequences(sequences, tokenizer, model, device=device, batch_size=esm_batch_size)
    hidden_size = embeddings[0].shape[1] if embeddings else 0
    for embedding, (item_index, chain_id) in zip(embeddings, refs):
        pending[item_index]["obj"]["protein_features"][chain_id]["esm2_embedding"] = embedding

    with output_env.begin(write=True) as txn:
        for payload in pending:
            txn.put(payload["key"].encode("utf-8"), pickle.dumps(payload["obj"]))
            state["processed_samples"] += 1
            state["processed_keys"].append(payload["key"])
    state["processed_chains"] += len(refs)
    state["hidden_size"] = hidden_size


def run(args):
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = args.device or "cuda:0"

    with open(args.index_pkl, "rb") as handle:
        rows = pickle.load(handle)
    os.makedirs(args.output_lmdb, exist_ok=True)
    output_index = args.output_index_pkl or os.path.join(args.output_lmdb, "index.pkl")
    os.makedirs(os.path.dirname(output_index) or ".", exist_ok=True)
    shutil.copyfile(args.index_pkl, output_index)

    input_env = lmdb.open(args.input_lmdb, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    output_env = lmdb.open(args.output_lmdb, map_size=args.map_size * 1024**3, subdir=True)

    state_path = checkpoint_state_path(args.output_lmdb)
    state = {
        "input_lmdb": os.path.abspath(args.input_lmdb),
        "output_lmdb": os.path.abspath(args.output_lmdb),
        "index_pkl": os.path.abspath(args.index_pkl),
        "esm_model_path": os.path.abspath(args.esm_model_path),
        "processed_samples": 0,
        "processed_chains": 0,
        "skipped_existing": 0,
        "processed_keys": [],
        "hidden_size": None,
    }
    if args.resume and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            old_state = json.load(handle)
        state.update(old_state)
        state.setdefault("processed_keys", [])

    tokenizer, model = load_esm(args.esm_model_path, device=device)
    hidden_size = int(model.config.hidden_size)
    if hidden_size != args.expected_hidden_size:
        raise ValueError(f"ESM hidden_size={hidden_size}, expected {args.expected_hidden_size}")

    pending = []
    processed_this_run = 0
    with input_env.begin(write=False) as txn:
        for row in rows:
            if args.max_samples and processed_this_run >= args.max_samples:
                break
            key = row["lmdb_key"]
            if args.resume and output_key_exists(output_env, key):
                state["skipped_existing"] += 1
                continue
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                raise KeyError(f"input LMDB key not found: {key}")
            obj = pickle.loads(raw)
            pending.append({"key": key, "obj": obj})
            processed_this_run += 1
            if len(pending) >= args.sample_batch_size:
                process_pending(pending, output_env, tokenizer, model, device, args.esm_batch_size, state)
                pending = []
                write_json_atomic(state_path, state)
                print(json.dumps({"processed_samples": state["processed_samples"], "skipped_existing": state["skipped_existing"]}), flush=True)

    process_pending(pending, output_env, tokenizer, model, device, args.esm_batch_size, state)
    write_json_atomic(state_path, state)
    output_env.close()
    input_env.close()
    print(json.dumps(state, indent=2))
    print(f"wrote LMDB: {args.output_lmdb}")
    print(f"wrote index: {output_index}")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract ESM2 embeddings into a PDBbind chain LMDB.")
    parser.add_argument("--input_lmdb", required=True)
    parser.add_argument("--index_pkl", required=True)
    parser.add_argument("--output_lmdb", required=True)
    parser.add_argument("--output_index_pkl", default=None)
    parser.add_argument("--esm_model_path", default="/home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D")
    parser.add_argument("--cuda", default=None, help="Physical GPU id, e.g. 7. Sets CUDA_VISIBLE_DEVICES before loading ESM.")
    parser.add_argument("--device", default=None, help="Torch device after CUDA_VISIBLE_DEVICES mapping; default cuda:0.")
    parser.add_argument("--sample_batch_size", type=int, default=8, help="Number of LMDB samples to embed before writing.")
    parser.add_argument("--esm_batch_size", type=int, default=16, help="Number of chains passed to ESM at once.")
    parser.add_argument("--expected_hidden_size", type=int, default=640)
    parser.add_argument("--map_size", type=int, default=200, help="Output LMDB map size in GB.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
