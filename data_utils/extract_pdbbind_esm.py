import argparse
import json
import os
import pickle
import sys
from typing import Iterable, List, Sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import torch


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def residue_ids_to_sequence(residue_ids: Iterable[int]) -> str:
    """Convert PDBbind residue ids 1..20 back to an ESM-compatible sequence."""
    sequence = []
    for residue_id in np.asarray(list(residue_ids)).reshape(-1):
        residue_id = int(residue_id)
        if 1 <= residue_id <= len(AA_LIST):
            sequence.append(AA_LIST[residue_id - 1])
        else:
            sequence.append("X")
    return "".join(sequence)


def sequence_for_sample(protein_feats, metadata=None, expected_length=None) -> str:
    metadata = metadata or {}
    if metadata.get("residue_sequence"):
        sequence = str(metadata["residue_sequence"])
    elif metadata.get("residue_ids"):
        sequence = residue_ids_to_sequence(metadata["residue_ids"])
    else:
        sequence = residue_ids_to_sequence(protein_feats)

    if expected_length is not None and len(sequence) != int(expected_length):
        raise ValueError(
            f"ESM sequence length {len(sequence)} does not match residue feature length {expected_length}"
        )
    return sequence


def checkpoint_chunk_files(checkpoint_dir: str) -> List[str]:
    return [
        os.path.join(checkpoint_dir, name)
        for name in sorted(os.listdir(checkpoint_dir))
        if name.startswith("chunk_") and name.endswith(".pkl")
    ]


def replace_protein_embeddings(chunk_data: list, embeddings: Sequence[np.ndarray]) -> list:
    """Write ESM embeddings into both 1D protein features and PyG graph nodes."""
    if len(chunk_data) != 10:
        raise ValueError(f"expected 10 data fields, got {len(chunk_data)}")
    if len(chunk_data[0]) != len(embeddings):
        raise ValueError(
            f"sample count mismatch: chunk has {len(chunk_data[0])}, embeddings has {len(embeddings)}"
        )

    for index, embedding in enumerate(embeddings):
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 2:
            raise ValueError(f"sample {index}: ESM embedding must be 2D, got {embedding.shape}")
        residue_count = len(chunk_data[0][index])
        if embedding.shape[0] != residue_count:
            raise ValueError(
                f"sample {index}: embedding length {embedding.shape[0]} != residue count {residue_count}"
            )

        # GerNA-Bind 的 1D 分支直接读取 data[0]，图分支读取 protein_graph.x。
        # 两处必须保持同一个 [L, 640] ESM2 表征，否则 1D/2D 分支看到的蛋白特征会不一致。
        chunk_data[0][index] = embedding
        graph = chunk_data[2][index]
        if not hasattr(graph, "x"):
            raise ValueError(f"sample {index}: protein graph has no x attribute")
        graph.x = torch.tensor(embedding, dtype=torch.float32)
    return chunk_data


def load_esm(model_path: str, device: torch.device):
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = EsmModel.from_pretrained(
        model_path,
        local_files_only=True,
        add_pooling_layer=False,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


def embed_sequences(
    sequences: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> List[np.ndarray]:
    embeddings = []
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("ESM model config does not expose a positive hidden_size")

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = list(sequences[start:start + batch_size])
            encoded = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            hidden = output.last_hidden_state.detach().cpu()

            for row, sequence in enumerate(batch_sequences):
                # HuggingFace ESM 在序列前后加 <cls>/<eos>；真实残基对应 1:1+L。
                length = len(sequence)
                embedding = hidden[row, 1:1 + length, :].numpy().astype(np.float32)
                if embedding.shape != (length, hidden_size):
                    raise ValueError(
                        f"ESM embedding shape {embedding.shape} does not match sequence length {length}"
                    )
                embeddings.append(embedding)
    return embeddings


def slice_chunk_payload(payload: dict, keep_count: int) -> dict:
    return {
        "data": [field[:keep_count] for field in payload["data"]],
        "metadata": payload.get("metadata", [])[:keep_count],
    }


def write_json_atomic(path: str, payload: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def write_pickle_atomic(path: str, payload: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle)
    os.replace(tmp_path, path)


def process_checkpoint_dir(args) -> dict:
    if not os.path.isdir(args.input_checkpoint_dir):
        raise FileNotFoundError(args.input_checkpoint_dir)
    os.makedirs(args.output_checkpoint_dir, exist_ok=True)

    input_chunks = checkpoint_chunk_files(args.input_checkpoint_dir)
    if not input_chunks:
        raise FileNotFoundError(f"no chunk_*.pkl found in {args.input_checkpoint_dir}")

    device = torch.device(args.device)
    tokenizer, model = load_esm(args.esm_model_path, device)
    hidden_size = int(model.config.hidden_size)
    if hidden_size != args.expected_hidden_size:
        raise ValueError(
            f"ESM hidden_size={hidden_size}, expected {args.expected_hidden_size}; "
            "change --expected_hidden_size only if model.py input_dim_rna is also changed"
        )

    state = {
        "input_checkpoint_dir": os.path.abspath(args.input_checkpoint_dir),
        "output_checkpoint_dir": os.path.abspath(args.output_checkpoint_dir),
        "esm_model_path": os.path.abspath(args.esm_model_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "hidden_size": hidden_size,
        "total_input_chunks": len(input_chunks),
        "processed_chunks": 0,
        "skipped_chunks": 0,
        "processed_samples": 0,
        "output_chunks": [],
    }

    processed_samples = 0
    processed_chunks = 0
    for input_path in input_chunks:
        if args.max_chunks and processed_chunks >= args.max_chunks:
            break
        if args.max_samples and processed_samples >= args.max_samples:
            break

        chunk_name = os.path.basename(input_path)
        output_path = os.path.join(args.output_checkpoint_dir, chunk_name)
        if os.path.exists(output_path):
            if args.resume:
                with open(output_path, "rb") as handle:
                    existing_payload = pickle.load(handle)
                existing_size = len(existing_payload["data"][-1])
                state["skipped_chunks"] += 1
                state["processed_samples"] += existing_size
                state["output_chunks"].append(chunk_name)
                processed_chunks += 1
                processed_samples += existing_size
                print(json.dumps({"skip_existing": chunk_name, "samples": existing_size}), flush=True)
                continue
            if not args.overwrite:
                raise RuntimeError(f"output chunk exists: {output_path}; pass --resume or --overwrite")

        with open(input_path, "rb") as handle:
            payload = pickle.load(handle)
        chunk_size = len(payload["data"][-1])
        if args.max_samples:
            keep_count = min(chunk_size, args.max_samples - processed_samples)
            payload = slice_chunk_payload(payload, keep_count)
            chunk_size = keep_count

        chunk_data = payload["data"]
        metadata = payload.get("metadata") or [{} for _ in range(chunk_size)]
        if len(metadata) < chunk_size:
            metadata = list(metadata) + [{} for _ in range(chunk_size - len(metadata))]
        sequences = [
            sequence_for_sample(
                chunk_data[3][index],
                metadata[index],
                expected_length=len(chunk_data[0][index]),
            )
            for index in range(chunk_size)
        ]
        embeddings = embed_sequences(
            sequences=sequences,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
        )
        replace_protein_embeddings(chunk_data, embeddings)
        payload["data"] = chunk_data
        payload["esm"] = {
            "model_path": os.path.abspath(args.esm_model_path),
            "hidden_size": hidden_size,
            "sequence_field": "protein_feats",
        }
        for item, sequence in zip(payload.get("metadata", []), sequences):
            item["esm_sequence_length"] = len(sequence)

        write_pickle_atomic(output_path, payload)
        processed_chunks += 1
        processed_samples += chunk_size
        state["processed_chunks"] += 1
        state["processed_samples"] += chunk_size
        state["output_chunks"].append(chunk_name)
        write_json_atomic(os.path.join(args.output_checkpoint_dir, "state.json"), state)
        print(json.dumps({"processed_chunk": chunk_name, "samples": chunk_size}), flush=True)

    write_json_atomic(os.path.join(args.output_checkpoint_dir, "state.json"), state)
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pre-extract local ESM2 embeddings for PDBbind protein-ligand chunks."
    )
    parser.add_argument("--input_checkpoint_dir", default="data/pdbbind_protein_ligand.pkl.checkpoint")
    parser.add_argument("--output_checkpoint_dir", default="data/pdbbind_protein_ligand_esm.pkl.checkpoint")
    parser.add_argument(
        "--esm_model_path",
        default="/home/user/wangzhiwen/DrugCLIP/pretrained_models/esm2_t30_150M_UR50D",
    )
    parser.add_argument("--cuda", default=None, help="Physical GPU id, e.g. 7. Sets CUDA_VISIBLE_DEVICES before model load.")
    parser.add_argument("--device", default=None, help="Torch device after CUDA_VISIBLE_DEVICES mapping, default cuda:0 if available else cpu.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--expected_hidden_size", type=int, default=640)
    parser.add_argument("--max_chunks", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = process_checkpoint_dir(args)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
