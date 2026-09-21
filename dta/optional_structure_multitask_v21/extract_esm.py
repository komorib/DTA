"""Extract full-sequence ESM embeddings with overlap stitching and adaptive pooling."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

from .data import read_jsonl, sha256_file


VERSION = "optional-structure-full-sequence-esm-v21"


def sequence_windows(sequence, window_size, overlap):
    if window_size <= overlap:
        raise ValueError("window size must exceed overlap")
    starts = list(range(0, max(len(sequence) - overlap, 1), window_size - overlap))
    windows = []
    for start in starts:
        stop = min(start + window_size, len(sequence))
        windows.append((start, sequence[start:stop]))
        if stop == len(sequence):
            break
    return windows


def adaptive_pool(embedding, maximum_tokens):
    if embedding.shape[0] <= int(maximum_tokens):
        return embedding
    tensor = torch.from_numpy(np.asarray(embedding, dtype=np.float32)).transpose(0, 1)[None]
    pooled = F.adaptive_avg_pool1d(tensor, int(maximum_tokens))[0].transpose(0, 1)
    return pooled.numpy()


def _lineage(args, manifests):
    return {
        "version": VERSION,
        "backend": args.backend,
        "model": args.model,
        "representation_layer": args.representation_layer,
        "maximum_tokens": args.maximum_tokens,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "manifest_sha256": {str(path): sha256_file(path) for path in manifests},
    }


def extract(args):
    manifests = [path.expanduser().resolve() for path in args.manifests]
    sequences = {}
    for manifest in manifests:
        for row in read_jsonl(manifest):
            sequence_id, sequence = str(row["sequence_id"]), str(row["sequence"])
            previous = sequences.setdefault(sequence_id, sequence)
            if previous != sequence:
                raise ValueError("sequence hash collision for %s" % sequence_id)
    output = args.output.expanduser().resolve()
    expected_lineage = _lineage(args, manifests)
    if output.exists() and any(output.iterdir()):
        lineage_path = output / "lineage.json"
        if not args.resume or not lineage_path.is_file():
            raise FileExistsError("non-empty ESM cache requires --resume: %s" % output)
        current = json.loads(lineage_path.read_text(encoding="utf-8"))
        if current != expected_lineage:
            raise ValueError("ESM cache lineage differs; refusing unsafe resume")
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "lineage.json").write_text(
            json.dumps(expected_lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.backend == "transformers":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        batch_converter = None
        layer = args.representation_layer
        embedding_width = int(model.config.hidden_size)
    else:
        import esm

        model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
        batch_converter = alphabet.get_batch_converter()
        layer = int(args.representation_layer or model.num_layers)
        tokenizer = None
        embedding_width = int(model.embed_dim)
    model.eval().to(args.device)
    if args.precision == "fp16":
        model.half()
    elif args.precision == "bf16":
        model.to(dtype=torch.bfloat16)
    environment = lmdb.open(
        str(output), map_size=int(args.map_size_gb * (1 << 30)), subdir=True, max_readers=32
    )
    with environment.begin(write=False) as transaction:
        remaining_sequences = {
            key: value for key, value in sequences.items() if transaction.get(key.encode("ascii")) is None
        }
    print(
        "ESM sequences total=%d cached=%d remaining=%d"
        % (len(sequences), len(sequences) - len(remaining_sequences), len(remaining_sequences)),
        flush=True,
    )
    jobs = []
    remaining = Counter()
    lengths = {}
    for sequence_id, sequence in sorted(remaining_sequences.items(), key=lambda item: len(item[1])):
        windows = sequence_windows(sequence, args.window_size, args.overlap)
        remaining[sequence_id] = len(windows)
        lengths[sequence_id] = len(sequence)
        jobs.extend((sequence_id, start, window) for start, window in windows)
    accumulators = {}
    completed = 0
    autocast_enabled = str(args.device).startswith("cuda") and args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    for offset in range(0, len(jobs), int(args.window_batch_size)):
        group = jobs[offset : offset + int(args.window_batch_size)]
        labels_and_sequences = [
            ("%s:%d" % (sequence_id, start), window) for sequence_id, start, window in group
        ]
        if args.backend == "transformers":
            encoded = tokenizer(
                [window for _, window in labels_and_sequences],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            tokens = encoded["input_ids"].to(args.device)
            attention_mask = encoded["attention_mask"].to(args.device)
        else:
            _, _, tokens = batch_converter(labels_and_sequences)
            tokens = tokens.to(args.device)
            attention_mask = None
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled
        ):
            if args.backend == "transformers":
                model_output = model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    output_hidden_states=layer is not None,
                )
                representations = (
                    model_output.last_hidden_state
                    if layer is None
                    else model_output.hidden_states[int(layer)]
                )
            else:
                representations = model(tokens, repr_layers=[layer], return_contacts=False)[
                    "representations"
                ][layer]
        finished = []
        for batch_index, (sequence_id, start, window) in enumerate(group):
            window_embedding = representations[batch_index, 1 : 1 + len(window)].float().cpu().numpy()
            if sequence_id not in accumulators:
                width = int(window_embedding.shape[1])
                accumulators[sequence_id] = (
                    np.zeros((lengths[sequence_id], width), dtype=np.float32),
                    np.zeros(lengths[sequence_id], dtype=np.float32),
                )
            total, count = accumulators[sequence_id]
            total[start : start + len(window)] += window_embedding
            count[start : start + len(window)] += 1.0
            remaining[sequence_id] -= 1
            if remaining[sequence_id] == 0:
                embedding = total / count[:, None].clip(1.0)
                embedding = adaptive_pool(embedding, args.maximum_tokens).astype(np.float16)
                finished.append(
                    (
                        sequence_id,
                        {
                            "embedding": embedding,
                            "original_length": int(lengths[sequence_id]),
                            "pooled_length": int(embedding.shape[0]),
                            "model": args.model,
                            "layer": "last" if layer is None else int(layer),
                        },
                    )
                )
                del accumulators[sequence_id]
        if finished:
            with environment.begin(write=True) as transaction:
                for sequence_id, record in finished:
                    transaction.put(
                        sequence_id.encode("ascii"),
                        pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    completed += 1
        if completed and completed % int(args.progress_every) < len(finished):
            print("ESM newly cached=%d/%d" % (completed, len(remaining_sequences)), flush=True)
    environment.sync()
    environment.close()
    metadata = dict(expected_lineage)
    metadata.update(
        {
            "count": len(sequences),
            "newly_cached": completed,
            "embedding_width": embedding_width,
            "storage_dtype": "float16",
            "full_sequence_policy": "overlap-stitch all residues, then adaptive mean pool",
            "status": "built",
        }
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/OptionalStructureMultitask_v21/proteins_esm2_t33.lmdb")
    )
    parser.add_argument("--backend", choices=("transformers", "fair_esm"), default="transformers")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--representation-layer", type=int)
    parser.add_argument("--maximum-tokens", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--window-batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--map-size-gb", type=float, default=64.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    extract(build_parser().parse_args())


if __name__ == "__main__":
    main()
