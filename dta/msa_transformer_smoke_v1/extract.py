"""Extract tiny official MSA Transformer residue caches for a real smoke test."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch

from .msa import (
    ENCODER_NAME,
    ESM_SOURCE_REVISION,
    prepare_inputs,
    sha256_file,
    sha256_text,
)


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_paths():
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    return (
        checkpoint_dir / (ENCODER_NAME + ".pt"),
        checkpoint_dir / (ENCODER_NAME + "-contact-regression.pt"),
    )


def load_official_model(args):
    """Load the fixed official implementation and both required checkpoint files."""

    import esm

    if args.model_file is not None and args.allow_download:
        raise ValueError("--model-file and --allow-download are mutually exclusive")
    if args.model_file is not None:
        model_path = Path(args.model_file).expanduser().resolve()
        if model_path.stem != ENCODER_NAME:
            raise ValueError("local model filename must identify %s" % ENCODER_NAME)
        regression_path = model_path.with_name(
            model_path.stem + "-contact-regression" + model_path.suffix
        )
        if not model_path.is_file() or not regression_path.is_file():
            raise FileNotFoundError(
                "local model and contact-regression checkpoints must both exist"
            )
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
    else:
        model_path, regression_path = _checkpoint_paths()
        if model_path.is_file() and regression_path.is_file():
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
        elif args.allow_download:
            model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        else:
            raise FileNotFoundError(
                "official MSA Transformer checkpoints are not cached; use --allow-download "
                "only for the approved first download"
            )
    if not model_path.is_file() or not regression_path.is_file():
        raise FileNotFoundError("official loader did not leave both required checkpoint files")
    metadata = {
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "contact_regression_path": str(regression_path),
        "contact_regression_size_bytes": regression_path.stat().st_size,
        "contact_regression_sha256": sha256_file(regression_path),
        "esm_version": getattr(esm, "__version__", None),
        "esm_module_path": str(Path(esm.__file__).resolve()),
        "esm_source_revision": ESM_SOURCE_REVISION,
    }
    return model, alphabet, metadata


def validate_model(model, alphabet):
    if int(model.num_layers) != 12:
        raise ValueError("unexpected MSA Transformer layer count: %s" % model.num_layers)
    if int(model.args.embed_dim) != 768:
        raise ValueError("unexpected MSA Transformer embedding width: %s" % model.args.embed_dim)
    if not bool(alphabet.prepend_bos) or bool(alphabet.append_eos):
        raise ValueError("unexpected MSA alphabet boundary-token configuration")
    if not bool(getattr(alphabet, "use_msa", False)):
        raise ValueError("loaded alphabet is not an MSA alphabet")


def _device_metadata(device):
    metadata = {"requested": str(device), "type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "logical_index": int(index),
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return metadata


def encode_prepared_msa(model, alphabet, prepared, device, maximum_tensor_tokens):
    """Run one validated MSA and return only row-0 real-residue representations."""

    converter = alphabet.get_batch_converter()
    _labels, _strings, tokens = converter([prepared.batch_converter_input()])
    expected_width = (
        prepared.width
        + int(bool(alphabet.prepend_bos))
        + int(bool(alphabet.append_eos))
    )
    expected_shape = (1, prepared.depth, expected_width)
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(
            "batch converter produced shape %s, expected %s"
            % (tuple(tokens.shape), expected_shape)
        )
    if int(tokens.numel()) > int(maximum_tensor_tokens):
        raise ValueError("actual tokenized MSA exceeds maximum_tensor_tokens before forward")
    maximum_positions = int(model.args.max_positions)
    if int(tokens.shape[-1]) > maximum_positions:
        raise ValueError(
            "tokenized row length %d exceeds model max_positions=%d"
            % (tokens.shape[-1], maximum_positions)
        )
    layer = int(model.num_layers)
    if layer != 12:
        raise ValueError("expected final layer 12, observed %d" % layer)
    residue_positions = torch.as_tensor(
        [
            int(bool(alphabet.prepend_bos)) + column
            for column in prepared.query_residue_columns
        ],
        dtype=torch.long,
        device=device,
    )
    tokens = tokens.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            tokens,
            repr_layers=[layer],
            need_head_weights=False,
            return_contacts=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if "representations" not in outputs or layer not in outputs["representations"]:
        raise ValueError("official model output is missing the requested final representation")
    representation = outputs["representations"][layer]
    if representation.ndim != 4 or tuple(representation.shape[:3]) != tuple(tokens.shape):
        raise ValueError("unexpected MSA representation layout: %s" % (tuple(representation.shape),))
    query_embedding = representation[0, 0].index_select(0, residue_positions)
    if query_embedding.shape[0] != len(prepared.query_sequence):
        raise ValueError("query representation does not map one-to-one to target residues")
    if not torch.isfinite(query_embedding).all():
        raise ValueError("MSA Transformer produced non-finite residue features")
    embedding = query_embedding.detach().to(dtype=torch.float32, device="cpu").numpy()
    run_metadata = {
        "layer": layer,
        "D": int(tokens.shape[1]),
        "K": int(prepared.width),
        "T": int(tokens.shape[2]),
        "tensor_tokens": int(tokens.numel()),
        "embedding_shape": list(embedding.shape),
        "embedding_dtype": str(embedding.dtype),
        "forward_seconds": float(elapsed),
        "max_positions": maximum_positions,
    }
    if device.type == "cuda":
        run_metadata.update(
            {
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return embedding, run_metadata


def _preflight_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("output directory is non-empty; refusing overwrite: %s" % path)
    return path


def extract(args, model_loader=None):
    """Validate all data first, then load the real model and create one new cache."""

    prepared_inputs = prepare_inputs(
        args.manifests,
        args.msa_manifest,
        args.mode,
        maximum_targets=args.maximum_targets,
        maximum_depth=args.maximum_depth,
        maximum_columns=args.maximum_columns,
        maximum_tensor_tokens=args.maximum_tensor_tokens,
    )
    output = _preflight_output(args.output)
    if float(args.map_size_gb) > 4.0:
        raise ValueError("map_size_gb exceeds the 4 GiB smoke maximum")
    loader = load_official_model if model_loader is None else model_loader
    load_started = time.perf_counter()
    model, alphabet, weight_metadata = loader(args)
    model_load_seconds = time.perf_counter() - load_started
    validate_model(model, alphabet)
    model.eval()
    model.requires_grad_(False)
    model.float()
    device = torch.device(args.device)
    model.to(device)

    output.mkdir(parents=True, exist_ok=True)
    lmdb_path = output / "proteins.lmdb"
    environment = lmdb.open(
        str(lmdb_path),
        subdir=True,
        map_size=int(float(args.map_size_gb) * (1 << 30)),
        lock=True,
        meminit=False,
    )
    records = []
    try:
        with environment.begin(write=True) as transaction:
            for sequence_id, prepared in sorted(prepared_inputs["prepared"].items()):
                embedding, runtime = encode_prepared_msa(
                    model,
                    alphabet,
                    prepared,
                    device,
                    args.maximum_tensor_tokens,
                )
                mapping = prepared_inputs["mappings"][sequence_id]
                record_metadata = {
                    **dict(prepared.metadata),
                    **runtime,
                    "sequence_id": sequence_id,
                    "query_sha256": sha256_text(prepared.query_sequence),
                    "provenance": mapping["provenance"],
                }
                cache_record = {
                    "embedding": embedding,
                    "encoder": ENCODER_NAME,
                    "feature_mode": args.mode,
                    "original_length": len(prepared.query_sequence),
                    "residue_mapping": [
                        {"output_index": index, "start": index, "stop": index + 1}
                        for index in range(len(prepared.query_sequence))
                    ],
                    "sequence_sha256": sha256_text(prepared.query_sequence),
                    "metadata": record_metadata,
                }
                inserted = transaction.put(
                    sequence_id.encode("utf-8"),
                    pickle.dumps(cache_record, protocol=pickle.HIGHEST_PROTOCOL),
                    overwrite=False,
                )
                if not inserted:
                    raise ValueError("duplicate cache key: %s" % sequence_id)
                records.append(record_metadata)
        environment.sync()
    finally:
        environment.close()

    lineage = {
        "status": "built",
        "encoder": ENCODER_NAME,
        "feature_mode": args.mode,
        "compute_precision": "fp32",
        "storage_dtype": "float32",
        "esm_source_revision": ESM_SOURCE_REVISION,
        "weights": weight_metadata,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_module_path": str(Path(torch.__file__).resolve()),
        "numpy_version": np.__version__,
        "device": _device_metadata(device),
        "model_load_seconds": float(model_load_seconds),
        "encoder_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "model_eval": not model.training,
        "source_manifests": prepared_inputs["source_manifests"],
        "msa_manifest": prepared_inputs["msa_manifest"],
        "target_count": len(prepared_inputs["targets"]),
        "maximum_depth": int(args.maximum_depth),
        "maximum_columns": int(args.maximum_columns),
        "maximum_tensor_tokens": int(args.maximum_tensor_tokens),
        "records": records,
        "protein_cache": str(lmdb_path),
    }
    _atomic_write_json(output / "metadata.json", lineage)
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(lineage, sort_keys=True), flush=True)
    return lineage


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--msa-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("query_only", "aligned_msa"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--maximum-targets", type=_positive_int, default=3)
    parser.add_argument("--maximum-depth", type=_positive_int, default=8)
    parser.add_argument("--maximum-columns", type=_positive_int, default=512)
    parser.add_argument("--maximum-tensor-tokens", type=_positive_int, default=8192)
    parser.add_argument("--map-size-gb", type=_positive_float, default=1.0)
    return parser


def main():
    extract(build_parser().parse_args())


if __name__ == "__main__":
    main()
