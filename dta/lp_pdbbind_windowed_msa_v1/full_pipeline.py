"""Prepare and extract train/valid full-chain features from the indexed MMseqs DB.

This module never opens the test manifest.  ``prepare`` is CPU-only; ``extract``
is run separately for each mode on an operator-selected GPU; ``audit`` checks
the completed caches against every unchanged train/valid sample.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import torch

from dta.msa_pocket_expert_v9.mmseqs_profile_db import _read_index
from dta.msa_transformer_smoke_v1.extract import load_official_model, validate_model
from dta.msa_transformer_smoke_v1.msa import ENCODER_NAME, sha256_file, sha256_text

from . import WINDOW_POLICY
from .cache import (
    SOURCE_MANIFEST_VERSION,
    FullChainMSAFeatureDataset,
    load_source_manifest,
    make_feature_record,
    validate_feature_record,
)
from .windowed import FullAlignment, encode_windowed_alignment, prepare_full_alignment
from .selector import (DEFAULT_CANDIDATE_POOL_SIZE, DIVERSE_SELECTION_POLICY,
                       LEGACY_SELECTION_POLICY)


MODES = ("query_only", "aligned_msa")
MAXIMUM_TENSOR_TOKENS = 8192


def _write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _input_paths(workspace):
    root = Path(workspace).expanduser().resolve()
    msa = root / "data/LP_PDBBind_MSA_v9"
    return {
        "sequence_manifest": msa / "sequences/sequence_manifest.json",
        "a3m_db": msa / "uniref30_full/work/uniref.a3m",
        "a3m_index": msa / "uniref30_full/work/uniref.a3m.index",
        "manifest_root": root / "data/PDBbind_chain_esm_residue_DTA/lp_pdbbind",
        "complex_root": msa / "complex_cache_full",
        "row_cache": root / "data/LP_PDBBind_MSA_v11/uniref30_rows_128",
    }


def _load_inventory(paths):
    sequence_path = paths["sequence_manifest"]
    sequences = _read_json(sequence_path)["sequences"]
    entries = _read_index(paths["a3m_index"])
    if len(entries) != len(sequences) or {x[0] for x in entries} != set(range(len(sequences))):
        raise ValueError("indexed A3M database does not cover the sequence inventory")
    indexed = {key: (offset, size) for key, offset, size in entries}
    size = paths["a3m_db"].stat().st_size
    for offset, length in indexed.values():
        if offset < 0 or length <= 0 or offset + length > size:
            raise ValueError("indexed A3M record is outside the database")
    return sequences, indexed


def _read_a3m(source, offset, length, expected_sequence_id):
    source.seek(int(offset))
    raw = source.read(int(length))
    if len(raw) != int(length):
        raise IOError("short indexed A3M read")
    raw = raw.rstrip(b"\x00")
    first = raw.split(b"\n", 1)[0]
    header = first[1:].split(None, 1) if first.startswith(b">") else []
    if not header:
        raise ValueError("indexed A3M query has an empty or missing identifier")
    identifier = header[0].decode("utf-8")
    if identifier != expected_sequence_id:
        raise ValueError("indexed A3M header differs from the sequence inventory")
    return raw


def _prepare_from_raw(raw, query, mode, temporary_path, **selection_kwargs):
    # The bytes hashed by prepare_full_alignment are exactly the bytes read
    # from the source database (after its MMseqs NUL record terminator).
    with Path(temporary_path).open("wb") as handle:
        handle.write(raw)
    return prepare_full_alignment(temporary_path, query, mode, **selection_kwargs)


def _query_alignment_from_aligned(aligned):
    metadata = dict(aligned.metadata)
    metadata["mode"] = "query_only"
    metadata["selected_source_identifiers"] = metadata["selected_source_identifiers"][:1]
    for field in ("selected_ids", "selected_source_indices", "selected_query_identity"):
        if field in metadata:
            metadata[field] = metadata[field][:1]
    if "selected_depth" in metadata:
        metadata["selected_depth"] = 1
    return FullAlignment(
        rows=aligned.rows[:1],
        query_residue_columns=aligned.query_residue_columns,
        query_sequence=aligned.query_sequence,
        mode="query_only",
        metadata=metadata,
    )


def _split_inventory(paths, sequences):
    manifests = {
        split: _read_json(paths["manifest_root"] / (split + "_manifest.json"))
        for split in ("train", "valid")
    }
    keys = {}
    for split, manifest in manifests.items():
        if manifest.get("split") != split:
            raise ValueError("train/valid manifest split mismatch")
        samples = manifest["samples"]
        sample_keys = [str(sample["lmdb_key"]) for sample in samples]
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError("duplicate complex key in %s manifest" % split)
        keys[split] = set(sample_keys)
    declared = {split: set() for split in keys}
    mapped = {split: set() for split in keys}
    index_by_sid = {}
    for index, sequence in enumerate(sequences):
        sid = str(sequence["sequence_id"])
        query = str(sequence["sequence"])
        if sid != "seq_" + sha256_text(query) or int(sequence["length"]) != len(query):
            raise ValueError("sequence inventory identity mismatch")
        if sid in index_by_sid:
            raise ValueError("duplicate sequence ID in inventory")
        index_by_sid[sid] = index
        for occurrence in sequence["occurrences"]:
            split = occurrence["split"]
            if split in keys and occurrence["lmdb_key"] in keys[split]:
                declared[split].add(index)
                mapped[split].add(occurrence["lmdb_key"])
    for split in keys:
        if mapped[split] != keys[split]:
            raise ValueError("sequence inventory does not map every %s complex" % split)
    used = {split: set() for split in keys}
    for split, manifest in manifests.items():
        environment = lmdb.open(
            str(paths["complex_root"] / split / "complexes.lmdb"),
            readonly=True, lock=False, readahead=False, meminit=False,
        )
        try:
            with environment.begin(write=False) as transaction:
                for sample in manifest["samples"]:
                    key = str(sample["lmdb_key"])
                    raw = transaction.get(key.encode("utf-8"))
                    if raw is None:
                        raise KeyError("missing complex cache key: %s" % key)
                    item = pickle.loads(bytes(raw))
                    if str(item["key"]) != key or abs(float(item["label"]) - float(sample["label"])) > 1e-5:
                        raise ValueError("complex cache identity differs from manifest")
                    profile_ids = tuple(str(value) for value in item["profile_ids"])
                    chain = np.asarray(item["pocket_chain_index"], dtype=np.int64)
                    if chain.ndim != 1 or not chain.size or np.any(chain < 0) or np.any(chain >= len(profile_ids)):
                        raise ValueError("invalid pocket chain mapping")
                    for chain_index in np.unique(chain):
                        sid = profile_ids[int(chain_index)]
                        if sid not in index_by_sid or index_by_sid[sid] not in declared[split]:
                            raise ValueError("used pocket chain is absent from the declared split inventory")
                        used[split].add(index_by_sid[sid])
        finally:
            environment.close()
    return manifests, used


def prepare(workspace, output, *, selection_policy=LEGACY_SELECTION_POLICY,
            candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
            baseline_cache_root=None, code_commit=None):
    """Freeze independent source identities before GPU extraction."""

    if baseline_cache_root is not None:
        from .selector_pipeline import prepare_selector_sources
        return prepare_selector_sources(
            workspace, output, baseline_cache_root=baseline_cache_root,
            selection_policy=selection_policy, candidate_pool_size=candidate_pool_size,
            code_commit=code_commit)
    if selection_policy != LEGACY_SELECTION_POLICY:
        raise ValueError("new selector requires an explicit historical --baseline-cache-root seal")
    paths = _input_paths(workspace)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("full feature output already exists")
    sequences, indexed = _load_inventory(paths)
    manifests, used = _split_inventory(paths, sequences)
    selected = sorted(used["train"] | used["valid"])
    sources = {}
    targets = []
    with tempfile.TemporaryDirectory(prefix="windowed-msa-source-") as temp_dir:
        temporary_path = Path(temp_dir) / "record.a3m"
        with paths["a3m_db"].open("rb", buffering=8 << 20) as db:
            for count, index in enumerate(selected, 1):
                sequence = sequences[index]
                sid = str(sequence["sequence_id"])
                offset, length = indexed[index]
                raw = _read_a3m(db, offset, length, sid)
                aligned = _prepare_from_raw(raw, sequence["sequence"], "aligned_msa", temporary_path)
                query_only = _query_alignment_from_aligned(aligned)
                sources[sid] = {
                    "query_sha256": aligned.metadata["query_sha256"],
                    "a3m_sha256": aligned.metadata["a3m_sha256"],
                    "selected_source_identifiers": {
                        "query_only": query_only.metadata["selected_source_identifiers"],
                        "aligned_msa": aligned.metadata["selected_source_identifiers"],
                    },
                }
                targets.append({
                    "sequence_id": sid,
                    "sequence_index": index,
                    "query_sequence": sequence["sequence"],
                    "a3m_offset": offset,
                    "a3m_size": length,
                })
                if count % 100 == 0:
                    print(json.dumps({"event": "source_preflight_progress", "processed": count, "total": len(selected)}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    source_path = output / "source_manifest.json"
    _write_json_new(source_path, {"version": SOURCE_MANIFEST_VERSION, "sequences": sources})
    _write_json_new(output / "targets.json", targets)
    report = {
        "status": "source_preflight_passed",
        "train_sample_count": len(manifests["train"]["samples"]),
        "valid_sample_count": len(manifests["valid"]["samples"]),
        "train_sequence_count": len(used["train"]),
        "valid_sequence_count": len(used["valid"]),
        "unique_sequence_count": len(selected),
        "source_manifest_sha256": sha256_file(source_path),
        "targets_sha256": sha256_file(output / "targets.json"),
        "sequence_manifest_sha256": sha256_file(paths["sequence_manifest"]),
        "a3m_index_sha256": sha256_file(paths["a3m_index"]),
        "a3m_db_sha256": sha256_file(paths["a3m_db"]),
        "a3m_db_size": paths["a3m_db"].stat().st_size,
        "train_manifest_sha256": sha256_file(paths["manifest_root"] / "train_manifest.json"),
        "valid_manifest_sha256": sha256_file(paths["manifest_root"] / "valid_manifest.json"),
        "test_manifest_read": False,
        "training_run": False,
    }
    _write_json_new(output / "preflight.json", report)
    return report


def _sealed_preflight(paths, output):
    report = _read_json(output / "preflight.json")
    if report.get("status") != "source_preflight_passed":
        raise ValueError("source preflight is incomplete")
    selector_seal = report.get("seal_version") is not None
    if selector_seal:
        from .selector_pipeline import validate_selector_seal
        validate_selector_seal(paths, output, report)
    checks = (
        (output / "source_manifest.json", "source_manifest_sha256"),
        (output / "targets.json", "targets_sha256"),
        (paths["sequence_manifest"], "sequence_manifest_sha256"),
        (paths["a3m_index"], "a3m_index_sha256"),
        (paths["manifest_root"] / "train_manifest.json", "train_manifest_sha256"),
        (paths["manifest_root"] / "valid_manifest.json", "valid_manifest_sha256"),
    )
    if not selector_seal:
        checks += ((paths["a3m_db"], "a3m_db_sha256"),)
    if any(sha256_file(path) != report[field] for path, field in checks):
        raise ValueError("full feature preflight input changed")
    if paths["a3m_db"].stat().st_size != report["a3m_db_size"]:
        raise ValueError("indexed A3M database size changed")
    targets = _read_json(output / "targets.json")
    sources = load_source_manifest(output / "source_manifest.json")
    if len(targets) != report["unique_sequence_count"] or {x["sequence_id"] for x in targets} != set(sources):
        raise ValueError("source manifest and target list differ")
    if selector_seal and any(
        entry.get("selector_policy") != report["selection_policy"]
        or entry.get("selector_config_sha256") != report["selector_config_sha256"]
        for entry in sources.values()
    ):
        raise ValueError("source entries differ from sealed selector identity")
    return report, targets, sources


def extract(workspace, output, model_file, mode, device, resume=False, map_size_gb=64):
    """Incrementally extract one arm; a partial LMDB can be explicitly resumed."""

    if mode not in MODES:
        raise ValueError("unknown feature mode")
    paths = _input_paths(workspace)
    output = Path(output).expanduser().resolve()
    report, targets, sources = _sealed_preflight(paths, output)
    selector_config = (_read_json(output / "selector_config.json")
                       if report.get("seal_version") else None)
    selector_kwargs = ({"selection_policy": selector_config["selection_policy"],
                        "candidate_pool_size": selector_config["candidate_pool_size"]}
                       if selector_config else {})
    mode_dir = output / mode
    if (mode_dir / "metadata.json").exists():
        raise FileExistsError("completed mode cache already exists")
    if mode_dir.exists() and not resume:
        raise FileExistsError("partial mode cache exists; pass --resume after inspection")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("full extraction requires an operator-selected CUDA device")
    if int(map_size_gb) <= 0:
        raise ValueError("LMDB map size must be positive")
    weight_hash = sha256_file(model_file)
    if selector_config and weight_hash != selector_config["encoder_weights_sha256"]:
        raise ValueError("selector extraction encoder checkpoint differs from historical baseline")
    config = {
        "mode": mode,
        "weights_sha256": weight_hash,
        "source_manifest_sha256": report["source_manifest_sha256"],
        "targets_sha256": report["targets_sha256"],
        "window_policy": WINDOW_POLICY,
        "encoder": ENCODER_NAME,
        "maximum_tensor_tokens": MAXIMUM_TENSOR_TOKENS,
        "map_size_gb": int(map_size_gb),
    }
    if selector_config:
        config.update(selector_policy=report["selection_policy"],
                      selector_config_sha256=report["selector_config_sha256"],
                      code_commit=report["code_commit"])
    if mode_dir.exists():
        if _read_json(mode_dir / "run_config.json") != config:
            raise ValueError("partial mode cache belongs to another source or weight")
    else:
        mode_dir.mkdir(exist_ok=False)
        _write_json_new(mode_dir / "run_config.json", config)
    model, alphabet, weight_metadata = load_official_model(SimpleNamespace(
        model_file=Path(model_file), allow_download=False,
    ))
    validate_model(model, alphabet)
    if weight_metadata["model_sha256"] != weight_hash:
        raise ValueError("loaded encoder weights differ from the specified file")
    model.eval().requires_grad_(False)
    model.float().to(device)
    environment = lmdb.open(
        str(mode_dir / "proteins.lmdb"), subdir=True,
        map_size=int(map_size_gb) << 30, lock=True, meminit=False,
    )
    newly_written = 0
    reused = 0
    try:
        with tempfile.TemporaryDirectory(prefix="windowed-msa-extract-") as temp_dir:
            temporary_path = Path(temp_dir) / "record.a3m"
            with paths["a3m_db"].open("rb", buffering=8 << 20) as db:
                for count, target in enumerate(targets, 1):
                    sid = target["sequence_id"]
                    with environment.begin(write=False) as transaction:
                        previous = transaction.get(sid.encode("ascii"))
                    alignment = None
                    if selector_config:
                        # An existing record cannot substitute for rereading its
                        # source when the whole-database digest is not recomputed.
                        raw = _read_a3m(db, target["a3m_offset"], target["a3m_size"], sid)
                        alignment = _prepare_from_raw(raw, target["query_sequence"], mode,
                                                      temporary_path, **selector_kwargs)
                        alignment.metadata.update(selector_policy=report["selection_policy"],
                                                  selector_config_sha256=report["selector_config_sha256"])
                        expected = sources[sid]
                        if (alignment.metadata["query_sha256"] != expected["query_sha256"]
                                or alignment.metadata["a3m_sha256"] != expected["a3m_sha256"]
                                or alignment.metadata["selected_source_identifiers"] != expected["selected_source_identifiers"][mode]):
                            raise ValueError("source A3M or sealed homolog selection changed")
                    if previous is not None:
                        validate_feature_record(
                            pickle.loads(bytes(previous)), sid, target["query_sequence"],
                            mode, weight_hash, sources[sid],
                        )
                        reused += 1
                        continue
                    if alignment is None:
                        raw = _read_a3m(db, target["a3m_offset"], target["a3m_size"], sid)
                        alignment = _prepare_from_raw(raw, target["query_sequence"], mode, temporary_path)
                    expected = sources[sid]
                    if (
                        alignment.metadata["query_sha256"] != expected["query_sha256"]
                        or alignment.metadata["a3m_sha256"] != expected["a3m_sha256"]
                        or alignment.metadata["selected_source_identifiers"]
                        != expected["selected_source_identifiers"][mode]
                    ):
                        raise ValueError("source A3M or homolog selection changed")
                    values, _runtime = encode_windowed_alignment(
                        model, alphabet, alignment, device,
                        maximum_tensor_tokens=MAXIMUM_TENSOR_TOKENS,
                    )
                    record = make_feature_record(sid, alignment, values, weight_hash)
                    validate_feature_record(record, sid, target["query_sequence"], mode, weight_hash, expected)
                    with environment.begin(write=True) as transaction:
                        if not transaction.put(sid.encode("ascii"), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL), overwrite=False):
                            raise ValueError("duplicate feature key during extraction")
                    newly_written += 1
                    if count % 100 == 0:
                        print(json.dumps({"event": "full_extraction_progress", "mode": mode, "processed": count, "total": len(targets)}), flush=True)
        environment.sync()
        if environment.stat()["entries"] != len(targets):
            raise ValueError("feature cache key count is incomplete")
    finally:
        environment.close()
    metadata = {
        "status": "built",
        "encoder": ENCODER_NAME,
        "feature_mode": mode,
        "weights_sha256": weight_hash,
        "window_policy": WINDOW_POLICY,
        "source_manifest_sha256": report["source_manifest_sha256"],
        "target_count": len(targets),
        "newly_written": newly_written,
        "reused": reused,
        "checkpoint": weight_metadata,
        "test_manifest_read": False,
        "training_run": False,
    }
    if selector_config:
        metadata.update(selector_policy=report["selection_policy"],
                        selector_config_sha256=report["selector_config_sha256"],
                        code_commit=report["code_commit"])
    _write_json_new(mode_dir / "metadata.json", metadata)
    return metadata


def audit(workspace, output):
    """Force both caches through every unchanged train/valid dataset item."""

    paths = _input_paths(workspace)
    output = Path(output).expanduser().resolve()
    report, targets, _sources = _sealed_preflight(paths, output)
    if (output / "coverage_audit.json").exists():
        raise FileExistsError("coverage audit already exists")
    modes = {}
    for mode in MODES:
        cache_dir = output / mode
        metadata = _read_json(cache_dir / "metadata.json")
        if (
            metadata.get("status") != "built"
            or metadata.get("target_count") != len(targets)
            or metadata.get("source_manifest_sha256") != report["source_manifest_sha256"]
        ):
            raise ValueError("mode cache is incomplete or from another source")
        if report.get("seal_version") and (
                metadata.get("selector_policy") != report["selection_policy"]
                or metadata.get("selector_config_sha256") != report["selector_config_sha256"]):
            raise ValueError("mode cache belongs to another selector configuration")
        split_counts = {}
        for split in ("train", "valid"):
            dataset = FullChainMSAFeatureDataset(
                paths["manifest_root"] / (split + "_manifest.json"),
                paths["complex_root"] / split,
                paths["row_cache"], cache_dir, mode,
                metadata["weights_sha256"], output / "source_manifest.json",
            )
            if len(dataset) != report[split + "_sample_count"]:
                raise ValueError("dataset sample count changed")
            for index in range(len(dataset)):
                item = dataset[index]
                values = np.asarray(item["fullchain_msa_pocket"])
                if values.ndim != 2 or values.shape[1] != 768 or not np.isfinite(values).all():
                    raise ValueError("invalid full-chain feature in %s %s" % (split, item["key"]))
            split_counts[split] = len(dataset)
        modes[mode] = {"weights_sha256": metadata["weights_sha256"], "samples": split_counts}
    if modes[MODES[0]]["weights_sha256"] != modes[MODES[1]]["weights_sha256"]:
        raise ValueError("A/B encoder weights differ")
    result = {
        "status": "full_cache_coverage_passed",
        "source_manifest_sha256": report["source_manifest_sha256"],
        "modes": modes,
        "test_manifest_read": False,
        "training_run": False,
    }
    if report.get("seal_version"):
        result.update(selector_policy=report["selection_policy"],
                      selector_config_sha256=report["selector_config_sha256"])
    _write_json_new(output / "coverage_audit.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "extract", "audit"))
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--map-size-gb", type=int, default=64)
    parser.add_argument("--selection-policy", choices=(LEGACY_SELECTION_POLICY, DIVERSE_SELECTION_POLICY),
                        default=LEGACY_SELECTION_POLICY)
    parser.add_argument("--candidate-pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    parser.add_argument("--baseline-cache-root", type=Path)
    parser.add_argument("--code-commit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.phase == "prepare":
        result = prepare(args.workspace, args.output, selection_policy=args.selection_policy,
                         candidate_pool_size=args.candidate_pool_size,
                         baseline_cache_root=args.baseline_cache_root, code_commit=args.code_commit)
    elif args.phase == "extract":
        if args.model_file is None or args.mode is None:
            raise ValueError("extract requires --model-file and --mode")
        result = extract(args.workspace, args.output, args.model_file, args.mode, args.device, args.resume, args.map_size_gb)
    else:
        result = audit(args.workspace, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
