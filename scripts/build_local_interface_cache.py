#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import lmdb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.local_interface import (  # noqa: E402
    LOCAL_INTERFACE_SCHEMA_VERSION,
    LocalInterfaceConfig,
    extract_local_interface_record,
    resolve_source_lmdb,
)
from utils.experiment_integrity import (  # noqa: E402
    atomic_write_json,
    hash_ordered_keys,
    manifest_fingerprint,
)


BUILDER_NAME = "local_interface_native_cache"


def _load_manifest(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    keys = [str(sample["lmdb_key"]) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate lmdb keys")
    return path, manifest, samples, keys


def _open_readonly_lmdb(path):
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )


def _open_output_lmdb(path, map_size_gb):
    map_size = int(float(map_size_gb) * 1024 ** 3)
    if map_size <= 0:
        raise ValueError("map_size_gb must be positive")
    return lmdb.open(
        str(path),
        map_size=map_size,
        subdir=True,
        meminit=False,
    )


def _lmdb_keys(environment):
    with environment.begin(write=False) as transaction:
        return sorted(
            key.decode("utf-8")
            for key in transaction.cursor().iternext(
                keys=True, values=False
            )
            if key != b"__metadata__"
        )


def _load_item(transaction, key, source_name):
    raw = transaction.get(key.encode("utf-8"))
    if raw is None:
        raise KeyError(f"{source_name} LMDB key not found: {key}")
    return pickle.loads(raw)


def _hash_ordered_lmdb_records(environment, ordered_keys):
    digest = hashlib.sha256()
    with environment.begin(write=False) as transaction:
        for key in ordered_keys:
            key_bytes = str(key).encode("utf-8")
            raw = transaction.get(key_bytes)
            if raw is None:
                raise KeyError(f"source LMDB key not found: {key}")
            for value in (key_bytes, raw):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
    return digest.hexdigest()


def _write_record(environment, key, value):
    with environment.begin(write=True) as transaction:
        transaction.put(str(key).encode("utf-8"), value)


def _provenance(
    manifest_path,
    source_path,
    source_keys,
    candidate_keys,
    candidate_record_sha256,
    max_samples,
    config,
):
    return {
        "builder": BUILDER_NAME,
        "schema_version": LOCAL_INTERFACE_SCHEMA_VERSION,
        "manifest": manifest_fingerprint(manifest_path),
        "source_lmdb": str(source_path),
        "source_key_count": len(source_keys),
        "source_key_sha256": hash_ordered_keys(source_keys),
        "max_samples": max_samples,
        "candidate_count": len(candidate_keys),
        "candidate_key_sha256": hash_ordered_keys(candidate_keys),
        "candidate_record_sha256": candidate_record_sha256,
        "config": {
            "core_cutoff": float(config.core_cutoff),
            "outer_cutoff": float(config.outer_cutoff),
            "shell_budget": int(config.shell_budget),
        },
    }


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _prepare_output(output_path, resume, provenance):
    metadata_path = output_path / "metadata.json"
    if resume:
        if not metadata_path.exists():
            raise ValueError(
                "resume provenance is unavailable: metadata.json is missing"
            )
        previous = _read_json(metadata_path)
        if previous.get("provenance") != provenance:
            raise ValueError(
                "resume provenance does not match manifest, source keys, "
                "schema, or cache configuration"
            )
        return previous
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"output LMDB is not empty: {output_path}; use --resume"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return None


def _coverage_report(environment):
    coverage_values = []
    shell_retention = []
    core_counts = []
    shell_counts = []
    protein_counts = []
    budget_exceeded = 0
    with environment.begin(write=False) as transaction:
        for key, raw in transaction.cursor():
            if key == b"__metadata__":
                continue
            record = pickle.loads(raw)
            coverage_values.append(
                float(record["ligand_neighbor_coverage"])
            )
            shell_retention.append(
                float(record["protein_shell_retention_fraction"])
            )
            core_counts.append(int(record["protein_core_atom_count"]))
            shell_counts.append(int(record["protein_shell_atom_count"]))
            protein_counts.append(int(record["protein_atom_count"]))
            budget_exceeded += int(
                bool(record["protein_shell_budget_exceeded"])
            )

    sample_count = len(coverage_values)

    def mean(values):
        return float(sum(values) / len(values)) if values else None

    return {
        "sample_count": sample_count,
        "full_ligand_neighbor_coverage_count": sum(
            value == 1.0 for value in coverage_values
        ),
        "mean_ligand_neighbor_coverage": mean(coverage_values),
        "mean_shell_retention_fraction": mean(shell_retention),
        "total_core_atoms": sum(core_counts),
        "total_shell_atoms": sum(shell_counts),
        "max_protein_atom_count": max(protein_counts, default=0),
        "shell_budget_exceeded_count": budget_exceeded,
    }


def _base_metadata(
    args,
    provenance,
    output_path,
    sample_count,
    candidate_count,
):
    return {
        "builder": BUILDER_NAME,
        "schema_version": LOCAL_INTERFACE_SCHEMA_VERSION,
        "output_lmdb": str(output_path),
        "provenance": provenance,
        "manifest": provenance["manifest"],
        "source_lmdb": provenance["source_lmdb"],
        "source_key_count": provenance["source_key_count"],
        "source_key_sha256": provenance["source_key_sha256"],
        "counts": {
            "manifest_samples": sample_count,
            "candidate_samples": candidate_count,
            "processed": 0,
            "written": 0,
            "errors": 0,
            "cached_total": 0,
        },
        "invocation_counts": {
            "processed": 0,
            "written": 0,
            "skipped_existing": 0,
            "errors": 0,
        },
        "failed_samples": {},
        "errors_preview": [],
        "coverage_report": {
            "sample_count": 0,
            "full_ligand_neighbor_coverage_count": 0,
        },
        "status": "in_progress",
        "max_samples": args.max_samples,
    }


def build_native_cache(args):
    manifest_path, manifest, samples, ordered_keys = _load_manifest(
        args.manifest
    )
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if args.progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if args.max_error_preview < 0:
        raise ValueError("max_error_preview must be non-negative")
    config = LocalInterfaceConfig(
        core_cutoff=args.core_cutoff,
        outer_cutoff=args.outer_cutoff,
        shell_budget=args.shell_budget,
    )
    source_path = Path(
        resolve_source_lmdb(manifest, manifest_path.parent)
    )
    source_environment = _open_readonly_lmdb(source_path)
    try:
        candidate_samples = (
            samples[: args.max_samples]
            if args.max_samples is not None
            else samples
        )
        candidate_keys = [
            str(sample["lmdb_key"]) for sample in candidate_samples
        ]
        source_keys = _lmdb_keys(source_environment)
        candidate_record_sha256 = _hash_ordered_lmdb_records(
            source_environment, candidate_keys
        )
        provenance = _provenance(
            manifest_path,
            source_path,
            source_keys,
            candidate_keys,
            candidate_record_sha256,
            args.max_samples,
            config,
        )
        output_path = Path(args.output_lmdb).resolve()
        previous = _prepare_output(
            output_path, bool(args.resume), provenance
        )
        metadata = _base_metadata(
            args,
            provenance,
            output_path,
            len(samples),
            len(candidate_samples),
        )
        if previous is not None:
            metadata["failed_samples"] = dict(
                previous.get("failed_samples", {})
            )
        atomic_write_json(output_path / "metadata.json", metadata)

        output_environment = _open_output_lmdb(
            output_path, args.map_size_gb
        )
        failed_samples = metadata["failed_samples"]
        try:
            with source_environment.begin(write=False) as source_transaction:
                for index, sample in enumerate(candidate_samples, start=1):
                    key = str(sample["lmdb_key"])
                    metadata["invocation_counts"]["processed"] += 1
                    with output_environment.begin(
                        write=False
                    ) as output_transaction:
                        exists = output_transaction.get(
                            key.encode("utf-8")
                        ) is not None
                    if exists:
                        metadata["invocation_counts"][
                            "skipped_existing"
                        ] += 1
                        failed_samples.pop(key, None)
                        continue
                    try:
                        payload = _load_item(
                            source_transaction, key, "source"
                        )
                        record = extract_local_interface_record(
                            payload,
                            key,
                            manifest["split"],
                            config,
                        )
                        serialized = pickle.dumps(
                            record, protocol=pickle.HIGHEST_PROTOCOL
                        )
                        _write_record(output_environment, key, serialized)
                        metadata["invocation_counts"]["written"] += 1
                        failed_samples.pop(key, None)
                    except lmdb.MapFullError:
                        raise
                    except Exception as error:
                        if manifest.get("split") in {"valid", "test"}:
                            raise
                        metadata["invocation_counts"]["errors"] += 1
                        failed_samples[key] = {
                            "key": key,
                            "error": str(error),
                        }
                    if (
                        args.progress_every
                        and index % args.progress_every == 0
                    ):
                        print(json.dumps(
                            metadata["invocation_counts"], sort_keys=True
                        ))

            cached_keys = _lmdb_keys(output_environment)
            cached_key_set = set(cached_keys)
            failed_samples = {
                key: value
                for key, value in failed_samples.items()
                if key not in cached_key_set
            }
            metadata["failed_samples"] = failed_samples
            metadata["counts"].update(
                {
                    "processed": len(cached_keys) + len(failed_samples),
                    "written": len(cached_keys),
                    "errors": len(failed_samples),
                    "cached_total": len(cached_keys),
                }
            )
            metadata["errors_preview"] = list(
                failed_samples.values()
            )[: args.max_error_preview]
            metadata["coverage_report"] = _coverage_report(
                output_environment
            )
            metadata["status"] = "complete"
            with output_environment.begin(
                write=True
            ) as output_transaction:
                output_transaction.put(
                    b"__metadata__",
                    pickle.dumps(
                        metadata, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
            atomic_write_json(output_path / "metadata.json", metadata)
        finally:
            output_environment.close()
    finally:
        source_environment.close()
    return metadata


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build schema-v1 native local-interface LMDB records."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--core-cutoff", type=float, default=6.0)
    parser.add_argument("--outer-cutoff", type=float, default=10.0)
    parser.add_argument("--shell-budget", type=int, default=128)
    parser.add_argument("--map-size-gb", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-error-preview", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    metadata = build_native_cache(build_arg_parser().parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
