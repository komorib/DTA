#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import lmdb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.local_interface_decoys import (  # noqa: E402
    DECOY_SCHEMA_VERSION,
    DecoyGenerationError,
    generate_decoy_set,
    soft_contact_statistics,
    validate_decoy_set,
)
from utils.experiment_integrity import (  # noqa: E402
    atomic_write_json,
    hash_ordered_keys,
    manifest_fingerprint,
    sha256_file,
)


BUILDER_NAME = "local_interface_decoy_cache"


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
            digest.update(len(key_bytes).to_bytes(8, "big"))
            digest.update(key_bytes)
            if raw is None:
                digest.update(b"\x00")
                continue
            digest.update(b"\x01")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _write_record(environment, key, value):
    with environment.begin(write=True) as transaction:
        transaction.put(str(key).encode("utf-8"), value)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_diagnostic(path):
    path = Path(path).resolve()
    payload = _read_json(path)
    keys = payload.get("keys")
    if not isinstance(keys, list):
        raise ValueError("diagnostic key file must contain a keys list")
    normalized = [str(key) for key in keys]
    if len(normalized) != len(set(normalized)):
        raise ValueError("diagnostic key file contains duplicates")
    return path, payload, normalized


def _native_metadata_summary(native_path):
    metadata_path = native_path / "metadata.json"
    if not metadata_path.exists():
        return {
            "native_metadata_sha256": None,
            "native_schema_version": None,
        }
    metadata = _read_json(metadata_path)
    return {
        "native_metadata_sha256": sha256_file(metadata_path),
        "native_schema_version": metadata.get("schema_version"),
    }


def _provenance(
    manifest_path,
    native_path,
    native_keys,
    candidate_keys,
    candidate_record_sha256,
    max_samples,
    diagnostic_path,
    diagnostic_keys,
    seed,
):
    return {
        "builder": BUILDER_NAME,
        "schema_version": DECOY_SCHEMA_VERSION,
        "manifest": manifest_fingerprint(manifest_path),
        "native_lmdb": str(native_path),
        "native_key_count": len(native_keys),
        "native_key_sha256": hash_ordered_keys(native_keys),
        "max_samples": max_samples,
        "candidate_count": len(candidate_keys),
        "candidate_key_sha256": hash_ordered_keys(candidate_keys),
        "candidate_record_sha256": candidate_record_sha256,
        **_native_metadata_summary(native_path),
        "diagnostic_keys_path": str(diagnostic_path),
        "diagnostic_keys_sha256": sha256_file(diagnostic_path),
        "diagnostic_ordered_key_sha256": hash_ordered_keys(
            diagnostic_keys
        ),
        "seed": int(seed),
    }


def _prepare_output(output_path, resume, provenance):
    report_path = output_path / "decoy_report.json"
    if resume:
        if not report_path.exists():
            raise ValueError(
                "resume provenance is unavailable: decoy_report.json "
                "is missing"
            )
        previous = _read_json(report_path)
        if previous.get("provenance") != provenance:
            raise ValueError(
                "resume provenance does not match manifest, native keys, "
                "diagnostic keys, schema, or seed"
            )
        return previous
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"output LMDB is not empty: {output_path}; use --resume"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return None


def _finite_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _sample_properties(sample, record, native_contact_mass=None):
    ligand_size = record.get("ligand_heavy_atom_count")
    if ligand_size is None and "ligand_coords" in record:
        ligand_size = len(record["ligand_coords"])
    pocket_size = record.get("protein_atom_count")
    if pocket_size is None and "protein_coords" in record:
        pocket_size = len(record["protein_coords"])
    return {
        "key": str(sample["lmdb_key"]),
        "affinity": _finite_or_none(sample.get("label")),
        "ligand_size": (
            int(ligand_size) if ligand_size is not None else None
        ),
        "pocket_size": (
            int(pocket_size) if pocket_size is not None else None
        ),
        "native_contact_mass": _finite_or_none(native_contact_mass),
    }


def _record_native_contact_mass(record):
    try:
        stats = soft_contact_statistics(
            record["protein_coords"],
            record["ligand_coords"],
            record["ligand_coords"],
        )
    except (KeyError, TypeError, ValueError):
        return None
    return _finite_or_none(stats["native_contact_mass"])


def _numeric_summary(values):
    cleaned = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not cleaned:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    middle = len(cleaned) // 2
    if len(cleaned) % 2:
        median = cleaned[middle]
    else:
        median = 0.5 * (
            cleaned[middle - 1] + cleaned[middle]
        )
    return {
        "count": len(cleaned),
        "min": cleaned[0],
        "max": cleaned[-1],
        "mean": float(sum(cleaned) / len(cleaned)),
        "median": float(median),
    }


def _selection_summary(rows):
    return {
        "sample_count": len(rows),
        "affinity": _numeric_summary(
            [row.get("affinity") for row in rows]
        ),
        "ligand_size": _numeric_summary(
            [row.get("ligand_size") for row in rows]
        ),
        "pocket_size": _numeric_summary(
            [row.get("pocket_size") for row in rows]
        ),
        "native_contact_mass": _numeric_summary(
            [row.get("native_contact_mass") for row in rows]
        ),
    }


def _collect_cached_report(
    output_environment,
    native_environment,
    sample_by_key,
):
    included_rows = []
    metrics = {
        "rmsd_sym": [],
        "soft_jaccard": [],
        "contact_mass_ratio": [],
        "minimum_distance_ratio": [],
        "max_overlap_severity": [],
        "sum_overlap_severity": [],
    }
    stratum_counts = Counter()
    rejection_counts = Counter()
    with output_environment.begin(
        write=False
    ) as output_transaction, native_environment.begin(
        write=False
    ) as native_transaction:
        for key_bytes, raw in output_transaction.cursor():
            if key_bytes == b"__metadata__":
                continue
            key = key_bytes.decode("utf-8")
            decoy_set = pickle.loads(raw)
            record = _load_item(
                native_transaction, key, "native interface"
            )
            sample = sample_by_key[key]
            included_rows.append(
                _sample_properties(
                    sample,
                    record,
                    decoy_set.get("native_contact_mass"),
                )
            )
            rejection_counts.update(
                decoy_set.get("rejection_counts", {})
            )
            for entry in decoy_set.get("decoys", []):
                stratum_counts[entry["stratum"]] += 1
                metrics["rmsd_sym"].append(entry["rmsd_sym"])
                metrics["soft_jaccard"].append(
                    entry["contact_stats"]["soft_jaccard"]
                )
                metrics["contact_mass_ratio"].append(
                    entry["contact_stats"]["contact_mass_ratio"]
                )
                metrics["minimum_distance_ratio"].append(
                    entry["overlap_stats"]["minimum_distance_ratio"]
                )
                metrics["max_overlap_severity"].append(
                    entry["overlap_stats"]["max_overlap_severity"]
                )
                metrics["sum_overlap_severity"].append(
                    entry["overlap_stats"]["sum_overlap_severity"]
                )
    return {
        "included_rows": included_rows,
        "accepted_transform_counts": dict(sorted(stratum_counts.items())),
        "generation_rejection_counts": dict(
            sorted(rejection_counts.items())
        ),
        "accepted_decoy_distributions": {
            name: _numeric_summary(values)
            for name, values in metrics.items()
        },
    }


def _base_report(
    args,
    provenance,
    output_path,
    manifest_count,
    eligible_count,
    diagnostic_excluded,
):
    return {
        "builder": BUILDER_NAME,
        "schema_version": DECOY_SCHEMA_VERSION,
        "output_lmdb": str(output_path),
        "provenance": provenance,
        "manifest": provenance["manifest"],
        "native_lmdb": provenance["native_lmdb"],
        "diagnostic_keys_path": provenance["diagnostic_keys_path"],
        "seed": int(args.seed),
        "counts": {
            "manifest_samples": manifest_count,
            "diagnostic_excluded": diagnostic_excluded,
            "eligible_samples": eligible_count,
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
        "accepted_transform_counts": {},
        "generation_rejection_counts": {},
        "accepted_decoy_distributions": {},
        "selection_bias": {
            "included": {"sample_count": 0},
            "excluded": {"sample_count": 0},
        },
        "included_keys": [],
        "excluded_samples": [],
        "errors_preview": [],
        "status": "in_progress",
        "max_samples": args.max_samples,
    }


def build_decoy_cache(args):
    manifest_path, manifest, samples, ordered_keys = _load_manifest(
        args.manifest
    )
    if manifest.get("split") != "train":
        raise ValueError("decoy cache requires manifest split=train")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if args.progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if args.max_error_preview < 0:
        raise ValueError("max_error_preview must be non-negative")
    diagnostic_path, diagnostic, diagnostic_keys = _load_diagnostic(
        args.diagnostic_keys
    )
    manifest_key_set = set(ordered_keys)
    unknown_diagnostic = sorted(
        set(diagnostic_keys) - manifest_key_set
    )
    if unknown_diagnostic:
        raise ValueError(
            "diagnostic keys are not present in the train manifest: "
            + ", ".join(unknown_diagnostic[:5])
        )
    diagnostic_manifest = diagnostic.get("manifest")
    current_manifest = manifest_fingerprint(manifest_path)
    if (
        diagnostic_manifest is not None
        and diagnostic_manifest.get("sha256")
        != current_manifest["sha256"]
    ):
        raise ValueError(
            "diagnostic keys were built from a different manifest"
        )

    native_path = Path(args.native_lmdb).resolve()
    native_environment = _open_readonly_lmdb(native_path)
    try:
        diagnostic_set = set(diagnostic_keys)
        eligible_samples = [
            sample
            for sample in samples
            if str(sample["lmdb_key"]) not in diagnostic_set
        ]
        diagnostic_excluded = len(samples) - len(eligible_samples)
        if args.max_samples is not None:
            eligible_samples = eligible_samples[: args.max_samples]
        candidate_keys = [
            str(sample["lmdb_key"]) for sample in eligible_samples
        ]
        native_keys = _lmdb_keys(native_environment)
        candidate_record_sha256 = _hash_ordered_lmdb_records(
            native_environment, candidate_keys
        )
        provenance = _provenance(
            manifest_path,
            native_path,
            native_keys,
            candidate_keys,
            candidate_record_sha256,
            args.max_samples,
            diagnostic_path,
            diagnostic_keys,
            args.seed,
        )
        sample_by_key = {
            str(sample["lmdb_key"]): sample
            for sample in samples
        }

        output_path = Path(args.output_lmdb).resolve()
        previous = _prepare_output(
            output_path, bool(args.resume), provenance
        )
        report = _base_report(
            args,
            provenance,
            output_path,
            len(samples),
            len(eligible_samples),
            diagnostic_excluded,
        )
        previous_excluded = (
            list(previous.get("excluded_samples", []))
            if previous is not None
            else []
        )
        report["excluded_samples"] = previous_excluded
        atomic_write_json(output_path / "decoy_report.json", report)

        output_environment = _open_output_lmdb(
            output_path, args.map_size_gb
        )
        excluded_by_key = {
            row["key"]: row for row in previous_excluded
        }
        try:
            with native_environment.begin(
                write=False
            ) as native_transaction:
                for index, sample in enumerate(
                    eligible_samples, start=1
                ):
                    key = str(sample["lmdb_key"])
                    report["invocation_counts"]["processed"] += 1
                    with output_environment.begin(
                        write=False
                    ) as output_transaction:
                        exists = output_transaction.get(
                            key.encode("utf-8")
                        ) is not None
                    if exists:
                        report["invocation_counts"][
                            "skipped_existing"
                        ] += 1
                        excluded_by_key.pop(key, None)
                        continue
                    record = None
                    try:
                        record = _load_item(
                            native_transaction,
                            key,
                            "native interface",
                        )
                        decoy_set = generate_decoy_set(
                            record, seed=args.seed
                        )
                        validate_decoy_set(
                            decoy_set, native_record=record
                        )
                        serialized = pickle.dumps(
                            decoy_set,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                        _write_record(
                            output_environment, key, serialized
                        )
                        report["invocation_counts"]["written"] += 1
                        excluded_by_key.pop(key, None)
                    except lmdb.MapFullError:
                        raise
                    except Exception as error:
                        properties = _sample_properties(
                            sample,
                            record or {},
                            (
                                _record_native_contact_mass(record)
                                if record is not None
                                else None
                            ),
                        )
                        properties["error_type"] = type(error).__name__
                        properties["error"] = str(error)
                        if isinstance(error, DecoyGenerationError):
                            properties[
                                "generation_rejection_counts"
                            ] = dict(
                                sorted(error.rejection_counts.items())
                            )
                        else:
                            properties[
                                "generation_rejection_counts"
                            ] = {}
                        excluded_by_key[key] = properties
                        report["invocation_counts"]["errors"] += 1
                    if (
                        args.progress_every
                        and index % args.progress_every == 0
                    ):
                        print(json.dumps(
                            report["invocation_counts"], sort_keys=True
                        ))

            cached = _collect_cached_report(
                output_environment,
                native_environment,
                sample_by_key,
            )
            cached_keys = {
                row["key"] for row in cached["included_rows"]
            }
            excluded_rows = [
                excluded_by_key[key]
                for key in sorted(excluded_by_key)
                if key not in cached_keys
            ]
            included_rows = cached["included_rows"]
            report["counts"].update(
                {
                    "processed": len(included_rows)
                    + len(excluded_rows),
                    "written": len(included_rows),
                    "errors": len(excluded_rows),
                    "cached_total": len(included_rows),
                }
            )
            report["included_keys"] = sorted(
                row["key"] for row in included_rows
            )
            report["excluded_samples"] = excluded_rows
            report["errors_preview"] = [
                {
                    "key": row["key"],
                    "error_type": row["error_type"],
                    "error": row["error"],
                }
                for row in excluded_rows[: args.max_error_preview]
            ]
            report["accepted_transform_counts"] = cached[
                "accepted_transform_counts"
            ]
            generation_failures = Counter(
                cached["generation_rejection_counts"]
            )
            for row in excluded_rows:
                generation_failures.update(
                    row.get("generation_rejection_counts", {})
                )
            report["generation_rejection_counts"] = dict(
                sorted(generation_failures.items())
            )
            report["accepted_decoy_distributions"] = cached[
                "accepted_decoy_distributions"
            ]
            report["selection_bias"] = {
                "included": _selection_summary(included_rows),
                "excluded": _selection_summary(excluded_rows),
            }
            report["status"] = "complete"
            with output_environment.begin(
                write=True
            ) as output_transaction:
                output_transaction.put(
                    b"__metadata__",
                    pickle.dumps(
                        report, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
            atomic_write_json(
                output_path / "decoy_report.json", report
            )
        finally:
            output_environment.close()
    finally:
        native_environment.close()
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build train-only symmetry-aware hard-pose decoys."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-lmdb", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--diagnostic-keys", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-size-gb", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-error-preview", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    report = build_decoy_cache(build_arg_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
