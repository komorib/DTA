import math


DECISION_THRESHOLDS = {
    "minimum_matched_complete": 20,
    "minimum_per_rank_quartile": 1,
    "minimum_upper_half_complete": 4,
    "maximum_no_vdw_safe_family_fraction": 0.90,
}
RANK_METRICS = (
    "ligand_size",
    "pocket_size",
    "native_contact_mass",
)


def _rank_bin_sizes(count, bin_count=4):
    base, remainder = divmod(count, bin_count)
    return [base + int(index < remainder) for index in range(bin_count)]


def _completion_rank_bins(records, complete_keys):
    result = {}
    for metric in RANK_METRICS:
        ordered = []
        for record in records:
            value = record["sample_properties"].get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(metric + " must be finite")
            ordered.append((float(value), str(record["key"])))
        ordered.sort(key=lambda row: (row[0], row[1]))
        bins = []
        offset = 0
        for index, size in enumerate(_rank_bin_sizes(len(ordered))):
            rows = ordered[offset : offset + size]
            keys = [key for _, key in rows]
            bins.append(
                {
                    "index": index,
                    "keys": keys,
                    "count": len(keys),
                    "value_min": rows[0][0] if rows else None,
                    "value_max": rows[-1][0] if rows else None,
                    "matched_complete_count": sum(
                        key in complete_keys for key in keys
                    ),
                }
            )
            offset += size
        result[metric] = {
            "bins": bins,
            "upper_half_matched_complete_count": sum(
                row["matched_complete_count"] for row in bins[2:]
            ),
        }
    return result


def aggregate_probe_report(
    requested_keys,
    native_exclusions,
    records,
    decision_eligible,
):
    requested_keys = [str(key) for key in requested_keys]
    if len(requested_keys) != len(set(requested_keys)):
        raise ValueError("requested probe keys must be unique")
    exclusion_keys = [str(row["key"]) for row in native_exclusions]
    if len(exclusion_keys) != len(set(exclusion_keys)):
        raise ValueError("native exclusion keys must be unique")
    requested_set = set(requested_keys)
    if not set(exclusion_keys).issubset(requested_set):
        raise ValueError("native exclusions are outside requested keys")

    record_by_key = {str(row["key"]): row for row in records}
    if len(record_by_key) != len(records):
        raise ValueError("probe record keys must be unique")
    eligible_keys = [
        key for key in requested_keys if key not in set(exclusion_keys)
    ]
    if set(record_by_key) != set(eligible_keys):
        raise ValueError("probe records do not exactly cover eligible keys")
    ordered_records = [record_by_key[key] for key in eligible_keys]

    family_fields = (
        "guided_any_vdw_safe",
        "guided_any_physical",
        "guided_any_matched",
        "expanded_any_vdw_safe",
        "expanded_any_physical",
        "expanded_any_matched",
        "fibonacci_added_physical",
        "fibonacci_added_matched",
    )
    family_totals = {field: 0 for field in family_fields}
    rejection_totals = {}
    direction_source_totals = {}
    total_rotation_families = 0
    matched_complete_keys = []
    physical_complete_keys = []
    for record in ordered_records:
        total_rotation_families += int(record["rotation_count"])
        counts = record["family_counts"]
        for field in family_fields:
            family_totals[field] += int(counts.get(field, 0))
        for reason, count in record["rejection_counts"].items():
            rejection_totals[reason] = rejection_totals.get(reason, 0) + int(count)
        for source, count in record["direction_source_counts"].items():
            direction_source_totals[source] = direction_source_totals.get(source, 0) + int(count)
        if int(counts.get("expanded_any_physical", 0)) >= 4:
            physical_complete_keys.append(record["key"])
        if int(counts.get("expanded_any_matched", 0)) >= 4:
            matched_complete_keys.append(record["key"])

    complete_set = set(matched_complete_keys)
    return {
        "status": "complete",
        "decision_eligible": bool(decision_eligible),
        "requested_count": len(requested_keys),
        "eligible_count": len(eligible_keys),
        "native_excluded_count": len(exclusion_keys),
        "requested_keys": requested_keys,
        "eligible_keys": eligible_keys,
        "native_exclusions": list(native_exclusions),
        "total_rotation_families": total_rotation_families,
        "family_totals": family_totals,
        "rejection_counts": dict(sorted(rejection_totals.items())),
        "direction_source_counts": dict(sorted(direction_source_totals.items())),
        "sample_completion": {
            "physical_complete_count": len(physical_complete_keys),
            "physical_complete_keys": physical_complete_keys,
            "matched_complete_count": len(matched_complete_keys),
            "matched_complete_keys": matched_complete_keys,
        },
        "completion_rank_bins": _completion_rank_bins(
            ordered_records, complete_set
        ),
    }


def evaluate_probe_decision(report):
    if not report.get("decision_eligible"):
        return None
    thresholds = dict(DECISION_THRESHOLDS)
    total = int(report["total_rotation_families"])
    safe = int(report["family_totals"]["expanded_any_vdw_safe"])
    no_vdw_safe_fraction = 1.0 - (float(safe) / total if total else 0.0)
    matched_complete = int(
        report["sample_completion"]["matched_complete_count"]
    )
    reasons = []
    if no_vdw_safe_fraction > thresholds[
        "maximum_no_vdw_safe_family_fraction"
    ]:
        reasons.append("expanded_no_vdw_safe_fraction_above_0_90")
    if matched_complete < thresholds["minimum_matched_complete"]:
        reasons.append("matched_complete_count_below_20")
    for metric in RANK_METRICS:
        summary = report["completion_rank_bins"][metric]
        for row in summary["bins"]:
            if row["matched_complete_count"] < thresholds[
                "minimum_per_rank_quartile"
            ]:
                reasons.append(
                    metric + "_quartile_%d_has_no_completion" % row["index"]
                )
        if summary["upper_half_matched_complete_count"] < thresholds[
            "minimum_upper_half_complete"
        ]:
            reasons.append(metric + "_upper_half_below_4")

    if "expanded_no_vdw_safe_fraction_above_0_90" in reasons:
        outcome = "translation_search_no_go_vdw"
    elif reasons:
        outcome = "translation_search_no_go_matching_or_stratum"
    else:
        outcome = "translation_search_go"
    return {
        "passed": not reasons,
        "outcome": outcome,
        "reasons": sorted(reasons),
        "metrics": {
            "matched_complete_count": matched_complete,
            "eligible_count": int(report["eligible_count"]),
            "expanded_no_vdw_safe_fraction": no_vdw_safe_fraction,
        },
        "thresholds": thresholds,
    }



import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import lmdb
import numpy as np
from rdkit import rdBase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils import local_interface_translation_probe_v4 as probe_core
from scripts import build_local_interface_decoys_v3 as builder_v3
from utils.experiment_integrity import (
    atomic_write_json,
    hash_ordered_keys,
    manifest_fingerprint,
    sha256_file,
)


REPORT_FILENAME = "translation_probe_report.json"
METADATA_KEY = b"__metadata__"
RECORD_DIGEST_PREFIX = b"__record_sha256__:"
EXPECTED_REQUESTED_COUNT = 64
EXPECTED_ELIGIBLE_COUNT = 56
NATIVE_EXCLUSION_REASONS = {"native_hard_clash", "native_contact_mass"}


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_v3_smoke_report(report):
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("builder") != "local_interface_decoy_cache_v3"
        or report.get("schema_version") != 3
        or report.get("cohort") != "generator_smoke"
    ):
        raise ValueError("a complete V3 smoke report is required")
    requested = report.get("requested_keys")
    if (
        not isinstance(requested, list)
        or len(requested) != EXPECTED_REQUESTED_COUNT
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("complete V3 smoke must contain 64 unique requested keys")
    cohort_from_v3_smoke_report(report)
    return report


def cohort_from_v3_smoke_report(report):
    requested = [str(key) for key in report.get("requested_keys", [])]
    excluded_rows = report.get("excluded_samples", [])
    if not isinstance(excluded_rows, list):
        raise ValueError("V3 smoke excluded_samples must be a list")
    native_rows = [
        row
        for row in excluded_rows
        if row.get("exclusion_reason") in NATIVE_EXCLUSION_REASONS
    ]
    native_keys = [str(row["key"]) for row in native_rows]
    if len(native_keys) != len(set(native_keys)):
        raise ValueError("V3 smoke native exclusions contain duplicate keys")
    requested_set = set(requested)
    if not set(native_keys).issubset(requested_set):
        raise ValueError("V3 smoke native exclusions are outside requested keys")
    eligible = [key for key in requested if key not in set(native_keys)]
    if len(requested) == EXPECTED_REQUESTED_COUNT and len(eligible) != EXPECTED_ELIGIBLE_COUNT:
        raise ValueError("complete V3 smoke must contain exactly 56 eligible keys")
    return {
        "requested_keys": requested,
        "native_exclusions": native_rows,
        "native_exclusion_keys": native_keys,
        "eligible_keys": eligible,
    }


def is_decision_eligible(
    max_eligible_samples,
    rotation_limit,
    eligible_count,
):
    return (
        max_eligible_samples is None
        and rotation_limit is None
        and int(eligible_count) == EXPECTED_ELIGIBLE_COUNT
    )


def prepare_output(output_path, resume, provenance):
    output_path = Path(output_path)
    report_path = output_path / REPORT_FILENAME
    if resume:
        if not report_path.is_file():
            raise ValueError("resume provenance is unavailable")
        previous = _read_json(report_path)
        if previous.get("provenance") != provenance:
            raise ValueError("resume provenance does not match")
        return previous
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            "output directory is not empty; use --resume"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return None


def _fibonacci_sha256():
    values = np.asarray(
        probe_core.fibonacci_directions(), dtype="<f8", order="C"
    )
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _build_provenance(
    args,
    manifest_path,
    v3_report_path,
    v3_report,
    native_path,
    native_environment,
    eligible_keys,
):
    return {
        "builder": "translation_feasibility_probe_v4",
        "schema": probe_core.PROBE_SCHEMA,
        "seed": int(args.seed),
        "manifest": manifest_fingerprint(manifest_path),
        "v3_smoke_report_path": str(v3_report_path),
        "v3_smoke_report_sha256": sha256_file(v3_report_path),
        "v3_generator_configuration": v3_report["provenance"][
            "generator_configuration"
        ],
        "native_lmdb": str(native_path),
        "eligible_key_count": len(eligible_keys),
        "eligible_key_sha256": hash_ordered_keys(eligible_keys),
        "eligible_record_sha256": builder_v3._hash_ordered_lmdb_records(
            native_environment, eligible_keys
        ),
        "max_eligible_samples": args.max_eligible_samples,
        "rotation_limit": args.rotation_limit,
        "configuration": {
            "target_stratum": probe_core.TARGET_STRATUM,
            "base_rotations": probe_core.BASE_ROTATIONS,
            "guided_direction_cap": probe_core.MAX_GUIDED_DIRECTIONS,
            "fibonacci_direction_count": (
                probe_core.FIBONACCI_DIRECTION_COUNT
            ),
            "fibonacci_sha256": _fibonacci_sha256(),
            "distance_grid": list(probe_core.HIGH_DISTANCE_GRID),
            "maximum_translation": probe_core.MAX_TRANSLATION,
            "hard_vdw_ratio": builder_v3.HARD_VDW_RATIO,
        },
        "runtime": {
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }


def _digest_key(key):
    return RECORD_DIGEST_PREFIX + str(key).encode("utf-8")


def _record_keys(environment):
    with environment.begin(write=False) as transaction:
        return sorted(
            key.decode("utf-8")
            for key in transaction.cursor().iternext(keys=True, values=False)
            if not key.startswith(b"__")
        )


def _write_record(environment, key, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest().encode("ascii")
    with environment.begin(write=True) as transaction:
        transaction.put(str(key).encode("utf-8"), payload)
        transaction.put(_digest_key(key), digest)


def _load_valid_record(
    transaction,
    key,
    expected_rotation_count,
):
    payload = transaction.get(str(key).encode("utf-8"))
    if payload is None:
        return None
    digest = transaction.get(_digest_key(key))
    expected_digest = hashlib.sha256(payload).hexdigest().encode("ascii")
    if digest != expected_digest:
        raise ValueError("cached probe record digest does not match")
    value = pickle.loads(payload)
    probe_core.validate_probe_record(
        value,
        expected_key=str(key),
        expected_rotation_count=expected_rotation_count,
    )
    return value


def _validate_positive_int(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(name + " must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(name + " is out of range")


def _validate_run_args(args):
    _validate_positive_int(args.seed, "seed", allow_zero=True)
    _validate_positive_int(
        args.progress_every, "progress_every", allow_zero=True
    )
    if args.max_eligible_samples is not None:
        _validate_positive_int(
            args.max_eligible_samples, "max_eligible_samples"
        )
    if args.rotation_limit is not None:
        _validate_positive_int(args.rotation_limit, "rotation_limit")
        if args.rotation_limit > probe_core.BASE_ROTATIONS:
            raise ValueError("rotation_limit cannot exceed 512")
    if float(args.map_size_gb) <= 0.0:
        raise ValueError("map_size_gb must be positive")


def run_probe(args, record_builder=probe_core.probe_record):
    _validate_run_args(args)
    manifest_path, manifest, samples, manifest_keys = (
        builder_v3._load_manifest(args.manifest)
    )
    if manifest.get("split") != "train":
        raise ValueError("translation probe requires manifest split=train")

    v3_report_path = Path(args.v3_smoke_report).resolve()
    v3_report = validate_v3_smoke_report(_read_json(v3_report_path))
    if int(v3_report["seed"]) != int(args.seed):
        raise ValueError("probe seed must match the V3 smoke seed")
    cohort = cohort_from_v3_smoke_report(v3_report)
    unknown = [
        key for key in cohort["requested_keys"] if key not in set(manifest_keys)
    ]
    if unknown:
        raise ValueError("V3 smoke keys are missing from the train manifest")

    all_eligible = cohort["eligible_keys"]
    selected_eligible = list(all_eligible)
    if args.max_eligible_samples is not None:
        selected_eligible = selected_eligible[: args.max_eligible_samples]
    decision_eligible = is_decision_eligible(
        args.max_eligible_samples,
        args.rotation_limit,
        len(all_eligible),
    )
    expected_rotation_count = (
        probe_core.BASE_ROTATIONS
        if args.rotation_limit is None
        else int(args.rotation_limit)
    )

    native_path = Path(args.native_lmdb).resolve()
    native_environment = builder_v3._open_readonly_lmdb(native_path)
    output_environment = None
    try:
        native_keys = set(builder_v3._lmdb_keys(native_environment))
        missing = [key for key in selected_eligible if key not in native_keys]
        if missing:
            raise ValueError("eligible keys are missing from native LMDB")
        provenance = _build_provenance(
            args,
            manifest_path,
            v3_report_path,
            v3_report,
            native_path,
            native_environment,
            selected_eligible,
        )
        output_path = Path(args.output_lmdb).resolve()
        prepare_output(output_path, bool(args.resume), provenance)
        output_environment = builder_v3._open_output_lmdb(
            output_path, args.map_size_gb
        )
        with output_environment.begin(write=True) as transaction:
            existing_metadata = transaction.get(METADATA_KEY)
            encoded = pickle.dumps(
                provenance, protocol=pickle.HIGHEST_PROTOCOL
            )
            if existing_metadata is None:
                transaction.put(METADATA_KEY, encoded)
            elif pickle.loads(existing_metadata) != provenance:
                raise ValueError("LMDB metadata provenance does not match")

        existing_keys = _record_keys(output_environment)
        unexpected = [
            key for key in existing_keys if key not in set(selected_eligible)
        ]
        if unexpected:
            raise ValueError("probe LMDB contains unexpected record keys")

        invocation = {
            "processed": 0,
            "written": 0,
            "skipped_existing": 0,
        }
        base_report = {
            "status": "in_progress",
            "output_lmdb": str(output_path),
            "provenance": provenance,
            "decision_eligible": decision_eligible,
            "cohort_requested_count": len(cohort["requested_keys"]),
            "cohort_eligible_count": len(all_eligible),
            "selected_eligible_count": len(selected_eligible),
            "invocation_counts": invocation,
        }
        atomic_write_json(output_path / REPORT_FILENAME, base_report)

        sample_by_key = {
            str(row["lmdb_key"]): row for row in samples
        }
        records = []
        with native_environment.begin(write=False) as native_transaction:
            for index, key in enumerate(selected_eligible, start=1):
                invocation["processed"] += 1
                with output_environment.begin(
                    write=False
                ) as output_transaction:
                    value = _load_valid_record(
                        output_transaction,
                        key,
                        expected_rotation_count,
                    )
                if value is not None:
                    invocation["skipped_existing"] += 1
                else:
                    native_record = builder_v3._load_item(
                        native_transaction, key, "native interface"
                    )
                    value = record_builder(
                        native_record,
                        seed=args.seed,
                        rotation_limit=args.rotation_limit,
                    )
                    _write_record(output_environment, key, value)
                    invocation["written"] += 1
                records.append(value)
                if (
                    args.progress_every
                    and index % args.progress_every == 0
                ):
                    print(
                        json.dumps(
                            {
                                "processed": index,
                                "selected": len(selected_eligible),
                                "written": invocation["written"],
                                "skipped_existing": invocation[
                                    "skipped_existing"
                                ],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                base_report["invocation_counts"] = dict(invocation)
                atomic_write_json(
                    output_path / REPORT_FILENAME, base_report
                )

        active_key_set = set(selected_eligible) | set(
            cohort["native_exclusion_keys"]
        )
        active_requested = [
            key
            for key in cohort["requested_keys"]
            if key in active_key_set
        ]
        final_report = aggregate_probe_report(
            active_requested,
            cohort["native_exclusions"],
            records,
            decision_eligible=decision_eligible,
        )
        final_report.update(
            {
                "output_lmdb": str(output_path),
                "provenance": provenance,
                "cohort_requested_count": len(cohort["requested_keys"]),
                "cohort_eligible_count": len(all_eligible),
                "selected_eligible_count": len(selected_eligible),
                "invocation_counts": dict(invocation),
            }
        )
        final_report["decision"] = evaluate_probe_decision(final_report)
        atomic_write_json(output_path / REPORT_FILENAME, final_report)
        return final_report
    finally:
        if output_environment is not None:
            output_environment.close()
        native_environment.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Probe V3 high rotations with wider V4 translations."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-lmdb", required=True)
    parser.add_argument("--v3-smoke-report", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-size-gb", type=float, default=16.0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-eligible-samples", type=int)
    parser.add_argument("--rotation-limit", type=int)
    return parser


def main():
    report = run_probe(build_arg_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_lmdb": report["output_lmdb"],
                "requested_count": report["requested_count"],
                "eligible_count": report["eligible_count"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
