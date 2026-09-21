#!/usr/bin/env python3
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.local_interface_decoys import apply_rigid_transform  # noqa: E402
from data_utils.local_interface_decoys_v2 import (  # noqa: E402
    BANK_NAMES,
    DECOY_SCHEMA_VERSION,
    STRATUM_ORDER,
    select_epoch_decoys_v2,
)
from scripts import build_local_interface_decoys_v2 as builder_v2  # noqa: E402
from utils.experiment_integrity import (  # noqa: E402
    atomic_write_json,
    hash_ordered_keys,
    sha256_file,
)
from utils.interface_decoy_audit import (  # noqa: E402
    HELDOUT_FEATURE_NAMES,
    evaluate_v2_data_gate,
    extract_coarse_matching_features,
    extract_heldout_interface_features,
    fit_grouped_pose_probes,
)
from utils.interface_diagnostics import SHORTCUT_FEATURE_NAMES  # noqa: E402


EVALUATOR_NAME = "local_interface_decoy_data_diagnostic_v2"
REPORT_FILENAME = "decoy_v2_data_diagnostic.json"
ROWS_FILENAME = "decoy_v2_data_rows.npz"
LOCK_FILENAME = ".decoy_v2_data_diagnostic.lock"
COARSE_DIMENSION = 9
HELDOUT_DIMENSION = 30


def _resolved_path(value, name):
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(f"{name} path must be a nonempty string")
    return Path(value).resolve()


def _read_json_snapshot(path, description):
    path = _resolved_path(path, description)
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return path, value, hashlib.sha256(payload).hexdigest()


def _manifest(path):
    resolved, value, digest = _read_json_snapshot(path, "manifest")
    lowered_name = resolved.name.lower()
    if (
        lowered_name.startswith(("valid", "test"))
        or "_valid" in lowered_name
        or "_test" in lowered_name
    ):
        raise ValueError("official valid/test manifest paths are not allowed")
    if value.get("split") != "train":
        raise ValueError("v2 data evaluation requires manifest split=train")
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    keys = []
    for sample in samples:
        if not isinstance(sample, dict) or "lmdb_key" not in sample:
            raise ValueError("every manifest sample must contain lmdb_key")
        if sample.get("split", "train") != "train":
            raise ValueError("all manifest entries must belong to train")
        keys.append(str(sample["lmdb_key"]))
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate lmdb keys")
    fingerprint = {
        "path": str(resolved),
        "sha256": digest,
        "split": "train",
        "sample_count": len(samples),
        "ordered_key_sha256": hash_ordered_keys(keys),
    }
    return resolved, keys, fingerprint


def _diagnostic(path, current_manifest, manifest_keys):
    resolved, value, digest = _read_json_snapshot(
        path, "diagnostic keys"
    )
    keys = value.get("keys")
    if not isinstance(keys, list):
        raise ValueError("diagnostic key file must contain a keys list")
    keys = [str(key) for key in keys]
    if len(keys) != len(set(keys)):
        raise ValueError("diagnostic key file contains duplicates")
    manifest_key_set = set(manifest_keys)
    unknown = [key for key in keys if key not in manifest_key_set]
    if unknown:
        raise ValueError(
            "diagnostic keys are not present in the train manifest: "
            + ", ".join(unknown[:5])
        )
    if value.get("manifest") != current_manifest:
        raise ValueError(
            "diagnostic keys were built from a different train manifest"
        )
    if (
        value.get("ordered_key_sha256") is not None
        and value["ordered_key_sha256"] != hash_ordered_keys(keys)
    ):
        raise ValueError("diagnostic ordered key hash does not match keys")
    return resolved, value, keys, digest


def _expected_requested_keys(cache_report, diagnostic_keys):
    provenance = cache_report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("completed cache provenance is missing")
    maximum = provenance.get("max_samples")
    if maximum is None:
        expected = list(diagnostic_keys)
    else:
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("cache max_samples provenance is invalid")
        expected = list(diagnostic_keys[:maximum])
    if cache_report.get("requested_keys") != expected:
        raise ValueError(
            "cache requested keys do not match the diagnostic key prefix"
        )
    return expected


def _provenance_path(provenance, name):
    return _resolved_path(
        provenance.get(name), f"cache provenance {name}"
    )


def _validate_cache_provenance(
    cache_report,
    decoy_path,
    current_manifest,
    native_path,
    native_environment,
    diagnostic_path,
    diagnostic,
    diagnostic_keys,
    diagnostic_digest,
    expected_keys,
    seed,
):
    provenance = cache_report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("completed cache provenance is missing")
    output_lmdb = _resolved_path(
        cache_report.get("output_lmdb"), "cache output_lmdb"
    )
    if (
        cache_report.get("builder") != builder_v2.BUILDER_NAME
        or cache_report.get("schema_version") != DECOY_SCHEMA_VERSION
        or cache_report.get("status") != "complete"
        or cache_report.get("cohort") != "diagnostic"
        or cache_report.get("seed") != int(seed)
        or output_lmdb != decoy_path
    ):
        raise ValueError(
            "completed cache seed/provenance is not the requested "
            "schema-v2 diagnostic cache"
        )
    if (
        provenance.get("builder") != builder_v2.BUILDER_NAME
        or provenance.get("schema_version") != DECOY_SCHEMA_VERSION
        or provenance.get("cohort") != "diagnostic"
        or provenance.get("seed") != int(seed)
        or provenance.get("manifest") != current_manifest
        or _provenance_path(provenance, "native_lmdb") != native_path
        or _provenance_path(provenance, "diagnostic_keys_path")
        != diagnostic_path
    ):
        raise ValueError("diagnostic cache provenance does not match inputs")
    if provenance.get("diagnostic_manifest") != diagnostic.get("manifest"):
        raise ValueError("diagnostic cache manifest provenance is mismatched")
    if (
        provenance.get("diagnostic_file_sha256") != diagnostic_digest
        or provenance.get("diagnostic_ordered_key_sha256")
        != hash_ordered_keys(diagnostic_keys)
    ):
        raise ValueError("diagnostic cache key-file provenance is mismatched")
    if (
        provenance.get("selected_key_count") != len(expected_keys)
        or provenance.get("selected_key_sha256")
        != hash_ordered_keys(expected_keys)
    ):
        raise ValueError("diagnostic cache selected-key provenance is mismatched")

    native_keys = builder_v2._lmdb_keys(native_environment)
    native_key_set = set(native_keys)
    if (
        provenance.get("native_key_count") != len(native_keys)
        or provenance.get("native_key_sha256") != hash_ordered_keys(native_keys)
    ):
        raise ValueError("native LMDB key inventory provenance is mismatched")
    missing_native = [
        key for key in expected_keys if key not in native_key_set
    ]
    if missing_native:
        raise ValueError(
            "diagnostic prefix keys are missing from native LMDB: "
            + ", ".join(missing_native[:5])
        )
    if (
        provenance.get("selected_native_record_sha256")
        != builder_v2._hash_ordered_lmdb_records(
            native_environment, expected_keys
        )
        or provenance.get("selected_automorphism_sha256")
        != builder_v2._automorphism_digest_aggregate(
            native_environment, expected_keys
        )
    ):
        raise ValueError("selected native record provenance is mismatched")
    native_metadata = builder_v2._native_metadata_summary(native_path)
    for name, value in native_metadata.items():
        if provenance.get(name) != value:
            raise ValueError("native metadata provenance is mismatched")


def _load_pickle(transaction, key, description):
    raw = transaction.get(str(key).encode("utf-8"))
    if raw is None:
        raise ValueError(f"{description} key is missing: {key}")
    try:
        value = pickle.loads(raw)
    except Exception as error:
        raise ValueError(
            f"{description} record is malformed: {key}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"{description} record must be a dictionary: {key}"
        )
    return value


def _validate_native_train_records(environment, ordered_keys):
    with environment.begin(write=False) as transaction:
        for key in ordered_keys:
            record = _load_pickle(
                transaction, key, "native diagnostic"
            )
            if record.get("key") != key:
                raise ValueError(
                    "native diagnostic record key does not match LMDB key: "
                    + str(key)
                )
            if record.get("split") != "train":
                raise ValueError(
                    "native diagnostic record must have split=train: "
                    + str(key)
                )


def _row_set(record, decoy_set, key, bank, seed):
    if record.get("key") != key:
        raise ValueError(
            "native diagnostic record key does not match LMDB key"
        )
    if record.get("split") != "train":
        raise ValueError(
            "native diagnostic record must have split=train"
        )
    if decoy_set.get("schema_version") != DECOY_SCHEMA_VERSION:
        raise ValueError("decoy schema version 2 is required")
    if decoy_set.get("key") != key:
        raise ValueError("decoy record key does not match LMDB key")
    if decoy_set.get("seed") != int(seed):
        raise ValueError("decoy record seed does not match evaluator seed")
    target = record.get("target_signature")
    if not isinstance(target, str) or not target:
        raise ValueError(f"native target_signature is missing: {key}")

    entries = select_epoch_decoys_v2(
        decoy_set,
        key=key,
        epoch=0,
        seed=int(seed),
        bank=bank,
    )
    if [entry["stratum"] for entry in entries] != list(STRATUM_ORDER):
        raise ValueError("epoch-0 decoy selection is not low/medium/high")
    native = np.asarray(record["ligand_coords"], dtype=np.float64)
    poses = [native]
    poses.extend(
        apply_rigid_transform(native, entry["transform"])
        for entry in entries
    )
    coarse = np.stack(
        [extract_coarse_matching_features(record, pose) for pose in poses],
        axis=0,
    ).astype(np.float64, copy=False)
    heldout = np.stack(
        [extract_heldout_interface_features(record, pose) for pose in poses],
        axis=0,
    ).astype(np.float64, copy=False)
    if (
        len(SHORTCUT_FEATURE_NAMES) != COARSE_DIMENSION
        or coarse.shape != (4, COARSE_DIMENSION)
    ):
        raise ValueError("coarse feature rows have an unexpected shape")
    if (
        len(HELDOUT_FEATURE_NAMES) != HELDOUT_DIMENSION
        or heldout.shape != (4, HELDOUT_DIMENSION)
    ):
        raise ValueError("held-out feature rows have an unexpected shape")
    if not np.all(np.isfinite(coarse)) or not np.all(
        np.isfinite(heldout)
    ):
        raise ValueError("diagnostic feature rows must be finite")
    return {
        "keys": np.asarray([key] * 4),
        "labels": np.asarray([1, 0, 0, 0], dtype=np.int64),
        "groups": np.asarray([target] * 4),
        "coarse_features": coarse,
        "heldout_features": heldout,
        "bank": np.asarray([bank] * 4),
        "stratum": np.asarray(["native", *STRATUM_ORDER]),
        "transform_fingerprint": np.asarray(
            ["native"]
            + [entry["transform_fingerprint"] for entry in entries]
        ),
    }


def _stack_rows(cohort, bank, ordered_keys, rows_by_key):
    if not ordered_keys:
        raise ValueError(f"{cohort} has no complete {bank} records")
    fields = (
        "keys",
        "labels",
        "groups",
        "coarse_features",
        "heldout_features",
        "bank",
        "stratum",
        "transform_fingerprint",
    )
    output = {
        field: np.concatenate(
            [rows_by_key[key][field] for key in ordered_keys], axis=0
        )
        for field in fields
    }
    output["cohort"] = np.asarray([cohort] * len(output["labels"]))
    return output


def _vector_sha256(value):
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        payload = array.astype("<i8", copy=False).tobytes(order="C")
        return hashlib.sha256(payload).hexdigest()
    return hash_ordered_keys(array.astype(str).tolist())


def _fold_assignment_sha256(labels, groups):
    from sklearn.model_selection import GroupKFold

    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    assignment = np.empty(len(labels), dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold_index, (_, test_index) in enumerate(
        splitter.split(np.zeros((len(labels), 1)), labels, groups=groups)
    ):
        assignment[test_index] = fold_index
    return hashlib.sha256(
        assignment.astype("<i8", copy=False).tobytes(order="C")
    ).hexdigest()


def _bounded_probe_summary(probe):
    return {
        key: value
        for key, value in probe.items()
        if key not in (
            "oof_linear_prediction",
            "oof_nonlinear_prediction",
        )
    }


def _bank_summary(rows):
    coarse = fit_grouped_pose_probes(
        rows["coarse_features"],
        rows["labels"],
        rows["groups"],
        SHORTCUT_FEATURE_NAMES,
    )
    heldout = fit_grouped_pose_probes(
        rows["heldout_features"],
        rows["labels"],
        rows["groups"],
        HELDOUT_FEATURE_NAMES,
    )
    return {
        "row_count": int(len(rows["labels"])),
        "labels_sha256": _vector_sha256(rows["labels"]),
        "groups_sha256": _vector_sha256(rows["groups"]),
        "fold_assignment_sha256": _fold_assignment_sha256(
            rows["labels"], rows["groups"]
        ),
        "coarse": _bounded_probe_summary(coarse),
        "heldout": _bounded_probe_summary(heldout),
    }


def _cohort(bank, keys, rows):
    return {
        "keys": list(keys),
        "ordered_key_sha256": hash_ordered_keys(keys),
        bank: _bank_summary(rows),
    }


def _atomic_write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp.npz", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _artifact_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _combined_npz(rows):
    names = (
        "cohort",
        "keys",
        "labels",
        "groups",
        "coarse_features",
        "heldout_features",
        "bank",
        "stratum",
        "transform_fingerprint",
    )
    return {
        name: np.concatenate([current[name] for current in rows], axis=0)
        for name in names
    }


def _require_unchanged(path, expected_sha256, description):
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{description} changed during evaluation")


def _validate_json_payload(value):
    try:
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "diagnostic report must be finite JSON data"
        ) from error


def evaluate_decoy_data(args):
    seed = int(args.seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    manifest_path, manifest_keys, current_manifest = _manifest(
        args.manifest
    )
    (
        diagnostic_path,
        diagnostic,
        diagnostic_keys,
        diagnostic_digest,
    ) = _diagnostic(
        args.diagnostic_keys, current_manifest, manifest_keys
    )

    native_path = _resolved_path(args.native_lmdb, "native LMDB")
    decoy_path = _resolved_path(
        args.diagnostic_decoy_lmdb, "diagnostic decoy LMDB"
    )
    with ExitStack() as stack:
        native_environment = builder_v2._open_readonly_lmdb(
            native_path
        )
        stack.callback(native_environment.close)
        decoy_environment = builder_v2._open_readonly_lmdb(decoy_path)
        stack.callback(decoy_environment.close)

        cache_report, _, completion_marker = (
            builder_v2._verify_completed_cache(
                decoy_path, decoy_environment
            )
        )
        cache_report_digest = completion_marker["report_sha256"]
        expected_keys = _expected_requested_keys(
            cache_report, diagnostic_keys
        )
        _validate_cache_provenance(
            cache_report=cache_report,
            decoy_path=decoy_path,
            current_manifest=current_manifest,
            native_path=native_path,
            native_environment=native_environment,
            diagnostic_path=diagnostic_path,
            diagnostic=diagnostic,
            diagnostic_keys=diagnostic_keys,
            diagnostic_digest=diagnostic_digest,
            expected_keys=expected_keys,
            seed=seed,
        )
        _validate_native_train_records(
            native_environment, expected_keys
        )

        included_keys = cache_report.get("included_keys")
        excluded_rows = cache_report.get("excluded_samples")
        if not isinstance(included_keys, list) or not isinstance(
            excluded_rows, list
        ):
            raise ValueError("completed cache key accounting is malformed")
        excluded_keys = [
            str(row.get("key"))
            for row in excluded_rows
            if isinstance(row, dict) and row.get("key") is not None
        ]
        if (
            len(included_keys) != len(set(included_keys))
            or len(excluded_keys) != len(set(excluded_keys))
            or set(included_keys).intersection(excluded_keys)
            or set(included_keys).union(excluded_keys) != set(expected_keys)
        ):
            raise ValueError(
                "completed cache does not account for every diagnostic prefix key"
            )
        expected_position = {
            key: index for index, key in enumerate(expected_keys)
        }
        if included_keys != sorted(
            included_keys, key=expected_position.__getitem__
        ):
            raise ValueError("completed cache included-key order is invalid")

        rows_by_bank = {bank: {} for bank in BANK_NAMES}
        random_keys = []
        matched_keys = []
        with native_environment.begin(
            write=False
        ) as native_transaction, decoy_environment.begin(
            write=False
        ) as decoy_transaction:
            for key in included_keys:
                record = _load_pickle(
                    native_transaction, key, "native diagnostic"
                )
                decoy_set = _load_pickle(
                    decoy_transaction, key, "v2 diagnostic decoy"
                )
                if decoy_set.get("schema_version") != DECOY_SCHEMA_VERSION:
                    raise ValueError("decoy schema version 2 is required")
                for bank in BANK_NAMES:
                    bank_value = decoy_set.get("banks", {}).get(bank)
                    if (
                        isinstance(bank_value, dict)
                        and bank_value.get("complete") is True
                    ):
                        rows_by_bank[bank][key] = _row_set(
                            record, decoy_set, key, bank, seed
                        )
                        if bank == "random_physical":
                            random_keys.append(key)
                        else:
                            matched_keys.append(key)

        paired_keys = [
            key
            for key in expected_keys
            if key in rows_by_bank["random_physical"]
            and key in rows_by_bank["matched"]
        ]
        if not paired_keys:
            raise ValueError("paired intersection is empty")
        paired_groups = {
            str(rows_by_bank["matched"][key]["groups"][0])
            for key in paired_keys
        }
        if len(paired_groups) < 5:
            raise ValueError(
                "paired diagnostic requires at least five target groups"
            )

        random_full_rows = _stack_rows(
            "random_physical_full",
            "random_physical",
            random_keys,
            rows_by_bank["random_physical"],
        )
        matched_full_rows = _stack_rows(
            "matched_full",
            "matched",
            matched_keys,
            rows_by_bank["matched"],
        )
        paired_random_rows = _stack_rows(
            "paired_intersection",
            "random_physical",
            paired_keys,
            rows_by_bank["random_physical"],
        )
        paired_matched_rows = _stack_rows(
            "paired_intersection",
            "matched",
            paired_keys,
            rows_by_bank["matched"],
        )
        if not (
            np.array_equal(
                paired_random_rows["labels"],
                paired_matched_rows["labels"],
            )
            and np.array_equal(
                paired_random_rows["groups"],
                paired_matched_rows["groups"],
            )
            and np.array_equal(
                paired_random_rows["keys"],
                paired_matched_rows["keys"],
            )
        ):
            raise ValueError("paired banks do not share row order and groups")

        cohorts = {
            "random_physical_full": _cohort(
                "random_physical", random_keys, random_full_rows
            ),
            "matched_full": _cohort(
                "matched", matched_keys, matched_full_rows
            ),
            "paired_intersection": {
                "keys": list(paired_keys),
                "ordered_key_sha256": hash_ordered_keys(paired_keys),
                "random_physical": _bank_summary(paired_random_rows),
                "matched": _bank_summary(paired_matched_rows),
            },
        }
        report = {
            "evaluator": EVALUATOR_NAME,
            "schema_version": DECOY_SCHEMA_VERSION,
            "status": "complete",
            "seed": seed,
            "inputs": {
                "manifest": current_manifest,
                "native_lmdb": str(native_path),
                "diagnostic_decoy_lmdb": str(decoy_path),
                "diagnostic_keys_path": str(diagnostic_path),
                "diagnostic_file_sha256": diagnostic_digest,
                "cache_report_sha256": cache_report_digest,
            },
            "cache_provenance": cache_report["provenance"],
            "bank_completion": cache_report["bank_completion"],
            "cohorts": cohorts,
        }
        report["gate"] = evaluate_v2_data_gate(report)

        output_dir = _resolved_path(args.output_dir, "output directory")
        rows_path = output_dir / ROWS_FILENAME
        report_path = output_dir / REPORT_FILENAME
        combined_rows = _combined_npz(
            [
                random_full_rows,
                matched_full_rows,
                paired_random_rows,
                paired_matched_rows,
            ]
        )
        with _artifact_lock(output_dir):
            _require_unchanged(
                manifest_path,
                current_manifest["sha256"],
                "manifest",
            )
            _require_unchanged(
                diagnostic_path,
                diagnostic_digest,
                "diagnostic key file",
            )
            _require_unchanged(
                decoy_path / builder_v2.REPORT_FILENAME,
                cache_report_digest,
                "cache report",
            )
            _atomic_write_npz(rows_path, combined_rows)
            rows_digest = sha256_file(rows_path)
            report["artifacts"] = {
                "report_json": {"filename": REPORT_FILENAME},
                "rows_npz": {
                    "filename": ROWS_FILENAME,
                    "sha256": rows_digest,
                },
            }
            _validate_json_payload(report)
            atomic_write_json(report_path, report)
            if sha256_file(rows_path) != rows_digest:
                raise RuntimeError(
                    "diagnostic row artifact changed during publication"
                )
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen train-only rotation-dominant v2 decoy banks "
            "before any GPU training."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-lmdb", required=True)
    parser.add_argument("--diagnostic-decoy-lmdb", required=True)
    parser.add_argument("--diagnostic-keys", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    report = evaluate_decoy_data(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate": report["gate"],
                "report": str(
                    Path(args.output_dir).resolve() / REPORT_FILENAME
                ),
                "rows": str(
                    Path(args.output_dir).resolve() / ROWS_FILENAME
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_pass and not report["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
