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
import numpy as np
from rdkit import rdBase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.local_interface_decoys import (  # noqa: E402
    GEOMETRY_ATOL,
    GEOMETRY_RTOL,
    STRATUM_BOUNDS,
    DecoyGenerationError,
    soft_contact_statistics,
)
from data_utils.local_interface_decoys_v2 import (  # noqa: E402
    BANK_NAMES,
    CENTROID_LIMITS,
    CONTACT_MASS_RATIO_RANGE,
    DECOY_SCHEMA_VERSION,
    HARD_VDW_RATIO,
    MATCH_LIMITS,
    NativeContactMassExclusion,
    PROPOSAL_RANGES,
    PROPOSALS_PER_STRATUM,
    RETAINED_PER_STRATUM,
    RNG_BIT_GENERATOR,
    SHORTLIST_SIZE,
    STRATUM_ORDER,
    automorphism_digest,
    generate_decoy_set_v2,
    validate_decoy_set_v2,
)
from utils.experiment_integrity import (  # noqa: E402
    atomic_write_json,
    hash_ordered_keys,
    manifest_fingerprint,
    sha256_file,
)


BUILDER_NAME = "local_interface_decoy_cache_v2"
REPORT_FILENAME = "decoy_report.json"
METADATA_KEY = b"__metadata__"
COMPLETION_KEY = b"__completion__"
RECORD_DIGEST_PREFIX = b"__record_sha256__:"
AUTOMORPHISM_CAP = 4096

DISTRIBUTION_METRICS = (
    "rmsd_sym",
    "soft_jaccard",
    "contact_mass_ratio",
    "delta_centroid_distance",
    "delta_minimum_distance",
    "delta_max_overlap_severity",
    "delta_sum_overlap_per_ligand_atom",
    "delta_count_below_0_9_per_ligand_atom",
    "delta_count_below_0_8_per_ligand_atom",
    "minimum_distance_ratio",
    "max_overlap_severity",
    "sum_overlap_severity",
    "count_below_0_9",
    "count_below_0_8",
    "match_distance",
)


def _canonical_json_value(value):
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest(path):
    path = Path(path).resolve()
    manifest = _read_json(path)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    try:
        keys = [str(sample["lmdb_key"]) for sample in samples]
    except (KeyError, TypeError) as error:
        raise ValueError("every manifest sample must contain lmdb_key") from error
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate lmdb keys")
    return path, manifest, samples, keys


def _load_diagnostic(path):
    path = Path(path).resolve()
    diagnostic = _read_json(path)
    keys = diagnostic.get("keys")
    if not isinstance(keys, list):
        raise ValueError("diagnostic key file must contain a keys list")
    normalized = [str(key) for key in keys]
    if len(normalized) != len(set(normalized)):
        raise ValueError("diagnostic key file contains duplicates")
    return path, diagnostic, normalized


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
    return lmdb.open(
        str(path),
        map_size=int(float(map_size_gb) * 1024 ** 3),
        subdir=True,
        meminit=False,
    )


def _lmdb_keys(environment):
    with environment.begin(write=False) as transaction:
        return sorted(
            key.decode("utf-8")
            for key in transaction.cursor().iternext(keys=True, values=False)
            if key != METADATA_KEY
        )


def _decode_output_key(key_bytes, label):
    try:
        return key_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"existing v2 decoy cache contains invalid {label} key bytes"
        ) from error


def _output_key_inventory(environment):
    record_keys = []
    digest_keys = []
    unexpected_reserved = []
    with environment.begin(write=False) as transaction:
        for key_bytes in transaction.cursor().iternext(
            keys=True, values=False
        ):
            if key_bytes in (METADATA_KEY, COMPLETION_KEY):
                continue
            if key_bytes.startswith(RECORD_DIGEST_PREFIX):
                subject = key_bytes[len(RECORD_DIGEST_PREFIX):]
                if not subject:
                    unexpected_reserved.append(
                        _decode_output_key(
                            key_bytes, "reserved"
                        )
                    )
                else:
                    digest_keys.append(
                        _decode_output_key(subject, "digest subject")
                    )
                continue
            if key_bytes.startswith(b"__"):
                unexpected_reserved.append(
                    _decode_output_key(key_bytes, "reserved")
                )
                continue
            record_keys.append(
                _decode_output_key(key_bytes, "record")
            )
    return (
        sorted(record_keys),
        sorted(digest_keys),
        sorted(unexpected_reserved),
    )


def _output_record_keys(environment):
    record_keys, _, _ = _output_key_inventory(environment)
    return record_keys


def _cache_integrity_summary(environment, ordered_keys):
    if not isinstance(ordered_keys, list) or any(
        not isinstance(key, str) for key in ordered_keys
    ):
        raise ValueError(
            "cache integrity included_keys must be a list of strings"
        )
    if len(ordered_keys) != len(set(ordered_keys)):
        raise ValueError(
            "cache integrity included_keys must not contain duplicates"
        )

    record_keys, digest_keys, unexpected_reserved = (
        _output_key_inventory(environment)
    )
    if unexpected_reserved:
        raise ValueError(
            "existing v2 decoy cache contains unexpected reserved keys: "
            + ", ".join(unexpected_reserved[:5])
        )
    record_key_set = set(record_keys)
    digest_key_set = set(digest_keys)
    orphan_digests = sorted(digest_key_set - record_key_set)
    if orphan_digests:
        raise ValueError(
            "existing v2 decoy cache contains orphan record digest: "
            + ", ".join(orphan_digests[:5])
        )
    missing_digests = sorted(record_key_set - digest_key_set)
    if missing_digests:
        raise ValueError(
            "existing v2 decoy cache contains missing record digest: "
            + ", ".join(missing_digests[:5])
        )
    if (
        len(ordered_keys) != len(record_keys)
        or set(ordered_keys) != record_key_set
    ):
        raise ValueError(
            "cache integrity record keys do not match included_keys"
        )

    rows = []
    with environment.begin(write=False) as transaction:
        for key in ordered_keys:
            raw = transaction.get(key.encode("utf-8"))
            stored_digest = transaction.get(_record_digest_key(key))
            if raw is None:
                raise ValueError(
                    "cache integrity record keys do not match included_keys"
                )
            record_sha256 = hashlib.sha256(raw).hexdigest()
            if stored_digest != record_sha256.encode("ascii"):
                raise ValueError(
                    "existing v2 decoy record is invalid: "
                    f"{key}: record digest mismatch"
                )
            rows.append(
                {"key": key, "record_sha256": record_sha256}
            )
    return {
        "record_count": len(rows),
        "ordered_key_sha256": hash_ordered_keys(ordered_keys),
        "ordered_record_digest_sha256": _canonical_sha256(rows),
    }


def _verify_completed_cache(output_path, environment):
    report_path = Path(output_path) / REPORT_FILENAME
    try:
        report = _read_json(report_path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "completion marker cannot authorize an unreadable report"
        ) from error
    if report.get("status") != "complete":
        raise ValueError(
            "completion marker does not authorize an incomplete report"
        )

    actual_cache_integrity = _cache_integrity_summary(
        environment, report.get("included_keys")
    )
    if report.get("cache_integrity") != actual_cache_integrity:
        raise ValueError(
            "completion cache integrity does not match actual records"
        )

    with environment.begin(write=False) as transaction:
        metadata_raw = transaction.get(METADATA_KEY)
        marker_raw = transaction.get(COMPLETION_KEY)
    if metadata_raw is None or marker_raw is None:
        raise ValueError(
            "completion marker or complete metadata is missing"
        )
    try:
        metadata = pickle.loads(metadata_raw)
        marker = pickle.loads(marker_raw)
    except Exception as error:
        raise ValueError("completion marker is malformed") from error

    expected_provenance = report.get("provenance")
    expected_report_sha256 = sha256_file(report_path)
    expected_provenance_sha256 = _canonical_sha256(
        expected_provenance
    )
    expected_integrity_sha256 = _canonical_sha256(
        actual_cache_integrity
    )
    if (
        metadata != report
        or not isinstance(marker, dict)
        or marker.get("builder") != BUILDER_NAME
        or marker.get("schema_version") != DECOY_SCHEMA_VERSION
        or marker.get("report_sha256") != expected_report_sha256
        or marker.get("provenance") != expected_provenance
        or marker.get("provenance_sha256")
        != expected_provenance_sha256
        or marker.get("cache_integrity")
        != actual_cache_integrity
        or marker.get("cache_integrity_sha256")
        != expected_integrity_sha256
    ):
        raise ValueError(
            "completion marker does not match report and metadata"
        )
    return report, metadata, marker


def _record_digest_key(key):
    return RECORD_DIGEST_PREFIX + str(key).encode("utf-8")


def _load_item(transaction, key, source_name):
    raw = transaction.get(str(key).encode("utf-8"))
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
                raise KeyError(f"native interface LMDB key not found: {key}")
            for value in (key_bytes, raw):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
    return digest.hexdigest()


def _automorphism_digest_aggregate(environment, ordered_keys):
    aggregate = hashlib.sha256()
    with environment.begin(write=False) as transaction:
        for key in ordered_keys:
            record = _load_item(transaction, key, "native interface")
            if "ligand_automorphisms" not in record:
                raise ValueError(
                    f"native interface record lacks ligand_automorphisms: {key}"
                )
            mappings = np.asarray(record["ligand_automorphisms"])
            digest = automorphism_digest(mappings)
            payload = json.dumps(
                [str(key), int(len(mappings)), digest],
                separators=(",", ":"),
            ).encode("utf-8")
            aggregate.update(len(payload).to_bytes(8, "big"))
            aggregate.update(payload)
    return aggregate.hexdigest()


def _write_record(environment, key, value):
    digest = hashlib.sha256(value).hexdigest().encode("ascii")
    with environment.begin(write=True) as transaction:
        transaction.put(str(key).encode("utf-8"), value)
        transaction.put(_record_digest_key(key), digest)


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


def _generator_configuration():
    configuration = {
        "bank_names": list(BANK_NAMES),
        "stratum_order": list(STRATUM_ORDER),
        "proposal_ranges": PROPOSAL_RANGES,
        "proposal_sampling": {
            "rotation_center": "ligand_centroid",
            "rotation_axis": "normalized_standard_gaussian_3d",
            "rotation_angle": "uniform_degrees_in_stratum_range",
            "translation_direction": (
                "independent_normalized_standard_gaussian_3d"
            ),
            "translation_distance": "uniform_in_stratum_range",
            "evaluate_complete_pool": True,
        },
        "physical_filters": {
            "finite_geometry": True,
            "hard_vdw_ratio": float(HARD_VDW_RATIO),
            "contact_pattern_nonempty": True,
            "complete_ligand_neighbor_coverage": True,
            "difficulty_definition": "stratum_bounds",
            "unique_transform_fingerprint": True,
        },
        "matching_rules": {
            "contact_mass_ratio_range": list(CONTACT_MASS_RATIO_RANGE),
            "centroid_limits_by_stratum": CENTROID_LIMITS,
            "native_relative_limits": MATCH_LIMITS,
            "distance": {
                "aggregation": "sum",
                "terms": [
                    {
                        "numerator": (
                            "abs(log(contact_mass_ratio))"
                        ),
                        "divisor": {
                            "formula": "log(1.15)",
                            "reference": float(
                                CONTACT_MASS_RATIO_RANGE[1]
                            ),
                        },
                    },
                    {
                        "numerator": (
                            "abs(delta_centroid_distance)"
                        ),
                        "divisor": {
                            "lookup": "centroid_limits[stratum]",
                            "values": CENTROID_LIMITS,
                        },
                    },
                    {
                        "numerator": (
                            "abs(delta_minimum_distance)"
                        ),
                        "divisor": float(
                            MATCH_LIMITS["minimum_distance"]
                        ),
                    },
                    {
                        "numerator": (
                            "abs(delta_max_overlap_severity)"
                        ),
                        "divisor": float(
                            MATCH_LIMITS[
                                "maximum_overlap_severity"
                            ]
                        ),
                    },
                    {
                        "numerator": (
                            "abs(delta_sum_overlap_per_ligand_atom)"
                        ),
                        "divisor": float(
                            MATCH_LIMITS[
                                "sum_overlap_per_ligand_atom"
                            ]
                        ),
                    },
                    {
                        "numerator": (
                            "abs(delta_count_below_0_9_per_ligand_atom)"
                        ),
                        "divisor": float(
                            MATCH_LIMITS[
                                "count_below_0_9_per_ligand_atom"
                            ]
                        ),
                    },
                    {
                        "numerator": (
                            "abs(delta_count_below_0_8_per_ligand_atom)"
                        ),
                        "divisor": float(
                            MATCH_LIMITS[
                                "count_below_0_8_per_ligand_atom"
                            ]
                        ),
                    },
                ],
            },
        },
        "native_eligibility": {
            "minimum_native_contact_mass": (
                "max(8.0,0.5*ligand_heavy_atom_count)"
            ),
            "native_hard_clash_ratio": float(HARD_VDW_RATIO),
        },
        "geometry_tolerances": {
            "absolute": float(GEOMETRY_ATOL),
            "relative": float(GEOMETRY_RTOL),
        },
        "continuous_contact_weights": {
            "distance_unit": "angstrom",
            "piecewise": [
                {"condition": "d<=5.0", "weight": "1.0"},
                {
                    "condition": "5.0<d<6.0",
                    "weight": (
                        "0.5*(cos(pi*(d-5.0))+1.0)"
                    ),
                },
                {"condition": "d>=6.0", "weight": "0.0"},
            ],
        },
        "overlap_statistics": {
            "distance_ratio": "distance/vdw_radius_sum",
            "count_below_0_9": {
                "comparison": "<",
                "threshold": 0.9,
            },
            "count_below_0_8": {
                "comparison": "<",
                "threshold": 0.8,
            },
            "hard_clash": {
                "comparison": "<",
                "threshold": float(HARD_VDW_RATIO),
            },
            "severity": "max(0.0,1.0-distance_ratio)",
        },
        "proposals_per_stratum": int(PROPOSALS_PER_STRATUM),
        "retained_per_stratum": int(RETAINED_PER_STRATUM),
        "shortlist_size": int(SHORTLIST_SIZE),
        "stratum_bounds": STRATUM_BOUNDS,
        "hard_vdw_ratio": float(HARD_VDW_RATIO),
        "contact_mass_ratio_range": list(CONTACT_MASS_RATIO_RANGE),
        "centroid_limits": CENTROID_LIMITS,
        "match_limits": MATCH_LIMITS,
        "rng_bit_generator": RNG_BIT_GENERATOR,
        "rng_purpose_namespace_templates": {
            "proposal": (
                "{seed}<NUL>{key}<NUL>{stratum}<NUL>proposal"
            ),
            "random_physical_retention": (
                "{seed}<NUL>{key}<NUL>{stratum}<NUL>random_physical"
            ),
            "matched_retention": (
                "{seed}<NUL>{key}<NUL>{stratum}<NUL>matched"
            ),
            "epoch_selection": (
                "{seed}<NUL>{key}<NUL>{stratum}<NUL>"
                "epoch_selection:{epoch}:{bank}"
            ),
        },
        "symmetry": {
            "source": "native_cache.ligand_automorphisms",
            "rdkit_molecule": "hydrogen_stripped_heavy_atom",
            "enumeration": {
                "uniquify": False,
                "use_chirality": True,
                "maximum_automorphisms": AUTOMORPHISM_CAP,
                "overflow_probe_max_matches": AUTOMORPHISM_CAP + 1,
                "overflow_policy": "reject_without_truncation",
            },
            "digest_dtype": "little_endian_signed_int64",
            "digest_order": "C",
            "rmsd": {
                "reduction": "minimum_over_cached_mappings",
                "rigid_alignment": "none",
                "coordinate_reordering": False,
            },
        },
        "shortlist_rule": {
            "bank": "matched",
            "sort": ["match_distance", "transform_fingerprint"],
            "take_prefix": int(SHORTLIST_SIZE),
        },
        "retention_rules": {
            "random_physical": {
                "eligible": "all_physical_candidates",
                "sort": ["transform_fingerprint"],
                "sample_without_replacement": int(RETAINED_PER_STRATUM),
            },
            "matched": {
                "eligible": "match_pass_true_in_shortlist",
                "sample_without_replacement": int(RETAINED_PER_STRATUM),
            },
            "bank_complete": (
                "exactly retained_per_stratum in every frozen stratum"
            ),
            "shared_proposal_pool": True,
        },
    }
    return _canonical_json_value(configuration)


def _provenance(
    manifest_path,
    native_path,
    native_keys,
    selected_keys,
    selected_record_sha256,
    selected_automorphism_sha256,
    max_samples,
    cohort,
    diagnostic_path,
    diagnostic,
    diagnostic_keys,
    seed,
):
    provenance = {
        "builder": BUILDER_NAME,
        "schema_version": DECOY_SCHEMA_VERSION,
        "cohort": str(cohort),
        "seed": int(seed),
        "max_samples": max_samples,
        "manifest": manifest_fingerprint(manifest_path),
        "selected_key_count": len(selected_keys),
        "selected_key_sha256": hash_ordered_keys(selected_keys),
        "native_lmdb": str(native_path),
        "native_key_count": len(native_keys),
        "native_key_sha256": hash_ordered_keys(native_keys),
        "selected_native_record_sha256": selected_record_sha256,
        "diagnostic_keys_path": str(diagnostic_path),
        "diagnostic_file_sha256": sha256_file(diagnostic_path),
        "diagnostic_ordered_key_sha256": hash_ordered_keys(diagnostic_keys),
        "diagnostic_manifest": diagnostic["manifest"],
        "generator_configuration": _generator_configuration(),
        "rdkit_runtime_version": str(rdBase.rdkitVersion),
        "selected_automorphism_sha256": selected_automorphism_sha256,
        **_native_metadata_summary(native_path),
    }
    return _canonical_json_value(provenance)


def _validate_integer_limit(value, name, allow_zero):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_args(args):
    if args.cohort not in ("diagnostic", "train"):
        raise ValueError("cohort must be diagnostic or train")
    _validate_integer_limit(args.seed, "seed", allow_zero=True)
    if args.max_samples is not None:
        _validate_integer_limit(
            args.max_samples, "max_samples", allow_zero=False
        )
    _validate_integer_limit(
        args.progress_every, "progress_every", allow_zero=True
    )
    _validate_integer_limit(
        args.max_error_preview, "max_error_preview", allow_zero=True
    )
    try:
        map_size_gb = float(args.map_size_gb)
    except (TypeError, ValueError) as error:
        raise ValueError("map_size_gb must be positive") from error
    if not math.isfinite(map_size_gb) or map_size_gb <= 0.0:
        raise ValueError("map_size_gb must be positive")


def _prepare_output(output_path, resume, provenance):
    report_path = output_path / REPORT_FILENAME
    if resume:
        if not report_path.exists():
            raise ValueError(
                "resume provenance is unavailable: decoy_report.json is missing"
            )
        previous = _read_json(report_path)
        if previous.get("provenance") != provenance:
            raise ValueError(
                "resume provenance does not match the complete v2 build "
                "configuration"
            )
        return previous
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"output LMDB is not empty: {output_path}; use --resume"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return None


def _preflight_existing_records(
    output_environment,
    native_environment,
    requested_keys,
    seed,
    previous_report,
):
    existing_keys, _, _ = _output_key_inventory(output_environment)
    requested_key_set = set(requested_keys)
    unexpected_keys = [
        key for key in existing_keys if key not in requested_key_set
    ]
    if unexpected_keys:
        raise ValueError(
            "existing v2 decoy cache contains unexpected keys: "
            + ", ".join(unexpected_keys[:5])
        )

    previous_complete = (
        previous_report is not None
        and previous_report.get("status") == "complete"
    )
    if previous_complete:
        ordered_existing_keys = previous_report.get("included_keys")
    else:
        existing_key_set = set(existing_keys)
        ordered_existing_keys = [
            key for key in requested_keys if key in existing_key_set
        ]
    actual_cache_integrity = _cache_integrity_summary(
        output_environment, ordered_existing_keys
    )
    if (
        previous_complete
        and previous_report.get("cache_integrity")
        != actual_cache_integrity
    ):
        raise ValueError(
            "completed cache integrity does not match actual records"
        )

    with output_environment.begin(
        write=False
    ) as output_transaction, native_environment.begin(
        write=False
    ) as native_transaction:
        for key in requested_keys:
            raw = output_transaction.get(key.encode("utf-8"))
            if raw is None:
                continue
            try:
                decoy_set = pickle.loads(raw)
                validate_decoy_set_v2(decoy_set)
                native_record = _load_item(
                    native_transaction, key, "native interface"
                )
                mappings = np.asarray(
                    native_record["ligand_automorphisms"]
                )
                expected_count = int(len(mappings))
                expected_automorphism_digest = automorphism_digest(
                    mappings
                )
                if decoy_set["key"] != key:
                    raise ValueError("cached key does not match LMDB key")
                if decoy_set["seed"] != int(seed):
                    raise ValueError("cached seed does not match build seed")
                if decoy_set["automorphism_count"] != expected_count:
                    raise ValueError(
                        "cached automorphism count does not match native"
                    )
                if (
                    decoy_set["automorphism_sha256"]
                    != expected_automorphism_digest
                ):
                    raise ValueError(
                        "cached automorphism digest does not match native"
                    )
            except lmdb.Error:
                raise
            except Exception as error:
                raise ValueError(
                    f"existing v2 decoy record is invalid: {key}"
                ) from error
    return actual_cache_integrity


def _finite_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _record_native_contact_mass(record):
    try:
        statistics = soft_contact_statistics(
            record["protein_coords"],
            record["ligand_coords"],
            record["ligand_coords"],
        )
    except (KeyError, TypeError, ValueError):
        return None
    return _finite_or_none(statistics.get("native_contact_mass"))


def _decoy_native_contact_mass(decoy_set):
    for bank_name in BANK_NAMES:
        for entry in decoy_set["banks"][bank_name]["decoys"]:
            value = _finite_or_none(
                entry.get("contact_stats", {}).get("native_contact_mass")
            )
            if value is not None:
                return value
    return None


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
        "ligand_size": int(ligand_size) if ligand_size is not None else None,
        "pocket_size": int(pocket_size) if pocket_size is not None else None,
        "native_contact_mass": _finite_or_none(native_contact_mass),
    }


def _numeric_summary(values):
    cleaned = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not cleaned:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    array = np.asarray(cleaned, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _selection_summary(rows):
    return {
        "sample_count": len(rows),
        "affinity": _numeric_summary([row.get("affinity") for row in rows]),
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


def _empty_distribution_values():
    return {
        bank: {
            stratum: {metric: [] for metric in DISTRIBUTION_METRICS}
            for stratum in STRATUM_ORDER
        }
        for bank in BANK_NAMES
    }


def _append_entry_metrics(values, bank, entry):
    stratum = entry["stratum"]
    contact = entry["contact_stats"]
    overlap = entry["overlap_stats"]
    match = entry["match_values"]
    row = values[bank][stratum]
    row["rmsd_sym"].append(entry["rmsd_sym"])
    row["soft_jaccard"].append(contact["soft_jaccard"])
    row["contact_mass_ratio"].append(contact["contact_mass_ratio"])
    for name in (
        "delta_centroid_distance",
        "delta_minimum_distance",
        "delta_max_overlap_severity",
        "delta_sum_overlap_per_ligand_atom",
        "delta_count_below_0_9_per_ligand_atom",
        "delta_count_below_0_8_per_ligand_atom",
    ):
        row[name].append(match[name])
    for name in (
        "minimum_distance_ratio",
        "max_overlap_severity",
        "sum_overlap_severity",
        "count_below_0_9",
        "count_below_0_8",
    ):
        row[name].append(overlap[name])
    row["match_distance"].append(entry["match_distance"])


def _distribution_summary(values):
    cleaned = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if len(cleaned) == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    quantiles = np.quantile(
        cleaned, [0.10, 0.25, 0.50, 0.75, 0.90]
    )
    return {
        "count": int(len(cleaned)),
        "min": float(np.min(cleaned)),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "max": float(np.max(cleaned)),
    }


def _summarize_distributions(values):
    return {
        bank: {
            stratum: {
                metric: _distribution_summary(metric_values)
                for metric, metric_values in values[bank][stratum].items()
            }
            for stratum in STRATUM_ORDER
        }
        for bank in BANK_NAMES
    }


def _empty_generation_counts():
    return {
        "random_physical": {
            stratum: {
                "proposed": 0,
                "physical_accepted": 0,
                "match_accepted": 0,
                "shortlisted": 0,
                "retained": 0,
            }
            for stratum in STRATUM_ORDER
        },
        "matched": {
            stratum: {
                "proposed": 0,
                "physical_accepted": 0,
                "match_accepted": 0,
                "shortlisted": 0,
                "retained": 0,
            }
            for stratum in STRATUM_ORDER
        },
    }


def _aggregate_generation(decoy_sets, excluded_rows):
    counts = _empty_generation_counts()
    rejection_by_stratum = {
        stratum: Counter() for stratum in STRATUM_ORDER
    }
    summaries = [
        (
            decoy_set.get("proposal_counts", {}),
            decoy_set.get("rejection_counts", {}),
        )
        for decoy_set in decoy_sets
    ]
    summaries.extend(
        (
            row.get("proposal_counts", {}),
            row.get("generation_rejection_counts_by_stratum", {}),
        )
        for row in excluded_rows
    )
    for proposal_counts, rejection_counts in summaries:
        for stratum in STRATUM_ORDER:
            current = proposal_counts.get(stratum)
            if current:
                random_counts = counts["random_physical"][stratum]
                matched_counts = counts["matched"][stratum]
                random_counts["proposed"] += int(current["proposed"])
                random_counts["physical_accepted"] += int(
                    current["physical_accepted"]
                )
                random_counts["match_accepted"] += int(
                    current["match_accepted"]
                )
                random_counts["shortlisted"] += int(
                    current["shortlisted"]
                )
                random_counts["retained"] += int(
                    current["random_physical_retained"]
                )
                matched_counts["proposed"] += int(current["proposed"])
                matched_counts["physical_accepted"] += int(
                    current["physical_accepted"]
                )
                matched_counts["match_accepted"] += int(
                    current["match_accepted"]
                )
                matched_counts["shortlisted"] += int(
                    current["shortlisted"]
                )
                matched_counts["retained"] += int(
                    current["matched_retained"]
                )
            rejection_by_stratum[stratum].update(
                rejection_counts.get(stratum, {})
            )

    flat_rejections = Counter()
    for counter in rejection_by_stratum.values():
        flat_rejections.update(counter)
    for row in excluded_rows:
        flat_rejections.update(row.get("generation_rejection_counts", {}))
    return (
        counts,
        {
            stratum: dict(sorted(counter.items()))
            for stratum, counter in rejection_by_stratum.items()
        },
        dict(sorted(flat_rejections.items())),
    )


def _preview_errors(rows, maximum):
    return [
        {
            "key": row["key"],
            "error_type": row["error_type"],
            "error": row["error"],
        }
        for row in rows[:maximum]
    ]


def _native_level_exclusion(row):
    return row.get("exclusion_reason") in {
        "native_hard_clash",
        "native_contact_mass",
    }


def _ordinary_error_row(sample, record, error):
    if isinstance(error, NativeContactMassExclusion):
        reason = "native_contact_mass"
        native_mass = error.native_contact_mass
        rejection_counts = {}
    elif isinstance(error, DecoyGenerationError):
        rejection_counts = dict(
            sorted(error.rejection_counts.items())
        )
        reason = (
            "native_hard_clash"
            if "native_hard_clash" in rejection_counts
            else "generation_error"
        )
        native_mass = _record_native_contact_mass(record)
    else:
        raise TypeError(
            "ordinary exclusions require an explicit generation outcome"
        )
    row = _sample_properties(sample, record, native_mass)
    row.update(
        {
            "error_type": type(error).__name__,
            "error": str(error),
            "exclusion_reason": reason,
            "bank_reasons": {
                bank: reason for bank in BANK_NAMES
            },
            "proposal_counts": {},
            "generation_rejection_counts_by_stratum": {},
            "generation_rejection_counts": rejection_counts,
        }
    )
    return row


def _incomplete_banks_row(sample, record, decoy_set):
    bank_reasons = {
        bank: decoy_set["banks"][bank]["failure_reason"]
        for bank in BANK_NAMES
    }
    native_mass = _decoy_native_contact_mass(decoy_set)
    if native_mass is None:
        native_mass = _record_native_contact_mass(record)
    row = _sample_properties(sample, record, native_mass)
    row.update(
        {
            "error_type": "IncompleteBanksError",
            "error": (
                "neither random_physical nor matched bank is complete"
            ),
            "exclusion_reason": "neither_bank_complete",
            "bank_reasons": bank_reasons,
            "proposal_counts": decoy_set["proposal_counts"],
            "generation_rejection_counts_by_stratum": (
                decoy_set["rejection_counts"]
            ),
            "generation_rejection_counts": {},
        }
    )
    return row


def _base_report(
    args,
    provenance,
    output_path,
    manifest_count,
    diagnostic_count,
    full_cohort_count,
    requested_keys,
):
    max_samples = args.max_samples
    return {
        "builder": BUILDER_NAME,
        "schema_version": DECOY_SCHEMA_VERSION,
        "status": "in_progress",
        "output_lmdb": str(output_path),
        "cohort": args.cohort,
        "seed": int(args.seed),
        "max_samples": max_samples,
        "manifest": provenance["manifest"],
        "native_lmdb": provenance["native_lmdb"],
        "diagnostic_keys_path": provenance["diagnostic_keys_path"],
        "provenance": provenance,
        "cohort_selection": {
            "full_cohort_count": full_cohort_count,
            "requested_count": len(requested_keys),
            "max_samples": max_samples,
            "max_samples_applied": max_samples is not None,
            "truncated_by_max_samples": (
                max_samples is not None
                and len(requested_keys) < full_cohort_count
            ),
        },
        "requested_keys": list(requested_keys),
        "counts": {
            "manifest_samples": manifest_count,
            "diagnostic_samples": diagnostic_count,
            "full_cohort_samples": full_cohort_count,
            "requested_samples": len(requested_keys),
        },
        "invocation_counts": {
            "processed": 0,
            "written": 0,
            "skipped_existing": 0,
            "errors": 0,
        },
        "cache_counts": {
            "cached_total": 0,
            "excluded_total": 0,
        },
        "included_keys": [],
        "excluded_samples": [],
        "errors_preview": [],
        "bank_completion": {},
        "eligible_bank_completion": {},
        "bank_exclusion_reasons": {
            bank: {} for bank in BANK_NAMES
        },
        "exclusion_reason_counts": {},
        "generation_counts": _empty_generation_counts(),
        "generation_rejection_counts_by_stratum": {
            stratum: {} for stratum in STRATUM_ORDER
        },
        "generation_rejection_counts": {},
        "distributions": _summarize_distributions(
            _empty_distribution_values()
        ),
        "selection_bias": {
            "included": _selection_summary([]),
            "excluded": _selection_summary([]),
        },
    }


def _fraction(count, denominator):
    return float(count / denominator) if denominator else None


def _completion_entry(count, denominator):
    return {
        "complete_count": int(count),
        "requested_fraction": _fraction(count, denominator),
    }


def _eligible_completion_entry(count, denominator):
    return {
        "complete_count": int(count),
        "eligible_fraction": _fraction(count, denominator),
    }


def _collect_final_report(
    report,
    output_environment,
    native_environment,
    requested_samples,
    excluded_by_key,
):
    sample_by_key = {
        str(sample["lmdb_key"]): sample for sample in requested_samples
    }
    requested_keys = [
        str(sample["lmdb_key"]) for sample in requested_samples
    ]
    cached_sets = []
    included_rows = []
    included_keys = []
    distribution_values = _empty_distribution_values()

    with output_environment.begin(
        write=False
    ) as output_transaction, native_environment.begin(
        write=False
    ) as native_transaction:
        for key in requested_keys:
            raw = output_transaction.get(key.encode("utf-8"))
            if raw is None:
                continue
            decoy_set = pickle.loads(raw)
            record = _load_item(
                native_transaction, key, "native interface"
            )
            cached_sets.append(decoy_set)
            included_keys.append(key)
            native_mass = _decoy_native_contact_mass(decoy_set)
            if native_mass is None:
                native_mass = _record_native_contact_mass(record)
            included_rows.append(
                _sample_properties(sample_by_key[key], record, native_mass)
            )
            for bank in BANK_NAMES:
                for entry in decoy_set["banks"][bank]["decoys"]:
                    _append_entry_metrics(
                        distribution_values, bank, entry
                    )

    included_key_set = set(included_keys)
    excluded_rows = [
        excluded_by_key[key]
        for key in requested_keys
        if key in excluded_by_key and key not in included_key_set
    ]

    complete_counts = {bank: 0 for bank in BANK_NAMES}
    intersection_count = 0
    bank_reasons = {bank: Counter() for bank in BANK_NAMES}
    for decoy_set in cached_sets:
        complete = {
            bank: bool(decoy_set["banks"][bank]["complete"])
            for bank in BANK_NAMES
        }
        for bank in BANK_NAMES:
            if complete[bank]:
                complete_counts[bank] += 1
            else:
                bank_reasons[bank][
                    decoy_set["banks"][bank]["failure_reason"]
                ] += 1
        intersection_count += int(all(complete.values()))
    for row in excluded_rows:
        for bank, reason in row.get("bank_reasons", {}).items():
            if reason is not None:
                bank_reasons[bank][reason] += 1

    requested_denominator = len(requested_keys)
    native_level_excluded = sum(
        _native_level_exclusion(row) for row in excluded_rows
    )
    eligible_denominator = requested_denominator - native_level_excluded

    generation_counts, rejection_by_stratum, rejections = (
        _aggregate_generation(cached_sets, excluded_rows)
    )
    exclusion_reasons = Counter(
        row.get("exclusion_reason", row["error_type"])
        for row in excluded_rows
    )

    report["cache_counts"] = {
        "cached_total": len(included_keys),
        "excluded_total": len(excluded_rows),
    }
    report["included_keys"] = included_keys
    report["cache_integrity"] = _cache_integrity_summary(
        output_environment, included_keys
    )
    report["excluded_samples"] = excluded_rows
    report["errors_preview"] = _preview_errors(
        excluded_rows, report["_max_error_preview"]
    )
    report["bank_completion"] = {
        "requested_denominator": requested_denominator,
        "random_physical": _completion_entry(
            complete_counts["random_physical"], requested_denominator
        ),
        "matched": _completion_entry(
            complete_counts["matched"], requested_denominator
        ),
        "intersection": _completion_entry(
            intersection_count, requested_denominator
        ),
    }
    report["eligible_bank_completion"] = {
        "eligible_denominator": eligible_denominator,
        "native_level_excluded_count": native_level_excluded,
        "random_physical": _eligible_completion_entry(
            complete_counts["random_physical"], eligible_denominator
        ),
        "matched": _eligible_completion_entry(
            complete_counts["matched"], eligible_denominator
        ),
        "intersection": _eligible_completion_entry(
            intersection_count, eligible_denominator
        ),
    }
    report["bank_exclusion_reasons"] = {
        bank: dict(sorted(reasons.items()))
        for bank, reasons in bank_reasons.items()
    }
    report["exclusion_reason_counts"] = dict(
        sorted(exclusion_reasons.items())
    )
    report["generation_counts"] = generation_counts
    report["generation_rejection_counts_by_stratum"] = (
        rejection_by_stratum
    )
    report["generation_rejection_counts"] = rejections
    report["distributions"] = _summarize_distributions(
        distribution_values
    )
    report["selection_bias"] = {
        "included": _selection_summary(included_rows),
        "excluded": _selection_summary(excluded_rows),
    }
    report["status"] = "complete"
    del report["_max_error_preview"]
    return _canonical_json_value(report)


def build_decoy_cache_v2(args):
    _validate_args(args)
    manifest_path, manifest, samples, manifest_keys = _load_manifest(
        args.manifest
    )
    if manifest.get("split") != "train":
        raise ValueError("v2 decoy cache requires manifest split=train")

    diagnostic_path, diagnostic, diagnostic_keys = _load_diagnostic(
        args.diagnostic_keys
    )
    current_manifest = manifest_fingerprint(manifest_path)
    if diagnostic.get("manifest") != current_manifest:
        raise ValueError(
            "diagnostic keys were built from a different manifest"
        )
    manifest_key_set = set(manifest_keys)
    unknown_diagnostic = [
        key for key in diagnostic_keys if key not in manifest_key_set
    ]
    if unknown_diagnostic:
        raise ValueError(
            "diagnostic keys are not present in the train manifest: "
            + ", ".join(unknown_diagnostic[:5])
        )

    sample_by_key = {
        str(sample["lmdb_key"]): sample for sample in samples
    }
    diagnostic_set = set(diagnostic_keys)
    if args.cohort == "diagnostic":
        full_cohort_samples = [
            sample_by_key[key] for key in diagnostic_keys
        ]
    else:
        full_cohort_samples = [
            sample
            for sample in samples
            if str(sample["lmdb_key"]) not in diagnostic_set
        ]
    requested_samples = (
        full_cohort_samples[: args.max_samples]
        if args.max_samples is not None
        else full_cohort_samples
    )
    requested_keys = [
        str(sample["lmdb_key"]) for sample in requested_samples
    ]

    native_path = Path(args.native_lmdb).resolve()
    native_environment = _open_readonly_lmdb(native_path)
    try:
        native_keys = _lmdb_keys(native_environment)
        missing_diagnostic = [
            key for key in diagnostic_keys if key not in set(native_keys)
        ]
        if missing_diagnostic:
            raise ValueError(
                "diagnostic keys are not present in the native LMDB: "
                + ", ".join(missing_diagnostic[:5])
            )
        selected_record_sha256 = _hash_ordered_lmdb_records(
            native_environment, requested_keys
        )
        selected_automorphism_sha256 = (
            _automorphism_digest_aggregate(
                native_environment, requested_keys
            )
        )
        provenance = _provenance(
            manifest_path=manifest_path,
            native_path=native_path,
            native_keys=native_keys,
            selected_keys=requested_keys,
            selected_record_sha256=selected_record_sha256,
            selected_automorphism_sha256=(
                selected_automorphism_sha256
            ),
            max_samples=args.max_samples,
            cohort=args.cohort,
            diagnostic_path=diagnostic_path,
            diagnostic=diagnostic,
            diagnostic_keys=diagnostic_keys,
            seed=args.seed,
        )

        output_path = Path(args.output_lmdb).resolve()
        previous = _prepare_output(
            output_path, bool(args.resume), provenance
        )
        if args.resume and (output_path / "data.mdb").is_file():
            preflight_environment = _open_readonly_lmdb(output_path)
            try:
                _preflight_existing_records(
                    preflight_environment,
                    native_environment,
                    requested_keys,
                    args.seed,
                    previous,
                )
            finally:
                preflight_environment.close()
        elif args.resume and previous.get("status") == "complete":
            raise ValueError(
                "completed cache integrity LMDB is missing"
            )

        report = _base_report(
            args=args,
            provenance=provenance,
            output_path=output_path,
            manifest_count=len(samples),
            diagnostic_count=len(diagnostic_keys),
            full_cohort_count=len(full_cohort_samples),
            requested_keys=requested_keys,
        )
        previous_excluded = (
            list(previous.get("excluded_samples", []))
            if previous is not None
            else []
        )
        report["excluded_samples"] = previous_excluded
        report["errors_preview"] = _preview_errors(
            previous_excluded, args.max_error_preview
        )
        report["_max_error_preview"] = int(args.max_error_preview)
        atomic_write_json(
            output_path / REPORT_FILENAME,
            _canonical_json_value(report),
        )

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
                    requested_samples, start=1
                ):
                    key = str(sample["lmdb_key"])
                    report["invocation_counts"]["processed"] += 1
                    with output_environment.begin(
                        write=False
                    ) as output_transaction:
                        exists = (
                            output_transaction.get(key.encode("utf-8"))
                            is not None
                        )
                    if exists:
                        report["invocation_counts"][
                            "skipped_existing"
                        ] += 1
                        excluded_by_key.pop(key, None)
                    else:
                        record = _load_item(
                            native_transaction,
                            key,
                            "native interface",
                        )
                        try:
                            decoy_set = generate_decoy_set_v2(
                                record, seed=args.seed
                            )
                        except NativeContactMassExclusion as error:
                            excluded_by_key[key] = _ordinary_error_row(
                                sample, record, error
                            )
                            report["invocation_counts"]["errors"] += 1
                        except DecoyGenerationError as error:
                            excluded_by_key[key] = _ordinary_error_row(
                                sample, record, error
                            )
                            report["invocation_counts"]["errors"] += 1
                        else:
                            validate_decoy_set_v2(decoy_set)
                            if not any(
                                decoy_set["banks"][bank]["complete"]
                                for bank in BANK_NAMES
                            ):
                                excluded_by_key[key] = (
                                    _incomplete_banks_row(
                                        sample, record, decoy_set
                                    )
                                )
                                report["invocation_counts"]["errors"] += 1
                            else:
                                serialized = pickle.dumps(
                                    decoy_set,
                                    protocol=pickle.HIGHEST_PROTOCOL,
                                )
                                _write_record(
                                    output_environment,
                                    key,
                                    serialized,
                                )
                                report["invocation_counts"]["written"] += 1
                                excluded_by_key.pop(key, None)

                    if (
                        args.progress_every
                        and index % args.progress_every == 0
                    ):
                        current_excluded = [
                            excluded_by_key[current_key]
                            for current_key in requested_keys
                            if current_key in excluded_by_key
                        ]
                        print(
                            json.dumps(
                                {
                                    "cohort": args.cohort,
                                    "progress": report[
                                        "invocation_counts"
                                    ],
                                    "errors_preview": _preview_errors(
                                        current_excluded,
                                        args.max_error_preview,
                                    ),
                                },
                                sort_keys=True,
                            )
                        )

            report = _collect_final_report(
                report,
                output_environment,
                native_environment,
                requested_samples,
                excluded_by_key,
            )
            report_path = output_path / REPORT_FILENAME
            atomic_write_json(report_path, report)
            completion_marker = {
                "builder": BUILDER_NAME,
                "schema_version": DECOY_SCHEMA_VERSION,
                "report_sha256": sha256_file(report_path),
                "provenance": report["provenance"],
                "provenance_sha256": _canonical_sha256(
                    report["provenance"]
                ),
                "cache_integrity": report["cache_integrity"],
                "cache_integrity_sha256": _canonical_sha256(
                    report["cache_integrity"]
                ),
            }
            with output_environment.begin(
                write=True
            ) as output_transaction:
                output_transaction.put(
                    METADATA_KEY,
                    pickle.dumps(
                        report, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
                output_transaction.put(
                    COMPLETION_KEY,
                    pickle.dumps(
                        completion_marker,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                )
        finally:
            output_environment.close()
    finally:
        native_environment.close()
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build provenance-locked rotation-dominant v2 decoy caches."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-lmdb", required=True)
    parser.add_argument("--diagnostic-keys", required=True)
    parser.add_argument(
        "--cohort",
        choices=("diagnostic", "train"),
        required=True,
    )
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-size-gb", type=float, default=16.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-error-preview", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser


def _terminal_summary(report):
    return {
        "status": report["status"],
        "cohort": report["cohort"],
        "requested_count": len(report["requested_keys"]),
        "invocation_counts": report["invocation_counts"],
        "cache_counts": report["cache_counts"],
        "bank_completion": report["bank_completion"],
        "exclusion_reason_counts": report["exclusion_reason_counts"],
        "errors_preview": report["errors_preview"],
    }


def main():
    report = build_decoy_cache_v2(build_arg_parser().parse_args())
    print(json.dumps(_terminal_summary(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
