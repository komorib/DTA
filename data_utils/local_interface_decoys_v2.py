from collections import Counter
import copy
import hashlib
import json

import numpy as np

from data_utils.local_interface_decoys import (
    CONTACT_STAT_FIELDS,
    OVERLAP_STAT_FIELDS,
    STRATUM_BOUNDS,
    TRANSFORM_FIELDS,
    DecoyGenerationError,
    GEOMETRY_ATOL,
    GEOMETRY_RTOL,
    apply_rigid_transform,
    soft_contact_statistics,
    symmetry_corrected_rmsd,
    vdw_overlap_statistics,
)


DECOY_SCHEMA_VERSION = 2
BANK_NAMES = ("random_physical", "matched")
STRATUM_ORDER = ("low", "medium", "high")
PROPOSALS_PER_STRATUM = 512
RETAINED_PER_STRATUM = 4
SHORTLIST_SIZE = 20
HARD_VDW_RATIO = 0.75
CONTACT_MASS_RATIO_RANGE = (0.80, 1.15)
RNG_BIT_GENERATOR = "PCG64"


class NativeContactMassExclusion(ValueError):
    def __init__(self, native_contact_mass, minimum_contact_mass):
        self.native_contact_mass = float(native_contact_mass)
        self.minimum_contact_mass = float(minimum_contact_mass)
        super().__init__(
            f"native contact mass {self.native_contact_mass:.6g} is below "
            f"{self.minimum_contact_mass:.6g}"
        )

PROPOSAL_RANGES = {
    "low": {
        "rotation_degrees": (5.0, 35.0),
        "translation": (0.0, 0.75),
    },
    "medium": {
        "rotation_degrees": (15.0, 80.0),
        "translation": (0.0, 1.25),
    },
    "high": {
        "rotation_degrees": (30.0, 150.0),
        "translation": (0.0, 2.0),
    },
}

CENTROID_LIMITS = {"low": 0.50, "medium": 1.00, "high": 1.50}
MATCH_LIMITS = {
    "minimum_distance": 0.50,
    "maximum_overlap_severity": 0.10,
    "sum_overlap_per_ligand_atom": 0.05,
    "count_below_0_9_per_ligand_atom": 0.12,
    "count_below_0_8_per_ligand_atom": 0.08,
}


def _as_coordinates(value, name):
    try:
        coordinates = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric coordinates") from error
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or len(coordinates) == 0
    ):
        raise ValueError(f"{name} must have shape [N,3] with N > 0")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return coordinates


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    return int(seed)


def _make_keyed_rng(seed, key, stratum, purpose):
    seed = _validate_seed(seed)
    digest = hashlib.sha256(
        f"{seed}\0{str(key)}\0{str(stratum)}\0{str(purpose)}".encode(
            "utf-8"
        )
    ).digest()
    entropy = int.from_bytes(digest, byteorder="big", signed=False)
    return np.random.Generator(np.random.PCG64(entropy))


def _random_unit_vector(rng):
    while True:
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            return vector / norm


def _rodrigues_rotation(axis, angle_radians):
    axis_cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    cosine = np.cos(angle_radians)
    return (
        np.eye(3) * cosine
        + (1.0 - cosine) * np.outer(axis, axis)
        + np.sin(angle_radians) * axis_cross
    )


def _transform_fingerprint(transform):
    canonical = json.dumps(
        transform, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def automorphism_digest(mappings):
    array = np.asarray(mappings, dtype="<i8", order="C")
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("automorphisms must be a nonempty [M,L] array")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def generate_proposal_pool(record, seed, stratum):
    if stratum not in PROPOSAL_RANGES:
        raise ValueError(
            f"stratum must be one of {', '.join(STRATUM_ORDER)}"
        )
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")

    coordinates = _as_coordinates(
        record.get("ligand_coords"), "ligand_coords"
    )
    if "key" not in record:
        raise ValueError("record must contain key")

    generator = _make_keyed_rng(seed, record["key"], stratum, "proposal")
    ranges = PROPOSAL_RANGES[stratum]
    center = np.mean(coordinates, axis=0)
    rows = []
    for _ in range(PROPOSALS_PER_STRATUM):
        axis = _random_unit_vector(generator)
        rotation_degrees = float(
            generator.uniform(*ranges["rotation_degrees"])
        )
        rotation = _rodrigues_rotation(
            axis, np.deg2rad(rotation_degrees)
        )

        translation_direction = _random_unit_vector(generator)
        translation_distance = float(
            generator.uniform(*ranges["translation"])
        )
        translation = translation_direction * translation_distance
        transform = {
            "center": center.astype(float).tolist(),
            "rotation_matrix": rotation.astype(float).tolist(),
            "translation": translation.astype(float).tolist(),
        }
        rows.append(
            {
                "transform": transform,
                "rotation_degrees": rotation_degrees,
                "translation_distance": translation_distance,
                "translation_direction": (
                    translation_direction.astype(float).tolist()
                ),
                "transform_fingerprint": _transform_fingerprint(transform),
            }
        )
    return rows


MATCH_VALUE_FIELDS = {
    "contact_mass_ratio",
    "delta_centroid_distance",
    "delta_minimum_distance",
    "delta_max_overlap_severity",
    "delta_sum_overlap_per_ligand_atom",
    "delta_count_below_0_9_per_ligand_atom",
    "delta_count_below_0_8_per_ligand_atom",
}
DECOY_ENTRY_FIELDS = {
    "transform",
    "transform_fingerprint",
    "rmsd_sym",
    "contact_stats",
    "overlap_stats",
    "stratum",
    "failure_reason",
    "match_values",
    "match_distance",
    "match_pass",
}
DECOY_SET_FIELDS = {
    "schema_version",
    "key",
    "seed",
    "proposal_count_per_stratum",
    "automorphism_count",
    "automorphism_sha256",
    "proposal_fingerprint_sha256",
    "banks",
    "proposal_counts",
    "rejection_counts",
}
BANK_FIELDS = {"complete", "failure_reason", "decoys"}
PROPOSAL_COUNT_FIELDS = {
    "proposed",
    "physical_accepted",
    "match_accepted",
    "shortlisted",
    "random_physical_retained",
    "matched_retained",
}


def _require_exact_fields(payload, expected_fields, label):
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a dictionary")
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(
            f"{label} must contain exactly its canonical schema fields "
            f"({', '.join(details)})"
        )


def _finite_number(payload, field, label):
    try:
        raw_value = payload[field]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} {field} must be numeric and finite") from error
    if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
        raw_value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} {field} must be numeric and finite")
    value = float(raw_value)
    if not np.isfinite(value):
        raise ValueError(f"{label} {field} must be numeric and finite")
    return value


def _nonnegative_integer(payload, field, label):
    try:
        value = payload[field]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{label} {field} must be a non-negative integer"
        ) from error
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{label} {field} must be a non-negative integer")
    return int(value)


def _validate_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256 hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{label} must be a SHA256 hexadecimal digest"
        ) from error
    return value


def _validate_automorphisms(automorphisms, atom_count):
    raw = np.asarray(automorphisms)
    if (
        raw.ndim != 2
        or raw.shape[1] != atom_count
        or len(raw) == 0
    ):
        raise ValueError("automorphisms must have shape [M,L]")
    try:
        mappings = raw.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("automorphisms must be atom permutations") from error
    if not np.array_equal(raw, mappings):
        raise ValueError("automorphisms must be atom permutations")
    expected = np.arange(atom_count, dtype=np.int64)
    if any(
        not np.array_equal(np.sort(mapping), expected)
        for mapping in mappings
    ):
        raise ValueError("automorphisms must be atom permutations")
    return mappings


def _prepare_native_geometry(record):
    if not isinstance(record, dict):
        raise ValueError("native record must be a dictionary")
    required = (
        "key",
        "ligand_coords",
        "protein_coords",
        "ligand_automorphisms",
        "ligand_atomic_numbers",
        "protein_element_symbol",
        "ligand_heavy_atom_count",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"native record is missing {missing[0]}")

    native = _as_coordinates(record["ligand_coords"], "ligand_coords")
    protein = _as_coordinates(record["protein_coords"], "protein_coords")
    ligand_count = record["ligand_heavy_atom_count"]
    if (
        isinstance(ligand_count, (bool, np.bool_))
        or not isinstance(ligand_count, (int, np.integer))
        or int(ligand_count) != len(native)
    ):
        raise ValueError(
            "ligand_heavy_atom_count is not aligned to ligand coordinates"
        )
    mappings = _validate_automorphisms(
        record["ligand_automorphisms"], len(native)
    )

    ligand_atomic_numbers = np.asarray(
        record["ligand_atomic_numbers"]
    ).reshape(-1)
    if len(ligand_atomic_numbers) != len(native):
        raise ValueError("ligand atomic numbers are not aligned")
    protein_symbols = np.asarray(
        record["protein_element_symbol"]
    ).reshape(-1)
    if len(protein_symbols) != len(protein):
        raise ValueError("protein element symbols are not aligned")

    return {
        "key": str(record["key"]),
        "native": native,
        "protein": protein,
        "automorphisms": mappings,
        "ligand_atomic_numbers": ligand_atomic_numbers,
        "protein_symbols": protein_symbols,
        "ligand_count": len(native),
    }


def centroid_distance(protein_coords, ligand_coords):
    protein = _as_coordinates(protein_coords, "protein_coords")
    ligand = _as_coordinates(ligand_coords, "ligand_coords")
    return float(np.linalg.norm(np.mean(protein, axis=0) - np.mean(ligand, axis=0)))


def minimum_pair_distance(protein_coords, ligand_coords):
    protein = _as_coordinates(protein_coords, "protein_coords")
    ligand = _as_coordinates(ligand_coords, "ligand_coords")
    distances = np.linalg.norm(
        protein[:, None, :] - ligand[None, :, :], axis=-1
    )
    return float(np.min(distances))


def _candidate_match_values_from_geometry(
    native_geometry,
    transformed,
    contact_stats,
    overlap_stats,
    native_overlap=None,
    native_centroid=None,
    native_minimum=None,
):
    ligand_count = native_geometry["ligand_count"]
    native = native_geometry["native"]
    protein = native_geometry["protein"]
    if native_overlap is None:
        native_overlap = vdw_overlap_statistics(
            protein,
            native_geometry["protein_symbols"],
            native,
            native_geometry["ligand_atomic_numbers"],
        )
    if native_centroid is None:
        native_centroid = centroid_distance(protein, native)
    if native_minimum is None:
        native_minimum = minimum_pair_distance(protein, native)
    return {
        "contact_mass_ratio": float(contact_stats["contact_mass_ratio"]),
        "delta_centroid_distance": float(
            centroid_distance(protein, transformed) - native_centroid
        ),
        "delta_minimum_distance": float(
            minimum_pair_distance(protein, transformed) - native_minimum
        ),
        "delta_max_overlap_severity": float(
            overlap_stats["max_overlap_severity"]
            - native_overlap["max_overlap_severity"]
        ),
        "delta_sum_overlap_per_ligand_atom": float(
            (
                overlap_stats["sum_overlap_severity"]
                - native_overlap["sum_overlap_severity"]
            )
            / ligand_count
        ),
        "delta_count_below_0_9_per_ligand_atom": float(
            (
                overlap_stats["count_below_0_9"]
                - native_overlap["count_below_0_9"]
            )
            / ligand_count
        ),
        "delta_count_below_0_8_per_ligand_atom": float(
            (
                overlap_stats["count_below_0_8"]
                - native_overlap["count_below_0_8"]
            )
            / ligand_count
        ),
    }


def candidate_match_values(record, transformed, contact_stats, overlap_stats):
    native_geometry = _prepare_native_geometry(record)
    transformed_coords = _as_coordinates(transformed, "transformed")
    if transformed_coords.shape != native_geometry["native"].shape:
        raise ValueError(
            "transformed ligand coordinates must align to native ligand atoms"
        )
    return _candidate_match_values_from_geometry(
        native_geometry,
        transformed_coords,
        contact_stats,
        overlap_stats,
    )


def _in_stratum(value, lower, upper, upper_inclusive):
    if value < lower:
        return False
    if upper_inclusive:
        return value <= upper
    return value < upper


def _physical_metrics_are_finite(rmsd_sym, contact_stats, overlap_stats):
    try:
        values = [
            float(rmsd_sym),
            float(contact_stats["native_contact_mass"]),
            float(contact_stats["decoy_contact_mass"]),
            float(contact_stats["contact_mass_ratio"]),
            float(contact_stats["soft_jaccard"]),
            float(contact_stats["native_neighbor_count"]),
            float(contact_stats["decoy_neighbor_count"]),
            float(overlap_stats["minimum_distance_ratio"]),
            float(overlap_stats["max_overlap_severity"]),
            float(overlap_stats["sum_overlap_severity"]),
            float(overlap_stats["count_below_0_9"]),
            float(overlap_stats["count_below_0_8"]),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def classify_physical_candidate(
    rmsd_sym,
    contact_stats,
    overlap_stats,
    target_stratum,
    transform_fingerprint,
    seen_fingerprints=None,
):
    if target_stratum not in STRATUM_ORDER:
        raise ValueError(
            f"target_stratum must be one of {', '.join(STRATUM_ORDER)}"
        )
    if not _physical_metrics_are_finite(
        rmsd_sym, contact_stats, overlap_stats
    ):
        reason = "nonfinite_metrics"
    elif bool(overlap_stats.get("has_hard_clash")):
        reason = "vdw_hard_clash"
    elif not bool(contact_stats.get("contact_pattern_nonempty")):
        reason = "empty_contact_pattern"
    elif not bool(contact_stats.get("coverage_retained")):
        reason = "ligand_neighbor_coverage"
    else:
        rmsd = float(rmsd_sym)
        jaccard = float(contact_stats["soft_jaccard"])
        geometry_stratum = None
        for stratum in STRATUM_ORDER:
            bounds = STRATUM_BOUNDS[stratum]
            if _in_stratum(
                rmsd,
                bounds["rmsd_min"],
                bounds["rmsd_max"],
                bounds["rmsd_max_inclusive"],
            ) and _in_stratum(
                jaccard,
                bounds["jaccard_min"],
                bounds["jaccard_max"],
                bounds["jaccard_max_inclusive"],
            ):
                geometry_stratum = stratum
                break
        if geometry_stratum is None:
            reason = "outside_frozen_strata"
        elif geometry_stratum != target_stratum:
            reason = "wrong_stratum"
        elif (
            seen_fingerprints is not None
            and transform_fingerprint in seen_fingerprints
        ):
            reason = "duplicate_transform"
        else:
            return {
                "accepted": True,
                "stratum": target_stratum,
                "failure_reason": None,
            }
    return {"accepted": False, "stratum": None, "failure_reason": reason}


def _match_values_as_floats(match_values):
    _require_exact_fields(
        match_values, MATCH_VALUE_FIELDS, "candidate match values"
    )
    values = {
        field: _finite_number(
            match_values, field, "candidate match values"
        )
        for field in MATCH_VALUE_FIELDS
    }
    if values["contact_mass_ratio"] <= 0.0:
        raise ValueError("candidate match contact_mass_ratio must be positive")
    return values


def normalized_match_distance(match_values, stratum):
    if stratum not in STRATUM_ORDER:
        raise ValueError(f"stratum must be one of {', '.join(STRATUM_ORDER)}")
    values = _match_values_as_floats(match_values)
    return float(
        abs(np.log(values["contact_mass_ratio"])) / np.log(1.15)
        + abs(values["delta_centroid_distance"]) / CENTROID_LIMITS[stratum]
        + abs(values["delta_minimum_distance"])
        / MATCH_LIMITS["minimum_distance"]
        + abs(values["delta_max_overlap_severity"])
        / MATCH_LIMITS["maximum_overlap_severity"]
        + abs(values["delta_sum_overlap_per_ligand_atom"])
        / MATCH_LIMITS["sum_overlap_per_ligand_atom"]
        + abs(values["delta_count_below_0_9_per_ligand_atom"])
        / MATCH_LIMITS["count_below_0_9_per_ligand_atom"]
        + abs(values["delta_count_below_0_8_per_ligand_atom"])
        / MATCH_LIMITS["count_below_0_8_per_ligand_atom"]
    )


def classify_matched_candidate(match_values, stratum):
    try:
        values = _match_values_as_floats(match_values)
        distance = normalized_match_distance(values, stratum)
    except ValueError:
        return {
            "match_pass": False,
            "failure_reason": "match_nonfinite_metrics",
            "match_distance": float("inf"),
        }

    ratio = values["contact_mass_ratio"]
    if not CONTACT_MASS_RATIO_RANGE[0] <= ratio <= CONTACT_MASS_RATIO_RANGE[1]:
        reason = "match_contact_mass_ratio"
    elif abs(values["delta_centroid_distance"]) > CENTROID_LIMITS[stratum]:
        reason = "match_delta_centroid_distance"
    elif (
        abs(values["delta_minimum_distance"])
        > MATCH_LIMITS["minimum_distance"]
    ):
        reason = "match_delta_minimum_distance"
    elif (
        abs(values["delta_max_overlap_severity"])
        > MATCH_LIMITS["maximum_overlap_severity"]
    ):
        reason = "match_delta_max_overlap_severity"
    elif (
        abs(values["delta_sum_overlap_per_ligand_atom"])
        > MATCH_LIMITS["sum_overlap_per_ligand_atom"]
    ):
        reason = "match_delta_sum_overlap_per_ligand_atom"
    elif (
        abs(values["delta_count_below_0_9_per_ligand_atom"])
        > MATCH_LIMITS["count_below_0_9_per_ligand_atom"]
    ):
        reason = "match_delta_count_below_0_9_per_ligand_atom"
    elif (
        abs(values["delta_count_below_0_8_per_ligand_atom"])
        > MATCH_LIMITS["count_below_0_8_per_ligand_atom"]
    ):
        reason = "match_delta_count_below_0_8_per_ligand_atom"
    else:
        return {
            "match_pass": True,
            "failure_reason": None,
            "match_distance": distance,
        }
    return {
        "match_pass": False,
        "failure_reason": reason,
        "match_distance": distance,
    }


def _sample_retained(candidates, seed, key, stratum, bank):
    count = min(RETAINED_PER_STRATUM, len(candidates))
    if count == 0:
        return []
    generator = _make_keyed_rng(seed, key, stratum, bank)
    indices = generator.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


def retain_random_physical(candidates, seed, key, stratum):
    ordered = sorted(
        candidates, key=lambda row: str(row["transform_fingerprint"])
    )
    return _sample_retained(
        ordered, seed, key, stratum, "random_physical"
    )


def retain_matched(candidates, seed, key, stratum):
    matched = [row for row in candidates if row.get("match_pass") is True]
    ordered = sorted(
        matched,
        key=lambda row: (
            float(row["match_distance"]),
            str(row["transform_fingerprint"]),
        ),
    )
    shortlist = ordered[: min(SHORTLIST_SIZE, len(ordered))]
    return _sample_retained(shortlist, seed, key, stratum, "matched")


def proposal_fingerprint_digest(proposal_pools):
    if not isinstance(proposal_pools, dict) or set(proposal_pools) != set(
        STRATUM_ORDER
    ):
        raise ValueError("proposal pools must contain every frozen stratum")
    payload = []
    for stratum in STRATUM_ORDER:
        rows = proposal_pools[stratum]
        if not isinstance(rows, (list, tuple)):
            raise ValueError("proposal pool rows must be a sequence")
        fingerprints = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("proposal rows must be dictionaries")
            fingerprint = row.get("transform_fingerprint")
            _validate_sha256(fingerprint, "proposal fingerprint")
            fingerprints.append(fingerprint)
        payload.append([stratum, fingerprints])
    canonical = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _evaluate_proposal(
    native_geometry, proposal, target_stratum, seen_fingerprints
):
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be a dictionary")
    transform = proposal.get("transform")
    fingerprint = proposal.get("transform_fingerprint")
    _validate_sha256(fingerprint, "proposal fingerprint")
    if fingerprint != _transform_fingerprint(transform):
        raise ValueError(
            "proposal fingerprint does not match rigid transform"
        )

    transformed = apply_rigid_transform(
        native_geometry["native"], transform
    )
    rmsd = symmetry_corrected_rmsd(
        native_geometry["native"],
        transformed,
        native_geometry["automorphisms"],
    )
    contact_stats = soft_contact_statistics(
        native_geometry["protein"],
        native_geometry["native"],
        transformed,
    )
    overlap_stats = vdw_overlap_statistics(
        native_geometry["protein"],
        native_geometry["protein_symbols"],
        transformed,
        native_geometry["ligand_atomic_numbers"],
    )
    physical = classify_physical_candidate(
        rmsd,
        contact_stats,
        overlap_stats,
        target_stratum=target_stratum,
        transform_fingerprint=fingerprint,
        seen_fingerprints=seen_fingerprints,
    )
    if not physical["accepted"]:
        return None, physical["failure_reason"], None

    seen_fingerprints.add(fingerprint)
    match_values = _candidate_match_values_from_geometry(
        native_geometry,
        transformed,
        contact_stats,
        overlap_stats,
        native_overlap=native_geometry["native_overlap"],
        native_centroid=native_geometry["native_centroid"],
        native_minimum=native_geometry["native_minimum"],
    )
    matched = classify_matched_candidate(match_values, target_stratum)
    candidate = {
        "transform": transform,
        "transform_fingerprint": fingerprint,
        "rmsd_sym": float(rmsd),
        "contact_stats": contact_stats,
        "overlap_stats": overlap_stats,
        "stratum": target_stratum,
        "failure_reason": None,
        "match_values": match_values,
        "match_distance": float(matched["match_distance"]),
        "match_pass": bool(matched["match_pass"]),
    }
    return candidate, None, matched["failure_reason"]


def _prepare_eligible_native_geometry(
    record, native_hard_clash_as_generation_error
):
    native_geometry = _prepare_native_geometry(record)
    native_overlap = vdw_overlap_statistics(
        native_geometry["protein"],
        native_geometry["protein_symbols"],
        native_geometry["native"],
        native_geometry["ligand_atomic_numbers"],
    )
    if native_overlap["has_hard_clash"]:
        message = (
            "native_hard_clash: native pose violates "
            "HARD_VDW_RATIO=0.75"
        )
        if native_hard_clash_as_generation_error:
            raise DecoyGenerationError(
                message, {"native_hard_clash": 1}
            )
        raise ValueError(message)

    native_contact_stats = soft_contact_statistics(
        native_geometry["protein"],
        native_geometry["native"],
        native_geometry["native"],
    )
    native_mass = float(native_contact_stats["native_contact_mass"])
    minimum_native_mass = max(
        8.0, 0.5 * native_geometry["ligand_count"]
    )
    if native_mass < minimum_native_mass:
        raise NativeContactMassExclusion(
            native_mass, minimum_native_mass
        )

    native_geometry["native_overlap"] = native_overlap
    native_geometry["native_centroid"] = centroid_distance(
        native_geometry["protein"], native_geometry["native"]
    )
    native_geometry["native_minimum"] = minimum_pair_distance(
        native_geometry["protein"], native_geometry["native"]
    )
    return native_geometry


def _bank_failure_reason(bank_name, retained_by_stratum):
    missing = [
        stratum
        for stratum in STRATUM_ORDER
        if len(retained_by_stratum[stratum]) != RETAINED_PER_STRATUM
    ]
    if not missing:
        return None
    return f"insufficient_{bank_name}_candidates:{','.join(missing)}"


def _evaluate_and_retain_proposal_pools(
    native_geometry, proposal_pools, seed
):
    seed = _validate_seed(seed)
    if not isinstance(proposal_pools, dict) or set(
        proposal_pools
    ) != set(STRATUM_ORDER):
        raise ValueError(
            "proposal pools must contain every frozen stratum"
        )
    for stratum in STRATUM_ORDER:
        rows = proposal_pools[stratum]
        if (
            not isinstance(rows, (list, tuple))
            or len(rows) != PROPOSALS_PER_STRATUM
        ):
            raise ValueError(
                f"{stratum} proposal pool must contain exactly "
                f"{PROPOSALS_PER_STRATUM} proposals"
            )

    retained_by_bank = {
        bank: {stratum: [] for stratum in STRATUM_ORDER}
        for bank in BANK_NAMES
    }
    proposal_counts = {}
    rejection_counts = {}
    seen_fingerprints = set()

    for stratum in STRATUM_ORDER:
        physical_candidates = []
        rejections = Counter()
        for proposal in proposal_pools[stratum]:
            candidate, physical_reason, match_reason = _evaluate_proposal(
                native_geometry,
                proposal,
                stratum,
                seen_fingerprints,
            )
            if candidate is None:
                rejections[physical_reason] += 1
                continue
            physical_candidates.append(candidate)
            if match_reason is not None:
                rejections[match_reason] += 1

        random_rows = retain_random_physical(
            physical_candidates,
            seed=seed,
            key=native_geometry["key"],
            stratum=stratum,
        )
        matched_rows = retain_matched(
            physical_candidates,
            seed=seed,
            key=native_geometry["key"],
            stratum=stratum,
        )
        retained_by_bank["random_physical"][stratum] = copy.deepcopy(
            random_rows
        )
        retained_by_bank["matched"][stratum] = copy.deepcopy(
            matched_rows
        )
        match_count = sum(
            1 for row in physical_candidates if row["match_pass"]
        )
        proposal_counts[stratum] = {
            "proposed": PROPOSALS_PER_STRATUM,
            "physical_accepted": len(physical_candidates),
            "match_accepted": match_count,
            "shortlisted": min(SHORTLIST_SIZE, match_count),
            "random_physical_retained": len(random_rows),
            "matched_retained": len(matched_rows),
        }
        rejection_counts[stratum] = dict(sorted(rejections.items()))

    banks = {}
    for bank_name in BANK_NAMES:
        failure_reason = _bank_failure_reason(
            bank_name, retained_by_bank[bank_name]
        )
        banks[bank_name] = {
            "complete": failure_reason is None,
            "failure_reason": failure_reason,
            "decoys": [
                row
                for stratum in STRATUM_ORDER
                for row in retained_by_bank[bank_name][stratum]
            ],
        }
    return {
        "banks": banks,
        "proposal_counts": proposal_counts,
        "rejection_counts": rejection_counts,
    }


def generate_decoy_set_v2(record, seed):
    seed = _validate_seed(seed)
    native_geometry = _prepare_eligible_native_geometry(
        record, native_hard_clash_as_generation_error=True
    )

    proposal_pools = {
        stratum: generate_proposal_pool(
            record, seed=seed, stratum=stratum
        )
        for stratum in STRATUM_ORDER
    }
    summary = _evaluate_and_retain_proposal_pools(
        native_geometry, proposal_pools, seed
    )
    decoy_set = {
        "schema_version": DECOY_SCHEMA_VERSION,
        "key": native_geometry["key"],
        "seed": seed,
        "proposal_count_per_stratum": PROPOSALS_PER_STRATUM,
        "automorphism_count": int(len(native_geometry["automorphisms"])),
        "automorphism_sha256": automorphism_digest(
            native_geometry["automorphisms"]
        ),
        "proposal_fingerprint_sha256": proposal_fingerprint_digest(
            proposal_pools
        ),
        "banks": summary["banks"],
        "proposal_counts": summary["proposal_counts"],
        "rejection_counts": summary["rejection_counts"],
    }
    return validate_decoy_set_v2(decoy_set)


def _validate_transform(transform):
    _require_exact_fields(transform, TRANSFORM_FIELDS, "transform")
    center = transform["center"]
    rotation = transform["rotation_matrix"]
    translation = transform["translation"]
    if (
        not isinstance(center, list)
        or len(center) != 3
        or not isinstance(translation, list)
        or len(translation) != 3
        or not isinstance(rotation, list)
        or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
    ):
        raise ValueError("transform must use canonical list metadata")
    apply_rigid_transform(np.asarray([[0.0, 0.0, 0.0]]), transform)


def _validate_cached_statistics(contact_stats, overlap_stats):
    _require_exact_fields(
        contact_stats, CONTACT_STAT_FIELDS, "contact statistics"
    )
    _require_exact_fields(
        overlap_stats, OVERLAP_STAT_FIELDS, "overlap statistics"
    )
    native_mass = _finite_number(
        contact_stats, "native_contact_mass", "contact"
    )
    decoy_mass = _finite_number(
        contact_stats, "decoy_contact_mass", "contact"
    )
    ratio = _finite_number(
        contact_stats, "contact_mass_ratio", "contact"
    )
    jaccard = _finite_number(contact_stats, "soft_jaccard", "contact")
    if native_mass <= 0.0 or decoy_mass <= 0.0:
        raise ValueError("contact masses must be positive")
    if not np.isclose(
        ratio,
        decoy_mass / native_mass,
        rtol=GEOMETRY_RTOL,
        atol=GEOMETRY_ATOL,
    ):
        raise ValueError("contact mass ratio does not match cached masses")
    if not 0.0 <= jaccard <= 1.0:
        raise ValueError("soft Jaccard must be in [0,1]")
    _nonnegative_integer(
        contact_stats, "native_neighbor_count", "contact"
    )
    _nonnegative_integer(
        contact_stats, "decoy_neighbor_count", "contact"
    )
    for field in ("coverage_retained", "contact_pattern_nonempty"):
        if not isinstance(contact_stats[field], (bool, np.bool_)):
            raise ValueError(f"contact {field} must be boolean")

    minimum_ratio = _finite_number(
        overlap_stats, "minimum_distance_ratio", "overlap"
    )
    maximum_severity = _finite_number(
        overlap_stats, "max_overlap_severity", "overlap"
    )
    summed_severity = _finite_number(
        overlap_stats, "sum_overlap_severity", "overlap"
    )
    if minimum_ratio < 0.0:
        raise ValueError("minimum distance ratio must be non-negative")
    if not 0.0 <= maximum_severity <= 1.0:
        raise ValueError("maximum overlap severity must be in [0,1]")
    if summed_severity < maximum_severity:
        raise ValueError("summed overlap severity must include the maximum")
    if not np.isclose(
        maximum_severity,
        max(0.0, 1.0 - minimum_ratio),
        rtol=GEOMETRY_RTOL,
        atol=GEOMETRY_ATOL,
    ):
        raise ValueError("maximum overlap severity is inconsistent")
    below_nine = _nonnegative_integer(
        overlap_stats, "count_below_0_9", "overlap"
    )
    below_eight = _nonnegative_integer(
        overlap_stats, "count_below_0_8", "overlap"
    )
    if below_eight > below_nine:
        raise ValueError("vdW threshold counts are inconsistent")
    hard_clash = overlap_stats["has_hard_clash"]
    if not isinstance(hard_clash, (bool, np.bool_)):
        raise ValueError("vdW hard-clash flag must be boolean")
    if bool(hard_clash) != bool(minimum_ratio < HARD_VDW_RATIO):
        raise ValueError("vdW hard-clash flag is inconsistent")


def _geometry_values_match(cached, recomputed):
    if isinstance(recomputed, (bool, np.bool_)):
        return isinstance(cached, (bool, np.bool_)) and bool(cached) == bool(
            recomputed
        )
    if isinstance(recomputed, (int, np.integer)):
        return (
            isinstance(cached, (int, np.integer))
            and not isinstance(cached, (bool, np.bool_))
            and int(cached) == int(recomputed)
        )
    try:
        cached_value = float(cached)
        recomputed_value = float(recomputed)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isclose(
            cached_value,
            recomputed_value,
            rtol=GEOMETRY_RTOL,
            atol=GEOMETRY_ATOL,
        )
    )


def _verify_geometry_statistics(cached, recomputed, label):
    for field, value in recomputed.items():
        if not _geometry_values_match(cached[field], value):
            raise ValueError(
                f"decoy geometry {label} {field} does not match "
                "native geometry"
            )


def _validate_proposal_counts(proposal_counts):
    _require_exact_fields(
        proposal_counts, set(STRATUM_ORDER), "proposal counts"
    )
    validated = {}
    for stratum in STRATUM_ORDER:
        counts = proposal_counts[stratum]
        _require_exact_fields(
            counts, PROPOSAL_COUNT_FIELDS, f"{stratum} proposal counts"
        )
        values = {
            field: _nonnegative_integer(
                counts, field, f"{stratum} proposal counts"
            )
            for field in PROPOSAL_COUNT_FIELDS
        }
        proposed = values["proposed"]
        physical = values["physical_accepted"]
        matched = values["match_accepted"]
        if proposed != PROPOSALS_PER_STRATUM:
            raise ValueError(
                "proposal count must equal proposal_count_per_stratum"
            )
        if physical > proposed:
            raise ValueError(
                "physical-accepted count exceeds proposed count"
            )
        if matched > physical:
            raise ValueError(
                "match-accepted count exceeds physical-accepted count"
            )
        if values["shortlisted"] != min(SHORTLIST_SIZE, matched):
            raise ValueError("shortlisted count is inconsistent")
        if values["random_physical_retained"] != min(
            RETAINED_PER_STRATUM, physical
        ):
            raise ValueError(
                "random-physical retained proposal count is inconsistent"
            )
        if values["matched_retained"] != min(
            RETAINED_PER_STRATUM, matched
        ):
            raise ValueError(
                "matched retained proposal count is inconsistent"
            )
        validated[stratum] = values
    return validated


def _validate_rejection_counts(rejection_counts, proposal_counts):
    _require_exact_fields(
        rejection_counts, set(STRATUM_ORDER), "rejection counts"
    )
    for stratum in STRATUM_ORDER:
        counts = rejection_counts[stratum]
        if not isinstance(counts, dict):
            raise ValueError(
                f"{stratum} rejection counts must be a dictionary"
            )
        physical_rejection_total = 0
        match_rejection_total = 0
        for reason, value in counts.items():
            if not isinstance(reason, str) or not reason:
                raise ValueError("rejection reason must be a nonempty string")
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(
                    "rejection counts must be non-negative integers"
                )
            if reason.startswith("match_"):
                match_rejection_total += int(value)
            else:
                physical_rejection_total += int(value)

        proposed = proposal_counts[stratum]["proposed"]
        physical = proposal_counts[stratum]["physical_accepted"]
        matched = proposal_counts[stratum]["match_accepted"]
        if physical_rejection_total != proposed - physical:
            raise ValueError(
                f"{stratum} physical rejection total is inconsistent"
            )
        if match_rejection_total != physical - matched:
            raise ValueError(
                f"{stratum} match rejection total is inconsistent"
            )
        if (
            physical_rejection_total + match_rejection_total
            != proposed - matched
        ):
            raise ValueError(
                f"{stratum} total rejection count is inconsistent"
            )


def _validate_decoy_entry(entry, bank_name, seen_fingerprints):
    _require_exact_fields(entry, DECOY_ENTRY_FIELDS, "decoy entry")
    if entry["failure_reason"] is not None:
        raise ValueError("retained decoy failure_reason must be None")
    if entry["stratum"] not in STRATUM_ORDER:
        raise ValueError("decoy stratum is invalid")

    transform = entry["transform"]
    _validate_transform(transform)
    fingerprint = _validate_sha256(
        entry["transform_fingerprint"], "transform fingerprint"
    )
    if fingerprint != _transform_fingerprint(transform):
        raise ValueError(
            "transform fingerprint does not match rigid transform"
        )
    if fingerprint in seen_fingerprints:
        raise ValueError(
            f"{bank_name} bank contains a duplicate transform fingerprint"
        )

    _validate_cached_statistics(
        entry["contact_stats"], entry["overlap_stats"]
    )
    rmsd = _finite_number(entry, "rmsd_sym", "decoy")
    physical = classify_physical_candidate(
        rmsd,
        entry["contact_stats"],
        entry["overlap_stats"],
        target_stratum=entry["stratum"],
        transform_fingerprint=fingerprint,
        seen_fingerprints=seen_fingerprints,
    )
    if not physical["accepted"]:
        raise ValueError(
            "decoy violates frozen physical geometry: "
            + str(physical["failure_reason"])
        )
    seen_fingerprints.add(fingerprint)

    values = _match_values_as_floats(entry["match_values"])
    if not _geometry_values_match(
        values["contact_mass_ratio"],
        entry["contact_stats"]["contact_mass_ratio"],
    ):
        raise ValueError(
            "candidate match contact mass ratio disagrees with contact statistics"
        )
    stored_distance = _finite_number(
        entry, "match_distance", "decoy match"
    )
    matched = classify_matched_candidate(values, entry["stratum"])
    if not _geometry_values_match(
        stored_distance, matched["match_distance"]
    ):
        raise ValueError("decoy match distance is inconsistent")
    if not isinstance(entry["match_pass"], (bool, np.bool_)):
        raise ValueError("decoy match_pass must be boolean")
    if bool(entry["match_pass"]) != bool(matched["match_pass"]):
        raise ValueError("decoy match_pass is inconsistent")
    if bank_name == "matched" and not matched["match_pass"]:
        raise ValueError("matched bank contains a match-filter rejection")
    return fingerprint


def validate_decoy_set_v2(decoy_set, native_record=None):
    if not isinstance(decoy_set, dict):
        raise ValueError("decoy set must be a dictionary")
    if decoy_set.get("schema_version") != DECOY_SCHEMA_VERSION:
        raise ValueError("decoy schema version 2 is required")
    _require_exact_fields(decoy_set, DECOY_SET_FIELDS, "decoy set")
    if not isinstance(decoy_set["key"], str):
        raise ValueError("decoy-set key must be a string")
    seed = _validate_seed(decoy_set["seed"])
    proposal_count = _nonnegative_integer(
        decoy_set,
        "proposal_count_per_stratum",
        "decoy set",
    )
    if proposal_count != PROPOSALS_PER_STRATUM:
        raise ValueError(
            f"proposal_count_per_stratum must be {PROPOSALS_PER_STRATUM}"
        )
    automorphism_count = _nonnegative_integer(
        decoy_set, "automorphism_count", "decoy set"
    )
    if automorphism_count == 0:
        raise ValueError("automorphism_count must be positive")
    _validate_sha256(
        decoy_set["automorphism_sha256"], "automorphism digest"
    )
    _validate_sha256(
        decoy_set["proposal_fingerprint_sha256"],
        "proposal fingerprint digest",
    )

    proposal_counts = _validate_proposal_counts(
        decoy_set["proposal_counts"]
    )
    _validate_rejection_counts(
        decoy_set["rejection_counts"], proposal_counts
    )
    banks = decoy_set["banks"]
    _require_exact_fields(banks, set(BANK_NAMES), "decoy banks")

    all_entries = []
    actual_counts = {}
    for bank_name in BANK_NAMES:
        bank = banks[bank_name]
        _require_exact_fields(
            bank, BANK_FIELDS, f"{bank_name} bank"
        )
        if not isinstance(bank["complete"], (bool, np.bool_)):
            raise ValueError(f"{bank_name} bank complete must be boolean")
        if not isinstance(bank["decoys"], list):
            raise ValueError(f"{bank_name} bank decoys must be a list")

        seen_fingerprints = set()
        counts = {stratum: 0 for stratum in STRATUM_ORDER}
        for entry in bank["decoys"]:
            fingerprint = _validate_decoy_entry(
                entry, bank_name, seen_fingerprints
            )
            counts[entry["stratum"]] += 1
            if counts[entry["stratum"]] > RETAINED_PER_STRATUM:
                raise ValueError(
                    f"{bank_name} bank retains more than "
                    f"{RETAINED_PER_STRATUM} {entry['stratum']} decoys"
                )
            all_entries.append((bank_name, entry, fingerprint))
        actual_counts[bank_name] = counts

        missing = [
            stratum
            for stratum in STRATUM_ORDER
            if counts[stratum] != RETAINED_PER_STRATUM
        ]
        expected_complete = not missing
        if bool(bank["complete"]) != expected_complete:
            raise ValueError(
                f"{bank_name} bank complete flag is inconsistent"
            )
        expected_reason = (
            None
            if expected_complete
            else f"insufficient_{bank_name}_candidates:{','.join(missing)}"
        )
        if bank["failure_reason"] != expected_reason:
            raise ValueError(
                f"{bank_name} bank failure_reason is inconsistent"
            )

    for stratum in STRATUM_ORDER:
        if (
            proposal_counts[stratum]["random_physical_retained"]
            != actual_counts["random_physical"][stratum]
            or proposal_counts[stratum]["matched_retained"]
            != actual_counts["matched"][stratum]
        ):
            raise ValueError("retained proposal counts do not match banks")

    if native_record is None:
        return decoy_set

    native_geometry = _prepare_eligible_native_geometry(
        native_record, native_hard_clash_as_generation_error=False
    )
    if decoy_set["key"] != native_geometry["key"]:
        raise ValueError(
            "decoy-set key does not match native geometry key"
        )
    expected_automorphism_digest = automorphism_digest(
        native_geometry["automorphisms"]
    )
    if (
        automorphism_count != len(native_geometry["automorphisms"])
        or decoy_set["automorphism_sha256"]
        != expected_automorphism_digest
    ):
        raise ValueError(
            "automorphism count or digest does not match cached native array"
        )

    proposal_pools = {
        stratum: generate_proposal_pool(
            native_record, seed=seed, stratum=stratum
        )
        for stratum in STRATUM_ORDER
    }
    if any(
        len(proposal_pools[stratum]) != PROPOSALS_PER_STRATUM
        for stratum in STRATUM_ORDER
    ):
        raise ValueError(
            "native proposal pools do not contain the frozen proposal count"
        )
    expected_proposal_digest = proposal_fingerprint_digest(proposal_pools)
    if (
        decoy_set["proposal_fingerprint_sha256"]
        != expected_proposal_digest
    ):
        raise ValueError(
            "proposal fingerprint digest does not match native proposal pool"
        )

    expected_summary = _evaluate_and_retain_proposal_pools(
        native_geometry, proposal_pools, seed
    )
    if decoy_set["proposal_counts"] != expected_summary["proposal_counts"]:
        raise ValueError(
            "proposal counts do not match deterministic proposal evaluation"
        )
    if decoy_set["rejection_counts"] != expected_summary["rejection_counts"]:
        raise ValueError(
            "rejection counts do not match deterministic proposal evaluation"
        )
    for bank_name in BANK_NAMES:
        actual_bank = decoy_set["banks"][bank_name]
        expected_bank = expected_summary["banks"][bank_name]
        if (
            actual_bank["complete"] != expected_bank["complete"]
            or actual_bank["failure_reason"]
            != expected_bank["failure_reason"]
        ):
            raise ValueError(
                f"{bank_name} bank completion metadata does not match "
                "deterministic retention"
            )
        actual_fingerprints = [
            row["transform_fingerprint"]
            for row in actual_bank["decoys"]
        ]
        expected_fingerprints = [
            row["transform_fingerprint"]
            for row in expected_bank["decoys"]
        ]
        if actual_fingerprints != expected_fingerprints:
            raise ValueError(
                f"{bank_name} retained fingerprint sequence does not "
                "match deterministic bank selection"
            )

    proposal_fingerprints = {
        row["transform_fingerprint"]
        for stratum in STRATUM_ORDER
        for row in proposal_pools[stratum]
    }

    for bank_name, entry, fingerprint in all_entries:
        if fingerprint not in proposal_fingerprints:
            raise ValueError(
                f"{bank_name} transform fingerprint is absent from "
                "the shared proposal pool"
            )
        transformed = apply_rigid_transform(
            native_geometry["native"], entry["transform"]
        )
        recomputed_rmsd = symmetry_corrected_rmsd(
            native_geometry["native"],
            transformed,
            native_geometry["automorphisms"],
        )
        recomputed_contact = soft_contact_statistics(
            native_geometry["protein"],
            native_geometry["native"],
            transformed,
        )
        recomputed_overlap = vdw_overlap_statistics(
            native_geometry["protein"],
            native_geometry["protein_symbols"],
            transformed,
            native_geometry["ligand_atomic_numbers"],
        )
        if not _geometry_values_match(
            entry["rmsd_sym"], recomputed_rmsd
        ):
            raise ValueError(
                "decoy geometry RMSD does not match native geometry"
            )
        _verify_geometry_statistics(
            entry["contact_stats"],
            recomputed_contact,
            "contact statistic",
        )
        _verify_geometry_statistics(
            entry["overlap_stats"],
            recomputed_overlap,
            "overlap statistic",
        )
        recomputed_physical = classify_physical_candidate(
            recomputed_rmsd,
            recomputed_contact,
            recomputed_overlap,
            target_stratum=entry["stratum"],
            transform_fingerprint=fingerprint,
            seen_fingerprints=set(),
        )
        if not recomputed_physical["accepted"]:
            raise ValueError(
                "decoy physical stratum does not match recomputed geometry"
            )
        recomputed_match_values = _candidate_match_values_from_geometry(
            native_geometry,
            transformed,
            recomputed_contact,
            recomputed_overlap,
            native_overlap=native_geometry["native_overlap"],
            native_centroid=native_geometry["native_centroid"],
            native_minimum=native_geometry["native_minimum"],
        )
        _verify_geometry_statistics(
            entry["match_values"],
            recomputed_match_values,
            "match value",
        )
        recomputed_match = classify_matched_candidate(
            recomputed_match_values, entry["stratum"]
        )
        if not _geometry_values_match(
            entry["match_distance"],
            recomputed_match["match_distance"],
        ):
            raise ValueError(
                "decoy match distance does not match native geometry"
            )
        if bool(entry["match_pass"]) != bool(
            recomputed_match["match_pass"]
        ):
            raise ValueError(
                "decoy match_pass does not match native geometry"
            )
        if bank_name == "matched" and not recomputed_match["match_pass"]:
            raise ValueError(
                "matched bank violates recomputed matching limits"
            )
    return decoy_set


def select_epoch_decoys_v2(decoy_set, key, epoch, seed, bank):
    validate_decoy_set_v2(decoy_set)
    if bank not in BANK_NAMES:
        raise ValueError(
            f"bank must be one of {', '.join(BANK_NAMES)}"
        )
    if str(decoy_set["key"]) != str(key):
        raise ValueError("selection key does not match decoy-set key")
    selected_bank = decoy_set["banks"][bank]
    if selected_bank["complete"] is not True:
        raise ValueError(f"requested bank {bank} must be complete")
    decoys = selected_bank["decoys"]
    epoch = _validate_seed(epoch)
    seed = _validate_seed(seed)

    selected = []
    for stratum in STRATUM_ORDER:
        candidates = [
            entry for entry in decoys if entry["stratum"] == stratum
        ]
        if len(candidates) != RETAINED_PER_STRATUM:
            raise ValueError(
                f"complete bank {bank} must contain "
                f"{RETAINED_PER_STRATUM} {stratum} decoys"
            )
        generator = _make_keyed_rng(
            seed,
            key,
            stratum,
            f"epoch_selection:{epoch}:{bank}",
        )
        index = int(generator.integers(0, len(candidates)))
        selected.append(copy.deepcopy(candidates[index]))
    return selected
