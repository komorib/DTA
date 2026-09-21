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
    _element_atomic_number,
    _vdw_radii,
    apply_rigid_transform,
    soft_contact_statistics,
    symmetry_corrected_rmsd,
    vdw_overlap_statistics,
)


DECOY_SCHEMA_VERSION = 3
BANK_NAMES = ("random_physical", "matched")
STRATUM_ORDER = ("low", "medium", "high")
BASE_ROTATIONS_PER_STRATUM = 512
TRANSLATIONS_PER_ROTATION = 4
PROPOSALS_PER_STRATUM = 2048
RETAINED_PER_STRATUM = 4
SHORTLIST_SIZE = 20
HARD_VDW_RATIO = 0.75
REPULSION_NEAR_RATIO = 1.10
ZERO_DISTANCE_EPSILON = 1e-12
CONTACT_MASS_RATIO_RANGE = (0.80, 1.15)
RNG_BIT_GENERATOR = "PCG64"
ROTATION_RANGES = {
    "low": (5.0, 35.0),
    "medium": (15.0, 80.0),
    "high": (30.0, 150.0),
}
TRANSLATION_GRIDS = {
    "low": (0.125, 0.250, 0.375, 0.500),
    "medium": (0.250, 0.500, 0.750, 1.000),
    "high": (0.375, 0.750, 1.125, 1.500),
}
DIRECTION_SOURCES = (
    "hard_clash_repulsion",
    "near_contact_repulsion",
    "rotated_contact_centroid",
    "native_interface_normal",
    "signed_rotation_axis",
)


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


def _hashed_signed_cartesian_axis(namespace):
    payload = "\0".join(str(value) for value in namespace).encode("utf-8")
    index = hashlib.sha256(payload).digest()[0] % 6
    axis = np.zeros(3, dtype=np.float64)
    axis[index // 2] = 1.0 if index % 2 == 0 else -1.0
    return axis


def _safe_pair_unit_vector(delta, ligand_centroid, protein_atom, namespace):
    delta = np.asarray(delta, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm >= ZERO_DISTANCE_EPSILON:
        return delta / norm
    fallback = np.asarray(ligand_centroid, dtype=np.float64) - np.asarray(
        protein_atom, dtype=np.float64
    )
    norm = float(np.linalg.norm(fallback))
    if norm >= ZERO_DISTANCE_EPSILON:
        return fallback / norm
    return _hashed_signed_cartesian_axis(namespace)


def _protein_vdw_radii(symbols):
    atomic_numbers = [_element_atomic_number(symbol) for symbol in symbols]
    return _vdw_radii(atomic_numbers, "protein atomic numbers")


def _vdw_pair_ratios(ligand, protein, ligand_radii, protein_radii):
    distances = np.linalg.norm(
        ligand[:, None, :] - protein[None, :, :], axis=-1
    )
    ratios = distances / (ligand_radii[:, None] + protein_radii[None, :])
    return distances, ratios


def _local_repulsion_direction(
    ligand, protein, ligand_radii, protein_radii, namespace
):
    delta = ligand[:, None, :] - protein[None, :, :]
    distances, ratios = _vdw_pair_ratios(
        ligand, protein, ligand_radii, protein_radii
    )
    hard = ratios < HARD_VDW_RATIO
    selected = hard if np.any(hard) else ratios < REPULSION_NEAR_RATIO
    source = (
        "hard_clash_repulsion"
        if np.any(hard)
        else "near_contact_repulsion"
    )
    if not np.any(selected):
        return None, None, 0

    force = np.zeros(3, dtype=np.float64)
    zero_count = 0
    centroid = np.mean(ligand, axis=0)
    for ligand_index, protein_index in np.argwhere(selected):
        zero_count += int(
            distances[ligand_index, protein_index] < ZERO_DISTANCE_EPSILON
        )
        unit = _safe_pair_unit_vector(
            delta[ligand_index, protein_index],
            centroid,
            protein[protein_index],
            (*namespace, int(ligand_index), int(protein_index)),
        )
        force += (
            REPULSION_NEAR_RATIO - ratios[ligand_index, protein_index]
        ) * unit
    norm = float(np.linalg.norm(force))
    if norm < ZERO_DISTANCE_EPSILON:
        return None, None, zero_count
    return force / norm, source, zero_count


def _contact_centroid_direction(ligand, protein):
    distances = np.linalg.norm(
        ligand[:, None, :] - protein[None, :, :], axis=-1
    )
    weights = np.clip(6.0 - distances, 0.0, 1.0)
    total = float(np.sum(weights))
    if total < ZERO_DISTANCE_EPSILON:
        return None
    ligand_weights = np.sum(weights, axis=1)
    protein_weights = np.sum(weights, axis=0)
    ligand_centroid = np.sum(ligand * ligand_weights[:, None], axis=0) / total
    protein_centroid = np.sum(protein * protein_weights[:, None], axis=0) / total
    direction = ligand_centroid - protein_centroid
    norm = float(np.linalg.norm(direction))
    return None if norm < ZERO_DISTANCE_EPSILON else direction / norm


def _endpoint_physical_score(
    rotated_ligand, protein, ligand_radii, protein_radii, direction, distance, sign_rank
):
    translated = rotated_ligand + direction * distance
    _, ratios = _vdw_pair_ratios(
        translated, protein, ligand_radii, protein_radii
    )
    severity = np.maximum(0.0, 1.0 - ratios)
    return (
        int(np.sum(ratios < HARD_VDW_RATIO)),
        -float(np.min(ratios)),
        float(np.sum(severity)),
        sign_rank,
    )


def _choose_signed_axis(
    rotated_ligand,
    protein,
    protein_symbols,
    ligand_atomic_numbers,
    axis,
    maximum_distance,
):
    ligand_radii = _vdw_radii(ligand_atomic_numbers, "ligand_atomic_numbers")
    protein_radii = _protein_vdw_radii(protein_symbols)
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= np.linalg.norm(unit_axis)
    candidates = [(unit_axis, 0), (-unit_axis, 1)]
    return min(
        candidates,
        key=lambda item: _endpoint_physical_score(
            rotated_ligand,
            protein,
            ligand_radii,
            protein_radii,
            item[0],
            maximum_distance,
            item[1],
        ),
    )


def _proposal_direction(
    rotated_ligand,
    native_ligand,
    protein,
    protein_symbols,
    ligand_atomic_numbers,
    axis,
    maximum_distance,
    seed,
    key,
    stratum,
    base_rotation_id,
):
    ligand_radii = _vdw_radii(ligand_atomic_numbers, "ligand_atomic_numbers")
    protein_radii = _protein_vdw_radii(protein_symbols)
    local_direction, local_source, zero_count = _local_repulsion_direction(
        rotated_ligand,
        protein,
        ligand_radii,
        protein_radii,
        (seed, key, stratum, base_rotation_id),
    )
    if local_direction is not None:
        return local_direction, local_source, zero_count
    rotated = _contact_centroid_direction(rotated_ligand, protein)
    if rotated is not None:
        return rotated, "rotated_contact_centroid", zero_count
    native = _contact_centroid_direction(native_ligand, protein)
    if native is not None:
        return native, "native_interface_normal", zero_count
    signed_axis, _ = _choose_signed_axis(
        rotated_ligand,
        protein,
        protein_symbols,
        ligand_atomic_numbers,
        axis,
        maximum_distance,
    )
    return signed_axis, "signed_rotation_axis", zero_count


def _base_rotation_fingerprint(center, rotation):
    return _transform_fingerprint(
        {
            "center": np.asarray(center, dtype=np.float64).astype(float).tolist(),
            "rotation_matrix": np.asarray(rotation, dtype=np.float64).astype(float).tolist(),
        }
    )


def generate_proposal_pool(record, seed, stratum):
    if stratum not in STRATUM_ORDER:
        raise ValueError(
            f"stratum must be one of {', '.join(STRATUM_ORDER)}"
        )
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")
    if "key" not in record:
        raise ValueError("record must contain key")

    native_ligand = _as_coordinates(record.get("ligand_coords"), "ligand_coords")
    protein = _as_coordinates(record.get("protein_coords"), "protein_coords")
    protein_symbols = np.asarray(record.get("protein_element_symbol")).reshape(-1)
    ligand_atomic_numbers = np.asarray(
        record.get("ligand_atomic_numbers")
    ).reshape(-1)
    if len(protein_symbols) != len(protein):
        raise ValueError("protein element symbols are not aligned to coordinates")
    if len(ligand_atomic_numbers) != len(native_ligand):
        raise ValueError("ligand atomic numbers are not aligned to coordinates")

    generator = _make_keyed_rng(seed, record["key"], stratum, "proposal")
    center = np.mean(native_ligand, axis=0)
    maximum_distance = max(TRANSLATION_GRIDS[stratum])
    rows = []
    for base_rotation_id in range(BASE_ROTATIONS_PER_STRATUM):
        axis = _random_unit_vector(generator)
        rotation_degrees = float(
            generator.uniform(*ROTATION_RANGES[stratum])
        )
        rotation = _rodrigues_rotation(axis, np.deg2rad(rotation_degrees))
        base_transform = {
            "center": center.astype(float).tolist(),
            "rotation_matrix": rotation.astype(float).tolist(),
            "translation": [0.0, 0.0, 0.0],
        }
        rotated_ligand = apply_rigid_transform(native_ligand, base_transform)
        direction, direction_source, zero_count = _proposal_direction(
            rotated_ligand,
            native_ligand,
            protein,
            protein_symbols,
            ligand_atomic_numbers,
            axis,
            maximum_distance,
            seed,
            record["key"],
            stratum,
            base_rotation_id,
        )
        zero_translation_vdw_safe = not vdw_overlap_statistics(
            protein, protein_symbols, rotated_ligand, ligand_atomic_numbers
        )["has_hard_clash"]
        base_rotation_fingerprint = _base_rotation_fingerprint(center, rotation)
        for translation_index, translation_distance in enumerate(
            TRANSLATION_GRIDS[stratum]
        ):
            translation = direction * translation_distance
            transform = {
                "center": center.astype(float).tolist(),
                "rotation_matrix": rotation.astype(float).tolist(),
                "translation": translation.astype(float).tolist(),
            }
            rows.append(
                {
                    "transform": transform,
                    "rotation_degrees": rotation_degrees,
                    "base_rotation_id": base_rotation_id,
                    "base_rotation_fingerprint": base_rotation_fingerprint,
                    "translation_index": translation_index,
                    "translation_distance": float(translation_distance),
                    "translation_direction": direction.astype(float).tolist(),
                    "direction_source": direction_source,
                    "zero_distance_pair_count": zero_count,
                    "zero_translation_vdw_safe": zero_translation_vdw_safe,
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
    "base_rotation_id",
    "base_rotation_fingerprint",
    "translation_index",
    "translation_distance",
    "translation_direction",
    "direction_source",
    "zero_distance_pair_count",
    "zero_translation_vdw_safe",
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
    "physical_rejection_counts",
    "match_rejection_counts",
    "direction_source_counts",
    "family_funnel_counts",
    "zero_distance_pair_count_by_stratum",
}
BANK_FIELDS = {"complete", "failure_reason", "decoys"}
PROPOSAL_COUNT_FIELDS = {
    "base_rotation_families",
    "proposed",
    "physical_accepted",
    "physical_family_count",
    "match_accepted",
    "match_family_count",
    "shortlisted_families",
    "random_physical_retained",
    "matched_retained",
}
DIRECTION_SOURCE_DIAGNOSTIC_FIELDS = {
    "proposed_families",
    "proposed_transforms",
    "families_with_any_physical_candidate",
    "families_with_any_matched_candidate",
    "physical_accepted_transforms",
    "match_accepted_transforms",
}
FAMILY_FUNNEL_DIAGNOSTIC_FIELDS = {
    "base_rotation_families",
    "zero_translation_vdw_safe",
    "any_grid_vdw_safe",
    "any_physical_target_candidate",
    "any_matched_candidate",
    "max_distance_still_hard_clash",
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


def _group_by_base_rotation(candidates):
    grouped = {}
    for row in candidates:
        family_id = int(row["base_rotation_id"])
        grouped.setdefault(family_id, []).append(row)
    return grouped


def retain_random_physical(candidates, seed, key, stratum):
    by_family = _group_by_base_rotation(candidates)
    family_ids = sorted(by_family)
    if not family_ids:
        return []
    rng = _make_keyed_rng(seed, key, stratum, "random_physical_family")
    indices = rng.choice(
        len(family_ids),
        size=min(RETAINED_PER_STRATUM, len(family_ids)),
        replace=False,
    )
    retained = []
    for index in indices:
        family_id = family_ids[int(index)]
        rows = sorted(
            by_family[family_id],
            key=lambda row: str(row["transform_fingerprint"]),
        )
        row_rng = _make_keyed_rng(
            seed,
            key,
            stratum,
            f"random_physical_translation:{family_id}",
        )
        retained.append(rows[int(row_rng.integers(0, len(rows)))])
    return retained


def retain_matched(candidates, seed, key, stratum):
    by_family = _group_by_base_rotation(
        [row for row in candidates if row.get("match_pass") is True]
    )
    representatives = [
        min(
            rows,
            key=lambda row: (
                float(row["match_distance"]),
                int(row["translation_index"]),
                str(row["transform_fingerprint"]),
            ),
        )
        for rows in by_family.values()
    ]
    representatives.sort(
        key=lambda row: (
            float(row["match_distance"]),
            int(row["translation_index"]),
            str(row["transform_fingerprint"]),
        )
    )
    shortlist = representatives[: min(SHORTLIST_SIZE, len(representatives))]
    return _sample_retained(shortlist, seed, key, stratum, "matched")


def _validate_distinct_family_ids(entries, bank_name):
    by_stratum = {stratum: [] for stratum in STRATUM_ORDER}
    for entry in entries:
        stratum = entry.get("stratum")
        if stratum in by_stratum:
            by_stratum[stratum].append(entry)
    for stratum, rows in by_stratum.items():
        if len(rows) == RETAINED_PER_STRATUM:
            ids = {int(row["base_rotation_id"]) for row in rows}
            if len(ids) != RETAINED_PER_STRATUM:
                raise ValueError(
                    f"{bank_name} {stratum} retained entries must use "
                    "distinct base rotation ids"
                )


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
        "base_rotation_id": proposal["base_rotation_id"],
        "base_rotation_fingerprint": proposal["base_rotation_fingerprint"],
        "translation_index": proposal["translation_index"],
        "translation_distance": proposal["translation_distance"],
        "translation_direction": proposal["translation_direction"],
        "direction_source": proposal["direction_source"],
        "zero_distance_pair_count": proposal["zero_distance_pair_count"],
        "zero_translation_vdw_safe": proposal["zero_translation_vdw_safe"],
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


def _summarize_proposal_diagnostics(rows, evaluations):
    if len(rows) != len(evaluations):
        raise ValueError("proposal rows and evaluations must align")
    families = {}
    zero_distance_pair_count = 0
    for row, outcome in zip(rows, evaluations):
        family_id = int(row["base_rotation_id"])
        source = row["direction_source"]
        if source not in DIRECTION_SOURCES:
            raise ValueError("proposal direction source is invalid")
        state = families.setdefault(
            family_id,
            {
                "source": source,
                "zero_translation_vdw_safe": bool(
                    row["zero_translation_vdw_safe"]
                ),
                "outcomes": [],
                "physical_transforms": 0,
                "matched_transforms": 0,
                "any_grid_vdw_safe": False,
            },
        )
        if (
            state["source"] != source
            or state["zero_translation_vdw_safe"]
            != bool(row["zero_translation_vdw_safe"])
        ):
            raise ValueError("proposal family metadata is inconsistent")
        candidate, physical_reason, match_reason = outcome
        physical = candidate is not None
        matched = physical and match_reason is None
        vdw_safe = physical or physical_reason not in {
            "vdw_hard_clash", "nonfinite_metrics"
        }
        state["physical_transforms"] += int(physical)
        state["matched_transforms"] += int(matched)
        state["any_grid_vdw_safe"] = (
            state["any_grid_vdw_safe"] or vdw_safe
        )
        state["outcomes"].append(
            (int(row["translation_index"]), physical_reason)
        )
        zero_distance_pair_count += int(row["zero_distance_pair_count"])

    direction_counts = {}
    family_counts = {
        "base_rotation_families": len(families),
        "zero_translation_vdw_safe": 0,
        "any_grid_vdw_safe": 0,
        "any_physical_target_candidate": 0,
        "any_matched_candidate": 0,
        "max_distance_still_hard_clash": 0,
    }
    for state in families.values():
        source = state["source"]
        counts = direction_counts.setdefault(
            source,
            {field: 0 for field in DIRECTION_SOURCE_DIAGNOSTIC_FIELDS},
        )
        physical_family = state["physical_transforms"] > 0
        matched_family = state["matched_transforms"] > 0
        counts["proposed_families"] += 1
        counts["proposed_transforms"] += len(state["outcomes"])
        counts["families_with_any_physical_candidate"] += int(
            physical_family
        )
        counts["families_with_any_matched_candidate"] += int(
            matched_family
        )
        counts["physical_accepted_transforms"] += state[
            "physical_transforms"
        ]
        counts["match_accepted_transforms"] += state[
            "matched_transforms"
        ]
        family_counts["zero_translation_vdw_safe"] += int(
            state["zero_translation_vdw_safe"]
        )
        family_counts["any_grid_vdw_safe"] += int(
            state["any_grid_vdw_safe"]
        )
        family_counts["any_physical_target_candidate"] += int(
            physical_family
        )
        family_counts["any_matched_candidate"] += int(matched_family)
        maximum_outcome = max(state["outcomes"], key=lambda item: item[0])
        family_counts["max_distance_still_hard_clash"] += int(
            maximum_outcome[1] == "vdw_hard_clash"
        )

    return {
        "direction_source_counts": direction_counts,
        "family_funnel_counts": family_counts,
        "zero_distance_pair_count": zero_distance_pair_count,
    }


def _evaluate_and_retain_proposal_pools(
    native_geometry, proposal_pools, seed
):
    seed = _validate_seed(seed)
    if not isinstance(proposal_pools, dict) or set(proposal_pools) != set(
        STRATUM_ORDER
    ):
        raise ValueError("proposal pools must contain every frozen stratum")
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
    physical_rejections = {}
    match_rejections = {}
    direction_source_counts = {}
    family_funnel_counts = {}
    zero_counts = {}
    seen_fingerprints = set()

    for stratum in STRATUM_ORDER:
        rows = proposal_pools[stratum]
        physical_candidates = []
        physical_counts = Counter()
        match_counts = Counter()
        evaluations = []
        for proposal in rows:
            candidate, physical_reason, match_reason = _evaluate_proposal(
                native_geometry,
                proposal,
                stratum,
                seen_fingerprints,
            )
            evaluations.append(
                (candidate, physical_reason, match_reason)
            )
            if candidate is None:
                physical_counts[physical_reason] += 1
                continue
            physical_candidates.append(candidate)
            if match_reason is not None:
                match_counts[match_reason] += 1

        diagnostics = _summarize_proposal_diagnostics(rows, evaluations)
        direction_source_counts[stratum] = diagnostics[
            "direction_source_counts"
        ]
        family_funnel_counts[stratum] = diagnostics[
            "family_funnel_counts"
        ]
        zero_counts[stratum] = diagnostics[
            "zero_distance_pair_count"
        ]

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
        retained_by_bank["matched"][stratum] = copy.deepcopy(matched_rows)
        matched_candidates = [
            row for row in physical_candidates if row["match_pass"]
        ]
        physical_family_count = len(
            {row["base_rotation_id"] for row in physical_candidates}
        )
        match_family_count = len(
            {row["base_rotation_id"] for row in matched_candidates}
        )
        values = {
            "base_rotation_families": BASE_ROTATIONS_PER_STRATUM,
            "proposed": len(rows),
            "physical_accepted": len(physical_candidates),
            "physical_family_count": physical_family_count,
            "match_accepted": len(matched_candidates),
            "match_family_count": match_family_count,
            "shortlisted_families": min(SHORTLIST_SIZE, match_family_count),
            "random_physical_retained": len(random_rows),
            "matched_retained": len(matched_rows),
        }
        proposal_counts[stratum] = values
        physical_rejections[stratum] = dict(sorted(physical_counts.items()))
        match_rejections[stratum] = dict(sorted(match_counts.items()))

    banks = {}
    for bank_name in BANK_NAMES:
        failure_reason = _bank_failure_reason(
            bank_name, retained_by_stratum=retained_by_bank[bank_name]
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
        "physical_rejection_counts": physical_rejections,
        "match_rejection_counts": match_rejections,
        "direction_source_counts": direction_source_counts,
        "family_funnel_counts": family_funnel_counts,
        "zero_distance_pair_count_by_stratum": zero_counts,
    }


def generate_decoy_set_v3(record, seed):
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
        "physical_rejection_counts": summary["physical_rejection_counts"],
        "match_rejection_counts": summary["match_rejection_counts"],
        "direction_source_counts": summary["direction_source_counts"],
        "family_funnel_counts": summary["family_funnel_counts"],
        "zero_distance_pair_count_by_stratum": (
            summary["zero_distance_pair_count_by_stratum"]
        ),
    }
    return validate_decoy_set_v3(decoy_set)


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
        if values["base_rotation_families"] != BASE_ROTATIONS_PER_STRATUM:
            raise ValueError("base rotation family count is inconsistent")
        if values["proposed"] != PROPOSALS_PER_STRATUM:
            raise ValueError(
                "proposal count must equal proposal_count_per_stratum"
            )
        if values["physical_accepted"] > values["proposed"]:
            raise ValueError("physical-accepted count exceeds proposed count")
        if values["physical_family_count"] > values["base_rotation_families"]:
            raise ValueError("physical family count exceeds proposed families")
        if values["match_accepted"] > values["physical_accepted"]:
            raise ValueError("match-accepted count exceeds physical-accepted count")
        if values["match_family_count"] > values["physical_family_count"]:
            raise ValueError("match family count exceeds physical family count")
        if values["shortlisted_families"] != min(
            SHORTLIST_SIZE, values["match_family_count"]
        ):
            raise ValueError("shortlisted family count is inconsistent")
        if values["random_physical_retained"] != min(
            RETAINED_PER_STRATUM, values["physical_family_count"]
        ):
            raise ValueError(
                "random-physical retained proposal count is inconsistent"
            )
        if values["matched_retained"] != min(
            RETAINED_PER_STRATUM, values["match_family_count"]
        ):
            raise ValueError("matched retained proposal count is inconsistent")
        validated[stratum] = values
    return validated


def _validate_rejection_counts(
    physical_rejection_counts, match_rejection_counts, proposal_counts
):
    for label, counts_by_stratum in (
        ("physical rejection counts", physical_rejection_counts),
        ("match rejection counts", match_rejection_counts),
    ):
        _require_exact_fields(
            counts_by_stratum, set(STRATUM_ORDER), label
        )
    for stratum in STRATUM_ORDER:
        totals = []
        for label, counts in (
            ("physical", physical_rejection_counts[stratum]),
            ("match", match_rejection_counts[stratum]),
        ):
            if not isinstance(counts, dict):
                raise ValueError(f"{stratum} {label} rejection counts must be a dictionary")
            total = 0
            for reason, value in counts.items():
                if not isinstance(reason, str) or not reason:
                    raise ValueError("rejection reason must be a nonempty string")
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                ):
                    raise ValueError("rejection counts must be non-negative integers")
                total += int(value)
            totals.append(total)
        proposed = proposal_counts[stratum]["proposed"]
        physical = proposal_counts[stratum]["physical_accepted"]
        matched = proposal_counts[stratum]["match_accepted"]
        if totals[0] != proposed - physical:
            raise ValueError(f"{stratum} physical rejection total is inconsistent")
        if totals[1] != physical - matched:
            raise ValueError(f"{stratum} match rejection total is inconsistent")


def _validate_funnel_diagnostics(decoy_set, proposal_counts):
    _require_exact_fields(
        decoy_set["direction_source_counts"],
        set(STRATUM_ORDER),
        "direction source counts",
    )
    _require_exact_fields(
        decoy_set["family_funnel_counts"],
        set(STRATUM_ORDER),
        "family funnel counts",
    )
    _require_exact_fields(
        decoy_set["zero_distance_pair_count_by_stratum"],
        set(STRATUM_ORDER),
        "zero-distance pair counts",
    )
    for stratum in STRATUM_ORDER:
        source_counts = decoy_set["direction_source_counts"][stratum]
        if not isinstance(source_counts, dict) or not set(
            source_counts
        ).issubset(set(DIRECTION_SOURCES)):
            raise ValueError("direction source counts are invalid")
        source_totals = Counter()
        for source, values in source_counts.items():
            _require_exact_fields(
                values,
                DIRECTION_SOURCE_DIAGNOSTIC_FIELDS,
                f"{stratum} {source} direction diagnostics",
            )
            for field in DIRECTION_SOURCE_DIAGNOSTIC_FIELDS:
                source_totals[field] += _nonnegative_integer(
                    values, field, "direction source diagnostic"
                )

        expected_source_totals = {
            "proposed_families": proposal_counts[stratum][
                "base_rotation_families"
            ],
            "proposed_transforms": proposal_counts[stratum]["proposed"],
            "families_with_any_physical_candidate": proposal_counts[
                stratum
            ]["physical_family_count"],
            "families_with_any_matched_candidate": proposal_counts[stratum][
                "match_family_count"
            ],
            "physical_accepted_transforms": proposal_counts[stratum][
                "physical_accepted"
            ],
            "match_accepted_transforms": proposal_counts[stratum][
                "match_accepted"
            ],
        }
        if dict(source_totals) != expected_source_totals:
            raise ValueError(
                "direction source diagnostics do not match proposal counts"
            )

        family_counts = decoy_set["family_funnel_counts"][stratum]
        _require_exact_fields(
            family_counts,
            FAMILY_FUNNEL_DIAGNOSTIC_FIELDS,
            f"{stratum} family funnel counts",
        )
        validated_family = {
            field: _nonnegative_integer(
                family_counts, field, "family funnel count"
            )
            for field in FAMILY_FUNNEL_DIAGNOSTIC_FIELDS
        }
        family_total = validated_family["base_rotation_families"]
        if family_total != proposal_counts[stratum][
            "base_rotation_families"
        ]:
            raise ValueError("family funnel total does not match proposals")
        if any(
            value > family_total
            for field, value in validated_family.items()
            if field != "base_rotation_families"
        ):
            raise ValueError("family funnel count exceeds family total")
        if (
            validated_family["any_physical_target_candidate"]
            != proposal_counts[stratum]["physical_family_count"]
            or validated_family["any_matched_candidate"]
            != proposal_counts[stratum]["match_family_count"]
        ):
            raise ValueError(
                "family funnel acceptance does not match proposal counts"
            )
        _nonnegative_integer(
            decoy_set["zero_distance_pair_count_by_stratum"],
            stratum,
            "zero-distance pair count",
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
    family_id = entry["base_rotation_id"]
    if (
        isinstance(family_id, (bool, np.bool_))
        or not isinstance(family_id, (int, np.integer))
        or not 0 <= int(family_id) < BASE_ROTATIONS_PER_STRATUM
    ):
        raise ValueError("base rotation id is invalid")
    base_fingerprint = _validate_sha256(
        entry["base_rotation_fingerprint"], "base rotation fingerprint"
    )
    if base_fingerprint != _base_rotation_fingerprint(
        transform["center"], transform["rotation_matrix"]
    ):
        raise ValueError("base rotation fingerprint does not match transform")
    translation_index = entry["translation_index"]
    if (
        isinstance(translation_index, (bool, np.bool_))
        or not isinstance(translation_index, (int, np.integer))
        or not 0 <= int(translation_index) < TRANSLATIONS_PER_ROTATION
    ):
        raise ValueError("translation index is invalid")
    distance = _finite_number(
        entry, "translation_distance", "decoy translation"
    )
    expected_distance = TRANSLATION_GRIDS[entry["stratum"]][
        int(translation_index)
    ]
    if not np.isclose(distance, expected_distance):
        raise ValueError("translation distance does not match frozen grid")
    direction = np.asarray(entry["translation_direction"], dtype=np.float64)
    if (
        direction.shape != (3,)
        or not np.all(np.isfinite(direction))
        or not np.isclose(np.linalg.norm(direction), 1.0)
    ):
        raise ValueError("translation direction must be a finite unit vector")
    if entry["direction_source"] not in DIRECTION_SOURCES:
        raise ValueError("direction source is invalid")
    _nonnegative_integer(
        entry, "zero_distance_pair_count", "decoy zero-distance pair"
    )
    if not isinstance(entry["zero_translation_vdw_safe"], (bool, np.bool_)):
        raise ValueError("zero translation VDW-safe flag must be boolean")
    if not np.allclose(
        np.asarray(transform["translation"], dtype=np.float64),
        direction * distance,
        rtol=GEOMETRY_RTOL,
        atol=GEOMETRY_ATOL,
    ):
        raise ValueError("transform translation disagrees with provenance")
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


def validate_decoy_set_v3(decoy_set, native_record=None):
    if not isinstance(decoy_set, dict):
        raise ValueError("decoy set must be a dictionary")
    if decoy_set.get("schema_version") != DECOY_SCHEMA_VERSION:
        raise ValueError("decoy schema version 3 is required")
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
        decoy_set["physical_rejection_counts"],
        decoy_set["match_rejection_counts"],
        proposal_counts,
    )
    _validate_funnel_diagnostics(decoy_set, proposal_counts)
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
            _validate_distinct_family_ids(bank["decoys"], bank_name)
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
    for field in (
        "physical_rejection_counts",
        "match_rejection_counts",
        "direction_source_counts",
        "family_funnel_counts",
        "zero_distance_pair_count_by_stratum",
    ):
        if decoy_set[field] != expected_summary[field]:
            raise ValueError(
                f"{field} does not match deterministic proposal evaluation"
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
        for actual_entry, expected_entry in zip(
            actual_bank["decoys"], expected_bank["decoys"]
        ):
            for field in (
                "base_rotation_id",
                "base_rotation_fingerprint",
                "translation_index",
                "translation_distance",
                "translation_direction",
                "direction_source",
                "zero_distance_pair_count",
                "zero_translation_vdw_safe",
            ):
                if actual_entry[field] != expected_entry[field]:
                    raise ValueError(
                        f"{bank_name} {field} does not match deterministic "
                        "proposal provenance"
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


def select_epoch_decoys_v3(decoy_set, key, epoch, seed, bank):
    validate_decoy_set_v3(decoy_set)
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
