from collections import Counter
import hashlib
import json

import numpy as np
from rdkit import Chem

from data_utils.local_interface import continuous_contact_weights


DECOY_SCHEMA_VERSION = 1
STRATUM_ORDER = ("low", "medium", "high")
MINIMUM_DECOYS = 12
MINIMUM_PER_STRATUM = 4
MAX_CANDIDATE_RETRIES = 64
HARD_VDW_RATIO = 0.75
CONTACT_RATIO_RANGE = (0.6, 1.2)
GEOMETRY_RTOL = 1e-7
GEOMETRY_ATOL = 1e-9

STRATUM_BOUNDS = {
    "low": {
        "rmsd_min": 0.4,
        "rmsd_max": 1.0,
        "rmsd_max_inclusive": False,
        "jaccard_min": 0.68,
        "jaccard_max": 0.95,
        "jaccard_max_inclusive": False,
    },
    "medium": {
        "rmsd_min": 1.0,
        "rmsd_max": 1.6,
        "rmsd_max_inclusive": False,
        "jaccard_min": 0.52,
        "jaccard_max": 0.82,
        "jaccard_max_inclusive": False,
    },
    "high": {
        "rmsd_min": 1.6,
        "rmsd_max": 2.5,
        "rmsd_max_inclusive": True,
        "jaccard_min": 0.32,
        "jaccard_max": 0.68,
        "jaccard_max_inclusive": False,
    },
}

# These ranges only propose candidates. Frozen acceptance is determined by
# STRATUM_BOUNDS and the geometry filters below.
PROPOSAL_RANGES = {
    "low": {
        "translation": (0.35, 0.95),
        "rotation_degrees": (0.0, 5.0),
    },
    "medium": {
        "translation": (0.85, 1.55),
        "rotation_degrees": (0.0, 10.0),
    },
    "high": {
        "translation": (1.35, 2.40),
        "rotation_degrees": (0.0, 15.0),
    },
}

DECOY_ENTRY_FIELDS = {
    "transform",
    "rmsd_sym",
    "contact_stats",
    "overlap_stats",
    "stratum",
    "failure_reason",
}
TRANSFORM_FIELDS = {"center", "rotation_matrix", "translation"}
CONTACT_STAT_FIELDS = {
    "native_contact_mass",
    "decoy_contact_mass",
    "contact_mass_ratio",
    "soft_jaccard",
    "native_neighbor_count",
    "decoy_neighbor_count",
    "coverage_retained",
    "contact_pattern_nonempty",
}
OVERLAP_STAT_FIELDS = {
    "minimum_distance_ratio",
    "max_overlap_severity",
    "sum_overlap_severity",
    "count_below_0_9",
    "count_below_0_8",
    "has_hard_clash",
}


class DecoyGenerationError(RuntimeError):
    def __init__(self, message, rejection_counts):
        super().__init__(message)
        self.rejection_counts = dict(rejection_counts)


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


def _as_vector3(value, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-three vector")
    return vector


def _validate_automorphisms(automorphisms, atom_count):
    raw = np.asarray(automorphisms)
    if raw.ndim != 2 or raw.shape[1] != atom_count or len(raw) == 0:
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


def symmetry_corrected_rmsd(native_coords, decoy_coords, automorphisms):
    native = _as_coordinates(native_coords, "native_coords")
    decoy = _as_coordinates(decoy_coords, "decoy_coords")
    if native.shape != decoy.shape:
        raise ValueError("native and decoy coordinates must have the same shape")
    mappings = _validate_automorphisms(automorphisms, len(native))
    differences = native[None, :, :] - decoy[mappings]
    squared_distances = np.sum(differences * differences, axis=-1)
    rmsd_values = np.sqrt(np.mean(squared_distances, axis=1))
    return float(np.min(rmsd_values))


def _validate_range(value, name, lower_limit=None, upper_limit=None):
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two finite values") from error
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError(f"{name} must be finite and ordered")
    if lower_limit is not None and lower < lower_limit:
        raise ValueError(f"{name} lower bound is out of range")
    if upper_limit is not None and upper > upper_limit:
        raise ValueError(f"{name} upper bound is out of range")
    return lower, upper


def _make_rng(seed=None, rng=None):
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        return rng
    return np.random.default_rng(seed)


def _random_unit_vector(rng):
    while True:
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            return vector / norm


def apply_rigid_transform(coords, transform):
    coordinates = _as_coordinates(coords, "coords")
    if not isinstance(transform, dict):
        raise ValueError("transform must be a dictionary")
    center = _as_vector3(transform.get("center"), "transform center")
    translation = _as_vector3(
        transform.get("translation"), "transform translation"
    )
    rotation = np.asarray(
        transform.get("rotation_matrix"), dtype=np.float64
    )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("transform rotation_matrix must be finite [3,3]")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("transform rotation_matrix must be orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("transform rotation_matrix must be a proper rotation")
    return (coordinates - center) @ rotation.T + center + translation


def random_rigid_transform(
    coords,
    seed=None,
    rng=None,
    translation_distance_range=(0.0, 6.0),
    rotation_degrees_range=(0.0, 180.0),
):
    coordinates = _as_coordinates(coords, "coords")
    generator = _make_rng(seed=seed, rng=rng)
    translation_range = _validate_range(
        translation_distance_range, "translation_distance_range", lower_limit=0.0
    )
    rotation_range = _validate_range(
        rotation_degrees_range,
        "rotation_degrees_range",
        lower_limit=0.0,
        upper_limit=180.0,
    )

    axis = _random_unit_vector(generator)
    angle = np.deg2rad(generator.uniform(*rotation_range))
    axis_cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * axis_cross
    )
    direction = _random_unit_vector(generator)
    distance = generator.uniform(*translation_range)
    translation = direction * distance
    center = np.mean(coordinates, axis=0)
    metadata = {
        "center": center.astype(float).tolist(),
        "rotation_matrix": rotation.astype(float).tolist(),
        "translation": translation.astype(float).tolist(),
    }
    transformed = apply_rigid_transform(coordinates, metadata)
    return transformed, metadata


def _element_atomic_number(symbol):
    normalized = str(symbol).strip()
    if not normalized:
        raise ValueError("protein element symbol must be nonempty")
    periodic_table = Chem.GetPeriodicTable()
    try:
        atomic_number = int(periodic_table.GetAtomicNumber(normalized))
    except RuntimeError as error:
        raise ValueError(f"unknown protein element symbol {normalized}") from error
    if atomic_number <= 0:
        raise ValueError(f"unknown protein element symbol {normalized}")
    return atomic_number


def _vdw_radii(atomic_numbers, name):
    numbers = np.asarray(atomic_numbers)
    if numbers.ndim != 1 or len(numbers) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    try:
        integer_numbers = numbers.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain atomic numbers") from error
    if not np.array_equal(numbers, integer_numbers):
        raise ValueError(f"{name} must contain atomic numbers")
    periodic_table = Chem.GetPeriodicTable()
    radii = []
    for atomic_number in integer_numbers:
        if atomic_number <= 0 or atomic_number > 118:
            raise ValueError(f"{name} contains invalid atomic number")
        radii.append(periodic_table.GetRvdw(int(atomic_number)))
    return np.asarray(radii, dtype=np.float64)


def vdw_overlap_statistics(
    protein_coords,
    protein_element_symbols,
    ligand_coords,
    ligand_atomic_numbers,
):
    protein = _as_coordinates(protein_coords, "protein_coords")
    ligand = _as_coordinates(ligand_coords, "ligand_coords")
    symbols = np.asarray(protein_element_symbols).reshape(-1)
    if len(symbols) != len(protein):
        raise ValueError("protein element symbols are not aligned to coordinates")
    protein_numbers = [_element_atomic_number(symbol) for symbol in symbols]
    protein_radii = _vdw_radii(protein_numbers, "protein atomic numbers")
    ligand_radii = _vdw_radii(
        ligand_atomic_numbers, "ligand_atomic_numbers"
    )
    if len(ligand_radii) != len(ligand):
        raise ValueError("ligand atomic numbers are not aligned to coordinates")

    distances = np.linalg.norm(
        protein[:, None, :] - ligand[None, :, :], axis=-1
    )
    radius_sums = protein_radii[:, None] + ligand_radii[None, :]
    ratios = distances / radius_sums
    severity = np.maximum(0.0, 1.0 - ratios)
    return {
        "minimum_distance_ratio": float(np.min(ratios)),
        "max_overlap_severity": float(np.max(severity)),
        "sum_overlap_severity": float(np.sum(severity)),
        "count_below_0_9": int(np.sum(ratios < 0.9)),
        "count_below_0_8": int(np.sum(ratios < 0.8)),
        "has_hard_clash": bool(np.any(ratios < HARD_VDW_RATIO)),
    }


def soft_contact_statistics(protein_coords, native_coords, decoy_coords):
    protein = _as_coordinates(protein_coords, "protein_coords")
    native = _as_coordinates(native_coords, "native_coords")
    decoy = _as_coordinates(decoy_coords, "decoy_coords")
    if native.shape != decoy.shape:
        raise ValueError("native and decoy ligand coordinates must align")
    native_distances = np.linalg.norm(
        protein[:, None, :] - native[None, :, :], axis=-1
    )
    decoy_distances = np.linalg.norm(
        protein[:, None, :] - decoy[None, :, :], axis=-1
    )
    native_weights = continuous_contact_weights(native_distances).astype(
        np.float64
    )
    decoy_weights = continuous_contact_weights(decoy_distances).astype(
        np.float64
    )
    native_mass = float(np.sum(native_weights))
    decoy_mass = float(np.sum(decoy_weights))
    union_mass = float(np.sum(np.maximum(native_weights, decoy_weights)))
    intersection_mass = float(
        np.sum(np.minimum(native_weights, decoy_weights))
    )
    native_neighbors = np.any(native_weights > 0.0, axis=0)
    decoy_neighbors = np.any(decoy_weights > 0.0, axis=0)
    coverage_retained = bool(
        np.all(np.logical_or(~native_neighbors, decoy_neighbors))
    )
    return {
        "native_contact_mass": native_mass,
        "decoy_contact_mass": decoy_mass,
        "contact_mass_ratio": (
            float(decoy_mass / native_mass)
            if native_mass > 0.0
            else float("inf")
        ),
        "soft_jaccard": (
            float(intersection_mass / union_mass)
            if union_mass > 0.0
            else 1.0
        ),
        "native_neighbor_count": int(np.sum(native_neighbors)),
        "decoy_neighbor_count": int(np.sum(decoy_neighbors)),
        "coverage_retained": coverage_retained,
        "contact_pattern_nonempty": bool(
            native_mass > 0.0 and decoy_mass > 0.0
        ),
    }


def _in_stratum(value, lower, upper, upper_inclusive):
    if value < lower:
        return False
    if upper_inclusive:
        return value <= upper
    return value < upper


def classify_decoy(rmsd_sym, contact_stats, overlap_stats):
    rmsd = float(rmsd_sym)
    ratio = float(contact_stats["contact_mass_ratio"])
    jaccard = float(contact_stats["soft_jaccard"])
    if not all(np.isfinite(value) for value in (rmsd, ratio, jaccard)):
        reason = "nonfinite_metrics"
    elif bool(overlap_stats["has_hard_clash"]):
        reason = "vdw_hard_clash"
    elif not bool(contact_stats["contact_pattern_nonempty"]):
        reason = "empty_contact_pattern"
    elif not bool(contact_stats["coverage_retained"]):
        reason = "ligand_neighbor_coverage"
    elif not CONTACT_RATIO_RANGE[0] <= ratio <= CONTACT_RATIO_RANGE[1]:
        reason = "contact_mass_ratio"
    else:
        for stratum in STRATUM_ORDER:
            bounds = STRATUM_BOUNDS[stratum]
            rmsd_matches = _in_stratum(
                rmsd,
                bounds["rmsd_min"],
                bounds["rmsd_max"],
                bounds["rmsd_max_inclusive"],
            )
            jaccard_matches = _in_stratum(
                jaccard,
                bounds["jaccard_min"],
                bounds["jaccard_max"],
                bounds["jaccard_max_inclusive"],
            )
            if rmsd_matches and jaccard_matches:
                return {
                    "accepted": True,
                    "stratum": stratum,
                    "failure_reason": None,
                }
        reason = "outside_frozen_strata"
    return {"accepted": False, "stratum": None, "failure_reason": reason}


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
            f"{label} must contain exactly its schema fields "
            f"({', '.join(details)})"
        )


def _validate_transform(transform):
    _require_exact_fields(transform, TRANSFORM_FIELDS, "transform")
    probe = np.array([[0.0, 0.0, 0.0]])
    apply_rigid_transform(probe, transform)


def _transform_fingerprint(transform):
    return hashlib.sha256(
        json.dumps(transform, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _finite_stat(stats, field, label):
    try:
        raw_value = stats[field]
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


def _nonnegative_count(stats, field, label):
    try:
        value = stats[field]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} {field} must be a non-negative integer") from error
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{label} {field} must be a non-negative integer")
    return int(value)


def _minimum_count(decoy_set, field, default):
    value = decoy_set.get(field, default)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _validate_cached_statistics(contact_stats, overlap_stats):
    _require_exact_fields(contact_stats, CONTACT_STAT_FIELDS, "contact statistics")
    _require_exact_fields(overlap_stats, OVERLAP_STAT_FIELDS, "overlap statistics")
    native_mass = _finite_stat(
        contact_stats, "native_contact_mass", "contact"
    )
    decoy_mass = _finite_stat(
        contact_stats, "decoy_contact_mass", "contact"
    )
    ratio = _finite_stat(contact_stats, "contact_mass_ratio", "contact")
    jaccard = _finite_stat(contact_stats, "soft_jaccard", "contact")
    if native_mass <= 0.0 or decoy_mass <= 0.0:
        raise ValueError("contact masses must be positive")
    if not np.isclose(ratio, decoy_mass / native_mass, rtol=1e-7, atol=1e-9):
        raise ValueError("contact mass ratio does not match cached masses")
    if not 0.0 <= jaccard <= 1.0:
        raise ValueError("soft Jaccard must be in [0,1]")
    _nonnegative_count(contact_stats, "native_neighbor_count", "contact")
    _nonnegative_count(contact_stats, "decoy_neighbor_count", "contact")
    for field in ("coverage_retained", "contact_pattern_nonempty"):
        if not isinstance(contact_stats.get(field), (bool, np.bool_)):
            raise ValueError(f"contact {field} must be boolean")

    minimum_ratio = _finite_stat(
        overlap_stats, "minimum_distance_ratio", "overlap"
    )
    maximum_severity = _finite_stat(
        overlap_stats, "max_overlap_severity", "overlap"
    )
    summed_severity = _finite_stat(
        overlap_stats, "sum_overlap_severity", "overlap"
    )
    if minimum_ratio < 0.0:
        raise ValueError("minimum distance ratio must be non-negative")
    if not 0.0 <= maximum_severity <= 1.0:
        raise ValueError("maximum overlap severity must be in [0,1]")
    if summed_severity < maximum_severity:
        raise ValueError("summed overlap severity must include the maximum")
    expected_maximum = max(0.0, 1.0 - minimum_ratio)
    if not np.isclose(maximum_severity, expected_maximum, atol=1e-7):
        raise ValueError("maximum overlap severity is inconsistent")
    below_nine = _nonnegative_count(
        overlap_stats, "count_below_0_9", "overlap"
    )
    below_eight = _nonnegative_count(
        overlap_stats, "count_below_0_8", "overlap"
    )
    if below_eight > below_nine:
        raise ValueError("vdW threshold counts are inconsistent")
    hard_clash = overlap_stats.get("has_hard_clash")
    if not isinstance(hard_clash, (bool, np.bool_)):
        raise ValueError("vdW hard-clash flag must be boolean")
    if bool(hard_clash) != bool(minimum_ratio < HARD_VDW_RATIO):
        raise ValueError("vdW hard-clash flag is inconsistent")


def _prepare_native_geometry(record):
    if not isinstance(record, dict):
        raise ValueError("native record must be a dictionary")
    try:
        key = str(record["key"])
        native = _as_coordinates(record["ligand_coords"], "ligand_coords")
        protein = _as_coordinates(record["protein_coords"], "protein_coords")
        automorphisms = _validate_automorphisms(
            record["ligand_automorphisms"], len(native)
        )
        ligand_atomic_numbers = np.asarray(
            record["ligand_atomic_numbers"]
        ).reshape(-1)
        protein_symbols = np.asarray(
            record["protein_element_symbol"]
        ).reshape(-1)
    except KeyError as error:
        raise ValueError(f"native record is missing {error.args[0]}") from error
    if len(ligand_atomic_numbers) != len(native):
        raise ValueError("ligand atomic numbers are not aligned")
    if len(protein_symbols) != len(protein):
        raise ValueError("protein element symbols are not aligned")
    return {
        "key": key,
        "native": native,
        "protein": protein,
        "automorphisms": automorphisms,
        "ligand_atomic_numbers": ligand_atomic_numbers,
        "protein_symbols": protein_symbols,
    }


def _geometry_values_match(cached, recomputed):
    if isinstance(recomputed, (bool, np.bool_)):
        return isinstance(cached, (bool, np.bool_)) and bool(cached) == bool(recomputed)
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
    for field, recomputed_value in recomputed.items():
        if not _geometry_values_match(cached[field], recomputed_value):
            raise ValueError(
                f"decoy geometry {label} {field} does not match native geometry"
            )


def validate_decoy_set(decoy_set, native_record=None):
    if not isinstance(decoy_set, dict):
        raise ValueError("decoy set must be a dictionary")
    if decoy_set.get("schema_version") != DECOY_SCHEMA_VERSION:
        raise ValueError("unsupported decoy schema version")
    decoys = decoy_set.get("decoys")
    if not isinstance(decoys, list):
        raise ValueError("decoys must be a list")
    native_contact_mass = _finite_stat(
        decoy_set, "native_contact_mass", "decoy set"
    )
    if native_contact_mass <= 0.0:
        raise ValueError("native contact mass must be positive")

    native_geometry = None
    if native_record is not None:
        native_geometry = _prepare_native_geometry(native_record)
        if str(decoy_set.get("key")) != native_geometry["key"]:
            raise ValueError("decoy-set key does not match native geometry key")
        recomputed_native_mass = _native_contact_mass(
            native_geometry["protein"], native_geometry["native"]
        )
        if not _geometry_values_match(
            native_contact_mass, recomputed_native_mass
        ):
            raise ValueError(
                "decoy-set native contact mass does not match native geometry"
            )

    required_total = max(
        MINIMUM_DECOYS,
        _minimum_count(
            decoy_set, "minimum_total", MINIMUM_DECOYS
        ),
    )
    required_per_stratum = max(
        MINIMUM_PER_STRATUM,
        _minimum_count(
            decoy_set, "minimum_per_stratum", MINIMUM_PER_STRATUM
        ),
    )
    if len(decoys) < required_total:
        raise ValueError(f"decoy set must contain at least {required_total} decoys")

    counts = Counter()
    fingerprints = set()
    for entry in decoys:
        if not isinstance(entry, dict):
            raise ValueError("each decoy must be a dictionary")
        if set(entry) != DECOY_ENTRY_FIELDS:
            raise ValueError(
                "decoy entries must contain transform metadata and statistics only"
            )
        if entry["failure_reason"] is not None:
            raise ValueError("accepted decoy failure reason must be None")
        transform = entry["transform"]
        _validate_transform(transform)
        fingerprint = _transform_fingerprint(transform)
        if fingerprint in fingerprints:
            raise ValueError("decoy set contains a duplicate transform")
        fingerprints.add(fingerprint)
        contact_stats = entry["contact_stats"]
        overlap_stats = entry["overlap_stats"]
        _validate_cached_statistics(contact_stats, overlap_stats)
        if not _geometry_values_match(
            contact_stats["native_contact_mass"], native_contact_mass
        ):
            raise ValueError(
                "decoy native contact mass does not match decoy-set native contact mass"
            )
        rmsd_sym = _finite_stat(entry, "rmsd_sym", "decoy")
        classification = classify_decoy(
            rmsd_sym,
            contact_stats,
            overlap_stats,
        )
        if not classification["accepted"]:
            raise ValueError(
                "decoy violates frozen geometry: "
                + str(classification["failure_reason"])
            )
        if entry["stratum"] != classification["stratum"]:
            raise ValueError("decoy stratum does not match frozen geometry")

        if native_geometry is not None:
            transformed = apply_rigid_transform(
                native_geometry["native"], transform
            )
            recomputed_rmsd = symmetry_corrected_rmsd(
                native_geometry["native"],
                transformed,
                native_geometry["automorphisms"],
            )
            recomputed_contact_stats = soft_contact_statistics(
                native_geometry["protein"],
                native_geometry["native"],
                transformed,
            )
            recomputed_overlap_stats = vdw_overlap_statistics(
                native_geometry["protein"],
                native_geometry["protein_symbols"],
                transformed,
                native_geometry["ligand_atomic_numbers"],
            )
            if not _geometry_values_match(rmsd_sym, recomputed_rmsd):
                raise ValueError(
                    "decoy geometry RMSD does not match native geometry"
                )
            _verify_geometry_statistics(
                contact_stats, recomputed_contact_stats, "contact statistic"
            )
            _verify_geometry_statistics(
                overlap_stats, recomputed_overlap_stats, "overlap statistic"
            )
            recomputed_classification = classify_decoy(
                recomputed_rmsd,
                recomputed_contact_stats,
                recomputed_overlap_stats,
            )
            if (
                not recomputed_classification["accepted"]
                or recomputed_classification["stratum"] != entry["stratum"]
            ):
                raise ValueError(
                    "decoy stratum does not match recomputed native geometry"
                )
        counts[entry["stratum"]] += 1
    for stratum in STRATUM_ORDER:
        if counts[stratum] < required_per_stratum:
            raise ValueError(
                f"decoy set requires at least {required_per_stratum} {stratum} decoys"
            )
    return decoy_set


def _stable_choice_index(key, epoch, seed, stratum, count):
    digest = hashlib.sha256(
        f"{int(seed)}\0{int(epoch)}\0{key}\0{stratum}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % count


def select_epoch_decoys(decoy_set, key, epoch, seed):
    decoys = (
        decoy_set.get("decoys")
        if isinstance(decoy_set, dict)
        else decoy_set
    )
    if not isinstance(decoys, (list, tuple)):
        raise ValueError("decoys must be a sequence")
    selected = []
    for stratum in STRATUM_ORDER:
        candidates = [
            entry for entry in decoys if entry.get("stratum") == stratum
        ]
        if not candidates:
            raise ValueError(f"missing stratum {stratum}")
        index = _stable_choice_index(key, epoch, seed, stratum, len(candidates))
        selected.append(candidates[index])
    return selected


def _native_contact_mass(protein_coords, ligand_coords):
    return soft_contact_statistics(
        protein_coords, ligand_coords, ligand_coords
    )["native_contact_mass"]


def _pocket_outward_direction(protein_coords, ligand_coords):
    distances = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :], axis=-1
    )
    weights = continuous_contact_weights(distances).astype(np.float64)
    contact_mass = float(np.sum(weights))
    if contact_mass <= 0.0:
        raise ValueError("native contact mass must be positive")
    pocket_center = np.sum(
        protein_coords[:, None, :] * weights[:, :, None], axis=(0, 1)
    ) / contact_mass
    direction = np.mean(ligand_coords, axis=0) - pocket_center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return direction / norm


def _make_keyed_rng(seed, key):
    digest = hashlib.sha256(
        f"{int(seed)}\0{str(key)}".encode("utf-8")
    ).digest()
    entropy = int.from_bytes(digest, byteorder="big", signed=False)
    return np.random.default_rng(entropy)


def generate_decoy_set(record, seed):
    native = _as_coordinates(record["ligand_coords"], "ligand_coords")
    protein = _as_coordinates(record["protein_coords"], "protein_coords")
    automorphisms = _validate_automorphisms(
        record["ligand_automorphisms"], len(native)
    )
    ligand_atomic_numbers = np.asarray(
        record["ligand_atomic_numbers"]
    ).reshape(-1)
    if len(ligand_atomic_numbers) != len(native):
        raise ValueError("ligand atomic numbers are not aligned")
    protein_symbols = np.asarray(record["protein_element_symbol"]).reshape(-1)
    if len(protein_symbols) != len(protein):
        raise ValueError("protein element symbols are not aligned")

    native_mass = _native_contact_mass(protein, native)
    minimum_native_mass = max(8.0, 0.5 * len(native))
    if native_mass < minimum_native_mass:
        raise ValueError(
            f"native contact mass {native_mass:.6g} is below "
            f"{minimum_native_mass:.6g}"
        )

    outward_direction = _pocket_outward_direction(protein, native)
    generator = _make_keyed_rng(seed, record["key"])
    decoys = []
    fingerprints = set()
    rejection_counts = Counter()
    for target_stratum in STRATUM_ORDER:
        proposal = PROPOSAL_RANGES[target_stratum]
        accepted_for_stratum = []
        candidate_budget = MINIMUM_PER_STRATUM * MAX_CANDIDATE_RETRIES
        for candidate_index in range(candidate_budget):
            transformed, transform = random_rigid_transform(
                native,
                rng=generator,
                translation_distance_range=(0.0, 0.0),
                rotation_degrees_range=proposal["rotation_degrees"],
            )
            random_direction = _random_unit_vector(generator)
            concentration = (0.5, 1.0, 2.0, 4.0)[candidate_index % 4]
            if outward_direction is None:
                translation_direction = random_direction
            else:
                translation_direction = (
                    random_direction + concentration * outward_direction
                )
                translation_direction /= np.linalg.norm(translation_direction)
            translation_distance = generator.uniform(*proposal["translation"])
            transform["translation"] = (
                translation_direction * translation_distance
            ).astype(float).tolist()
            transformed = apply_rigid_transform(native, transform)
            rmsd = symmetry_corrected_rmsd(
                native, transformed, automorphisms
            )
            contact_stats = soft_contact_statistics(
                protein, native, transformed
            )
            overlap_stats = vdw_overlap_statistics(
                protein,
                protein_symbols,
                transformed,
                ligand_atomic_numbers,
            )
            classification = classify_decoy(
                rmsd, contact_stats, overlap_stats
            )
            if not classification["accepted"]:
                rejection_counts[classification["failure_reason"]] += 1
                continue
            if classification["stratum"] != target_stratum:
                rejection_counts["wrong_stratum"] += 1
                continue
            fingerprint = _transform_fingerprint(transform)
            if fingerprint in fingerprints:
                rejection_counts["duplicate_transform"] += 1
                continue
            fingerprints.add(fingerprint)
            accepted_for_stratum.append(
                {
                    "transform": transform,
                    "rmsd_sym": float(rmsd),
                    "contact_stats": contact_stats,
                    "overlap_stats": overlap_stats,
                    "stratum": target_stratum,
                    "failure_reason": None,
                }
            )
            if len(accepted_for_stratum) == MINIMUM_PER_STRATUM:
                break
        if len(accepted_for_stratum) < MINIMUM_PER_STRATUM:
            raise DecoyGenerationError(
                "failed to generate required "
                f"{target_stratum} decoys within "
                f"{candidate_budget} candidates "
                f"({MAX_CANDIDATE_RETRIES} per required transform)",
                rejection_counts,
            )
        decoys.extend(accepted_for_stratum)

    decoy_set = {
        "schema_version": DECOY_SCHEMA_VERSION,
        "key": str(record["key"]),
        "seed": int(seed),
        "minimum_total": MINIMUM_DECOYS,
        "minimum_per_stratum": MINIMUM_PER_STRATUM,
        "native_contact_mass": float(native_mass),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "max_candidate_retries_per_transform": MAX_CANDIDATE_RETRIES,
        "candidate_budget_per_stratum": candidate_budget,
        "decoys": decoys,
    }
    return validate_decoy_set(decoy_set, native_record=record)
