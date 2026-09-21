import numpy as np

from data_utils import local_interface_decoys_v3 as v3


PROBE_SCHEMA = "rotation_feasibility_probe_v4b"
TARGET_STRATUM = "high"
MAX_CENTROID_MOTION = 1.5
SLIDE_GRID = tuple(float(index) * 0.15 for index in range(11))
BASE_ROTATIONS = v3.BASE_ROTATIONS_PER_STRATUM
_PIVOT_RTOL = 1e-10
_PIVOT_ATOL = 1e-12


def _normalize_direction(vector):
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm < v3.ZERO_DISTANCE_EPSILON:
        return None
    return value / norm


def _least_aligned_basis(normal):
    unit = _normalize_direction(normal)
    if unit is None:
        raise ValueError("outward direction must be a finite nonzero vector")
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(unit)))]
    first = _normalize_direction(np.cross(unit, axis))
    second = _normalize_direction(np.cross(unit, first))
    if first is None or second is None:
        raise ValueError("failed to construct tangent basis")
    return first, second


def iter_v3_high_rotation_seeds(record, seed):
    if not isinstance(record, dict) or "key" not in record:
        raise ValueError("record must contain key")
    native = v3._as_coordinates(record.get("ligand_coords"), "ligand_coords")
    center = np.mean(native, axis=0)
    generator = v3._make_keyed_rng(
        seed, record["key"], TARGET_STRATUM, "proposal"
    )
    for base_rotation_id in range(BASE_ROTATIONS):
        raw_axis = v3._random_unit_vector(generator)
        raw_degrees = float(
            generator.uniform(*v3.ROTATION_RANGES[TARGET_STRATUM])
        )
        raw_rotation_matrix = v3._rodrigues_rotation(
            raw_axis, np.deg2rad(raw_degrees)
        )
        yield {
            "base_rotation_id": int(base_rotation_id),
            "center": center.astype(float).tolist(),
            "raw_axis": raw_axis.astype(float).tolist(),
            "raw_rotation_degrees": raw_degrees,
            "raw_rotation_matrix": raw_rotation_matrix.astype(float).tolist(),
            "v3_base_rotation_fingerprint": v3._base_rotation_fingerprint(
                center, raw_rotation_matrix
            ),
        }


def native_outward_direction(native_ligand, protein, seed, key):
    ligand = v3._as_coordinates(native_ligand, "native_ligand")
    protein_coords = v3._as_coordinates(protein, "protein")
    direction = v3._contact_centroid_direction(ligand, protein_coords)
    if direction is not None:
        return direction.astype(np.float64)
    fallback = _normalize_direction(np.mean(ligand, axis=0) - np.mean(protein_coords, axis=0))
    if fallback is not None:
        return fallback
    return v3._hashed_signed_cartesian_axis(
        (int(seed), str(key), TARGET_STRATUM, "native_outward")
    )


def tangent_axis(raw_axis, outward, namespace):
    unit_axis = _normalize_direction(raw_axis)
    normal = _normalize_direction(outward)
    if unit_axis is None or normal is None:
        raise ValueError("raw axis and outward direction must be finite nonzero vectors")
    projected = _normalize_direction(unit_axis - float(np.dot(unit_axis, normal)) * normal)
    if projected is not None:
        return projected
    first, second = _least_aligned_basis(normal)
    first_dot = float(np.dot(unit_axis, first))
    second_dot = float(np.dot(unit_axis, second))
    use_first = abs(first_dot) >= abs(second_dot)
    basis = first if use_first else second
    signed_dot = first_dot if use_first else second_dot
    if abs(signed_dot) < v3.ZERO_DISTANCE_EPSILON:
        fallback = v3._hashed_signed_cartesian_axis(namespace)
        signed_dot = float(np.dot(fallback, basis))
    return basis if signed_dot >= 0.0 else -basis


def _append_pivot(rows, pivot_id, coordinates, pivot_atom_index=None):
    pivot = np.asarray(coordinates, dtype=np.float64)
    if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
        return False
    if any(
        np.allclose(
            pivot,
            np.asarray(row["pivot"], dtype=np.float64),
            rtol=_PIVOT_RTOL,
            atol=_PIVOT_ATOL,
        )
        for row in rows
    ):
        return False
    rows.append(
        {
            "pivot_id": str(pivot_id),
            "pivot": pivot.astype(float).tolist(),
            "pivot_atom_index": pivot_atom_index,
        }
    )
    return True


def build_pivot_rows(native_ligand, protein, outward):
    ligand = v3._as_coordinates(native_ligand, "native_ligand")
    protein_coords = v3._as_coordinates(protein, "protein")
    normal = _normalize_direction(outward)
    if normal is None:
        raise ValueError("outward direction must be a finite nonzero vector")

    rows = []
    centroid = np.mean(ligand, axis=0)
    _append_pivot(rows, "ligand_centroid", centroid)

    distances = np.linalg.norm(
        ligand[:, None, :] - protein_coords[None, :, :], axis=-1
    )
    weights = 0.5 * (np.cos(np.pi * (distances - 5.0)) + 1.0)
    weights[distances <= 5.0] = 1.0
    weights[distances >= 6.0] = 0.0
    ligand_weights = np.sum(weights, axis=1)
    total_weight = float(np.sum(ligand_weights))
    if total_weight >= v3.ZERO_DISTANCE_EPSILON:
        contact_anchor = np.sum(ligand * ligand_weights[:, None], axis=0) / total_weight
        _append_pivot(rows, "contact_anchor", contact_anchor)

    projections = (ligand - centroid) @ normal
    exposed_atom_index = int(np.argmax(projections))
    _append_pivot(
        rows,
        "exposed_atom",
        ligand[exposed_atom_index],
        pivot_atom_index=exposed_atom_index,
    )
    return rows



def build_rotation_variants(native_ligand, protein, family_seed, outward):
    if not isinstance(family_seed, dict):
        raise ValueError("family seed must be a dictionary")
    required = {
        "base_rotation_id",
        "raw_axis",
        "raw_rotation_degrees",
        "v3_base_rotation_fingerprint",
    }
    if not required.issubset(family_seed):
        raise ValueError("family seed is incomplete")
    ligand = v3._as_coordinates(native_ligand, "native_ligand")
    protein_coords = v3._as_coordinates(protein, "protein")
    normal = _normalize_direction(outward)
    if normal is None:
        raise ValueError("outward direction must be a finite nonzero vector")

    axis = tangent_axis(
        family_seed["raw_axis"],
        normal,
        (int(family_seed["base_rotation_id"]), "tangent_axis"),
    )
    matrix = v3._rodrigues_rotation(
        axis,
        np.deg2rad(float(family_seed["raw_rotation_degrees"])),
    )
    variants = []
    for rotation_variant_id, pivot_row in enumerate(
        build_pivot_rows(ligand, protein_coords, normal)
    ):
        pivot = np.asarray(pivot_row["pivot"], dtype=np.float64)
        transform = {
            "center": pivot.astype(float).tolist(),
            "rotation_matrix": matrix.astype(float).tolist(),
            "translation": [0.0, 0.0, 0.0],
        }
        variants.append(
            {
                **family_seed,
                **pivot_row,
                "rotation_variant_id": int(rotation_variant_id),
                "rotation_source": "tangent_axis",
                "tangent_axis": axis.astype(float).tolist(),
                "rotation_matrix": matrix.astype(float).tolist(),
                "rotation_variant_fingerprint": v3._transform_fingerprint(transform),
            }
        )
    return variants


def build_coupled_proposals(native_ligand, outward, variant):
    if not isinstance(variant, dict):
        raise ValueError("rotation variant must be a dictionary")
    required = {
        "base_rotation_id",
        "v3_base_rotation_fingerprint",
        "rotation_variant_id",
        "pivot",
        "rotation_matrix",
        "tangent_axis",
    }
    if not required.issubset(variant):
        raise ValueError("rotation variant is incomplete")
    native = v3._as_coordinates(native_ligand, "native_ligand")
    normal = _normalize_direction(outward)
    if normal is None:
        raise ValueError("outward direction must be a finite nonzero vector")
    pivot = np.asarray(variant["pivot"], dtype=np.float64)
    base_transform = {
        "center": pivot.astype(float).tolist(),
        "rotation_matrix": variant["rotation_matrix"],
        "translation": [0.0, 0.0, 0.0],
    }
    rotated = v3.apply_rigid_transform(native, base_transform)
    rows = []
    for slide_index, slide_distance in enumerate(SLIDE_GRID):
        translation = normal * float(slide_distance)
        transformed = rotated + translation
        transform = {
            "center": pivot.astype(float).tolist(),
            "rotation_matrix": variant["rotation_matrix"],
            "translation": translation.astype(float).tolist(),
        }
        rows.append(
            {
                "transform": transform,
                "transform_fingerprint": v3._transform_fingerprint(transform),
                "base_rotation_id": int(variant["base_rotation_id"]),
                "base_rotation_fingerprint": str(variant["v3_base_rotation_fingerprint"]),
                "translation_index": int(slide_index),
                "translation_distance": float(slide_distance),
                "translation_direction": normal.astype(float).tolist(),
                "direction_source": "outward_rotate_slide",
                "zero_distance_pair_count": 0,
                "zero_translation_vdw_safe": False,
                "rotation_variant_id": int(variant["rotation_variant_id"]),
                "rotation_source": str(variant["rotation_source"]),
                "tangent_axis": list(variant["tangent_axis"]),
                "pivot_id": str(variant["pivot_id"]),
                "pivot": pivot.astype(float).tolist(),
                "pivot_atom_index": variant.get("pivot_atom_index"),
                "slide_index": int(slide_index),
                "slide_distance": float(slide_distance),
                "motion_limit_pass": bool(
                    np.linalg.norm(
                        np.mean(transformed, axis=0) - np.mean(native, axis=0)
                    )
                    <= MAX_CENTROID_MOTION + 1e-12
                ),
            }
        )
    return rows


def _evaluate_variant_proposals(
    native_geometry, outward, variant, seen_fingerprints, full_evaluator
):
    from data_utils import local_interface_translation_probe_v4 as v4

    proposals = build_coupled_proposals(
        native_geometry["native"], outward, variant
    )
    valid = [row for row in proposals if row["motion_limit_pass"]]
    outcomes = [
        {
            "proposal": row,
            "fast_vdw_safe": False,
            "candidate": None,
            "physical_reason": "centroid_motion_limit",
            "match_reason": None,
        }
        for row in proposals
        if not row["motion_limit_pass"]
    ]
    if not valid:
        return outcomes

    pivot = np.asarray(variant["pivot"], dtype=np.float64)
    rotated = v3.apply_rigid_transform(
        native_geometry["native"],
        {
            "center": pivot.astype(float).tolist(),
            "rotation_matrix": variant["rotation_matrix"],
            "translation": [0.0, 0.0, 0.0],
        },
    )
    translations = np.asarray(
        [row["transform"]["translation"] for row in valid],
        dtype=np.float64,
    )
    safe = v4.fast_vdw_safe_mask(
        rotated,
        native_geometry["protein"],
        native_geometry["protein_symbols"],
        native_geometry["ligand_atomic_numbers"],
        translations,
    )
    for proposal, is_safe in zip(valid, safe):
        if not is_safe:
            candidate = None
            physical_reason = "vdw_hard_clash"
            match_reason = None
        else:
            candidate, physical_reason, match_reason = full_evaluator(
                native_geometry,
                proposal,
                TARGET_STRATUM,
                seen_fingerprints,
            )
            if candidate is not None:
                candidate = dict(candidate)
                for field in (
                    "rotation_variant_id",
                    "rotation_source",
                    "tangent_axis",
                    "pivot_id",
                    "pivot",
                    "pivot_atom_index",
                    "slide_index",
                    "slide_distance",
                ):
                    candidate[field] = proposal[field]
        outcomes.append(
            {
                "proposal": proposal,
                "fast_vdw_safe": bool(is_safe),
                "candidate": candidate,
                "physical_reason": physical_reason,
                "match_reason": match_reason,
            }
        )
    return outcomes


def _rejection_counts(outcomes):
    counts = {}
    for outcome in outcomes:
        reason = outcome["physical_reason"]
        if reason is None and outcome["candidate"] is not None:
            reason = outcome["match_reason"]
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _select_physical_representative(candidates):
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            int(row["rotation_variant_id"]),
            int(row["slide_index"]),
            str(row["transform_fingerprint"]),
        ),
    )


def _select_matched_representative(candidates):
    matched = [row for row in candidates if row.get("match_pass") is True]
    if not matched:
        return None
    return min(
        matched,
        key=lambda row: (
            float(row["match_distance"]),
            int(row["rotation_variant_id"]),
            int(row["slide_index"]),
            str(row["transform_fingerprint"]),
        ),
    )


def evaluate_rotation_family(
    native_geometry, family_seed, full_evaluator=v3._evaluate_proposal, outward=None
):
    native = v3._as_coordinates(native_geometry["native"], "native")
    protein = v3._as_coordinates(native_geometry["protein"], "protein")
    if outward is None:
        outward = native_outward_direction(
            native,
            protein,
            seed=0,
            key="rotation_probe",
        )
    else:
        outward = _normalize_direction(outward)
        if outward is None:
            raise ValueError("outward direction must be a finite nonzero vector")
    variants = build_rotation_variants(native, protein, family_seed, outward)
    seen_fingerprints = set()
    outcomes = []
    variant_summaries = []
    for variant in variants:
        variant_outcomes = _evaluate_variant_proposals(
            native_geometry,
            outward,
            variant,
            seen_fingerprints,
            full_evaluator,
        )
        outcomes.extend(variant_outcomes)
        variant_candidates = [
            row["candidate"]
            for row in variant_outcomes
            if row["candidate"] is not None
        ]
        variant_matched = [
            row
            for row in variant_candidates
            if row.get("match_pass") is True
        ]
        variant_summaries.append(
            {
                **variant,
                "counts": {
                    "proposed": len(variant_outcomes),
                    "centroid_motion_limit": sum(
                        row["physical_reason"] == "centroid_motion_limit"
                        for row in variant_outcomes
                    ),
                    "vdw_safe": sum(
                        row["fast_vdw_safe"] for row in variant_outcomes
                    ),
                    "physical": len(variant_candidates),
                    "matched": len(variant_matched),
                },
                "rejection_counts": _rejection_counts(variant_outcomes),
                "any_vdw_safe": any(
                    row["fast_vdw_safe"] for row in variant_outcomes
                ),
                "any_physical": bool(variant_candidates),
                "any_matched": bool(variant_matched),
            }
        )
    candidates = [
        outcome["candidate"]
        for outcome in outcomes
        if outcome["candidate"] is not None
    ]
    matched = [row for row in candidates if row.get("match_pass") is True]
    return {
        "base_rotation_id": int(family_seed["base_rotation_id"]),
        "v3_base_rotation_fingerprint": str(family_seed["v3_base_rotation_fingerprint"]),
        "raw_axis": list(family_seed["raw_axis"]),
        "raw_rotation_degrees": float(family_seed["raw_rotation_degrees"]),
        "outward_direction": outward.astype(float).tolist(),
        "variants": variant_summaries,
        "counts": {
            "variants": len(variants),
            "proposed": len(outcomes),
            "centroid_motion_limit": sum(
                outcome["physical_reason"] == "centroid_motion_limit"
                for outcome in outcomes
            ),
            "vdw_safe": sum(outcome["fast_vdw_safe"] for outcome in outcomes),
            "physical": len(candidates),
            "matched": len(matched),
        },
        "rejection_counts": _rejection_counts(outcomes),
        "any_vdw_safe": any(outcome["fast_vdw_safe"] for outcome in outcomes),
        "any_physical": bool(candidates),
        "any_matched": bool(matched),
        "best_physical": _select_physical_representative(candidates),
        "best_matched": _select_matched_representative(candidates),
    }



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
            if isinstance(value, bool) or not np.isfinite(float(value)):
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
                    "index": int(index),
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


def _family_state_counts(families):
    fields = ("any_vdw_safe", "any_physical", "any_matched")
    counts = {field: 0 for field in fields}
    for family in families:
        for field in fields:
            counts[field] += int(bool(family[field]))
    return counts


def probe_record(record, seed, rotation_limit=None):
    if rotation_limit is not None:
        if (
            isinstance(rotation_limit, bool)
            or not isinstance(rotation_limit, (int, np.integer))
            or not 1 <= int(rotation_limit) <= BASE_ROTATIONS
        ):
            raise ValueError("rotation_limit must be between 1 and 512")
        rotation_limit = int(rotation_limit)
    native_geometry = v3._prepare_eligible_native_geometry(
        record,
        native_hard_clash_as_generation_error=False,
    )
    native = native_geometry["native"]
    protein = native_geometry["protein"]
    native_contact = v3.soft_contact_statistics(protein, native, native)
    outward = native_outward_direction(native, protein, seed, record["key"])
    families = []
    rejection_counts = {}
    for index, family_seed in enumerate(
        iter_v3_high_rotation_seeds(record, seed)
    ):
        if rotation_limit is not None and index >= rotation_limit:
            break
        family = evaluate_rotation_family(
            native_geometry,
            family_seed,
            outward=outward,
        )
        families.append(family)
        for reason, count in family["rejection_counts"].items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + int(count)
    value = {
        "schema": PROBE_SCHEMA,
        "key": str(record["key"]),
        "seed": int(seed),
        "target_stratum": TARGET_STRATUM,
        "rotation_count": len(families),
        "decision_eligible": rotation_limit is None,
        "sample_properties": {
            "ligand_size": int(len(native)),
            "pocket_size": int(len(protein)),
            "native_contact_mass": float(native_contact["native_contact_mass"]),
        },
        "families": families,
        "family_counts": _family_state_counts(families),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    validate_probe_record(
        value,
        expected_key=str(record["key"]),
        expected_rotation_count=(
            BASE_ROTATIONS if rotation_limit is None else rotation_limit
        ),
    )
    return value


def validate_probe_record(
    value, expected_key=None, expected_rotation_count=BASE_ROTATIONS
):
    if not isinstance(value, dict):
        raise ValueError("probe record must be a dictionary")
    required = {
        "schema",
        "key",
        "seed",
        "target_stratum",
        "rotation_count",
        "decision_eligible",
        "sample_properties",
        "families",
        "family_counts",
        "rejection_counts",
    }
    if not required.issubset(value):
        raise ValueError("probe record is incomplete")
    if value["schema"] != PROBE_SCHEMA:
        raise ValueError("probe record schema is invalid")
    if expected_key is not None and value["key"] != str(expected_key):
        raise ValueError("probe record key does not match")
    if value["target_stratum"] != TARGET_STRATUM:
        raise ValueError("probe record target stratum is invalid")
    rotation_count = value["rotation_count"]
    if (
        isinstance(rotation_count, bool)
        or not isinstance(rotation_count, (int, np.integer))
        or int(rotation_count) != int(expected_rotation_count)
    ):
        raise ValueError("probe record rotation count does not match")
    if value["decision_eligible"] and rotation_count != BASE_ROTATIONS:
        raise ValueError("decision-eligible probe record requires 512 rotations")
    families = value["families"]
    if not isinstance(families, list) or len(families) != rotation_count:
        raise ValueError("probe record families do not match rotation count")
    identifiers = [int(row["base_rotation_id"]) for row in families]
    if identifiers != list(range(rotation_count)):
        raise ValueError("probe record rotation ids must be contiguous")
    expected_counts = _family_state_counts(families)
    if value["family_counts"] != expected_counts:
        raise ValueError("probe record family counts do not match families")
    for metric in RANK_METRICS:
        metric_value = value["sample_properties"].get(metric)
        if isinstance(metric_value, bool) or not np.isfinite(float(metric_value)):
            raise ValueError(metric + " must be finite")


def aggregate_probe_report(
    records,
    requested_keys,
    native_exclusions,
    v4_report=None,
    decision_eligible=True,
):
    requested = [str(key) for key in requested_keys]
    if len(requested) != len(set(requested)):
        raise ValueError("requested keys must be unique")
    exclusions = list(native_exclusions)
    exclusion_keys = [str(row["key"]) for row in exclusions]
    if len(exclusion_keys) != len(set(exclusion_keys)):
        raise ValueError("native exclusion keys must be unique")
    requested_set = set(requested)
    if not set(exclusion_keys).issubset(requested_set):
        raise ValueError("native exclusions are outside requested keys")
    record_by_key = {str(row["key"]): row for row in records}
    if len(record_by_key) != len(records):
        raise ValueError("probe record keys must be unique")
    eligible_keys = [key for key in requested if key not in set(exclusion_keys)]
    if set(record_by_key) != set(eligible_keys):
        raise ValueError("probe records do not exactly cover eligible keys")
    ordered_records = [record_by_key[key] for key in eligible_keys]

    family_fields = (
        "any_vdw_safe",
        "any_physical",
        "any_matched",
        "proposed",
        "centroid_motion_limit",
        "vdw_safe",
        "physical",
        "matched",
    )
    family_totals = {field: 0 for field in family_fields}
    rejection_counts = {}
    pivot_totals = {}
    total_rotation_families = 0
    physical_complete_keys = []
    matched_complete_keys = []
    for record in ordered_records:
        total_rotation_families += int(record["rotation_count"])
        for field in ("any_vdw_safe", "any_physical", "any_matched"):
            family_totals[field] += int(record["family_counts"][field])
        for family in record["families"]:
            for field in (
                "proposed",
                "centroid_motion_limit",
                "vdw_safe",
                "physical",
                "matched",
            ):
                family_totals[field] += int(family["counts"][field])
            for variant in family["variants"]:
                pivot_id = str(variant["pivot_id"])
                summary = pivot_totals.setdefault(
                    pivot_id,
                    {
                        "variants": 0,
                        "proposed": 0,
                        "centroid_motion_limit": 0,
                        "vdw_safe": 0,
                        "physical": 0,
                        "matched": 0,
                        "rejection_counts": {},
                    },
                )
                summary["variants"] += 1
                for field in (
                    "proposed",
                    "centroid_motion_limit",
                    "vdw_safe",
                    "physical",
                    "matched",
                ):
                    summary[field] += int(variant["counts"][field])
                for reason, count in variant["rejection_counts"].items():
                    reasons = summary["rejection_counts"]
                    reasons[reason] = reasons.get(reason, 0) + int(count)
        for reason, count in record["rejection_counts"].items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + int(count)
        if int(record["family_counts"]["any_physical"]) >= 4:
            physical_complete_keys.append(record["key"])
        if int(record["family_counts"]["any_matched"]) >= 4:
            matched_complete_keys.append(record["key"])
    complete_keys = set(matched_complete_keys)
    report = {
        "status": "complete",
        "decision_eligible": bool(decision_eligible),
        "requested_count": len(requested),
        "eligible_count": len(eligible_keys),
        "native_excluded_count": len(exclusion_keys),
        "requested_keys": requested,
        "eligible_keys": eligible_keys,
        "native_exclusions": exclusions,
        "total_rotation_families": total_rotation_families,
        "family_totals": family_totals,
        "pivot_totals": {
            key: {
                **value,
                "rejection_counts": dict(
                    sorted(value["rejection_counts"].items())
                ),
            }
            for key, value in sorted(pivot_totals.items())
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "sample_completion": {
            "physical_complete_count": len(physical_complete_keys),
            "physical_complete_keys": physical_complete_keys,
            "matched_complete_count": len(matched_complete_keys),
            "matched_complete_keys": matched_complete_keys,
        },
        "completion_rank_bins": _completion_rank_bins(
            ordered_records,
            complete_keys,
        ),
    }
    if v4_report is not None:
        report["v4_comparison"] = {
            "matched_complete_count": v4_report["sample_completion"]["matched_complete_count"],
            "expanded_no_vdw_safe_fraction": v4_report["decision"]["metrics"]["expanded_no_vdw_safe_fraction"],
        }
    return report


def evaluate_probe_decision(report):
    if report.get("decision_eligible") is False:
        return None
    total = int(report["total_rotation_families"])
    if total <= 0:
        raise ValueError("total rotation families must be positive")
    safe = int(report["family_totals"]["any_vdw_safe"])
    no_vdw_safe_fraction = 1.0 - float(safe) / total
    matched_complete = int(report["sample_completion"]["matched_complete_count"])
    reasons = []
    if no_vdw_safe_fraction > DECISION_THRESHOLDS[
        "maximum_no_vdw_safe_family_fraction"
    ]:
        reasons.append("no_vdw_safe_family_fraction_above_0_90")
    if matched_complete < DECISION_THRESHOLDS["minimum_matched_complete"]:
        reasons.append("matched_complete_count_below_20")
    for metric in RANK_METRICS:
        summary = report["completion_rank_bins"][metric]
        for row in summary["bins"]:
            if row["matched_complete_count"] < DECISION_THRESHOLDS[
                "minimum_per_rank_quartile"
            ]:
                reasons.append(metric + "_quartile_%d_has_no_completion" % row["index"])
        if summary["upper_half_matched_complete_count"] < DECISION_THRESHOLDS[
            "minimum_upper_half_complete"
        ]:
            reasons.append(metric + "_upper_half_below_4")
    if "no_vdw_safe_family_fraction_above_0_90" in reasons:
        outcome = "rotation_probe_no_go_vdw"
    elif reasons:
        outcome = "rotation_probe_no_go_completion"
    else:
        outcome = "rotation_probe_pass"
    return {
        "passed": not reasons,
        "outcome": outcome,
        "reasons": sorted(reasons),
        "metrics": {
            "eligible_count": int(report["eligible_count"]),
            "matched_complete_count": matched_complete,
            "no_vdw_safe_family_fraction": no_vdw_safe_fraction,
        },
        "thresholds": dict(DECISION_THRESHOLDS),
    }
