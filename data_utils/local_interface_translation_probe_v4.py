import numpy as np

from data_utils import local_interface_decoys_v3 as v3


PROBE_SCHEMA = "translation_feasibility_probe_v4"
TARGET_STRATUM = "high"
HIGH_DISTANCE_GRID = tuple(float(index) * 0.15 for index in range(1, 11))
MAX_TRANSLATION = 1.5
MAX_GUIDED_DIRECTIONS = 22
FIBONACCI_DIRECTION_COUNT = 64
BASE_ROTATIONS = v3.BASE_ROTATIONS_PER_STRATUM
MAX_SEVERE_PAIR_DIRECTIONS = 8
MAX_LIGAND_CLUSTER_DIRECTIONS = 6
DIRECTION_RTOL = 1e-10
DIRECTION_ATOL = 1e-12


def _normalize_direction(vector):
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm < v3.ZERO_DISTANCE_EPSILON:
        return None
    return value / norm


def _append_unique_direction(rows, vector, source, **metadata):
    unit = _normalize_direction(vector)
    if unit is None:
        return False
    if any(
        np.allclose(
            np.asarray(row["direction"], dtype=np.float64),
            unit,
            rtol=DIRECTION_RTOL,
            atol=DIRECTION_ATOL,
        )
        for row in rows
    ):
        return False
    row = dict(metadata)
    row.update(
        {
            "direction_family_id": len(rows),
            "direction_source": str(source),
            "direction": unit.astype(float).tolist(),
            "search_stage": "guided",
        }
    )
    rows.append(row)
    return True


def _least_aligned_basis(normal):
    normal = _normalize_direction(normal)
    if normal is None:
        raise ValueError("outward normal must be a finite nonzero vector")
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    first = _normalize_direction(np.cross(normal, axis))
    second = _normalize_direction(np.cross(normal, first))
    if first is None or second is None:
        raise ValueError("failed to construct tangent basis")
    return first, second


def fibonacci_directions(count=FIBONACCI_DIRECTION_COUNT):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("Fibonacci direction count must be an integer")
    count = int(count)
    if count <= 0:
        raise ValueError("Fibonacci direction count must be positive")
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / float(count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = indices * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.column_stack(
        (radius * np.cos(angle), radius * np.sin(angle), z)
    )


def iter_v3_high_rotations(record, seed):
    if not isinstance(record, dict) or "key" not in record:
        raise ValueError("record must contain key")
    native = v3._as_coordinates(record.get("ligand_coords"), "ligand_coords")
    center = np.mean(native, axis=0)
    generator = v3._make_keyed_rng(
        seed, record["key"], TARGET_STRATUM, "proposal"
    )
    for rotation_id in range(BASE_ROTATIONS):
        axis = v3._random_unit_vector(generator)
        degrees = float(
            generator.uniform(*v3.ROTATION_RANGES[TARGET_STRATUM])
        )
        matrix = v3._rodrigues_rotation(axis, np.deg2rad(degrees))
        yield {
            "base_rotation_id": rotation_id,
            "base_rotation_fingerprint": v3._base_rotation_fingerprint(
                center, matrix
            ),
            "center": center.astype(float).tolist(),
            "axis": axis.astype(float).tolist(),
            "rotation_degrees": degrees,
            "rotation_matrix": matrix.astype(float).tolist(),
        }


def _selected_clash_pairs(
    ligand,
    protein,
    ligand_radii,
    protein_radii,
    namespace,
):
    distances, ratios = v3._vdw_pair_ratios(
        ligand, protein, ligand_radii, protein_radii
    )
    hard = ratios < v3.HARD_VDW_RATIO
    selected = hard if np.any(hard) else ratios < v3.REPULSION_NEAR_RATIO
    if not np.any(selected):
        return [], "near_contact"
    source = "hard_clash" if np.any(hard) else "near_contact"
    centroid = np.mean(ligand, axis=0)
    rows = []
    for ligand_index, protein_index in np.argwhere(selected):
        ligand_index = int(ligand_index)
        protein_index = int(protein_index)
        unit = v3._safe_pair_unit_vector(
            ligand[ligand_index] - protein[protein_index],
            centroid,
            protein[protein_index],
            (*namespace, ligand_index, protein_index),
        )
        rows.append(
            {
                "q": float(ratios[ligand_index, protein_index]),
                "ligand_atom_index": ligand_index,
                "protein_atom_index": protein_index,
                "direction": unit,
                "pair_source": source,
            }
        )
    rows.sort(
        key=lambda row: (
            row["q"],
            row["ligand_atom_index"],
            row["protein_atom_index"],
        )
    )
    return rows, source


def _cluster_directions(pair_rows):
    grouped = {}
    for row in pair_rows:
        grouped.setdefault(row["ligand_atom_index"], []).append(row)
    clusters = []
    for ligand_index, rows in grouped.items():
        force = np.zeros(3, dtype=np.float64)
        for row in rows:
            force += (v3.REPULSION_NEAR_RATIO - row["q"]) * row[
                "direction"
            ]
        direction = _normalize_direction(force)
        if direction is None:
            continue
        clusters.append(
            {
                "ligand_atom_index": int(ligand_index),
                "minimum_q": min(row["q"] for row in rows),
                "direction": direction,
            }
        )
    clusters.sort(
        key=lambda row: (row["minimum_q"], row["ligand_atom_index"])
    )
    return clusters


def build_guided_directions(
    rotated_ligand,
    native_ligand,
    protein,
    protein_symbols,
    ligand_atomic_numbers,
    axis,
    seed,
    key,
    base_rotation_id,
):
    rotated_ligand = v3._as_coordinates(rotated_ligand, "rotated_ligand")
    native_ligand = v3._as_coordinates(native_ligand, "native_ligand")
    protein = v3._as_coordinates(protein, "protein")
    protein_symbols = np.asarray(protein_symbols).reshape(-1)
    ligand_atomic_numbers = np.asarray(ligand_atomic_numbers).reshape(-1)
    axis = np.asarray(axis, dtype=np.float64)
    if len(protein_symbols) != len(protein):
        raise ValueError("protein symbols are not aligned to coordinates")
    if len(ligand_atomic_numbers) != len(rotated_ligand):
        raise ValueError("ligand atomic numbers are not aligned to coordinates")

    ligand_radii = v3._vdw_radii(
        ligand_atomic_numbers, "ligand_atomic_numbers"
    )
    protein_radii = v3._protein_vdw_radii(protein_symbols)
    namespace = (seed, key, TARGET_STRATUM, int(base_rotation_id))
    rows = []

    v3_direction, v3_source, zero_count = v3._proposal_direction(
        rotated_ligand,
        native_ligand,
        protein,
        protein_symbols,
        ligand_atomic_numbers,
        axis,
        MAX_TRANSLATION,
        seed,
        key,
        TARGET_STRATUM,
        int(base_rotation_id),
    )
    _append_unique_direction(
        rows,
        v3_direction,
        "v3_" + v3_source,
        zero_distance_pair_count=int(zero_count),
    )

    pair_rows, pair_source = _selected_clash_pairs(
        rotated_ligand,
        protein,
        ligand_radii,
        protein_radii,
        namespace,
    )
    for pair in pair_rows[:MAX_SEVERE_PAIR_DIRECTIONS]:
        _append_unique_direction(
            rows,
            pair["direction"],
            "severe_pair",
            pair_source=pair_source,
            ligand_atom_index=pair["ligand_atom_index"],
            protein_atom_index=pair["protein_atom_index"],
            q=pair["q"],
        )

    for cluster in _cluster_directions(pair_rows)[
        :MAX_LIGAND_CLUSTER_DIRECTIONS
    ]:
        _append_unique_direction(
            rows,
            cluster["direction"],
            "ligand_atom_cluster",
            pair_source=pair_source,
            ligand_atom_index=cluster["ligand_atom_index"],
            minimum_q=cluster["minimum_q"],
        )

    rotated_outward = v3._contact_centroid_direction(
        rotated_ligand, protein
    )
    native_outward = v3._contact_centroid_direction(native_ligand, protein)
    _append_unique_direction(
        rows, rotated_outward, "rotated_interface_outward"
    )
    _append_unique_direction(rows, native_outward, "native_interface_outward")

    outward = next(
        (
            value
            for value in (rotated_outward, native_outward, v3_direction)
            if _normalize_direction(value) is not None
        ),
        None,
    )
    if outward is not None:
        first, second = _least_aligned_basis(outward)
        for tangent_index, tangent in enumerate(
            (first, -first, second, -second)
        ):
            _append_unique_direction(
                rows,
                np.asarray(outward) + 0.5 * tangent,
                "outward_tangent",
                tangent_index=tangent_index,
            )

    signed_axis, sign_rank = v3._choose_signed_axis(
        rotated_ligand,
        protein,
        protein_symbols,
        ligand_atomic_numbers,
        axis,
        MAX_TRANSLATION,
    )
    _append_unique_direction(
        rows, signed_axis, "signed_rotation_axis", sign_rank=int(sign_rank)
    )
    return rows[:MAX_GUIDED_DIRECTIONS]



def _translation_array(value):
    translations = np.asarray(value, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must have shape [N,3]")
    if not np.all(np.isfinite(translations)):
        raise ValueError("translations must be finite")
    if np.any(np.linalg.norm(translations, axis=1) > MAX_TRANSLATION + 1e-12):
        raise ValueError("translation exceeds the high-stratum limit")
    return translations


def fast_vdw_safe_mask(
    rotated_ligand,
    protein,
    protein_symbols,
    ligand_atomic_numbers,
    translations,
    chunk_size=128,
):
    rotated_ligand = v3._as_coordinates(
        rotated_ligand, "rotated_ligand"
    )
    protein = v3._as_coordinates(protein, "protein")
    protein_symbols = np.asarray(protein_symbols).reshape(-1)
    ligand_atomic_numbers = np.asarray(ligand_atomic_numbers).reshape(-1)
    translations = _translation_array(translations)
    if len(protein_symbols) != len(protein):
        raise ValueError("protein symbols are not aligned to coordinates")
    if len(ligand_atomic_numbers) != len(rotated_ligand):
        raise ValueError("ligand atomic numbers are not aligned to coordinates")
    if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_size = int(chunk_size)

    ligand_radii = v3._vdw_radii(
        ligand_atomic_numbers, "ligand_atomic_numbers"
    )
    protein_radii = v3._protein_vdw_radii(protein_symbols)
    deltas = (
        rotated_ligand[:, None, :] - protein[None, :, :]
    )
    pair_distances = np.linalg.norm(deltas, axis=-1)
    hard_distances = v3.HARD_VDW_RATIO * (
        ligand_radii[:, None] + protein_radii[None, :]
    )
    relevant = pair_distances < hard_distances + MAX_TRANSLATION
    if not np.any(relevant):
        return np.ones(len(translations), dtype=bool)

    pair_deltas = deltas[relevant]
    hard_threshold_squared = np.square(hard_distances[relevant])
    safe = np.empty(len(translations), dtype=bool)
    for start in range(0, len(translations), chunk_size):
        stop = min(start + chunk_size, len(translations))
        shifted = (
            pair_deltas[None, :, :]
            + translations[start:stop, None, :]
        )
        distance_squared = np.einsum(
            "bpi,bpi->bp", shifted, shifted
        )
        safe[start:stop] = np.all(
            distance_squared >= hard_threshold_squared[None, :],
            axis=1,
        )
    return safe


def build_translation_proposals(rotation, directions):
    required = {
        "base_rotation_id",
        "base_rotation_fingerprint",
        "center",
        "rotation_matrix",
    }
    if not isinstance(rotation, dict) or not required.issubset(rotation):
        raise ValueError("rotation metadata is incomplete")
    proposals = []
    for direction_row in directions:
        direction = _normalize_direction(direction_row.get("direction"))
        if direction is None:
            raise ValueError("proposal direction must be finite and nonzero")
        stage = str(direction_row.get("search_stage", "guided"))
        direction_id = int(direction_row["direction_family_id"])
        source = str(direction_row["direction_source"])
        for distance_index, distance in enumerate(HIGH_DISTANCE_GRID):
            translation = direction * distance
            transform = {
                "center": list(rotation["center"]),
                "rotation_matrix": rotation["rotation_matrix"],
                "translation": translation.astype(float).tolist(),
            }
            proposals.append(
                {
                    "transform": transform,
                    "transform_fingerprint": v3._transform_fingerprint(
                        transform
                    ),
                    "base_rotation_id": int(
                        rotation["base_rotation_id"]
                    ),
                    "base_rotation_fingerprint": str(
                        rotation["base_rotation_fingerprint"]
                    ),
                    "translation_index": distance_index,
                    "distance_index": distance_index,
                    "translation_distance": float(distance),
                    "translation_direction": direction.astype(float).tolist(),
                    "direction_family_id": direction_id,
                    "direction_source": source,
                    "search_stage": stage,
                    "zero_distance_pair_count": int(
                        direction_row.get("zero_distance_pair_count", 0)
                    ),
                    "zero_translation_vdw_safe": bool(
                        direction_row.get(
                            "zero_translation_vdw_safe", False
                        )
                    ),
                }
            )
    fingerprints = [
        row["transform_fingerprint"] for row in proposals
    ]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("translation proposal fingerprints must be unique")
    return proposals


def evaluate_proposal_batch(
    native_geometry,
    rotated_ligand,
    proposals,
    seen_fingerprints,
    full_evaluator=v3._evaluate_proposal,
):
    translations = np.asarray(
        [row["transform"]["translation"] for row in proposals],
        dtype=np.float64,
    )
    safe = fast_vdw_safe_mask(
        rotated_ligand,
        native_geometry["protein"],
        native_geometry["protein_symbols"],
        native_geometry["ligand_atomic_numbers"],
        translations,
    )
    outcomes = []
    for proposal, is_safe in zip(proposals, safe):
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
                    "direction_family_id",
                    "distance_index",
                    "search_stage",
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


def select_physical_representative(candidates):
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: str(row["transform_fingerprint"]),
    )


def select_matched_representative(candidates):
    matched = [row for row in candidates if row.get("match_pass") is True]
    if not matched:
        return None
    return min(
        matched,
        key=lambda row: (
            float(row["match_distance"]),
            int(row["direction_family_id"]),
            int(row["distance_index"]),
            str(row["transform_fingerprint"]),
        ),
    )


def _stage_summary(outcomes):
    candidates = [
        row["candidate"]
        for row in outcomes
        if row["candidate"] is not None
    ]
    return {
        "any_vdw_safe": any(row["fast_vdw_safe"] for row in outcomes),
        "any_physical": bool(candidates),
        "any_matched": any(
            row.get("match_reason") is None
            and row["candidate"] is not None
            and row["candidate"].get("match_pass") is True
            for row in outcomes
        ),
    }


def _rejection_counts(outcomes):
    counts = {}
    for row in outcomes:
        reason = row["physical_reason"]
        if reason is None and row["candidate"] is not None:
            reason = row["match_reason"]
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def evaluate_rotation_family(
    native_geometry,
    rotation,
    guided_directions,
    fibonacci_direction_rows,
    full_evaluator=v3._evaluate_proposal,
):
    base_transform = {
        "center": rotation["center"],
        "rotation_matrix": rotation["rotation_matrix"],
        "translation": [0.0, 0.0, 0.0],
    }
    rotated_ligand = v3.apply_rigid_transform(
        native_geometry["native"], base_transform
    )
    seen_fingerprints = set()
    guided_proposals = build_translation_proposals(
        rotation, guided_directions
    )
    guided_outcomes = evaluate_proposal_batch(
        native_geometry,
        rotated_ligand,
        guided_proposals,
        seen_fingerprints,
        full_evaluator=full_evaluator,
    )
    guided_summary = _stage_summary(guided_outcomes)

    fibonacci_outcomes = []
    if not guided_summary["any_matched"]:
        fallback_proposals = build_translation_proposals(
            rotation, fibonacci_direction_rows
        )
        guided_fingerprints = {
            row["transform_fingerprint"] for row in guided_proposals
        }
        fallback_proposals = [
            row
            for row in fallback_proposals
            if row["transform_fingerprint"] not in guided_fingerprints
        ]
        fibonacci_outcomes = evaluate_proposal_batch(
            native_geometry,
            rotated_ligand,
            fallback_proposals,
            seen_fingerprints,
            full_evaluator=full_evaluator,
        )

    all_outcomes = guided_outcomes + fibonacci_outcomes
    candidates = [
        row["candidate"]
        for row in all_outcomes
        if row["candidate"] is not None
    ]
    return {
        "base_rotation_id": int(rotation["base_rotation_id"]),
        "base_rotation_fingerprint": str(
            rotation["base_rotation_fingerprint"]
        ),
        "guided": guided_summary,
        "expanded": _stage_summary(all_outcomes),
        "counts": {
            "guided_proposed": len(guided_proposals),
            "fibonacci_proposed": len(fibonacci_outcomes),
            "vdw_safe": sum(
                row["fast_vdw_safe"] for row in all_outcomes
            ),
            "physical": len(candidates),
            "matched": sum(
                row.get("match_reason") is None
                and row["candidate"] is not None
                and row["candidate"].get("match_pass") is True
                for row in all_outcomes
            ),
        },
        "rejection_counts": _rejection_counts(all_outcomes),
        "best_physical": select_physical_representative(candidates),
        "best_matched": select_matched_representative(candidates),
        "fibonacci_added_physical": (
            not guided_summary["any_physical"]
            and _stage_summary(all_outcomes)["any_physical"]
        ),
        "fibonacci_added_matched": (
            not guided_summary["any_matched"]
            and _stage_summary(all_outcomes)["any_matched"]
        ),
    }



def _fibonacci_direction_rows():
    return [
        {
            "direction_family_id": index,
            "direction_source": "fibonacci",
            "direction": direction.astype(float).tolist(),
            "search_stage": "fibonacci",
        }
        for index, direction in enumerate(fibonacci_directions())
    ]


def _family_state_counts(families):
    fields = (
        "guided_any_vdw_safe",
        "guided_any_physical",
        "guided_any_matched",
        "expanded_any_vdw_safe",
        "expanded_any_physical",
        "expanded_any_matched",
        "fibonacci_added_physical",
        "fibonacci_added_matched",
    )
    counts = {field: 0 for field in fields}
    for family in families:
        for stage in ("guided", "expanded"):
            for state in ("any_vdw_safe", "any_physical", "any_matched"):
                counts[stage + "_" + state] += int(
                    family[stage][state]
                )
        counts["fibonacci_added_physical"] += int(
            family["fibonacci_added_physical"]
        )
        counts["fibonacci_added_matched"] += int(
            family["fibonacci_added_matched"]
        )
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
        record, native_hard_clash_as_generation_error=False
    )
    native = native_geometry["native"]
    protein = native_geometry["protein"]
    protein_symbols = native_geometry["protein_symbols"]
    ligand_atomic_numbers = native_geometry["ligand_atomic_numbers"]
    native_contact_stats = v3.soft_contact_statistics(
        protein, native, native
    )
    properties = {
        "ligand_size": int(len(native)),
        "pocket_size": int(len(protein)),
        "native_contact_mass": float(
            native_contact_stats["native_contact_mass"]
        ),
    }

    fallback = _fibonacci_direction_rows()
    families = []
    rejection_counts = {}
    direction_source_counts = {}
    rotations = iter_v3_high_rotations(record, seed)
    for rotation_index, rotation in enumerate(rotations):
        if rotation_limit is not None and rotation_index >= rotation_limit:
            break
        base_transform = {
            "center": rotation["center"],
            "rotation_matrix": rotation["rotation_matrix"],
            "translation": [0.0, 0.0, 0.0],
        }
        rotated = v3.apply_rigid_transform(native, base_transform)
        guided = build_guided_directions(
            rotated_ligand=rotated,
            native_ligand=native,
            protein=protein,
            protein_symbols=protein_symbols,
            ligand_atomic_numbers=ligand_atomic_numbers,
            axis=rotation["axis"],
            seed=seed,
            key=record["key"],
            base_rotation_id=rotation["base_rotation_id"],
        )
        for row in guided:
            source = row["direction_source"]
            direction_source_counts[source] = (
                direction_source_counts.get(source, 0) + 1
            )
        family = evaluate_rotation_family(
            native_geometry,
            rotation,
            guided,
            fallback,
        )
        for reason, count in family["rejection_counts"].items():
            rejection_counts[reason] = (
                rejection_counts.get(reason, 0) + int(count)
            )
        families.append(family)

    value = {
        "schema": PROBE_SCHEMA,
        "key": str(record["key"]),
        "seed": int(seed),
        "target_stratum": TARGET_STRATUM,
        "rotation_count": len(families),
        "decision_eligible": rotation_limit is None,
        "sample_properties": properties,
        "families": families,
        "family_counts": _family_state_counts(families),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "direction_source_counts": dict(
            sorted(direction_source_counts.items())
        ),
    }
    validate_probe_record(
        value,
        expected_key=str(record["key"]),
        expected_rotation_count=(
            BASE_ROTATIONS if rotation_limit is None else rotation_limit
        ),
    )
    return value


def _validate_nonnegative_counts(value, name):
    if not isinstance(value, dict):
        raise ValueError(name + " must be a dictionary")
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or int(count) < 0
        ):
            raise ValueError(name + " must contain nonnegative integer counts")


def validate_probe_record(
    value,
    expected_key=None,
    expected_rotation_count=BASE_ROTATIONS,
):
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
        "direction_source_counts",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("probe record fields are invalid")
    if value["schema"] != PROBE_SCHEMA:
        raise ValueError("probe record schema is invalid")
    if expected_key is not None and value["key"] != expected_key:
        raise ValueError("probe record key does not match")
    if value["target_stratum"] != TARGET_STRATUM:
        raise ValueError("probe record target stratum must be high")
    if not isinstance(value["decision_eligible"], bool):
        raise ValueError("decision_eligible must be boolean")
    if value["decision_eligible"] and value["rotation_count"] != BASE_ROTATIONS:
        raise ValueError("decision-eligible probe record requires 512 rotations")
    if value["rotation_count"] != expected_rotation_count:
        raise ValueError("probe record rotation count does not match")
    families = value["families"]
    if not isinstance(families, list) or len(families) != value["rotation_count"]:
        raise ValueError("probe record families do not match rotation count")
    family_ids = [row["base_rotation_id"] for row in families]
    if family_ids != list(range(len(families))):
        raise ValueError("probe record rotation ids must be contiguous")
    properties = value["sample_properties"]
    if set(properties) != {
        "ligand_size",
        "pocket_size",
        "native_contact_mass",
    }:
        raise ValueError("probe sample properties are invalid")
    for name, number in properties.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float, np.integer, np.floating))
            or not np.isfinite(number)
            or float(number) <= 0.0
        ):
            raise ValueError(name + " must be finite and positive")
    _validate_nonnegative_counts(value["family_counts"], "family_counts")
    _validate_nonnegative_counts(
        value["rejection_counts"], "rejection_counts"
    )
    _validate_nonnegative_counts(
        value["direction_source_counts"], "direction_source_counts"
    )
    for family in families:
        for field in ("best_physical", "best_matched"):
            representative = family[field]
            if representative is not None and int(
                representative["base_rotation_id"]
            ) != int(family["base_rotation_id"]):
                raise ValueError(
                    "family representative has a different rotation id"
                )
    return value
