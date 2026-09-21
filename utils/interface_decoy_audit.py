import math

import numpy as np

from data_utils.local_interface import continuous_contact_weights
from utils.interface_diagnostics import extract_shortcut_features


HELDOUT_FEATURE_NAMES = (
    "ligand_contact_entropy_normalized",
    "ligand_contact_max_share",
    "ligand_effective_contact_atom_count",
    "ligand_covalent_edge_total_variation",
    "residue_contact_entropy_normalized",
    "residue_contact_max_share",
    "residue_effective_contact_count",
    *tuple(f"contact_residue_type_{index}" for index in range(20)),
    "contact_direction_eigenvalue_0",
    "contact_direction_eigenvalue_1",
    "contact_direction_eigenvalue_2",
)


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _coordinates(value, name):
    try:
        array = _as_numpy(value).astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric coordinates") from error
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or len(array) == 0
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{name} must be finite [N,3] coordinates with N > 0"
        )
    return array


def _integer_vector(value, length, name):
    array = _as_numpy(value)
    if (
        array.ndim != 1
        or len(array) != length
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be an aligned integer vector")
    return array.astype(np.int64, copy=False)


def _distribution_statistics(probabilities, support_size):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or len(probabilities) != support_size
        or support_size <= 0
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError("contact distribution is malformed")
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    normalized = (
        float(entropy / math.log(support_size))
        if support_size > 1
        else 0.0
    )
    return normalized, float(np.max(probabilities)), float(np.exp(entropy))


def _unique_ligand_edges(edge_index, ligand_count):
    edge_index = _as_numpy(edge_index)
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or not np.issubdtype(edge_index.dtype, np.integer)
    ):
        raise ValueError("ligand edge index must have integer shape [2,E]")
    edge_index = edge_index.astype(np.int64, copy=False)
    if edge_index.size and (
        int(np.min(edge_index)) < 0
        or int(np.max(edge_index)) >= ligand_count
    ):
        raise ValueError("ligand edge index is out of range")
    edges = set()
    for source, target in edge_index.T:
        source = int(source)
        target = int(target)
        if source == target:
            raise ValueError("ligand edge index contains a self edge")
        edges.add(tuple(sorted((source, target))))
    return sorted(edges)


def extract_heldout_interface_features(record, pose_coords):
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")
    ligand = _coordinates(record.get("ligand_coords"), "ligand_coords")
    protein = _coordinates(record.get("protein_coords"), "protein_coords")
    pose = _coordinates(pose_coords, "pose_coords")
    if pose.shape != ligand.shape:
        raise ValueError("pose_coords must align with ligand atoms")

    ligand_count = record.get("ligand_heavy_atom_count", len(ligand))
    if (
        isinstance(ligand_count, (bool, np.bool_))
        or not isinstance(ligand_count, (int, np.integer))
        or int(ligand_count) != len(ligand)
    ):
        raise ValueError(
            "ligand_heavy_atom_count must align with ligand atoms"
        )
    ligand_count = int(ligand_count)

    parents = _integer_vector(
        record.get("protein_parent_residue"),
        len(protein),
        "protein parent residue",
    )
    if np.any(parents < 0):
        raise ValueError("protein parent residue must be non-negative")
    if "residue_count" in record:
        residue_count = record["residue_count"]
        if (
            isinstance(residue_count, (bool, np.bool_))
            or not isinstance(residue_count, (int, np.integer))
            or int(residue_count) <= 0
            or np.any(parents >= int(residue_count))
        ):
            raise ValueError(
                "protein parent residue is outside residue_count"
            )

    residue_types = _integer_vector(
        record.get("protein_residue_type"),
        len(protein),
        "protein residue type",
    )
    if np.any(residue_types < 0) or np.any(residue_types >= 20):
        raise ValueError("protein residue type must be in [0,19]")
    for parent in np.unique(parents):
        if len(np.unique(residue_types[parents == parent])) != 1:
            raise ValueError(
                "protein residue type must be consistent within each parent"
            )

    edges = _unique_ligand_edges(
        record.get("ligand_edge_index"), ligand_count
    )
    displacement = protein[:, None, :] - pose[None, :, :]
    distances = np.linalg.norm(displacement, axis=-1)
    weights = continuous_contact_weights(distances).astype(
        np.float64, copy=False
    )
    contact_mass = float(np.sum(weights))
    if not math.isfinite(contact_mass) or contact_mass <= 0.0:
        raise ValueError("held-out features require positive contact mass")
    if np.any((distances <= 0.0) & (weights > 0.0)):
        raise ValueError(
            "zero-distance contact has undefined direction"
        )

    ligand_distribution = np.sum(weights, axis=0) / contact_mass
    ligand_entropy, ligand_max, ligand_effective = (
        _distribution_statistics(ligand_distribution, ligand_count)
    )
    if edges:
        edge_variation = float(
            np.mean(
                [
                    abs(
                        ligand_distribution[source]
                        - ligand_distribution[target]
                    )
                    for source, target in edges
                ]
            )
        )
    else:
        edge_variation = 0.0

    parent_ids, parent_inverse = np.unique(
        parents, return_inverse=True
    )
    protein_contact_mass = np.sum(weights, axis=1)
    residue_mass = np.bincount(
        parent_inverse,
        weights=protein_contact_mass,
        minlength=len(parent_ids),
    ).astype(np.float64)
    residue_distribution = residue_mass / contact_mass
    residue_entropy, residue_max, residue_effective = (
        _distribution_statistics(
            residue_distribution, len(parent_ids)
        )
    )

    composition = np.bincount(
        residue_types,
        weights=protein_contact_mass,
        minlength=20,
    ).astype(np.float64)[:20]
    composition /= contact_mass

    unit_directions = displacement / distances[:, :, None]
    direction_matrix = np.einsum(
        "pl,pli,plj->ij",
        weights,
        unit_directions,
        unit_directions,
        optimize=True,
    )
    direction_matrix /= contact_mass
    direction_matrix = 0.5 * (
        direction_matrix + direction_matrix.T
    )
    eigenvalues = np.linalg.eigvalsh(direction_matrix)[::-1]

    features = np.concatenate(
        [
            np.asarray(
                [
                    ligand_entropy,
                    ligand_max,
                    ligand_effective,
                    edge_variation,
                    residue_entropy,
                    residue_max,
                    residue_effective,
                ],
                dtype=np.float64,
            ),
            composition,
            eigenvalues.astype(np.float64, copy=False),
        ]
    )
    if (
        features.shape != (len(HELDOUT_FEATURE_NAMES),)
        or not np.all(np.isfinite(features))
    ):
        raise ValueError("held-out interface features must be finite")
    return features


def extract_coarse_matching_features(record, pose_coords):
    return extract_shortcut_features(record, pose_coords)


DATA_GATE_THRESHOLDS = {
    "maximum_matched_auroc": 0.80,
    "minimum_random_minus_matched_auroc": 0.08,
    "minimum_matched_requested_fraction": 0.30,
}


def _probe_inputs(features, labels, target_groups, feature_names):
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("features must be numeric") from error
    names = tuple(feature_names)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError(
            "feature_names must be unique nonempty strings"
        )
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(names)
    ):
        raise ValueError(
            "features must be a matrix aligned with feature_names"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")

    label_vector = np.asarray(labels).reshape(-1)
    group_vector = np.asarray(target_groups).reshape(-1)
    if len(label_vector) != len(matrix):
        raise ValueError("labels must align with features")
    if len(group_vector) != len(matrix):
        raise ValueError("target_groups must align with features")
    if not (
        np.issubdtype(label_vector.dtype, np.integer)
        or np.issubdtype(label_vector.dtype, np.bool_)
    ):
        raise ValueError("labels must contain binary classes")
    label_vector = label_vector.astype(np.int64, copy=False)
    if set(label_vector.tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes")
    if len(np.unique(group_vector)) < 5:
        raise ValueError(
            "grouped probes require at least five target groups"
        )
    return matrix, label_vector, group_vector, names


def fit_grouped_pose_probes(
    features,
    labels,
    target_groups,
    feature_names,
):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    matrix, label_vector, group_vector, names = _probe_inputs(
        features, labels, target_groups, feature_names
    )
    row_count = len(matrix)
    linear_prediction = np.empty(row_count, dtype=np.float64)
    nonlinear_prediction = np.empty(row_count, dtype=np.float64)
    univariate_prediction = np.empty(
        (row_count, len(names)), dtype=np.float64
    )

    splitter = GroupKFold(n_splits=5)
    folds = list(
        splitter.split(
            matrix, label_vector, groups=group_vector
        )
    )
    for train_index, test_index in folds:
        if set(label_vector[train_index].tolist()) != {0, 1}:
            raise ValueError(
                "each grouped training fold must contain both classes"
            )
        linear = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=0,
            ),
        )
        nonlinear = GradientBoostingClassifier(
            n_estimators=32,
            max_depth=3,
            random_state=0,
        )
        linear.fit(matrix[train_index], label_vector[train_index])
        nonlinear.fit(matrix[train_index], label_vector[train_index])
        linear_prediction[test_index] = linear.predict_proba(
            matrix[test_index]
        )[:, 1]
        nonlinear_prediction[test_index] = nonlinear.predict_proba(
            matrix[test_index]
        )[:, 1]

        for feature_index in range(len(names)):
            univariate = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=0,
                ),
            )
            univariate.fit(
                matrix[train_index, feature_index : feature_index + 1],
                label_vector[train_index],
            )
            univariate_prediction[
                test_index, feature_index
            ] = univariate.predict_proba(
                matrix[test_index, feature_index : feature_index + 1]
            )[:, 1]

    return {
        "feature_names": list(names),
        "row_count": int(row_count),
        "target_group_count": int(len(np.unique(group_vector))),
        "cv_splits": 5,
        "linear_cv_auroc": float(
            roc_auc_score(label_vector, linear_prediction)
        ),
        "nonlinear_cv_auroc": float(
            roc_auc_score(label_vector, nonlinear_prediction)
        ),
        "univariate_cv_auroc": [
            {
                "feature": name,
                "auroc": float(
                    roc_auc_score(
                        label_vector,
                        univariate_prediction[:, feature_index],
                    )
                ),
            }
            for feature_index, name in enumerate(names)
        ],
        "oof_linear_prediction": linear_prediction,
        "oof_nonlinear_prediction": nonlinear_prediction,
    }


def _finite_gate_metric(value, name):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _completion_count(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def evaluate_v2_data_gate(report):
    try:
        paired = report["cohorts"]["paired_intersection"]
        random_auroc = paired["random_physical"]["coarse"][
            "nonlinear_cv_auroc"
        ]
        matched_auroc = paired["matched"]["coarse"][
            "nonlinear_cv_auroc"
        ]
        completion = report["bank_completion"]
        requested_count = completion["requested_denominator"]
        matched_count = completion["matched"]["complete_count"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "v2 data report is missing required gate fields"
        ) from error

    random_auroc = _finite_gate_metric(
        random_auroc, "random physical coarse AUROC"
    )
    matched_auroc = _finite_gate_metric(
        matched_auroc, "matched coarse AUROC"
    )
    if not 0.0 <= random_auroc <= 1.0:
        raise ValueError(
            "random physical coarse AUROC must be in [0,1]"
        )
    if not 0.0 <= matched_auroc <= 1.0:
        raise ValueError("matched coarse AUROC must be in [0,1]")

    requested_count = _completion_count(
        requested_count, "requested denominator"
    )
    matched_count = _completion_count(
        matched_count, "matched complete count"
    )
    if requested_count <= 0:
        raise ValueError("requested denominator must be positive")
    if matched_count < 0 or matched_count > requested_count:
        raise ValueError(
            "matched complete count must be within requested denominator"
        )

    difference = float(random_auroc - matched_auroc)
    matched_fraction = float(matched_count / requested_count)
    reasons = []
    if matched_auroc >= DATA_GATE_THRESHOLDS[
        "maximum_matched_auroc"
    ]:
        reasons.append("matched_coarse_auroc_too_high")
    if difference < DATA_GATE_THRESHOLDS[
        "minimum_random_minus_matched_auroc"
    ]:
        reasons.append(
            "paired_coarse_auroc_reduction_too_small"
        )
    if matched_fraction <= DATA_GATE_THRESHOLDS[
        "minimum_matched_requested_fraction"
    ]:
        reasons.append("matched_requested_fraction_too_low")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": dict(DATA_GATE_THRESHOLDS),
        "metrics": {
            "matched_coarse_nonlinear_cv_auroc": matched_auroc,
            "random_physical_coarse_nonlinear_cv_auroc": random_auroc,
            "random_minus_matched_coarse_auroc": difference,
            "matched_complete_count": matched_count,
            "requested_count": requested_count,
            "matched_requested_fraction": matched_fraction,
        },
    }
