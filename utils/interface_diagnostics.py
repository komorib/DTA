import numpy as np

from data_utils.local_interface_decoys import (
    soft_contact_statistics,
    vdw_overlap_statistics,
)


SHORTCUT_FEATURE_NAMES = (
    "contact_mass",
    "minimum_distance",
    "ligand_pocket_centroid_distance",
    "ligand_atom_count",
    "protein_atom_count",
    "maximum_overlap_severity",
    "summed_overlap_severity",
    "count_below_0_9_vdw_sum",
    "count_below_0_8_vdw_sum",
)

GATE_THRESHOLDS = {
    "maximum_nonlinear_score_r2": 0.50,
    "maximum_nonlinear_shortcut_auroc": 0.80,
    "minimum_interface_shortcut_gap": 0.10,
    "minimum_interface_auroc": 0.75,
    "minimum_large_tertile_interface_auroc": 0.65,
    "minimum_tertile_set_count": 100,
}


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_shortcut_features(record, pose_coords):
    protein_coords = _as_numpy(record["protein_coords"]).astype(
        np.float64, copy=False
    )
    native_coords = _as_numpy(record["ligand_coords"]).astype(
        np.float64, copy=False
    )
    pose_coords = _as_numpy(pose_coords).astype(
        np.float64, copy=False
    )
    if (
        protein_coords.ndim != 2
        or protein_coords.shape[1] != 3
        or native_coords.ndim != 2
        or native_coords.shape[1] != 3
        or pose_coords.shape != native_coords.shape
    ):
        raise ValueError(
            "shortcut geometry must contain aligned [N,3] coordinates"
        )
    distances = np.linalg.norm(
        protein_coords[:, None, :] - pose_coords[None, :, :],
        axis=-1,
    )
    contacts = soft_contact_statistics(
        protein_coords, native_coords, pose_coords
    )
    overlap = vdw_overlap_statistics(
        protein_coords,
        _as_numpy(record["protein_element_symbol"]),
        pose_coords,
        _as_numpy(record["ligand_atomic_numbers"]),
    )
    centroid_distance = float(
        np.linalg.norm(
            np.mean(protein_coords, axis=0)
            - np.mean(pose_coords, axis=0)
        )
    )
    features = np.asarray(
        [
            contacts["decoy_contact_mass"],
            float(np.min(distances)),
            centroid_distance,
            float(len(pose_coords)),
            float(len(protein_coords)),
            overlap["max_overlap_severity"],
            overlap["sum_overlap_severity"],
            float(overlap["count_below_0_9"]),
            float(overlap["count_below_0_8"]),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("shortcut features must be finite")
    return features


def _as_finite_matrix(features):
    features = np.asarray(features, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] != len(SHORTCUT_FEATURE_NAMES)
        or not np.all(np.isfinite(features))
    ):
        raise ValueError(
            "features must be a finite [N,F] shortcut matrix"
        )
    return features


def _as_aligned_vector(value, row_count, name, dtype=None):
    vector = np.asarray(value, dtype=dtype).reshape(-1)
    if len(vector) != row_count:
        raise ValueError(f"{name} must align with features")
    return vector


def _mean_vector(vectors):
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(
        float
    ).tolist()


def _contact_mass_tertiles(
    labels,
    interface_scores,
    set_ids,
    contact_mass,
):
    from sklearn.metrics import roc_auc_score

    ordered_sets = []
    set_mass = {}
    for set_id, mass in zip(set_ids, contact_mass):
        set_id = str(set_id)
        mass = float(mass)
        if set_id not in set_mass:
            ordered_sets.append(set_id)
            set_mass[set_id] = mass
        elif not np.isclose(
            set_mass[set_id], mass, rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                "contact mass must be constant within a pose set"
            )
    masses = np.asarray(
        [set_mass[set_id] for set_id in ordered_sets],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(masses)):
        raise ValueError("contact mass must be finite")
    lower, upper = np.quantile(masses, [1.0 / 3.0, 2.0 / 3.0])
    definitions = (
        ("low", lambda value: value <= lower),
        (
            "medium",
            lambda value: value > lower and value <= upper,
        ),
        ("high", lambda value: value > upper),
    )
    output = []
    string_set_ids = np.asarray(
        [str(set_id) for set_id in set_ids]
    )
    for name, selector in definitions:
        selected_sets = {
            set_id
            for set_id in ordered_sets
            if selector(set_mass[set_id])
        }
        row_mask = np.asarray(
            [
                set_id in selected_sets
                for set_id in string_set_ids
            ],
            dtype=bool,
        )
        selected_labels = labels[row_mask]
        if (
            len(selected_labels)
            and len(np.unique(selected_labels)) == 2
        ):
            auroc = float(
                roc_auc_score(
                    selected_labels,
                    interface_scores[row_mask],
                )
            )
        else:
            auroc = None
        output.append(
            {
                "name": name,
                "set_count": len(selected_sets),
                "row_count": int(np.sum(row_mask)),
                "auroc": auroc,
                "contact_mass_lower": (
                    None
                    if not selected_sets
                    else float(
                        min(set_mass[key] for key in selected_sets)
                    )
                ),
                "contact_mass_upper": (
                    None
                    if not selected_sets
                    else float(
                        max(set_mass[key] for key in selected_sets)
                    )
                ),
            }
        )
    return output


def fit_shortcut_probes(
    features,
    interface_scores,
    labels,
    target_groups,
    set_ids,
    contact_mass,
):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    features = _as_finite_matrix(features)
    row_count = features.shape[0]
    interface_scores = _as_aligned_vector(
        interface_scores,
        row_count,
        "interface_scores",
        dtype=np.float64,
    )
    labels = _as_aligned_vector(
        labels, row_count, "labels", dtype=np.int64
    )
    target_groups = _as_aligned_vector(
        target_groups, row_count, "target_groups"
    )
    set_ids = _as_aligned_vector(
        set_ids, row_count, "set_ids"
    )
    contact_mass = _as_aligned_vector(
        contact_mass,
        row_count,
        "contact_mass",
        dtype=np.float64,
    )
    if not np.all(np.isfinite(interface_scores)):
        raise ValueError("interface_scores must be finite")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes")
    if len(np.unique(target_groups)) < 5:
        raise ValueError(
            "shortcut probes require at least five target groups"
        )

    linear_score_prediction = np.empty(row_count, dtype=np.float64)
    tree_score_prediction = np.empty(row_count, dtype=np.float64)
    linear_pose_prediction = np.empty(row_count, dtype=np.float64)
    boosted_pose_prediction = np.empty(row_count, dtype=np.float64)
    linear_score_coefficients = []
    tree_importances = []
    linear_pose_coefficients = []
    boosted_importances = []
    splitter = GroupKFold(n_splits=5)
    for train_index, test_index in splitter.split(
        features, labels, groups=target_groups
    ):
        linear_score = make_pipeline(
            StandardScaler(), LinearRegression()
        )
        tree_score = DecisionTreeRegressor(
            max_depth=3, random_state=0
        )
        linear_pose = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=0,
            ),
        )
        boosted_pose = GradientBoostingClassifier(
            n_estimators=32,
            max_depth=3,
            random_state=0,
        )
        linear_score.fit(
            features[train_index],
            interface_scores[train_index],
        )
        tree_score.fit(
            features[train_index],
            interface_scores[train_index],
        )
        linear_pose.fit(
            features[train_index], labels[train_index]
        )
        boosted_pose.fit(
            features[train_index], labels[train_index]
        )
        linear_score_prediction[test_index] = linear_score.predict(
            features[test_index]
        )
        tree_score_prediction[test_index] = tree_score.predict(
            features[test_index]
        )
        linear_pose_prediction[test_index] = (
            linear_pose.predict_proba(features[test_index])[:, 1]
        )
        boosted_pose_prediction[test_index] = (
            boosted_pose.predict_proba(features[test_index])[:, 1]
        )
        linear_score_coefficients.append(
            linear_score[-1].coef_
        )
        tree_importances.append(tree_score.feature_importances_)
        linear_pose_coefficients.append(
            linear_pose[-1].coef_[0]
        )
        boosted_importances.append(
            boosted_pose.feature_importances_
        )

    report = {
        "feature_names": list(SHORTCUT_FEATURE_NAMES),
        "row_count": row_count,
        "set_count": int(len(np.unique(set_ids))),
        "target_group_count": int(
            len(np.unique(target_groups))
        ),
        "cv_splits": 5,
        "score_probe": {
            "linear_cv_r2": float(
                r2_score(
                    interface_scores,
                    linear_score_prediction,
                )
            ),
            "nonlinear_cv_r2": float(
                r2_score(
                    interface_scores,
                    tree_score_prediction,
                )
            ),
            "linear_coefficients": _mean_vector(
                linear_score_coefficients
            ),
            "nonlinear_feature_importances": _mean_vector(
                tree_importances
            ),
        },
        "pose_probe": {
            "linear_cv_auroc": float(
                roc_auc_score(labels, linear_pose_prediction)
            ),
            "nonlinear_cv_auroc": float(
                roc_auc_score(labels, boosted_pose_prediction)
            ),
            "linear_coefficients": _mean_vector(
                linear_pose_coefficients
            ),
            "nonlinear_feature_importances": _mean_vector(
                boosted_importances
            ),
        },
        "interface": {
            "auroc": float(
                roc_auc_score(labels, interface_scores)
            ),
            "contact_mass_tertiles": _contact_mass_tertiles(
                labels,
                interface_scores,
                set_ids,
                contact_mass,
            ),
        },
    }
    report["gate"] = evaluate_shortcut_gate(report)
    return report


def _finite_metric(report, section, field):
    try:
        value = float(report[section][field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"diagnostic report is missing {section}.{field}"
        ) from error
    if not np.isfinite(value):
        raise ValueError(
            f"diagnostic metric {section}.{field} must be finite"
        )
    return value


def evaluate_shortcut_gate(report):
    nonlinear_r2 = _finite_metric(
        report, "score_probe", "nonlinear_cv_r2"
    )
    shortcut_auroc = _finite_metric(
        report, "pose_probe", "nonlinear_cv_auroc"
    )
    interface_auroc = _finite_metric(
        report, "interface", "auroc"
    )
    reasons = []
    if (
        nonlinear_r2
        >= GATE_THRESHOLDS["maximum_nonlinear_score_r2"]
    ):
        reasons.append("nonlinear_score_r2_too_high")
    if (
        shortcut_auroc
        >= GATE_THRESHOLDS[
            "maximum_nonlinear_shortcut_auroc"
        ]
    ):
        reasons.append("nonlinear_shortcut_auroc_too_high")
    if (
        interface_auroc - shortcut_auroc
        < GATE_THRESHOLDS["minimum_interface_shortcut_gap"]
    ):
        reasons.append("interface_shortcut_gap_too_small")
    if (
        interface_auroc
        < GATE_THRESHOLDS["minimum_interface_auroc"]
    ):
        reasons.append("interface_auroc_too_low")

    try:
        tertiles = report["interface"][
            "contact_mass_tertiles"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "diagnostic report is missing contact-mass tertiles"
        ) from error
    for tertile in tertiles:
        name = str(tertile.get("name", "unknown"))
        set_count = int(tertile.get("set_count", 0))
        auroc = tertile.get("auroc")
        if (
            set_count
            >= GATE_THRESHOLDS["minimum_tertile_set_count"]
        ):
            if auroc is None or not np.isfinite(float(auroc)):
                raise ValueError(
                    f"large contact-mass tertile {name} needs AUROC"
                )
            if (
                float(auroc)
                < GATE_THRESHOLDS[
                    "minimum_large_tertile_interface_auroc"
                ]
            ):
                reasons.append(
                    "interface_tertile_auroc_too_low:" + name
                )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": dict(GATE_THRESHOLDS),
    }
