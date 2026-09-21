"""Seal a strict outer-OOF contact-aware MSA pair teacher for V31.

The five neural teachers are trained/exported by :mod:`teacher_folds`.  This
CPU-only stage fits one PCA+ExtraTrees residual reader per outer fold, using
only that fold's outer-fit pair latents and labels, then predicts its blind
outer holdout.  The resulting scalar affinity target is safe to attach to the
V31 student *training* rows and is never consulted by validation/test forward
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

from dta.optional_structure_staged_v22.data import manifest_fingerprint, read_jsonl, sha256_file
from dta.rank_fusion_v4.metrics import affinity_metrics

from .folds import load_fold_metadata, lp_train_rows


VERSION = "optional-structure-privileged-msa-pair-teacher-v31"
SCHEMA_VERSION = 2
DEFAULT_COMPONENTS = 128
DEFAULT_ESTIMATORS = 800
DEFAULT_MINIMUM_LEAF = 5
DEFAULT_MAX_FEATURES = 1.0


def _atomic_npz(path, **arrays):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rows(train_manifest, limit=None):
    rows = lp_train_rows(train_manifest, limit)
    keys = [str(row["structure_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("V31 teacher train structure keys are not unique")
    return rows


def _load_features(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("V31 outer teacher feature archive is missing: %s" % path)
    with np.load(str(path), allow_pickle=False) as archive:
        required = {"keys", "labels", "predictions", "pair_latents"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("V31 teacher feature archive misses %s" % sorted(missing))
        keys = np.asarray(archive["keys"]).astype(str)
        labels = np.asarray(archive["labels"], dtype=np.float64)
        predictions = np.asarray(archive["predictions"], dtype=np.float64)
        pair_latents = np.asarray(archive["pair_latents"], dtype=np.float32)
    if (
        keys.ndim != 1
        or labels.shape != keys.shape
        or predictions.shape != keys.shape
        or pair_latents.ndim != 2
        or pair_latents.shape[0] != keys.size
        or pair_latents.shape[1] != 1024
    ):
        raise ValueError("V31 outer teacher feature shapes are invalid")
    if len(set(keys.tolist())) != keys.size:
        raise ValueError("V31 outer teacher feature keys are not unique")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all() or not np.isfinite(pair_latents).all():
        raise ValueError("V31 outer teacher features contain non-finite values")
    return {"path": path, "keys": keys, "labels": labels, "predictions": predictions, "pair_latents": pair_latents}


def _load_v22_train_predictions(path, rows, train_manifest):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("V31 V22 train prediction archive is missing")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError("V31 V22 train prediction archive lacks its lineage sidecar")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "optional-structure-privileged-msa-pair-v22-train-export-v31":
        raise ValueError("V31 V22 train export version mismatch")
    if metadata.get("status") != "complete" or metadata.get("output_sha256") != sha256_file(path):
        raise ValueError("V31 V22 train export is incomplete or changed")
    if metadata.get("official_valid_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
        raise ValueError("V31 V22 train export accessed official validation/test")
    if metadata.get("train_manifest_sha256") != manifest_fingerprint(train_manifest):
        raise ValueError("V31 V22 train export train-manifest lineage mismatch")
    with np.load(str(path), allow_pickle=False) as archive:
        required = {"sample_ids", "labels", "dataset_index", "predictions"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("V31 V22 train archive misses %s" % sorted(missing))
        ids = np.asarray(archive["sample_ids"]).astype(str)
        labels = np.asarray(archive["labels"], dtype=np.float64)
        dataset_index = np.asarray(archive["dataset_index"], dtype=np.int64)
        predictions = np.asarray(archive["predictions"], dtype=np.float64)
    if ids.ndim != 1 or labels.shape != ids.shape or predictions.shape != ids.shape or dataset_index.shape != ids.shape:
        raise ValueError("V31 V22 train prediction shapes are invalid")
    if len(set(ids.tolist())) != ids.size or not np.all(dataset_index == 0):
        raise ValueError("V31 V22 train archive must contain unique LP rows only")
    requested_ids = np.asarray([str(row["sample_id"]) for row in rows])
    requested_labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    if ids.size != requested_ids.size or set(ids.tolist()) != set(requested_ids.tolist()):
        raise ValueError("V31 V22 train archive does not exactly cover LP train rows")
    positions = {key: index for index, key in enumerate(ids.tolist())}
    order = np.asarray([positions[key] for key in requested_ids], dtype=np.int64)
    if not np.allclose(labels[order], requested_labels, rtol=0.0, atol=1.0e-6):
        raise ValueError("V31 V22 train archive labels differ from train manifest")
    ordered = predictions[order]
    if not np.isfinite(ordered).all():
        raise ValueError("V31 V22 train predictions contain non-finite values")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "metadata_path": metadata_path,
        "metadata_sha256": sha256_file(metadata_path),
        "sealed_v22_checkpoint": metadata.get("sealed_v22_checkpoint"),
        "sealed_v22_checkpoint_sha256": metadata.get("sealed_v22_checkpoint_sha256"),
        "predictions_by_sample_id": {key: float(value) for key, value in zip(requested_ids.tolist(), ordered.tolist())},
    }


def _align_feature_rows(features, wanted_rows, name):
    wanted_keys = np.asarray([str(row["structure_key"]) for row in wanted_rows])
    wanted_labels = np.asarray([float(row["label"]) for row in wanted_rows], dtype=np.float64)
    if features["keys"].size != wanted_keys.size or set(features["keys"].tolist()) != set(wanted_keys.tolist()):
        raise ValueError("V31 %s teacher feature keys differ from its fold manifest" % name)
    positions = {key: index for index, key in enumerate(features["keys"].tolist())}
    order = np.asarray([positions[key] for key in wanted_keys], dtype=np.int64)
    if not np.allclose(features["labels"][order], wanted_labels, rtol=0.0, atol=1.0e-6):
        raise ValueError("V31 %s teacher feature labels differ from its fold manifest" % name)
    return {
        "keys": features["keys"][order],
        "labels": features["labels"][order],
        "pair_latents": features["pair_latents"][order],
    }


def _read_fold_rows(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(value.get("split")) != "train" or not isinstance(value.get("samples"), list):
        raise ValueError("V31 fold feature manifest is invalid")
    return value["samples"]


def _manifest_rows_from_keys(source_rows, samples):
    requested = [str(sample.get("lmdb_key", "")) for sample in samples]
    source = {str(row["structure_key"]): row for row in source_rows}
    if len(requested) != len(set(requested)) or set(requested).difference(source):
        raise ValueError("V31 fold manifest keys do not map to source train rows")
    rows = [source[key] for key in requested]
    labels = np.asarray([float(sample["label"]) for sample in samples], dtype=np.float64)
    expected = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    if not np.allclose(labels, expected, rtol=0.0, atol=1.0e-6):
        raise ValueError("V31 fold manifest labels changed from source train rows")
    return rows


def _fit_readout(features, target, holdout_features, args, fold):
    components = min(int(args.components), int(features.shape[0]) - 1, int(features.shape[1]))
    if components <= 0:
        raise ValueError("V31 fold has insufficient outer-fit rows for PCA")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=int(args.seed) + 1009 * int(fold),
    )
    fit = pca.fit_transform(features)
    holdout = pca.transform(holdout_features)
    forest = ExtraTreesRegressor(
        n_estimators=int(args.estimators),
        min_samples_leaf=int(args.minimum_leaf),
        max_features=float(args.max_features),
        n_jobs=int(args.n_jobs),
        random_state=int(args.seed) + 1009 * int(fold) + 29,
    )
    forest.fit(fit, target)
    prediction = forest.predict(holdout).astype(np.float64)
    return prediction, components


def _validate_args(args):
    if int(args.components) < 1 or int(args.estimators) < 1 or int(args.minimum_leaf) < 1:
        raise ValueError("V31 PCA/forest settings must be positive")
    if not 0.0 < float(args.max_features) <= 1.0:
        raise ValueError("V31 max_features must be in (0, 1]")
    if int(args.n_jobs) == 0:
        raise ValueError("V31 n_jobs cannot be zero")


def build_teacher_cache(args):
    _validate_args(args)
    train_manifest = Path(args.train_manifest).expanduser().resolve()
    fold_root = Path(args.fold_root).expanduser().resolve()
    teacher_root = Path(args.teacher_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("V31 teacher cache output exists; choose a new CACHE_ID")
    fold_metadata = load_fold_metadata(fold_root, train_manifest, require_full=bool(args.require_full_folds))
    source_rows = _rows(train_manifest, fold_metadata.get("limit"))
    v22 = _load_v22_train_predictions(args.v22_train_predictions, source_rows, train_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s.building-" % output.name, dir=str(output.parent)))
    try:
        ids = np.asarray([str(row["sample_id"]) for row in source_rows])
        keys = np.asarray([str(row["structure_key"]) for row in source_rows])
        labels = np.asarray([float(row["label"]) for row in source_rows], dtype=np.float64)
        expected_fold = np.asarray(
            [int(fold_metadata["sequence_to_outer_fold"][str(row["sequence_id"])]) for row in source_rows],
            dtype=np.int64,
        )
        teacher_predictions = np.full(labels.shape, np.nan, dtype=np.float64)
        teacher_deltas = np.full(labels.shape, np.nan, dtype=np.float64)
        positions = {key: index for index, key in enumerate(keys.tolist())}
        recipes, teacher_lineages = [], []
        for fold in range(int(fold_metadata["folds"])):
            info = fold_metadata["fold_artifacts"][fold]
            fold_dir = fold_root / ("fold_%d" % fold)
            result_path = teacher_root / ("outer_fold_%d" % fold) / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError("V31 outer teacher result is missing for fold %d" % fold)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("schema_version") != 1 or result.get("version") != "optional-structure-privileged-msa-pair-outer-teacher-v31":
                raise ValueError("V31 outer teacher result schema/version mismatch")
            if int(result.get("fold", -1)) != fold or result.get("status") != "complete":
                raise ValueError("V31 outer teacher result is incomplete for fold %d" % fold)
            if result.get("official_valid_manifest_read") is not False or result.get("test_manifest_read") is not False:
                raise ValueError("V31 outer teacher accessed official validation/test")
            if result.get("outer_holdout_opened_before_selection") is not False or result.get("outer_holdout_labels_used_for_checkpoint_selection") is not False:
                raise ValueError("V31 outer holdout was used during teacher checkpoint selection")
            if result.get("manifest_sha256") != info["manifest_sha256"]:
                raise ValueError("V31 outer teacher fold-manifest lineage mismatch")
            fit_path = Path(result["outer_fit_features"]["path"]).expanduser().resolve()
            holdout_path = Path(result["outer_holdout_features"]["path"]).expanduser().resolve()
            if result["outer_fit_features"].get("sha256") != sha256_file(fit_path):
                raise ValueError("V31 outer-fit feature checksum mismatch")
            if result["outer_holdout_features"].get("sha256") != sha256_file(holdout_path):
                raise ValueError("V31 outer-holdout feature checksum mismatch")
            fit_rows = _manifest_rows_from_keys(source_rows, _read_fold_rows(fold_dir / "outer_fit.json"))
            holdout_rows = _manifest_rows_from_keys(source_rows, _read_fold_rows(fold_dir / "outer_holdout.json"))
            fit = _align_feature_rows(_load_features(fit_path), fit_rows, "outer-fit")
            holdout = _align_feature_rows(_load_features(holdout_path), holdout_rows, "outer-holdout")
            fit_v22 = np.asarray([v22["predictions_by_sample_id"][str(row["sample_id"])] for row in fit_rows])
            holdout_v22 = np.asarray([v22["predictions_by_sample_id"][str(row["sample_id"])] for row in holdout_rows])
            target = fit["labels"] - fit_v22
            prediction_delta, components = _fit_readout(
                fit["pair_latents"], target, holdout["pair_latents"], args, fold
            )
            prediction = holdout_v22 + prediction_delta
            indices = np.asarray([positions[str(row["structure_key"])] for row in holdout_rows], dtype=np.int64)
            if not np.all(expected_fold[indices] == fold):
                raise RuntimeError("V31 outer teacher prediction does not match fold assignment")
            if np.isfinite(teacher_predictions[indices]).any():
                raise RuntimeError("V31 outer teacher writes a row more than once")
            teacher_predictions[indices] = prediction
            teacher_deltas[indices] = prediction_delta
            recipes.append(
                {
                    "fold": fold,
                    "components": int(components),
                    "outer_fit_count": int(fit["keys"].size),
                    "outer_holdout_count": int(holdout["keys"].size),
                    "outer_fit_features": str(fit_path),
                    "outer_fit_features_sha256": sha256_file(fit_path),
                    "outer_holdout_features": str(holdout_path),
                    "outer_holdout_features_sha256": sha256_file(holdout_path),
                }
            )
            teacher_lineages.append(
                {
                    "fold": fold,
                    "result": str(result_path),
                    "result_sha256": sha256_file(result_path),
                    "initialization": result["initialization"],
                    "refit": result.get("refit", {}),
                }
            )
        if not np.isfinite(teacher_predictions).all() or not np.isfinite(teacher_deltas).all():
            raise RuntimeError("V31 outer OOF teacher is incomplete")
        archive = temporary / "train_teacher_oof.npz"
        _atomic_npz(
            archive,
            sample_ids=ids,
            structure_keys=keys,
            labels=labels,
            dataset_index=np.zeros(labels.shape, dtype=np.int64),
            fold_ids=expected_fold,
            teacher_predictions=teacher_predictions,
            teacher_deltas=teacher_deltas,
        )
        initialization_policies = sorted({str(item["initialization"].get("policy")) for item in teacher_lineages})
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "version": VERSION,
            "status": "built",
            "output": str(output),
            "archive": str(output / "train_teacher_oof.npz"),
            "archive_sha256": sha256_file(archive),
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": manifest_fingerprint(train_manifest),
            "fold_root": str(fold_root),
            "folds_sha256": sha256_file(fold_root / "folds.json"),
            "teacher_root": str(teacher_root),
            "labels_scope": "train_only",
            "official_valid_manifest_read": False,
            "valid_manifest_read": False,
            "test_manifest_read": False,
            "teacher_target_regime": "strict_outer_sequence_grouped_contact_pair_latent_oof_v22_delta",
            "teacher_target": "V22_full_train_prediction + outer-fold-pair-latent OOF residual",
            "fold_group": "sequence_id",
            "fold_allocation": fold_metadata["fold_allocation"],
            "folds": int(fold_metadata["folds"]),
            "fold_ids_sha256": _sha256_array(expected_fold),
            "sample_ids_sha256": _sha256_array(ids),
            "structure_keys_sha256": _sha256_array(keys),
            "teacher_predictions_sha256": _sha256_array(teacher_predictions),
            "teacher_deltas_sha256": _sha256_array(teacher_deltas),
            "sample_count": int(ids.size),
            "pair_latent_width": 1024,
            "v22_train_predictions": str(v22["path"]),
            "v22_train_predictions_sha256": v22["sha256"],
            "v22_train_export_metadata": str(v22["metadata_path"]),
            "v22_train_export_metadata_sha256": v22["metadata_sha256"],
            "sealed_v22_checkpoint": v22["sealed_v22_checkpoint"],
            "sealed_v22_checkpoint_sha256": v22["sealed_v22_checkpoint_sha256"],
            "v22_train_predictions_are_training_only": True,
            "teacher_metrics_on_train_oof": {
                key: float(value) for key, value in affinity_metrics(labels, teacher_predictions).items()
            },
            "readout_recipe": {
                "target": "label_minus_frozen_v22_full",
                "pca_components_requested": int(args.components),
                "pca_whiten": True,
                "estimator": "ExtraTreesRegressor",
                "estimators": int(args.estimators),
                "minimum_leaf": int(args.minimum_leaf),
                "max_features": float(args.max_features),
                "seed": int(args.seed),
                "fold_recipes": recipes,
            },
            "teacher_lineage": teacher_lineages,
            "initialization_policies": initialization_policies,
            "full_ancestry_train_only": initialization_policies == ["fold_local_pretraining"],
        }
        _atomic_json(temporary / "metadata.json", metadata)
        os.replace(str(temporary), str(output))
    except BaseException:
        raise
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def load_train_teacher_cache(cache_dir, train_manifest, expected_width=None):
    cache_dir = Path(cache_dir).expanduser().resolve()
    metadata_path = cache_dir / "metadata.json"
    archive_path = cache_dir / "train_teacher_oof.npz"
    if not metadata_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("V31 train-only pair teacher cache is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("version") != VERSION:
        raise ValueError("V31 pair teacher cache schema/version mismatch")
    if metadata.get("status") != "built" or metadata.get("labels_scope") != "train_only":
        raise ValueError("V31 pair teacher cache is not a completed train-only artifact")
    if (
        metadata.get("official_valid_manifest_read") is not False
        or metadata.get("valid_manifest_read") is not False
        or metadata.get("test_manifest_read") is not False
    ):
        raise ValueError("V31 pair teacher cache accessed official validation/test")
    if metadata.get("teacher_target_regime") != "strict_outer_sequence_grouped_contact_pair_latent_oof_v22_delta":
        raise ValueError("V31 pair teacher target regime mismatch")
    if metadata.get("train_manifest_sha256") != manifest_fingerprint(train_manifest):
        raise ValueError("V31 pair teacher train-manifest lineage mismatch")
    if Path(metadata.get("archive", "")).expanduser().resolve() != archive_path:
        raise ValueError("V31 pair teacher archive path mismatch")
    if metadata.get("archive_sha256") != sha256_file(archive_path):
        raise ValueError("V31 pair teacher archive checksum mismatch")
    fold_root = Path(metadata.get("fold_root", "")).expanduser().resolve()
    fold_metadata = load_fold_metadata(fold_root, train_manifest)
    if metadata.get("folds_sha256") != sha256_file(fold_root / "folds.json"):
        raise ValueError("V31 pair teacher fold root changed after cache build")
    with np.load(str(archive_path), allow_pickle=False) as archive:
        required = {
            "sample_ids", "structure_keys", "labels", "dataset_index", "fold_ids", "teacher_predictions", "teacher_deltas"
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("V31 pair teacher archive misses %s" % sorted(missing))
        values = {name: np.asarray(archive[name]) for name in required}
    ids = values["sample_ids"].astype(str)
    keys = values["structure_keys"].astype(str)
    labels = np.asarray(values["labels"], dtype=np.float64)
    indices = np.asarray(values["dataset_index"], dtype=np.int64)
    fold_ids = np.asarray(values["fold_ids"], dtype=np.int64)
    teacher = np.asarray(values["teacher_predictions"], dtype=np.float64)
    deltas = np.asarray(values["teacher_deltas"], dtype=np.float64)
    if (
        ids.ndim != 1
        or labels.shape != ids.shape
        or keys.shape != ids.shape
        or indices.shape != ids.shape
        or fold_ids.shape != ids.shape
        or teacher.shape != ids.shape
        or deltas.shape != ids.shape
    ):
        raise ValueError("V31 pair teacher archive arrays have incompatible shapes")
    if len(set(ids.tolist())) != ids.size or len(set(keys.tolist())) != keys.size:
        raise ValueError("V31 pair teacher archive IDs/keys are not unique")
    if not np.all(indices == 0) or not np.isfinite(labels).all() or not np.isfinite(teacher).all() or not np.isfinite(deltas).all():
        raise ValueError("V31 pair teacher archive values are invalid")
    rows = _rows(train_manifest, fold_metadata.get("limit"))
    expected_ids = np.asarray([str(row["sample_id"]) for row in rows])
    expected_keys = np.asarray([str(row["structure_key"]) for row in rows])
    expected_labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    expected_folds = np.asarray(
        [int(fold_metadata["sequence_to_outer_fold"][str(row["sequence_id"])]) for row in rows], dtype=np.int64
    )
    if ids.size != expected_ids.size or set(ids.tolist()) != set(expected_ids.tolist()):
        raise ValueError("V31 pair teacher archive does not exactly cover LP train rows")
    positions = {key: index for index, key in enumerate(ids.tolist())}
    order = np.asarray([positions[key] for key in expected_ids], dtype=np.int64)
    if not np.allclose(labels[order], expected_labels, rtol=0.0, atol=1.0e-6):
        raise ValueError("V31 pair teacher archive labels differ from train manifest")
    if not np.array_equal(keys[order], expected_keys) or not np.array_equal(fold_ids[order], expected_folds):
        raise ValueError("V31 pair teacher archive key/fold alignment changed")
    checksums = {
        "sample_ids_sha256": _sha256_array(ids),
        "structure_keys_sha256": _sha256_array(keys),
        "fold_ids_sha256": _sha256_array(fold_ids),
        "teacher_predictions_sha256": _sha256_array(teacher),
        "teacher_deltas_sha256": _sha256_array(deltas),
    }
    for name, observed in checksums.items():
        if metadata.get(name) != observed:
            raise ValueError("V31 pair teacher checksum mismatch: %s" % name)
    if int(metadata.get("sample_count", -1)) != int(ids.size):
        raise ValueError("V31 pair teacher sample count mismatch")
    if expected_width is not None and int(expected_width) != 1:
        raise ValueError("V31 pair scalar teacher width must be one")
    return {
        "metadata": metadata,
        "cache_dir": str(cache_dir),
        "metadata_sha256": sha256_file(metadata_path),
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "predictions_by_sample_id": {key: float(value) for key, value in zip(ids.tolist(), teacher.tolist())},
        "deltas_by_sample_id": {key: float(value) for key, value in zip(ids.tolist(), deltas.tolist())},
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--v22-train-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--estimators", type=int, default=DEFAULT_ESTIMATORS)
    parser.add_argument("--minimum-leaf", type=int, default=DEFAULT_MINIMUM_LEAF)
    parser.add_argument("--max-features", type=float, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--require-full-folds", action="store_true")
    return parser


def main(argv=None):
    return 0 if build_teacher_cache(build_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    main()
