"""Build a train-only OOF privileged-MSA affinity teacher for V30.

This module deliberately has no validation/test-manifest argument.  It uses
only the LP-PDBBind rows assigned to the supplied training manifest and the
frozen V13 *pretraining* feature archive.  In particular it rejects the
validation-fitted V20 final model and does not attempt to reconstruct V26.
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

from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.optional_structure_staged_v22.data import manifest_fingerprint, read_jsonl, sha256_file


VERSION = "optional-structure-privileged-msa-teacher-v30"
SCHEMA_VERSION = 1
DEFAULT_COMPONENTS = 96
DEFAULT_ESTIMATORS = 400
DEFAULT_MINIMUM_LEAF = 3
DEFAULT_MAX_FEATURES = 0.85


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
            # fdopen takes ownership of the descriptor.  Do not close it
            # between mkstemp and fdopen, otherwise every cache build fails
            # with EBADF before metadata can be sealed.
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


def _lp_rows(train_manifest):
    rows = [row for row in read_jsonl(train_manifest) if str(row["dataset"]) == "LP_PDBBind"]
    if not rows:
        raise ValueError("V30 teacher training manifest has no LP-PDBBind rows")
    if any(str(row.get("split", "")) != "train" for row in rows):
        raise ValueError("V30 teacher manifest must contain LP-PDBBind train rows only")
    keys = [str(row.get("structure_key", "")) for row in rows]
    if not all(keys):
        raise ValueError("V30 LP rows require structure_key values")
    if len(set(keys)) != len(keys):
        raise ValueError("V30 LP structure_key values are not unique")
    return rows


def _grouped_hash_folds(rows, count):
    count = int(count)
    if count < 2:
        raise ValueError("V30 OOF fold count must be at least two")
    # Sequence-level grouping ensures that an exact sequence ID never appears
    # in both the OOF fit and holdout partitions.  It is not a claim of full
    # homolog- or ligand-level decontamination.
    groups = [str(row["sequence_id"]) for row in rows]
    folds = np.asarray(
        [
            int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:16], 16) % count
            for group in groups
        ],
        dtype=np.int64,
    )
    unique = np.unique(folds)
    if unique.size != count:
        raise ValueError("V30 OOF sequence folds are empty")
    return folds


def _validate_build_arguments(args):
    """Reject underspecified OOF recipes before opening a large feature archive."""

    if int(args.folds) < 2:
        raise ValueError("V30 OOF fold count must be at least two")
    if int(args.components) < 1:
        raise ValueError("V30 OOF PCA component count must be positive")
    if int(args.estimators) < 1:
        raise ValueError("V30 OOF estimator count must be positive")
    if int(args.minimum_leaf) < 1:
        raise ValueError("V30 OOF minimum leaf must be positive")
    if not 0.0 < float(args.max_features) <= 1.0:
        raise ValueError("V30 OOF max_features must be in (0, 1]")
    if int(args.n_jobs) == 0:
        raise ValueError("V30 OOF n_jobs cannot be zero")


def _load_aligned_pretrain_features(feature_path, rows):
    feature_path = Path(feature_path).expanduser().resolve()
    with np.load(str(feature_path), allow_pickle=False) as archive:
        required = {"keys", "labels", "pretrain_features"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("V30 MSA feature archive is missing: %s" % sorted(missing))
        source_keys = np.asarray(archive["keys"]).astype(str)
        source_labels = np.asarray(archive["labels"], dtype=np.float64)
        source_features = np.asarray(archive["pretrain_features"], dtype=np.float32)
    if source_features.ndim != 2 or source_features.shape[0] != source_keys.size:
        raise ValueError("V30 pretraining features have an invalid shape")
    if source_labels.shape != (source_keys.size,):
        raise ValueError("V30 pretraining feature labels have an invalid shape")
    if not np.isfinite(source_features).all():
        raise ValueError("V30 pretraining features contain non-finite values")
    if len(set(source_keys.tolist())) != source_keys.size:
        raise ValueError("V30 pretraining feature keys are not unique")
    positions = {key: index for index, key in enumerate(source_keys.tolist())}
    requested = [str(row["structure_key"]) for row in rows]
    if source_keys.size != len(requested) or set(source_keys.tolist()) != set(requested):
        raise ValueError("V30 pretraining feature keys do not exactly match LP train rows")
    try:
        order = np.asarray([positions[key] for key in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError("V30 pretraining features do not cover all LP train rows") from error
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    if not np.allclose(source_labels[order], labels, rtol=0.0, atol=1.0e-6):
        raise ValueError("V30 MSA feature labels differ from train manifest")
    return feature_path, labels, source_features[order]


def _fit_fold(features, labels, train, holdout, args, fold):
    components = min(int(args.components), int(train.sum()) - 1, int(features.shape[1]))
    if components <= 0:
        raise ValueError("V30 OOF PCA component count is invalid")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=int(args.seed) + 1009 * int(fold),
    )
    train_features = pca.fit_transform(features[train])
    holdout_features = pca.transform(features[holdout])
    tree = ExtraTreesRegressor(
        n_estimators=int(args.estimators),
        min_samples_leaf=int(args.minimum_leaf),
        max_features=float(args.max_features),
        n_jobs=int(args.n_jobs),
        random_state=int(args.seed) + 1009 * int(fold) + 17,
    )
    tree.fit(train_features, labels[train])
    return tree.predict(holdout_features).astype(np.float64), components


def build_teacher_cache(args):
    _validate_build_arguments(args)
    train_manifest = Path(args.train_manifest).expanduser().resolve()
    feature_path = Path(args.msa_train_features).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("V30 teacher output exists; reuse it directly or choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build beneath a sibling first, then atomically publish the directory.
    # An interrupted PCA/forest fit consequently never leaves a half-built
    # cache at the requested final path, and a retry with the same CACHE_ID
    # cannot overwrite an ambiguous partial artifact.
    temporary_output = Path(
        tempfile.mkdtemp(prefix=".%s.building-" % output.name, dir=str(output.parent))
    )
    rows = _lp_rows(train_manifest)
    feature_path, labels, features = _load_aligned_pretrain_features(feature_path, rows)
    folds = _grouped_hash_folds(rows, args.folds)
    prediction = np.full(labels.shape, np.nan, dtype=np.float64)
    components_by_fold = {}
    for fold in range(int(args.folds)):
        holdout = folds == fold
        train = ~holdout
        # _fit_fold safely clips requested PCA width to train_count - 1.  A
        # small smoke should therefore be valid whenever a fold has at least
        # two fit rows, even if it retains the formal components=96 default.
        if not holdout.any() or train.sum() < 2:
            raise ValueError("V30 OOF split is too small for PCA")
        current, components = _fit_fold(features, labels, train, holdout, args, fold)
        prediction[holdout] = current
        components_by_fold[str(fold)] = int(components)
    if not np.isfinite(prediction).all():
        raise RuntimeError("V30 OOF teacher prediction is incomplete")
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    structure_keys = np.asarray([str(row["structure_key"]) for row in rows])
    metrics = {key: float(value) for key, value in affinity_metrics(labels, prediction).items()}
    temporary_archive_path = temporary_output / "train_teacher_oof.npz"
    archive_path = output / "train_teacher_oof.npz"
    _atomic_npz(
        temporary_archive_path,
        sample_ids=sample_ids,
        structure_keys=structure_keys,
        labels=labels,
        dataset_index=np.zeros(labels.shape, dtype=np.int64),
        fold_ids=folds,
        teacher_predictions=prediction,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "built",
        "output": str(output),
        "archive": str(archive_path),
        "archive_sha256": sha256_file(temporary_archive_path),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": manifest_fingerprint(train_manifest),
        "msa_train_features": str(feature_path),
        "msa_train_features_sha256": sha256_file(feature_path),
        "feature_field": "pretrain_features",
        "labels_scope": "train_only",
        "teacher_target_regime": "sequence_grouped_cross_fitted_oof",
        "valid_manifest_read": False,
        "test_manifest_read": False,
        "fold_group": "sequence_id",
        "folds": int(args.folds),
        "fold_ids_sha256": _sha256_array(folds),
        "sample_ids_sha256": _sha256_array(sample_ids),
        "structure_keys_sha256": _sha256_array(structure_keys),
        "teacher_predictions_sha256": _sha256_array(prediction),
        "sample_count": int(sample_ids.size),
        "teacher_metrics_on_train_oof": metrics,
        "recipe": {
            "pca_components_requested": int(args.components),
            "pca_components_by_fold": components_by_fold,
            "pca_whiten": True,
            "estimator": "ExtraTreesRegressor",
            "estimators": int(args.estimators),
            "minimum_leaf": int(args.minimum_leaf),
            "max_features": float(args.max_features),
            "seed": int(args.seed),
        },
    }
    _atomic_json(temporary_output / "metadata.json", metadata)
    os.replace(str(temporary_output), str(output))
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def load_train_teacher_cache(cache_dir, train_manifest, expected_width=None):
    cache_dir = Path(cache_dir).expanduser().resolve()
    metadata_path = cache_dir / "metadata.json"
    archive_path = cache_dir / "train_teacher_oof.npz"
    if not metadata_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("V30 train-only teacher cache is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("V30 teacher cache schema mismatch")
    if metadata.get("version") != VERSION:
        raise ValueError("V30 teacher cache version mismatch")
    if metadata.get("status") != "built":
        raise ValueError("V30 teacher cache is not a completed build")
    if metadata.get("feature_field") != "pretrain_features":
        raise ValueError("V30 teacher cache used a non-pretraining feature field")
    if metadata.get("labels_scope") != "train_only":
        raise ValueError("V30 teacher cache was not built from train-only labels")
    if metadata.get("valid_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
        raise ValueError("V30 teacher cache accessed validation/test inputs")
    if metadata.get("teacher_target_regime") != "sequence_grouped_cross_fitted_oof":
        raise ValueError("V30 teacher cache target regime mismatch")
    if metadata.get("fold_group") != "sequence_id" or int(metadata.get("folds", 0)) < 2:
        raise ValueError("V30 teacher cache fold protocol mismatch")
    if metadata.get("train_manifest_sha256") != manifest_fingerprint(train_manifest):
        raise ValueError("V30 teacher cache train-manifest lineage mismatch")
    if Path(metadata.get("archive", "")).expanduser().resolve() != archive_path:
        raise ValueError("V30 teacher cache archive path mismatch")
    if metadata.get("archive_sha256") != sha256_file(archive_path):
        raise ValueError("V30 teacher cache archive checksum mismatch")
    with np.load(str(archive_path), allow_pickle=False) as archive:
        required = {
            "sample_ids",
            "structure_keys",
            "labels",
            "dataset_index",
            "fold_ids",
            "teacher_predictions",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("V30 teacher archive is missing: %s" % sorted(missing))
        values = {name: np.asarray(archive[name]) for name in required}
    ids = values["sample_ids"].astype(str)
    structure_keys = values["structure_keys"].astype(str)
    labels = np.asarray(values["labels"], dtype=np.float64)
    dataset_index = np.asarray(values["dataset_index"], dtype=np.int64)
    fold_ids = np.asarray(values["fold_ids"], dtype=np.int64)
    teacher = np.asarray(values["teacher_predictions"], dtype=np.float64)
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("V30 teacher archive sample IDs are not unique")
    if len(set(structure_keys.tolist())) != structure_keys.size:
        raise ValueError("V30 teacher archive structure keys are not unique")
    if (
        labels.shape != teacher.shape
        or dataset_index.shape != teacher.shape
        or fold_ids.shape != teacher.shape
        or structure_keys.shape != teacher.shape
    ):
        raise ValueError("V30 teacher archive arrays have incompatible shapes")
    if not np.all(dataset_index == 0):
        raise ValueError("V30 teacher archive must contain LP-PDBBind only")
    if not np.isfinite(labels).all() or not np.isfinite(teacher).all():
        raise ValueError("V30 teacher archive contains non-finite values")
    fold_count = int(metadata["folds"])
    if np.any(fold_ids < 0) or np.any(fold_ids >= fold_count) or set(fold_ids.tolist()) != set(range(fold_count)):
        raise ValueError("V30 teacher archive fold IDs are invalid")
    checksums = {
        "sample_ids_sha256": _sha256_array(ids),
        "structure_keys_sha256": _sha256_array(structure_keys),
        "fold_ids_sha256": _sha256_array(fold_ids),
        "teacher_predictions_sha256": _sha256_array(teacher),
    }
    for name, observed in checksums.items():
        if metadata.get(name) != observed:
            raise ValueError("V30 teacher cache %s mismatch" % name)
    if int(metadata.get("sample_count", -1)) != int(ids.size):
        raise ValueError("V30 teacher cache sample count mismatch")
    rows = _lp_rows(train_manifest)
    requested_ids = np.asarray([str(row["sample_id"]) for row in rows])
    requested_keys = np.asarray([str(row["structure_key"]) for row in rows])
    if ids.size != requested_ids.size or set(ids.tolist()) != set(requested_ids.tolist()):
        raise ValueError("V30 teacher archive sample IDs do not exactly match train LP rows")
    positions = {key: index for index, key in enumerate(ids.tolist())}
    try:
        order = np.asarray([positions[key] for key in requested_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("V30 teacher archive does not cover train LP rows") from error
    manifest_labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    if not np.allclose(labels[order], manifest_labels, rtol=0.0, atol=1.0e-6):
        raise ValueError("V30 teacher archive labels differ from train manifest")
    if not np.array_equal(structure_keys[order], requested_keys):
        raise ValueError("V30 teacher archive structure keys differ from train manifest")
    result = {
        "metadata": metadata,
        "cache_dir": str(cache_dir),
        "metadata_sha256": sha256_file(metadata_path),
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "predictions_by_sample_id": {
            key: float(value) for key, value in zip(ids.tolist(), teacher.tolist())
        },
    }
    if expected_width is not None and int(expected_width) != 1:
        raise ValueError("V30 scalar teacher width must be one")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--msa-train-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--estimators", type=int, default=DEFAULT_ESTIMATORS)
    parser.add_argument("--minimum-leaf", type=int, default=DEFAULT_MINIMUM_LEAF)
    parser.add_argument("--max-features", type=float, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser


def main(argv=None):
    return 0 if build_teacher_cache(build_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    main()
