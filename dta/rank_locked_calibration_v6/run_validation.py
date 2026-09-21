"""Train/valid-only validation orchestration for calibration v6."""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np
import torch

from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import sha256_file

from . import CANDIDATE_NAMES, FORMAL_COUNTS, VERSION
from .calibrators import MonotonicSplineCalibrator, PositiveAffineCalibrator
from .data import (
    RankLockedCalibrationDataset,
    load_validation_caches,
    reconstruct_v5_anchor,
)
from .folds import assign_grouped_folds, load_fold_rows
from .projection import rank_locked_project
from .selection import (
    ValidationCandidate,
    seal_validation_package,
    select_candidate,
    validate_candidate_inventory,
)
from .training import (
    ConditionalTrainingConfig,
    NonfiniteTrainingError,
    build_run_lineage,
    cross_fit_analytic,
    ensemble_then_project,
    fit_conditional_seed,
    predict_conditional_checkpoint,
)


def _canonical_json_bytes(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("validation payload must be canonical JSON compatible") from error


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + "-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class ValidationMaterial:
    candidates: tuple
    cache_identities: Mapping
    normalization: Mapping
    fold_map: Mapping
    selected_models: Mapping
    valid_keys: np.ndarray
    valid_labels: np.ndarray
    rollback: Mapping

    def __post_init__(self):
        candidates = validate_candidate_inventory(self.candidates)
        if not isinstance(self.cache_identities, Mapping) or set(
            self.cache_identities
        ) != {"train", "valid"}:
            raise ValueError("validation cache identities must contain train and valid")
        if not isinstance(self.normalization, Mapping):
            raise ValueError("validation normalization must be a mapping")
        if not isinstance(self.fold_map, Mapping):
            raise ValueError("validation fold map must be a mapping")
        if not isinstance(self.selected_models, Mapping) or set(
            self.selected_models
        ) != set(CANDIDATE_NAMES):
            raise ValueError("validation selected model inventory does not match")
        keys = np.asarray(self.valid_keys)
        labels = np.asarray(self.valid_labels)
        if keys.ndim != 1 or keys.size == 0 or labels.shape != keys.shape:
            raise ValueError("validation key and label arrays do not align")
        if len(set(str(value) for value in keys)) != keys.shape[0]:
            raise ValueError("validation keys must be unique")
        if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
            raise ValueError("validation labels must be finite")
        if any(candidate.predictions.shape != keys.shape for candidate in candidates):
            raise ValueError("candidate predictions do not match validation rows")
        if not isinstance(self.rollback, Mapping):
            raise ValueError("validation rollback must be a mapping")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self, "cache_identities", MappingProxyType(dict(self.cache_identities))
        )
        object.__setattr__(
            self, "normalization", MappingProxyType(dict(self.normalization))
        )
        object.__setattr__(self, "fold_map", MappingProxyType(dict(self.fold_map)))
        object.__setattr__(
            self, "selected_models", MappingProxyType(dict(self.selected_models))
        )
        object.__setattr__(self, "valid_keys", keys.copy())
        object.__setattr__(self, "valid_labels", labels.astype(np.float64))
        object.__setattr__(self, "rollback", MappingProxyType(dict(self.rollback)))


def _limit_value(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError("limit must be a positive integer")
    return int(value)


def _inventory_lock(output_root, limit):
    payload = {
        "schema_version": 1,
        "status": "locked",
        "version": VERSION,
        "mode": "formal" if limit is None else "smoke",
        "limit": limit,
        "candidate_names": list(CANDIDATE_NAMES),
    }
    path = output_root / "candidate_inventory_lock.json"
    if path.exists():
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError("candidate inventory lock is unavailable") from error
        if raw != _canonical_json_bytes(payload):
            raise ValueError("candidate inventory lock does not match")
    else:
        _atomic_json(path, payload)
    return payload


def run_validation(args, *, hooks=None):
    if hooks is None:
        hooks = DefaultValidationHooks()
    for name in ("load_train", "fit_train", "load_valid", "evaluate"):
        if not callable(getattr(hooks, name, None)):
            raise ValueError("validation hooks are incomplete")
    output_value = getattr(args, "output_root", None)
    if not isinstance(output_value, (str, os.PathLike)):
        raise ValueError("output root is invalid")
    output_root = Path(output_value).expanduser().resolve()
    if output_root.is_symlink():
        raise ValueError("output root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    limit = _limit_value(getattr(args, "limit", None))

    train_context = hooks.load_train(args)
    lock = _inventory_lock(output_root, limit)
    train_artifacts = hooks.fit_train(train_context, args)
    valid_context = hooks.load_valid(train_context, train_artifacts, args)
    material = hooks.evaluate(
        train_context, train_artifacts, valid_context, args
    )
    if not isinstance(material, ValidationMaterial):
        raise ValueError("validation hooks returned invalid material")
    candidates = validate_candidate_inventory(material.candidates)
    selected = select_candidate(candidates)

    summary = {
        "schema_version": 1,
        "version": VERSION,
        "mode": lock["mode"],
        "limit": limit,
        "candidate_names": list(CANDIDATE_NAMES),
        "candidate_metrics": {
            candidate.name: dict(candidate.validation_metrics)
            for candidate in candidates
        },
    }
    if limit is not None:
        result = dict(summary)
        result.update(
            {
                "status": "smoke_complete",
                "selected_name": None if selected is None else selected.name,
                "official_test_allowed": False,
            }
        )
        _atomic_json(output_root / "smoke_result.json", result)
        return result
    if selected is None:
        result = dict(summary)
        result.update(
            {
                "status": "validation_failed",
                "selected_name": None,
                "official_test_allowed": False,
                "fallback": dict(material.rollback),
            }
        )
        _atomic_json(output_root / "validation_failed.json", result)
        return result

    package = output_root / "validation_package"
    seal_validation_package(
        package,
        selected=selected,
        candidates=candidates,
        cache_identities=material.cache_identities,
        normalization=material.normalization,
        fold_map=material.fold_map,
        selected_model=material.selected_models[selected.name],
        valid_keys=material.valid_keys,
        valid_labels=material.valid_labels,
        valid_predictions=selected.predictions,
        rollback=material.rollback,
    )
    result = dict(summary)
    result.update(
        {
            "status": "passed",
            "selected_name": selected.name,
            "selected_validation_metrics": dict(selected.validation_metrics),
            "official_test_allowed": True,
            "package": str(package),
        }
    )
    _atomic_json(output_root / "validation_result.json", result)
    return result


class DefaultValidationHooks:
    def __init__(self, *, conditional_runner=None):
        self.conditional_runner = (
            fit_conditional_seed
            if conditional_runner is None
            else conditional_runner
        )
        if not callable(self.conditional_runner):
            raise ValueError("conditional runner must be callable")

    def load_train(self, args):
        limit = _limit_value(getattr(args, "limit", None))
        references = load_validation_caches(
            args.cache_root,
            {"train": FORMAL_COUNTS["train"], "valid": FORMAL_COUNTS["valid"]},
            splits=("train", "valid"),
        )
        anchor = reconstruct_v5_anchor(
            args.v5_package,
            args.v5_validation_root,
            references,
        )
        train_reference = references["train"]
        train_count = len(train_reference.cache.keys)
        selected_count = train_count if limit is None else min(limit, train_count)
        if selected_count < 5:
            raise ValueError("train limit must include at least five rows")
        selected_indices = np.arange(selected_count, dtype=np.int64)
        selected_keys = tuple(
            train_reference.cache.keys[index] for index in selected_indices
        )
        rows = load_fold_rows(
            args.metadata_csv,
            selected_keys,
            expected_split="train",
            allow_invalid_scaffold_fallback=True,
        )
        assignment = assign_grouped_folds(rows, fold_count=5, seed=42)
        fold_ids = np.asarray(
            [assignment.by_key[key] for key in selected_keys], dtype=np.int64
        )
        full_dataset = RankLockedCalibrationDataset(
            train_reference.cache, anchor.train_predictions
        )
        dataset = (
            full_dataset
            if selected_count == train_count
            else torch.utils.data.Subset(full_dataset, selected_indices.tolist())
        )
        labels = np.asarray(
            train_reference.cache.labels[selected_indices], dtype=np.float64
        )
        base = np.asarray(
            anchor.train_predictions[selected_indices], dtype=np.float64
        )
        try:
            normalization = json.loads(
                (train_reference.directory / "normalization.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ValueError("train normalization is malformed") from error
        maximum_epochs = 200
        patience = 20
        if limit is not None:
            maximum_epochs = int(getattr(args, "smoke_epochs", 2))
            if maximum_epochs <= 0 or maximum_epochs > 200:
                raise ValueError("smoke epochs must be between one and 200")
            patience = min(20, maximum_epochs)
        training_config = ConditionalTrainingConfig(
            maximum_epochs=maximum_epochs,
            patience=patience,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-3,
            gradient_clip=1.0,
            device=args.device,
        )
        anchor_hash = hashlib.sha256(
            np.asarray(anchor.train_predictions, dtype=np.float64).tobytes()
        ).hexdigest()
        package = Path(args.v5_package).expanduser().resolve()
        package_seal = package / "selection_seal.json"
        if not package_seal.is_file():
            raise ValueError("v5 package selection seal is unavailable")
        package_hash = sha256_file(package_seal)
        normalization_hash = train_reference.file_sha256["normalization.json"]
        lineages = {
            seed: build_run_lineage(
                candidate="conditional_seed_" + str(seed),
                seed=seed,
                config=training_config,
                source_revision=args.source_revision,
                train_identity=train_reference.identity,
                valid_identity=references["valid"].identity,
                v5_package_sha256=package_hash,
                v5_anchor_sha256=anchor_hash,
                fold_map_sha256=assignment.sha256,
                normalization_sha256=normalization_hash,
            )
            for seed in (1, 7, 21)
        }
        fallback_result = Path(args.v5_fallback_result).expanduser().resolve()
        if fallback_result.is_symlink() or not fallback_result.is_file():
            raise ValueError("v5 fallback result is unavailable")
        rollback = {
            "schema_version": 1,
            "promotion_status": "not_promoted",
            "v5_validation_package": str(package),
            "v5_selection_seal_sha256": package_hash,
            "v5_official_test_result": str(fallback_result),
            "v5_official_test_result_sha256": sha256_file(fallback_result),
        }
        return {
            "references": references,
            "anchor": anchor,
            "dataset": dataset,
            "fold_ids": fold_ids,
            "base_predictions": base,
            "labels": labels,
            "anchor_descriptor": {
                "type": "v5_static_anchor",
                "selected_name": anchor.selected_name,
                "static_spec": anchor.static_spec.to_dict(),
            },
            "training_config": training_config,
            "conditional_output_root": Path(args.output_root).expanduser().resolve()
            / "conditional_runs",
            "lineages": lineages,
            "resume": bool(getattr(args, "resume", False)),
            "selected_train_indices": selected_indices,
            "selected_valid_count": (
                FORMAL_COUNTS["valid"] if limit is None else min(limit, FORMAL_COUNTS["valid"])
            ),
            "cache_identities": {
                "train": train_reference.identity,
                "valid": references["valid"].identity,
            },
            "normalization": normalization,
            "fold_map": {
                "keys": np.asarray(selected_keys, dtype=np.str_),
                "fold_ids": fold_ids,
            },
            "rollback": rollback,
        }

    def fit_train(self, train_context, args):
        if not isinstance(train_context, Mapping):
            raise ValueError("default train context is invalid")
        required = {
            "dataset",
            "fold_ids",
            "base_predictions",
            "labels",
            "anchor_descriptor",
            "training_config",
            "conditional_output_root",
            "lineages",
            "resume",
        }
        if not required.issubset(train_context):
            raise ValueError("default train context is incomplete")
        dataset = train_context["dataset"]
        fold_ids = np.asarray(train_context["fold_ids"], dtype=np.int64)
        base = np.asarray(train_context["base_predictions"], dtype=np.float64)
        labels = np.asarray(train_context["labels"], dtype=np.float64)
        if base.shape != labels.shape or base.shape != fold_ids.shape:
            raise ValueError("default train arrays do not align")
        artifacts = {
            "identity": {
                "status": "complete",
                "model_descriptor": train_context["anchor_descriptor"],
                "oof_summary": {
                    "status": "complete",
                    "metrics": affinity_metrics(labels, base),
                    "note": "frozen expert train predictions are not expert OOF",
                },
            }
        }
        analytic_specs = (
            ("affine_mse", PositiveAffineCalibrator, {"objective": "mse"}),
            ("affine_huber", PositiveAffineCalibrator, {"objective": "huber"}),
            ("spline_8_knots", MonotonicSplineCalibrator, {"knot_count": 8}),
            ("spline_16_knots", MonotonicSplineCalibrator, {"knot_count": 16}),
        )
        for name, model_class, keyword in analytic_specs:
            oof = cross_fit_analytic(
                dataset, fold_ids, candidate=name
            )
            model = model_class.fit(base, labels, **keyword)
            artifacts[name] = {
                "status": "complete",
                "model_descriptor": model.to_dict(),
                "oof_summary": {
                    "status": "complete",
                    "metrics": dict(oof.metrics),
                    "held_out_count_minimum": int(oof.held_out_count.min()),
                    "held_out_count_maximum": int(oof.held_out_count.max()),
                },
            }
        lineages = train_context["lineages"]
        if not isinstance(lineages, Mapping) or set(lineages) != {1, 7, 21}:
            raise ValueError("conditional lineage inventory does not match")
        conditional_statuses = []
        for seed in (1, 7, 21):
            name = "conditional_seed_" + str(seed)
            try:
                completed = self.conditional_runner(
                    dataset,
                    fold_ids,
                    seed=seed,
                    config=train_context["training_config"],
                    output=Path(train_context["conditional_output_root"])
                    / ("seed_" + str(seed)),
                    lineage=lineages[seed],
                    resume=bool(train_context["resume"]),
                )
                artifacts[name] = {
                    "status": "complete",
                    "model_descriptor": completed.checkpoint_best,
                    "oof_summary": {
                        "status": "complete",
                        "result": dict(completed.result),
                        "diagnostics": dict(completed.diagnostics),
                    },
                }
            except NonfiniteTrainingError as error:
                artifacts[name] = {
                    "status": "failed_nonfinite",
                    "model_descriptor": {
                        "type": "conditional",
                        "seed": seed,
                        "error": str(error),
                    },
                    "oof_summary": {
                        "status": "failed_nonfinite",
                        "error": str(error),
                        "diagnostics": dict(error.diagnostics),
                    },
                }
            conditional_statuses.append(artifacts[name]["status"])
        ensemble_status = (
            "complete"
            if all(status == "complete" for status in conditional_statuses)
            else "failed_missing_seed"
        )
        artifacts["conditional_ensemble"] = {
            "status": ensemble_status,
            "model_descriptor": {
                "type": "conditional_ensemble",
                "rule": "arithmetic_mean_before_one_rank_projection",
                "members": [1, 7, 21],
            },
            "oof_summary": {
                "status": ensemble_status,
                "members": [1, 7, 21],
            },
        }
        return {name: artifacts[name] for name in CANDIDATE_NAMES}

    def load_valid(self, train_context, train_artifacts, args):
        if not isinstance(train_context, Mapping):
            raise ValueError("default train context is invalid")
        references = train_context["references"]
        anchor = train_context["anchor"]
        valid_reference = references["valid"]
        full_dataset = RankLockedCalibrationDataset(
            valid_reference.cache, anchor.valid_predictions
        )
        count = int(train_context["selected_valid_count"])
        indices = np.arange(count, dtype=np.int64)
        dataset = (
            full_dataset
            if count == len(valid_reference.cache.keys)
            else torch.utils.data.Subset(full_dataset, indices.tolist())
        )
        keys = np.asarray(
            [valid_reference.cache.keys[index] for index in indices],
            dtype=np.str_,
        )
        labels = np.asarray(
            valid_reference.cache.labels[indices], dtype=np.float64
        )
        base = np.asarray(
            anchor.valid_predictions[indices], dtype=np.float64
        )
        if not np.array_equal(
            base,
            np.asarray(anchor.stored_valid_predictions, dtype=np.float64)[indices],
        ):
            raise ValueError("stored v5 valid anchor does not match")
        return {
            "dataset": dataset,
            "keys": keys,
            "labels": labels,
            "base_predictions": base,
        }

    def evaluate(self, train_context, train_artifacts, valid_context, args):
        if not isinstance(train_context, Mapping) or not isinstance(
            train_artifacts, Mapping
        ) or not isinstance(valid_context, Mapping):
            raise ValueError("default validation contexts are invalid")
        if set(train_artifacts) != set(CANDIDATE_NAMES):
            raise ValueError("train artifact inventory does not match")
        keys = np.asarray(valid_context.get("keys"))
        labels = np.asarray(valid_context.get("labels"), dtype=np.float64)
        base = np.asarray(
            valid_context.get("base_predictions"), dtype=np.float64
        )
        if (
            keys.ndim != 1
            or keys.size == 0
            or labels.shape != keys.shape
            or base.shape != keys.shape
            or not np.isfinite(labels).all()
            or not np.isfinite(base).all()
        ):
            raise ValueError("default validation arrays do not align")
        identity_metrics = affinity_metrics(labels, base)
        candidate_values = {}
        selected_models = {}
        unprojected_conditional = {}
        device = getattr(args, "device", "cuda:0")
        prediction_config = ConditionalTrainingConfig(
            maximum_epochs=1,
            patience=1,
            batch_size=256,
            device=device,
        )

        def add_candidate(
            name,
            status,
            predictions,
            descriptor,
            oof,
            parameters,
            deploy_model=None,
        ):
            predictions = np.asarray(predictions, dtype=np.float64)
            predictions.setflags(write=False)
            metrics = affinity_metrics(labels, predictions)
            if status == "complete" and (
                metrics["CI"] != identity_metrics["CI"]
                or metrics["Spearman"] != identity_metrics["Spearman"]
            ):
                raise ValueError("rank-locked candidate changed anchor rank metrics")
            candidate = ValidationCandidate(
                name=name,
                candidate_type=(
                    "identity"
                    if name == "identity"
                    else "analytic"
                    if name.startswith("affine_") or name.startswith("spline_")
                    else "ensemble"
                    if name == "conditional_ensemble"
                    else "conditional"
                ),
                validation_metrics=metrics,
                parameter_count=parameters,
                predictions=predictions,
                model_descriptor=descriptor,
                oof_summary=oof,
                status=status,
            )
            candidate_values[name] = candidate
            selected_models[name] = {
                "schema_version": 1,
                "name": name,
                "candidate_type": candidate.candidate_type,
                "status": status,
                "model": descriptor if deploy_model is None else deploy_model,
            }

        identity = train_artifacts["identity"]
        add_candidate(
            "identity",
            identity["status"],
            base,
            identity["model_descriptor"],
            identity["oof_summary"],
            0,
        )
        for name in (
            "affine_mse",
            "affine_huber",
            "spline_8_knots",
            "spline_16_knots",
        ):
            artifact = train_artifacts[name]
            descriptor = artifact["model_descriptor"]
            if descriptor.get("type") == "positive_affine":
                model = PositiveAffineCalibrator.from_dict(dict(descriptor))
                parameter_count = 2
            elif descriptor.get("type") == "monotonic_spline":
                model = MonotonicSplineCalibrator.from_dict(dict(descriptor))
                parameter_count = 2 * len(descriptor["x_knots"])
            else:
                raise ValueError("analytic model descriptor does not match")
            proposed = model.predict(base)
            projected, _ = rank_locked_project(base, proposed)
            add_candidate(
                name,
                artifact["status"],
                projected,
                descriptor,
                artifact["oof_summary"],
                parameter_count,
            )
        for seed in (1, 7, 21):
            name = "conditional_seed_" + str(seed)
            artifact = train_artifacts[name]
            status = artifact["status"]
            descriptor = artifact["model_descriptor"]
            if status == "complete":
                proposed = predict_conditional_checkpoint(
                    descriptor,
                    valid_context["dataset"],
                    config=prediction_config,
                )
                projected, _ = rank_locked_project(base, proposed)
                unprojected_conditional[seed] = proposed
                state = descriptor.get("model_state_dict", {})
                parameter_count = sum(
                    int(value.numel())
                    for value in state.values()
                    if hasattr(value, "numel")
                )
                manifest_descriptor = {
                    "type": "conditional_checkpoint",
                    "seed": seed,
                    "model_config": descriptor["model_config"],
                    "lineage_sha256": descriptor.get("lineage", {}).get(
                        "lineage_sha256"
                    ),
                }
            else:
                projected = base.copy()
                parameter_count = 0
                manifest_descriptor = descriptor
            add_candidate(
                name,
                status,
                projected,
                manifest_descriptor,
                artifact["oof_summary"],
                parameter_count,
                deploy_model=descriptor,
            )
        ensemble_artifact = train_artifacts["conditional_ensemble"]
        if len(unprojected_conditional) == 3:
            ensemble_predictions, _, _ = ensemble_then_project(
                base, unprojected_conditional
            )
            ensemble_status = "complete"
            ensemble_parameters = sum(
                candidate_values["conditional_seed_" + str(seed)].parameter_count
                for seed in (1, 7, 21)
            )
            ensemble_descriptor = {
                "type": "conditional_ensemble",
                "rule": "arithmetic_mean_before_one_rank_projection",
                "members": {
                    str(seed): train_artifacts[
                        "conditional_seed_" + str(seed)
                    ]["model_descriptor"]
                    for seed in (1, 7, 21)
                },
            }
            ensemble_manifest_descriptor = {
                "type": "conditional_ensemble",
                "rule": "arithmetic_mean_before_one_rank_projection",
                "members": [1, 7, 21],
            }
        else:
            ensemble_predictions = base.copy()
            ensemble_status = "failed_missing_seed"
            ensemble_parameters = 0
            ensemble_descriptor = ensemble_artifact["model_descriptor"]
            ensemble_manifest_descriptor = ensemble_descriptor
        add_candidate(
            "conditional_ensemble",
            ensemble_status,
            ensemble_predictions,
            ensemble_manifest_descriptor,
            ensemble_artifact["oof_summary"],
            ensemble_parameters,
            deploy_model=ensemble_descriptor,
        )
        return ValidationMaterial(
            candidates=tuple(candidate_values[name] for name in CANDIDATE_NAMES),
            cache_identities=train_context["cache_identities"],
            normalization=train_context["normalization"],
            fold_map=train_context["fold_map"],
            selected_models=selected_models,
            valid_keys=keys,
            valid_labels=labels,
            rollback=train_context["rollback"],
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--v5-validation-root", required=True)
    parser.add_argument("--v5-package", required=True)
    parser.add_argument("--v5-fallback-result", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = run_validation(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DefaultValidationHooks",
    "ValidationMaterial",
    "build_arg_parser",
    "run_validation",
]
