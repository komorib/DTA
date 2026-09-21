"""Train/validation-only orchestration for anchored expert fusion v5."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from dta.anchored_expert_fusion_v5 import data as data_module
from dta.anchored_expert_fusion_v5 import training as training_module
from dta.anchored_expert_fusion_v5 import selection as selection_module
from dta.anchored_expert_fusion_v5.selection import (
    CandidateRecord,
    FORMAL_BUDGETS,
    FORMAL_SEEDS,
    build_candidate_inventory,
    build_residual_ensembles,
    persist_prediction_artifact,
    seal_validation_package,
    select_validation_candidate,
    validate_relative_path,
)
from dta.anchored_expert_fusion_v5.static_fusion import (
    assign_folds,
    cross_fit_static,
    report_sha256,
)
from dta.rank_fusion_v4.metrics import affinity_metrics
from utils.experiment_integrity import atomic_write_json, sha256_file


BUDGETS = FORMAL_BUDGETS
SEEDS = FORMAL_SEEDS


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run anchored expert fusion validation selection."
    )
    parser.add_argument("--train-cache-root", required=True)
    parser.add_argument("--valid-cache-root", required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--valid-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _arg(args, name, default=None):
    return getattr(args, name, default)


def _validated_inventory_controls(args):
    """Return mode and canonical fixed inventory controls."""
    limit = _arg(args, "limit", None)
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 5
    ):
        raise ValueError("limit must be an integer of at least five")
    supplied_budgets = _arg(args, "budgets", BUDGETS)
    try:
        supplied_budgets = tuple(float(value) for value in supplied_budgets)
    except (TypeError, ValueError) as error:
        raise ValueError("budgets are fixed to the formal values") from error
    if supplied_budgets != BUDGETS:
        raise ValueError("budgets are fixed to 0.10, 0.25, and 0.50")

    try:
        supplied_seeds = tuple(int(value) for value in args.seeds)
    except (TypeError, ValueError) as error:
        raise ValueError("seeds are invalid") from error
    if len(supplied_seeds) != len(set(supplied_seeds)):
        raise ValueError("seeds must be unique")
    if limit is None:
        if set(supplied_seeds) != set(SEEDS) or len(supplied_seeds) != len(SEEDS):
            raise ValueError("formal validation requires exactly all formal seeds")
        return "formal", BUDGETS, SEEDS
    if (
        len(supplied_seeds) < 2
        or any(seed not in SEEDS for seed in supplied_seeds)
    ):
        raise ValueError(
            "smoke validation requires at least two unique formal seeds"
        )
    canonical = tuple(seed for seed in SEEDS if seed in supplied_seeds)
    return "smoke", BUDGETS, canonical


def _ordered_key_hash(keys):
    encoded = json.dumps(
        list(keys), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _limit_reference(reference, limit):
    if limit is None:
        return reference
    cache = reference.cache
    if limit > len(cache.keys):
        raise ValueError("limit exceeds cache population")
    if limit == len(cache.keys):
        return reference
    keys = tuple(cache.keys[:limit])
    metadata = dict(cache.metadata)
    metadata["count"] = limit
    metadata["ordered_key_sha256"] = _ordered_key_hash(keys)
    limited_cache = replace(
        cache,
        keys=keys,
        labels=cache.labels[:limit].copy(),
        predictions=cache.predictions[:limit].copy(),
        standardized_predictions=cache.standardized_predictions[:limit].copy(),
        available=cache.available[:limit].copy(),
        upper_bound_latents=cache.upper_bound_latents[:limit].copy(),
        boltz_latents=cache.boltz_latents[:limit].copy(),
        three_dprot_latents=cache.three_dprot_latents[:limit].copy(),
        metadata=metadata,
    )
    identity = {
        "split": reference.identity["split"],
        "count": limit,
        "manifest_sha256": reference.identity["manifest_sha256"],
        "ordered_key_sha256": metadata["ordered_key_sha256"],
        "files": dict(reference.identity["files"]),
    }
    return data_module.ImmutableCacheRef(
        cache=limited_cache, identity=identity
    )


def _load_validation_inputs(args):
    train_root = Path(args.train_cache_root).expanduser().resolve()
    valid_root = Path(args.valid_cache_root).expanduser().resolve()
    if train_root == valid_root:
        refs = data_module.load_immutable_v4_bundle(
            train_root,
            {"train": int(args.train_count), "valid": int(args.valid_count)},
            splits=("train", "valid"),
        )
    else:
        train = data_module.load_immutable_v4_bundle(
            train_root,
            {"train": int(args.train_count)},
            splits=("train",),
        )
        valid = data_module.load_immutable_v4_bundle(
            valid_root,
            {"valid": int(args.valid_count)},
            splits=("valid",),
        )
        refs = {"train": train["train"], "valid": valid["valid"]}
    limit = _arg(args, "limit", None)
    return {
        split: _limit_reference(reference, limit)
        for split, reference in refs.items()
    }


def _cross_fit(inputs, args):
    train_ref = inputs["train"]
    valid_ref = inputs["valid"]
    common = {
        "train_labels": train_ref.cache.labels,
        "train_predictions": train_ref.cache.predictions,
        "train_available": train_ref.cache.available,
        "validation_keys": valid_ref.cache.keys,
        "validation_labels": valid_ref.cache.labels,
        "validation_predictions": valid_ref.cache.predictions,
        "validation_available": valid_ref.cache.available,
        "seed": 42,
        "fold_count": 5,
    }
    return {
        family: cross_fit_static(**common, family=family)
        for family in ("raw", "rank")
    }


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return str(value)


def _metric_payload(metrics):
    return {
        name: float(metrics[name])
        for name in ("CI", "MAE", "RMSE", "Pearson", "Spearman")
    }


def _report_payload(report):
    canonical_hash = report_sha256(report)
    if report.report_hash != canonical_hash:
        raise ValueError("cross-fit report hash does not match canonical report")
    payload = {
        "family": report.family,
        "validation_keys": [str(value) for value in report.validation_keys],
        "fold_ids": np.asarray(report.fold_ids, dtype=np.int64).tolist(),
        "fold_map_hash": report.fold_map_hash,
        "held_out_count": np.asarray(
            report.held_out_count, dtype=np.int64
        ).tolist(),
        "predictions": np.asarray(
            report.predictions, dtype=np.float64
        ).tolist(),
        "pooled_metrics": _metric_payload(report.pooled_metrics),
        "upper_bound_pooled_metrics": _metric_payload(
            report.upper_bound_pooled_metrics
        ),
        "folds": [
            {
                "fold_id": int(fold.fold_id),
                "held_out_indices": np.asarray(
                    fold.held_out_indices, dtype=np.int64
                ).tolist(),
                "selected_weights": np.asarray(
                    fold.selected_weights, dtype=np.float64
                ).tolist(),
                "candidate_metrics": _metric_payload(fold.candidate_metrics),
                "upper_bound_metrics": _metric_payload(
                    fold.upper_bound_metrics
                ),
                "search_metrics": _metric_payload(fold.search_metrics),
                "slope": float(fold.slope),
                "intercept": float(fold.intercept),
                "stage_counts": [int(value) for value in fold.stage_counts],
                "searched_count": int(fold.searched_count),
            }
            for fold in report.folds
        ],
        "fold_weights": np.asarray(
            report.fold_weights, dtype=np.float64
        ).tolist(),
        "deploy_weights": np.asarray(
            report.deploy_weights, dtype=np.float64
        ).tolist(),
        "gate": {
            "passed": bool(report.gate.passed),
            "reasons": [str(value) for value in report.gate.reasons],
        },
        "validation_available": np.asarray(
            report.validation_available, dtype=np.bool_
        ).tolist(),
        "report_hash": canonical_hash,
    }
    if report_sha256(report) != payload["report_hash"]:
        raise ValueError("cross-fit report serialization is incomplete")
    return payload


def _lineage(
    train_ref,
    valid_ref,
    report_hash=None,
    static_spec_hash=None,
):
    if (report_hash is None) != (static_spec_hash is None):
        raise ValueError("report and static spec lineage must be bound together")
    result = {
        "train_manifest": train_ref.identity["manifest_sha256"],
        "train_ordered_keys": train_ref.identity["ordered_key_sha256"],
        "valid_manifest": valid_ref.identity["manifest_sha256"],
        "valid_ordered_keys": valid_ref.identity["ordered_key_sha256"],
    }
    if report_hash is not None:
        result["cross_fit_report"] = report_hash
        result["static_spec_sha256"] = static_spec_hash
    return result


def _prediction_metrics(labels, predictions):
    result = affinity_metrics(
        np.asarray(labels, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
    )
    return {
        key: float(result[key])
        for key in ("CI", "MAE", "RMSE", "Pearson", "Spearman")
    }



def _persisted_candidate(
    output,
    valid_ref,
    *,
    name,
    candidate_type,
    family,
    predictions,
    validation_metrics,
    lineage_hashes,
):
    artifact_predictions = np.asarray(predictions, dtype=np.float32)
    computed_metrics = _prediction_metrics(
        valid_ref.cache.labels, artifact_predictions
    )
    try:
        supplied_metrics = {
            key: float(validation_metrics[key])
            for key in ("CI", "MAE", "RMSE", "Pearson", "Spearman")
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " validation metrics are malformed") from error
    if not all(np.isfinite(value) for value in supplied_metrics.values()):
        raise ValueError(name + " validation metrics are nonfinite")
    path, digest = persist_prediction_artifact(
        output,
        name,
        valid_ref.cache.keys,
        valid_ref.cache.labels,
        artifact_predictions,
    )
    return CandidateRecord(
        name=name,
        type=candidate_type,
        family=family,
        prediction_path=path,
        prediction_sha256=digest,
        validation_metrics=computed_metrics,
        parent_static_name=None,
        checkpoint_path=None,
        checkpoint_sha256=None,
        lineage_hashes=lineage_hashes,
        member_hashes={},
        budget=None,
        seed=None,
        members=(),
        ensemble_rule=None,
        predictions=predictions,
    )


def _static_candidate(
    family,
    report,
    train_ref,
    valid_ref,
    output,
    lineage_hashes,
):
    return _persisted_candidate(
        output,
        valid_ref,
        name="static_" + family,
        candidate_type="static",
        family=family,
        predictions=report.predictions,
        validation_metrics=dict(report.pooled_metrics),
        lineage_hashes=lineage_hashes,
    )


def _static_spec(family, report, train_ref, valid_ref):
    slopes = [float(fold.slope) for fold in report.folds]
    intercepts = [float(fold.intercept) for fold in report.folds]
    return training_module.StaticSpec(
        family=family,
        deploy_weights=tuple(float(value) for value in report.deploy_weights),
        rank_slope=float(np.median(slopes)) if family == "rank" else 1.0,
        rank_intercept=(
            float(np.median(intercepts)) if family == "rank" else 0.0
        ),
        cross_fit_report_hash=report.report_hash,
        train_cache_identity=train_ref.identity,
        valid_cache_identity=valid_ref.identity,
    )


def _bound_static_lineage(report, spec, train_ref, valid_ref):
    canonical_hash = report_sha256(report)
    slopes = [float(fold.slope) for fold in report.folds]
    intercepts = [float(fold.intercept) for fold in report.folds]
    expected_slope = float(np.median(slopes)) if report.family == "rank" else 1.0
    expected_intercept = (
        float(np.median(intercepts)) if report.family == "rank" else 0.0
    )
    if (
        report.report_hash != canonical_hash
        or spec.family != report.family
        or spec.cross_fit_report_hash != canonical_hash
        or spec.train_cache_identity != train_ref.identity
        or spec.valid_cache_identity != valid_ref.identity
        or not np.array_equal(
            np.asarray(spec.deploy_weights, dtype=np.float64),
            np.asarray(report.deploy_weights, dtype=np.float64),
        )
        or spec.rank_slope != expected_slope
        or spec.rank_intercept != expected_intercept
    ):
        raise ValueError("static spec does not match canonical report lineage")
    return _lineage(
        train_ref,
        valid_ref,
        canonical_hash,
        training_module.static_spec_sha256(spec),
    )


def _relative_artifact(output, value, digest, name):
    if not isinstance(value, (str, Path)):
        raise ValueError(name + " path is missing")
    root = Path(output).expanduser().resolve()
    supplied = Path(value)
    resolved = (
        supplied.expanduser().resolve()
        if supplied.is_absolute()
        else (root / supplied).resolve()
    )
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(name + " path leaves the output root") from error
    validate_relative_path(relative, name + " path")
    if not isinstance(digest, str):
        raise ValueError(name + " artifact hash does not match")
    content = selection_module.read_verified_bytes(
        resolved, expected_sha256=digest, name=name + " artifact"
    )
    return relative, digest, content


def _load_residual_predictions(content, valid_ref):
    try:
        arrays = selection_module.load_verified_npz_bytes(
            content, name="residual prediction artifact"
        )
        if set(arrays) != {"keys", "labels", "predictions", "epoch", "variant"}:
            raise ValueError("residual prediction artifact schema is invalid")
        raw_keys = np.asarray(arrays["keys"])
        raw_labels = np.asarray(arrays["labels"])
        raw_predictions = np.asarray(arrays["predictions"])
        if (raw_keys.ndim != 1 or raw_keys.dtype.kind not in ("U", "S") or
                raw_labels.dtype != np.float32 or
                raw_predictions.dtype != np.float32):
            raise ValueError("residual prediction artifact dtypes are invalid")
        keys = tuple(str(value) for value in raw_keys.tolist())
        labels = raw_labels.reshape(-1)
        predictions = raw_predictions.reshape(-1)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("residual prediction artifact is invalid") from error
    if keys != tuple(valid_ref.cache.keys):
        raise ValueError("residual prediction keys do not match validation")
    if (labels.shape != valid_ref.cache.labels.shape or
            not np.array_equal(labels, valid_ref.cache.labels)):
        raise ValueError("residual prediction labels do not match validation")
    if (predictions.shape != valid_ref.cache.labels.shape or
            not np.isfinite(predictions).all()):
        raise ValueError("residual predictions are invalid")
    return predictions


def _residual_candidate(
    result,
    family,
    budget,
    seed,
    spec,
    train_ref,
    valid_ref,
    output,
    lineage_hashes,
):
    if not isinstance(result, dict) or result.get("status") != "complete":
        raise ValueError("residual result is incomplete")
    prediction_relative, prediction_hash, prediction_bytes = _relative_artifact(
        output,
        result.get("prediction_path"),
        result.get("prediction_sha256"),
        "residual prediction",
    )
    checkpoint_relative, checkpoint_hash, checkpoint_bytes = _relative_artifact(
        output,
        result.get("checkpoint_path"),
        result.get("checkpoint_sha256"),
        "residual checkpoint",
    )
    predictions = _load_residual_predictions(prediction_bytes, valid_ref)
    checkpoint_payload = selection_module.load_verified_torch_bytes(
        checkpoint_bytes, map_location="cpu", name="residual checkpoint"
    )
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("residual checkpoint metadata is missing")
    if (checkpoint_payload.get("epoch") != int(result.get("best_epoch", -1)) or
            checkpoint_payload.get("metrics") != result.get("best_metrics") or
            checkpoint_payload.get("run_config") != result.get("run_config") or
            checkpoint_payload.get("budget") != float(budget) or
            checkpoint_payload.get("seed") != int(seed) or
            checkpoint_payload.get("static_spec_sha256") !=
            training_module.static_spec_sha256(spec) or
            checkpoint_payload.get("cache_lineage") != {
                "train": train_ref.identity,
                "valid": valid_ref.identity,
            }):
        raise ValueError("residual checkpoint metadata does not match result")
    computed = _prediction_metrics(valid_ref.cache.labels, predictions)
    supplied = result.get("best_metrics")
    if not isinstance(supplied, dict) or set(supplied) != set(computed):
        raise ValueError("residual result metrics are malformed")
    if any(abs(float(supplied[key]) - computed[key]) > 1e-6
           for key in computed):
        raise ValueError("residual result metrics do not match prediction artifact")
    budget_name = "%02d" % int(round(float(budget) * 100))
    return CandidateRecord(
        name="residual_%s_%s_%d" % (
            family, budget_name, int(seed)
        ),
        type="residual",
        family=family,
        prediction_path=prediction_relative,
        prediction_sha256=prediction_hash,
        validation_metrics=computed,
        parent_static_name="static_" + family,
        checkpoint_path=checkpoint_relative,
        checkpoint_sha256=checkpoint_hash,
        lineage_hashes=lineage_hashes,
        member_hashes={},
        budget=float(budget),
        seed=int(seed),
        members=(),
        ensemble_rule=None,
        predictions=predictions,
    )


def _expert_candidates(valid_ref, output):
    cache = valid_ref.cache
    predictions = data_module.fallback_predictions(
        cache.predictions, cache.available
    ).astype(np.float64)
    result = []
    for index, name in enumerate(data_module.EXPERT_ORDER):
        values = predictions[:, index]
        result.append(_persisted_candidate(
            output,
            valid_ref,
            name=name,
            candidate_type="expert",
            family="expert",
            predictions=values,
            validation_metrics=affinity_metrics(cache.labels, values),
            lineage_hashes={
                "valid_manifest": valid_ref.identity["manifest_sha256"],
                "valid_ordered_keys": valid_ref.identity[
                    "ordered_key_sha256"
                ],
                "prediction_source": valid_ref.identity["files"][
                    "features.npz"
                ],
            },
        ))
    return tuple(result)


def _fold_map(valid_ref, reports):
    fold_ids = assign_folds(valid_ref.cache.keys, seed=42, fold_count=5)
    for report in reports.values():
        if not np.array_equal(report.fold_ids, fold_ids):
            raise ValueError("cross-fit report fold map does not match")
    return {
        "keys": list(valid_ref.cache.keys),
        "fold_ids": fold_ids.astype(int).tolist(),
    }


def run_validation(args):
    """Run validation-only cross-fitting, residual inventory, and sealing."""
    inventory_mode, budgets, seeds = _validated_inventory_controls(args)
    inputs = _load_validation_inputs(args)
    reports = _cross_fit(inputs, args)
    failed = {
        family: list(report.gate_reasons)
        for family, report in reports.items()
        if not report.gate_passed
    }
    output = Path(args.output).expanduser().resolve()
    if len(failed) == 2:
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "failed_static_gate",
            "reasons": failed,
            "inventory_mode": inventory_mode,
        }
        atomic_write_json(output / "status.json", payload)
        return payload

    train_ref, valid_ref = inputs["train"], inputs["valid"]
    passing_families = tuple(
        family for family in ("raw", "rank")
        if reports[family].gate_passed
    )
    static_candidates = []
    specs = {}
    lineages = {}
    for family in passing_families:
        report = reports[family]
        spec = _static_spec(family, report, train_ref, valid_ref)
        specs[family] = spec
        lineages[family] = _bound_static_lineage(
            report, spec, train_ref, valid_ref
        )
        static_candidates.append(
            _static_candidate(
                family,
                report,
                train_ref,
                valid_ref,
                output,
                lineages[family],
            )
        )

    residuals = []
    for family in passing_families:
        spec = specs[family]
        for budget in budgets:
            for seed in seeds:
                run_dir = output / "residual_runs" / (
                    "%s_%0.2f_%d" % (family, budget, seed)
                )
                resume_run = (
                    bool(_arg(args, "resume", False))
                    and run_dir.is_dir()
                    and any(run_dir.iterdir())
                )
                result = training_module.fit_one_residual(
                    train_cache=train_ref,
                    valid_cache=valid_ref,
                    static_spec=spec,
                    budget=budget,
                    seed=seed,
                    output_dir=run_dir,
                    device=_arg(args, "device", "cpu"),
                    maximum_epochs=int(_arg(args, "maximum_epochs", 60)),
                    patience=int(_arg(args, "patience", 8)),
                    batch_size=int(_arg(args, "batch_size", 256)),
                    learning_rate=float(_arg(args, "learning_rate", 1e-4)),
                    weight_decay=float(_arg(args, "weight_decay", 1e-3)),
                    gradient_clip=float(_arg(args, "gradient_clip", 1.0)),
                    resume=resume_run,
                )
                residuals.append(_residual_candidate(
                    result,
                    family,
                    budget,
                    seed,
                    spec,
                    train_ref,
                    valid_ref,
                    output,
                    lineages[family],
                ))

    ensembles = build_residual_ensembles(
        residuals,
        labels=valid_ref.cache.labels,
        validation_keys=valid_ref.cache.keys,
        output_root=output,
        expected_families=passing_families,
        expected_budgets=budgets,
        expected_seeds=seeds,
        rank_calibrations={
            family: (specs[family].rank_slope, specs[family].rank_intercept)
            for family in passing_families
        },
    )
    experts = _expert_candidates(valid_ref, output)
    inventory = build_candidate_inventory(
        experts,
        static_candidates,
        residuals,
        ensembles,
        expected_families=passing_families,
        expected_budgets=budgets,
        expected_seeds=seeds,
    )
    selected = select_validation_candidate(
        experts, static_candidates, residuals, ensembles
    )
    selected_references = {
        "name": selected.name,
        "prediction_path": selected.prediction_path,
        "prediction_sha256": selected.prediction_sha256,
        "checkpoint_path": selected.checkpoint_path,
        "checkpoint_sha256": selected.checkpoint_sha256,
    }
    selection = {
        "status": "complete",
        "selected_name": selected.name,
        "selected_type": selected.type,
        "selected_family": selected.family,
        "selected_validation_metrics": dict(
            selected.validation_metrics
        ),
        "candidate_names": [item.name for item in inventory],
        "candidate_count": len(inventory),
        "inventory_mode": inventory_mode,
        "inventory_controls": {
            "mode": inventory_mode,
            "budgets": list(budgets),
            "seeds": list(seeds),
            "limit": _arg(args, "limit", None),
        },
        "future_test_context": {
            "target_ci": 0.706858,
            "comparison": (
                "A future evaluation compares CI strictly greater "
                "than target_ci."
            ),
        },
    }
    payload = {
        "selection": selection,
        "fold_map": _fold_map(valid_ref, reports),
        "raw_report": _report_payload(reports["raw"]),
        "rank_report": _report_payload(reports["rank"]),
        "static_specs": {
            family: spec.to_dict() for family, spec in specs.items()
        },
        "candidate_manifest": [item.to_dict() for item in inventory],
        "selected_references": selected_references,
        "cache_identities": {
            "train": _plain(train_ref.identity),
            "valid": _plain(valid_ref.identity),
        },
    }
    package = output / "validation_package"
    seal_validation_package(
        package, payload, artifact_root=output
    )
    return dict(selection, package=str(package))


def run(args):
    return run_validation(args)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = run_validation(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
