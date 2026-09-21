#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.experiment_integrity import atomic_write_json


REPORT_FILENAME = "paired_smoke_report.json"
STAGE1_METRIC_NAMES = (
    "total_loss",
    "pose_loss",
    "consistency_loss",
    "pose_accuracy",
    "score_margin",
    "gradient_norm",
    "jitter_identity_fallbacks",
    "samples",
    "seconds",
)
MODEL_DIAGNOSTIC_METRIC_NAMES = (
    "interface_auroc",
    "pose_accuracy",
    "score_margin",
    "coarse_linear_cv_auroc",
    "coarse_nonlinear_cv_auroc",
    "heldout_linear_cv_auroc",
    "heldout_nonlinear_cv_auroc",
    "interface_minus_coarse_linear_auroc",
    "interface_minus_coarse_nonlinear_auroc",
    "interface_minus_heldout_linear_auroc",
    "interface_minus_heldout_nonlinear_auroc",
)


def _read_json(path, label):
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, value


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(output):
        raise ValueError(f"{label} must be finite")
    return output


def _integer(value, label, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        output = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if output < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return output


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA256 hex digest") from error
    return value


def _history(path):
    path = Path(path).resolve()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"metrics history is unreadable: {path}") from error
    if len(rows) != 1:
        raise ValueError("paired smoke requires exactly one metrics-history row")
    row = rows[0]
    missing = [name for name in STAGE1_METRIC_NAMES if name not in row]
    if missing:
        raise ValueError(
            "metrics history is missing: " + ", ".join(missing)
        )
    values = {"epoch": _integer(row.get("epoch"), "history epoch", 1)}
    if values["epoch"] != 1:
        raise ValueError("paired smoke metrics history must be epoch 1")
    for name in STAGE1_METRIC_NAMES:
        value = _finite(row[name], f"history {name}")
        if name in ("samples", "jitter_identity_fallbacks"):
            values[name] = _integer(value, f"history {name}")
        else:
            values[name] = value
    if values["samples"] <= 0:
        raise ValueError("paired smoke samples must be positive")
    if values["seconds"] < 0:
        raise ValueError("paired smoke elapsed seconds must be non-negative")
    return path, values


def _model_diagnostic(path, expected_bank, expected_seed):
    path, report = _read_json(path, "model diagnostic report")
    if report.get("schema_version") != 2:
        raise ValueError("model diagnostic report must use schema version 2")
    if report.get("status") != "complete":
        raise ValueError("model diagnostic report is incomplete")
    if report.get("bank") != expected_bank:
        raise ValueError("model diagnostic report bank does not match")
    if report.get("cohort") != "paired_intersection":
        raise ValueError("model diagnostic report must use paired intersection")
    if report.get("paired_intersection") is not True:
        raise ValueError("model diagnostic report must use paired intersection")
    if _integer(report.get("seed"), "model diagnostic seed") != expected_seed:
        raise ValueError("model diagnostic seed does not match")
    gate = _mapping(report.get("gate"), "model diagnostic gate")
    if gate.get("passed") is not True:
        raise ValueError("model diagnostic report gate did not pass")
    metrics = _mapping(report.get("metrics"), "model diagnostic metrics")
    values = {
        name: _finite(metrics.get(name), f"model diagnostic {name}")
        for name in MODEL_DIAGNOSTIC_METRIC_NAMES
    }
    key_count = _integer(
        report.get("key_count"), "model diagnostic key_count", 1
    )
    row_count = _integer(
        report.get("row_count"), "model diagnostic row_count", 1
    )
    if row_count != key_count * 4:
        raise ValueError(
            "model diagnostic row_count must equal four poses per key"
        )
    return {
        "path": str(path),
        "key_count": key_count,
        "row_count": row_count,
        "ordered_key_sha256": _digest(
            report.get("ordered_key_sha256"),
            "model diagnostic ordered_key_sha256",
        ),
        "metrics": values,
    }


def _optimizer_signature(provenance):
    training = _mapping(
        provenance.get("training"), "run-config provenance training"
    )
    fields = (
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "pose_temperature",
        "consistency_weight",
    )
    return {
        "name": "AdamW",
        **{
            name: _finite(training.get(name), f"optimizer {name}")
            for name in fields
        },
    }


def _hyperparameter_signature(arguments, provenance):
    model = _mapping(provenance.get("model"), "run-config provenance model")
    training = _mapping(
        provenance.get("training"), "run-config provenance training"
    )
    argument_fields = (
        "epochs",
        "num_workers",
        "ligand_gnn_depth",
        "ligand_dropout",
        "device",
        "decoy_schema_version",
        "require_paired_bank_intersection",
    )
    missing = [name for name in argument_fields if name not in arguments]
    if missing:
        raise ValueError(
            "run config arguments are missing: " + ", ".join(missing)
        )
    if int(arguments["decoy_schema_version"]) != 2:
        raise ValueError("paired smoke must use decoy schema version 2")
    if arguments["require_paired_bank_intersection"] is not True:
        raise ValueError("paired smoke must require bank intersection")
    if str(arguments["device"]) != "cuda:0":
        raise ValueError("paired smoke must use cuda:0")
    return {
        "arguments": {
            name: arguments[name] for name in argument_fields
        },
        "model": {
            "ligand_gnn_depth": _integer(
                model.get("ligand_gnn_depth"),
                "model ligand_gnn_depth",
                1,
            ),
            "ligand_dropout": _finite(
                model.get("ligand_dropout"), "model ligand_dropout"
            ),
            "residue_to_atom_mode": model.get("residue_to_atom_mode"),
        },
        "training": {
            "epochs": _integer(
                training.get("epochs"), "training epochs", 1
            ),
            "seed": _integer(training.get("seed"), "training seed"),
            "learning_rate": _finite(
                training.get("learning_rate"), "training learning_rate"
            ),
            "weight_decay": _finite(
                training.get("weight_decay"), "training weight_decay"
            ),
            "gradient_clip": _finite(
                training.get("gradient_clip"), "training gradient_clip"
            ),
            "pose_temperature": _finite(
                training.get("pose_temperature"),
                "training pose_temperature",
            ),
            "consistency_weight": _finite(
                training.get("consistency_weight"),
                "training consistency_weight",
            ),
        },
    }


def _run_artifacts(root, expected_bank):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"smoke output directory is missing: {root}")
    run_config_path, run_config = _read_json(
        root / "run_config.json", "run config"
    )
    arguments = _mapping(run_config.get("arguments"), "run config arguments")
    provenance = _mapping(
        run_config.get("provenance"), "run config provenance"
    )
    if arguments.get("decoy_bank") != expected_bank:
        raise ValueError("run config decoy bank does not match")
    seed = _integer(arguments.get("seed"), "run config seed")
    if provenance.get("decoy_schema_version") != 2:
        raise ValueError("run config provenance must use schema version 2")
    initial_state_sha256 = _digest(
        _mapping(provenance.get("model"), "run-config provenance model").get(
            "initial_state_sha256"
        ),
        "initial_state_sha256",
    )
    stage1_key_sha256 = _digest(
        provenance.get("stage1_included_ordered_key_sha256"),
        "stage1_included_ordered_key_sha256",
    )
    history_path, history = _history(root / "metrics_history.csv")
    diagnostic = _model_diagnostic(
        root / "model_diagnostic_report.json", expected_bank, seed
    )
    return {
        "root": str(root),
        "run_config": str(run_config_path),
        "metrics_history": str(history_path),
        "seed": seed,
        "initial_state_sha256": initial_state_sha256,
        "stage1_included_ordered_key_sha256": stage1_key_sha256,
        "optimizer": _optimizer_signature(provenance),
        "hyperparameters": _hyperparameter_signature(
            arguments, provenance
        ),
        "history": history,
        "model_diagnostic": diagnostic,
    }


def _require_equal(left, right, name):
    if left != right:
        raise ValueError(f"paired smoke {name} does not match")


def _delta(left, right, names):
    return {
        name: float(right[name] - left[name])
        for name in names
    }


def compare_paired_smokes(random_output_dir, matched_output_dir, output_dir):
    random_run = _run_artifacts(random_output_dir, "random_physical")
    matched_run = _run_artifacts(matched_output_dir, "matched")
    _require_equal(
        random_run["initial_state_sha256"],
        matched_run["initial_state_sha256"],
        "initial_state_sha256",
    )
    _require_equal(
        random_run["stage1_included_ordered_key_sha256"],
        matched_run["stage1_included_ordered_key_sha256"],
        "stage1_included_ordered_key_sha256",
    )
    _require_equal(random_run["seed"], matched_run["seed"], "seed")
    _require_equal(
        random_run["optimizer"], matched_run["optimizer"], "optimizer"
    )
    _require_equal(
        random_run["hyperparameters"],
        matched_run["hyperparameters"],
        "hyperparameters",
    )
    _require_equal(
        random_run["model_diagnostic"]["ordered_key_sha256"],
        matched_run["model_diagnostic"]["ordered_key_sha256"],
        "model diagnostic ordered key hash",
    )
    _require_equal(
        random_run["model_diagnostic"]["key_count"],
        matched_run["model_diagnostic"]["key_count"],
        "model diagnostic key count",
    )

    report = {
        "schema_version": 1,
        "status": "complete",
        "descriptive_only": True,
        "paired_provenance": {
            "seed": random_run["seed"],
            "initial_state_sha256": random_run["initial_state_sha256"],
            "stage1_included_ordered_key_sha256": random_run[
                "stage1_included_ordered_key_sha256"
            ],
            "optimizer": random_run["optimizer"],
            "hyperparameters": random_run["hyperparameters"],
            "diagnostic_ordered_key_sha256": random_run[
                "model_diagnostic"
            ]["ordered_key_sha256"],
            "diagnostic_key_count": random_run["model_diagnostic"][
                "key_count"
            ],
        },
        "banks": {
            "random_physical": {
                "artifacts": {
                    "run_config": random_run["run_config"],
                    "metrics_history": random_run["metrics_history"],
                    "model_diagnostic_report": random_run[
                        "model_diagnostic"
                    ]["path"],
                },
                "stage1": random_run["history"],
                "model_diagnostic": random_run["model_diagnostic"][
                    "metrics"
                ],
                "elapsed_seconds": random_run["history"]["seconds"],
            },
            "matched": {
                "artifacts": {
                    "run_config": matched_run["run_config"],
                    "metrics_history": matched_run["metrics_history"],
                    "model_diagnostic_report": matched_run[
                        "model_diagnostic"
                    ]["path"],
                },
                "stage1": matched_run["history"],
                "model_diagnostic": matched_run["model_diagnostic"][
                    "metrics"
                ],
                "elapsed_seconds": matched_run["history"]["seconds"],
            },
        },
        "delta": {
            "stage1": _delta(
                random_run["history"],
                matched_run["history"],
                STAGE1_METRIC_NAMES,
            ),
            "model_diagnostic": _delta(
                random_run["model_diagnostic"]["metrics"],
                matched_run["model_diagnostic"]["metrics"],
                MODEL_DIAGNOSTIC_METRIC_NAMES,
            ),
            "elapsed_seconds": float(
                matched_run["history"]["seconds"]
                - random_run["history"]["seconds"]
            ),
        },
    }
    output_dir = Path(output_dir).resolve()
    report_path = output_dir / REPORT_FILENAME
    atomic_write_json(report_path, report)
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired one-epoch v2 random-physical and matched "
            "Stage-1 smoke artifacts."
        )
    )
    parser.add_argument("--random-output-dir", required=True)
    parser.add_argument("--matched-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    report = compare_paired_smokes(
        args.random_output_dir, args.matched_output_dir, args.output_dir
    )
    print(
        json.dumps(
            {
                "report": str(
                    Path(args.output_dir).resolve() / REPORT_FILENAME
                ),
                "initial_state_sha256": report["paired_provenance"][
                    "initial_state_sha256"
                ],
                "stage1_included_ordered_key_sha256": (
                    report["paired_provenance"][
                        "stage1_included_ordered_key_sha256"
                    ]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
