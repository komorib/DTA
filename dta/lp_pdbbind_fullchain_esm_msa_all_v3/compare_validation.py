"""Recompute v2 and v3 metrics on identical validation keys and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dta.lp_pdbbind_fullchain_esm_msa_all_v2 import VERSION as V2_VERSION
from dta.lp_pdbbind_windowed_msa_v1.full_pipeline import _read_json, _write_json_new
from dta.msa_transformer_smoke_v1.msa import sha256_file
from dta.rank_fusion_v4.metrics import affinity_metrics

from . import MODE, VERSION


def _archive(path, expected_sha):
    path = Path(path).expanduser().resolve()
    if sha256_file(path) != expected_sha:
        raise ValueError("validation prediction archive checksum differs: %s" % path)
    with np.load(str(path), allow_pickle=False) as archive:
        keys = np.asarray(archive["keys"]).astype(str)
        labels = np.asarray(archive["labels"], dtype=np.float64)
        predictions = np.asarray(archive["predictions"], dtype=np.float64)
    if (keys.ndim != 1 or not len(keys) or len(set(keys.tolist())) != len(keys)
            or labels.shape != keys.shape or predictions.shape != keys.shape
            or not np.isfinite(labels).all() or not np.isfinite(predictions).all()):
        raise ValueError("validation prediction archive has invalid arrays")
    return keys, labels, predictions


def compare(summary_path):
    summary_path = Path(summary_path).expanduser().resolve()
    v3 = _read_json(summary_path)
    if (v3.get("version") != VERSION
            or v3.get("status") != "single_model_training_complete"
            or v3.get("mode") != MODE
            or v3.get("test_manifest_read") is not False):
        raise ValueError("v3 summary is not a completed validation-only single-model run")
    reference = v3["v2_reference"]
    v2_path = Path(reference["summary"]).expanduser().resolve()
    if sha256_file(v2_path) != reference["summary_sha256"]:
        raise ValueError("v2 reference summary changed")
    v2 = _read_json(v2_path)
    if (v2.get("version") != V2_VERSION
            or v2.get("status") != "paired_training_complete"
            or v2.get("test_manifest_read") is not False):
        raise ValueError("v2 reference is not the completed validation-only run")
    for name in (
        "source_manifest_sha256", "encoder_weights_sha256",
        "train_manifest_sha256", "valid_manifest_sha256",
        "initial_weight_fingerprint",
    ):
        if v3.get(name) != v2.get(name):
            raise ValueError("v2/v3 provenance differs: %s" % name)
    old = v2["arms"][MODE]
    new = v3[MODE]
    if (old.get("mode") != MODE or new.get("mode") != MODE
            or old.get("test_manifest_read") is not False
            or new.get("test_manifest_read") is not False):
        raise ValueError("validation arms or test boundary differ")
    old_keys, old_labels, old_predictions = _archive(
        old["valid_predictions"], old["valid_predictions_sha256"]
    )
    new_keys, new_labels, new_predictions = _archive(
        new["valid_predictions"], new["valid_predictions_sha256"]
    )
    if old_keys.size != new_keys.size or set(old_keys.tolist()) != set(new_keys.tolist()):
        raise ValueError("v2/v3 validation keys differ")
    positions = {key: index for index, key in enumerate(new_keys.tolist())}
    order = np.asarray([positions[key] for key in old_keys.tolist()], dtype=np.int64)
    if not np.allclose(new_labels[order], old_labels, rtol=0.0, atol=1e-6):
        raise ValueError("v2/v3 validation labels differ")
    old_metrics = affinity_metrics(old_labels, old_predictions)
    new_metrics = affinity_metrics(old_labels, new_predictions[order])
    for archived, recomputed, label in (
        (old["validation_metrics"], old_metrics, "v2"),
        (new["validation_metrics"], new_metrics, "v3"),
    ):
        if any(abs(float(archived[name]) - float(recomputed[name])) > 1e-6
               for name in recomputed):
            raise ValueError("%s archived metrics differ from predictions" % label)
    return {
        "status": "same_validation_keys_verified",
        "count": int(old_keys.size),
        "v2_summary": str(v2_path),
        "v2_summary_sha256": reference["summary_sha256"],
        "v3_summary": str(summary_path),
        "v3_summary_sha256": sha256_file(summary_path),
        "v2_metrics": old_metrics,
        "v3_metrics": new_metrics,
        "v3_minus_v2": {name: float(new_metrics[name] - old_metrics[name])
                        for name in old_metrics},
        "v3_better_than_v2": {
            name: bool(new_metrics[name] < old_metrics[name])
            if name in ("RMSE", "MAE") else bool(new_metrics[name] > old_metrics[name])
            for name in old_metrics
        },
        "v3_v20_gate_passed": bool(new["validation_gate"]["passed"]),
        "test_manifest_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = compare(args.v3_summary)
    if args.output:
        _write_json_new(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
