"""Fixed alpha=0 objective for a FullChain v3 single-network control."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dta.lp_pdbbind_fullchain_esm_msa_all_v3 import (
    MODE, MODEL_DROPOUT, VERSION as V3_VERSION,
)
from dta.lp_pdbbind_fullchain_esm_msa_all_v3.compare_validation import _archive
from dta.lp_pdbbind_fullchain_esm_msa_all_v3 import run_training as v3_training
from dta.lp_pdbbind_windowed_msa_v1.full_pipeline import _read_json
from dta.msa_interface_pair_pooling_v18 import training as v18_training
from dta.msa_transformer_smoke_v1.msa import sha256_file
from dta.rank_fusion_v4.metrics import affinity_metrics


VERSION = "lp-pdbbind-fullchain-esm-msa-all-v3-affinity-only-v1"
LOSS_PROTOCOL = "affinity_only_v1"
AUXILIARY_WEIGHTS = {
    "contact": 0.05,
    "residue_contact": 0.03,
    "affinity_class": 0.05,
    "distogram": 0.02,
    "interface_distogram": 0.08,
    "interface_contact": 0.05,
    "match": 0.02,
}
PROTOCOL_FIELDS = {
    "loss_protocol": LOSS_PROTOCOL,
    "aux_loss_scale": 0.0,
    "affinity_loss_weights": {"smooth_l1": 1.0, "mse": 0.15, "rank": 0.40},
    "auxiliary_original_weights": AUXILIARY_WEIGHTS,
    "auxiliary_effective_weights": {name: 0.0 for name in AUXILIARY_WEIGHTS},
    "planned_optimizer_steps": 20700,
    "controlled_change_from_v3": {
        "auxiliary_loss_group": "all_seven_off",
        "model_graph_inputs_initialization_and_other_training_controls": "matched",
    },
}


def affinity_only_loss(output, normalized_labels, batch, config):
    """Use only the original affinity terms for backward; audit auxiliaries detached."""

    smooth = F.smooth_l1_loss(output.prediction, normalized_labels, beta=0.5)
    mse = F.mse_loss(output.prediction, normalized_labels)
    rank = v18_training._pairwise_rank_loss(
        output.prediction, normalized_labels, config.minimum_rank_gap,
    )
    total = smooth + config.mse_weight * mse + config.rank_weight * rank
    components = {
        "smooth_l1": float(smooth.detach()),
        "mse": float(mse.detach()),
        "rank": float(rank.detach()),
        "loss_affinity": float(total.detach()),
        "loss_total": float(total.detach()),
        "loss_auxiliary_effective": 0.0,
    }
    # Reuse this forward's tensors. no_grad creates no auxiliary backward graph,
    # does not run the model again, and does not consume random numbers.
    with torch.no_grad():
        _, observed = v18_training.msa_interface_loss(
            output, normalized_labels, batch, config,
        )
    valid = 0
    original_total = 0.0
    for name, weight in AUXILIARY_WEIGHTS.items():
        value = observed.get(name)
        value = float(value) if value is not None else float("nan")
        if math.isfinite(value):
            components[name] = value
            original_total += weight * value
            valid += 1
        else:
            components[name] = None
    components["loss_auxiliary_original"] = (
        original_total if valid == len(AUXILIARY_WEIGHTS) else None
    )
    components["aux_diagnostic_valid_components"] = valid
    return total, components


def verify_v3_reference(path, lineage, valid_data, config, baseline,
                        initial_fingerprint, v2_reference):
    """Seal the same v3 archive, configuration, and random initialization."""

    path = Path(path).expanduser().resolve()
    summary = _read_json(path)
    arm = summary.get(MODE, {})
    if (
        summary.get("version") != V3_VERSION
        or summary.get("status") != "single_model_training_complete"
        or summary.get("mode") != MODE
        or summary.get("initialization") != "random_clean_start_no_stage1_weights"
        or summary.get("test_manifest_read") is not False
        or summary.get("test_used_for_selection") is not False
        or summary.get("initial_weight_fingerprint") != initial_fingerprint
        or summary.get("matched_hyperparameters") != v3_training.MATCHED
        or summary.get("model_config") != asdict(config)
        or summary.get("v2_reference", {}).get("summary_sha256")
        != v2_reference["summary_sha256"]
        or summary.get("v20_baseline_metrics") != baseline
        or not summary.get("gpu_smoke_result", {}).get("sha256")
        or arm.get("mode") != MODE
        or arm.get("test_manifest_read") is not False
        or arm.get("test_used_for_selection") is not False
    ):
        raise ValueError("v3 reference configuration or identity differs")
    for name, expected in {**lineage, **v3_training.PROTOCOL}.items():
        if summary.get(name) != expected or arm.get(name) != expected:
            raise ValueError("v3 reference lineage differs: %s" % name)
    archive_path = Path(arm["valid_predictions"]).expanduser().resolve()
    keys, labels, predictions = _archive(
        archive_path, arm["valid_predictions_sha256"],
    )
    if (
        keys.size != len(valid_data)
        or tuple(keys.tolist()) != tuple(valid_data.keys)
        or not np.allclose(labels, valid_data.labels, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("v3 reference validation keys or labels differ")
    metrics = affinity_metrics(labels, predictions)
    if any(abs(float(metrics[name]) - float(arm["validation_metrics"][name])) > 1e-6
           for name in metrics):
        raise ValueError("v3 reference validation metrics differ")
    return {
        "summary": str(path),
        "summary_sha256": sha256_file(path),
        "valid_predictions": str(archive_path),
        "valid_predictions_sha256": arm["valid_predictions_sha256"],
        "validation_count": int(keys.size),
        "validation_metrics": metrics,
        "initial_weight_fingerprint": initial_fingerprint,
        "gpu_smoke_sha256": summary["gpu_smoke_result"]["sha256"],
    }


def run(args):
    args.dropout = MODEL_DROPOUT
    root = Path(__file__).resolve().parents[2]
    sources = (
        "dta/lp_pdbbind_fullchain_esm_msa_all_v3/affinity_only.py",
        "dta/lp_pdbbind_fullchain_esm_msa_all_v3/run_training.py",
        "dta/lp_pdbbind_fullchain_esm_msa_cross_v1/run_training.py",
        "dta/msa_interface_pair_pooling_v18/training.py",
    )
    experiment_fields = {
        **PROTOCOL_FIELDS,
        "source_file_sha256": {name: sha256_file(root / name) for name in sources},
    }
    return v3_training.run(
        args,
        loss_function=affinity_only_loss,
        experiment_version=VERSION,
        experiment_fields=experiment_fields,
        reference_validator=verify_v3_reference,
    )


def build_parser():
    parser = v3_training.build_parser()
    parser.description = __doc__
    parser.add_argument("--v3-summary", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
