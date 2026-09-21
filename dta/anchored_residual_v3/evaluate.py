"""Post-selection test-only evaluation for anchored residual v3."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from dta.hybrid_cached_training_v2 import _metrics
from utils.experiment_integrity import atomic_write_json


def _variant_name(value):
    value = str(value)
    if not value or any(not (character.isalnum() or character == "_") for character in value):
        raise ValueError("variant must contain only letters, digits, and underscores")
    return value


def write_test_artifact(
    output_dir,
    evaluation,
    *,
    split,
    variant,
    manifest_sha256,
    checkpoint_sha256,
):
    if split != "test":
        raise ValueError("test evaluator requires split=test")
    variant = _variant_name(variant)
    if len(str(manifest_sha256)) != 64 or len(str(checkpoint_sha256)) != 64:
        raise ValueError("test provenance requires SHA256 strings")
    keys = np.asarray(evaluation["keys"]).astype(str).reshape(-1)
    labels = np.asarray(evaluation["labels"], dtype=np.float64).reshape(-1)
    predictions = np.asarray(evaluation["predictions"], dtype=np.float64).reshape(-1)
    if not len(keys) or len(keys) != len(labels) or len(keys) != len(predictions):
        raise ValueError("test keys, labels, and predictions must align")
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("test keys must be unique")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("test labels and predictions must be finite")
    metrics = dict(evaluation.get("metrics") or _metrics(labels, predictions))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"test_predictions_{variant}.npz"
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(output_dir)
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle, keys=keys, labels=labels, predictions=predictions
            )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    metadata = {
        "status": "complete",
        "split": "test",
        "variant": variant,
        "count": len(keys),
        "manifest_sha256": str(manifest_sha256),
        "checkpoint_sha256": str(checkpoint_sha256),
        "metrics": metrics,
    }
    atomic_write_json(output_dir / f"test_metrics_{variant}.json", metadata)
    return metadata


def evaluate_teacher_lookup(lookup):
    return {
        "keys": list(lookup.keys),
        "labels": lookup.labels.copy(),
        "predictions": lookup.predictions.copy(),
        "metrics": _metrics(lookup.labels, lookup.predictions),
    }


def evaluate_selected_model(model, loader, device, *, amp_enabled=False):
    from dta.anchored_residual_v3.training import evaluate_anchored

    return evaluate_anchored(model, loader, device, amp_enabled=amp_enabled)



def build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Report cached teacher test metrics.")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv=None):
    from dta.anchored_residual_v3.teacher import load_teacher_artifact
    from utils.experiment_integrity import manifest_fingerprint, sha256_file
    args = build_arg_parser().parse_args(argv)
    fingerprint = manifest_fingerprint(args.manifest)
    if fingerprint["split"] != "test":
        raise ValueError("teacher report requires a test manifest")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    teacher_dir = Path(args.teacher).expanduser().resolve()
    lookup = load_teacher_artifact(
        teacher_dir / "predictions.npz", teacher_dir / "metadata.json",
        expected_split="test", expected_manifest_sha256=fingerprint["sha256"],
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    evaluation = evaluate_teacher_lookup(lookup)
    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(lookup):
            raise ValueError("limit is outside the teacher artifact")
        evaluation = {"keys": evaluation["keys"][:args.limit],
                      "labels": evaluation["labels"][:args.limit],
                      "predictions": evaluation["predictions"][:args.limit]}
    result = write_test_artifact(
        args.output, evaluation, split="test", variant="cached_teacher",
        manifest_sha256=fingerprint["sha256"], checkpoint_sha256=checkpoint_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
