"""Verify the V22 structure validation seal before reading test exactly once."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .data import OptionalStructureDataset, manifest_fingerprint, sha256_file
from .model import OptionalStructureStagedDTA
from .training import TrainingConfig, VERSION, cache_lineage, predict, save_predictions


V20_REFERENCE = {
    "CI": 0.736092,
    "RMSE": 1.297841,
    "MAE": 1.002948,
    "Pearson": 0.674478,
    "Spearman": 0.650757,
}


def _verify_selection(selection_output):
    selection_path = selection_output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != VERSION:
        raise ValueError("V22 selection version mismatch")
    if selection.get("phase") != "structure_validation_selection":
        raise ValueError("V22 structure selection phase mismatch")
    if selection.get("status") != "validation_passed" or not selection.get("validation_selected"):
        raise ValueError("V22 structure validation gate did not pass")
    if sha256_file(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise ValueError("sealed V22 checkpoint checksum mismatch")
    if sha256_file(selection["valid_predictions"]) != selection["valid_predictions_sha256"]:
        raise ValueError("sealed V22 validation predictions checksum mismatch")
    if sha256_file(selection["sequence_selection"]) != selection["sequence_selection_sha256"]:
        raise ValueError("V22 sequence selection checksum mismatch")
    sequence_selection = json.loads(Path(selection["sequence_selection"]).read_text(encoding="utf-8"))
    if sequence_selection["checkpoint_sha256"] != selection["sequence_checkpoint_sha256"]:
        raise ValueError("V22 sequence checkpoint lineage mismatch")
    if sha256_file(sequence_selection["checkpoint"]) != selection["sequence_checkpoint_sha256"]:
        raise ValueError("V22 sequence checkpoint checksum mismatch")
    if manifest_fingerprint(selection["train_manifest"]) != selection["train_manifest_sha256"]:
        raise ValueError("training manifest changed after V22 validation selection")
    if manifest_fingerprint(selection["valid_manifest"]) != selection["valid_manifest_sha256"]:
        raise ValueError("validation manifest changed after V22 validation selection")
    return selection_path, selection


def run(args):
    selection_output = args.selection_output.expanduser().resolve()
    selection_path, selection = _verify_selection(selection_output)
    result_path = selection_output / "result.json"
    prediction_path = selection_output / "test_predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite existing V22 test report")
    checkpoint = torch.load(str(selection["checkpoint"]), map_location="cpu")
    if checkpoint.get("version") != VERSION or checkpoint.get("phase") != "structure":
        raise ValueError("sealed V22 checkpoint phase mismatch")
    if checkpoint.get("train_manifest_sha256") != selection["train_manifest_sha256"]:
        raise ValueError("sealed V22 checkpoint training manifest lineage mismatch")
    if checkpoint.get("valid_manifest_sha256") != selection["valid_manifest_sha256"]:
        raise ValueError("sealed V22 checkpoint validation manifest lineage mismatch")
    if checkpoint.get("protein_cache") != cache_lineage(selection["protein_cache"]):
        raise ValueError("sealed V22 protein cache lineage mismatch")
    if checkpoint.get("ligand_cache") != cache_lineage(selection["ligand_cache"]):
        raise ValueError("sealed V22 ligand cache lineage mismatch")
    sequence_source = checkpoint.get("sources", {}).get("sequence_checkpoint", {})
    if sequence_source.get("sha256") != selection["sequence_checkpoint_sha256"]:
        raise ValueError("sealed V22 structure checkpoint sequence lineage mismatch")
    config_values = dict(checkpoint["config"])
    config_values["device"] = args.device or config_values["device"]
    if args.num_workers is not None:
        config_values["num_workers"] = args.num_workers
    if args.evaluation_batch_size is not None:
        config_values["evaluation_batch_size"] = args.evaluation_batch_size
    config = TrainingConfig(**config_values)
    test_manifest = args.test_manifest.expanduser().resolve()
    print("V22 validation seal verified; reading test manifest now", flush=True)
    test = OptionalStructureDataset(
        test_manifest,
        selection["protein_cache"],
        selection["ligand_cache"],
        with_structure=True,
        limit=args.limit_test,
    )
    model = OptionalStructureStagedDTA(**checkpoint["model_kwargs"]).to(config.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_result = predict(model, test, config, checkpoint["label_statistics"])
    save_predictions(prediction_path, test_result)
    lp_metrics = test_result["metrics"].get("LP_PDBBind")
    v20_delta = None
    if lp_metrics is not None:
        v20_delta = {key: float(lp_metrics[key] - value) for key, value in V20_REFERENCE.items()}
    result = {
        "version": VERSION,
        "phase": "test_report",
        "status": "reported",
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selection_checkpoint_sha256": selection["checkpoint_sha256"],
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": manifest_fingerprint(test_manifest),
        "test_metrics": test_result["metrics"],
        "test_base_only_metrics": test_result["base_metrics"],
        "v20_reference": V20_REFERENCE,
        "lp_pdbbind_delta_vs_v20": v20_delta,
        "counts": {"test": len(test)},
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "test_used_for_selection": False,
        "inference_config": asdict(config),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--limit-test", type=int)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
