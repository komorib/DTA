"""Export frozen V22 full predictions on LP train only for the V31 teacher.

This is a GPU inference utility, not a model-selection command.  It has no
validation/test argument and only exposes the LP rows of the supplied train
manifest.  The archive is used to make the outer pair-latent teacher predict
the residual to the same sealed V22 backbone that V31 deploys.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from dta.optional_structure_staged_v22.data import (
    DATASET_INDEX,
    OptionalStructureCollator,
    OptionalStructureDataset,
    manifest_fingerprint,
    move_batch_to_device,
    sha256_file,
)
from dta.optional_structure_staged_v22.model import OptionalStructureStagedDTA
from dta.optional_structure_staged_v22.training import _amp_dtype, _normalizers


VERSION = "optional-structure-privileged-msa-pair-v22-train-export-v31"


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


def run(args):
    train_manifest = Path(args.train_manifest).expanduser().resolve()
    protein_cache = Path(args.protein_cache).expanduser().resolve()
    ligand_cache = Path(args.ligand_cache).expanduser().resolve()
    checkpoint_path = Path(args.sealed_v22_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("V31 V22 train export already exists")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("version") != "optional-structure-staged-v22" or checkpoint.get("phase") != "structure":
        raise ValueError("V31 needs a sealed V22 structure checkpoint")
    if checkpoint.get("train_manifest_sha256") != manifest_fingerprint(train_manifest):
        raise ValueError("V31 V22 export train manifest differs from sealed V22 lineage")
    sealed_config = checkpoint.get("config", {})
    geometry = {
        "evaluation_batch_size": int(args.evaluation_batch_size),
        "maximum_ligand_atoms": int(args.maximum_ligand_atoms),
        "maximum_pocket_tokens": int(args.maximum_pocket_tokens),
        "amp_dtype": str(args.amp_dtype),
    }
    expected_geometry = {
        name: sealed_config.get(name) for name in geometry
    }
    if any(expected_geometry[name] != value for name, value in geometry.items()):
        raise ValueError("V31 V22 train export must preserve sealed V22 inference geometry")
    model = OptionalStructureStagedDTA(**checkpoint["model_kwargs"]).to(torch.device(args.device))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source = OptionalStructureDataset(train_manifest, protein_cache, ligand_cache, with_structure=True)
    indices = [index for index, row in enumerate(source.rows) if str(row["dataset"]) == "LP_PDBBind"]
    if not indices:
        raise ValueError("V31 V22 export source has no LP_PDBBind train rows")
    subset = torch.utils.data.Subset(source, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=int(args.evaluation_batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=OptionalStructureCollator(
            int(args.maximum_ligand_atoms), int(args.maximum_pocket_tokens)
        ),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device)
    means, standard_deviations = _normalizers(checkpoint["label_statistics"], device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    sample_ids, labels, dataset_index, predictions = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if not torch.all(batch["dataset_index"] == DATASET_INDEX["LP_PDBBind"]):
                raise RuntimeError("V31 V22 export loader contains a non-LP row")
            sample_ids.extend(str(value) for value in batch["sample_ids"])
            labels.append(batch["labels"].numpy().astype(np.float64))
            dataset_index.append(batch["dataset_index"].numpy().astype(np.int64))
            moved = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                value = model(moved)
            mean = means[moved["dataset_index"]]
            standard_deviation = standard_deviations[moved["dataset_index"]]
            predictions.append(
                (value.prediction * standard_deviation + mean).float().cpu().numpy().astype(np.float64)
            )
    labels = np.concatenate(labels)
    dataset_index = np.concatenate(dataset_index)
    predictions = np.concatenate(predictions)
    if len(sample_ids) != len(set(sample_ids)) or not np.isfinite(predictions).all():
        raise RuntimeError("V31 V22 train export is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output,
        sample_ids=np.asarray(sample_ids),
        labels=labels,
        dataset_index=dataset_index,
        predictions=predictions,
    )
    result = {
        "version": VERSION,
        "status": "complete",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "count": int(labels.size),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": manifest_fingerprint(train_manifest),
        "sealed_v22_checkpoint": str(checkpoint_path),
        "sealed_v22_checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_batch_size": int(args.evaluation_batch_size),
        "inference_geometry": geometry,
        "official_valid_manifest_read": False,
        "test_manifest_read": False,
    }
    _atomic_json(metadata_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--protein-cache", type=Path, required=True)
    parser.add_argument("--ligand-cache", type=Path, required=True)
    parser.add_argument("--sealed-v22-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--maximum-ligand-atoms", type=int, default=128)
    parser.add_argument("--maximum-pocket-tokens", type=int, default=256)
    return parser


def main(argv=None):
    return 0 if run(build_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    main()
