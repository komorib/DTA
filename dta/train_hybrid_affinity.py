import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import random
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np
import torch

from utils.experiment_integrity import (
    atomic_write_json,
    manifest_fingerprint,
    sha256_file,
)


RANK_WEIGHTS = (0.1, 0.2, 0.4)
FUSION_TYPES = ("mlp", "transformer")


def build_gerna_backbone(model_class=None):
    if model_class is None:
        from dta.model_dta import GerNADTA

        model_class = GerNADTA
    return model_class(
        [4, 2, 128, 128],
        trigonometry=True,
        mol_graph=True,
        rna_repre=True,
        rna_graph=False,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=None,
        rna_3d_input_dim=640,
        disable_protein_2d=True,
    )


def build_interface_backbone(model_class=None):
    if model_class is None:
        from net.interface_first import InterfaceFirstGerNADTA

        model_class = InterfaceFirstGerNADTA
    return model_class(
        ligand_gnn_depth=3,
        ligand_dropout=0.1,
        affinity_dropout=0.2,
        residue_to_atom_mode="gated_residual",
    )


def load_backbone_checkpoint(model, checkpoint_path):
    path = os.path.abspath(os.fspath(checkpoint_path))
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        checkpoint_format = "model_state_dict"
    else:
        state_dict = payload
        checkpoint_format = "state_dict"
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint state dict must be a mapping")
    model.load_state_dict(state_dict, strict=True)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "format": checkpoint_format,
    }


def validate_manifest_split(manifest_path, expected_split):
    if expected_split not in {"train", "valid"}:
        raise ValueError("expected split must be train or valid")
    path = Path(manifest_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    actual = manifest.get("split")
    if actual != expected_split:
        raise ValueError(
            f"manifest split is {actual!r}; expected {expected_split}: {path}"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    try:
        keys = [str(sample["lmdb_key"]) for sample in samples]
    except (KeyError, TypeError) as error:
        raise ValueError("manifest sample is missing lmdb_key") from error
    if len(keys) != len(set(keys)):
        raise ValueError(f"{expected_split} manifest contains duplicate keys")
    return manifest


def validate_train_valid_manifests(train_path, valid_path):
    train = validate_manifest_split(train_path, "train")
    valid = validate_manifest_split(valid_path, "valid")
    train_keys = {
        str(sample["lmdb_key"]) for sample in train["samples"]
    }
    valid_keys = {
        str(sample["lmdb_key"]) for sample in valid["samples"]
    }
    overlap = sorted(train_keys.intersection(valid_keys))
    if overlap:
        raise ValueError(
            "train/valid key overlap: " + ", ".join(overlap[:5])
        )
    return {"train": train, "valid": valid}


def build_hybrid_dataset(
    manifest_path,
    native_cache_path,
    expected_split,
    gerna_dataset_class=None,
    interface_dataset_class=None,
    hybrid_dataset_class=None,
):
    if gerna_dataset_class is None:
        from data_utils.lmdb_dataset import PDBbindLmdbDataset

        gerna_dataset_class = PDBbindLmdbDataset
    if interface_dataset_class is None:
        from data_utils.local_interface_dataset import NativeInterfaceDataset

        interface_dataset_class = NativeInterfaceDataset
    if hybrid_dataset_class is None:
        from data_utils.hybrid_affinity_dataset import HybridAffinityDataset

        hybrid_dataset_class = HybridAffinityDataset

    manifest_path = os.path.abspath(os.fspath(manifest_path))
    native_cache_path = os.path.abspath(os.fspath(native_cache_path))
    manifest = validate_manifest_split(
        manifest_path, expected_split
    )
    samples = manifest["samples"]

    gerna_dataset = gerna_dataset_class(
        manifest_path,
        protein_3d_mode="ligand_pocket_residue_esm",
        protein_embedding_mode="esm",
        ligand_pocket_cutoff=8.0,
        disable_protein_2d=True,
    )
    interface_dataset = interface_dataset_class(
        manifest_path, native_cache_path
    )
    return hybrid_dataset_class(
        samples, gerna_dataset, interface_dataset
    )


PREFLIGHT_BINDING_FIELDS = (
    "manifest_sha256",
    "native_metadata_sha256",
    "ordered_key_sha256",
    "sample_count",
)


def build_preflight_binding(manifest_path, native_cache_path):
    manifest = manifest_fingerprint(manifest_path)
    metadata_path = (
        Path(native_cache_path).expanduser().resolve() / "metadata.json"
    )
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete":
        raise RuntimeError(
            f"native cache is not complete: {metadata_path.parent}"
        )
    if metadata.get("manifest") != manifest:
        raise RuntimeError(
            "native cache manifest provenance does not match "
            + os.fspath(manifest_path)
        )
    return {
        "manifest_sha256": manifest["sha256"],
        "native_metadata_sha256": sha256_file(metadata_path),
        "ordered_key_sha256": manifest["ordered_key_sha256"],
        "sample_count": int(manifest["sample_count"]),
    }


def preflight_report_matches(report, binding):
    if not isinstance(report, dict) or report.get("status") != "complete":
        return False
    actual = report.get("binding")
    if not isinstance(actual, dict) or not all(
        actual.get(field) == binding.get(field)
        for field in PREFLIGHT_BINDING_FIELDS
    ):
        return False
    sample_count = binding.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or report.get("sample_count") != sample_count
        or report.get("ordered_key_sha256")
        != binding.get("ordered_key_sha256")
    ):
        return False
    expected_largest = min(4, sample_count)
    indices = report.get("largest_indices")
    keys = report.get("largest_keys")
    if (
        not isinstance(indices, list)
        or not isinstance(keys, list)
        or len(indices) != expected_largest
        or len(keys) != expected_largest
        or not all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < sample_count
            for index in indices
        )
        or len(set(indices)) != expected_largest
        or not all(isinstance(key, str) and key for key in keys)
    ):
        return False
    return True


def run_or_reuse_dataset_preflight(
    dataset,
    manifest_path,
    native_cache_path,
    report_path,
    preflight_fn=None,
):
    if preflight_fn is None:
        from data_utils.hybrid_affinity_dataset import (
            preflight_hybrid_dataset,
        )

        preflight_fn = preflight_hybrid_dataset
    report_path = Path(report_path).expanduser().resolve()
    binding = build_preflight_binding(manifest_path, native_cache_path)
    if report_path.is_file():
        try:
            with report_path.open(encoding="utf-8") as handle:
                cached = json.load(handle)
        except (OSError, TypeError, ValueError):
            cached = None
        if preflight_report_matches(cached, binding):
            reused = dict(cached)
            reused["reused"] = True
            return reused

    result = preflight_fn(dataset)
    if not isinstance(result, dict):
        raise RuntimeError("dataset preflight must return a mapping")
    if int(result.get("sample_count", -1)) != binding["sample_count"]:
        raise RuntimeError("dataset preflight sample count does not match manifest")
    if result.get("ordered_key_sha256") != binding["ordered_key_sha256"]:
        raise RuntimeError("dataset preflight ordered keys do not match manifest")
    if set(result).intersection({"status", "binding", "reused"}):
        raise RuntimeError("dataset preflight returned reserved fields")
    payload = {
        "status": "complete",
        "binding": binding,
        **result,
    }
    atomic_write_json(report_path, payload)
    returned = dict(payload)
    returned["reused"] = False
    return returned


def _run_gpu_probe_group(model, samples, device, rank_weight=0.2):
    from dta.hybrid_affinity_training import (
        ModelEMA,
        _prediction_from_output,
        hybrid_affinity_loss,
        prepare_hybrid_sample,
    )

    device = torch.device(device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=2e-6,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ema = ModelEMA(model, decay=0.999)
    model.zero_grad(set_to_none=True)
    predictions = []
    targets = []
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for sample in samples:
                gerna_inputs, interface_item, target = prepare_hybrid_sample(
                    sample, device
                )
                output = model(gerna_inputs, interface_item)
                predictions.append(
                    _prediction_from_output(output, sample["key"])
                )
                targets.append(target.reshape(1))
            loss = hybrid_affinity_loss(
                torch.cat(predictions),
                torch.cat(targets),
                rank_weight=rank_weight,
            )["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)


def _is_cuda_oom(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def run_gpu_memory_preflight(
    model,
    dataset,
    largest_indices,
    device,
    report_path,
    memory_budget_bytes=40 * 1024 ** 3,
    probe_fn=None,
    cuda_api=None,
    configure_fn=None,
):
    from dta.hybrid_affinity_training import configure_training_stage

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("GPU memory preflight requires a CUDA device")
    memory_budget_bytes = int(memory_budget_bytes)
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    indices = [int(index) for index in largest_indices]
    if len(indices) < 4 or len(set(indices[:4])) != 4:
        raise ValueError("memory preflight requires four distinct largest samples")
    if probe_fn is None:
        probe_fn = _run_gpu_probe_group
    if cuda_api is None:
        cuda_api = torch.cuda
    if configure_fn is None:
        configure_fn = configure_training_stage

    configure_fn(model, "full")
    report = {
        "status": "running",
        "device": str(device),
        "memory_budget_bytes": memory_budget_bytes,
        "amp": True,
        "grad_scaler": True,
        "optimizer": "AdamW",
        "optimizer_step": True,
        "ema_decay": 0.999,
        "activation_checkpointing": False,
        "largest_indices": indices[:4],
        "attempts": [],
    }
    for group_size in (4, 2):
        samples = [dataset[index] for index in indices[:group_size]]
        cuda_api.reset_peak_memory_stats(device)
        attempt = {"group_size": group_size}
        try:
            probe_fn(model, samples, device)
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            attempt.update(
                {
                    "status": "oom",
                    "error": str(error),
                    "max_allocated_bytes": int(
                        cuda_api.max_memory_allocated(device)
                    ),
                    "max_reserved_bytes": int(
                        cuda_api.max_memory_reserved(device)
                    ),
                }
            )
        else:
            allocated = int(cuda_api.max_memory_allocated(device))
            reserved = int(cuda_api.max_memory_reserved(device))
            attempt.update(
                {
                    "max_allocated_bytes": allocated,
                    "max_reserved_bytes": reserved,
                    "status": (
                        "complete"
                        if reserved <= memory_budget_bytes
                        else "over_budget"
                    ),
                }
            )
        finally:
            cuda_api.empty_cache()
        report["attempts"].append(attempt)
        if attempt["status"] == "complete":
            report["status"] = "complete"
            report["selected_group_size"] = group_size
            atomic_write_json(report_path, report)
            return report

    report["status"] = "failed"
    report["selected_group_size"] = None
    atomic_write_json(report_path, report)
    raise RuntimeError("GPU memory preflight failed for group sizes 4 and 2")


def run_isolated_gpu_memory_preflight(
    model,
    *,
    runner=run_gpu_memory_preflight,
    **kwargs,
):
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    was_training = bool(model.training)
    try:
        return runner(model=model, **kwargs)
    finally:
        model.load_state_dict(state, strict=True)
        model.train(was_training)


def validate_cli_args(args):
    if args.fusion_type not in FUSION_TYPES:
        raise ValueError("unsupported fusion type")
    if float(args.rank_weight) not in RANK_WEIGHTS:
        raise ValueError("unsupported rank weight")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal hybrid training requires a CUDA device")
    if int(args.num_workers) != 0:
        raise ValueError("exact resume requires num_workers=0")
    memory_budget_gb = float(args.memory_budget_gb)
    if not np.isfinite(memory_budget_gb) or memory_budget_gb <= 0.0:
        raise ValueError("memory budget must be finite and positive")
    epochs = (
        int(args.head_epochs),
        int(args.high_epochs),
        int(args.full_epochs),
    )
    if any(value < 0 for value in epochs) or sum(epochs) <= 0:
        raise ValueError("stage epochs must be nonnegative and nonempty")
    seed = int(args.seed)
    if seed < 0 or seed >= 2 ** 32:
        raise ValueError("seed must be in [0, 2**32)")

    output = Path(args.output_dir).expanduser()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output directory is nonempty; use --resume: {output}"
        )
    return args


def build_training_records(
    args,
    checkpoint_artifacts,
    preflights,
    group_size,
):
    splits = {}
    split_paths = {
        "train": (args.train_manifest, args.train_native_cache),
        "valid": (args.valid_manifest, args.valid_native_cache),
    }
    for split, (manifest_path, native_cache_path) in split_paths.items():
        binding = dict(preflights[split]["binding"])
        splits[split] = {
            "manifest_path": os.path.abspath(os.fspath(manifest_path)),
            "native_cache_path": os.path.abspath(
                os.fspath(native_cache_path)
            ),
            **binding,
        }

    run_config = {
        "schema_version": 1,
        "fusion_type": str(args.fusion_type),
        "rank_weight": float(args.rank_weight),
        "seed": int(args.seed),
        "device": str(torch.device(args.device)),
        "group_size": int(group_size),
        "num_workers": int(args.num_workers),
        "amp": True,
    }
    provenance = {
        "schema_version": 1,
        "dataset": "LP-PDBBind",
        "architecture": "hybrid_affinity_dual_encoder",
        "initial_checkpoints": {
            name: dict(checkpoint_artifacts[name])
            for name in ("gerna", "interface")
        },
        "splits": splits,
        "gerna_dataset": {
            "protein_3d_mode": "ligand_pocket_residue_esm",
            "protein_embedding_mode": "esm",
            "ligand_pocket_cutoff": 8.0,
            "disable_protein_2d": True,
        },
        "gerna_model": {
            "params": [4, 2, 128, 128],
            "rna_3d_input_dim": 640,
            "disable_protein_2d": True,
        },
        "interface_model": {
            "ligand_gnn_depth": 3,
            "ligand_dropout": 0.1,
            "affinity_dropout": 0.2,
            "residue_to_atom_mode": "gated_residual",
        },
        "fusion": {
            "type": str(args.fusion_type),
            "token_dim": 128,
            "dropout": 0.1,
        },
    }
    provenance_hashes = {
        "gerna_checkpoint_sha256": checkpoint_artifacts["gerna"]["sha256"],
        "interface_checkpoint_sha256": checkpoint_artifacts["interface"][
            "sha256"
        ],
    }
    for split in ("train", "valid"):
        for field in (
            "manifest_sha256",
            "native_metadata_sha256",
            "ordered_key_sha256",
        ):
            provenance_hashes[f"{split}_{field}"] = splits[split][field]
        provenance_hashes[f"{split}_sample_count"] = splits[split][
            "sample_count"
        ]
    return {
        "run_config": run_config,
        "provenance": provenance,
        "provenance_hashes": provenance_hashes,
    }


def validate_input_paths(args):
    for name in (
        "train_manifest",
        "valid_manifest",
        "gerna_checkpoint",
        "interface_checkpoint",
    ):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    validate_train_valid_manifests(
        args.train_manifest, args.valid_manifest
    )
    for name in ("train_native_cache", "valid_native_cache"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"{name}: {path}")


def set_random_seeds(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_initialized_hybrid_model(args):
    from net.hybrid_affinity import (
        HybridAffinityFusion,
        HybridAffinityModel,
    )

    gerna = build_gerna_backbone()
    interface = build_interface_backbone()
    artifacts = {
        "gerna": load_backbone_checkpoint(
            gerna, args.gerna_checkpoint
        ),
        "interface": load_backbone_checkpoint(
            interface, args.interface_checkpoint
        ),
    }
    fusion = HybridAffinityFusion(
        fusion_type=args.fusion_type,
        token_dim=128,
        dropout=0.1,
    )
    return HybridAffinityModel(gerna, interface, fusion), artifacts


def build_hybrid_loaders(
    train_dataset,
    valid_dataset,
    group_size,
    seed,
    num_workers=0,
):
    from torch.utils.data import DataLoader
    from data_utils.hybrid_affinity_dataset import (
        LabelSpreadBatchSampler,
        collate_hybrid_group,
    )

    if int(num_workers) != 0:
        raise ValueError("exact resume requires num_workers=0")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    batch_sampler = LabelSpreadBatchSampler(
        train_dataset.labels,
        group_size=int(group_size),
        seed=int(seed),
        min_gap=0.1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=collate_hybrid_group,
        generator=generator,
        pin_memory=False,
        persistent_workers=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_hybrid_group,
        generator=generator,
        pin_memory=False,
        persistent_workers=False,
    )
    return train_loader, valid_loader, generator


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _close_hybrid_dataset(dataset):
    for name in ("gerna_dataset", "interface_dataset"):
        owner = getattr(dataset, name, None)
        close = getattr(owner, "close", None)
        if callable(close):
            close()


def resolve_dataset_preflight_paths(args, output_dir):
    output = Path(output_dir).expanduser().resolve()
    configured = {
        "train": args.train_preflight_report,
        "valid": args.valid_preflight_report,
    }
    return {
        split: (
            Path(value).expanduser().resolve()
            if value
            else output / f"dataset_preflight_{split}.json"
        )
        for split, value in configured.items()
    }


def _run_training(args, fit_fn):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_random_seeds(args.seed)
    model, checkpoint_artifacts = build_initialized_hybrid_model(args)

    datasets = {}
    try:
        datasets["train"] = build_hybrid_dataset(
            args.train_manifest,
            args.train_native_cache,
            expected_split="train",
        )
        datasets["valid"] = build_hybrid_dataset(
            args.valid_manifest,
            args.valid_native_cache,
            expected_split="valid",
        )
        output = Path(args.output_dir).expanduser().resolve()
        preflight_paths = resolve_dataset_preflight_paths(args, output)
        preflights = {
            "train": run_or_reuse_dataset_preflight(
                datasets["train"],
                args.train_manifest,
                args.train_native_cache,
                preflight_paths["train"],
            ),
            "valid": run_or_reuse_dataset_preflight(
                datasets["valid"],
                args.valid_manifest,
                args.valid_native_cache,
                preflight_paths["valid"],
            ),
        }
        memory = run_isolated_gpu_memory_preflight(
            model,
            runner=run_gpu_memory_preflight,
            dataset=datasets["train"],
            largest_indices=preflights["train"]["largest_indices"],
            device=device,
            report_path=output / "memory_preflight.json",
            memory_budget_bytes=int(
                float(args.memory_budget_gb) * 1024 ** 3
            ),
        )
        set_random_seeds(args.seed)
        group_size = int(memory["selected_group_size"])
        train_loader, valid_loader, generator = build_hybrid_loaders(
            datasets["train"],
            datasets["valid"],
            group_size=group_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        records = build_training_records(
            args,
            checkpoint_artifacts,
            preflights,
            group_size=group_size,
        )
        return fit_fn(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            output_dir=str(output),
            device=device,
            rank_weight=float(args.rank_weight),
            run_config=records["run_config"],
            provenance=records["provenance"],
            provenance_hashes=records["provenance_hashes"],
            data_loader_generator=generator,
            stage_epochs={
                "head": int(args.head_epochs),
                "high": int(args.high_epochs),
                "full": int(args.full_epochs),
            },
            amp_enabled=True,
            max_grad_norm=1.0,
            resume=bool(args.resume),
        )
    finally:
        for dataset in datasets.values():
            _close_hybrid_dataset(dataset)


def run_training(args, fit_fn=None):
    validate_cli_args(args)
    validate_input_paths(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "formal hybrid training requires a visible CUDA GPU"
        )
    if fit_fn is None:
        from dta.hybrid_affinity_training import fit_hybrid_affinity

        fit_fn = fit_hybrid_affinity

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "launcher.log"
    mode = "a" if args.resume else "w"
    with log_path.open(mode, encoding="utf-8") as log_handle:
        stdout = _Tee(sys.stdout, log_handle)
        stderr = _Tee(sys.stderr, log_handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(
                json.dumps(
                    {
                        "event": "hybrid_training_start",
                        "device": str(args.device),
                        "fusion_type": str(args.fusion_type),
                        "rank_weight": float(args.rank_weight),
                        "seed": int(args.seed),
                        "resume": bool(args.resume),
                    },
                    sort_keys=True,
                )
            )
            result = _run_training(args, fit_fn)
            print(
                json.dumps(
                    {
                        "event": "hybrid_training_complete",
                        "result": result,
                    },
                    default=str,
                    sort_keys=True,
                )
            )
            return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train the LP-PDBBind hybrid affinity model."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--train-native-cache", required=True)
    parser.add_argument("--valid-native-cache", required=True)
    parser.add_argument("--gerna-checkpoint", required=True)
    parser.add_argument("--interface-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fusion-type", required=True, choices=FUSION_TYPES
    )
    parser.add_argument(
        "--rank-weight", required=True, type=float, choices=RANK_WEIGHTS
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--high-epochs", type=int, default=15)
    parser.add_argument("--full-epochs", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-budget-gb", type=float, default=40.0)
    parser.add_argument("--train-preflight-report")
    parser.add_argument("--valid-preflight-report")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
