import contextlib
from copy import deepcopy
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


STAGE2_CHECKPOINT_SCHEMA_VERSION = 1
CI_SELECTION_TOLERANCE = 1e-6
METRIC_NAMES = ("RMSE", "MAE", "Pearson", "Spearman", "CI")
HISTORY_FIELDS = (
    "epoch",
    "phase",
    "train_loss",
    "local_lr",
    "global_lr",
    "affinity_lr",
    "valid_RMSE",
    "valid_MAE",
    "valid_Pearson",
    "valid_Spearman",
    "valid_CI",
    "best_epoch",
    "best_valid_CI",
    "improved",
)

STAGE2_ARTIFACT_NAMES = (
    "Model_dta_best.pth",
    "Model_dta_last.pth",
    "training_checkpoint.pth",
    "metrics_history.csv",
    "valid_predictions.npz",
    "test_predictions.npz",
    "best_metrics.json",
    "run_config.json",
    "train.log",
)


def _regression_metrics(labels, predictions):
    from utils.metrics import get_score_metrics

    rmse, mae, pearson, spearman, ci = get_score_metrics(
        predictions, labels
    )
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Pearson": float(pearson),
        "Spearman": float(spearman),
        "CI": float(ci),
    }


def evaluate_interface_first(model, data_loader, device):
    from data_utils.local_interface_dataset import recursive_to_device

    model.eval()
    keys = []
    labels = []
    predictions = []
    with torch.no_grad():
        for cpu_item in data_loader:
            key = str(cpu_item["key"])
            item = recursive_to_device(cpu_item, device)
            prediction = model(item)[0].reshape(-1)
            label = item["label"].float().reshape(-1)
            if prediction.numel() != 1 or label.numel() != 1:
                raise ValueError(
                    "interface-first evaluation requires one complex per batch"
                )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(
                    f"nonfinite interface-first prediction for {key}"
                )
            if not torch.isfinite(label).all():
                raise FloatingPointError(
                    f"nonfinite interface-first label for {key}"
                )
            keys.append(key)
            labels.append(float(label.item()))
            predictions.append(float(prediction.item()))
    if not keys:
        raise RuntimeError("interface-first evaluation loader is empty")
    label_array = np.asarray(labels, dtype=np.float32)
    prediction_array = np.asarray(predictions, dtype=np.float32)
    return {
        "metrics": _regression_metrics(label_array, prediction_array),
        "keys": np.asarray(keys, dtype=str),
        "labels": label_array,
        "predictions": prediction_array,
    }


def save_keyed_predictions(path, output):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = np.asarray(output["keys"], dtype=str).reshape(-1)
    labels = np.asarray(output["labels"], dtype=np.float32).reshape(-1)
    predictions = np.asarray(
        output["predictions"], dtype=np.float32
    ).reshape(-1)
    if not (len(keys) == len(labels) == len(predictions)):
        raise ValueError("prediction keys, labels, and values must align")
    temporary = path + ".tmp.npz"
    try:
        np.savez_compressed(
            temporary,
            keys=keys,
            labels=labels,
            predictions=predictions,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metrics_with_epoch(metrics, epoch):
    output = {name: float(metrics[name]) for name in METRIC_NAMES}
    output["Epoch"] = int(epoch)
    return output


def _is_better_ci(current, best, tolerance=CI_SELECTION_TOLERANCE):
    if best is None:
        return True
    tolerance = float(tolerance)
    current_ci = float(current["CI"])
    best_ci = float(best["CI"])
    if not np.isfinite(current_ci):
        return False
    if not np.isfinite(best_ci):
        return True
    if current_ci > best_ci + tolerance:
        return True
    if abs(current_ci - best_ci) > tolerance:
        return False
    for name in ("RMSE", "MAE"):
        current_value = float(current.get(name, float("inf")))
        best_value = float(best.get(name, float("inf")))
        if not np.isfinite(current_value):
            current_value = float("inf")
        if not np.isfinite(best_value):
            best_value = float("inf")
        if current_value < best_value:
            return True
        if current_value > best_value:
            return False
    return int(current["Epoch"]) < int(best["Epoch"])


def _atomic_torch_save(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_history(path, history):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for row in history:
                writer.writerow({name: row.get(name, "") for name in HISTORY_FIELDS})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _optimizer_learning_rates(optimizer):
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _train_interface_epoch(
    model, data_loader, optimizer, device, grad_clip
):
    from data_utils.local_interface_dataset import recursive_to_device

    model.train()
    total_loss = 0.0
    sample_count = 0
    for cpu_item in data_loader:
        item = recursive_to_device(cpu_item, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(item)[0].reshape(-1)
        label = item["label"].float().reshape(-1)
        if prediction.numel() != 1 or label.numel() != 1:
            raise ValueError(
                "interface-first training requires one complex per batch"
            )
        loss = F.smooth_l1_loss(prediction, label)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"nonfinite interface-first loss for {cpu_item['key']}"
            )
        loss.backward()
        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(grad_clip)
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"nonfinite interface-first gradient for {cpu_item['key']}"
            )
        optimizer.step()
        total_loss += float(loss.detach().item())
        sample_count += 1
    if sample_count == 0:
        raise RuntimeError("interface-first training loader is empty")
    return total_loss / sample_count


def _capture_random_state(train_loader):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "train_loader_generator": None,
    }
    generator = getattr(train_loader, "generator", None)
    if generator is not None:
        state["train_loader_generator"] = generator.get_state()
    return state


def _restore_random_state(state, train_loader):
    if not isinstance(state, dict):
        raise RuntimeError("Stage-2 checkpoint has no random state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Stage-2 CUDA random state cannot be restored without CUDA"
            )
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_state = state.get("train_loader_generator")
    generator = getattr(train_loader, "generator", None)
    if loader_state is not None:
        if generator is None:
            raise RuntimeError(
                "Stage-2 train loader generator cannot be restored"
            )
        generator.set_state(loader_state)


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    best_metrics,
    best_epoch,
    history,
    provenance,
    model_config,
    train_loader,
    best_model_state,
):
    return {
        "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "next_epoch": int(epoch),
        "model_state_dict": deepcopy(model.state_dict()),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": deepcopy(scheduler.state_dict()),
        "best_valid_metrics": deepcopy(best_metrics),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "history": deepcopy(history),
        "provenance": deepcopy(provenance),
        "model_config": deepcopy(model_config),
        "random_state": _capture_random_state(train_loader),
        "best_model_state_dict": deepcopy(best_model_state),
    }


def _load_resume_checkpoint(
    path, model, optimizer, scheduler, provenance, train_loader
):
    payload = torch.load(os.fspath(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("Stage-2 checkpoint must be a dictionary")
    if payload.get("schema_version") != STAGE2_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported Stage-2 checkpoint schema")
    validate_resume_provenance(payload, provenance)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_random_state(
        payload.get("random_state"), train_loader
    )
    return payload


def fit_interface_first(
    model,
    train_loader,
    valid_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    output_dir,
    epochs,
    warmup_epochs,
    grad_clip,
    provenance,
    evaluate_fn=evaluate_interface_first,
    resume_checkpoint=None,
    model_config=None,
):
    epochs = int(epochs)
    warmup_epochs = int(warmup_epochs)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if float(grad_clip) <= 0:
        raise ValueError("grad_clip must be positive")
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    model_config = {} if model_config is None else dict(model_config)
    best_path = os.path.join(output_dir, "Model_dta_best.pth")
    last_path = os.path.join(output_dir, "Model_dta_last.pth")
    checkpoint_path = os.path.join(output_dir, "training_checkpoint.pth")
    history_path = os.path.join(output_dir, "metrics_history.csv")
    valid_prediction_path = os.path.join(
        output_dir, "valid_predictions.npz"
    )
    test_prediction_path = os.path.join(
        output_dir, "test_predictions.npz"
    )

    start_epoch = 0
    history = []
    best_metrics = None
    best_epoch = None
    best_model_state = None
    if resume_checkpoint:
        payload = _load_resume_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            provenance,
            train_loader,
        )
        start_epoch = int(payload["next_epoch"])
        history = list(payload["history"])
        best_metrics = payload.get("best_valid_metrics")
        best_epoch = payload.get("best_epoch")
        best_model_state = payload.get("best_model_state_dict")
        if len(history) != start_epoch:
            raise RuntimeError(
                "Stage-2 metrics history does not match resume epoch"
            )
        if best_metrics is not None and best_model_state is None:
            raise RuntimeError(
                "Stage-2 resume checkpoint has no best model state"
            )
        if best_model_state is not None and not os.path.exists(best_path):
            _atomic_torch_save(
                best_path,
                {
                    "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
                    "model_state_dict": deepcopy(best_model_state),
                    "model_config": deepcopy(model_config),
                    "valid_metrics": deepcopy(best_metrics),
                    "provenance": deepcopy(provenance),
                },
            )

    for epoch_index in range(start_epoch, epochs):
        if epoch_index < warmup_epochs:
            model.configure_stage2_warmup()
            phase = "warmup"
        else:
            model.configure_stage2_full()
            phase = "full"
        train_loss = _train_interface_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip,
        )
        valid_output = evaluate_fn(model, valid_loader, device)
        current_metrics = _metrics_with_epoch(
            valid_output["metrics"], epoch_index + 1
        )
        improved = _is_better_ci(
            current_metrics,
            best_metrics,
            tolerance=CI_SELECTION_TOLERANCE,
        )
        if improved:
            best_metrics = current_metrics
            best_epoch = epoch_index + 1
            best_model_state = deepcopy(model.state_dict())
            _atomic_torch_save(
                best_path,
                {
                    "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
                    "model_state_dict": deepcopy(best_model_state),
                    "model_config": deepcopy(model_config),
                    "valid_metrics": deepcopy(best_metrics),
                    "provenance": deepcopy(provenance),
                },
            )
            save_keyed_predictions(
                valid_prediction_path, valid_output
            )
        learning_rates = _optimizer_learning_rates(optimizer)
        history.append(
            {
                "epoch": epoch_index + 1,
                "phase": phase,
                "train_loss": float(train_loss),
                "local_lr": learning_rates.get("local", ""),
                "global_lr": learning_rates.get("global", ""),
                "affinity_lr": learning_rates.get("affinity", ""),
                **{
                    f"valid_{name}": float(current_metrics[name])
                    for name in METRIC_NAMES
                },
                "best_epoch": int(best_epoch),
                "best_valid_CI": float(best_metrics["CI"]),
                "improved": int(improved),
            }
        )
        _atomic_write_history(history_path, history)
        scheduler.step()
        _atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch=epoch_index + 1,
                best_metrics=best_metrics,
                best_epoch=best_epoch,
                history=history,
                provenance=provenance,
                model_config=model_config,
                train_loader=train_loader,
                best_model_state=best_model_state,
            ),
        )
        print(
            json.dumps(
                {
                    "epoch": epoch_index + 1,
                    "phase": phase,
                    "train_loss": train_loss,
                    "valid": current_metrics,
                    "best_epoch": best_epoch,
                    "best_valid_CI": best_metrics["CI"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if best_epoch is None or not os.path.exists(best_path):
        raise RuntimeError("Stage-2 training did not produce a best checkpoint")
    _atomic_torch_save(
        last_path,
        {
            "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": deepcopy(model.state_dict()),
            "model_config": deepcopy(model_config),
            "provenance": deepcopy(provenance),
        },
    )
    best_payload = torch.load(best_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    test_output = evaluate_fn(model, test_loader, device)
    save_keyed_predictions(test_prediction_path, test_output)
    from utils.experiment_integrity import atomic_write_json

    result = {
        "selection_metric": "CI",
        "selection_tolerance": CI_SELECTION_TOLERANCE,
        "best_epoch": int(best_epoch),
        "valid": {
            name: float(best_metrics[name]) for name in METRIC_NAMES
        },
        "test": {
            name: float(test_output["metrics"][name])
            for name in METRIC_NAMES
        },
        "checkpoint_path": best_path,
        "last_model_path": last_path,
        "prediction_paths": {
            "valid": valid_prediction_path,
            "test": test_prediction_path,
        },
        "provenance": deepcopy(provenance),
    }
    atomic_write_json(
        os.path.join(output_dir, "best_metrics.json"), result
    )
    return result


def _set_random_seeds(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _cache_provenance(path, manifest_path, split, shell_budget):
    from utils.experiment_integrity import manifest_fingerprint, sha256_file

    path = os.path.abspath(os.fspath(path))
    metadata_path = os.path.join(path, "metadata.json")
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"native cache is not complete: {path}")
    expected_manifest = manifest_fingerprint(manifest_path)
    if metadata.get("manifest") != expected_manifest:
        raise RuntimeError(
            f"native cache manifest provenance does not match {split}"
        )
    config = metadata.get("provenance", {}).get("config", {})
    if int(config.get("shell_budget", -1)) != int(shell_budget):
        raise RuntimeError(
            f"native cache shell budget does not match {split}"
        )
    return {
        "path": path,
        "metadata_path": metadata_path,
        "metadata_sha256": sha256_file(metadata_path),
        "manifest": expected_manifest,
    }


def _stage1_marker_artifact_path(marker, name):
    value = marker.get(name)
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        return None
    return os.path.abspath(value)


def _validate_stage1_gate(args):
    from utils.experiment_integrity import sha256_file

    checkpoint = os.path.abspath(args.interface_stage1_checkpoint)
    report_path = os.path.abspath(args.interface_diagnostic_report)
    marker_path = os.path.abspath(args.interface_stage2_ready)
    report = _read_json(report_path)
    if not bool(report.get("gate", {}).get("passed")):
        reasons = report.get("gate", {}).get("reasons", [])
        raise RuntimeError(
            "Stage-1 diagnostic gate did not pass: "
            + ", ".join(str(reason) for reason in reasons)
        )
    marker = _read_json(marker_path)
    if (
        _stage1_marker_artifact_path(marker, "stage1_checkpoint")
        != checkpoint
    ):
        raise RuntimeError("Stage-2-ready marker checkpoint does not match")
    marker_report = _stage1_marker_artifact_path(
        marker, "diagnostic_report"
    )
    if marker_report is None:
        marker_report = _stage1_marker_artifact_path(
            marker, "model_diagnostic_report"
        )
    if marker_report != report_path:
        raise RuntimeError(
            "Stage-2-ready marker diagnostic report does not match"
        )
    return {
        "checkpoint": {
            "path": checkpoint,
            "sha256": sha256_file(checkpoint),
        },
        "diagnostic_report": {
            "path": report_path,
            "sha256": sha256_file(report_path),
        },
        "stage2_ready": {
            "path": marker_path,
            "sha256": sha256_file(marker_path),
        },
    }


def build_interface_initialization(args):
    if args.interface_init_mode == "scratch":
        return {"mode": "scratch"}
    return {
        "mode": "stage1",
        "artifacts": _validate_stage1_gate(args),
    }


def validate_resume_provenance(payload, provenance):
    if payload.get("provenance") != provenance:
        raise RuntimeError("Stage-2 resume provenance does not match")


def interface_model_config_from_args(args):
    return {
        "ligand_gnn_depth": int(args.interface_ligand_gnn_depth),
        "ligand_dropout": float(args.interface_ligand_dropout),
        "affinity_dropout": float(args.interface_affinity_dropout),
        "residue_to_atom_mode": args.interface_residue_to_atom_mode,
    }


def _make_loader(dataset, num_workers, shuffle, seed):
    from data_utils.local_interface_dataset import collate_single_interface

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        collate_fn=collate_single_interface,
        persistent_workers=False,
        pin_memory=False,
    )


def train_interface_first(args):
    from data_utils.local_interface_dataset import (
        NativeInterfaceDataset,
        validate_native_interface_inventory,
    )
    from net.interface_first import InterfaceFirstGerNADTA
    from utils.experiment_integrity import atomic_write_json

    _set_random_seeds(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "interface-first Stage-2 training requires a visible CUDA GPU"
        )
    device = torch.device(f"cuda:{args.cuda}")
    paths = {
        "train": args.train_path,
        "valid": args.valid_path,
        "test": args.test_path,
    }
    cache_paths = {
        "train": args.interface_train_native_lmdb,
        "valid": args.interface_valid_native_lmdb,
        "test": args.interface_test_native_lmdb,
    }
    model_config = interface_model_config_from_args(args)
    initialization = build_interface_initialization(args)
    datasets = {}
    try:
        for split in ("train", "valid", "test"):
            datasets[split] = NativeInterfaceDataset(
                paths[split], cache_paths[split]
            )
        native_cache_preflight = {
            split: validate_native_interface_inventory(
                paths[split], cache_paths[split]
            )
            for split in ("train", "valid", "test")
        }
        provenance = {
            "architecture": "interface_first",
            "initialization": initialization,
            "native_caches": {
                split: _cache_provenance(
                    cache_paths[split],
                    paths[split],
                    split,
                    args.interface_shell_budget,
                )
                for split in ("train", "valid", "test")
            },
            "native_cache_preflight": native_cache_preflight,
            "seed": int(args.seed),
            "fp32": True,
            "loss": "smooth_l1",
            "selection_metric": "CI",
            "selection_tolerance": CI_SELECTION_TOLERANCE,
            "test_selection_evaluations": 0,
            "model": model_config,
            "training": {
                "warmup_epochs": int(args.interface_warmup_epochs),
                "local_lr": float(args.interface_local_lr),
                "global_lr": float(args.interface_global_lr),
                "affinity_lr": float(args.interface_affinity_lr),
                "weight_decay": float(args.weight_decay),
                "lr_decay_step": int(args.lr_decay_step),
                "lr_decay_gamma": float(args.lr_decay_gamma),
                "grad_clip": float(args.grad_clip),
                "batch_size": 1,
            },
        }
        model = InterfaceFirstGerNADTA(**model_config).to(device=device)
        if initialization["mode"] == "stage1":
            from dta.pretrain_local_interface import load_stage1_checkpoint

            stage1_payload = load_stage1_checkpoint(
                args.interface_stage1_checkpoint, model, strict=True
            )
            stage1_model = stage1_payload.get(
                "provenance", {}
            ).get("model", {})
            expected_stage1_model = {
                "ligand_gnn_depth": model_config["ligand_gnn_depth"],
                "ligand_dropout": model_config["ligand_dropout"],
                "residue_to_atom_mode": model_config[
                    "residue_to_atom_mode"
                ],
            }
            if stage1_model != expected_stage1_model:
                raise RuntimeError(
                    "Stage-1 model configuration does not match Stage-2: "
                    f"{stage1_model} != {expected_stage1_model}"
                )
        model.configure_stage2_full()
        optimizer = torch.optim.AdamW(
            model.optimizer_parameter_groups(
                local_lr=args.interface_local_lr,
                global_lr=args.interface_global_lr,
                affinity_lr=args.interface_affinity_lr,
            ),
            weight_decay=float(args.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.lr_decay_step),
            gamma=float(args.lr_decay_gamma),
        )
        loaders = {
            "train": _make_loader(
                datasets["train"], args.num_workers, True, args.seed
            ),
            "valid": _make_loader(
                datasets["valid"], args.num_workers, False, args.seed
            ),
            "test": _make_loader(
                datasets["test"], args.num_workers, False, args.seed
            ),
        }
        output_dir = os.path.abspath(args.model_output_path)
        os.makedirs(output_dir, exist_ok=True)
        atomic_write_json(
            os.path.join(output_dir, "run_config.json"),
            {
                "arguments": vars(args),
                "model_config": model_config,
                "provenance": provenance,
                "device": str(device),
            },
        )
        return fit_interface_first(
            model=model,
            train_loader=loaders["train"],
            valid_loader=loaders["valid"],
            test_loader=loaders["test"],
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            epochs=args.epoch,
            warmup_epochs=args.interface_warmup_epochs,
            grad_clip=args.grad_clip,
            provenance=provenance,
            resume_checkpoint=args.resume_training_checkpoint,
            model_config=model_config,
        )
    finally:
        for dataset in datasets.values():
            dataset.close()


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def prepare_stage2_output_dir(
    output_dir, overwrite=False, resume_checkpoint=None
):
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if resume_checkpoint:
        if not os.path.isfile(os.fspath(resume_checkpoint)):
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {resume_checkpoint}"
            )
        return output_dir
    existing = [
        os.path.join(output_dir, name)
        for name in STAGE2_ARTIFACT_NAMES
        if os.path.exists(os.path.join(output_dir, name))
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Stage-2 output artifacts already exist; pass "
            "--overwrite_output_path to replace them: "
            + ", ".join(existing)
        )
    if overwrite:
        for path in existing:
            os.unlink(path)
    return output_dir


def train_interface_first_with_logging(args):
    output_dir = prepare_stage2_output_dir(
        args.model_output_path,
        overwrite=args.overwrite_output_path,
        resume_checkpoint=args.resume_training_checkpoint,
    )
    log_path = os.path.join(output_dir, "train.log")
    mode = "a" if args.resume_training_checkpoint else "w"
    with open(log_path, mode, encoding="utf-8", buffering=1) as handle:
        with contextlib.redirect_stdout(
            _TeeStream(sys.stdout, handle)
        ), contextlib.redirect_stderr(_TeeStream(sys.stderr, handle)):
            print("train_log:", log_path, flush=True)
            return train_interface_first(args)


def load_interface_first_checkpoint(path, device):
    from net.interface_first import InterfaceFirstGerNADTA

    payload = torch.load(os.fspath(path), map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(
            "interface-first checkpoint must contain model_state_dict"
        )
    model = InterfaceFirstGerNADTA(**dict(payload.get("model_config", {})))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device=device)
    model.eval()
    return model, payload
