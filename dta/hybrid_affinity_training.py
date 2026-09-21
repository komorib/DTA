from contextlib import contextmanager, nullcontext
import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STAGE_EPOCHS = {"head": 2, "high": 15, "full": 40}
WARMUP_FRACTION = 0.1
MAX_AMP_OVERFLOW_RETRIES = 16


STAGE_LRS = {
    "head": {"fusion": 3e-4, "interface": 0.0, "gerna": 0.0},
    "high": {"fusion": 1e-4, "interface": 3e-5, "gerna": 1e-5},
    "full": {"fusion": 5e-5, "interface": 5e-6, "gerna": 2e-6},
}


def pairwise_logistic_rank_loss(
    prediction,
    target,
    min_gap=0.1,
    temperature=0.5,
):
    prediction = prediction.reshape(-1)
    if prediction.numel() == 0:
        raise ValueError("prediction and target must be nonempty")
    compute_dtype = (
        torch.float32
        if prediction.dtype in (torch.float16, torch.bfloat16)
        else prediction.dtype
    )
    prediction = prediction.to(dtype=compute_dtype)
    target = target.reshape(-1).to(
        device=prediction.device, dtype=compute_dtype
    )
    if prediction.numel() != target.numel():
        raise ValueError("prediction and target sizes must match")
    min_gap = float(min_gap)
    temperature = float(temperature)
    if not math.isfinite(min_gap) or min_gap < 0.0:
        raise ValueError("min_gap must be nonnegative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")

    delta_target = target[:, None] - target[None, :]
    mask = torch.triu(
        (delta_target.abs() >= min_gap) & (delta_target != 0),
        diagonal=1,
    )
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    margin = delta_target.sign() * (
        prediction[:, None] - prediction[None, :]
    )
    margin = margin / temperature
    return F.softplus(-margin[mask]).mean()


def hybrid_affinity_loss(
    prediction,
    target,
    rank_weight,
    min_gap=0.1,
    temperature=0.5,
):
    prediction = prediction.reshape(-1)
    if prediction.numel() == 0:
        raise ValueError("prediction and target must be nonempty")
    compute_dtype = (
        torch.float32
        if prediction.dtype in (torch.float16, torch.bfloat16)
        else prediction.dtype
    )
    prediction = prediction.to(dtype=compute_dtype)
    target = target.reshape(-1).to(
        device=prediction.device, dtype=compute_dtype
    )
    if prediction.numel() != target.numel():
        raise ValueError("prediction and target sizes must match")
    rank_weight = float(rank_weight)
    if not math.isfinite(rank_weight) or rank_weight < 0.0:
        raise ValueError("rank_weight must be nonnegative")
    smooth_l1 = F.smooth_l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    rank = pairwise_logistic_rank_loss(
        prediction,
        target,
        min_gap=min_gap,
        temperature=temperature,
    )
    total = smooth_l1 + 0.10 * mse + rank_weight * rank
    return {
        "loss": total,
        "smooth_l1": smooth_l1,
        "mse": mse,
        "rank": rank,
    }


def _set_trainable(parameters, trainable):
    for parameter in parameters:
        parameter.requires_grad = bool(trainable)
        if not trainable:
            parameter.grad = None


def _interface_high_parameters(interface):
    modules = [
        *list(interface.local_path.interface_blocks[-2:]),
        interface.local_path.interface_pool,
        interface.protein_global,
        interface.protein_conditioned_pool,
        interface.ligand_conditioned_pool,
        interface.affinity_head,
    ]
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
    ]


def configure_training_stage(model, stage, optimizer=None):
    if stage not in STAGE_LRS:
        raise ValueError(
            "stage must be one of " + ", ".join(STAGE_LRS)
        )
    if optimizer is not None:
        names = [group.get("name") for group in optimizer.param_groups]
        if names != ["fusion", "interface", "gerna"]:
            raise ValueError(
                "optimizer groups must be fusion, interface, gerna"
            )
    _set_trainable(model.parameters(), False)
    _set_trainable(model.fusion.parameters(), True)
    if stage == "high":
        _set_trainable(
            model.gerna.high_level_affinity_parameters(), True
        )
        _set_trainable(
            _interface_high_parameters(model.interface), True
        )
    elif stage == "full":
        _set_trainable(model.parameters(), True)

    if optimizer is not None:
        groups = {group["name"]: group for group in optimizer.param_groups}
        for name, learning_rate in STAGE_LRS[stage].items():
            groups[name]["lr"] = learning_rate
    return STAGE_LRS[stage].copy()


def build_hybrid_optimizer(
    model,
    stage="head",
    weight_decay=1e-4,
):
    if stage not in STAGE_LRS:
        raise ValueError("unknown training stage")
    definitions = (
        ("fusion", list(model.fusion.parameters())),
        ("interface", list(model.interface.parameters())),
        ("gerna", list(model.gerna.parameters())),
    )
    seen = set()
    groups = []
    for name, parameters in definitions:
        identifiers = {id(parameter) for parameter in parameters}
        if len(identifiers) != len(parameters):
            raise ValueError(f"{name} contains duplicate parameters")
        if seen.intersection(identifiers):
            raise ValueError("optimizer parameter groups overlap")
        seen.update(identifiers)
        groups.append(
            {
                "name": name,
                "params": parameters,
                "lr": STAGE_LRS[stage][name],
            }
        )
    model_identifiers = {id(parameter) for parameter in model.parameters()}
    if seen != model_identifiers:
        raise ValueError("optimizer groups do not cover all model parameters")
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(weight_decay)
    )
    configure_training_stage(model, stage, optimizer)
    return optimizer


class ModelEMA:
    def __init__(self, model, decay=0.999):
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        if set(current) != set(self.shadow):
            raise RuntimeError("EMA and model state keys differ")
        for name, value in current.items():
            shadow = self.shadow[name]
            value = value.detach().to(
                device=shadow.device, dtype=shadow.dtype
            )
            if shadow.is_floating_point() or shadow.is_complex():
                shadow.mul_(self.decay).add_(
                    value, alpha=1.0 - self.decay
                )
            else:
                shadow.copy_(value)

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {
                name: value.detach().clone()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict) or set(state_dict) != {
            "decay",
            "shadow",
        }:
            raise RuntimeError("invalid EMA state")
        decay = float(state_dict["decay"])
        if not 0.0 <= decay < 1.0:
            raise RuntimeError("invalid EMA decay")
        incoming = state_dict["shadow"]
        if set(incoming) != set(self.shadow):
            raise RuntimeError("EMA state keys differ")
        for name, value in incoming.items():
            if not torch.is_tensor(value):
                raise RuntimeError(f"EMA value is not a tensor for {name}")
            if value.shape != self.shadow[name].shape:
                raise RuntimeError(f"EMA shape mismatch for {name}")
            if value.dtype != self.shadow[name].dtype:
                raise RuntimeError(f"EMA dtype mismatch for {name}")
        self.decay = decay
        for name, value in incoming.items():
            self.shadow[name].copy_(
                value.detach().to(device=self.shadow[name].device)
            )

    @contextmanager
    def average_parameters(self, model):
        backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        try:
            model.load_state_dict(self.shadow, strict=True)
            yield model
        finally:
            model.load_state_dict(backup, strict=True)

GERNA_INPUT_INDICES = {
    "RNA_repre": 0,
    "Seq_mask": 1,
    "RNA_Graph": 3,
    "Mol_Graph": 2,
    "RNA_feats": 4,
    "RNA_C4_coors": 5,
    "RNA_coors": 6,
    "RNA_mask": 7,
    "Mol_feats": 8,
    "Mol_coors": 9,
    "Mol_mask": 10,
    "Mol_LAS_dis": 11,
    "DTI_protein_tokens": 13,
    "DTI_protein_mask": 14,
    "DTI_global": 15,
    "Ligand_unimol_tokens": 16,
    "Ligand_unimol_mask": 17,
    "Ligand_unimol2_tokens": 18,
    "Ligand_unimol2_mask": 19,
    "PBCNet_global": 20,
    "PBCNet_graph": 21,
    "Boltz_interface": 22,
}


def recursive_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: recursive_to_device(nested, device)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [recursive_to_device(nested, device) for nested in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_device(nested, device) for nested in value)
    move = getattr(value, "to", None)
    if callable(move):
        return move(device)
    return value


def prepare_hybrid_sample(
    sample,
    device,
    collate_fn=None,
    unpack_fn=None,
):
    if not isinstance(sample, dict):
        raise ValueError("hybrid sample must be a dict")
    missing = [
        name
        for name in ("key", "label", "gerna", "interface")
        if name not in sample
    ]
    if missing:
        raise ValueError("hybrid sample is missing " + ", ".join(missing))
    if collate_fn is None:
        from data_utils.dataset import custom_collate_fn

        collate_fn = custom_collate_fn
    if unpack_fn is None:
        from dta.train_dta import unpack_batch_for_device

        unpack_fn = unpack_batch_for_device

    unpacked = unpack_fn(collate_fn([sample["gerna"]]), device)
    if not isinstance(unpacked, (tuple, list)) or len(unpacked) != 23:
        raise RuntimeError("unpacked GerNA sample must contain 23 fields")
    gerna_inputs = {
        name: unpacked[index]
        for name, index in GERNA_INPUT_INDICES.items()
    }
    interface_item = recursive_to_device(sample["interface"], device)
    target = torch.tensor(
        [float(sample["label"])], dtype=torch.float32, device=device
    )
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError(
            f"nonfinite target for sample {sample['key']}"
        )
    return gerna_inputs, interface_item, target


def _tensor_statistics(value):
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite].float()
    statistics = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "finite_count": int(finite.sum().item()),
    }
    if finite_values.numel():
        statistics.update(
            {
                "minimum": float(finite_values.min().cpu()),
                "maximum": float(finite_values.max().cpu()),
                "maximum_absolute": float(
                    finite_values.abs().max().cpu()
                ),
                "mean": float(finite_values.mean().cpu()),
            }
        )
    else:
        statistics.update(
            {
                "minimum": None,
                "maximum": None,
                "maximum_absolute": None,
                "mean": None,
            }
        )
    return statistics


class NonfiniteTrainingError(FloatingPointError):
    def __init__(self, phase, keys, tensors):
        keys = [str(key) for key in keys]
        self.diagnostics = {
            "phase": str(phase),
            "keys": keys,
            "tensors": {
                str(name): _tensor_statistics(value)
                for name, value in tensors.items()
            },
        }
        super().__init__(
            f"nonfinite {phase} for samples " + ", ".join(keys)
        )


class AMPOverflowRetryError(FloatingPointError):
    def __init__(self, diagnostics, max_retries):
        self.diagnostics = dict(diagnostics)
        self.diagnostics["max_amp_overflow_retries"] = int(max_retries)
        keys = [str(key) for key in self.diagnostics.get("keys", [])]
        super().__init__(
            f"AMP gradient overflow persisted after {int(max_retries)} "
            f"retries for samples " + ", ".join(keys)
        )


def _prediction_from_output(output, key):
    if not isinstance(output, dict) or "prediction" not in output:
        raise RuntimeError(
            f"hybrid model output for sample {key} must contain prediction"
        )
    prediction = output["prediction"]
    if not torch.is_tensor(prediction) or prediction.numel() != 1:
        raise RuntimeError(
            f"prediction for sample {key} must contain one tensor value"
        )
    prediction = prediction.reshape(1)
    if not bool(torch.isfinite(prediction).all()):
        raise NonfiniteTrainingError(
            "prediction", [key], {"prediction": prediction}
        )
    return prediction


def _autocast_context(device, enabled, dtype):
    if not enabled:
        return nullcontext()
    if torch.device(device).type != "cuda":
        raise ValueError("AMP is supported only on CUDA")
    return torch.cuda.amp.autocast(dtype=dtype)


def train_hybrid_group(
    model,
    optimizer,
    group,
    device,
    rank_weight,
    prepare_fn=prepare_hybrid_sample,
    scaler=None,
    amp_enabled=False,
    amp_dtype=torch.float16,
    max_grad_norm=1.0,
    min_gap=0.1,
    temperature=0.5,
    ema=None,
):
    samples = list(group)
    if not samples:
        raise ValueError("training group must be nonempty")
    keys = [str(sample["key"]) for sample in samples]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = []
    targets = []
    try:
        with _autocast_context(device, amp_enabled, amp_dtype):
            for sample in samples:
                gerna_inputs, interface_item, target = prepare_fn(
                    sample, device
                )
                prediction = _prediction_from_output(
                    model(gerna_inputs, interface_item), sample["key"]
                )
                predictions.append(prediction)
                targets.append(target.reshape(1))
            prediction = torch.cat(predictions, dim=0)
            target = torch.cat(targets, dim=0)
            components = hybrid_affinity_loss(
                prediction,
                target,
                rank_weight=rank_weight,
                min_gap=min_gap,
                temperature=temperature,
            )
            loss = components["loss"]
        if not bool(torch.isfinite(loss).all()):
            raise NonfiniteTrainingError(
                "loss", keys, {"loss": loss}
            )
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        bad_gradients = {
            name: gradient
            for name, gradient in gradients.items()
            if not bool(torch.isfinite(gradient).all())
        }
        optimizer_step = True
        loss_scale_before = None
        loss_scale_after = None
        overflow_diagnostics = None
        if bad_gradients:
            error = NonfiniteTrainingError(
                "gradient", keys, bad_gradients
            )
            if scaler is None:
                raise error
            loss_scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            loss_scale_after = float(scaler.get_scale())
            if loss_scale_after >= loss_scale_before:
                raise error
            optimizer.zero_grad(set_to_none=True)
            optimizer_step = False
            overflow_diagnostics = error.diagnostics
        else:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    float(max_grad_norm),
                )
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            if ema is not None:
                ema.update(model)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise

    result = {
        "keys": keys,
        "labels": target.detach().float().cpu().numpy(),
        "predictions": prediction.detach().float().cpu().numpy(),
        "optimizer_step": optimizer_step,
        "loss_scale_before": loss_scale_before,
        "loss_scale_after": loss_scale_after,
        "overflow_diagnostics": overflow_diagnostics,
    }
    result.update(
        {
            name: float(value.detach().float().cpu())
            for name, value in components.items()
        }
    )
    return result


def train_hybrid_group_with_amp_retries(
    model,
    optimizer,
    group,
    device,
    rank_weight,
    prepare_fn=prepare_hybrid_sample,
    scaler=None,
    amp_enabled=False,
    amp_dtype=torch.float16,
    max_grad_norm=1.0,
    min_gap=0.1,
    temperature=0.5,
    ema=None,
    max_amp_overflow_retries=MAX_AMP_OVERFLOW_RETRIES,
    overflow_callback=None,
):
    max_amp_overflow_retries = int(max_amp_overflow_retries)
    if max_amp_overflow_retries < 0:
        raise ValueError("max_amp_overflow_retries must be nonnegative")
    if overflow_callback is not None and not callable(overflow_callback):
        raise TypeError("overflow_callback must be callable")
    samples = list(group)
    overflow_retries = 0
    while True:
        result = train_hybrid_group(
            model,
            optimizer,
            samples,
            device,
            rank_weight,
            prepare_fn=prepare_fn,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_grad_norm=max_grad_norm,
            min_gap=min_gap,
            temperature=temperature,
            ema=ema,
        )
        if result["optimizer_step"]:
            result["amp_overflow_retries"] = overflow_retries
            return result

        event = {
            "event": "amp_overflow",
            "attempt": overflow_retries + 1,
            "sample_keys": list(result["keys"]),
            "loss_scale_before": result["loss_scale_before"],
            "loss_scale_after": result["loss_scale_after"],
        }
        if overflow_callback is not None:
            overflow_callback(event)
        if overflow_retries >= max_amp_overflow_retries:
            diagnostics = dict(result["overflow_diagnostics"])
            diagnostics["amp_overflow_attempts"] = overflow_retries + 1
            diagnostics["loss_scale_before"] = result["loss_scale_before"]
            diagnostics["loss_scale_after"] = result["loss_scale_after"]
            raise AMPOverflowRetryError(
                diagnostics, max_amp_overflow_retries
            )
        overflow_retries += 1


def evaluate_hybrid_affinity(
    model,
    data_loader,
    device,
    prepare_fn=prepare_hybrid_sample,
    amp_enabled=False,
    amp_dtype=torch.float16,
):
    was_training = model.training
    model.eval()
    keys = []
    labels = []
    predictions = []
    try:
        with torch.inference_mode(), _autocast_context(
            device, amp_enabled, amp_dtype
        ):
            for group in data_loader:
                for sample in group:
                    key = str(sample["key"])
                    gerna_inputs, interface_item, target = prepare_fn(
                        sample, device
                    )
                    prediction = _prediction_from_output(
                        model(gerna_inputs, interface_item), key
                    )
                    keys.append(key)
                    labels.append(float(target.reshape(-1)[0].cpu()))
                    predictions.append(
                        float(prediction.reshape(-1)[0].float().cpu())
                    )
    finally:
        model.train(was_training)
    if len(keys) != len(set(keys)):
        raise ValueError("evaluation keys must be unique")
    from utils.metrics import get_score_metrics

    metric_values = get_score_metrics(predictions, labels)
    metrics = dict(
        zip(("RMSE", "MAE", "Pearson", "Spearman", "CI"), metric_values)
    )
    metric_tensor = torch.tensor(
        list(metrics.values()), dtype=torch.float64
    )
    if not bool(torch.isfinite(metric_tensor).all()):
        raise NonfiniteTrainingError(
            "metrics", keys, {"metrics": metric_tensor}
        )
    return {
        "keys": keys,
        "labels": np.asarray(labels, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64),
        "metrics": metrics,
    }

def capture_rng_state(data_loader_generator=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "data_loader_generator": (
            None
            if data_loader_generator is None
            else data_loader_generator.get_state()
        ),
    }


def restore_rng_state(state, data_loader_generator=None):
    required = {
        "python", "numpy", "torch", "cuda", "data_loader_generator"
    }
    if not isinstance(state, dict) or set(state) != required:
        raise RuntimeError("invalid RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["cuda"])
    generator_state = state["data_loader_generator"]
    if (generator_state is None) != (data_loader_generator is None):
        raise RuntimeError(
            "DataLoader generator presence differs from checkpoint"
        )
    if data_loader_generator is not None:
        data_loader_generator.set_state(generator_state)


def atomic_torch_save(payload, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    schedulers,
    scaler,
    ema,
    training_state,
    top_three,
    history,
    data_loader_generator,
    provenance_hashes,
    training_config,
):
    scheduler_states = {
        str(name): scheduler.state_dict()
        for name, scheduler in schedulers.items()
    }
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedulers": scheduler_states,
        "scaler": None if scaler is None else scaler.state_dict(),
        "ema": ema.state_dict(),
        "training_state": dict(training_state),
        "top_three": list(top_three),
        "history": list(history),
        "rng": capture_rng_state(data_loader_generator),
        "provenance_hashes": dict(provenance_hashes),
        "training_config": _normalize_json_mapping(
            training_config, "training_config"
        ),
    }
    atomic_torch_save(payload, path)


def _training_configs_match_for_resume(
    saved,
    expected,
    allow_stage_extension,
):
    if saved == expected:
        return True
    if not allow_stage_extension:
        return False
    if not isinstance(saved, dict) or not isinstance(expected, dict):
        return False
    saved_without_epochs = dict(saved)
    expected_without_epochs = dict(expected)
    saved_epochs = saved_without_epochs.pop("stage_epochs", None)
    expected_epochs = expected_without_epochs.pop("stage_epochs", None)
    if saved_without_epochs != expected_without_epochs:
        return False
    if not isinstance(saved_epochs, dict) or not isinstance(
        expected_epochs, dict
    ):
        return False
    if set(saved_epochs) != set(STAGE_EPOCHS) or set(
        expected_epochs
    ) != set(STAGE_EPOCHS):
        return False
    stage_order = ("head", "high", "full")
    last_started = max(
        (
            index
            for index, stage in enumerate(stage_order)
            if int(saved_epochs[stage]) > 0
        ),
        default=-1,
    )
    for index, stage in enumerate(stage_order):
        previous = int(saved_epochs[stage])
        requested = int(expected_epochs[stage])
        if previous == requested:
            continue
        if (
            previous != 0
            or requested <= 0
            or index <= last_started
        ):
            return False
    return True


def _read_training_checkpoint_payload(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise RuntimeError("unsupported training checkpoint")
    return payload


def _validate_training_checkpoint_metadata(
    payload,
    expected_provenance_hashes,
    expected_training_config,
    allow_stage_extension,
):
    saved_hashes = payload.get("provenance_hashes")
    if saved_hashes != dict(expected_provenance_hashes):
        raise RuntimeError(
            "training checkpoint provenance hashes differ"
        )
    saved_training_config = payload.get("training_config")
    expected_training_config = _normalize_json_mapping(
        expected_training_config, "expected_training_config"
    )
    if not _training_configs_match_for_resume(
        saved_training_config,
        expected_training_config,
        allow_stage_extension=allow_stage_extension,
    ):
        raise RuntimeError(
            "training checkpoint training config differs"
        )
    return saved_hashes


def _validate_state_dict_schema(reference, incoming, name):
    if not isinstance(incoming, dict) or set(incoming) != set(reference):
        raise RuntimeError(f"{name} state keys differ")
    for key, expected in reference.items():
        actual = incoming[key]
        if not torch.is_tensor(actual):
            raise RuntimeError(f"{name} state value is not a tensor: {key}")
        if actual.shape != expected.shape:
            raise RuntimeError(f"{name} state shape differs: {key}")
        if actual.dtype != expected.dtype:
            raise RuntimeError(f"{name} state dtype differs: {key}")


def _preflight_fit_checkpoint_payload(payload, model, amp_enabled):
    reference = model.state_dict()
    _validate_state_dict_schema(
        reference, payload.get("model"), "model"
    )
    ema_state = payload.get("ema")
    if not isinstance(ema_state, dict):
        raise RuntimeError("EMA checkpoint state is invalid")
    _validate_state_dict_schema(
        reference, ema_state.get("shadow"), "EMA"
    )
    saved_optimizer = payload.get("optimizer")
    if not isinstance(saved_optimizer, dict):
        raise RuntimeError("optimizer checkpoint state is invalid")
    saved_groups = saved_optimizer.get("param_groups")
    expected_groups = (
        ("fusion", list(model.fusion.parameters())),
        ("interface", list(model.interface.parameters())),
        ("gerna", list(model.gerna.parameters())),
    )
    if not isinstance(saved_groups, list) or len(saved_groups) != 3:
        raise RuntimeError("optimizer checkpoint groups differ")
    for saved, (name, parameters) in zip(
        saved_groups, expected_groups
    ):
        if (
            saved.get("name") != name
            or len(saved.get("params", [])) != len(parameters)
        ):
            raise RuntimeError(
                f"optimizer checkpoint group differs: {name}"
            )
    schedulers = payload.get("schedulers")
    if not isinstance(schedulers, dict) or set(schedulers) != {"hybrid"}:
        raise RuntimeError("training checkpoint scheduler sets differ")
    if (payload.get("scaler") is None) == bool(amp_enabled):
        raise RuntimeError(
            "training checkpoint scaler presence differs"
        )
    rng = payload.get("rng")
    required_rng = {
        "python",
        "numpy",
        "torch",
        "cuda",
        "data_loader_generator",
    }
    if not isinstance(rng, dict) or set(rng) != required_rng:
        raise RuntimeError("invalid RNG state")
    cpu_rng_tensors = [rng["torch"], rng["data_loader_generator"]]
    if rng["cuda"] is not None:
        cpu_rng_tensors.extend(rng["cuda"])
    for value in cpu_rng_tensors:
        if value is not None and (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
        ):
            raise RuntimeError(
                "checkpoint RNG tensors must remain on CPU"
            )
    top_three = payload.get("top_three")
    if not isinstance(top_three, list):
        raise RuntimeError("checkpoint top_three is invalid")
    for index, candidate in enumerate(top_three):
        if not isinstance(candidate, dict):
            raise RuntimeError("checkpoint candidate is invalid")
        _validate_state_dict_schema(
            reference,
            candidate.get("model_state_dict"),
            f"validation rank {index + 1}",
        )
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise RuntimeError("checkpoint training_state is invalid")
    best_ema = training_state.get("best_ema")
    if best_ema is not None:
        _validate_state_dict_schema(
            reference,
            best_ema.get("model_state_dict"),
            "best EMA",
        )


def load_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    schedulers,
    scaler,
    ema,
    data_loader_generator,
    expected_provenance_hashes,
    expected_training_config,
    map_location="cpu",
    allow_stage_extension=False,
    checkpoint_payload=None,
):
    payload = (
        _read_training_checkpoint_payload(
            path, map_location=map_location
        )
        if checkpoint_payload is None
        else checkpoint_payload
    )
    saved_hashes = _validate_training_checkpoint_metadata(
        payload,
        expected_provenance_hashes,
        expected_training_config,
        allow_stage_extension=allow_stage_extension,
    )
    saved_schedulers = payload.get("schedulers")
    if (
        not isinstance(saved_schedulers, dict)
        or set(saved_schedulers) != set(schedulers)
    ):
        raise RuntimeError("training checkpoint scheduler sets differ")
    saved_scaler = payload.get("scaler")
    if (saved_scaler is None) != (scaler is None):
        raise RuntimeError("training checkpoint scaler presence differs")

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(saved_schedulers[name])
    if scaler is not None:
        scaler.load_state_dict(saved_scaler)
    ema.load_state_dict(payload["ema"])
    restore_rng_state(payload["rng"], data_loader_generator)
    return {
        "training_state": payload["training_state"],
        "top_three": payload["top_three"],
        "history": payload["history"],
        "provenance_hashes": saved_hashes,
    }

def _normalize_json_mapping(payload, name):
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a dict")
    try:
        return json.loads(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error


def validation_candidate_is_better(candidate, incumbent, ci_tolerance=1e-6):
    if incumbent is None:
        return True
    tolerance = float(ci_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("ci_tolerance must be finite and nonnegative")

    candidate_metrics = candidate.get("metrics", {})
    incumbent_metrics = incumbent.get("metrics", {})
    required = ("CI", "Spearman", "RMSE")
    try:
        candidate_values = {
            name: float(candidate_metrics[name]) for name in required
        }
        incumbent_values = {
            name: float(incumbent_metrics[name]) for name in required
        }
        candidate_epoch = int(candidate["epoch"])
        incumbent_epoch = int(incumbent["epoch"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid validation candidate") from error
    if not all(
        math.isfinite(value)
        for value in list(candidate_values.values())
        + list(incumbent_values.values())
    ):
        raise ValueError("validation metrics must be finite")

    ci_difference = candidate_values["CI"] - incumbent_values["CI"]
    if abs(ci_difference) > tolerance:
        return ci_difference > 0.0
    spearman_difference = (
        candidate_values["Spearman"] - incumbent_values["Spearman"]
    )
    if spearman_difference != 0.0:
        return spearman_difference > 0.0
    rmse_difference = (
        candidate_values["RMSE"] - incumbent_values["RMSE"]
    )
    if rmse_difference != 0.0:
        return rmse_difference < 0.0
    return candidate_epoch < incumbent_epoch


class StageWarmupCosineScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.active_stage = None
        self.base_lrs = []
        self.total_steps = 0
        self.warmup_steps = 0
        self.step_count = 0

    def start_stage(self, stage, total_steps, warmup_fraction=0.1):
        if stage not in STAGE_LRS:
            raise ValueError("unknown training stage")
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        fraction = float(warmup_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("warmup_fraction must be in [0, 1]")
        names = [group.get("name") for group in self.optimizer.param_groups]
        if names != ["fusion", "interface", "gerna"]:
            raise ValueError(
                "optimizer groups must be fusion, interface, gerna"
            )
        self.active_stage = stage
        self.base_lrs = [
            float(STAGE_LRS[stage][name]) for name in names
        ]
        self.total_steps = total_steps
        self.warmup_steps = min(
            total_steps,
            max(1, int(math.ceil(total_steps * fraction))),
        )
        self.step_count = 0
        self.apply_current()

    def _factor(self):
        if self.active_stage is None:
            return 1.0
        if self.step_count < self.warmup_steps:
            return float(self.step_count + 1) / float(self.warmup_steps)
        remaining = max(1, self.total_steps - self.warmup_steps)
        progress = min(
            1.0,
            max(
                0.0,
                float(self.step_count - self.warmup_steps)
                / float(remaining),
            ),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply_current(self):
        if self.active_stage is None:
            return
        factor = self._factor()
        for group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs
        ):
            group["lr"] = base_lr * factor

    def step(self):
        if self.active_stage is None:
            raise RuntimeError("scheduler stage is not initialized")
        self.step_count = min(self.total_steps, self.step_count + 1)
        self.apply_current()

    def state_dict(self):
        return {
            "active_stage": self.active_stage,
            "base_lrs": list(self.base_lrs),
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict):
        required = {
            "active_stage",
            "base_lrs",
            "total_steps",
            "warmup_steps",
            "step_count",
        }
        if not isinstance(state_dict, dict) or set(state_dict) != required:
            raise RuntimeError("invalid stage scheduler state")
        base_lrs = [float(value) for value in state_dict["base_lrs"]]
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError("scheduler optimizer group count differs")
        self.active_stage = state_dict["active_stage"]
        self.base_lrs = base_lrs
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.step_count = int(state_dict["step_count"])
        if self.active_stage not in STAGE_LRS:
            raise RuntimeError("invalid active scheduler stage")
        if not (
            0 < self.warmup_steps <= self.total_steps
            and 0 <= self.step_count <= self.total_steps
        ):
            raise RuntimeError("invalid stage scheduler counters")
        self.apply_current()


def _temporary_path(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    return destination, temporary


def _atomic_write_text(path, text):
    destination, temporary = _temporary_path(path)
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path, payload):
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_csv(path, rows, fieldnames):
    destination, temporary = _temporary_path(path)
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_predictions(path, evaluation):
    keys = [str(key) for key in evaluation["keys"]]
    labels = np.asarray(evaluation["labels"], dtype=np.float64)
    predictions = np.asarray(
        evaluation["predictions"], dtype=np.float64
    )
    if len(keys) != len(set(keys)):
        raise ValueError("prediction keys must be unique")
    if labels.shape != predictions.shape or labels.shape != (len(keys),):
        raise ValueError("prediction arrays must align with keys")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("prediction arrays must be finite")
    destination, temporary = _temporary_path(path)
    try:
        with open(temporary, "wb") as handle:
            np.savez(
                handle,
                keys=np.asarray(keys, dtype=np.str_),
                labels=labels,
                predictions=predictions,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _clone_state_to_cpu(state_dict):
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def _candidate_metadata(candidate):
    if candidate is None:
        return None
    return {
        "epoch": int(candidate["epoch"]),
        "stage": str(candidate["stage"]),
        "variant": str(candidate["variant"]),
        "metrics": {
            name: float(value)
            for name, value in candidate["metrics"].items()
        },
    }


def _candidate_from_evaluation(
    evaluation,
    epoch,
    stage,
    variant,
    state_dict,
):
    return {
        "epoch": int(epoch),
        "stage": str(stage),
        "variant": str(variant),
        "metrics": {
            name: float(value)
            for name, value in evaluation["metrics"].items()
        },
        "keys": list(evaluation["keys"]),
        "labels": np.asarray(
            evaluation["labels"], dtype=np.float64
        ).copy(),
        "predictions": np.asarray(
            evaluation["predictions"], dtype=np.float64
        ).copy(),
        "model_state_dict": _clone_state_to_cpu(state_dict),
    }


def _insert_validation_candidate(top_three, candidate):
    result = list(top_three)
    inserted = False
    for index, existing in enumerate(result):
        if validation_candidate_is_better(candidate, existing):
            result.insert(index, candidate)
            inserted = True
            break
    if not inserted:
        result.append(candidate)
    return result[:3]


def _set_loader_epoch(data_loader, epoch):
    for owner_name in ("batch_sampler", "sampler"):
        owner = getattr(data_loader, owner_name, None)
        set_epoch = getattr(owner, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
            return


def _validate_exact_resume_loader(
    data_loader,
    data_loader_generator,
    loader_name,
):
    workers = int(getattr(data_loader, "num_workers", 0))
    if workers != 0:
        raise ValueError(
            f"{loader_name} must use num_workers=0 for exact resume"
        )
    owners = [
        data_loader,
        getattr(data_loader, "sampler", None),
        getattr(data_loader, "batch_sampler", None),
    ]
    batch_sampler = getattr(data_loader, "batch_sampler", None)
    owners.append(getattr(batch_sampler, "sampler", None))
    for owner in owners:
        generator = getattr(owner, "generator", None)
        if (
            generator is not None
            and generator is not data_loader_generator
        ):
            raise ValueError(
                f"{loader_name} uses an independent generator"
            )


def _next_stage_schedule(stage_epochs):
    schedule = []
    for stage in ("head", "high", "full"):
        count = int(stage_epochs[stage])
        if count < 0:
            raise ValueError("stage epoch counts must be nonnegative")
        schedule.extend((stage, local_epoch) for local_epoch in range(count))
    if not schedule:
        raise ValueError("at least one training epoch is required")
    return schedule


def _best_checkpoint_payload(best_raw, best_ema):
    selected = best_raw
    if validation_candidate_is_better(best_ema, selected):
        selected = best_ema
    return {
        "format_version": 1,
        "model_state_dict": selected["model_state_dict"],
        "selected": _candidate_metadata(selected),
        "raw_best": _candidate_metadata(best_raw),
        "raw_best_state_dict": best_raw["model_state_dict"],
        "ema_best": _candidate_metadata(best_ema),
        "ema_best_state_dict": best_ema["model_state_dict"],
    }, selected


def _history_fieldnames():
    fields = [
        "epoch",
        "stage",
        "stage_epoch",
        "train_loss",
        "train_smooth_l1",
        "train_mse",
        "train_rank",
        "fusion_lr",
        "interface_lr",
        "gerna_lr",
    ]
    for prefix in ("valid", "ema_valid"):
        fields.extend(
            f"{prefix}_{name}"
            for name in ("RMSE", "MAE", "Pearson", "Spearman", "CI")
        )
    return fields


def _initial_log_lines(output, resume):
    path = Path(output) / "train.log"
    if not resume:
        return [
            json.dumps(
                {"event": "start", "resume": False},
                sort_keys=True,
            )
        ]
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("resume train.log is empty")
    try:
        for line in lines:
            json.loads(line)
    except (TypeError, ValueError) as error:
        raise RuntimeError("resume train.log contains invalid JSON") from error
    lines.append(
        json.dumps({"event": "resume", "resume": True}, sort_keys=True)
    )
    return lines


def _write_committed_artifacts(
    *,
    output,
    model,
    ema,
    top_three,
    best_ema,
    history,
    log_lines,
    last_epoch,
    last_stage,
):
    if not top_three or best_ema is None:
        raise RuntimeError(
            "committed training state is missing validation candidates"
        )
    best_payload, selected = _best_checkpoint_payload(
        top_three[0], best_ema
    )
    atomic_torch_save(
        best_payload, output / "Model_hybrid_best.pth"
    )
    atomic_torch_save(
        {
            "format_version": 1,
            "epoch": int(last_epoch),
            "stage": str(last_stage),
            "model_state_dict": _clone_state_to_cpu(
                model.state_dict()
            ),
            "ema_state_dict": _clone_state_to_cpu(ema.shadow),
        },
        output / "Model_hybrid_last.pth",
    )
    for rank, candidate in enumerate(top_three, start=1):
        atomic_torch_save(
            {
                "format_version": 1,
                "rank": rank,
                "epoch": candidate["epoch"],
                "stage": candidate["stage"],
                "metrics": candidate["metrics"],
                "model_state_dict": candidate[
                    "model_state_dict"
                ],
            },
            output / f"validation_rank_{rank}.pth",
        )
    atomic_write_predictions(
        output / "valid_predictions.npz",
        {
            "keys": selected["keys"],
            "labels": selected["labels"],
            "predictions": selected["predictions"],
        },
    )
    atomic_write_json(
        output / "best_metrics.json",
        _candidate_metadata(selected),
    )
    atomic_write_csv(
        output / "metrics_history.csv",
        history,
        _history_fieldnames(),
    )
    _atomic_write_text(
        output / "train.log", "\n".join(log_lines) + "\n"
    )
    return selected


def _record_nonfinite_failure(
    *,
    output,
    log_lines,
    error,
    stage,
    epoch,
    stage_epoch,
    global_step,
    optimizer,
    sample_keys=None,
    validation_variant=None,
):
    diagnostics = getattr(
        error,
        "diagnostics",
        {
            "phase": "unknown",
            "keys": [] if sample_keys is None else sample_keys,
            "tensors": {},
        },
    )
    if sample_keys is None:
        sample_keys = diagnostics.get("keys", [])
    sample_keys = [str(key) for key in sample_keys]
    failure = {
        "status": "failed_nonfinite",
        "error": str(error),
        "stage": str(stage),
        "epoch": int(epoch),
        "stage_epoch": int(stage_epoch),
        "global_step": int(global_step),
        "sample_keys": sample_keys,
        "learning_rates": {
            group["name"]: float(group["lr"])
            for group in optimizer.param_groups
        },
        "diagnostics": diagnostics,
    }
    if validation_variant is not None:
        failure["validation_variant"] = str(validation_variant)
    atomic_write_json(output / "training_failure.json", failure)
    log_lines.append(
        json.dumps(
            {"event": "failed_nonfinite", **failure},
            sort_keys=True,
        )
    )
    _atomic_write_text(
        output / "train.log", "\n".join(log_lines) + "\n"
    )
    return failure


def fit_hybrid_affinity(
    *,
    model,
    train_loader,
    valid_loader,
    output_dir,
    device,
    rank_weight,
    run_config,
    provenance,
    provenance_hashes,
    data_loader_generator,
    prepare_fn=prepare_hybrid_sample,
    stage_epochs=None,
    amp_enabled=True,
    max_grad_norm=1.0,
    resume=False,
):
    epochs = dict(STAGE_EPOCHS if stage_epochs is None else stage_epochs)
    if set(epochs) != set(STAGE_EPOCHS):
        raise ValueError("stage_epochs must define head, high and full")
    schedule = _next_stage_schedule(epochs)
    try:
        groups_per_epoch = len(train_loader)
    except TypeError as error:
        raise ValueError("train_loader must have a stable length") from error
    if groups_per_epoch <= 0:
        raise ValueError("train_loader must be nonempty")
    device = torch.device(device)
    amp_enabled = bool(amp_enabled)
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP training requires CUDA")

    _validate_exact_resume_loader(
        train_loader,
        data_loader_generator,
        loader_name="train_loader",
    )
    _validate_exact_resume_loader(
        valid_loader,
        data_loader_generator,
        loader_name="valid_loader",
    )

    normalized_run_config = _normalize_json_mapping(
        run_config, "run_config"
    )
    normalized_provenance = _normalize_json_mapping(
        provenance, "provenance"
    )
    training_config = _normalize_json_mapping(
        {
            "schema_version": 1,
            "run_config": normalized_run_config,
            "stage_epochs": {
                name: int(value) for name, value in epochs.items()
            },
            "stage_lrs": STAGE_LRS,
            "groups_per_epoch": groups_per_epoch,
            "rank_weight": float(rank_weight),
            "rank_min_gap": 0.1,
            "rank_temperature": 0.5,
            "amp_enabled": amp_enabled,
            "amp_dtype": "float16",
            "max_amp_overflow_retries": MAX_AMP_OVERFLOW_RETRIES,
            "max_grad_norm": (
                None
                if max_grad_norm is None
                else float(max_grad_norm)
            ),
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "ema_decay": 0.999,
            "scheduler": {
                "name": "linear_warmup_cosine",
                "warmup_fraction": WARMUP_FRACTION,
            },
        },
        "training_config",
    )

    output = Path(output_dir)
    checkpoint_path = output / "training_checkpoint.pth"
    run_config_path = output / "run_config.json"
    provenance_path = output / "provenance.json"
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(str(output))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))
        for path, expected, name in (
            (run_config_path, normalized_run_config, "run_config"),
            (provenance_path, normalized_provenance, "provenance"),
        ):
            if not path.is_file():
                raise RuntimeError(f"resume requires existing {name}")
            try:
                with path.open(encoding="utf-8") as handle:
                    actual = json.load(handle)
            except (OSError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"resume {name} is unreadable"
                ) from error
            if actual != expected:
                raise RuntimeError(f"resume {name} differs")
        log_lines = _initial_log_lines(output, resume=True)
        resume_payload = _read_training_checkpoint_payload(
            checkpoint_path, map_location="cpu"
        )
        _validate_training_checkpoint_metadata(
            resume_payload,
            provenance_hashes,
            training_config,
            allow_stage_extension=True,
        )
        _preflight_fit_checkpoint_payload(
            resume_payload,
            model,
            amp_enabled=amp_enabled,
        )
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                "training checkpoint already exists; use resume"
            )
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_config_path, normalized_run_config)
        atomic_write_json(provenance_path, normalized_provenance)
        log_lines = _initial_log_lines(output, resume=False)
        resume_payload = None

    model.to(device)
    optimizer = build_hybrid_optimizer(
        model,
        stage=schedule[0][0],
        weight_decay=1e-4,
    )
    scheduler = StageWarmupCosineScheduler(optimizer)
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if amp_enabled
        else None
    )
    ema = ModelEMA(model, decay=0.999)
    history = []
    top_three = []
    best_ema = None
    next_epoch = 0
    global_step = 0

    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))
        loaded = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            schedulers={"hybrid": scheduler},
            scaler=scaler,
            ema=ema,
            data_loader_generator=data_loader_generator,
            expected_provenance_hashes=provenance_hashes,
            expected_training_config=training_config,
            map_location="cpu",
            allow_stage_extension=True,
            checkpoint_payload=resume_payload,
        )
        training_state = loaded["training_state"]
        next_epoch = int(training_state["next_epoch"])
        global_step = int(training_state["global_step"])
        best_ema = training_state.get("best_ema")
        top_three = list(loaded["top_three"])
        history = list(loaded["history"])
        committed_log_lines = training_state.get("log_lines")
        if not isinstance(committed_log_lines, list) or not all(
            isinstance(line, str) for line in committed_log_lines
        ):
            raise RuntimeError(
                "training checkpoint is missing committed log lines"
            )
        for line in committed_log_lines:
            try:
                json.loads(line)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "training checkpoint contains invalid log lines"
                ) from error
        log_lines = list(committed_log_lines)
        log_lines.append(
            json.dumps(
                {"event": "resume", "resume": True},
                sort_keys=True,
            )
        )
        if next_epoch > len(schedule):
            raise RuntimeError(
                "resume schedule is shorter than completed training"
            )
        selected = _write_committed_artifacts(
            output=output,
            model=model,
            ema=ema,
            top_three=top_three,
            best_ema=best_ema,
            history=history,
            log_lines=log_lines,
            last_epoch=next_epoch - 1,
            last_stage=training_state["stage"],
        )
        if training_state.get("complete") and next_epoch >= len(schedule):
            return {
                "history": history,
                "best": _candidate_metadata(selected),
                "top_three": [
                    _candidate_metadata(candidate)
                    for candidate in top_three
                ],
            }
    for global_epoch in range(next_epoch, len(schedule)):
        stage, stage_epoch = schedule[global_epoch]
        if scheduler.active_stage != stage:
            configure_training_stage(model, stage, optimizer)
            scheduler.start_stage(
                stage,
                total_steps=groups_per_epoch * int(epochs[stage]),
                warmup_fraction=WARMUP_FRACTION,
            )
        else:
            configure_training_stage(model, stage, optimizer=None)
            scheduler.apply_current()
        _set_loader_epoch(train_loader, global_epoch)

        totals = {
            "loss": 0.0,
            "smooth_l1": 0.0,
            "mse": 0.0,
            "rank": 0.0,
        }
        sample_count = 0

        def record_amp_overflow(event):
            payload = dict(event)
            payload.update(
                {
                    "stage": str(stage),
                    "epoch": int(global_epoch),
                    "stage_epoch": int(stage_epoch),
                    "global_step": int(global_step),
                }
            )
            log_lines.append(json.dumps(payload, sort_keys=True))
            _atomic_write_text(
                output / "train.log", "\n".join(log_lines) + "\n"
            )

        for group in train_loader:
            try:
                result = train_hybrid_group_with_amp_retries(
                    model,
                    optimizer,
                    group,
                    device,
                    rank_weight=rank_weight,
                    prepare_fn=prepare_fn,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    max_grad_norm=max_grad_norm,
                    ema=ema,
                    max_amp_overflow_retries=MAX_AMP_OVERFLOW_RETRIES,
                    overflow_callback=record_amp_overflow,
                )
            except FloatingPointError as error:
                _record_nonfinite_failure(
                    output=output,
                    log_lines=log_lines,
                    error=error,
                    stage=stage,
                    epoch=global_epoch,
                    stage_epoch=stage_epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    sample_keys=[
                        str(sample.get("key", "<missing>"))
                        for sample in group
                    ],
                )
                raise
            count = len(result["keys"])
            sample_count += count
            for name in totals:
                totals[name] += result[name] * count
            scheduler.step()
            global_step += 1
        if sample_count == 0:
            raise RuntimeError("training epoch contained no samples")
        train_metrics = {
            name: value / sample_count
            for name, value in totals.items()
        }

        try:
            raw_evaluation = evaluate_hybrid_affinity(
                model,
                valid_loader,
                device,
                prepare_fn=prepare_fn,
                amp_enabled=amp_enabled,
            )
        except FloatingPointError as error:
            _record_nonfinite_failure(
                output=output,
                log_lines=log_lines,
                error=error,
                stage=stage,
                epoch=global_epoch,
                stage_epoch=stage_epoch,
                global_step=global_step,
                optimizer=optimizer,
                validation_variant="raw",
            )
            raise
        try:
            with ema.average_parameters(model):
                ema_evaluation = evaluate_hybrid_affinity(
                    model,
                    valid_loader,
                    device,
                    prepare_fn=prepare_fn,
                    amp_enabled=amp_enabled,
                )
        except FloatingPointError as error:
            _record_nonfinite_failure(
                output=output,
                log_lines=log_lines,
                error=error,
                stage=stage,
                epoch=global_epoch,
                stage_epoch=stage_epoch,
                global_step=global_step,
                optimizer=optimizer,
                validation_variant="ema",
            )
            raise

        raw_metadata = {
            "epoch": global_epoch,
            "stage": stage,
            "variant": "raw",
            "metrics": raw_evaluation["metrics"],
        }
        qualifies = (
            len(top_three) < 3
            or validation_candidate_is_better(
                raw_metadata, top_three[-1]
            )
        )
        if qualifies:
            raw_candidate = _candidate_from_evaluation(
                raw_evaluation,
                global_epoch,
                stage,
                "raw",
                model.state_dict(),
            )
            top_three = _insert_validation_candidate(
                top_three, raw_candidate
            )
        best_raw = top_three[0]

        ema_metadata = {
            "epoch": global_epoch,
            "stage": stage,
            "variant": "ema",
            "metrics": ema_evaluation["metrics"],
        }
        if validation_candidate_is_better(ema_metadata, best_ema):
            best_ema = _candidate_from_evaluation(
                ema_evaluation,
                global_epoch,
                stage,
                "ema",
                ema.shadow,
            )

        lr_by_name = {
            group["name"]: float(group["lr"])
            for group in optimizer.param_groups
        }
        row = {
            "epoch": global_epoch,
            "stage": stage,
            "stage_epoch": stage_epoch,
            "train_loss": train_metrics["loss"],
            "train_smooth_l1": train_metrics["smooth_l1"],
            "train_mse": train_metrics["mse"],
            "train_rank": train_metrics["rank"],
            "fusion_lr": lr_by_name["fusion"],
            "interface_lr": lr_by_name["interface"],
            "gerna_lr": lr_by_name["gerna"],
        }
        for prefix, evaluation in (
            ("valid", raw_evaluation),
            ("ema_valid", ema_evaluation),
        ):
            for name, value in evaluation["metrics"].items():
                row[f"{prefix}_{name}"] = float(value)
        history.append(row)

        best_payload, selected = _best_checkpoint_payload(
            best_raw, best_ema
        )
        log_lines.append(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": global_epoch,
                    "stage": stage,
                    "train_loss": train_metrics["loss"],
                    "valid": raw_evaluation["metrics"],
                    "ema_valid": ema_evaluation["metrics"],
                    "selected": _candidate_metadata(selected),
                },
                sort_keys=True,
            )
        )
        complete = global_epoch + 1 >= len(schedule)
        training_state = {
            "stage": stage,
            "next_epoch": global_epoch + 1,
            "global_step": global_step,
            "complete": complete,
            "best_ema": best_ema,
            "log_lines": list(log_lines),
        }
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            schedulers={"hybrid": scheduler},
            scaler=scaler,
            ema=ema,
            training_state=training_state,
            top_three=top_three,
            history=history,
            data_loader_generator=data_loader_generator,
            provenance_hashes=provenance_hashes,
            training_config=training_config,
        )
        _write_committed_artifacts(
            output=output,
            model=model,
            ema=ema,
            top_three=top_three,
            best_ema=best_ema,
            history=history,
            log_lines=log_lines,
            last_epoch=global_epoch,
            last_stage=stage,
        )

    best_payload, selected = _best_checkpoint_payload(
        top_three[0], best_ema
    )
    return {
        "history": history,
        "best": _candidate_metadata(selected),
        "top_three": [
            _candidate_metadata(candidate) for candidate in top_three
        ],
    }
