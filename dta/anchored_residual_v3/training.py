"""Guarded staged training primitives for anchored residual v3."""

from copy import deepcopy

import torch

from dta.hybrid_cached_training_v2 import is_better_validation, upper_bound_affinity_loss


STAGE_DEFAULTS = {
    "residual": {"residual_lr": 1e-4, "consistency_weight": 0.10},
    "fusion": {"residual_lr": 1e-4, "fusion_lr": 1e-5, "consistency_weight": 0.03},
    "selective_e2e": {
        "residual_lr": 1e-4,
        "fusion_lr": 1e-5,
        "interface_lr": 1e-6,
        "gerna_lr": 3e-7,
        "consistency_weight": 0.01,
    },
}


def _direct_parameter_modules(component):
    modules = []
    for name, module in component.named_modules():
        parameters = list(module.parameters(recurse=False))
        if parameters:
            modules.append((name or "root", module, parameters))
    return modules


def _enable_top_modules(component, count):
    modules = _direct_parameter_modules(component)
    if count <= 0 or not modules:
        return []
    selected = modules[-int(count):]
    for _, _, parameters in selected:
        for parameter in parameters:
            parameter.requires_grad = True
    return [(name, parameters) for name, _, parameters in selected]


def _group(name, component, parameters, learning_rate):
    return {
        "name": name,
        "component": component,
        "params": list(parameters),
        "lr": float(learning_rate),
    }


def configure_stage(
    model,
    stage,
    *,
    residual_learning_rate=1e-4,
    fusion_learning_rate=1e-5,
    interface_learning_rate=1e-6,
    gerna_learning_rate=3e-7,
    gerna_top_owners=1,
    interface_top_owners=1,
    weight_decay=1e-4,
):
    """Freeze all parameters, then enable the exact components for one stage."""
    if stage not in STAGE_DEFAULTS:
        raise ValueError("stage must be residual, fusion, or selective_e2e")
    for name in ("residual", "fusion", "interface", "gerna"):
        if not hasattr(model, name):
            raise ValueError(f"model is missing {name}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    groups = [
        _group(
            "residual", "residual", model.residual.parameters(), residual_learning_rate
        )
    ]
    for parameter in model.residual.parameters():
        parameter.requires_grad = True
    selected = {"residual": ["all"], "fusion": [], "interface": [], "gerna": []}
    if stage in {"fusion", "selective_e2e"}:
        for parameter in model.fusion.parameters():
            parameter.requires_grad = True
        groups.append(_group("fusion", "fusion", model.fusion.parameters(), fusion_learning_rate))
        selected["fusion"] = ["all"]
    if stage == "selective_e2e":
        for component_name, learning_rate, count in (
            ("interface", interface_learning_rate, interface_top_owners),
            ("gerna", gerna_learning_rate, gerna_top_owners),
        ):
            enabled = _enable_top_modules(getattr(model, component_name), count)
            for owner, parameters in enabled:
                groups.append(
                    _group(
                        f"{component_name}.{owner}",
                        component_name,
                        parameters,
                        learning_rate,
                    )
                )
            selected[component_name] = [owner for owner, _ in enabled]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("stage optimizer parameter groups overlap")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    return optimizer, {"stage": stage, "trainable_owners": selected}


def snapshot_trainable_parameters(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def anchored_affinity_loss(
    output,
    target,
    *,
    consistency_weight,
    named_parameters=None,
    reference_parameters=None,
    l2_sp_weight=0.0,
):
    if not isinstance(output, dict) or {"prediction", "delta", "expert_weights"}.difference(output):
        raise ValueError("anchored output is missing prediction, delta, or expert_weights")
    result = dict(
        upper_bound_affinity_loss(
            output["prediction"], target, output["expert_weights"]
        )
    )
    consistency = output["delta"].float().square().mean()
    l2_sp = consistency.new_zeros(())
    if named_parameters is not None or reference_parameters is not None:
        if not isinstance(named_parameters, dict) or not isinstance(reference_parameters, dict):
            raise ValueError("named_parameters and reference_parameters must both be mappings")
        for name, parameter in named_parameters.items():
            if name not in reference_parameters:
                raise ValueError(f"L2-SP reference is missing {name}")
            l2_sp = l2_sp + (parameter - reference_parameters[name].to(parameter)).square().mean()
    result["consistency"] = consistency
    result["l2_sp"] = l2_sp
    result["loss"] = result["loss"] + float(consistency_weight) * consistency + float(l2_sp_weight) * l2_sp
    return result


class ValidationGuard:
    """Keep only validation improvements and restore the accepted state otherwise."""

    def __init__(self, model, metrics, *, variant):
        self.metrics = dict(metrics)
        self.variant = str(variant)
        self._state = deepcopy(model.state_dict())

    def consider(self, model, metrics, variant):
        if not is_better_validation(metrics, self.metrics):
            return False
        self.metrics = dict(metrics)
        self.variant = str(variant)
        self._state = deepcopy(model.state_dict())
        return True

    def restore(self, model):
        model.load_state_dict(self._state, strict=True)



def _autocast(device, enabled):
    from contextlib import nullcontext
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def train_anchored_group(model, optimizer, group, device, *, consistency_weight,
                         reference_parameters, l2_sp_weight=1e-5, prepare_fn=None,
                         scaler=None, amp_enabled=False, max_grad_norm=1.0):
    from dta.anchored_residual_v3.data import prepare_anchored_group
    from dta.hybrid_affinity_training import NonfiniteTrainingError
    prepare_fn = prepare_anchored_group if prepare_fn is None else prepare_fn
    keys, gerna_inputs, packed, targets, teacher = prepare_fn(group, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, amp_enabled):
        output = model(gerna_inputs, packed, teacher)
        trainable = {name: parameter for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        components = anchored_affinity_loss(
            output, targets, consistency_weight=consistency_weight,
            named_parameters=trainable, reference_parameters=reference_parameters,
            l2_sp_weight=l2_sp_weight,
        )
        loss = components["loss"]
    if not bool(torch.isfinite(loss).all()):
        raise NonfiniteTrainingError("loss", keys, {"loss": loss})
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    nonfinite = {name: parameter.grad for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None
                 and not bool(torch.isfinite(parameter.grad).all())}
    if nonfinite:
        raise NonfiniteTrainingError("gradient", keys, nonfinite)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(max_grad_norm),
    )
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return {"keys": list(keys), **{name: float(value.detach().float().cpu())
            for name, value in components.items()}}


@torch.inference_mode()
def evaluate_anchored(model, loader, device, *, prepare_fn=None, amp_enabled=False):
    import numpy as np
    from dta.anchored_residual_v3.data import prepare_anchored_group
    from dta.hybrid_cached_training_v2 import _metrics
    prepare_fn = prepare_anchored_group if prepare_fn is None else prepare_fn
    was_training = model.training
    model.eval()
    keys, labels, predictions, teachers = [], [], [], []
    try:
        with _autocast(device, amp_enabled):
            for group in loader:
                group_keys, gerna_inputs, packed, targets, teacher = prepare_fn(group, device)
                output = model(gerna_inputs, packed, teacher)
                prediction = output["prediction"].reshape(-1).float()
                if prediction.shape != targets.reshape(-1).shape:
                    raise RuntimeError("anchored prediction and label sizes differ")
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("anchored prediction is nonfinite")
                keys.extend(str(key) for key in group_keys)
                labels.extend(targets.reshape(-1).float().cpu().tolist())
                predictions.extend(prediction.cpu().tolist())
                teachers.extend(teacher.reshape(-1).float().cpu().tolist())
    finally:
        model.train(was_training)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("anchored evaluation keys must be nonempty and unique")
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    teachers = np.asarray(teachers, dtype=np.float64)
    return {"keys": keys, "labels": labels, "predictions": predictions,
            "teacher_predictions": teachers, "metrics": _metrics(labels, predictions),
            "teacher_metrics": _metrics(labels, teachers)}


def fit_guarded_stages(model, train_loader, valid_loader, device, *, stage_epochs,
                       patience=2, prepare_fn=None, amp_enabled=False, on_accept=None):
    import numpy as np
    baseline = evaluate_anchored(model, valid_loader, device,
                                 prepare_fn=prepare_fn, amp_enabled=amp_enabled)
    if not np.array_equal(baseline["predictions"], baseline["teacher_predictions"]):
        raise RuntimeError("zero-init residual does not exactly reproduce teacher")
    guard = ValidationGuard(model, baseline["metrics"], variant="cached_teacher")
    history = [{"stage": "baseline", "epoch": -1,
                "metrics": dict(baseline["metrics"]), "accepted": True}]
    if on_accept is not None:
        on_accept(model, baseline, "cached_teacher", -1)
    for stage in ("residual", "fusion", "selective_e2e"):
        epochs = int(stage_epochs.get(stage, 0))
        if epochs <= 0:
            continue
        guard.restore(model)
        optimizer, stage_metadata = configure_stage(model, stage)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        references = snapshot_trainable_parameters(model)
        stale = 0
        for epoch in range(epochs):
            losses = []
            sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for group in train_loader:
                result = train_anchored_group(
                    model, optimizer, group, device,
                    consistency_weight=STAGE_DEFAULTS[stage]["consistency_weight"],
                    reference_parameters=references, prepare_fn=prepare_fn,
                    amp_enabled=amp_enabled,
                )
                losses.append(result["loss"])
            scheduler.step()
            evaluation = evaluate_anchored(model, valid_loader, device,
                                           prepare_fn=prepare_fn, amp_enabled=amp_enabled)
            accepted = guard.consider(model, evaluation["metrics"], stage)
            stale = 0 if accepted else stale + 1
            history.append({"stage": stage, "epoch": epoch,
                            "train_loss": float(np.mean(losses)),
                            "metrics": dict(evaluation["metrics"]),
                            "accepted": accepted, "stage_metadata": stage_metadata})
            if accepted and on_accept is not None:
                on_accept(model, evaluation, stage, epoch)
            if stale >= int(patience):
                break
        guard.restore(model)
    return {"baseline": baseline, "history": history, "selected": guard,
            "improved_over_teacher": guard.variant != "cached_teacher"}
