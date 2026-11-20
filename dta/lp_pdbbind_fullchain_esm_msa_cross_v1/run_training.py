"""Matched clean-start full-chain ESM/MSA cross-attention training; no test."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from data_utils.hybrid_affinity_dataset import LabelSpreadBatchSampler
from dta.lp_pdbbind_e2e_msa_structure_v3.fusion_training import validation_gate
from dta.lp_pdbbind_e2e_msa_structure_v3.run_fusion_training import (
    DEFAULT_V20_SELECTION, _atomic_torch_save, _load_v20_baseline, _manifest_rows,
)
from dta.lp_pdbbind_windowed_msa_v1.full_pipeline import (
    MODES, _input_paths, _read_json, _sealed_preflight, _write_json_new,
)
from dta.msa_interface_pair_pooling_v18 import training as v18_training
from dta.msa_interface_pair_pooling_v18.training import TrainingConfig
from dta.msa_pocket_expert_v9.dataset import move_batch_to_device
from dta.msa_transformer_smoke_v1.msa import sha256_file
from dta.rank_fusion_v4.metrics import affinity_metrics

from . import VERSION
from .dataset import FullChainESMMSACollator, FullChainESMMSADataset
from .model import FullChainESMMSACrossV1


def _fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _config(args):
    # Architecture dimensions mirror v3 for interpretability, but absolutely
    # no v3 or old pocket-MSA parameters are loaded into this model.
    return TrainingConfig(
        pretrained_checkpoint="none", device=args.device, seed=args.seed,
        batch_size=args.batch_size, evaluation_batch_size=args.evaluation_batch_size,
        num_workers=args.num_workers, amp_dtype="bfloat16",
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        maximum_epochs=args.maximum_epochs, model_width=512, layers=8, heads=8,
        dropout=getattr(args, "dropout", 0.10),
        latent_width=1024, msa_width=192, msa_layers=4, msa_heads=6,
        pair_factor=8, pair_width=96, interface_blocks=2,
        pair_token_width=256, pair_token_heads=8, pair_token_layers=2,
        pair_token_count=96, inference_subsets=1,
    )


def _dataset(paths, cache_root, mode, weight_sha, split,
             dataset_type=FullChainESMMSADataset):
    return dataset_type(
        paths["manifest_root"] / (split + "_manifest.json"),
        paths["complex_root"] / split, cache_root / mode, mode,
        cache_root / "source_manifest.json", cache_root / "targets.json", weight_sha,
    )


def _loader(dataset, config, training, seed=0, label_std=1.0,
            collator_type=FullChainESMMSACollator):
    common = dict(
        dataset=dataset, collate_fn=collator_type(),
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    if config.num_workers:
        common["prefetch_factor"] = 2
    if training:
        sampler = LabelSpreadBatchSampler(
            dataset.labels, config.batch_size, seed,
            config.minimum_rank_gap * label_std,
        )
        return torch.utils.data.DataLoader(batch_sampler=sampler, **common), sampler
    return torch.utils.data.DataLoader(
        batch_size=config.evaluation_batch_size, shuffle=False, **common,
    ), None


def predict(model, dataset, config, label_mean, label_std, loader=None,
            collator_type=FullChainESMMSACollator):
    model.eval()
    if loader is None:
        loader, _ = _loader(dataset, config, False, collator_type=collator_type)
    device = torch.device(config.device)
    dtype = v18_training._autocast_dtype(config.amp_dtype)
    keys, labels, predictions = [], [], []
    with torch.no_grad():
        for batch in loader:
            keys.extend(batch["keys"])
            labels.append(batch["labels"].numpy().astype(np.float64))
            moved = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=dtype):
                output = model(moved)
            predictions.append(
                output.prediction.float().cpu().numpy().astype(np.float64)
                * label_std + label_mean
            )
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    if len(keys) != len(dataset) or not np.isfinite(predictions).all():
        raise ValueError("full-chain predictions are incomplete or nonfinite")
    return dict(keys=tuple(keys), labels=labels, predictions=predictions,
                metrics=affinity_metrics(labels, predictions))


def _score(metrics, baseline):
    gate = validation_gate(metrics, baseline)
    return (int(gate["passed"]), metrics["CI"], -metrics["RMSE"],
            -metrics["MAE"], metrics["Pearson"], metrics["Spearman"])


def _selection_score(metrics, baseline, selection_policy=None):
    """The default is the historical V20 gate; new policies are opt-in only."""
    if selection_policy is None:
        return _score(metrics, baseline)
    if selection_policy != "ci_lexicographic_v1":
        raise ValueError("unknown validation selection policy")
    return (metrics["CI"], -metrics["RMSE"], -metrics["MAE"],
            metrics["Pearson"], metrics["Spearman"])


def _selection_report(metrics, baseline, selection_policy=None):
    if selection_policy is None:
        gate = validation_gate(metrics, baseline)
        return dict(status="validation_passed" if gate["passed"] else "validation_failed",
                    validation_gate=gate, validation_selected=bool(gate["passed"]))
    _selection_score(metrics, baseline, selection_policy)
    return dict(status="training_complete", selection_policy=selection_policy,
                validation_selection_performed=True, v20_gate_used=False)


def _fit_arm(mode, initial_state, config, train_data, valid_data, args,
             label_mean, label_std, baseline, output, lineage,
             model_type=FullChainESMMSACrossV1,
             collator_type=FullChainESMMSACollator, version=VERSION,
             protocol_fields=None, loss_function=None, training_step=None,
             selection_policy=None):
    protocol_fields = dict(protocol_fields or {})
    if selection_policy not in (None, "ci_lexicographic_v1"):
        raise ValueError("unknown validation selection policy")
    started = time.monotonic() if loss_function is not None else None
    v18_training._seed_everything(args.seed)
    device = torch.device(args.device)
    model = model_type(**config.model_kwargs())
    model.load_state_dict(initial_state, strict=True)
    model.freeze_fusion_head()
    model.to(device)
    ema = deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader, sampler = _loader(train_data, config, True, args.seed, label_std,
                              collator_type=collator_type)
    valid_loader, _ = _loader(valid_data, config, False,
                              collator_type=collator_type)
    total_steps = max(1, len(loader) * args.maximum_epochs)
    expected_steps = protocol_fields.get("planned_optimizer_steps")
    if expected_steps is not None and total_steps != int(expected_steps):
        raise ValueError("planned optimizer steps differ from sealed protocol")
    print(json.dumps({
        "event": "fullchain_cross_arm_start", "mode": mode,
        "train_count": len(train_data), "valid_count": len(valid_data),
        "batch_size": config.batch_size, "maximum_epochs": args.maximum_epochs,
        "planned_optimizer_steps": total_steps,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
    }, sort_keys=True), flush=True)
    warmup = max(1, int(total_steps * 0.08))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.475 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    amp_dtype = v18_training._autocast_dtype(config.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    history, best, steps = [], None, 0
    if training_step is not None and (loss_function is None or scaler.is_enabled()):
        raise ValueError("custom training step requires an unscaled explicit loss")
    for epoch in range(args.maximum_epochs):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        component_sums, component_counts = {}, {}
        step_sums = {"loss_perturbed": 0.0, "first_gradient_norm": 0.0,
                     "perturbation_norm": 0.0}
        zero_gradient_count = 0
        for batch in loader:
            moved = v18_training._mask_interface_for_training(
                move_batch_to_device(batch, device), config.interface_mask_probability,
            )
            normalized = (moved["labels"] - label_mean) / label_std
            optimizer.zero_grad(set_to_none=True)
            if training_step is None:
                with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                    output_value = model(moved)
                    selected_loss = loss_function or v18_training.msa_interface_loss
                    loss, _components = selected_loss(
                        output_value, normalized, moved, config,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite clean-start full-chain loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                step_info = training_step(
                    model, optimizer, moved, normalized, config, loss_function,
                    amp_dtype=amp_dtype, audit_restore=steps < 2,
                )
                loss, _components = step_info["loss_clean"], step_info["components"]
            norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                config.gradient_clip,
            )
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite clean-start full-chain gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            v18_training._update_ema(ema, model, config.ema_decay)
            loss_sum += float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            if training_step is not None:
                for name in step_sums:
                    step_sums[name] += step_info[name]
                zero_gradient_count += int(step_info["zero_gradient_norm"])
                if steps < 2:
                    print(json.dumps({"event": "sam_update_audit", "step": steps + 1,
                                      "loss_clean": step_info["loss_clean"],
                                      "loss_perturbed": step_info["loss_perturbed"],
                                      "first_gradient_norm": step_info["first_gradient_norm"],
                                      "perturbation_norm": step_info["perturbation_norm"],
                                      "restored_exact": step_info["restored_exact"]},
                                     sort_keys=True), flush=True)
            if loss_function is not None:
                for name, value in _components.items():
                    if value is None:
                        component_counts.setdefault(name, 0)
                        continue
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        component_counts.setdefault(name, 0)
                        continue
                    component_sums[name] = component_sums.get(name, 0.0) + numeric
                    component_counts[name] = component_counts.get(name, 0) + 1
            steps += 1
        candidates = []
        for variant, candidate in (("raw", model), ("ema", ema)):
            result = predict(candidate, valid_data, config, label_mean, label_std,
                             loader=valid_loader)
            candidates.append((_selection_score(result["metrics"], baseline, selection_policy),
                               variant, candidate, result))
        score, variant, selected, result = max(candidates, key=lambda entry: entry[0])
        row = dict(epoch=epoch, mode=mode, selected_variant=variant,
                   train_loss=loss_sum / len(loader), validation_metrics=result["metrics"])
        if training_step is not None:
            row["loss_clean"] = row["train_loss"]
            row["loss_perturbed"] = step_sums["loss_perturbed"] / len(loader)
            row["sam_diagnostics"] = {
                "first_gradient_norm_mean": step_sums["first_gradient_norm"] / len(loader),
                "perturbation_norm_mean": step_sums["perturbation_norm"] / len(loader),
                "zero_gradient_norm_count": zero_gradient_count,
                "backward_passes": 2 * len(loader),
                "base_optimizer_updates": len(loader),
            }
        if loss_function is not None:
            row["train_components"] = {
                "means": {name: component_sums[name] / count if count else None
                          for name, count in component_counts.items()},
                "valid_counts": component_counts,
                "batch_count": len(loader),
            }
        history.append(row)
        print(json.dumps({"event": "fullchain_cross_epoch", **row}, sort_keys=True), flush=True)
        if best is None or score > best["score"]:
            checkpoint = dict(
                version=version, mode=mode, config=asdict(config),
                model_kwargs=config.model_kwargs(),
                model_state={key: value.detach().cpu().clone()
                             for key, value in selected.state_dict().items()},
                label_mean=label_mean, label_std=label_std,
                selected_epoch=epoch, selected_variant=variant,
                initialization="random_clean_start_no_stage1_weights",
                old_pocket_msa_weights_transferred=False,
                pocket_msa_input=False, test_manifest_read=False,
                **lineage, **protocol_fields,
            )
            if training_step is not None:
                checkpoint["sam_optimizer_steps_at_save"] = steps
                checkpoint["sam_backward_passes_at_save"] = 2 * steps
            if selection_policy is not None:
                checkpoint["selection_policy"] = selection_policy
            _atomic_torch_save(checkpoint, output / "best.pt")
            best = dict(score=score, result=result, epoch=epoch, variant=variant)
    _write_json_new(output / "history.json", history)
    if expected_steps is not None and steps != int(expected_steps):
        raise ValueError("actual optimizer steps differ from sealed protocol")
    checkpoint_path = output / "best.pt"
    reloaded = torch.load(str(checkpoint_path), map_location="cpu")
    restored = model_type(**config.model_kwargs())
    restored.load_state_dict(reloaded["model_state"], strict=True)
    restored.to(device)
    after = predict(restored, valid_data, config, label_mean, label_std,
                    loader=valid_loader)
    before = best["result"]
    if tuple(after["keys"]) != tuple(before["keys"]) or not np.array_equal(
        after["labels"], before["labels"]
    ):
        raise ValueError("reloaded validation identities differ")
    delta = float(np.max(np.abs(after["predictions"] - before["predictions"])))
    if delta > 1e-6:
        raise ValueError("reloaded predictions differ")
    np.savez_compressed(
        str(output / "valid_predictions.npz"),
        keys=np.asarray(after["keys"]), labels=after["labels"],
        predictions=after["predictions"],
    )
    result = dict(
        **_selection_report(after["metrics"], baseline, selection_policy),
        mode=mode, validation_metrics=after["metrics"], selected_epoch=best["epoch"],
        selected_variant=best["variant"], optimizer_steps=steps,
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_reload_maximum_prediction_delta=delta,
        valid_predictions=str(output / "valid_predictions.npz"),
        valid_predictions_sha256=sha256_file(output / "valid_predictions.npz"),
        initialization="random_clean_start_no_stage1_weights",
        old_pocket_msa_weights_transferred=False, pocket_msa_input=False,
        test_manifest_read=False, test_used_for_selection=False,
        **lineage, **protocol_fields,
    )
    if training_step is not None:
        result["training_backward_passes"] = 2 * steps
        result["sam_base_optimizer_updates"] = steps
    if loss_function is not None:
        result["runtime"] = {
            "wall_seconds": time.monotonic() - started,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "visible_physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else None,
        }
    return result


def run(args, model_type=FullChainESMMSACrossV1,
        dataset_type=FullChainESMMSADataset,
        collator_type=FullChainESMMSACollator, version=VERSION,
        protocol_fields=None):
    protocol_fields = dict(protocol_fields or {})
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("an operator-selected CUDA device is required")
    paths = _input_paths(args.workspace)
    cache_root = Path(args.cache_root).expanduser().resolve()
    preflight, targets, _sources = _sealed_preflight(paths, cache_root)
    audit = _read_json(cache_root / "coverage_audit.json")
    if (audit.get("status") != "full_cache_coverage_passed"
            or audit.get("source_manifest_sha256") != preflight["source_manifest_sha256"]
            or audit.get("test_manifest_read") is not False):
        raise ValueError("train/valid full-chain cache audit is not sealed")
    weights = set()
    for mode in MODES:
        metadata = _read_json(cache_root / mode / "metadata.json")
        if (metadata.get("status") != "built"
                or metadata.get("target_count") != len(targets)
                or metadata.get("source_manifest_sha256") != preflight["source_manifest_sha256"]
                or audit.get("modes", {}).get(mode, {}).get("weights_sha256")
                != metadata.get("weights_sha256")):
            raise ValueError("full-chain cache mode is incomplete")
        weights.add(metadata["weights_sha256"])
    if len(weights) != 1:
        raise ValueError("paired modes have different encoder weights")
    weight_sha = next(iter(weights))
    smoke_path = Path(args.smoke_result).expanduser().resolve()
    smoke = _read_json(smoke_path)
    if (smoke.get("version") != version
            or smoke.get("status") != "real_gpu_smoke_passed"
            or smoke.get("diagnostic_only") is not True
            or smoke.get("batch_size") != args.batch_size
            or smoke.get("seed") != args.seed
            or smoke.get("initialization") != "random_clean_start_no_stage1_weights"
            or smoke.get("old_pocket_msa_weights_transferred") is not False
            or smoke.get("old_pocket_msa_batch_keys") != []
            or smoke.get("fullchain_esm_projection_updated") is not True
            or smoke.get("reload_maximum_prediction_delta", 1.0) > 1e-6
            or smoke.get("source_manifest_sha256") != preflight["source_manifest_sha256"]
            or smoke.get("encoder_weights_sha256") != weight_sha
            or smoke.get("test_manifest_read") is not False):
        raise ValueError("matching clean-start GPU smoke is required")
    if any(smoke.get(key) != value for key, value in protocol_fields.items()):
        raise ValueError("GPU smoke does not match the requested input protocol")
    _, valid_rows = _manifest_rows(paths["manifest_root"] / "valid_manifest.json", "valid")
    baseline_path, baseline, _sources, _baseline = _load_v20_baseline(
        args.v20_selection, valid_rows,
    )
    config = _config(args)
    train_data = _dataset(paths, cache_root, MODES[0], weight_sha, "train",
                          dataset_type=dataset_type)
    label_mean = float(train_data.labels.mean())
    label_std = float(train_data.labels.std())
    if not np.isfinite(label_mean) or not np.isfinite(label_std) or label_std <= 0:
        raise ValueError("train labels cannot be normalized")
    v18_training._seed_everything(args.seed)
    initial = model_type(**config.model_kwargs())
    initial.freeze_fusion_head()
    initial_fingerprint = _fingerprint(initial)
    if smoke.get("initial_weight_fingerprint") != initial_fingerprint:
        raise ValueError("clean-start initialization differs from GPU smoke")
    initial_state = {key: value.detach().cpu().clone()
                     for key, value in initial.state_dict().items()}
    del initial
    lineage = dict(
        source_manifest_sha256=preflight["source_manifest_sha256"],
        encoder_weights_sha256=weight_sha,
        train_manifest_sha256=preflight["train_manifest_sha256"],
        valid_manifest_sha256=preflight["valid_manifest_sha256"],
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("full-chain cross output already exists")
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for mode in MODES:
        arm = output / mode
        arm.mkdir(exist_ok=False)
        train_data = _dataset(paths, cache_root, mode, weight_sha, "train",
                              dataset_type=dataset_type)
        valid_data = _dataset(paths, cache_root, mode, weight_sha, "valid",
                              dataset_type=dataset_type)
        result = _fit_arm(mode, initial_state, config, train_data, valid_data,
                          args, label_mean, label_std, baseline, arm, lineage,
                          model_type=model_type, collator_type=collator_type,
                          version=version, protocol_fields=protocol_fields)
        _write_json_new(arm / "selection.json", result)
        results[mode] = result
    if results[MODES[0]]["optimizer_steps"] != results[MODES[1]]["optimizer_steps"]:
        raise ValueError("paired arms used different optimizer budgets")
    summary = dict(
        version=version, status="paired_training_complete", arms=results,
        matched_hyperparameters=dict(seed=args.seed, batch_size=args.batch_size,
                                     learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay,
                                     maximum_epochs=args.maximum_epochs),
        initialization="random_clean_start_no_stage1_weights",
        initial_weight_fingerprint=initial_fingerprint,
        old_pocket_msa_weights_transferred=False,
        v20_baseline_selection=str(baseline_path), v20_baseline_metrics=baseline,
        gpu_smoke_result=str(smoke_path), gpu_smoke_result_sha256=sha256_file(smoke_path),
        test_manifest_read=False, test_used_for_selection=False,
        **lineage, **protocol_fields,
    )
    _write_json_new(output / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--v20-selection", type=Path, default=DEFAULT_V20_SELECTION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-epochs", type=int, default=30)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.batch_size < 2 or args.evaluation_batch_size <= 0
            or args.num_workers < 0 or args.maximum_epochs <= 0
            or args.learning_rate <= 0 or args.weight_decay < 0):
        raise ValueError("invalid clean-start training configuration")
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
