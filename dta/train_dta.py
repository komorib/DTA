import argparse
import contextlib
import json
import os
import pickle
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

from data_utils.dataset import GerNAChunkedDataset, GerNA_dataset, custom_collate_fn
from data_utils.lmdb_dataset import PDBbindLmdbDataset, lmdb_manifest_labels
from dta.model_dta import GerNADTA
from utils.metrics import get_score_metrics

#固定随机种子
def set_random_seeds(seed_value=99, seed_cuda=False):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if seed_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_training_device(cuda_arg):
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            "CUDA GPU is required for this training script, but PyTorch cannot see CUDA. "
            f"CUDA_VISIBLE_DEVICES={visible}. Check that this shell is on a GPU node and that the selected GPU is usable."
        )
    return torch.device(f"cuda:{cuda_arg}")


def resolve_dataset_paths(args):
    if args.train_path and args.valid_path and args.test_path:
        return args.train_path, args.valid_path, args.test_path
    if args.train_path or args.valid_path or args.test_path:
        raise ValueError("train_path, valid_path and test_path must be provided together")
    base_dir = os.path.join(args.data_root, args.dataset, args.split_method)
    manifest_paths = (
        os.path.join(base_dir, "train_manifest.json"),
        os.path.join(base_dir, "valid_manifest.json"),
        os.path.join(base_dir, "test_manifest.json"),
    )
    if all(os.path.exists(path) for path in manifest_paths):
        return manifest_paths
    return (
        os.path.join(base_dir, "train_data.pkl"),
        os.path.join(base_dir, "valid_data.pkl"),
        os.path.join(base_dir, "test_data.pkl"),
    )


def load_json_manifest(dataset_path):
    if not dataset_path.endswith(".json"):
        return None
    with open(dataset_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_for_path(dataset_path, args):
    manifest = load_json_manifest(dataset_path)
    if manifest and manifest.get("dataset_type") == "pdbbind_lmdb":
        return PDBbindLmdbDataset(
            dataset_path,
            protein_3d_mode=args.protein_3d_mode,
            protein_edge_mode=args.protein_edge_mode,
            protein_contact_cutoff=args.protein_contact_cutoff,
            protein_knn_k=args.protein_knn_k,
            protein_embedding_mode=args.protein_embedding_mode,
            expected_protein_embedding_dim=args.expected_protein_embedding_dim,
            ligand_pocket_cutoff=args.ligand_pocket_cutoff,
            disable_protein_2d=args.disable_protein_2d,
            dti_fusion_lmdb_path=getattr(args, "dti_fusion_lmdb_path", None),
            ligand_unimol_lmdb_path=getattr(args, "ligand_unimol_lmdb_path", None),
            ligand_unimol2_lmdb_path=getattr(args, "ligand_unimol2_lmdb_path", None),
            pbcnet_fusion_lmdb_path=getattr(args, "pbcnet_fusion_lmdb_path", None),
            pbcnet_graph_lmdb_path=getattr(args, "pbcnet_graph_lmdb_path", None),
            boltz_interface_lmdb_path=getattr(args, "boltz_interface_lmdb_path", None),
            boltz_pocket_radius=getattr(args, "boltz_pocket_radius", 10.0),
            boltz_min_pocket_residues=getattr(args, "boltz_min_pocket_residues", 16),
            boltz_max_pocket_residues=getattr(args, "boltz_max_pocket_residues", 64),
        )
    if dataset_path.endswith("_manifest.json"):
        return GerNAChunkedDataset(dataset_path)
    return GerNA_dataset(dataset_path)


def load_labels_for_path(dataset_path):
    manifest = load_json_manifest(dataset_path)
    if manifest and manifest.get("dataset_type") == "pdbbind_lmdb":
        return lmdb_manifest_labels(dataset_path)
    if dataset_path.endswith("_manifest.json"):
        with open(dataset_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return [float(sample["label"]) for sample in manifest["samples"]]
    with open(dataset_path, "rb") as handle:
        *_, label = pickle.load(handle)
    return [float(item) for item in label]


def infer_rna_num_tokens(dataset_name, explicit_value=None, protein_3d_mode=None):
    if explicit_value is not None:
        return explicit_value
    if protein_3d_mode == "all_atom":
        return 8
    if protein_3d_mode == "pocket_pharmacophore":
        return 8
    if protein_3d_mode in {"pocket_pharmacophore_esm", "pocket_residue_esm", "ligand_pocket_residue_esm", "full_residue_esm"}:
        return None
    if protein_3d_mode == "atom5":
        return 6
    if protein_3d_mode == "residue":
        return 21
    dataset_name = dataset_name.lower()
    if dataset_name.startswith("pdbbind") and "atom5" in dataset_name:
        return 6
    if dataset_name.startswith("pdbbind"):
        return 21
    return None


def infer_rna_3d_input_dim(protein_3d_mode):
    if protein_3d_mode == "pocket_pharmacophore_esm":
        return 648
    if protein_3d_mode in {"pocket_residue_esm", "ligand_pocket_residue_esm", "full_residue_esm"}:
        return 640
    return None


def resolve_protein_3d_mode(args, train_path):
    if args.protein_3d_mode is not None:
        return args.protein_3d_mode
    manifest = load_json_manifest(train_path)
    if manifest and manifest.get("dataset_type") == "pdbbind_lmdb":
        return manifest.get("protein_3d_mode") or "residue"
    return None


def unpack_batch_for_device(batch, device):
    # custom_collate_fn returns a variable-length tuple depending on enabled optional caches.
    # Base 13 tensors include RNA_repre [B,L,640], Mol_feats [B,A], Mol_coors [B,A,3].
    # Optional tensors are DTI protein [B,L,256], DTI global [B,512], old Uni-Mol [B,A,512], Uni-Mol2 [B,A,768].
    # Keep the mapping explicit so each fusion path receives the intended tensors and disabled paths stay as None.
    dti_protein_tokens = None
    dti_protein_mask = None
    dti_global = None
    ligand_unimol_tokens = None
    ligand_unimol_mask = None
    ligand_unimol2_tokens = None
    ligand_unimol2_mask = None
    pbcnet_global = None
    pbcnet_graph = None
    boltz_interface = None

    if len(batch) > 13 and isinstance(batch[-1], dict) and "pocket_emb" in batch[-1] and "ligand_emb" in batch[-1]:
        boltz_interface = batch[-1]
        batch = batch[:-1]

    if len(batch) == 13:
        batch = batch[:13]
    elif len(batch) == 15:
        ligand_unimol_tokens = batch[13]
        ligand_unimol_mask = batch[14]
        batch = batch[:13]
    elif len(batch) == 16:
        dti_protein_tokens = batch[13]
        dti_protein_mask = batch[14]
        dti_global = batch[15]
        batch = batch[:13]
    elif len(batch) == 18:
        dti_protein_tokens = batch[13]
        dti_protein_mask = batch[14]
        dti_global = batch[15]
        ligand_unimol_tokens = batch[16]
        ligand_unimol_mask = batch[17]
        batch = batch[:13]
    elif len(batch) == 20:
        # Full optional layout without PBCNet: DTI fields, old ligand Uni-Mol 2D fields, then Uni-Mol2 3D fields.
        dti_protein_tokens = batch[13]
        dti_protein_mask = batch[14]
        dti_global = batch[15]
        ligand_unimol_tokens = batch[16]
        ligand_unimol_mask = batch[17]
        ligand_unimol2_tokens = batch[18]
        ligand_unimol2_mask = batch[19]
        batch = batch[:13]
    elif len(batch) == 21:
        # Full optional layout with PBCNet global at the end. Disabled optional branches are explicit None.
        dti_protein_tokens = batch[13]
        dti_protein_mask = batch[14]
        dti_global = batch[15]
        ligand_unimol_tokens = batch[16]
        ligand_unimol_mask = batch[17]
        ligand_unimol2_tokens = batch[18]
        ligand_unimol2_mask = batch[19]
        pbcnet_global = batch[20]
        batch = batch[:13]
    elif len(batch) == 22:
        # Full optional layout with PBCNet global and trainable PBCNet graph at the end.
        dti_protein_tokens = batch[13]
        dti_protein_mask = batch[14]
        dti_global = batch[15]
        ligand_unimol_tokens = batch[16]
        ligand_unimol_mask = batch[17]
        ligand_unimol2_tokens = batch[18]
        ligand_unimol2_mask = batch[19]
        pbcnet_global = batch[20]
        pbcnet_graph = batch[21]
        batch = batch[:13]
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    (
        batch_RNA_repre,
        batch_seq_mask,
        batch_Mol_Graph,
        batch_RNA_Graph,
        batch_RNA_feats,
        batch_RNA_C4_coors,
        batch_RNA_coors,
        batch_RNA_mask,
        batch_Mol_feats,
        batch_Mol_coors,
        batch_Mol_mask,
        batch_Mol_LAS,
        batch_label,
    ) = batch

    def move_optional(value):
        return None if value is None else value.to(device)

    def move_boltz_interface(value):
        return None if value is None else {key: tensor.to(device) for key, tensor in value.items()}

    return (
        batch_RNA_repre.to(device),
        batch_seq_mask.to(device),
        batch_Mol_Graph.to(device),
        batch_RNA_Graph.to(device),
        batch_RNA_feats.to(device),
        batch_RNA_C4_coors.to(device),
        batch_RNA_coors.to(device),
        batch_RNA_mask.to(device),
        batch_Mol_feats.to(device),
        batch_Mol_coors.to(device),
        batch_Mol_mask.to(device),
        batch_Mol_LAS.to(device),
        batch_label.to(device),
        move_optional(dti_protein_tokens),
        move_optional(dti_protein_mask),
        move_optional(dti_global),
        move_optional(ligand_unimol_tokens),
        move_optional(ligand_unimol_mask),
        move_optional(ligand_unimol2_tokens),
        move_optional(ligand_unimol2_mask),
        move_optional(pbcnet_global),
        pbcnet_graph,
        move_boltz_interface(boltz_interface),
    )


def regression_loss(pred, target, loss_type="mse"):
    pred = pred.reshape(-1)
    target = target.float().reshape(-1)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError("loss_type must be one of: mse, smooth_l1")


def update_regression_early_stop(current_metric, best_metric, bad_epochs, patience, min_delta=0.0):
    improved = current_metric < best_metric - min_delta
    if improved:
        return current_metric, 0, False, True
    bad_epochs += 1
    if patience <= 0:
        # patience <= 0 disables stopping but still records consecutive non-improving epochs.
        return best_metric, bad_epochs, False, False
    return best_metric, bad_epochs, bad_epochs >= patience, False


LOWER_IS_BETTER_REGRESSION_METRICS = {"RMSE", "MAE"}


def is_better_regression_checkpoint(current_metrics, best_metrics, select_metric="RMSE", min_delta=0.0):
    if best_metrics is None:
        return True
    if select_metric not in REGRESSION_METRIC_NAMES:
        raise ValueError(f"select_metric must be one of: {', '.join(REGRESSION_METRIC_NAMES)}")
    current_value = float(current_metrics.get(select_metric, float("nan")))
    best_value = float(best_metrics.get(select_metric, float("nan")))
    if not np.isfinite(current_value):
        return False
    if not np.isfinite(best_value):
        return True

    delta = float(min_delta or 0.0)
    if select_metric == "CI":
        if current_value > best_value + delta:
            return True
        if abs(current_value - best_value) > delta:
            return False
        for metric_name in ("RMSE", "MAE"):
            current_tiebreak = float(current_metrics.get(metric_name, float("inf")))
            best_tiebreak = float(best_metrics.get(metric_name, float("inf")))
            if not np.isfinite(current_tiebreak):
                current_tiebreak = float("inf")
            if not np.isfinite(best_tiebreak):
                best_tiebreak = float("inf")
            if current_tiebreak < best_tiebreak:
                return True
            if current_tiebreak > best_tiebreak:
                return False
        if "Epoch" not in current_metrics or "Epoch" not in best_metrics:
            return False
        return int(current_metrics["Epoch"]) < int(best_metrics["Epoch"])

    lower_is_better = select_metric in LOWER_IS_BETTER_REGRESSION_METRICS
    if lower_is_better:
        if current_value < best_value - delta:
            return True
        if abs(current_value - best_value) <= delta:
            return float(current_metrics.get("CI", -float("inf"))) > float(best_metrics.get("CI", -float("inf"))) + delta
        return False

    if current_value > best_value + delta:
        return True
    if abs(current_value - best_value) <= delta:
        return float(current_metrics.get("RMSE", float("inf"))) < float(best_metrics.get("RMSE", float("inf"))) - delta
    return False


def update_regression_checkpoint_state(current_metrics, best_metrics, bad_epochs, patience, select_metric="RMSE", min_delta=0.0):
    improved = is_better_regression_checkpoint(current_metrics, best_metrics, select_metric, min_delta)
    if improved:
        return metrics_to_float_dict(current_metrics), 0, False, True
    bad_epochs += 1
    if patience <= 0:
        return best_metrics, bad_epochs, False, False
    return best_metrics, bad_epochs, bad_epochs >= patience, False


def pairwise_rank_loss(pred, target, min_gap=0.10, temperature=0.5):
    pred = pred.reshape(-1)
    target = target.float().reshape(-1)
    if pred.numel() < 2:
        return pred.new_zeros(())
    pred_delta = pred[:, None] - pred[None, :]
    target_delta = target[:, None] - target[None, :]
    valid = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
    valid = valid & (target_delta.abs() >= float(min_gap))
    if not bool(valid.any()):
        return pred.new_zeros(())
    direction = target_delta.sign()
    temp = max(float(temperature), 1e-6)
    logits = direction[valid] * pred_delta[valid] / temp
    return F.softplus(-logits).mean()


def build_dta_optimizer(net, args):
    pbcnet_lr = float(getattr(args, "pbcnet_lr", 0.0) or 0.0)
    weight_decay = float(getattr(args, "weight_decay", 0.0) or 0.0)
    use_pbcnet_group = (
        bool(getattr(args, "pbcnet_trainable", False))
        and pbcnet_lr > 0.0
        and hasattr(net, "pbcnet_encoder")
    )
    if not use_pbcnet_group:
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)

    pbcnet_params = list(net.pbcnet_encoder.parameters())
    pbcnet_param_ids = {id(param) for param in pbcnet_params}
    main_params = [param for param in net.parameters() if id(param) not in pbcnet_param_ids]
    param_groups = []
    if main_params:
        param_groups.append({"name": "main", "params": main_params, "lr": args.lr})
    if pbcnet_params:
        param_groups.append({"name": "pbcnet_encoder", "params": pbcnet_params, "lr": pbcnet_lr})
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def optimizer_group_summaries(optimizer):
    summaries = []
    for index, group in enumerate(optimizer.param_groups):
        params = list(group["params"])
        summaries.append({
            "index": index,
            "name": group.get("name", f"group_{index}"),
            "lr": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "n_params": int(sum(param.numel() for param in params)),
        })
    return summaries


def set_pbcnet_trainable_for_epoch(net, epoch, freeze_pbcnet_epochs):
    if not hasattr(net, "pbcnet_encoder"):
        return True
    freeze_epochs = int(freeze_pbcnet_epochs or 0)
    trainable = int(epoch) >= freeze_epochs
    for param in net.pbcnet_encoder.parameters():
        param.requires_grad = trainable
    return trainable


def evaluate(net, data_loader, device):
    net.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for batch in data_loader:
            (
                batch_RNA_repre,
                batch_seq_mask,
                batch_Mol_Graph,
                batch_RNA_Graph,
                batch_RNA_feats,
                batch_RNA_C4_coors,
                batch_RNA_coors,
                batch_RNA_mask,
                batch_Mol_feats,
                batch_Mol_coors,
                batch_Mol_mask,
                batch_Mol_LAS,
                batch_label,
                dti_protein_tokens,
                dti_protein_mask,
                dti_global,
                ligand_unimol_tokens,
                ligand_unimol_mask,
                ligand_unimol2_tokens,
                ligand_unimol2_mask,
                pbcnet_global,
                pbcnet_graph,
                boltz_interface,
            ) = unpack_batch_for_device(batch, device)

            pred, _ = net(
                batch_RNA_repre,
                batch_seq_mask,
                batch_RNA_Graph,
                batch_Mol_Graph,
                batch_RNA_feats,
                batch_RNA_C4_coors,
                batch_RNA_coors,
                batch_RNA_mask,
                batch_Mol_feats,
                batch_Mol_coors,
                batch_Mol_mask,
                batch_Mol_LAS,
                dti_protein_tokens,
                dti_protein_mask,
                dti_global,
                ligand_unimol_tokens,
                ligand_unimol_mask,
                ligand_unimol2_tokens,
                ligand_unimol2_mask,
                pbcnet_global,
                pbcnet_graph,
                boltz_interface,
            )
            flat_pred = pred.reshape(-1).detach()
            finite_mask = torch.isfinite(flat_pred)
            if not bool(finite_mask.all()):
                print("warning_eval_nonfinite_pred batch", len(pred_list), "nonfinite", int((~finite_mask).sum().item()), "total", int(flat_pred.numel()))
            finite_pred = flat_pred[finite_mask]
            if finite_pred.numel() and float(finite_pred.abs().max().cpu()) > 100.0:
                print("warning_eval_large_pred batch", len(pred_list), "max_abs", round(float(finite_pred.abs().max().cpu()), 6))
            pred_list.extend(flat_pred.cpu().numpy().tolist())
            label_list.extend(batch_label.reshape(-1).detach().cpu().numpy().tolist())
    rmse, mae, pearson, spearman, ci = get_score_metrics(pred_list, label_list)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
        "CI": ci,
    }, np.asarray(label_list), np.asarray(pred_list)


REGRESSION_METRIC_NAMES = ["RMSE", "MAE", "Pearson", "Spearman", "CI"]
METRICS_HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "learning_rate",
    "valid_RMSE",
    "valid_MAE",
    "valid_Pearson",
    "valid_Spearman",
    "valid_CI",
    "test_RMSE",
    "test_MAE",
    "test_Pearson",
    "test_Spearman",
    "test_CI",
    "selection_metric",
    "best_selection_value",
    "best_valid_rmse",
    "bad_epochs",
    "improved",
)


def metrics_to_float_dict(metrics):
    result = {name: float(metrics[name]) for name in REGRESSION_METRIC_NAMES}
    if "Epoch" in metrics:
        result["Epoch"] = int(metrics["Epoch"])
    return result


def build_best_metrics_payload(
    epoch,
    train_loss,
    learning_rate,
    valid_metrics,
    test_metrics,
    checkpoint_path,
    selection_metric="RMSE",
    prediction_paths=None,
):
    payload = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "learning_rate": float(learning_rate),
        "checkpoint_path": checkpoint_path,
        "selection_metric": selection_metric,
        "selection_value": float(valid_metrics[selection_metric]),
        "valid": metrics_to_float_dict(valid_metrics),
        "test": metrics_to_float_dict(test_metrics),
    }
    if prediction_paths:
        payload["prediction_paths"] = dict(prediction_paths)
    return payload


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_prediction_file(path, labels, preds):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    if labels.shape[0] != preds.shape[0]:
        raise ValueError(f"labels and preds must have the same length: {labels.shape[0]} != {preds.shape[0]}")
    np.save(path, np.stack([preds, labels], axis=1))


def print_metrics(prefix, n_samples, metrics):
    values = " ".join(f"{name} {round(float(metrics[name]), 6)}" for name in REGRESSION_METRIC_NAMES)
    print(prefix, n_samples, values)


TRAINING_ARTIFACT_NAMES = ("Model_dta_best.pth", "Model_dta_last.pth", "best_metrics.json", "valid_pred.npy", "test_pred.npy")


class TeeStream:
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

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def training_log_path(model_output_path):
    return os.path.join(model_output_path, "train.log")


def training_checkpoint_path(model_output_path):
    return os.path.join(model_output_path, "training_checkpoint.pth")


def metrics_history_path(model_output_path):
    return os.path.join(model_output_path, "metrics_history.csv")


def format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def append_metrics_history(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8") as handle:
        if write_header:
            handle.write(",".join(METRICS_HISTORY_FIELDS) + "\n")
        values = [format_csv_value(row.get(field)) for field in METRICS_HISTORY_FIELDS]
        handle.write(",".join(values) + "\n")


def build_metrics_history_row(
    epoch,
    train_loss,
    learning_rate,
    valid_metrics,
    test_metrics,
    best_rmse,
    bad_epochs,
    improved,
    selection_metric="RMSE",
    best_selection_value=None,
):
    row = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "learning_rate": float(learning_rate),
        "selection_metric": selection_metric,
        "best_selection_value": None if best_selection_value is None else float(best_selection_value),
        "best_valid_rmse": float(best_rmse),
        "bad_epochs": int(bad_epochs),
        "improved": bool(improved),
    }
    for name in REGRESSION_METRIC_NAMES:
        row[f"valid_{name}"] = float(valid_metrics[name])
        row[f"test_{name}"] = None if test_metrics is None else float(test_metrics[name])
    return row


def build_training_checkpoint_payload(
    epoch,
    best_rmse,
    bad_epochs,
    model_state,
    optimizer_state,
    scheduler_state,
    args,
    best_valid_metrics=None,
    selection_metric="RMSE",
):
    return {
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "best_rmse": float(best_rmse),
        "best_valid_metrics": None if best_valid_metrics is None else metrics_to_float_dict(best_valid_metrics),
        "selection_metric": selection_metric,
        "bad_epochs": int(bad_epochs),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "args": dict(args),
    }


def save_training_checkpoint(path, epoch, best_rmse, bad_epochs, net, optimizer, scheduler, args, best_valid_metrics=None):
    payload = build_training_checkpoint_payload(
        epoch=epoch,
        best_rmse=best_rmse,
        bad_epochs=bad_epochs,
        model_state=net.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        args=vars(args),
        best_valid_metrics=best_valid_metrics,
        selection_metric=getattr(args, "select_metric", "RMSE"),
    )
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def existing_training_artifacts(model_output_path):
    return [
        os.path.join(model_output_path, name)
        for name in TRAINING_ARTIFACT_NAMES
        if os.path.exists(os.path.join(model_output_path, name))
    ]


def prepare_model_output_dir(model_output_path, overwrite=False):
    artifacts = existing_training_artifacts(model_output_path)
    if artifacts and not overwrite:
        joined = ", ".join(artifacts)
        raise FileExistsError(
            "Refusing to overwrite existing DTA training artifacts. "
            f"Use a new --model_output_path or pass --overwrite_output_path. Existing: {joined}"
        )
    os.makedirs(model_output_path, exist_ok=True)


def train_and_eval(args):
    device = resolve_training_device(args.cuda)
    set_random_seeds(args.seed, seed_cuda=True)

    if not getattr(args, "_model_output_dir_prepared", False):
        prepare_model_output_dir(
            args.model_output_path,
            overwrite=args.overwrite_output_path or bool(args.resume_training_checkpoint),
        )

    train_path, valid_path, test_path = resolve_dataset_paths(args)
    args.protein_3d_mode = resolve_protein_3d_mode(args, train_path)
    print("train_path:", train_path)
    print("valid_path:", valid_path)
    print("test_path:", test_path)
    print("protein_3d_mode:", args.protein_3d_mode)
    print("protein_embedding_mode:", args.protein_embedding_mode)
    print("ligand_pocket_cutoff:", float(args.ligand_pocket_cutoff))
    print("disable_protein_2d:", bool(args.disable_protein_2d))
    print("only_3d:", bool(args.only_3d))
    print("dti_front_fusion:", args.dti_front_fusion)
    print("dti_late_fusion:", args.dti_late_fusion)
    print("ligand_unimol_fusion:", args.ligand_unimol_fusion)
    print("ligand_unimol2_3d_fusion:", args.ligand_unimol2_3d_fusion)
    print("ligand_unimol2_3d_fusion_stage:", args.ligand_unimol2_3d_fusion_stage)
    print("pbcnet_late_fusion:", args.pbcnet_late_fusion)
    print("boltz_interface_fusion:", bool(args.boltz_interface_fusion))
    print("select_metric:", args.select_metric)
    print("rank_loss_weight:", float(args.rank_loss_weight))
    print("pbcnet_trainable:", bool(args.pbcnet_trainable))
    if args.pbcnet_init_checkpoint:
        print("pbcnet_init_checkpoint:", args.pbcnet_init_checkpoint)
    print("pbcnet_lr:", float(args.pbcnet_lr))
    print("freeze_pbcnet_epochs:", int(args.freeze_pbcnet_epochs))
    if args.ligand_unimol_lmdb_path:
        print("ligand_unimol_lmdb_path:", args.ligand_unimol_lmdb_path)
    if args.ligand_unimol2_lmdb_path:
        print("ligand_unimol2_lmdb_path:", args.ligand_unimol2_lmdb_path)
    if args.dti_fusion_lmdb_path:
        print("dti_fusion_lmdb_path:", args.dti_fusion_lmdb_path)
    if args.pbcnet_fusion_lmdb_path:
        print("pbcnet_fusion_lmdb_path:", args.pbcnet_fusion_lmdb_path)
    if args.pbcnet_graph_lmdb_path:
        print("pbcnet_graph_lmdb_path:", args.pbcnet_graph_lmdb_path)
    if args.boltz_interface_lmdb_path:
        print("boltz_interface_lmdb_path:", args.boltz_interface_lmdb_path)
    train_dataset = load_dataset_for_path(train_path, args)
    valid_dataset = load_dataset_for_path(valid_path, args)
    test_dataset = load_dataset_for_path(test_path, args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    params = [args.GNN_depth, args.DMA_depth, args.hidden_size1, args.hidden_size2]
    rna_num_tokens = infer_rna_num_tokens(args.dataset, args.rna_num_tokens, args.protein_3d_mode)
    rna_3d_input_dim = infer_rna_3d_input_dim(args.protein_3d_mode)
    net = GerNADTA(
        params,
        trigonometry=True,
        mol_graph=not args.only_3d,
        rna_repre=not args.only_3d,
        rna_graph=(not args.disable_protein_2d) and (not args.only_3d),
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=rna_num_tokens,
        rna_3d_input_dim=rna_3d_input_dim,
        disable_protein_2d=args.disable_protein_2d,
        dti_front_fusion=args.dti_front_fusion,
        dti_front_input_dim=args.dti_front_input_dim if args.dti_front_input_dim > 0 else None,
        dti_late_fusion=args.dti_late_fusion,
        dti_late_input_dim=args.dti_late_input_dim if args.dti_late_input_dim > 0 else None,
        dti_late_hidden_dim=args.dti_late_hidden_dim if args.dti_late_hidden_dim > 0 else None,
        ligand_unimol_fusion=args.ligand_unimol_fusion,
        ligand_unimol_input_dim=args.ligand_unimol_input_dim if args.ligand_unimol_input_dim > 0 else None,
        ligand_unimol2_3d_fusion=args.ligand_unimol2_3d_fusion,
        ligand_unimol2_input_dim=args.ligand_unimol2_input_dim if args.ligand_unimol2_input_dim > 0 else None,
        ligand_unimol2_3d_fusion_stage=args.ligand_unimol2_3d_fusion_stage,
        pbcnet_late_fusion=args.pbcnet_late_fusion,
        pbcnet_input_dim=args.pbcnet_input_dim if args.pbcnet_input_dim > 0 else None,
        pbcnet_hidden_dim=args.pbcnet_hidden_dim if args.pbcnet_hidden_dim > 0 else None,
        pbcnet_trainable=args.pbcnet_trainable,
        pbcnet_root=args.pbcnet_root,
        pbcnet_model_hidden_dim=args.pbcnet_model_hidden_dim,
        pbcnet_init_checkpoint=args.pbcnet_init_checkpoint,
        boltz_interface_fusion=args.boltz_interface_fusion,
        boltz_interface_input_dim=args.boltz_interface_input_dim,
        boltz_interface_hidden_dim=args.boltz_interface_hidden_dim,
    ).to(device)
    if args.pbcnet_trainable:
        print("pbcnet_trainable_device:", next(net.pbcnet_encoder.parameters()).device)
    optimizer = build_dta_optimizer(net, args)
    print("optimizer_param_groups:", optimizer_group_summaries(optimizer))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    best_model_path = os.path.join(args.model_output_path, "Model_dta_best.pth")
    last_model_path = os.path.join(args.model_output_path, "Model_dta_last.pth")
    best_metrics_path = os.path.join(args.model_output_path, "best_metrics.json")
    valid_pred_path = os.path.join(args.model_output_path, "valid_pred.npy")
    test_pred_path = os.path.join(args.model_output_path, "test_pred.npy")
    checkpoint_path = training_checkpoint_path(args.model_output_path)
    metrics_history_file = metrics_history_path(args.model_output_path)
    best_rmse = float("inf")
    best_valid_metrics = None
    bad_epochs = 0
    start_epoch = 0

    if args.resume_training_checkpoint:
        checkpoint = torch.load(args.resume_training_checkpoint, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_rmse = float(checkpoint.get("best_rmse", best_rmse))
        best_valid_metrics = checkpoint.get("best_valid_metrics")
        if best_valid_metrics is None and np.isfinite(best_rmse) and args.select_metric == "RMSE":
            best_valid_metrics = {"RMSE": best_rmse, "MAE": float("inf"), "Pearson": -float("inf"), "Spearman": -float("inf"), "CI": -float("inf")}
        bad_epochs = int(checkpoint.get("bad_epochs", bad_epochs))
        start_epoch = int(checkpoint.get("next_epoch", int(checkpoint.get("epoch", -1)) + 1))
        print("resume_training_checkpoint:", args.resume_training_checkpoint)
        print("resume_start_epoch:", start_epoch)
        print("resume_best_rmse:", round(best_rmse, 6), "resume_bad_epochs:", bad_epochs)
        if best_valid_metrics is not None:
            print("resume_best_{}:".format(args.select_metric), round(float(best_valid_metrics.get(args.select_metric, float("nan"))), 6))
    elif args.init_model_checkpoint:
        checkpoint = torch.load(args.init_model_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        net.load_state_dict(state_dict)
        print("init_model_checkpoint:", args.init_model_checkpoint)
        print("init_model_checkpoint_mode: model weights only; optimizer/scheduler/early-stop state reset")

    if start_epoch >= args.epoch:
        print("No training epochs left: start_epoch", start_epoch, ">= epoch", args.epoch)
        return

    last_pbcnet_trainable_state = None
    for epoch in range(start_epoch, args.epoch):
        if args.pbcnet_trainable:
            pbcnet_trainable_now = set_pbcnet_trainable_for_epoch(net, epoch, args.freeze_pbcnet_epochs)
            if pbcnet_trainable_now != last_pbcnet_trainable_state:
                print(
                    "pbcnet_epoch_trainable:",
                    bool(pbcnet_trainable_now),
                    "epoch:",
                    epoch,
                    "freeze_pbcnet_epochs:",
                    int(args.freeze_pbcnet_epochs),
                )
                last_pbcnet_trainable_state = pbcnet_trainable_now
        net.train()
        total_loss = 0.0
        total_count = 0
        for batch_index, batch in enumerate(train_loader):
            (
                batch_RNA_repre,
                batch_seq_mask,
                batch_Mol_Graph,
                batch_RNA_Graph,
                batch_RNA_feats,
                batch_RNA_C4_coors,
                batch_RNA_coors,
                batch_RNA_mask,
                batch_Mol_feats,
                batch_Mol_coors,
                batch_Mol_mask,
                batch_Mol_LAS,
                batch_label,
                dti_protein_tokens,
                dti_protein_mask,
                dti_global,
                ligand_unimol_tokens,
                ligand_unimol_mask,
                ligand_unimol2_tokens,
                ligand_unimol2_mask,
                pbcnet_global,
                pbcnet_graph,
                boltz_interface,
            ) = unpack_batch_for_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            pred, _ = net(
                batch_RNA_repre,
                batch_seq_mask,
                batch_RNA_Graph,
                batch_Mol_Graph,
                batch_RNA_feats,
                batch_RNA_C4_coors,
                batch_RNA_coors,
                batch_RNA_mask,
                batch_Mol_feats,
                batch_Mol_coors,
                batch_Mol_mask,
                batch_Mol_LAS,
                dti_protein_tokens,
                dti_protein_mask,
                dti_global,
                ligand_unimol_tokens,
                ligand_unimol_mask,
                ligand_unimol2_tokens,
                ligand_unimol2_mask,
                pbcnet_global,
                pbcnet_graph,
                boltz_interface,
            )
            loss = regression_loss(pred, batch_label, args.loss)
            if args.rank_loss_weight > 0:
                loss = loss + float(args.rank_loss_weight) * pairwise_rank_loss(
                    pred,
                    batch_label,
                    min_gap=args.rank_min_gap,
                    temperature=args.rank_temperature,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_label)
            total_count += len(batch_label)
            if args.step_log_every and batch_index % args.step_log_every == 0:
                print("epoch", epoch, "batch", batch_index)
        scheduler.step()

        epoch_loss = total_loss / max(total_count, 1)
        print("epoch", epoch, "train_loss", round(epoch_loss, 6), "learning rate:", optimizer.param_groups[0]["lr"])
        if args.eval_train_every > 0 and epoch % args.eval_train_every == 0:
            train_metrics, train_label, train_pred = evaluate(net, eval_train_loader, device)
            print_metrics("train", len(train_pred), train_metrics)
        valid_metrics, valid_label, valid_pred = evaluate(net, valid_loader, device)
        print_metrics("valid", len(valid_pred), valid_metrics)

        valid_checkpoint_metrics = dict(valid_metrics)
        valid_checkpoint_metrics["Epoch"] = epoch
        best_valid_metrics, bad_epochs, should_stop, improved = update_regression_checkpoint_state(
            current_metrics=valid_checkpoint_metrics,
            best_metrics=best_valid_metrics,
            bad_epochs=bad_epochs,
            patience=args.early_stop_patience,
            select_metric=args.select_metric,
            min_delta=args.early_stop_min_delta,
        )
        if improved:
            best_rmse = float(valid_metrics["RMSE"])
        test_metrics_for_history = None
        if improved:
            torch.save(net.state_dict(), best_model_path)
            write_prediction_file(valid_pred_path, valid_label, valid_pred)
            test_metrics, test_label, test_pred = evaluate(net, test_loader, device)
            write_prediction_file(test_pred_path, test_label, test_pred)
            test_metrics_for_history = test_metrics
            print_metrics("test ", len(test_pred), test_metrics)
            best_payload = build_best_metrics_payload(
                epoch=epoch,
                train_loss=epoch_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
                valid_metrics=valid_metrics,
                test_metrics=test_metrics,
                checkpoint_path=best_model_path,
                selection_metric=args.select_metric,
                prediction_paths={"valid": valid_pred_path, "test": test_pred_path},
            )
            best_payload.update({
                "dataset": args.dataset,
                "split_method": args.split_method,
                "protein_3d_mode": args.protein_3d_mode,
                "protein_embedding_mode": args.protein_embedding_mode,
                "ligand_pocket_cutoff": float(args.ligand_pocket_cutoff),
                "disable_protein_2d": bool(args.disable_protein_2d),
                "only_3d": bool(args.only_3d),
                "dti_fusion_lmdb_path": args.dti_fusion_lmdb_path,
                "dti_front_fusion": args.dti_front_fusion,
                "dti_front_input_dim": int(args.dti_front_input_dim),
                "dti_late_fusion": args.dti_late_fusion,
                "dti_late_input_dim": int(args.dti_late_input_dim),
                "dti_late_hidden_dim": int(args.dti_late_hidden_dim),
                "ligand_unimol_lmdb_path": args.ligand_unimol_lmdb_path,
                "ligand_unimol_fusion": args.ligand_unimol_fusion,
                "ligand_unimol_input_dim": int(args.ligand_unimol_input_dim),
                "ligand_unimol2_lmdb_path": args.ligand_unimol2_lmdb_path,
                "ligand_unimol2_3d_fusion": args.ligand_unimol2_3d_fusion,
                "ligand_unimol2_3d_fusion_stage": args.ligand_unimol2_3d_fusion_stage,
                "ligand_unimol2_input_dim": int(args.ligand_unimol2_input_dim),
                "pbcnet_fusion_lmdb_path": args.pbcnet_fusion_lmdb_path,
                "pbcnet_graph_lmdb_path": args.pbcnet_graph_lmdb_path,
                "boltz_interface_lmdb_path": args.boltz_interface_lmdb_path,
                "boltz_interface_fusion": bool(args.boltz_interface_fusion),
                "boltz_interface_input_dim": int(args.boltz_interface_input_dim),
                "boltz_interface_hidden_dim": int(args.boltz_interface_hidden_dim),
                "boltz_pocket_radius": float(args.boltz_pocket_radius),
                "boltz_min_pocket_residues": int(args.boltz_min_pocket_residues),
                "boltz_max_pocket_residues": int(args.boltz_max_pocket_residues),
                "pbcnet_trainable": bool(args.pbcnet_trainable),
                "pbcnet_init_checkpoint": args.pbcnet_init_checkpoint,
                "pbcnet_input_dim": int(args.pbcnet_input_dim),
                "pbcnet_hidden_dim": int(args.pbcnet_hidden_dim),
                "pbcnet_model_hidden_dim": int(args.pbcnet_model_hidden_dim),
                "pbcnet_lr": float(args.pbcnet_lr),
                "freeze_pbcnet_epochs": int(args.freeze_pbcnet_epochs),
            })
            write_json(best_metrics_path, best_payload)
        best_selection_value = None if best_valid_metrics is None else best_valid_metrics.get(args.select_metric)
        if args.early_stop_patience > 0:
            print(
                "early_stop_bad_epochs:", bad_epochs, "/", args.early_stop_patience,
                "best_valid_{}:".format(args.select_metric),
                "nan" if best_selection_value is None else round(float(best_selection_value), 6),
                "best_valid_RMSE:", round(best_rmse, 6),
            )
        else:
            print(
                "early_stop_bad_epochs:", bad_epochs, "/ disabled",
                "best_valid_{}:".format(args.select_metric),
                "nan" if best_selection_value is None else round(float(best_selection_value), 6),
                "best_valid_RMSE:", round(best_rmse, 6),
            )
        append_metrics_history(
            metrics_history_file,
            build_metrics_history_row(
                epoch=epoch,
                train_loss=epoch_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
                valid_metrics=valid_metrics,
                test_metrics=test_metrics_for_history,
                best_rmse=best_rmse,
                bad_epochs=bad_epochs,
                improved=improved,
                selection_metric=args.select_metric,
                best_selection_value=best_selection_value,
            ),
        )
        save_training_checkpoint(checkpoint_path, epoch, best_rmse, bad_epochs, net, optimizer, scheduler, args, best_valid_metrics=best_valid_metrics)
        print("training_checkpoint:", checkpoint_path)
        if should_stop:
            print(
                "Early stopping at epoch", epoch,
                "best_valid_{}:".format(args.select_metric),
                "nan" if best_selection_value is None else round(float(best_selection_value), 6),
                "best_valid_RMSE:", round(best_rmse, 6),
            )
            break

    torch.save(net.state_dict(), last_model_path)
    print("Finished DTA training")
    print("best_model:", best_model_path)
    print("best_metrics:", best_metrics_path)
    print("last_model:", last_model_path)
    print("best_valid_pred:", valid_pred_path)
    print("best_test_pred:", test_pred_path)
    print("metrics_history:", metrics_history_file)


def train_and_eval_with_logging(args):
    prepare_model_output_dir(
        args.model_output_path,
        overwrite=args.overwrite_output_path or bool(args.resume_training_checkpoint),
    )
    args._model_output_dir_prepared = True
    log_path = training_log_path(args.model_output_path)
    log_mode = "a" if args.resume_training_checkpoint else "w"
    with open(log_path, log_mode, encoding="utf-8", buffering=1) as log_handle:
        tee_stdout = TeeStream(sys.stdout, log_handle)
        tee_stderr = TeeStream(sys.stderr, log_handle)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print("train_log:", log_path)
            if args.resume_training_checkpoint:
                print("=" * 80)
                print("Resuming training run")
                print("=" * 80)
            train_and_eval(args)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train GerNA-Bind protein-ligand DTA pAffinity regression.")
    parser.add_argument("--model_arch", choices=["gerna", "interface_first"], default="gerna")
    parser.add_argument("--dataset", type=str, default="PDBbind_atom5_dta")
    parser.add_argument("--split_method", type=str, default="random", help="Subdirectory under data/<dataset>/ used for split manifests, e.g. random, protein, mol, both, fixed_seed42.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--model_output_path", type=str, default="Model/PDBbind_atom5_dta_random/")
    parser.add_argument("--overwrite_output_path", action="store_true", help="Allow overwriting existing Model_dta_best.pth, Model_dta_last.pth, or best_metrics.json in --model_output_path.")
    parser.add_argument("--resume_training_checkpoint", type=str, default=None, help="Resume full training state from a training_checkpoint.pth file; --epoch is the final target epoch count.")
    parser.add_argument("--interface_train_native_lmdb", type=str, default=None)
    parser.add_argument("--interface_valid_native_lmdb", type=str, default=None)
    parser.add_argument("--interface_test_native_lmdb", type=str, default=None)
    parser.add_argument(
        "--interface_init_mode",
        choices=("stage1", "scratch"),
        default="stage1",
    )
    parser.add_argument("--interface_stage1_checkpoint", type=str, default=None)
    parser.add_argument("--interface_diagnostic_report", type=str, default=None)
    parser.add_argument("--interface_stage2_ready", type=str, default=None)
    parser.add_argument("--interface_local_lr", type=float, default=1e-4)
    parser.add_argument("--interface_global_lr", type=float, default=3e-5)
    parser.add_argument("--interface_affinity_lr", type=float, default=1e-4)
    parser.add_argument("--interface_warmup_epochs", type=int, default=3)
    parser.add_argument("--interface_shell_budget", type=int, default=128)
    parser.add_argument("--interface_ligand_gnn_depth", type=int, default=3)
    parser.add_argument("--interface_ligand_dropout", type=float, default=0.1)
    parser.add_argument("--interface_affinity_dropout", type=float, default=0.2)
    parser.add_argument(
        "--interface_residue_to_atom_mode",
        choices=["gated_residual", "concat", "no_esm"],
        default="gated_residual",
    )
    parser.add_argument("--init_model_checkpoint", type=str, default=None, help="Initialize model weights from a Model_dta_best.pth/state_dict checkpoint, but start a fresh optimizer/scheduler/early-stop run.")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss", type=str, default="smooth_l1", choices=["mse", "smooth_l1"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_decay_step", type=int, default=10)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Request AMP (unsupported by interface-first Stage 2).")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)
    parser.add_argument("--select_metric", choices=REGRESSION_METRIC_NAMES, default="RMSE", help="Validation metric used to save Model_dta_best.pth and count non-improving epochs.")
    parser.add_argument("--rank_loss_weight", type=float, default=0.0, help="Optional auxiliary pairwise ranking loss weight added to the regression loss.")
    parser.add_argument("--rank_min_gap", type=float, default=0.10, help="Minimum absolute label gap included in the auxiliary rank loss.")
    parser.add_argument("--rank_temperature", type=float, default=0.5, help="Temperature for the auxiliary pairwise rank loss.")
    parser.add_argument("--step_log_every", type=int, default=1000)
    parser.add_argument("--eval_train_every", type=int, default=0, help="Evaluate the full train set every N epochs; 0 disables expensive train-set evaluation.")
    parser.add_argument("--GNN_depth", type=int, default=4)
    parser.add_argument("--DMA_depth", type=int, default=2)
    parser.add_argument("--hidden_size1", type=int, default=128)
    parser.add_argument("--hidden_size2", type=int, default=128)
    parser.add_argument("--rna_num_tokens", type=int, default=None)
    parser.add_argument("--protein_3d_mode", choices=["residue", "all_atom", "atom5", "pocket_pharmacophore", "pocket_pharmacophore_esm", "pocket_residue_esm", "ligand_pocket_residue_esm", "full_residue_esm"], default=None, help="Protein 3D tokens. ligand_pocket_residue_esm dynamically selects full-chain ESM CA residues near the ligand.")
    parser.add_argument("--protein_embedding_mode", choices=["esm", "residue", "auto"], default="esm", help="Protein 1D/2D residue features for LMDB manifests. esm fails fast if esm2_embedding is missing; residue is the non-ESM baseline; auto falls back silently.")
    parser.add_argument("--expected_protein_embedding_dim", type=int, default=640)
    parser.add_argument("--ligand_pocket_cutoff", type=float, default=8.0, help="CA-to-ligand distance cutoff in Angstrom for ligand_pocket_residue_esm.")
    parser.add_argument("--disable_protein_2d", action="store_true", help="Disable protein 2D graph and protein-side 2D triangle geometry while keeping chain ESM, ligand 2D, and 3D branches.")
    parser.add_argument("--only_3d", action="store_true", help="Use only ligand/protein 3D Equiformer branches; disables protein 1D/2D and ligand 2D pairwise branches.")
    parser.add_argument("--dti_fusion_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key; values are pickled dicts with protein_tokens [L,D] and/or global [D].")
    parser.add_argument("--dti_front_fusion", choices=["none", "add", "concat"], default="none", help="Fuse external unpooled DTI protein tokens into GerNA protein 1D features.")
    parser.add_argument("--dti_front_input_dim", type=int, default=0, help="Feature dimension of DTI protein_tokens; required when --dti_front_fusion is not none.")
    parser.add_argument("--dti_late_fusion", choices=["none", "concat"], default="none", help="Fuse external pooled/global DTI vector before the affinity head.")
    parser.add_argument("--dti_late_input_dim", type=int, default=0, help="Feature dimension of DTI global vector; required when --dti_late_fusion is not none.")
    parser.add_argument("--dti_late_hidden_dim", type=int, default=128, help="Projected dimension appended to GerNA kronecker vector for late concat fusion.")
    parser.add_argument("--ligand_unimol_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key with GerNA-order ligand Uni-Mol token embeddings.")
    parser.add_argument("--ligand_unimol_fusion", choices=["none", "add", "concat"], default="none", help="Fuse GerNA-order ligand Uni-Mol atom tokens into the ligand 2D graph atom features.")
    parser.add_argument("--ligand_unimol_input_dim", type=int, default=0, help="Feature dimension of ligand Uni-Mol token embeddings; required when --ligand_unimol_fusion is not none.")
    parser.add_argument("--ligand_unimol2_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key with GerNA-order ligand Uni-Mol2 atom token embeddings.")
    parser.add_argument("--ligand_unimol2_3d_fusion", choices=["none", "add", "concat"], default="none", help="Fuse GerNA-order ligand Uni-Mol2 atom tokens into ligand 3D atom features only.")
    parser.add_argument("--ligand_unimol2_3d_fusion_stage", choices=["pre", "post"], default="post", help="Where to fuse Uni-Mol2 ligand atom tokens: post fuses after ligand Equiformer; pre fuses before ligand Equiformer.")
    parser.add_argument("--ligand_unimol2_input_dim", type=int, default=0, help="Feature dimension of ligand Uni-Mol2 token embeddings; required when --ligand_unimol2_3d_fusion is not none.")
    parser.add_argument("--pbcnet_fusion_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key with PBCNet2.0 pocket-ligand global embeddings.")
    parser.add_argument("--pbcnet_graph_lmdb_path", type=str, default=None, help="Optional LMDB keyed by sample lmdb_key with PBCNet2.0 DGL graphs for trainable PBCNet fusion.")
    parser.add_argument("--boltz_interface_lmdb_path", type=str, default=None, help="Optional Boltz merged LMDB with frozen pocket residue and ligand atom embeddings.")
    parser.add_argument("--boltz_interface_fusion", action="store_true", help="Fuse frozen Boltz local pair and global interface features before the affinity head.")
    parser.add_argument("--boltz_interface_input_dim", type=int, default=384)
    parser.add_argument("--boltz_interface_hidden_dim", type=int, default=128)
    parser.add_argument("--boltz_pocket_radius", type=float, default=10.0)
    parser.add_argument("--boltz_min_pocket_residues", type=int, default=16)
    parser.add_argument("--boltz_max_pocket_residues", type=int, default=64)
    parser.add_argument("--pbcnet_trainable", action="store_true", help="Enable end-to-end trainable PBCNet2.0 from graph batches. Use --pbcnet_init_checkpoint to warm-start from pretrained weights, or omit it for random init.")
    parser.add_argument("--pbcnet_init_checkpoint", type=str, default=None, help="Optional pretrained PBCNet checkpoint used to initialize the trainable PBCNet encoder before full fine-tuning.")
    parser.add_argument("--pbcnet_root", type=str, default="external/PBCNet2.0", help="Root directory containing PBCNet2.0 model_code for --pbcnet_trainable.")
    parser.add_argument("--pbcnet_model_hidden_dim", type=int, default=96, help="Hidden/readout dimension for randomly initialized trainable PBCNetv2.")
    parser.add_argument("--pbcnet_lr", type=float, default=0.0, help="Optional separate learning rate for trainable PBCNet encoder; 0 uses the main --lr.")
    parser.add_argument("--freeze_pbcnet_epochs", type=int, default=0, help="Freeze trainable PBCNet encoder for the first N epochs, then unfreeze it.")
    parser.add_argument("--pbcnet_late_fusion", choices=["none", "concat"], default="none", help="Fuse PBCNet2.0 _readout global vector before the affinity head.")
    parser.add_argument("--pbcnet_input_dim", type=int, default=0, help="Feature dimension of PBCNet global vector; expected 128 for PBCNet2.0 default weights.")
    parser.add_argument("--pbcnet_hidden_dim", type=int, default=64, help="Projected dimension appended to GerNA kronecker vector for PBCNet late concat fusion.")
    parser.add_argument("--protein_edge_mode", choices=["radius", "knn"], default="radius", help="Protein 2D graph edge construction for LMDB manifests.")
    parser.add_argument("--protein_contact_cutoff", type=float, default=6.0, help="Radius cutoff in Angstrom for LMDB protein graph edges when --protein_edge_mode radius.")
    parser.add_argument("--protein_knn_k", type=int, default=16, help="K for LMDB protein graph edges when --protein_edge_mode knn.")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=99)
    return parser


def validate_training_args(parser, args):
    if args.ligand_pocket_cutoff <= 0:
        parser.error("--ligand_pocket_cutoff must be positive")
    if args.boltz_interface_fusion and not args.boltz_interface_lmdb_path:
        parser.error("--boltz_interface_lmdb_path is required when --boltz_interface_fusion is enabled")
    if args.boltz_interface_input_dim <= 0 or args.boltz_interface_hidden_dim <= 0:
        parser.error("Boltz interface dimensions must be positive")
    if args.boltz_pocket_radius <= 0 or args.boltz_min_pocket_residues <= 0 or args.boltz_max_pocket_residues < args.boltz_min_pocket_residues:
        parser.error("Invalid Boltz interface pocket parameters")
    if args.dti_front_fusion != "none" and args.dti_front_input_dim <= 0:
        parser.error("--dti_front_input_dim must be positive when --dti_front_fusion is enabled")
    if args.dti_late_fusion != "none" and args.dti_late_input_dim <= 0:
        parser.error("--dti_late_input_dim must be positive when --dti_late_fusion is enabled")
    if (args.dti_front_fusion != "none" or args.dti_late_fusion != "none") and not args.dti_fusion_lmdb_path:
        parser.error("--dti_fusion_lmdb_path is required when any DTI fusion mode is enabled")
    if args.ligand_unimol_fusion != "none" and args.ligand_unimol_input_dim <= 0:
        parser.error("--ligand_unimol_input_dim must be positive when --ligand_unimol_fusion is enabled")
    if args.ligand_unimol_fusion != "none" and not args.ligand_unimol_lmdb_path:
        parser.error("--ligand_unimol_lmdb_path is required when --ligand_unimol_fusion is enabled")
    if args.ligand_unimol2_3d_fusion != "none" and args.ligand_unimol2_input_dim <= 0:
        parser.error("--ligand_unimol2_input_dim must be positive when --ligand_unimol2_3d_fusion is enabled")
    if args.ligand_unimol2_3d_fusion != "none" and not args.ligand_unimol2_lmdb_path:
        parser.error("--ligand_unimol2_lmdb_path is required when --ligand_unimol2_3d_fusion is enabled")
    if args.pbcnet_trainable and args.pbcnet_input_dim <= 0:
        if args.pbcnet_init_checkpoint:
            from net.model import infer_pbcnet_hidden_dim_from_state_dict, load_pbcnet_state_dict
            inferred_state_dict = load_pbcnet_state_dict(
                args.pbcnet_init_checkpoint,
                pbcnet_root=args.pbcnet_root,
            )
            inferred_dim = infer_pbcnet_hidden_dim_from_state_dict(inferred_state_dict)
            if inferred_dim is None:
                parser.error("Could not infer --pbcnet_input_dim from --pbcnet_init_checkpoint")
            args.pbcnet_input_dim = inferred_dim
            args.pbcnet_model_hidden_dim = inferred_dim
        else:
            args.pbcnet_input_dim = args.pbcnet_model_hidden_dim
    if args.pbcnet_late_fusion != "none" and args.pbcnet_input_dim <= 0:
        parser.error("--pbcnet_input_dim must be positive when --pbcnet_late_fusion is enabled")
    if args.pbcnet_late_fusion != "none" and args.pbcnet_trainable and not args.pbcnet_graph_lmdb_path:
        parser.error("--pbcnet_graph_lmdb_path is required when --pbcnet_trainable is enabled")
    if args.pbcnet_late_fusion != "none" and not args.pbcnet_trainable and not args.pbcnet_fusion_lmdb_path:
        parser.error("--pbcnet_fusion_lmdb_path is required for non-trainable PBCNet late fusion")
    if args.pbcnet_trainable and args.pbcnet_fusion_lmdb_path:
        parser.error("Do not pass --pbcnet_fusion_lmdb_path with --pbcnet_trainable; use --pbcnet_graph_lmdb_path")
    if args.pbcnet_lr < 0:
        parser.error("--pbcnet_lr must be non-negative")
    if args.freeze_pbcnet_epochs < 0:
        parser.error("--freeze_pbcnet_epochs must be non-negative")
    if args.rank_loss_weight < 0:
        parser.error("--rank_loss_weight must be non-negative")
    if args.rank_min_gap < 0:
        parser.error("--rank_min_gap must be non-negative")
    if args.rank_temperature <= 0:
        parser.error("--rank_temperature must be positive")
    if args.amp:
        parser.error("--amp is not supported by this training entry point")
    if args.model_arch == "interface_first":
        native_cache_values = {
            "--interface_train_native_lmdb": args.interface_train_native_lmdb,
            "--interface_valid_native_lmdb": args.interface_valid_native_lmdb,
            "--interface_test_native_lmdb": args.interface_test_native_lmdb,
        }
        missing_native_caches = [
            name for name, value in native_cache_values.items() if not value
        ]
        if missing_native_caches:
            parser.error(
                "interface_first requires "
                + ", ".join(missing_native_caches)
            )
        stage1_values = {
            "--interface_stage1_checkpoint": args.interface_stage1_checkpoint,
            "--interface_diagnostic_report": args.interface_diagnostic_report,
            "--interface_stage2_ready": args.interface_stage2_ready,
        }
        if args.interface_init_mode == "stage1":
            missing_stage1 = [
                name for name, value in stage1_values.items() if not value
            ]
            if missing_stage1:
                parser.error(
                    "interface_first requires "
                    + ", ".join(missing_stage1)
                )
        else:
            supplied_stage1 = [
                name for name, value in stage1_values.items() if value
            ]
            if supplied_stage1:
                parser.error(
                    "scratch interface_first does not accept "
                    + ", ".join(supplied_stage1)
                )
            if args.interface_warmup_epochs != 0:
                parser.error(
                    "scratch interface_first requires "
                    "--interface_warmup_epochs 0"
                )
        if args.batch_size != 1:
            parser.error("interface_first requires --batch_size 1")
        if args.loss != "smooth_l1":
            parser.error("interface_first requires --loss smooth_l1")
        if args.early_stop_patience != 0:
            parser.error(
                "interface_first requires --early_stop_patience 0"
            )
        if args.rank_loss_weight != 0:
            parser.error(
                "interface_first does not permit affinity rank loss"
            )
        if args.select_metric != "CI":
            parser.error("interface_first requires --select_metric CI")
        if args.interface_warmup_epochs < 0:
            parser.error(
                "--interface_warmup_epochs must be non-negative"
            )
        if args.interface_shell_budget < 0:
            parser.error(
                "--interface_shell_budget must be non-negative"
            )
        if min(
            args.interface_local_lr,
            args.interface_global_lr,
            args.interface_affinity_lr,
        ) <= 0:
            parser.error("interface-first learning rates must be positive")
        if args.epoch <= 0:
            parser.error("--epoch must be positive")
        if args.num_workers < 0:
            parser.error("--num_workers must be non-negative")
        if args.lr_decay_step <= 0:
            parser.error("--lr_decay_step must be positive")
        if args.lr_decay_gamma <= 0:
            parser.error("--lr_decay_gamma must be positive")
        if args.weight_decay < 0:
            parser.error("--weight_decay must be non-negative")
        if args.grad_clip <= 0:
            parser.error("--grad_clip must be positive")
        if args.interface_ligand_gnn_depth <= 0:
            parser.error("--interface_ligand_gnn_depth must be positive")
        if not 0 <= args.interface_ligand_dropout < 1:
            parser.error("--interface_ligand_dropout must be in [0,1)")
        if not 0 <= args.interface_affinity_dropout < 1:
            parser.error("--interface_affinity_dropout must be in [0,1)")
        external_fusion_enabled = any(
            (
                args.dti_fusion_lmdb_path,
                args.dti_front_fusion != "none",
                args.dti_late_fusion != "none",
                args.ligand_unimol_lmdb_path,
                args.ligand_unimol_fusion != "none",
                args.ligand_unimol2_lmdb_path,
                args.ligand_unimol2_3d_fusion != "none",
                args.pbcnet_fusion_lmdb_path,
                args.pbcnet_graph_lmdb_path,
                args.pbcnet_trainable,
                args.pbcnet_init_checkpoint,
                args.pbcnet_late_fusion != "none",
                args.boltz_interface_lmdb_path,
                args.boltz_interface_fusion,
            )
        )
        if external_fusion_enabled:
            parser.error(
                "interface_first does not permit external fusion inputs"
            )
        if args.only_3d or args.disable_protein_2d:
            parser.error(
                "legacy GerNA branch flags do not apply to interface_first"
            )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_training_args(parser, args)
    if args.model_arch == "interface_first":
        train_path, valid_path, test_path = resolve_dataset_paths(args)
        args.train_path = train_path
        args.valid_path = valid_path
        args.test_path = test_path
        from dta.interface_first_training import (
            train_interface_first_with_logging,
        )

        train_interface_first_with_logging(args)
        return
    train_and_eval_with_logging(args)


if __name__ == "__main__":
    main()
