"""Keyed feature export for the frozen official 3DProtDTA expert."""

import hashlib
import importlib.util
import json
import math
import pickle
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import torch

from utils.experiment_integrity import (
    atomic_write_json,
    manifest_fingerprint,
    sha256_file,
)

from .artifact import write_expert_artifact


ADAPTER_VERSION = "three-dprot-official-v1"
EXPERT_NAME = "three_dprot"
MAX_LATENT_DIMENSION = 1024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREE_DPROT_ROOT = PROJECT_ROOT / "external_benchmarks" / "3d-prot-dta"
PREPARED_FILENAMES = (
    "full.csv",
    "folds/train_fold_setting1.txt",
    "folds/test_fold_setting1.txt",
    "ligand_to_graph.pkl",
    "protein_to_graph.pkl",
    "ligand_to_ecfp.pkl",
)


def forward_three_dprot_features(model, data):
    """Reproduce the upstream forward while returning its four hidden states."""
    ecfp = model.chem_ecfp_post_fc(data["e1_fp"])
    ligand = model.chem_graph_encoder(data["e1_graph"])
    protein = model.prot_graph_encoder(data["e2_graph"])
    joint = model.fc(torch.cat((ecfp, ligand, protein), dim=1))
    prediction = model.out(joint).reshape(-1)
    return prediction, (ecfp, ligand, protein, joint)


def _load_module(name, path, package_locations=None):
    specification = importlib.util.spec_from_file_location(
        name,
        str(path),
        submodule_search_locations=(
            None if package_locations is None else [str(package_locations)]
        ),
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load 3DProt dependency {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _temporary_modules(replacements):
    class ModuleAliases:
        def __enter__(self):
            self.previous = {
                name: sys.modules.get(name) for name in replacements
            }
            sys.modules.update(replacements)

        def __exit__(self, exc_type, exc_value, traceback):
            for name, previous in self.previous.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    return ModuleAliases()


def _build_dependencies():
    """Load upstream absolute-import modules without shadowing project packages."""
    prefix = "_rank_fusion_v4_three_dprot"
    params = _load_module(prefix + "_params", THREE_DPROT_ROOT / "params.py")
    general = _load_module(
        prefix + "_general", THREE_DPROT_ROOT / "utils" / "general.py"
    )
    upstream_utils = types.ModuleType(prefix + "_utils")
    for name in ("logger", "GraphPairsBatchConstructor", "load_pkl"):
        setattr(upstream_utils, name, getattr(general, name))
    with _temporary_modules({"params": params, "utils": upstream_utils}):
        helpers = _load_module(prefix + "_helpers", THREE_DPROT_ROOT / "helpers.py")
        models = _load_module(prefix + "_models", THREE_DPROT_ROOT / "models.py")
    return (
        helpers.CustomDataLoader,
        helpers.CustomTrial,
        models.DTIProtGraphChemGraphECFP,
        dict(params.HP),
    )


def _load_manifest(path, expected_split):
    path = Path(path).expanduser().resolve()
    fingerprint = manifest_fingerprint(path)
    if fingerprint["split"] != expected_split:
        raise ValueError("manifest split does not match requested split")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a nonempty list")
    keys, labels, pdb_ids = [], [], []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("manifest sample must be a mapping")
        key = sample.get("lmdb_key")
        if not isinstance(key, str) or not key:
            raise ValueError("manifest lmdb_key must be nonempty")
        suffix = "_protein"
        if not key.lower().endswith(suffix) or len(key) <= len(suffix):
            raise ValueError(f"3DProt cannot map lmdb_key to PDB ID: {key}")
        try:
            label = float(sample["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"manifest label is invalid for {key}") from error
        if not math.isfinite(label):
            raise ValueError(f"manifest label is nonfinite for {key}")
        keys.append(key)
        labels.append(label)
        pdb_ids.append(key[: -len(suffix)].lower())
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate keys")
    if len(pdb_ids) != len(set(pdb_ids)):
        raise ValueError("manifest maps multiple keys to one PDB ID")
    return (
        fingerprint,
        keys,
        np.asarray(labels, dtype=np.float32),
        pdb_ids,
    )


def _resolve_data_dir(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    missing = [name for name in PREPARED_FILENAMES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "prepared 3DProt data is incomplete at "
            + str(path)
            + ": "
            + ", ".join(missing)
        )
    return path


def _load_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("3DProt checkpoint must be a mapping")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("3DProt checkpoint model_state_dict is missing")
    if not isinstance(checkpoint.get("args"), dict):
        raise ValueError("3DProt checkpoint args is missing")
    return path, checkpoint


def _load_prepared_data(data_dir):
    frame = pd.read_csv(data_dir / "full.csv")
    required = {"ligand", "protein", "label"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "prepared 3DProt full.csv is missing " + ", ".join(missing_columns)
        )
    frame = frame[["ligand", "protein", "label"]].copy()
    frame["protein"] = frame["protein"].astype(str).str.lower()
    if frame["protein"].duplicated().any():
        raise ValueError("prepared 3DProt full.csv contains duplicate protein keys")
    for name in (
        "ligand_to_graph.pkl",
        "protein_to_graph.pkl",
        "ligand_to_ecfp.pkl",
    ):
        with (data_dir / name).open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"prepared 3DProt {name} must contain a mapping")
        if name == "ligand_to_graph.pkl":
            ligand_graphs = value
        elif name == "protein_to_graph.pkl":
            protein_graphs = value
        else:
            ligand_ecfp = value
    return frame, protein_graphs, ligand_graphs, ligand_ecfp


def _module_output_width(module):
    width = getattr(module, "n_out", None)
    if isinstance(width, int) and width > 0:
        return width
    width = getattr(module, "out_features", None)
    if isinstance(width, int) and width > 0:
        return width
    children = list(module.children())
    for child in reversed(children):
        try:
            return _module_output_width(child)
        except ValueError:
            continue
    raise ValueError(f"cannot determine output width for {type(module).__name__}")


def _latent_width(model):
    return sum(
        _module_output_width(module)
        for module in (
            model.chem_ecfp_post_fc,
            model.chem_graph_encoder,
            model.prot_graph_encoder,
            model.fc,
        )
    )


def _projection(checkpoint_sha256, input_dimension):
    if input_dimension <= MAX_LATENT_DIMENSION:
        return None, None, None
    digest = hashlib.sha256(str(checkpoint_sha256).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
    generator = np.random.RandomState(seed)
    matrix = generator.standard_normal(
        (input_dimension, MAX_LATENT_DIMENSION)
    ).astype(np.float32)
    matrix /= np.float32(math.sqrt(MAX_LATENT_DIMENSION))
    projection_sha256 = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    return seed, matrix, projection_sha256


def _move_data(data, device):
    moved = {}
    for name, value in data.items():
        if hasattr(value, "to"):
            moved[name] = value.to(device)
        else:
            moved[name] = value
    return moved


def _load_one_batch(
    frame,
    CustomDataLoader,
    device,
    ligand_graphs,
    protein_graphs,
    ligand_ecfp,
):
    loader = CustomDataLoader(
        df=frame[["ligand", "protein", "label"]].copy(),
        batch_size=len(frame),
        device=device,
        e1_key_to_graph=ligand_graphs,
        e2_key_to_graph=protein_graphs,
        e1_key_to_fp=ligand_ecfp,
        shuffle=False,
    )
    iterator = iter(loader)
    labels, data = next(iterator)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("3DProt loader produced more than one batch")
    return labels, data


def _run_batch(
    frame,
    row_indices,
    model,
    CustomDataLoader,
    device,
    ligand_graphs,
    protein_graphs,
    ligand_ecfp,
    projection_tensor,
    predictions,
    latents,
    available,
):
    labels, data = _load_one_batch(
        frame,
        CustomDataLoader,
        device,
        ligand_graphs,
        protein_graphs,
        ligand_ecfp,
    )
    labels = labels.reshape(-1).float()
    expected = torch.as_tensor(frame["label"].to_numpy(), dtype=torch.float32)
    if labels.shape != expected.shape or not torch.allclose(
        labels.cpu(), expected, rtol=0.0, atol=1e-5
    ):
        raise ValueError("3DProt loader labels do not match prepared rows")
    data = _move_data(data, device)
    with torch.inference_mode():
        output, parts = forward_three_dprot_features(model, data)
        hidden = torch.cat(tuple(part.float() for part in parts), dim=1)
        if projection_tensor is not None:
            hidden = hidden @ projection_tensor
    if output.shape != (len(row_indices),) or hidden.shape[0] != len(row_indices):
        raise RuntimeError("3DProt model output has an invalid batch shape")
    if not torch.isfinite(output).all() or not torch.isfinite(hidden).all():
        raise FloatingPointError("3DProt model produced nonfinite output")
    output = output.float().cpu().numpy()
    hidden = hidden.float().cpu().numpy()
    for offset, row_index in enumerate(row_indices):
        predictions[row_index] = output[offset]
        latents[row_index] = hidden[offset]
        available[row_index] = True


def export_three_dprot(
    manifest_path,
    checkpoint_path,
    output_dir,
    *,
    split,
    device,
    batch_size=32,
    limit=None,
    data_dir=None,
):
    """Export a manifest-aligned official 3DProtDTA artifact."""
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    fingerprint, keys, labels, pdb_ids = _load_manifest(manifest_path, split)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        if not 1 <= limit <= len(keys):
            raise ValueError("limit must be between 1 and manifest count")
        keys, labels, pdb_ids = keys[:limit], labels[:limit], pdb_ids[:limit]

    checkpoint_path, checkpoint = _load_checkpoint(checkpoint_path)
    selected_data_dir = data_dir or checkpoint["args"].get("data_dir")
    if not selected_data_dir:
        raise ValueError("3DProt checkpoint args is missing data_dir")
    selected_data_dir = _resolve_data_dir(selected_data_dir)
    frame, protein_graphs, ligand_graphs, ligand_ecfp = _load_prepared_data(
        selected_data_dir
    )
    prepared_by_pdb = {
        str(row.protein).lower(): row
        for row in frame.itertuples(index=False)
    }

    CustomDataLoader, CustomTrial, Model, hp = _build_dependencies()
    model = Model(trial=CustomTrial(hp=hp))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(device)
    model = model.to(device).eval()
    input_dimension = _latent_width(model)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    projection_seed, projection, projection_sha256 = _projection(
        checkpoint_sha256, input_dimension
    )
    output_dimension = min(input_dimension, MAX_LATENT_DIMENSION)
    projection_tensor = (
        None
        if projection is None
        else torch.from_numpy(projection).to(device=device)
    )

    predictions = np.zeros(len(keys), dtype=np.float32)
    latents = np.zeros((len(keys), output_dimension), dtype=np.float32)
    available = np.zeros(len(keys), dtype=np.bool_)
    missing = []
    candidates = []
    for row_index, (key, pdb_id, expected_label) in enumerate(
        zip(keys, pdb_ids, labels)
    ):
        prepared = prepared_by_pdb.get(pdb_id)
        if prepared is None:
            missing.append({"key": key, "reason": "not_in_prepared_data"})
            continue
        try:
            prepared_label = float(prepared.label)
        except (TypeError, ValueError) as error:
            raise ValueError(f"prepared 3DProt label is invalid for {key}") from error
        if not math.isfinite(prepared_label) or not math.isclose(
            prepared_label, float(expected_label), rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(f"prepared 3DProt label does not match manifest for {key}")
        candidates.append(
            (
                row_index,
                key,
                {
                    "ligand": prepared.ligand,
                    "protein": pdb_id,
                    "label": prepared_label,
                },
            )
        )

    safe_record_errors = (ValueError, KeyError, IndexError)
    for start in range(0, len(candidates), int(batch_size)):
        chunk = candidates[start : start + int(batch_size)]
        chunk_frame = pd.DataFrame([item[2] for item in chunk])
        chunk_indices = [item[0] for item in chunk]
        try:
            loaded = _load_one_batch(
                chunk_frame,
                CustomDataLoader,
                device,
                ligand_graphs,
                protein_graphs,
                ligand_ecfp,
            )
        except safe_record_errors:
            for row_index, key, row in chunk:
                row_frame = pd.DataFrame([row])
                try:
                    _run_batch(
                        row_frame,
                        [row_index],
                        model,
                        CustomDataLoader,
                        device,
                        ligand_graphs,
                        protein_graphs,
                        ligand_ecfp,
                        projection_tensor,
                        predictions,
                        latents,
                        available,
                    )
                except safe_record_errors as error:
                    missing.append(
                        {"key": key, "reason": "record_error", "error": repr(error)}
                    )
            continue
        labels_batch, data_batch = loaded
        expected_batch = torch.as_tensor(
            chunk_frame["label"].to_numpy(), dtype=torch.float32
        )
        if labels_batch.reshape(-1).shape != expected_batch.shape or not torch.allclose(
            labels_batch.reshape(-1).float().cpu(),
            expected_batch,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError("3DProt loader labels do not match prepared rows")
        data_batch = _move_data(data_batch, device)
        with torch.inference_mode():
            output, parts = forward_three_dprot_features(model, data_batch)
            hidden = torch.cat(tuple(part.float() for part in parts), dim=1)
            if projection_tensor is not None:
                hidden = hidden @ projection_tensor
        if output.shape != (len(chunk),) or hidden.shape != (
            len(chunk),
            output_dimension,
        ):
            raise RuntimeError("3DProt model output has an invalid batch shape")
        if not torch.isfinite(output).all() or not torch.isfinite(hidden).all():
            raise FloatingPointError("3DProt model produced nonfinite output")
        output_array = output.float().cpu().numpy()
        hidden_array = hidden.float().cpu().numpy()
        for offset, (row_index, _, _) in enumerate(chunk):
            predictions[row_index] = output_array[offset]
            latents[row_index] = hidden_array[offset]
            available[row_index] = True

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_metadata = write_expert_artifact(
        output_path,
        keys=keys,
        labels=labels,
        predictions=predictions,
        latents=latents,
        available=available,
        expert=EXPERT_NAME,
        split=split,
        manifest_sha256=fingerprint["sha256"],
        checkpoint_sha256=checkpoint_sha256,
        adapter_version=ADAPTER_VERSION,
    )
    atomic_write_json(
        output_path / "adapter_metadata.json",
        {
            "status": "complete",
            "adapter_version": ADAPTER_VERSION,
            "expert": EXPERT_NAME,
            "split": split,
            "manifest_sha256": fingerprint["sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "source_count": fingerprint["sample_count"],
            "selected_count": len(keys),
            "available_count": int(available.sum()),
            "missing_count": int((~available).sum()),
            "missing": missing,
            "data_dir": str(selected_data_dir),
            "latent_input_dimension": input_dimension,
            "latent_dimension": output_dimension,
            "projection_seed": projection_seed,
            "projection_input_dimension": input_dimension,
            "projection_output_dimension": output_dimension,
            "projection_sha256": projection_sha256,
            "latent_parts": ["ecfp", "ligand_graph", "protein_graph", "joint"],
        },
    )
    return artifact_metadata
