import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from data_utils.local_interface_decoys import (
    soft_contact_statistics,
    vdw_overlap_statistics,
)


STAGE1_CHECKPOINT_SCHEMA_VERSION = 1
POSE_TEMPERATURE = 0.2
CONSISTENCY_WEIGHT = 0.1
JITTER_TRANSLATION_RADIUS = 0.10
JITTER_MAX_ROTATION_DEGREES = 3.0
JITTER_MINIMUM_CONTACT_RETENTION = 0.95
FROZEN_DIAGNOSTIC_KEY_COUNT = 1104
FROZEN_DIAGNOSTIC_ORDERED_KEY_SHA256 = (
    "43a7fa8b74f928fb8712713b80f0fd46aeb626aa0b2fec0392adf84f1f7d4005"
)


def pose_set_loss(native_score, decoy_scores, temperature=POSE_TEMPERATURE):
    if not torch.is_tensor(native_score) or native_score.numel() != 1:
        raise ValueError("native_score must contain one scalar")
    if not torch.is_tensor(decoy_scores) or decoy_scores.numel() != 3:
        raise ValueError("pose_set_loss requires exactly K=3 decoys")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    logits = torch.cat(
        [native_score.reshape(1), decoy_scores.reshape(3)],
        dim=0,
    )
    return -F.log_softmax(logits / temperature, dim=0)[0]


def pretrain_objective(
    native_score,
    decoy_scores,
    native_projection,
    jitter_projection,
    temperature=POSE_TEMPERATURE,
    consistency_weight=CONSISTENCY_WEIGHT,
):
    if (
        not torch.is_tensor(decoy_scores)
        or decoy_scores.numel() != 3
    ):
        raise ValueError(
            "pretrain_objective requires exactly K=3 decoys"
        )
    if (
        not torch.is_tensor(native_projection)
        or not torch.is_tensor(jitter_projection)
        or native_projection.ndim != 2
        or jitter_projection.shape != native_projection.shape
        or native_projection.shape[0] != 1
        or native_projection.shape[1] == 0
    ):
        raise ValueError(
            "native and jitter projections must align as [1,D]"
        )
    consistency_weight = float(consistency_weight)
    if (
        not math.isfinite(consistency_weight)
        or consistency_weight < 0
    ):
        raise ValueError(
            "consistency_weight must be finite and non-negative"
        )
    pose = pose_set_loss(
        native_score, decoy_scores, temperature=temperature
    )
    consistency = 1.0 - F.cosine_similarity(
        native_projection,
        jitter_projection,
        dim=-1,
        eps=1e-8,
    ).mean()
    return (
        pose + consistency_weight * consistency,
        pose,
        consistency,
    )


def stage1_forward_loss(
    model,
    item,
    jitter_coords,
    temperature=POSE_TEMPERATURE,
    consistency_weight=CONSISTENCY_WEIGHT,
):
    decoy_coords = item.get("decoy_ligand_coords")
    if (
        not torch.is_tensor(decoy_coords)
        or decoy_coords.ndim != 3
        or decoy_coords.shape[0] != 3
        or decoy_coords.shape[1:] != item["ligand_coords"].shape
    ):
        raise ValueError(
            "Stage-1 item requires exactly K=3 aligned decoy poses"
        )
    if (
        not torch.is_tensor(jitter_coords)
        or jitter_coords.shape != item["ligand_coords"].shape
    ):
        raise ValueError(
            "jitter_coords must align with native ligand coordinates"
        )
    static_state = model.local_path.encode_static(item)
    native = model.local_path.forward_geometry(
        item, static_state
    )
    decoys = [
        model.local_path.forward_geometry(
            item,
            static_state,
            ligand_coords=coordinates,
        )
        for coordinates in decoy_coords
    ]
    jitter = model.local_path.forward_geometry(
        item,
        static_state,
        ligand_coords=jitter_coords,
    )
    decoy_scores = torch.cat(
        [output.interface_score for output in decoys],
        dim=0,
    )
    total, pose, consistency = pretrain_objective(
        native.interface_score,
        decoy_scores,
        native.projection,
        jitter.projection,
        temperature=temperature,
        consistency_weight=consistency_weight,
    )
    return {
        "total_loss": total,
        "pose_loss": pose,
        "consistency_loss": consistency,
        "native_score": native.interface_score,
        "decoy_scores": decoy_scores,
    }


def _as_numpy(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    return array


def _random_unit_vector(generator):
    while True:
        vector = generator.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            return vector / norm


def _rotation_matrix(axis, angle_radians):
    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis
    cosine = math.cos(float(angle_radians))
    sine = math.sin(float(angle_radians))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float64,
    )


def _jitter_candidate(native_coords, generator):
    axis = _random_unit_vector(generator)
    angle_degrees = float(
        generator.uniform(0.0, JITTER_MAX_ROTATION_DEGREES)
    )
    rotation = _rotation_matrix(
        axis, math.radians(angle_degrees)
    )
    direction = _random_unit_vector(generator)
    radius = JITTER_TRANSLATION_RADIUS * float(
        generator.random() ** (1.0 / 3.0)
    )
    translation = direction * radius
    center = np.mean(native_coords, axis=0)
    coordinates = (
        (native_coords - center) @ rotation.T
        + center
        + translation
    )
    return coordinates.astype(np.float32), {
        "rotation_axis": axis.astype(float).tolist(),
        "rotation_angle_degrees": angle_degrees,
        "rotation_matrix": rotation.astype(float).tolist(),
        "translation": translation.astype(float).tolist(),
        "translation_norm": radius,
    }


def sample_valid_ligand_jitter(record, seed, max_attempts=64):
    if int(max_attempts) != max_attempts or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    native_coords = _as_numpy(
        record["ligand_coords"], "ligand_coords"
    ).astype(np.float64, copy=False)
    protein_coords = _as_numpy(
        record["protein_coords"], "protein_coords"
    ).astype(np.float64, copy=False)
    if (
        native_coords.ndim != 2
        or native_coords.shape[1] != 3
        or protein_coords.ndim != 2
        or protein_coords.shape[1] != 3
    ):
        raise ValueError(
            "jitter coordinates must have shape [N,3]"
        )
    protein_symbols = _as_numpy(
        record["protein_element_symbol"],
        "protein_element_symbol",
    )
    ligand_atomic_numbers = _as_numpy(
        record["ligand_atomic_numbers"],
        "ligand_atomic_numbers",
    )
    native_overlap = vdw_overlap_statistics(
        protein_coords,
        protein_symbols,
        native_coords,
        ligand_atomic_numbers,
    )
    generator = np.random.default_rng(int(seed))
    for attempt in range(1, int(max_attempts) + 1):
        coordinates, metadata = _jitter_candidate(
            native_coords, generator
        )
        overlap = vdw_overlap_statistics(
            protein_coords,
            protein_symbols,
            coordinates,
            ligand_atomic_numbers,
        )
        contacts = soft_contact_statistics(
            protein_coords, native_coords, coordinates
        )
        retention = float(contacts["contact_mass_ratio"])
        if (
            not overlap["has_hard_clash"]
            and contacts["coverage_retained"]
            and retention >= JITTER_MINIMUM_CONTACT_RETENTION
        ):
            return {
                "coords": coordinates,
                "attempt": attempt,
                "fallback_identity": False,
                "contact_mass_retention": retention,
                "contact_statistics": contacts,
                "overlap_statistics": overlap,
                "native_hard_clash": bool(
                    native_overlap["has_hard_clash"]
                ),
                **metadata,
            }

    overlap = native_overlap
    contacts = soft_contact_statistics(
        protein_coords, native_coords, native_coords
    )
    return {
        "coords": native_coords.astype(np.float32),
        "attempt": int(max_attempts),
        "fallback_identity": True,
        "contact_mass_retention": 1.0,
        "contact_statistics": contacts,
        "overlap_statistics": overlap,
        "native_hard_clash": bool(
            native_overlap["has_hard_clash"]
        ),
        "rotation_axis": [1.0, 0.0, 0.0],
        "rotation_angle_degrees": 0.0,
        "rotation_matrix": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0],
        "translation_norm": 0.0,
    }


def save_stage1_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    provenance,
):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "stage1_state": model.stage1_state_dict(),
        "optimizer_state": (
            None
            if optimizer is None
            else deepcopy(optimizer.state_dict())
        ),
        "metrics": deepcopy(dict(metrics)),
        "provenance": deepcopy(dict(provenance)),
    }
    temporary_path = path + ".tmp"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return payload


def load_stage1_checkpoint(
    path,
    model,
    optimizer=None,
    strict=True,
):
    payload = torch.load(
        os.fspath(path), map_location="cpu"
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Stage-1 checkpoint must be a dictionary")
    if (
        payload.get("schema_version")
        != STAGE1_CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "unsupported Stage-1 checkpoint schema"
        )
    model.load_stage1_state_dict(
        payload.get("stage1_state"), strict=bool(strict)
    )
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is None:
            raise RuntimeError(
                "Stage-1 checkpoint has no optimizer state"
            )
        optimizer.load_state_dict(optimizer_state)
    return payload


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train the complete local interface pathway on "
            "official LP-PDBBind train-only hard poses."
        )
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--native-lmdb", required=True)
    parser.add_argument("--decoy-lmdb", required=True)
    parser.add_argument("--diagnostic-keys", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--decoy-schema-version", type=int, choices=(1, 2), default=1
    )
    parser.add_argument(
        "--decoy-bank",
        choices=("random_physical", "matched"),
        default=None,
    )
    parser.add_argument(
        "--require-paired-bank-intersection",
        action="store_true",
    )
    parser.add_argument("--diagnostic-decoy-lmdb", default=None)
    parser.add_argument("--data-diagnostic-report", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--temperature", type=float, default=POSE_TEMPERATURE
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=CONSISTENCY_WEIGHT,
    )
    parser.add_argument("--ligand-gnn-depth", type=int, default=3)
    parser.add_argument("--ligand-dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser


def _stable_seed(seed, key, epoch, purpose):
    digest = hashlib.sha256()
    for value in (seed, key, epoch, purpose):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "little") % (
        2 ** 32
    )


def _set_random_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    return device


def _artifact_provenance(path):
    from utils.experiment_integrity import sha256_file

    path = os.path.abspath(os.fspath(path))
    output = {"path": path}
    candidates = (
        os.path.join(path, "metadata.json"),
        os.path.join(path, "decoy_report.json"),
    )
    reports = {}
    for candidate in candidates:
        if os.path.isfile(candidate):
            reports[os.path.basename(candidate)] = {
                "path": candidate,
                "sha256": sha256_file(candidate),
            }
    if reports:
        output["reports"] = reports
    return output


def initial_model_state_sha256(model):
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"model state value is not a tensor: {name}"
            )
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            np.asarray(value.shape, dtype="<i8").tobytes(order="C")
        )
        digest.update(b"\0")
        digest.update(
            value.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json_artifact(path, description):
    from utils.experiment_integrity import sha256_file

    path = os.path.abspath(os.fspath(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return path, value, sha256_file(path)


def validate_frozen_diagnostic_keys(
    diagnostic_keys_path,
    train_manifest,
    expected_count=FROZEN_DIAGNOSTIC_KEY_COUNT,
    expected_ordered_key_sha256=(
        FROZEN_DIAGNOSTIC_ORDERED_KEY_SHA256
    ),
):
    from utils.experiment_integrity import (
        hash_ordered_keys,
        manifest_fingerprint,
    )

    with open(
        os.fspath(diagnostic_keys_path), "r", encoding="utf-8"
    ) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            "diagnostic key file must contain a JSON object"
        )
    keys = _load_diagnostic_keys(diagnostic_keys_path)
    ordered_key_sha256 = hash_ordered_keys(keys)
    if (
        len(keys) != int(expected_count)
        or ordered_key_sha256 != expected_ordered_key_sha256
    ):
        raise ValueError(
            "diagnostic key file does not match the frozen holdout"
        )
    if (
        payload.get("ordered_key_sha256") != ordered_key_sha256
        or payload.get("manifest")
        != manifest_fingerprint(train_manifest)
    ):
        raise ValueError(
            "diagnostic key file provenance does not match train manifest"
        )
    return keys


def validate_v2_native_cache_binding(
    provenance,
    native_lmdb,
    selected_keys,
):
    from scripts import build_local_interface_decoys_v2 as builder_v2
    from utils.experiment_integrity import hash_ordered_keys

    if not isinstance(provenance, dict):
        raise ValueError("v2 cache native provenance is missing")
    native_path = os.path.abspath(os.fspath(native_lmdb))
    if os.path.abspath(
        os.fspath(provenance.get("native_lmdb", ""))
    ) != native_path:
        raise ValueError(
            "v2 cache native provenance path does not match"
        )
    environment = builder_v2._open_readonly_lmdb(native_path)
    try:
        native_keys = builder_v2._lmdb_keys(environment)
        actual = {
            "native_key_count": len(native_keys),
            "native_key_sha256": hash_ordered_keys(native_keys),
            "selected_native_record_sha256": (
                builder_v2._hash_ordered_lmdb_records(
                    environment, selected_keys
                )
            ),
            "selected_automorphism_sha256": (
                builder_v2._automorphism_digest_aggregate(
                    environment, selected_keys
                )
            ),
        }
    finally:
        environment.close()
    actual.update(builder_v2._native_metadata_summary(
        Path(native_path)
    ))
    if any(
        provenance.get(name) != value
        for name, value in actual.items()
    ):
        raise ValueError(
            "v2 cache native provenance does not match current records"
        )
    return actual


def invalidate_stage2_ready(output_dir):
    ready_path = os.path.abspath(
        os.path.join(os.fspath(output_dir), "stage2_ready.json")
    )
    if os.path.lexists(ready_path):
        if os.path.isdir(ready_path) and not os.path.islink(
            ready_path
        ):
            raise RuntimeError(
                "stage2_ready.json exists but is not a file"
            )
        os.unlink(ready_path)
    return ready_path


def validate_v2_training_decoy_cache(
    args,
    expected_diagnostic_key_count=FROZEN_DIAGNOSTIC_KEY_COUNT,
    expected_diagnostic_ordered_key_sha256=(
        FROZEN_DIAGNOSTIC_ORDERED_KEY_SHA256
    ),
):
    from scripts import build_local_interface_decoys_v2 as builder_v2
    from utils.experiment_integrity import (
        hash_ordered_keys,
        manifest_fingerprint,
        sha256_file,
    )

    cache_path = os.path.abspath(os.fspath(args.decoy_lmdb))
    environment = builder_v2._open_readonly_lmdb(cache_path)
    try:
        report, _, _ = builder_v2._verify_completed_cache(
            cache_path, environment
        )
    finally:
        environment.close()
    report_path = os.path.join(
        cache_path, builder_v2.REPORT_FILENAME
    )
    if (
        report.get("builder") != builder_v2.BUILDER_NAME
        or report.get("schema_version")
        != builder_v2.DECOY_SCHEMA_VERSION
        or report.get("status") != "complete"
        or report.get("cohort") != "train"
        or report.get("seed") != int(args.seed)
        or report.get("max_samples") is not None
        or os.path.abspath(
            os.fspath(report.get("output_lmdb", ""))
        )
        != cache_path
    ):
        raise ValueError(
            "v2 training decoy cache identity does not match"
        )
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            "v2 training decoy cache provenance is missing"
        )
    current_manifest = manifest_fingerprint(args.train_manifest)
    diagnostic_keys_path = os.path.abspath(
        os.fspath(args.diagnostic_keys)
    )
    native_path = os.path.abspath(os.fspath(args.native_lmdb))
    diagnostic_keys = validate_frozen_diagnostic_keys(
        diagnostic_keys_path,
        args.train_manifest,
        expected_count=expected_diagnostic_key_count,
        expected_ordered_key_sha256=(
            expected_diagnostic_ordered_key_sha256
        ),
    )
    if (
        provenance.get("builder") != builder_v2.BUILDER_NAME
        or provenance.get("schema_version")
        != builder_v2.DECOY_SCHEMA_VERSION
        or provenance.get("cohort") != "train"
        or provenance.get("seed") != int(args.seed)
        or provenance.get("max_samples") is not None
        or provenance.get("manifest") != current_manifest
        or os.path.abspath(
            os.fspath(provenance.get("native_lmdb", ""))
        )
        != native_path
        or os.path.abspath(
            os.fspath(
                provenance.get("diagnostic_keys_path", "")
            )
        )
        != diagnostic_keys_path
        or provenance.get("diagnostic_file_sha256")
        != sha256_file(diagnostic_keys_path)
        or provenance.get("diagnostic_ordered_key_sha256")
        != hash_ordered_keys(diagnostic_keys)
    ):
        raise ValueError(
            "v2 training decoy cache provenance does not match inputs"
        )

    with open(
        os.fspath(args.train_manifest), "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    if manifest.get("split") != "train":
        raise ValueError(
            "v2 training decoy cache requires manifest split=train"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    excluded = set(diagnostic_keys)
    expected_keys = [
        str(sample["lmdb_key"])
        for sample in samples
        if str(sample["lmdb_key"]) not in excluded
    ]
    if (
        provenance.get("selected_key_count") != len(expected_keys)
        or provenance.get("selected_key_sha256")
        != hash_ordered_keys(expected_keys)
    ):
        raise ValueError(
            "v2 training decoy cache does not cover the full train cohort"
        )
    validate_v2_native_cache_binding(
        provenance, native_path, expected_keys
    )
    return {
        "path": report_path,
        "sha256": sha256_file(report_path),
        "report": report,
        "cache_integrity": report["cache_integrity"],
    }


def validate_v2_data_diagnostic(
    args,
    expected_diagnostic_key_count=FROZEN_DIAGNOSTIC_KEY_COUNT,
    expected_diagnostic_ordered_key_sha256=(
        FROZEN_DIAGNOSTIC_ORDERED_KEY_SHA256
    ),
):
    from scripts.evaluate_local_interface_decoys_v2 import (
        EVALUATOR_NAME,
        REPORT_FILENAME,
        ROWS_FILENAME,
    )
    from utils.experiment_integrity import (
        manifest_fingerprint,
        sha256_file,
    )

    report_value = getattr(args, "data_diagnostic_report", None)
    if not report_value:
        raise ValueError(
            "schema-v2 training requires --data-diagnostic-report"
        )
    diagnostic_decoy_value = getattr(
        args, "diagnostic_decoy_lmdb", None
    )
    if not diagnostic_decoy_value:
        raise ValueError(
            "schema-v2 training requires --diagnostic-decoy-lmdb"
        )

    report_path, report, report_sha256 = _load_json_artifact(
        report_value, "v2 data diagnostic report"
    )
    if (
        report.get("evaluator") != EVALUATOR_NAME
        or report.get("schema_version") != 2
        or report.get("status") != "complete"
        or report.get("seed") != int(args.seed)
    ):
        raise ValueError(
            "v2 data diagnostic report identity does not match training"
        )
    from utils.interface_decoy_audit import evaluate_v2_data_gate

    gate = report.get("gate")
    recomputed_gate = evaluate_v2_data_gate(report)
    if gate != recomputed_gate:
        raise ValueError(
            "v2 data diagnostic gate does not match frozen metrics"
        )
    if recomputed_gate["passed"] is not True:
        raise ValueError("v2 data diagnostic gate did not pass")

    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(
            "v2 data diagnostic report inputs are missing"
        )
    current_manifest = manifest_fingerprint(args.train_manifest)
    if inputs.get("manifest") != current_manifest:
        raise ValueError(
            "v2 data diagnostic manifest provenance does not match"
        )

    native_path = os.path.abspath(os.fspath(args.native_lmdb))
    diagnostic_decoy_path = os.path.abspath(
        os.fspath(diagnostic_decoy_value)
    )
    diagnostic_keys_path = os.path.abspath(
        os.fspath(args.diagnostic_keys)
    )
    if (
        os.path.abspath(os.fspath(inputs.get("native_lmdb", "")))
        != native_path
        or os.path.abspath(
            os.fspath(inputs.get("diagnostic_decoy_lmdb", ""))
        )
        != diagnostic_decoy_path
        or os.path.abspath(
            os.fspath(inputs.get("diagnostic_keys_path", ""))
        )
        != diagnostic_keys_path
    ):
        raise ValueError(
            "v2 data diagnostic input paths do not match training"
        )
    if inputs.get("diagnostic_file_sha256") != sha256_file(
        diagnostic_keys_path
    ):
        raise ValueError(
            "v2 data diagnostic key-file provenance does not match"
        )

    cache_report_path = os.path.join(
        diagnostic_decoy_path, "decoy_report.json"
    )
    if (
        not os.path.isfile(cache_report_path)
        or inputs.get("cache_report_sha256")
        != sha256_file(cache_report_path)
    ):
        raise ValueError(
            "v2 diagnostic decoy cache report provenance does not match"
        )
    from scripts import build_local_interface_decoys_v2 as builder_v2

    cache_environment = builder_v2._open_readonly_lmdb(
        diagnostic_decoy_path
    )
    try:
        cache_report, _, _ = builder_v2._verify_completed_cache(
            diagnostic_decoy_path, cache_environment
        )
    finally:
        cache_environment.close()
    cache_provenance = report.get("cache_provenance")
    if (
        not isinstance(cache_provenance, dict)
        or cache_report.get("provenance") != cache_provenance
        or cache_provenance.get("max_samples") is not None
    ):
        raise ValueError(
            "v2 data diagnostic cannot use a reduced max_samples cache"
        )
    if (
        cache_report.get("builder") != builder_v2.BUILDER_NAME
        or cache_report.get("schema_version")
        != builder_v2.DECOY_SCHEMA_VERSION
        or cache_report.get("status") != "complete"
        or cache_report.get("cohort") != "diagnostic"
        or cache_report.get("seed") != int(args.seed)
        or os.path.abspath(
            os.fspath(cache_report.get("output_lmdb", ""))
        )
        != diagnostic_decoy_path
    ):
        raise ValueError(
            "v2 diagnostic cache completion identity does not match"
        )
    diagnostic_keys = validate_frozen_diagnostic_keys(
        diagnostic_keys_path,
        args.train_manifest,
        expected_count=expected_diagnostic_key_count,
        expected_ordered_key_sha256=(
            expected_diagnostic_ordered_key_sha256
        ),
    )
    from utils.experiment_integrity import hash_ordered_keys

    if (
        cache_provenance.get("selected_key_count")
        != len(diagnostic_keys)
        or cache_provenance.get("selected_key_sha256")
        != hash_ordered_keys(diagnostic_keys)
    ):
        raise ValueError(
            "v2 data diagnostic does not cover the full diagnostic key file"
        )
    validate_v2_native_cache_binding(
        cache_provenance, native_path, diagnostic_keys
    )

    artifacts = report.get("artifacts")
    rows = (
        artifacts.get("rows_npz")
        if isinstance(artifacts, dict)
        else None
    )
    if (
        not isinstance(rows, dict)
        or rows.get("filename") != ROWS_FILENAME
        or report.get("artifacts", {})
        .get("report_json", {})
        .get("filename")
        != REPORT_FILENAME
    ):
        raise ValueError(
            "v2 data diagnostic artifact declaration is incomplete"
        )
    rows_path = os.path.join(
        os.path.dirname(report_path), rows["filename"]
    )
    if (
        not os.path.isfile(rows_path)
        or rows.get("sha256") != sha256_file(rows_path)
    ):
        raise ValueError(
            "v2 data diagnostic row artifact provenance does not match"
        )

    return {
        "path": report_path,
        "sha256": report_sha256,
        "report": report,
        "rows_path": rows_path,
        "rows_sha256": rows["sha256"],
        "diagnostic_cache_report_path": cache_report_path,
        "diagnostic_cache_report_sha256": inputs[
            "cache_report_sha256"
        ],
    }


def build_stage1_provenance(
    args,
    dataset,
    initial_state_sha256=None,
    v2_data_diagnostic=None,
    v2_training_cache=None,
):
    from utils.experiment_integrity import (
        hash_ordered_keys,
        manifest_fingerprint,
        sha256_file,
    )

    missing_decoys = list(dataset.stage1_missing_decoy_keys)
    incomplete_banks = list(
        getattr(dataset, "stage1_incomplete_bank_keys", [])
    )
    included_keys = list(
        getattr(
            dataset,
            "stage1_included_keys",
            [
                str(sample["lmdb_key"])
                for sample in dataset.samples
            ],
        )
    )
    provenance = {
        "train_manifest": manifest_fingerprint(
            args.train_manifest
        ),
        "native_lmdb": _artifact_provenance(args.native_lmdb),
        "decoy_lmdb": _artifact_provenance(args.decoy_lmdb),
        "diagnostic_keys": {
            "path": os.path.abspath(args.diagnostic_keys),
            "sha256": sha256_file(args.diagnostic_keys),
        },
        "decoy_schema_version": int(
            getattr(args, "decoy_schema_version", 1)
        ),
        "stage1_sample_count": len(dataset),
        "stage1_included_ordered_key_sha256": getattr(
            dataset,
            "stage1_included_ordered_key_sha256",
            hash_ordered_keys(included_keys),
        ),
        "stage1_missing_decoy_count": len(missing_decoys),
        "stage1_missing_decoy_ordered_key_sha256": (
            hash_ordered_keys(missing_decoys)
        ),
        "stage1_missing_decoy_preview": missing_decoys[:50],
        "stage1_incomplete_bank_count": len(incomplete_banks),
        "stage1_incomplete_bank_ordered_key_sha256": (
            hash_ordered_keys(incomplete_banks)
        ),
        "stage1_incomplete_bank_preview": incomplete_banks[:50],
        "model": {
            "ligand_gnn_depth": int(args.ligand_gnn_depth),
            "ligand_dropout": float(args.ligand_dropout),
            "residue_to_atom_mode": "gated_residual",
            "initial_state_sha256": initial_state_sha256,
        },
        "training": {
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "gradient_clip": float(args.grad_clip),
            "pose_temperature": float(args.temperature),
            "consistency_weight": float(
                args.consistency_weight
            ),
        },
    }
    if provenance["decoy_schema_version"] == 2:
        if not isinstance(v2_data_diagnostic, dict):
            raise ValueError(
                "schema-v2 provenance requires validated data diagnostic"
            )
        if not isinstance(v2_training_cache, dict):
            raise ValueError(
                "schema-v2 provenance requires validated training cache"
            )
        provenance.update(
            {
                "decoy_bank": str(args.decoy_bank),
                "required_complete_banks": list(
                    dataset.require_complete_banks
                ),
                "diagnostic_decoy_lmdb": {
                    **_artifact_provenance(
                        args.diagnostic_decoy_lmdb
                    ),
                    "decoy_report_sha256": v2_data_diagnostic[
                        "diagnostic_cache_report_sha256"
                    ],
                },
                "data_diagnostic": {
                    "path": v2_data_diagnostic["path"],
                    "sha256": v2_data_diagnostic["sha256"],
                    "rows_path": v2_data_diagnostic["rows_path"],
                    "rows_sha256": v2_data_diagnostic[
                        "rows_sha256"
                    ],
                },
                "training_decoy_cache": {
                    "path": v2_training_cache["path"],
                    "sha256": v2_training_cache["sha256"],
                    "cache_integrity": deepcopy(
                        v2_training_cache["cache_integrity"]
                    ),
                },
            }
        )
    return provenance


def _write_metrics_history(path, history):
    path = os.path.abspath(os.fspath(path))
    temporary_path = path + ".tmp"
    fieldnames = (
        "epoch",
        "total_loss",
        "pose_loss",
        "consistency_loss",
        "pose_accuracy",
        "score_margin",
        "gradient_norm",
        "jitter_identity_fallbacks",
        "samples",
        "seconds",
    )
    try:
        with open(
            temporary_path, "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames
            )
            writer.writeheader()
            for row in history:
                writer.writerow(
                    {name: row[name] for name in fieldnames}
                )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _read_metrics_history(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    integer_fields = {
        "epoch",
        "jitter_identity_fallbacks",
        "samples",
    }
    for row in rows:
        output.append(
            {
                key: (
                    int(value)
                    if key in integer_fields
                    else float(value)
                )
                for key, value in row.items()
            }
        )
    return output


def _make_epoch_loader(dataset, args, epoch):
    generator = torch.Generator()
    generator.manual_seed(
        _stable_seed(args.seed, "loader", epoch, "shuffle")
    )
    from data_utils.local_interface_dataset import (
        collate_single_interface,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=int(args.num_workers),
        collate_fn=collate_single_interface,
        persistent_workers=False,
    )


def train_stage1_epoch(
    model,
    dataset,
    optimizer,
    device,
    args,
    epoch,
):
    from data_utils.local_interface_dataset import (
        recursive_to_device,
    )

    dataset.set_epoch(epoch)
    loader = _make_epoch_loader(dataset, args, epoch)
    model.train()
    totals = {
        "total_loss": 0.0,
        "pose_loss": 0.0,
        "consistency_loss": 0.0,
        "pose_accuracy": 0.0,
        "score_margin": 0.0,
        "gradient_norm": 0.0,
        "jitter_identity_fallbacks": 0,
        "jitter_native_clash_samples": 0,
        "jitter_native_clash_identity_fallbacks": 0,
        "samples": 0,
    }
    started = time.time()
    for batch_index, cpu_item in enumerate(loader, start=1):
        jitter = sample_valid_ligand_jitter(
            cpu_item,
            seed=_stable_seed(
                args.seed,
                cpu_item["key"],
                epoch,
                "rigid-jitter",
            ),
        )
        item = recursive_to_device(cpu_item, device)
        jitter_coords = torch.from_numpy(
            np.ascontiguousarray(jitter["coords"])
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        output = stage1_forward_loss(
            model,
            item,
            jitter_coords,
            temperature=args.temperature,
            consistency_weight=args.consistency_weight,
        )
        if not torch.isfinite(output["total_loss"]):
            raise FloatingPointError(
                f"nonfinite Stage-1 loss for {cpu_item['key']}"
            )
        output["total_loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            float(args.grad_clip),
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"nonfinite Stage-1 gradient for {cpu_item['key']}"
            )
        optimizer.step()

        native_score = float(
            output["native_score"].detach().item()
        )
        maximum_decoy = float(
            output["decoy_scores"].detach().max().item()
        )
        totals["total_loss"] += float(
            output["total_loss"].detach().item()
        )
        totals["pose_loss"] += float(
            output["pose_loss"].detach().item()
        )
        totals["consistency_loss"] += float(
            output["consistency_loss"].detach().item()
        )
        totals["pose_accuracy"] += float(
            native_score > maximum_decoy
        )
        totals["score_margin"] += (
            native_score - maximum_decoy
        )
        totals["gradient_norm"] += float(gradient_norm.item())
        totals["jitter_identity_fallbacks"] += int(
            jitter["fallback_identity"]
        )
        totals["jitter_native_clash_samples"] += int(
            jitter["native_hard_clash"]
        )
        totals["jitter_native_clash_identity_fallbacks"] += int(
            jitter["native_hard_clash"]
            and jitter["fallback_identity"]
        )
        totals["samples"] += 1
        if (
            int(args.log_every) > 0
            and batch_index % int(args.log_every) == 0
        ):
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_index,
                        "samples": totals["samples"],
                        "mean_loss": (
                            totals["total_loss"]
                            / totals["samples"]
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if totals["samples"] == 0:
        raise RuntimeError("Stage-1 dataset contains no samples")
    sample_count = totals["samples"]
    metrics = {
        key: (
            value
            if key in {
                "jitter_identity_fallbacks",
                "jitter_native_clash_samples",
                "jitter_native_clash_identity_fallbacks",
                "samples",
            }
            else value / sample_count
        )
        for key, value in totals.items()
    }
    metrics["epoch"] = epoch + 1
    metrics["seconds"] = time.time() - started
    return metrics


def _load_diagnostic_keys(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    keys = payload.get("keys")
    if not isinstance(keys, list):
        raise ValueError(
            "diagnostic key file must contain a keys list"
        )
    keys = [str(key) for key in keys]
    if len(keys) != len(set(keys)):
        raise ValueError("diagnostic keys must be unique")
    return keys


def run_shortcut_diagnostics(
    model,
    train_manifest,
    native_lmdb,
    diagnostic_keys_path,
    output_dir,
    seed,
    device,
):
    from data_utils.local_interface_dataset import (
        NativeInterfaceDataset,
        recursive_to_device,
    )
    from data_utils.local_interface_decoys import (
        DecoyGenerationError,
        apply_rigid_transform,
        generate_decoy_set,
        select_epoch_decoys,
    )
    from utils.experiment_integrity import atomic_write_json
    from utils.interface_diagnostics import (
        extract_shortcut_features,
        fit_shortcut_probes,
    )

    diagnostic_keys = _load_diagnostic_keys(
        diagnostic_keys_path
    )
    dataset = NativeInterfaceDataset(
        train_manifest, native_lmdb
    )
    index_by_key = {
        str(sample["lmdb_key"]): index
        for index, sample in enumerate(dataset.samples)
    }
    unknown = sorted(set(diagnostic_keys) - set(index_by_key))
    if unknown:
        dataset.close()
        raise ValueError(
            "diagnostic keys are absent from train manifest: "
            + ", ".join(unknown[:5])
        )

    features = []
    scores = []
    labels = []
    target_groups = []
    set_ids = []
    contact_mass = []
    failures = []
    model.eval()
    try:
        for key in diagnostic_keys:
            cpu_item = dataset[index_by_key[key]]
            try:
                decoy_set = generate_decoy_set(
                    cpu_item, seed=int(seed)
                )
            except (ValueError, DecoyGenerationError) as error:
                failures.append(
                    {"key": key, "error": str(error)}
                )
                continue
            selected = select_epoch_decoys(
                decoy_set,
                key,
                epoch=0,
                seed=int(seed),
            )
            native_coords = cpu_item[
                "ligand_coords"
            ].detach().cpu().numpy()
            pose_coordinates = [
                native_coords,
                *[
                    apply_rigid_transform(
                        native_coords, entry["transform"]
                    ).astype(np.float32)
                    for entry in selected
                ],
            ]
            moved_item = recursive_to_device(cpu_item, device)
            with torch.no_grad():
                static_state = model.local_path.encode_static(
                    moved_item
                )
                pose_scores = [
                    float(
                        model.local_path.forward_geometry(
                            moved_item,
                            static_state,
                            ligand_coords=torch.from_numpy(
                                np.ascontiguousarray(coordinates)
                            ).to(device=device),
                        ).interface_score.item()
                    )
                    for coordinates in pose_coordinates
                ]
            for pose_index, (coordinates, score) in enumerate(
                zip(pose_coordinates, pose_scores)
            ):
                features.append(
                    extract_shortcut_features(
                        cpu_item, coordinates
                    )
                )
                scores.append(score)
                labels.append(int(pose_index == 0))
                target_groups.append(
                    str(cpu_item["target_signature"])
                )
                set_ids.append(key)
                contact_mass.append(
                    float(decoy_set["native_contact_mass"])
                )
    finally:
        dataset.close()

    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    generation = {
        "requested_keys": len(diagnostic_keys),
        "accepted_keys": len(set(set_ids)),
        "failed_keys": len(failures),
        "failures": failures,
    }
    if not features:
        report = {
            "diagnostic_generation": generation,
            "gate": {
                "passed": False,
                "reasons": ["no_eligible_diagnostic_pose_sets"],
            },
        }
    else:
        try:
            report = fit_shortcut_probes(
                np.asarray(features),
                np.asarray(scores),
                np.asarray(labels),
                np.asarray(target_groups),
                np.asarray(set_ids),
                np.asarray(contact_mass),
            )
        except ValueError as error:
            report = {
                "diagnostic_generation": generation,
                "probe_error": str(error),
                "gate": {
                    "passed": False,
                    "reasons": [
                        "diagnostic_probe_unavailable"
                    ],
                },
            }
        else:
            report["diagnostic_generation"] = generation
            np.savez_compressed(
                os.path.join(
                    output_dir, "diagnostic_pose_rows.npz"
                ),
                features=np.asarray(features),
                interface_scores=np.asarray(scores),
                labels=np.asarray(labels),
                target_groups=np.asarray(target_groups),
                set_ids=np.asarray(set_ids),
                contact_mass=np.asarray(contact_mass),
            )
    atomic_write_json(
        os.path.join(
            output_dir, "shortcut_diagnostic_report.json"
        ),
        report,
    )
    return report


def _v2_model_diagnostic_cohort(data_report, bank, paired_intersection):
    from utils.experiment_integrity import hash_ordered_keys

    cohorts = data_report.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("v2 data diagnostic cohorts are missing")
    cohort_name = (
        "paired_intersection"
        if paired_intersection
        else f"{bank}_full"
    )
    cohort = cohorts.get(cohort_name)
    if not isinstance(cohort, dict):
        raise ValueError(
            f"v2 data diagnostic cohort is missing: {cohort_name}"
        )
    keys = cohort.get("keys")
    bank_report = cohort.get(bank)
    if (
        not isinstance(keys, list)
        or not keys
        or not isinstance(bank_report, dict)
    ):
        raise ValueError(
            f"v2 data diagnostic cohort {cohort_name} is incomplete"
        )
    keys = [str(key) for key in keys]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "v2 model diagnostic keys must be unique"
        )
    expected_hash = cohort.get("ordered_key_sha256")
    if (
        expected_hash is not None
        and expected_hash != hash_ordered_keys(keys)
    ):
        raise ValueError(
            "v2 model diagnostic ordered-key hash does not match"
        )
    return cohort_name, keys, bank_report


def _probe_auroc(bank_report, family, model_name):
    try:
        value = float(bank_report[family][model_name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"v2 data diagnostic {family} {model_name} is missing"
        ) from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"v2 data diagnostic {family} {model_name} is invalid"
        )
    return value


def run_v2_model_diagnostics(
    model,
    train_manifest,
    native_lmdb,
    diagnostic_decoy_lmdb,
    data_report,
    output_dir,
    seed,
    device,
    bank,
    paired_intersection,
    data_report_sha256=None,
):
    from data_utils.local_interface_dataset import (
        NativeInterfaceDataset,
        _load_pickled,
        _open_readonly_lmdb,
        recursive_to_device,
    )
    from data_utils.local_interface_decoys import (
        apply_rigid_transform,
    )
    from data_utils.local_interface_decoys_v2 import (
        BANK_NAMES,
        STRATUM_ORDER,
        select_epoch_decoys_v2,
    )
    from sklearn.metrics import roc_auc_score
    from utils.experiment_integrity import (
        atomic_write_json,
        hash_ordered_keys,
        sha256_file,
    )

    if bank not in BANK_NAMES:
        raise ValueError("invalid v2 diagnostic bank")
    cohort_name, keys, bank_report = _v2_model_diagnostic_cohort(
        data_report, bank, paired_intersection
    )
    dataset = NativeInterfaceDataset(train_manifest, native_lmdb)
    index_by_key = {
        str(sample["lmdb_key"]): index
        for index, sample in enumerate(dataset.samples)
    }
    unknown = [key for key in keys if key not in index_by_key]
    if unknown:
        dataset.close()
        raise ValueError(
            "v2 diagnostic keys are absent from train manifest: "
            + ", ".join(unknown[:5])
        )

    decoy_environment = _open_readonly_lmdb(
        diagnostic_decoy_lmdb
    )
    labels = []
    scores = []
    row_keys = []
    strata = []
    fingerprints = []
    model.eval()
    try:
        for key in keys:
            cpu_item = dataset[index_by_key[key]]
            decoy_set = _load_pickled(
                decoy_environment, key, "v2 diagnostic decoy"
            )
            if decoy_set.get("seed") != int(seed):
                raise ValueError(
                    f"v2 diagnostic decoy seed does not match: {key}"
                )
            if paired_intersection and not all(
                decoy_set.get("banks", {})
                .get(required_bank, {})
                .get("complete")
                is True
                for required_bank in BANK_NAMES
            ):
                raise ValueError(
                    f"paired diagnostic banks are incomplete: {key}"
                )
            selected = select_epoch_decoys_v2(
                decoy_set,
                key=key,
                epoch=0,
                seed=int(seed),
                bank=bank,
            )
            if [entry["stratum"] for entry in selected] != list(
                STRATUM_ORDER
            ):
                raise ValueError(
                    "v2 diagnostic selection must be low/medium/high"
                )
            native_coords = (
                cpu_item["ligand_coords"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            poses = [native_coords]
            poses.extend(
                apply_rigid_transform(
                    native_coords, entry["transform"]
                ).astype(np.float32)
                for entry in selected
            )
            moved_item = recursive_to_device(cpu_item, device)
            with torch.no_grad():
                static_state = model.local_path.encode_static(
                    moved_item
                )
                pose_scores = []
                for coordinates in poses:
                    score = model.local_path.forward_geometry(
                        moved_item,
                        static_state,
                        ligand_coords=torch.from_numpy(
                            np.ascontiguousarray(coordinates)
                        ).to(device=device),
                    ).interface_score
                    if not torch.is_tensor(score) or score.numel() != 1:
                        raise ValueError(
                            "interface diagnostic score must be scalar"
                        )
                    pose_scores.append(float(score.item()))
            if not np.all(np.isfinite(pose_scores)):
                raise RuntimeError(
                    f"nonfinite v2 model diagnostic score: {key}"
                )
            labels.extend((1, 0, 0, 0))
            scores.extend(pose_scores)
            row_keys.extend((key,) * 4)
            strata.extend(("native", *STRATUM_ORDER))
            fingerprints.extend(
                (
                    "native",
                    *[
                        entry["transform_fingerprint"]
                        for entry in selected
                    ],
                )
            )
    finally:
        decoy_environment.close()
        dataset.close()

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    score_sets = score_array.reshape(len(keys), 4)
    interface_auroc = float(
        roc_auc_score(label_array, score_array)
    )
    pose_accuracy = float(
        np.mean(np.argmax(score_sets, axis=1) == 0)
    )
    score_margin = float(
        np.mean(score_sets[:, 0] - np.max(score_sets[:, 1:], axis=1))
    )
    coarse_linear = _probe_auroc(
        bank_report, "coarse", "linear_cv_auroc"
    )
    coarse_nonlinear = _probe_auroc(
        bank_report, "coarse", "nonlinear_cv_auroc"
    )
    heldout_linear = _probe_auroc(
        bank_report, "heldout", "linear_cv_auroc"
    )
    heldout_nonlinear = _probe_auroc(
        bank_report, "heldout", "nonlinear_cv_auroc"
    )
    metrics = {
        "interface_auroc": interface_auroc,
        "pose_accuracy": pose_accuracy,
        "score_margin": score_margin,
        "coarse_linear_cv_auroc": coarse_linear,
        "coarse_nonlinear_cv_auroc": coarse_nonlinear,
        "heldout_linear_cv_auroc": heldout_linear,
        "heldout_nonlinear_cv_auroc": heldout_nonlinear,
        "interface_minus_coarse_linear_auroc": (
            interface_auroc - coarse_linear
        ),
        "interface_minus_coarse_nonlinear_auroc": (
            interface_auroc - coarse_nonlinear
        ),
        "interface_minus_heldout_linear_auroc": (
            interface_auroc - heldout_linear
        ),
        "interface_minus_heldout_nonlinear_auroc": (
            interface_auroc - heldout_nonlinear
        ),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(
            "v2 model diagnostic produced nonfinite metrics"
        )

    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    rows_path = os.path.join(
        output_dir, "model_diagnostic_rows.npz"
    )
    temporary_rows_path = rows_path + ".tmp.npz"
    try:
        np.savez_compressed(
            temporary_rows_path,
            labels=label_array,
            interface_scores=score_array,
            keys=np.asarray(row_keys),
            strata=np.asarray(strata),
            transform_fingerprints=np.asarray(fingerprints),
        )
        os.replace(temporary_rows_path, rows_path)
    finally:
        if os.path.exists(temporary_rows_path):
            os.unlink(temporary_rows_path)

    report = {
        "schema_version": 2,
        "status": "complete",
        "descriptive_only": True,
        "seed": int(seed),
        "bank": bank,
        "cohort": cohort_name,
        "paired_intersection": bool(paired_intersection),
        "key_count": len(keys),
        "ordered_key_sha256": hash_ordered_keys(keys),
        "row_count": len(label_array),
        "metrics": metrics,
        "gate": {
            "passed": True,
            "reasons": [],
            "policy": "finite_metrics_and_complete_artifacts_only",
        },
        "inputs": {
            "train_manifest": os.path.abspath(
                os.fspath(train_manifest)
            ),
            "native_lmdb": os.path.abspath(os.fspath(native_lmdb)),
            "diagnostic_decoy_lmdb": os.path.abspath(
                os.fspath(diagnostic_decoy_lmdb)
            ),
            "data_diagnostic_report_sha256": data_report_sha256,
        },
        "artifacts": {
            "rows_npz": {
                "path": rows_path,
                "sha256": sha256_file(rows_path),
            }
        },
    }
    report_path = os.path.join(
        output_dir, "model_diagnostic_report.json"
    )
    atomic_write_json(report_path, report)
    if not os.path.isfile(report_path):
        raise RuntimeError(
            "v2 model diagnostic report was not published"
        )
    return report


def train_stage1(args, run_diagnostics=True):
    from data_utils.local_interface_dataset import (
        PosePretrainDataset,
    )
    from data_utils.local_interface_decoys_v2 import BANK_NAMES
    from net.interface_first import InterfaceFirstGerNADTA
    from utils.experiment_integrity import (
        atomic_write_json,
        sha256_file,
    )

    if int(args.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if float(args.grad_clip) <= 0:
        raise ValueError("grad_clip must be positive")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ready_path = invalidate_stage2_ready(output_dir)

    decoy_schema_version = int(
        getattr(args, "decoy_schema_version", 1)
    )
    decoy_bank = getattr(args, "decoy_bank", None)
    paired_intersection = bool(
        getattr(args, "require_paired_bank_intersection", False)
    )
    diagnostic_decoy_lmdb = getattr(
        args, "diagnostic_decoy_lmdb", None
    )
    data_diagnostic_path = getattr(
        args, "data_diagnostic_report", None
    )
    if decoy_schema_version == 1:
        if (
            decoy_bank is not None
            or paired_intersection
            or diagnostic_decoy_lmdb is not None
            or data_diagnostic_path is not None
        ):
            raise ValueError(
                "schema-v1 training does not accept v2 decoy options"
            )
        required_banks = ()
        v2_data_diagnostic = None
        v2_training_cache = None
    elif decoy_schema_version == 2:
        if decoy_bank not in BANK_NAMES:
            raise ValueError(
                "schema-v2 training requires --decoy-bank"
            )
        required_banks = (
            tuple(BANK_NAMES)
            if paired_intersection
            else (decoy_bank,)
        )
        v2_training_cache = validate_v2_training_decoy_cache(
            args
        )
        v2_data_diagnostic = validate_v2_data_diagnostic(args)
    else:
        raise ValueError("decoy_schema_version must be 1 or 2")

    _set_random_seed(args.seed)
    device = _resolve_device(args.device)
    dataset = PosePretrainDataset(
        args.train_manifest,
        args.native_lmdb,
        args.decoy_lmdb,
        args.diagnostic_keys,
        seed=args.seed,
        decoy_schema_version=decoy_schema_version,
        decoy_bank=decoy_bank,
        require_complete_banks=required_banks,
    )
    if len(dataset) == 0:
        dataset.close()
        raise RuntimeError(
            "no Stage-1 samples have complete decoy sets"
        )

    model = InterfaceFirstGerNADTA(
        ligand_gnn_depth=args.ligand_gnn_depth,
        ligand_dropout=args.ligand_dropout,
    ).to(device)
    model.configure_stage1()
    initial_state_sha256 = initial_model_state_sha256(model)
    provenance = build_stage1_provenance(
        args,
        dataset,
        initial_state_sha256=initial_state_sha256,
        v2_data_diagnostic=v2_data_diagnostic,
        v2_training_cache=v2_training_cache,
    )
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(
            local_lr=args.lr,
            global_lr=args.lr,
            affinity_lr=args.lr,
        ),
        weight_decay=float(args.weight_decay),
    )
    checkpoint_path = os.path.join(
        output_dir, "training_checkpoint.pth"
    )
    metrics_path = os.path.join(
        output_dir, "metrics_history.csv"
    )
    start_epoch = 0
    history = []
    if args.resume:
        if not os.path.exists(checkpoint_path):
            dataset.close()
            raise FileNotFoundError(
                "resume requested but training checkpoint is absent"
            )
        payload = load_stage1_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            strict=True,
        )
        if payload.get("provenance") != provenance:
            dataset.close()
            raise RuntimeError(
                "Stage-1 resume provenance does not match"
            )
        start_epoch = int(payload["epoch"])
        history = _read_metrics_history(metrics_path)
        if len(history) != start_epoch:
            dataset.close()
            raise RuntimeError(
                "metrics history does not match resume epoch"
            )
    run_config_path = os.path.join(
        output_dir, "run_config.json"
    )
    run_config = {
        "arguments": vars(args),
        "provenance": provenance,
        "device": str(device),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
    }
    atomic_write_json(run_config_path, run_config)
    try:
        for epoch in range(start_epoch, int(args.epochs)):
            metrics = train_stage1_epoch(
                model,
                dataset,
                optimizer,
                device,
                args,
                epoch,
            )
            history.append(metrics)
            _write_metrics_history(metrics_path, history)
            save_stage1_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch + 1,
                metrics=metrics,
                provenance=provenance,
            )
            print(
                json.dumps(metrics, sort_keys=True),
                flush=True,
            )
    finally:
        dataset.close()

    if not history:
        raise RuntimeError(
            "Stage-1 training produced no epoch metrics"
        )
    final_checkpoint = os.path.join(
        output_dir, "stage1_final.pth"
    )
    save_stage1_checkpoint(
        final_checkpoint,
        model,
        optimizer,
        epoch=int(args.epochs),
        metrics=history[-1],
        provenance=provenance,
    )
    result = {
        "final_checkpoint": final_checkpoint,
        "metrics": history[-1],
    }
    if not run_diagnostics:
        return result

    if decoy_schema_version == 2:
        current_training_cache = validate_v2_training_decoy_cache(
            args
        )
        current_data_diagnostic = validate_v2_data_diagnostic(args)
        if (
            current_training_cache["sha256"]
            != v2_training_cache["sha256"]
            or current_training_cache["cache_integrity"]
            != v2_training_cache["cache_integrity"]
            or current_data_diagnostic["sha256"]
            != v2_data_diagnostic["sha256"]
            or current_data_diagnostic["rows_sha256"]
            != v2_data_diagnostic["rows_sha256"]
        ):
            raise RuntimeError(
                "schema-v2 diagnostic or training artifacts changed "
                "during Stage-1"
            )
        required_finite_metrics = (
            "total_loss",
            "pose_loss",
            "consistency_loss",
            "gradient_norm",
        )
        for name in required_finite_metrics:
            try:
                value = float(history[-1][name])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Stage-1 final metric is missing: {name}"
                ) from error
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Stage-1 final metric is nonfinite: {name}"
                )
        report = run_v2_model_diagnostics(
            model,
            train_manifest=args.train_manifest,
            native_lmdb=args.native_lmdb,
            diagnostic_decoy_lmdb=args.diagnostic_decoy_lmdb,
            data_report=v2_data_diagnostic["report"],
            output_dir=output_dir,
            seed=args.seed,
            device=device,
            bank=decoy_bank,
            paired_intersection=paired_intersection,
            data_report_sha256=v2_data_diagnostic["sha256"],
        )
        result["diagnostic_report"] = report
        model_report_path = os.path.join(
            output_dir, "model_diagnostic_report.json"
        )
        model_rows_path = os.path.join(
            output_dir, "model_diagnostic_rows.npz"
        )
        required_artifacts = (
            final_checkpoint,
            metrics_path,
            run_config_path,
            model_report_path,
            model_rows_path,
        )
        missing_artifacts = [
            path for path in required_artifacts
            if not os.path.isfile(path)
        ]
        if missing_artifacts:
            raise RuntimeError(
                "schema-v2 Stage-1 artifacts are incomplete: "
                + ", ".join(missing_artifacts)
            )
        atomic_write_json(
            ready_path,
            {
                "schema_version": 2,
                "status": "complete",
                "completed_epochs": int(args.epochs),
                "stage1_checkpoint": {
                    "path": final_checkpoint,
                    "sha256": sha256_file(final_checkpoint),
                },
                "model_diagnostic_report": {
                    "path": model_report_path,
                    "sha256": sha256_file(model_report_path),
                    "descriptive_only": True,
                },
                "data_diagnostic_report": {
                    "path": v2_data_diagnostic["path"],
                    "sha256": v2_data_diagnostic["sha256"],
                },
                "initial_state_sha256": initial_state_sha256,
                "stage1_included_ordered_key_sha256": provenance[
                    "stage1_included_ordered_key_sha256"
                ],
                "decoy_bank": decoy_bank,
                "required_complete_banks": list(required_banks),
            },
        )
        return result

    report = run_shortcut_diagnostics(
        model,
        args.train_manifest,
        args.native_lmdb,
        args.diagnostic_keys,
        output_dir,
        args.seed,
        device,
    )
    result["diagnostic_report"] = report
    if report["gate"]["passed"]:
        atomic_write_json(
            ready_path,
            {
                "stage1_checkpoint": final_checkpoint,
                "diagnostic_report": os.path.join(
                    output_dir,
                    "shortcut_diagnostic_report.json",
                ),
            },
        )
    else:
        if os.path.exists(ready_path):
            os.unlink(ready_path)
        raise RuntimeError(
            "Stage-1 shortcut diagnostic gate failed: "
            + ", ".join(report["gate"]["reasons"])
        )
    return result


def main():
    args = build_arg_parser().parse_args()
    train_stage1(args)


if __name__ == "__main__":
    main()
