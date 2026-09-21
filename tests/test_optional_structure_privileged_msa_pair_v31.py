"""CPU regression coverage for V31 privileged contact-pair distillation."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dta.optional_structure_privileged_msa_pair_v31.data import DATASET_INDEX, TargetAwareBalancedBatchSampler
from dta.optional_structure_privileged_msa_pair_v31.folds import (
    balanced_sequence_folds,
    build_fold_manifests,
    build_parser as build_fold_parser,
    load_fold_metadata,
)
from dta.optional_structure_privileged_msa_pair_v31.losses import LossConfig, privileged_msa_loss
from dta.optional_structure_privileged_msa_pair_v31.model import (
    OptionalStructurePrivilegedMSAPairDTA,
    load_sealed_v22_backbone,
)
from dta.optional_structure_privileged_msa_pair_v31.run_test import (
    _verify_requested_test_configuration,
    run as run_test,
)
from dta.optional_structure_privileged_msa_pair_v31.run_training import (
    _formal_validation_policy,
    _verify_teacher_v22_lineage,
    _verify_validation_anchor_archive,
)
from dta.optional_structure_privileged_msa_pair_v31.selection import (
    apply_sealed_blends,
    select_validation_blends,
)
from dta.optional_structure_privileged_msa_pair_v31.teacher_cache import (
    build_parser as build_teacher_parser,
    build_teacher_cache,
    load_train_teacher_cache,
)
from dta.optional_structure_privileged_msa_pair_v31.teacher_folds import (
    _validate_pretraining_checkpoint,
    build_parser as build_outer_teacher_parser,
)
from dta.optional_structure_privileged_msa_pair_v31.training import TrainingConfig, VERSION
from dta.optional_structure_staged_v22.data import manifest_fingerprint, sha256_file


def _small_model():
    torch.manual_seed(31)
    return OptionalStructurePrivilegedMSAPairDTA(
        protein_input_width=16,
        width=32,
        heads=4,
        protein_layers=1,
        ligand_layers=1,
        cross_layers=1,
        latent_width=48,
        pair_width=32,
        structure_self_layers=1,
        structure_pair_layers=1,
        structure_token_layers=1,
        structure_top_pairs=6,
        maximum_protein_tokens=16,
        dataset_count=3,
        adapter_width=16,
        dropout=0.0,
        residual_width=16,
        maximum_residual=0.5,
    )


def _structured_batch(batch_size=4):
    torch.manual_seed(32)
    pocket_coords = torch.randn(batch_size, 3, 3)
    ligand_coords = torch.randn(batch_size, 4, 3)
    return {
        "protein_embeddings": torch.randn(batch_size, 5, 16),
        "protein_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "ligand_categorical": torch.randint(0, 2, (batch_size, 4, 8)),
        "ligand_continuous": torch.randn(batch_size, 4, 4),
        "bond_features": torch.randn(batch_size, 4, 4, 7),
        "ligand_mask": torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.bool),
        "dataset_index": torch.zeros(batch_size, dtype=torch.long),
        "has_3d": torch.tensor([True, True] + [False] * (batch_size - 2)),
        "pocket_esm": torch.randn(batch_size, 3, 640),
        "pocket_scalar": torch.randn(batch_size, 3, 15),
        "pocket_aa": torch.randint(0, 21, (batch_size, 3)),
        "pocket_coords": pocket_coords,
        "pocket_mask": torch.tensor([[1, 1, 0]] * batch_size, dtype=torch.bool),
        "structure_ligand_categorical": torch.randn(batch_size, 4, 19),
        "structure_ligand_continuous": torch.randn(batch_size, 4, 2),
        "structure_ligand_coords": ligand_coords,
        "structure_ligand_mask": torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.bool),
        "residue_ligand_distance": torch.cdist(pocket_coords, ligand_coords),
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _toy_rows(label_offset=None):
    # Ten exact sequence groups, each with three ligands.  This is enough to
    # exercise both outer grouping and an inner train/valid partition.
    label_offset = label_offset or {}
    rows = []
    for sequence in range(10):
        for ligand in range(3):
            index = 3 * sequence + ligand
            rows.append(
                {
                    "dataset": "LP_PDBBind",
                    "split": "train",
                    "sample_id": "LP_PDBBind:toy:%03d" % index,
                    "sequence_id": "sequence-%02d" % sequence,
                    "ligand_id": "ligand-%03d" % index,
                    "structure_key": "toy-key-%03d" % index,
                    "pdb_id": "toy-%03d" % index,
                    "label": float(index) / 7.0 + float(label_offset.get(sequence, 0.0)),
                }
            )
    return rows


def _write_msa_manifest(path, rows):
    path.write_text(
        json.dumps(
            {
                "split": "train",
                "samples": [
                    {"lmdb_key": row["structure_key"], "label": row["label"], "pdb_id": row["pdb_id"]}
                    for row in rows
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _feature_archive(path, rows, seed):
    generator = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        keys=np.asarray([row["structure_key"] for row in rows]),
        labels=np.asarray([row["label"] for row in rows], dtype=np.float64),
        predictions=np.asarray([row["label"] - 0.1 for row in rows], dtype=np.float64),
        pair_latents=generator.normal(size=(len(rows), 1024)).astype(np.float32),
    )


def _make_strict_teacher_cache(tmp_path, label_offset=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = _toy_rows(label_offset)
    train_manifest = tmp_path / "train.jsonl"
    msa_manifest = tmp_path / "msa_train.json"
    _write_jsonl(train_manifest, rows)
    _write_msa_manifest(msa_manifest, rows)
    fold_root = tmp_path / "folds"
    fold_args = SimpleNamespace(
        train_manifest=train_manifest,
        msa_train_manifest=msa_manifest,
        output=fold_root,
        folds=5,
        inner_valid_fraction=0.2,
        seed=31,
        limit=None,
    )
    fold_metadata = build_fold_manifests(fold_args)
    teacher_root = tmp_path / "teachers"
    teacher_root.mkdir()
    for fold, info in enumerate(fold_metadata["fold_artifacts"]):
        root = teacher_root / ("outer_fold_%d" % fold)
        root.mkdir()
        fold_dir = fold_root / ("fold_%d" % fold)
        archives = {}
        for name in ("outer_fit", "outer_holdout"):
            manifest = json.loads((fold_dir / (name + ".json")).read_text(encoding="utf-8"))
            source = {row["structure_key"]: row for row in rows}
            partition = [source[sample["lmdb_key"]] for sample in manifest["samples"]]
            path = root / (name + "_features.npz")
            _feature_archive(path, partition, seed=1000 + 17 * fold + len(partition))
            archives[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "count": len(partition),
                "pair_latent_width": 1024,
            }
        result = {
            "schema_version": 1,
            "version": "optional-structure-privileged-msa-pair-outer-teacher-v31",
            "fold": fold,
            "status": "complete",
            "official_valid_manifest_read": False,
            "test_manifest_read": False,
            "outer_holdout_opened_before_selection": False,
            "outer_holdout_labels_used_for_checkpoint_selection": False,
            "manifest_sha256": info["manifest_sha256"],
            "initialization": {"policy": "fold_local_pretraining"},
            "refit": {"enabled": True, "selection_free": True},
            "outer_fit_features": archives["outer_fit"],
            "outer_holdout_features": archives["outer_holdout"],
        }
        (root / "result.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    v22_path = tmp_path / "v22_train.npz"
    np.savez_compressed(
        v22_path,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        labels=np.asarray([row["label"] for row in rows], dtype=np.float64),
        dataset_index=np.zeros(len(rows), dtype=np.int64),
        # The frozen V22 train prediction is deliberately held fixed when a
        # synthetic outer-holdout label is perturbed below.
        predictions=np.asarray(
            [0.11 * int(str(row["sample_id"]).rsplit(":", 1)[1]) for row in rows],
            dtype=np.float64,
        ),
    )
    v22_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": "optional-structure-privileged-msa-pair-v22-train-export-v31",
                "status": "complete",
                "output_sha256": sha256_file(v22_path),
                "train_manifest_sha256": manifest_fingerprint(train_manifest),
                "sealed_v22_checkpoint": "/tmp/toy-v22.pt",
                "sealed_v22_checkpoint_sha256": "toy",
                "official_valid_manifest_read": False,
                "test_manifest_read": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "teacher_cache"
    args = SimpleNamespace(
        train_manifest=train_manifest,
        fold_root=fold_root,
        teacher_root=teacher_root,
        v22_train_predictions=v22_path,
        output=output,
        components=4,
        estimators=5,
        minimum_leaf=1,
        max_features=1.0,
        n_jobs=1,
        seed=31,
        require_full_folds=False,
    )
    return rows, train_manifest, fold_root, output, args


def test_sequence_balanced_outer_folds_and_v18_compatible_manifests(tmp_path):
    rows, train_manifest, fold_root, _, _ = _make_strict_teacher_cache(tmp_path)
    metadata = load_fold_metadata(fold_root, train_manifest)
    assert metadata["outer_row_loads"] == [6, 6, 6, 6, 6]
    assert metadata["official_valid_manifest_read"] is False
    assert metadata["test_manifest_read"] is False
    for info in metadata["fold_artifacts"]:
        outer = json.loads(Path(info["manifests"]["outer_holdout"]).read_text(encoding="utf-8"))
        assert outer["split"] == "train"
        assert len(outer["samples"]) == 6
        assert all("lmdb_key" in sample for sample in outer["samples"])
    folds, assignment, loads, groups = balanced_sequence_folds(rows, 5, 31)
    assert folds.shape == (len(rows),)
    assert loads == [6, 6, 6, 6, 6]
    assert min(groups) > 0
    assert len(assignment) == 10


def test_outer_teacher_cli_has_no_official_valid_test_and_rejects_affinity_checkpoint(tmp_path):
    destinations = {action.dest for action in build_outer_teacher_parser()._actions}
    assert "valid_manifest" not in destinations and "test_manifest" not in destinations
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    allowed = fold_dir / "best_pretrain.pt"
    torch.save({"schema_version": 1, "encoder_state": {"toy": torch.ones(1)}}, allowed)
    lineage = _validate_pretraining_checkpoint(allowed, "fold_local_pretraining", fold_dir)
    assert lineage["policy"] == "fold_local_pretraining"
    forbidden = tmp_path / "historical_affinity.pt"
    torch.save(
        {"schema_version": 1, "encoder_state": {"toy": torch.ones(1)}, "model_state": {"x": torch.ones(1)}},
        forbidden,
    )
    with pytest.raises(ValueError, match="affinity-model checkpoint"):
        _validate_pretraining_checkpoint(forbidden, "historical_self_supervised", fold_dir)


def test_strict_outer_pair_teacher_cache_is_train_only_and_has_sequence_oof(tmp_path):
    rows, train_manifest, fold_root, output, args = _make_strict_teacher_cache(tmp_path)
    destinations = {action.dest for action in build_fold_parser()._actions}
    assert "valid_manifest" not in destinations and "test_manifest" not in destinations
    destinations = {action.dest for action in build_teacher_parser()._actions}
    assert "valid_manifest" not in destinations and "test_manifest" not in destinations
    metadata = build_teacher_cache(args)
    assert metadata["labels_scope"] == "train_only"
    assert metadata["official_valid_manifest_read"] is False
    assert metadata["test_manifest_read"] is False
    assert metadata["teacher_target_regime"] == "strict_outer_sequence_grouped_contact_pair_latent_oof_v22_delta"
    assert metadata["full_ancestry_train_only"] is True
    loaded = load_train_teacher_cache(output, train_manifest)
    assert set(loaded["predictions_by_sample_id"]) == {row["sample_id"] for row in rows}
    with np.load(output / "train_teacher_oof.npz", allow_pickle=False) as archive:
        fold_ids = archive["fold_ids"]
        ids = archive["sample_ids"].astype(str)
        assert set(fold_ids.tolist()) == set(range(5))
    lookup = {row["sample_id"]: row["sequence_id"] for row in rows}
    by_sequence = {}
    for sample_id, fold in zip(ids, fold_ids):
        by_sequence.setdefault(lookup[sample_id], set()).add(int(fold))
    assert all(len(value) == 1 for value in by_sequence.values())


def test_outer_holdout_label_change_does_not_change_that_fold_teacher_output(tmp_path):
    rows, train_manifest, _, output, args = _make_strict_teacher_cache(tmp_path / "baseline")
    baseline = build_teacher_cache(args)
    with np.load(output / "train_teacher_oof.npz", allow_pickle=False) as archive:
        baseline_ids = archive["sample_ids"].astype(str)
        baseline_folds = archive["fold_ids"].copy()
        baseline_prediction = archive["teacher_predictions"].copy()
    changed_sequence = _toy_rows()[0]["sequence_id"]
    changed_rows, changed_manifest, _, changed_output, changed_args = _make_strict_teacher_cache(
        tmp_path / "changed", {0: 100.0}
    )
    changed = build_teacher_cache(changed_args)
    with np.load(changed_output / "train_teacher_oof.npz", allow_pickle=False) as archive:
        changed_ids = archive["sample_ids"].astype(str)
        changed_folds = archive["fold_ids"].copy()
        changed_prediction = archive["teacher_predictions"].copy()
    np.testing.assert_array_equal(changed_ids, baseline_ids)
    np.testing.assert_array_equal(changed_folds, baseline_folds)
    heldout_fold = int(baseline_folds[np.where(baseline_ids == "LP_PDBBind:toy:000")[0][0]])
    heldout = baseline_folds == heldout_fold
    # This entire exact-sequence outer fold never enters that fold's V18-like
    # feature/readout fit.  The helper uses a frozen V22 prediction, so the
    # changed heldout labels are not an input to its own OOF output either.
    np.testing.assert_array_equal(changed_prediction[heldout], baseline_prediction[heldout])
    assert baseline["teacher_target_regime"] == changed["teacher_target_regime"]


def test_teacher_cache_rejects_tampered_fold_or_validation_lineage(tmp_path):
    _, train_manifest, fold_root, output, args = _make_strict_teacher_cache(tmp_path)
    build_teacher_cache(args)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["official_valid_manifest_read"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="official validation/test"):
        load_train_teacher_cache(output, train_manifest)
    metadata["official_valid_manifest_read"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    folds = json.loads((fold_root / "folds.json").read_text(encoding="utf-8"))
    folds["sequence_to_outer_fold"][next(iter(folds["sequence_to_outer_fold"]))] = 99
    (fold_root / "folds.json").write_text(json.dumps(folds), encoding="utf-8")
    with pytest.raises(ValueError, match="fold root changed"):
        load_train_teacher_cache(output, train_manifest)


def test_v31_requires_matching_v22_teacher_and_sealed_validation_archives(tmp_path):
    checkpoint = tmp_path / "v22.pt"
    checkpoint.write_bytes(b"sealed-v22-state")
    teacher = {
        "metadata": {
            "sealed_v22_checkpoint": str(checkpoint),
            "sealed_v22_checkpoint_sha256": sha256_file(checkpoint),
        }
    }
    lineage = _verify_teacher_v22_lineage(teacher, checkpoint)
    assert lineage["matched"] is True
    mismatched = tmp_path / "different-v22.pt"
    mismatched.write_bytes(b"different-state")
    with pytest.raises(ValueError, match="differs from frozen student"):
        _verify_teacher_v22_lineage(teacher, mismatched)

    archive = tmp_path / "valid_predictions.npz"
    np.savez_compressed(archive, sample_ids=np.asarray(["toy"]), predictions=np.asarray([0.0]))
    selection = {
        "version": "optional-structure-simplex-router-v26",
        "phase": "validation_selection_sealed",
        "status": "validation_passed",
        "validation_selected": True,
        "test_inputs_read": False,
        "valid_predictions": str(archive),
        "valid_predictions_sha256": sha256_file(archive),
    }
    selection_path = archive.with_name("selection.json")
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    checked = _verify_validation_anchor_archive(
        archive,
        "V26",
        "optional-structure-simplex-router-v26",
        "validation_selection_sealed",
        "test_inputs_read",
    )
    assert checked["validation_predictions_sha256"] == sha256_file(archive)
    selection["valid_predictions_sha256"] = "tampered"
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        _verify_validation_anchor_archive(
            archive,
            "V26",
            "optional-structure-simplex-router-v26",
            "validation_selection_sealed",
            "test_inputs_read",
        )


def test_lp_only_sampler_allows_empty_zero_probability_datasets():
    class Dataset:
        rows = [
            {"dataset": "LP_PDBBind", "sequence_id": "target-a", "ligand_id": "ligand-a"},
            {"dataset": "LP_PDBBind", "sequence_id": "target-a", "ligand_id": "ligand-b"},
        ]
        dataset_to_indices = {name: [] for name in DATASET_INDEX}

    dataset = Dataset()
    dataset.dataset_to_indices["LP_PDBBind"] = [0, 1]
    sampler = TargetAwareBalancedBatchSampler(
        dataset,
        batch_size=4,
        steps_per_epoch=3,
        probabilities={"LP_PDBBind": 1.0, "DAVIS": 0.0, "KIBA": 0.0},
        pair_batch_fraction=1.0,
        seed=31,
    )
    assert all(set(batch).issubset({0, 1}) for batch in sampler)


def test_cross_gated_adapter_zero_identity_and_frozen_backbone_gradients(tmp_path):
    model = _small_model()
    model.freeze_backbone()
    model.train()
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    output = model(_structured_batch())
    torch.testing.assert_close(output.prediction, output.backbone_prediction)
    torch.testing.assert_close(output.teacher_prediction, output.backbone_prediction)
    assert output.effective_has_3d.tolist() == [True, True, False, False]
    checkpoint_path = tmp_path / "v22.pt"
    torch.save(
        {
            "version": "optional-structure-staged-v22",
            "phase": "structure",
            "model_kwargs": model.backbone_kwargs(),
            "model_state": model.backbone.state_dict(),
        },
        checkpoint_path,
    )
    target = _small_model()
    load_sealed_v22_backbone(target, checkpoint_path)
    batch = _structured_batch()
    batch.update(
        {
            "teacher_available": torch.tensor([True, True, False, False]),
            "teacher_predictions_normalized": torch.tensor([0.1, -0.2, 0.0, 0.0]),
            "target_group": torch.tensor([0, 0, 1, 1]),
        }
    )
    value = target(batch)
    loss, _ = privileged_msa_loss(value, torch.tensor([0.2, -0.3, 0.0, 0.0]), batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert target.label_head.weight.grad is not None
    assert target.teacher_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in target.backbone.parameters())


def test_teacher_loss_excludes_nonstructured_or_unavailable_rows():
    output = SimpleNamespace(
        prediction=torch.tensor([0.2, 2.0, -3.0], requires_grad=True),
        teacher_prediction=torch.tensor([0.4, 4.0, -5.0], requires_grad=True),
        backbone_prediction=torch.zeros(3),
        structure_delta=torch.tensor([0.1, 0.2, 0.3], requires_grad=True),
        structure_gate=torch.tensor([0.6, 0.7, 0.8], requires_grad=True),
        teacher_delta=torch.tensor([0.1, 0.2, 0.3], requires_grad=True),
        teacher_mix=torch.tensor([0.5, 0.4, 0.3], requires_grad=True),
        effective_has_3d=torch.tensor([True, True, False]),
    )
    batch = {
        "teacher_available": torch.tensor([True, False, True]),
        "teacher_predictions_normalized": torch.tensor([0.5, 1000.0, -1000.0]),
        "target_group": torch.tensor([0, 0, 0]),
    }
    config = LossConfig(rank_weight=0.0, teacher_rank_weight=0.0, backbone_anchor_weight=0.0, delta_l2_weight=0.0, gate_center_weight=0.0, mix_center_weight=0.0)
    first, _ = privileged_msa_loss(output, torch.tensor([0.3, 100.0, -100.0]), batch, config)
    changed, _ = privileged_msa_loss(
        output,
        torch.tensor([0.3, -999.0, 999.0]),
        dict(batch, teacher_predictions_normalized=torch.tensor([0.5, -777.0, 777.0])),
        config,
    )
    torch.testing.assert_close(first, changed)


def _validation_arrays(count=120):
    labels, anchor, candidate, sample_ids, dataset_index = [], [], [], [], []
    for name, index in DATASET_INDEX.items():
        target = np.linspace(-2.0, 2.0, count)
        noise = 0.30 * np.sin(np.arange(count) * 1.7 + index)
        labels.append(target)
        anchor.append(target + noise)
        candidate.append(target if name == "LP_PDBBind" else -target)
        sample_ids.extend("%s:%04d" % (name, value) for value in range(count))
        dataset_index.extend([index] * count)
    return np.asarray(sample_ids), np.concatenate(labels), np.concatenate(anchor), np.concatenate(candidate), np.asarray(dataset_index)


def test_validation_selection_and_test_seal_stay_validation_only(tmp_path):
    sample_ids, labels, anchor, candidate, dataset_index = _validation_arrays()
    result = select_validation_blends(sample_ids, labels, dataset_index, anchor, candidate, minimum_ci_gain=0.0001, minimum_positive_folds=3)
    assert result["any_promoted"]
    assert result["datasets"]["LP_PDBBind"]["effective_v31_weight"] > 0.0
    np.testing.assert_allclose(apply_sealed_blends(dataset_index, anchor, candidate, result), result["predictions"])
    output = tmp_path / "failed"
    output.mkdir()
    (output / "selection.json").write_text(json.dumps({"version": VERSION, "phase": "validation_selection", "status": "validation_failed", "validation_selected": False}), encoding="utf-8")
    forbidden = tmp_path / "must_not_read.jsonl"
    with pytest.raises(ValueError, match="validation gate did not pass"):
        run_test(SimpleNamespace(selection_output=output, test_manifest=forbidden, anchor_test=forbidden, v22_test=forbidden, device="cpu", num_workers=0, evaluation_batch_size=1, limit_test=1))


def test_formal_test_rejects_subsets_and_path_substitution_and_smoke_policy():
    expected = {"test_manifest": "/tmp/expected_test.jsonl", "anchor_test": "/tmp/expected_anchor.npz", "v22_test": "/tmp/expected_v22.npz"}
    checkpoint = {"config": {"evaluation_batch_size": 4}}
    with pytest.raises(ValueError, match="does not permit --limit-test"):
        _verify_requested_test_configuration(SimpleNamespace(limit_test=1, evaluation_batch_size=None, test_manifest=Path(expected["test_manifest"]), anchor_test=Path(expected["anchor_test"]), v22_test=Path(expected["v22_test"])), {"expected_test_inputs": expected}, checkpoint)
    with pytest.raises(ValueError, match="test_manifest differs"):
        _verify_requested_test_configuration(SimpleNamespace(limit_test=None, evaluation_batch_size=4, test_manifest=Path("/tmp/substituted.jsonl"), anchor_test=Path(expected["anchor_test"]), v22_test=Path(expected["v22_test"])), {"expected_test_inputs": expected}, checkpoint)
    policy = _formal_validation_policy(
        SimpleNamespace(limit_train=128, limit_valid=64, minimum_anchor_ci_gain=0.0005, maximum_anchor_rmse_increase=0.003, minimum_anchor_spearman_gain=0.0, minimum_positive_folds=3),
        TrainingConfig(device="cpu"),
        {"required": False},
    )
    assert not policy["passed"]
