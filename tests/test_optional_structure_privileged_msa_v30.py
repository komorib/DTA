"""CPU regression coverage for V30 train-only privileged-MSA distillation."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dta.optional_structure_privileged_msa_v30.data import (
    DATASET_INDEX,
    TargetAwareBalancedBatchSampler,
    manifest_fingerprint,
)
from dta.optional_structure_privileged_msa_v30.losses import (
    LossConfig,
    privileged_msa_loss,
)
from dta.optional_structure_privileged_msa_v30.model import (
    OptionalStructurePrivilegedMSADTA,
    load_sealed_v22_backbone,
)
from dta.optional_structure_privileged_msa_v30.run_test import (
    _verify_requested_test_configuration,
    run as run_test,
)
from dta.optional_structure_privileged_msa_v30.run_training import (
    _formal_validation_policy,
)
from dta.optional_structure_privileged_msa_v30.selection import (
    apply_sealed_blends,
    select_validation_blends,
)
from dta.optional_structure_privileged_msa_v30.teacher_cache import (
    build_parser as build_teacher_parser,
    build_teacher_cache,
    load_train_teacher_cache,
)
from dta.optional_structure_privileged_msa_v30.training import TrainingConfig, VERSION


def _small_model():
    torch.manual_seed(30)
    return OptionalStructurePrivilegedMSADTA(
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
        maximum_residual=0.50,
    )


def _structured_batch(batch_size=4):
    torch.manual_seed(31)
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
        "structure_ligand_mask": torch.tensor(
            [[1, 1, 1, 0]] * batch_size, dtype=torch.bool
        ),
        "residue_ligand_distance": torch.cdist(pocket_coords, ligand_coords),
    }


def _sequence_ids_covering_all_folds(folds=5, rows_per_sequence=2):
    by_fold = {}
    candidate = 0
    while len(by_fold) < folds:
        sequence_id = "v30-sequence-%d" % candidate
        fold = int(
            hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:16], 16
        ) % folds
        by_fold.setdefault(fold, sequence_id)
        candidate += 1
    return [
        sequence_id
        for fold in range(folds)
        for sequence_id in [by_fold[fold]] * rows_per_sequence
    ]


def _build_toy_teacher_cache(tmp_path):
    sequence_ids = _sequence_ids_covering_all_folds()
    rows = []
    for index, sequence_id in enumerate(sequence_ids):
        rows.append(
            {
                "dataset": "LP_PDBBind",
                "split": "train",
                "sample_id": "LP_PDBBind:toy:%02d" % index,
                "sequence_id": sequence_id,
                "ligand_id": "ligand:%02d" % index,
                "structure_key": "toy-structure-%02d" % index,
                "label": float(index) / 10.0,
            }
        )
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    feature_path = tmp_path / "pretrain_features.npz"
    np.savez_compressed(
        feature_path,
        keys=np.asarray([row["structure_key"] for row in rows]),
        labels=np.asarray([row["label"] for row in rows], dtype=np.float64),
        pretrain_features=np.arange(len(rows) * 6, dtype=np.float32).reshape(len(rows), 6),
    )
    output = tmp_path / "teacher_cache"
    arguments = SimpleNamespace(
        train_manifest=train_manifest,
        msa_train_features=feature_path,
        output=output,
        folds=5,
        components=2,
        estimators=3,
        minimum_leaf=1,
        max_features=1.0,
        n_jobs=1,
        seed=30,
    )
    return rows, train_manifest, output, arguments


def test_teacher_cache_is_oof_train_only_and_rejects_validation_test_lineage(tmp_path):
    rows, train_manifest, output, arguments = _build_toy_teacher_cache(tmp_path)
    assert not hasattr(arguments, "valid_manifest")
    assert not hasattr(arguments, "test_manifest")
    parser_destinations = {action.dest for action in build_teacher_parser()._actions}
    assert "valid_manifest" not in parser_destinations
    assert "test_manifest" not in parser_destinations

    invalid_split_rows = [dict(row) for row in rows]
    invalid_split_rows[0]["split"] = "valid"
    invalid_split_manifest = tmp_path / "invalid_split.jsonl"
    invalid_split_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in invalid_split_rows),
        encoding="utf-8",
    )
    invalid_split_arguments = SimpleNamespace(
        **dict(
            vars(arguments),
            train_manifest=invalid_split_manifest,
            output=tmp_path / "invalid_split_cache",
        )
    )
    with pytest.raises(ValueError, match="train rows only"):
        build_teacher_cache(invalid_split_arguments)

    metadata = build_teacher_cache(arguments)
    assert metadata["labels_scope"] == "train_only"
    assert metadata["valid_manifest_read"] is False
    assert metadata["test_manifest_read"] is False
    assert metadata["teacher_target_regime"] == "sequence_grouped_cross_fitted_oof"
    assert set(metadata["recipe"]["pca_components_by_fold"]) == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }

    loaded = load_train_teacher_cache(output, train_manifest)
    assert set(loaded["predictions_by_sample_id"]) == {
        row["sample_id"] for row in rows
    }
    assert all(np.isfinite(value) for value in loaded["predictions_by_sample_id"].values())

    metadata_path = output / "metadata.json"
    # The cache is deliberately sealed to the exact LP training IDs, labels,
    # and manifest fingerprint, rather than merely being a partial lookup.
    wrong_ids = [dict(row) for row in rows]
    wrong_ids[0]["sample_id"] = "LP_PDBBind:unexpected-id"
    train_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in wrong_ids),
        encoding="utf-8",
    )
    lineage = json.loads(metadata_path.read_text(encoding="utf-8"))
    lineage["train_manifest_sha256"] = manifest_fingerprint(train_manifest)
    metadata_path.write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(ValueError, match="sample IDs do not exactly match"):
        load_train_teacher_cache(output, train_manifest)

    wrong_labels = [dict(row) for row in rows]
    wrong_labels[0]["label"] += 0.125
    train_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in wrong_labels),
        encoding="utf-8",
    )
    lineage["train_manifest_sha256"] = manifest_fingerprint(train_manifest)
    metadata_path.write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(ValueError, match="labels differ"):
        load_train_teacher_cache(output, train_manifest)

    unsafe = json.loads(metadata_path.read_text(encoding="utf-8"))
    unsafe["valid_manifest_read"] = True
    metadata_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(ValueError, match="accessed validation/test inputs"):
        load_train_teacher_cache(output, train_manifest)


def test_oof_teacher_never_uses_its_heldout_sequence_fold_labels(tmp_path):
    rows, train_manifest, output, arguments = _build_toy_teacher_cache(tmp_path)
    build_teacher_cache(arguments)
    with np.load(output / "train_teacher_oof.npz", allow_pickle=False) as archive:
        baseline_ids = archive["sample_ids"].astype(str)
        baseline_folds = np.asarray(archive["fold_ids"], dtype=np.int64)
        baseline_predictions = np.asarray(archive["teacher_predictions"], dtype=np.float64)
    heldout_fold = 0
    heldout_sequences = {
        row["sequence_id"]
        for row in rows
        if int(
            hashlib.sha256(row["sequence_id"].encode("utf-8")).hexdigest()[:16], 16
        )
        % 5
        == heldout_fold
    }
    assert heldout_sequences
    changed_rows = [dict(row) for row in rows]
    for row in changed_rows:
        if row["sequence_id"] in heldout_sequences:
            row["label"] += 100.0
    changed_manifest = tmp_path / "changed_train.jsonl"
    changed_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in changed_rows),
        encoding="utf-8",
    )
    # Feature values remain byte-for-byte unchanged.  The feature archive's
    # labels are adjusted only because V30 validates their train-manifest
    # lineage before fitting the OOF trees.
    changed_features = tmp_path / "changed_pretrain_features.npz"
    with np.load(arguments.msa_train_features, allow_pickle=False) as archive:
        np.savez_compressed(
            changed_features,
            keys=archive["keys"],
            labels=np.asarray([row["label"] for row in changed_rows], dtype=np.float64),
            pretrain_features=archive["pretrain_features"],
        )
    changed_output = tmp_path / "changed_teacher_cache"
    changed_arguments = SimpleNamespace(
        **dict(
            vars(arguments),
            train_manifest=changed_manifest,
            msa_train_features=changed_features,
            output=changed_output,
        )
    )
    build_teacher_cache(changed_arguments)
    with np.load(changed_output / "train_teacher_oof.npz", allow_pickle=False) as archive:
        changed_ids = archive["sample_ids"].astype(str)
        changed_folds = np.asarray(archive["fold_ids"], dtype=np.int64)
        changed_predictions = np.asarray(archive["teacher_predictions"], dtype=np.float64)
    np.testing.assert_array_equal(changed_ids, baseline_ids)
    np.testing.assert_array_equal(changed_folds, baseline_folds)
    heldout = baseline_folds == heldout_fold
    assert heldout.any()
    # Each fold's PCA/tree is fit only on its complement.  Therefore changing
    # all labels in this heldout sequence-fold cannot alter its OOF outputs;
    # predictions in the other folds are intentionally not constrained.
    np.testing.assert_array_equal(
        changed_predictions[heldout], baseline_predictions[heldout]
    )


def test_lp_only_sampler_allows_empty_zero_probability_datasets():
    class Dataset:
        rows = [
            {
                "dataset": "LP_PDBBind",
                "sequence_id": "target-a",
                "ligand_id": "ligand-a",
            },
            {
                "dataset": "LP_PDBBind",
                "sequence_id": "target-a",
                "ligand_id": "ligand-b",
            },
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
        seed=30,
    )
    batches = list(sampler)
    assert len(batches) == 3
    assert all(len(batch) == 4 for batch in batches)
    assert all(set(batch).issubset({0, 1}) for batch in batches)


def test_zero_initialized_dual_heads_preserve_v22_and_backbone_stays_frozen():
    model = _small_model()
    model.freeze_backbone()
    model.train()
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.label_head.weight.requires_grad
    assert model.teacher_head.weight.requires_grad

    output = model(_structured_batch())
    assert output.effective_has_3d.tolist() == [True, True, False, False]
    assert output.contact_logits.shape == (4, 3, 4)
    assert output.distance_logits.shape == (4, 3, 4, 16)
    torch.testing.assert_close(output.prediction, output.backbone_prediction)
    torch.testing.assert_close(output.teacher_prediction, output.backbone_prediction)
    torch.testing.assert_close(output.structure_delta, torch.zeros_like(output.structure_delta))
    torch.testing.assert_close(output.teacher_delta, torch.zeros_like(output.teacher_delta))
    torch.testing.assert_close(
        output.prediction[~output.effective_has_3d],
        output.backbone_prediction[~output.effective_has_3d],
    )


def test_sealed_v22_loader_strict_loads_and_prevents_backbone_gradients(tmp_path):
    source = _small_model()
    with torch.no_grad():
        source.backbone.delta_heads[0][-1].weight.fill_(0.03125)
    checkpoint_path = tmp_path / "v22_structure.pt"
    torch.save(
        {
            "version": "optional-structure-staged-v22",
            "phase": "structure",
            "model_kwargs": source.backbone_kwargs(),
            "model_state": source.backbone.state_dict(),
        },
        checkpoint_path,
    )

    target = _small_model()
    load_sealed_v22_backbone(target, checkpoint_path)
    target.train()
    assert not target.backbone.training
    assert not any(parameter.requires_grad for parameter in target.backbone.parameters())
    for name, value in source.backbone.state_dict().items():
        torch.testing.assert_close(value, target.backbone.state_dict()[name])

    batch = _structured_batch()
    batch.update(
        {
            "teacher_available": torch.tensor([True, True, False, False]),
            "teacher_predictions_normalized": torch.tensor([0.1, -0.2, 0.0, 0.0]),
            "target_group": torch.tensor([0, 0, 1, 1]),
        }
    )
    model_output = target(batch)
    loss, _ = privileged_msa_loss(
        model_output,
        torch.tensor([0.2, -0.3, 0.0, 0.0]),
        batch,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert target.label_head.weight.grad is not None
    assert target.teacher_head.weight.grad is not None
    assert torch.isfinite(target.label_head.weight.grad).all()
    assert torch.isfinite(target.teacher_head.weight.grad).all()
    assert all(parameter.grad is None for parameter in target.backbone.parameters())


def test_teacher_loss_excludes_rows_without_teacher_or_structure():
    prediction = torch.tensor([0.2, 2.0, -3.0], requires_grad=True)
    teacher_prediction = torch.tensor([0.4, 4.0, -5.0], requires_grad=True)
    structure_delta = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    structure_gate = torch.tensor([0.6, 0.7, 0.8], requires_grad=True)
    teacher_delta = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    teacher_mix = torch.tensor([0.5, 0.4, 0.3], requires_grad=True)
    output = SimpleNamespace(
        prediction=prediction,
        teacher_prediction=teacher_prediction,
        backbone_prediction=torch.zeros(3),
        structure_delta=structure_delta,
        structure_gate=structure_gate,
        teacher_delta=teacher_delta,
        teacher_mix=teacher_mix,
        effective_has_3d=torch.tensor([True, True, False]),
    )
    batch = {
        "teacher_available": torch.tensor([True, False, True]),
        "teacher_predictions_normalized": torch.tensor([0.5, 1000.0, -1000.0]),
        "target_group": torch.tensor([0, 0, 0]),
    }
    labels = torch.tensor([0.3, 100.0, -100.0])
    config = LossConfig(
        rank_weight=0.0,
        teacher_rank_weight=0.0,
        backbone_anchor_weight=0.0,
        delta_l2_weight=0.0,
        gate_center_weight=0.0,
        mix_center_weight=0.0,
    )
    loss, components = privileged_msa_loss(output, labels, batch, config)
    changed_labels = labels.clone()
    changed_labels[1:] = torch.tensor([-999.0, 999.0])
    changed_teacher = batch["teacher_predictions_normalized"].clone()
    changed_teacher[1:] = torch.tensor([-777.0, 777.0])
    changed_batch = dict(batch, teacher_predictions_normalized=changed_teacher)
    changed_loss, _ = privileged_msa_loss(output, changed_labels, changed_batch, config)
    torch.testing.assert_close(loss, changed_loss)
    assert components["relative_pairs"] == 0.0
    assert components["teacher_relative_pairs"] == 0.0

    loss.backward()
    # These are the two supervised affinity branches.  The remaining tensors
    # are only regularized when their corresponding non-zero config weights
    # are enabled, so they intentionally need not receive a gradient here.
    for tensor in (prediction, teacher_prediction):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad[1:], torch.zeros_like(tensor.grad[1:]))


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
    return (
        np.asarray(sample_ids),
        np.concatenate(labels),
        np.concatenate(anchor),
        np.concatenate(candidate),
        np.asarray(dataset_index),
    )


def test_validation_selection_requires_nonzero_v30_weight_and_preserves_other_datasets():
    sample_ids, labels, anchor, candidate, dataset_index = _validation_arrays()
    result = select_validation_blends(
        sample_ids,
        labels,
        dataset_index,
        anchor,
        candidate,
        minimum_ci_gain=0.0001,
        minimum_positive_folds=3,
    )
    lp = result["datasets"]["LP_PDBBind"]
    assert result["any_promoted"]
    assert lp["passed"]
    assert lp["effective_v30_weight"] > 0.0
    assert lp["positive_folds"] >= 3
    applied = apply_sealed_blends(dataset_index, anchor, candidate, result)
    np.testing.assert_allclose(applied, result["predictions"])
    for name, index in DATASET_INDEX.items():
        if name == "LP_PDBBind":
            continue
        assert result["datasets"][name]["effective_v30_weight"] == 0.0
        np.testing.assert_array_equal(applied[dataset_index == index], anchor[dataset_index == index])


def test_validation_failure_prevents_v30_test_input_access(tmp_path):
    output = tmp_path / "selection"
    output.mkdir()
    (output / "selection.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "phase": "validation_selection",
                "status": "validation_failed",
                "validation_selected": False,
            }
        ),
        encoding="utf-8",
    )
    arguments = SimpleNamespace(
        selection_output=output,
        test_manifest=tmp_path / "must_not_be_read.jsonl",
        anchor_test=tmp_path / "must_not_be_read_anchor.npz",
        v22_test=tmp_path / "must_not_be_read_v22.npz",
        device="cpu",
        num_workers=0,
        evaluation_batch_size=1,
        limit_test=1,
    )
    with pytest.raises(ValueError, match="validation gate did not pass"):
        run_test(arguments)


class _MustNotReadTestInput:
    def expanduser(self):
        raise AssertionError("test input was accessed before its validation seal")


def test_smoke_selection_is_rejected_before_any_test_path_is_touched(tmp_path):
    output = tmp_path / "smoke_selection"
    output.mkdir()
    (output / "selection.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "phase": "validation_selection",
                "status": "validation_passed",
                "validation_selected": True,
                "formal_validation_selection": False,
            }
        ),
        encoding="utf-8",
    )
    forbidden = _MustNotReadTestInput()
    arguments = SimpleNamespace(
        selection_output=output,
        test_manifest=forbidden,
        anchor_test=forbidden,
        v22_test=forbidden,
        device="cpu",
        num_workers=0,
        evaluation_batch_size=1,
        limit_test=None,
    )
    with pytest.raises(ValueError, match="smoke/relaxed validation"):
        run_test(arguments)


def test_formal_test_request_rejects_limit_and_sealed_path_substitution(tmp_path):
    expected = {
        "test_manifest": str(tmp_path / "expected_test.jsonl"),
        "anchor_test": str(tmp_path / "expected_anchor.npz"),
        "v22_test": str(tmp_path / "expected_v22.npz"),
    }
    selection = {"expected_test_inputs": expected}
    checkpoint = {"config": {"evaluation_batch_size": 4}}
    forbidden = _MustNotReadTestInput()
    limited = SimpleNamespace(
        limit_test=1,
        evaluation_batch_size=None,
        test_manifest=forbidden,
        anchor_test=forbidden,
        v22_test=forbidden,
    )
    with pytest.raises(ValueError, match="does not permit --limit-test"):
        _verify_requested_test_configuration(limited, selection, checkpoint)

    substituted = SimpleNamespace(
        limit_test=None,
        evaluation_batch_size=4,
        test_manifest=tmp_path / "substituted_test.jsonl",
        anchor_test=tmp_path / "expected_anchor.npz",
        v22_test=tmp_path / "expected_v22.npz",
    )
    with pytest.raises(ValueError, match="test_manifest differs"):
        _verify_requested_test_configuration(substituted, selection, checkpoint)


def test_smoke_limits_and_relaxed_geometry_cannot_be_marked_formal():
    arguments = SimpleNamespace(
        limit_train=128,
        limit_valid=64,
        minimum_anchor_ci_gain=0.0005,
        maximum_anchor_rmse_increase=0.003,
        minimum_anchor_spearman_gain=0.0,
        minimum_positive_folds=3,
    )
    policy = _formal_validation_policy(
        arguments,
        TrainingConfig(device="cpu"),
        {"required": False},
    )
    assert not policy["passed"]
    assert not policy["checks"]["full_train_manifest"]
    assert not policy["checks"]["full_validation_manifest"]
    assert not policy["checks"]["v22_archive_batch_geometry"]
