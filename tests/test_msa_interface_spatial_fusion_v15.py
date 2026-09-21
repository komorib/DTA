import copy
import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from dta.msa_interface_spatial_fusion_v15.features import (
    _group_features,
    extract_spatial_interface_features,
    spatial_feature_dimension,
)
from dta.msa_interface_spatial_fusion_v15.run_features import _run_test_only
from dta.msa_interface_spatial_fusion_v15.run_fusion import (
    _family_raw,
    _gate,
    _normalize_fold_ids,
    run,
)
from dta.msa_interface_pretrained_v13.model import PocketMSAInterfaceV13


def _batch():
    batch_size, depth, pocket, ligand = 2, 4, 5, 4
    generator = torch.Generator().manual_seed(1501)
    pocket_coords = torch.randn(batch_size, pocket, 3, generator=generator)
    ligand_coords = torch.randn(batch_size, ligand, 3, generator=generator)
    distances = torch.cdist(pocket_coords, ligand_coords)
    ligand_categorical = torch.zeros(batch_size, ligand, 19)
    ligand_categorical[:, :, 0] = torch.tensor([6.0, 7.0, 8.0, 17.0])
    return {
        "labels": torch.tensor([5.0, 7.0]),
        "pocket_msa": torch.randn(batch_size, pocket, 50, generator=generator),
        "pocket_msa_tokens": torch.randint(
            0, 22, (batch_size, depth, pocket), generator=generator
        ),
        "pocket_msa_insertions": torch.rand(
            batch_size, depth, pocket, generator=generator
        ),
        "pocket_msa_row_features": torch.rand(
            batch_size, depth, pocket, 2, generator=generator
        ),
        "pocket_msa_row_mask": torch.ones(
            batch_size, depth, pocket, dtype=torch.bool
        ),
        "pocket_esm": torch.randn(batch_size, pocket, 640, generator=generator),
        "pocket_scalar": torch.randn(batch_size, pocket, 15, generator=generator),
        "pocket_aa": torch.tensor([[1, 3, 7, 12, 20], [1, 3, 7, 12, 20]]),
        "pocket_coords": pocket_coords,
        "pocket_chain": torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]),
        "pocket_position": torch.tensor([[1, 7, 12, 2, 9], [1, 7, 12, 2, 9]]),
        "pocket_mask": torch.ones(batch_size, pocket, dtype=torch.bool),
        "ligand_categorical": ligand_categorical,
        "ligand_continuous": torch.randn(
            batch_size, ligand, 2, generator=generator
        ),
        "ligand_coords": ligand_coords,
        "ligand_mask": torch.ones(batch_size, ligand, dtype=torch.bool),
        "residue_ligand_distance": distances,
        "contact_label": distances <= 4.5,
        "contact_mask": torch.ones(
            batch_size, pocket, ligand, dtype=torch.bool
        ),
        "bond_features": torch.zeros(batch_size, ligand, ligand, 7),
    }


def _encoder():
    return PocketMSAInterfaceV13(
        width=32,
        layers=1,
        heads=4,
        pair_factor=4,
        pair_width=16,
        interface_blocks=2,
        dropout=0.0,
    ).eval()


def test_spatial_features_have_expected_shape_and_finite_values():
    with torch.no_grad():
        features = extract_spatial_interface_features(_encoder(), _batch())
    assert features.shape == (2, spatial_feature_dimension(16))
    assert features.shape[1] == 56 * 16 + 96
    assert torch.isfinite(features).all()


def test_spatial_features_ignore_padded_values():
    encoder = _encoder()
    original = _batch()
    original["pocket_mask"][:, -1] = False
    original["pocket_msa_row_mask"][:, :, -1] = False
    original["ligand_mask"][:, -1] = False
    original["contact_mask"][:, -1, :] = False
    original["contact_mask"][:, :, -1] = False
    changed = copy.deepcopy(original)
    changed["pocket_msa_tokens"][:, :, -1] = 21
    changed["pocket_msa_insertions"][:, :, -1] = 1000.0
    changed["pocket_msa_row_features"][:, :, -1] = 1000.0
    changed["pocket_aa"][:, -1] = 6
    changed["ligand_categorical"][:, -1, 0] = 53
    changed["ligand_continuous"][:, -1] = 1000.0
    changed["residue_ligand_distance"][:, -1, :] = 1000.0
    changed["residue_ligand_distance"][:, :, -1] = 1000.0
    with torch.no_grad():
        expected = extract_spatial_interface_features(encoder, original)
        actual = extract_spatial_interface_features(encoder, changed)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_spatial_features_are_rigid_motion_invariant():
    encoder = _encoder()
    original = _batch()
    moved = copy.deepcopy(original)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([8.0, -3.0, 5.0])
    moved["pocket_coords"] = original["pocket_coords"] @ rotation.T + translation
    moved["ligand_coords"] = original["ligand_coords"] @ rotation.T + translation
    moved["residue_ligand_distance"] = torch.cdist(
        moved["pocket_coords"], moved["ligand_coords"]
    )
    with torch.no_grad():
        expected = extract_spatial_interface_features(encoder, original)
        actual = extract_spatial_interface_features(encoder, moved)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_residue_and_ligand_group_occupancies_each_cover_pairs():
    batch = _batch()
    pair = torch.randn(2, 5, 4, 8)
    pair_mask = batch["contact_mask"]
    _, _, occupancy = _group_features(
        pair,
        pair_mask,
        batch["pocket_aa"],
        batch["ligand_categorical"][..., 0].long(),
    )
    assert occupancy.shape == (2, 12)
    torch.testing.assert_close(
        occupancy[:, :6].sum(dim=1), torch.ones(2), rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        occupancy[:, 6:].sum(dim=1), torch.ones(2), rtol=0.0, atol=1e-6
    )


def test_feature_families_and_non_contiguous_folds():
    pretrain = np.arange(24, dtype=np.float64).reshape(6, 4)
    affinity = pretrain + np.linspace(-1.0, 1.0, 24).reshape(6, 4)
    values = {"pretrain_features": pretrain, "affinity_features": affinity}
    np.testing.assert_array_equal(_family_raw(values, "affinity"), affinity)
    np.testing.assert_array_equal(_family_raw(values, "pretrain"), pretrain)
    np.testing.assert_allclose(_family_raw(values, "delta"), affinity - pretrain)
    assert _family_raw(values, "joint").shape == (6, 8)
    folds = _normalize_fold_ids(np.asarray([3, 8, 3, 21, 8, 21]), (6,))
    np.testing.assert_array_equal(folds, np.asarray([0, 1, 0, 2, 1, 2]))


def _gate_args(**overrides):
    values = {
        "allow_zero_budget": False,
        "minimum_validation_ci_gain": 0.001,
        "minimum_spearman_gain": 0.0,
        "maximum_rmse_increase": 0.005,
        "minimum_positive_folds": 4,
    }
    values.update(overrides)
    return Namespace(**values)


def test_formal_gate_compares_against_v14_and_requires_all_conditions():
    benchmark = {"CI": 0.70, "Spearman": 0.60, "RMSE": 1.20}
    selection = {
        "residual": {
            "selected": {
                "budget": 0.35,
                "positive_folds": 4,
                "metrics": {"CI": 0.702, "Spearman": 0.601, "RMSE": 1.204},
            }
        }
    }
    assert _gate(selection, benchmark, _gate_args())["passed"]
    changed = copy.deepcopy(selection)
    changed["residual"]["selected"]["budget"] = 0.0
    assert not _gate(changed, benchmark, _gate_args())["passed"]
    changed = copy.deepcopy(selection)
    changed["residual"]["selected"]["positive_folds"] = 3
    assert not _gate(changed, benchmark, _gate_args())["passed"]
    changed = copy.deepcopy(selection)
    changed["residual"]["selected"]["metrics"]["CI"] = 0.7005
    assert not _gate(changed, benchmark, _gate_args())["passed"]
    changed = copy.deepcopy(selection)
    changed["residual"]["selected"]["metrics"]["Spearman"] = 0.599
    assert not _gate(changed, benchmark, _gate_args())["passed"]
    changed = copy.deepcopy(selection)
    changed["residual"]["selected"]["metrics"]["RMSE"] = 1.206
    assert not _gate(changed, benchmark, _gate_args())["passed"]


def _base(path, keys, labels, predictions, fold_ids=None):
    values = {
        "keys": keys,
        "labels": labels,
        "v11_fusion_predictions": predictions + 0.03,
        "v12_predictions": predictions - 0.04,
        "predictions": predictions,
    }
    if fold_ids is not None:
        values["fold_ids"] = fold_ids
    np.savez_compressed(path, **values)


def _benchmark(path, keys, labels, predictions, fold_ids=None):
    values = {
        "keys": keys,
        "labels": labels,
        "anchor_predictions": predictions + 0.02,
        "predictions": predictions,
    }
    if fold_ids is not None:
        values["fold_ids"] = fold_ids
    np.savez_compressed(path, **values)


def _features(path, keys, labels, pretrain, affinity):
    np.savez_compressed(
        path,
        keys=keys,
        labels=labels,
        pretrain_features=pretrain.astype(np.float32),
        affinity_features=affinity.astype(np.float32),
    )


def _fusion_args(base, benchmark, features, output, minimum_gain):
    return Namespace(
        base_root=str(base),
        benchmark_root=str(benchmark),
        feature_root=str(features),
        output=str(output),
        limit=None,
        pca_components=6,
        minimum_validation_ci_gain=minimum_gain,
        minimum_spearman_gain=-1.0,
        maximum_rmse_increase=100.0,
        minimum_positive_folds=0,
        allow_zero_budget=True,
        oof_estimators=3,
        final_estimators=5,
        n_jobs=1,
        seed=1511,
    )


def _validation_artifacts(tmp_path):
    generator = np.random.RandomState(1517)
    count = 50
    keys = np.asarray(["v%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    pretrain = generator.normal(size=(count, 14))
    affinity = pretrain + generator.normal(0.0, 0.5, size=(count, 14))
    base_predictions = labels + generator.normal(0.0, 0.55, size=count)
    benchmark_predictions = labels + generator.normal(0.0, 0.45, size=count)
    base = tmp_path / "base"
    benchmark = tmp_path / "benchmark"
    features = tmp_path / "features"
    base.mkdir()
    benchmark.mkdir()
    features.mkdir()
    fold_values = np.asarray([3, 8, 14, 21, 34], dtype=np.int64)
    folds = fold_values[np.arange(count) % fold_values.size]
    _base(base / "valid_predictions.npz", keys, labels, base_predictions, folds)
    _benchmark(
        benchmark / "valid_predictions.npz",
        keys,
        labels,
        benchmark_predictions,
        folds,
    )
    _features(features / "valid_features.npz", keys, labels, pretrain, affinity)
    return generator, base, benchmark, features


def test_validation_failed_does_not_read_test(tmp_path):
    _, base, benchmark, features = _validation_artifacts(tmp_path)
    output = tmp_path / "failed"
    result = run(
        _fusion_args(base, benchmark, features, output, minimum_gain=2.0)
    )
    assert result["status"] == "validation_failed"
    assert not (output / "test_predictions.npz").exists()
    assert not (base / "test_predictions.npz").exists()
    assert not (benchmark / "test_predictions.npz").exists()
    assert not (features / "test_features.npz").exists()


def test_validation_then_final_test_are_separate(tmp_path):
    generator, base, benchmark, features = _validation_artifacts(tmp_path)
    selection_output = tmp_path / "selection"
    selection = run(
        _fusion_args(base, benchmark, features, selection_output, minimum_gain=-1.0)
    )
    assert selection["status"] == "validation_passed"
    assert not (selection_output / "test_predictions.npz").exists()

    count = 24
    keys = np.asarray(["t%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    pretrain = generator.normal(size=(count, 14))
    affinity = pretrain + generator.normal(0.0, 0.5, size=(count, 14))
    _base(
        base / "test_predictions.npz",
        keys,
        labels,
        labels + generator.normal(0.0, 0.55, size=count),
    )
    _benchmark(
        benchmark / "test_predictions.npz",
        keys,
        labels,
        labels + generator.normal(0.0, 0.45, size=count),
    )
    _features(features / "test_features.npz", keys, labels, pretrain, affinity)
    final_output = tmp_path / "final"
    final = run(
        _fusion_args(base, benchmark, features, final_output, minimum_gain=-1.0)
    )
    assert final["status"] in ("passed", "not_improved")
    assert final["count"] == count
    assert set(np.load(final_output / "test_predictions.npz").files) == {
        "keys",
        "labels",
        "v12_base_predictions",
        "v14_anchor_predictions",
        "predictions",
    }
    assert (final_output / "model.joblib").is_file()


def test_feature_test_only_rejects_wrong_validation_sha(tmp_path):
    output = tmp_path / "features"
    output.mkdir()
    np.savez_compressed(
        output / "valid_features.npz",
        keys=np.asarray(["a", "b"]),
        labels=np.asarray([1.0, 2.0]),
        pretrain_features=np.zeros((2, 3), dtype=np.float32),
        affinity_features=np.zeros((2, 3), dtype=np.float32),
    )
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "validation_complete"}, handle)
    selection_path = tmp_path / "selection.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "validation_passed",
                "inputs": {"v15_valid_sha256": "0" * 64},
            },
            handle,
        )
    with pytest.raises(ValueError, match="SHA256"):
        _run_test_only(
            Namespace(output=str(output), selection_result=str(selection_path))
        )
