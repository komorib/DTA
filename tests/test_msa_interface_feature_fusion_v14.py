import copy
import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from dta.msa_interface_feature_fusion_v14.features import (
    extract_interface_features,
    interface_feature_dimension,
    load_interface_encoders,
)
from dta.msa_interface_feature_fusion_v14.run_features import _run_test_only
from dta.msa_interface_feature_fusion_v14.run_fusion import (
    _family_raw,
    _gate,
    _normalize_fold_ids,
    run,
)
from dta.msa_interface_pretrained_v13.model import (
    MSAInterfaceExpertV13,
    PocketMSAInterfaceV13,
)


def _batch():
    batch_size, depth, pocket, ligand = 2, 4, 5, 3
    generator = torch.Generator().manual_seed(1401)
    pocket_coords = torch.randn(batch_size, pocket, 3, generator=generator)
    ligand_coords = torch.randn(batch_size, ligand, 3, generator=generator)
    distances = torch.cdist(pocket_coords, ligand_coords)
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
        "pocket_aa": torch.randint(0, 21, (batch_size, pocket), generator=generator),
        "pocket_coords": pocket_coords,
        "pocket_chain": torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]),
        "pocket_position": torch.tensor([[1, 7, 12, 2, 9], [1, 7, 12, 2, 9]]),
        "pocket_mask": torch.ones(batch_size, pocket, dtype=torch.bool),
        "ligand_categorical": torch.zeros(batch_size, ligand, 19),
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


def _encoder_kwargs(pair_width=16):
    return {
        "width": 32,
        "layers": 1,
        "heads": 4,
        "pair_factor": 4,
        "pair_width": pair_width,
        "interface_blocks": 2,
        "dropout": 0.0,
    }


def _checkpoint_config(pair_width=16):
    return {
        "msa_width": 32,
        "msa_layers": 1,
        "msa_heads": 4,
        "pair_factor": 4,
        "pair_width": pair_width,
        "interface_blocks": 2,
        "dropout": 0.0,
    }


def test_interface_features_have_expected_shape_and_finite_values():
    encoder = PocketMSAInterfaceV13(**_encoder_kwargs()).eval()
    with torch.no_grad():
        features = extract_interface_features(encoder, _batch())
    assert features.shape == (2, interface_feature_dimension(16))
    assert features.shape[1] == 9 * 16 + 39
    assert torch.isfinite(features).all()


def test_interface_features_ignore_padded_residue_and_ligand_values():
    encoder = PocketMSAInterfaceV13(**_encoder_kwargs()).eval()
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
    changed["ligand_categorical"][:, -1] = 1000.0
    changed["ligand_continuous"][:, -1] = 1000.0
    changed["residue_ligand_distance"][:, -1, :] = 1000.0
    changed["residue_ligand_distance"][:, :, -1] = 1000.0
    with torch.no_grad():
        expected = extract_interface_features(encoder, original)
        actual = extract_interface_features(encoder, changed)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_interface_features_are_rigid_motion_invariant():
    encoder = PocketMSAInterfaceV13(**_encoder_kwargs()).eval()
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
        expected = extract_interface_features(encoder, original)
        actual = extract_interface_features(encoder, moved)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_v13_pretrain_and_affinity_encoders_load_strictly(tmp_path):
    torch.manual_seed(1411)
    pretrain_encoder = PocketMSAInterfaceV13(**_encoder_kwargs())
    affinity_model = MSAInterfaceExpertV13(
        model_width=64,
        layers=1,
        heads=4,
        latent_width=96,
        msa_width=32,
        msa_layers=1,
        msa_heads=4,
        pair_factor=4,
        pair_width=16,
        interface_blocks=2,
        dropout=0.0,
    )
    pretrain_path = tmp_path / "best_pretrain.pt"
    affinity_path = tmp_path / "best.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 3,
            "variant": "ema",
            "metrics": {"loss": 1.25},
            "config": _checkpoint_config(),
            "encoder_state": pretrain_encoder.state_dict(),
        },
        pretrain_path,
    )
    torch.save(
        {
            "schema_version": 1,
            "epoch": 7,
            "variant": "ema",
            "metrics": {"CI": 0.70},
            "config": _checkpoint_config(),
            "model_state": affinity_model.state_dict(),
        },
        affinity_path,
    )
    loaded_pretrain, loaded_affinity, lineage = load_interface_encoders(
        pretrain_path, affinity_path
    )
    assert lineage["pretrain"]["epoch"] == 3
    assert lineage["affinity"]["epoch"] == 7
    assert len(lineage["pretrain"]["sha256"]) == 64
    for name, value in pretrain_encoder.state_dict().items():
        torch.testing.assert_close(loaded_pretrain.state_dict()[name], value)
    for name, value in affinity_model.msa_pair_encoder.state_dict().items():
        torch.testing.assert_close(loaded_affinity.state_dict()[name], value)


def test_feature_families_and_non_contiguous_fold_mapping():
    pretrain = np.arange(24, dtype=np.float64).reshape(6, 4)
    affinity = pretrain + np.linspace(-1.0, 1.0, 24).reshape(6, 4)
    values = {"pretrain_features": pretrain, "affinity_features": affinity}
    np.testing.assert_array_equal(_family_raw(values, "pretrain"), pretrain)
    np.testing.assert_array_equal(_family_raw(values, "affinity"), affinity)
    np.testing.assert_allclose(_family_raw(values, "delta"), affinity - pretrain)
    assert _family_raw(values, "combined").shape == (6, 12)
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


def test_formal_validation_gate_requires_all_conditions():
    anchor = {"CI": 0.70, "Spearman": 0.60, "RMSE": 1.20}
    selection = {
        "residual": {
            "budget": 0.1,
            "metrics": {"CI": 0.702, "Spearman": 0.601, "RMSE": 1.204},
        },
        "summary": {"positive_folds": 4},
    }
    assert _gate(selection, anchor, _gate_args())["passed"]
    for mutation in (
        {"budget": 0.0},
        {"metrics": {"CI": 0.7005, "Spearman": 0.601, "RMSE": 1.204}},
        {"metrics": {"CI": 0.702, "Spearman": 0.599, "RMSE": 1.204}},
        {"metrics": {"CI": 0.702, "Spearman": 0.601, "RMSE": 1.206}},
    ):
        changed = copy.deepcopy(selection)
        changed["residual"].update(mutation)
        assert not _gate(changed, anchor, _gate_args())["passed"]
    changed = copy.deepcopy(selection)
    changed["summary"]["positive_folds"] = 3
    assert not _gate(changed, anchor, _gate_args())["passed"]


def _anchor(path, keys, labels, predictions, fold_ids=None):
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


def _features(path, keys, labels, pretrain, affinity):
    np.savez_compressed(
        path,
        keys=keys,
        labels=labels,
        pretrain_features=pretrain.astype(np.float32),
        affinity_features=affinity.astype(np.float32),
    )


def _fusion_args(anchor, features, output, minimum_gain):
    return Namespace(
        anchor_root=str(anchor),
        feature_root=str(features),
        output=str(output),
        limit=None,
        pca_components=6,
        minimum_validation_ci_gain=minimum_gain,
        minimum_spearman_gain=-1.0,
        maximum_rmse_increase=100.0,
        minimum_positive_folds=0,
        allow_zero_budget=True,
        oof_estimators=4,
        final_estimators=6,
        n_jobs=1,
        seed=1417,
    )


def _validation_artifacts(tmp_path):
    generator = np.random.RandomState(1423)
    count = 50
    keys = np.asarray(["v%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    pretrain = generator.normal(size=(count, 12))
    affinity = pretrain + generator.normal(0.0, 0.5, size=(count, 12))
    anchor_predictions = labels + generator.normal(0.0, 0.5, size=count)
    anchor = tmp_path / "anchor"
    features = tmp_path / "features"
    anchor.mkdir()
    features.mkdir()
    fold_values = np.asarray([3, 8, 14, 21, 34], dtype=np.int64)
    _anchor(
        anchor / "valid_predictions.npz",
        keys,
        labels,
        anchor_predictions,
        fold_values[np.arange(count) % fold_values.size],
    )
    _features(features / "valid_features.npz", keys, labels, pretrain, affinity)
    return generator, anchor, features


def test_validation_failed_does_not_require_or_read_test_artifacts(tmp_path):
    _, anchor, features = _validation_artifacts(tmp_path)
    output = tmp_path / "failed"
    result = run(_fusion_args(anchor, features, output, minimum_gain=2.0))
    assert result["status"] == "validation_failed"
    assert not (output / "test_predictions.npz").exists()
    assert not (output / "model.joblib").exists()
    assert not (anchor / "test_predictions.npz").exists()
    assert not (features / "test_features.npz").exists()


def test_validation_selection_then_final_test_are_separate_stages(tmp_path):
    generator, anchor, features = _validation_artifacts(tmp_path)
    selection_output = tmp_path / "selection"
    selection = run(_fusion_args(anchor, features, selection_output, minimum_gain=-1.0))
    assert selection["status"] == "validation_passed"
    assert not (selection_output / "test_predictions.npz").exists()

    count = 24
    keys = np.asarray(["t%d" % index for index in range(count)])
    labels = generator.normal(6.0, 1.0, size=count)
    pretrain = generator.normal(size=(count, 12))
    affinity = pretrain + generator.normal(0.0, 0.5, size=(count, 12))
    _anchor(
        anchor / "test_predictions.npz",
        keys,
        labels,
        labels + generator.normal(0.0, 0.5, size=count),
    )
    _features(features / "test_features.npz", keys, labels, pretrain, affinity)
    final_output = tmp_path / "final"
    final = run(_fusion_args(anchor, features, final_output, minimum_gain=-1.0))
    assert final["status"] in ("passed", "not_improved")
    assert final["count"] == count
    assert set(np.load(final_output / "test_predictions.npz").files) == {
        "keys",
        "labels",
        "anchor_predictions",
        "predictions",
    }
    assert (final_output / "model.joblib").is_file()


def test_feature_test_only_rejects_wrong_validation_sha_before_loading_test(tmp_path):
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
                "inputs": {"v14_valid_sha256": "0" * 64},
            },
            handle,
        )
    with pytest.raises(ValueError, match="SHA256"):
        _run_test_only(
            Namespace(output=str(output), selection_result=str(selection_path))
        )
