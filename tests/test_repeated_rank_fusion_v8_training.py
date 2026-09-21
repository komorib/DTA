import tempfile

import numpy as np

from dta.rank_locked_calibration_v6.data import collate_rank_locked_calibration
from dta.rank_unlocked_fusion_v7.training import TrainingConfig
from dta.repeated_rank_fusion_v8.training import fit_repeat
from tests.test_rank_unlocked_fusion_v7_training import TinyFusionDataset


def test_repeat_training_covers_every_oof_row_once():
    dataset = TinyFusionDataset(count=12)
    fold_ids = np.arange(len(dataset), dtype=np.int64) % 2
    config = TrainingConfig(
        device="cpu",
        batch_size=6,
        maximum_epochs=1,
        patience=1,
        model_width=16,
        transformer_layers=1,
        attention_heads=4,
        latent_widths=(4, 3, 5),
        dropout=0.0,
    )

    with tempfile.TemporaryDirectory() as directory:
        result = fit_repeat(
            dataset,
            fold_ids,
            repeat_name="repeat_test",
            fold_seed=17,
            model_seed=23,
            output=directory,
            config=config,
            collate_fn=collate_rank_locked_calibration,
            resume=False,
        )

    assert result["predictions"].shape == (len(dataset),)
    assert np.isfinite(result["predictions"]).all()
    assert np.all(result["coverage"] == 1)
    assert len(result["checkpoints"]) == 2


def test_repeat_resume_rejects_changed_training_configuration():
    dataset = TinyFusionDataset(count=12)
    fold_ids = np.arange(len(dataset), dtype=np.int64) % 2
    base = TrainingConfig(
        device="cpu", batch_size=6, maximum_epochs=1, patience=1,
        model_width=16, transformer_layers=1, attention_heads=4,
        latent_widths=(4, 3, 5), dropout=0.0,
    )
    changed = TrainingConfig(
        device="cpu", batch_size=4, maximum_epochs=1, patience=1,
        model_width=16, transformer_layers=1, attention_heads=4,
        latent_widths=(4, 3, 5), dropout=0.0,
    )

    with tempfile.TemporaryDirectory() as directory:
        fit_repeat(
            dataset, fold_ids, repeat_name="repeat_test", fold_seed=17,
            model_seed=23, output=directory, config=base,
            collate_fn=collate_rank_locked_calibration, resume=False,
        )
        try:
            fit_repeat(
                dataset, fold_ids, repeat_name="repeat_test", fold_seed=17,
                model_seed=23, output=directory, config=changed,
                collate_fn=collate_rank_locked_calibration, resume=True,
            )
        except ValueError as error:
            assert "configuration" in str(error)
        else:
            raise AssertionError("changed V8 resume configuration was accepted")


def test_repeat_resume_rejects_changed_data_lineage():
    dataset = TinyFusionDataset(count=12)
    fold_ids = np.arange(len(dataset), dtype=np.int64) % 2
    config = TrainingConfig(
        device="cpu", batch_size=6, maximum_epochs=1, patience=1,
        model_width=16, transformer_layers=1, attention_heads=4,
        latent_widths=(4, 3, 5), dropout=0.0,
    )

    with tempfile.TemporaryDirectory() as directory:
        fit_repeat(
            dataset, fold_ids, repeat_name="repeat_test", fold_seed=17,
            model_seed=23, data_lineage_sha256="a" * 64,
            output=directory, config=config,
            collate_fn=collate_rank_locked_calibration, resume=False,
        )
        try:
            fit_repeat(
                dataset, fold_ids, repeat_name="repeat_test", fold_seed=17,
                model_seed=23, data_lineage_sha256="b" * 64,
                output=directory, config=config,
                collate_fn=collate_rank_locked_calibration, resume=True,
            )
        except ValueError as error:
            assert "data lineage" in str(error)
        else:
            raise AssertionError("changed V8 resume data lineage was accepted")
