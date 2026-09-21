import numpy as np
import torch

from dta.rank_locked_calibration_v6.data import collate_rank_locked_calibration
from dta.rank_unlocked_fusion_v7.training import (
    TrainingConfig,
    fit_fixed_epochs,
    predict_checkpoint,
)


class TinyFusionDataset(torch.utils.data.Dataset):
    def __init__(self, count=12):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        label = float(index) / 3.0
        return {
            "key": "key_%d" % index,
            "label": label,
            "base_prediction": label + (0.3 if index % 2 else -0.3),
            "raw_predictions": np.asarray(
                [label + 0.1, label - 0.2, label + 0.2], dtype=np.float32
            ),
            "standardized_predictions": np.asarray(
                [label, label * 0.5, -label * 0.25], dtype=np.float32
            ),
            "available": np.asarray([True, True, True], dtype=np.bool_),
            "latents": (
                np.full(4, label, dtype=np.float32),
                np.full(3, label, dtype=np.float32),
                np.full(5, label, dtype=np.float32),
            ),
        }


def test_fixed_epoch_checkpoint_predicts_every_requested_row():
    dataset = TinyFusionDataset()
    config = TrainingConfig(
        device="cpu",
        batch_size=6,
        maximum_epochs=2,
        patience=1,
        model_width=16,
        transformer_layers=1,
        attention_heads=4,
        latent_widths=(4, 3, 5),
        dropout=0.0,
    )
    indices = np.arange(len(dataset), dtype=np.int64)

    checkpoint = fit_fixed_epochs(
        dataset=dataset,
        indices=indices,
        seed=1,
        epochs=2,
        config=config,
        collate_fn=collate_rank_locked_calibration,
    )
    predictions = predict_checkpoint(
        checkpoint,
        dataset,
        indices,
        config=config,
        collate_fn=collate_rank_locked_calibration,
    )

    assert predictions.shape == (len(dataset),)
    assert np.isfinite(predictions).all()
    assert checkpoint["epoch"] == 2
    assert checkpoint["history"][-1]["loss"] <= checkpoint["history"][0]["loss"] + 1.0
