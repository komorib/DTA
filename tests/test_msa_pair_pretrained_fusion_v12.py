from argparse import Namespace

import numpy as np

from dta.msa_pair_pretrained_v12.run_fusion import run


def _anchor(path, keys, labels, predictions, fold_ids=None):
    values = {
        "keys": keys,
        "labels": labels,
        "v10_predictions": predictions + 0.03,
        "v11_predictions": predictions - 0.04,
        "predictions": predictions,
    }
    if fold_ids is not None:
        values["fold_ids"] = fold_ids
    np.savez_compressed(path, **values)


def _expert(path, keys, labels, predictions, latents):
    np.savez_compressed(
        path,
        keys=keys,
        labels=labels,
        predictions=predictions,
        latents=latents,
    )


def test_v12_fusion_runs_end_to_end(tmp_path):
    generator = np.random.RandomState(31)
    valid_count, test_count = 60, 30
    valid_labels = generator.normal(6.5, 1.2, size=valid_count)
    test_labels = generator.normal(6.5, 1.2, size=test_count)
    valid_latents = generator.normal(size=(valid_count, 12))
    test_latents = generator.normal(size=(test_count, 12))
    valid_anchor = valid_labels + generator.normal(0.0, 0.45, size=valid_count)
    test_anchor = test_labels + generator.normal(0.0, 0.45, size=test_count)
    valid_expert = valid_labels + 0.15 * valid_latents[:, 0] + generator.normal(
        0.0, 0.35, size=valid_count
    )
    test_expert = test_labels + 0.15 * test_latents[:, 0] + generator.normal(
        0.0, 0.35, size=test_count
    )
    valid_keys = np.asarray(["v%d" % index for index in range(valid_count)])
    test_keys = np.asarray(["t%d" % index for index in range(test_count)])
    anchor = tmp_path / "v11_fusion"
    v12 = tmp_path / "v12"
    output = tmp_path / "output"
    anchor.mkdir()
    v12.mkdir()
    _anchor(
        anchor / "valid_predictions.npz",
        valid_keys,
        valid_labels,
        valid_anchor,
        np.asarray([2, 4, 7, 9, 11], dtype=np.int64)[
            np.arange(valid_count) % 5
        ],
    )
    _anchor(anchor / "test_predictions.npz", test_keys, test_labels, test_anchor)
    _expert(
        v12 / "valid_predictions.npz",
        valid_keys,
        valid_labels,
        valid_expert,
        valid_latents,
    )
    _expert(
        v12 / "test_predictions.npz",
        test_keys,
        test_labels,
        test_expert,
        test_latents,
    )
    result = run(
        Namespace(
            anchor_root=str(anchor),
            v12_root=str(v12),
            output=str(output),
            limit=None,
            pca_components=6,
            oof_estimators=8,
            final_estimators=12,
            n_jobs=1,
            seed=7,
        )
    )
    assert result["count"] == test_count
    assert result["status"] in ("passed", "not_improved")
    assert (output / "result.json").is_file()
    assert (output / "model.joblib").is_file()
