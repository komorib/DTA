import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dta.rank_fusion_v4.cache import FusionCache
from dta.rank_fusion_v4.evaluate_test import (
    build_arg_parser,
    evaluate_sealed_selection,
)
from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.rank_fusion_v4.model import RankFusionV4
from utils.experiment_integrity import sha256_file


def test_cache(count=8):
    labels = np.linspace(2.0, 9.0, count, dtype=np.float32)
    upper = labels + np.linspace(-0.2, 0.2, count, dtype=np.float32)
    boltz = labels + 0.4
    three = labels - 0.3
    predictions = np.stack((upper, boltz, three), axis=1).astype(np.float32)
    available = np.ones((count, 3), dtype=np.bool_)
    available[1, 1] = False
    available[2, 2] = False
    predictions[~available] = 0.0
    zeros = np.zeros((count, 1), dtype=np.float32)
    return FusionCache(
        keys=tuple(f"t{index}_protein" for index in range(count)),
        labels=labels,
        predictions=predictions,
        standardized_predictions=predictions.copy(),
        available=available,
        upper_bound_latents=zeros.copy(),
        boltz_latents=zeros.copy(),
        three_dprot_latents=zeros.copy(),
        metadata={
            "status": "complete",
            "split": "test",
            "count": count,
            "manifest_sha256": "test-manifest",
            "ordered_key_sha256": "test-keys",
            "normalization_sha256": "normalization",
            "expert_checkpoint_sha256": {
                "upper_bound": "upper-checkpoint",
                "boltz_pairwise": "boltz-checkpoint",
                "three_dprot": "three-checkpoint",
            },
        },
    )


def checkpoint(path, residual):
    model = RankFusionV4(
        (1, 1, 1), model_dim=8, layers=1, heads=2,
        dropout=0.0, drop_path_rate=0.0,
    )
    with torch.no_grad():
        model.residual_head[-1].bias.fill_(residual)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "class": "RankFusionV4",
                "latent_dims": [1, 1, 1],
                "model_dim": 8,
                "layers": 1,
                "heads": 2,
                "dropout": 0.0,
                "drop_path_rate": 0.0,
            },
        },
        path,
    )


def seal_package(root, selected_type, members, cache, *, selected_name=None):
    package = root / (selected_type + "_package")
    package.mkdir()
    checkpoint_hashes = {}
    if selected_type != "single_expert":
        checkpoint_root = package / "checkpoints"
        checkpoint_root.mkdir()
        for member in members:
            path = checkpoint_root / f"{member}.pt"
            checkpoint(path, {"1": 0.1, "7": 0.2, "21": 0.3}[member])
            checkpoint_hashes[member] = sha256_file(path)
    selection = {
        "status": "complete",
        "selected_type": selected_type,
        "selected_name": selected_name or selected_type,
        "members": members,
        "selected_validation_metrics": {
            "CI": 0.7, "MAE": 1.0, "RMSE": 1.3,
            "Pearson": 0.5, "Spearman": 0.5,
        },
        "ensemble_rule": (
            "stable_average_percentile_mean"
            if selected_type == "rank_ensemble"
            else "arithmetic_mean" if selected_type == "raw_ensemble" else None
        ),
        "affine_calibration": (
            {"slope": 2.0, "intercept": 1.0}
            if selected_type == "rank_ensemble"
            else {"slope": 1.0, "intercept": 0.0}
        ),
        "target_manifest_sha256": "valid-manifest",
        "target_ordered_key_sha256": "valid-keys",
        "target_count": 8,
        "normalization_sha256": cache.metadata["normalization_sha256"],
        "expert_checkpoint_sha256": cache.metadata["expert_checkpoint_sha256"],
        "packaged_checkpoint_sha256": checkpoint_hashes,
        "expert_coverage": {},
        "all_candidates": [],
    }
    (package / "selection.json").write_text(json.dumps(selection))
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(package))] = sha256_file(path)
    (package / "seal.json").write_text(
        json.dumps({"status": "sealed", "schema_version": 1, "files": files})
    )
    return package


class RankFusionTestEvaluatorTest(unittest.TestCase):
    def test_selected_single_expert_uses_upper_bound_fallback(self):
        cache = test_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = seal_package(
                root, "single_expert", ["boltz_pairwise"], cache,
                selected_name="boltz_pairwise",
            )
            output = root / "result"
            summary = evaluate_sealed_selection(
                package, cache, output, device="cpu", expected_count=len(cache.keys)
            )
            expected = np.where(
                cache.available[:, 1], cache.predictions[:, 1], cache.predictions[:, 0]
            )
            with np.load(output / "test_predictions_selected.npz") as archive:
                np.testing.assert_allclose(archive["predictions"], expected)
                self.assertEqual(tuple(archive["keys"].tolist()), cache.keys)
            self.assertEqual(summary["test_metrics"], affinity_metrics(cache.labels, expected))
            self.assertEqual(summary["test_count"], len(cache.keys))
            self.assertTrue(summary["output_was_empty"])

    def test_selected_v4_seed_runs_packaged_checkpoint(self):
        cache = test_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = seal_package(root, "v4_seed", ["1"], cache)
            output = root / "result"
            evaluate_sealed_selection(
                package, cache, output, device="cpu", expected_count=len(cache.keys)
            )
            with np.load(output / "test_predictions_selected.npz") as archive:
                np.testing.assert_allclose(
                    archive["predictions"], cache.predictions[:, 0] + 0.1,
                    rtol=1e-5, atol=1e-5,
                )

    def test_raw_and_rank_ensembles_apply_only_sealed_rules(self):
        cache = test_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_package = seal_package(
                root, "raw_ensemble", ["1", "7", "21"], cache
            )
            raw_output = root / "raw-result"
            evaluate_sealed_selection(
                raw_package, cache, raw_output,
                device="cpu", expected_count=len(cache.keys),
            )
            with np.load(raw_output / "test_predictions_selected.npz") as archive:
                np.testing.assert_allclose(
                    archive["predictions"], cache.predictions[:, 0] + 0.2,
                    rtol=1e-5, atol=1e-5,
                )

            rank_package = seal_package(
                root, "rank_ensemble", ["1", "7", "21"], cache
            )
            rank_output = root / "rank-result"
            evaluate_sealed_selection(
                rank_package, cache, rank_output,
                device="cpu", expected_count=len(cache.keys),
            )
            order = np.argsort(cache.predictions[:, 0], kind="mergesort")
            ranks = np.empty(len(order), dtype=np.float64)
            ranks[order] = np.arange(len(order), dtype=np.float64)
            expected = 1.0 + 2.0 * ((ranks + 0.5) / len(order))
            with np.load(rank_output / "test_predictions_selected.npz") as archive:
                np.testing.assert_allclose(archive["predictions"], expected)

    def test_rejects_invalid_seal_lineage_count_keys_and_existing_output(self):
        cache = test_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = seal_package(root, "single_expert", ["upper_bound"], cache)
            (package / "selection.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "seal"):
                evaluate_sealed_selection(
                    package, cache, root / "bad-seal", device="cpu",
                    expected_count=len(cache.keys),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = seal_package(root, "single_expert", ["upper_bound"], cache)
            cache.metadata["normalization_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "normalization"):
                evaluate_sealed_selection(
                    package, cache, root / "bad-lineage", device="cpu",
                    expected_count=len(cache.keys),
                )

        cache = test_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = seal_package(root, "single_expert", ["upper_bound"], cache)
            with self.assertRaisesRegex(ValueError, "expected_count"):
                evaluate_sealed_selection(
                    package, cache, root / "bad-count", device="cpu",
                    expected_count=4751,
                )
            duplicate = test_cache()
            duplicate = FusionCache(
                keys=(duplicate.keys[0],) + duplicate.keys[:-1],
                labels=duplicate.labels, predictions=duplicate.predictions,
                standardized_predictions=duplicate.standardized_predictions,
                available=duplicate.available,
                upper_bound_latents=duplicate.upper_bound_latents,
                boltz_latents=duplicate.boltz_latents,
                three_dprot_latents=duplicate.three_dprot_latents,
                metadata=duplicate.metadata,
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluate_sealed_selection(
                    package, duplicate, root / "duplicate", device="cpu",
                    expected_count=len(duplicate.keys),
                )
            output = root / "existing"
            output.mkdir()
            (output / "summary.json").write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "empty"):
                evaluate_sealed_selection(
                    package, cache, output, device="cpu",
                    expected_count=len(cache.keys),
                )

    def test_cli_defaults_to_full_official_test_count(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--selection", "selection", "--test-cache", "cache",
                "--output", "result", "--device", "cpu",
            ]
        )
        self.assertEqual(args.expected_count, 4751)


if __name__ == "__main__":
    unittest.main()
