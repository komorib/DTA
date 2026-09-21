import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dta.rank_fusion_v4.cache import FusionCache
from dta.rank_fusion_v4.metrics import affinity_metrics
from dta.rank_fusion_v4.selection import (
    rank_percentiles,
    select_validation_package,
    verify_sealed_package,
)


def valid_cache(count=12):
    labels = np.linspace(1.0, 12.0, count, dtype=np.float32)
    predictions = np.stack(
        (labels + 1.0, labels[::-1], labels + 0.5), axis=1
    ).astype(np.float32)
    available = np.ones((count, 3), dtype=np.bool_)
    available[::4, 1] = False
    available[1::5, 2] = False
    predictions[~available] = 0.0
    zeros = np.zeros((count, 1), dtype=np.float32)
    return FusionCache(
        keys=tuple(f"v{i}_protein" for i in range(count)),
        labels=labels,
        predictions=predictions,
        standardized_predictions=predictions.copy(),
        available=available,
        upper_bound_latents=zeros,
        boltz_latents=zeros,
        three_dprot_latents=zeros,
        metadata={
            "status": "complete", "split": "valid", "count": count,
            "manifest_sha256": "valid-manifest",
            "ordered_key_sha256": "valid-keys",
            "normalization_sha256": "normalization",
            "expert_checkpoint_sha256": {
                "upper_bound": "upper", "boltz_pairwise": "boltz", "three_dprot": "three"
            },
        },
    )


def write_run(root, seed, cache, prediction):
    run = root / f"seed{seed}"
    run.mkdir()
    checkpoint = run / "checkpoint_best.pt"
    torch.save({"seed": seed}, checkpoint)
    np.savez_compressed(
        run / "valid_predictions_best.npz",
        keys=np.asarray(cache.keys), labels=cache.labels,
        predictions=np.asarray(prediction, dtype=np.float32),
        expert_predictions=cache.predictions,
        mixture_weights=np.ones((len(cache.keys), 3), dtype=np.float32) / 3,
        rank_residuals=np.zeros(len(cache.keys), dtype=np.float32),
    )
    summary = {
        "status": "complete", "seed": seed,
        "validation_count": len(cache.keys),
        "run_config": {
            "valid_manifest_sha256": cache.metadata["manifest_sha256"],
            "normalization_sha256": cache.metadata["normalization_sha256"],
        },
        "best_metrics": affinity_metrics(cache.labels, prediction),
    }
    (run / "summary.json").write_text(json.dumps(summary))
    return run


class RankFusionSelectionTest(unittest.TestCase):
    def test_rank_percentiles_use_stable_average_ties(self):
        ranks = rank_percentiles(np.asarray([2.0, 1.0, 2.0, 4.0]))
        np.testing.assert_allclose(ranks, [(1.5 + 0.5) / 4, 0.5 / 4, (1.5 + 0.5) / 4, 3.5 / 4])

    def test_selects_validation_candidate_and_seals_complete_package(self):
        cache = valid_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {}
            for seed, noise in zip((1, 7, 21, 42, 87), (0.6, 0.5, 0.4, 0.3, 0.2)):
                prediction = cache.labels + np.sin(np.arange(len(cache.keys))) * noise
                runs[str(seed)] = write_run(root, seed, cache, prediction)
            package = root / "selection"
            selection = select_validation_package(
                cache, runs, package, expected_count=len(cache.keys)
            )
            self.assertEqual(selection["status"], "complete")
            self.assertEqual(selection["target_count"], len(cache.keys))
            self.assertEqual(selection["target_manifest_sha256"], "valid-manifest")
            self.assertIn(selection["selected_type"], {"v4_seed", "raw_ensemble", "rank_ensemble", "single_expert"})
            self.assertEqual(len(selection["all_candidates"]), 10)
            self.assertEqual(selection["expert_coverage"]["boltz_pairwise"]["fallback_count"], 3)
            self.assertTrue((package / "selection.json").is_file())
            self.assertTrue((package / "seal.json").is_file())
            verify_sealed_package(package)
            if selection["selected_type"] != "single_expert":
                for member in selection["members"]:
                    self.assertTrue((package / "checkpoints" / f"{member}.pt").is_file())

    def test_baseline_guard_wins_when_v4_does_not_strictly_improve_ci(self):
        cache = valid_cache()
        cache.predictions[:, 0] = cache.labels
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                str(seed): write_run(root, seed, cache, cache.labels + 0.1)
                for seed in (1, 7, 21, 42, 87)
            }
            selected = select_validation_package(
                cache, runs, root / "selection", expected_count=len(cache.keys)
            )
            self.assertEqual(selected["selected_type"], "single_expert")
            self.assertEqual(selected["members"], ["upper_bound"])

    def test_rejects_run_key_label_or_lineage_mismatch(self):
        cache = valid_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                str(seed): write_run(root, seed, cache, cache.labels + seed * 0.01)
                for seed in (1, 7, 21, 42, 87)
            }
            prediction_path = runs["1"] / "valid_predictions_best.npz"
            with np.load(prediction_path) as archive:
                payload = {name: archive[name] for name in archive.files}
            payload["keys"] = payload["keys"][::-1]
            np.savez_compressed(prediction_path, **payload)
            with self.assertRaisesRegex(ValueError, "keys"):
                select_validation_package(
                    cache, runs, root / "bad", expected_count=len(cache.keys)
                )

    def test_seal_detects_mutation(self):
        cache = valid_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                str(seed): write_run(root, seed, cache, cache.labels + seed * 0.001)
                for seed in (1, 7, 21, 42, 87)
            }
            package = root / "selection"
            select_validation_package(cache, runs, package, expected_count=len(cache.keys))
            (package / "selection.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "seal"):
                verify_sealed_package(package)


    def test_smoke_can_select_two_explicit_seeds_without_ensembles(self):
        cache = valid_cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                str(seed): write_run(root, seed, cache, cache.labels + seed * 0.001)
                for seed in (42, 87)
            }
            selection = select_validation_package(
                cache,
                runs,
                root / "selection",
                expected_count=len(cache.keys),
                expected_seeds=("42", "87"),
            )
            names = {candidate["name"] for candidate in selection["all_candidates"]}
            self.assertEqual(
                names,
                {"upper_bound", "boltz_pairwise", "three_dprot", "seed42", "seed87"},
            )


if __name__ == "__main__":
    unittest.main()
