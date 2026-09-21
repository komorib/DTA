import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


def write_predictions(path, keys, labels, predictions):
    with Path(path).open("wb") as handle:
        np.savez(
            handle,
            keys=np.asarray(keys, dtype=np.str_),
            labels=np.asarray(labels, dtype=np.float64),
            predictions=np.asarray(predictions, dtype=np.float64),
        )


def metrics(ci, spearman=0.5, rmse=1.0):
    return {
        "RMSE": float(rmse),
        "MAE": float(rmse * 0.8),
        "Pearson": float(spearman),
        "Spearman": float(spearman),
        "CI": float(ci),
    }


def candidate(
    value,
    *,
    ci,
    variant="raw",
    epoch=1,
    keys=("a", "b", "c"),
    labels=(1.0, 2.0, 3.0),
    predictions=(1.1, 2.1, 2.9),
):
    return {
        "epoch": int(epoch),
        "stage": "high",
        "variant": variant,
        "metrics": metrics(ci),
        "keys": list(keys),
        "labels": np.asarray(labels, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64),
        "model_state_dict": {
            "weight": torch.tensor([float(value)]),
            "counter": torch.tensor([7], dtype=torch.int64),
        },
    }


def write_run(
    run_dir,
    *,
    fusion_type="mlp",
    rank_weight=0.1,
    seed=42,
    raw_ci=0.65,
    ema_ci=0.66,
    full_epochs=0,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)
    run_config = {
        "schema_version": 1,
        "fusion_type": fusion_type,
        "rank_weight": float(rank_weight),
        "seed": int(seed),
        "device": "cuda:0",
        "group_size": 4,
        "num_workers": 0,
        "amp": True,
    }
    provenance = {
        "schema_version": 1,
        "dataset": "LP-PDBBind",
        "initial_checkpoints": {
            "gerna": {"path": "/fixed/gerna.pth", "sha256": "a" * 64},
            "interface": {
                "path": "/fixed/interface.pth",
                "sha256": "b" * 64,
            },
        },
        "splits": {
            "train": {
                "manifest_path": "/fixed/train.json",
                "native_cache_path": "/fixed/native_train",
                "manifest_sha256": "c" * 64,
                "native_metadata_sha256": "d" * 64,
                "ordered_key_sha256": "e" * 64,
                "sample_count": 3,
            },
            "valid": {
                "manifest_path": "/fixed/valid.json",
                "native_cache_path": "/fixed/native_valid",
                "manifest_sha256": "f" * 64,
                "native_metadata_sha256": "1" * 64,
                "ordered_key_sha256": "2" * 64,
                "sample_count": 3,
            },
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config), encoding="utf-8"
    )
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )

    top_three = [
        candidate(1.0, ci=raw_ci, epoch=3),
        candidate(2.0, ci=raw_ci - 0.01, epoch=2),
        candidate(3.0, ci=raw_ci - 0.02, epoch=1),
    ]
    best_ema = candidate(
        4.0,
        ci=ema_ci,
        variant="ema",
        epoch=4,
        predictions=(1.0, 2.0, 3.0),
    )
    checkpoint = {
        "format_version": 1,
        "top_three": top_three,
        "training_state": {
            "complete": True,
            "best_ema": best_ema,
        },
        "training_config": {
            "run_config": run_config,
            "stage_epochs": {
                "head": 2,
                "high": 15,
                "full": int(full_epochs),
            }
        },
        "provenance_hashes": {
            "gerna_checkpoint_sha256": "a" * 64,
            "interface_checkpoint_sha256": "b" * 64,
            "train_manifest_sha256": "c" * 64,
            "train_native_metadata_sha256": "d" * 64,
            "train_ordered_key_sha256": "e" * 64,
            "train_sample_count": 3,
            "valid_manifest_sha256": "f" * 64,
            "valid_native_metadata_sha256": "1" * 64,
            "valid_ordered_key_sha256": "2" * 64,
            "valid_sample_count": 3,
        },
    }
    torch.save(checkpoint, run_dir / "training_checkpoint.pth")
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": top_three[0]["model_state_dict"],
        },
        run_dir / "validation_rank_1.pth",
    )
    write_predictions(
        run_dir / "valid_predictions.npz",
        best_ema["keys"],
        best_ema["labels"],
        best_ema["predictions"],
    )
    return run_dir


class TestPredictionAlignment(unittest.TestCase):
    def test_archive_rejects_duplicate_keys_and_nonfinite_values(self):
        from scripts.select_hybrid_affinity_runs import (
            load_prediction_archive,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.npz"
            write_predictions(
                duplicate,
                ["a", "a"],
                [1.0, 2.0],
                [1.1, 2.1],
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_prediction_archive(duplicate)

            nonfinite = root / "nonfinite.npz"
            write_predictions(
                nonfinite,
                ["a", "b"],
                [1.0, 2.0],
                [1.1, float("nan")],
            )
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_prediction_archive(nonfinite)

    def test_alignment_rejects_key_and_label_mismatch(self):
        from scripts.select_hybrid_affinity_runs import (
            align_prediction_archives,
            load_prediction_archive,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name, keys, labels in (
                ("base", ["a", "b"], [1.0, 2.0]),
                ("keys", ["b", "a"], [1.0, 2.0]),
                ("labels", ["a", "b"], [1.0, 2.1]),
            ):
                path = root / f"{name}.npz"
                write_predictions(path, keys, labels, [1.2, 1.8])
                paths.append(path)
            archives = [load_prediction_archive(path) for path in paths]
            with self.assertRaisesRegex(ValueError, "key"):
                align_prediction_archives(archives[:2])
            with self.assertRaisesRegex(ValueError, "label"):
                align_prediction_archives(
                    [archives[0], archives[2]]
                )


class TestCandidateSelection(unittest.TestCase):
    def test_ordering_uses_ci_spearman_rmse_then_epoch(self):
        from scripts.select_hybrid_affinity_runs import (
            candidate_is_better,
        )

        incumbent = {
            "metrics": metrics(0.7, spearman=0.5, rmse=1.0),
            "epoch": 5,
        }
        cases = (
            {"metrics": metrics(0.700002, 0.1, 2.0), "epoch": 9},
            {"metrics": metrics(0.7000005, 0.6, 2.0), "epoch": 9},
            {"metrics": metrics(0.7, 0.5, 0.9), "epoch": 9},
            {"metrics": metrics(0.7, 0.5, 1.0), "epoch": 4},
        )
        for challenger in cases:
            with self.subTest(challenger=challenger):
                self.assertTrue(
                    candidate_is_better(challenger, incumbent)
                )

    def test_state_average_requires_matching_schema_and_nonfloats(self):
        from scripts.select_hybrid_affinity_runs import (
            average_state_dicts,
        )

        first = {
            "weight": torch.tensor([1.0, 3.0]),
            "counter": torch.tensor([7], dtype=torch.int64),
        }
        second = {
            "weight": torch.tensor([3.0, 5.0]),
            "counter": torch.tensor([7], dtype=torch.int64),
        }
        averaged = average_state_dicts([first, second])
        torch.testing.assert_close(
            averaged["weight"], torch.tensor([2.0, 4.0])
        )
        torch.testing.assert_close(averaged["counter"], first["counter"])

        changed_counter = dict(second)
        changed_counter["counter"] = torch.tensor(
            [8], dtype=torch.int64
        )
        with self.assertRaisesRegex(ValueError, "nonfloating"):
            average_state_dicts([first, changed_counter])

        changed_shape = dict(second)
        changed_shape["weight"] = torch.tensor([[3.0, 5.0]])
        with self.assertRaisesRegex(ValueError, "shape"):
            average_state_dicts([first, changed_shape])

    def test_run_selection_evaluates_average_and_freezes_hashes(self):
        from scripts.select_hybrid_affinity_runs import select_run

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = write_run(Path(temporary) / "mlp_rank0.1")

            def evaluate_average(
                actual_run, run_config, provenance, state_dict
            ):
                self.assertEqual(Path(actual_run), run_dir)
                self.assertEqual(run_config["fusion_type"], "mlp")
                self.assertEqual(provenance["dataset"], "LP-PDBBind")
                torch.testing.assert_close(
                    state_dict["weight"], torch.tensor([2.0])
                )
                return {
                    "keys": ["a", "b", "c"],
                    "labels": np.asarray([1.0, 2.0, 3.0]),
                    "predictions": np.asarray([1.0, 2.0, 3.0]),
                    "metrics": metrics(0.70, 0.70, 0.8),
                }

            selected = select_run(
                run_dir, evaluate_state_dict=evaluate_average
            )
            self.assertEqual(selected["selected_variant"], "top3_average")
            self.assertEqual(selected["validation_metrics"]["CI"], 0.70)
            self.assertEqual(
                selected["run_config"]["fusion_type"], "mlp"
            )
            self.assertEqual(
                selected["provenance_hashes"][
                    "gerna_checkpoint_sha256"
                ],
                "a" * 64,
            )
            self.assertEqual(len(selected["model_sha256"]), 64)
            self.assertEqual(
                len(selected["training_checkpoint_sha256"]), 64
            )
            self.assertTrue(Path(selected["model_path"]).is_file())
            self.assertTrue(Path(selected["prediction_path"]).is_file())
            self.assertTrue(
                (run_dir / "selected_candidate.json").is_file()
            )

    def test_run_selection_requires_saved_validation_predictions(self):
        from scripts.select_hybrid_affinity_runs import select_run

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = write_run(Path(temporary) / "mlp_rank0.1")
            (run_dir / "valid_predictions.npz").unlink()
            with self.assertRaisesRegex(
                FileNotFoundError, "valid_predictions"
            ):
                select_run(
                    run_dir,
                    evaluate_state_dict=lambda *args: self.fail(
                        "average evaluation must not start"
                    ),
                )

    def test_run_selection_rejects_config_changed_after_training(self):
        from scripts.select_hybrid_affinity_runs import select_run

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = write_run(Path(temporary) / "mlp_rank0.1")
            config_path = run_dir / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["rank_weight"] = 0.4
            config_path.write_text(
                json.dumps(config), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "training config"
            ):
                select_run(
                    run_dir,
                    evaluate_state_dict=lambda *args: self.fail(
                        "average evaluation must not start"
                    ),
                )

    def test_default_average_evaluator_moves_model_to_device(self):
        from scripts.select_hybrid_affinity_runs import (
            _default_average_evaluator,
        )

        class FakeOwner:
            def close(self):
                pass

        class FakeDataset:
            def __init__(self):
                self.gerna_dataset = FakeOwner()
                self.interface_dataset = FakeOwner()

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"index": index}

        class FakeModel:
            def __init__(self):
                self.moved_to = None

            def load_state_dict(self, state_dict, strict):
                self.state_dict = state_dict
                self.strict = strict

            def to(self, device):
                self.moved_to = torch.device(device)
                return self

        model = FakeModel()

        def evaluate(actual_model, loader, device, amp_enabled):
            self.assertIs(actual_model, model)
            self.assertEqual(model.moved_to, torch.device("cpu"))
            return {
                "keys": ["a"],
                "labels": np.asarray([1.0]),
                "predictions": np.asarray([1.0]),
                "metrics": metrics(0.5),
            }

        provenance = {
            "splits": {
                "valid": {
                    "manifest_path": "/fixed/valid.json",
                    "native_cache_path": "/fixed/native_valid",
                }
            }
        }
        with mock.patch(
            "dta.train_hybrid_affinity.build_gerna_backbone",
            return_value=object(),
        ), mock.patch(
            "dta.train_hybrid_affinity.build_interface_backbone",
            return_value=object(),
        ), mock.patch(
            "dta.train_hybrid_affinity.build_hybrid_dataset",
            return_value=FakeDataset(),
        ), mock.patch(
            "net.hybrid_affinity.HybridAffinityFusion",
            return_value=object(),
        ), mock.patch(
            "net.hybrid_affinity.HybridAffinityModel",
            return_value=model,
        ), mock.patch(
            "scripts.select_hybrid_affinity_runs.evaluate_hybrid_affinity",
            side_effect=evaluate,
        ):
            _default_average_evaluator(
                "/run",
                {"fusion_type": "mlp"},
                provenance,
                {"weight": torch.tensor([1.0])},
                device="cpu",
            )


    def test_full_stage_selection_skips_high_only_runs(self):
        from scripts.select_hybrid_affinity_runs import select_runs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs"
            write_run(
                run_root / "mlp_rank0.1",
                rank_weight=0.1,
                full_epochs=0,
            )
            write_run(
                run_root / "transformer_rank0.2",
                fusion_type="transformer",
                rank_weight=0.2,
                full_epochs=40,
            )

            def evaluate(run_dir, run_config, provenance, state_dict):
                return {
                    "keys": ["a", "b", "c"],
                    "labels": np.asarray([1.0, 2.0, 3.0]),
                    "predictions": np.asarray([1.0, 2.0, 3.0]),
                    "metrics": metrics(0.70),
                }

            selected = select_runs(
                run_root,
                top_k=1,
                output_path=root / "winner.json",
                evaluate_state_dict=evaluate,
                require_full_stage=True,
            )
            self.assertEqual(len(selected["entries"]), 1)
            self.assertEqual(
                selected["entries"][0]["fusion_type"], "transformer"
            )



class TestBoundedEnsemble(unittest.TestCase):
    def test_bounded_fit_is_deterministic_and_respects_weight_limits(self):
        from scripts.fit_hybrid_affinity_ensemble import (
            fit_bounded_linear_ensemble,
        )

        x = np.linspace(-1.0, 1.0, 41)
        matrix = np.stack([x, -x, x * x], axis=1)
        labels = 3.0 * x + 0.5
        first = fit_bounded_linear_ensemble(
            matrix, labels, l2_penalty=1e-3
        )
        second = fit_bounded_linear_ensemble(
            matrix, labels, l2_penalty=1e-3
        )
        np.testing.assert_allclose(first["weights"], second["weights"])
        self.assertTrue(
            np.all(np.abs(first["weights"]) <= 2.0 + 1e-12)
        )
        self.assertTrue(np.isfinite(first["intercept"]))

    def test_selection_manifest_contains_only_validation_frozen_data(self):
        from scripts.fit_hybrid_affinity_ensemble import (
            fit_ensemble_selection,
        )
        from utils.experiment_integrity import hash_ordered_keys

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys = ["a", "b", "c", "d"]
            labels = [1.0, 2.0, 3.0, 4.0]
            records = []
            for index, seed in enumerate((1, 7, 21, 42, 87)):
                model = root / f"model_{seed}.pth"
                model.write_bytes(f"model:{seed}".encode("ascii"))
                prediction = root / f"valid_{seed}.npz"
                write_predictions(
                    prediction,
                    keys,
                    labels,
                    np.asarray(labels) + 0.1 * (index - 2),
                )
                records.append(
                    {
                        "seed": seed,
                        "fusion_type": "transformer",
                        "rank_weight": 0.2,
                        "model_path": str(model),
                        "prediction_path": str(prediction),
                    }
                )
            output = root / "final_selection.json"
            result = fit_ensemble_selection(records, output)
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(len(result["model_paths"]), 5)
            self.assertEqual(len(result["weights"]), 5)
            self.assertTrue(
                all(abs(weight) <= 2.0 for weight in result["weights"])
            )
            self.assertEqual(
                result["prediction_key_sha256"],
                hash_ordered_keys(keys),
            )
            self.assertEqual(set(result["validation_metrics"]), {
                "RMSE", "MAE", "Pearson", "Spearman", "CI"
            })
            self.assertFalse(
                any("test" in key.lower() for key in result)
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                result,
            )

    def test_parsers_expose_no_test_inputs(self):
        from scripts.fit_hybrid_affinity_ensemble import (
            build_arg_parser as build_ensemble_parser,
        )
        from scripts.select_hybrid_affinity_runs import (
            build_arg_parser as build_selection_parser,
        )

        for parser in (
            build_selection_parser(),
            build_ensemble_parser(),
        ):
            destinations = {action.dest for action in parser._actions}
            self.assertFalse(
                any("test" in destination.lower() for destination in destinations)
            )


if __name__ == "__main__":
    unittest.main()
