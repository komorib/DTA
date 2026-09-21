import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn


class TinyInterfaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Linear(1, 1, bias=False)
        self.global_encoder = nn.Linear(1, 1, bias=False)
        self.affinity = nn.Linear(1, 1, bias=False)
        self.trainability_seen = []

    @staticmethod
    def _set_trainable(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad = bool(enabled)

    def configure_stage2_warmup(self):
        self._set_trainable(self, False)
        self._set_trainable(self.local, True)
        self._set_trainable(self.affinity, True)

    def configure_stage2_full(self):
        self._set_trainable(self, True)

    def optimizer_parameter_groups(
        self, local_lr, global_lr, affinity_lr=None
    ):
        return [
            {
                "name": "local",
                "params": list(self.local.parameters()),
                "lr": float(local_lr),
            },
            {
                "name": "global",
                "params": list(self.global_encoder.parameters()),
                "lr": float(global_lr),
            },
            {
                "name": "affinity",
                "params": list(self.affinity.parameters()),
                "lr": float(affinity_lr),
            },
        ]

    def forward(self, item):
        if self.training:
            self.trainability_seen.append(
                (
                    self.local.weight.requires_grad,
                    self.global_encoder.weight.requires_grad,
                    self.affinity.weight.requires_grad,
                )
            )
        value = self.local(item["x"])
        value = value + self.global_encoder(item["x"])
        prediction = self.affinity(value)
        return prediction, None, None, None


class StochasticTinyInterfaceModel(TinyInterfaceModel):
    def forward(self, item):
        changed = dict(item)
        changed["x"] = torch.nn.functional.dropout(
            item["x"], p=0.5, training=self.training
        )
        return super().forward(changed)


def make_loader(name):
    return [
        {
            "key": f"{name}_key",
            "x": torch.ones(1, 1),
            "label": torch.tensor(1.0),
        }
    ]


class TestInterfaceFineTuning(unittest.TestCase):
    def test_isolated_flow_warms_up_then_unfreezes_and_tests_once(self):
        import dta.interface_first_training as training

        fit_interface_first = training.fit_interface_first

        model = TinyInterfaceModel()
        model.configure_stage2_full()
        optimizer = torch.optim.SGD(
            model.optimizer_parameter_groups(0.05, 0.01, 0.05)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.5
        )
        train_loader = make_loader("train")
        valid_loader = make_loader("valid")
        test_loader = make_loader("test")
        validation_metrics = [
            {"RMSE": 1.0, "MAE": 0.8, "Pearson": 0.5,
             "Spearman": 0.5, "CI": 0.80},
            {"RMSE": 0.9, "MAE": 0.7, "Pearson": 0.5,
             "Spearman": 0.5, "CI": 0.70},
            {"RMSE": 0.8, "MAE": 0.6, "Pearson": 0.5,
             "Spearman": 0.5, "CI": 0.70},
            {"RMSE": 0.7, "MAE": 0.5, "Pearson": 0.5,
             "Spearman": 0.5, "CI": 0.70},
        ]
        calls = []
        best_weight = {}

        def fake_evaluate(current_model, loader, device):
            if loader is valid_loader:
                index = sum(name == "valid" for name, _ in calls)
                calls.append(("valid", float(current_model.local.weight)))
                if index == 0:
                    best_weight["value"] = float(
                        current_model.local.weight
                    )
                return {
                    "metrics": validation_metrics[index],
                    "keys": np.asarray(["valid_key"]),
                    "labels": np.asarray([1.0], dtype=np.float32),
                    "predictions": np.asarray(
                        [0.5 + index], dtype=np.float32
                    ),
                }
            self.assertIs(loader, test_loader)
            calls.append(("test", float(current_model.local.weight)))
            return {
                "metrics": {
                    "RMSE": 1.1,
                    "MAE": 0.9,
                    "Pearson": 0.4,
                    "Spearman": 0.4,
                    "CI": 0.65,
                },
                "keys": np.asarray(["test_key"]),
                "labels": np.asarray([2.0], dtype=np.float32),
                "predictions": np.asarray([1.5], dtype=np.float32),
            }

        with tempfile.TemporaryDirectory() as temporary:
            rank_loss = mock.Mock(
                side_effect=AssertionError("rank loss must not run"),
            )
            with mock.patch.object(
                training,
                "pairwise_rank_loss",
                rank_loss,
                create=True,
            ):
                result = fit_interface_first(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=torch.device("cpu"),
                    output_dir=temporary,
                    epochs=4,
                    warmup_epochs=3,
                    grad_clip=1.0,
                    provenance={"run": "unit-test"},
                    evaluate_fn=fake_evaluate,
                )
            rank_loss.assert_not_called()

            self.assertEqual(
                model.trainability_seen[:3],
                [(True, False, True)] * 3,
            )
            self.assertEqual(
                model.trainability_seen[3], (True, True, True)
            )
            self.assertEqual(
                [name for name, _ in calls],
                ["valid", "valid", "valid", "valid", "test"],
            )
            self.assertAlmostEqual(
                calls[-1][1], best_weight["value"], places=7
            )
            self.assertEqual(result["best_epoch"], 1)

            valid = np.load(
                Path(temporary) / "valid_predictions.npz"
            )
            test = np.load(
                Path(temporary) / "test_predictions.npz"
            )
            self.assertEqual(
                set(valid.files), {"keys", "labels", "predictions"}
            )
            self.assertEqual(
                set(test.files), {"keys", "labels", "predictions"}
            )
            self.assertEqual(valid["keys"].tolist(), ["valid_key"])
            self.assertEqual(test["keys"].tolist(), ["test_key"])

    def test_evaluate_returns_keyed_regression_predictions(self):
        from dta.interface_first_training import (
            evaluate_interface_first,
        )

        model = TinyInterfaceModel()
        output = evaluate_interface_first(
            model, make_loader("valid"), torch.device("cpu")
        )

        self.assertEqual(output["keys"].tolist(), ["valid_key"])
        self.assertEqual(output["labels"].shape, (1,))
        self.assertEqual(output["predictions"].shape, (1,))
        self.assertEqual(
            set(output["metrics"]),
            {"RMSE", "MAE", "Pearson", "Spearman", "CI"},
        )

    def test_resume_matches_uninterrupted_shuffle_and_dropout_training(self):
        from dta.interface_first_training import fit_interface_first

        samples = [
            {
                "key": "a",
                "x": torch.tensor([[1.0]]),
                "label": torch.tensor(0.5),
            },
            {
                "key": "b",
                "x": torch.tensor([[2.0]]),
                "label": torch.tensor(1.5),
            },
            {
                "key": "c",
                "x": torch.tensor([[3.0]]),
                "label": torch.tensor(2.5),
            },
        ]

        def loader():
            generator = torch.Generator()
            generator.manual_seed(909)
            return torch.utils.data.DataLoader(
                samples,
                batch_size=None,
                shuffle=True,
                generator=generator,
            )

        def evaluator(model, current_loader, device):
            return {
                "metrics": {
                    "RMSE": 1.0,
                    "MAE": 1.0,
                    "Pearson": 0.0,
                    "Spearman": 0.0,
                    "CI": 0.5,
                },
                "keys": np.asarray(["x"]),
                "labels": np.asarray([1.0], dtype=np.float32),
                "predictions": np.asarray([1.0], dtype=np.float32),
            }

        def make_components():
            model = StochasticTinyInterfaceModel()
            model.configure_stage2_full()
            optimizer = torch.optim.SGD(
                model.optimizer_parameter_groups(0.05, 0.01, 0.05)
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=2, gamma=0.5
            )
            return model, optimizer, scheduler

        def train(root, epochs, resume=None):
            model, optimizer, scheduler = make_components()
            fit_interface_first(
                model=model,
                train_loader=loader(),
                valid_loader=object(),
                test_loader=object(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
                output_dir=root,
                epochs=epochs,
                warmup_epochs=1,
                grad_clip=1.0,
                provenance={"run": "resume-test"},
                evaluate_fn=evaluator,
                resume_checkpoint=resume,
            )
            return torch.load(
                Path(root) / "training_checkpoint.pth",
                map_location="cpu",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous_dir = root / "continuous"
            resumed_dir = root / "resumed"

            torch.manual_seed(811)
            np.random.seed(811)
            continuous = train(continuous_dir, epochs=4)

            torch.manual_seed(811)
            np.random.seed(811)
            train(resumed_dir, epochs=2)
            moved_dir = root / "moved_checkpoint_only"
            moved_dir.mkdir()
            moved_checkpoint = moved_dir / "training_checkpoint.pth"
            moved_checkpoint.write_bytes(
                (resumed_dir / "training_checkpoint.pth").read_bytes()
            )
            torch.manual_seed(1777)
            np.random.seed(1777)
            resumed = train(
                moved_dir,
                epochs=4,
                resume=moved_checkpoint,
            )

        self.assertEqual(
            continuous["history"], resumed["history"]
        )
        self.assertEqual(
            set(continuous["model_state_dict"]),
            set(resumed["model_state_dict"]),
        )
        for name, expected in continuous["model_state_dict"].items():
            torch.testing.assert_close(
                expected, resumed["model_state_dict"][name]
            )

    def test_fresh_stage2_run_refuses_to_overwrite_model_artifacts(self):
        from dta.interface_first_training import (
            prepare_stage2_output_dir,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "Model_dta_best.pth").write_bytes(b"old")
            with self.assertRaisesRegex(
                FileExistsError, "overwrite_output_path"
            ):
                prepare_stage2_output_dir(
                    output_dir,
                    overwrite=False,
                    resume_checkpoint=None,
                )

    def test_interface_evaluation_rejects_external_fusion_inputs(self):
        from dta.eval_dta import build_arg_parser, validate_eval_args

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--checkpoint_path",
                "model.pth",
                "--model_arch",
                "interface_first",
                "--interface_valid_native_lmdb",
                "valid.lmdb",
                "--interface_test_native_lmdb",
                "test.lmdb",
                "--boltz_interface_lmdb_path",
                "boltz.lmdb",
            ]
        )
        with self.assertRaises(SystemExit):
            validate_eval_args(parser, args)

    def test_interface_training_rejects_rank_loss(self):
        from dta.train_dta import (
            build_arg_parser,
            validate_training_args,
        )

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--model_arch",
                "interface_first",
                "--interface_train_native_lmdb",
                "train.lmdb",
                "--interface_valid_native_lmdb",
                "valid.lmdb",
                "--interface_test_native_lmdb",
                "test.lmdb",
                "--interface_stage1_checkpoint",
                "stage1.pth",
                "--interface_diagnostic_report",
                "diagnostic.json",
                "--interface_stage2_ready",
                "ready.json",
                "--batch_size",
                "1",
                "--early_stop_patience",
                "0",
                "--select_metric",
                "CI",
                "--rank_loss_weight",
                "0.1",
            ]
        )
        with self.assertRaises(SystemExit):
            validate_training_args(parser, args)

    def test_stage2_gate_accepts_schema_v2_ready_marker(self):
        from dta.interface_first_training import _validate_stage1_gate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "stage1_final.pth"
            report = root / "model_diagnostic_report.json"
            marker = root / "stage2_ready.json"
            checkpoint.write_bytes(b"checkpoint")
            report.write_text(
                json.dumps({"gate": {"passed": True}}),
                encoding="utf-8",
            )
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "stage1_checkpoint": {"path": str(checkpoint)},
                        "model_diagnostic_report": {"path": str(report)},
                    }
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "interface_stage1_checkpoint": str(checkpoint),
                    "interface_diagnostic_report": str(report),
                    "interface_stage2_ready": str(marker),
                },
            )()

            gate = _validate_stage1_gate(args)

            self.assertEqual(gate["checkpoint"]["path"], str(checkpoint))
            self.assertEqual(
                gate["diagnostic_report"]["path"], str(report)
            )

    def test_training_parser_keeps_legacy_architecture_as_default(self):
        from dta.train_dta import build_arg_parser

        args = build_arg_parser().parse_args([])
        self.assertEqual(args.model_arch, "gerna")

    def test_evaluation_parser_keeps_legacy_architecture_as_default(self):
        from dta.eval_dta import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--checkpoint_path", "model.pth"])
        self.assertEqual(args.model_arch, "gerna")


if __name__ == "__main__":
    unittest.main()
