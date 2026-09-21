import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dta.rank_fusion_v4.build_cache import build_arg_parser as build_cache_parser
from dta.rank_fusion_v4.export_expert import (
    build_arg_parser as export_parser,
    main as export_main,
)


class CliPolicyTest(unittest.TestCase):
    def test_export_requires_explicit_expert_specific_paths(self):
        parser = export_parser()
        base = [
            "--expert", "upper_bound",
            "--manifest", "manifest.json",
            "--checkpoint", "model.pt",
            "--artifact-root", "artifacts",
            "--split", "valid",
            "--device", "cpu",
        ]
        args = parser.parse_args(base + ["--upper-bound-cache", "cache"])
        self.assertEqual(args.expert, "upper_bound")
        with self.assertRaises(SystemExit):
            parser.parse_args(base)

        boltz = list(base)
        boltz[1] = "boltz_pairwise"
        args = parser.parse_args(
            boltz + ["--boltz-lmdb-dir", "lmdb", "--boltz-index-path", "index"]
        )
        self.assertEqual(args.boltz_lmdb_dir, "lmdb")

        three = list(base)
        three[1] = "three_dprot"
        args = parser.parse_args(three + ["--three-dprot-data-dir", "prepared"])
        self.assertEqual(args.three_dprot_data_dir, "prepared")

    def test_cache_cli_separates_train_fit_from_eval_reuse(self):
        parser = build_cache_parser()
        train = parser.parse_args(
            [
                "--manifest", "train.json",
                "--artifact-root", "artifacts",
                "--output", "cache/train",
                "--split", "train",
                "--normalization-output", "normalization.json",
            ]
        )
        self.assertEqual(train.split, "train")
        valid = parser.parse_args(
            [
                "--manifest", "valid.json",
                "--artifact-root", "artifacts",
                "--output", "cache/valid",
                "--split", "valid",
                "--normalization-input", "normalization.json",
            ]
        )
        self.assertEqual(valid.split, "valid")

    def test_existing_export_is_reused_only_with_current_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"split": "valid", "samples": [{"lmdb_key": "x_protein", "label": 1.0}]}))
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "artifacts" / "valid" / "upper_bound"
            output.mkdir(parents=True)
            (output / "metadata.json").write_text("{}")
            loader = mock.Mock(return_value=SimpleNamespace(metadata={"status": "complete"}))
            with mock.patch("dta.rank_fusion_v4.export_expert.load_expert_artifact", loader):
                export_main([
                    "--expert", "upper_bound",
                    "--manifest", str(manifest),
                    "--checkpoint", str(checkpoint),
                    "--artifact-root", str(root / "artifacts"),
                    "--split", "valid",
                    "--device", "cpu",
                    "--upper-bound-cache", "cache",
                ])
            loader.assert_called_once_with(
                output,
                expected_expert="upper_bound",
                expected_split="valid",
                expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )

    def test_training_and_selection_clis_never_expose_test_options(self):
        package = Path(__file__).resolve().parents[1] / "dta" / "rank_fusion_v4"
        for filename in ("train.py", "select.py"):
            path = package / filename
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                    continue
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        self.assertNotIn("test", argument.value.lower(), filename)

    def test_package_never_references_legacy_positional_boltz_predictions(self):
        package = Path(__file__).resolve().parents[1] / "dta" / "rank_fusion_v4"
        forbidden = ("valid_" + "pred.npy", "test_" + "pred.npy")
        for path in package.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for name in forbidden:
                self.assertNotIn(name, text, str(path))


if __name__ == "__main__":
    unittest.main()
