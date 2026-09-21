import unittest


class TestAnchoredResidualV3Cli(unittest.TestCase):
    def test_cli_requires_three_splits_and_teacher_artifacts(self):
        from dta.anchored_residual_v3.train import build_arg_parser

        args = build_arg_parser().parse_args([
            "--train-manifest", "train.json",
            "--valid-manifest", "valid.json",
            "--test-manifest", "test.json",
            "--train-native-cache", "native-train",
            "--valid-native-cache", "native-valid",
            "--test-native-cache", "native-test",
            "--teacher-train", "teacher-train",
            "--teacher-valid", "teacher-valid",
            "--teacher-test", "teacher-test",
            "--gerna-checkpoint", "gerna.pth",
            "--interface-checkpoint", "interface.pth",
            "--cached-checkpoint", "cached.pth",
            "--output", "v3-output",
            "--seed", "87",
            "--device", "cuda:0",
        ])
        self.assertEqual(args.residual_epochs, 5)
        self.assertEqual(args.fusion_epochs, 5)
        self.assertEqual(args.selective_epochs, 5)
        self.assertEqual(args.test_manifest, "test.json")
        self.assertFalse(args.amp)
        self.assertFalse(hasattr(args, "test_patience"))


if __name__ == "__main__":
    unittest.main()
