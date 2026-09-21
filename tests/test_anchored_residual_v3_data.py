import unittest

import torch


class Dataset:
    def __init__(self):
        self.samples = [
            {"key": "b", "label": 2.0},
            {"key": "a", "label": 1.0},
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return dict(self.samples[index])


class Lookup:
    def value_for(self, key):
        return {"a": 1.5, "b": 2.5}[key]


class TestAnchoredData(unittest.TestCase):
    def test_dataset_attaches_teacher_by_key_without_mutating_source(self):
        from dta.anchored_residual_v3.data import TeacherAnchoredDataset

        source = Dataset()
        wrapped = TeacherAnchoredDataset(source, Lookup())
        self.assertEqual(wrapped[0]["teacher_prediction"], 2.5)
        self.assertNotIn("teacher_prediction", source.samples[0])
        self.assertEqual(wrapped.keys, ("b", "a"))

    def test_prepare_appends_ordered_teacher_tensor(self):
        from dta.anchored_residual_v3.data import prepare_anchored_group

        samples = [
            {"key": "b", "teacher_prediction": 2.5},
            {"key": "a", "teacher_prediction": 1.5},
        ]

        def base_prepare(group, device):
            return [item["key"] for item in group], {"x": 1}, object(), torch.tensor([2.0, 1.0])

        keys, gerna, packed, labels, teacher = prepare_anchored_group(
            samples, "cpu", prepare_fn=base_prepare
        )
        self.assertEqual(keys, ["b", "a"])
        self.assertEqual(tuple(teacher.shape), (2, 1))
        torch.testing.assert_close(teacher[:, 0], torch.tensor([2.5, 1.5]))
        self.assertEqual(labels.tolist(), [2.0, 1.0])

    def test_prepare_rejects_missing_and_nonfinite_teacher_with_key(self):
        from dta.anchored_residual_v3.data import prepare_anchored_group

        def base_prepare(group, device):
            return [item["key"] for item in group], {}, object(), torch.ones(len(group))

        with self.assertRaisesRegex(ValueError, "bad.*teacher_prediction"):
            prepare_anchored_group(
                [{"key": "bad"}], "cpu", prepare_fn=base_prepare
            )
        with self.assertRaisesRegex(ValueError, "nan.*finite"):
            prepare_anchored_group(
                [{"key": "nan", "teacher_prediction": float("nan")}],
                "cpu",
                prepare_fn=base_prepare,
            )


if __name__ == "__main__":
    unittest.main()
