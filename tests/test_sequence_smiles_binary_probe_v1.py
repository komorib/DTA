import struct
from pathlib import Path

import numpy as np

from dta.sequence_smiles_binary_probe_v1.core import (
    _INDEX_MAGIC,
    TargetStatisticsEncoder,
    best_decision_threshold,
    binary_metrics,
    load_split,
    select_score_expression,
)


def _write_indexed(prefix: Path, sequences, vocabulary):
    arrays = []
    pointers = []
    pointer = 0
    for sequence in sequences:
        tokens = np.asarray([0, *[vocabulary[c] + 4 for c in sequence], 2], dtype="<u2")
        arrays.append(tokens)
        pointers.append(pointer)
        pointer += tokens.nbytes
    index = bytearray(_INDEX_MAGIC)
    index.extend(struct.pack("<Q", 1))
    index.extend(bytes([8]))
    index.extend(struct.pack("<Q", len(arrays)))
    index.extend(np.asarray([array.size for array in arrays], dtype="<i4").tobytes())
    index.extend(np.asarray(pointers, dtype="<i8").tobytes())
    prefix.with_suffix(".idx").write_bytes(index)
    prefix.with_suffix(".bin").write_bytes(b"".join(array.tobytes() for array in arrays))


def test_fairseq_mmap_decoder_round_trip(tmp_path):
    root = tmp_path / "DAVIS"
    for branch in ("input0", "input1", "label"):
        (root / branch).mkdir(parents=True)
    (root / "input0" / "dict.txt").write_text("C 2\nO 1\n", encoding="utf-8")
    (root / "input1" / "dict.txt").write_text("A 2\nG 1\n", encoding="utf-8")
    _write_indexed(root / "input0" / "train", ["CCO", "OC"], {"C": 0, "O": 1})
    _write_indexed(root / "input1" / "train", ["AAG", "GA"], {"A": 0, "G": 1})
    (root / "label" / "train.label").write_text("7.0\n8.0\n", encoding="utf-8")
    split = load_split(root, "train")
    assert split.smiles == ["CCO", "OC"]
    assert split.proteins == ["AAG", "GA"]
    np.testing.assert_allclose(split.affinity, [7.0, 8.0])


def test_leave_one_out_statistics_and_novel_pair_mask():
    smiles = ["CC", "CC", "CO", "CO"]
    proteins = ["AA", "GG", "AA", "AA"]
    affinity = np.asarray([6.0, 8.0, 7.0, 9.0])
    binary = np.asarray([0, 1, 0, 1], dtype=np.int8)
    encoder = TargetStatisticsEncoder().fit(smiles, proteins, affinity, binary)
    train_features = encoder.transform(
        smiles, proteins, affinity=affinity, binary=binary, leave_one_out=True
    )
    probe_features = encoder.transform(["CC", "CN"], ["AA", "AA"])
    assert np.isfinite(train_features).all()
    assert probe_features.shape == (2, len(encoder.feature_names))
    assert encoder.novel_pair_mask(["CC", "CN"], ["AA", "AA"]).tolist() == [False, True]


def test_validation_score_and_threshold_selection():
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    base_scores = {
        "weak": np.asarray([0.1, 0.4, 0.2, 0.5, 0.7, 0.6]),
        "strong": np.asarray([0.05, 0.1, 0.2, 0.8, 0.9, 0.7]),
    }
    expression, candidates = select_score_expression(labels, base_scores)
    threshold, objective = best_decision_threshold(labels, base_scores["strong"])
    metrics = binary_metrics(labels, base_scores["strong"], threshold)
    assert expression == {"kind": "base", "name": "strong"}
    assert candidates[0]["AUPRC"] == 1.0
    assert objective["MCC"] == 1.0
    assert metrics["ROC-AUC"] == 1.0
    assert metrics["MCC"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
