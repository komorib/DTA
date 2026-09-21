import json
import pickle

import lmdb
import numpy as np

from dta.msa_pocket_expert_v9.sequence_manifest import (
    build_sequence_manifest,
    normalize_query_sequence,
)


def _write_lmdb(path, records):
    env = lmdb.open(str(path), map_size=1 << 20, subdir=True)
    with env.begin(write=True) as transaction:
        for key, value in records.items():
            transaction.put(key.encode("utf-8"), pickle.dumps(value))
    env.close()


def test_normalize_query_sequence_replaces_noncanonical_residues():
    assert normalize_query_sequence(np.asarray(list("acdu-z"))) == "ACDXXX"


def test_build_sequence_manifest_deduplicates_across_splits(tmp_path):
    lmdb_path = tmp_path / "features.lmdb"
    _write_lmdb(
        lmdb_path,
        {
            "a_protein": {
                "protein_features": {
                    "B": {"sequence": np.asarray(list("GGGX"))},
                    "A": {"sequence": np.asarray(list("ACDE"))},
                }
            },
            "b_protein": {
                "protein_features": {"A": {"sequence": np.asarray(list("ACDE"))}}
            },
        },
    )
    train = tmp_path / "train.json"
    valid = tmp_path / "valid.json"
    train.write_text(
        json.dumps({"split": "train", "lmdb_path": str(lmdb_path), "samples": [{"lmdb_key": "a_protein"}]}),
        encoding="utf-8",
    )
    valid.write_text(
        json.dumps({"split": "valid", "lmdb_path": str(lmdb_path), "samples": [{"lmdb_key": "b_protein"}]}),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    result = build_sequence_manifest(
        [("train", train), ("valid", valid)], output
    )

    payload = json.loads((output / "sequence_manifest.json").read_text(encoding="utf-8"))
    assert result["sequence_count"] == 2
    assert result["occurrence_count"] == 3
    assert sorted(len(item["occurrences"]) for item in payload["sequences"]) == [1, 2]
    fasta = (output / "queries.fasta").read_text(encoding="utf-8")
    assert fasta.count(">seq_") == 2
    assert "ACDE" in fasta
