import json
import pickle

import lmdb

from dta.msa_pocket_expert_v9.mmseqs_profile_db import build_mmseqs_profile_cache


def test_build_mmseqs_profile_cache_reads_offset_order(tmp_path):
    sequences = [
        {
            "sequence_id": "seq_zero",
            "sequence_sha256": "zero",
            "sequence": "ACDE",
        },
        {
            "sequence_id": "seq_one",
            "sequence_sha256": "one",
            "sequence": "FGHI",
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sequences": sequences}), encoding="utf-8")
    raw_one = b">seq_one\nFGHI\n>hit\nF-HI\n\x00"
    raw_zero = b">seq_zero\nACDE\n>hit\nACde-E\n\x00"
    database = tmp_path / "result.a3m"
    database.write_bytes(raw_one + raw_zero)
    (tmp_path / "result.a3m.index").write_text(
        f"0\t{len(raw_one)}\t{len(raw_zero)}\n1\t0\t{len(raw_one)}\n",
        encoding="utf-8",
    )

    output = tmp_path / "profiles"
    report = build_mmseqs_profile_cache(
        database, manifest, output, map_size_gb=0.01, temp_dir=tmp_path
    )
    assert report["processed"] == 2
    env = lmdb.open(str(output / "profiles.lmdb"), readonly=True, lock=False)
    with env.begin(write=False) as transaction:
        zero = pickle.loads(bytes(transaction.get(b"seq_zero")))
        one = pickle.loads(bytes(transaction.get(b"seq_one")))
    env.close()
    assert zero["query_sequence"] == "ACDE"
    assert one["query_sequence"] == "FGHI"
    assert zero["mmseqs_key"] == 0
    assert one["mmseqs_key"] == 1
