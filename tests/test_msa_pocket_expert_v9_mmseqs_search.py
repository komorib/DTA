from pathlib import Path

import pytest

from dta.msa_pocket_expert_v9.mmseqs_search import read_fasta, validate_a3m


def test_read_fasta_rejects_duplicate_identifiers(tmp_path):
    path = tmp_path / "queries.fasta"
    path.write_text(">same\nACD\n>same\nEFG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_fasta(path)


def test_validate_a3m_accepts_insertions_and_rejects_bad_columns(tmp_path):
    path = tmp_path / "query.a3m"
    path.write_text(">query\nACDE\n>hit\nACde-E\n>hit\nA-DE\n", encoding="utf-8")
    assert validate_a3m(path, "query", "ACDE") == 3
    path.write_text(">query\nACDE\n>hit\nACD\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        validate_a3m(path, "query", "ACDE")
