import numpy as np
import pytest

from dta.msa_pocket_expert_v9.msa_profile import (
    PROFILE_FEATURE_NAMES,
    build_msa_profile,
)


def test_build_msa_profile_returns_finite_residue_features(tmp_path):
    path = tmp_path / "query.a3m"
    path.write_text(
        ">query\nACDE\n>close\nACDE\n>insertion\nACde-E\n>deletion\nA-DE\n",
        encoding="utf-8",
    )
    profile = build_msa_profile(path, "ACDE")
    features = profile["residue_features"]
    assert features.shape == (4, len(PROFILE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert profile["depth"] == 4
    assert 1.0 <= profile["henikoff_neff"] <= 4.0
    insertion_index = PROFILE_FEATURE_NAMES.index("insertion_fraction")
    assert features[1, insertion_index] > 0.0


def test_build_msa_profile_rejects_wrong_query(tmp_path):
    path = tmp_path / "query.a3m"
    path.write_text(">query\nACDE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="query"):
        build_msa_profile(path, "ACDF")
