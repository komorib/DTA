import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")


def test_project_protein_tokens_uses_checkpoint_linear_projection():
    from dta.extract_dti_v12_fusion_lmdb import project_protein_tokens

    protein_repre = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    weight = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.5, -0.5, 1.0, 0.0], dtype=np.float32)

    payload = project_protein_tokens(protein_repre, weight, bias)

    assert payload["protein_tokens"].dtype == np.float32
    assert payload["protein_tokens"].shape == (2, 4)
    assert payload["protein_mask"].dtype == np.bool_
    assert payload["protein_mask"].tolist() == [True, True]
    np.testing.assert_allclose(
        payload["protein_tokens"],
        np.array(
            [
                [1.5, 1.5, 4.0, 6.0],
                [4.5, 4.5, 7.0, 15.0],
            ],
            dtype=np.float32,
        ),
    )


def test_attach_global_feature_adds_flat_float32_vector():
    from dta.extract_dti_v12_fusion_lmdb import attach_global_feature

    payload = {
        "protein_tokens": np.zeros((3, 2), dtype=np.float32),
        "protein_mask": np.ones((3,), dtype=np.bool_),
    }
    global_feature = np.arange(6, dtype=np.float32).reshape(2, 3)

    fused = attach_global_feature(payload, global_feature)

    assert fused["global"].dtype == np.float32
    assert fused["global"].shape == (6,)
    np.testing.assert_allclose(fused["global"], np.arange(6, dtype=np.float32))


if __name__ == "__main__":
    test_project_protein_tokens_uses_checkpoint_linear_projection()
    test_attach_global_feature_adds_flat_float32_vector()
    print("ok")
