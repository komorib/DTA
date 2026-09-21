import json
import pickle

import lmdb
import numpy as np

from dta.msa_pocket_expert_v9.complex_cache import build_complex_cache


def _write_lmdb(path, values):
    env = lmdb.open(str(path), map_size=1 << 24, subdir=True)
    with env.begin(write=True) as transaction:
        for key, value in values.items():
            transaction.put(key.encode("utf-8"), pickle.dumps(value, protocol=4))
    env.close()


def test_build_complex_cache_maps_profile_and_selects_nearest_residues(tmp_path):
    key = "sample_protein"
    sequence = "ACD"
    residue_count = len(sequence)
    chain = {
        "sequence": np.asarray(list(sequence)),
        "ca_coords": np.asarray([[0, 0, 0], [5, 0, 0], [15, 0, 0]], dtype=np.float32),
        "atom_coords": np.asarray([[0, 0, 0], [5, 0, 0], [15, 0, 0]], dtype=np.float32),
        "atom_residue_indices": np.arange(residue_count, dtype=np.int32),
        "esm2_embedding": np.ones((residue_count, 640), dtype=np.float32),
    }
    for field in (
        "hydrophobicity", "charge", "polarity", "aromatic", "hbd", "hba",
        "phi_sin", "phi_cos", "psi_sin", "psi_cos", "omega_sin", "omega_cos",
        "backbone_complete_mask", "sidechain_complete_mask", "cb_valid_mask",
    ):
        chain[field] = np.zeros(residue_count, dtype=np.float32)
    ligand = {
        "atom_coords": np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        "bond_edge_idx": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
    }
    for field in (
        "atom_atomic_numbers", "atom_is_donor", "atom_is_acceptor", "atom_formal_charge",
        "atom_hybridization", "atom_is_aromatic", "atom_chiral_tag", "atom_num_rings",
        "atom_degree", "atom_total_valence", "atom_num_hydrogens", "atom_is_in_ring",
        "atom_min_ring_size", "atom_is_hydrophobe", "atom_is_polar", "atom_is_positive",
        "atom_is_negative", "atom_is_halogen", "atom_is_metal_binding_candidate",
        "atom_logp_contribution", "atom_partial_charge",
    ):
        ligand[field] = np.zeros(2, dtype=np.float32)
    for field in (
        "bond_type", "bond_is_rotatable", "bond_is_aromatic", "bond_is_in_ring",
        "bond_is_conjugated", "bond_stereo",
    ):
        ligand[field] = np.zeros(2, dtype=np.int32)
    feature_lmdb = tmp_path / "features.lmdb"
    _write_lmdb(
        feature_lmdb,
        {key: {"protein_features": {"A": chain}, "ligand_features": ligand}},
    )
    profile_lmdb = tmp_path / "profiles.lmdb"
    _write_lmdb(
        profile_lmdb,
        {
            "seq_test": {
                "query_sequence": sequence,
                "residue_features": np.ones((residue_count, 50), dtype=np.float32),
            }
        },
    )
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "train",
                "lmdb_path": str(feature_lmdb),
                "samples": [{"lmdb_key": key, "label": 7.0, "pdb_id": "sample"}],
            }
        ),
        encoding="utf-8",
    )
    sequence_manifest = tmp_path / "sequences.json"
    sequence_manifest.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "sequence_id": "seq_test",
                        "occurrences": [
                            {"split": "train", "lmdb_key": key, "chain_id": "A"}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = build_complex_cache(
        manifest,
        sequence_manifest,
        profile_lmdb,
        output,
        minimum_pocket_residues=2,
        maximum_pocket_residues=2,
        maximum_ligand_atoms=8,
        map_size_gb=0.01,
    )
    assert report["processed"] == 1
    env = lmdb.open(str(output / "complexes.lmdb"), readonly=True, lock=False)
    cached = pickle.loads(bytes(env.begin().get(key.encode("utf-8"))))
    env.close()
    assert cached["pocket_msa"].shape == (2, 50)
    assert cached["pocket_esm"].shape == (2, 640)
    assert cached["residue_ligand_distance"].shape == (2, 2)
    assert cached["contact_label"].any()
