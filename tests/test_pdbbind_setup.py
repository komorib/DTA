import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np

from data_utils.split_pdbbind_dataset import stratified_split_indices


def test_stratified_split_indices_cover_all_samples_without_overlap():
    labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]

    train_idx, valid_idx, test_idx = stratified_split_indices(
        labels,
        train_ratio=0.6,
        valid_ratio=0.2,
        seed=7,
    )

    all_indices = train_idx + valid_idx + test_idx
    assert sorted(all_indices) == list(range(len(labels)))
    assert len(all_indices) == len(set(all_indices))
    assert len(train_idx) == 6
    assert len(valid_idx) == 2
    assert len(test_idx) == 2

    for split in (train_idx, valid_idx, test_idx):
        split_labels = np.array(labels)[split]
        assert set(split_labels.tolist()) == {0, 1}


def test_gerna_accepts_protein_residue_token_count():
    from net.model import GerNA

    model = GerNA([2, 1, 8, 8], rna_num_tokens=21)
    assert model.rna_num_tokens == 21


def test_make_balanced_indices_repeats_minority_class():
    from train_model import make_balanced_indices

    labels = [1, 1, 1, 0]
    indices = make_balanced_indices(labels, strategy="minority")
    balanced_labels = [labels[i] for i in indices]
    assert balanced_labels.count(0) == 3
    assert balanced_labels.count(1) == 3


def test_chunked_dataset_reads_manifest_sample(tmp_dir="/tmp/gernabind_chunked_dataset_test"):
    import json
    import os
    import pickle
    import shutil
    from data_utils.dataset import GerNAChunkedDataset

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    chunk_payload = {
        "data": [["r0", "r1"], ["mg0", "mg1"], ["pg0", "pg1"], ["pf0", "pf1"], ["pc0", "pc1"], ["pco0", "pco1"], ["mf0", "mf1"], ["mc0", "mc1"], ["las0", "las1"], [0, 1]],
        "metadata": [{"pdb_id": "a", "label": 0}, {"pdb_id": "b", "label": 1}],
    }
    with open(os.path.join(tmp_dir, "chunk_000000.pkl"), "wb") as handle:
        pickle.dump(chunk_payload, handle)
    manifest = {
        "checkpoint_dir": tmp_dir,
        "samples": [{"chunk": "chunk_000000.pkl", "local_index": 1, "label": 1}],
    }
    manifest_path = os.path.join(tmp_dir, "train_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    dataset = GerNAChunkedDataset(manifest_path)
    assert len(dataset) == 1
    assert dataset[0] == ("r1", "mg1", "pg1", "pf1", "pc1", "pco1", "mf1", "mc1", "las1", 1)


def test_contact_targets_from_ca_ligand_distance():
    import torch
    from train_model import build_contact_targets

    protein_ca = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    ligand_xyz = torch.tensor([[[1.0, 0.0, 0.0], [20.0, 0.0, 0.0]]])
    protein_mask = torch.tensor([[1, 1]], dtype=torch.bool)
    ligand_mask = torch.tensor([[1, 1]], dtype=torch.bool)

    target, mask = build_contact_targets(protein_ca, ligand_xyz, protein_mask, ligand_mask, cutoff=4.5)
    assert target.tolist() == [[[1.0, 0.0], [0.0, 0.0]]]
    assert mask.tolist() == [[[1.0, 1.0], [1.0, 1.0]]]


def test_pdbbind_pocket_parser_keeps_residue_ca_and_atom5_coords():
    from data_utils.process_pdbbind import PROTEIN_ATOM5_TOKEN, read_pocket_from_pdb_text

    pdb_text = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.000   1.000   0.000  1.00  0.00           C
ATOM      6  N   GLY A   2       4.000   0.000   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       6.000   0.000   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       7.000   0.000   0.000  1.00  0.00           O
END
"""

    repre, graph, residue_ids, ca_coords, atom_tokens, atom_coords, residue_sequence = read_pocket_from_pdb_text(
        pdb_text,
        distance_cutoff=8.0,
    )

    assert residue_sequence == "AG"
    assert repre.shape == (2, 640)
    assert graph.x.shape[0] == 2
    assert residue_ids.tolist() == [1, 6]
    assert ca_coords.shape == (2, 3)
    assert atom_coords.shape == (9, 3)
    assert atom_tokens.tolist() == [
        PROTEIN_ATOM5_TOKEN["N"],
        PROTEIN_ATOM5_TOKEN["CA"],
        PROTEIN_ATOM5_TOKEN["C"],
        PROTEIN_ATOM5_TOKEN["O"],
        PROTEIN_ATOM5_TOKEN["CB"],
        PROTEIN_ATOM5_TOKEN["N"],
        PROTEIN_ATOM5_TOKEN["CA"],
        PROTEIN_ATOM5_TOKEN["C"],
        PROTEIN_ATOM5_TOKEN["O"],
    ]


def test_masked_pairwise_bce_respects_mask_and_positive_weight():
    import torch
    from train_model import masked_pairwise_bce

    pred = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    loss = masked_pairwise_bce(pred, target, mask, positive_weight=1.0)
    expected = torch.nn.functional.binary_cross_entropy(pred[:, 0, :], target[:, 0, :])
    assert torch.allclose(loss, expected)


def test_group_split_keeps_group_keys_out_of_multiple_splits():
    from data_utils.split_pdbbind_dataset import grouped_split_indices

    labels = [1, 0, 1, 0, 1, 0]
    groups = ["p1", "p1", "p2", "p2", "p3", "p4"]
    train_idx, valid_idx, test_idx = grouped_split_indices(
        labels,
        groups,
        train_ratio=0.5,
        valid_ratio=0.25,
        seed=7,
    )

    splits = [set(train_idx), set(valid_idx), set(test_idx)]
    assert sorted(train_idx + valid_idx + test_idx) == list(range(len(labels)))
    for group in set(groups):
        touched = [any(groups[index] == group for index in split) for split in splits]
        assert sum(touched) == 1


def test_both_cold_group_uses_bipartite_components():
    from data_utils.split_pdbbind_dataset import build_both_cold_groups

    metadata = [
        {"protein_group": "p1", "ligand_group": "m1"},
        {"protein_group": "p2", "ligand_group": "m1"},
        {"protein_group": "p3", "ligand_group": "m2"},
    ]

    groups = build_both_cold_groups(metadata)

    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_early_stop_state_tracks_valid_auc_patience():
    from train_model import update_early_stop_state

    best, bad_epochs, should_stop, improved = update_early_stop_state(
        current_metric=0.80,
        best_metric=0.75,
        bad_epochs=2,
        patience=3,
    )
    assert best == 0.80
    assert bad_epochs == 0
    assert should_stop is False
    assert improved is True

    best, bad_epochs, should_stop, improved = update_early_stop_state(
        current_metric=0.79,
        best_metric=0.80,
        bad_epochs=2,
        patience=3,
    )
    assert best == 0.80
    assert bad_epochs == 3
    assert should_stop is True
    assert improved is False

    best, bad_epochs, should_stop, improved = update_early_stop_state(
        current_metric=0.79,
        best_metric=0.80,
        bad_epochs=100,
        patience=0,
    )
    assert best == 0.80
    assert bad_epochs == 0
    assert should_stop is False
    assert improved is False


def test_pdbbind_variant_dataset_infers_protein_token_count():
    from train_model import infer_rna_num_tokens

    assert infer_rna_num_tokens("PDBbind", None) == 21
    assert infer_rna_num_tokens("PDBbind_esm", None) == 21
    assert infer_rna_num_tokens("PDBbind_atom5", None) == 6
    assert infer_rna_num_tokens("PDBbind_atom5_esm", None) == 6
    assert infer_rna_num_tokens("Robin", None) is None
    assert infer_rna_num_tokens("PDBbind_esm", 31) == 31


def test_pdbbind_residue_ids_convert_to_esm_sequence():
    from data_utils.extract_pdbbind_esm import residue_ids_to_sequence

    assert residue_ids_to_sequence([1, 2, 20, 0, 21]) == "ACYXX"


def test_pdbbind_esm_sequence_uses_metadata_for_atom5_chunks():
    import numpy as np
    from data_utils.extract_pdbbind_esm import sequence_for_sample

    atom5_tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    metadata = {"residue_sequence": "A"}

    assert sequence_for_sample(atom5_tokens, metadata, expected_length=1) == "A"


def test_pdbbind_esm_embedding_replaces_repre_and_graph_x():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from data_utils.extract_pdbbind_esm import replace_protein_embeddings

    chunk_data = [
        [np.zeros((3, 640), dtype=np.float32)],
        ["mol_graph"],
        [Data(x=torch.zeros((3, 640)), edge_index=torch.empty((2, 0), dtype=torch.long))],
        [np.array([1, 2, 3, 4, 5], dtype=np.int64)],
        ["ca"],
        ["coors"],
        ["mol_feats"],
        ["mol_coors"],
        ["las"],
        [1],
    ]
    esm_embedding = np.ones((3, 640), dtype=np.float32)

    replace_protein_embeddings(chunk_data, [esm_embedding])

    assert np.array_equal(chunk_data[0][0], esm_embedding)
    assert torch.equal(chunk_data[2][0].x, torch.tensor(esm_embedding, dtype=torch.float32))


def test_score_metrics_for_binary_probability_scores():
    from utils.metrics import get_score_metrics

    rmse, mae, pearson, spearman, ci = get_score_metrics(
        [0.1, 0.2, 0.8, 0.9],
        [0.0, 0.0, 1.0, 1.0],
    )

    assert round(rmse, 6) == 0.158114
    assert round(mae, 6) == 0.15
    assert round(pearson, 6) == 0.989949
    assert round(spearman, 6) == 0.894427
    assert round(ci, 6) == 1.0


def test_train_model_cli_accepts_protein_split_method():
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    result = subprocess.run(
        [sys.executable, "train_model.py", "--help"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "{random,RNA,protein,mol,both}" in result.stdout


if __name__ == "__main__":
    test_stratified_split_indices_cover_all_samples_without_overlap()
    test_gerna_accepts_protein_residue_token_count()
    test_make_balanced_indices_repeats_minority_class()
    test_chunked_dataset_reads_manifest_sample()
    test_contact_targets_from_ca_ligand_distance()
    test_pdbbind_pocket_parser_keeps_residue_ca_and_atom5_coords()
    test_masked_pairwise_bce_respects_mask_and_positive_weight()
    test_group_split_keeps_group_keys_out_of_multiple_splits()
    test_both_cold_group_uses_bipartite_components()
    test_early_stop_state_tracks_valid_auc_patience()
    test_pdbbind_variant_dataset_infers_protein_token_count()
    test_pdbbind_residue_ids_convert_to_esm_sequence()
    test_pdbbind_esm_sequence_uses_metadata_for_atom5_chunks()
    test_pdbbind_esm_embedding_replaces_repre_and_graph_x()
    test_score_metrics_for_binary_probability_scores()
    test_train_model_cli_accepts_protein_split_method()
    print("ok")
def test_full_residue_esm_mode_uses_continuous_full_chain_esm_inputs():
    from dta.train_dta import infer_rna_3d_input_dim, infer_rna_num_tokens
    from data_utils.lmdb_dataset import build_full_residue_esm_inputs

    chains = [
        {
            "sequence": "AC",
            "esm2_embedding": np.arange(6, dtype=np.float32).reshape(2, 3),
            "ca_coords": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        },
        {
            "sequence": "D",
            "esm2_embedding": np.asarray([[6.0, 7.0, 8.0]], dtype=np.float32),
            "ca_coords": np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        },
    ]

    features, coords = build_full_residue_esm_inputs(chains, expected_dim=3)

    assert infer_rna_num_tokens("PDBbind", None, protein_3d_mode="full_residue_esm") is None
    assert infer_rna_3d_input_dim("full_residue_esm") == 640
    assert features.shape == (3, 3)
    assert coords.shape == (3, 3)
    assert features.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]

def test_lmdb_dataset_allows_full_residue_esm_override_for_residue_manifests(tmp_dir="/tmp/gernabind_full_residue_esm_manifest_test"):
    import json
    import os
    import shutil
    from data_utils.lmdb_dataset import PDBbindLmdbDataset

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    manifest_path = os.path.join(tmp_dir, "train_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_type": "pdbbind_lmdb",
                "lmdb_path": "dummy_feature_lmdb",
                "protein_3d_mode": "residue",
                "samples": [],
            },
            handle,
        )

    dataset = PDBbindLmdbDataset(manifest_path, protein_3d_mode="full_residue_esm")

    assert dataset.protein_3d_mode == "full_residue_esm"

def test_ligand_pocket_residue_esm_selects_only_residues_near_ligand():
    from data_utils.lmdb_dataset import build_ligand_pocket_residue_esm_inputs

    chains = [
        {
            "sequence": "ACD",
            "ca_coords": np.asarray(
                [[0.0, 0.0, 0.0], [7.9, 0.0, 0.0], [8.1, 0.0, 0.0]],
                dtype=np.float32,
            ),
        }
    ]
    protein_repre = np.arange(9, dtype=np.float32).reshape(3, 3)
    ligand_coords = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    features, coords = build_ligand_pocket_residue_esm_inputs(
        chains,
        protein_repre,
        ligand_coords,
        cutoff=8.0,
    )

    assert features.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert coords.tolist() == [[0.0, 0.0, 0.0], [7.900000095367432, 0.0, 0.0]]


def test_lmdb_dataset_allows_ligand_pocket_residue_esm_override_for_residue_manifests(tmp_dir="/tmp/gernabind_ligand_pocket_manifest_test"):
    import json
    import shutil
    from data_utils.lmdb_dataset import PDBbindLmdbDataset

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    manifest_path = os.path.join(tmp_dir, "train_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_type": "pdbbind_lmdb",
                "lmdb_path": "dummy_feature_lmdb",
                "protein_3d_mode": "residue",
                "samples": [],
            },
            handle,
        )

    dataset = PDBbindLmdbDataset(manifest_path, protein_3d_mode="ligand_pocket_residue_esm")

    assert dataset.protein_3d_mode == "ligand_pocket_residue_esm"


def test_ligand_pocket_residue_esm_uses_continuous_esm_equiformer_inputs():
    from dta.train_dta import infer_rna_3d_input_dim, infer_rna_num_tokens

    assert infer_rna_num_tokens("LP_PDBBind", None, protein_3d_mode="ligand_pocket_residue_esm") is None
    assert infer_rna_3d_input_dim("ligand_pocket_residue_esm") == 640


def test_boltz_interface_head_ignores_masked_padding():
    import torch
    from net.boltz_interface import BoltzInterfaceHead

    torch.manual_seed(7)
    head = BoltzInterfaceHead(emb_dim=4, hidden_dim=3, dropout=0.0).eval()
    batch = {
        "pocket_emb": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]]]),
        "pocket_mask": torch.tensor([[True, True, False]]),
        "ligand_emb": torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [4.0, 4.0, 4.0, 4.0]]]),
        "ligand_mask": torch.tensor([[True, True, False]]),
        "global_feat": torch.ones((1, 8)),
    }
    expected = head(batch)
    altered = {key: value.clone() for key, value in batch.items()}
    altered["pocket_emb"][0, 2] = -1000.0
    altered["ligand_emb"][0, 2] = 1000.0

    actual = head(altered)

    assert expected.shape == (1, 6)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_boltz_interface_payload_uses_nearest_residue_pocket():
    from data_utils.lmdb_dataset import build_boltz_interface_payload

    payload = {
        "protein_boltz_emb": np.arange(12, dtype=np.float32).reshape(3, 4),
        "ligand_boltz_emb": np.arange(8, dtype=np.float32).reshape(2, 4),
        "atom_coords": np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "atom_mask": np.asarray([[True, False], [True, False], [True, False]]),
        "ligand_xyz": np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "ligand_mask": np.asarray([True, True]),
    }

    out = build_boltz_interface_payload(payload, pocket_radius=3.0, min_pocket_residues=2, max_pocket_residues=2)

    assert out["pocket_emb"].tolist() == [[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]
    assert out["ligand_emb"].shape == (2, 4)
    assert out["global_feat"].shape == (8,)


def test_evaluate_forwards_boltz_interface_payload_to_model():
    import torch
    from dta.train_dta import evaluate

    class CaptureNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_args = None

        def forward(self, *args):
            self.last_args = args
            return torch.zeros((1, 1)), None

    base = tuple(torch.zeros((1, 1)) for _ in range(13))
    boltz = {
        "pocket_emb": torch.zeros((1, 1, 4)),
        "pocket_mask": torch.ones((1, 1), dtype=torch.bool),
        "ligand_emb": torch.zeros((1, 1, 4)),
        "ligand_mask": torch.ones((1, 1), dtype=torch.bool),
        "global_feat": torch.zeros((1, 8)),
    }
    net = CaptureNet()

    evaluate(net, [base + (boltz,)], torch.device("cpu"))

    assert len(net.last_args) == 22
    assert net.last_args[-1]["global_feat"].shape == (1, 8)


def test_gerna_forward_declares_boltz_interface_argument():
    import inspect
    from net.model import GerNA

    assert "Boltz_interface" in inspect.signature(GerNA.forward).parameters


def test_boltz_coverage_reports_missing_manifest_keys(tmp_dir="/tmp/gernabind_boltz_coverage_test"):
    import json
    import os
    import shutil
    import lmdb
    from scripts.check_boltz_interface_coverage import inspect_coverage

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    manifest_path = os.path.join(tmp_dir, "train_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"samples": [{"lmdb_key": "present"}, {"lmdb_key": "absent"}]}, handle)
    lmdb_path = os.path.join(tmp_dir, "boltz.lmdb")
    env = lmdb.open(lmdb_path, map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        txn.put(b"present", b"payload")
    env.close()

    report = inspect_coverage([manifest_path], lmdb_path)

    assert report["total"] == 2
    assert report["present"] == 1
    assert report["missing"] == 1
    assert report["manifests"][0]["missing_preview"] == ["absent"]


def test_boltz_coverage_writes_intersected_manifests(tmp_dir="/tmp/gernabind_boltz_filter_test"):
    import json
    import os
    import shutil
    import lmdb
    from scripts.check_boltz_interface_coverage import write_intersected_manifests

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    source_path = os.path.join(tmp_dir, "train_manifest.json")
    with open(source_path, "w", encoding="utf-8") as handle:
        json.dump({"dataset_type": "pdbbind_lmdb", "samples": [{"lmdb_key": "present"}, {"lmdb_key": "absent"}]}, handle)
    lmdb_path = os.path.join(tmp_dir, "boltz.lmdb")
    env = lmdb.open(lmdb_path, map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        txn.put(b"present", b"payload")
    env.close()

    written = write_intersected_manifests([source_path], lmdb_path, os.path.join(tmp_dir, "filtered"))

    with open(written[0], "r", encoding="utf-8") as handle:
        filtered = json.load(handle)
    assert filtered["samples"] == [{"lmdb_key": "present"}]


def test_boltz_numpy_pickle_compat_registers_numpy2_module_aliases():
    import sys
    from data_utils.lmdb_dataset import install_boltz_numpy_pickle_compat

    install_boltz_numpy_pickle_compat()

    assert "numpy._core" in sys.modules
    assert "numpy._core.multiarray" in sys.modules


def test_gernadta_only_3d_configuration_uses_single_3d_affinity_branch():
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 8, 8],
        mol_graph=False,
        rna_repre=False,
        rna_graph=False,
        coors=True,
        rna_num_tokens=None,
        rna_3d_input_dim=640,
    )

    assert model.mol_graph is False
    assert model.rna_repre is False
    assert model.rna_graph is False
    assert model.coors is True
    assert model.affinity_input_dim == 8 * 8

def test_gernadta_trainable_pbcnet_initializes_random_encoder():
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 8, 8],
        pbcnet_late_fusion="concat",
        pbcnet_input_dim=96,
        pbcnet_hidden_dim=4,
        pbcnet_trainable=True,
        pbcnet_root="external/PBCNet2.0",
        pbcnet_model_hidden_dim=96,
    )

    assert model.pbcnet_trainable is True
    assert model.pbcnet_input_dim == 96
    assert hasattr(model, "pbcnet_encoder")
    assert model.affinity_input_dim == 8 * 8 * 2 + 4


def test_gernadta_trainable_pbcnet_loads_init_checkpoint_and_keeps_trainable(tmp_dir="/tmp/gernabind_pbcnet_init_test"):
    import os
    import shutil
    import torch
    from dta.model_dta import GerNADTA
    from net.model import load_pbcnet_state_dict, infer_pbcnet_hidden_dim_from_state_dict

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    seed_model = GerNADTA(
        [2, 1, 8, 8],
        pbcnet_late_fusion="concat",
        pbcnet_input_dim=96,
        pbcnet_hidden_dim=4,
        pbcnet_trainable=True,
        pbcnet_root="external/PBCNet2.0",
        pbcnet_model_hidden_dim=96,
    )
    checkpoint_path = os.path.join(tmp_dir, "pbcnet_init.pth")
    torch.save({"model_state_dict": seed_model.pbcnet_encoder.state_dict()}, checkpoint_path)

    model = GerNADTA(
        [2, 1, 8, 8],
        pbcnet_late_fusion="concat",
        pbcnet_input_dim=96,
        pbcnet_hidden_dim=4,
        pbcnet_trainable=True,
        pbcnet_root="external/PBCNet2.0",
        pbcnet_model_hidden_dim=96,
        pbcnet_init_checkpoint=checkpoint_path,
    )

    loaded_state = load_pbcnet_state_dict(checkpoint_path)
    assert infer_pbcnet_hidden_dim_from_state_dict(loaded_state) == 96
    assert torch.equal(
        model.pbcnet_encoder.state_dict()["norm.weight"],
        loaded_state["norm.weight"],
    )
    assert all(param.requires_grad for param in model.pbcnet_encoder.parameters())

def test_build_dta_optimizer_uses_lower_pbcnet_learning_rate():
    import torch
    from types import SimpleNamespace
    from dta.train_dta import build_dta_optimizer

    class DummyDTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pbcnet_encoder = torch.nn.Linear(2, 2)
            self.affinity_head = torch.nn.Linear(2, 1)

    model = DummyDTA()
    args = SimpleNamespace(
        lr=1e-4,
        pbcnet_lr=1e-5,
        weight_decay=1e-4,
        pbcnet_trainable=True,
    )

    optimizer = build_dta_optimizer(model, args)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-5
    pbcnet_param_ids = {id(param) for param in model.pbcnet_encoder.parameters()}
    main_param_ids = {id(param) for param in optimizer.param_groups[0]["params"]}
    pbcnet_group_ids = {id(param) for param in optimizer.param_groups[1]["params"]}
    assert not (main_param_ids & pbcnet_param_ids)
    assert pbcnet_group_ids == pbcnet_param_ids


def test_set_pbcnet_trainable_for_epoch_freezes_then_unfreezes():
    import torch
    from dta.train_dta import set_pbcnet_trainable_for_epoch

    class DummyDTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pbcnet_encoder = torch.nn.Linear(2, 2)

    model = DummyDTA()

    assert set_pbcnet_trainable_for_epoch(model, epoch=0, freeze_pbcnet_epochs=5) is False
    assert all(not param.requires_grad for param in model.pbcnet_encoder.parameters())

    assert set_pbcnet_trainable_for_epoch(model, epoch=5, freeze_pbcnet_epochs=5) is True
    assert all(param.requires_grad for param in model.pbcnet_encoder.parameters())

