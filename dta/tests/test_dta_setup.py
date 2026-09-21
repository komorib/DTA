import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")


def test_paffinity_from_molar_value():
    from dta.process_pdbbind_dta import molar_to_paffinity

    assert molar_to_paffinity(1e-9) == 9.0
    assert molar_to_paffinity(1e-6) == 6.0


def test_dta_record_filter_keeps_exact_affinity_only():
    from data_utils.process_pdbbind import BindingRecord
    from dta.process_pdbbind_dta import record_to_paffinity

    exact = BindingRecord("1abc", "Kd", "=", 1e-8, "Kd=10nM")
    upper_bound = BindingRecord("2abc", "Ki", "<", 1e-8, "Ki<10nM")

    assert record_to_paffinity(exact) == 8.0
    assert record_to_paffinity(upper_bound) is None


def test_protein_3d_mode_selects_residue_ca_by_default():
    import numpy as np
    from dta.process_pdbbind_dta import select_protein_3d_inputs

    residue_ids = np.array([1, 2], dtype=np.int64)
    ca_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    atom5_tokens = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4], dtype=np.int64)
    atom5_coords = np.zeros((9, 3), dtype=np.float32)

    tokens, coords, metadata = select_protein_3d_inputs(
        residue_ids,
        ca_coords,
        atom5_tokens,
        atom5_coords,
        protein_3d_mode="residue",
    )

    assert tokens.tolist() == [1, 2]
    assert coords.shape == (2, 3)
    assert metadata["protein_3d_mode"] == "residue"
    assert metadata["protein_3d_atoms"] == ["CA"]
    assert metadata["protein_3d_num_tokens"] == 21


def test_protein_3d_mode_can_still_select_atom5():
    import numpy as np
    from dta.process_pdbbind_dta import select_protein_3d_inputs

    residue_ids = np.array([1], dtype=np.int64)
    ca_coords = np.zeros((1, 3), dtype=np.float32)
    atom5_tokens = np.array([1, 2, 3], dtype=np.int64)
    atom5_coords = np.zeros((3, 3), dtype=np.float32)

    tokens, coords, metadata = select_protein_3d_inputs(
        residue_ids,
        ca_coords,
        atom5_tokens,
        atom5_coords,
        protein_3d_mode="atom5",
    )

    assert tokens.tolist() == [1, 2, 3]
    assert coords.shape == (3, 3)
    assert metadata["protein_3d_mode"] == "atom5"
    assert metadata["protein_3d_num_tokens"] == 6



def test_lmdb_protein_edges_support_radius_and_knn_without_cross_chain_backbone():
    import numpy as np
    from data_utils.lmdb_dataset import build_protein_edges

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    radius_edges = build_protein_edges(coords, chain_lengths=[1, 2], edge_mode="radius", cutoff=1.0)
    radius_pairs = {tuple(edge) for edge in radius_edges.t().tolist()}
    assert (1, 2) in radius_pairs
    assert (2, 1) in radius_pairs
    assert (0, 1) not in radius_pairs
    assert (1, 0) not in radius_pairs

    knn_edges = build_protein_edges(coords, chain_lengths=[1, 1, 1], edge_mode="knn", knn_k=1)
    knn_pairs = {tuple(edge) for edge in knn_edges.t().tolist()}
    assert knn_pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_lmdb_can_use_dummy_protein_graph_when_protein_2d_disabled():
    from data_utils.lmdb_dataset import build_disabled_protein_graph

    graph = build_disabled_protein_graph(feature_dim=640)

    assert graph.x.shape == (1, 640)
    assert graph.edge_index.shape == (2, 0)


def test_lmdb_protein_3d_inputs_select_residue_or_all_atom_tokens():
    import numpy as np
    from data_utils.lmdb_dataset import select_lmdb_protein_3d_inputs

    chains = [
        {
            "sequence": np.array(list("AC")),
            "ca_coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            "atom_element": np.array([0, 2], dtype=np.int32),
            "atom_coords": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        }
    ]

    residue_tokens, residue_coords, residue_meta = select_lmdb_protein_3d_inputs(chains, "residue")
    assert residue_tokens.tolist() == [1, 2]
    assert residue_coords.shape == (2, 3)
    assert residue_meta["protein_3d_num_tokens"] == 21

    atom_tokens, atom_coords, atom_meta = select_lmdb_protein_3d_inputs(chains, "all_atom")
    assert atom_tokens.tolist() == [1, 3]
    assert atom_coords.shape == (2, 3)
    assert atom_meta["protein_3d_num_tokens"] == 8



def test_lmdb_residue_embedding_requires_esm_by_default():
    import numpy as np
    from data_utils.lmdb_dataset import build_residue_representations

    chains = [{"sequence": np.array(list("AC")), "ca_coords": np.zeros((2, 3), dtype=np.float32)}]
    try:
        build_residue_representations(chains, embedding_mode="esm")
    except ValueError as exc:
        assert "missing esm2_embedding" in str(exc)
    else:
        raise AssertionError("missing ESM embedding should fail in esm mode")

    manual = build_residue_representations(chains, embedding_mode="residue")
    assert manual.shape == (2, 640)



def test_lmdb_protein_3d_inputs_can_build_pocket_pharmacophore_tokens():
    import numpy as np
    from data_utils.lmdb_dataset import select_lmdb_protein_3d_inputs

    chains = [
        {
            "sequence": np.array(list("AD")),
            "ca_coords": np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
            "atom_names": np.array(["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB", "OD1", "OD2"]),
            "atom_coords": np.arange(36, dtype=np.float32).reshape(12, 3),
            "atom_element": np.array([1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 2, 2], dtype=np.int32),
            "atom_residue_indices": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
            "atom_residue_type": np.array([0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32),
            "atom_is_backbone": np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32),
            "atom_is_ca": np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int32),
            "atom_is_donor": np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            "atom_is_acceptor": np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1], dtype=np.int32),
            "atom_is_aromatic": np.zeros(12, dtype=np.int32),
            "atom_is_charged": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int32),
            "atom_is_hydrophobic": np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0], dtype=np.int32),
        }
    ]

    tokens, coords, metadata = select_lmdb_protein_3d_inputs(chains, "pocket_pharmacophore")

    assert metadata["protein_3d_mode"] == "pocket_pharmacophore"
    assert metadata["protein_3d_num_tokens"] == 8
    assert len(tokens) == len(coords)
    assert len(tokens) < len(chains[0]["atom_coords"])
    assert min(tokens.tolist()) >= 1
    assert max(tokens.tolist()) <= 7
    assert coords.shape[1] == 3


def test_pocket_pharmacophore_esm_features_align_to_chain_esm():
    import numpy as np
    from data_utils.lmdb_dataset import build_pocket_pharmacophore_esm_inputs

    feature_chain = {
        "sequence": np.array(list("WAC")),
    }
    protein_repre = np.zeros((3, 640), dtype=np.float32)
    protein_repre[1, 0] = 11.0
    protein_repre[2, 0] = 22.0

    structure_chain = {
        "sequence": np.array(list("AC")),
        "atom_names": np.array(["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB"]),
        "atom_coords": np.arange(30, dtype=np.float32).reshape(10, 3),
        "atom_residue_indices": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
        "atom_is_backbone": np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=np.int32),
        "atom_is_ca": np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int32),
        "atom_is_donor": np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32),
        "atom_is_acceptor": np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.int32),
        "atom_is_aromatic": np.zeros(10, dtype=np.int32),
        "atom_is_charged": np.zeros(10, dtype=np.int32),
        "atom_is_hydrophobic": np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=np.int32),
    }

    features, coords = build_pocket_pharmacophore_esm_inputs(
        [feature_chain],
        [structure_chain],
        protein_repre,
    )

    assert features.shape[1] == 648
    assert coords.shape[1] == 3
    assert features[0, :8].sum() == 1.0
    assert features[0, 1] == 1.0
    assert features[0, 8] == 11.0
    residue_one_points = np.where(features[:, 8] == 22.0)[0]
    assert len(residue_one_points) > 0



def test_pocket_residue_esm_features_align_to_chain_esm():
    import numpy as np
    from data_utils.lmdb_dataset import build_pocket_residue_esm_inputs

    feature_chain = {
        "sequence": np.array(list("WAC")),
    }
    protein_repre = np.zeros((3, 640), dtype=np.float32)
    protein_repre[1, 0] = 11.0
    protein_repre[2, 0] = 22.0

    structure_chain = {
        "sequence": np.array(list("AC")),
        "ca_coords": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    }

    features, coords = build_pocket_residue_esm_inputs(
        [feature_chain],
        [structure_chain],
        protein_repre,
    )

    assert features.shape == (2, 640)
    assert coords.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert features[0, 0] == 11.0
    assert features[1, 0] == 22.0



def test_pocket_residue_esm_prefers_ca_coordinate_mapping_over_repeated_sequence():
    import numpy as np
    from data_utils.lmdb_dataset import build_pocket_residue_esm_inputs

    feature_chain = {
        "sequence": np.array(list("WAA")),
        "ca_coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32),
        "chain_id": "A",
    }
    protein_repre = np.zeros((3, 640), dtype=np.float32)
    protein_repre[1, 0] = 11.0
    protein_repre[2, 0] = 22.0

    structure_chain = {
        "sequence": np.array(list("A")),
        "ca_coords": np.array([[9.0, 0.0, 0.0]], dtype=np.float32),
        "chain_id": "A",
    }

    features, coords = build_pocket_residue_esm_inputs(
        [feature_chain],
        [structure_chain],
        protein_repre,
    )

    assert features.shape == (1, 640)
    assert features[0, 0] == 22.0
    assert coords.tolist() == [[9.0, 0.0, 0.0]]


def test_lmdb_split_manifest_can_use_separate_structure_lmdb(tmp_dir="/tmp/gernabind_dual_lmdb_split_test"):
    import os
    import pickle
    import shutil

    import lmdb
    import numpy as np

    from dta.split_pdbbind_lmdb import write_split_manifests

    shutil.rmtree(tmp_dir, ignore_errors=True)
    feature_lmdb = os.path.join(tmp_dir, "feature_lmdb")
    structure_lmdb = os.path.join(tmp_dir, "structure_lmdb")
    index_pkl = os.path.join(tmp_dir, "index.pkl")
    output_dir = os.path.join(tmp_dir, "split")
    os.makedirs(feature_lmdb, exist_ok=True)
    os.makedirs(structure_lmdb, exist_ok=True)

    key = "1abc_protein"
    feature_obj = {
        "protein_features": {
            "A": {
                "sequence": np.array(list("AC")),
                "ca_coords": np.zeros((2, 3), dtype=np.float32),
                "num_atoms": 2,
                "esm2_embedding": np.zeros((2, 640), dtype=np.float32),
            }
        },
        "ligand_features": {"smiles": "CCO", "num_atoms": 3},
        "affinity": 7.5,
    }
    structure_obj = {
        "protein_features": {
            "A": {
                "sequence": np.array(list("AC")),
                "ca_coords": np.zeros((2, 3), dtype=np.float32),
                "atom_coords": np.zeros((11, 3), dtype=np.float32),
                "atom_element": np.zeros((11,), dtype=np.int64),
                "num_atoms": 11,
            }
        },
        "ligand_features": {"smiles": "CCO", "num_atoms": 3},
        "affinity": 7.5,
    }

    for lmdb_path, obj in [(feature_lmdb, feature_obj), (structure_lmdb, structure_obj)]:
        env = lmdb.open(lmdb_path, map_size=1024 * 1024, subdir=True)
        with env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(obj))
        env.close()

    with open(index_pkl, "wb") as handle:
        pickle.dump([{"lmdb_key": key, "affinity": 7.5}], handle)

    summary = write_split_manifests(
        lmdb_path=feature_lmdb,
        index_pkl=index_pkl,
        output_dir=output_dir,
        structure_lmdb_path=structure_lmdb,
        split_method="random",
        train_ratio=1.0,
        valid_ratio=0.0,
        protein_3d_mode="all_atom",
        seed=99,
    )

    import json
    with open(os.path.join(output_dir, "train_manifest.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert summary["structure_lmdb"] == os.path.abspath(structure_lmdb)
    assert manifest["structure_lmdb_path"] == os.path.abspath(structure_lmdb)
    assert manifest["protein_3d_mode"] == "all_atom"
    assert manifest["samples"][0]["n_protein_atom"] == 11


def test_split_manifest_size_filter_preserves_chunk_indices(tmp_dir="/tmp/gernabind_dta_split_filter_test"):
    import os
    import pickle
    import shutil
    from dta.split_pdbbind_dta import write_split_manifests

    shutil.rmtree(tmp_dir, ignore_errors=True)
    checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
    output_dir = os.path.join(tmp_dir, "split")
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "data": [["r0", "r1"], ["mg0", "mg1"], ["pg0", "pg1"], ["pf0", "pf1"], ["pc0", "pc1"], ["pco0", "pco1"], ["mf0", "mf1"], ["mc0", "mc1"], ["las0", "las1"], [7.0, 8.0]],
        "metadata": [
            {"pdb_id": "small", "label": 7.0, "paffinity": 7.0, "n_protein_3d_token": 10, "n_ligand_atom": 10, "residue_sequence": "AA", "ligand_scaffold_smiles": "C"},
            {"pdb_id": "large", "label": 8.0, "paffinity": 8.0, "n_protein_3d_token": 1000, "n_ligand_atom": 1000, "residue_sequence": "BB", "ligand_scaffold_smiles": "N"},
        ],
    }
    with open(os.path.join(checkpoint_dir, "chunk_000000.pkl"), "wb") as handle:
        pickle.dump(payload, handle)

    summary = write_split_manifests(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        split_method="random",
        max_pair_tokens=10000,
    )

    assert summary["raw_total"] == 2
    assert summary["total"] == 1
    assert summary["filtered_out"] == 1



def test_lmdb_split_manifest_can_use_fixed_tsv_split(tmp_dir="/tmp/gernabind_fixed_tsv_split_test"):
    import json
    import os
    import pickle
    import shutil

    import lmdb
    import numpy as np

    from dta.split_pdbbind_lmdb import write_split_manifests

    shutil.rmtree(tmp_dir, ignore_errors=True)
    feature_lmdb = os.path.join(tmp_dir, "feature_lmdb")
    index_pkl = os.path.join(tmp_dir, "index.pkl")
    fixed_tsv = os.path.join(tmp_dir, "fixed.tsv")
    output_dir = os.path.join(tmp_dir, "split")
    os.makedirs(feature_lmdb, exist_ok=True)

    rows = []
    env = lmdb.open(feature_lmdb, map_size=1024 * 1024, subdir=True)
    try:
        with env.begin(write=True) as txn:
            for pdb_id in ("1abc", "2abc", "3abc"):
                key = f"{pdb_id}_protein"
                obj = {
                    "protein_features": {
                        "A": {
                            "sequence": np.array(list("AC")),
                            "ca_coords": np.zeros((2, 3), dtype=np.float32),
                            "num_atoms": 2,
                            "esm2_embedding": np.zeros((2, 640), dtype=np.float32),
                        }
                    },
                    "ligand_features": {"smiles": "CCO", "num_atoms": 3},
                    "affinity": 7.5,
                }
                txn.put(key.encode("utf-8"), pickle.dumps(obj))
                rows.append({"lmdb_key": key, "affinity": 7.5})
    finally:
        env.close()

    with open(index_pkl, "wb") as handle:
        pickle.dump(rows, handle)

    with open(fixed_tsv, "w", encoding="utf-8") as handle:
        handle.write("complex_id\tsplit\n")
        handle.write("1abc\ttrain\n")
        handle.write("2abc\tvalid\n")
        handle.write("3abc\ttest\n")

    summary = write_split_manifests(
        lmdb_path=feature_lmdb,
        index_pkl=index_pkl,
        output_dir=output_dir,
        split_method="fixed",
        fixed_split_tsv=fixed_tsv,
        protein_3d_mode="residue",
    )

    with open(os.path.join(output_dir, "train_manifest.json"), "r", encoding="utf-8") as handle:
        train_manifest = json.load(handle)
    with open(os.path.join(output_dir, "valid_manifest.json"), "r", encoding="utf-8") as handle:
        valid_manifest = json.load(handle)
    with open(os.path.join(output_dir, "test_manifest.json"), "r", encoding="utf-8") as handle:
        test_manifest = json.load(handle)

    assert summary["split_method"] == "fixed"
    assert summary["fixed_split_tsv"] == os.path.abspath(fixed_tsv)
    assert [sample["pdb_id"] for sample in train_manifest["samples"]] == ["1abc"]
    assert [sample["pdb_id"] for sample in valid_manifest["samples"]] == ["2abc"]
    assert [sample["pdb_id"] for sample in test_manifest["samples"]] == ["3abc"]


def test_dta_model_normalizes_continuous_3d_inputs():
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 32, 32],
        trigonometry=True,
        rna_graph=True,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=None,
        rna_3d_input_dim=648,
    )

    assert model.rna_3d_input_norm.normalized_shape == (648,)


def test_regression_loss_supports_mse_and_smooth_l1():
    import torch
    from dta.train_dta import regression_loss

    pred = torch.tensor([[7.0], [8.0]])
    target = torch.tensor([6.0, 8.0])

    assert torch.isclose(regression_loss(pred, target, "mse"), torch.tensor(0.5))
    assert regression_loss(pred, target, "smooth_l1") > 0


def test_tee_stream_writes_to_terminal_and_log_streams():
    import io
    from dta.train_dta import TeeStream

    terminal = io.StringIO()
    log = io.StringIO()
    tee = TeeStream(terminal, log)

    written = tee.write("epoch 0 train_loss 1.0\n")

    assert written == len("epoch 0 train_loss 1.0\n")
    assert terminal.getvalue() == "epoch 0 train_loss 1.0\n"
    assert log.getvalue() == "epoch 0 train_loss 1.0\n"


def test_training_log_path_uses_model_output_dir():
    from dta.train_dta import training_log_path

    assert training_log_path("Model/run_x") == "Model/run_x/train.log"


def test_training_checkpoint_payload_records_resume_state():
    from dta.train_dta import build_training_checkpoint_payload

    payload = build_training_checkpoint_payload(
        epoch=3,
        best_rmse=1.25,
        bad_epochs=2,
        model_state={"w": 1},
        optimizer_state={"opt": 2},
        scheduler_state={"sch": 3},
        args={"dataset": "demo"},
    )

    assert payload["epoch"] == 3
    assert payload["next_epoch"] == 4
    assert payload["best_rmse"] == 1.25
    assert payload["bad_epochs"] == 2
    assert payload["model_state_dict"] == {"w": 1}
    assert payload["optimizer_state_dict"] == {"opt": 2}
    assert payload["scheduler_state_dict"] == {"sch": 3}
    assert payload["args"] == {"dataset": "demo"}


def test_training_checkpoint_path_uses_model_output_dir():
    from dta.train_dta import training_checkpoint_path

    assert training_checkpoint_path("Model/run_x") == "Model/run_x/training_checkpoint.pth"


def test_prepare_model_output_dir_refuses_to_overwrite_existing_artifacts(tmp_path):
    from dta.train_dta import prepare_model_output_dir

    output_dir = tmp_path / "existing_run"
    output_dir.mkdir()
    (output_dir / "best_metrics.json").write_text("{}\n", encoding="utf-8")

    try:
        prepare_model_output_dir(str(output_dir), overwrite=False)
    except FileExistsError as exc:
        assert "Refusing to overwrite" in str(exc)
        assert "best_metrics.json" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")

    prepare_model_output_dir(str(output_dir), overwrite=True)


def test_dta_cli_exposes_training_and_fusion_flags_without_amp():
    from pathlib import Path

    train_source = Path(__file__).resolve().parents[1] / "train_dta.py"
    source = train_source.read_text(encoding="utf-8")

    assert 'add_argument("--amp"' not in source
    assert 'add_argument("--overwrite_output_path"' in source
    assert 'add_argument("--resume_training_checkpoint"' in source
    assert 'add_argument("--disable_protein_2d"' in source
    assert 'add_argument("--dti_fusion_lmdb_path"' in source
    assert 'add_argument("--dti_front_fusion"' in source
    assert 'add_argument("--dti_late_fusion"' in source
    assert 'add_argument("--ligand_unimol_lmdb_path"' in source
    assert 'add_argument("--ligand_unimol_fusion"' in source
    assert 'add_argument("--ligand_unimol_input_dim"' in source
    assert 'add_argument("--ligand_unimol2_lmdb_path"' in source
    assert 'add_argument("--ligand_unimol2_3d_fusion"' in source
    assert 'add_argument("--ligand_unimol2_input_dim"' in source
    assert 'add_argument("--pbcnet_fusion_lmdb_path"' in source
    assert 'add_argument("--pbcnet_late_fusion"' in source



def test_early_stop_for_regression_minimizes_rmse():
    from dta.train_dta import update_regression_early_stop

    best, bad_epochs, should_stop, improved = update_regression_early_stop(
        current_metric=1.0,
        best_metric=1.2,
        bad_epochs=2,
        patience=3,
        min_delta=0.0,
    )
    assert best == 1.0
    assert bad_epochs == 0
    assert should_stop is False
    assert improved is True

    best, bad_epochs, should_stop, improved = update_regression_early_stop(
        current_metric=1.1,
        best_metric=1.0,
        bad_epochs=2,
        patience=3,
        min_delta=0.0,
    )
    assert best == 1.0
    assert bad_epochs == 3
    assert should_stop is True
    assert improved is False


def test_best_metrics_payload_records_all_regression_metrics():
    from dta.train_dta import build_best_metrics_payload

    valid_metrics = {
        "RMSE": 1.2,
        "MAE": 0.9,
        "Pearson": 0.5,
        "Spearman": 0.6,
        "CI": 0.7,
    }
    test_metrics = {
        "RMSE": 1.3,
        "MAE": 1.0,
        "Pearson": 0.4,
        "Spearman": 0.5,
        "CI": 0.6,
    }

    payload = build_best_metrics_payload(
        epoch=3,
        train_loss=0.8,
        learning_rate=0.0003,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        checkpoint_path="Model/example/Model_dta_best.pth",
    )

    assert payload["epoch"] == 3
    assert payload["checkpoint_path"] == "Model/example/Model_dta_best.pth"
    assert payload["valid"]["RMSE"] == 1.2
    assert payload["valid"]["MAE"] == 0.9
    assert payload["valid"]["Pearson"] == 0.5
    assert payload["valid"]["Spearman"] == 0.6
    assert payload["valid"]["CI"] == 0.7
    assert payload["test"]["RMSE"] == 1.3
    assert payload["train_loss"] == 0.8
    assert payload["learning_rate"] == 0.0003


def test_dti_fusion_payload_supports_front_and_late_features():
    import numpy as np
    from data_utils.lmdb_dataset import normalize_dti_fusion_payload

    payload = {
        "protein_tokens": np.ones((3, 32), dtype=np.float32),
        "global": np.arange(64, dtype=np.float32),
    }

    normalized = normalize_dti_fusion_payload(payload, expected_protein_length=3)

    assert normalized["protein_tokens"].shape == (3, 32)
    assert normalized["protein_mask"].tolist() == [True, True, True]
    assert normalized["global"].shape == (64,)


def test_pbcnet_fusion_payload_normalizes_global_feature():
    import numpy as np
    from data_utils.lmdb_dataset import normalize_pbcnet_fusion_payload

    payload = {
        "pbcnet_global": np.arange(128, dtype=np.float32).reshape(1, 128),
    }

    normalized = normalize_pbcnet_fusion_payload(payload)

    assert normalized["payload_type"] == "pbcnet"
    assert normalized["global"].shape == (128,)
    assert normalized["global"].dtype == np.float32


def test_custom_collate_pads_optional_dti_fusion_features():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from data_utils.dataset import custom_collate_fn

    def item(length, ligand_atoms, dti_dim=5, global_dim=7):
        rna_repre = np.zeros((length, 640), dtype=np.float32)
        mol_graph = Data(
            x=torch.zeros((ligand_atoms, 55), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_graph = Data(
            x=torch.zeros((length, 640), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_feats = np.arange(length, dtype=np.int64) + 1
        rna_coords = np.zeros((length, 3), dtype=np.float32)
        mol_feats = np.arange(ligand_atoms, dtype=np.int64) + 1
        mol_coords = np.zeros((ligand_atoms, 3), dtype=np.float32)
        mol_las = np.zeros((ligand_atoms, ligand_atoms), dtype=np.float32)
        dti = {
            "protein_tokens": np.ones((length, dti_dim), dtype=np.float32),
            "protein_mask": np.ones((length,), dtype=bool),
            "global": np.ones((global_dim,), dtype=np.float32),
        }
        return (
            rna_repre,
            mol_graph,
            rna_graph,
            rna_feats,
            rna_coords,
            rna_coords,
            mol_feats,
            mol_coords,
            mol_las,
            7.0,
            dti,
        )

    batch = custom_collate_fn([item(2, 3), item(4, 2)])

    assert len(batch) == 16
    dti_protein_tokens, dti_protein_mask, dti_global = batch[13], batch[14], batch[15]
    assert dti_protein_tokens.shape == (2, 4, 5)
    assert dti_protein_mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert dti_global.shape == (2, 7)


def test_custom_collate_stacks_optional_pbcnet_global_features():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from data_utils.dataset import custom_collate_fn

    def item(length, ligand_atoms, global_dim=9):
        rna_repre = np.zeros((length, 640), dtype=np.float32)
        mol_graph = Data(
            x=torch.zeros((ligand_atoms, 55), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_graph = Data(
            x=torch.zeros((length, 640), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_feats = np.arange(length, dtype=np.int64) + 1
        rna_coords = np.zeros((length, 3), dtype=np.float32)
        mol_feats = np.arange(ligand_atoms, dtype=np.int64) + 1
        mol_coords = np.zeros((ligand_atoms, 3), dtype=np.float32)
        mol_las = np.zeros((ligand_atoms, ligand_atoms), dtype=np.float32)
        pbcnet = {
            "payload_type": "pbcnet",
            "global": np.ones((global_dim,), dtype=np.float32),
        }
        return (
            rna_repre,
            mol_graph,
            rna_graph,
            rna_feats,
            rna_coords,
            rna_coords,
            mol_feats,
            mol_coords,
            mol_las,
            7.0,
            pbcnet,
        )

    batch = custom_collate_fn([item(2, 3), item(4, 2)])

    assert len(batch) == 21
    assert batch[13] is None
    assert batch[16] is None
    assert batch[18] is None
    pbcnet_global = batch[20]
    assert pbcnet_global.shape == (2, 9)


def test_ligand_unimol_heavy_atom_tokens_expand_to_gerna_all_atom_length():
    import numpy as np
    from rdkit import Chem
    from data_utils.lmdb_dataset import align_ligand_unimol_to_gerna_atoms

    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom("C"))
    mol.AddAtom(Chem.Atom("H"))
    mol.AddAtom(Chem.Atom("O"))
    mol = mol.GetMol()
    ligand_unimol = {
        "ligand_tokens": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "ligand_mask": np.asarray([True, True]),
    }

    aligned = align_ligand_unimol_to_gerna_atoms(ligand_unimol, mol, expected_ligand_length=3)

    assert aligned["ligand_tokens"].shape == (3, 2)
    assert aligned["ligand_tokens"].tolist() == [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]
    assert aligned["ligand_mask"].tolist() == [True, False, True]


def test_ligand_unimol_atoms_must_match_gerna_atom_order_when_present():
    import numpy as np
    from rdkit import Chem
    from data_utils.lmdb_dataset import align_ligand_unimol_to_gerna_atoms

    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom("C"))
    mol.AddAtom(Chem.Atom("O"))
    mol.AddAtom(Chem.Atom("N"))
    mol = mol.GetMol()
    ligand_unimol = {
        "ligand_tokens": np.ones((3, 2), dtype=np.float32),
        "ligand_mask": np.asarray([True, True, True]),
        "atoms": ["C", "N", "O"],
    }

    try:
        align_ligand_unimol_to_gerna_atoms(ligand_unimol, mol, expected_ligand_length=3)
    except ValueError as exc:
        assert "atom order mismatch" in str(exc)
    else:
        raise AssertionError("Expected atom order mismatch for same-length but reordered Uni-Mol tokens")


def test_custom_collate_pads_optional_ligand_unimol_features():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from data_utils.dataset import custom_collate_fn

    def item(length, ligand_atoms, ligand_dim=6):
        rna_repre = np.zeros((length, 640), dtype=np.float32)
        mol_graph = Data(
            x=torch.zeros((ligand_atoms, 55), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_graph = Data(
            x=torch.zeros((length, 640), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_feats = np.arange(length, dtype=np.int64) + 1
        rna_coords = np.zeros((length, 3), dtype=np.float32)
        mol_feats = np.arange(ligand_atoms, dtype=np.int64) + 1
        mol_coords = np.zeros((ligand_atoms, 3), dtype=np.float32)
        mol_las = np.zeros((ligand_atoms, ligand_atoms), dtype=np.float32)
        ligand_unimol = {
            "ligand_tokens": np.ones((ligand_atoms, ligand_dim), dtype=np.float32),
            "ligand_mask": np.ones((ligand_atoms,), dtype=bool),
        }
        return (
            rna_repre,
            mol_graph,
            rna_graph,
            rna_feats,
            rna_coords,
            rna_coords,
            mol_feats,
            mol_coords,
            mol_las,
            7.0,
            ligand_unimol,
        )

    batch = custom_collate_fn([item(2, 3), item(4, 2)])

    assert len(batch) == 15
    ligand_tokens, ligand_mask = batch[13], batch[14]
    assert ligand_tokens.shape == (2, 3, 6)
    assert ligand_mask.tolist() == [[True, True, True], [True, True, False]]


def test_custom_collate_pads_optional_ligand_unimol2_features():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from data_utils.dataset import custom_collate_fn

    def item(length, ligand_atoms, ligand_dim=8):
        rna_repre = np.zeros((length, 640), dtype=np.float32)
        mol_graph = Data(
            x=torch.zeros((ligand_atoms, 55), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_graph = Data(
            x=torch.zeros((length, 640), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        rna_feats = np.arange(length, dtype=np.int64) + 1
        rna_coords = np.zeros((length, 3), dtype=np.float32)
        mol_feats = np.arange(ligand_atoms, dtype=np.int64) + 1
        mol_coords = np.zeros((ligand_atoms, 3), dtype=np.float32)
        mol_las = np.zeros((ligand_atoms, ligand_atoms), dtype=np.float32)
        ligand_unimol2 = {
            "payload_type": "ligand_unimol2",
            "ligand_tokens": np.ones((ligand_atoms, ligand_dim), dtype=np.float32),
            "ligand_mask": np.ones((ligand_atoms,), dtype=bool),
        }
        return (
            rna_repre,
            mol_graph,
            rna_graph,
            rna_feats,
            rna_coords,
            rna_coords,
            mol_feats,
            mol_coords,
            mol_las,
            7.0,
            ligand_unimol2,
        )

    batch = custom_collate_fn([item(2, 3), item(4, 2)])

    assert len(batch) == 20
    assert batch[13] is None
    assert batch[16] is None
    ligand_unimol2_tokens, ligand_unimol2_mask = batch[18], batch[19]
    assert ligand_unimol2_tokens.shape == (2, 3, 8)
    assert ligand_unimol2_mask.tolist() == [[True, True, True], [True, True, False]]


def test_dta_model_accepts_ligand_unimol_front_fusion_option():
    import torch
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 32, 32],
        trigonometry=True,
        rna_graph=False,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=21,
        disable_protein_2d=True,
        ligand_unimol_fusion="concat",
        ligand_unimol_input_dim=512,
    )

    assert model.ligand_unimol_fusion == "concat"
    assert model.ligand_unimol_2d_merge[1].out_features == 32
    assert not hasattr(model, "ligand_unimol_3d_proj")
    assert not hasattr(model, "ligand_unimol_3d_merge")

    mol_feats = torch.randn(1, 3, 16)
    ligand_tokens = torch.randn(1, 3, 512)
    returned = model.apply_ligand_unimol_fusion_3d(mol_feats, torch.ones(1, 3), ligand_tokens, torch.ones(1, 3))
    assert returned is mol_feats


def test_dta_model_accepts_ligand_unimol2_3d_fusion_option():
    import torch
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 32, 32],
        trigonometry=True,
        rna_graph=False,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=21,
        disable_protein_2d=True,
        ligand_unimol2_3d_fusion="concat",
        ligand_unimol2_input_dim=768,
    )

    assert model.ligand_unimol2_3d_fusion == "concat"
    assert model.ligand_unimol2_3d_merge[1].out_features == 16
    assert not hasattr(model, "ligand_unimol2_2d_proj")

    mol_feats = torch.randn(1, 3, 16)
    ligand_tokens = torch.randn(1, 3, 768)
    fused = model.apply_ligand_unimol2_fusion_3d(mol_feats, torch.ones(1, 3), ligand_tokens, torch.ones(1, 3))
    assert fused.shape == mol_feats.shape
    assert fused is not mol_feats



def test_dta_model_accepts_ligand_unimol2_pre_equiformer_fusion_option():
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 32, 32],
        trigonometry=True,
        rna_graph=False,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=21,
        disable_protein_2d=True,
        ligand_unimol2_3d_fusion="concat",
        ligand_unimol2_3d_fusion_stage="pre",
        ligand_unimol2_input_dim=768,
    )

    assert model.ligand_unimol2_3d_fusion_stage == "pre"
    assert model.ligand_unimol2_pre_proj.out_features == 16
    assert model.ligand_unimol2_pre_merge[1].out_features == 16
    assert hasattr(model, "mol_token_emb")


def test_dta_model_accepts_dti_front_and_late_fusion_options():
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 32, 32],
        trigonometry=True,
        rna_graph=False,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=21,
        disable_protein_2d=True,
        dti_front_fusion="add",
        dti_front_input_dim=24,
        dti_late_fusion="concat",
        dti_late_input_dim=48,
        dti_late_hidden_dim=16,
    )

    assert model.dti_front_fusion == "add"
    assert model.dti_late_fusion == "concat"
    assert model.W_out[0].in_features == 2 * 32 * 32 + 16


def test_dta_model_accepts_pbcnet_late_fusion_option():
    import torch
    from dta.model_dta import GerNADTA

    model = GerNADTA(
        [2, 1, 32, 32],
        trigonometry=True,
        rna_graph=False,
        coors=True,
        coors_3_bead=True,
        uncertainty=False,
        rna_num_tokens=21,
        disable_protein_2d=True,
        pbcnet_late_fusion="concat",
        pbcnet_input_dim=128,
        pbcnet_hidden_dim=16,
    )

    kroneck = torch.zeros((2, 2 * 32 * 32), dtype=torch.float32)
    pbcnet_global = torch.ones((2, 128), dtype=torch.float32)
    fused = model.apply_pbcnet_late_fusion(kroneck, pbcnet_global)

    assert model.pbcnet_late_fusion == "concat"
    assert model.W_out[0].in_features == 2 * 32 * 32 + 16
    assert fused.shape == (2, 2 * 32 * 32 + 16)


def test_eval_dta_cli_exposes_checkpoint_and_json_output():
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.abspath(__file__)) + "/../.."
    result = subprocess.run(
        [sys.executable, "dta/eval_dta.py", "--help"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--checkpoint_path" in result.stdout
    assert "--output_json" in result.stdout
    assert "--eval_splits" in result.stdout
    assert "--dti_fusion_lmdb_path" in result.stdout
    assert "--dti_front_fusion" in result.stdout
    assert "--dti_late_fusion" in result.stdout
    assert "--ligand_unimol_lmdb_path" in result.stdout
    assert "--ligand_unimol_fusion" in result.stdout
    assert "--ligand_unimol_input_dim" in result.stdout
    assert "--pbcnet_fusion_lmdb_path" in result.stdout
    assert "--pbcnet_late_fusion" in result.stdout


if __name__ == "__main__":
    test_paffinity_from_molar_value()
    test_dta_record_filter_keeps_exact_affinity_only()
    test_protein_3d_mode_selects_residue_ca_by_default()
    test_protein_3d_mode_can_still_select_atom5()
    test_lmdb_protein_edges_support_radius_and_knn_without_cross_chain_backbone()
    test_lmdb_can_use_dummy_protein_graph_when_protein_2d_disabled()
    test_lmdb_protein_3d_inputs_select_residue_or_all_atom_tokens()
    test_lmdb_protein_3d_inputs_can_build_pocket_pharmacophore_tokens()
    test_pocket_pharmacophore_esm_features_align_to_chain_esm()
    test_dta_model_normalizes_continuous_3d_inputs()
    test_lmdb_residue_embedding_requires_esm_by_default()
    test_split_manifest_size_filter_preserves_chunk_indices()
    test_regression_loss_supports_mse_and_smooth_l1()
    test_tee_stream_writes_to_terminal_and_log_streams()
    test_training_log_path_uses_model_output_dir()
    test_training_checkpoint_payload_records_resume_state()
    test_training_checkpoint_path_uses_model_output_dir()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_prepare_model_output_dir_refuses_to_overwrite_existing_artifacts(Path(tmpdir))
    test_dta_cli_exposes_training_and_fusion_flags_without_amp()
    test_early_stop_for_regression_minimizes_rmse()
    test_best_metrics_payload_records_all_regression_metrics()
    test_dti_fusion_payload_supports_front_and_late_features()
    test_pbcnet_fusion_payload_normalizes_global_feature()
    test_custom_collate_pads_optional_dti_fusion_features()
    test_custom_collate_stacks_optional_pbcnet_global_features()
    test_ligand_unimol_heavy_atom_tokens_expand_to_gerna_all_atom_length()
    test_ligand_unimol_atoms_must_match_gerna_atom_order_when_present()
    test_custom_collate_pads_optional_ligand_unimol_features()
    test_custom_collate_pads_optional_ligand_unimol2_features()
    test_dta_model_accepts_ligand_unimol_front_fusion_option()
    test_dta_model_accepts_ligand_unimol2_3d_fusion_option()
    test_dta_model_accepts_dti_front_and_late_fusion_options()
    test_dta_model_accepts_pbcnet_late_fusion_option()
    test_eval_dta_cli_exposes_checkpoint_and_json_output()
    print("ok")

