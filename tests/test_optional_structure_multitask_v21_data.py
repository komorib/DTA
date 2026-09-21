import pickle

import lmdb
import numpy as np
import torch

from dta.optional_structure_multitask_v21.build_ligand_cache import (
    molecule_graph,
    structure_cache_graph,
)
from dta.optional_structure_multitask_v21.data import (
    OptionalStructureCollator,
    OptionalStructureDataset,
    read_jsonl,
    write_jsonl,
)
from dta.optional_structure_multitask_v21.extract_esm import adaptive_pool, sequence_windows
from dta.optional_structure_multitask_v21.losses import optional_structure_loss
from dta.optional_structure_multitask_v21.model import OptionalStructureDTA
from dta.optional_structure_multitask_v21.training import label_statistics


def test_full_sequence_windows_cover_every_residue_and_pool():
    sequence = "A" * 2500
    windows = sequence_windows(sequence, 1022, 128)
    coverage = np.zeros(len(sequence), dtype=np.int64)
    for start, window in windows:
        coverage[start : start + len(window)] += 1
    assert coverage.min() >= 1
    embedding = np.arange(2500 * 4, dtype=np.float32).reshape(2500, 4)
    pooled = adaptive_pool(embedding, 512)
    assert pooled.shape == (512, 4)
    assert np.isfinite(pooled).all()


def test_rdkit_and_structure_backed_graph_schemas():
    graph = molecule_graph("CC(=O)O")
    assert graph["categorical"].shape == (4, 8)
    assert graph["continuous"].shape == (4, 4)
    assert graph["bond_edges"].shape[0] == 2
    assert graph["bond_features"].shape[1] == 6

    record = {
        "ligand_categorical": np.asarray(
            [[6, 0, 0, 0, 3, 0, 0, 0, 1, 4, 3, 0, 0, 1, 0, 0, 0, 0, 0]],
            dtype=np.int16,
        ),
        "ligand_continuous": np.asarray([[0.1, -0.2]], dtype=np.float32),
        "bond_edges": np.empty((2, 0), dtype=np.int64),
        "bond_features": np.empty((0, 6), dtype=np.int16),
    }
    fallback = structure_cache_graph(record)
    assert fallback["categorical"].shape == (1, 8)
    assert fallback["continuous"].shape == (1, 4)
    assert fallback["source"] == "structure_cache_2d"



def test_smoke_training_prefix_has_label_variance_for_every_dataset():
    dataset = OptionalStructureDataset(
        "data/OptionalStructureMultitask_v21/manifests_v2/train.jsonl",
        "data/OptionalStructureMultitask_v21/proteins_esm2_t33_v21.lmdb",
        "data/OptionalStructureMultitask_v21/ligands_v2.lmdb",
        limit=64,
    )
    statistics = label_statistics(dataset)
    assert set(statistics) == {"LP_PDBBind", "DAVIS", "KIBA"}
    assert all(value["count"] == 64 for value in statistics.values())
    assert all(value["std"] > 0 for value in statistics.values())


def test_real_mixed_dataset_collation_and_backward(tmp_path):
    manifest_root = "data/OptionalStructureMultitask_v21/manifests_v2"
    rows = read_jsonl(manifest_root + "/train.jsonl")
    selected = []
    for dataset in ("DAVIS", "KIBA"):
        selected.append(next(row for row in rows if row["dataset"] == dataset))
    selected.append(
        next(
            row
            for row in rows
            if row["dataset"] == "LP_PDBBind"
            and row["ligand_source"] == "structure_cache_2d"
        )
    )
    manifest = tmp_path / "mixed.jsonl"
    write_jsonl(manifest, selected)
    protein_cache = tmp_path / "proteins.lmdb"
    environment = lmdb.open(str(protein_cache), map_size=1 << 24, subdir=True)
    with environment.begin(write=True) as transaction:
        for row in selected:
            record = {"embedding": np.random.default_rng(4).normal(size=(7, 16)).astype(np.float16)}
            transaction.put(row["sequence_id"].encode("ascii"), pickle.dumps(record))
    environment.close()
    dataset = OptionalStructureDataset(
        manifest,
        protein_cache,
        "data/OptionalStructureMultitask_v21/ligands_v2.lmdb",
    )
    batch = OptionalStructureCollator(maximum_ligand_atoms=32, maximum_pocket_tokens=16)(
        [dataset[index] for index in range(len(dataset))]
    )
    assert batch["has_3d"].tolist() == [False, False, True]
    assert batch["ligand_categorical"].shape[-1] == 8
    assert batch["pocket_mask"][2].any()
    model = OptionalStructureDTA(
        protein_input_width=16,
        width=32,
        heads=4,
        protein_layers=1,
        ligand_layers=1,
        cross_layers=1,
        latent_width=48,
        pair_width=32,
        structure_self_layers=1,
        structure_pair_layers=1,
        structure_token_layers=1,
        structure_top_pairs=8,
        maximum_protein_tokens=8,
        dataset_count=3,
        dropout=0.0,
    )
    output = model(batch)
    loss, _ = optional_structure_loss(output, torch.tensor([-0.2, 0.3, 0.8]), batch)
    loss.backward()
    assert torch.isfinite(loss)
