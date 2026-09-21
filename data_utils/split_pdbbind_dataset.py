import argparse
import json
import os
import pickle
import random
import tarfile

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass


AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def stratified_split_indices(labels, train_ratio=0.8, valid_ratio=0.1, seed=99):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")

    rng = random.Random(seed)
    by_label = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(idx)

    train_idx = []
    valid_idx = []
    test_idx = []

    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        n_total = len(label_indices)
        n_train = int(round(n_total * train_ratio))
        n_valid = int(round(n_total * valid_ratio))
        if n_total >= 3:
            n_train = min(max(n_train, 1), n_total - 2)
            n_valid = min(max(n_valid, 1), n_total - n_train - 1)
        else:
            n_train = min(n_train, n_total)
            n_valid = min(n_valid, n_total - n_train)

        train_idx.extend(label_indices[:n_train])
        valid_idx.extend(label_indices[n_train:n_train + n_valid])
        test_idx.extend(label_indices[n_train + n_valid:])

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)
    return train_idx, valid_idx, test_idx


def grouped_split_indices(labels, group_keys, train_ratio=0.8, valid_ratio=0.1, seed=99):
    if len(labels) != len(group_keys):
        raise ValueError("labels and group_keys must have the same length")
    if not labels:
        return [], [], []

    groups = {}
    for idx, group_key in enumerate(group_keys):
        if group_key is None or group_key == "":
            raise ValueError(f"missing group key for sample index {idx}")
        groups.setdefault(str(group_key), []).append(idx)

    split_names = ["train", "valid", "test"]
    split_ratios = {
        "train": train_ratio,
        "valid": valid_ratio,
        "test": 1.0 - train_ratio - valid_ratio,
    }
    total_count = len(labels)
    label_values = sorted(set(int(label) for label in labels))
    total_label_counts = {label: sum(1 for item in labels if int(item) == label) for label in label_values}
    target_sizes = {name: total_count * ratio for name, ratio in split_ratios.items()}
    target_label_counts = {
        name: {label: total_label_counts[label] * split_ratios[name] for label in label_values}
        for name in split_names
    }

    rng = random.Random(seed)
    group_items = []
    for group_key, indices in groups.items():
        counts = {label: 0 for label in label_values}
        for idx in indices:
            counts[int(labels[idx])] += 1
        group_items.append((group_key, indices, counts, rng.random()))
    group_items.sort(key=lambda item: (-len(item[1]), item[3], item[0]))

    assigned = {name: [] for name in split_names}
    assigned_counts = {name: 0 for name in split_names}
    assigned_label_counts = {name: {label: 0 for label in label_values} for name in split_names}

    def candidate_score(split_name, indices, counts):
        score = 0.0
        for name in split_names:
            future_size = assigned_counts[name] + (len(indices) if name == split_name else 0)
            size_delta = (future_size - target_sizes[name]) / max(target_sizes[name], 1.0)
            overfill = max(0.0, future_size - target_sizes[name]) / max(target_sizes[name], 1.0)
            score += size_delta * size_delta + 5.0 * overfill * overfill
            for label in label_values:
                future_label_count = assigned_label_counts[name][label] + (counts[label] if name == split_name else 0)
                label_delta = (future_label_count - target_label_counts[name][label]) / max(target_label_counts[name][label], 1.0)
                score += 2.0 * label_delta * label_delta
        return score

    for _, indices, counts, _ in group_items:
        split_name = min(split_names, key=lambda name: candidate_score(name, indices, counts))
        assigned[split_name].extend(indices)
        assigned_counts[split_name] += len(indices)
        for label in label_values:
            assigned_label_counts[split_name][label] += counts[label]

    for name in split_names:
        rng.shuffle(assigned[name])
    return assigned["train"], assigned["valid"], assigned["test"]


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_both_cold_groups(metadata):
    union_find = UnionFind()
    for item in metadata:
        protein_group = item.get("protein_group")
        ligand_group = item.get("ligand_group")
        if not protein_group or not ligand_group:
            raise ValueError("both split requires protein_group and ligand_group for every sample")
        union_find.union("protein:" + str(protein_group), "ligand:" + str(ligand_group))

    component_ids = {}
    groups = []
    for item in metadata:
        root = union_find.find("protein:" + str(item["protein_group"]))
        if root not in component_ids:
            component_ids[root] = f"both_component_{len(component_ids)}"
        groups.append(component_ids[root])
    return groups


def subset_dataset(data, indices):
    return [[field[i] for i in indices] for field in data]


def label_counts(labels):
    counts = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def write_pickle(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def empty_dataset_lists():
    return [[] for _ in range(10)]


def extend_sample(dataset, sample):
    for target, value in zip(dataset, sample):
        target.append(value)


def checkpoint_chunk_files(checkpoint_dir):
    return [
        os.path.join(checkpoint_dir, name)
        for name in sorted(os.listdir(checkpoint_dir))
        if name.startswith("chunk_") and name.endswith(".pkl")
    ]


def load_checkpoint_metadata(checkpoint_dir):
    metadata = []
    for chunk_path in checkpoint_chunk_files(checkpoint_dir):
        with open(chunk_path, "rb") as handle:
            payload = pickle.load(handle)
        metadata.extend(payload.get("metadata", []))
    return metadata


def pdb_text_to_residue_sequence(pdb_text):
    residues = []
    seen = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        residue_name = line[17:20].strip().upper()
        aa = AA3_TO_1.get(residue_name)
        if aa is None:
            continue
        residue_key = (line[21:22], line[22:27].strip())
        if residue_key in seen:
            continue
        seen.add(residue_key)
        residues.append(aa)
    return "".join(residues)


def parse_ligand_block(block, kind):
    from rdkit import Chem

    if kind == "sdf":
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    elif kind == "mol2":
        mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
    else:
        raise ValueError(f"unknown ligand block kind: {kind}")
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() == 0:
        return None
    return mol


def ligand_smiles_and_scaffold(mol):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    smiles = Chem.MolToSmiles(mol, canonical=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        scaffold = smiles
    return smiles, scaffold


def build_aux_metadata_from_pdbbind(pdbbind_dir, pdb_ids):
    pl_tar_path = os.path.join(pdbbind_dir, "P-L.tar.gz")
    if not os.path.exists(pl_tar_path):
        raise FileNotFoundError(pl_tar_path)

    needed_ids = set(pdb_ids)
    aux = {pdb_id: {} for pdb_id in needed_ids}
    with tarfile.open(pl_tar_path, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if "_" not in base:
                continue
            pdb_id = base.split("_", 1)[0].lower()
            if pdb_id not in needed_ids:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            text = handle.read().decode("utf-8", errors="replace")
            if base.endswith("_pocket.pdb"):
                sequence = pdb_text_to_residue_sequence(text)
                if sequence:
                    aux[pdb_id]["residue_sequence"] = sequence
            elif base.endswith("_ligand.sdf") and "ligand_smiles" not in aux[pdb_id]:
                try:
                    mol = parse_ligand_block(text, "sdf")
                    if mol is not None:
                        smiles, scaffold = ligand_smiles_and_scaffold(mol)
                        aux[pdb_id]["ligand_smiles"] = smiles
                        aux[pdb_id]["ligand_scaffold_smiles"] = scaffold
                except Exception:
                    pass
            elif base.endswith("_ligand.mol2") and "ligand_smiles" not in aux[pdb_id]:
                try:
                    mol = parse_ligand_block(text, "mol2")
                    if mol is not None:
                        smiles, scaffold = ligand_smiles_and_scaffold(mol)
                        aux[pdb_id]["ligand_smiles"] = smiles
                        aux[pdb_id]["ligand_scaffold_smiles"] = scaffold
                except Exception:
                    pass
    return aux


def merge_aux_metadata(metadata, pdbbind_dir=None):
    merged = [dict(item) for item in metadata]
    needs_aux = any(
        not item.get("residue_sequence") or not item.get("ligand_scaffold_smiles")
        for item in merged
    )
    if not needs_aux:
        return merged, None
    if not pdbbind_dir:
        return merged, None

    pdb_ids = [item.get("pdb_id") for item in merged if item.get("pdb_id")]
    aux_by_id = build_aux_metadata_from_pdbbind(pdbbind_dir, pdb_ids)
    for item in merged:
        item.update(aux_by_id.get(item.get("pdb_id"), {}))
    return merged, os.path.abspath(pdbbind_dir)


def get_protein_group(item):
    return item.get("protein_group") or item.get("residue_sequence")


def get_ligand_group(item):
    return item.get("ligand_group") or item.get("ligand_scaffold_smiles") or item.get("ligand_smiles")


def prepare_split_metadata(metadata, split_method, pdbbind_dir=None):
    metadata, aux_source = merge_aux_metadata(metadata, pdbbind_dir=pdbbind_dir)
    prepared = []
    for index, item in enumerate(metadata):
        prepared_item = dict(item)
        prepared_item["protein_group"] = get_protein_group(prepared_item)
        prepared_item["ligand_group"] = get_ligand_group(prepared_item)
        if split_method in {"protein", "both"} and not prepared_item["protein_group"]:
            raise ValueError(
                f"{split_method} split requires residue_sequence/protein_group for sample {index}; pass --pdbbind_dir PDBbind or regenerate metadata"
            )
        if split_method in {"mol", "both"} and not prepared_item["ligand_group"]:
            raise ValueError(
                f"{split_method} split requires ligand_scaffold_smiles/ligand_group for sample {index}; pass --pdbbind_dir PDBbind or regenerate metadata"
            )
        prepared.append(prepared_item)
    return prepared, aux_source


def split_indices_for_method(metadata, labels, split_method, train_ratio=0.8, valid_ratio=0.1, seed=99):
    if split_method == "random":
        return stratified_split_indices(labels, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed), None
    if split_method == "protein":
        group_keys = [item["protein_group"] for item in metadata]
    elif split_method == "mol":
        group_keys = [item["ligand_group"] for item in metadata]
    elif split_method == "both":
        group_keys = build_both_cold_groups(metadata)
    else:
        raise ValueError("split_method must be one of: random, protein, mol, both")
    return grouped_split_indices(labels, group_keys, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed), group_keys


def write_split_from_checkpoint(checkpoint_dir, output_path, selected_indices):
    selected_indices = set(selected_indices)
    dataset = empty_dataset_lists()
    global_index = 0
    for chunk_path in checkpoint_chunk_files(checkpoint_dir):
        with open(chunk_path, "rb") as handle:
            payload = pickle.load(handle)
        chunk_data = payload["data"]
        chunk_size = len(chunk_data[-1])
        for local_index in range(chunk_size):
            if global_index in selected_indices:
                extend_sample(dataset, [field[local_index] for field in chunk_data])
            global_index += 1
    write_pickle(output_path, dataset)


def build_sample_refs(checkpoint_dir, metadata):
    sample_refs = []
    global_index = 0
    for chunk_path in checkpoint_chunk_files(checkpoint_dir):
        chunk_name = os.path.basename(chunk_path)
        with open(chunk_path, "rb") as handle:
            payload = pickle.load(handle)
        chunk_metadata = payload.get("metadata", [])
        for local_index, _ in enumerate(chunk_metadata):
            item = metadata[global_index]
            sample_refs.append({
                "chunk": chunk_name,
                "local_index": local_index,
                "label": int(item["label"]),
                "pdb_id": item.get("pdb_id"),
                "protein_group": item.get("protein_group"),
                "ligand_group": item.get("ligand_group"),
            })
            global_index += 1
    return sample_refs


def group_overlap_report(sample_refs, splits, key):
    split_groups = {}
    for split_name, indices in splits.items():
        split_groups[split_name] = {
            sample_refs[index].get(key)
            for index in indices
            if sample_refs[index].get(key)
        }
    overlaps = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sorted(split_groups[left] & split_groups[right])
            overlaps[f"{left}_vs_{right}"] = len(overlap)
    return overlaps


def write_split_manifests_from_checkpoint(checkpoint_dir, output_dir, train_ratio=0.8, valid_ratio=0.1, seed=99, split_method="random", pdbbind_dir=None):
    raw_metadata = load_checkpoint_metadata(checkpoint_dir)
    metadata, aux_source = prepare_split_metadata(raw_metadata, split_method=split_method, pdbbind_dir=pdbbind_dir)
    labels = [int(item["label"]) for item in metadata]
    (train_idx, valid_idx, test_idx), group_keys = split_indices_for_method(
        metadata,
        labels,
        split_method=split_method,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    splits = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir_abs = os.path.abspath(checkpoint_dir)
    sample_refs = build_sample_refs(checkpoint_dir, metadata)
    for split_name, indices in splits.items():
        manifest = {
            "checkpoint_dir": checkpoint_dir_abs,
            "split": split_name,
            "split_method": split_method,
            "samples": [sample_refs[i] for i in indices],
        }
        with open(os.path.join(output_dir, f"{split_name}_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    summary = build_summary(
        input_path=checkpoint_dir_abs,
        output_dir=output_dir,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        labels=labels,
        splits=splits,
        split_method=split_method,
        sample_refs=sample_refs,
        group_keys=group_keys,
        aux_metadata_source=aux_source,
    )
    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def split_dataset_from_checkpoint(checkpoint_dir, output_dir, train_ratio=0.8, valid_ratio=0.1, seed=99, split_method="random", pdbbind_dir=None):
    metadata, aux_source = prepare_split_metadata(load_checkpoint_metadata(checkpoint_dir), split_method=split_method, pdbbind_dir=pdbbind_dir)
    labels = [int(item["label"]) for item in metadata]
    (train_idx, valid_idx, test_idx), group_keys = split_indices_for_method(
        metadata,
        labels,
        split_method=split_method,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    splits = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }
    os.makedirs(output_dir, exist_ok=True)
    for split_name, indices in splits.items():
        write_split_from_checkpoint(
            checkpoint_dir,
            os.path.join(output_dir, f"{split_name}_data.pkl"),
            indices,
        )

    sample_refs = [
        {
            "label": int(item["label"]),
            "pdb_id": item.get("pdb_id"),
            "protein_group": item.get("protein_group"),
            "ligand_group": item.get("ligand_group"),
        }
        for item in metadata
    ]
    summary = build_summary(
        input_path=checkpoint_dir,
        output_dir=output_dir,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        labels=labels,
        splits=splits,
        split_method=split_method,
        sample_refs=sample_refs,
        group_keys=group_keys,
        aux_metadata_source=aux_source,
    )
    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_summary(input_path, output_dir, seed, train_ratio, valid_ratio, labels, splits, split_method="random", sample_refs=None, group_keys=None, aux_metadata_source=None):
    summary = {
        "input_path": input_path,
        "output_dir": output_dir,
        "seed": seed,
        "split_method": split_method,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": 1.0 - train_ratio - valid_ratio,
        "total": len(labels),
        "aux_metadata_source": aux_metadata_source,
        "splits": {},
    }
    for split_name, indices in splits.items():
        split_labels = [labels[i] for i in indices]
        split_summary = {
            "size": len(indices),
            "label_counts": label_counts(split_labels),
        }
        if group_keys is not None:
            split_summary["group_count"] = len({group_keys[i] for i in indices})
        summary["splits"][split_name] = split_summary
    if sample_refs is not None and split_method in {"protein", "both"}:
        summary["protein_group_overlap"] = group_overlap_report(sample_refs, splits, "protein_group")
    if sample_refs is not None and split_method in {"mol", "both"}:
        summary["ligand_group_overlap"] = group_overlap_report(sample_refs, splits, "ligand_group")
    return summary


def split_dataset(input_path, output_dir, train_ratio=0.8, valid_ratio=0.1, seed=99, split_method="random"):
    if split_method != "random":
        raise ValueError("protein/mol/both splits require checkpoint metadata; use --checkpoint_dir")
    with open(input_path, "rb") as handle:
        data = pickle.load(handle)

    labels = data[-1]
    train_idx, valid_idx, test_idx = stratified_split_indices(
        labels,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    splits = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }
    for split_name, indices in splits.items():
        write_pickle(
            os.path.join(output_dir, f"{split_name}_data.pkl"),
            subset_dataset(data, indices),
        )

    summary = build_summary(
        input_path=input_path,
        output_dir=output_dir,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        labels=labels,
        splits=splits,
        split_method=split_method,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Create train/valid/test splits for PDBbind pkl.")
    parser.add_argument("--input", default="data/pdbbind_protein_ligand.pkl")
    parser.add_argument("--checkpoint_dir", default=None, help="Optional process_pdbbind checkpoint dir for memory-light splitting")
    parser.add_argument("--output_dir", default="data/PDBbind/random")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--split_method", choices=["random", "protein", "mol", "both"], default="random")
    parser.add_argument("--pdbbind_dir", default=None, help="Optional raw PDBbind dir used to fill missing residue_sequence/ligand scaffold metadata")
    parser.add_argument("--output_format", choices=["manifest", "pkl"], default="manifest")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        default_checkpoint_dir = args.input + ".checkpoint"
        if os.path.isdir(default_checkpoint_dir):
            checkpoint_dir = default_checkpoint_dir

    if checkpoint_dir and args.output_format == "manifest":
        summary = write_split_manifests_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            split_method=args.split_method,
            pdbbind_dir=args.pdbbind_dir,
        )
    elif checkpoint_dir:
        summary = split_dataset_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            split_method=args.split_method,
            pdbbind_dir=args.pdbbind_dir,
        )
    else:
        summary = split_dataset(
            input_path=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            split_method=args.split_method,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
