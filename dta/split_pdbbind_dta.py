import argparse
import json
import os
import pickle
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from data_utils.split_pdbbind_dataset import build_both_cold_groups


def checkpoint_chunk_files(checkpoint_dir):
    return [
        os.path.join(checkpoint_dir, name)
        for name in sorted(os.listdir(checkpoint_dir))
        if name.startswith("chunk_") and name.endswith(".pkl")
    ]


def load_checkpoint_metadata(checkpoint_dir):
    metadata = []
    for chunk_path in checkpoint_chunk_files(checkpoint_dir):
        chunk_name = os.path.basename(chunk_path)
        with open(chunk_path, "rb") as handle:
            payload = pickle.load(handle)
        for local_index, item in enumerate(payload.get("metadata", [])):
            item_with_ref = dict(item)
            item_with_ref["_chunk"] = chunk_name
            item_with_ref["_local_index"] = local_index
            metadata.append(item_with_ref)
    return metadata


def random_split_indices(n_samples, train_ratio=0.8, valid_ratio=0.1, seed=99):
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")
    rng = random.Random(seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)
    n_train = int(round(n_samples * train_ratio))
    n_valid = int(round(n_samples * valid_ratio))
    return indices[:n_train], indices[n_train:n_train + n_valid], indices[n_train + n_valid:]


def grouped_split_indices(group_keys, train_ratio=0.8, valid_ratio=0.1, seed=99):
    groups = {}
    for idx, group_key in enumerate(group_keys):
        if not group_key:
            raise ValueError(f"missing group key for sample index {idx}")
        groups.setdefault(str(group_key), []).append(idx)

    split_names = ["train", "valid", "test"]
    split_ratios = {"train": train_ratio, "valid": valid_ratio, "test": 1.0 - train_ratio - valid_ratio}
    target_sizes = {name: len(group_keys) * ratio for name, ratio in split_ratios.items()}
    assigned = {name: [] for name in split_names}
    assigned_counts = {name: 0 for name in split_names}

    rng = random.Random(seed)
    group_items = [(key, indices, rng.random()) for key, indices in groups.items()]
    group_items.sort(key=lambda item: (-len(item[1]), item[2], item[0]))

    for _, indices, _ in group_items:
        split_name = min(
            split_names,
            key=lambda name: ((assigned_counts[name] + len(indices)) - target_sizes[name]) ** 2 / max(target_sizes[name], 1.0),
        )
        assigned[split_name].extend(indices)
        assigned_counts[split_name] += len(indices)

    for name in split_names:
        rng.shuffle(assigned[name])
    return assigned["train"], assigned["valid"], assigned["test"]


def get_protein_group(item):
    return item.get("protein_group") or item.get("residue_sequence")


def get_ligand_group(item):
    return item.get("ligand_group") or item.get("ligand_scaffold_smiles") or item.get("ligand_smiles")


def sample_size_score(item):
    return int(item.get("n_protein_3d_token", 0)) * int(item.get("n_ligand_atom", 0))


def metadata_passes_size_filter(item, max_residue=0, max_protein_3d_token=0, max_ligand_atom=0, max_pair_tokens=0):
    if max_residue and int(item.get("n_residue", 0)) > max_residue:
        return False
    if max_protein_3d_token and int(item.get("n_protein_3d_token", 0)) > max_protein_3d_token:
        return False
    if max_ligand_atom and int(item.get("n_ligand_atom", 0)) > max_ligand_atom:
        return False
    if max_pair_tokens and sample_size_score(item) > max_pair_tokens:
        return False
    return True


def prepare_metadata(metadata, max_residue=0, max_protein_3d_token=0, max_ligand_atom=0, max_pair_tokens=0):
    prepared = []
    filtered_out = 0
    for item in metadata:
        prepared_item = dict(item)
        prepared_item["protein_group"] = get_protein_group(prepared_item)
        prepared_item["ligand_group"] = get_ligand_group(prepared_item)
        if not metadata_passes_size_filter(
            prepared_item,
            max_residue=max_residue,
            max_protein_3d_token=max_protein_3d_token,
            max_ligand_atom=max_ligand_atom,
            max_pair_tokens=max_pair_tokens,
        ):
            filtered_out += 1
            continue
        prepared.append(prepared_item)
    return prepared, filtered_out


def split_indices_for_method(metadata, split_method, train_ratio=0.8, valid_ratio=0.1, seed=99):
    if split_method == "random":
        return random_split_indices(len(metadata), train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed), None
    if split_method == "protein":
        group_keys = [item["protein_group"] for item in metadata]
    elif split_method == "mol":
        group_keys = [item["ligand_group"] for item in metadata]
    elif split_method == "both":
        group_keys = build_both_cold_groups(metadata)
    else:
        raise ValueError("split_method must be one of: random, protein, mol, both")
    return grouped_split_indices(group_keys, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed), group_keys


def build_sample_refs(checkpoint_dir, metadata):
    sample_refs = []
    for item in metadata:
        sample_refs.append({
            "chunk": item["_chunk"],
            "local_index": int(item["_local_index"]),
            "label": float(item.get("label", item.get("paffinity"))),
            "pdb_id": item.get("pdb_id"),
            "protein_group": item.get("protein_group"),
            "ligand_group": item.get("ligand_group"),
        })
    return sample_refs


def label_stats(labels):
    if not labels:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(labels) / len(labels)
    var = sum((item - mean) ** 2 for item in labels) / len(labels)
    return {
        "count": len(labels),
        "mean": mean,
        "std": var ** 0.5,
        "min": min(labels),
        "max": max(labels),
    }


def group_overlap_report(sample_refs, splits, key):
    split_groups = {}
    for split_name, indices in splits.items():
        split_groups[split_name] = {sample_refs[index].get(key) for index in indices if sample_refs[index].get(key)}
    overlaps = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps[f"{left}_vs_{right}"] = len(split_groups[left] & split_groups[right])
    return overlaps


def write_split_manifests(checkpoint_dir, output_dir, train_ratio=0.8, valid_ratio=0.1, seed=99, split_method="random", max_residue=0, max_protein_3d_token=0, max_ligand_atom=0, max_pair_tokens=0):
    raw_metadata = load_checkpoint_metadata(checkpoint_dir)
    metadata, filtered_out = prepare_metadata(
        raw_metadata,
        max_residue=max_residue,
        max_protein_3d_token=max_protein_3d_token,
        max_ligand_atom=max_ligand_atom,
        max_pair_tokens=max_pair_tokens,
    )
    labels = [float(item.get("label", item.get("paffinity"))) for item in metadata]
    (train_idx, valid_idx, test_idx), group_keys = split_indices_for_method(
        metadata,
        split_method,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    splits = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir_abs = os.path.abspath(checkpoint_dir)
    sample_refs = build_sample_refs(checkpoint_dir, metadata)

    for split_name, indices in splits.items():
        manifest = {
            "checkpoint_dir": checkpoint_dir_abs,
            "split": split_name,
            "split_method": split_method,
            "task": "dta_regression",
            "samples": [sample_refs[i] for i in indices],
        }
        with open(os.path.join(output_dir, f"{split_name}_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    summary = {
        "input_path": checkpoint_dir_abs,
        "output_dir": output_dir,
        "seed": seed,
        "split_method": split_method,
        "task": "dta_regression",
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": 1.0 - train_ratio - valid_ratio,
        "total": len(labels),
        "raw_total": len(raw_metadata),
        "filtered_out": filtered_out,
        "size_filter": {
            "max_residue": max_residue,
            "max_protein_3d_token": max_protein_3d_token,
            "max_ligand_atom": max_ligand_atom,
            "max_pair_tokens": max_pair_tokens,
        },
        "splits": {},
    }
    for split_name, indices in splits.items():
        split_labels = [labels[i] for i in indices]
        split_summary = {"size": len(indices), "label_stats": label_stats(split_labels)}
        if group_keys is not None:
            split_summary["group_count"] = len({group_keys[i] for i in indices})
        summary["splits"][split_name] = split_summary
    if split_method in {"protein", "both"}:
        summary["protein_group_overlap"] = group_overlap_report(sample_refs, splits, "protein_group")
    if split_method in {"mol", "both"}:
        summary["ligand_group_overlap"] = group_overlap_report(sample_refs, splits, "ligand_group")

    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create DTA pAffinity train/valid/test manifests from checkpoint chunks.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", default="data/PDBbind_atom5_dta/random")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--split_method", choices=["random", "protein", "mol", "both"], default="random")
    parser.add_argument("--max_residue", type=int, default=0, help="Drop samples with more residue-level protein tokens than this; 0 disables.")
    parser.add_argument("--max_protein_3d_token", type=int, default=0, help="Drop samples with more atom5 protein tokens than this; 0 disables.")
    parser.add_argument("--max_ligand_atom", type=int, default=0, help="Drop samples with more ligand atoms than this; 0 disables.")
    parser.add_argument("--max_pair_tokens", type=int, default=0, help="Drop samples where n_protein_3d_token * n_ligand_atom exceeds this; 0 disables.")
    args = parser.parse_args()

    summary = write_split_manifests(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        split_method=args.split_method,
        max_residue=args.max_residue,
        max_protein_3d_token=args.max_protein_3d_token,
        max_ligand_atom=args.max_ligand_atom,
        max_pair_tokens=args.max_pair_tokens,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
