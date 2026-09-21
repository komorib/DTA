import argparse
import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import lmdb

from data_utils.lmdb_dataset import count_lmdb_protein_3d_tokens, pocket_pharmacophore_can_align_to_feature_chains
from dta.split_pdbbind_dta import (
    group_overlap_report,
    grouped_split_indices,
    label_stats,
    random_split_indices,
    split_indices_for_method,
)


def sequence_to_string(sequence_value):
    if hasattr(sequence_value, "reshape") and hasattr(sequence_value, "tolist"):
        return "".join(str(item) for item in sequence_value.reshape(-1).tolist())
    if isinstance(sequence_value, (list, tuple)):
        return "".join(str(item) for item in sequence_value)
    if isinstance(sequence_value, bytes):
        return sequence_value.decode("utf-8")
    return str(sequence_value)


def ordered_chains(protein_features):
    if isinstance(protein_features, dict):
        return [protein_features[key] for key in sorted(protein_features)]
    return list(protein_features)


def pdb_id_from_key(key):
    key = str(key)
    if key.endswith("_protein"):
        return key[:-8]
    return key.split("_", 1)[0]


def load_index_rows(index_pkl):
    with open(index_pkl, "rb") as handle:
        return pickle.load(handle)


def protein_sequence_from_obj(obj):
    chains = ordered_chains(obj["protein_features"])
    return "|".join(sequence_to_string(chain.get("sequence", "")) for chain in chains)


def metadata_from_lmdb(lmdb_path, index_pkl, structure_lmdb_path=None, protein_3d_mode="residue"):
    rows = load_index_rows(index_pkl)
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    structure_env = None
    if structure_lmdb_path and os.path.abspath(structure_lmdb_path) != os.path.abspath(lmdb_path):
        structure_env = lmdb.open(structure_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)

    metadata = []
    missing = 0
    missing_structure = 0
    alignment_errors = 0

    def append_metadata_rows(txn, structure_txn):
        nonlocal missing, missing_structure, alignment_errors
        for row in rows:
            key = row["lmdb_key"]
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                missing += 1
                continue
            obj = pickle.loads(raw)

            if structure_txn is not None:
                structure_raw = structure_txn.get(key.encode("utf-8"))
                if structure_raw is None:
                    missing_structure += 1
                    continue
                structure_obj = pickle.loads(structure_raw)
            else:
                structure_obj = obj

            chains = ordered_chains(obj["protein_features"])
            structure_chains = ordered_chains(structure_obj["protein_features"])
            if protein_3d_mode in {"pocket_pharmacophore_esm", "pocket_residue_esm"} and not pocket_pharmacophore_can_align_to_feature_chains(chains, structure_chains):
                alignment_errors += 1
                continue
            sequence = protein_sequence_from_obj(obj)
            n_residue = sum(len(sequence_to_string(chain.get("sequence", ""))) for chain in chains)
            n_protein_atom = sum(
                int(chain.get("num_atoms", len(chain.get("atom_coords", []))))
                for chain in structure_chains
            )
            n_protein_3d_token = count_lmdb_protein_3d_tokens(
                structure_chains,
                protein_3d_mode=protein_3d_mode,
            )
            ligand_features = obj["ligand_features"]
            ligand_smiles = ligand_features.get("smiles", "")
            n_ligand_atom = int(ligand_features.get("num_atoms", len(ligand_features.get("atom_coords", []))))
            has_esm2_embedding = all(
                "esm2_embedding" in chain
                and getattr(chain["esm2_embedding"], "shape", (0, 0))[0]
                == len(sequence_to_string(chain.get("sequence", "")))
                for chain in chains
            )
            label = float(row.get("affinity", obj.get("affinity")))
            metadata.append({
                "lmdb_key": key,
                "pdb_id": pdb_id_from_key(key),
                "label": label,
                "paffinity": label,
                "protein_group": sequence,
                "ligand_group": ligand_smiles,
                "residue_sequence": sequence,
                "ligand_smiles": ligand_smiles,
                "ligand_scaffold_smiles": ligand_smiles,
                "n_residue": n_residue,
                "n_protein_atom": n_protein_atom,
                "n_protein_3d_token": n_protein_3d_token,
                "n_ligand_atom": n_ligand_atom,
                "has_esm2_embedding": bool(has_esm2_embedding),
            })

    try:
        with env.begin(write=False) as txn:
            if structure_env is None:
                append_metadata_rows(txn, None)
            else:
                with structure_env.begin(write=False) as structure_txn:
                    append_metadata_rows(txn, structure_txn)
    finally:
        env.close()
        if structure_env is not None:
            structure_env.close()

    return metadata, {
        "index_total": len(rows),
        "missing_lmdb_key": missing,
        "missing_structure_lmdb_key": missing_structure,
        "alignment_errors": alignment_errors,
    }



def load_fixed_split_map(tsv_path):
    split_map = {}
    with open(tsv_path, 'r', encoding='utf-8') as handle:
        header = handle.readline().strip().split('	')
        if header != ['complex_id', 'split']:
            raise ValueError(f"fixed split TSV must have columns complex_id and split, got {header}")
        for line_no, line in enumerate(handle, start=2):
            text = line.strip()
            if not text:
                continue
            parts = text.split('	')
            if len(parts) != 2:
                raise ValueError(f"invalid fixed split TSV row at line {line_no}: {text}")
            complex_id, split_name = parts
            split_name = split_name.strip().lower()
            if split_name not in {'train', 'valid', 'test'}:
                raise ValueError(f"invalid split name at line {line_no}: {split_name}")
            split_map[complex_id.strip()] = split_name
    return split_map


def fixed_split_indices(metadata, fixed_split_tsv):
    if not fixed_split_tsv:
        raise ValueError('fixed_split_tsv is required when split_method=fixed')
    split_map = load_fixed_split_map(fixed_split_tsv)
    assigned = {'train': [], 'valid': [], 'test': []}
    missing = []
    for index, item in enumerate(metadata):
        pdb_id = str(item.get('pdb_id', '')).strip()
        split_name = split_map.get(pdb_id)
        if split_name is None:
            missing.append(pdb_id)
            continue
        assigned[split_name].append(index)
    if missing:
        preview = missing[:20]
        raise ValueError(f"fixed split TSV is missing {len(missing)} pdb_ids, first few: {preview}")
    return assigned['train'], assigned['valid'], assigned['test']

def protein_3d_token_count(item, protein_3d_mode):
    if "n_protein_3d_token" in item:
        return int(item["n_protein_3d_token"])
    if protein_3d_mode == "all_atom":
        return int(item["n_protein_atom"])
    return int(item["n_residue"])


def pair_size(item, protein_3d_mode):
    return protein_3d_token_count(item, protein_3d_mode) * int(item["n_ligand_atom"])


def metadata_passes_filter(item, protein_3d_mode="residue", max_residue=0, max_protein_atom=0, max_ligand_atom=0, max_pair_tokens=0):
    if max_residue and int(item["n_residue"]) > max_residue:
        return False
    if max_protein_atom and protein_3d_token_count(item, protein_3d_mode) > max_protein_atom:
        return False
    if max_ligand_atom and int(item["n_ligand_atom"]) > max_ligand_atom:
        return False
    if max_pair_tokens and pair_size(item, protein_3d_mode) > max_pair_tokens:
        return False
    return True


def prepare_metadata(metadata, protein_3d_mode="residue", max_residue=0, max_protein_atom=0, max_ligand_atom=0, max_pair_tokens=0):
    prepared = []
    filtered_out = 0
    for item in metadata:
        if metadata_passes_filter(
            item,
            protein_3d_mode=protein_3d_mode,
            max_residue=max_residue,
            max_protein_atom=max_protein_atom,
            max_ligand_atom=max_ligand_atom,
            max_pair_tokens=max_pair_tokens,
        ):
            prepared.append(item)
        else:
            filtered_out += 1
    return prepared, filtered_out


def write_split_manifests(
    lmdb_path,
    index_pkl,
    output_dir,
    structure_lmdb_path=None,
    split_method="random",
    train_ratio=0.8,
    valid_ratio=0.1,
    seed=99,
    protein_3d_mode="residue",
    max_residue=0,
    max_protein_atom=0,
    max_ligand_atom=0,
    max_pair_tokens=0,
    fixed_split_tsv=None,
):
    raw_metadata, load_stats = metadata_from_lmdb(lmdb_path, index_pkl, structure_lmdb_path=structure_lmdb_path, protein_3d_mode=protein_3d_mode)
    metadata, filtered_out = prepare_metadata(
        raw_metadata,
        protein_3d_mode=protein_3d_mode,
        max_residue=max_residue,
        max_protein_atom=max_protein_atom,
        max_ligand_atom=max_ligand_atom,
        max_pair_tokens=max_pair_tokens,
    )
    labels = [float(item["label"]) for item in metadata]
    if split_method == "random":
        train_idx, valid_idx, test_idx = random_split_indices(
            len(metadata),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
        group_keys = None
    elif split_method == "fixed":
        train_idx, valid_idx, test_idx = fixed_split_indices(metadata, fixed_split_tsv=fixed_split_tsv)
        group_keys = None
    else:
        (train_idx, valid_idx, test_idx), group_keys = split_indices_for_method(
            metadata,
            split_method,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
    splits = {"train": train_idx, "valid": valid_idx, "test": test_idx}

    os.makedirs(output_dir, exist_ok=True)
    lmdb_path_abs = os.path.abspath(lmdb_path)
    index_pkl_abs = os.path.abspath(index_pkl)
    structure_lmdb_path_abs = os.path.abspath(structure_lmdb_path) if structure_lmdb_path else None
    sample_refs = [
        {
            "lmdb_key": item["lmdb_key"],
            "label": float(item["label"]),
            "pdb_id": item["pdb_id"],
            "protein_group": item["protein_group"],
            "ligand_group": item["ligand_group"],
            "n_residue": int(item["n_residue"]),
            "n_protein_atom": int(item["n_protein_atom"]),
            "n_protein_3d_token": int(item.get("n_protein_3d_token", item["n_protein_atom"])),
            "n_ligand_atom": int(item["n_ligand_atom"]),
            "has_esm2_embedding": bool(item.get("has_esm2_embedding", False)),
        }
        for item in metadata
    ]

    for split_name, indices in splits.items():
        manifest = {
            "dataset_type": "pdbbind_lmdb",
            "task": "dta_regression",
            "split": split_name,
            "split_method": split_method,
            "lmdb_path": lmdb_path_abs,
            "index_pkl": index_pkl_abs,
            "structure_lmdb_path": structure_lmdb_path_abs,
            "protein_3d_mode": protein_3d_mode,
            "fixed_split_tsv": os.path.abspath(fixed_split_tsv) if fixed_split_tsv else None,
            "samples": [sample_refs[index] for index in indices],
        }
        with open(os.path.join(output_dir, f"{split_name}_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    summary = {
        "input_lmdb": lmdb_path_abs,
        "input_index": index_pkl_abs,
        "structure_lmdb": structure_lmdb_path_abs,
        "output_dir": output_dir,
        "seed": seed,
        "split_method": split_method,
        "fixed_split_tsv": os.path.abspath(fixed_split_tsv) if fixed_split_tsv else None,
        "task": "dta_regression",
        "protein_3d_mode": protein_3d_mode,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": 1.0 - train_ratio - valid_ratio,
        "raw_total": len(raw_metadata),
        "total": len(metadata),
        "filtered_out": filtered_out,
        "load_stats": load_stats,
        "esm_coverage": {
            "raw_samples_all_chains_with_esm": sum(1 for item in raw_metadata if item.get("has_esm2_embedding")),
            "samples_all_chains_with_esm": sum(1 for item in metadata if item.get("has_esm2_embedding")),
        },
        "size_filter": {
            "max_residue": max_residue,
            "max_protein_atom": max_protein_atom,
            "max_ligand_atom": max_ligand_atom,
            "max_pair_tokens": max_pair_tokens,
        },
        "splits": {},
    }
    for split_name, indices in splits.items():
        split_labels = [labels[index] for index in indices]
        split_summary = {"size": len(indices), "label_stats": label_stats(split_labels)}
        if group_keys is not None:
            split_summary["group_count"] = len({group_keys[index] for index in indices})
        summary["splits"][split_name] = split_summary
    if split_method in {"protein", "both"}:
        summary["protein_group_overlap"] = group_overlap_report(sample_refs, splits, "protein_group")
    if split_method in {"mol", "both"}:
        summary["ligand_group_overlap"] = group_overlap_report(sample_refs, splits, "ligand_group")

    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create train/valid/test manifests from PDBbind LMDB.")
    parser.add_argument("--lmdb_path", required=True, help="Feature LMDB: provides chain ESM, 2D protein graph source, ligand, and labels.")
    parser.add_argument("--index_pkl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--structure_lmdb_path", default=None, help="Optional structure LMDB: provides protein 3D tokens/coordinates, e.g. pocket all-atom.")
    parser.add_argument("--split_method", choices=["random", "protein", "mol", "both", "fixed"], default="random")
    parser.add_argument("--fixed_split_tsv", default=None, help="TSV with columns complex_id and split; required when split_method=fixed.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--protein_3d_mode", choices=["residue", "all_atom", "pocket_pharmacophore", "pocket_pharmacophore_esm", "pocket_residue_esm"], default="residue")
    parser.add_argument("--max_residue", type=int, default=0)
    parser.add_argument("--max_protein_atom", type=int, default=0)
    parser.add_argument("--max_ligand_atom", type=int, default=0)
    parser.add_argument("--max_pair_tokens", type=int, default=0)
    args = parser.parse_args()

    summary = write_split_manifests(
        lmdb_path=args.lmdb_path,
        index_pkl=args.index_pkl,
        output_dir=args.output_dir,
        structure_lmdb_path=args.structure_lmdb_path,
        split_method=args.split_method,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        protein_3d_mode=args.protein_3d_mode,
        max_residue=args.max_residue,
        max_protein_atom=args.max_protein_atom,
        max_ligand_atom=args.max_ligand_atom,
        max_pair_tokens=args.max_pair_tokens,
        fixed_split_tsv=args.fixed_split_tsv,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
