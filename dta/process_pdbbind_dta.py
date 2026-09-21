import argparse
import json
import math
import os
import pickle
import sys
import tarfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from data_utils.process_pdbbind import (
    PROTEIN_ATOM5_NAMES,
    PROTEIN_ATOM5_NUM_TOKENS,
    add_sample,
    empty_dataset_lists,
    extend_dataset_lists,
    ligand_features,
    ligand_smiles_and_scaffold,
    parse_binding_record,
    read_ligand_from_blocks,
    read_pocket_from_pdb_text,
)

PROTEIN_RESIDUE_NUM_TOKENS = 21


def select_protein_3d_inputs(residue_ids, ca_coords, atom5_tokens, atom5_coords, protein_3d_mode="residue"):
    if protein_3d_mode == "residue":
        metadata = {
            "protein_3d_mode": "residue",
            "protein_3d_atoms": ["CA"],
            "protein_3d_num_tokens": PROTEIN_RESIDUE_NUM_TOKENS,
        }
        return residue_ids, ca_coords, metadata
    if protein_3d_mode == "atom5":
        metadata = {
            "protein_3d_mode": "atom5",
            "protein_3d_atoms": list(PROTEIN_ATOM5_NAMES),
            "protein_3d_num_tokens": PROTEIN_ATOM5_NUM_TOKENS,
        }
        return atom5_tokens, atom5_coords, metadata
    raise ValueError("protein_3d_mode must be one of: residue, atom5")


def molar_to_paffinity(value_molar):
    if value_molar <= 0:
        raise ValueError("value_molar must be positive")
    return round(-math.log10(float(value_molar)), 6)


def record_to_paffinity(record, include_bounds=False, affinity_types=None):
    affinity_types = set(affinity_types or ("Kd", "Ki", "IC50"))
    if record.binding_type not in affinity_types:
        return None
    if record.operator in ("=", "=="):
        return molar_to_paffinity(record.value_molar)
    if include_bounds and record.operator in ("<", "<=", ">", ">="):
        return molar_to_paffinity(record.value_molar)
    return None


def load_affinity_records(index_tar_path, include_bounds=False, affinity_types=None):
    records = []
    with tarfile.open(index_tar_path, "r:gz") as tar:
        index_file = tar.extractfile("index/INDEX_general_PL.2020R1.lst")
        if index_file is None:
            raise FileNotFoundError("index/INDEX_general_PL.2020R1.lst not found")
        for raw_line in index_file:
            line = raw_line.decode("utf-8", errors="replace").strip()
            record = parse_binding_record(line)
            if record is None:
                continue
            paffinity = record_to_paffinity(record, include_bounds=include_bounds, affinity_types=affinity_types)
            if paffinity is None:
                continue
            records.append((record, paffinity))
    return records


def checkpoint_dir_for(args):
    return args.checkpoint_dir or (args.output + ".checkpoint")


def checkpoint_paths(args):
    ckpt_dir = checkpoint_dir_for(args)
    return ckpt_dir, os.path.join(ckpt_dir, "state.json")


def load_checkpoint(args):
    ckpt_dir, state_path = checkpoint_paths(args)
    state = {
        "completed_ids": [],
        "failed_ids": [],
        "stats": None,
        "next_chunk_index": 0,
    }
    if not os.path.isdir(ckpt_dir):
        return state
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            state.update(json.load(handle))
    return state


def write_checkpoint_state(args, state):
    ckpt_dir, state_path = checkpoint_paths(args)
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    os.replace(tmp_path, state_path)


def write_checkpoint_chunk(args, chunk_data, chunk_metadata, state):
    if not chunk_metadata:
        return
    ckpt_dir, _ = checkpoint_paths(args)
    os.makedirs(ckpt_dir, exist_ok=True)
    chunk_index = int(state.get("next_chunk_index", 0))
    chunk_path = os.path.join(ckpt_dir, f"chunk_{chunk_index:06d}.pkl")
    tmp_path = chunk_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump({"data": chunk_data, "metadata": chunk_metadata}, handle)
    os.replace(tmp_path, chunk_path)
    state["next_chunk_index"] = chunk_index + 1
    write_checkpoint_state(args, state)


def load_checkpoint_dataset(args):
    ckpt_dir, _ = checkpoint_paths(args)
    dataset = empty_dataset_lists()
    metadata = []
    if not os.path.isdir(ckpt_dir):
        return dataset, metadata
    chunk_files = sorted(name for name in os.listdir(ckpt_dir) if name.startswith("chunk_") and name.endswith(".pkl"))
    for name in chunk_files:
        with open(os.path.join(ckpt_dir, name), "rb") as handle:
            payload = pickle.load(handle)
        extend_dataset_lists(dataset, payload["data"])
        metadata.extend(payload.get("metadata", []))
    return dataset, metadata


def build_dataset(args):
    index_tar_path = os.path.join(args.pdbbind_dir, "index.tar.gz")
    pl_tar_path = os.path.join(args.pdbbind_dir, "P-L.tar.gz")
    if not os.path.exists(index_tar_path):
        raise FileNotFoundError(index_tar_path)
    if not os.path.exists(pl_tar_path):
        raise FileNotFoundError(pl_tar_path)

    ckpt_dir, _ = checkpoint_paths(args)
    if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir) and not args.resume:
        raise RuntimeError(f"checkpoint dir exists: {ckpt_dir}; pass --resume to continue or choose a different --checkpoint_dir")

    affinity_types = tuple(item.strip() for item in args.affinity_types.split(",") if item.strip())
    records = load_affinity_records(index_tar_path, include_bounds=args.include_bounds, affinity_types=affinity_types)
    record_by_id = {record.pdb_id: (record, paffinity) for record, paffinity in records}

    state = load_checkpoint(args) if args.resume else {
        "completed_ids": [],
        "failed_ids": [],
        "stats": None,
        "next_chunk_index": 0,
    }
    completed_pdb_ids = set(state.get("completed_ids", []))
    failed_pdb_ids = set(state.get("failed_ids", []))
    stats = state.get("stats") or {
        "candidate_records": len(records),
        "kept": len(completed_pdb_ids),
        "missing_structure": 0,
        "parse_errors": len(failed_pdb_ids),
    }
    stats["candidate_records"] = len(records)

    pending = {}
    chunk_data = empty_dataset_lists()
    chunk_metadata = []

    def persist_chunk():
        nonlocal chunk_data, chunk_metadata
        if not chunk_metadata:
            return
        state["completed_ids"] = sorted(completed_pdb_ids)
        state["failed_ids"] = sorted(failed_pdb_ids)
        state["stats"] = stats
        write_checkpoint_chunk(args, chunk_data, chunk_metadata, state)
        chunk_data = empty_dataset_lists()
        chunk_metadata = []

    def process_pending_entry(pdb_id):
        if pdb_id in completed_pdb_ids or pdb_id in failed_pdb_ids:
            pending.pop(pdb_id, None)
            return
        entry = pending.get(pdb_id, {})
        if "pocket_text" not in entry or ("sdf_text" not in entry and "mol2_text" not in entry):
            return

        record, paffinity = record_by_id[pdb_id]
        pending.pop(pdb_id, None)
        try:
            p_repre, p_graph, p_residue_ids, p_ca, p_atom5_feats, p_atom5_coors, p_residue_sequence = read_pocket_from_pdb_text(
                entry["pocket_text"],
                distance_cutoff=args.protein_contact_cutoff,
            )
            p_3d_feats, p_3d_coors, protein_3d_metadata = select_protein_3d_inputs(
                p_residue_ids,
                p_ca,
                p_atom5_feats,
                p_atom5_coors,
                protein_3d_mode=args.protein_3d_mode,
            )
            ligand_mol = read_ligand_from_blocks(entry.get("sdf_text"), entry.get("mol2_text"))
            l_graph, l_feats, l_coors, l_las = ligand_features(ligand_mol)
            ligand_smiles, ligand_scaffold = ligand_smiles_and_scaffold(ligand_mol)
        except Exception as exc:
            failed_pdb_ids.add(pdb_id)
            stats["parse_errors"] += 1
            state["completed_ids"] = sorted(completed_pdb_ids)
            state["failed_ids"] = sorted(failed_pdb_ids)
            state["stats"] = stats
            write_checkpoint_state(args, state)
            if args.verbose:
                print(f"skip {pdb_id}: {exc}", flush=True)
            return

        sample = (
            p_repre,
            l_graph,
            p_graph,
            p_3d_feats,
            p_ca,
            p_3d_coors,
            l_feats,
            l_coors,
            l_las,
            float(paffinity),
        )
        item_metadata = {
            "pdb_id": pdb_id,
            "binding": record.raw,
            "binding_type": record.binding_type,
            "binding_operator": record.operator,
            "binding_value_molar": float(record.value_molar),
            "paffinity": float(paffinity),
            "label": float(paffinity),
            "n_residue": int(len(p_residue_ids)),
            "n_protein_3d_token": int(len(p_3d_feats)),
            "n_ligand_atom": int(len(l_feats)),
            "ligand_smiles": ligand_smiles,
            "ligand_scaffold_smiles": ligand_scaffold,
            "residue_sequence": p_residue_sequence,
            "residue_ids": p_residue_ids.tolist(),
            **protein_3d_metadata,
        }
        add_sample(chunk_data, sample)
        chunk_metadata.append(item_metadata)
        completed_pdb_ids.add(pdb_id)
        stats["kept"] += 1

        if args.progress_every and stats["kept"] % args.progress_every == 0:
            print(json.dumps(stats), flush=True)
        if args.checkpoint_every and len(chunk_metadata) >= args.checkpoint_every:
            persist_chunk()

    with tarfile.open(pl_tar_path, "r|gz") as tar:
        for member in tar:
            if args.max_samples and stats["kept"] >= args.max_samples:
                break
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if "_" not in base:
                continue
            pdb_id = base.split("_", 1)[0].lower()
            if pdb_id not in record_by_id or pdb_id in completed_pdb_ids or pdb_id in failed_pdb_ids:
                continue

            field = None
            if base.endswith("_pocket.pdb"):
                field = "pocket_text"
            elif base.endswith("_ligand.sdf"):
                field = "sdf_text"
            elif base.endswith("_ligand.mol2"):
                field = "mol2_text"
            if field is None:
                continue

            handle = tar.extractfile(member)
            if handle is None:
                continue
            pending.setdefault(pdb_id, {})[field] = handle.read().decode("utf-8", errors="replace")
            process_pending_entry(pdb_id)

    for pdb_id in list(pending):
        if args.max_samples and stats["kept"] >= args.max_samples:
            break
        process_pending_entry(pdb_id)

    persist_chunk()
    if not args.max_samples:
        stats["missing_structure"] = len(record_by_id) - len(completed_pdb_ids) - len(failed_pdb_ids)
    state["completed_ids"] = sorted(completed_pdb_ids)
    state["failed_ids"] = sorted(failed_pdb_ids)
    state["stats"] = stats
    write_checkpoint_state(args, state)
    dataset, metadata = load_checkpoint_dataset(args)
    return dataset, metadata, stats


def main():
    parser = argparse.ArgumentParser(description="Convert PDBbind protein-ligand data to DTA pAffinity regression data.")
    parser.add_argument("--pdbbind_dir", default="PDBbind")
    parser.add_argument("--output", default="data/pdbbind_protein_ligand_dta.pkl")
    parser.add_argument("--metadata_output", default="data/pdbbind_protein_ligand_dta_metadata.json")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--affinity_types", default="Kd,Ki,IC50")
    parser.add_argument("--include_bounds", action="store_true", help="Include <, <=, >, >= records as approximate labels. Default keeps exact values only.")
    parser.add_argument("--protein_contact_cutoff", type=float, default=8.0)
    parser.add_argument("--protein_3d_mode", choices=["residue", "atom5"], default="residue", help="Protein 3D representation: residue uses amino-acid ids with CA coordinates; atom5 uses N/CA/C/O/CB tokens.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data, metadata, stats = build_dataset(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(data, handle)
    os.makedirs(os.path.dirname(args.metadata_output) or ".", exist_ok=True)
    with open(args.metadata_output, "w", encoding="utf-8") as handle:
        json.dump({"stats": stats, "samples": metadata}, handle, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote dataset: {args.output}")
    print(f"wrote metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
