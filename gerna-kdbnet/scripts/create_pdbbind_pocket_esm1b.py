#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kdbnet.pdbbind import BACKBONE_ATOMS, THREE_TO_ONE


def parse_pocket_residues(pdb_path):
    residues = []
    residue_index = {}
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom not in BACKBONE_ATOMS:
                continue
            resname = line[17:20].strip().upper()
            aa = THREE_TO_ONE.get(resname)
            if aa is None:
                continue
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key not in residue_index:
                residue_index[key] = len(residues)
                residues.append({"key": key, "chain": chain, "aa": aa, "atoms": {}})
            try:
                coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            residues[residue_index[key]]["atoms"][atom] = coord

    valid = [residue for residue in residues
             if all(atom in residue["atoms"] for atom in BACKBONE_ATOMS)]
    if not valid:
        raise ValueError("pocket PDB contains no residues with complete N/CA/C/O backbone")
    return valid


def make_chain_sequences(residues):
    chain_to_indices = {}
    for idx, residue in enumerate(residues):
        chain_to_indices.setdefault(residue["chain"], []).append(idx)
    chain_records = []
    for chain, indices in chain_to_indices.items():
        seq = "".join(residues[idx]["aa"] for idx in indices)
        chain_records.append((chain, indices, seq))
    return chain_records


def load_esm_model(model_name, model_path, device):
    import esm

    if model_path is not None:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        if not hasattr(esm.pretrained, model_name):
            raise ValueError("Unknown ESM model function: {}".format(model_name))
        model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(device)
    return model, alphabet


def embed_sequences(model, alphabet, records, repr_layer, device):
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([(name, seq) for name, seq in records])
    tokens = tokens.to(device)
    with torch.no_grad():
        output = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = output["representations"][repr_layer].detach().cpu()
    return [reps[i, 1:1 + len(seq)].clone() for i, (_, seq) in enumerate(records)]


def main():
    parser = argparse.ArgumentParser(
        description="Create pocket-level ESM-1b embeddings for PDBbind PL."
    )
    parser.add_argument(
        "--manifest",
        default="data/PDBbind_full/processed/pdbbind_pl_manifest.tsv",
        help="PDBbind PL manifest TSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/PDBbind_full/esm1b",
        help="Directory for generated .pt files.",
    )
    parser.add_argument(
        "--output_manifest",
        default="data/PDBbind_full/processed/pdbbind_pl_manifest_esm.tsv",
        help="Manifest copy with an added esm_pt column.",
    )
    parser.add_argument(
        "--model_name",
        default="esm1b_t33_650M_UR50S",
        help="Function name under esm.pretrained used when --model_path is not set.",
    )
    parser.add_argument(
        "--model_path",
        help="Local ESM checkpoint path. If unset, fair-esm loads by --model_name.",
    )
    parser.add_argument("--repr_layer", type=int, default=33, help="ESM representation layer.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device.")
    parser.add_argument("--limit", type=int, help="Only process the first N manifest rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files.")
    parser.add_argument("--report", default="data/PDBbind_full/processed/pdbbind_pl_esm_report.json")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_manifest = Path(args.output_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_table(manifest_path)
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()
    required_cols = {"complex_id", "pocket_pdb"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError("Manifest missing columns: {}".format(sorted(missing)))

    model, alphabet = load_esm_model(args.model_name, args.model_path, device)

    skipped = []
    output_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="esm"):
        complex_id = str(row["complex_id"]).lower()
        pocket_pdb = str(row["pocket_pdb"])
        output_path = output_dir / "{}.pt".format(complex_id)
        if output_path.exists() and not args.overwrite:
            output_paths.append(str(output_path))
            continue
        try:
            residues = parse_pocket_residues(pocket_pdb)
            chain_records = make_chain_sequences(residues)
            embeddings = torch.empty((len(residues), 1280), dtype=torch.float32)
            records = [("{}:{}".format(complex_id, chain), seq)
                       for chain, _, seq in chain_records]
            chain_embeddings = embed_sequences(
                model, alphabet, records, args.repr_layer, device)
            for (_, indices, seq), chain_embedding in zip(chain_records, chain_embeddings):
                if chain_embedding.shape[0] != len(seq):
                    raise ValueError("ESM length mismatch for {}".format(complex_id))
                embeddings[indices] = chain_embedding
            torch.save(embeddings, output_path)
            output_paths.append(str(output_path))
        except Exception as exc:
            skipped.append({
                "complex_id": complex_id,
                "pocket_pdb": pocket_pdb,
                "reason": type(exc).__name__,
                "message": str(exc),
            })
            if output_path.exists():
                output_path.unlink()
            output_paths.append("")

    df["esm_pt"] = output_paths
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest, sep="\t", index=False)

    report = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "output_manifest": str(output_manifest),
        "rows": len(df),
        "created_or_existing": int(sum(bool(path) and Path(path).exists()
                                       for path in output_paths)),
        "skipped": len(skipped),
        "skipped_examples": skipped[:50],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("wrote_manifest={}".format(output_manifest))
    print("wrote_report={}".format(report_path))
    print("created_or_existing={}/{}".format(report["created_or_existing"], len(df)))
    print("skipped={}".format(len(skipped)))


if __name__ == "__main__":
    main()
