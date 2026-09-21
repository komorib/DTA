#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem

# Inlined from data_utils.lmdb_dataset so this script is fully self-contained (no GerNA repo needed).
def ordered_chain_items(protein_features):
    if isinstance(protein_features, dict):
        return [(key, protein_features[key]) for key in sorted(protein_features)]
    return [(str(index), chain) for index, chain in enumerate(protein_features)]


def resolve_manifest_path(manifest_dir, path_value):
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(manifest_dir, path_value)

META_KEY = b'__metadata__'

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    'X': 'UNK',
}


def load_manifest(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('dataset_type') != 'pdbbind_lmdb':
        raise ValueError(f'not a PDBbind LMDB manifest: {manifest_path}')
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    lmdb_path = resolve_manifest_path(manifest_dir, manifest['lmdb_path'])
    split_name = manifest.get('split') or Path(manifest_path).stem.replace('_manifest', '')
    return manifest, lmdb_path, split_name


def collect_samples(manifest_paths, feature_lmdb_override=None):
    samples = []
    seen = set()
    lmdb_paths = []
    split_counts = {}
    for manifest_path in manifest_paths:
        manifest, lmdb_path, split_name = load_manifest(manifest_path)
        if lmdb_path is not None:
            lmdb_paths.append(os.path.abspath(lmdb_path))
        split_counts[split_name] = len(manifest.get('samples', []))
        for sample in manifest.get('samples', []):
            key = sample['lmdb_key']
            if key in seen:
                continue
            seen.add(key)
            samples.append(sample)
    # The manifest's own lmdb_path is often a stale absolute path from another machine.
    # --feature-lmdb-path overrides it with the real feature LMDB location on THIS server.
    if feature_lmdb_override:
        return samples, os.path.abspath(feature_lmdb_override), split_counts
    unique_lmdb_paths = sorted(set(lmdb_paths))
    if len(unique_lmdb_paths) != 1:
        raise ValueError(
            f'manifests do not agree on one feature LMDB ({unique_lmdb_paths}); pass --feature-lmdb-path to override'
        )
    return samples, unique_lmdb_paths[0], split_counts


def open_read_lmdb(path):
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True, max_readers=256)


def open_write_lmdb(path, map_size_gb):
    Path(path).mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(path),
        subdir=True,
        readonly=False,
        create=True,
        lock=True,
        meminit=False,
        map_size=int(map_size_gb) * (1024 ** 3),
        max_readers=256,
    )


def load_feature_item(txn, key):
    raw = txn.get(key.encode('utf-8'))
    if raw is None:
        raise KeyError(f'feature LMDB key not found: {key}')
    return pickle.loads(raw)


def output_contains(txn, key):
    return txn.get(key.encode('utf-8')) is not None


def infer_element(atom_name):
    text = ''.join(ch for ch in str(atom_name).strip() if ch.isalpha()).upper()
    if not text:
        return 'C'
    if len(text) >= 2 and text[:2] in {'CL', 'BR', 'FE', 'ZN', 'MG', 'MN', 'CA', 'NA'}:
        return text[:2].title()
    return text[0]


def write_ligand_sdf(feature_item, path):
    ligand_features = feature_item['ligand_features']
    if 'mol' not in ligand_features:
        raise ValueError('feature LMDB ligand_features does not contain RDKit mol')
    mol = Chem.Mol(ligand_features['mol'])
    coords = np.asarray(ligand_features.get('atom_coords'), dtype=np.float64)
    if coords.ndim == 2 and coords.shape[0] == mol.GetNumAtoms():
        if mol.GetNumConformers() == 0:
            conf = Chem.Conformer(mol.GetNumAtoms())
            mol.AddConformer(conf, assignId=True)
        conf = mol.GetConformer()
        for idx, xyz in enumerate(coords):
            conf.SetAtomPosition(idx, tuple(float(v) for v in xyz))
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def write_pocket_pdb(feature_item, path):
    serial = 1
    lines = []
    chain_items = ordered_chain_items(feature_item['protein_features'])
    for chain_order, (chain_key, chain) in enumerate(chain_items):
        chain_id = str(chain.get('chain_id') or chain_key or chr(ord('A') + chain_order))[0]
        atom_names = np.asarray(chain['atom_names']).reshape(-1)
        coords = np.asarray(chain['atom_coords'], dtype=np.float64)
        residue_names = np.asarray(chain.get('atom_residue_names', ['X'] * len(atom_names))).reshape(-1)
        residue_indices = np.asarray(chain.get('atom_residue_indices', np.arange(len(atom_names))), dtype=np.int64).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != len(atom_names):
            raise ValueError(f'invalid protein atom coords shape: {coords.shape}')
        for atom_name, res_name, res_idx, xyz in zip(atom_names, residue_names, residue_indices, coords):
            atom_name = str(atom_name).strip()[:4] or 'C'
            res_name = str(res_name).strip()
            res3 = ONE_TO_THREE.get(res_name, res_name[:3].upper() if len(res_name) >= 3 else 'UNK')
            element = infer_element(atom_name)
            x, y, z = (float(v) for v in xyz)
            lines.append(
                f'ATOM  {serial:5d} {atom_name:>4s} {res3:>3s} {chain_id:1s}{int(res_idx) + 1:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n'
            )
            serial += 1
            if serial > 99999:
                break
    lines.append('END\n')
    Path(path).write_text(''.join(lines), encoding='ascii')


def import_pbcnet_modules(pbcnet_root):
    pbcnet_root = Path(pbcnet_root).resolve()
    model_code = pbcnet_root / 'model_code'
    if not model_code.exists():
        raise FileNotFoundError(f'PBCNet model_code not found under {pbcnet_root}')
    for entry in (str(pbcnet_root), str(model_code)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from Graph2pickle import Graph_Information
    return Graph_Information


def load_pbcnet_model(checkpoint, device):
    try:
        model = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint, map_location=device)
    model.to(device)
    model.eval()
    return model


import dgl.function as fn


def src_norm_dst(src, dst, out):
    def func(edges):
        return {out: torch.sqrt(torch.sum((edges.src[src] - edges.dst[dst]) ** 2, dim=1))}
    return func


def extract_per_atom(model, graph, device):
    graph = graph.to(device)
    graph.edges['int'].data['bond_scalar'] = torch.clip(graph.edges['int'].data['bond_scalar'].long(), min=0, max=4)
    with torch.no_grad():
        with graph.local_scope():
            graph.apply_edges(src_norm_dst('pos', 'pos', 'dist'), etype='int')
            graph.apply_edges(fn.v_sub_u('pos', 'pos', 'vec'), etype='int')
            emb, _ = model.encoder(graph, False)        # [N_total, hidden]
            type_mask = graph.nodes['atom'].data['type']  # 1=ligand, 0=pocket
    lig = (type_mask == 1)
    return emb[lig].detach().cpu().float().numpy().astype('float32')  # [N_lig, hidden]


def ligand_atom_symbols(ligand_sdf):
    lig_mol = Chem.RemoveAllHs(Chem.MolFromMolFile(str(ligand_sdf)))
    if lig_mol is None:
        raise ValueError(f'could not parse ligand SDF for atom symbols: {ligand_sdf}')
    return [atom.GetSymbol() for atom in lig_mol.GetAtoms()]


def write_metadata(env, metadata):
    with env.begin(write=True) as txn:
        txn.put(META_KEY, json.dumps(metadata, indent=2, ensure_ascii=True).encode('utf-8'))


def extract(args):
    Graph_Information = import_pbcnet_modules(args.pbcnet_root)
    device = torch.device(f'cuda:{args.cuda}' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    model = load_pbcnet_model(args.checkpoint, device)
    hidden_dim = int(getattr(model, 'hidden_channels', args.pbcnet_dim))

    samples, feature_lmdb_path, split_counts = collect_samples(args.manifest, feature_lmdb_override=args.feature_lmdb_path)
    # Deterministic sharding for CPU multi-process parallelism: each shard processes a disjoint
    # subset (sample index % num_shards == shard) and writes its own --output-lmdb; merge afterwards.
    if int(args.num_shards) > 1:
        samples = [s for idx, s in enumerate(samples) if idx % int(args.num_shards) == int(args.shard)]
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    feature_env = open_read_lmdb(feature_lmdb_path)
    output_env = open_write_lmdb(args.output_lmdb, args.map_size_gb)

    tmp_root = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix='pbcnet_extract_'))
    tmp_root.mkdir(parents=True, exist_ok=True)
    cleanup_tmp = args.tmp_dir is None

    written = 0
    skipped_existing = 0
    errors = []
    processed = 0
    write_txn = output_env.begin(write=True)
    try:
        with feature_env.begin(write=False) as feature_txn:
            for sample in samples:
                processed += 1
                key = sample['lmdb_key']
                if args.resume and output_contains(write_txn, key):
                    skipped_existing += 1
                    continue
                try:
                    feature_item = load_feature_item(feature_txn, key)
                    sample_dir = tmp_root / key
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    ligand_sdf = sample_dir / 'ligand.sdf'
                    pocket_pdb = sample_dir / 'pocket.pdb'
                    write_ligand_sdf(feature_item, ligand_sdf)
                    write_pocket_pdb(feature_item, pocket_pdb)
                    graph = Graph_Information(str(ligand_sdf), str(pocket_pdb))
                    lig_emb = extract_per_atom(model, graph, device)
                    if lig_emb.ndim != 2 or lig_emb.shape[1] != hidden_dim:
                        raise ValueError(f'PBCNet per-atom embedding dim mismatch: got {lig_emb.shape}, expected [N_lig, {hidden_dim}]')
                    atoms = ligand_atom_symbols(ligand_sdf)
                    if len(atoms) != lig_emb.shape[0]:
                        raise ValueError(f'ligand atom count mismatch: symbols={len(atoms)}, embeddings={lig_emb.shape[0]}')
                    payload = {
                        'payload_type': 'pbcnet_atom',
                        'ligand_tokens': lig_emb,
                        'ligand_mask': np.ones(lig_emb.shape[0], bool),
                        'atoms': atoms,
                        'source': 'PBCNet2.0.encoder_per_atom',
                    }
                    write_txn.put(key.encode('utf-8'), pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                    written += 1
                    if args.clean_tmp_each:
                        shutil.rmtree(sample_dir, ignore_errors=True)
                except Exception as exc:
                    if len(errors) < args.error_preview:
                        errors.append({'key': key, 'error': str(exc)})
                if processed % args.commit_every == 0:
                    write_txn.commit()
                    write_txn = output_env.begin(write=True)
                if args.progress_every and processed % args.progress_every == 0:
                    print(json.dumps({
                        'processed': processed,
                        'candidate_samples': len(samples),
                        'written': written,
                        'skipped_existing': skipped_existing,
                        'errors': len(errors),
                    }, ensure_ascii=True))
        write_txn.commit()
        write_txn = None
    finally:
        if write_txn is not None:
            write_txn.abort()
        if cleanup_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        feature_env.close()

    metadata = {
        'manifest_paths': [os.path.abspath(path) for path in args.manifest],
        'feature_lmdb': os.path.abspath(feature_lmdb_path),
        'output_lmdb': os.path.abspath(args.output_lmdb),
        'pbcnet_root': os.path.abspath(args.pbcnet_root),
        'checkpoint': os.path.abspath(args.checkpoint),
        'split_counts': split_counts,
        'candidate_samples': len(samples),
        'written': written,
        'skipped_existing': skipped_existing,
        'errors': len(errors),
        'hidden_dim': hidden_dim,
        'representation': 'PBCNet2.0_encoder_per_atom',
        'errors_preview': errors,
    }
    write_metadata(output_env, metadata)
    output_env.close()
    if args.metadata_output:
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description='Extract PBCNet2.0 encoder per-atom ligand embeddings into LMDB.')
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--manifest', action='append', required=True, help='PDBbind LMDB manifest path. Pass train/valid/test manifests by repeating this flag.')
    parser.add_argument('--feature-lmdb-path', required=True, help='Path to the feature LMDB (PDBbind_chain_lmdb_esm_gernabind_v2) on THIS server. Overrides the stale lmdb_path inside the manifests.')
    parser.add_argument('--pbcnet-root', default=os.path.join(_here, 'PBCNet2.0'), help='Directory containing Graph2pickle.py and model_code/ (default: ./PBCNet2.0).')
    parser.add_argument('--checkpoint', default=os.path.join(_here, 'PBCNet2.0', 'PBCNet2.pth'), help='PBCNet2.0 torch checkpoint (default: ./PBCNet2.0/PBCNet2.pth).')
    parser.add_argument('--output-lmdb', required=True)
    parser.add_argument('--metadata-output', default=None)
    parser.add_argument('--map-size-gb', type=int, default=8)
    parser.add_argument('--pbcnet-dim', type=int, default=128)
    parser.add_argument('--commit-every', type=int, default=100)
    parser.add_argument('--progress-every', type=int, default=500)
    parser.add_argument('--error-preview', type=int, default=30)
    parser.add_argument('--tmp-dir', default=None)
    parser.add_argument('--clean-tmp-each', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--num-shards', type=int, default=1, help='Split samples into this many disjoint shards for CPU multi-process parallelism.')
    parser.add_argument('--shard', type=int, default=0, help='Shard index in [0, num_shards) handled by this process.')
    parser.add_argument('--max-samples', type=int, default=0, help='If >0, only process the first N samples (after sharding). For smoke tests.')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
