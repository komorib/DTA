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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.lmdb_dataset import ordered_chain_items, resolve_manifest_path

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


def collect_samples(manifest_paths):
    samples = []
    seen = set()
    lmdb_paths = []
    split_counts = {}
    for manifest_path in manifest_paths:
        manifest, lmdb_path, split_name = load_manifest(manifest_path)
        lmdb_paths.append(os.path.abspath(lmdb_path))
        split_counts[split_name] = len(manifest.get('samples', []))
        for sample in manifest.get('samples', []):
            key = sample['lmdb_key']
            if key in seen:
                continue
            seen.add(key)
            samples.append(sample)
    unique_lmdb_paths = sorted(set(lmdb_paths))
    if len(unique_lmdb_paths) != 1:
        raise ValueError(f'all manifests must point to the same feature LMDB, got {unique_lmdb_paths}')
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


def extract_readout(model, graph, device):
    graph = graph.to(device)
    graph.edges['int'].data['bond_scalar'] = torch.clip(graph.edges['int'].data['bond_scalar'].long(), min=0, max=4)
    with torch.no_grad():
        emb, _ = model._readout(graph, False)
    return emb.detach().cpu().float().numpy().reshape(-1).astype(np.float32)


def write_metadata(env, metadata):
    with env.begin(write=True) as txn:
        txn.put(META_KEY, json.dumps(metadata, indent=2, ensure_ascii=True).encode('utf-8'))


def extract(args):
    Graph_Information = import_pbcnet_modules(args.pbcnet_root)
    device = torch.device(f'cuda:{args.cuda}' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    model = load_pbcnet_model(args.checkpoint, device)
    hidden_dim = int(getattr(model, 'hidden_channels', args.pbcnet_dim))

    samples, feature_lmdb_path, split_counts = collect_samples(args.manifest)
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
                    global_feature = extract_readout(model, graph, device)
                    if global_feature.shape[0] != hidden_dim:
                        raise ValueError(f'PBCNet embedding dim mismatch: got {global_feature.shape[0]}, expected {hidden_dim}')
                    payload = {
                        'payload_type': 'pbcnet',
                        'pbcnet_global': global_feature,
                        'global': global_feature,
                        'source': 'PBCNet2.0._readout',
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
        'representation': 'PBCNet2.0_readout_global',
        'errors_preview': errors,
    }
    write_metadata(output_env, metadata)
    output_env.close()
    if args.metadata_output:
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description='Extract PBCNet2.0 _readout pocket-ligand global embeddings into LMDB.')
    parser.add_argument('--manifest', action='append', required=True, help='PDBbind LMDB manifest path. Pass train/valid/test manifests by repeating this flag.')
    parser.add_argument('--pbcnet-root', required=True, help='Directory containing Graph2pickle.py, model_code/, and PBCNet2.pth.')
    parser.add_argument('--checkpoint', required=True, help='PBCNet2.0 torch checkpoint, usually PBCNet2.pth.')
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
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
