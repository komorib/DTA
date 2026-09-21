#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace

import lmdb
import numpy as np
import torch
from rdkit import Chem


def add_repo_paths(repo_root, unicore_root):
    unimol2_root = os.path.join(repo_root, 'external', 'Uni-Mol', 'unimol2')
    for path in [unicore_root, unimol2_root]:
        if path and path not in sys.path:
            sys.path.insert(0, path)


def build_args():
    return SimpleNamespace(
        encoder_layers=12,
        encoder_embed_dim=768,
        pair_embed_dim=512,
        pair_hidden_dim=64,
        encoder_ffn_embed_dim=768,
        encoder_attention_heads=48,
        dropout=0.1,
        emb_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pooler_dropout=0.0,
        max_seq_len=512,
        activation_fn='gelu',
        pooler_activation_fn='tanh',
        post_ln=False,
        masked_token_loss=-1.0,
        masked_coord_loss=-1.0,
        masked_dist_loss=-1.0,
        masked_coord_dist_loss=-1.0,
        x_norm_loss=-1.0,
        delta_pair_repr_norm_loss=-1.0,
        notri=False,
        gaussian_std_width=1.0,
        gaussian_mean_start=0.0,
        gaussian_mean_stop=9.0,
        mode='infer',
        droppath_prob=0.0,
    )


def open_lmdb(path, readonly=True, map_size_gb=16):
    return lmdb.open(
        path,
        readonly=readonly,
        lock=False if readonly else True,
        readahead=False,
        meminit=False,
        subdir=True,
        map_size=int(map_size_gb * 1024 ** 3),
    )


def load_lmdb_item(env, key):
    with env.begin(write=False) as txn:
        raw = txn.get(key.encode('utf-8'))
    if raw is None:
        raise KeyError(key)
    return pickle.loads(raw)


def heavy_atom_symbols(mol):
    return [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']


def remove_hs_mol(mol):
    return Chem.RemoveHs(mol, sanitize=False)


def make_item(record, mol):
    from unimol2.data.molecule_dataset import get_graph
    from unimol2.data.unimol2_dataset import get_graph_features

    coords = np.asarray(record['coordinates'], dtype=np.float32)
    atoms = list(record['atoms'])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'bad coordinates shape: {coords.shape}')
    if len(atoms) != coords.shape[0]:
        raise ValueError(f'atoms/coords length mismatch: {len(atoms)} vs {coords.shape[0]}')

    mol_no_h = remove_hs_mol(mol)
    mol_atoms = heavy_atom_symbols(mol_no_h)
    if mol_no_h.GetNumAtoms() != len(atoms):
        raise ValueError(f'atom count mismatch: rdkit={mol_no_h.GetNumAtoms()} json={len(atoms)}')
    if mol_atoms != atoms:
        raise ValueError(f'atom order mismatch: rdkit={mol_atoms[:20]} json={atoms[:20]}')

    node_attr, edge_index, edge_attr = get_graph(mol_no_h)
    feat = get_graph_features(edge_attr, edge_index, node_attr, drop_feat=False)
    src_pos = coords - coords.mean(axis=0, keepdims=True)
    src_token = np.asarray([atom.GetAtomicNum() for atom in mol_no_h.GetAtoms()], dtype=np.int64)

    item = {
        'src_token': torch.from_numpy(src_token).long(),
        'src_pos': torch.from_numpy(src_pos).float(),
    }
    item.update(feat)
    return item


def collate_items(items):
    from unimol2.data.unimol2_dataset import pad_1d, pad_1d_feat, pad_2d, pad_2d_feat, pad_attn_bias

    pad_fns = {
        'src_token': pad_1d,
        'src_pos': pad_1d_feat,
        'atom_feat': pad_1d_feat,
        'atom_mask': pad_1d,
        'edge_feat': pad_2d_feat,
        'shortest_path': pad_2d,
        'degree': pad_1d,
        'pair_type': pad_2d_feat,
        'attn_bias': pad_attn_bias,
    }
    max_node_num = max(item['atom_mask'].shape[0] for item in items)
    max_node_num = (max_node_num + 1 + 3) // 4 * 4 - 1
    return {key: pad_fns[key]([item[key] for item in items], max_node_num) for key in pad_fns}


def move_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def load_model(checkpoint_path, device, fp16=False):
    from unimol2.models.unimol2 import UniMol2Model

    args = build_args()
    model = UniMol2Model(args, dictionary=None)
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state = state.get('model', state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        print('[unimol2] unexpected checkpoint keys:', unexpected[:10])
    if missing:
        print('[unimol2] missing checkpoint keys:', missing[:10])
    model.eval()
    if fp16:
        model.half()
    model.to(device)
    return model


def flush(txn, env):
    if txn is not None:
        txn.commit()
    return env.begin(write=True)


def extract(args):
    repo_root = os.path.abspath(args.repo_root)
    add_repo_paths(repo_root, args.unicore_root)

    device = torch.device('cuda:0' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device, fp16=args.fp16 and device.type == 'cuda')

    feature_env = open_lmdb(args.feature_lmdb, readonly=True)
    out_env = open_lmdb(args.output_lmdb, readonly=False, map_size_gb=args.map_size_gb)

    pending = []
    written = 0
    skipped_existing = 0
    errors = []
    total = 0
    txn = out_env.begin(write=True)

    def write_batch(batch_records):
        nonlocal txn, written
        items = [rec['item'] for rec in batch_records]
        lengths = [int(rec['length']) for rec in batch_records]
        batch = move_to_device(collate_items(items), device)
        with torch.no_grad():
            x, pair = model(batch)
        token_reprs = x[:, 1:, :].detach().float().cpu().numpy()
        for rec, length, arr in zip(batch_records, lengths, token_reprs):
            tokens = np.ascontiguousarray(arr[:length].astype(np.float32))
            if tokens.shape != (length, 768):
                raise ValueError(f'bad output shape for {rec["lmdb_key"]}: {tokens.shape}')
            payload = {
                'ligand_tokens': tokens,
                'ligand_mask': np.ones((length,), dtype=bool),
                'atoms': rec['atoms'],
                'coordinates': rec['coordinates'],
                'encoder': 'unimol2_84m',
                'representation': 'atom_token',
                'hidden_size': 768,
            }
            txn.put(rec['lmdb_key'].encode('utf-8'), pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            written += 1
            if written % args.commit_every == 0:
                txn = flush(txn, out_env)

    with open(args.input_jsonl, 'r', encoding='utf-8') as handle:
        for line in handle:
            total += 1
            record = json.loads(line)
            key = record['lmdb_key']
            with out_env.begin(write=False) as read_txn:
                if args.resume and read_txn.get(key.encode('utf-8')) is not None:
                    skipped_existing += 1
                    continue
            try:
                feature_item = load_lmdb_item(feature_env, key)
                mol = feature_item['ligand_features']['mol']
                item = make_item(record, mol)
                pending.append({
                    'lmdb_key': key,
                    'atoms': record['atoms'],
                    'coordinates': np.asarray(record['coordinates'], dtype=np.float32),
                    'length': len(record['atoms']),
                    'item': item,
                })
                if len(pending) >= args.batch_size:
                    write_batch(pending)
                    pending.clear()
            except Exception as exc:
                errors.append({'key': key, 'error': str(exc)})
            if total % args.progress_every == 0:
                print(json.dumps({
                    'total': total,
                    'written': written,
                    'skipped_existing': skipped_existing,
                    'errors': len(errors),
                }, ensure_ascii=False))

    if pending:
        write_batch(pending)
        pending.clear()
    txn = flush(txn, out_env)
    txn.abort()

    meta = {
        'input_jsonl': os.path.abspath(args.input_jsonl),
        'feature_lmdb': os.path.abspath(args.feature_lmdb),
        'checkpoint': os.path.abspath(args.checkpoint),
        'output_lmdb': os.path.abspath(args.output_lmdb),
        'total': total,
        'written': written,
        'skipped_existing': skipped_existing,
        'errors': len(errors),
        'hidden_size': 768,
        'encoder': 'unimol2_84m',
        'errors_preview': errors[:20],
    }
    with out_env.begin(write=True) as meta_txn:
        meta_txn.put(b'__metadata__', pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
    if args.metadata_output:
        with open(args.metadata_output, 'w', encoding='utf-8') as out:
            json.dump(meta, out, ensure_ascii=False, indent=2)
            out.write('\n')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Extract Uni-Mol2 atom-token features aligned to GerNA ligand atom order.')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--unicore-root', default='/home/user/wangzhiwen/DrugCLIP/DrugCLIP/Uni-Core')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--feature-lmdb', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-lmdb', required=True)
    parser.add_argument('--metadata-output', default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--map-size-gb', type=int, default=64)
    parser.add_argument('--commit-every', type=int, default=100)
    parser.add_argument('--progress-every', type=int, default=500)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
