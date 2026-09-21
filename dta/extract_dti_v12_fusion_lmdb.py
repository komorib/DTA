#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.lmdb_dataset import build_residue_representations, ordered_chains, resolve_manifest_path  # noqa: E402

DRUGCLIP_DIR = Path('/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen')
if str(DRUGCLIP_DIR) not in sys.path:
    sys.path.insert(0, str(DRUGCLIP_DIR))

from lmdb_embedding_cache import EmbeddingCacheReader, read_cache_metadata  # noqa: E402
from models_seq_interaction_fourroute_cached_v1 import FourRouteCachedSequenceInteractionEncoder  # noqa: E402
from smiles_key_utils import canonicalize_smiles_key  # noqa: E402

META_KEY = b'__metadata__'


def load_checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        return checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
    return checkpoint


def resolve_hidden_dim(metadata, label):
    if 'hidden_size' in metadata:
        return int(metadata['hidden_size'])
    if 'hidden_dim' in metadata:
        return int(metadata['hidden_dim'])
    raise KeyError(f'{label} metadata is missing hidden_size/hidden_dim: {metadata}')


def load_prot_proj(state_dict):
    for prefix in ('backbone.prot_proj', 'prot_proj'):
        weight_key = f'{prefix}.weight'
        bias_key = f'{prefix}.bias'
        if weight_key in state_dict and bias_key in state_dict:
            weight = state_dict[weight_key].detach().cpu().float().numpy()
            bias = state_dict[bias_key].detach().cpu().float().numpy()
            if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
                raise ValueError(f'invalid prot_proj shape: weight={weight.shape}, bias={bias.shape}')
            return weight.astype(np.float32), bias.astype(np.float32), {'weight_key': weight_key, 'bias_key': bias_key}
    preview = list(state_dict.keys())[:20]
    raise KeyError(f'checkpoint does not contain prot_proj weights; first keys={preview}')


def project_protein_tokens(protein_repre, weight, bias):
    protein_repre = np.asarray(protein_repre, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if protein_repre.ndim != 2:
        raise ValueError(f'protein_repre must be [L,D], got {protein_repre.shape}')
    if protein_repre.shape[1] != weight.shape[1]:
        raise ValueError(
            'protein embedding dim does not match checkpoint prot_proj input dim: '
            f'protein={protein_repre.shape[1]}, checkpoint={weight.shape[1]}'
        )
    if bias.shape[0] != weight.shape[0]:
        raise ValueError(f'bias dim does not match weight out dim: {bias.shape[0]} vs {weight.shape[0]}')
    tokens = protein_repre @ weight.T
    tokens = tokens + bias.reshape(1, -1)
    return {
        'protein_tokens': np.asarray(tokens, dtype=np.float32),
        'protein_mask': np.ones((protein_repre.shape[0],), dtype=np.bool_),
    }


def attach_global_feature(payload, global_feature):
    global_feature = np.asarray(global_feature, dtype=np.float32).reshape(-1)
    if global_feature.ndim != 1 or global_feature.shape[0] == 0:
        raise ValueError('global feature must flatten to a non-empty vector')
    out = dict(payload)
    out['global'] = global_feature
    return out


def downsample_token_numpy(embedding, max_len):
    embedding = np.asarray(embedding, dtype=np.float32)
    length = int(embedding.shape[0])
    max_len = int(max_len)
    if max_len <= 0 or length <= max_len:
        return embedding, np.ones((length,), dtype=np.int64)
    boundaries = np.linspace(0, length, max_len + 1, dtype=np.int64)
    reduced = np.zeros((max_len, embedding.shape[1]), dtype=np.float32)
    for idx in range(max_len):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
        if end <= start:
            end = min(length, start + 1)
        reduced[idx] = embedding[start:end].mean(axis=0)
    return reduced, np.ones((max_len,), dtype=np.int64)


def load_manifest(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('dataset_type') != 'pdbbind_lmdb':
        raise ValueError(f'not a PDBbind LMDB manifest: {manifest_path}')
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    lmdb_path = resolve_manifest_path(manifest_dir, manifest['lmdb_path'])
    return manifest, lmdb_path


def collect_samples(manifest_paths):
    samples = []
    seen = set()
    lmdb_paths = []
    split_counts = {}
    for manifest_path in manifest_paths:
        manifest, lmdb_path = load_manifest(manifest_path)
        lmdb_paths.append(lmdb_path)
        split_name = manifest.get('split') or Path(manifest_path).stem.replace('_manifest', '')
        split_counts[split_name] = len(manifest.get('samples', []))
        for sample in manifest.get('samples', []):
            key = sample['lmdb_key']
            if key in seen:
                continue
            seen.add(key)
            samples.append(sample)
    unique_lmdb_paths = sorted(set(os.path.abspath(path) for path in lmdb_paths))
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


def write_metadata(env, metadata):
    with env.begin(write=True) as txn:
        txn.put(META_KEY, json.dumps(metadata, indent=2, ensure_ascii=True).encode('utf-8'))


def try_load_ligand_embedding(reader, smiles_text):
    queries = []
    canonical = canonicalize_smiles_key(smiles_text)
    if canonical:
        queries.append(canonical)
    raw = str(smiles_text).strip()
    if raw and raw not in queries:
        queries.append(raw)
    last_error = None
    for query in queries:
        try:
            embedding, mask, meta = reader.get_numpy(query)
            return query, embedding, mask, meta
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise KeyError(f'empty ligand cache query for smiles={smiles_text!r}')


def build_token_batch(protein_repre, ligand_token_embedding, cfg):
    prot_embedding, prot_mask = downsample_token_numpy(protein_repre, cfg['prot_token_max_len'])
    lig_embedding, lig_mask = downsample_token_numpy(ligand_token_embedding, cfg['lig_token_max_len'])
    return {
        'prot_embeddings': torch.from_numpy(prot_embedding).unsqueeze(0),
        'prot_embedding_mask': torch.from_numpy(prot_mask).unsqueeze(0),
        'mol_embeddings': torch.from_numpy(lig_embedding).unsqueeze(0),
        'mol_embedding_mask': torch.from_numpy(lig_mask).unsqueeze(0),
    }


def build_dti_global_feature(model, protein_repre, ligand_token_embedding, cfg):
    batch = build_token_batch(protein_repre, ligand_token_embedding, cfg)
    with torch.no_grad():
        prot_mask_t = batch['prot_embedding_mask'].bool()
        lig_mask_t = batch['mol_embedding_mask'].bool()
        prot_tokens_t = model.prot_proj(batch['prot_embeddings'].float())
        lig_tokens_t = model.mol_proj(batch['mol_embeddings'].float())
        prot_tokens_t, prot_mask_t = model.prot_token_compressor(prot_tokens_t, prot_mask_t, model.token_interaction_prot_len)
        lig_tokens_t, lig_mask_t = model.lig_token_compressor(lig_tokens_t, lig_mask_t, model.token_interaction_lig_len)
        for block in model.t_blocks:
            prot_tokens_t, lig_tokens_t = block(prot_tokens_t, lig_tokens_t, prot_mask_t, lig_mask_t)
        prot_vec, lig_vec = model._encode_branch_views(
            prot_tokens_t,
            lig_tokens_t,
            prot_mask_t,
            lig_mask_t,
            model.prot_attn_t,
            model.lig_attn_t,
            model.prot_gate_t,
            model.lig_gate_t,
        )
        global_feature = torch.cat([prot_vec, lig_vec], dim=-1).squeeze(0)
    return global_feature.detach().cpu().float().numpy().astype(np.float32)


def load_drugclip_backbone(state_dict, config, protein_dim, ligand_dim):
    model = FourRouteCachedSequenceInteractionEncoder(
        mol_dim=int(ligand_dim),
        prot_dim=int(protein_dim),
        hidden_dim=int(config['hidden_dim']),
        heads=int(config['cross_heads']),
        layers=int(config['cross_layers']),
        dropout=float(config['dropout']),
        cnn_vocab_size_protein=int(config['cnn_vocab_size_protein']),
        cnn_vocab_size_ligand=int(config['cnn_vocab_size_ligand']),
        cnn_embed_dim=int(config['cnn_embed_dim']),
        cnn_channels=int(config['cnn_channels']),
        cnn_kernel_size=int(config['cnn_kernel_size']),
        cnn_layers=int(config['cnn_layers']),
        token_interaction_prot_len=int(config['token_interaction_prot_len']),
        token_interaction_lig_len=int(config['token_interaction_lig_len']),
        cnn_interaction_prot_len=int(config['cnn_interaction_prot_len']),
        cnn_interaction_lig_len=int(config['cnn_interaction_lig_len']),
        use_esm_token=bool(config.get('use_esm_token', True)),
        use_ouroboros_token=bool(config.get('use_ouroboros_token', True)),
        use_protein_cnn=bool(config.get('use_protein_cnn', False)),
        use_smiles_cnn=bool(config.get('use_smiles_cnn', False)),
    )
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            inner_key = key[len('backbone.'):]
            if inner_key in model.state_dict():
                cleaned[inner_key] = value
        elif key in model.state_dict():
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing = [key for key in missing if not key.startswith('head.')]
    if missing:
        raise ValueError(f'failed to load DrugCLIP backbone weights, missing keys: {missing[:20]}')
    unexpected = [key for key in unexpected if key != 'residual_scale']
    if unexpected:
        raise ValueError(f'unexpected checkpoint keys for DrugCLIP backbone: {unexpected[:20]}')
    model.eval()
    return model


def extract(args):
    manifest_paths = [str(Path(args.manifest_dir) / f'{split}_manifest.json') for split in args.splits]
    for manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'manifest not found: {manifest_path}')

    state_dict = load_checkpoint_state_dict(args.checkpoint)
    weight, bias, checkpoint_meta = load_prot_proj(state_dict)
    drugclip_cfg = json.load(open(args.drugclip_config, 'r', encoding='utf-8'))
    ligand_meta = read_cache_metadata(args.ligand_lmdb)
    ligand_dim = resolve_hidden_dim(ligand_meta, 'ligand_lmdb')
    backbone = load_drugclip_backbone(state_dict, drugclip_cfg, protein_dim=int(weight.shape[1]), ligand_dim=ligand_dim)

    samples, feature_lmdb_path, split_counts = collect_samples(manifest_paths)
    if args.max_samples > 0:
        samples = samples[: int(args.max_samples)]

    ligand_cache = EmbeddingCacheReader(args.ligand_lmdb)
    feature_env = open_read_lmdb(feature_lmdb_path)
    output_env = open_write_lmdb(args.output, args.map_size_gb)

    stats = {
        'manifest_dir': os.path.abspath(args.manifest_dir),
        'feature_lmdb': os.path.abspath(feature_lmdb_path),
        'ligand_lmdb': os.path.abspath(args.ligand_lmdb),
        'checkpoint': os.path.abspath(args.checkpoint),
        'output': os.path.abspath(args.output),
        'split_counts': split_counts,
        'candidate_samples': len(samples),
        'written': 0,
        'skipped_existing': 0,
        'feature_missing': 0,
        'ligand_cache_miss': 0,
        'shape_errors': 0,
        'prot_proj': {
            'input_dim': int(weight.shape[1]),
            'output_dim': int(weight.shape[0]),
            **checkpoint_meta,
        },
        'dti_global': {
            'dim': 512,
            'source': 'DrugCLIP_v12_token_cross_concat(protein_vec,ligand_vec)',
        },
    }
    errors = []

    try:
        with feature_env.begin(write=False) as feature_txn:
            pending = []
            for index, sample in enumerate(samples, start=1):
                key = sample['lmdb_key']
                if args.resume:
                    with output_env.begin(write=False) as output_read_txn:
                        if output_contains(output_read_txn, key):
                            stats['skipped_existing'] += 1
                            continue
                try:
                    feature_item = load_feature_item(feature_txn, key)
                    chains = ordered_chains(feature_item['protein_features'])
                    protein_repre = build_residue_representations(
                        chains,
                        embedding_mode='esm',
                        expected_dim=int(weight.shape[1]),
                    )
                    expected_len = sample.get('n_residue')
                    if expected_len is not None and int(expected_len) != int(protein_repre.shape[0]):
                        raise ValueError(f'n_residue mismatch: sample={expected_len}, esm={protein_repre.shape[0]}')
                    ligand_key, ligand_embedding, ligand_mask, _lig_meta = try_load_ligand_embedding(ligand_cache, sample['ligand_group'])
                    valid_ligand_len = int(np.asarray(ligand_mask).sum())
                    ligand_embedding = np.asarray(ligand_embedding[:valid_ligand_len], dtype=np.float32)
                    payload = project_protein_tokens(protein_repre, weight, bias)
                    payload = attach_global_feature(
                        payload,
                        build_dti_global_feature(backbone, protein_repre, ligand_embedding, drugclip_cfg),
                    )
                    payload['source'] = {
                        'lmdb_key': key,
                        'pdb_id': sample.get('pdb_id'),
                        'feature': 'DrugCLIP_v12_backbone.prot_proj(GerNA_chain_ESM)',
                        'global_feature': 'DrugCLIP_v12_token_cross_concat(protein_vec,ligand_vec)',
                        'ligand_cache_key': ligand_key,
                    }
                    pending.append((key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)))
                except KeyError as exc:
                    if 'cache miss' in str(exc):
                        stats['ligand_cache_miss'] += 1
                    else:
                        stats['feature_missing'] += 1
                    errors.append({'key': key, 'error': str(exc)})
                    if args.fail_on_error:
                        raise
                except Exception as exc:
                    stats['shape_errors'] += 1
                    errors.append({'key': key, 'error': str(exc)})
                    if args.fail_on_error:
                        raise

                if len(pending) >= int(args.commit_every):
                    with output_env.begin(write=True) as output_txn:
                        for out_key, blob in pending:
                            output_txn.put(out_key.encode('utf-8'), blob)
                    stats['written'] += len(pending)
                    pending = []

                if args.progress_every > 0 and index % int(args.progress_every) == 0:
                    print(
                        json.dumps(
                            {k: stats[k] for k in ('candidate_samples', 'written', 'skipped_existing', 'feature_missing', 'ligand_cache_miss', 'shape_errors')},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

            if pending:
                with output_env.begin(write=True) as output_txn:
                    for out_key, blob in pending:
                        output_txn.put(out_key.encode('utf-8'), blob)
                stats['written'] += len(pending)

        stats['errors_preview'] = errors[:20]
        write_metadata(output_env, stats)
        output_env.sync()
    finally:
        feature_env.close()
        output_env.close()
        ligand_cache.close()

    if args.metadata_output:
        Path(args.metadata_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata_output).write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Extract GerNA-compatible DrugCLIP v12 fusion LMDB for PDBbind DTA.')
    parser.add_argument('--manifest_dir', default='data/PDBbind_chain_esm_residue_DTA/random')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--checkpoint', default='/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen/runs/seq_interaction_tokenonly_unimol_v12_pooledfusion/20260517_224200/best_pearson.pt')
    parser.add_argument('--drugclip_config', default='/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen/runs/seq_interaction_tokenonly_unimol_v12_pooledfusion/20260517_224200/config.json')
    parser.add_argument('--ligand_lmdb', default='/home/user/wangzhiwen/DrugCLIP/dta_seq_nextgen/cache/ligand_unimol_token.lmdb')
    parser.add_argument('--output', default='data/PDBbind_dti_v12_fusion_lmdb')
    parser.add_argument('--metadata_output', default='data/PDBbind_dti_v12_fusion_lmdb_metadata.json')
    parser.add_argument('--map_size_gb', type=int, default=16)
    parser.add_argument('--commit_every', type=int, default=100)
    parser.add_argument('--progress_every', type=int, default=500)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fail_on_error', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extract(args)


if __name__ == '__main__':
    main()
