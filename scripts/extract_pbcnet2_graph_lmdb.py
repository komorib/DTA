#!/usr/bin/env python
import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import lmdb

from extract_pbcnet2_global_lmdb import (
    collect_samples,
    import_pbcnet_modules,
    load_feature_item,
    open_read_lmdb,
    open_write_lmdb,
    output_contains,
    write_ligand_sdf,
    write_metadata,
    write_pocket_pdb,
)


def extract(args):
    graph_information = import_pbcnet_modules(args.pbcnet_root)
    samples, feature_lmdb_path, split_counts = collect_samples(args.manifest)
    feature_env = open_read_lmdb(feature_lmdb_path)
    output_env = open_write_lmdb(args.output_lmdb, args.map_size_gb)

    tmp_root = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix='pbcnet_graph_extract_'))
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
                    graph = graph_information(str(ligand_sdf), str(pocket_pdb))
                    payload = {
                        'payload_type': 'pbcnet_graph',
                        'graph': graph,
                        'source': 'PBCNet2.0.Graph_Information',
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
                    }), flush=True)
        write_txn.commit()
        write_txn = None
        metadata = {
            'manifests': args.manifest,
            'pbcnet_root': str(Path(args.pbcnet_root).resolve()),
            'output_lmdb': str(Path(args.output_lmdb).resolve()),
            'split_counts': split_counts,
            'candidate_samples': len(samples),
            'written': written,
            'skipped_existing': skipped_existing,
            'errors': len(errors),
            'errors_preview': errors,
            'payload_type': 'pbcnet_graph',
        }
        write_metadata(output_env, metadata)
    finally:
        feature_env.close()
        output_env.close()
        if cleanup_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
    if args.metadata_output:
        with open(args.metadata_output, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description='Extract static PBCNet2.0 DGL graphs into LMDB for trainable PBCNet fusion.')
    parser.add_argument('--manifest', action='append', required=True, help='PDBbind LMDB manifest. Pass multiple times for train/valid/test.')
    parser.add_argument('--pbcnet-root', required=True)
    parser.add_argument('--output-lmdb', required=True)
    parser.add_argument('--metadata-output', default=None)
    parser.add_argument('--map-size-gb', type=float, default=32.0)
    parser.add_argument('--commit-every', type=int, default=100)
    parser.add_argument('--progress-every', type=int, default=500)
    parser.add_argument('--error-preview', type=int, default=30)
    parser.add_argument('--tmp-dir', default=None)
    parser.add_argument('--clean-tmp-each', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
