#!/usr/bin/env python3
"""Merge several LMDBs (e.g. PBCNet per-atom extraction shards) into one output LMDB.

All keys are copied. The reserved b'__metadata__' key, if present, is taken from the
first input that has it and an aggregated 'written' count is recomputed from real keys.
"""
import argparse
import json
import pickle
from pathlib import Path

import lmdb

META_KEY = b'__metadata__'


def main():
    ap = argparse.ArgumentParser(description='Merge LMDB shards into one LMDB.')
    ap.add_argument('--input', action='append', required=True, help='Input LMDB dir (repeat).')
    ap.add_argument('--output', required=True, help='Output LMDB dir.')
    ap.add_argument('--map-size-gb', type=int, default=16)
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out = lmdb.open(args.output, subdir=True, readonly=False, create=True, lock=True,
                    map_size=int(args.map_size_gb) * (1024 ** 3), max_readers=256)
    meta = None
    total = 0
    wtxn = out.begin(write=True)
    n_since_commit = 0
    for src in args.input:
        env = lmdb.open(src, readonly=True, lock=False, readahead=False, subdir=True, max_readers=256)
        with env.begin(write=False) as rtxn:
            cur = rtxn.cursor()
            for k, v in cur:
                if k == META_KEY:
                    if meta is None:
                        try:
                            meta = json.loads(v.decode('utf-8'))
                        except Exception:
                            meta = None
                    continue
                wtxn.put(k, v)
                total += 1
                n_since_commit += 1
                if n_since_commit >= 2000:
                    wtxn.commit(); wtxn = out.begin(write=True); n_since_commit = 0
        env.close()
        print(f'merged {src}: running total {total}')
    if meta is None:
        meta = {}
    meta['merged_inputs'] = [str(s) for s in args.input]
    meta['written'] = total
    wtxn.put(META_KEY, json.dumps(meta, ensure_ascii=True).encode('utf-8'))
    wtxn.commit()
    out.close()
    print(json.dumps({'output': args.output, 'total_keys': total, 'inputs': len(args.input)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
