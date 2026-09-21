#!/usr/bin/env python3
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.local_interface import (  # noqa: E402
    canonical_chain_signature,
    resolve_source_lmdb,
)
from utils.experiment_integrity import (  # noqa: E402
    atomic_write_json,
    hash_ordered_keys,
    manifest_fingerprint,
)


DIAGNOSTIC_SCHEMA_VERSION = 1


def _load_manifest(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    keys = [str(sample["lmdb_key"]) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate lmdb keys")
    return path, manifest, samples, keys


def _open_readonly_lmdb(path):
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )


def _source_key_summary(environment):
    with environment.begin(write=False) as transaction:
        keys = sorted(
            key.decode("utf-8")
            for key in transaction.cursor().iternext(
                keys=True, values=False
            )
            if key != b"__metadata__"
        )
    return {
        "source_key_count": len(keys),
        "source_key_sha256": hash_ordered_keys(keys),
    }


def _load_payload(transaction, key):
    raw = transaction.get(key.encode("utf-8"))
    if raw is None:
        raise KeyError(f"source LMDB key not found: {key}")
    return pickle.loads(raw)


def _target_hash(payload):
    signature = canonical_chain_signature(payload["protein_features"])
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _choose_groups(groups, fraction, seed):
    signatures = sorted(groups)
    target_count = max(
        1,
        int(np.floor(sum(len(groups[key]) for key in signatures) * fraction + 0.5)),
    )
    generator = np.random.default_rng(int(seed))
    shuffled = [
        signatures[int(index)]
        for index in generator.permutation(len(signatures))
    ]
    selected = []
    selected_count = 0
    for signature in shuffled:
        selected.append(signature)
        selected_count += len(groups[signature])
        if selected_count >= target_count:
            break
    return selected, target_count


def build_diagnostic_split(args):
    fraction = float(args.fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between zero and one")
    manifest_path, manifest, samples, ordered_keys = _load_manifest(
        args.manifest
    )
    if manifest.get("split") != "train":
        raise ValueError("diagnostic split requires manifest split=train")
    source_path = Path(
        resolve_source_lmdb(manifest, manifest_path.parent)
    )
    environment = _open_readonly_lmdb(source_path)
    try:
        source_summary = _source_key_summary(environment)
        groups = defaultdict(list)
        key_to_target = {}
        with environment.begin(write=False) as transaction:
            for sample, key in zip(samples, ordered_keys):
                payload = _load_payload(transaction, key)
                target = _target_hash(payload)
                groups[target].append(key)
                key_to_target[key] = target
    finally:
        environment.close()

    selected_targets, requested_count = _choose_groups(
        groups, fraction, args.seed
    )
    selected_target_set = set(selected_targets)
    selected_keys = [
        key
        for key in ordered_keys
        if key_to_target[key] in selected_target_set
    ]
    result = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "manifest": manifest_fingerprint(manifest_path),
        "source_lmdb": str(source_path),
        **source_summary,
        "seed": int(args.seed),
        "fraction": fraction,
        "requested_sample_count": requested_count,
        "selected_sample_count": len(selected_keys),
        "total_target_count": len(groups),
        "selected_target_count": len(selected_targets),
        "selected_target_hashes": sorted(selected_targets),
        "ordered_key_sha256": hash_ordered_keys(selected_keys),
        "keys": selected_keys,
    }
    atomic_write_json(args.output, result)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build a train-only exact-target diagnostic holdout."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    result = build_diagnostic_split(build_arg_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
