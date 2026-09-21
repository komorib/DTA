import hashlib
import json
import os
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_ordered_keys(keys):
    digest = hashlib.sha256()
    for key in keys:
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    descriptor, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def manifest_fingerprint(path):
    path = os.path.abspath(os.fspath(path))
    with open(path, "rb") as handle:
        manifest_bytes = handle.read()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    samples = manifest["samples"]
    ordered_keys = [sample["lmdb_key"] for sample in samples]
    return {
        "path": path,
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split": manifest["split"],
        "sample_count": len(samples),
        "ordered_key_sha256": hash_ordered_keys(ordered_keys),
    }
