"""Mode-sealed chain features and exact chain-to-pocket projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
import torch

from dta.msa_pair_expert_v11.dataset import MSAPairCollator, MSAPairComplexDataset
from dta.msa_transformer_smoke_v1.msa import ENCODER_NAME, sha256_text

from . import WINDOW_POLICY


FEATURE_WIDTH = 768
SOURCE_MANIFEST_VERSION = "fullchain_msa_sources_v1"
_FEATURE_MODES = ("query_only", "aligned_msa")


def _selector_identity(source):
    policy, digest = source.get("selector_policy"), source.get("selector_config_sha256")
    row_policy = source.get("row_validation_policy")
    if row_policy is not None and (row_policy != "first_valid_identifier_occurrence_v2"
                                   or policy is None or digest is None):
        raise ValueError("invalid row validation cache identity")
    if policy is None and digest is None:
        return {}
    if (policy not in ("query_then_first_valid_unique_homologs_v1", "query_then_diverse_maxmin_v1")
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("invalid selector cache identity")
    result = {"selector_policy": policy, "selector_config_sha256": digest}
    if row_policy is not None:
        result["row_validation_policy"] = row_policy
    return result


def _validate_source_entry(source):
    if not isinstance(source, dict):
        raise ValueError("full-chain source entry must be an object")
    for field in ("query_sha256", "a3m_sha256"):
        value = source.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("full-chain source entry has invalid %s" % field)
    selections = source.get("selected_source_identifiers")
    if not isinstance(selections, dict) or set(selections) != set(_FEATURE_MODES):
        raise ValueError("full-chain source entry must specify both feature modes")
    maximum_depth = source.get("maximum_depth", 8)
    if type(maximum_depth) is not int or maximum_depth not in (8, 32):
        raise ValueError("unsupported sealed source maximum_depth")
    if maximum_depth == 32 and not _selector_identity(source):
        raise ValueError("depth-32 source requires an explicit sealed selection configuration")
    for mode in _FEATURE_MODES:
        identifiers = selections[mode]
        if (
            not isinstance(identifiers, list)
            or not 1 <= len(identifiers) <= maximum_depth
            or any(not isinstance(value, str) or not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("full-chain source entry has invalid %s selection" % mode)
    if (
        len(selections["query_only"]) != 1
        or selections["aligned_msa"][0] != selections["query_only"][0]
    ):
        raise ValueError("full-chain source modes must share the query row")
    _selector_identity(source)
    return source


def load_source_manifest(path):
    """Load expected identities from this run's A3M inventory, not the feature cache."""

    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != SOURCE_MANIFEST_VERSION:
        raise ValueError("full-chain source manifest version mismatch")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, dict) or not sequences:
        raise ValueError("full-chain source manifest has no sequences")
    for sequence_id, source in sequences.items():
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("full-chain source manifest has an invalid sequence ID")
        _validate_source_entry(source)
    return sequences


def _identity_fields(sequence_id, query_sha256, a3m_sha256, mode, weight_sha256, source_ids,
                     selector_identity=None):
    if mode not in ("query_only", "aligned_msa"):
        raise ValueError("unknown MSA feature mode")
    return {
        "encoder": ENCODER_NAME,
        "weights_sha256": str(weight_sha256),
        "feature_mode": mode,
        "sequence_id": str(sequence_id),
        "query_sha256": str(query_sha256),
        "a3m_sha256": str(a3m_sha256),
        "selected_source_identifiers": list(source_ids),
        "window_policy": WINDOW_POLICY,
        **(selector_identity or {}),
    }


def _identity_hash(fields):
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_feature_record(sequence_id, alignment, embedding, weight_sha256):
    """Package a full-chain output without discarding non-pocket residues."""

    values = np.ascontiguousarray(embedding, dtype=np.float32)
    if values.shape != (len(alignment.query_sequence), FEATURE_WIDTH):
        raise ValueError("full-chain feature shape does not match query")
    if not np.isfinite(values).all():
        raise ValueError("full-chain features contain non-finite values")
    fields = _identity_fields(
        sequence_id,
        sha256_text(alignment.query_sequence),
        alignment.metadata["a3m_sha256"],
        alignment.mode,
        weight_sha256,
        alignment.metadata["selected_source_identifiers"],
        _selector_identity(alignment.metadata),
    )
    return {
        **fields,
        "cache_identity_sha256": _identity_hash(fields),
        "embedding_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "original_length": len(alignment.query_sequence),
        "embedding": values,
    }


def validate_feature_record(record, sequence_id, query_sequence, mode, weight_sha256, expected_source):
    """Require both internal integrity and the independently specified source."""

    _validate_source_entry(expected_source)
    expected_query_sha = sha256_text(query_sequence)
    if expected_source["query_sha256"] != expected_query_sha:
        raise ValueError("full-chain source query differs from the row cache")
    fields = _identity_fields(
        record.get("sequence_id"),
        record.get("query_sha256"),
        record.get("a3m_sha256"),
        record.get("feature_mode"),
        record.get("weights_sha256"),
        record.get("selected_source_identifiers", ()),
        _selector_identity(record),
    )
    if (
        fields["sequence_id"] != str(sequence_id)
        or fields["query_sha256"] != expected_query_sha
        or fields["feature_mode"] != mode
        or fields["weights_sha256"] != str(weight_sha256)
        or record.get("encoder") != ENCODER_NAME
        or record.get("window_policy") != WINDOW_POLICY
        or record.get("cache_identity_sha256") != _identity_hash(fields)
    ):
        raise ValueError("full-chain feature cache identity mismatch")
    if (
        fields["a3m_sha256"] != expected_source["a3m_sha256"]
        or fields["selected_source_identifiers"]
        != expected_source["selected_source_identifiers"][mode]
    ):
        raise ValueError("full-chain feature cache source mismatch")
    if _selector_identity(record) != _selector_identity(expected_source):
        raise ValueError("full-chain feature cache selector identity mismatch")
    values = np.asarray(record.get("embedding"))
    if values.shape != (len(query_sequence), FEATURE_WIDTH) or not np.isfinite(values).all():
        raise ValueError("full-chain feature cache has invalid shape or values")
    values = np.ascontiguousarray(values, dtype=np.float32)
    if record.get("embedding_sha256") != hashlib.sha256(values.tobytes()).hexdigest():
        raise ValueError("full-chain feature cache embedding hash mismatch")
    return values


def gather_pocket_features(item, records, queries, mode, weight_sha256, expected_sources):
    """Use the existing chain index and noncontiguous sequence positions."""

    profile_ids = tuple(str(value) for value in item["profile_ids"])
    chain = np.asarray(item["pocket_chain_index"], dtype=np.int64)
    position = np.asarray(item["pocket_sequence_position"], dtype=np.int64)
    if chain.ndim != 1 or position.shape != chain.shape or chain.size == 0:
        raise ValueError("pocket chain and position arrays are malformed")
    if np.any(chain < 0) or np.any(chain >= len(profile_ids)):
        raise ValueError("pocket chain index is outside profile_ids")
    gathered = np.empty((chain.size, FEATURE_WIDTH), dtype=np.float32)
    for chain_index in np.unique(chain):
        sid = profile_ids[int(chain_index)]
        if sid not in records or sid not in queries or sid not in expected_sources:
            raise KeyError("full-chain feature, query, or expected source is missing for %s" % sid)
        values = validate_feature_record(
            records[sid], sid, queries[sid], mode, weight_sha256, expected_sources[sid]
        )
        selected = np.flatnonzero(chain == chain_index)
        indices = position[selected]
        if np.any(indices < 0) or np.any(indices >= values.shape[0]):
            raise ValueError("pocket position is outside its complete chain")
        gathered[selected] = values[indices]
    return gathered


class FullChainMSAFeatureDataset(MSAPairComplexDataset):
    """Read a sealed feature cache without changing the existing v3 samples."""

    def __init__(self, manifest_path, complex_cache, row_cache, feature_cache, mode, weight_sha256, source_manifest):
        super().__init__(manifest_path, complex_cache, row_cache)
        self.feature_cache = Path(feature_cache).expanduser().resolve()
        self.feature_mode = str(mode)
        self.weight_sha256 = str(weight_sha256)
        metadata_path = self.feature_cache / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("encoder") != ENCODER_NAME
            or metadata.get("feature_mode") != self.feature_mode
            or metadata.get("weights_sha256") != self.weight_sha256
            or metadata.get("window_policy") != WINDOW_POLICY
        ):
            raise ValueError("full-chain cache metadata differs from requested arm")
        if not (self.feature_cache / "proteins.lmdb").is_dir():
            raise FileNotFoundError("full-chain feature LMDB is missing")
        self.expected_sources = load_source_manifest(source_manifest)
        self._feature_env = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_feature_env"] = None
        return state

    def _feature_environment(self):
        if self._feature_env is None:
            self._feature_env = lmdb.open(
                str(self.feature_cache / "proteins.lmdb"),
                readonly=True, lock=False, readahead=False, meminit=False,
            )
        return self._feature_env

    def __getitem__(self, index):
        item = dict(super().__getitem__(index))
        profile_ids = tuple(str(value) for value in item["profile_ids"])
        used = {profile_ids[int(i)] for i in np.unique(item["pocket_chain_index"])}
        queries = {}
        records = {}
        with self._row_environment().begin(write=False) as row_txn, self._feature_environment().begin(write=False) as feature_txn:
            for sid in used:
                row_raw = row_txn.get(sid.encode("ascii"))
                feature_raw = feature_txn.get(sid.encode("ascii"))
                if row_raw is None or feature_raw is None:
                    raise KeyError("row or full-chain feature cache is missing %s" % sid)
                queries[sid] = str(pickle.loads(bytes(row_raw))["query_sequence"])
                records[sid] = pickle.loads(bytes(feature_raw))
        item["fullchain_msa_pocket"] = gather_pocket_features(
            item, records, queries, self.feature_mode, self.weight_sha256,
            self.expected_sources,
        )
        return item


class FullChainMSACollator(MSAPairCollator):
    def __call__(self, items):
        batch = super().__call__(items)
        features = torch.zeros(
            (len(items), batch["pocket_mask"].shape[1], FEATURE_WIDTH),
            dtype=torch.float32,
        )
        for index, item in enumerate(items):
            value = np.asarray(item.get("fullchain_msa_pocket"))
            pocket_count = int(item["pocket_chain_index"].shape[0])
            if value.shape != (pocket_count, FEATURE_WIDTH) or not np.isfinite(value).all():
                raise ValueError("full-chain pocket features are missing or invalid")
            features[index, :pocket_count] = torch.as_tensor(value, dtype=torch.float32)
        batch["fullchain_msa_pocket"] = features
        return batch


__all__ = [
    "FEATURE_WIDTH",
    "SOURCE_MANIFEST_VERSION",
    "FullChainMSACollator",
    "FullChainMSAFeatureDataset",
    "gather_pocket_features",
    "load_source_manifest",
    "make_feature_record",
    "validate_feature_record",
]
