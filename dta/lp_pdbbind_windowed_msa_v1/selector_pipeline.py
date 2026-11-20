"""Explicit experiment seals for selector-only runs, without hashing the A3M DB.

The historical source seal is a comparison reference. Every used indexed
record is parsed afresh and checked against its historical per-record digest.
The database-wide historical digest is declared, never recomputed here.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import pickle
import subprocess
import tempfile
import time

import lmdb
import numpy as np

from dta.msa_transformer_smoke_v1.msa import ENCODER_NAME, sha256_file
from . import WINDOW_POLICY
from .cache import SOURCE_MANIFEST_VERSION, load_source_manifest
from .selector import (DEFAULT_CANDIDATE_POOL_SIZE, DIVERSE_SELECTION_POLICY,
                       LEGACY_SELECTION_POLICY, aligned_sequence_identity)


SELECTOR_SEAL_VERSION = "selector_only_record_seal_v1"


def _stat_identity(path):
    value = Path(path).stat()
    return {"size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "quantiles": None}
    return {"count": int(array.size), "mean": float(array.mean()),
            "median": float(np.median(array)),
            "quantiles": [float(x) for x in np.quantile(array, [0, .25, .5, .75, 1])]}


def _row_summary(alignment):
    from experiments.lp_pdbbind.fullchain_msa_quality_diagnostic_v1.analyze import chain_quality
    rows = [(row.identifier, row.aligned_sequence) for row in alignment.rows]
    identities = [aligned_sequence_identity(rows[0][1], row[1]) for row in rows]
    diversity = [1.0 - aligned_sequence_identity(left[1], right[1])
                 for left, right in combinations(rows[1:], 2)]
    quality = chain_quality(rows, np.arange(len(alignment.query_sequence)))
    return {"selected_depth": len(rows), "selected_query_identity": identities,
            "selected_pairwise_diversity_summary": {
                "mean": float(np.mean(diversity)) if diversity else 0.0,
                "median": float(np.median(diversity)) if diversity else 0.0,
                "min": float(min(diversity)) if diversity else 0.0,
                "max": float(max(diversity)) if diversity else 0.0},
            "selected_row_coverage": quality["pocket_coverage"],
            "coverage_scope": "all_query_residue_columns_using_existing_chain_quality"}


def _source_indices(raw, identifiers):
    # Header positions only; row validity and sequence parsing remain in the
    # unchanged parser/validator called by prepare_full_alignment.
    first, index = {}, 0
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith(b">"):
            identifier = line[1:].strip().split(None, 1)[0].decode("utf-8")
            first.setdefault(identifier, index)
            index += 1
    return [first[identifier] for identifier in identifiers]


def _write_jsonl_new(path, records):
    with Path(path).open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _code_provenance(code_commit):
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root)).decode().strip()
    if code_commit is None or str(code_commit) != head:
        raise ValueError("selector prepare requires --code-commit equal to checkout HEAD")
    files = ["dta/lp_pdbbind_windowed_msa_v1/" + name for name in
             ("windowed.py", "selector.py", "selector_pipeline.py", "full_pipeline.py", "cache.py")]
    frozen = ["dta/msa_transformer_smoke_v1/msa.py", "dta/msa_transformer_smoke_v1/extract.py",
              "dta/lp_pdbbind_fullchain_esm_msa_all_v2/model.py",
              "dta/lp_pdbbind_fullchain_esm_msa_all_v2/dataset.py",
              "dta/lp_pdbbind_fullchain_esm_msa_all_v2/run_training.py",
              "dta/lp_pdbbind_fullchain_esm_msa_all_v2/run_gpu_smoke.py",
              "dta/lp_pdbbind_fullchain_esm_msa_cross_v1/run_training.py",
              "dta/lp_pdbbind_fullchain_esm_msa_cross_v1/dataset.py",
              "dta/msa_interface_pair_pooling_v18/training.py",
              "experiments/lp_pdbbind/fullchain_msa_quality_diagnostic_v1/analyze.py"]
    if subprocess.check_output(["git", "diff", "66fdb1fb6a8de223badd0772f6e3a02b2a11cc80", "--"] + frozen,
                               cwd=str(root)).strip():
        raise ValueError("frozen parser/encoder/model/training/validation code differs from experiment base")
    files += frozen
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--"] + files,
                                    cwd=str(root)).decode().strip()
    if dirty:
        raise ValueError("commit selector pipeline implementation before preparing real sources")
    return {"git_commit": head, "files_sha256": {name: sha256_file(root / name) for name in files}}


def _validation_quality(paths, manifest, alignments, selections):
    from experiments.lp_pdbbind.fullchain_msa_quality_diagnostic_v1.analyze import (
        _weighted_quality, chain_quality,
    )
    records = []
    complex_env = lmdb.open(str(paths["complex_root"] / "valid" / "complexes.lmdb"),
                            readonly=True, lock=False, readahead=False, meminit=False)
    profile_path = paths["profile_cache"]
    profile_env = lmdb.open(str(profile_path), readonly=True, lock=False,
                            readahead=False, meminit=False)
    neff_cache = {}
    try:
        with complex_env.begin() as complexes, profile_env.begin() as profiles:
            for sample in manifest["samples"]:
                key = str(sample["lmdb_key"])
                raw = complexes.get(key.encode())
                if raw is None:
                    raise KeyError("validation complex missing during quality manifest creation")
                item = pickle.loads(bytes(raw))
                sids = tuple(str(sid) for sid in item["profile_ids"])
                chains = np.asarray(item["pocket_chain_index"], dtype=np.int64)
                positions = np.asarray(item["pocket_sequence_position"], dtype=np.int64)
                parts, used = [], []
                for index in np.unique(chains):
                    sid = sids[int(index)]
                    alignment = alignments[sid]
                    if sid not in neff_cache:
                        profile_raw = profiles.get(sid.encode())
                        if profile_raw is None:
                            raise KeyError("validation profile missing: " + sid)
                        profile = pickle.loads(bytes(profile_raw))
                        if (profile["sequence_id"] != sid
                                or profile["query_sequence"] != alignment.query_sequence
                                or profile["a3m_sha256"] != alignment.metadata["a3m_sha256"]):
                            raise ValueError("full-profile identity differs from selected source")
                        value = float(profile["henikoff_neff"])
                        if not np.isfinite(value) or value < 1:
                            raise ValueError("invalid full-profile Neff")
                        neff_cache[sid] = value
                    part = chain_quality([(row.identifier, row.aligned_sequence) for row in alignment.rows],
                                         positions[chains == index])
                    part["full_profile_neff"] = neff_cache[sid]
                    parts.append(part)
                    used.append(sid)
                records.append({"key": key, "protein_chain_ids": sorted(set(used)),
                                "quality": _weighted_quality(parts),
                                "quality_scope": "same_complex_pocket_positions_as_original_diagnostic",
                                "fallback_used": any(selections[sid]["fallback_used"] for sid in used),
                                "selected_set_changed": any(selections[sid]["selected_set_changed"] for sid in used)})
    finally:
        complex_env.close()
        profile_env.close()
    return records


def prepare_selector_sources(workspace, output, *, baseline_cache_root, selection_policy,
                             candidate_pool_size, code_commit):
    from .full_pipeline import (_input_paths, _load_inventory, _prepare_from_raw, _read_a3m,
                                _read_json, _split_inventory, _write_json_new)
    if selection_policy not in (LEGACY_SELECTION_POLICY, DIVERSE_SELECTION_POLICY):
        raise ValueError("unknown selector experiment policy")
    if int(candidate_pool_size) != DEFAULT_CANDIDATE_POOL_SIZE:
        raise ValueError("controlled experiment requires candidate_pool_size=256")
    provenance = _code_provenance(code_commit)
    paths = _input_paths(workspace)
    paths["profile_cache"] = Path(workspace).resolve() / "data/LP_PDBBind_MSA_v9/uniref30_profiles/profiles.lmdb"
    output, baseline_root = Path(output).resolve(), Path(baseline_cache_root).resolve()
    if output.exists() or baseline_root == output or baseline_root in output.parents:
        raise FileExistsError("selector output must be new and independent of baseline cache")
    baseline = _read_json(baseline_root / "preflight.json")
    if baseline.get("status") != "source_preflight_passed" or baseline.get("test_manifest_read") is not False:
        raise ValueError("historical source preflight is incomplete")
    for path, field in ((baseline_root / "source_manifest.json", "source_manifest_sha256"),
                        (baseline_root / "targets.json", "targets_sha256"),
                        (paths["sequence_manifest"], "sequence_manifest_sha256"),
                        (paths["a3m_index"], "a3m_index_sha256"),
                        (paths["manifest_root"] / "train_manifest.json", "train_manifest_sha256"),
                        (paths["manifest_root"] / "valid_manifest.json", "valid_manifest_sha256")):
        if sha256_file(path) != baseline[field]:
            raise ValueError("historical source/config manifest differs: " + field)
    if paths["a3m_db"].stat().st_size != baseline["a3m_db_size"]:
        raise ValueError("historical A3M database size differs")
    historical_sources = load_source_manifest(baseline_root / "source_manifest.json")
    historical_targets = _read_json(baseline_root / "targets.json")
    historical_modes = [_read_json(baseline_root / mode / "metadata.json")
                        for mode in ("query_only", "aligned_msa")]
    weight_hashes = {metadata["weights_sha256"] for metadata in historical_modes}
    if len(weight_hashes) != 1 or any(metadata.get("status") != "built" for metadata in historical_modes):
        raise ValueError("historical encoder metadata is incomplete or inconsistent")
    weight_hash = next(iter(weight_hashes))
    sequences, indexed = _load_inventory(paths)
    manifests, used = _split_inventory(paths, sequences)
    selected = sorted(used["train"] | used["valid"])
    if (len(selected) != baseline["unique_sequence_count"]
            or len(used["train"]) != baseline["train_sequence_count"]
            or len(used["valid"]) != baseline["valid_sequence_count"]
            or len(manifests["train"]["samples"]) != baseline["train_sample_count"]
            or len(manifests["valid"]["samples"]) != baseline["valid_sample_count"]):
        raise ValueError("selector experiment inventory differs from historical baseline")
    # v2 consumes every profile_id, while the inherited source inventory is
    # pocket-mapped. Prove full-chain coverage without expanding that inventory.
    selected_sids = {str(sequences[index]["sequence_id"]) for index in selected}
    all_chain_counts = {}
    for split in ("train", "valid"):
        env = lmdb.open(str(paths["complex_root"] / split / "complexes.lmdb"),
                        readonly=True, lock=False, readahead=False, meminit=False)
        try:
            all_sids = set()
            with env.begin() as transaction:
                for sample in manifests[split]["samples"]:
                    item = pickle.loads(bytes(transaction.get(str(sample["lmdb_key"]).encode())))
                    all_sids.update(str(sid) for sid in item["profile_ids"])
            if not all_sids <= selected_sids:
                raise ValueError("v2 full-chain profile_ids exceed unchanged prepared inventory")
            all_chain_counts[split] = len(all_sids)
        finally:
            env.close()
    config = {"version": SELECTOR_SEAL_VERSION, "selection_policy": selection_policy,
              "candidate_pool_size": int(candidate_pool_size), "maximum_depth": 8,
              "identity_definition": "matches_on_jointly_non_gap_columns_v1",
              "coverage_tiebreak_available": False, "window_policy": WINDOW_POLICY,
              "encoder": ENCODER_NAME, "encoder_weights_sha256": weight_hash,
              "code": provenance}
    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / "selector_config.json", config)
    config_hash = sha256_file(output / "selector_config.json")
    _write_json_new(output / "prepare_started.json", {"status": "preparing", "code": provenance,
                    "test_manifest_read": False, "a3m_database_rehashed": False})
    started = time.monotonic()
    initial_stat = _stat_identity(paths["a3m_db"])
    sources, targets, selections, valid_alignments, baseline_stats = {}, [], {}, {}, []
    try:
        with tempfile.TemporaryDirectory(prefix="selector-only-source-") as temporary:
            temporary_path = Path(temporary) / "record.a3m"
            with paths["a3m_db"].open("rb", buffering=8 << 20) as database:
                for count, index in enumerate(selected, 1):
                    sequence = sequences[index]
                    sid = str(sequence["sequence_id"])
                    offset, length = indexed[index]
                    raw = _read_a3m(database, offset, length, sid)
                    old = _prepare_from_raw(raw, sequence["sequence"], "aligned_msa", temporary_path)
                    archived = historical_sources[sid]
                    if (old.metadata["a3m_sha256"] != archived["a3m_sha256"]
                            or old.metadata["query_sha256"] != archived["query_sha256"]
                            or old.metadata["selected_source_identifiers"] != archived["selected_source_identifiers"]["aligned_msa"]):
                        raise ValueError("historical indexed source/legacy selection differs: " + sid)
                    alignment = old if selection_policy == LEGACY_SELECTION_POLICY else _prepare_from_raw(
                        raw, sequence["sequence"], "aligned_msa", temporary_path,
                        selection_policy=selection_policy, candidate_pool_size=candidate_pool_size)
                    if (alignment.query_residue_columns != old.query_residue_columns
                            or alignment.rows[0] != old.rows[0]
                            or len(alignment.rows) != len(old.rows)):
                        raise ValueError("selector changed query mapping or selected depth")
                    identifiers = alignment.metadata["selected_source_identifiers"]
                    old_ids = old.metadata["selected_source_identifiers"]
                    stats = _row_summary(alignment)
                    old_stats = _row_summary(old)
                    baseline_stats.append(old_stats)
                    fallback = len(alignment.rows) < 8
                    entry = {"protein_chain_id": sid, "selector_policy": selection_policy,
                             "candidate_pool_size": candidate_pool_size,
                             "candidate_count": min(candidate_pool_size, alignment.metadata["valid_homolog_count"]),
                             "valid_homolog_count": alignment.metadata["valid_homolog_count"],
                             "query_id": identifiers[0], "selected_ids": identifiers,
                             "selected_source_indices": (alignment.metadata["selected_source_indices"]
                                                          if "selected_source_indices" in alignment.metadata
                                                          else _source_indices(raw, identifiers)),
                             "fallback_used": alignment.metadata.get("fallback_used", fallback),
                             "fallback_count": alignment.metadata.get("fallback_count", 0),
                             "selection_shortfall_count": 8 - len(alignment.rows),
                             "fallback_reason": alignment.metadata.get("fallback_reason", "valid_homologs_exhausted" if fallback else None),
                             "coverage_tiebreak_available": False,
                             "identity_zero_denominator_count": alignment.metadata.get("identity_zero_denominator_count", 0),
                             "selected_set_changed": set(identifiers) != set(old_ids),
                             "old_selected_ids": old_ids,
                             "selected_homolog_id_overlap_fraction": (len(set(identifiers[1:]) & set(old_ids[1:])) / (len(old_ids) - 1)
                                                                       if len(old_ids) > 1 else 1.0),
                             "baseline_statistics": old_stats, **stats}
                    selections[sid] = entry
                    sources[sid] = {"query_sha256": alignment.metadata["query_sha256"],
                                    "a3m_sha256": alignment.metadata["a3m_sha256"],
                                    "selected_source_identifiers": {"query_only": identifiers[:1], "aligned_msa": identifiers},
                                    "selector_policy": selection_policy, "selector_config_sha256": config_hash}
                    targets.append({"sequence_id": sid, "sequence_index": index,
                                    "query_sequence": sequence["sequence"], "a3m_offset": offset, "a3m_size": length})
                    if index in used["valid"]:
                        valid_alignments[sid] = alignment
                    if count % 25 == 0 or count == len(selected):
                        print(json.dumps({"event": "selector_prepare_progress", "processed": count,
                                          "total": len(selected), "seconds": time.monotonic() - started}), flush=True)
        if targets != historical_targets or set(sources) != set(historical_sources):
            raise ValueError("regenerated targets differ from historical source inventory")
        if _stat_identity(paths["a3m_db"]) != initial_stat:
            raise ValueError("A3M database changed during selector prepare")
        _write_json_new(output / "source_manifest.json", {"version": SOURCE_MANIFEST_VERSION,
                        "selector_config_sha256": config_hash, "selection_policy": selection_policy,
                        "code_commit": provenance["git_commit"], "sequences": sources})
        _write_json_new(output / "targets.json", targets)
        source_hash = sha256_file(output / "source_manifest.json")
        for entry in selections.values():
            entry["source_a3m_manifest"] = str(output / "source_manifest.json")
            entry["source_manifest_sha256"] = source_hash
            entry["feature_cache_path"] = str(output / "aligned_msa")
        _write_jsonl_new(output / "selection_manifest.jsonl", selections.values())
        qualities = _validation_quality(paths, manifests["valid"], valid_alignments, selections)
        _write_jsonl_new(output / "validation_quality_manifest.jsonl", qualities)
        summary = {"status": "selector_inventory_prepared", "chain_count": len(sources),
                   "validation_chain_count": len(valid_alignments), "validation_sample_count": len(qualities),
                   "selection_policy": selection_policy, "candidate_pool_size": candidate_pool_size,
                   "changed_chain_count": sum(entry["selected_set_changed"] for entry in selections.values()),
                   "fallback_chain_count": sum(entry["fallback_used"] for entry in selections.values()),
                   "fallback_count": sum(entry["fallback_count"] for entry in selections.values()),
                   "shortfall_chain_count": sum(entry["selection_shortfall_count"] > 0 for entry in selections.values()),
                   "selected_depth_distribution": dict(Counter(str(entry["selected_depth"]) for entry in selections.values())),
                   "new": {}, "baseline": {}, "all_profile_ids_coverage": all_chain_counts,
                   "selected_homolog_overlap": _distribution([entry["selected_homolog_id_overlap_fraction"] for entry in selections.values()]),
                   "test_manifest_read": False, "full_database_hash": False,
                   "seconds": time.monotonic() - started}
        for key, rows in (("new", list(selections.values())), ("baseline", baseline_stats)):
            summary[key] = {"query_identity": _distribution([identity for row in rows for identity in row["selected_query_identity"][1:]]),
                            "chain_pairwise_diversity": _distribution([row["selected_pairwise_diversity_summary"]["mean"] for row in rows]),
                            "chain_row_coverage": _distribution([row["selected_row_coverage"] for row in rows])}
        _write_json_new(output / "selection_summary.json", summary)
        report = {"status": "source_preflight_passed", "seal_version": SELECTOR_SEAL_VERSION,
                  "selection_policy": selection_policy, "selector_config_sha256": config_hash,
                  "code_commit": provenance["git_commit"],
                  "train_sample_count": len(manifests["train"]["samples"]),
                  "valid_sample_count": len(manifests["valid"]["samples"]),
                  "train_sequence_count": len(used["train"]), "valid_sequence_count": len(used["valid"]),
                  "unique_sequence_count": len(selected), "source_manifest_sha256": source_hash,
                  "targets_sha256": sha256_file(output / "targets.json"),
                  "selection_manifest_sha256": sha256_file(output / "selection_manifest.jsonl"),
                  "validation_quality_manifest_sha256": sha256_file(output / "validation_quality_manifest.jsonl"),
                  "sequence_manifest_sha256": baseline["sequence_manifest_sha256"],
                  "a3m_index_sha256": baseline["a3m_index_sha256"],
                  "train_manifest_sha256": baseline["train_manifest_sha256"],
                  "valid_manifest_sha256": baseline["valid_manifest_sha256"],
                  "declared_reference_a3m_db_sha256": baseline.get("a3m_db_sha256"),
                  "a3m_db_sha256_recomputed": False, "a3m_db_size": initial_stat["size_bytes"],
                  "a3m_db_stat": initial_stat, "source_record_hashes_verified": len(sources),
                  "historical_baseline_cache_root": str(baseline_root),
                  "test_manifest_read": False, "training_run": False}
        _write_json_new(output / "preflight.json", report)
        return report
    except Exception as error:
        _write_json_new(output / "prepare_failed.json", {"status": "selector_prepare_failed",
                        "error_type": type(error).__name__, "error": str(error),
                        "processed_chain_count": len(sources), "test_manifest_read": False,
                        "full_database_hash": False})
        raise


def validate_selector_seal(paths, output, report):
    from .full_pipeline import _read_json
    if report.get("seal_version") != SELECTOR_SEAL_VERSION:
        raise ValueError("unknown selector experiment seal")
    if report.get("a3m_db_sha256_recomputed") is not False:
        raise ValueError("selector seal must declare no whole-database rehash")
    if _stat_identity(paths["a3m_db"]) != report["a3m_db_stat"]:
        raise ValueError("selector source database stat changed; re-audit required")
    for name, field in (("selector_config.json", "selector_config_sha256"),
                        ("selection_manifest.jsonl", "selection_manifest_sha256"),
                        ("validation_quality_manifest.jsonl", "validation_quality_manifest_sha256")):
        if sha256_file(output / name) != report[field]:
            raise ValueError("selector generated manifest changed: " + name)
    config = _read_json(output / "selector_config.json")
    if (config["selection_policy"] != report["selection_policy"]
            or config["candidate_pool_size"] != 256 or config["maximum_depth"] != 8
            or config["window_policy"] != WINDOW_POLICY):
        raise ValueError("selector sealed configuration differs")
    root = Path(__file__).resolve().parents[2]
    for name, expected in config["code"]["files_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError("selector implementation changed after prepare: " + name)
    return config
