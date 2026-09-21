import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import os
import pickle
import subprocess
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


CI_TOLERANCE = 1e-8
DEFAULT_BOOTSTRAP_DRAWS = 2000
DEFAULT_BOOTSTRAP_SEED = 42
MMSEQS_MIN_IDENTITY = 0.90
MMSEQS_COVERAGE = 0.80
MMSEQS_COVERAGE_MODE = 0
METRIC_NAMES = ("RMSE", "MAE", "Pearson", "Spearman", "CI")


def _as_string_keys(values):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("prediction keys must be one-dimensional")
    output = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        value = str(value)
        if not value:
            raise ValueError("prediction keys must be nonempty")
        output.append(value)
    return np.asarray(output, dtype=str)


def load_keyed_predictions(path):
    path = os.path.abspath(os.fspath(path))
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(
            f"could not load keyed prediction archive {path}: {error}"
        ) from error
    with archive:
        required = {"keys", "labels", "predictions"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                "prediction archive is missing: " + ", ".join(missing)
            )
        keys = _as_string_keys(archive["keys"])
        labels = np.asarray(archive["labels"], dtype=np.float64)
        predictions = np.asarray(
            archive["predictions"], dtype=np.float64
        )
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            "prediction labels and values must be one-dimensional"
        )
    if not (len(keys) == len(labels) == len(predictions)):
        raise ValueError(
            "prediction keys, labels, and values must have equal length"
        )
    if len(keys) == 0:
        raise ValueError("prediction archive must be nonempty")
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("prediction archive contains duplicate keys")
    if not np.all(np.isfinite(labels)) or not np.all(
        np.isfinite(predictions)
    ):
        raise ValueError(
            "prediction labels and values must contain only finite values"
        )
    return {
        "path": path,
        "keys": keys,
        "labels": labels,
        "predictions": predictions,
    }


class _FenwickTree:
    def __init__(self, size):
        self.values = np.zeros(int(size) + 1, dtype=np.int64)

    def add(self, index):
        index = int(index) + 1
        while index < len(self.values):
            self.values[index] += 1
            index += index & -index

    def prefix_count(self, end):
        total = 0
        index = int(end)
        while index > 0:
            total += int(self.values[index])
            index -= index & -index
        return total


def _fast_concordance_index(
    labels, predictions, tolerance=CI_TOLERANCE
):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(
        predictions, dtype=np.float64
    ).reshape(-1)
    tolerance = float(tolerance)
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("CI tolerance must be finite and non-negative")
    if not np.all(np.isfinite(labels)) or not np.all(
        np.isfinite(predictions)
    ):
        raise ValueError("CI inputs must contain only finite values")
    if len(labels) < 2:
        return 0.0

    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    sorted_predictions = predictions[order]
    prediction_coordinates = np.unique(sorted_predictions).tolist()
    tree = _FenwickTree(len(prediction_coordinates))
    added = 0
    comparable = 0
    concordant = 0.0
    for index in range(len(sorted_labels)):
        lower_label = sorted_labels[index] - tolerance
        while (
            added < index
            and sorted_labels[added] < lower_label
        ):
            rank = bisect_left(
                prediction_coordinates,
                float(sorted_predictions[added]),
            )
            tree.add(rank)
            added += 1
        eligible = added
        prediction = float(sorted_predictions[index])
        less_end = bisect_left(
            prediction_coordinates, prediction - tolerance
        )
        tie_end = bisect_right(
            prediction_coordinates, prediction + tolerance
        )
        lower_count = tree.prefix_count(less_end)
        tie_count = (
            tree.prefix_count(tie_end) - lower_count
        )
        comparable += eligible
        concordant += lower_count + 0.5 * tie_count
    if comparable == 0:
        return 0.0
    return float(concordant / comparable)


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _safe_pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _regression_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    difference = predictions - labels
    return {
        "RMSE": float(np.sqrt(np.mean(difference * difference))),
        "MAE": float(np.mean(np.abs(difference))),
        "Pearson": _safe_pearson(predictions, labels),
        "Spearman": _safe_pearson(
            _average_ranks(predictions), _average_ranks(labels)
        ),
        "CI": _fast_concordance_index(labels, predictions),
    }


def _seed_sort_key(seed):
    try:
        return (0, int(seed))
    except (TypeError, ValueError):
        return (1, str(seed))


def _aligned_pair(baseline, interface, context):
    if not np.array_equal(baseline["keys"], interface["keys"]):
        raise ValueError(f"prediction key alignment failed for {context}")
    if not np.array_equal(
        baseline["labels"], interface["labels"]
    ):
        raise ValueError(f"prediction label alignment failed for {context}")


def _load_matched_records(baseline_paths, interface_paths):
    if not isinstance(baseline_paths, dict) or not isinstance(
        interface_paths, dict
    ):
        raise ValueError("prediction paths must be seed-to-path mappings")
    if set(baseline_paths) != set(interface_paths):
        raise ValueError(
            "baseline and interface-first seeds must match exactly"
        )
    if not baseline_paths:
        raise ValueError("at least one matched seed is required")
    seeds = sorted(baseline_paths, key=_seed_sort_key)
    baseline_records = []
    interface_records = []
    reference = None
    for seed in seeds:
        baseline = load_keyed_predictions(baseline_paths[seed])
        interface = load_keyed_predictions(interface_paths[seed])
        _aligned_pair(baseline, interface, f"seed {seed}")
        if reference is not None:
            _aligned_pair(reference, baseline, f"baseline seed {seed}")
        else:
            reference = baseline
        baseline_records.append(baseline)
        interface_records.append(interface)
    return seeds, baseline_records, interface_records


def _metric_summary(rows):
    return {
        "mean": {
            name: float(np.mean([row[name] for row in rows]))
            for name in METRIC_NAMES
        },
        "std": {
            name: float(
                np.std(
                    [row[name] for row in rows],
                    ddof=1 if len(rows) > 1 else 0,
                )
            )
            for name in METRIC_NAMES
        },
    }


def summarize_matched_seeds(baseline_paths, interface_paths):
    from utils.experiment_integrity import hash_ordered_keys

    seeds, baseline_records, interface_records = (
        _load_matched_records(baseline_paths, interface_paths)
    )
    baseline_metrics = []
    interface_metrics = []
    delta_metrics = []
    per_seed = []
    for seed, baseline, interface in zip(
        seeds, baseline_records, interface_records
    ):
        baseline_row = _regression_metrics(
            baseline["labels"], baseline["predictions"]
        )
        interface_row = _regression_metrics(
            interface["labels"], interface["predictions"]
        )
        delta_row = {
            name: float(interface_row[name] - baseline_row[name])
            for name in METRIC_NAMES
        }
        baseline_metrics.append(baseline_row)
        interface_metrics.append(interface_row)
        delta_metrics.append(delta_row)
        per_seed.append(
            {
                "seed": seed,
                "baseline": baseline_row,
                "interface_first": interface_row,
                "delta_interface_minus_baseline": delta_row,
                "baseline_prediction_path": baseline["path"],
                "interface_prediction_path": interface["path"],
            }
        )
    keys = baseline_records[0]["keys"].tolist()
    return {
        "seeds": seeds,
        "sample_count": len(keys),
        "ordered_key_sha256": hash_ordered_keys(keys),
        "per_seed": per_seed,
        "baseline": _metric_summary(baseline_metrics),
        "interface_first": _metric_summary(interface_metrics),
        "delta": _metric_summary(delta_metrics),
    }


def _sample_complex_indices(sample_count, generator):
    return generator.randint(
        0, int(sample_count), size=int(sample_count)
    ).astype(np.int64)


def _sample_group_indices(group_ids, generator):
    group_ids = np.asarray(group_ids).reshape(-1)
    if len(group_ids) == 0:
        raise ValueError("group ids must be nonempty")
    ordered_groups = list(dict.fromkeys(group_ids.tolist()))
    members = {
        group: np.flatnonzero(group_ids == group)
        for group in ordered_groups
    }
    selected = generator.randint(
        0, len(ordered_groups), size=len(ordered_groups)
    )
    return np.concatenate(
        [members[ordered_groups[index]] for index in selected]
    ).astype(np.int64)


def _validate_record_lists(baseline_records, interface_records):
    if len(baseline_records) != len(interface_records):
        raise ValueError(
            "baseline and interface-first record counts must match"
        )
    if not baseline_records:
        raise ValueError("at least one matched seed is required")
    reference = baseline_records[0]
    for index, (baseline, interface) in enumerate(
        zip(baseline_records, interface_records)
    ):
        _aligned_pair(baseline, interface, f"seed index {index}")
        _aligned_pair(reference, baseline, f"baseline seed index {index}")
    return reference


def paired_bootstrap_delta_ci(
    baseline_records,
    interface_records,
    group_ids=None,
    n_bootstrap=DEFAULT_BOOTSTRAP_DRAWS,
    seed=DEFAULT_BOOTSTRAP_SEED,
):
    reference = _validate_record_lists(
        baseline_records, interface_records
    )
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    sample_count = len(reference["keys"])
    if group_ids is not None:
        group_ids = np.asarray(group_ids).reshape(-1)
        if len(group_ids) != sample_count:
            raise ValueError("group ids must align with prediction keys")
    point_deltas = [
        _fast_concordance_index(
            interface["labels"], interface["predictions"]
        )
        - _fast_concordance_index(
            baseline["labels"], baseline["predictions"]
        )
        for baseline, interface in zip(
            baseline_records, interface_records
        )
    ]
    generator = np.random.RandomState(int(seed))
    draws = []
    for _ in range(n_bootstrap):
        if group_ids is None:
            indices = _sample_complex_indices(
                sample_count, generator
            )
        else:
            indices = _sample_group_indices(group_ids, generator)
        seed_deltas = []
        for baseline, interface in zip(
            baseline_records, interface_records
        ):
            labels = baseline["labels"][indices]
            baseline_ci = _fast_concordance_index(
                labels, baseline["predictions"][indices]
            )
            interface_ci = _fast_concordance_index(
                labels, interface["predictions"][indices]
            )
            seed_deltas.append(interface_ci - baseline_ci)
        draws.append(float(np.mean(seed_deltas)))
    draw_array = np.asarray(draws, dtype=np.float64)
    interval = np.percentile(draw_array, [2.5, 97.5])
    return {
        "unit": "complex" if group_ids is None else "target_cluster",
        "draw_count": n_bootstrap,
        "seed": int(seed),
        "point_estimate": float(np.mean(point_deltas)),
        "interval_95": [float(interval[0]), float(interval[1])],
        "draws": [float(value) for value in draw_array],
    }


def _normalize_sequence(sequence):
    if isinstance(sequence, bytes):
        sequence = sequence.decode("utf-8")
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("target chain sequences must be nonempty")
    return normalized


def _chain_multiset_hash(tokens):
    normalized = sorted(str(token) for token in tokens)
    if not normalized:
        raise ValueError("each target must contain at least one chain")
    canonical = "|".join(
        f"{len(token)}:{token}" for token in normalized
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_chains(keys, chain_sequences_by_key):
    output = {}
    for key in keys:
        key = str(key)
        if key not in chain_sequences_by_key:
            raise KeyError(f"target chain sequences are missing key: {key}")
        chains = chain_sequences_by_key[key]
        if isinstance(chains, (str, bytes)) or not isinstance(
            chains, (list, tuple)
        ):
            raise ValueError(
                f"target chain sequences must be a list for {key}"
            )
        output[key] = [_normalize_sequence(chain) for chain in chains]
        if not output[key]:
            raise ValueError(f"target has no chain sequences: {key}")
    return output


def exact_target_groups(keys, chain_sequences_by_key):
    keys = [str(key) for key in keys]
    chains = _normalized_chains(keys, chain_sequences_by_key)
    return [
        _chain_multiset_hash(chains[key])
        for key in keys
    ]


def run_mmseqs_target_clustering(
    keys,
    chain_sequences_by_key,
    output_dir,
    mmseqs_binary="mmseqs",
    min_identity=MMSEQS_MIN_IDENTITY,
    coverage=MMSEQS_COVERAGE,
    threads=1,
    runner=subprocess.run,
):
    from utils.experiment_integrity import atomic_write_json, sha256_file

    keys = [str(key) for key in keys]
    chains = _normalized_chains(keys, chain_sequences_by_key)
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    unique_sequences = sorted(
        {sequence for values in chains.values() for sequence in values}
    )
    sequence_to_id = {
        sequence: "seq_" + hashlib.sha256(
            sequence.encode("utf-8")
        ).hexdigest()
        for sequence in unique_sequences
    }
    if len(set(sequence_to_id.values())) != len(sequence_to_id):
        raise RuntimeError("sequence identifier hash collision")
    fasta_path = os.path.join(output_dir, "target_chains.fasta")
    with open(fasta_path, "w", encoding="utf-8") as handle:
        for sequence in unique_sequences:
            handle.write(f">{sequence_to_id[sequence]}\n{sequence}\n")

    version_result = runner(
        [str(mmseqs_binary), "--version"],
        check=True,
        text=True,
        capture_output=True,
    )
    version = (
        str(getattr(version_result, "stdout", "") or "").strip()
        or str(getattr(version_result, "stderr", "") or "").strip()
    )
    prefix = os.path.join(output_dir, "related_targets")
    temporary_dir = os.path.join(output_dir, "tmp")
    os.makedirs(temporary_dir, exist_ok=True)
    cluster_path = prefix + "_cluster.tsv"
    if os.path.exists(cluster_path):
        os.unlink(cluster_path)
    command = [
        str(mmseqs_binary),
        "easy-cluster",
        fasta_path,
        prefix,
        temporary_dir,
        "--min-seq-id",
        f"{float(min_identity):g}",
        "-c",
        f"{float(coverage):g}",
        "--cov-mode",
        str(MMSEQS_COVERAGE_MODE),
        "--threads",
        str(int(threads)),
    ]
    runner(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    if not os.path.isfile(cluster_path):
        raise RuntimeError(
            f"MMseqs2 did not create cluster TSV: {cluster_path}"
        )
    member_to_representative = {}
    with open(cluster_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not all(fields):
                raise RuntimeError(
                    f"invalid MMseqs2 cluster row {line_number}"
                )
            representative, member = fields
            previous = member_to_representative.setdefault(
                member, representative
            )
            if previous != representative:
                raise RuntimeError(
                    f"MMseqs2 member has multiple clusters: {member}"
                )
    missing_ids = sorted(
        set(sequence_to_id.values()) - set(member_to_representative)
    )
    if missing_ids:
        raise RuntimeError(
            "MMseqs2 cluster TSV omits sequences: "
            + ", ".join(missing_ids[:5])
        )

    group_ids = []
    key_to_group = {}
    for key in keys:
        cluster_tokens = [
            member_to_representative[sequence_to_id[sequence]]
            for sequence in chains[key]
        ]
        group = _chain_multiset_hash(cluster_tokens)
        group_ids.append(group)
        key_to_group[key] = group
    groups_path = os.path.join(
        output_dir, "related_target_groups.json"
    )
    atomic_write_json(
        groups_path,
        {
            "ordered_keys": keys,
            "group_ids": group_ids,
            "key_to_group": key_to_group,
        },
    )
    metadata = {
        "version": version,
        "command": command,
        "minimum_sequence_identity": float(min_identity),
        "coverage": float(coverage),
        "coverage_mode": MMSEQS_COVERAGE_MODE,
        "threads": int(threads),
        "fasta_path": fasta_path,
        "fasta_sha256": sha256_file(fasta_path),
        "cluster_path": cluster_path,
        "cluster_sha256": sha256_file(cluster_path),
        "groups_path": groups_path,
        "unique_chain_count": len(unique_sequences),
        "complex_count": len(keys),
    }
    metadata_path = os.path.join(
        output_dir, "mmseqs_metadata.json"
    )
    atomic_write_json(metadata_path, metadata)
    return {
        "group_ids": group_ids,
        "key_to_group": key_to_group,
        "metadata": metadata,
        "metadata_path": metadata_path,
    }


def _positive_interval(result):
    lower, upper = [
        float(value) for value in result["interval_95"]
    ]
    if lower > upper:
        raise ValueError("bootstrap interval is reversed")
    return lower, upper


def interpret_bootstrap(complex_result, exact_result, related_result):
    complex_interval = _positive_interval(complex_result)
    exact_interval = _positive_interval(exact_result)
    related_interval = _positive_interval(related_result)
    if complex_interval[1] < 0:
        status = "degradation"
        stable = False
        message = (
            "The complex-level interval is below zero; the interface-first "
            "model is worse on CI under this comparison."
        )
    elif complex_interval[0] <= 0:
        status = "inconclusive"
        stable = False
        message = (
            "The complex-level interval includes zero; no CI improvement "
            "is established."
        )
    elif exact_interval[0] <= 0 or related_interval[0] <= 0:
        status = "target_repeat_driven"
        stable = False
        message = (
            "Complex-level improvement is positive, but a target-cluster "
            "interval includes zero; do not claim stable cross-target "
            "improvement."
        )
    else:
        status = "stable_cross_target_improvement"
        stable = True
        message = (
            "Complex, exact-target, and related-target intervals are all "
            "above zero."
        )
    return {
        "status": status,
        "stable_cross_target_improvement": stable,
        "message": message,
    }


def load_chain_sequences_from_native_lmdb(path, keys):
    import lmdb

    path = os.path.abspath(os.fspath(path))
    environment = lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    output = {}
    try:
        with environment.begin(write=False) as transaction:
            for key in keys:
                key = str(key)
                raw = transaction.get(key.encode("utf-8"))
                if raw is None:
                    raise KeyError(
                        f"native interface LMDB key not found: {key}"
                    )
                record = pickle.loads(raw)
                chain_indices = np.asarray(
                    record["residue_chain_index"]
                ).reshape(-1)
                sequence = np.asarray(
                    record["residue_sequence"]
                ).reshape(-1)
                if len(chain_indices) != len(sequence):
                    raise ValueError(
                        f"native target chain fields do not align for {key}"
                    )
                chains = []
                for chain_index in sorted(
                    set(chain_indices.tolist())
                ):
                    mask = chain_indices == chain_index
                    chains.append(
                        "".join(str(value) for value in sequence[mask])
                    )
                output[key] = chains
    finally:
        environment.close()
    return output


def _load_target_metadata(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(
        payload.get("samples"), list
    ):
        output = {}
        for sample in payload["samples"]:
            key = str(sample["lmdb_key"])
            if key in output:
                raise ValueError(
                    f"target metadata contains duplicate key: {key}"
                )
            output[key] = sample["chain_sequences"]
        return output
    if isinstance(payload, dict) and isinstance(
        payload.get("targets"), dict
    ):
        payload = payload["targets"]
    if not isinstance(payload, dict):
        raise ValueError("target metadata must be a mapping or samples list")
    return {str(key): value for key, value in payload.items()}


def _parse_seed_paths(values, option_name):
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"{option_name} values must use SEED=PATH"
            )
        seed_text, path = value.split("=", 1)
        seed = int(seed_text)
        if seed in output:
            raise ValueError(
                f"{option_name} contains duplicate seed {seed}"
            )
        if not path:
            raise ValueError(f"{option_name} path cannot be empty")
        output[seed] = path
    return output


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize five matched LP-PDBBind seeds and paired "
            "complex/target-cluster CI uncertainty."
        )
    )
    parser.add_argument(
        "--baseline-prediction",
        action="append",
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument(
        "--interface-prediction",
        action="append",
        required=True,
        metavar="SEED=PATH",
    )
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--target-metadata")
    targets.add_argument("--target-native-lmdb")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mmseqs-output-dir", default=None)
    parser.add_argument("--mmseqs-binary", default="mmseqs")
    parser.add_argument("--mmseqs-threads", type=int, default=1)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=DEFAULT_BOOTSTRAP_DRAWS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser


def summarize_from_args(args):
    from utils.experiment_integrity import atomic_write_json

    baseline_paths = _parse_seed_paths(
        args.baseline_prediction, "--baseline-prediction"
    )
    interface_paths = _parse_seed_paths(
        args.interface_prediction, "--interface-prediction"
    )
    summary = summarize_matched_seeds(
        baseline_paths, interface_paths
    )
    seeds, baseline_records, interface_records = (
        _load_matched_records(baseline_paths, interface_paths)
    )
    keys = baseline_records[0]["keys"].tolist()
    if args.target_metadata:
        chain_sequences = _load_target_metadata(args.target_metadata)
        target_source = {
            "type": "json",
            "path": os.path.abspath(args.target_metadata),
        }
    else:
        chain_sequences = load_chain_sequences_from_native_lmdb(
            args.target_native_lmdb, keys
        )
        target_source = {
            "type": "native_lmdb",
            "path": os.path.abspath(args.target_native_lmdb),
        }
    exact_groups = exact_target_groups(keys, chain_sequences)
    output_path = os.path.abspath(args.output_json)
    mmseqs_output_dir = (
        os.path.abspath(args.mmseqs_output_dir)
        if args.mmseqs_output_dir
        else os.path.join(
            os.path.dirname(output_path), "mmseqs_related_targets"
        )
    )
    related = run_mmseqs_target_clustering(
        keys,
        chain_sequences,
        mmseqs_output_dir,
        mmseqs_binary=args.mmseqs_binary,
        threads=args.mmseqs_threads,
    )
    complex_bootstrap = paired_bootstrap_delta_ci(
        baseline_records,
        interface_records,
        n_bootstrap=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    exact_bootstrap = paired_bootstrap_delta_ci(
        baseline_records,
        interface_records,
        group_ids=exact_groups,
        n_bootstrap=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    related_bootstrap = paired_bootstrap_delta_ci(
        baseline_records,
        interface_records,
        group_ids=related["group_ids"],
        n_bootstrap=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    result = {
        "matched_seed_summary": summary,
        "bootstrap": {
            "complex": complex_bootstrap,
            "exact_target": exact_bootstrap,
            "related_target_90_identity": related_bootstrap,
        },
        "interpretation": interpret_bootstrap(
            complex_bootstrap,
            exact_bootstrap,
            related_bootstrap,
        ),
        "target_grouping": {
            "source": target_source,
            "exact_key_to_group": dict(zip(keys, exact_groups)),
            "related_key_to_group": related["key_to_group"],
            "mmseqs_metadata_path": related["metadata_path"],
        },
        "configuration": {
            "seeds": seeds,
            "bootstrap_draws": int(args.bootstrap_draws),
            "bootstrap_seed": int(args.bootstrap_seed),
            "ci_tolerance": CI_TOLERANCE,
            "mmseqs_min_identity": MMSEQS_MIN_IDENTITY,
            "mmseqs_coverage": MMSEQS_COVERAGE,
        },
    }
    atomic_write_json(output_path, result)
    return result


def main():
    args = build_arg_parser().parse_args()
    result = summarize_from_args(args)
    print(json.dumps(result["interpretation"], sort_keys=True))
    print("summary:", os.path.abspath(args.output_json))


if __name__ == "__main__":
    main()
