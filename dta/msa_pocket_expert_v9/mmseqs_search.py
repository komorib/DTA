"""Run a resumable ColabFold-style UniRef30 search and validate its A3M output."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


AA_WITH_GAP = frozenset("ACDEFGHIKLMNPQRSTVWYXBZJUO-")
INSERTION_RE = re.compile(r"[a-z]")


def read_fasta(path, uppercase=True, require_unique=True):
    records = []
    name = None
    chunks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequence = "".join(chunks)
                    records.append((name, sequence.upper() if uppercase else sequence))
                name = line[1:].split()[0]
                chunks = []
            elif name is None:
                raise ValueError("FASTA sequence appears before its header")
            else:
                chunks.append(line)
    if name is not None:
        sequence = "".join(chunks)
        records.append((name, sequence.upper() if uppercase else sequence))
    if not records:
        raise ValueError("query FASTA is empty")
    names = [name for name, _sequence in records]
    if require_unique and len(names) != len(set(names)):
        raise ValueError("query FASTA contains duplicate identifiers")
    return records


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + "-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_step(name, command, output_dbtype, state_dir, log_path, env=None):
    marker = state_dir / (name + ".done.json")
    output_dbtype = Path(output_dbtype)
    if marker.is_file() and output_dbtype.is_file():
        print(json.dumps({"event": "mmseqs_step_skipped", "step": name}, sort_keys=True), flush=True)
        return
    started = time.time()
    print(json.dumps({"command": [str(value) for value in command], "event": "mmseqs_step_start", "step": name}, sort_keys=True), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[" + name + "] " + " ".join(str(value) for value in command) + "\n")
        log.flush()
        subprocess.run([str(value) for value in command], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    if not output_dbtype.is_file():
        raise RuntimeError(f"MMseqs step {name} did not create {output_dbtype}")
    _atomic_json(
        marker,
        {
            "command": [str(value) for value in command],
            "elapsed_seconds": time.time() - started,
            "output_dbtype": str(output_dbtype),
            "status": "complete",
        },
    )
    print(json.dumps({"elapsed_seconds": time.time() - started, "event": "mmseqs_step_complete", "step": name}, sort_keys=True), flush=True)


def validate_a3m(path, expected_name, expected_sequence):
    records = read_fasta(path, uppercase=False, require_unique=False)
    query_name, query_sequence = records[0]
    query_columns = INSERTION_RE.sub("", query_sequence)
    if query_columns != expected_sequence:
        raise ValueError(f"A3M query does not match input for {expected_name}: {query_name}")
    for row_name, row in records:
        columns = INSERTION_RE.sub("", row)
        if len(columns) != len(expected_sequence):
            raise ValueError(f"A3M row {row_name} has {len(columns)} columns, expected {len(expected_sequence)}")
        invalid = set(columns) - AA_WITH_GAP
        if invalid:
            raise ValueError(f"A3M row {row_name} contains invalid residues: {sorted(invalid)}")
    return len(records)


def run_uniref30_search(query_fasta, database_prefix, output_dir, mmseqs, threads=32, sensitivity=8.0, unpack=True):
    query_fasta = Path(query_fasta).expanduser().resolve()
    database_prefix = Path(database_prefix).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    mmseqs = Path(mmseqs).expanduser().resolve()
    records = read_fasta(query_fasta)
    for suffix in (".dbtype", "_seq.dbtype", "_aln.dbtype"):
        if not Path(str(database_prefix) + suffix).is_file():
            raise FileNotFoundError(str(database_prefix) + suffix)
    if not mmseqs.is_file():
        raise FileNotFoundError(mmseqs)

    work = output_dir / "work"
    state = output_dir / "state"
    a3m_dir = output_dir / "a3m"
    work.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    a3m_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "mmseqs.log"
    lineage_path = output_dir / "lineage.json"
    lineage = {
        "database_prefix": str(database_prefix),
        "mmseqs": str(mmseqs),
        "query_count": len(records),
        "query_fasta": str(query_fasta),
        "query_fasta_sha256": _sha256(query_fasta),
        "sensitivity": float(sensitivity),
        "threads": int(threads),
        "version": "colabfold-uniref30-v1",
    }
    if lineage_path.is_file():
        existing = json.loads(lineage_path.read_text(encoding="utf-8"))
        if existing != lineage:
            raise ValueError("output directory lineage does not match this invocation")
    else:
        _atomic_json(lineage_path, lineage)

    env = dict(os.environ)
    env["MMSEQS_CALL_DEPTH"] = "1"
    db = str(database_prefix)
    db_seq = db + "_seq"
    db_aln = db + "_aln"
    qdb = work / "qdb"
    res = work / "res"
    tmp = work / "tmp"
    profile = work / "prof_res"
    expanded = work / "res_exp"
    realigned = work / "res_exp_realign"
    filtered = work / "res_exp_realign_filter"
    msa_db = work / "uniref.a3m"
    common_search = [
        "--threads", str(threads), "--num-iterations", "3", "--db-load-mode", "0",
        "-a", "-e", "0.1", "--max-seqs", "10000", "--prefilter-mode", "0",
        "-s", f"{float(sensitivity):.1f}",
    ]

    _run_step("01_createdb", [mmseqs, "createdb", query_fasta, qdb, "--shuffle", "0", "--dbtype", "1"], str(qdb) + ".dbtype", state, log_path, env)
    _run_step("02_search", [mmseqs, "search", qdb, db, res, tmp] + common_search, str(res) + ".dbtype", state, log_path, env)
    _run_step("03_profile", [mmseqs, "mvdb", tmp / "latest/profile_1", profile], str(profile) + ".dbtype", state, log_path, env)
    _run_step("04_profile_headers", [mmseqs, "lndb", str(qdb) + "_h", str(profile) + "_h"], str(profile) + "_h.dbtype", state, log_path, env)
    _run_step(
        "05_expandaln",
        [mmseqs, "expandaln", qdb, db_seq, res, db_aln, expanded, "--db-load-mode", "0", "--threads", str(threads), "--expansion-mode", "0", "-e", "inf", "--expand-filter-clusters", "1", "--max-seq-id", "0.95"],
        str(expanded) + ".dbtype", state, log_path, env,
    )
    _run_step(
        "06_align",
        [mmseqs, "align", profile, db_seq, expanded, realigned, "--db-load-mode", "0", "-e", "10", "--max-accept", "100000", "--threads", str(threads), "--alt-ali", "10", "-a"],
        str(realigned) + ".dbtype", state, log_path, env,
    )
    _run_step(
        "07_filterresult",
        [mmseqs, "filterresult", qdb, db_seq, realigned, filtered, "--db-load-mode", "0", "--qid", "0", "--qsc", "0.8", "--diff", "0", "--threads", str(threads), "--max-seq-id", "1.0", "--filter-min-enable", "100"],
        str(filtered) + ".dbtype", state, log_path, env,
    )
    _run_step(
        "08_result2msa",
        [mmseqs, "result2msa", qdb, db_seq, filtered, msa_db, "--msa-format-mode", "6", "--db-load-mode", "0", "--threads", str(threads), "--filter-msa", "1", "--filter-min-enable", "1000", "--diff", "3000", "--qid", "0.0,0.2,0.4,0.6,0.8,1.0", "--qsc", "0", "--max-seq-id", "0.95"],
        str(msa_db) + ".dbtype", state, log_path, env,
    )
    if not unpack:
        report = {
            "a3m_database": str(msa_db),
            "database_prefix": str(database_prefix),
            "output": str(output_dir),
            "query_count": len(records),
            "status": "database_complete",
        }
        _atomic_json(output_dir / "search_report.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return report

    unpack_marker = state / "09_unpack.done.json"
    if not unpack_marker.is_file():
        command = [mmseqs, "unpackdb", msa_db, a3m_dir, "--unpack-name-mode", "0", "--unpack-suffix", ".a3m"]
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n[09_unpack] " + " ".join(str(value) for value in command) + "\n")
            log.flush()
            subprocess.run([str(value) for value in command], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        _atomic_json(unpack_marker, {"elapsed_seconds": time.time() - started, "status": "complete"})

    depths = []
    for index, (name, sequence) in enumerate(records):
        numeric_path = a3m_dir / f"{index}.a3m"
        named_path = a3m_dir / f"{name}.a3m"
        if numeric_path.is_file() and numeric_path != named_path:
            os.replace(numeric_path, named_path)
        if not named_path.is_file():
            raise FileNotFoundError(named_path)
        depths.append(validate_a3m(named_path, name, sequence))
    report = {
        "database_prefix": str(database_prefix),
        "depth": {"maximum": max(depths), "mean": sum(depths) / len(depths), "minimum": min(depths)},
        "output": str(output_dir),
        "query_count": len(records),
        "status": "complete",
    }
    _atomic_json(output_dir / "search_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-fasta", required=True)
    parser.add_argument("--database-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mmseqs", default="/home/user/wangzhiwen/miniconda3/envs/msa_v9/bin/mmseqs")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--sensitivity", type=float, default=8.0)
    parser.add_argument("--unpack", type=int, choices=(0, 1), default=1)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_uniref30_search(
        args.query_fasta,
        args.database_prefix,
        args.output,
        args.mmseqs,
        threads=args.threads,
        sensitivity=args.sensitivity,
        unpack=bool(args.unpack),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
