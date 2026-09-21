"""Unified explicit exporter for the three Rank Fusion v4 experts."""

import argparse
import json
from pathlib import Path

from utils.experiment_integrity import manifest_fingerprint, sha256_file

from .artifact import load_expert_artifact


EXPERTS = ("upper_bound", "boltz_pairwise", "three_dprot")


class ExpertArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        requirements = {
            "upper_bound": ("upper_bound_cache",),
            "boltz_pairwise": ("boltz_lmdb_dir", "boltz_index_path"),
            "three_dprot": ("three_dprot_data_dir",),
        }
        missing = [
            name for name in requirements[parsed.expert] if not getattr(parsed, name)
        ]
        if missing:
            self.error(
                parsed.expert
                + " requires "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        return parsed


def build_arg_parser():
    parser = ExpertArgumentParser(
        description="Export one keyed Rank Fusion v4 expert"
    )
    parser.add_argument("--expert", required=True, choices=EXPERTS)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--split", required=True, choices=("train", "valid", "test"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--upper-bound-cache")
    parser.add_argument("--boltz-lmdb-dir")
    parser.add_argument("--boltz-index-path")
    parser.add_argument("--three-dprot-data-dir")
    return parser


def _reuse_existing(args, output):
    manifest = manifest_fingerprint(args.manifest)
    if manifest["split"] != args.split:
        raise ValueError("manifest split does not match requested split")
    return load_expert_artifact(
        output,
        expected_expert=args.expert,
        expected_split=args.split,
        expected_manifest_sha256=manifest["sha256"],
        expected_checkpoint_sha256=sha256_file(args.checkpoint),
    )


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    output = (
        Path(args.artifact_root).expanduser().resolve() / args.split / args.expert
    )
    if (output / "metadata.json").is_file():
        artifact = _reuse_existing(args, output)
        print(
            json.dumps({"status": "reused", **artifact.metadata}, sort_keys=True),
            flush=True,
        )
        return
    common = {
        "manifest_path": args.manifest,
        "checkpoint_path": args.checkpoint,
        "output_dir": output,
        "split": args.split,
        "device": args.device,
        "batch_size": args.batch_size,
        "limit": args.limit,
    }
    if args.expert == "upper_bound":
        from .upper_bound_export import export_upper_bound

        metadata = export_upper_bound(cache_dir=args.upper_bound_cache, **common)
    elif args.expert == "boltz_pairwise":
        from .boltz_export import export_boltz_pairwise

        metadata = export_boltz_pairwise(
            lmdb_dir=args.boltz_lmdb_dir,
            index_path=args.boltz_index_path,
            **common,
        )
    else:
        from .three_dprot_export import export_three_dprot

        metadata = export_three_dprot(
            data_dir=args.three_dprot_data_dir,
            **common,
        )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
