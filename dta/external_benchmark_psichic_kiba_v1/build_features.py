from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import build_protein_cache, write_ligand_cache


def main():
    parser = argparse.ArgumentParser(description="Build PSICHIC-sequence-edge ESM2 residue and ligand caches.")
    parser.add_argument("--data-root", default="data/ExternalBenchmarkKIBA_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--window-size", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ligand = write_ligand_cache(args.data_root, output)
    protein = build_protein_cache(args.data_root, output, args.model, args.device, args.window_size, args.overlap, args.precision, args.overwrite)
    result = {"version": "psichic-sequence-edge-feature-build-v1", "ligand": ligand, "protein": protein}
    (output / "build_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
