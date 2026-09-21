# KIBA binary benchmark V1

This is an independent data/CPU validation bundle for comparing external DTI
implementations on the repository's fixed KIBA split. It does not modify any
upstream model repository and does not launch GPU work.

The task is binary classification with `label = (affinity >= 12.1)`. The
fixed source split is decoded from `data/KIBA`. Exact duplicate
sequence–SMILES pairs found in train and valid/test are removed from train;
the official valid/test rows are preserved and any valid/test duplicate pairs
are reported in metadata. The resulting leak-audited counts are
82,506/11,825/23,651 train/valid/test examples. Epoch/checkpoint selection
must use validation only; the test split is for the final report.

Prepare the bundle on CPU:

```bash
# Small/simple invocation (suitable for a limited smoke bundle)
bash experiments/external_benchmarks/kiba_binary_v1/prepare_cpu.sh

# Full 118,254-row bundle; processes train/valid/test separately to bound
# wall-clock and memory usage.
bash experiments/external_benchmarks/kiba_binary_v1/prepare_full_cpu.sh

bash experiments/external_benchmarks/kiba_binary_v1/smoke_cpu.sh
```

For a small CPU smoke bundle, use `LIMIT=24`; this writes to a separate output
directory, for example `OUTPUT_DIR=/tmp/kiba_binary_v1_smoke`.

The generated model-specific files are adapters only:

- `flat_*.csv` and `common/*.jsonl`: DeepDTA source/audit formats;
- `graphdta/*.csv`: graph-DTA column names plus the binary label;
- `transformercpi/*.txt`: SMILES, sequence, binary label;
- `graphormer/*.txt`: stable ligand/protein IDs, sequence, binary label;
- `psichic/*.csv`: both regression and classification labels;
- `fmca/*.data`: FMCA's whitespace format;
- `common/*.jsonl`: canonical audit source with both labels.
