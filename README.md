# DTA: LP-PDBBind Early K/V Fusion

This directory is the frozen, standalone code release for the final LP-PDBBind
early-fusion model. It contains the complete local Python source closure needed
to train and evaluate the model. It does not import code from a parent checkout
and it does not include previous ablation implementations.

## Model

The final model uses the original full-chain protein and geometric trunk. The
pooled MoLFormer representation is projected once and appended as one token to
the protein memory used by the cross-attention block:

```text
MoLFormer [B, 768]
  -> LayerNorm -> Linear(768, 512) -> LayerNorm
  -> append one molecule token to protein K/V
  -> original full-chain queries, geometric trunk, masked pooling, affinity head
```

The package is early K/V fusion only. It contains no late molecule branch,
joint-K/V branch, or molecule-fusion fallback.

## Layout

- `model.py`: public model entry point.
- `train.py`: training entry point.
- `evaluate.py`: metrics entry point for saved prediction arrays.
- `data.py`, `losses.py`: dataset adapter and affinity loss.
- `dta/`, `data_utils/`, `utils/`: complete source closure for the final model.
- `config.py`: frozen experiment constants for reporting and reproduction.
- `smoke.py`: dependency and early-fusion contract check.
- `requirements.txt`: Python dependencies.
- `SOURCE_MANIFEST.json`: hashes for every released source and documentation file.

## Environment

Create an environment with Python 3.8 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the local contract check from this directory:

```bash
PYTHONPATH=. python smoke.py
```

The check does not need data or a GPU. It verifies that the release imports
locally and that the molecule representation is appended to K/V exactly once.

## Data and training

The repository intentionally excludes private or large data artifacts. A run
requires the LP-PDBBind manifests, complex/ESM/MSA feature stores, and the
MoLFormer cache described by the command-line help of `train.py`. Run the
entry points from this directory with `PYTHONPATH=.` so
all imports resolve inside the release:

```bash
PYTHONPATH=. python train.py --help
PYTHONPATH=. python evaluate.py --help
```

Checkpoints are produced from a fresh random initialization. No checkpoint,
cache or path outside this standalone package is read implicitly.

## Reproducibility

The release is source-frozen. `config.py` records the final experiment
constants, and `SOURCE_MANIFEST.json` records SHA-256 hashes for the released
files. The reported result still depends on the exact external data manifests,
feature caches, and checkpoint-selection protocol supplied to the entry points.
