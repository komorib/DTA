# DTA Experiment Directory

This directory stores reproducible DTA experiment entrypoints. Source code stays in `dta/`, `data_utils/`, and `net/`; model weights and metrics stay in `Model/`.

## Convention

Each experiment has its own directory:

```text
experiments/dta/<number>_<experiment_name>/
  run.sh        # full training command
  NOTES.md      # purpose, data, features, expected comparison, result slot
  logs/         # terminal stdout/stderr from each run
```

`run.sh` writes terminal output to:

```text
experiments/dta/<experiment_name>/logs/YYYYmmdd_HHMMSS.log
```

The model output remains under:

```text
Model/<model_output_name>/
```

For future experiments, copy `template/` or an existing experiment directory, rename both the experiment directory and `MODEL_DIR`, then update `NOTES.md`.
