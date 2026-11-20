"""Frozen constants for the final LP-PDBBind early K/V experiment."""

SEED = 20261001
EPOCHS = 30
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.995
MODEL_WIDTH = 512
LATENT_WIDTH = 1024
MAIN_LAYERS = 4
HEADS = 8
DROPOUT = 0.2
MOLFORMER_WIDTH = 768


def model_kwargs():
    """Return the architecture constants used by the final model."""
    return {
        "model_width": MODEL_WIDTH,
        "layers": MAIN_LAYERS,
        "heads": HEADS,
        "dropout": DROPOUT,
        "latent_width": LATENT_WIDTH,
    }
