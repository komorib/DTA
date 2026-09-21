"""V21 missing-modality-aware sequence/2D/optional-3D DTA model."""

VERSION = "optional-structure-multitask-v21"

from .model import OptionalStructureDTA, OptionalStructureOutput

__all__ = ["VERSION", "OptionalStructureDTA", "OptionalStructureOutput"]
