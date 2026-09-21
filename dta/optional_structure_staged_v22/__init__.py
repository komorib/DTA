"""V22 staged sequence/2D/optional-3D DTA architecture."""

VERSION = "optional-structure-staged-v22"

from .model import OptionalStructureOutput, OptionalStructureStagedDTA

__all__ = ["VERSION", "OptionalStructureStagedDTA", "OptionalStructureOutput"]
