"""V31 train-only privileged contact-aware MSA-pair distillation package.

The deployed predictor remains a single no-MSA optional-structure network.
Contact-aware MSA pair features exist only in the train-only offline teacher
cache; runtime forward consumes only sequence, ligand graph, and optional 3D.
"""

from .model import OptionalStructurePrivilegedMSAPairDTA
from .training import TrainingConfig, VERSION

__all__ = ["OptionalStructurePrivilegedMSAPairDTA", "TrainingConfig", "VERSION"]
