"""V30 train-only privileged-MSA distillation package.

The deployed predictor remains a single no-MSA optional-structure network.
MSA features exist only in the train-only offline teacher cache.
"""

from .model import OptionalStructurePrivilegedMSADTA
from .training import TrainingConfig, VERSION

__all__ = ["OptionalStructurePrivilegedMSADTA", "TrainingConfig", "VERSION"]
