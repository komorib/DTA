"""V17 spatial feature contract.

V17 deliberately keeps the frozen V15 representation unchanged.  The new
architecture changes where supervision is applied: the complete official
training subset is exported and used to fit the downstream affinity head.
"""

from dta.msa_interface_spatial_fusion_v15.features import (  # noqa: F401
    extract_spatial_interface_features,
    load_interface_encoders,
    spatial_feature_dimension,
)

__all__ = (
    "extract_spatial_interface_features",
    "load_interface_encoders",
    "spatial_feature_dimension",
)

