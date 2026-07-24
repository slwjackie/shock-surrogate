"""Compressible-flow extensions for the learning-augmented shock surrogate.

The package contains a structured-grid Euler/Navier--Stokes reference solver,
positivity-preserving state projections, a local--global neural operator, and
an unstructured-mesh graph surrogate.  It is intentionally independent from
the original 1-D Burgers testbed so both pipelines remain reproducible.
"""

from hypersonic.state import (
    conservative_to_primitive,
    primitive_to_conservative,
    primitive_to_conservative_3d,
    pressure,
    sound_speed,
)
from hypersonic.positivity import positivity_preserving_blend

__all__ = [
    "conservative_to_primitive",
    "primitive_to_conservative",
    "primitive_to_conservative_3d",
    "pressure",
    "sound_speed",
    "positivity_preserving_blend",
]
