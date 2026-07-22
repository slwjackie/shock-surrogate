"""Neural surrogates for structured and unstructured compressible-flow states."""

from hypersonic.models.lgno2d import LocalGlobalNeuralOperator2d
from hypersonic.models.mesh_gnn import ConservativeMeshGNN

__all__ = ["LocalGlobalNeuralOperator2d", "ConservativeMeshGNN"]
