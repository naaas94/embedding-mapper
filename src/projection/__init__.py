"""
Dimensionality reduction and projection module.

Provides interfaces for projecting high-dimensional embeddings to lower dimensions:
- UMAP for preserving local and global structure
- t-SNE for local structure preservation
- PCA for linear dimensionality reduction
"""

from .reducer import DimensionalityReducer
from .umap_projector import UMAPProjector

__all__ = [
    "DimensionalityReducer",
    "UMAPProjector"
] 