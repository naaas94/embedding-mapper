"""
Clustering and structure discovery module.

Provides interfaces for discovering clusters and structure in embeddings:
- K-means for centroid-based clustering
- DBSCAN for density-based clustering
- HDBSCAN for hierarchical density-based clustering
"""

from .clusterer import Clusterer

__all__ = [
    "Clusterer"
] 