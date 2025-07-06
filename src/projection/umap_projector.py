"""
UMAP-based dimensionality reduction.

Provides UMAP projection for preserving both local and global structure
in high-dimensional embeddings.
"""

from typing import Optional, Dict, Any
import numpy as np
import umap

from .reducer import DimensionalityReducer, EstimatorProtocol


class UMAPProjector(DimensionalityReducer):
    """
    UMAP-based dimensionality reduction.
    
    UMAP (Uniform Manifold Approximation and Projection) is particularly
    good at preserving both local and global structure in the data.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        cache_dir: Optional[str] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize UMAP projector.
        
        Args:
            n_components: Number of dimensions to project to
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points
            metric: Distance metric to use
            cache_dir: Directory to cache results
            random_state: Random seed for reproducibility
        """
        super().__init__(n_components, cache_dir, random_state)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
    
    def _create_model(self) -> EstimatorProtocol:
        """Create UMAP model."""
        return umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            verbose=False
        )
    
    def fit_transform(
        self,
        embeddings: np.ndarray,
        use_cache: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Fit UMAP and transform embeddings.
        
        Args:
            embeddings: Input embeddings
            use_cache: Whether to use cached results
            **kwargs: Additional UMAP parameters
            
        Returns:
            UMAP projection
        """
        return super().fit_transform(embeddings, use_cache, **kwargs)
    
    def get_umap_parameters(self) -> Dict[str, Any]:
        """
        Get current UMAP parameters.
        
        Returns:
            Dictionary of UMAP parameters
        """
        if self.model is None:
            return {}
        
        return {
            'n_components': self.model.n_components,
            'n_neighbors': self.model.n_neighbors,
            'min_dist': self.model.min_dist,
            'metric': self.model.metric,
            'random_state': self.model.random_state
        }
    
    def get_connectivity_graph(self) -> Optional[np.ndarray]:
        """
        Get the connectivity graph from UMAP (if available).
        
        Returns:
            Connectivity graph or None if not available
        """
        if self.model is None or not self.is_fitted:
            return None
        
        return getattr(self.model, 'graph_', None) 