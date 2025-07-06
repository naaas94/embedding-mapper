"""
Base clustering interface.

Provides a common interface for all clustering algorithms
with support for caching, parameter management, and result analysis.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator


class Clusterer(ABC):
    """
    Abstract base class for clustering algorithms.
    
    Provides a unified interface for discovering clusters and structure
    in embeddings with caching and parameter management.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize the clusterer.
        
        Args:
            cache_dir: Directory to cache results
            random_state: Random seed for reproducibility
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/clusters")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.model: Optional[BaseEstimator] = None
        self.is_fitted = False
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying clustering model."""
        pass
    
    def _get_cache_path(self, embeddings: np.ndarray, params: Dict[str, Any]) -> Path:
        """Generate cache file path based on embeddings and parameters."""
        import hashlib
        
        # Create hash from embeddings shape and parameters
        data_hash = hashlib.md5(
            f"{embeddings.shape}_{str(sorted(params.items()))}".encode()
        ).hexdigest()[:16]
        
        return self.cache_dir / f"{self.__class__.__name__}_{data_hash}.pkl"
    
    def fit_predict(
        self,
        embeddings: np.ndarray,
        use_cache: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Fit the model and predict cluster labels.
        
        Args:
            embeddings: Input embeddings (n_samples, n_features)
            use_cache: Whether to use cached results if available
            **kwargs: Additional parameters for the specific method
            
        Returns:
            Cluster labels (n_samples,)
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(embeddings, kwargs)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data['params'] == kwargs:
                        print(f"Using cached clustering from {cache_path}")
                        self.labels_ = cached_data['labels']
                        self.n_clusters_ = cached_data['n_clusters']
                        self.model = cached_data['model']
                        self.is_fitted = True
                        return self.labels_
        
        # Create and fit model
        self.model = self._create_model()
        self._set_parameters(**kwargs)
        
        # Fit and predict
        self.labels_ = self.model.fit_predict(embeddings)
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ != -1]))
        self.is_fitted = True
        
        # Cache results
        if use_cache:
            cache_path = self._get_cache_path(embeddings, kwargs)
            cache_data = {
                'labels': self.labels_,
                'n_clusters': self.n_clusters_,
                'params': kwargs,
                'model': self.model
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        return self.labels_
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new embeddings.
        
        Args:
            embeddings: Input embeddings to cluster
            
        Returns:
            Cluster labels
            
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        
        return self.model.predict(embeddings)
    
    def fit(self, embeddings: np.ndarray, **kwargs) -> 'Clusterer':
        """
        Fit the model without predicting.
        
        Args:
            embeddings: Input embeddings
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        self.model = self._create_model()
        self._set_parameters(**kwargs)
        self.model.fit(embeddings)
        self.is_fitted = True
        return self
    
    def _set_parameters(self, **kwargs):
        """Set parameters on the underlying model."""
        if self.model is not None:
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers (if available).
        
        Returns:
            Array of cluster centers or None if not available
        """
        if self.model is None or not self.is_fitted:
            return None
        
        return getattr(self.model, 'cluster_centers_', None)
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get the size of each cluster.
        
        Returns:
            Dictionary mapping cluster labels to cluster sizes
        """
        if self.labels_ is None:
            return {}
        
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(unique_labels, counts))
    
    def get_cluster_indices(self) -> Dict[int, np.ndarray]:
        """
        Get indices of points in each cluster.
        
        Returns:
            Dictionary mapping cluster labels to arrays of indices
        """
        if self.labels_ is None:
            return {}
        
        cluster_indices = {}
        for label in np.unique(self.labels_):
            cluster_indices[label] = np.where(self.labels_ == label)[0]
        
        return cluster_indices
    
    def get_cluster_embeddings(self, embeddings: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get embeddings for each cluster.
        
        Args:
            embeddings: Original embeddings
            
        Returns:
            Dictionary mapping cluster labels to cluster embeddings
        """
        if self.labels_ is None:
            return {}
        
        cluster_embeddings = {}
        for label in np.unique(self.labels_):
            mask = self.labels_ == label
            cluster_embeddings[label] = embeddings[mask]
        
        return cluster_embeddings
    
    def get_noise_points(self) -> np.ndarray:
        """
        Get indices of noise points (label == -1).
        
        Returns:
            Array of indices for noise points
        """
        if self.labels_ is None:
            return np.array([])
        
        return np.where(self.labels_ == -1)[0]
    
    def clear_cache(self):
        """Clear all cached clustering results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Cleared cache directory: {self.cache_dir}") 