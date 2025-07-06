"""
Base dimensionality reduction interface.

Provides a common interface for all dimensionality reduction techniques
with support for caching, parameter management, and result storage.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Protocol
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator


class EstimatorProtocol(Protocol):
    """Protocol for sklearn-style estimators."""
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit(self, X: np.ndarray) -> 'EstimatorProtocol': ...


class DimensionalityReducer(ABC):
    """
    Abstract base class for dimensionality reduction techniques.
    
    Provides a unified interface for projecting high-dimensional embeddings
    to lower dimensions with caching and parameter management.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        cache_dir: Optional[str] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize the dimensionality reducer.
        
        Args:
            n_components: Number of dimensions to project to
            cache_dir: Directory to cache results
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/projections")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.model: Optional[EstimatorProtocol] = None
        self.is_fitted = False
    
    @abstractmethod
    def _create_model(self) -> EstimatorProtocol:
        """Create the underlying reduction model."""
        pass
    
    def _get_cache_path(self, embeddings: np.ndarray, params: Dict[str, Any]) -> Path:
        """Generate cache file path based on embeddings and parameters."""
        import hashlib
        
        # Create hash from embeddings shape and parameters
        data_hash = hashlib.md5(
            f"{embeddings.shape}_{str(sorted(params.items()))}".encode()
        ).hexdigest()[:16]
        
        return self.cache_dir / f"{self.__class__.__name__}_{data_hash}.pkl"
    
    def fit_transform(
        self,
        embeddings: np.ndarray,
        use_cache: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Fit the model and transform the embeddings.
        
        Args:
            embeddings: Input embeddings (n_samples, n_features)
            use_cache: Whether to use cached results if available
            **kwargs: Additional parameters for the specific method
            
        Returns:
            Projected embeddings (n_samples, n_components)
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(embeddings, kwargs)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data['params'] == kwargs:
                        print(f"Using cached projection from {cache_path}")
                        return cached_data['projection']
        
        # Create and fit model
        self.model = self._create_model()
        self._set_parameters(**kwargs)
        
        # Fit and transform
        projection = self.model.fit_transform(embeddings)
        self.is_fitted = True
        
        # Cache results
        if use_cache:
            cache_path = self._get_cache_path(embeddings, kwargs)
            cache_data = {
                'projection': projection,
                'params': kwargs,
                'model': self.model
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        return projection
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings using the fitted model.
        
        Args:
            embeddings: Input embeddings to transform
            
        Returns:
            Projected embeddings
            
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before calling transform()")
        
        return self.model.transform(embeddings)
    
    def fit(self, embeddings: np.ndarray, **kwargs) -> 'DimensionalityReducer':
        """
        Fit the model without transforming.
        
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
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio (if available).
        
        Returns:
            Array of explained variance ratios or None if not available
        """
        if self.model is None or not self.is_fitted:
            return None
        
        # Use getattr to safely access the attribute
        return getattr(self.model, 'explained_variance_ratio_', None)
    
    def clear_cache(self):
        """Clear all cached projections."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Cleared cache directory: {self.cache_dir}") 