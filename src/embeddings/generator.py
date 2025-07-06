"""
Core embedding generation functionality.

Provides a unified interface for generating embeddings from various models
with support for caching, batch processing, and model switching.
"""

import os
import pickle
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Type alias for better type checking
SentenceTransformerType = SentenceTransformer


class EmbeddingGenerator:
    """
    Unified interface for generating embeddings from multiple models.
    
    Supports caching, batch processing, and easy model switching.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache embeddings
            device: Device to use for computation ('cpu', 'cuda', etc.)
            normalize: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.normalize = normalize
        self.model: Optional[SentenceTransformerType] = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Loaded model: {self.model_name} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def _get_cache_path(self, texts: List[str]) -> Path:
        """Generate cache file path based on model and text content."""
        # Create a hash of the texts for cache identification
        import hashlib
        text_hash = hashlib.md5(str(sorted(texts)).encode()).hexdigest()[:16]
        return self.cache_dir / f"{self.model_name}_{text_hash}.pkl"
    
    def generate(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: Single text string or list of texts
            use_cache: Whether to use cached embeddings if available
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(texts)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data['model_name'] == self.model_name:
                        print(f"Using cached embeddings from {cache_path}")
                        return cached_data['embeddings']
        
        # Generate embeddings
        embeddings = self._generate_embeddings(
            texts, batch_size=batch_size, show_progress=show_progress
        )
        
        # Cache results
        if use_cache:
            cache_path = self._get_cache_path(texts)
            cache_data = {
                'model_name': self.model_name,
                'embeddings': embeddings,
                'texts': texts
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        return embeddings
    
    def _generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Internal method to generate embeddings."""
        if self.model is None:
            self._load_model()
            
        embeddings = []
        
        # Process in batches
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Generating embeddings with {self.model_name}",
            disable=not show_progress
        ):
            batch_texts = texts[i:i + batch_size]
            assert self.model is not None  # Type assertion for linter
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def switch_model(self, model_name: str):
        """Switch to a different model."""
        self.model_name = model_name
        self._load_model()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by the current model."""
        if self.model is None:
            self._load_model()
        assert self.model is not None  # Type assertion for linter
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Could not determine embedding dimension for model {self.model_name}")
        return dimension
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Cleared cache directory: {self.cache_dir}") 