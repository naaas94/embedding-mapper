"""
Data validation and processing utilities.

Provides functions for validating embeddings, normalizing data,
and ensuring data quality for the embedding-mapper system.
"""

import numpy as np
from typing import Optional, Union, List


def validate_embeddings(
    embeddings: np.ndarray,
    min_samples: int = 1,
    min_dimensions: int = 1,
    check_finite: bool = True
) -> bool:
    """
    Validate embedding array.
    
    Args:
        embeddings: Embedding array to validate
        min_samples: Minimum number of samples required
        min_dimensions: Minimum number of dimensions required
        check_finite: Whether to check for finite values
        
    Returns:
        True if valid, False otherwise
    """
    if embeddings is None:
        return False
    
    if not isinstance(embeddings, np.ndarray):
        return False
    
    if embeddings.ndim != 2:
        return False
    
    if embeddings.shape[0] < min_samples:
        return False
    
    if embeddings.shape[1] < min_dimensions:
        return False
    
    if check_finite and not np.all(np.isfinite(embeddings)):
        return False
    
    return True


def normalize_embeddings(
    embeddings: np.ndarray,
    method: str = 'l2',
    axis: int = 1
) -> np.ndarray:
    """
    Normalize embeddings using various methods.
    
    Args:
        embeddings: Input embeddings
        method: Normalization method ('l2', 'l1', 'max', 'zscore')
        axis: Axis along which to normalize
        
    Returns:
        Normalized embeddings
    """
    if not validate_embeddings(embeddings):
        raise ValueError("Invalid embeddings for normalization")
    
    if method == 'l2':
        # L2 normalization (unit norm)
        norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    elif method == 'l1':
        # L1 normalization
        norms = np.sum(np.abs(embeddings), axis=axis, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    elif method == 'max':
        # Max normalization
        max_vals = np.max(np.abs(embeddings), axis=axis, keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)
        return embeddings / max_vals
    
    elif method == 'zscore':
        # Z-score normalization
        mean = np.mean(embeddings, axis=axis, keepdims=True)
        std = np.std(embeddings, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (embeddings - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_similarity_matrix(
    embeddings: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute similarity matrix between embeddings.
    
    Args:
        embeddings: Input embeddings
        metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        Similarity matrix
    """
    if not validate_embeddings(embeddings):
        raise ValueError("Invalid embeddings for similarity computation")
    
    if metric == 'cosine':
        # Normalize embeddings for cosine similarity
        normalized = normalize_embeddings(embeddings, method='l2')
        return np.dot(normalized, normalized.T)
    
    elif metric == 'euclidean':
        # Convert to similarity (1 / (1 + distance))
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings)
        return 1 / (1 + distances)
    
    elif metric == 'dot':
        # Dot product similarity
        return np.dot(embeddings, embeddings.T)
    
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def subsample_embeddings(
    embeddings: np.ndarray,
    n_samples: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Subsample embeddings randomly.
    
    Args:
        embeddings: Input embeddings
        n_samples: Number of samples to keep
        random_state: Random seed for reproducibility
        
    Returns:
        Subsampled embeddings
    """
    if not validate_embeddings(embeddings):
        raise ValueError("Invalid embeddings for subsampling")
    
    if n_samples >= embeddings.shape[0]:
        return embeddings
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.choice(
        embeddings.shape[0],
        size=n_samples,
        replace=False
    )
    
    return embeddings[indices]


def get_embedding_statistics(embeddings: np.ndarray) -> dict:
    """
    Get statistics about embeddings.
    
    Args:
        embeddings: Input embeddings
        
    Returns:
        Dictionary with embedding statistics
    """
    if not validate_embeddings(embeddings):
        raise ValueError("Invalid embeddings for statistics")
    
    stats = {
        'n_samples': embeddings.shape[0],
        'n_dimensions': embeddings.shape[1],
        'mean': np.mean(embeddings),
        'std': np.std(embeddings),
        'min': np.min(embeddings),
        'max': np.max(embeddings),
        'sparsity': np.mean(embeddings == 0),
        'finite_ratio': np.mean(np.isfinite(embeddings))
    }
    
    # Add norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    stats.update({
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'min_norm': np.min(norms),
        'max_norm': np.max(norms)
    })
    
    return stats


def validate_texts(
    texts: Union[str, List[str]],
    min_length: int = 1,
    max_length: Optional[int] = None
) -> bool:
    """
    Validate text data.
    
    Args:
        texts: Text or list of texts to validate
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        True if valid, False otherwise
    """
    if isinstance(texts, str):
        texts = [texts]
    
    if not isinstance(texts, list):
        return False
    
    if len(texts) == 0:
        return False
    
    for text in texts:
        if not isinstance(text, str):
            return False
        
        if len(text) < min_length:
            return False
        
        if max_length is not None and len(text) > max_length:
            return False
    
    return True 