"""
Embedding generation module.

Provides interfaces for generating embeddings from multiple models:
- Sentence transformers (MiniLM, MPNet, multilingual)
- Custom fine-tuned models
- Batch processing and caching
"""

from .generator import EmbeddingGenerator
from .models import get_available_models, load_model, get_recommended_model

__all__ = ["EmbeddingGenerator", "get_available_models", "load_model", "get_recommended_model"] 