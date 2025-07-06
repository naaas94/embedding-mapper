"""
Model management utilities for embedding generation.

Provides functions to list available models, load models, and manage model configurations.
"""

from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer


# Predefined model configurations
AVAILABLE_MODELS = {
    # Fast, general-purpose models
    "all-MiniLM-L6-v2": {
        "description": "Fast, general-purpose model (384d)",
        "dimension": 384,
        "speed": "fast",
        "quality": "good"
    },
    "all-MiniLM-L12-v2": {
        "description": "Balanced speed/quality model (384d)",
        "dimension": 384,
        "speed": "medium",
        "quality": "good"
    },
    
    # High-quality models
    "all-mpnet-base-v2": {
        "description": "High-quality general-purpose model (768d)",
        "dimension": 768,
        "speed": "medium",
        "quality": "excellent"
    },
    "all-mpnet-base-v2-sentence": {
        "description": "Sentence-optimized MPNet model (768d)",
        "dimension": 768,
        "speed": "medium",
        "quality": "excellent"
    },
    
    # Multilingual models
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "description": "Multilingual model supporting 50+ languages (384d)",
        "dimension": 384,
        "speed": "medium",
        "quality": "good",
        "multilingual": True
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "description": "High-quality multilingual model (768d)",
        "dimension": 768,
        "speed": "slow",
        "quality": "excellent",
        "multilingual": True
    },
    
    # Specialized models
    "all-distilroberta-v1": {
        "description": "Distilled RoBERTa model (768d)",
        "dimension": 768,
        "speed": "medium",
        "quality": "good"
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "description": "Question-answering optimized model (384d)",
        "dimension": 384,
        "speed": "fast",
        "quality": "good",
        "specialized": "qa"
    }
}


def get_available_models() -> Dict[str, Dict]:
    """
    Get information about all available models.
    
    Returns:
        Dictionary mapping model names to their configurations
    """
    return AVAILABLE_MODELS.copy()


def list_models_by_category() -> Dict[str, List[str]]:
    """
    List models organized by category.
    
    Returns:
        Dictionary with categories as keys and model lists as values
    """
    categories = {
        "fast": [],
        "balanced": [],
        "high_quality": [],
        "multilingual": [],
        "specialized": []
    }
    
    for model_name, config in AVAILABLE_MODELS.items():
        if config.get("multilingual"):
            categories["multilingual"].append(model_name)
        elif config.get("specialized"):
            categories["specialized"].append(model_name)
        elif config["speed"] == "fast":
            categories["fast"].append(model_name)
        elif config["quality"] == "excellent":
            categories["high_quality"].append(model_name)
        else:
            categories["balanced"].append(model_name)
    
    return categories


def get_model_info(model_name: str) -> Optional[Dict]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary or None if not found
    """
    return AVAILABLE_MODELS.get(model_name)


def load_model(model_name: str, device: Optional[str] = None) -> SentenceTransformer:
    """
    Load a sentence transformer model.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on ('cpu', 'cuda', etc.)
        
    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        ValueError: If model name is not recognized
        RuntimeError: If model loading fails
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    try:
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def get_recommended_model(
    use_case: str = "general",
    speed_priority: bool = False,
    multilingual: bool = False
) -> str:
    """
    Get a recommended model based on requirements.
    
    Args:
        use_case: Intended use case ('general', 'qa', 'multilingual')
        speed_priority: Whether to prioritize speed over quality
        multilingual: Whether multilingual support is needed
        
    Returns:
        Recommended model name
    """
    if multilingual:
        if speed_priority:
            return "paraphrase-multilingual-MiniLM-L12-v2"
        else:
            return "paraphrase-multilingual-mpnet-base-v2"
    
    if use_case == "qa":
        return "multi-qa-MiniLM-L6-cos-v1"
    
    if speed_priority:
        return "all-MiniLM-L6-v2"
    else:
        return "all-mpnet-base-v2"


def validate_model_name(model_name: str) -> bool:
    """
    Check if a model name is valid.
    
    Args:
        model_name: Name to validate
        
    Returns:
        True if valid, False otherwise
    """
    return model_name in AVAILABLE_MODELS 