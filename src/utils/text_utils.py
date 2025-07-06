"""
Text processing utilities.

Provides functions for cleaning, tokenizing, and analyzing text data
for embedding generation and analysis.
"""

import re
from typing import List, Optional
from collections import Counter


def clean_text(text: str, remove_punctuation: bool = False) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    # Remove punctuation if requested
    if remove_punctuation:
        text = re.sub(r'[^\\w\\s]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize_text(text: str, method: str = 'simple') -> List[str]:
    """
    Tokenize text into words or subwords.
    
    Args:
        text: Input text
        method: Tokenization method ('simple', 'whitespace', 'word')
        
    Returns:
        List of tokens
    """
    if method == 'simple':
        # Simple word tokenization
        tokens = re.findall(r'\\b\\w+\\b', text.lower())
    elif method == 'whitespace':
        # Split on whitespace
        tokens = text.split()
    elif method == 'word':
        # More sophisticated word tokenization
        tokens = re.findall(r'\\b[a-zA-Z]+\\b', text.lower())
    else:
        raise ValueError(f"Unknown tokenization method: {method}")
    
    return tokens


def extract_keywords(
    text: str,
    top_k: int = 10,
    min_length: int = 3,
    stop_words: Optional[List[str]] = None
) -> List[tuple]:
    """
    Extract keywords from text using frequency analysis.
    
    Args:
        text: Input text
        top_k: Number of top keywords to return
        min_length: Minimum word length
        stop_words: List of stop words to exclude
        
    Returns:
        List of (word, frequency) tuples
    """
    # Default stop words
    if stop_words is None:
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    # Clean and tokenize
    cleaned_text = clean_text(text, remove_punctuation=True)
    tokens = tokenize_text(cleaned_text)
    
    # Filter tokens
    filtered_tokens = [
        token for token in tokens
        if len(token) >= min_length and token not in stop_words
    ]
    
    # Count frequencies
    word_counts = Counter(filtered_tokens)
    
    # Return top k keywords
    return word_counts.most_common(top_k)


def get_text_statistics(text: str) -> dict:
    """
    Get basic statistics about text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    # Basic counts
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text))
    
    # Word length statistics
    words = text.split()
    word_lengths = [len(word) for word in words if word]
    
    stats = {
        'characters': char_count,
        'words': word_count,
        'sentences': sentence_count,
        'avg_word_length': sum(word_lengths) / len(word_lengths) if word_lengths else 0,
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0
    }
    
    return stats


def normalize_text_length(text: str, max_length: int = 512) -> str:
    """
    Normalize text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum number of characters
        
    Returns:
        Normalized text
    """
    if len(text) <= max_length:
        return text
    
    # Truncate to max_length
    truncated = text[:max_length]
    
    # Try to break at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If we can break at a reasonable point
        return truncated[:last_space] + '...'
    
    return truncated + '...' 