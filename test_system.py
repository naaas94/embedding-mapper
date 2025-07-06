#!/usr/bin/env python3
"""
Simple test script to verify the embedding-mapper system works.

This script tests basic imports and functionality without requiring
heavy dependencies like sentence transformers.
"""

import sys
import os

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core package
        from src import __version__, __author__
        print(f"✓ Core package: {__version__} by {__author__}")
        
        # Test embeddings module
        from src.embeddings import get_available_models, get_recommended_model
        print("✓ Embeddings module imported")
        
        # Test projection module
        from src.projection import UMAPProjector
        print("✓ Projection module imported")
        
        # Test clustering module
        from src.clustering import Clusterer
        print("✓ Clustering module imported")
        
        # Test visualization module
        from src.visualization import InteractiveVisualizer
        print("✓ Visualization module imported")
        
        # Test utils module
        from src.utils import clean_text, tokenize_text
        print("✓ Utils module imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_model_config():
    """Test model configuration functionality."""
    print("\nTesting model configuration...")
    
    try:
        from src.embeddings import get_available_models, get_recommended_model
        
        # Test getting available models
        models = get_available_models()
        print(f"✓ Found {len(models)} available models")
        
        # Test model recommendation
        recommended = get_recommended_model(use_case="general", speed_priority=True)
        print(f"✓ Recommended model: {recommended}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model config error: {e}")
        return False

def test_text_utils():
    """Test text processing utilities."""
    print("\nTesting text utilities...")
    
    try:
        from src.utils import clean_text, tokenize_text, extract_keywords
        
        # Test text cleaning
        test_text = "  This is a TEST text with   extra spaces!  "
        cleaned = clean_text(test_text)
        print(f"✓ Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test tokenization
        tokens = tokenize_text(cleaned)
        print(f"✓ Tokenization: {len(tokens)} tokens")
        
        # Test keyword extraction
        keywords = extract_keywords(cleaned, top_k=3)
        print(f"✓ Keyword extraction: {keywords}")
        
        return True
        
    except Exception as e:
        print(f"✗ Text utils error: {e}")
        return False

def test_projector_creation():
    """Test projector creation (without fitting)."""
    print("\nTesting projector creation...")
    
    try:
        from src.projection import UMAPProjector
        
        # Test creating projector
        projector = UMAPProjector(n_components=2, n_neighbors=5)
        print(f"✓ Created UMAP projector with {projector.n_components} components")
        
        return True
        
    except Exception as e:
        print(f"✗ Projector creation error: {e}")
        return False

def test_visualizer_creation():
    """Test visualizer creation."""
    print("\nTesting visualizer creation...")
    
    try:
        from src.visualization import InteractiveVisualizer
        
        # Test creating visualizer
        visualizer = InteractiveVisualizer()
        print(f"✓ Created interactive visualizer with theme: {visualizer.theme}")
        
        # Test color map generation
        import numpy as np
        test_labels = np.array([0, 1, 2, 0, 1])
        color_map = visualizer.create_cluster_color_map(test_labels)
        print(f"✓ Generated color map for {len(color_map)} clusters")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualizer creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("embedding-mapper System Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_config,
        test_text_utils,
        test_projector_creation,
        test_visualizer_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the notebooks in the notebooks/ directory")
        print("3. Start experimenting with your own data!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 