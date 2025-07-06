# embedding-mapper

A modular, reusable, expandable environment for prototyping and iterating on core components of semantic representation and reasoning with text.

## 🎯 Project Purpose

**embedding-mapper** is a living technical system designed to master the tools, abstractions, and workflows that support symbolic intelligence through vector-based language modeling.

This is not a product, nor a single demo notebook. It's a research environment where you can:

- **Experiment** with different embedding models and their semantic properties
- **Discover** structure in high-dimensional spaces through clustering and projection
- **Visualize** and explore semantic relationships interactively
- **Build** reusable components for symbolic reasoning systems
- **Iterate** on the core abstractions of vector-based NLP

## 🧱 Core Capabilities

### ✅ Representation
- **Multiple Models**: MiniLM, MPNet, multilingual, custom fine-tuned models
- **Batch Processing**: Efficient generation with caching and progress tracking
- **Model Switching**: Seamless comparison across different embedding spaces

### ✅ Projection
- **UMAP**: Preserves both local and global structure
- **t-SNE**: Focuses on local structure preservation
- **PCA**: Linear dimensionality reduction
- **Caching**: Reproducible results with parameter tracking

### ✅ Clustering
- **K-Means**: Centroid-based clustering
- **DBSCAN**: Density-based clustering
- **HDBSCAN**: Hierarchical density-based clustering
- **Analysis**: Cluster centers, sizes, and relationships

### ✅ Visualization
- **Interactive Plots**: Plotly-based exploration with hover information
- **Cluster Overlays**: Color-mapped themes and centroid visualization
- **Comparison Tools**: Side-by-side analysis of different methods
- **Export**: HTML files for sharing and documentation

### ✅ Search & Similarity
- **FAISS**: Fast similarity search
- **Annoy**: Approximate nearest neighbors
- **ScaNN**: Scalable nearest neighbors
- **Cosine Similarity**: Standard vector similarity metrics

### ✅ Abstractions
- **Modular Design**: Clean interfaces, decoupled components
- **Config-Driven**: Reproducible experiments with parameter tracking
- **Caching**: Efficient reuse of expensive computations
- **Testing**: Unit tests for core functionality

## 🏗️ Project Structure

```
embedding-mapper/
├── notebooks/                    # Research notebooks
│   ├── 01_embedding_exploration.ipynb
│   ├── 02_clustering_tests.ipynb
│   └── 03_visualization_sandbox.ipynb
├── src/                         # Core modules
│   ├── embeddings/              # Embedding generation
│   │   ├── generator.py         # Main embedding interface
│   │   └── models.py           # Model configurations
│   ├── projection/              # Dimensionality reduction
│   │   ├── reducer.py          # Base projection interface
│   │   ├── umap_projector.py   # UMAP implementation
│   │   ├── tsne_projector.py   # t-SNE implementation
│   │   └── pca_projector.py    # PCA implementation
│   ├── clustering/              # Structure discovery
│   │   ├── clusterer.py        # Base clustering interface
│   │   ├── kmeans_clusterer.py # K-means implementation
│   │   ├── dbscan_clusterer.py # DBSCAN implementation
│   │   └── hdbscan_clusterer.py # HDBSCAN implementation
│   ├── visualization/           # Interactive visualization
│   │   ├── visualizer.py       # Base visualization interface
│   │   ├── interactive_visualizer.py # Plotly implementation
│   │   └── scatter_visualizer.py # Static plots
│   ├── search/                  # Vector search (future)
│   ├── labeling/                # Semantic labeling (future)
│   └── utils/                   # Common utilities
│       ├── text_utils.py       # Text processing
│       ├── data_utils.py       # Data validation
│       └── config_utils.py     # Configuration management
├── data/                        # Data storage
├── outputs/                     # Generated visualizations
├── cache/                       # Cached computations
├── tests/                       # Unit tests
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd embedding-mapper

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Import core modules
from src.embeddings import EmbeddingGenerator
from src.projection import UMAPProjector
from src.visualization import InteractiveVisualizer

# Generate embeddings
generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
embeddings = generator.generate(["Your text here", "Another text"])

# Project to 2D
projector = UMAPProjector(n_components=2)
projection = projector.fit_transform(embeddings)

# Visualize interactively
visualizer = InteractiveVisualizer()
fig = visualizer.create_scatter_plot(projection)
fig.show()
```

## 🔍 Design Principles

### Modularity
Every component is designed to be:
- **Decoupled**: Minimal dependencies between modules
- **Swappable**: Easy to replace implementations
- **Testable**: Clear interfaces for unit testing

### Reproducibility
- **Caching**: Expensive computations are cached and reused
- **Parameter Tracking**: All experiments track their parameters
- **Version Control**: Code and configurations are versioned

### Extensibility
- **Plugin Architecture**: Easy to add new models, algorithms, visualizations
- **Configuration-Driven**: Experiments defined in config files
- **Clean Interfaces**: Abstract base classes for all major components

## 🧠 Research Focus

This system is designed for researchers and engineers who want to:

1. **Understand Semantic Structure**: How do different models capture meaning?
2. **Discover Patterns**: What clusters and relationships emerge in embedding spaces?
3. **Build Symbolic Tools**: How can we extract symbolic knowledge from vectors?
4. **Iterate on Abstractions**: What are the right interfaces for semantic reasoning?

## 🎯 Your Edge

The goal is to become the kind of ML/NLP engineer who can:
- **Build tools that reveal meaning** in high-dimensional spaces
- **Structure ambiguity** through clustering and visualization
- **Enable symbolic inference** through vector geometry
- **Design systems** that support both technical depth and symbolic clarity

## 📚 Next Steps

1. **Start with notebooks**: Explore the existing notebooks to understand the system
2. **Experiment with models**: Try different embedding models on your data
3. **Extend the system**: Add new clustering algorithms, visualization techniques
4. **Build abstractions**: Develop new interfaces for semantic reasoning
5. **Share insights**: Document your discoveries and contribute back

## 🤝 Contributing

This is a research environment - contributions should focus on:
- **New algorithms**: Clustering, projection, or search methods
- **Better abstractions**: Cleaner interfaces and modular design
- **Documentation**: Notebooks, examples, and insights
- **Testing**: Unit tests and validation for core components

## 📄 License

This project is for research and educational purposes. Please respect the intellectual property of the underlying models and libraries.

---

**Remember**: This is about mastering the tools of symbolic intelligence through vector-based language modeling. Focus on the abstractions, the interfaces, and the workflows that reveal meaning in text.