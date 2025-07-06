"""
Base visualization interface.

Provides common functionality for all visualization types
including color schemes, themes, and utility functions.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np


class Visualizer(ABC):
    """
    Abstract base class for visualization components.
    
    Provides common functionality for creating visualizations
    of embeddings, clusters, and relationships.
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        self.color_schemes = {
            'default': self.default_colors,
            'qualitative': [
                '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                '#ffff33', '#a65628', '#f781bf', '#999999'
            ],
            'diverging': [
                '#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598',
                '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'
            ]
        }
    
    def get_color_map(self, n_colors: int, scheme: str = 'default') -> List[str]:
        """
        Get a color map with the specified number of colors.
        
        Args:
            n_colors: Number of colors needed
            scheme: Color scheme to use
            
        Returns:
            List of color codes
        """
        colors = self.color_schemes.get(scheme, self.default_colors)
        
        if n_colors <= len(colors):
            return colors[:n_colors]
        else:
            # Extend the color palette if needed
            extended_colors = colors.copy()
            while len(extended_colors) < n_colors:
                extended_colors.extend(colors)
            return extended_colors[:n_colors]
    
    def create_cluster_color_map(
        self,
        labels: np.ndarray,
        scheme: str = 'default'
    ) -> Dict[int, str]:
        """
        Create a color mapping for cluster labels.
        
        Args:
            labels: Cluster labels
            scheme: Color scheme to use
            
        Returns:
            Dictionary mapping cluster labels to colors
        """
        unique_labels = np.unique(labels)
        colors = self.get_color_map(len(unique_labels), scheme)
        
        color_map = {}
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                color_map[label] = '#cccccc'  # Light gray
            else:
                color_map[label] = colors[i]
        
        return color_map
    
    @abstractmethod
    def create_scatter_plot(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
        title: str = "Embedding Visualization",
        **kwargs
    ):
        """
        Create a scatter plot visualization.
        
        Args:
            embeddings: 2D embeddings to visualize
            labels: Cluster labels for coloring
            texts: Text labels for annotations
            title: Plot title
            **kwargs: Additional visualization parameters
        """
        pass
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate that embeddings are suitable for visualization.
        
        Args:
            embeddings: Embeddings to validate
            
        Returns:
            True if valid, False otherwise
        """
        if embeddings is None or len(embeddings) == 0:
            return False
        
        if embeddings.ndim != 2:
            return False
        
        if embeddings.shape[1] < 2:
            return False
        
        return True
    
    def prepare_embeddings_for_plotting(
        self,
        embeddings: np.ndarray,
        max_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Prepare embeddings for plotting, including subsampling if needed.
        
        Args:
            embeddings: Input embeddings
            max_points: Maximum number of points to plot
            
        Returns:
            Prepared embeddings
        """
        if not self.validate_embeddings(embeddings):
            raise ValueError("Invalid embeddings for visualization")
        
        if max_points is not None and len(embeddings) > max_points:
            # Randomly sample points
            indices = np.random.choice(
                len(embeddings),
                size=max_points,
                replace=False
            )
            return embeddings[indices]
        
        return embeddings
    
    def get_plot_dimensions(self, n_plots: int) -> tuple:
        """
        Calculate optimal subplot dimensions.
        
        Args:
            n_plots: Number of plots needed
            
        Returns:
            Tuple of (rows, cols)
        """
        if n_plots <= 1:
            return (1, 1)
        elif n_plots <= 2:
            return (1, 2)
        elif n_plots <= 4:
            return (2, 2)
        elif n_plots <= 6:
            return (2, 3)
        elif n_plots <= 9:
            return (3, 3)
        else:
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
            return (rows, cols) 