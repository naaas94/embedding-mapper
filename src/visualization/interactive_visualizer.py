"""
Interactive visualization using Plotly.

Provides interactive visualizations for exploring embeddings and clusters
with hover information, zooming, and selection capabilities.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .visualizer import Visualizer


class InteractiveVisualizer(Visualizer):
    """
    Interactive visualization using Plotly.
    
    Provides rich interactive visualizations for exploring embeddings,
    clusters, and relationships in the data.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the interactive visualizer.
        
        Args:
            theme: Plotly theme to use
        """
        super().__init__()
        self.theme = theme
    
    def create_scatter_plot(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
        title: str = "Embedding Visualization",
        x_label: str = "Dimension 1",
        y_label: str = "Dimension 2",
        color_map: Optional[Dict[int, str]] = None,
        hover_data: Optional[Dict[str, List]] = None
    ) -> go.Figure:
        """
        Create an interactive scatter plot.
        
        Args:
            embeddings: 2D embeddings to visualize
            labels: Cluster labels for coloring
            texts: Text labels for hover information
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            color_map: Custom color mapping for clusters
            hover_data: Additional data for hover tooltips
            
        Returns:
            Plotly figure object
        """
        # Prepare data
        x, y = embeddings[:, 0], embeddings[:, 1]
        
        # Create hover text
        hover_text = []
        if texts is not None:
            for i, text in enumerate(texts):
                hover_info = f"Text: {text[:100]}{'...' if len(text) > 100 else ''}"
                if labels is not None:
                    hover_info += f"<br>Cluster: {labels[i]}"
                if hover_data:
                    for key, values in hover_data.items():
                        if i < len(values):
                            hover_info += f"<br>{key}: {values[i]}"
                hover_text.append(hover_info)
        
        # Create scatter plot
        if labels is not None:
            # Color by clusters
            unique_labels = np.unique(labels)
            fig = go.Figure()
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    color = "gray"
                    name = "Noise"
                else:
                    color = color_map.get(label, f"Cluster {label}") if color_map else None
                    name = f"Cluster {label}"
                
                mask = labels == label
                fig.add_trace(go.Scatter(
                    x=x[mask],
                    y=y[mask],
                    mode='markers',
                    name=name,
                    marker=dict(
                        color=color,
                        size=8,
                        opacity=0.7
                    ),
                    text=hover_text[mask] if hover_text else None,
                    hoverinfo='text'
                ))
        else:
            # Single color
            fig = go.Figure(data=go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    color='blue',
                    size=8,
                    opacity=0.7
                ),
                text=hover_text if hover_text else None,
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def create_cluster_comparison(
        self,
        embeddings: np.ndarray,
        cluster_results: Dict[str, np.ndarray],
        title: str = "Cluster Comparison",
        max_clusters: int = 4
    ) -> go.Figure:
        """
        Create a comparison of different clustering results.
        
        Args:
            embeddings: 2D embeddings
            cluster_results: Dictionary mapping method names to cluster labels
            title: Plot title
            max_clusters: Maximum number of clustering methods to show
            
        Returns:
            Plotly figure with subplots
        """
        methods = list(cluster_results.keys())[:max_clusters]
        n_methods = len(methods)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=n_methods,
            subplot_titles=methods,
            specs=[[{"secondary_y": False}] * n_methods]
        )
        
        for i, method in enumerate(methods):
            labels = cluster_results[method]
            x, y = embeddings[:, 0], embeddings[:, 1]
            
            # Add scatter plot for this method
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                fig.add_trace(
                    go.Scatter(
                        x=x[mask],
                        y=y[mask],
                        mode='markers',
                        name=f"{method} - Cluster {label}",
                        marker=dict(size=6, opacity=0.7),
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_embedding_explorer(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        title: str = "Embedding Explorer"
    ) -> go.Figure:
        """
        Create an interactive embedding explorer with text search.
        
        Args:
            embeddings: 2D embeddings
            texts: Text labels
            labels: Cluster labels
            title: Plot title
            
        Returns:
            Interactive Plotly figure
        """
        fig = self.create_scatter_plot(
            embeddings=embeddings,
            labels=labels,
            texts=texts,
            title=title
        )
        
        # Add search functionality
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.1,
                    y=1.1,
                    showactive=False,
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="Show All",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False] * len(fig.data)}],
                            label="Hide All",
                            method="update"
                        )
                    ])
                )
            ]
        )
        
        return fig
    
    def create_cluster_analysis(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_centers: Optional[np.ndarray] = None,
        title: str = "Cluster Analysis"
    ) -> go.Figure:
        """
        Create a detailed cluster analysis visualization.
        
        Args:
            embeddings: 2D embeddings
            labels: Cluster labels
            cluster_centers: Cluster center coordinates
            title: Plot title
            
        Returns:
            Plotly figure with cluster analysis
        """
        fig = self.create_scatter_plot(
            embeddings=embeddings,
            labels=labels,
            title=title
        )
        
        # Add cluster centers if available
        if cluster_centers is not None:
            fig.add_trace(go.Scatter(
                x=cluster_centers[:, 0],
                y=cluster_centers[:, 1],
                mode='markers',
                name='Cluster Centers',
                marker=dict(
                    symbol='x',
                    size=15,
                    color='red',
                    line=dict(width=2)
                )
            ))
        
        # Add cluster statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_stats = []
        for label, count in zip(unique_labels, counts):
            if label != -1:  # Skip noise
                cluster_stats.append(f"Cluster {label}: {count} points")
        
        if cluster_stats:
            fig.add_annotation(
                text="<br>".join(cluster_stats),
                xref="paper", yref="paper",
                x=1.02, y=1,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        return fig
    
    def save_interactive_html(
        self,
        fig: go.Figure,
        filename: str,
        include_plotlyjs: bool = True
    ):
        """
        Save the figure as an interactive HTML file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            include_plotlyjs: Whether to include Plotly.js in the file
        """
        fig.write_html(
            filename,
            include_plotlyjs=include_plotlyjs,
            full_html=True
        )
        print(f"Saved interactive visualization to {filename}") 