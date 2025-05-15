"""
Base Causal Algorithm Module

This module provides the base class for causal algorithms.
"""
import logging
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class BaseCausalAlgorithm:
    """
    Base class for all causal algorithms.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base causal algorithm.
        
        Args:
            config: Configuration parameters for the algorithm
        """
        self.config = config or {}
    
    def discover_causal_relationships(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Discover causal relationships in the data.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            A directed graph representing causal relationships
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def visualize_graph(self, graph: nx.DiGraph, output_path: Optional[str] = None) -> None:
        """
        Visualize the causal graph.
        
        Args:
            graph: Directed graph representing causal relationships
            output_path: Path to save the visualization (if None, display interactively)
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Use spring layout for node positioning
            pos = nx.spring_layout(graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue')
            
            # Draw edges with weights as labels
            edge_weights = nx.get_edge_attributes(graph, 'weight')
            nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights)
            
            # Draw node labels
            nx.draw_networkx_labels(graph, pos, font_size=10)
            
            plt.title("Causal Graph")
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Graph visualization saved to {output_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib is required for graph visualization")
    
    def export_graph(self, graph: nx.DiGraph, format: str = 'json') -> Dict[str, Any]:
        """
        Export the causal graph to a specified format.
        
        Args:
            graph: Directed graph representing causal relationships
            format: Export format ('json', 'adjacency_list', 'adjacency_matrix')
            
        Returns:
            Dictionary containing the exported graph
        """
        if format == 'json':
            # Convert to node-link format
            import json
            data = nx.node_link_data(graph)
            return data
        
        elif format == 'adjacency_list':
            # Convert to adjacency list
            adj_list = {}
            for node in graph.nodes():
                adj_list[node] = []
                for neighbor in graph.neighbors(node):
                    weight = graph[node][neighbor].get('weight', 1.0)
                    adj_list[node].append({'target': neighbor, 'weight': weight})
            return adj_list
        
        elif format == 'adjacency_matrix':
            # Convert to adjacency matrix
            import numpy as np
            nodes = list(graph.nodes())
            n = len(nodes)
            matrix = np.zeros((n, n))
            
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if graph.has_edge(source, target):
                        matrix[i, j] = graph[source][target].get('weight', 1.0)
            
            return {
                'nodes': nodes,
                'matrix': matrix.tolist()
            }
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_graph(self, data: Dict[str, Any], format: str = 'json') -> nx.DiGraph:
        """
        Import a causal graph from a specified format.
        
        Args:
            data: Dictionary containing the graph data
            format: Import format ('json', 'adjacency_list', 'adjacency_matrix')
            
        Returns:
            Directed graph representing causal relationships
        """
        if format == 'json':
            # Convert from node-link format
            graph = nx.node_link_graph(data)
            return graph
        
        elif format == 'adjacency_list':
            # Convert from adjacency list
            graph = nx.DiGraph()
            
            for source, targets in data.items():
                for target_info in targets:
                    target = target_info['target']
                    weight = target_info.get('weight', 1.0)
                    graph.add_edge(source, target, weight=weight)
            
            return graph
        
        elif format == 'adjacency_matrix':
            # Convert from adjacency matrix
            import numpy as np
            nodes = data['nodes']
            matrix = np.array(data['matrix'])
            
            graph = nx.DiGraph()
            
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if matrix[i, j] != 0:
                        graph.add_edge(source, target, weight=matrix[i, j])
            
            return graph
        
        else:
            raise ValueError(f"Unsupported import format: {format}")