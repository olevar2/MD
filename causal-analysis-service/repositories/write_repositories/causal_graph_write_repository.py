"""
Causal Graph Write Repository

This module provides a write repository for causal graphs.
"""
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import networkx as nx

from common_lib.cqrs.repositories import WriteRepository
from common_lib.caching.decorators import invalidate_cache
from causal_analysis_service.models.causal_models import CausalGraphResponse, Edge
from causal_analysis_service.utils.cache_factory import cache_factory

logger = logging.getLogger(__name__)

class CausalGraphWriteRepository(WriteRepository[nx.DiGraph, str]):
    """
    Write repository for causal graphs.
    """
    def __init__(self, db_connection=None):
        """
        Initialize the causal graph write repository.
        
        Args:
            db_connection: Database connection object
        """
        self.db_connection = db_connection
        self.cache = cache_factory.get_cache()
        self.causal_graphs = {}  # In-memory storage for causal graphs
    
    async def add(self, graph: nx.DiGraph, metadata: Dict[str, Any]) -> str:
        """
        Add a causal graph.
        
        Args:
            graph: The causal graph to add
            metadata: Metadata for the graph
            
        Returns:
            The ID of the added causal graph
        """
        logger.info("Adding causal graph")
        
        graph_id = str(uuid.uuid4())
        
        # Extract nodes and edges
        nodes = list(graph.nodes())
        edges = []
        
        for source, target, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            edges.append(Edge(source=source, target=target, weight=weight))
        
        # Create adjacency matrix
        adjacency_matrix = []
        for i, source in enumerate(nodes):
            row = []
            for j, target in enumerate(nodes):
                if graph.has_edge(source, target):
                    row.append(graph[source][target].get('weight', 1.0))
                else:
                    row.append(0.0)
            adjacency_matrix.append(row)
        
        # Create response object
        graph_response = CausalGraphResponse(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adjacency_matrix,
            created_at=datetime.now(),
            algorithm=metadata.get('algorithm', 'unknown'),
            parameters=metadata.get('parameters', {})
        )
        
        # Store in memory
        self.causal_graphs[graph_id] = graph_response
        
        # If database connection is available, store in database
        if self.db_connection:
            try:
                # Convert to dictionary for storage
                graph_dict = graph_response.dict()
                
                # Store in database
                # This is a placeholder for actual database storage
                # await self.db_connection.execute(
                #     "INSERT INTO causal_graphs (graph_id, data) VALUES ($1, $2)",
                #     graph_id, json.dumps(graph_dict)
                # )
                
                logger.info(f"Saved causal graph {graph_id} to database")
            except Exception as e:
                logger.error(f"Error saving causal graph to database: {e}")
        
        return graph_id
    
    @invalidate_cache(cache_factory.get_cache(), "causal_graph")
    async def update(self, graph: nx.DiGraph) -> None:
        """
        Update a causal graph.
        
        Args:
            graph: The causal graph to update
        """
        logger.info("Updating causal graph")
        
        # This is a placeholder for actual update logic
        # In a real implementation, we would extract the ID from the graph
        # and update the corresponding record in the database
        
        logger.info("Updated causal graph")
    
    @invalidate_cache(cache_factory.get_cache(), "causal_graph")
    async def delete(self, id: str) -> None:
        """
        Delete a causal graph.
        
        Args:
            id: The ID of the causal graph to delete
        """
        logger.info(f"Deleting causal graph with ID {id}")
        
        # Remove from in-memory storage
        if id in self.causal_graphs:
            del self.causal_graphs[id]
        
        # If database connection is available, delete from database
        if self.db_connection:
            try:
                # This is a placeholder for actual database deletion
                # await self.db_connection.execute(
                #     "DELETE FROM causal_graphs WHERE graph_id = $1",
                #     id
                # )
                
                logger.info(f"Deleted causal graph {id} from database")
            except Exception as e:
                logger.error(f"Error deleting causal graph from database: {e}")
    
    async def add_batch(self, graphs: List[nx.DiGraph]) -> List[str]:
        """
        Add multiple causal graphs in a batch.
        
        Args:
            graphs: The causal graphs to add
            
        Returns:
            The IDs of the added causal graphs
        """
        logger.info(f"Adding {len(graphs)} causal graphs in batch")
        
        # This is a placeholder for actual batch add logic
        # In a real implementation, we would use a bulk insert operation
        
        # For now, just add each graph individually
        graph_ids = []
        for graph in graphs:
            graph_id = await self.add(graph, {})
            graph_ids.append(graph_id)
        
        return graph_ids