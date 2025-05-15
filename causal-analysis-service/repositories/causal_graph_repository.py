from typing import Dict, Optional, Any
import json
import os
from datetime import datetime

from causal_analysis_service.models.causal_models import CausalGraphResponse

class CausalGraphRepository:
    """
    Repository for storing and retrieving causal graphs.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the repository with a storage path.
        """
        self.storage_path = storage_path or os.environ.get("CAUSAL_GRAPH_STORAGE_PATH", "./data/causal_graphs")
        os.makedirs(self.storage_path, exist_ok=True)
        self.graphs: Dict[str, CausalGraphResponse] = {}
    
    def save_graph(self, graph_id: str, graph_data: CausalGraphResponse) -> None:
        """
        Save a causal graph to the repository.
        """
        self.graphs[graph_id] = graph_data
        
        # Also persist to disk for durability
        file_path = os.path.join(self.storage_path, f"{graph_id}.json")
        with open(file_path, "w") as f:
            # Convert to dict for JSON serialization
            graph_dict = graph_data.dict()
            # Convert datetime to string
            graph_dict["timestamp"] = graph_dict["timestamp"].isoformat()
            json.dump(graph_dict, f, indent=2)
    
    def get_graph(self, graph_id: str) -> Optional[CausalGraphResponse]:
        """
        Retrieve a causal graph from the repository.
        """
        # Try to get from in-memory cache first
        if graph_id in self.graphs:
            return self.graphs[graph_id]
        
        # If not in memory, try to load from disk
        file_path = os.path.join(self.storage_path, f"{graph_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                graph_dict = json.load(f)
                # Convert string back to datetime
                graph_dict["timestamp"] = datetime.fromisoformat(graph_dict["timestamp"])
                graph_data = CausalGraphResponse(**graph_dict)
                # Cache in memory for future use
                self.graphs[graph_id] = graph_data
                return graph_data
        
        return None
    
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a causal graph from the repository.
        """
        if graph_id in self.graphs:
            del self.graphs[graph_id]
        
        file_path = os.path.join(self.storage_path, f"{graph_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        
        return False
    
    def list_graphs(self) -> Dict[str, Any]:
        """
        List all causal graphs in the repository.
        """
        result = {}
        
        # List files in the storage directory
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                graph_id = filename[:-5]  # Remove .json extension
                file_path = os.path.join(self.storage_path, filename)
                with open(file_path, "r") as f:
                    graph_dict = json.load(f)
                    # Include basic metadata in the listing
                    result[graph_id] = {
                        "algorithm_used": graph_dict["algorithm_used"],
                        "timestamp": graph_dict["timestamp"],
                        "node_count": len(graph_dict["node_metadata"])
                    }
        
        return result