from typing import Dict, List, Any, Optional
import os
import httpx
from common_lib.resilience.decorators import with_standard_resilience

class AnalysisCoordinatorAdapter:
    """
    Adapter for interacting with the analysis coordinator service.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the adapter with the analysis coordinator service URL.
        """
        self.base_url = base_url or os.environ.get("ANALYSIS_COORDINATOR_URL", "http://analysis-coordinator-service:8000")
        self.timeout = int(os.environ.get("ANALYSIS_COORDINATOR_TIMEOUT", "30"))
    
    @with_standard_resilience()
    async def notify_causal_graph_created(self, graph_id: str) -> Dict[str, Any]:
        """
        Notify the analysis coordinator that a new causal graph has been created.
        """
        payload = {
            "graph_id": graph_id,
            "service": "causal-analysis-service"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/v1/notifications/causal-graph", json=payload)
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_analysis_context(self, context_id: str) -> Dict[str, Any]:
        """
        Get analysis context from the analysis coordinator.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/context/{context_id}")
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def update_analysis_status(self, analysis_id: str, status: str, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update the status of an analysis in the analysis coordinator.
        """
        payload = {
            "status": status
        }
        
        if results:
            payload["results"] = results
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(f"{self.base_url}/api/v1/analysis/{analysis_id}/status", json=payload)
            response.raise_for_status()
            return response.json()