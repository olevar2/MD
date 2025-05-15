import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from analysis_coordinator_service.utils.resilience import with_retry, with_circuit_breaker

logger = logging.getLogger(__name__)

class CausalAnalysisAdapter:
    """
    Adapter for communicating with the causal analysis service.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def _make_request(self, method: str, url: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the causal analysis service.
        """
        async with aiohttp.ClientSession() as session:
            if method == "POST":
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        raise Exception(f"Causal graph request failed: {error_text}")
                    
                    return await response.json()
            elif method == "GET":
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        raise Exception(f"Causal graph request failed: {error_text}")
                    
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def generate_causal_graph(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        variables: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a causal graph from market data.
        """
        if variables is None:
            variables = ["price", "volume", "volatility"]
            
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/causal/causal-graph"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "variables": variables,
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending causal graph request to {url}")
        
        return await self._make_request("POST", url, payload)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_causal_impact(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        intervention_date: datetime = None,
        variables: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze causal impact of an intervention.
        """
        if variables is None:
            variables = ["price", "volume", "volatility"]
            
        if parameters is None:
            parameters = {}
            
        if intervention_date is None:
            raise ValueError("Intervention date is required for causal impact analysis")
            
        url = f"{self.base_url}/api/v1/causal/causal-impact"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "intervention_date": intervention_date.isoformat(),
            "variables": variables,
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending causal impact request to {url}")
        
        return await self._make_request("POST", url, payload)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_granger_causality(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        variables: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform Granger causality test.
        """
        if variables is None:
            variables = ["price", "volume", "volatility"]
            
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/causal/granger-causality"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "variables": variables,
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending Granger causality request to {url}")
        
        return await self._make_request("POST", url, payload)
