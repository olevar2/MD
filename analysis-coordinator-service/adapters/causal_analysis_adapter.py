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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Causal graph request failed with status {response.status}: {error_text}")
                    raise Exception(f"Causal graph request failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Causal graph request successful")
                return result
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def calculate_intervention_effect(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        intervention: Dict[str, Any] = None,
        target: str = "price",
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate the effect of an intervention on the causal graph.
        """
        if intervention is None:
            intervention = {}
            
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/causal/intervention-effect"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "intervention": intervention,
            "target": target,
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending intervention effect request to {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Intervention effect request failed with status {response.status}: {error_text}")
                    raise Exception(f"Intervention effect request failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Intervention effect request successful")
                return result
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def generate_counterfactual_scenario(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        scenario: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual scenario based on the causal graph.
        """
        if scenario is None:
            scenario = {}
            
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/causal/counterfactual-scenario"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "scenario": scenario,
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending counterfactual scenario request to {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Counterfactual scenario request failed with status {response.status}: {error_text}")
                    raise Exception(f"Counterfactual scenario request failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Counterfactual scenario request successful")
                return result