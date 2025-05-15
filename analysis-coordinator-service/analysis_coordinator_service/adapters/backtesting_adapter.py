import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from analysis_coordinator_service.utils.resilience import with_retry, with_circuit_breaker

logger = logging.getLogger(__name__)

class BacktestingAdapter:
    """
    Adapter for communicating with the backtesting service.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def _make_request(self, method: str, url: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the backtesting service.
        """
        async with aiohttp.ClientSession() as session:
            if method == "POST":
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        raise Exception(f"Backtest request failed: {error_text}")
                    
                    return await response.json()
            elif method == "GET":
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        raise Exception(f"Backtest request failed: {error_text}")
                    
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def run_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        strategy_config: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest with the specified configuration.
        """
        if strategy_config is None:
            strategy_config = {}
            
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/backtest/run"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "strategy_config": strategy_config,
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending backtest request to {url}")
        
        return await self._make_request("POST", url, payload)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_backtest_result(self, backtest_id: str) -> Dict[str, Any]:
        """
        Get the result of a previously run backtest.
        """
        url = f"{self.base_url}/api/v1/backtest/{backtest_id}"
        
        logger.info(f"Sending backtest result request to {url}")
        
        return await self._make_request("GET", url)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_available_strategies(self) -> Dict[str, Any]:
        """
        Get a list of available backtesting strategies.
        """
        url = f"{self.base_url}/api/v1/backtest/strategies"
        
        logger.info(f"Sending available strategies request to {url}")
        
        return await self._make_request("GET", url)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get the parameters for a specific backtesting strategy.
        """
        url = f"{self.base_url}/api/v1/backtest/strategies/{strategy_name}/parameters"
        
        logger.info(f"Sending strategy parameters request to {url}")
        
        return await self._make_request("GET", url)
