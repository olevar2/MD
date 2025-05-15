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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Backtest request failed with status {response.status}: {error_text}")
                    raise Exception(f"Backtest request failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Backtest request successful")
                return result
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_backtest_result(self, backtest_id: str) -> Dict[str, Any]:
        """
        Get the result of a previously run backtest.
        """
        url = f"{self.base_url}/api/v1/backtest/{backtest_id}"
        
        logger.info(f"Sending backtest result request to {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Backtest result request failed with status {response.status}: {error_text}")
                    raise Exception(f"Backtest result request failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Backtest result request successful")
                return result
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_backtest_status(self, backtest_id: str) -> Dict[str, Any]:
        """
        Get the status of a previously run backtest.
        """
        url = f"{self.base_url}/api/v1/backtest/{backtest_id}/status"
        
        logger.info(f"Sending backtest status request to {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Backtest status request failed with status {response.status}: {error_text}")
                    raise Exception(f"Backtest status request failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Backtest status request successful")
                return result