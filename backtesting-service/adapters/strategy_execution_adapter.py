from typing import Dict, List, Any, Optional
import os
import httpx
from common_lib.resilience.decorators import with_standard_resilience

class StrategyExecutionAdapter:
    """
    Adapter for interacting with the strategy execution engine.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the adapter with the strategy execution engine URL.
        """
        self.base_url = base_url or os.environ.get("STRATEGY_EXECUTION_URL", "http://strategy-execution-engine:8000")
        self.timeout = int(os.environ.get("STRATEGY_EXECUTION_TIMEOUT", "120"))
    
    @with_standard_resilience()
    async def get_strategy_config(
        self,
        strategy_id: Optional[str],
        strategy_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get strategy configuration from the strategy execution engine.
        """
        payload = {
            "strategy_type": strategy_type,
            "parameters": parameters
        }
        
        if strategy_id:
            payload["strategy_id"] = strategy_id
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/v1/strategies/config", json=payload)
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def run_backtest(
        self,
        historical_data: Dict[str, Any],
        strategy_config: Dict[str, Any],
        initial_capital: float,
        commission: float,
        slippage: float,
        leverage: float,
        additional_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest using the strategy execution engine.
        """
        payload = {
            "historical_data": historical_data,
            "strategy_config": strategy_config,
            "initial_capital": initial_capital,
            "commission": commission,
            "slippage": slippage,
            "leverage": leverage
        }
        
        if additional_parameters:
            payload["additional_parameters"] = additional_parameters
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/v1/backtest", json=payload)
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Get available strategies from the strategy execution engine.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/strategies")
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_strategy_details(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get details of a specific strategy from the strategy execution engine.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/strategies/{strategy_id}")
            response.raise_for_status()
            return response.json()