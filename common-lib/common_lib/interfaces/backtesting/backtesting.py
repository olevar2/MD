from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class IBacktestingService(ABC):
    """
    Interface for backtesting service.
    """
    
    @abstractmethod
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
        
        Args:
            symbol: Symbol to backtest
            timeframe: Timeframe to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_config: Strategy configuration
            parameters: Additional parameters for the backtest
            
        Returns:
            Backtest task information
        """
        pass
        
    @abstractmethod
    async def get_backtest_result(self, backtest_id: str) -> Dict[str, Any]:
        """
        Get the result of a previously run backtest.
        
        Args:
            backtest_id: ID of the backtest
            
        Returns:
            Backtest result
        """
        pass
        
    @abstractmethod
    async def get_backtest_status(self, backtest_id: str) -> Dict[str, Any]:
        """
        Get the status of a previously run backtest.
        
        Args:
            backtest_id: ID of the backtest
            
        Returns:
            Backtest status
        """
        pass
        
    @abstractmethod
    async def cancel_backtest(self, backtest_id: str) -> bool:
        """
        Cancel a running backtest.
        
        Args:
            backtest_id: ID of the backtest
            
        Returns:
            True if the backtest was cancelled, False otherwise
        """
        pass
        
    @abstractmethod
    async def list_backtests(
        self,
        limit: int = 10,
        offset: int = 0,
        strategy_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all backtests with optional filtering.
        
        Args:
            limit: Maximum number of backtests to return
            offset: Offset for pagination
            strategy_id: Filter by strategy ID
            status: Filter by status
            
        Returns:
            List of backtests
        """
        pass