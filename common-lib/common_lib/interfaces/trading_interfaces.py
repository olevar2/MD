"""
Trading Service Interfaces

This module defines interfaces for trading services used by the analysis engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IRiskManager(ABC):
    """Interface for risk management services."""
    
    @abstractmethod
    async def evaluate_risk(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk for a potential trade.
        
        Args:
            trade_params: Parameters for the trade
            
        Returns:
            Dictionary of risk evaluation results
        """
        pass


class ITradingGateway(ABC):
    """Interface for trading gateway services."""
    
    @abstractmethod
    async def get_market_status(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market status for symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dictionary of market status information
        """
        pass
