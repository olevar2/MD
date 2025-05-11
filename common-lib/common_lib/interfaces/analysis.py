"""
Analysis Provider Interface

This module defines the interface for analysis providers in the forex trading platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime


class IAnalysisProvider(ABC):
    """
    Interface for analysis providers.
    
    This interface defines the contract for services that provide technical analysis
    and market analysis functionality.
    """
    
    @abstractmethod
    async def calculate_indicator(
        self,
        indicator_name: str,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate a technical indicator.
        
        Args:
            indicator_name: Name of the indicator to calculate
            data: Input data for the calculation
            params: Parameters for the indicator
            
        Returns:
            DataFrame with the indicator values
        """
        pass
    
    @abstractmethod
    async def detect_market_regime(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100
    ) -> Dict[str, Any]:
        """
        Detect the current market regime for a symbol.
        
        Args:
            symbol: Symbol to detect regime for
            timeframe: Timeframe for analysis
            lookback_bars: Number of bars to use for detection
            
        Returns:
            Dictionary containing market regime information
        """
        pass
    
    @abstractmethod
    async def get_technical_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get technical indicators for a symbol.
        
        Args:
            symbol: Symbol to get indicators for
            timeframe: Timeframe for analysis
            indicators: List of indicators to calculate
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of data points
            
        Returns:
            Dictionary containing indicator values
        """
        pass
