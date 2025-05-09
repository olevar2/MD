"""
Analysis Engine Interface

This module defines the interface for interacting with the analysis engine service.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from common_lib.models import MarketData

class IAnalysisEngine(ABC):
    """Interface defining required analysis capabilities"""
    
    @abstractmethod
    async def run_analysis(
        self, 
        analyzer_name: str, 
        market_data: Union[MarketData, Dict[str, MarketData]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a specific type of analysis on market data"""
        pass
    
    @abstractmethod
    async def get_market_regime(
        self,
        symbol: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get the market regime analysis for a symbol"""
        pass
    
    @abstractmethod
    async def get_confluence_analysis(
        self,
        market_data: Dict[str, MarketData],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get confluence analysis across multiple timeframes/indicators"""
        pass

    @abstractmethod
    async def analyze_pattern_effectiveness(
        self,
        pattern_name: str,
        symbol: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze the effectiveness of a specific pattern"""
        pass
