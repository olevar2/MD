"""
Interface definitions for analysis engine integration
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

class IAnalysisProvider(ABC):
    @abstractmethod
    async def get_market_analysis(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get market analysis results"""
        pass

    @abstractmethod
    async def get_causal_analysis(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        variables: List[str]
    ) -> Dict[str, Any]:
        """Get causal analysis results"""
        pass

    @abstractmethod
    async def get_regime_analysis(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get market regime analysis"""
        pass
