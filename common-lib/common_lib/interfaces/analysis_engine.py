
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

class IAnalysisProvider(ABC):
    """Interface for analysis providers"""

    @abstractmethod
    async def analyze_market(self,
                            symbol: str,
                            timeframe: str,
                            analysis_type: str,
                            start_time: datetime,
                            end_time: Optional[datetime] = None,
                            parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform market analysis"""
        pass

    @abstractmethod
    async def get_analysis_types(self) -> List[Dict[str, Any]]:
        """Get available analysis types"""
        pass

    @abstractmethod
    async def backtest_strategy(self,
                               strategy_id: str,
                               symbol: str,
                               timeframe: str,
                               start_time: datetime,
                               end_time: datetime,
                               parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backtest a trading strategy"""
        pass

class IIndicatorProvider(ABC):
    """Interface for indicator providers"""

    @abstractmethod
    async def calculate_indicator(self,
                                 indicator_name: str,
                                 data: pd.DataFrame,
                                 parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Calculate a technical indicator"""
        pass

    @abstractmethod
    async def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """Get information about an indicator"""
        pass

    @abstractmethod
    async def list_indicators(self) -> List[str]:
        """List available indicators"""
        pass

class IPatternRecognizer(ABC):
    """Interface for pattern recognition"""

    @abstractmethod
    async def recognize_patterns(self,
                                data: pd.DataFrame,
                                pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Recognize patterns in market data"""
        pass

    @abstractmethod
    async def get_pattern_types(self) -> List[Dict[str, Any]]:
        """Get available pattern types"""
        pass
