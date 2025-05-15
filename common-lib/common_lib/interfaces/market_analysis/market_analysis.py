from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class IMarketAnalysisService(ABC):
    """
    Interface for market analysis service.
    """
    
    @abstractmethod
    async def analyze_market(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
            
        Returns:
            Market analysis data
        """
        pass
        
    @abstractmethod
    async def recognize_patterns(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Recognize chart patterns in market data.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            patterns: Patterns to recognize
            
        Returns:
            Pattern recognition data
        """
        pass
        
    @abstractmethod
    async def identify_support_resistance(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Identify support and resistance levels.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
            
        Returns:
            Support and resistance data
        """
        pass
        
    @abstractmethod
    async def detect_market_regime(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Detect market regime (trending, ranging, volatile).
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
            
        Returns:
            Market regime data
        """
        pass
        
    @abstractmethod
    async def analyze_correlations(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        symbols: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlations between symbols.
        
        Args:
            symbol: Primary symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            symbols: Additional symbols to analyze correlations with
            parameters: Additional parameters for the analysis
            
        Returns:
            Correlation analysis data
        """
        pass
        
    @abstractmethod
    async def get_available_patterns(self) -> List[Dict[str, Any]]:
        """
        Get available chart patterns for recognition.
        
        Returns:
            List of available patterns
        """
        pass
        
    @abstractmethod
    async def get_available_regimes(self) -> List[Dict[str, Any]]:
        """
        Get available market regimes for detection.
        
        Returns:
            List of available regimes
        """
        pass
        
    @abstractmethod
    async def get_available_methods(self) -> Dict[str, List[str]]:
        """
        Get available analysis methods.
        
        Returns:
            Dictionary of available methods
        """
        pass