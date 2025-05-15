"""
Interface for the Market Analysis Service.

This module defines the interface for the Market Analysis Service, which provides
market analysis capabilities, including pattern recognition, support/resistance detection,
market regime detection, and correlation analysis.
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime


class IMarketAnalysisService(ABC):
    """Interface for the Market Analysis Service."""

    @abstractmethod
    async def analyze_market(self, 
                            symbol: str,
                            timeframe: str,
                            data: Dict[str, Any],
                            analysis_types: List[str],
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis for the specified symbol and timeframe.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the analysis results
        """
        pass

    @abstractmethod
    async def detect_patterns(self, 
                             symbol: str,
                             timeframe: str,
                             data: Dict[str, Any],
                             pattern_types: Optional[List[str]] = None,
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect chart patterns in the market data.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            pattern_types: Optional list of pattern types to detect
            config: Optional configuration parameters

        Returns:
            A dictionary containing the detected patterns
        """
        pass

    @abstractmethod
    async def detect_support_resistance(self, 
                                       symbol: str,
                                       timeframe: str,
                                       data: Dict[str, Any],
                                       methods: Optional[List[str]] = None,
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect support and resistance levels in the market data.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            methods: Optional list of detection methods to use
            config: Optional configuration parameters

        Returns:
            A dictionary containing the support and resistance levels
        """
        pass

    @abstractmethod
    async def detect_market_regime(self, 
                                  symbol: str,
                                  timeframe: str,
                                  data: Dict[str, Any],
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect the current market regime based on the market data.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            config: Optional configuration parameters

        Returns:
            A dictionary containing the market regime information
        """
        pass

    @abstractmethod
    async def get_regime_history(self,
                                symbol: str,
                                timeframe: str,
                                limit: Optional[int] = 10,
                                config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get historical regime data for a specific symbol and timeframe.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            limit: Maximum number of historical regimes to return
            config: Optional configuration parameters

        Returns:
            A list of dictionaries containing historical regime information
        """
        pass

    @abstractmethod
    async def analyze_correlation(self, 
                                 symbols: List[str],
                                 timeframe: str,
                                 data: Dict[str, Dict[str, Any]],
                                 method: Optional[str] = "pearson",
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze correlation between multiple symbols.

        Args:
            symbols: List of symbols to analyze
            timeframe: The timeframe to analyze
            data: Dictionary of market data for each symbol
            method: Correlation method to use
            config: Optional configuration parameters

        Returns:
            A dictionary containing the correlation analysis
        """
        pass

    @abstractmethod
    async def find_optimal_market_conditions(self,
                                           tool_id: str,
                                           min_sample_size: int = 10,
                                           timeframe: Optional[str] = None,
                                           instrument: Optional[str] = None,
                                           from_date: Optional[datetime] = None,
                                           to_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Find the optimal market conditions for a specific tool.

        Args:
            tool_id: The ID of the tool to analyze
            min_sample_size: Minimum sample size for analysis
            timeframe: Optional timeframe filter
            instrument: Optional instrument filter
            from_date: Optional start date filter
            to_date: Optional end date filter

        Returns:
            A dictionary containing the optimal market conditions
        """
        pass

    @abstractmethod
    async def recommend_tools_for_current_regime(self,
                                               current_regime: str,
                                               instrument: Optional[str] = None,
                                               timeframe: Optional[str] = None,
                                               min_sample_size: int = 10,
                                               min_win_rate: float = 50.0,
                                               top_n: int = 3) -> Dict[str, Any]:
        """
        Recommend the best trading tools for the current market regime.

        Args:
            current_regime: The current market regime
            instrument: Optional instrument filter
            timeframe: Optional timeframe filter
            min_sample_size: Minimum sample size for analysis
            min_win_rate: Minimum win rate for recommended tools
            top_n: Number of top tools to recommend

        Returns:
            A dictionary containing the recommended tools
        """
        pass