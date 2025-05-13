"""
Analysis Interfaces Module

This module provides interfaces for analysis components used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime


class MarketRegimeType(str, Enum):
    """Market regime types for analysis."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_NARROW = "ranging_narrow"
    RANGING_WIDE = "ranging_wide"
    VOLATILE = "volatile"
    CHOPPY = "choppy"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class AnalysisTimeframe(str, Enum):
    """Standard timeframes for analysis."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class IMarketRegimeAnalyzer(ABC):
    """Interface for market regime analysis."""
    
    @abstractmethod
    async def detect_regime(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe],
        lookback_periods: int = 100
    ) -> Dict[str, Any]:
        """
        Detect the current market regime for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with regime information including:
            - regime_type: The detected regime type
            - confidence: Confidence score for the detection
            - regime_metrics: Additional metrics about the regime
            - regime_history: Recent regime changes
        """
        pass
    
    @abstractmethod
    async def get_regime_probabilities(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe]
    ) -> Dict[MarketRegimeType, float]:
        """
        Get probability distribution across different regime types.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            
        Returns:
            Dictionary mapping regime types to their probabilities
        """
        pass
    
    @abstractmethod
    async def get_regime_transition_probability(
        self,
        symbol: str,
        from_regime: MarketRegimeType,
        to_regime: MarketRegimeType,
        timeframe: Union[str, AnalysisTimeframe]
    ) -> float:
        """
        Get the probability of transitioning between regimes.
        
        Args:
            symbol: The trading symbol
            from_regime: Starting regime type
            to_regime: Target regime type
            timeframe: The timeframe to analyze
            
        Returns:
            Probability of transition (0.0 to 1.0)
        """
        pass


class IMultiAssetAnalyzer(ABC):
    """Interface for multi-asset analysis."""
    
    @abstractmethod
    async def get_correlated_assets(
        self,
        symbol: str,
        min_correlation: float = 0.7,
        lookback_periods: int = 100
    ) -> Dict[str, float]:
        """
        Get assets correlated with the specified symbol.
        
        Args:
            symbol: The trading symbol
            min_correlation: Minimum correlation coefficient
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary mapping correlated symbols to their correlation coefficients
        """
        pass
    
    @abstractmethod
    async def get_currency_strength(
        self,
        currencies: Optional[List[str]] = None,
        timeframe: Union[str, AnalysisTimeframe] = AnalysisTimeframe.H1
    ) -> Dict[str, float]:
        """
        Get relative strength of currencies.
        
        Args:
            currencies: Optional list of currencies to analyze
            timeframe: The timeframe to analyze
            
        Returns:
            Dictionary mapping currencies to their strength scores
        """
        pass
    
    @abstractmethod
    async def get_cross_pair_opportunities(
        self,
        base_currency: str,
        quote_currency: str,
        related_pairs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze opportunities across related currency pairs.
        
        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            related_pairs: Optional list of related pairs to analyze
            
        Returns:
            Dictionary with cross-pair analysis results
        """
        pass


class IPatternRecognizer(ABC):
    """Interface for pattern recognition."""
    
    @abstractmethod
    async def detect_patterns(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe],
        lookback_periods: int = 100,
        pattern_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect chart patterns for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to analyze
            pattern_types: Optional list of pattern types to detect
            
        Returns:
            List of detected patterns with details
        """
        pass
    
    @abstractmethod
    async def get_pattern_statistics(
        self,
        symbol: str,
        pattern_type: str,
        timeframe: Union[str, AnalysisTimeframe],
        lookback_periods: int = 500
    ) -> Dict[str, Any]:
        """
        Get statistics about pattern effectiveness.
        
        Args:
            symbol: The trading symbol
            pattern_type: Type of pattern to analyze
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with pattern statistics
        """
        pass


class IAnalysisEngine(ABC):
    """Interface for the core analysis engine."""
    
    @abstractmethod
    async def get_technical_analysis(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe],
        indicators: List[Dict[str, Any]],
        lookback_periods: int = 100
    ) -> Dict[str, Any]:
        """
        Get technical analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            indicators: List of indicators to calculate
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with technical analysis results
        """
        pass
    
    @abstractmethod
    async def get_confluence_analysis(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe],
        lookback_periods: int = 100
    ) -> Dict[str, Any]:
        """
        Get confluence analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with confluence analysis results
        """
        pass
    
    @abstractmethod
    async def get_multi_timeframe_analysis(
        self,
        symbol: str,
        timeframes: List[Union[str, AnalysisTimeframe]],
        indicators: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get multi-timeframe analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframes: List of timeframes to analyze
            indicators: List of indicators to calculate
            
        Returns:
            Dictionary with multi-timeframe analysis results
        """
        pass
    
    @abstractmethod
    async def get_integrated_analysis(
        self,
        symbol: str,
        timeframe: Union[str, AnalysisTimeframe],
        include_components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get integrated analysis from multiple components.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            include_components: Optional list of components to include
            
        Returns:
            Dictionary with integrated analysis results
        """
        pass
