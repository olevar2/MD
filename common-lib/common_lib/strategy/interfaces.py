"""
Strategy Interfaces Module

This module provides interfaces for strategy execution and analysis components,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime


class SignalDirection(str, Enum):
    """Direction of a trading signal."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


class SignalTimeframe(str, Enum):
    """Timeframes for trading signals."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class SignalSource(str, Enum):
    """Source of a trading signal."""
    TECHNICAL_ANALYSIS = "technical_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    MACHINE_LEARNING = "machine_learning"
    SENTIMENT = "sentiment"
    MARKET_REGIME = "market_regime"
    MULTI_ASSET = "multi_asset"
    ECONOMIC_CALENDAR = "economic_calendar"
    CORRELATION = "correlation"
    CUSTOM = "custom"


class MarketRegimeType(str, Enum):
    """Types of market regimes."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_NARROW = "ranging_narrow"
    RANGING_WIDE = "ranging_wide"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class ISignal(ABC):
    """Interface for trading signals."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Get the source identifier of the signal."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> SignalSource:
        """Get the source type of the signal."""
        pass

    @property
    @abstractmethod
    def direction(self) -> SignalDirection:
        """Get the direction of the signal."""
        pass

    @property
    @abstractmethod
    def symbol(self) -> str:
        """Get the symbol the signal applies to."""
        pass

    @property
    @abstractmethod
    def timeframe(self) -> SignalTimeframe:
        """Get the timeframe of the signal."""
        pass

    @property
    @abstractmethod
    def strength(self) -> float:
        """Get the strength of the signal (0.0 to 1.0)."""
        pass

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp when the signal was generated."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get additional metadata about the signal."""
        pass


class IAnalysisProvider(ABC):
    """Interface for analysis providers."""

    @abstractmethod
    async def get_technical_analysis(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100
    ) -> Dict[str, Any]:
        """
        Get technical analysis for a symbol.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_bars: Number of bars to analyze

        Returns:
            Dictionary with technical analysis results
        """
        pass

    @abstractmethod
    async def get_pattern_recognition(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100
    ) -> Dict[str, Any]:
        """
        Get pattern recognition analysis for a symbol.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_bars: Number of bars to analyze

        Returns:
            Dictionary with pattern recognition results
        """
        pass

    @abstractmethod
    async def get_market_regime(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get current market regime for a symbol.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze

        Returns:
            Dictionary with market regime information
        """
        pass

    @abstractmethod
    async def get_multi_timeframe_analysis(
        self,
        symbol: str,
        timeframes: List[str],
        lookback_bars: int = 100
    ) -> Dict[str, Any]:
        """
        Get multi-timeframe analysis for a symbol.

        Args:
            symbol: The trading symbol
            timeframes: List of timeframes to analyze
            lookback_bars: Number of bars to analyze

        Returns:
            Dictionary with multi-timeframe analysis results
        """
        pass

    @abstractmethod
    async def get_multi_asset_analysis(
        self,
        symbol: str,
        related_symbols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get multi-asset analysis for a symbol.

        Args:
            symbol: The trading symbol
            related_symbols: Optional list of related symbols

        Returns:
            Dictionary with multi-asset analysis results
        """
        pass

    @abstractmethod
    async def get_integrated_analysis(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100,
        include_components: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get integrated analysis from multiple components.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_bars: Number of bars to analyze
            include_components: List of components to include

        Returns:
            Dictionary with integrated analysis results
        """
        pass


class IStrategyExecutor(ABC):
    """Interface for strategy execution."""

    @abstractmethod
    async def execute_strategy(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a trading strategy.

        Args:
            strategy_id: The ID of the strategy to execute
            symbol: The trading symbol
            timeframe: The timeframe to use
            parameters: Optional strategy parameters

        Returns:
            Dictionary with execution results
        """
        pass

    @abstractmethod
    async def backtest_strategy(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy.

        Args:
            strategy_id: The ID of the strategy to backtest
            symbol: The trading symbol
            timeframe: The timeframe to use
            start_date: Start date for backtesting
            end_date: End date for backtesting
            parameters: Optional strategy parameters

        Returns:
            Dictionary with backtest results
        """
        pass

    @abstractmethod
    async def get_strategy_signals(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get signals from a trading strategy.

        Args:
            strategy_id: The ID of the strategy
            symbol: The trading symbol
            timeframe: The timeframe to use
            parameters: Optional strategy parameters

        Returns:
            List of signal dictionaries
        """
        pass

    @abstractmethod
    async def optimize_strategy(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        parameters_range: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize a trading strategy.

        Args:
            strategy_id: The ID of the strategy to optimize
            symbol: The trading symbol
            timeframe: The timeframe to use
            start_date: Start date for optimization
            end_date: End date for optimization
            parameters_range: Range of parameters to optimize

        Returns:
            Dictionary with optimization results
        """
        pass


class ISignalAggregator(ABC):
    """Interface for signal aggregation."""

    @abstractmethod
    async def aggregate_signals(
        self,
        signals: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        market_regime: str = None
    ) -> Dict[str, Any]:
        """
        Aggregate multiple trading signals.

        Args:
            signals: List of signal dictionaries
            symbol: The trading symbol
            timeframe: The timeframe of the signals
            market_regime: Optional market regime

        Returns:
            Dictionary with aggregated signal
        """
        pass

    @abstractmethod
    async def get_signal_effectiveness(
        self,
        source_id: str,
        market_regime: str = None,
        timeframe: str = None
    ) -> Dict[str, float]:
        """
        Get effectiveness metrics for a signal source.

        Args:
            source_id: The source identifier
            market_regime: Optional market regime
            timeframe: Optional timeframe

        Returns:
            Dictionary with effectiveness metrics
        """
        pass


class IStrategyEvaluator(ABC):
    """Interface for strategy evaluation."""

    @abstractmethod
    async def evaluate_strategy(
        self,
        strategy_id: str,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy based on backtest results.

        Args:
            strategy_id: The ID of the strategy
            backtest_results: Results from backtesting

        Returns:
            Dictionary with evaluation metrics
        """
        pass

    @abstractmethod
    async def compare_strategies(
        self,
        strategy_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies.

        Args:
            strategy_results: Dictionary mapping strategy IDs to their results

        Returns:
            Dictionary with comparison results
        """
        pass

    @abstractmethod
    async def get_strategy_performance(
        self,
        strategy_id: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a strategy.

        Args:
            strategy_id: The ID of the strategy
            start_date: Optional start date for the period
            end_date: Optional end date for the period

        Returns:
            Dictionary with performance metrics
        """
        pass


class IStrategyEnhancer(ABC):
    """Interface for strategy enhancement functionality"""

    @abstractmethod
    async def enhance_strategy(
        self,
        strategy_id: str,
        enhancement_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance a strategy with additional functionality.

        Args:
            strategy_id: Strategy identifier
            enhancement_type: Type of enhancement to apply
            parameters: Enhancement parameters

        Returns:
            Dictionary with enhancement results
        """
        pass

    @abstractmethod
    async def get_enhancement_types(self) -> List[Dict[str, Any]]:
        """
        Get available enhancement types.

        Returns:
            List of enhancement type information dictionaries
        """
        pass

    @abstractmethod
    async def get_enhancement_history(
        self,
        strategy_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get enhancement history for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            List of enhancement history entries
        """
        pass

    @abstractmethod
    async def compare_enhancements(
        self,
        strategy_id: str,
        enhancement_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple enhancements for a strategy.

        Args:
            strategy_id: Strategy identifier
            enhancement_ids: List of enhancement identifiers to compare

        Returns:
            Dictionary with comparison results
        """
        pass


class ICausalStrategyEnhancer(IStrategyEnhancer):
    """Interface for causal strategy enhancement functionality"""

    @abstractmethod
    async def identify_causal_factors(
        self,
        strategy_id: str,
        data_period: Dict[str, Any],
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Identify causal factors affecting strategy performance.

        Args:
            strategy_id: Strategy identifier
            data_period: Dictionary with start and end dates for analysis
            significance_threshold: Threshold for statistical significance

        Returns:
            Dictionary with causal factors and their significance
        """
        pass

    @abstractmethod
    async def generate_causal_graph(
        self,
        strategy_id: str,
        data_period: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a causal graph for a strategy.

        Args:
            strategy_id: Strategy identifier
            data_period: Dictionary with start and end dates for analysis

        Returns:
            Dictionary with causal graph data
        """
        pass

    @abstractmethod
    async def apply_causal_enhancement(
        self,
        strategy_id: str,
        causal_factors: List[Dict[str, Any]],
        enhancement_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply causal enhancement to a strategy.

        Args:
            strategy_id: Strategy identifier
            causal_factors: List of causal factors to consider
            enhancement_parameters: Parameters for the enhancement

        Returns:
            Dictionary with enhancement results
        """
        pass
