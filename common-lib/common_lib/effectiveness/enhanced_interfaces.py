"""
Enhanced Tool Effectiveness Interfaces Module

This module provides enhanced interfaces for tool effectiveness analysis,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from common_lib.effectiveness.interfaces import (
    MarketRegimeEnum,
    TimeFrameEnum,
    SignalEvent,
    SignalOutcome,
    IToolEffectivenessTracker
)


class IEnhancedToolEffectivenessTracker(IToolEffectivenessTracker):
    """Enhanced interface for tool effectiveness tracking with advanced analytics"""

    @abstractmethod
    def get_tool_performance_history(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical performance data for a tool.

        Args:
            tool_id: Tool identifier
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of historical performance data points
        """
        pass

    @abstractmethod
    def get_regime_transition_performance(
        self,
        tool_id: str,
        from_regime: str,
        to_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get tool performance during market regime transitions.

        Args:
            tool_id: Tool identifier
            from_regime: Starting market regime
            to_regime: Ending market regime
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter

        Returns:
            Dictionary containing transition performance metrics
        """
        pass

    @abstractmethod
    def get_tool_correlation_matrix(
        self,
        tool_ids: List[str],
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get correlation matrix between different tools.

        Args:
            tool_ids: List of tool identifiers
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter

        Returns:
            Dictionary mapping tool IDs to dictionaries of correlation coefficients
        """
        pass

    @abstractmethod
    def get_tool_effectiveness_confidence(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get confidence metrics for tool effectiveness.

        Args:
            tool_id: Tool identifier
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter

        Returns:
            Dictionary containing confidence metrics (e.g., confidence intervals)
        """
        pass

    @abstractmethod
    def get_optimal_tool_combination(
        self,
        market_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        max_tools: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get optimal combination of tools for a market regime.

        Args:
            market_regime: Market regime to analyze
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            max_tools: Maximum number of tools to include

        Returns:
            List of tools with weights and expected performance
        """
        pass


class IAdaptiveLayerService(ABC):
    """Interface for adaptive layer services"""

    @abstractmethod
    async def get_adaptive_parameters(
        self,
        symbol: str,
        timeframe: str,
        strategy_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get adaptive parameters for a trading strategy.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            strategy_id: Optional strategy identifier
            context: Optional contextual information

        Returns:
            Dictionary with adaptive parameters
        """
        pass

    @abstractmethod
    async def record_strategy_performance(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        performance_metrics: Dict[str, Any],
        parameters_used: Dict[str, Any]
    ) -> bool:
        """
        Record strategy performance for adaptive learning.

        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            timeframe: Chart timeframe
            performance_metrics: Performance metrics
            parameters_used: Parameters used for the strategy

        Returns:
            Success flag
        """
        pass

    @abstractmethod
    async def get_tool_signal_weights(
        self,
        market_regime: str,
        tools: List[str],
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get signal weights for tools based on their effectiveness.

        Args:
            market_regime: Current market regime
            tools: List of tools to get weights for
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter

        Returns:
            Dictionary mapping tool IDs to weights
        """
        pass

    @abstractmethod
    async def run_adaptation_cycle(
        self,
        market_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Run an adaptation cycle to adjust parameters based on current conditions.

        Args:
            market_regime: Current market regime
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            lookback_hours: Hours of data to consider

        Returns:
            Dictionary with adaptation results
        """
        pass

    @abstractmethod
    async def get_adaptation_recommendations(
        self,
        symbol: str,
        timeframe: str,
        current_market_data: Dict[str, Any],
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get recommendations for strategy adaptation.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            current_market_data: Current market data
            strategy_id: Optional strategy identifier

        Returns:
            Dictionary with adaptation recommendations
        """
        pass
