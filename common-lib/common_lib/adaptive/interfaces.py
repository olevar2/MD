"""
Adaptive Layer Interfaces Module

This module provides interfaces for adaptive layer components used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime


class AdaptationLevelEnum(str, Enum):
    """Adaptation aggressiveness levels."""
    NONE = "none"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


class IAdaptiveStrategyService(ABC):
    """Interface for adaptive strategy services."""

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

    @abstractmethod
    def set_adaptation_level(
        self,
        level: Union[str, AdaptationLevelEnum]
    ) -> None:
        """
        Set the adaptation aggressiveness level.

        Args:
            level: Adaptation level
        """
        pass

    @abstractmethod
    def get_adaptation_level(self) -> str:
        """
        Get the current adaptation aggressiveness level.

        Returns:
            Current adaptation level
        """
        pass
