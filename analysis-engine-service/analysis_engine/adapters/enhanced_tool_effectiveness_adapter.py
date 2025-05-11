"""
Enhanced Tool Effectiveness Adapter Module

This module provides adapters for enhanced tool effectiveness functionality,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid

from common_lib.effectiveness.interfaces import (
    MarketRegimeEnum,
    TimeFrameEnum,
    SignalEvent,
    SignalOutcome
)
from common_lib.effectiveness.enhanced_interfaces import (
    IEnhancedToolEffectivenessTracker,
    IAdaptiveLayerService
)
from analysis_engine.services.enhanced_tool_effectiveness import EnhancedToolEffectivenessTracker
from analysis_engine.services.adaptive_layer import AdaptiveLayer
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository

logger = logging.getLogger(__name__)


class EnhancedToolEffectivenessTrackerAdapter(IEnhancedToolEffectivenessTracker):
    """
    Adapter for enhanced tool effectiveness tracking that implements the common interface.

    This adapter wraps the actual EnhancedToolEffectivenessTracker implementation.
    """

    def __init__(self, tracker_instance: Optional[EnhancedToolEffectivenessTracker] = None):
        """
        Initialize the adapter.

        Args:
            tracker_instance: Optional actual tracker instance to wrap
        """
        self.tracker = tracker_instance or EnhancedToolEffectivenessTracker()

    def record_signal(self, signal: SignalEvent) -> str:
        """Record a new trading signal."""
        return self.tracker.record_signal(signal)

    def record_outcome(self, outcome: SignalOutcome) -> bool:
        """Record the outcome of a signal."""
        return self.tracker.record_outcome(outcome)

    def get_tool_effectiveness(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get effectiveness metrics for a tool."""
        return self.tracker.get_tool_effectiveness(
            tool_id=tool_id,
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

    def get_best_tools(
        self,
        market_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        min_signals: int = 10,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get best performing tools for a market regime."""
        return self.tracker.get_best_tools(
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol,
            min_signals=min_signals,
            top_n=top_n
        )

    def get_tool_performance_history(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get historical performance data for a tool."""
        return self.tracker.get_performance_history(
            tool_id=tool_id,
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

    def get_regime_transition_performance(
        self,
        tool_id: str,
        from_regime: str,
        to_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get tool performance during market regime transitions."""
        return self.tracker.get_regime_transition_performance(
            tool_id=tool_id,
            from_regime=from_regime,
            to_regime=to_regime,
            timeframe=timeframe,
            symbol=symbol
        )

    def get_tool_correlation_matrix(
        self,
        tool_ids: List[str],
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix between different tools."""
        return self.tracker.get_correlation_matrix(
            tool_ids=tool_ids,
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol
        )

    def get_tool_effectiveness_confidence(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get confidence metrics for tool effectiveness."""
        return self.tracker.get_effectiveness_confidence(
            tool_id=tool_id,
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol
        )

    def get_optimal_tool_combination(
        self,
        market_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        max_tools: int = 5
    ) -> List[Dict[str, Any]]:
        """Get optimal combination of tools for a market regime."""
        return self.tracker.get_optimal_combination(
            market_regime=market_regime,
            timeframe=timeframe,
            symbol=symbol,
            max_tools=max_tools
        )


class AdaptiveLayerServiceAdapter(IAdaptiveLayerService):
    """
    Adapter for adaptive layer service that implements the common interface.

    This adapter wraps the actual AdaptiveLayer implementation.
    """

    def __init__(
        self,
        adaptive_layer_instance: Optional[AdaptiveLayer] = None,
        repository: Optional[ToolEffectivenessRepository] = None
    ):
        """
        Initialize the adapter.

        Args:
            adaptive_layer_instance: Optional actual adaptive layer instance to wrap
            repository: Optional repository for effectiveness data
        """
        self.repository = repository or ToolEffectivenessRepository()
        self.adaptive_layer = adaptive_layer_instance or AdaptiveLayer(self.repository)

    async def get_adaptive_parameters(
        self,
        symbol: str,
        timeframe: str,
        strategy_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get adaptive parameters for a trading strategy."""
        # Convert timeframe string to enum if needed
        if isinstance(timeframe, str):
            try:
                timeframe_enum = TimeFrameEnum(timeframe)
            except ValueError:
                timeframe_enum = None
        else:
            timeframe_enum = timeframe

        # Get price data from context if available
        price_data = context.get("price_data") if context else None

        # Get available tools from context if available
        available_tools = context.get("available_tools") if context else None

        # Get recent signals from context if available
        recent_signals = context.get("recent_signals") if context else None

        # Generate adaptive parameters
        return self.adaptive_layer.generate_adaptive_parameters(
            symbol=symbol,
            timeframe=timeframe_enum or timeframe,
            price_data=price_data,
            available_tools=available_tools,
            recent_signals=recent_signals
        )

    async def record_strategy_performance(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        performance_metrics: Dict[str, Any],
        parameters_used: Dict[str, Any]
    ) -> bool:
        """Record strategy performance for adaptive learning."""
        try:
            # Record performance in the repository
            await self.repository.save_strategy_performance(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                performance_metrics=performance_metrics,
                parameters_used=parameters_used
            )
            return True
        except Exception as e:
            logger.error(f"Error recording strategy performance: {str(e)}")
            return False

    async def get_adaptation_recommendations(
        self,
        symbol: str,
        timeframe: str,
        current_market_data: Dict[str, Any],
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get recommendations for strategy adaptation."""
        # Convert timeframe string to enum if needed
        if isinstance(timeframe, str):
            try:
                timeframe_enum = TimeFrameEnum(timeframe)
            except ValueError:
                timeframe_enum = None
        else:
            timeframe_enum = timeframe

        # Detect market regime
        market_regime = self.adaptive_layer.market_regime_service.detect_regime(
            price_data=current_market_data.get("price_data"),
            symbol=symbol,
            timeframe=timeframe_enum or timeframe
        )

        # Get tool effectiveness metrics
        tool_metrics = {}
        available_tools = current_market_data.get("available_tools", [])
        for tool_id in available_tools:
            metrics = await self.repository.get_tool_metrics(
                tool_id=tool_id,
                market_regime=market_regime,
                timeframe=timeframe
            )
            tool_metrics[tool_id] = metrics

        # Generate recommendations
        recommendations = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy_id": strategy_id,
            "market_regime": market_regime,
            "confidence": 0.8,  # Default confidence
            "recommended_tools": [],
            "parameter_adjustments": {},
            "risk_recommendation": "neutral",
            "timestamp": datetime.now().isoformat()
        }

        # Add recommended tools based on effectiveness
        sorted_tools = sorted(
            tool_metrics.items(),
            key=lambda x: x[1].get("effectiveness_score", 0.0),
            reverse=True
        )

        recommendations["recommended_tools"] = [
            {
                "tool_id": tool_id,
                "effectiveness_score": metrics.get("effectiveness_score", 0.0),
                "weight": max(0.1, min(1.0, metrics.get("effectiveness_score", 0.5) * 2))
            }
            for tool_id, metrics in sorted_tools[:5]  # Top 5 tools
        ]

        return recommendations

