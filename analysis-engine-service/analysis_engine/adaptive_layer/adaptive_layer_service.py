"""
AdaptiveLayerService

This module provides integration of all the adaptive components including:
- AdaptiveWeightCalculator
- ConfluenceAnalyzer
- Advanced Technical Analysis
- Market Regime Detection
- Tool Effectiveness Tracking

It serves as the main entry point for all adaptive functionality in the platform.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from common_lib.adaptive.interfaces import AdaptationLevelEnum, IAdaptiveStrategyService
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.adaptive_layer import AdaptiveLayer
from analysis_engine.adaptive_layer.adaptive_weight_calculator import AdaptiveWeightCalculator
from analysis_engine.learning_from_mistakes.error_pattern_recognition import ErrorPatternRecognitionSystem
from analysis_engine.adaptive_layer.market_regime_aware_adapter import MarketRegimeAwareAdapter
from analysis_engine.adapters.adaptive_strategy_adapter import AdaptiveStrategyServiceAdapter


class AdaptiveLayerService(IAdaptiveStrategyService):
    """
    Enhanced service that integrates all adaptive components for Phase 4
    of the project. This class serves as the main entry point for strategies
    to access adaptive functionality.

    Implements the IAdaptiveStrategyService interface to break circular dependencies.
    """

    def __init__(
        self,
        repository: ToolEffectivenessRepository,
        market_regime_service: Optional[MarketRegimeService] = None,
        adaptive_layer: Optional[AdaptiveLayer] = None,
        weight_calculator: Optional[AdaptiveWeightCalculator] = None,
        error_pattern_system: Optional[ErrorPatternRecognitionSystem] = None
    ):
        """
        Initialize the Adaptive Layer Service with all required components.

        Args:
            repository: Repository for effectiveness data
            market_regime_service: Service for detecting market regimes
            adaptive_layer: Core adaptive layer component
            weight_calculator: Component for calculating adaptive weights
            error_pattern_system: Component for learning from mistakes
        """
        self.logger = logging.getLogger(__name__)
        self.repository = repository
        self.market_regime_service = market_regime_service or MarketRegimeService()
        self.adaptive_layer = adaptive_layer or AdaptiveLayer(repository, self.market_regime_service)
        self.weight_calculator = weight_calculator or AdaptiveWeightCalculator()
        self.error_pattern_system = error_pattern_system or ErrorPatternRecognitionSystem()
        self.adaptation_level = AdaptationLevelEnum.MODERATE

        # Create regime-aware adapter
        self.regime_adapter = MarketRegimeAwareAdapter(self.market_regime_service)

        # Initialize cache for optimization
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=15)

    # IAdaptiveStrategyService interface implementation

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
        try:
            # Extract price data from context if available
            price_data = context.get("price_data") if context else None
            if price_data is None:
                self.logger.warning(f"No price data provided in context for {symbol} {timeframe}")
                price_data = pd.DataFrame()  # Empty dataframe as fallback

            # Extract available tools from context or use default
            available_tools = context.get("available_tools", []) if context else []

            # Extract recent signals from context if available
            recent_signals = context.get("recent_signals") if context else None

            # Get adaptive parameters using the existing method
            return await self.get_adaptive_parameters_internal(
                symbol=symbol,
                timeframe=timeframe,
                price_data=price_data,
                available_tools=available_tools,
                recent_signals=recent_signals,
                strategy_id=strategy_id
            )
        except Exception as e:
            self.logger.error(f"Error getting adaptive parameters: {str(e)}", exc_info=True)
            # Return default parameters as fallback
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy_id": strategy_id,
                "adaptation_level": self.adaptation_level.value,
                "parameters": {
                    "stop_loss_pips": 20,
                    "take_profit_pips": 40,
                    "max_trades": 3,
                    "risk_per_trade": 0.02
                },
                "signal_weights": {
                    "technical_analysis": 0.4,
                    "pattern_recognition": 0.3,
                    "machine_learning": 0.3
                }
            }

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
        try:
            self.logger.info(
                f"Recording strategy performance: {strategy_id}, {symbol}, {timeframe}, "
                f"metrics: {performance_metrics}"
            )

            # TODO: Implement actual recording logic
            # For now, just log the performance

            return True
        except Exception as e:
            self.logger.error(f"Error recording strategy performance: {str(e)}", exc_info=True)
            return False

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
        try:
            # Convert string market_regime to enum if needed
            regime = market_regime
            if isinstance(market_regime, str):
                try:
                    regime = MarketRegime[market_regime.upper()]
                except (KeyError, AttributeError):
                    # Create a temporary enum value
                    regime = MarketRegime.UNKNOWN

            # Convert string timeframe to enum if needed
            tf = timeframe
            if isinstance(timeframe, str):
                try:
                    tf = TimeFrame[timeframe.upper()]
                except (KeyError, AttributeError):
                    tf = None

            # Get tool metrics for each tool
            tool_metrics = {}
            for tool_id in tools:
                metrics = self._get_tool_metrics(tool_id, symbol, tf, regime)
                if metrics:
                    tool_metrics[tool_id] = metrics

            # Calculate weights based on effectiveness
            weights = self._calculate_signal_weights(tool_metrics, regime)

            return weights
        except Exception as e:
            self.logger.error(f"Error getting tool signal weights: {str(e)}", exc_info=True)

            # Fallback implementation - distribute weights evenly
            weight = 1.0 / len(tools) if tools else 0.0
            return {tool_id: weight for tool_id in tools}

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
        try:
            self.logger.info(
                f"Running adaptation cycle: {market_regime}, {timeframe}, {symbol}, "
                f"lookback_hours: {lookback_hours}"
            )

            # TODO: Implement actual adaptation cycle
            # For now, return a placeholder result

            return {
                "market_regime": market_regime,
                "timeframe": timeframe,
                "symbol": symbol,
                "adaptations_made": 0,
                "status": "not_implemented"
            }
        except Exception as e:
            self.logger.error(f"Error running adaptation cycle: {str(e)}", exc_info=True)

            return {
                "market_regime": market_regime,
                "timeframe": timeframe,
                "symbol": symbol,
                "adaptations_made": 0,
                "status": "error",
                "error": str(e)
            }

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
        try:
            self.logger.info(
                f"Getting adaptation recommendations: {symbol}, {timeframe}, "
                f"strategy_id: {strategy_id}"
            )

            # TODO: Implement actual recommendation logic
            # For now, return a placeholder result

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy_id": strategy_id,
                "recommendations": {
                    "stop_loss": "no_change",
                    "take_profit": "no_change",
                    "entry_criteria": "no_change"
                },
                "confidence": 0.5,
                "status": "not_implemented"
            }
        except Exception as e:
            self.logger.error(f"Error getting adaptation recommendations: {str(e)}", exc_info=True)

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy_id": strategy_id,
                "recommendations": {
                    "stop_loss": "no_change",
                    "take_profit": "no_change",
                    "entry_criteria": "no_change"
                },
                "confidence": 0.5,
                "status": "error",
                "error": str(e)
            }

    def set_adaptation_level(
        self,
        level: Union[str, AdaptationLevelEnum]
    ) -> None:
        """
        Set the adaptation aggressiveness level.

        Args:
            level: Adaptation level
        """
        if isinstance(level, str):
            try:
                self.adaptation_level = AdaptationLevelEnum(level.lower())
            except ValueError:
                self.logger.warning(f"Invalid adaptation level: {level}, using MODERATE")
                self.adaptation_level = AdaptationLevelEnum.MODERATE
        else:
            self.adaptation_level = level

        self.logger.info(f"Adaptation level set to: {self.adaptation_level.value}")

    def get_adaptation_level(self) -> str:
        """
        Get the current adaptation aggressiveness level.

        Returns:
            Current adaptation level
        """
        return self.adaptation_level.value

    async def get_adaptive_parameters_internal(
        self,
        symbol: str,
        timeframe: str,
        price_data: pd.DataFrame,
        available_tools: List[str],
        recent_signals: Optional[List[Dict[str, Any]]] = None,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get adaptive parameters for trading, enhanced with Phase 4 capabilities.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_data: Recent price data
            available_tools: List of available analysis tools
            recent_signals: Recent trading signals if available
            strategy_id: Optional strategy ID for strategy-specific adaptations

        Returns:
            Dictionary with adaptive parameters
        """
        try:
            # First get base adaptive parameters from core adaptive layer
            base_params = self.adaptive_layer.generate_adaptive_parameters(
                symbol=symbol,
                timeframe=timeframe,
                price_data=price_data,
                available_tools=available_tools,
                recent_signals=recent_signals
            )

            # Extract current regime and certainty
            current_regime = base_params.get("detected_regime", {}).get("final_regime", MarketRegime.UNKNOWN)
            regime_certainty = base_params.get("detected_regime", {}).get("certainty", 0.5)

            # Enhance signal weights with our advanced weight calculator
            tool_metrics = {}
            for tool_id in available_tools:
                metrics = self._get_tool_metrics(tool_id, symbol, timeframe, current_regime)
                if metrics:
                    tool_metrics[tool_id] = metrics

            # Calculate enhanced weights using our advanced calculator
            enhanced_weights = self.weight_calculator.calculate_signal_weights(
                effectiveness_metrics=tool_metrics,
                current_regime=current_regime,
                regime_certainty=regime_certainty
            )

            # Check if there are recent performance trends to apply
            if recent_signals and len(recent_signals) > 0:
                trend_factors = self._calculate_trend_factors(recent_signals, timeframe)
                enhanced_weights = self.weight_calculator.apply_trend_factors(
                    enhanced_weights, trend_factors
                )

            # Apply strategy-specific adjustments if a strategy ID is provided
            if strategy_id:
                strategy_adjustments = await self._get_strategy_specific_adjustments(
                    strategy_id, current_regime
                )
                enhanced_weights = self._apply_strategy_weight_adjustments(
                    enhanced_weights, strategy_adjustments
                )

            # Update the base params with enhanced weights and additional information
            base_params["signal_weights"] = enhanced_weights

            # Add error pattern analysis if we have recent signals
            if recent_signals:
                error_patterns = self.error_pattern_system.analyze_recent_signals(
                    recent_signals, current_regime
                )
                if error_patterns:
                    base_params["error_patterns"] = error_patterns

            # Add confluence information if available
            confluence_data = self.get_confluence_data(symbol, timeframe, price_data)
            if confluence_data:
                base_params["confluence_data"] = confluence_data

            # Add Phase 4 specific parameter recommendations
            base_params["advanced_parameters"] = self._generate_advanced_parameters(
                current_regime, regime_certainty, tool_metrics
            )

            # Add statistical significance information
            base_params["statistical_significance"] = self._calculate_statistical_significance(
                tool_metrics, current_regime
            )

            return base_params

        except Exception as e:
            self.logger.error(f"Error generating enhanced adaptive parameters: {str(e)}", exc_info=True)
            return self.adaptive_layer._get_default_parameters(available_tools)

    def get_confluence_data(
        self,
        symbol: str,
        timeframe: str,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get confluence analysis for a specific symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_data: Recent price data

        Returns:
            Dictionary with confluence information
        """
        try:
            # This is a placeholder - will be replaced with actual implementation
            # when we create the ConfluenceAnalyzer in a later step
            return {
                "support_resistance_confluence": [],
                "indicator_confluences": [],
                "multi_timeframe_confluences": [],
                "harmonic_patterns": []
            }
        except Exception as e:
            self.logger.error(f"Error getting confluence data: {str(e)}", exc_info=True)
            return {}

    def _get_tool_metrics(
        self,
        tool_id: str,
        symbol: str,
        timeframe: str,
        market_regime: MarketRegime
    ) -> Dict[str, Any]:
        """
        Get effectiveness metrics for a tool, with caching for performance.

        Args:
            tool_id: ID of the analysis tool
            symbol: Trading symbol
            timeframe: Chart timeframe
            market_regime: Current market regime

        Returns:
            Dictionary with tool effectiveness metrics
        """
        cache_key = f"{tool_id}_{symbol}_{timeframe}_{market_regime.value}"

        # Check if we have cached metrics
        if (cache_key in self.cache and
            cache_key in self.cache_expiry and
            self.cache_expiry[cache_key] > datetime.now()):
            return self.cache[cache_key]

        try:
            # Get metrics from the repository
            metrics = self.adaptive_layer._get_cached_effectiveness_metrics(
                tool_id=tool_id,
                timeframe=timeframe,
                instrument=symbol,
                market_regime=market_regime
            )

            # Cache the results
            self.cache[cache_key] = metrics
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration

            return metrics
        except Exception as e:
            self.logger.error(f"Error getting tool metrics: {str(e)}")
            return {}

    def _calculate_trend_factors(
        self,
        recent_signals: List[Dict[str, Any]],
        timeframe: str
    ) -> Dict[str, float]:
        """
        Calculate trend factors based on recent signal performance.

        Args:
            recent_signals: List of recent signals with outcomes
            timeframe: Chart timeframe

        Returns:
            Dictionary mapping tool IDs to trend factors
        """
        trend_factors = {}

        # Group signals by tool
        tool_signals = {}
        for signal in recent_signals:
            tool_id = signal.get("tool_id")
            if not tool_id:
                continue

            if tool_id not in tool_signals:
                tool_signals[tool_id] = []

            tool_signals[tool_id].append(signal)

        # Calculate trend factor for each tool
        for tool_id, signals in tool_signals.items():
            # Calculate recent performance
            successful = sum(1 for s in signals if s.get("outcome") == "success")
            total = len(signals)

            if total < 3:  # Require at least 3 signals
                trend_factors[tool_id] = 1.0
                continue

            success_rate = successful / total if total > 0 else 0.5

            # Convert to trend factor (1.0 is neutral, >1.0 is improving, <1.0 is declining)
            # Scale: 0.7 to 1.3 based on success rate (0% to 100%)
            trend_factor = 0.7 + (success_rate * 0.6)

            trend_factors[tool_id] = trend_factor

        return trend_factors

    async def _get_strategy_specific_adjustments(
        self,
        strategy_id: str,
        market_regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Get strategy-specific weight adjustments.

        Args:
            strategy_id: Strategy ID
            market_regime: Current market regime

        Returns:
            Dictionary mapping tool IDs to adjustment factors
        """
        # This would typically come from a database of strategy-specific settings
        # For now, we'll return an empty dict as a placeholder
        return {}

    def _apply_strategy_weight_adjustments(
        self,
        weights: Dict[str, float],
        adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply strategy-specific adjustments to weights.

        Args:
            weights: Current weights
            adjustments: Adjustment factors

        Returns:
            Dictionary with adjusted weights
        """
        if not adjustments:
            return weights

        adjusted = {}
        total = 0.0

        for tool_id, weight in weights.items():
            if tool_id in adjustments:
                new_weight = weight * adjustments[tool_id]
            else:
                new_weight = weight

            adjusted[tool_id] = new_weight
            total += new_weight

        # Re-normalize
        if total > 0:
            return {k: v/total for k, v in adjusted.items()}

        return weights

    def _generate_advanced_parameters(
        self,
        market_regime: MarketRegime,
        regime_certainty: float,
        tool_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate advanced parameters for Phase 4 strategies.

        Args:
            market_regime: Current market regime
            regime_certainty: Confidence in regime detection
            tool_metrics: Tool effectiveness metrics

        Returns:
            Dictionary with advanced parameters
        """
        # This will contain regime-specific advanced parameters for Forex strategies
        advanced_params = {}

        # Parameters for Harmonic Pattern Strategy
        advanced_params["harmonic_patterns"] = {
            "detection_sensitivity": self._calculate_harmonic_sensitivity(market_regime),
            "confirmation_threshold": self._calculate_confirmation_threshold(market_regime, regime_certainty)
        }

        # Parameters for Elliott Wave Strategy
        advanced_params["elliott_wave"] = {
            "wave_detection_mode": self._get_wave_detection_mode(market_regime),
            "correction_detection_sensitivity": self._calculate_correction_sensitivity(market_regime)
        }

        # Parameters for Multi-Timeframe Confluence Strategy
        advanced_params["multi_timeframe_confluence"] = {
            "timeframe_weights": self._calculate_timeframe_weights(market_regime),
            "minimum_confluence_score": self._calculate_min_confluence_score(market_regime, regime_certainty)
        }

        # Parameters for Advanced Breakout Strategy
        advanced_params["breakout"] = {
            "breakout_confirmation_periods": self._calculate_confirmation_periods(market_regime),
            "fibonacci_level_importance": self._calculate_fib_level_importance(market_regime)
        }

        # Parameters for Adaptive MA Strategy
        advanced_params["adaptive_ma"] = {
            "fast_period_adjustment": self._calculate_ma_period_adjustment(market_regime, "fast"),
            "slow_period_adjustment": self._calculate_ma_period_adjustment(market_regime, "slow"),
            "ma_type": self._get_preferred_ma_type(market_regime)
        }

        return advanced_params

    def _calculate_statistical_significance(
        self,
        tool_metrics: Dict[str, Dict[str, Any]],
        market_regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Calculate statistical significance of tool metrics.

        Args:
            tool_metrics: Tool effectiveness metrics
            market_regime: Current market regime

        Returns:
            Dictionary with statistical significance scores
        """
        significance = {}

        for tool_id, metrics in tool_metrics.items():
            # Basic significance based on sample size
            sample_size = metrics.get("signal_count", 0)

            # Use AdaptiveWeightCalculator's method
            significance[tool_id] = self.weight_calculator.calculate_statistical_significance(sample_size)

        return significance

    def _calculate_harmonic_sensitivity(self, market_regime: MarketRegime) -> float:
        """Calculate sensitivity for harmonic pattern detection based on market regime"""
        sensitivity_map = {
            MarketRegime.TRENDING_UP: 0.7,    # Less sensitive in trending markets (avoid false patterns)
            MarketRegime.TRENDING_DOWN: 0.7,  # Less sensitive in trending markets
            MarketRegime.RANGING: 0.9,        # More sensitive in ranging markets (more valid patterns)
            MarketRegime.VOLATILE: 0.5,       # Least sensitive in volatile markets (many false patterns)
            MarketRegime.CHOPPY: 0.6,         # Less sensitive in choppy markets
            MarketRegime.BREAKOUT: 0.7,       # Moderate sensitivity in breakout markets
            MarketRegime.UNKNOWN: 0.7         # Default moderate sensitivity
        }

        return sensitivity_map.get(market_regime, 0.7)

    def _calculate_confirmation_threshold(
        self,
        market_regime: MarketRegime,
        regime_certainty: float
    ) -> float:
        """Calculate confirmation threshold based on market regime and certainty"""
        base_threshold_map = {
            MarketRegime.TRENDING_UP: 0.6,
            MarketRegime.TRENDING_DOWN: 0.6,
            MarketRegime.RANGING: 0.7,
            MarketRegime.VOLATILE: 0.85,
            MarketRegime.CHOPPY: 0.8,
            MarketRegime.BREAKOUT: 0.75,
            MarketRegime.UNKNOWN: 0.75
        }

        base_threshold = base_threshold_map.get(market_regime, 0.75)

        # Adjust based on regime certainty
        # Less certain = higher threshold (more conservative)
        adjustment = (1.0 - regime_certainty) * 0.1

        return base_threshold + adjustment

    def _get_wave_detection_mode(self, market_regime: MarketRegime) -> str:
        """Get preferred Elliott Wave detection mode based on market regime"""
        mode_map = {
            MarketRegime.TRENDING_UP: "impulse_focus",
            MarketRegime.TRENDING_DOWN: "impulse_focus",
            MarketRegime.RANGING: "correction_focus",
            MarketRegime.VOLATILE: "adaptive",
            MarketRegime.CHOPPY: "correction_focus",
            MarketRegime.BREAKOUT: "impulse_focus",
            MarketRegime.UNKNOWN: "balanced"
        }

        return mode_map.get(market_regime, "balanced")

    def _calculate_correction_sensitivity(self, market_regime: MarketRegime) -> float:
        """Calculate Elliott Wave correction detection sensitivity"""
        sensitivity_map = {
            MarketRegime.TRENDING_UP: 0.6,
            MarketRegime.TRENDING_DOWN: 0.6,
            MarketRegime.RANGING: 0.9,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.CHOPPY: 0.8,
            MarketRegime.BREAKOUT: 0.5,
            MarketRegime.UNKNOWN: 0.7
        }

        return sensitivity_map.get(market_regime, 0.7)

    def _calculate_timeframe_weights(self, market_regime: MarketRegime) -> Dict[str, float]:
        """Calculate weights for different timeframes based on market regime"""
        # This reflects the importance of each timeframe in multi-timeframe analysis
        if market_regime == MarketRegime.TRENDING_UP or market_regime == MarketRegime.TRENDING_DOWN:
            # In trending markets, give more weight to higher timeframes
            return {
                "1m": 0.05,
                "5m": 0.07,
                "15m": 0.10,
                "30m": 0.13,
                "1h": 0.20,
                "4h": 0.25,
                "1d": 0.15,
                "1w": 0.05
            }
        elif market_regime == MarketRegime.RANGING:
            # In ranging markets, give more weight to medium timeframes
            return {
                "1m": 0.05,
                "5m": 0.10,
                "15m": 0.20,
                "30m": 0.25,
                "1h": 0.20,
                "4h": 0.10,
                "1d": 0.07,
                "1w": 0.03
            }
        elif market_regime == MarketRegime.VOLATILE:
            # In volatile markets, give more weight to higher timeframes to filter noise
            return {
                "1m": 0.02,
                "5m": 0.05,
                "15m": 0.08,
                "30m": 0.10,
                "1h": 0.15,
                "4h": 0.30,
                "1d": 0.25,
                "1w": 0.05
            }
        elif market_regime == MarketRegime.CHOPPY:
            # In choppy markets, give more weight to higher timeframes for clear direction
            return {
                "1m": 0.02,
                "5m": 0.03,
                "15m": 0.05,
                "30m": 0.10,
                "1h": 0.15,
                "4h": 0.30,
                "1d": 0.25,
                "1w": 0.10
            }
        elif market_regime == MarketRegime.BREAKOUT:
            # In breakout markets, give more weight to lower timeframes for early detection
            return {
                "1m": 0.15,
                "5m": 0.20,
                "15m": 0.15,
                "30m": 0.15,
                "1h": 0.15,
                "4h": 0.10,
                "1d": 0.07,
                "1w": 0.03
            }

        # Default weights for unknown regime
        return {
            "1m": 0.05,
            "5m": 0.10,
            "15m": 0.15,
            "30m": 0.15,
            "1h": 0.20,
            "4h": 0.20,
            "1d": 0.10,
            "1w": 0.05
        }

    def _calculate_min_confluence_score(
        self,
        market_regime: MarketRegime,
        regime_certainty: float
    ) -> float:
        """Calculate minimum confluence score required for signal generation"""
        base_score_map = {
            MarketRegime.TRENDING_UP: 0.6,
            MarketRegime.TRENDING_DOWN: 0.6,
            MarketRegime.RANGING: 0.7,
            MarketRegime.VOLATILE: 0.8,
            MarketRegime.CHOPPY: 0.75,
            MarketRegime.BREAKOUT: 0.65,
            MarketRegime.UNKNOWN: 0.7
        }

        base_score = base_score_map.get(market_regime, 0.7)

        # Adjust based on regime certainty
        # Less certain = higher threshold (more conservative)
        adjustment = (1.0 - regime_certainty) * 0.1

        return base_score + adjustment

    def _calculate_confirmation_periods(self, market_regime: MarketRegime) -> int:
        """Calculate number of confirmation periods for breakout strategies"""
        periods_map = {
            MarketRegime.TRENDING_UP: 2,
            MarketRegime.TRENDING_DOWN: 2,
            MarketRegime.RANGING: 3,
            MarketRegime.VOLATILE: 4,
            MarketRegime.CHOPPY: 4,
            MarketRegime.BREAKOUT: 2,
            MarketRegime.UNKNOWN: 3
        }

        return periods_map.get(market_regime, 3)

    def _calculate_fib_level_importance(self, market_regime: MarketRegime) -> Dict[str, float]:
        """Calculate importance weights for different Fibonacci levels"""
        if market_regime == MarketRegime.RANGING:
            # In ranging markets, focus on retracement levels
            return {
                "0.382": 0.2,
                "0.5": 0.3,
                "0.618": 0.3,
                "0.786": 0.1,
                "1.0": 0.05,
                "1.272": 0.025,
                "1.618": 0.025
            }
        elif market_regime == MarketRegime.TRENDING_UP or market_regime == MarketRegime.TRENDING_DOWN:
            # In trending markets, focus on extension levels
            return {
                "0.382": 0.05,
                "0.5": 0.1,
                "0.618": 0.15,
                "0.786": 0.1,
                "1.0": 0.15,
                "1.272": 0.25,
                "1.618": 0.2
            }
        elif market_regime == MarketRegime.BREAKOUT:
            # In breakout markets, focus on extension levels
            return {
                "0.382": 0.05,
                "0.5": 0.05,
                "0.618": 0.1,
                "0.786": 0.1,
                "1.0": 0.2,
                "1.272": 0.25,
                "1.618": 0.25
            }

        # Default balanced weights
        return {
            "0.382": 0.15,
            "0.5": 0.15,
            "0.618": 0.2,
            "0.786": 0.1,
            "1.0": 0.15,
            "1.272": 0.15,
            "1.618": 0.1
        }

    def _calculate_ma_period_adjustment(
        self,
        market_regime: MarketRegime,
        ma_type: str
    ) -> int:
        """Calculate period adjustment for moving averages"""
        # Adjust MA periods based on market regime
        if ma_type == "fast":
            # Fast MA adjustments
            adjustment_map = {
                MarketRegime.TRENDING_UP: -1,      # Shorter in trending markets
                MarketRegime.TRENDING_DOWN: -1,    # Shorter in trending markets
                MarketRegime.RANGING: 0,           # Standard in ranging markets
                MarketRegime.VOLATILE: 2,          # Longer in volatile markets
                MarketRegime.CHOPPY: 3,            # Longest in choppy markets
                MarketRegime.BREAKOUT: -2,         # Shortest in breakout markets
                MarketRegime.UNKNOWN: 0            # Standard for unknown
            }
        else:
            # Slow MA adjustments
            adjustment_map = {
                MarketRegime.TRENDING_UP: -2,      # Shorter in trending markets
                MarketRegime.TRENDING_DOWN: -2,    # Shorter in trending markets
                MarketRegime.RANGING: 0,           # Standard in ranging markets
                MarketRegime.VOLATILE: 3,          # Longer in volatile markets
                MarketRegime.CHOPPY: 5,            # Longest in choppy markets
                MarketRegime.BREAKOUT: -3,         # Shortest in breakout markets
                MarketRegime.UNKNOWN: 0            # Standard for unknown
            }

        return adjustment_map.get(market_regime, 0)

    def _get_preferred_ma_type(self, market_regime: MarketRegime) -> str:
        """Get preferred moving average type based on market regime"""
        ma_type_map = {
            MarketRegime.TRENDING_UP: "ema",      # EMA responds faster in trends
            MarketRegime.TRENDING_DOWN: "ema",    # EMA responds faster in trends
            MarketRegime.RANGING: "sma",          # SMA more stable in ranges
            MarketRegime.VOLATILE: "wma",         # WMA balances responsiveness and smoothing
            MarketRegime.CHOPPY: "smma",          # Smoothed MA best for choppy markets
            MarketRegime.BREAKOUT: "ema",         # EMA responds faster to breakouts
            MarketRegime.UNKNOWN: "sma"           # SMA is a good default
        }

        return ma_type_map.get(market_regime, "sma")
