"""
Adaptive Layer Service

This module provides the core functionality for the Adaptive Layer,
allowing strategies to dynamically adjust to changing market conditions
based on effectiveness metrics and market regime detection.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pydantic import BaseModel
from enum import Enum
import asyncio
import json

from analysis_engine.services.tool_effectiveness import (
    ToolEffectivenessTracker, MarketRegime, TimeFrame
)
from analysis_engine.services.enhanced_tool_effectiveness import EnhancedToolEffectivenessTracker
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository


class AdaptationLevel(Enum):
    """Defines how aggressively the system should adapt to changing conditions"""
    NONE = 0        # No adaptation
    CONSERVATIVE = 1  # Minimal changes, very gradual adaptation
    MODERATE = 2    # Balanced adaptation
    AGGRESSIVE = 3   # Rapid adaptation to changing conditions
    EXPERIMENTAL = 4  # Experimental features, highest adaptation rate


class AdaptationContext(BaseModel):
    """Context information for adaptation decisions"""
    current_market_regime: str
    previous_market_regime: Optional[str]
    regime_change_detected: bool
    current_volatility_percentile: float
    current_liquidity_score: float
    trading_session: str  # 'asian', 'european', 'american', 'overlap'
    time_in_regime: int  # minutes
    effectiveness_trends: Dict[str, List[float]]  # tool_id -> recent effectiveness scores
    sample_size_sufficient: bool
    confidence_level: float


class ParameterAdjustment(BaseModel):
    """Represents an adjustment to a strategy parameter"""
    parameter_name: str
    previous_value: Any
    new_value: Any
    adjustment_reason: str
    confidence_level: float
    reversion_threshold: Optional[float] = None  # Threshold to revert this change
    is_experimental: bool = False


class AdaptationResult(BaseModel):
    """Result of an adaptation cycle"""
    adaptation_id: str
    timestamp: datetime
    market_regime: str
    adaptations_applied: List[ParameterAdjustment]
    expected_impact: Dict[str, float]  # Metrics expected to improve
    tools_affected: List[str]
    is_reversion: bool = False  # Whether this adaptation reverses a previous one
    metadata: Dict[str, Any] = {}


class AdaptiveLayerService:
    """
    Core service for the Adaptive Layer that enables strategies to
    dynamically adjust to changing market conditions.
    """
    
    def __init__(
        self, 
        effectiveness_repository: ToolEffectivenessRepository,
        initial_adaptation_level: AdaptationLevel = AdaptationLevel.MODERATE
    ):
        """
        Initialize the Adaptive Layer service
        
        Args:
            effectiveness_repository: Repository for tool effectiveness data
            initial_adaptation_level: Initial adaptation aggressiveness level
        """
        self.effectiveness_repository = effectiveness_repository
        self.adaptation_level = initial_adaptation_level
        self.logger = logging.getLogger(__name__)
        
        # Trackers for effectiveness metrics
        self.enhanced_tracker = EnhancedToolEffectivenessTracker()
        
        # Registry of adaptation handlers for different tools/strategies
        self.adaptation_handlers: Dict[str, Callable] = {}
        
        # History of adaptations for learning
        self.adaptation_history: List[AdaptationResult] = []
        
        # Current parameter settings for each tool/strategy
        self.current_parameters: Dict[str, Dict[str, Any]] = {}
        
        # Parameter constraints (min/max values, etc.)
        self.parameter_constraints: Dict[str, Dict[str, Any]] = {}
        
        # Map of market regimes to optimal parameters (learned over time)
        self.regime_optimal_parameters: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Initialize effectiveness decline detection
        self._initialize_decline_detection()
        
        # Cache for signal weights to avoid recalculation
        self.signal_weights_cache = {}
        self.signal_weights_cache_expiry = {}
        self.signal_weights_cache_duration = timedelta(minutes=15)  # Cache duration
        
        # Enhanced metrics tracking for Phase 4
        self.effectiveness_history = {}
        self.weight_adjustment_factors = {}
        self.recent_performance_window = timedelta(days=3)  # Window for recent performance tracking

    async def get_tool_signal_weights(
        self,
        market_regime: MarketRegime,
        tools: List[str],
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate signal weights based on tool effectiveness metrics for the SignalAggregator
        
        Args:
            market_regime: Current market regime
            tools: List of tools/indicators for which to calculate weights
            timeframe: Trading timeframe
            symbol: Trading instrument symbol
            
        Returns:
            Dictionary mapping tool IDs to weights based on their effectiveness
        """
        # Check cache first
        cache_key = f"{market_regime.value}:{','.join(sorted(tools))}:{timeframe.value if timeframe else 'all'}:{symbol or 'all'}"
        now = datetime.now()
        
        if (cache_key in self.signal_weights_cache and 
            cache_key in self.signal_weights_cache_expiry and
            now < self.signal_weights_cache_expiry[cache_key]):
            self.logger.debug(f"Using cached signal weights for {cache_key}")
            return self.signal_weights_cache[cache_key]
        
        # Calculate weights based on tool effectiveness metrics
        weights = {}
        total_weight = 0.0
        effectiveness_metrics = {}
        
        # Get effectiveness metrics for each tool
        for tool_id in tools:
            metrics = await self.effectiveness_repository.get_tool_effectiveness_metrics_async(
                tool_id=tool_id,
                timeframe=timeframe.value if timeframe else None,
                instrument=symbol,
                market_regime=market_regime
            )
            
            if metrics:
                effectiveness_metrics[tool_id] = metrics
        
        # Calculate weight for each tool based on its effectiveness
        for tool_id, metrics in effectiveness_metrics.items():
            # Calculate weight based on win rate, profit factor, and expected payoff
            win_rate = metrics.get("win_rate", 0.5)
            profit_factor = metrics.get("profit_factor", 1.0)
            expected_payoff = metrics.get("expected_payoff", 0.0)
            
            # Get regime-specific metrics if available
            regime_metrics = metrics.get("regime_metrics", {}).get(market_regime.value, {})
            regime_win_rate = regime_metrics.get("win_rate", win_rate)
            regime_profit_factor = regime_metrics.get("profit_factor", profit_factor)
            
            # Enhanced Phase 4: Calculate effectiveness score using a comprehensive formula
            effectiveness_score = await self._calculate_comprehensive_effectiveness_score(
                tool_id, metrics, market_regime, timeframe, symbol
            )
            
            # Apply recent performance adjustment factor
            recent_factor = self._get_recent_performance_factor(tool_id, market_regime)
            
            # Calculate base weight using the effectiveness score
            base_weight = effectiveness_score * recent_factor
            
            # Adjust weight based on sample size
            sample_size = metrics.get("signal_count", 0)
            sample_size_factor = min(sample_size / 100, 1.0)  # Cap at 100 samples
            
            # Apply sample size factor
            weight = base_weight * (0.5 + 0.5 * sample_size_factor)
            
            # Apply adaptation level factor
            adaptation_factor = {
                AdaptationLevel.NONE: 0.0,  # No adaptation
                AdaptationLevel.CONSERVATIVE: 0.7,
                AdaptationLevel.MODERATE: 1.0,
                AdaptationLevel.AGGRESSIVE: 1.3,
                AdaptationLevel.EXPERIMENTAL: 1.5
            }
            
            weight *= adaptation_factor.get(self.adaptation_level, 1.0)
            
            # Store the calculated weight
            weights[tool_id] = max(0.1, min(weight, 1.0))  # Clamp between 0.1 and 1.0
            total_weight += weights[tool_id]
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # If no effectiveness data, assign equal weights
            equal_weight = 1.0 / len(tools) if tools else 0.0
            weights = {tool_id: equal_weight for tool_id in tools}
        
        # Store in cache
        self.signal_weights_cache[cache_key] = weights
        self.signal_weights_cache_expiry[cache_key] = now + self.signal_weights_cache_duration
        
        self.logger.info(f"Calculated signal weights based on tool effectiveness: {weights}")
        return weights

    async def _calculate_comprehensive_effectiveness_score(
        self, 
        tool_id: str, 
        metrics: Dict[str, Any],
        market_regime: MarketRegime,
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None
    ) -> float:
        """
        Calculate a comprehensive effectiveness score for a tool based on all available metrics
        
        Args:
            tool_id: Tool identifier
            metrics: Tool effectiveness metrics dictionary
            market_regime: Current market regime
            timeframe: Trading timeframe
            symbol: Trading instrument symbol
            
        Returns:
            Effectiveness score between 0.0 and 1.0
        """
        # Extract base metrics
        win_rate = metrics.get("win_rate", 0.5)
        profit_factor = metrics.get("profit_factor", 1.0)
        expected_payoff = metrics.get("expected_payoff", 0.0)
        max_drawdown = metrics.get("max_drawdown", 0.0)
        recovery_factor = metrics.get("recovery_factor", 0.0)
        
        # Extract regime-specific metrics if available
        regime_metrics = metrics.get("regime_metrics", {}).get(market_regime.value, {})
        regime_win_rate = regime_metrics.get("win_rate", win_rate)
        regime_profit_factor = regime_metrics.get("profit_factor", profit_factor)
        regime_expected_payoff = regime_metrics.get("expected_payoff", expected_payoff)
        
        # Get historical trend of effectiveness
        trend_factor = await self._calculate_effectiveness_trend_factor(tool_id, market_regime)
        
        # Calculate component scores
        win_rate_score = regime_win_rate  # Win rate is already 0.0-1.0
        
        # Profit factor score (normalize to 0.0-1.0 range)
        pf_score = min(regime_profit_factor / 3.0, 1.0)
        
        # Expected payoff score (normalize to 0.0-1.0 range)
        ep_score = min(max(regime_expected_payoff, 0.0) / 0.01, 1.0)
        
        # Drawdown penalty (less drawdown is better)
        dd_penalty = max(0.0, min(abs(max_drawdown) / 0.2, 1.0))
        
        # Recovery factor bonus
        rf_bonus = min(recovery_factor / 3.0, 0.15)
        
        # Combine all factors with weightings
        effectiveness_score = (
            0.35 * win_rate_score +
            0.30 * pf_score +
            0.20 * ep_score -
            0.10 * dd_penalty +
            rf_bonus
        )
        
        # Apply trend factor
        effectiveness_score *= trend_factor
        
        # Ensure score is within valid range
        return max(0.1, min(effectiveness_score, 1.0))

    async def _calculate_effectiveness_trend_factor(
        self,
        tool_id: str,
        market_regime: MarketRegime
    ) -> float:
        """
        Calculate a factor based on the trend of a tool's effectiveness
        
        Args:
            tool_id: Tool identifier
            market_regime: Current market regime
            
        Returns:
            Trend factor between 0.8 and 1.2
        """
        # If no history, return neutral factor
        if tool_id not in self.effectiveness_history:
            return 1.0
            
        history = self.effectiveness_history[tool_id]
        
        # If less than 3 data points, return neutral factor
        if len(history) < 3:
            return 1.0
            
        # Get recent history for this regime
        regime_history = [
            entry for entry in history
            if entry.get("market_regime") == market_regime.value
        ]
        
        # If insufficient regime-specific data, use all history
        if len(regime_history) < 3:
            regime_history = history
            
        # Sort by timestamp
        sorted_history = sorted(
            regime_history,
            key=lambda x: x.get("timestamp", datetime.min)
        )
        
        # Get win rate trend
        win_rates = [entry.get("win_rate", 0.5) for entry in sorted_history[-5:]]
        
        # Calculate trend direction
        if len(win_rates) >= 3:
            # Simple linear regression slope
            n = len(win_rates)
            x = list(range(n))
            slope = (n * sum(i*j for i,j in zip(x, win_rates)) - sum(x) * sum(win_rates)) / \
                    (n * sum(i*i for i in x) - sum(x)**2)
                    
            # Convert slope to factor (1.0 is neutral, >1.0 is improving, <1.0 is declining)
            trend_factor = 1.0 + slope * 5.0  # Scale the slope
            
            # Limit the factor range
            return max(0.8, min(trend_factor, 1.2))
        
        return 1.0

    def _get_recent_performance_factor(
        self,
        tool_id: str,
        market_regime: MarketRegime
    ) -> float:
        """
        Get a factor based on very recent performance (last few days)
        
        Args:
            tool_id: Tool identifier
            market_regime: Current market regime
            
        Returns:
            Recent performance factor between 0.8 and 1.2
        """
        # Default to neutral factor
        if tool_id not in self.weight_adjustment_factors:
            return 1.0
            
        # Get regime-specific factor if available
        regime_factors = self.weight_adjustment_factors.get(tool_id, {})
        return regime_factors.get(market_regime.value, 1.0)
        
    async def update_weight_adjustment_factor(
        self,
        tool_id: str,
        market_regime: MarketRegime,
        success: bool,
        impact: float = 0.05
    ) -> None:
        """
        Update weight adjustment factor based on very recent tool performance
        
        Args:
            tool_id: Tool identifier
            market_regime: Current market regime
            success: Whether the tool was successful
            impact: How much to adjust the factor by
        """
        # Initialize if needed
        if tool_id not in self.weight_adjustment_factors:
            self.weight_adjustment_factors[tool_id] = {}
        
        # Get current factor
        current_factor = self.weight_adjustment_factors[tool_id].get(
            market_regime.value, 1.0
        )
        
        # Update factor
        if success:
            # Increase factor for success
            new_factor = min(current_factor + impact, 1.2)
        else:
            # Decrease factor for failure
            new_factor = max(current_factor - impact, 0.8)
            
        # Store updated factor
        self.weight_adjustment_factors[tool_id][market_regime.value] = new_factor
        
        self.logger.debug(
            f"Updated weight adjustment factor for {tool_id} in {market_regime.value}: "
            f"{current_factor:.2f} -> {new_factor:.2f}"
        )

    async def get_aggregator_weights(
        self,
        market_regime: MarketRegime,
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get weights for the SignalAggregator categories based on current market regime
        and tool effectiveness metrics
        
        Args:
            market_regime: Current market regime
            timeframe: Trading timeframe
            symbol: Trading instrument symbol
            
        Returns:
            Dictionary of weights for different signal categories
        """
        # Define base weights for each market regime
        regime_base_weights = {
            MarketRegime.TRENDING: {
                "technical_analysis": 0.55,
                "machine_learning": 0.25,
                "market_regime": 0.15,
                "correlation": 0.05
            },
            MarketRegime.RANGING: {
                "technical_analysis": 0.5,
                "machine_learning": 0.2,
                "market_regime": 0.1,
                "correlation": 0.2
            },
            MarketRegime.VOLATILE: {
                "technical_analysis": 0.4,
                "machine_learning": 0.3,
                "market_regime": 0.25,
                "correlation": 0.05
            },
            MarketRegime.BREAKOUT: {
                "technical_analysis": 0.6,
                "machine_learning": 0.2,
                "market_regime": 0.15,
                "correlation": 0.05
            }
        }
        
        # Get base weights for the current regime
        base_weights = regime_base_weights.get(market_regime, {
            "technical_analysis": 0.5,
            "machine_learning": 0.3,
            "market_regime": 0.1,
            "correlation": 0.1
        })
        
        # Get effectiveness metrics for each category
        categories = list(base_weights.keys())
        category_metrics = {}
        
        for category in categories:
            # Get tools for this category
            category_tools = await self.effectiveness_repository.get_tools_by_category_async(category)
            
            if not category_tools:
                continue
                
            # Get average effectiveness for tools in this category
            effectiveness_sum = 0
            tool_count = 0
            
            for tool_id in category_tools:
                metrics = await self.effectiveness_repository.get_tool_effectiveness_metrics_async(
                    tool_id=tool_id,
                    timeframe=timeframe.value if timeframe else None,
                    instrument=symbol,
                    market_regime=market_regime
                )
                
                if metrics:
                    win_rate = metrics.get("win_rate", 0.5)
                    effectiveness_sum += win_rate
                    tool_count += 1
            
            if tool_count > 0:
                category_metrics[category] = effectiveness_sum / tool_count
        
        # Adjust weights based on effectiveness metrics
        adjusted_weights = {}
        total_adjustment = 0
        
        for category, base_weight in base_weights.items():
            if category in category_metrics:
                # Calculate adjustment factor (higher effectiveness = higher weight)
                effectiveness = category_metrics[category]
                adjustment_factor = 1.0 + (effectiveness - 0.5) * 0.5  # Scale adjustment
                adjusted_weight = base_weight * adjustment_factor
            else:
                adjusted_weight = base_weight
                
            adjusted_weights[category] = adjusted_weight
            total_adjustment += adjusted_weight
        
        # Normalize weights
        if total_adjustment > 0:
            normalized_weights = {k: v / total_adjustment for k, v in adjusted_weights.items()}
        else:
            normalized_weights = base_weights
            
        self.logger.info(f"Calculated aggregator weights for {market_regime.value}: {normalized_weights}")
        return normalized_weights
    
    async def run_adaptation_cycle(
        self,
        market_regime: MarketRegime,
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None,
        lookback_hours: int = 24
    ) -> Dict[str, AdaptationResult]:
        """
        Run an adaptation cycle to adjust parameters based on 
        current market conditions and effectiveness metrics
        
        Args:
            market_regime: Current market regime
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            lookback_hours: Hours of data to consider
            
        Returns:
            Dictionary mapping tool IDs to adaptation results
        """
        try:
            # Skip if adaptation is disabled
            if self.adaptation_level == AdaptationLevel.NONE:
                self.logger.info("Adaptation is disabled (level NONE)")
                return {}
            
            # Get active tools with sufficient data
            active_tool_ids = await self._get_active_tools(lookback_hours)
            
            if not active_tool_ids:
                self.logger.info("No active tools found with sufficient data for adaptation")
                return {}
                
            # Build adaptation context with current market conditions
            context = await self._build_adaptation_context(
                market_regime=market_regime,
                timeframe=timeframe,
                symbol=symbol,
                lookback_hours=lookback_hours
            )
            
            results = {}
            
            # Process each tool that has a registered adaptation handler
            for tool_id in active_tool_ids:
                if tool_id in self.adaptation_handlers:
                    # Get current parameters
                    current_params = self.current_parameters.get(
                        tool_id, self._get_tool_parameters(tool_id)
                    )
                    
                    # Run the adaptation handler
                    handler = self.adaptation_handlers[tool_id]
                    adjustments = handler(context, current_params)
                    
                    # Apply constraints to the adjustments
                    adjustments = self._apply_parameter_constraints(tool_id, adjustments)
                    
                    # Record and apply the adaptations
                    if adjustments:
                        # Create adaptation result
                        result = await self._create_adaptation_result(
                            tool_id=tool_id,
                            market_regime=market_regime.value,
                            adjustments=adjustments,
                            context=context
                        )
                        
                        # Apply the parameter changes
                        self._apply_parameter_adjustments(tool_id, adjustments)
                        
                        # Record adaptation
                        self.adaptation_history.append(result)
                        results[tool_id] = result
                        
                        self.logger.info(
                            f"Applied {len(adjustments)} adaptations to {tool_id} "
                            f"in {market_regime.value} regime"
                        )
                    else:
                        self.logger.info(f"No adaptations needed for {tool_id}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in adaptation cycle: {str(e)}", exc_info=True)
            return {}
            
    async def _get_active_tools(self, lookback_hours: int) -> List[str]:
        """Get active tools with sufficient data for adaptation"""
        try:
            # Query the repository for active tools with data in the lookback period
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            active_tools = await self.effectiveness_repository.get_active_tools_async(
                start_date=start_date,
                end_date=end_date,
                min_sample_count=self.decline_thresholds[self.adaptation_level]["minimum_sample_size"]
            )
            
            return [t.tool_id for t in active_tools]
            
        except Exception as e:
            self.logger.error(f"Error getting active tools: {str(e)}")
            return []
    
    async def _build_adaptation_context(
        self,
        market_regime: MarketRegime,
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None,
        lookback_hours: int = 24
    ) -> AdaptationContext:
        """Build context for adaptation decisions"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=lookback_hours)
        
        # Get previous market regime
        previous_regime_data = await self.effectiveness_repository.get_previous_regime_async(
            current_regime=market_regime.value,
            before_date=end_date,
            max_lookback_hours=lookback_hours * 2  # Look back further for regime history
        )
        
        previous_regime = previous_regime_data.get("regime") if previous_regime_data else None
        
        # Determine current trading session based on time
        trading_session = self._determine_trading_session(datetime.utcnow())
        
        # Get time in current regime
        time_in_regime_minutes = previous_regime_data.get("minutes_since_change", 0) if previous_regime_data else 0
        
        # Get effectiveness trends for active tools
        effectiveness_trends = {}
        active_tool_ids = await self._get_active_tools(lookback_hours)
        
        for tool_id in active_tool_ids:
            metrics = await self.effectiveness_repository.get_tool_metrics_async(
                tool_id=tool_id,
                metric_type="composite_score",
                start_date=start_date,
                end_date=end_date,
                order_by_date=True
            )
            
            trend = [m.value for m in metrics if m.value is not None]
            effectiveness_trends[tool_id] = trend
        
        # Get market volatility and liquidity scores
        market_metrics = await self.effectiveness_repository.get_market_metrics_async(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol
        )
        
        volatility_percentile = market_metrics.get("volatility_percentile", 0.5)
        liquidity_score = market_metrics.get("liquidity_score", 0.5)
        
        # Determine if we have sufficient sample size for confident decisions
        sample_size_sufficient = True  # Default assumption
        min_samples = self.decline_thresholds[self.adaptation_level]["minimum_sample_size"]
        
        for tool_id in active_tool_ids:
            count = await self.effectiveness_repository.count_tool_outcomes_async(
                tool_id=tool_id,
                start_date=start_date,
                end_date=end_date
            )
            if count < min_samples:
                sample_size_sufficient = False
                break
        
        # Calculate confidence level based on sample size and adaptation level
        confidence_base = {
            AdaptationLevel.CONSERVATIVE: 0.7,
            AdaptationLevel.MODERATE: 0.8,
            AdaptationLevel.AGGRESSIVE: 0.85,
            AdaptationLevel.EXPERIMENTAL: 0.6  # Lower confidence for experimental features
        }
        
        confidence_level = confidence_base[self.adaptation_level]
        if not sample_size_sufficient:
            confidence_level *= 0.7  # Reduce confidence with insufficient data
            
        # Build the context object
        context = AdaptationContext(
            current_market_regime=market_regime.value,
            previous_market_regime=previous_regime,
            regime_change_detected=previous_regime != market_regime.value,
            current_volatility_percentile=volatility_percentile,
            current_liquidity_score=liquidity_score,
            trading_session=trading_session,
            time_in_regime=time_in_regime_minutes,
            effectiveness_trends=effectiveness_trends,
            sample_size_sufficient=sample_size_sufficient,
            confidence_level=confidence_level
        )
        
        return context
    
    def _determine_trading_session(self, timestamp: datetime) -> str:
        """Determine the current trading session based on UTC time"""
        # Convert to hour in UTC
        hour = timestamp.hour
        
        # Define trading sessions based on UTC hours
        if 0 <= hour < 8:  # Asian session (approx)
            return "asian"
        elif 8 <= hour < 12:  # Asian-European overlap
            return "asian_european_overlap"
        elif 12 <= hour < 16:  # European session
            return "european"
        elif 16 <= hour < 20:  # European-American overlap
            return "european_american_overlap"
        else:  # 20-24 American session
            return "american"
    
    def _get_tool_parameters(self, tool_id: str) -> Dict[str, Any]:
        """Get current parameters for a tool (fetching from database)"""
        # This would typically fetch from a parameter repository
        # For now, return empty dict if not in our cache
        return self.current_parameters.get(tool_id, {})
    
    def _apply_parameter_constraints(
        self, 
        tool_id: str, 
        adjustments: List[ParameterAdjustment]
    ) -> List[ParameterAdjustment]:
        """Apply constraints to parameter adjustments"""
        if tool_id not in self.parameter_constraints:
            return adjustments  # No constraints defined
            
        constraints = self.parameter_constraints[tool_id]
        valid_adjustments = []
        
        for adj in adjustments:
            param_name = adj.parameter_name
            
            if param_name in constraints:
                constraint = constraints[param_name]
                new_value = adj.new_value
                
                # Apply min/max constraints
                if "min" in constraint and new_value < constraint["min"]:
                    new_value = constraint["min"]
                if "max" in constraint and new_value > constraint["max"]:
                    new_value = constraint["max"]
                    
                # Apply step size
                if "step" in constraint and constraint["step"] > 0:
                    step = constraint["step"]
                    new_value = round(new_value / step) * step
                    
                # Check if value is in allowed values list
                if "values" in constraint and new_value not in constraint["values"]:
                    # Find closest allowed value
                    allowed_values = constraint["values"]
                    new_value = min(allowed_values, key=lambda x: abs(x - new_value))
                
                # Update the adjustment with constrained value
                if new_value != adj.new_value:
                    adj.new_value = new_value
                    adj.adjustment_reason += " (constrained)"
                
            valid_adjustments.append(adj)
        
        return valid_adjustments
    
    def _apply_parameter_adjustments(
        self, 
        tool_id: str, 
        adjustments: List[ParameterAdjustment]
    ) -> None:
        """Apply parameter adjustments to tool"""
        # Ensure we have an entry for this tool
        if tool_id not in self.current_parameters:
            self.current_parameters[tool_id] = {}
            
        # Apply each adjustment
        for adj in adjustments:
            param_name = adj.parameter_name
            new_value = adj.new_value
            
            # Update the parameter
            self.current_parameters[tool_id][param_name] = new_value
    
    async def _create_adaptation_result(
        self,
        tool_id: str,
        market_regime: str,
        adjustments: List[ParameterAdjustment],
        context: AdaptationContext
    ) -> AdaptationResult:
        """Create result object for an adaptation"""
        # Generate a unique ID
        adaptation_id = f"{tool_id}_{market_regime}_{datetime.utcnow().isoformat()}"
        
        # Determine if this is reverting a previous adaptation
        is_reversion = any("revert" in adj.adjustment_reason.lower() for adj in adjustments)
        
        # Calculate expected impact based on historical data and adjustment magnitude
        expected_impact = await self._calculate_expected_impact(tool_id, adjustments, context)
        
        return AdaptationResult(
            adaptation_id=adaptation_id,
            timestamp=datetime.utcnow(),
            market_regime=market_regime,
            adaptations_applied=adjustments,
            expected_impact=expected_impact,
            tools_affected=[tool_id],
            is_reversion=is_reversion,
            metadata={
                "adaptation_level": self.adaptation_level.name,
                "context": context.dict()
            }
        )
    
    async def _calculate_expected_impact(
        self,
        tool_id: str,
        adjustments: List[ParameterAdjustment],
        context: AdaptationContext
    ) -> Dict[str, float]:
        """Calculate expected impact of adaptations on key metrics"""
        # This is a simplified version - in reality would use ML predictions
        # Based on historical data of similar adjustments
        
        # Get baseline metrics
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=24)
        
        baseline_metrics = await self.effectiveness_repository.get_tool_metrics_summary_async(
            tool_id=tool_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate expected improvements based on adjustment magnitude
        # and adaptation confidence
        expected_impact = {}
        total_adjustment_magnitude = sum(
            abs((adj.new_value - adj.previous_value) / max(adj.previous_value, 1)) 
            for adj in adjustments 
            if isinstance(adj.previous_value, (int, float)) and isinstance(adj.new_value, (int, float))
        )
        
        # Scale expected improvements based on adaptation level
        improvement_factor = {
            AdaptationLevel.CONSERVATIVE: 0.02,  # 2% improvement
            AdaptationLevel.MODERATE: 0.05,      # 5% improvement
            AdaptationLevel.AGGRESSIVE: 0.10,    # 10% improvement
            AdaptationLevel.EXPERIMENTAL: 0.15   # 15% potential improvement (but higher risk)
        }[self.adaptation_level]
        
        # Apply confidence level
        improvement_factor *= context.confidence_level
        
        # Calculate expected impact for each metric
        if "win_rate" in baseline_metrics:
            base_win_rate = baseline_metrics["win_rate"]
            max_possible_improvement = min(0.95 - base_win_rate, 0.2)  # Cap at 95% win rate, max 20% improvement
            expected_impact["win_rate"] = min(improvement_factor * total_adjustment_magnitude, max_possible_improvement)
            
        if "profit_factor" in baseline_metrics:
            base_pf = baseline_metrics["profit_factor"]
            expected_impact["profit_factor"] = base_pf * improvement_factor * total_adjustment_magnitude
            
        if "expected_payoff" in baseline_metrics:
            base_payoff = baseline_metrics["expected_payoff"]
            expected_impact["expected_payoff"] = base_payoff * improvement_factor * total_adjustment_magnitude
        
        return expected_impact
    
    async def detect_effectiveness_decline(
        self,
        tool_id: str,
        lookback_hours: int = 24,
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect if a tool's effectiveness is declining
        
        Args:
            tool_id: ID of tool to check
            lookback_hours: Hours to look back for comparison
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            
        Returns:
            Dictionary with decline detection results
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            # Get thresholds based on adaptation level
            thresholds = self.decline_thresholds[self.adaptation_level]
            
            # Get metrics for recent periods
            recent_metrics = await self.effectiveness_repository.get_tool_metrics_async(
                tool_id=tool_id,
                start_date=start_date,
                end_date=end_date,
                order_by_date=True
            )
            
            # Ensure we have enough data
            if len(recent_metrics) < thresholds["lookback_periods"] + 1:
                return {
                    "tool_id": tool_id,
                    "decline_detected": False,
                    "reason": "Insufficient data for comparison"
                }
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in recent_metrics:
                if metric.metric_type not in metrics_by_type:
                    metrics_by_type[metric.metric_type] = []
                metrics_by_type[metric.metric_type].append(metric)
            
            # Check for win rate decline
            decline_detected = False
            decline_details = {}
            
            if "win_rate" in metrics_by_type:
                win_rate_metrics = metrics_by_type["win_rate"]
                
                if len(win_rate_metrics) >= 2:
                    # Compare most recent to previous periods
                    current = win_rate_metrics[-1].value
                    previous_avg = sum(m.value for m in win_rate_metrics[-1-thresholds["lookback_periods"]:-1]) / thresholds["lookback_periods"]
                    
                    if current < previous_avg * (1 - thresholds["win_rate_decline"]):
                        decline_detected = True
                        decline_details["win_rate"] = {
                            "current": current,
                            "previous_avg": previous_avg,
                            "decline_pct": (previous_avg - current) / previous_avg
                        }
            
            # Check for profit factor decline
            if "profit_factor" in metrics_by_type:
                pf_metrics = metrics_by_type["profit_factor"]
                
                if len(pf_metrics) >= 2:
                    # Compare most recent to previous periods
                    current = pf_metrics[-1].value
                    previous_avg = sum(m.value for m in pf_metrics[-1-thresholds["lookback_periods"]:-1]) / thresholds["lookback_periods"]
                    
                    if current < previous_avg * (1 - thresholds["profit_factor_decline"]):
                        decline_detected = True
                        decline_details["profit_factor"] = {
                            "current": current,
                            "previous_avg": previous_avg,
                            "decline_pct": (previous_avg - current) / previous_avg
                        }
            
            return {
                "tool_id": tool_id,
                "decline_detected": decline_detected,
                "details": decline_details,
                "adaptation_level": self.adaptation_level.name,
                "thresholds": thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting effectiveness decline: {str(e)}")
            return {
                "tool_id": tool_id,
                "decline_detected": False,
                "error": str(e)
            }
    
    async def get_optimal_parameters(
        self,
        tool_id: str,
        market_regime: MarketRegime,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get optimal parameters for a tool in the given market regime
        
        Args:
            tool_id: ID of the tool
            market_regime: Current market regime
            context: Additional context (volatility, etc.)
            
        Returns:
            Dictionary of optimal parameters
        """
        # Check if we have learned optimal parameters for this regime
        regime_key = market_regime.value
        
        if regime_key in self.regime_optimal_parameters and tool_id in self.regime_optimal_parameters[regime_key]:
            return self.regime_optimal_parameters[regime_key][tool_id]
        
        # If we don't have specific learned parameters, return current parameters
        return self.current_parameters.get(tool_id, {})
    
    async def record_adaptation_feedback(
        self,
        adaptation_id: str,
        metrics: Dict[str, float],
        success: bool,
        notes: str = ""
    ) -> None:
        """
        Record feedback on an adaptation's actual impact
        
        Args:
            adaptation_id: ID of the adaptation
            metrics: Actual metrics observed after adaptation
            success: Whether the adaptation was successful
            notes: Additional notes
        """
        # Find the adaptation in history
        for adaptation in self.adaptation_history:
            if adaptation.adaptation_id == adaptation_id:
                # Update the adaptation with feedback
                adaptation.metadata["feedback"] = {
                    "actual_metrics": metrics,
                    "success": success,
                    "notes": notes,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # If successful and confident, update regime optimal parameters
                if success and adaptation.metadata.get("context", {}).get("confidence_level", 0) > 0.7:
                    tool_id = adaptation.tools_affected[0]
                    regime = adaptation.market_regime
                    
                    # Get the parameter changes
                    parameter_updates = {
                        adj.parameter_name: adj.new_value
                        for adj in adaptation.adaptations_applied
                    }
                    
                    # Ensure we have an entry for this regime
                    if regime not in self.regime_optimal_parameters:
                        self.regime_optimal_parameters[regime] = {}
                    
                    # Ensure we have an entry for this tool
                    if tool_id not in self.regime_optimal_parameters[regime]:
                        self.regime_optimal_parameters[regime][tool_id] = {}
                    
                    # Update optimal parameters
                    self.regime_optimal_parameters[regime][tool_id].update(parameter_updates)
                    
                    self.logger.info(
                        f"Updated optimal parameters for {tool_id} in {regime} regime "
                        f"based on successful adaptation feedback"
                    )
                
                self.logger.info(f"Recorded feedback for adaptation {adaptation_id}")
                break
