"""
RL-Based Risk Parameter Optimization

This module provides components that dynamically adjust risk parameters
based on insights from reinforcement learning models, market conditions,
and prediction confidence levels.

Key components:
- RLRiskParameterOptimizer: Core optimization engine
- RiskRegimeDetector: Detects current risk regime based on market conditions
- ConfidenceBasedPositionSizer: Adjusts position sizes based on model confidence
- AdaptiveRiskParameterService: Service interface for other components
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
import json
import os

from core_foundations.utils.logger import get_logger
from core_foundations.models.order import Order, OrderSide, OrderType
from risk_management_service.models.risk_profile import RiskProfile, RiskLimit
from risk_management_service.models.risk_check import RiskCheckResult
from risk_management_service.orchestration.risk_check_orchestrator import RiskCheckOrchestrator
from common_lib.simulation.interfaces import MarketRegimeType

logger = get_logger(__name__)

class RLRiskParameterOptimizer:
    """
    Core engine for optimizing risk parameters based on reinforcement learning insights.

    This class uses RL model predictions, confidence scores, and market regime information
    to recommend optimal risk parameters like position sizes, stop distances, and risk limits.
    """

    def __init__(
        self,
        base_risk_profile: RiskProfile,
        risk_limits: Dict[str, RiskLimit],
        rl_confidence_threshold: float = 0.7,
        max_position_adjustment: float = 0.3,
        max_stop_adjustment: float = 0.2,
        volatility_lookback_periods: int = 20
    ):
        """
        Initialize the RL Risk Parameter Optimizer.

        Args:
            base_risk_profile: Base risk profile with default settings
            risk_limits: Dictionary of risk limits by category
            rl_confidence_threshold: Minimum confidence level to apply RL suggestions
            max_position_adjustment: Maximum adjustment to position size (as fraction)
            max_stop_adjustment: Maximum adjustment to stop loss distance (as fraction)
            volatility_lookback_periods: Number of periods to consider for volatility
        """
        self.base_risk_profile = base_risk_profile
        self.risk_limits = risk_limits
        self.rl_confidence_threshold = rl_confidence_threshold
        self.max_position_adjustment = max_position_adjustment
        self.max_stop_adjustment = max_stop_adjustment
        self.volatility_lookback_periods = volatility_lookback_periods

        # Keep track of parameter history
        self.parameter_history = []

        logger.info(f"Initialized RL Risk Parameter Optimizer with confidence threshold {rl_confidence_threshold}")

    def optimize_position_size(
        self,
        symbol: str,
        current_price: float,
        base_position_size: float,
        rl_confidence_score: float,
        market_volatility: float,
        market_regime: MarketRegimeType,
        account_balance: float
    ) -> float:
        """
        Optimize the position size based on RL model confidence and market conditions.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            current_price: Current market price
            base_position_size: Default position size (in lots)
            rl_confidence_score: Confidence score from RL model (0-1)
            market_volatility: Current market volatility (normalized)
            market_regime: Current market regime
            account_balance: Current account balance

        Returns:
            Optimized position size
        """
        # Apply confidence adjustment only if above threshold
        if rl_confidence_score >= self.rl_confidence_threshold:
            # Scale position size based on confidence
            confidence_factor = self._calculate_confidence_factor(rl_confidence_score)
            position_size = base_position_size * confidence_factor
        else:
            # Use base position size if confidence is low
            position_size = base_position_size

        # Apply market regime adjustment
        regime_factor = self._get_regime_position_factor(market_regime)
        position_size *= regime_factor

        # Apply volatility adjustment (reduce size when volatility is high)
        volatility_factor = 1.0 / (1.0 + market_volatility)
        position_size *= volatility_factor

        # Ensure position size doesn't exceed risk limits
        max_allowed = self._calculate_max_position_size(symbol, current_price, account_balance)
        position_size = min(position_size, max_allowed)

        # Record the adjustment
        self._record_parameter_adjustment(
            "position_size",
            base_position_size,
            position_size,
            {
                "confidence_score": rl_confidence_score,
                "confidence_factor": confidence_factor if rl_confidence_score >= self.rl_confidence_threshold else 1.0,
                "regime_factor": regime_factor,
                "volatility_factor": volatility_factor,
                "market_regime": str(market_regime)
            }
        )

        return position_size

    def optimize_stop_loss_distance(
        self,
        symbol: str,
        order_side: OrderSide,
        current_price: float,
        base_stop_distance: float,
        rl_predicted_volatility: float,
        market_volatility: float,
        market_regime: MarketRegimeType
    ) -> float:
        """
        Optimize the stop loss distance based on RL predictions and market conditions.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            order_side: Direction of the trade (BUY or SELL)
            current_price: Current market price
            base_stop_distance: Default stop distance in pips
            rl_predicted_volatility: Volatility prediction from RL model
            market_volatility: Current market volatility (normalized)
            market_regime: Current market regime

        Returns:
            Optimized stop loss distance (in pips)
        """
        # Compare RL predicted volatility with current market volatility
        if rl_predicted_volatility > 0:
            # Use the ratio to adjust stop distance
            volatility_ratio = rl_predicted_volatility / max(market_volatility, 0.001)
            volatility_factor = min(max(volatility_ratio, 0.8), 1.5)  # Limit adjustment range
        else:
            volatility_factor = 1.0

        # Apply market regime adjustment
        regime_factor = self._get_regime_stop_factor(market_regime)

        # Calculate adjusted stop distance
        adjusted_stop = base_stop_distance * volatility_factor * regime_factor

        # Ensure the adjustment is within limits
        max_adjustment = base_stop_distance * self.max_stop_adjustment
        if adjusted_stop > base_stop_distance + max_adjustment:
            adjusted_stop = base_stop_distance + max_adjustment
        elif adjusted_stop < base_stop_distance - max_adjustment:
            adjusted_stop = base_stop_distance - max_adjustment

        # Record the adjustment
        self._record_parameter_adjustment(
            "stop_distance",
            base_stop_distance,
            adjusted_stop,
            {
                "volatility_factor": volatility_factor,
                "regime_factor": regime_factor,
                "market_regime": str(market_regime)
            }
        )

        return adjusted_stop

    def optimize_take_profit_distance(
        self,
        symbol: str,
        order_side: OrderSide,
        current_price: float,
        stop_loss_distance: float,
        base_risk_reward: float,
        rl_predicted_direction_strength: float,
        market_regime: MarketRegimeType
    ) -> float:
        """
        Optimize the take profit distance based on RL predictions and risk/reward.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            order_side: Direction of the trade (BUY or SELL)
            current_price: Current market price
            stop_loss_distance: Stop loss distance in pips
            base_risk_reward: Default risk/reward ratio
            rl_predicted_direction_strength: Strength of directional prediction (0-1)
            market_regime: Current market regime

        Returns:
            Optimized take profit distance (in pips)
        """
        # Adjust risk/reward based on direction strength
        if rl_predicted_direction_strength >= self.rl_confidence_threshold:
            # Higher confidence = higher risk/reward target
            direction_factor = 1.0 + (rl_predicted_direction_strength - self.rl_confidence_threshold)
        else:
            # Lower confidence = default or lower risk/reward
            direction_factor = max(0.8, rl_predicted_direction_strength / self.rl_confidence_threshold)

        # Apply market regime adjustment
        regime_factor = self._get_regime_target_factor(market_regime)

        # Calculate adjusted risk/reward ratio
        adjusted_risk_reward = base_risk_reward * direction_factor * regime_factor

        # Calculate take profit distance based on stop loss and risk/reward
        take_profit_distance = stop_loss_distance * adjusted_risk_reward

        # Record the adjustment
        self._record_parameter_adjustment(
            "take_profit_distance",
            stop_loss_distance * base_risk_reward,
            take_profit_distance,
            {
                "direction_factor": direction_factor,
                "regime_factor": regime_factor,
                "adjusted_risk_reward": adjusted_risk_reward,
                "market_regime": str(market_regime)
            }
        )

        return take_profit_distance

    def optimize_risk_limits(
        self,
        default_risk_limits: Dict[str, float],
        market_regime: MarketRegimeType,
        market_volatility: float,
        rl_risk_assessment: float = None
    ) -> Dict[str, float]:
        """
        Optimize risk limits based on market conditions and RL assessments.

        Args:
            default_risk_limits: Default risk limits by category
            market_regime: Current market regime
            market_volatility: Current market volatility (normalized)
            rl_risk_assessment: Risk level assessment from RL model (0-1, optional)

        Returns:
            Optimized risk limits
        """
        optimized_limits = {}

        # Apply market regime adjustment
        regime_factor = self._get_regime_risk_limit_factor(market_regime)

        # Apply volatility adjustment (tighter limits when volatility is high)
        volatility_factor = 1.0 / (1.0 + market_volatility * 0.5)

        # Apply RL risk assessment if available
        if rl_risk_assessment is not None and rl_risk_assessment >= self.rl_confidence_threshold:
            # Higher risk assessment = tighter limits
            rl_factor = 1.0 - ((rl_risk_assessment - self.rl_confidence_threshold) / (1.0 - self.rl_confidence_threshold) * 0.3)
        else:
            rl_factor = 1.0

        # Calculate combined adjustment factor
        combined_factor = regime_factor * volatility_factor * rl_factor

        # Apply adjustment to each limit
        for limit_name, limit_value in default_risk_limits.items():
            optimized_limits[limit_name] = limit_value * combined_factor

        return optimized_limits

    def _calculate_confidence_factor(self, confidence_score: float) -> float:
        """Calculate position size adjustment factor based on confidence score."""
        # Scale confidence to an adjustment factor
        # At threshold: factor = 1.0 (no adjustment)
        # At max confidence (1.0): factor = 1.0 + max_adjustment
        if confidence_score <= self.rl_confidence_threshold:
            return 1.0

        adjustment_range = confidence_score - self.rl_confidence_threshold
        adjustment_percentage = adjustment_range / (1.0 - self.rl_confidence_threshold)
        factor = 1.0 + (adjustment_percentage * self.max_position_adjustment)

        return factor

    def _get_regime_position_factor(self, regime: MarketRegimeType) -> float:
        """Get position size adjustment factor based on market regime."""
        # Define adjustment factors for different regimes
        regime_factors = {
            MarketRegimeType.TRENDING_BULLISH: 1.2,    # More aggressive in strong trends
            MarketRegimeType.TRENDING_BEARISH: 1.2,
            MarketRegimeType.RANGING: 0.9,             # More conservative in ranges
            MarketRegimeType.VOLATILE: 0.7,            # Much more conservative in volatile markets
            MarketRegimeType.BREAKOUT: 1.1,            # Slightly more aggressive in breakouts
            MarketRegimeType.REVERSAL: 0.8,            # More conservative in reversals
            MarketRegimeType.NORMAL: 1.0,              # Neutral in normal markets
            MarketRegimeType.LIQUIDITY_CRISIS: 0.5     # Very conservative in crisis
        }

        return regime_factors.get(regime, 1.0)

    def _get_regime_stop_factor(self, regime: MarketRegimeType) -> float:
        """Get stop loss adjustment factor based on market regime."""
        # Define adjustment factors for different regimes
        regime_factors = {
            MarketRegimeType.TRENDING_BULLISH: 0.9,    # Tighter stops in trends
            MarketRegimeType.TRENDING_BEARISH: 0.9,
            MarketRegimeType.RANGING: 1.1,             # Wider stops in ranges
            MarketRegimeType.VOLATILE: 1.3,            # Much wider stops in volatile markets
            MarketRegimeType.BREAKOUT: 1.2,            # Wider stops in breakouts
            MarketRegimeType.REVERSAL: 1.0,            # Normal stops in reversals
            MarketRegimeType.NORMAL: 1.0,              # Neutral in normal markets
            MarketRegimeType.LIQUIDITY_CRISIS: 1.5     # Very wide stops in crisis
        }

        return regime_factors.get(regime, 1.0)

    def _get_regime_target_factor(self, regime: MarketRegimeType) -> float:
        """Get take profit adjustment factor based on market regime."""
        # Define adjustment factors for different regimes
        regime_factors = {
            MarketRegimeType.TRENDING_BULLISH: 1.2,    # Wider targets in trends
            MarketRegimeType.TRENDING_BEARISH: 1.2,
            MarketRegimeType.RANGING: 0.8,             # Tighter targets in ranges
            MarketRegimeType.VOLATILE: 1.0,            # Normal targets in volatile markets
            MarketRegimeType.BREAKOUT: 1.3,            # Wider targets in breakouts
            MarketRegimeType.REVERSAL: 1.1,            # Slightly wider targets in reversals
            MarketRegimeType.NORMAL: 1.0,              # Neutral in normal markets
            MarketRegimeType.LIQUIDITY_CRISIS: 0.7     # Tighter targets in crisis
        }

        return regime_factors.get(regime, 1.0)

    def _get_regime_risk_limit_factor(self, regime: MarketRegimeType) -> float:
        """Get risk limit adjustment factor based on market regime."""
        # Define adjustment factors for different regimes
        regime_factors = {
            MarketRegimeType.TRENDING_BULLISH: 1.1,    # Higher limits in trends
            MarketRegimeType.TRENDING_BEARISH: 1.1,
            MarketRegimeType.RANGING: 1.0,             # Normal limits in ranges
            MarketRegimeType.VOLATILE: 0.8,            # Lower limits in volatile markets
            MarketRegimeType.BREAKOUT: 0.9,            # Slightly lower limits in breakouts
            MarketRegimeType.REVERSAL: 0.9,            # Slightly lower limits in reversals
            MarketRegimeType.NORMAL: 1.0,              # Neutral in normal markets
            MarketRegimeType.LIQUIDITY_CRISIS: 0.5     # Much lower limits in crisis
        }

        return regime_factors.get(regime, 1.0)

    def _calculate_max_position_size(self, symbol: str, price: float, account_balance: float) -> float:
        """Calculate maximum position size based on risk limits and account balance."""
        # Get risk limit for this symbol
        risk_limit = self.risk_limits.get(symbol, self.risk_limits.get('default', RiskLimit(max_position_size=0.01)))

        # Calculate max position size as percentage of account
        account_based_max = account_balance * risk_limit.max_account_risk_percent / 100.0 / price

        # Respect absolute maximum
        return min(account_based_max, risk_limit.max_position_size)

    def _record_parameter_adjustment(
        self,
        parameter_name: str,
        base_value: float,
        adjusted_value: float,
        metadata: Dict[str, Any]
    ) -> None:
        """Record a parameter adjustment for analysis and auditing."""
        adjustment = {
            "timestamp": datetime.now().isoformat(),
            "parameter": parameter_name,
            "base_value": base_value,
            "adjusted_value": adjusted_value,
            "adjustment_percent": ((adjusted_value / base_value) - 1.0) * 100 if base_value > 0 else 0,
            "metadata": metadata
        }

        self.parameter_history.append(adjustment)

        # Keep history manageable
        if len(self.parameter_history) > 1000:
            self.parameter_history = self.parameter_history[-1000:]


class RiskRegimeDetector:
    """
    Detects the current risk regime based on market indicators and conditions.

    This component analyzes market volatility, correlation patterns, and other
    factors to determine the current risk environment, which influences the
    risk parameter optimization.
    """

    def __init__(
        self,
        volatility_window: int = 20,
        correlation_window: int = 50,
        market_stress_indicators: List[str] = None,
        volatility_threshold_high: float = 1.5,
        volatility_threshold_low: float = 0.5
    ):
        """
        Initialize the Risk Regime Detector.

        Args:
            volatility_window: Window size for volatility calculation
            correlation_window: Window size for correlation calculation
            market_stress_indicators: List of market stress indicators to monitor
            volatility_threshold_high: Threshold for high volatility classification
            volatility_threshold_low: Threshold for low volatility classification
        """
        self.volatility_window = volatility_window
        self.correlation_window = correlation_window
        self.market_stress_indicators = market_stress_indicators or ["VIX", "TED_SPREAD", "FINANCIAL_STRESS_INDEX"]
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low

        # Risk regime history
        self.regime_history = []

        logger.info(f"Initialized Risk Regime Detector with volatility window {volatility_window}")

    def detect_risk_regime(
        self,
        recent_price_data: Dict[str, List[float]],
        market_volatility: float,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        stress_indicators: Optional[Dict[str, float]] = None,
        volatility_z_score: Optional[float] = None
    ) -> str:
        """
        Detect the current risk regime based on market data.

        Args:
            recent_price_data: Dictionary of recent price data by symbol
            market_volatility: Current market volatility level
            correlation_matrix: Correlation matrix between symbols (optional)
            stress_indicators: Market stress indicator values (optional)
            volatility_z_score: Z-score of current volatility (optional)

        Returns:
            Detected risk regime
        """
        # Analyze volatility level
        if volatility_z_score is None:
            volatility_z_score = self._calculate_volatility_zscore(market_volatility)

        if market_volatility > self.volatility_threshold_high or volatility_z_score > 2.0:
            volatility_regime = "HIGH"
        elif market_volatility < self.volatility_threshold_low or volatility_z_score < -0.5:
            volatility_regime = "LOW"
        else:
            volatility_regime = "NORMAL"

        # Analyze correlation structure
        if correlation_matrix:
            correlation_regime = self._analyze_correlation_structure(correlation_matrix)
        else:
            correlation_regime = "UNKNOWN"

        # Check for market stress
        if stress_indicators:
            stress_level = self._calculate_stress_level(stress_indicators)
            if stress_level > 0.7:
                stress_regime = "HIGH_STRESS"
            elif stress_level > 0.4:
                stress_regime = "MODERATE_STRESS"
            else:
                stress_regime = "LOW_STRESS"
        else:
            stress_regime = "UNKNOWN"

        # Determine overall risk regime
        if volatility_regime == "HIGH" or stress_regime == "HIGH_STRESS":
            risk_regime = "HIGH_RISK"
        elif volatility_regime == "LOW" and stress_regime != "MODERATE_STRESS":
            risk_regime = "LOW_RISK"
        else:
            risk_regime = "NORMAL_RISK"

        # Record the regime detection
        self._record_regime_detection(
            risk_regime,
            {
                "volatility_regime": volatility_regime,
                "correlation_regime": correlation_regime,
                "stress_regime": stress_regime,
                "volatility_level": market_volatility,
                "volatility_z_score": volatility_z_score
            }
        )

        return risk_regime

    def _calculate_volatility_zscore(self, current_volatility: float) -> float:
        """Calculate z-score of current volatility relative to history."""
        if not self.regime_history or len(self.regime_history) < 3:
            return 0.0

        # Extract volatility values from history
        volatility_history = [record["metadata"]["volatility_level"] for record in self.regime_history
                             if "volatility_level" in record["metadata"]]

        if not volatility_history:
            return 0.0

        # Calculate mean and standard deviation
        mean_volatility = np.mean(volatility_history)
        std_volatility = np.std(volatility_history)

        # Calculate z-score
        if std_volatility > 0:
            z_score = (current_volatility - mean_volatility) / std_volatility
        else:
            z_score = 0.0

        return z_score

    def _analyze_correlation_structure(self, correlation_matrix: Dict[str, Dict[str, float]]) -> str:
        """Analyze the correlation structure between symbols."""
        # Extract correlation values (upper triangle only)
        correlation_values = []
        symbols = list(correlation_matrix.keys())

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                    correlation_values.append(abs(correlation_matrix[symbol1][symbol2]))

        if not correlation_values:
            return "UNKNOWN"

        # Calculate average absolute correlation
        avg_correlation = np.mean(correlation_values)

        # Determine correlation regime
        if avg_correlation > 0.7:
            return "HIGH_CORRELATION"
        elif avg_correlation < 0.3:
            return "LOW_CORRELATION"
        else:
            return "MODERATE_CORRELATION"

    def _calculate_stress_level(self, stress_indicators: Dict[str, float]) -> float:
        """Calculate overall market stress level from indicators."""
        # Define normal ranges and weights for each indicator
        indicator_ranges = {
            "VIX": {"normal_mean": 15, "normal_std": 5, "weight": 0.4},
            "TED_SPREAD": {"normal_mean": 0.3, "normal_std": 0.1, "weight": 0.3},
            "FINANCIAL_STRESS_INDEX": {"normal_mean": 0, "normal_std": 1, "weight": 0.3}
        }

        # Calculate weighted stress level
        total_weight = 0
        weighted_stress = 0

        for indicator, value in stress_indicators.items():
            if indicator in indicator_ranges:
                # Calculate how many standard deviations from normal
                params = indicator_ranges[indicator]
                z_score = abs(value - params["normal_mean"]) / params["normal_std"]

                # Convert to 0-1 stress level (capped at 3 std devs)
                indicator_stress = min(z_score / 3, 1.0)

                # Add to weighted sum
                weighted_stress += indicator_stress * params["weight"]
                total_weight += params["weight"]

        # Return normalized stress level
        if total_weight > 0:
            return weighted_stress / total_weight
        else:
            return 0.0

    def _record_regime_detection(self, regime: str, metadata: Dict[str, Any]) -> None:
        """Record a regime detection for history and analysis."""
        detection = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "metadata": metadata
        }

        self.regime_history.append(detection)

        # Keep history manageable
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]


class ConfidenceBasedPositionSizer:
    """
    Adjusts position sizes based on model confidence, market conditions, and risk constraints.

    This component translates RL model confidence scores into specific position sizing
    decisions while respecting risk limits and accounting for market conditions.
    """

    def __init__(
        self,
        risk_optimizer: RLRiskParameterOptimizer,
        min_confidence_threshold: float = 0.6,
        max_position_multiplier: float = 3.0,
        enable_progressive_sizing: bool = True
    ):
        """
        Initialize the Confidence-Based Position Sizer.

        Args:
            risk_optimizer: Risk parameter optimizer instance
            min_confidence_threshold: Minimum confidence to increase position size
            max_position_multiplier: Maximum position size multiplier
            enable_progressive_sizing: Whether to enable progressive position sizing
        """
        self.risk_optimizer = risk_optimizer
        self.min_confidence_threshold = min_confidence_threshold
        self.max_position_multiplier = max_position_multiplier
        self.enable_progressive_sizing = enable_progressive_sizing

        # Position sizing history
        self.sizing_history = []

        logger.info(f"Initialized Confidence-Based Position Sizer with threshold {min_confidence_threshold}")

    def calculate_position_size(
        self,
        symbol: str,
        base_size: float,
        confidence_score: float,
        market_regime: MarketRegimeType,
        current_price: float,
        volatility_ratio: float,
        account_balance: float,
        risk_per_trade_percent: float = 0.02
    ) -> float:
        """
        Calculate position size based on confidence and market conditions.

        Args:
            symbol: Trading symbol
            base_size: Base position size
            confidence_score: Model confidence score (0-1)
            market_regime: Current market regime
            current_price: Current market price
            volatility_ratio: Current volatility relative to normal
            account_balance: Current account balance
            risk_per_trade_percent: Base risk per trade as percentage

        Returns:
            Calculated position size
        """
        # Calculate risk-based position size
        risk_amount = account_balance * risk_per_trade_percent

        # Delegate to risk optimizer for core calculation
        position_size = self.risk_optimizer.optimize_position_size(
            symbol=symbol,
            current_price=current_price,
            base_position_size=base_size,
            rl_confidence_score=confidence_score,
            market_volatility=volatility_ratio,
            market_regime=market_regime,
            account_balance=account_balance
        )

        # Apply progressive sizing if enabled
        if self.enable_progressive_sizing:
            position_size = self._apply_progressive_sizing(position_size, confidence_score)

        # Record the position sizing decision
        self._record_position_sizing(
            symbol, base_size, position_size, confidence_score, market_regime, volatility_ratio
        )

        return position_size

    def calculate_size_for_signals(
        self,
        symbol: str,
        signals: List[Dict[str, Any]],
        max_total_size: float,
        account_balance: float,
        current_price: float,
        market_regime: MarketRegimeType,
        volatility_ratio: float
    ) -> List[Dict[str, Any]]:
        """
        Calculate position sizes for multiple signals, respecting overall risk limits.

        Args:
            symbol: Trading symbol
            signals: List of trading signals with confidence scores
            max_total_size: Maximum total position size allowed
            account_balance: Current account balance
            current_price: Current market price
            market_regime: Current market regime
            volatility_ratio: Current volatility relative to normal

        Returns:
            List of signals with calculated position sizes
        """
        # Sort signals by confidence (descending)
        sorted_signals = sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True)

        # Calculate initial sizes without considering total limit
        for signal in sorted_signals:
            confidence = signal.get('confidence', 0.5)
            base_size = signal.get('base_size', 0.01)

            signal['position_size'] = self.calculate_position_size(
                symbol=symbol,
                base_size=base_size,
                confidence_score=confidence,
                market_regime=market_regime,
                current_price=current_price,
                volatility_ratio=volatility_ratio,
                account_balance=account_balance
            )

        # Check if total size exceeds limit
        total_size = sum(signal['position_size'] for signal in sorted_signals)
        if total_size > max_total_size and total_size > 0:
            # Scale down all position sizes proportionally
            scale_factor = max_total_size / total_size
            for signal in sorted_signals:
                signal['position_size'] *= scale_factor
                signal['size_adjusted'] = True

            logger.info(f"Scaled down position sizes with factor {scale_factor:.2f} to respect total limit")

        return sorted_signals

    def _apply_progressive_sizing(self, position_size: float, confidence_score: float) -> float:
        """Apply progressive position sizing based on confidence score."""
        if confidence_score <= self.min_confidence_threshold:
            return position_size

        # Scale from 1.0 at min threshold to max_multiplier at full confidence
        confidence_range = 1.0 - self.min_confidence_threshold
        if confidence_range > 0:
            size_scale = 1.0 + (self.max_position_multiplier - 1.0) * (confidence_score - self.min_confidence_threshold) / confidence_range
            return position_size * size_scale
        else:
            return position_size

    def _record_position_sizing(
        self,
        symbol: str,
        base_size: float,
        actual_size: float,
        confidence_score: float,
        market_regime: MarketRegimeType,
        volatility_ratio: float
    ) -> None:
        """Record a position sizing decision for analysis."""
        sizing_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "base_size": base_size,
            "actual_size": actual_size,
            "size_factor": actual_size / base_size if base_size > 0 else 1.0,
            "confidence_score": confidence_score,
            "market_regime": str(market_regime),
            "volatility_ratio": volatility_ratio
        }

        self.sizing_history.append(sizing_record)

        # Keep history manageable
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-1000:]


class AdaptiveRiskParameterService:
    """
    Main service interface for RL-based adaptive risk parameter management.

    This service coordinates the operation of all risk parameter optimization
    components and provides a unified interface for other parts of the system.
    """

    def __init__(
        self,
        risk_profile: RiskProfile,
        risk_limits: Dict[str, RiskLimit],
        risk_check_orchestrator: Optional[RiskCheckOrchestrator] = None,
        storage_path: str = None
    ):
        """
        Initialize the Adaptive Risk Parameter Service.

        Args:
            risk_profile: Base risk profile
            risk_limits: Dictionary of risk limits
            risk_check_orchestrator: Risk check orchestrator instance (optional)
            storage_path: Path for storing adjustment history
        """
        self.risk_profile = risk_profile
        self.risk_limits = risk_limits
        self.risk_check_orchestrator = risk_check_orchestrator
        self.storage_path = storage_path or os.path.join("data", "risk_parameters")

        # Create component instances
        self.risk_optimizer = RLRiskParameterOptimizer(risk_profile, risk_limits)
        self.risk_regime_detector = RiskRegimeDetector()
        self.position_sizer = ConfidenceBasedPositionSizer(self.risk_optimizer)

        # Ensure storage directory exists
        if self.storage_path:
            os.makedirs(self.storage_path, exist_ok=True)

        logger.info("Initialized Adaptive Risk Parameter Service")

        # Track active sessions for each symbol
        self.active_sessions = {}

    def start_risk_session(self, symbol: str, account_id: str = None) -> str:
        """
        Start a new risk parameter session for a symbol.

        Args:
            symbol: Trading symbol
            account_id: Account identifier (optional)

        Returns:
            Session ID
        """
        session_id = f"{symbol}_{account_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.active_sessions[session_id] = {
            "symbol": symbol,
            "account_id": account_id,
            "start_time": datetime.now(),
            "market_regime": MarketRegimeType.NORMAL,
            "volatility_ratio": 1.0,
            "risk_regime": "NORMAL_RISK",
            "adjustments": []
        }

        logger.info(f"Started risk session {session_id} for {symbol}")
        return session_id

    def update_market_conditions(
        self,
        session_id: str,
        market_regime: MarketRegimeType,
        volatility_data: Dict[str, float],
        price_data: Dict[str, List[float]] = None,
        stress_indicators: Dict[str, float] = None
    ) -> None:
        """
        Update market conditions for a risk session.

        Args:
            session_id: Risk session ID
            market_regime: Current market regime
            volatility_data: Dictionary of volatility metrics
            price_data: Recent price data (optional)
            stress_indicators: Market stress indicators (optional)
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        # Update market regime
        session["market_regime"] = market_regime

        # Update volatility ratio
        if "current" in volatility_data and "average" in volatility_data and volatility_data["average"] > 0:
            session["volatility_ratio"] = volatility_data["current"] / volatility_data["average"]
        else:
            session["volatility_ratio"] = 1.0

        # Detect risk regime
        if price_data:
            risk_regime = self.risk_regime_detector.detect_risk_regime(
                recent_price_data=price_data,
                market_volatility=session["volatility_ratio"],
                stress_indicators=stress_indicators
            )
            session["risk_regime"] = risk_regime

        logger.debug(f"Updated market conditions for session {session_id}: {market_regime}, vol={session['volatility_ratio']:.2f}")

    def get_optimized_position_size(
        self,
        session_id: str,
        base_size: float,
        confidence_score: float,
        current_price: float,
        account_balance: float
    ) -> float:
        """
        Get optimized position size based on model confidence and market conditions.

        Args:
            session_id: Risk session ID
            base_size: Base position size
            confidence_score: Model confidence score (0-1)
            current_price: Current market price
            account_balance: Current account balance

        Returns:
            Optimized position size
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        # Calculate optimized position size
        position_size = self.position_sizer.calculate_position_size(
            symbol=session["symbol"],
            base_size=base_size,
            confidence_score=confidence_score,
            market_regime=session["market_regime"],
            current_price=current_price,
            volatility_ratio=session["volatility_ratio"],
            account_balance=account_balance
        )

        # Record adjustment
        self._record_adjustment(
            session_id,
            "position_size",
            base_size,
            position_size,
            confidence_score
        )

        return position_size

    def get_optimized_stop_loss(
        self,
        session_id: str,
        order_side: OrderSide,
        base_stop_distance: float,
        current_price: float,
        rl_predicted_volatility: float = None
    ) -> float:
        """
        Get optimized stop loss distance.

        Args:
            session_id: Risk session ID
            order_side: Order side (BUY or SELL)
            base_stop_distance: Base stop distance in pips
            current_price: Current market price
            rl_predicted_volatility: RL-predicted volatility (optional)

        Returns:
            Optimized stop loss distance
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        # Use session volatility if RL prediction not provided
        if rl_predicted_volatility is None:
            rl_predicted_volatility = session["volatility_ratio"]

        # Calculate optimized stop loss
        stop_distance = self.risk_optimizer.optimize_stop_loss_distance(
            symbol=session["symbol"],
            order_side=order_side,
            current_price=current_price,
            base_stop_distance=base_stop_distance,
            rl_predicted_volatility=rl_predicted_volatility,
            market_volatility=session["volatility_ratio"],
            market_regime=session["market_regime"]
        )

        # Record adjustment
        self._record_adjustment(
            session_id,
            "stop_loss",
            base_stop_distance,
            stop_distance,
            rl_predicted_volatility
        )

        return stop_distance

    def get_optimized_take_profit(
        self,
        session_id: str,
        order_side: OrderSide,
        stop_loss_distance: float,
        current_price: float,
        base_risk_reward: float = 2.0,
        rl_direction_strength: float = None
    ) -> float:
        """
        Get optimized take profit distance.

        Args:
            session_id: Risk session ID
            order_side: Order side (BUY or SELL)
            stop_loss_distance: Stop loss distance in pips
            current_price: Current market price
            base_risk_reward: Base risk/reward ratio
            rl_direction_strength: RL-predicted direction strength (optional)

        Returns:
            Optimized take profit distance
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        # Default direction strength if not provided
        if rl_direction_strength is None:
            rl_direction_strength = 0.5

        # Calculate optimized take profit
        take_profit_distance = self.risk_optimizer.optimize_take_profit_distance(
            symbol=session["symbol"],
            order_side=order_side,
            current_price=current_price,
            stop_loss_distance=stop_loss_distance,
            base_risk_reward=base_risk_reward,
            rl_predicted_direction_strength=rl_direction_strength,
            market_regime=session["market_regime"]
        )

        # Record adjustment
        self._record_adjustment(
            session_id,
            "take_profit",
            stop_loss_distance * base_risk_reward,
            take_profit_distance,
            rl_direction_strength
        )

        return take_profit_distance

    def get_optimized_risk_limits(
        self,
        session_id: str,
        default_limits: Dict[str, float] = None,
        rl_risk_assessment: float = None
    ) -> Dict[str, float]:
        """
        Get optimized risk limits for a session.

        Args:
            session_id: Risk session ID
            default_limits: Default risk limits to optimize (optional)
            rl_risk_assessment: RL model risk assessment (optional)

        Returns:
            Optimized risk limits
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]

        # Use system defaults if not provided
        if default_limits is None:
            symbol = session["symbol"]
            if symbol in self.risk_limits:
                limit = self.risk_limits[symbol]
            else:
                limit = self.risk_limits.get('default', RiskLimit(max_position_size=0.01))

            default_limits = {
                "max_position_size": limit.max_position_size,
                "max_account_risk_percent": limit.max_account_risk_percent,
                "max_drawdown_percent": limit.max_drawdown_percent,
                "max_daily_loss_percent": limit.max_daily_loss_percent
            }

        # Get optimized limits
        optimized_limits = self.risk_optimizer.optimize_risk_limits(
            default_risk_limits=default_limits,
            market_regime=session["market_regime"],
            market_volatility=session["volatility_ratio"],
            rl_risk_assessment=rl_risk_assessment
        )

        # Record adjustment
        base_values = {k: v for k, v in default_limits.items()}
        self._record_adjustment(
            session_id,
            "risk_limits",
            base_values,
            optimized_limits,
            rl_risk_assessment if rl_risk_assessment else 0.5
        )

        return optimized_limits

    def end_risk_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a risk parameter session and get summary.

        Args:
            session_id: Risk session ID

        Returns:
            Session summary data
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]
        session["end_time"] = datetime.now()
        session["duration"] = (session["end_time"] - session["start_time"]).total_seconds()

        # Create summary
        summary = {
            "session_id": session_id,
            "symbol": session["symbol"],
            "account_id": session["account_id"],
            "start_time": session["start_time"].isoformat(),
            "end_time": session["end_time"].isoformat(),
            "duration_seconds": session["duration"],
            "market_regime": str(session["market_regime"]),
            "risk_regime": session["risk_regime"],
            "adjustment_count": len(session["adjustments"]),
            "average_position_factor": self._calculate_average_adjustment(session, "position_size"),
            "average_stop_factor": self._calculate_average_adjustment(session, "stop_loss"),
            "average_target_factor": self._calculate_average_adjustment(session, "take_profit")
        }

        # Save session data if storage path is set
        if self.storage_path:
            self._save_session_data(session_id, session)

        # Remove from active sessions
        del self.active_sessions[session_id]

        logger.info(f"Ended risk session {session_id}, duration: {session['duration']:.1f}s")
        return summary

    def _record_adjustment(
        self,
        session_id: str,
        adjustment_type: str,
        base_value: Any,
        adjusted_value: Any,
        model_score: float
    ) -> None:
        """Record a parameter adjustment within a session."""
        session = self.active_sessions[session_id]

        adjustment = {
            "timestamp": datetime.now().isoformat(),
            "type": adjustment_type,
            "base_value": base_value,
            "adjusted_value": adjusted_value,
            "model_score": model_score,
            "market_regime": str(session["market_regime"]),
            "volatility_ratio": session["volatility_ratio"],
            "risk_regime": session["risk_regime"]
        }

        session["adjustments"].append(adjustment)

    def _calculate_average_adjustment(self, session: Dict[str, Any], adjustment_type: str) -> float:
        """Calculate the average adjustment factor for a specific parameter type."""
        adjustments = [adj for adj in session["adjustments"] if adj["type"] == adjustment_type]

        if not adjustments:
            return 1.0

        factors = []
        for adj in adjustments:
            base = adj["base_value"]
            adjusted = adj["adjusted_value"]

            # Handle different data types
            if isinstance(base, dict) and isinstance(adjusted, dict):
                # For dictionaries (like risk limits), calculate average ratio
                ratios = []
                for key in base:
                    if key in adjusted and base[key] > 0:
                        ratios.append(adjusted[key] / base[key])
                if ratios:
                    factors.append(sum(ratios) / len(ratios))
            elif isinstance(base, (int, float)) and isinstance(adjusted, (int, float)) and base > 0:
                factors.append(adjusted / base)

        if factors:
            return sum(factors) / len(factors)
        else:
            return 1.0

    def _save_session_data(self, session_id: str, session: Dict[str, Any]) -> None:
        """Save session data to storage."""
        try:
            # Create a serializable copy of the session data
            serializable_session = {
                "symbol": session["symbol"],
                "account_id": session["account_id"],
                "start_time": session["start_time"].isoformat(),
                "end_time": session["end_time"].isoformat() if "end_time" in session else None,
                "market_regime": str(session["market_regime"]),
                "volatility_ratio": session["volatility_ratio"],
                "risk_regime": session["risk_regime"],
                "adjustments": session["adjustments"]
            }

            # Save to file
            file_path = os.path.join(self.storage_path, f"{session_id}.json")
            with open(file_path, 'w') as f:
                json.dump(serializable_session, f, indent=2)

            logger.debug(f"Saved session data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
