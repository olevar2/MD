"""
Integration between Reinforcement Learning models and Risk Management Service.

This module provides components for adjusting risk parameters based on RL model insights
and adapting to current market conditions.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from risk_management_service.risk_check_orchestrator import RiskCheckOrchestrator
from risk_management_service.models.risk_parameters import RiskParameters, PositionSizingParams, StopLossParams
from common_lib.simulation.interfaces import MarketRegimeType

logger = get_logger(__name__)


class RLRiskParameterSuggester:
    """
    Component that suggests risk parameter adjustments based on RL model insights.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk parameter suggester.

        Args:
            config: Configuration parameters for the suggester
        """
        self.config = config
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)
        self.max_position_size_factor = config.get('max_position_size_factor', 1.2)
        self.min_position_size_factor = config.get('min_position_size_factor', 0.8)
        self.high_confidence_threshold = config.get('high_confidence_threshold', 0.8)
        self.low_confidence_threshold = config.get('low_confidence_threshold', 0.4)
        self.volatility_adjustment_factor = config.get('volatility_adjustment_factor', 1.0)
        self.sl_adjustment_factor = config.get('sl_adjustment_factor', 1.0)
        self.tp_adjustment_factor = config.get('tp_adjustment_factor', 1.0)

        # Track suggestion history
        self.suggestion_history = []

    def suggest_parameter_adjustments(
        self,
        rl_insights: Dict[str, Any],
        current_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest risk parameter adjustments based on RL model insights.

        Args:
            rl_insights: Insights from the RL model
            current_parameters: Current risk parameters

        Returns:
            Suggested parameter adjustments
        """
        # Default to no change
        adjustments = {
            "position_sizing": 1.0,  # Multiplier for position size
            "stop_loss_distance": 1.0,  # Multiplier for SL distance
            "take_profit_distance": 1.0,  # Multiplier for TP distance
            "max_open_positions": 0,  # Adjustment to max open positions
            "confidence": 0.5,  # Overall confidence in suggestions
        }

        # Check if we have minimum required insights
        if not rl_insights or 'confidence' not in rl_insights:
            logger.warning("Insufficient RL insights provided for parameter suggestions")
            return adjustments

        # Extract key insights with fallbacks
        confidence = rl_insights.get('confidence', 0.5)
        predicted_volatility = rl_insights.get('predicted_volatility', None)
        expected_return = rl_insights.get('expected_return', None)
        trade_duration = rl_insights.get('expected_trade_duration', None)

        # Skip suggestions if confidence is too low
        if confidence < self.min_confidence_threshold:
            logger.info(f"RL confidence too low ({confidence:.2f}) to suggest parameter adjustments")
            return adjustments

        # Update overall confidence in suggestions
        adjustments['confidence'] = confidence

        # --- Position Sizing Adjustments ---
        # Scale position size based on confidence
        if confidence >= self.high_confidence_threshold:
            # High confidence allows for larger positions
            position_factor = 1.0 + (confidence - self.high_confidence_threshold) * (
                self.max_position_size_factor - 1.0) / (1.0 - self.high_confidence_threshold)
            position_factor = min(position_factor, self.max_position_size_factor)
        elif confidence <= self.low_confidence_threshold:
            # Low confidence reduces position sizes
            position_factor = self.min_position_size_factor + (confidence / self.low_confidence_threshold) * (
                1.0 - self.min_position_size_factor)
        else:
            # Medium confidence uses linear scaling
            position_factor = 1.0

        # Further adjust position size based on predicted volatility
        if predicted_volatility is not None:
            # Baseline volatility from config
            baseline_volatility = self.config.get('baseline_volatility', 0.0005)

            # Decrease position size when volatility is higher than baseline
            if predicted_volatility > baseline_volatility:
                volatility_ratio = baseline_volatility / predicted_volatility
                volatility_adjustment = max(
                    0.5,
                    min(1.0, volatility_ratio ** self.volatility_adjustment_factor)
                )
                position_factor *= volatility_adjustment
            # Increase position size when volatility is lower than baseline (but more conservatively)
            elif predicted_volatility < baseline_volatility:
                volatility_ratio = predicted_volatility / baseline_volatility
                # More conservative scaling for low volatility
                volatility_boost = 1.0 + max(
                    0.0,
                    min(0.2, (1.0 - volatility_ratio) * self.volatility_adjustment_factor * 0.5)
                )
                position_factor *= volatility_boost

        adjustments["position_sizing"] = position_factor

        # --- Stop Loss Adjustments ---
        sl_adjustment = 1.0

        # Adjust SL based on predicted volatility
        if predicted_volatility is not None:
            baseline_volatility = self.config.get('baseline_volatility', 0.0005)

            # Wider stops for higher volatility
            if predicted_volatility > baseline_volatility:
                volatility_ratio = predicted_volatility / baseline_volatility
                sl_adjustment = min(2.0, volatility_ratio ** self.sl_adjustment_factor)
            # Tighter stops for lower volatility
            else:
                volatility_ratio = baseline_volatility / predicted_volatility
                sl_adjustment = max(0.7, 1.0 / (volatility_ratio ** self.sl_adjustment_factor))

        adjustments["stop_loss_distance"] = sl_adjustment

        # --- Take Profit Adjustments ---
        tp_adjustment = 1.0

        # Adjust TP based on expected return if available
        if expected_return is not None:
            baseline_return = self.config.get('baseline_return', 0.002)  # 0.2%

            # Scale TP based on expected return
            return_ratio = expected_return / baseline_return
            tp_adjustment = max(0.8, min(1.5, return_ratio ** self.tp_adjustment_factor))

        # Also consider trade duration for TP adjustment
        if trade_duration is not None:
            baseline_duration = self.config.get('baseline_duration', 60)  # minutes

            # Shorter expected trades can have closer TPs
            duration_ratio = trade_duration / baseline_duration
            duration_factor = max(0.8, min(1.2, duration_ratio ** 0.5))
            tp_adjustment *= duration_factor

        adjustments["take_profit_distance"] = tp_adjustment

        # --- Max Open Positions Adjustment ---
        # High confidence can allow more concurrent positions
        max_positions_adjustment = 0
        if confidence > 0.7:
            max_positions_adjustment = 1
        if confidence > 0.9:
            max_positions_adjustment = 2

        adjustments["max_open_positions"] = max_positions_adjustment

        # Log the suggestions
        logger.info(f"RL-based risk parameter adjustments: {adjustments}")

        # Record in history
        self.suggestion_history.append({
            'timestamp': datetime.now(),
            'adjustments': adjustments,
            'insights': rl_insights
        })

        return adjustments

    def get_suggestion_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent suggestion history."""
        return self.suggestion_history[-limit:] if self.suggestion_history else []


class DynamicRiskAdapter:
    """
    Component that adapts risk parameters based on RL model insights and market conditions.
    """

    def __init__(
        self,
        risk_orchestrator: RiskCheckOrchestrator,
        rl_suggester: RLRiskParameterSuggester,
        config: Dict[str, Any]
    ):
        """
        Initialize the dynamic risk adapter.

        Args:
            risk_orchestrator: Risk check orchestrator to integrate with
            rl_suggester: RL risk parameter suggester
            config: Configuration parameters
        """
        self.risk_orchestrator = risk_orchestrator
        self.rl_suggester = rl_suggester
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.enable_position_size_adjustment = config.get('enable_position_size_adjustment', True)
        self.enable_stop_loss_adjustment = config.get('enable_stop_loss_adjustment', True)
        self.enable_take_profit_adjustment = config.get('enable_take_profit_adjustment', True)
        self.detected_market_regime = None
        self.regime_adjustment_factors = config.get('regime_adjustment_factors', {
            'trending_bullish': {'position': 1.1, 'sl': 1.1, 'tp': 1.1},
            'trending_bearish': {'position': 1.1, 'sl': 1.1, 'tp': 1.1},
            'ranging': {'position': 0.9, 'sl': 0.9, 'tp': 0.9},
            'volatile': {'position': 0.8, 'sl': 1.2, 'tp': 1.2},
            'breakout': {'position': 1.0, 'sl': 1.2, 'tp': 1.2},
            'liquidity_crisis': {'position': 0.5, 'sl': 1.5, 'tp': 1.5},
            'choppy': {'position': 0.7, 'sl': 1.0, 'tp': 0.9},
            'low_volatility': {'position': 1.0, 'sl': 0.8, 'tp': 0.8},
        })

        self.adjustment_history = []

    def set_market_regime(self, regime: MarketRegimeType) -> None:
        """
        Set the current detected market regime.

        Args:
            regime: Detected market regime
        """
        self.detected_market_regime = regime
        logger.info(f"Market regime set to {regime}")

    def process_rl_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process insights from the RL model and adapt risk parameters.

        Args:
            insights: Dictionary of insights from the RL model

        Returns:
            Dictionary of applied adjustments
        """
        # Get current risk parameters
        current_params = self._get_current_risk_parameters()

        # Get suggested adjustments from RL
        rl_adjustments = self.rl_suggester.suggest_parameter_adjustments(
            insights, current_params)

        # Apply regime-specific adjustments
        final_adjustments = self._apply_regime_adjustments(rl_adjustments)

        # Only apply if confidence exceeds threshold
        if final_adjustments['confidence'] >= self.confidence_threshold:
            self._apply_risk_parameter_adjustments(final_adjustments)
            logger.info("Applied RL-suggested risk parameter adjustments")
        else:
            logger.info("Confidence below threshold, no adjustments applied")

        # Record adjustment in history
        self.adjustment_history.append({
            'timestamp': datetime.now(),
            'insights': insights,
            'rl_adjustments': rl_adjustments,
            'final_adjustments': final_adjustments,
            'applied': final_adjustments['confidence'] >= self.confidence_threshold
        })

        return final_adjustments

    def _get_current_risk_parameters(self) -> Dict[str, Any]:
        """Get current risk parameters from orchestrator."""
        try:
            # This is a simplified representation - in practice, would extract from risk orchestrator
            params = self.risk_orchestrator.get_risk_parameters()
            return {
                'position_sizing': params.position_sizing.max_position_size,
                'stop_loss_distance': params.stop_loss.default_distance_pips,
                'take_profit_distance': params.take_profit.default_distance_pips,
                'max_open_positions': params.position_sizing.max_positions,
            }
        except AttributeError:
            # Fallback if methods don't exist
            logger.warning("Could not retrieve current risk parameters, using defaults")
            return {
                'position_sizing': 1.0,
                'stop_loss_distance': 30.0,  # pips
                'take_profit_distance': 60.0,  # pips
                'max_open_positions': 5,
            }

    def _apply_regime_adjustments(self, rl_adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime-specific adjustments to RL suggestions."""
        if not self.detected_market_regime:
            return rl_adjustments

        # Get regime adjustment factors
        regime_name = self.detected_market_regime.value
        factors = self.regime_adjustment_factors.get(regime_name, {
            'position': 1.0, 'sl': 1.0, 'tp': 1.0
        })

        # Create a copy of the adjustments to modify
        adjusted = rl_adjustments.copy()

        # Apply regime-specific scaling
        adjusted['position_sizing'] *= factors.get('position', 1.0)
        adjusted['stop_loss_distance'] *= factors.get('sl', 1.0)
        adjusted['take_profit_distance'] *= factors.get('tp', 1.0)

        # Ensure values are within reasonable bounds
        adjusted['position_sizing'] = max(0.5, min(1.5, adjusted['position_sizing']))
        adjusted['stop_loss_distance'] = max(0.5, min(2.0, adjusted['stop_loss_distance']))
        adjusted['take_profit_distance'] = max(0.5, min(2.0, adjusted['take_profit_distance']))

        # Log the regime adjustment
        logger.info(f"Applied {regime_name} regime adjustments to RL suggestions")

        return adjusted

    def _apply_risk_parameter_adjustments(self, adjustments: Dict[str, Any]) -> None:
        """Apply finalized risk parameter adjustments to the orchestrator."""
        try:
            # Get current parameters
            current_params = self.risk_orchestrator.get_risk_parameters()

            # Create updated parameters
            updated_params = current_params.copy()

            # Update position sizing if enabled
            if self.enable_position_size_adjustment:
                max_size = current_params.position_sizing.max_position_size
                updated_params.position_sizing.max_position_size = max_size * adjustments['position_sizing']

                # Also adjust max positions if suggested
                if adjustments['max_open_positions'] != 0:
                    max_positions = current_params.position_sizing.max_positions
                    updated_params.position_sizing.max_positions = max_positions + adjustments['max_open_positions']

            # Update stop loss if enabled
            if self.enable_stop_loss_adjustment:
                sl_distance = current_params.stop_loss.default_distance_pips
                updated_params.stop_loss.default_distance_pips = sl_distance * adjustments['stop_loss_distance']

            # Update take profit if enabled
            if self.enable_take_profit_adjustment:
                tp_distance = current_params.take_profit.default_distance_pips
                updated_params.take_profit.default_distance_pips = tp_distance * adjustments['take_profit_distance']

            # Apply the updated parameters
            self.risk_orchestrator.update_risk_parameters(updated_params)

        except (AttributeError, Exception) as e:
            logger.error(f"Error applying risk parameter adjustments: {e}")

    def get_adjustment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent adjustment history."""
        return self.adjustment_history[-limit:] if self.adjustment_history else []


# Mock RiskCheckOrchestrator for testing/demonstration
class MockRiskOrchestrator:
    """Mock implementation of RiskCheckOrchestrator for testing."""

    def __init__(self):
    """
      init  .
    
    """

        self.risk_parameters = {
            'position_sizing': {
                'max_position_size': 1.0,  # in lots
                'max_positions': 5
            },
            'stop_loss': {
                'default_distance_pips': 30.0
            },
            'take_profit': {
                'default_distance_pips': 60.0
            }
        }

    def get_risk_parameters(self):
    """
    Get risk parameters.
    
    """

        return self.risk_parameters

    def update_risk_parameters(self, params):
    """
    Update risk parameters.
    
    Args:
        params: Description of params
    
    """

        self.risk_parameters = params
        print(f"Updated risk parameters: {self.risk_parameters}")


# Example usage
if __name__ == "__main__":
    # Configuration for suggester
    suggester_config = {
        'min_confidence_threshold': 0.3,
        'max_position_size_factor': 1.2,
        'min_position_size_factor': 0.8,
        'high_confidence_threshold': 0.8,
        'low_confidence_threshold': 0.4,
        'baseline_volatility': 0.0006,
        'baseline_return': 0.002,
        'baseline_duration': 60,
        'volatility_adjustment_factor': 1.0,
        'sl_adjustment_factor': 0.8,
        'tp_adjustment_factor': 0.5
    }

    # Create components
    mock_orchestrator = MockRiskOrchestrator()
    rl_suggester = RLRiskParameterSuggester(suggester_config)
    dynamic_adapter = DynamicRiskAdapter(mock_orchestrator, rl_suggester, {})

    # --- Simulate receiving RL insights ---

    print("\n--- Scenario 1: High Confidence RL Insight ---")
    high_confidence_insights = {'confidence': 0.9, 'predicted_volatility': 0.0006}
    dynamic_adapter.process_rl_insights(high_confidence_insights)
    # Expected: Position size increased to 12000

    print("\n--- Scenario 2: Low Confidence RL Insight ---")
    low_confidence_insights = {'confidence': 0.3, 'predicted_volatility': 0.0008}
    dynamic_adapter.process_rl_insights(low_confidence_insights)
    # Expected: Position size decreased from 12000 to 9600

    print("\n--- Scenario 3: Medium Confidence RL Insight ---")
    medium_confidence_insights = {'confidence': 0.6, 'predicted_volatility': 0.0012}
    dynamic_adapter.process_rl_insights(medium_confidence_insights)
    # Expected: Position size decreased due to higher volatility

    print("\n--- Scenario 4: Market Regime Change ---")
    dynamic_adapter.set_market_regime(MarketRegimeType.VOLATILE)
    volatile_insights = {'confidence': 0.8, 'predicted_volatility': 0.0009}
    dynamic_adapter.process_rl_insights(volatile_insights)
    # Expected: Position size further decreased due to volatile regime
