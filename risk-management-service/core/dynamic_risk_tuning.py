"""
Dynamic Risk Tuning Integration

This module provides components that adapt risk parameters based on RL model insights,
enabling dynamic position sizing, adaptive stop-loss placement, and risk regime detection.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from common_lib.simulation.interfaces import MarketRegimeType
from common_lib.reinforcement.interfaces import IRLEnvironment, IRLModel, IRLOptimizer
from common_lib.risk.interfaces import (
    IRiskParameters, IRiskRegimeDetector, IDynamicRiskTuner, RiskRegimeType
)
from adapters.simulation_adapters import (
    MarketRegimeSimulatorAdapter,
    BrokerSimulatorAdapter
)
from adapters.ml_adapters import (
    RLModelAdapter, RLOptimizerAdapter, RLEnvironmentAdapter, MarketRegimeAnalyzerAdapter
)
from adapters.dynamic_risk_adapter import SimpleRiskParameters

logger = get_logger(__name__)


# Using RiskRegimeType from common_lib.risk.interfaces


@dataclass
class RiskParameters(IRiskParameters):
    """Collection of risk management parameters that can be dynamically tuned"""
    # Position sizing parameters
    position_size_pct: float = 0.02  # Percentage of account balance to risk per trade
    max_position_size: float = 5.0  # Maximum position size in lots
    min_position_size: float = 0.01  # Minimum position size in lots

    # Stop-loss parameters
    stop_loss_pct: float = 0.01  # Stop-loss percentage
    stop_loss_atr_multiplier: float = 2.0  # ATR multiplier for stop-loss placement
    trailing_stop_activation_pct: float = 0.01  # When to activate trailing stop
    trailing_stop_distance_pct: float = 0.005  # How far to trail price

    # Take-profit parameters
    take_profit_pct: float = 0.03  # Take-profit percentage
    take_profit_atr_multiplier: float = 3.0  # ATR multiplier for take-profit placement

    # Risk-per-trade parameters
    max_risk_per_trade_pct: float = 0.02  # Maximum risk percentage per trade
    max_total_risk_pct: float = 0.10  # Maximum total risk exposure

    # Correlation and diversification parameters
    max_correlation_threshold: float = 0.7  # Maximum allowed correlation between positions
    min_diversification_score: float = 0.3  # Minimum diversification score

    # Volatility parameters
    volatility_scaling_factor: float = 1.0  # Factor to scale position size based on volatility
    volatility_lookback_periods: int = 20  # Lookback periods for calculating volatility

    # Drawdown parameters
    max_drawdown_pct: float = 0.15  # Maximum allowed drawdown
    drawdown_reduction_factor: float = 0.5  # Factor to reduce position sizes during drawdown

    # Profit-taking parameters
    profit_taking_threshold_pct: float = 0.05  # When to take partial profits
    profit_taking_size_pct: float = 0.3  # Percentage of position to close when taking profits

    # Time-based parameters
    max_holding_time_minutes: int = 1440  # Maximum position holding time (in minutes)
    time_decay_factor: float = 0.2  # Factor for time decay of position sizing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'RiskParameters':
        """Create from dictionary"""
        return cls(**params_dict)

    def adjust_for_regime(self, regime_type: RiskRegimeType) -> 'RiskParameters':
        """Create a copy of risk parameters adjusted for the given risk regime"""
        params = self.to_dict()

        if regime_type == RiskRegimeType.LOW_RISK:
            # Conservative parameters
            params['position_size_pct'] *= 0.5
            params['max_position_size'] *= 0.7
            params['stop_loss_atr_multiplier'] *= 1.5
            params['max_risk_per_trade_pct'] *= 0.7
            params['volatility_scaling_factor'] *= 0.8

        elif regime_type == RiskRegimeType.MODERATE_RISK:
            # Default parameters, no changes needed
            pass

        elif regime_type == RiskRegimeType.HIGH_RISK:
            # More aggressive parameters
            params['position_size_pct'] *= 1.2
            params['max_position_size'] *= 1.1
            params['stop_loss_atr_multiplier'] *= 0.9
            params['take_profit_atr_multiplier'] *= 1.2
            params['max_risk_per_trade_pct'] *= 1.1
            params['volatility_scaling_factor'] *= 1.1

        elif regime_type == RiskRegimeType.EXTREME_RISK:
            # Very conservative parameters
            params['position_size_pct'] *= 0.3
            params['max_position_size'] *= 0.5
            params['stop_loss_atr_multiplier'] *= 1.8
            params['max_risk_per_trade_pct'] *= 0.5
            params['max_total_risk_pct'] *= 0.6
            params['volatility_scaling_factor'] *= 0.6

        elif regime_type == RiskRegimeType.CRISIS:
            # Ultra-conservative parameters
            params['position_size_pct'] *= 0.2
            params['max_position_size'] *= 0.3
            params['stop_loss_atr_multiplier'] *= 2.0
            params['max_risk_per_trade_pct'] *= 0.3
            params['max_total_risk_pct'] *= 0.4
            params['volatility_scaling_factor'] *= 0.4
            params['drawdown_reduction_factor'] *= 1.5

        return RiskParameters.from_dict(params)


class RiskRegimeDetector(IRiskRegimeDetector):
    """
    Detector for identifying current risk regime based on market indicators.
    Adapts risk parameters based on detected market conditions.
    """

    def __init__(
        self,
        base_risk_parameters: Optional[RiskParameters] = None,
        lookback_periods: int = 50,
        volatility_threshold_multiplier: float = 1.5,
        detection_frequency_minutes: int = 60,
        symbols: List[str] = None
    ):
        """
        Initialize risk regime detector.

        Args:
            base_risk_parameters: Base risk parameters to adjust
            lookback_periods: Number of periods to look back for regime detection
            volatility_threshold_multiplier: Multiplier for volatility thresholds
            detection_frequency_minutes: How often to re-check regime (in minutes)
            symbols: Symbols to monitor for regime detection
        """
        self.base_risk_parameters = base_risk_parameters or RiskParameters()
        self.lookback_periods = lookback_periods
        self.volatility_threshold_multiplier = volatility_threshold_multiplier
        self.detection_frequency_minutes = detection_frequency_minutes
        self.symbols = symbols or ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]

        # Current regime state
        self.current_regime = RiskRegimeType.MODERATE_RISK
        self.last_detection_time = datetime.now()
        self.regime_history = []

        # Market indicators history
        self.indicators_history = {
            'volatility': [],
            'correlation': [],
            'drawdown': [],
            'liquidity': [],
            'gap_frequency': [],
            'price_jumps': []
        }

    def detect_regime(self, market_data: pd.DataFrame) -> RiskRegimeType:
        """
        Detect current risk regime based on market data.

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            Detected risk regime type
        """
        # Ensure we have enough data
        if len(market_data) < self.lookback_periods:
            logger.warning(f"Not enough data for regime detection (need {self.lookback_periods}, got {len(market_data)})")
            return self.current_regime

        # Calculate key indicators
        volatility = self._calculate_volatility(market_data)
        correlation = self._calculate_correlation(market_data)
        drawdown = self._calculate_drawdown(market_data)
        liquidity = self._calculate_liquidity_proxy(market_data)
        gap_frequency = self._calculate_gap_frequency(market_data)
        price_jumps = self._calculate_price_jumps(market_data)

        # Update indicators history
        self._update_indicators_history(volatility, correlation, drawdown, liquidity, gap_frequency, price_jumps)

        # Determine regime based on indicators
        regime = self._determine_regime(volatility, correlation, drawdown, liquidity, gap_frequency, price_jumps)

        # Update current regime and history
        self.current_regime = regime
        self.last_detection_time = datetime.now()
        self.regime_history.append((self.last_detection_time, regime))

        return regime

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate market volatility"""
        # Use standard deviation of returns
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(self.lookback_periods)
        return volatility

    def _calculate_correlation(self, data: pd.DataFrame) -> float:
        """Calculate correlation between symbols"""
        # If only one symbol in data, return 0
        if len(self.symbols) <= 1 or 'symbol' not in data.columns:
            return 0.0

        correlations = []
        returns_by_symbol = {}

        for symbol in self.symbols:
            symbol_data = data[data['symbol'] == symbol] if 'symbol' in data.columns else data
            if len(symbol_data) > 0:
                returns_by_symbol[symbol] = symbol_data['close'].pct_change().dropna()

        for i, symbol1 in enumerate(self.symbols[:-1]):
            for symbol2 in self.symbols[i+1:]:
                if symbol1 in returns_by_symbol and symbol2 in returns_by_symbol:
                    s1_returns = returns_by_symbol[symbol1]
                    s2_returns = returns_by_symbol[symbol2]

                    # Align the indexes
                    s1_returns, s2_returns = s1_returns.align(s2_returns, join='inner')

                    if len(s1_returns) > 1:
                        corr = s1_returns.corr(s2_returns)
                        if not pd.isna(corr):
                            correlations.append(abs(corr))

        # Average absolute correlation
        return np.mean(correlations) if correlations else 0.0

    def _calculate_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate maximum drawdown in the lookback period"""
        # Calculate the maximum drawdown
        prices = data['close'].values
        peak = prices[0]
        max_drawdown = 0

        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_liquidity_proxy(self, data: pd.DataFrame) -> float:
        """Calculate a proxy for market liquidity"""
        # Use volume as liquidity proxy if available, otherwise use bid-ask spread
        if 'volume' in data.columns:
            recent_volume = data['volume'].iloc[-self.lookback_periods//4:].mean()
            all_volume = data['volume'].mean()
            return recent_volume / all_volume if all_volume > 0 else 1.0
        elif 'spread' in data.columns:
            recent_spread = data['spread'].iloc[-self.lookback_periods//4:].mean()
            all_spread = data['spread'].mean()
            # Inverse relationship: higher spread = lower liquidity
            return all_spread / recent_spread if recent_spread > 0 else 1.0
        else:
            return 1.0

    def _calculate_gap_frequency(self, data: pd.DataFrame) -> float:
        """Calculate frequency of price gaps"""
        # Calculate gaps between candles
        open_prices = data['open'].values[1:]
        prev_close_prices = data['close'].values[:-1]

        # Calculate gaps as percentage of previous close
        gaps = np.abs(open_prices - prev_close_prices) / prev_close_prices

        # Count significant gaps (e.g., >0.1%)
        significant_gaps = np.sum(gaps > 0.001)

        return significant_gaps / (len(gaps) if len(gaps) > 0 else 1)

    def _calculate_price_jumps(self, data: pd.DataFrame) -> float:
        """Calculate frequency and size of price jumps within candles"""
        # Calculate candle range as percentage of price
        high_low_range = (data['high'] - data['low']) / data['close']

        # Calculate average range
        avg_range = high_low_range.mean()

        # Identify extreme ranges (e.g., >2x average)
        extreme_ranges = high_low_range > (2 * avg_range)
        extreme_count = extreme_ranges.sum()

        return extreme_count / len(data)

    def _update_indicators_history(
        self,
        volatility: float,
        correlation: float,
        drawdown: float,
        liquidity: float,
        gap_frequency: float,
        price_jumps: float
    ):
        """Update indicators history"""
        self.indicators_history['volatility'].append(volatility)
        self.indicators_history['correlation'].append(correlation)
        self.indicators_history['drawdown'].append(drawdown)
        self.indicators_history['liquidity'].append(liquidity)
        self.indicators_history['gap_frequency'].append(gap_frequency)
        self.indicators_history['price_jumps'].append(price_jumps)

        # Keep history to a reasonable size
        max_history = 100
        for key in self.indicators_history:
            if len(self.indicators_history[key]) > max_history:
                self.indicators_history[key] = self.indicators_history[key][-max_history:]

    def _determine_regime(
        self,
        volatility: float,
        correlation: float,
        drawdown: float,
        liquidity: float,
        gap_frequency: float,
        price_jumps: float
    ) -> RiskRegimeType:
        """Determine risk regime based on market indicators"""
        # Calculate historical volatility stats for reference
        if len(self.indicators_history['volatility']) > 10:
            volatility_mean = np.mean(self.indicators_history['volatility'])
            volatility_std = np.std(self.indicators_history['volatility'])
        else:
            volatility_mean = volatility
            volatility_std = volatility * 0.2  # Rough estimate

        # Score system for risk regime determination
        crisis_score = 0
        extreme_score = 0
        high_score = 0
        moderate_score = 0
        low_score = 0

        # Volatility scoring
        vol_z_score = (volatility - volatility_mean) / volatility_std if volatility_std > 0 else 0
        if vol_z_score > 3.0:
            crisis_score += 1
        elif vol_z_score > 2.0:
            extreme_score += 1
        elif vol_z_score > 1.0:
            high_score += 1
        elif vol_z_score < -1.0:
            low_score += 1
        else:
            moderate_score += 1

        # Correlation scoring
        if correlation > 0.9:
            extreme_score += 1
        elif correlation > 0.75:
            high_score += 1
        elif correlation < 0.3:
            low_score += 1
        else:
            moderate_score += 1

        # Drawdown scoring
        if drawdown > 0.15:
            crisis_score += 1
        elif drawdown > 0.1:
            extreme_score += 1
        elif drawdown > 0.05:
            high_score += 1
        elif drawdown < 0.02:
            low_score += 1
        else:
            moderate_score += 1

        # Liquidity scoring
        if liquidity < 0.3:
            crisis_score += 1
        elif liquidity < 0.5:
            extreme_score += 1
        elif liquidity < 0.8:
            high_score += 1
        elif liquidity > 1.2:
            low_score += 1
        else:
            moderate_score += 1

        # Gap frequency scoring
        if gap_frequency > 0.2:
            crisis_score += 1
        elif gap_frequency > 0.1:
            extreme_score += 1
        elif gap_frequency > 0.05:
            high_score += 1
        elif gap_frequency < 0.01:
            low_score += 1
        else:
            moderate_score += 1

        # Price jumps scoring
        if price_jumps > 0.2:
            crisis_score += 1
        elif price_jumps > 0.1:
            extreme_score += 1
        elif price_jumps > 0.05:
            high_score += 1
        elif price_jumps < 0.01:
            low_score += 1
        else:
            moderate_score += 1

        # Determine regime based on highest score
        scores = {
            RiskRegimeType.CRISIS: crisis_score,
            RiskRegimeType.EXTREME_RISK: extreme_score,
            RiskRegimeType.HIGH_RISK: high_score,
            RiskRegimeType.MODERATE_RISK: moderate_score,
            RiskRegimeType.LOW_RISK: low_score
        }

        return max(scores, key=scores.get)

    def should_detect(self, current_time: datetime) -> bool:
        """Check if it's time to detect regime based on frequency setting"""
        time_since_last = current_time - self.last_detection_time
        return time_since_last.total_seconds() >= (self.detection_frequency_minutes * 60)

    def get_current_risk_parameters(self) -> RiskParameters:
        """Get risk parameters adjusted for current regime"""
        return self.base_risk_parameters.adjust_for_regime(self.current_regime)

    def get_regime_history(self) -> List[Tuple[datetime, RiskRegimeType]]:
        """Get history of regime changes"""
        return self.regime_history.copy()

    def save_state(self, filepath: str) -> None:
        """Save detector state to file"""
        state = {
            'current_regime': self.current_regime.value,
            'last_detection_time': self.last_detection_time.isoformat(),
            'regime_history': [(dt.isoformat(), regime.value) for dt, regime in self.regime_history],
            'indicators_history': self.indicators_history
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)

    def load_state(self, filepath: str) -> None:
        """Load detector state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.current_regime = RiskRegimeType(state['current_regime'])
        self.last_detection_time = datetime.fromisoformat(state['last_detection_time'])
        self.regime_history = [(datetime.fromisoformat(dt), RiskRegimeType(regime)) for dt, regime in state['regime_history']]
        self.indicators_history = state['indicators_history']


class DynamicRiskTuner(IDynamicRiskTuner):
    """
    Component that adapts risk parameters based on RL model insights and market conditions.
    """

    def __init__(
        self,
        base_risk_parameters: Optional[RiskParameters] = None,
        regime_detector: Optional[RiskRegimeDetector] = None,
        rl_confidence_weight: float = 0.5,
        adaptation_speed: float = 0.1,
        max_param_change_pct: float = 0.3,
        update_frequency_minutes: int = 30
    ):
        """
        Initialize dynamic risk tuner.

        Args:
            base_risk_parameters: Base risk parameters
            regime_detector: Risk regime detector
            rl_confidence_weight: Weight given to RL confidence scores (vs. rule-based)
            adaptation_speed: Speed of parameter adaptation (0.0-1.0)
            max_param_change_pct: Maximum percentage change per update
            update_frequency_minutes: How often to update parameters
        """
        self.base_risk_parameters = base_risk_parameters or RiskParameters()
        self.regime_detector = regime_detector or RiskRegimeDetector(base_risk_parameters=self.base_risk_parameters)
        self.rl_confidence_weight = rl_confidence_weight
        self.adaptation_speed = adaptation_speed
        self.max_param_change_pct = max_param_change_pct
        self.update_frequency_minutes = update_frequency_minutes

        # Current parameters and history
        self.current_parameters = self.base_risk_parameters
        self.parameter_history = []
        self.last_update_time = datetime.now()

        # RL model confidence scores
        self.rl_confidence_scores = {
            'position_sizing': 0.5,
            'stop_loss': 0.5,
            'take_profit': 0.5,
            'overall': 0.5
        }

    def update_rl_confidence_scores(self, confidence_scores: Dict[str, float]) -> None:
        """
        Update confidence scores from RL model.

        Args:
            confidence_scores: Dictionary of confidence scores (0.0-1.0)
        """
        for key, value in confidence_scores.items():
            if key in self.rl_confidence_scores:
                # Ensure values are in valid range
                self.rl_confidence_scores[key] = max(0.0, min(1.0, value))

    def update_parameters(
        self,
        market_data: Optional[Any] = None,
        current_time: Optional[datetime] = None,
        rl_recommendations: Optional[Dict[str, float]] = None,
        force_update: bool = False
    ) -> IRiskParameters:
        """
        Update risk parameters based on regime, RL insights, and time.

        Args:
            market_data: Optional market data for regime detection
            current_time: Current time (defaults to now)
            rl_recommendations: Optional parameter recommendations from RL
            force_update: Whether to force update regardless of time

        Returns:
            Updated risk parameters
        """
        current_time = current_time or datetime.now()
        time_since_update = current_time - self.last_update_time

        # Check if it's time to update
        if not force_update and time_since_update.total_seconds() < (self.update_frequency_minutes * 60):
            return self.current_parameters

        # Update regime if market data is provided
        if market_data is not None and self.regime_detector.should_detect(current_time):
            self.regime_detector.detect_regime(market_data)

        # Get regime-adjusted parameters as starting point
        regime_params = self.regime_detector.get_current_risk_parameters()

        # Apply RL recommendations if provided
        if rl_recommendations:
            self.current_parameters = self._apply_rl_recommendations(regime_params, rl_recommendations)
        else:
            self.current_parameters = regime_params

        # Record update
        self.last_update_time = current_time
        self.parameter_history.append((current_time, self.current_parameters.to_dict()))

        return self.current_parameters

    def _apply_rl_recommendations(
        self,
        base_params: RiskParameters,
        recommendations: Dict[str, float]
    ) -> RiskParameters:
        """Apply RL recommendations to risk parameters"""
        param_dict = base_params.to_dict()

        # Apply each recommendation
        for param_name, recommended_value in recommendations.items():
            if param_name in param_dict:
                current_value = param_dict[param_name]
                confidence = self.rl_confidence_scores.get('overall', 0.5)

                # Get specific confidence if available
                for conf_key in self.rl_confidence_scores:
                    if conf_key in param_name:
                        confidence = self.rl_confidence_scores[conf_key]
                        break

                # Apply change with weighting based on confidence
                weight = self.rl_confidence_weight * confidence
                max_change = current_value * self.max_param_change_pct

                # Calculate weighted target value
                weighted_target = (current_value * (1 - weight)) + (recommended_value * weight)

                # Limit change by adaptation speed and max change
                change = (weighted_target - current_value) * self.adaptation_speed
                change = max(-max_change, min(max_change, change))

                # Apply change
                param_dict[param_name] = current_value + change

        return RiskParameters.from_dict(param_dict)

    def get_current_parameters(self) -> RiskParameters:
        """Get current risk parameters"""
        return self.current_parameters

    def get_parameter_history(self) -> List[Tuple[datetime, Dict]]:
        """Get history of parameter updates"""
        return self.parameter_history.copy()

    def save_state(self, filepath: str) -> None:
        """Save tuner state to file"""
        state = {
            'current_parameters': self.current_parameters.to_dict(),
            'last_update_time': self.last_update_time.isoformat(),
            'rl_confidence_scores': self.rl_confidence_scores,
            'parameter_history': [(dt.isoformat(), params) for dt, params in self.parameter_history]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)

    def load_state(self, filepath: str) -> None:
        """Load tuner state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.current_parameters = RiskParameters.from_dict(state['current_parameters'])
        self.last_update_time = datetime.fromisoformat(state['last_update_time'])
        self.rl_confidence_scores = state['rl_confidence_scores']
        self.parameter_history = [(datetime.fromisoformat(dt), params) for dt, params in state['parameter_history']]


class RLRiskOptimizer(IRLOptimizer):
    """
    Optimizer that extracts risk parameter recommendations from RL models.
    """

    def __init__(
        self,
        rl_model=None,
        base_risk_parameters: Optional[RiskParameters] = None,
        confidence_threshold: float = 0.6,
        min_training_steps: int = 50000,
        feature_importance_threshold: float = 0.1
    ):
        """
        Initialize RL risk optimizer.

        Args:
            rl_model: Reinforcement learning model
            base_risk_parameters: Base risk parameters
            confidence_threshold: Minimum confidence required for recommendations
            min_training_steps: Minimum training steps before making recommendations
            feature_importance_threshold: Minimum feature importance for recommendations
        """
        self.rl_model = rl_model
        self.base_risk_parameters = base_risk_parameters or RiskParameters()
        self.confidence_threshold = confidence_threshold
        self.min_training_steps = min_training_steps
        self.feature_importance_threshold = feature_importance_threshold

        # Track recommendations and their performance
        self.recommendations_history = []
        self.performance_metrics = {
            'position_sizing': {
                'accuracy': 0.0,
                'improvement': 0.0,
                'confidence': 0.0
            },
            'stop_loss': {
                'accuracy': 0.0,
                'improvement': 0.0,
                'confidence': 0.0
            },
            'take_profit': {
                'accuracy': 0.0,
                'improvement': 0.0,
                'confidence': 0.0
            }
        }

    def set_model(self, model) -> None:
        """Set the RL model"""
        self.rl_model = model

    def evaluate_model_confidence(self, test_env=None) -> Dict[str, float]:
        """
        Evaluate the model's confidence for risk parameter recommendations.

        Args:
            test_env: Environment for testing

        Returns:
            Dictionary of confidence scores
        """
        if not self.rl_model:
            return {key: 0.5 for key in self.performance_metrics}

        # Get model training steps if available
        training_steps = getattr(self.rl_model, 'num_timesteps', 0)
        if training_steps < self.min_training_steps:
            logger.info(f"Model has insufficient training steps ({training_steps}/{self.min_training_steps})")
            return {key: 0.5 for key in self.performance_metrics}

        # Run evaluation if environment is provided
        if test_env:
            self._run_evaluation(test_env)

        # Return current confidence scores
        return {
            'position_sizing': self.performance_metrics['position_sizing']['confidence'],
            'stop_loss': self.performance_metrics['stop_loss']['confidence'],
            'take_profit': self.performance_metrics['take_profit']['confidence'],
            'overall': np.mean([
                self.performance_metrics['position_sizing']['confidence'],
                self.performance_metrics['stop_loss']['confidence'],
                self.performance_metrics['take_profit']['confidence']
            ])
        }

    def _run_evaluation(self, env) -> None:
        """Run evaluation to assess model performance"""
        try:
            # Run model on environment for multiple episodes
            n_episodes = 10
            total_rewards = []
            actions_data = []

            for _ in range(n_episodes):
                obs = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action, _ = self.rl_model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward

                    # Record action details
                    if isinstance(action, (list, np.ndarray)) and len(action) >= 3:
                        actions_data.append({
                            'action_type': action[0],
                            'position_size': action[1],
                            'stop_loss': action[2],
                            'take_profit': action[3] if len(action) > 3 else 0.0,
                            'reward': reward,
                            'info': info
                        })

                total_rewards.append(episode_reward)

            # Calculate metrics from the evaluation
            self._update_metrics_from_evaluation(total_rewards, actions_data)

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")

    def _update_metrics_from_evaluation(
        self,
        rewards: List[float],
        actions_data: List[Dict]
    ) -> None:
        """Update performance metrics based on evaluation results"""
        if not actions_data:
            return

        # Analyze position sizing
        position_sizes = [a['position_size'] for a in actions_data if a['action_type'] != 0]
        if position_sizes:
            position_size_rewards = [a['reward'] for a in actions_data if a['action_type'] != 0]
            position_size_accuracy = np.mean([r > 0 for r in position_size_rewards])
            position_size_avg_reward = np.mean(position_size_rewards)
            position_size_confidence = min(1.0, position_size_accuracy * 1.5)

            self.performance_metrics['position_sizing'] = {
                'accuracy': position_size_accuracy,
                'improvement': position_size_avg_reward,
                'confidence': position_size_confidence
            }

        # Analyze stop loss placement
        stop_losses = [a['stop_loss'] for a in actions_data if a['action_type'] != 0 and a['stop_loss'] > 0]
        if stop_losses:
            # Check how often stop losses saved from worse losses
            good_stops = sum(1 for a in actions_data if
                            a['action_type'] != 0 and
                            a['stop_loss'] > 0 and
                            'stopped_out' in a['info'] and
                            a['info'].get('max_adverse_excursion', 0) > a['stop_loss'])
            stop_loss_accuracy = good_stops / len(stop_losses) if stop_losses else 0.5
            stop_loss_confidence = min(1.0, 0.5 + 0.5 * stop_loss_accuracy)

            self.performance_metrics['stop_loss'] = {
                'accuracy': stop_loss_accuracy,
                'improvement': 0.0,  # Hard to quantify
                'confidence': stop_loss_confidence
            }

        # Analyze take profit placement
        take_profits = [a['take_profit'] for a in actions_data if a['action_type'] != 0 and a['take_profit'] > 0]
        if take_profits:
            # Check how often take profits captured optimal gains
            good_tps = sum(1 for a in actions_data if
                          a['action_type'] != 0 and
                          a['take_profit'] > 0 and
                          'profit_target_hit' in a['info'] and
                          a['info'].get('max_favorable_excursion', 0) < a['take_profit'] * 1.5)
            take_profit_accuracy = good_tps / len(take_profits) if take_profits else 0.5
            take_profit_confidence = min(1.0, 0.5 + 0.5 * take_profit_accuracy)

            self.performance_metrics['take_profit'] = {
                'accuracy': take_profit_accuracy,
                'improvement': 0.0,  # Hard to quantify
                'confidence': take_profit_confidence
            }

    def get_risk_parameter_recommendations(self,
                                          market_state: Dict[str, Any],
                                          current_params: RiskParameters) -> Dict[str, float]:
        """
        Get risk parameter recommendations based on current market state.

        Args:
            market_state: Current market state features
            current_params: Current risk parameters

        Returns:
            Dictionary of recommended parameter values
        """
        if not self.rl_model:
            return {}

        try:
            # Get confidences
            confidence_scores = self.evaluate_model_confidence()
            overall_confidence = confidence_scores.get('overall', 0.0)

            # Don't make recommendations if confidence is too low
            if overall_confidence < self.confidence_threshold:
                logger.info(f"Confidence too low for recommendations: {overall_confidence:.2f}")
                return {}

            # Extract features from market state for prediction
            features = self._extract_prediction_features(market_state)

            # Get model prediction (action) based on features
            # This is a simplified approach - would need adaptation for actual model
            model_output = self._get_model_prediction(features)

            # Convert model output to parameter recommendations
            recommendations = self._convert_model_output_to_recommendations(model_output, current_params)

            # Filter recommendations by confidence
            filtered_recommendations = {}
            for param, value in recommendations.items():
                param_type = 'position_sizing'
                if 'stop' in param:
                    param_type = 'stop_loss'
                elif 'profit' in param:
                    param_type = 'take_profit'

                param_confidence = self.performance_metrics[param_type]['confidence']

                if param_confidence >= self.confidence_threshold:
                    filtered_recommendations[param] = value

            # Record recommendations
            self.recommendations_history.append({
                'timestamp': datetime.now(),
                'market_state': {k: v for k, v in market_state.items() if isinstance(v, (int, float, str, bool))},
                'current_params': current_params.to_dict(),
                'recommendations': filtered_recommendations,
                'confidence': confidence_scores
            })

            return filtered_recommendations

        except Exception as e:
            logger.error(f"Error getting parameter recommendations: {e}")
            return {}

    def _extract_prediction_features(self, market_state: Dict[str, Any]) -> np.ndarray:
        """Extract features for prediction from market state"""
        # Extract relevant features for prediction
        # This would need to be adapted to match the model's input features
        features = []

        # Include standard market metrics if available
        for key in ['volatility', 'trend_strength', 'current_drawdown']:
            if key in market_state:
                features.append(market_state[key])

        # Include price action features
        for key in ['last_return', 'return_volatility', 'sharpe_ratio']:
            if key in market_state:
                features.append(market_state[key])

        # Add risk regime as one-hot if available
        if 'risk_regime' in market_state:
            regime_dict = {r.value: 0 for r in RiskRegimeType}
            regime_dict[market_state['risk_regime']] = 1
            features.extend(list(regime_dict.values()))

        # Ensure we have features
        if not features:
            # Use placeholder features if necessary
            features = [0.5, 0.5, 0.5]

        return np.array(features)

    def _get_model_prediction(self, features: np.ndarray) -> np.ndarray:
        """Get model prediction based on features"""
        # This is a placeholder - actual implementation would depend on model type
        if not self.rl_model:
            return np.array([0.0, 0.5, 0.5, 0.5])

        # For stable-baselines3 type models
        if hasattr(self.rl_model, 'predict'):
            try:
                # Reshape features if needed
                if features.ndim == 1:
                    features = features.reshape(1, -1)

                # Get prediction
                action, _ = self.rl_model.predict(features, deterministic=True)
                return action
            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                return np.array([0.0, 0.5, 0.5, 0.5])

        # Fallback for other model types
        return np.array([0.0, 0.5, 0.5, 0.5])

    def _convert_model_output_to_recommendations(
        self,
        model_output: np.ndarray,
        current_params: RiskParameters
    ) -> Dict[str, float]:
        """Convert model output (action) to parameter recommendations"""
        recommendations = {}

        # Assume model output format is:
        # [action_type, position_size_pct, stop_loss_pips, take_profit_pips]
        if len(model_output) >= 4:
            # Position sizing recommendation
            position_size_pct = model_output[1]
            recommendations['position_size_pct'] = position_size_pct

            # Stop loss recommendation
            stop_loss_atr = model_output[2]
            if stop_loss_atr > 0:
                recommendations['stop_loss_atr_multiplier'] = stop_loss_atr

            # Take profit recommendation
            take_profit_atr = model_output[3]
            if take_profit_atr > 0:
                recommendations['take_profit_atr_multiplier'] = take_profit_atr

        # Calculate maximum total risk based on position sizing
        if 'position_size_pct' in recommendations:
            max_risk_per_trade = recommendations['position_size_pct'] * 0.5
            recommendations['max_risk_per_trade_pct'] = max_risk_per_trade
            recommendations['max_total_risk_pct'] = max_risk_per_trade * 5

        return recommendations

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    def get_recommendations_history(self) -> List[Dict]:
        """Get history of recommendations"""
        return self.recommendations_history.copy()

    async def optimize_parameters(
        self,
        parameter_type: str,
        current_values: Dict[str, Any],
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize parameters using RL insights.

        Args:
            parameter_type: Type of parameters to optimize
            current_values: Current parameter values
            context: Contextual information for optimization
            constraints: Optional constraints on parameter values

        Returns:
            Optimized parameter values
        """
        # Convert context to market state format
        market_state = {
            'volatility': context.get('volatility', 0.01),
            'trend_strength': context.get('trend_strength', 0.5),
            'current_drawdown': context.get('drawdown', 0.0),
            'risk_regime': context.get('risk_regime', RiskRegimeType.MODERATE_RISK.value)
        }

        # Create RiskParameters object from current values
        current_params = RiskParameters.from_dict(current_values)

        # Get recommendations
        recommendations = self.get_risk_parameter_recommendations(market_state, current_params)

        if not recommendations:
            return current_values

        # Apply recommendations with constraints
        result = current_values.copy()
        for param, value in recommendations.items():
            if param in result:
                result[param] = value

                # Apply constraints if provided
                if constraints:
                    min_val = constraints.get(f"{param}_min")
                    max_val = constraints.get(f"{param}_max")

                    if min_val is not None and result[param] < min_val:
                        result[param] = min_val

                    if max_val is not None and result[param] > max_val:
                        result[param] = max_val

        return result

    def get_optimization_confidence(self) -> float:
        """
        Get the confidence level of the last optimization.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence_scores = self.evaluate_model_confidence()
        return confidence_scores.get('overall', 0.5)

    async def update_models(self) -> bool:
        """
        Update the underlying RL models.

        Returns:
            Success status of the update
        """
        # This would reload or update the model in a real implementation
        logger.info("Updating RL models")
        return True

    def save_state(self, filepath: str) -> None:
        """Save optimizer state to file"""
        state = {
            'performance_metrics': self.performance_metrics,
            'recommendations_history': [
                {
                    'timestamp': r['timestamp'].isoformat(),
                    'market_state': r['market_state'],
                    'current_params': r['current_params'],
                    'recommendations': r['recommendations'],
                    'confidence': r['confidence']
                }
                for r in self.recommendations_history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)

    def load_state(self, filepath: str) -> None:
        """Load optimizer state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.performance_metrics = state['performance_metrics']
        self.recommendations_history = [
            {
                'timestamp': datetime.fromisoformat(r['timestamp']),
                'market_state': r['market_state'],
                'current_params': r['current_params'],
                'recommendations': r['recommendations'],
                'confidence': r['confidence']
            }
            for r in state['recommendations_history']
        ]
