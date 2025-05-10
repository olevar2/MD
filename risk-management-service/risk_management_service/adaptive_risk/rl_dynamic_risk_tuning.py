"""
Dynamic Risk Parameter Tuning based on RL Model Insights

This module provides components for adapting risk parameters based on reinforcement learning
model insights and detected market regimes.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from risk_management_service.models.risk_metrics import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_value_at_risk, calculate_expected_shortfall
)
from common_lib.simulation.interfaces import (
    MarketRegimeType,
    NewsImpactLevel,
    SentimentLevel
)
from risk_management_service.adapters.simulation_adapters import NewsSentimentSimulatorAdapter

logger = get_logger(__name__)


class RiskParameterType(Enum):
    """Types of risk parameters that can be dynamically adjusted."""
    POSITION_SIZE = "position_size"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_EXPOSURE = "max_exposure"
    VOLATILITY_SCALING = "volatility_scaling"
    CORRELATION_LIMIT = "correlation_limit"
    NEWS_SENSITIVITY = "news_sensitivity"


@dataclass
class RiskRegimeConfig:
    """Risk configuration for a specific market regime."""
    regime: MarketRegimeType
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    max_drawdown_limit: float = 0.02  # 2% default
    max_exposure_pct: float = 0.1  # 10% default
    volatility_scaling_factor: float = 1.0
    correlation_limit: float = 0.7
    news_sensitivity: float = 1.0


class RiskRegimeDetector:
    """
    Detects the current market risk regime based on market data,
    volatility, and other factors.
    """

    def __init__(
        self,
        lookback_window: int = 30,
        volatility_window: int = 14,
        regime_change_threshold: float = 0.6,
    ):
        """
        Initialize the risk regime detector.

        Args:
            lookback_window: Days to look back for regime detection
            volatility_window: Window for volatility calculation
            regime_change_threshold: Threshold for confirming regime change
        """
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.regime_change_threshold = regime_change_threshold
        self.regime_history = []
        self.current_regime = MarketRegimeType.NORMAL

    def detect_regime(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        news_events: Optional[List[Dict]] = None,
        external_indicators: Optional[Dict] = None
    ) -> MarketRegimeType:
        """
        Detect the current market regime based on various data sources.

        Args:
            price_data: DataFrame with price data (OHLCV)
            volume_data: Optional volume data
            news_events: Optional list of recent news events
            external_indicators: Optional external market indicators

        Returns:
            The detected market regime
        """
        if len(price_data) < self.lookback_window:
            logger.warning(f"Insufficient data for regime detection: {len(price_data)} < {self.lookback_window}")
            return MarketRegimeType.NORMAL

        # Calculate key metrics
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1] * np.sqrt(252)

        # Check for extreme volatility (crisis)
        if volatility > 0.4:  # 40% annualized volatility
            regime = MarketRegimeType.CRISIS

        # Check for breakout pattern
        elif self._detect_breakout(price_data):
            regime = MarketRegimeType.BREAKOUT

        # Check for trending market
        elif self._detect_trend(price_data):
            regime = MarketRegimeType.TRENDING

        # Check for high volatility (but not crisis)
        elif volatility > 0.25:  # 25% annualized volatility
            regime = MarketRegimeType.VOLATILE

        # Check for ranging market
        elif self._detect_range(price_data):
            regime = MarketRegimeType.RANGING

        # Default to normal
        else:
            regime = MarketRegimeType.NORMAL

        # Consider news events for regime detection
        if news_events:
            # Check if we're dealing with NewsEvent objects or dictionaries
            if hasattr(news_events[0], 'impact_level') if news_events else False:
                # Using NewsEvent objects
                high_impact_news = [e for e in news_events
                                   if e.impact_level in (NewsImpactLevel.HIGH, NewsImpactLevel.CRITICAL)]
            else:
                # Using dictionary format
                high_impact_news = [e for e in news_events
                                   if e.get('impact') in (NewsImpactLevel.HIGH.value, NewsImpactLevel.CRITICAL.value)]

            if high_impact_news and len(high_impact_news) > 2:
                # Multiple high-impact news suggests volatile or crisis regime
                if regime == MarketRegimeType.NORMAL or regime == MarketRegimeType.RANGING:
                    regime = MarketRegimeType.VOLATILE

        # Check if regime has changed
        if regime != self.current_regime:
            # Count how many of the recent regime detections match the new regime
            self.regime_history.append(regime)
            if len(self.regime_history) > 5:
                self.regime_history.pop(0)

            # Only change regime if we have consistent signals
            if (self.regime_history.count(regime) / len(self.regime_history)) >= self.regime_change_threshold:
                self.current_regime = regime
                logger.info(f"Risk regime changed to {regime.name}")
            else:
                # Not enough evidence to change regime yet
                return self.current_regime
        else:
            # Regime hasn't changed, add to history
            self.regime_history.append(regime)
            if len(self.regime_history) > 5:
                self.regime_history.pop(0)

        return self.current_regime

    def _detect_trend(self, price_data: pd.DataFrame) -> bool:
        """Detect if the market is in a trending state."""
        # Calculate short and long moving averages
        short_ma = price_data['close'].rolling(window=20).mean()
        long_ma = price_data['close'].rolling(window=50).mean()

        # Check if short MA is consistently above/below long MA
        last_n = 10  # Check last 10 periods
        if len(short_ma) < last_n or len(long_ma) < last_n:
            return False

        # Check for uptrend or downtrend
        is_uptrend = all(short_ma.iloc[-i] > long_ma.iloc[-i] for i in range(1, last_n))
        is_downtrend = all(short_ma.iloc[-i] < long_ma.iloc[-i] for i in range(1, last_n))

        return is_uptrend or is_downtrend

    def _detect_range(self, price_data: pd.DataFrame) -> bool:
        """Detect if the market is in a ranging state."""
        if len(price_data) < self.lookback_window:
            return False

        # Calculate price range
        recent_data = price_data.iloc[-self.lookback_window:]
        price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()

        # Calculate how many times price crosses its mean
        mean_price = recent_data['close'].mean()
        crosses = 0
        above = recent_data['close'].iloc[0] > mean_price

        for _, row in recent_data.iterrows():
            current_above = row['close'] > mean_price
            if current_above != above:
                crosses += 1
                above = current_above

        # Ranging market typically has narrow range and multiple crosses
        return price_range < 0.05 and crosses >= 4  # 5% range and at least 4 crosses

    def _detect_breakout(self, price_data: pd.DataFrame) -> bool:
        """Detect if the market is experiencing a breakout."""
        if len(price_data) < self.lookback_window:
            return False

        # Calculate recent price action
        recent_data = price_data.iloc[-self.lookback_window:]

        # Calculate upper and lower bands (simple approach using recent highs/lows)
        lookback = min(self.lookback_window, 20)  # Use at most 20 days for band calculation
        upper_band = recent_data['high'].iloc[:-5].max()  # High of the recent range, excluding last 5 periods
        lower_band = recent_data['low'].iloc[:-5].min()   # Low of the recent range, excluding last 5 periods

        # Check if latest close has broken out of the range
        latest_close = recent_data['close'].iloc[-1]
        prev_close = recent_data['close'].iloc[-2]

        # Breakout with increased volume is a stronger signal
        volume_surge = False
        if 'volume' in recent_data.columns:
            avg_volume = recent_data['volume'].iloc[:-1].mean()
            latest_volume = recent_data['volume'].iloc[-1]
            volume_surge = latest_volume > (avg_volume * 1.5)  # 50% higher than average

        # Check for price breakout
        upside_breakout = latest_close > upper_band and prev_close <= upper_band
        downside_breakout = latest_close < lower_band and prev_close >= lower_band

        return (upside_breakout or downside_breakout) and (volume_surge or not 'volume' in recent_data.columns)


class DynamicRiskParamOptimizer:
    """
    Adapts risk parameters based on RL model predictions and detected market conditions.
    """

    def __init__(
        self,
        base_params: Dict[str, float],
        regime_configs: Dict[MarketRegimeType, RiskRegimeConfig] = None,
        adaptation_rate: float = 0.2,  # How quickly to adapt to new suggestions (0-1)
        rl_confidence_weight: float = 0.5,  # Weight given to RL suggestions vs rule-based
    ):
        """
        Initialize the dynamic risk parameter optimizer.

        Args:
            base_params: Base risk parameters
            regime_configs: Risk configurations for each market regime
            adaptation_rate: How quickly to adapt parameters (0-1)
            rl_confidence_weight: Weight given to RL suggestions vs rules
        """
        self.base_params = base_params
        self.current_params = base_params.copy()
        self.adaptation_rate = adaptation_rate
        self.rl_confidence_weight = rl_confidence_weight
        self.regime_detector = RiskRegimeDetector()

        # Initialize regime configurations
        if regime_configs is None:
            # Create default configurations for each regime
            self.regime_configs = {
                MarketRegimeType.NORMAL: RiskRegimeConfig(
                    regime=MarketRegimeType.NORMAL,
                    position_size_multiplier=1.0,
                    stop_loss_multiplier=1.0,
                    take_profit_multiplier=1.0
                ),
                MarketRegimeType.TRENDING: RiskRegimeConfig(
                    regime=MarketRegimeType.TRENDING,
                    position_size_multiplier=1.2,  # Increase position size in trends
                    stop_loss_multiplier=1.2,      # Wider stops in trends
                    take_profit_multiplier=1.5     # Larger profit targets in trends
                ),
                MarketRegimeType.RANGING: RiskRegimeConfig(
                    regime=MarketRegimeType.RANGING,
                    position_size_multiplier=0.8,  # Smaller positions in ranges
                    stop_loss_multiplier=0.7,      # Tighter stops in ranges
                    take_profit_multiplier=0.7     # Smaller profit targets in ranges
                ),
                MarketRegimeType.VOLATILE: RiskRegimeConfig(
                    regime=MarketRegimeType.VOLATILE,
                    position_size_multiplier=0.6,  # Reduced size in volatile markets
                    stop_loss_multiplier=1.5,      # Wider stops for volatility
                    take_profit_multiplier=1.2,    # Slightly higher profit targets
                    max_exposure_pct=0.07          # Reduced overall exposure
                ),
                MarketRegimeType.BREAKOUT: RiskRegimeConfig(
                    regime=MarketRegimeType.BREAKOUT,
                    position_size_multiplier=1.3,  # Larger positions for breakouts
                    stop_loss_multiplier=0.8,      # Tighter stops on breakouts
                    take_profit_multiplier=1.5,    # Larger profit targets on breakouts
                    news_sensitivity=1.5           # More sensitive to news during breakouts
                ),
                MarketRegimeType.CRISIS: RiskRegimeConfig(
                    regime=MarketRegimeType.CRISIS,
                    position_size_multiplier=0.3,  # Drastically reduced positions
                    stop_loss_multiplier=1.8,      # Much wider stops if trading
                    take_profit_multiplier=0.8,    # Smaller profit targets
                    max_exposure_pct=0.03,         # Very low overall exposure
                    volatility_scaling_factor=0.3, # Scale down for crisis volatility
                    news_sensitivity=2.0           # Highly sensitive to news
                )
            }
        else:
            self.regime_configs = regime_configs

        # Keep history of parameter changes
        self.parameter_history = []
        self.regime_history = []

    def update_parameters(
        self,
        market_data: pd.DataFrame,
        current_time: datetime,
        rl_suggestions: Optional[Dict[str, float]] = None,
        rl_confidence: float = 0.5,
        news_events: Optional[List[Dict]] = None,
        news_simulator: Optional[NewsSentimentSimulatorAdapter] = None,
    ) -> Dict[str, float]:
        """
        Update risk parameters based on current market conditions and RL suggestions.

        Args:
            market_data: Recent market data for regime detection
            current_time: Current timestamp
            rl_suggestions: Optional parameter suggestions from RL model
            rl_confidence: Confidence level in RL suggestions (0-1)
            news_events: Recent news events that might affect risk
            news_simulator: Optional news sentiment simulator adapter

        Returns:
            Updated risk parameters
        """
        # Get news events if not provided but simulator is available
        if news_events is None and news_simulator is not None:
            # Get current news for major currencies
            news_events = []
            for currency in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
                news_events.extend(news_simulator.get_current_news(currency))

        # Detect the current market regime
        current_regime = self.regime_detector.detect_regime(market_data, news_events=news_events)
        regime_config = self.regime_configs[current_regime]

        # Start with base parameters adjusted for the current regime
        new_params = self.base_params.copy()

        # Apply regime-specific adjustments
        new_params['position_size'] *= regime_config.position_size_multiplier
        new_params['stop_loss_pips'] *= regime_config.stop_loss_multiplier
        new_params['take_profit_pips'] *= regime_config.take_profit_multiplier
        new_params['max_drawdown'] = regime_config.max_drawdown_limit
        new_params['max_exposure'] = regime_config.max_exposure_pct
        new_params['volatility_scaling'] = regime_config.volatility_scaling_factor

        # If we have RL suggestions, blend them with rule-based parameters
        if rl_suggestions:
            # Calculate effective weight for RL suggestions based on confidence
            effective_weight = self.rl_confidence_weight * rl_confidence

            # Blend parameters
            for param_name, rl_value in rl_suggestions.items():
                if param_name in new_params:
                    rule_value = new_params[param_name]
                    blended_value = (rule_value * (1 - effective_weight)) + (rl_value * effective_weight)
                    new_params[param_name] = blended_value

        # Apply adaptation rate for smooth transitions
        for param_name, new_value in new_params.items():
            if param_name in self.current_params:
                current_value = self.current_params[param_name]
                self.current_params[param_name] = current_value + (self.adaptation_rate * (new_value - current_value))
            else:
                self.current_params[param_name] = new_value

        # Store history
        self.parameter_history.append({
            'timestamp': current_time,
            'regime': current_regime.name,
            'parameters': self.current_params.copy(),
            'rl_confidence': rl_confidence if rl_suggestions else 0.0
        })

        if len(self.regime_history) == 0 or self.regime_history[-1] != current_regime:
            self.regime_history.append(current_regime)
            logger.info(f"Risk regime changed to {current_regime.name}, updating parameters")

        return self.current_params

    def suggest_position_size(
        self,
        symbol: str,
        direction: str,
        account_balance: float,
        current_exposure: float,
        market_data: Optional[pd.DataFrame] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Suggest position size based on current risk parameters and market conditions.

        Args:
            symbol: The trading symbol
            direction: Trade direction ('buy' or 'sell')
            account_balance: Current account balance
            current_exposure: Current exposure as percentage of account
            market_data: Optional recent market data
            volatility: Optional pre-calculated volatility

        Returns:
            Suggested position size in standard lots
        """
        # Get base position size from current parameters
        base_position_size = self.current_params.get('position_size', 0.01)

        # Adjust for account balance (simple scaling)
        account_factor = account_balance / 10000.0  # Normalize to $10k base
        position_size = base_position_size * account_factor

        # Apply volatility scaling if available
        if volatility is not None:
            vol_scaling = self.current_params.get('volatility_scaling', 1.0)
            # Adjust position inversely to volatility
            norm_vol = min(volatility / 0.01, 5.0)  # Cap at 5x normal volatility
            position_size *= (vol_scaling / norm_vol)

        # Respect maximum exposure
        max_exposure = self.current_params.get('max_exposure', 0.1)  # Default 10%
        if current_exposure + position_size > max_exposure:
            position_size = max(0, max_exposure - current_exposure)

        # Apply any RL-specific adjustments for the direction
        if direction == 'buy':
            position_size *= self.current_params.get('long_bias', 1.0)
        else:
            position_size *= self.current_params.get('short_bias', 1.0)

        return position_size

    def suggest_stop_loss(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        market_data: Optional[pd.DataFrame] = None,
        atr_value: Optional[float] = None
    ) -> float:
        """
        Suggest stop loss price based on current risk parameters and market conditions.

        Args:
            symbol: The trading symbol
            direction: Trade direction ('buy' or 'sell')
            entry_price: Trade entry price
            market_data: Optional recent market data
            atr_value: Optional pre-calculated ATR value

        Returns:
            Suggested stop loss price
        """
        # Get base stop loss in pips from current parameters
        base_stop_pips = self.current_params.get('stop_loss_pips', 50)

        # Convert pips to price for forex (assuming 4 decimal places for most pairs)
        pip_value = 0.0001
        if symbol.endswith('JPY'):
            pip_value = 0.01  # JPY pairs typically have 2 decimal places

        # Calculate ATR-based stop if ATR is available
        if atr_value is not None:
            atr_multiplier = self.current_params.get('atr_stop_multiplier', 2.0)
            atr_stop_pips = atr_value * atr_multiplier / pip_value
            # Blend fixed and ATR-based stops
            stop_pips = (base_stop_pips + atr_stop_pips) / 2
        else:
            stop_pips = base_stop_pips

        # Calculate stop loss price
        if direction == 'buy':
            stop_loss = entry_price - (stop_pips * pip_value)
        else:
            stop_loss = entry_price + (stop_pips * pip_value)

        return stop_loss

    def suggest_take_profit(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        market_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Suggest take profit price based on current risk parameters and market conditions.

        Args:
            symbol: The trading symbol
            direction: Trade direction ('buy' or 'sell')
            entry_price: Trade entry price
            stop_loss: Stop loss price
            market_data: Optional recent market data

        Returns:
            Suggested take profit price
        """
        # Get base take profit in pips or risk:reward ratio
        base_tp_pips = self.current_params.get('take_profit_pips', 100)
        risk_reward_ratio = self.current_params.get('risk_reward_ratio', 2.0)

        # Convert pips to price for forex (assuming 4 decimal places for most pairs)
        pip_value = 0.0001
        if symbol.endswith('JPY'):
            pip_value = 0.01  # JPY pairs typically have 2 decimal places

        # Calculate risk in price terms
        risk_price = abs(entry_price - stop_loss)

        # Calculate take profit based on risk:reward ratio
        reward_price = risk_price * risk_reward_ratio

        # Also calculate fixed take profit
        fixed_tp = entry_price + (base_tp_pips * pip_value) if direction == 'buy' else entry_price - (base_tp_pips * pip_value)

        # Use the larger of the two for reward maximization
        if direction == 'buy':
            take_profit = max(entry_price + reward_price, fixed_tp)
        else:
            take_profit = min(entry_price - reward_price, fixed_tp)

        return take_profit

    def get_parameter_history(self) -> pd.DataFrame:
        """Get historical parameter changes as a DataFrame."""
        if not self.parameter_history:
            return pd.DataFrame()

        # Extract data into a format suitable for DataFrame
        data = []
        for record in self.parameter_history:
            row = {'timestamp': record['timestamp'], 'regime': record['regime'], 'rl_confidence': record['rl_confidence']}
            for param, value in record['parameters'].items():
                row[param] = value
            data.append(row)

        return pd.DataFrame(data)
