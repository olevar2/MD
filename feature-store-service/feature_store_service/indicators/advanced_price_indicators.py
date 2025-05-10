"""
Advanced Price Indicators Module.

This module provides implementations of advanced price-based chart indicators.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from feature_store_service.indicators.base_indicator import BaseIndicator


class IchimokuCloud(BaseIndicator):
    """
    Ichimoku Kinko Hyo (Cloud) indicator with all five components and signal optimization.

    This is a comprehensive indicator that includes multiple components to identify
    trend direction, support and resistance levels, and generate signals.
    """

    category = "price"

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        chikou_shift: int = 26,
        displacement: int = 26,
        include_cloud_strength: bool = True,
        optimize_signals: bool = False,
        signal_optimization_method: str = "historical_accuracy",
        **kwargs
    ):
        """
        Initialize Ichimoku Cloud indicator.

        Args:
            tenkan_period: Period for Tenkan-sen (Conversion Line)
            kijun_period: Period for Kijun-sen (Base Line) and Chikou shift
            senkou_b_period: Period for Senkou Span B
            chikou_shift: Period for shifting Chikou Span (typically same as kijun_period)
            displacement: Forward displacement for Senkou spans (cloud)
            include_cloud_strength: Whether to calculate cloud strength analysis
            optimize_signals: Whether to optimize signal thresholds
            signal_optimization_method: Method for signal optimization
                ('historical_accuracy', 'profit_maximization', 'risk_adjusted')
            **kwargs: Additional parameters
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.chikou_shift = chikou_shift
        self.displacement = displacement
        self.include_cloud_strength = include_cloud_strength
        self.optimize_signals = optimize_signals
        self.signal_optimization_method = signal_optimization_method

        # Define output column names
        self.name_tenkan = f"ichimoku_tenkan_{tenkan_period}"
        self.name_kijun = f"ichimoku_kijun_{kijun_period}"
        self.name_senkou_a = f"ichimoku_senkou_a_{tenkan_period}_{kijun_period}"
        self.name_senkou_b = f"ichimoku_senkou_b_{senkou_b_period}"
        self.name_chikou = f"ichimoku_chikou_{chikou_shift}"
        self.name_cloud_color = "ichimoku_cloud_color"
        self.name_cloud_strength = "ichimoku_cloud_strength"
        self.name_lead_signal = "ichimoku_lead_signal"  # Early signal
        self.name_confirm_signal = "ichimoku_confirm_signal"  # Confirmed signal
        self.name_strong_signal = "ichimoku_strong_signal"  # Strong signal (all conditions)
        self.name_kumo_breakout = "ichimoku_kumo_breakout"  # Cloud breakout

    def _midpoint(self, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """Calculate the midpoint between highest high and lowest low for a period."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return (highest_high + lowest_low) / 2

    def _optimize_signal_thresholds(self, data: pd.DataFrame) -> dict:
        """
        Optimize signal thresholds based on historical performance.

        Args:
            data: DataFrame with OHLCV data and calculated Ichimoku components

        Returns:
            Dictionary with optimized parameters
        """
        # Default thresholds
        thresholds = {
            'tk_cross_weight': 1.0,
            'price_above_cloud_weight': 1.0,
            'chikou_weight': 1.0,
            'cloud_flat_penalty': 0.5,
            'min_signal_strength': 2.0  # Minimum combined score to generate signal
        }

        if not self.optimize_signals or len(data) < 200:
            return thresholds

        # Simple returns for performance evaluation
        data['returns'] = data['close'].pct_change()

        # Calculate signal success metrics for different threshold combinations
        best_score = -np.inf
        best_thresholds = thresholds.copy()

        # Grid search for optimal parameters
        for tk_weight in [0.8, 1.0, 1.2]:
            for cloud_weight in [0.8, 1.0, 1.2]:
                for chikou_weight in [0.8, 1.0, 1.2]:
                    for min_strength in [1.5, 2.0, 2.5]:
                        # Calculate signal strength with these weights
                        signal_strength = (
                            (data[self.name_tenkan] > data[self.name_kijun]) * tk_weight +
                            ((data['close'] > data[self.name_senkou_a]) &
                             (data['close'] > data[self.name_senkou_b])) * cloud_weight +
                            (data[self.name_chikou] > data['close'].shift(self.chikou_shift)) * chikou_weight
                        )

                        # Generate test signals
                        test_signal = np.where(signal_strength >= min_strength, 1,
                                               np.where(signal_strength <= -min_strength, -1, 0))

                        # Evaluate performance
                        if self.signal_optimization_method == 'historical_accuracy':
                            # Test if signals predict next day's direction
                            correct_predictions = (test_signal * np.sign(data['returns'].shift(-1))) > 0
                            score = correct_predictions.mean()

                        elif self.signal_optimization_method == 'profit_maximization':
                            # Simple strategy returns
                            strategy_returns = test_signal * data['returns'].shift(-1)
                            score = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0

                        elif self.signal_optimization_method == 'risk_adjusted':
                            # Sharpe ratio-like metric
                            strategy_returns = test_signal * data['returns'].shift(-1)
                            if strategy_returns.std() > 0:
                                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                                drawdown = 1 - (1 + strategy_returns).cumprod() / (1 + strategy_returns).cumprod().cummax()
                                max_dd = drawdown.max() if not np.isnan(drawdown.max()) else 1
                                score = sharpe * (1 - max_dd)
                            else:
                                score = 0

                        # Update best parameters if better score found
                        if score > best_score:
                            best_score = score
                            best_thresholds = {
                                'tk_cross_weight': tk_weight,
                                'price_above_cloud_weight': cloud_weight,
                                'chikou_weight': chikou_weight,
                                'cloud_flat_penalty': 0.5,  # Keep constant for now
                                'min_signal_strength': min_strength
                            }

        return best_thresholds

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Ichimoku Cloud values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate Tenkan-sen (Conversion Line) - short-term trend
        result[self.name_tenkan] = self._midpoint(
            result['high'], result['low'], self.tenkan_period
        )

        # Calculate Kijun-sen (Base Line) - medium-term trend
        result[self.name_kijun] = self._midpoint(
            result['high'], result['low'], self.kijun_period
        )

        # Calculate Senkou Span A (Leading Span A) - first cloud boundary
        # Note: Should be shifted forward by displacement period for display
        senkou_a = (result[self.name_tenkan] + result[self.name_kijun]) / 2
        result[self.name_senkou_a] = senkou_a.shift(self.displacement)

        # Calculate Senkou Span B (Leading Span B) - second cloud boundary
        # Note: Should be shifted forward by displacement period for display
        senkou_b = self._midpoint(result['high'], result['low'], self.senkou_b_period)
        result[self.name_senkou_b] = senkou_b.shift(self.displacement)

        # Calculate Chikou Span (Lagging Span) - momentum indicator
        # Note: Shifted backwards, different from the others
        result[self.name_chikou] = result['close'].shift(-self.chikou_shift)

        # Store current Senkou values (unshifted) for signal generation
        # These represent where the cloud will be in the future
        current_senkou_a = senkou_a
        current_senkou_b = senkou_b

        # Implement cloud color (bullish/bearish indicator)
        # Green/bullish when Senkou A > Senkou B, Red/bearish when Senkou A < Senkou B
        result[self.name_cloud_color] = np.where(
            result[self.name_senkou_a] > result[self.name_senkou_b], 1,
            np.where(result[self.name_senkou_a] < result[self.name_senkou_b], -1, 0)
        )

        # Add cloud strength analysis if requested
        if self.include_cloud_strength:
            # Calculate cloud thickness as percentage of price
            cloud_thickness = abs(result[self.name_senkou_a] - result[self.name_senkou_b])
            cloud_flat_penalty = np.where(cloud_thickness / result['close'] < 0.001, 0.5, 1.0)
            result[self.name_cloud_strength] = cloud_thickness / result['close'] * 100 * cloud_flat_penalty

        # Optimize signal thresholds if requested
        if self.optimize_signals and len(result) >= 200:
            thresholds = self._optimize_signal_thresholds(result)
        else:
            thresholds = {
                'tk_cross_weight': 1.0,
                'price_above_cloud_weight': 1.0,
                'chikou_weight': 1.0,
                'cloud_flat_penalty': 0.5,
                'min_signal_strength': 2.0
            }

        # Leading Signal (TK Cross) - Early entry signal
        result[self.name_lead_signal] = np.where(
            result[self.name_tenkan] > result[self.name_kijun], 1,
            np.where(result[self.name_tenkan] < result[self.name_kijun], -1, 0)
        )

        # Confirmation Signal (Price relative to cloud)
        result[self.name_confirm_signal] = np.where(
            (result['close'] > result[self.name_senkou_a]) &
            (result['close'] > result[self.name_senkou_b]), 1,
            np.where(
                (result['close'] < result[self.name_senkou_a]) &
                (result['close'] < result[self.name_senkou_b]), -1, 0
            )
        )

        # Kumo (Cloud) Breakout Signal
        # Detects when price breaks through the cloud
        price_cross_senkou_a = np.sign(result['close'] - result[self.name_senkou_a]) != \
                               np.sign(result['close'].shift(1) - result[self.name_senkou_a].shift(1))
        price_cross_senkou_b = np.sign(result['close'] - result[self.name_senkou_b]) != \
                               np.sign(result['close'].shift(1) - result[self.name_senkou_b].shift(1))

        result[self.name_kumo_breakout] = 0  # Default no breakout

        # Bullish breakout (breaking above cloud)
        bullish_breakout = (
            price_cross_senkou_a | price_cross_senkou_b) & \
            (result['close'] > result[self.name_senkou_a]) & \
            (result['close'] > result[self.name_senkou_b]) & \
            (result['close'] > result['close'].shift(1)
        )
        result.loc[bullish_breakout, self.name_kumo_breakout] = 1

        # Bearish breakout (breaking below cloud)
        bearish_breakout = (
            price_cross_senkou_a | price_cross_senkou_b) & \
            (result['close'] < result[self.name_senkou_a]) & \
            (result['close'] < result[self.name_senkou_b]) & \
            (result['close'] < result['close'].shift(1)
        )
        result.loc[bearish_breakout, self.name_kumo_breakout] = -1

        # Calculate Strong Signal (all conditions aligned)
        # Use weighted scoring based on thresholds
        bullish_score = (
            (result[self.name_tenkan] > result[self.name_kijun]) * thresholds['tk_cross_weight'] +
            ((result['close'] > result[self.name_senkou_a]) &
             (result['close'] > result[self.name_senkou_b])) * thresholds['price_above_cloud_weight'] +
            (result[self.name_chikou] > result['close'].shift(self.chikou_shift)) * thresholds['chikou_weight']
        )

        bearish_score = (
            (result[self.name_tenkan] < result[self.name_kijun]) * thresholds['tk_cross_weight'] +
            ((result['close'] < result[self.name_senkou_a]) &
             (result['close'] < result[self.name_senkou_b])) * thresholds['price_above_cloud_weight'] +
            (result[self.name_chikou] < result['close'].shift(self.chikou_shift)) * thresholds['chikou_weight']
        )

        # Apply cloud strength penalty
        if self.include_cloud_strength:
            cloud_flat_condition = result[self.name_cloud_strength] < 0.5
            bullish_score = bullish_score * np.where(cloud_flat_condition, thresholds['cloud_flat_penalty'], 1.0)
            bearish_score = bearish_score * np.where(cloud_flat_condition, thresholds['cloud_flat_penalty'], 1.0)

        # Generate final strong signal based on threshold
        result[self.name_strong_signal] = np.where(
            bullish_score >= thresholds['min_signal_strength'], 1,
            np.where(bearish_score >= thresholds['min_signal_strength'], -1, 0)
        )

        return result


class HeikinAshi(BaseIndicator):
    """
    Heikin-Ashi indicator with integrated automatic trend analysis.

    This indicator uses a modified candlestick calculation method to better
    visualize the trend direction and strength by smoothing price action.
    It includes automatic trend detection, strength measurement, and reversal detection.
    """

    category = "price"

    def __init__(
        self,
        trend_window: int = 5,
        trend_threshold: float = 3.0,
        reversal_window: int = 3,
        smooth_factor: int = 1,
        include_body_size_analysis: bool = True,
        include_shadow_analysis: bool = True,
        **kwargs
    ):
        """
        Initialize Heikin-Ashi indicator with integrated trend analysis.

        Args:
            trend_window: Window size for trend analysis
            trend_threshold: Threshold for determining strong trends
            reversal_window: Window for detecting trend reversals
            smooth_factor: Additional smoothing for HA calculations
            include_body_size_analysis: Whether to analyze candle body size
            include_shadow_analysis: Whether to analyze candle shadows
            **kwargs: Additional parameters
        """
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.reversal_window = reversal_window
        self.smooth_factor = smooth_factor
        self.include_body_size_analysis = include_body_size_analysis
        self.include_shadow_analysis = include_shadow_analysis

        # Define output column names
        self.name_open = "ha_open"
        self.name_high = "ha_high"
        self.name_low = "ha_low"
        self.name_close = "ha_close"
        self.name_trend = "ha_trend"
        self.name_strength = "ha_strength"
        self.name_reversal = "ha_reversal"
        self.name_body_size = "ha_body_size"
        self.name_upper_shadow = "ha_upper_shadow"
        self.name_lower_shadow = "ha_lower_shadow"
        self.name_consecutive = "ha_consecutive_candles"
        self.name_signal = "ha_signal"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candles with integrated trend analysis.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Heikin-Ashi values and trend analysis
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Initialize Heikin-Ashi columns with NaN values
        result[self.name_open] = np.nan
        result[self.name_high] = np.nan
        result[self.name_low] = np.nan
        result[self.name_close] = np.nan

        # Calculate first values (use regular candle for the first entry)
        result.loc[0, self.name_close] = (result.loc[0, 'open'] +
                                         result.loc[0, 'high'] +
                                         result.loc[0, 'low'] +
                                         result.loc[0, 'close']) / 4
        result.loc[0, self.name_open] = (result.loc[0, 'open'] +
                                        result.loc[0, 'close']) / 2
        result.loc[0, self.name_high] = result.loc[0, 'high']
        result.loc[0, self.name_low] = result.loc[0, 'low']

        # Calculate HA candles for the rest of the data
        for i in range(1, len(result)):
            # Calculate close - average of O, H, L, C for current candle
            result.loc[i, self.name_close] = (result.loc[i, 'open'] +
                                             result.loc[i, 'high'] +
                                             result.loc[i, 'low'] +
                                             result.loc[i, 'close']) / 4

            # Apply additional smoothing if requested
            if self.smooth_factor > 1:
                if i >= self.smooth_factor:
                    # Calculate smoothed close using previous values
                    result.loc[i, self.name_close] = (
                        result.loc[i, self.name_close] +
                        result.loc[i-1:i-self.smooth_factor:-1, self.name_close].sum()
                    ) / (self.smooth_factor + 1)

            # Calculate open - average of previous HA open and close
            result.loc[i, self.name_open] = (result.loc[i-1, self.name_open] +
                                            result.loc[i-1, self.name_close]) / 2

            # Calculate high - maximum of current high, HA open, or HA close
            result.loc[i, self.name_high] = max(result.loc[i, 'high'],
                                              result.loc[i, self.name_open],
                                              result.loc[i, self.name_close])

            # Calculate low - minimum of current low, HA open, or HA close
            result.loc[i, self.name_low] = min(result.loc[i, 'low'],
                                             result.loc[i, self.name_open],
                                             result.loc[i, self.name_close])

        # Calculate body size and shadows if requested
        if self.include_body_size_analysis:
            result[self.name_body_size] = abs(result[self.name_close] - result[self.name_open])
            result[self.name_body_size + "_pct"] = result[self.name_body_size] / result[self.name_close] * 100

        if self.include_shadow_analysis:
            # Upper shadow
            result[self.name_upper_shadow] = np.where(
                result[self.name_close] >= result[self.name_open],
                result[self.name_high] - result[self.name_close],  # If bullish
                result[self.name_high] - result[self.name_open]    # If bearish
            )

            # Lower shadow
            result[self.name_lower_shadow] = np.where(
                result[self.name_close] >= result[self.name_open],
                result[self.name_open] - result[self.name_low],    # If bullish
                result[self.name_close] - result[self.name_low]    # If bearish
            )

            # Calculate shadow-to-body ratio (avoid division by zero)
            total_shadow = result[self.name_upper_shadow] + result[self.name_lower_shadow]
            body_size_adj = result[self.name_body_size].replace(0, np.nan)
            result["ha_shadow_body_ratio"] = (total_shadow / body_size_adj).fillna(0)

        # Determine candle color (bullish=1, bearish=-1)
        result[self.name_trend] = np.sign(result[self.name_close] - result[self.name_open])

        # Count consecutive candles of the same color
        result[self.name_consecutive] = 0
        for i in range(1, len(result)):
            if result.loc[i, self.name_trend] == result.loc[i-1, self.name_trend]:
                result.loc[i, self.name_consecutive] = result.loc[i-1, self.name_consecutive] + 1
            else:
                result.loc[i, self.name_consecutive] = 1

        # Calculate trend strength
        # Use a rolling window to detect sustained movement
        result[self.name_strength] = 0.0

        # Only start trend calculations after we have enough data
        if len(result) > self.trend_window:
            # Baseline trend using rolling percentage change
            pct_change = ((result[self.name_close] / result[self.name_close].shift(self.trend_window)) - 1) * 100

            # Scale to 0-10 range with a sigmoid-like transformation
            trend_intensity = pct_change.abs().clip(0, 5) * 2

            # Determine direction and final strength
            result[self.name_strength] = np.sign(pct_change) * trend_intensity

            # Account for consecutive candles
            consecutive_factor = result[self.name_consecutive].clip(0, 5) / 5
            result[self.name_strength] = result[self.name_strength] * (1 + consecutive_factor * 0.5)

        # Identify trend reversals
        result[self.name_reversal] = 0  # No reversal by default

        # Need at least reversal_window + 1 candles to detect a reversal
        if len(result) > self.reversal_window + 1:
            for i in range(self.reversal_window + 1, len(result)):
                # Check for bullish reversal (previously bearish, now bullish)
                if (result.loc[i-self.reversal_window:i-1, self.name_trend].mean() < -0.5 and
                    result.loc[i, self.name_trend] > 0 and
                    result.loc[i, self.name_consecutive] <= 2):
                    result.loc[i, self.name_reversal] = 1

                # Check for bearish reversal (previously bullish, now bearish)
                elif (result.loc[i-self.reversal_window:i-1, self.name_trend].mean() > 0.5 and
                      result.loc[i, self.name_trend] < 0 and
                      result.loc[i, self.name_consecutive] <= 2):
                    result.loc[i, self.name_reversal] = -1

        # Generate trading signals
        result[self.name_signal] = 0  # Neutral by default

        # Strong trend signals
        strong_uptrend = (
            (result[self.name_trend] > 0) &
            (result[self.name_strength] > self.trend_threshold) &
            (result[self.name_consecutive] >= 3)
        )
        result.loc[strong_uptrend, self.name_signal] = 1

        strong_downtrend = (
            (result[self.name_trend] < 0) &
            (result[self.name_strength] < -self.trend_threshold) &
            (result[self.name_consecutive] >= 3)
        )
        result.loc[strong_downtrend, self.name_signal] = -1

        # Reversal signals
        bullish_reversal = result[self.name_reversal] == 1
        result.loc[bullish_reversal, self.name_signal] = 2  # Strong buy signal

        bearish_reversal = result[self.name_reversal] == -1
        result.loc[bearish_reversal, self.name_signal] = -2  # Strong sell signal

        return result


class RenkoCharts(BaseIndicator):
    """
    Renko Charts system with adaptive box size options.

    Renko charts filter out noise by only showing price movements of a specified size,
    ignoring time and focusing purely on price action.
    """

    category = "price"

    def __init__(
        self,
        box_size: float = None,
        box_size_pct: float = None,
        box_method: str = "atr",
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        wicks: bool = False,
        adaptive_size: bool = False,
        volatility_lookback: int = 50,
        min_box_size_pct: float = 0.25,
        max_box_size_pct: float = 2.0,
        **kwargs
    ):
        """
        Initialize Renko Charts system.

        Args:
            box_size: Fixed box size in price units (specify either this or box_size_pct)
            box_size_pct: Box size as percentage of price (specify either this or box_size)
            box_method: Method to determine box size if not provided ('atr', 'volatility', 'fixed')
            atr_period: Period for ATR calculation when box_method is 'atr'
            atr_multiplier: Multiplier for ATR when determining box size
            wicks: Whether to include wicks in Renko charts
            adaptive_size: Whether to dynamically adjust box size based on volatility
            volatility_lookback: Period for volatility calculation when adaptive_size is True
            min_box_size_pct: Minimum box size percentage when using adaptive sizing
            max_box_size_pct: Maximum box size percentage when using adaptive sizing
            **kwargs: Additional parameters
        """
        # Validate box size inputs
        if box_size is not None and box_size_pct is not None:
            raise ValueError("Specify either box_size or box_size_pct, not both")

        self.box_size = box_size
        self.box_size_pct = box_size_pct
        self.box_method = box_method.lower()
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.wicks = wicks
        self.adaptive_size = adaptive_size
        self.volatility_lookback = volatility_lookback
        self.min_box_size_pct = min_box_size_pct
        self.max_box_size_pct = max_box_size_pct

        # Define output column names
        self.name_open = "renko_open"
        self.name_high = "renko_high"
        self.name_low = "renko_low"
        self.name_close = "renko_close"
        self.name_direction = "renko_direction"
        self.name_box_size = "renko_box_size"
        self.name_box_count = "renko_box_count"
        self.name_trend_strength = "renko_trend_strength"
        self.name_reversal = "renko_reversal"
        self.name_signal = "renko_signal"
        self.name_noise_ratio = "renko_noise_ratio"

    def _calculate_box_size(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the box size for Renko charts.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with box size values
        """
        if self.box_size is not None:
            # Fixed box size in price units
            return pd.Series(self.box_size, index=data.index)

        elif self.box_size_pct is not None:
            # Box size as percentage of price
            return data['close'] * (self.box_size_pct / 100)

        elif self.box_method == 'atr':
            # Box size based on ATR
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))

            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=self.atr_period).mean()

            return atr * self.atr_multiplier

        elif self.box_method == 'volatility':
            # Box size based on price volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=self.atr_period).std() * np.sqrt(self.atr_period)
            return data['close'] * volatility * self.atr_multiplier

        else:  # Default to 1% of price
            return data['close'] * 0.01

    def _adjust_box_size_for_volatility(self, data: pd.DataFrame, base_box_size: pd.Series) -> pd.Series:
        """
        Adjust box size based on recent volatility trends.

        Args:
            data: DataFrame with OHLCV data
            base_box_size: Initial calculated box size

        Returns:
            Adjusted box size Series
        """
        if not self.adaptive_size:
            return base_box_size

        # Calculate recent volatility ratio compared to longer-term volatility
        short_vol = data['close'].pct_change().rolling(window=self.atr_period).std()
        long_vol = data['close'].pct_change().rolling(window=self.volatility_lookback).std()

        # Avoid division by zero
        vol_ratio = short_vol / long_vol.replace(0, np.nan)
        vol_ratio = vol_ratio.fillna(1.0)

        # Scale box size based on volatility ratio
        adjusted_size = base_box_size * vol_ratio

        # Cap the adjustment within reasonable bounds
        min_box = data['close'] * (self.min_box_size_pct / 100)
        max_box = data['close'] * (self.max_box_size_pct / 100)

        return adjusted_size.clip(lower=min_box, upper=max_box)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Renko chart data for the given price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Renko chart values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate box size
        base_box_size = self._calculate_box_size(data)
        box_size = self._adjust_box_size_for_volatility(data, base_box_size)
        result[self.name_box_size] = box_size

        # Initialize Renko columns
        result[self.name_open] = np.nan
        result[self.name_high] = np.nan
        result[self.name_low] = np.nan
        result[self.name_close] = np.nan
        result[self.name_direction] = 0  # 1 for up brick, -1 for down brick
        result[self.name_box_count] = 0  # Number of boxes in the current move

        # Set initial values
        # We'll use the first close price as the anchor price
        anchor_price = result.iloc[0]['close']
        current_direction = 0
        box_count = 0

        # Process each price point
        for i in range(len(result)):
            current_price = result.iloc[i]['close']
            current_box_size = result.iloc[i][self.name_box_size]

            if i == 0:
                # Initialize first Renko bar
                result.loc[result.index[i], self.name_open] = anchor_price
                result.loc[result.index[i], self.name_close] = anchor_price
                result.loc[result.index[i], self.name_high] = result.iloc[i]['high']
                result.loc[result.index[i], self.name_low] = result.iloc[i]['low']
                continue

            # Check if price moved up by at least one box size
            if current_price >= anchor_price + current_box_size:
                # Calculate number of bricks to add
                boxes_to_add = int((current_price - anchor_price) / current_box_size)

                # Update anchor price
                new_anchor = anchor_price + (boxes_to_add * current_box_size)

                # Update direction and box count
                if current_direction <= 0:  # Direction changed or first brick
                    current_direction = 1
                    box_count = boxes_to_add
                else:
                    box_count += boxes_to_add

                # Add Renko bar
                result.loc[result.index[i], self.name_open] = anchor_price
                result.loc[result.index[i], self.name_close] = new_anchor
                result.loc[result.index[i], self.name_direction] = current_direction
                result.loc[result.index[i], self.name_box_count] = box_count

                # Update high/low if including wicks
                if self.wicks:
                    result.loc[result.index[i], self.name_high] = max(new_anchor, result.iloc[i]['high'])
                    result.loc[result.index[i], self.name_low] = min(anchor_price, result.iloc[i]['low'])
                else:
                    result.loc[result.index[i], self.name_high] = new_anchor
                    result.loc[result.index[i], self.name_low] = anchor_price

                anchor_price = new_anchor

            # Check if price moved down by at least one box size
            elif current_price <= anchor_price - current_box_size:
                # Calculate number of bricks to add
                boxes_to_add = int((anchor_price - current_price) / current_box_size)

                # Update anchor price
                new_anchor = anchor_price - (boxes_to_add * current_box_size)

                # Update direction and box count
                if current_direction >= 0:  # Direction changed or first brick
                    current_direction = -1
                    box_count = boxes_to_add
                else:
                    box_count += boxes_to_add

                # Add Renko bar
                result.loc[result.index[i], self.name_open] = anchor_price
                result.loc[result.index[i], self.name_close] = new_anchor
                result.loc[result.index[i], self.name_direction] = current_direction
                result.loc[result.index[i], self.name_box_count] = box_count

                # Update high/low if including wicks
                if self.wicks:
                    result.loc[result.index[i], self.name_high] = max(anchor_price, result.iloc[i]['high'])
                    result.loc[result.index[i], self.name_low] = min(new_anchor, result.iloc[i]['low'])
                else:
                    result.loc[result.index[i], self.name_high] = anchor_price
                    result.loc[result.index[i], self.name_low] = new_anchor

                anchor_price = new_anchor

            else:
                # No new brick, copy values from previous row
                result.loc[result.index[i], self.name_open] = result.loc[result.index[i-1], self.name_open]
                result.loc[result.index[i], self.name_close] = result.loc[result.index[i-1], self.name_close]
                result.loc[result.index[i], self.name_high] = result.loc[result.index[i-1], self.name_high]
                result.loc[result.index[i], self.name_low] = result.loc[result.index[i-1], self.name_low]
                result.loc[result.index[i], self.name_direction] = result.loc[result.index[i-1], self.name_direction]
                result.loc[result.index[i], self.name_box_count] = result.loc[result.index[i-1], self.name_box_count]

        # Calculate trend strength based on consecutive boxes
        result[self.name_trend_strength] = result[self.name_box_count] * np.sign(result[self.name_direction])

        # Detect reversals (direction changes)
        result[self.name_reversal] = result[self.name_direction] != result[self.name_direction].shift(1)

        # Calculate noise ratio (measure of how much price action is filtered out)
        # Higher values indicate more noise in original price data
        price_movement = abs(result['close'] - result['close'].shift(1)).cumsum()
        renko_movement = abs(result[self.name_close] - result[self.name_close].shift(1)).cumsum()

        # Avoid division by zero
        result[self.name_noise_ratio] = np.where(
            renko_movement > 0,
            1 - (renko_movement / price_movement),
            np.nan
        )

        # Generate signals
        result[self.name_signal] = 0

        # Strong trend signals (3+ consecutive boxes)
        strong_uptrend = (result[self.name_direction] > 0) & (result[self.name_box_count] >= 3)
        result.loc[strong_uptrend, self.name_signal] = 1

        strong_downtrend = (result[self.name_direction] < 0) & (result[self.name_box_count] >= 3)
        result.loc[strong_downtrend, self.name_signal] = -1

        # Reversal signals
        reversal_up = result[self.name_reversal] & (result[self.name_direction] > 0)
        result.loc[reversal_up, self.name_signal] = 2  # Strong buy signal

        reversal_down = result[self.name_reversal] & (result[self.name_direction] < 0)
        result.loc[reversal_down, self.name_signal] = -2  # Strong sell signal

        return result


class PointAndFigure(BaseIndicator):
    """
    Point & Figure charts with major pattern identification.

    This indicator filters out small price movements and focuses on significant price changes,
    making it useful for identifying support/resistance levels and trend changes.
    """

    category = "price"

    def __init__(
        self,
        box_size: float = None,
        box_size_pct: float = 0.01,
        reversal_size: int = 3,
        box_method: str = "atr",
        atr_period: int = 14,
        atr_multiplier: float = 0.5,
        use_log_scale: bool = False,
        identify_patterns: bool = True,
        high_low_method: bool = True,
        **kwargs
    ):
        """
        Initialize Point & Figure charts.

        Args:
            box_size: Fixed box size in price units (specify either this or box_size_pct)
            box_size_pct: Box size as percentage of price (specify either this or box_size)
            reversal_size: Number of boxes needed to trigger a reversal
            box_method: Method to determine box size if not provided ('atr', 'volatility', 'fixed')
            atr_period: Period for ATR calculation when box_method is 'atr'
            atr_multiplier: Multiplier for ATR when determining box size
            use_log_scale: Whether to use logarithmic scale for box size
            identify_patterns: Whether to identify classic P&F patterns
            high_low_method: Whether to use high/low or close prices
            **kwargs: Additional parameters
        """
        # Validate box size inputs
        if box_size is not None and box_size_pct is not None:
            raise ValueError("Specify either box_size or box_size_pct, not both")

        self.box_size = box_size
        self.box_size_pct = box_size_pct
        self.reversal_size = reversal_size
        self.box_method = box_method.lower()
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_log_scale = use_log_scale
        self.identify_patterns = identify_patterns
        self.high_low_method = high_low_method

        # Define output column names
        self.name_boxes = "pnf_boxes"  # String representation of boxes
        self.name_direction = "pnf_direction"  # 1 for X column, -1 for O column
        self.name_price = "pnf_price"  # Current price level
        self.name_pattern = "pnf_pattern"  # Identified pattern
        self.name_strength = "pnf_strength"  # Pattern strength
        self.name_support = "pnf_support"  # Support level
        self.name_resistance = "pnf_resistance"  # Resistance level
        self.name_signal = "pnf_signal"  # Trading signal
        self.name_box_size_value = "pnf_box_size"  # Actual box size

    def _calculate_box_size(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the box size for Point & Figure charts.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with box size values
        """
        if self.box_size is not None:
            # Fixed box size in price units
            return pd.Series(self.box_size, index=data.index)

        elif self.box_size_pct is not None:
            # Box size as percentage of price
            if self.use_log_scale:
                # Logarithmic scale - more appropriate for long-term charts
                return data['close'] * (10 ** (np.log10(self.box_size_pct / 100)))
            else:
                return data['close'] * (self.box_size_pct / 100)

        elif self.box_method == 'atr':
            # Box size based on ATR
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))

            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=self.atr_period).mean()

            return atr * self.atr_multiplier

        else:  # Default to 1% of price
            return data['close'] * 0.01

    def _identify_pattern(self, column_history: list, current_direction: int) -> tuple:
        """
        Identify classic Point & Figure chart patterns.

        Args:
            column_history: List of column heights (positive for X, negative for O)
            current_direction: Current column direction (1 for X, -1 for O)

        Returns:
            Tuple of (pattern_name, pattern_strength)
        """
        if len(column_history) < 5:
            return None, 0

        # Extract the last 5 columns for pattern recognition
        last_cols = column_history[-5:] if len(column_history) >= 5 else column_history

        # Double Top/Bottom patterns (breakout above previous X column high or below previous O column low)
        if len(last_cols) >= 3:
            if current_direction == 1:  # X column
                # Double Top Breakout: Current X column exceeds previous X column high
                if len(last_cols) >= 3 and last_cols[-1] > 0 and last_cols[-3] > 0 and last_cols[-1] > last_cols[-3]:
                    return "Double Top Breakout", 7

            else:  # O column
                # Double Bottom Breakout: Current O column exceeds previous O column low
                if len(last_cols) >= 3 and last_cols[-1] < 0 and last_cols[-3] < 0 and abs(last_cols[-1]) > abs(last_cols[-3]):
                    return "Double Bottom Breakdown", 7

        # Triple Top/Bottom patterns
        if len(last_cols) >= 5:
            if current_direction == 1:  # X column
                # Check if we have 3 X columns with similar highs and current breaks above
                if (last_cols[-1] > 0 and last_cols[-3] > 0 and last_cols[-5] > 0 and
                    abs(last_cols[-1] - last_cols[-3]) <= 1 and
                    abs(last_cols[-3] - last_cols[-5]) <= 1 and
                    last_cols[-1] > max(last_cols[-3], last_cols[-5])):
                    return "Triple Top Breakout", 8

            else:  # O column
                # Check if we have 3 O columns with similar lows and current breaks below
                if (last_cols[-1] < 0 and last_cols[-3] < 0 and last_cols[-5] < 0 and
                    abs(abs(last_cols[-1]) - abs(last_cols[-3])) <= 1 and
                    abs(abs(last_cols[-3]) - abs(last_cols[-5])) <= 1 and
                    abs(last_cols[-1]) > max(abs(last_cols[-3]), abs(last_cols[-5]))):
                    return "Triple Bottom Breakdown", 8

        # Bullish/Bearish Signal Reversed
        if len(last_cols) >= 3:
            # Bullish signal: O column fails to make a new low, followed by X column making a higher high
            if (current_direction == 1 and last_cols[-2] < 0 and last_cols[-3] < 0 and
                abs(last_cols[-2]) < abs(last_cols[-3]) and last_cols[-1] > 3):
                return "Bullish Catapult", 9

            # Bearish signal: X column fails to make a new high, followed by O column making a lower low
            elif (current_direction == -1 and last_cols[-2] > 0 and last_cols[-3] > 0 and
                  last_cols[-2] < last_cols[-3] and abs(last_cols[-1]) > 3):
                return "Bearish Catapult", 9

        # Ascending/Descending Triple Top/Bottom
        if len(last_cols) >= 5:
            if current_direction == 1:  # X column
                # Each X column higher than previous, with O columns between them
                if (last_cols[-1] > 0 and last_cols[-3] > 0 and last_cols[-5] > 0 and
                    last_cols[-1] > last_cols[-3] > last_cols[-5]):
                    return "Ascending Triple Top", 8

            else:  # O column
                # Each O column lower than previous, with X columns between them
                if (last_cols[-1] < 0 and last_cols[-3] < 0 and last_cols[-5] < 0 and
                    abs(last_cols[-1]) > abs(last_cols[-3]) > abs(last_cols[-5])):
                    return "Descending Triple Bottom", 8

        # Bullish/Bearish Triangle
        if len(last_cols) >= 6:
            # Look for converging pattern (higher lows, lower highs)
            x_cols = [col for col in last_cols if col > 0]
            o_cols = [col for col in last_cols if col < 0]

            if len(x_cols) >= 3 and len(o_cols) >= 3:
                # Check if X columns show lower highs
                x_trend = all(x_cols[i] < x_cols[i-1] for i in range(1, len(x_cols)))
                # Check if O columns show higher lows (less negative)
                o_trend = all(abs(o_cols[i]) < abs(o_cols[i-1]) for i in range(1, len(o_cols)))

                if x_trend and o_trend:
                    return "Triangle Consolidation", 6

        # No recognized pattern
        return None, 0

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Point & Figure chart data for the given price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Point & Figure chart values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate box size
        box_size = self._calculate_box_size(data)
        result[self.name_box_size_value] = box_size

        # Initialize P&F columns
        result[self.name_boxes] = ""
        result[self.name_direction] = 0
        result[self.name_price] = np.nan
        result[self.name_pattern] = None
        result[self.name_strength] = 0
        result[self.name_support] = np.nan
        result[self.name_resistance] = np.nan
        result[self.name_signal] = 0

        # Set initial values
        if self.high_low_method:
            # For high/low method, use high for X columns and low for O columns
            anchor_price = data.iloc[0]['close']
        else:
            # For close method, just use closing prices
            anchor_price = data.iloc[0]['close']

        current_direction = 0  # 0 is initial, 1 for X column, -1 for O column
        current_column_boxes = 0
        column_history = []  # Stores height of each column (positive for X, negative for O)

        for i in range(len(result)):
            current_box_size = result.iloc[i][self.name_box_size_value]

            if self.high_low_method:
                high_price = result.iloc[i]['high']
                low_price = result.iloc[i]['low']
            else:
                high_price = result.iloc[i]['close']
                low_price = result.iloc[i]['close']

            # Initial column determination
            if i == 0 or current_direction == 0:
                if high_price >= anchor_price + current_box_size:
                    current_direction = 1  # X column
                    current_column_boxes = int((high_price - anchor_price) / current_box_size)
                    anchor_price += current_column_boxes * current_box_size
                    column_history.append(current_column_boxes)
                elif low_price <= anchor_price - current_box_size:
                    current_direction = -1  # O column
                    current_column_boxes = int((anchor_price - low_price) / current_box_size)
                    anchor_price -= current_column_boxes * current_box_size
                    column_history.append(-current_column_boxes)
                else:
                    current_direction = 0
                    current_column_boxes = 0

            # X column - check if we continue X, reverse to O, or do nothing
            elif current_direction == 1:
                if high_price >= anchor_price + current_box_size:
                    # Continue X column
                    new_boxes = int((high_price - anchor_price) / current_box_size)
                    anchor_price += new_boxes * current_box_size
                    current_column_boxes += new_boxes
                    column_history[-1] = current_column_boxes
                elif low_price <= anchor_price - (current_box_size * self.reversal_size):
                    # Reverse to O column
                    new_boxes = int((anchor_price - low_price) / current_box_size)
                    current_direction = -1
                    anchor_price -= new_boxes * current_box_size
                    current_column_boxes = new_boxes
                    column_history.append(-current_column_boxes)

            # O column - check if we continue O, reverse to X, or do nothing
            elif current_direction == -1:
                if low_price <= anchor_price - current_box_size:
                    # Continue O column
                    new_boxes = int((anchor_price - low_price) / current_box_size)
                    anchor_price -= new_boxes * current_box_size
                    current_column_boxes += new_boxes
                    column_history[-1] = -current_column_boxes
                elif high_price >= anchor_price + (current_box_size * self.reversal_size):
                    # Reverse to X column
                    new_boxes = int((high_price - anchor_price) / current_box_size)
                    current_direction = 1
                    anchor_price += new_boxes * current_box_size
                    current_column_boxes = new_boxes
                    column_history.append(current_column_boxes)

            # Record the current state
            result.loc[result.index[i], self.name_direction] = current_direction
            result.loc[result.index[i], self.name_price] = anchor_price

            # Create box string (e.g., "XXXXX" or "OOOOO")
            if current_direction == 1:
                result.loc[result.index[i], self.name_boxes] = "X" * current_column_boxes
            elif current_direction == -1:
                result.loc[result.index[i], self.name_boxes] = "O" * current_column_boxes

            # Identify patterns if requested and we have enough history
            if self.identify_patterns and len(column_history) >= 3:
                pattern, strength = self._identify_pattern(column_history, current_direction)
                result.loc[result.index[i], self.name_pattern] = pattern
                result.loc[result.index[i], self.name_strength] = strength

                # Generate trading signals based on patterns
                if pattern is not None:
                    # Bullish patterns
                    if pattern in ["Double Top Breakout", "Triple Top Breakout", "Bullish Catapult"]:
                        result.loc[result.index[i], self.name_signal] = strength / 10  # Scale to 0-1
                    # Bearish patterns
                    elif pattern in ["Double Bottom Breakdown", "Triple Bottom Breakdown", "Bearish Catapult"]:
                        result.loc[result.index[i], self.name_signal] = -strength / 10  # Scale to -1-0

            # Calculate support and resistance levels
            if len(column_history) >= 3:
                # Support is highest low of recent O columns
                o_columns = [abs(col) * current_box_size for col in column_history[-5:] if col < 0]
                if o_columns:
                    support_level = anchor_price - max(o_columns) if current_direction == 1 else anchor_price
                    result.loc[result.index[i], self.name_support] = support_level

                # Resistance is highest high of recent X columns
                x_columns = [col * current_box_size for col in column_history[-5:] if col > 0]
                if x_columns:
                    resistance_level = anchor_price if current_direction == -1 else anchor_price + max(x_columns)
                    result.loc[result.index[i], self.name_resistance] = resistance_level

        return result
