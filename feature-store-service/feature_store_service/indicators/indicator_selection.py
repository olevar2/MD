"""
Intelligent Indicator Selection System Module.

This module provides mechanisms for dynamically selecting technical indicators
based on market conditions and tracking their performance over time.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
import pandas as pd
import numpy as np
from enum import Enum
import json
import datetime
from pathlib import Path
import logging
import warnings
from collections import defaultdict

from feature_store_service.indicators.base_indicator import BaseIndicator


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Enum representing different market regimes/conditions."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_VOLATILE = "ranging_volatile"
    RANGING_TIGHT = "ranging_tight"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class IndicatorCategory(Enum):
    """Enum representing different indicator categories."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN = "pattern"
    OSCILLATOR = "oscillator"
    CUSTOM = "custom"


class IndicatorPerformanceMetrics:
    """Class for tracking and calculating indicator performance metrics."""
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize performance metrics tracking.
        
        Args:
            lookback_periods: List of periods to track (e.g., [5, 10, 20, 50])
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]
        self.signal_accuracy = {}  # Accuracy of indicator signals
        self.prediction_power = {}  # Correlation with future price movements
        self.timeliness = {}  # Early signal vs. lagging signal metrics
        self.consistency = {}  # Consistency of signals across timeframes
        self.noise_ratio = {}  # Signal-to-noise ratio
        
    def calculate_metrics(
        self, 
        indicator_values: pd.Series, 
        price_data: pd.DataFrame, 
        signals: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for an indicator.
        
        Args:
            indicator_values: Series with indicator values
            price_data: DataFrame with OHLCV data
            signals: Optional series with binary signals derived from indicator
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Skip calculation if insufficient data
        if len(indicator_values) < max(self.lookback_periods):
            return {'error': 'Insufficient data for metrics calculation'}
            
        # Calculate metrics for each lookback period
        for period in self.lookback_periods:
            metrics[f'period_{period}'] = self._calculate_period_metrics(
                indicator_values.iloc[-period:], 
                price_data.iloc[-period:],
                signals.iloc[-period:] if signals is not None else None
            )
            
        # Calculate overall metrics
        metrics['overall'] = self._calculate_overall_metrics(metrics)
        
        return metrics
        
    def _calculate_period_metrics(
        self, 
        indicator_values: pd.Series, 
        price_data: pd.DataFrame, 
        signals: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate metrics for a specific lookback period."""
        metrics = {}
        
        # Get close prices for various calculations
        close_prices = price_data['close']
        
        # 1. Calculate prediction power (correlation with future returns)
        # We'll check correlation with future returns at different horizons
        for horizon in [1, 3, 5, 10]:
            future_returns = close_prices.pct_change(horizon).shift(-horizon)
            if future_returns.iloc[:-horizon].isna().sum() < len(future_returns) - horizon:
                correlation = indicator_values[:-horizon].corr(future_returns[:-horizon])
                metrics[f'prediction_power_{horizon}'] = correlation
                
        # 2. Calculate noise ratio (signal-to-noise)
        if len(indicator_values) > 1:
            # Simple measure: ratio of directional changes to total periods
            direction_changes = np.sum(np.diff(np.sign(np.diff(indicator_values))) != 0)
            noise_ratio = direction_changes / (len(indicator_values) - 2) if len(indicator_values) > 2 else 0
            metrics['noise_ratio'] = noise_ratio
            
        # 3. Calculate signal accuracy if signals are provided
        if signals is not None and not signals.isna().all():
            # For simplicity, we'll assume signals are binary (1: buy, -1: sell, 0: neutral)
            # and evaluate if the market moved in the predicted direction
            
            # Check different prediction horizons
            for horizon in [1, 3, 5, 10]:
                if len(signals) <= horizon:
                    continue
                    
                # Calculate future returns for the horizon
                future_returns = close_prices.pct_change(horizon).shift(-horizon)
                
                # Calculate accuracy: correct predictions / total predictions
                buy_signals = signals == 1
                sell_signals = signals == -1
                
                # Only evaluate non-NaN instances
                valid_buy = buy_signals & ~future_returns.shift(-horizon).isna()
                valid_sell = sell_signals & ~future_returns.shift(-horizon).isna()
                
                if valid_buy.sum() > 0:
                    buy_accuracy = (future_returns[valid_buy] > 0).mean()
                    metrics[f'buy_accuracy_{horizon}'] = buy_accuracy
                    
                if valid_sell.sum() > 0:
                    sell_accuracy = (future_returns[valid_sell] < 0).mean()
                    metrics[f'sell_accuracy_{horizon}'] = sell_accuracy
                
                # Calculate overall accuracy
                if (valid_buy.sum() + valid_sell.sum()) > 0:
                    correct_predictions = ((future_returns[valid_buy] > 0).sum() + 
                                        (future_returns[valid_sell] < 0).sum())
                    total_predictions = valid_buy.sum() + valid_sell.sum()
                    metrics[f'signal_accuracy_{horizon}'] = correct_predictions / total_predictions
                    
        # 4. Calculate timeliness (how early the indicator signals change)
        if len(indicator_values) > 1 and len(close_prices) > 1:
            # Get the turning points in the price series
            price_direction = np.sign(close_prices.diff())
            price_turns = (price_direction.shift(1) != price_direction) & (price_direction != 0)
            price_turn_indices = price_turns[price_turns].index.tolist()
            
            # Get the turning points in the indicator
            indicator_direction = np.sign(indicator_values.diff())
            indicator_turns = (indicator_direction.shift(1) != indicator_direction) & (indicator_direction != 0)
            indicator_turn_indices = indicator_turns[indicator_turns].index.tolist()
            
            # Calculate average lead/lag between indicator turns and price turns
            lead_lag_periods = []
            
            for ind_turn in indicator_turn_indices:
                # Find the closest price turn after this indicator turn
                future_price_turns = [pt for pt in price_turn_indices if pt > ind_turn]
                if future_price_turns:
                    closest_turn = min(future_price_turns)
                    # Calculate the difference in periods
                    try:
                        lead_periods = close_prices.index.get_indexer([closest_turn])[0] - \
                                    close_prices.index.get_indexer([ind_turn])[0]
                        lead_lag_periods.append(lead_periods)
                    except:
                        # Handle case where indices may not be directly comparable
                        pass
                        
            if lead_lag_periods:
                metrics['avg_lead_periods'] = sum(lead_lag_periods) / len(lead_lag_periods)
                # Positive value means indicator leads price, negative means it lags
                
        return metrics
        
    def _calculate_overall_metrics(self, period_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall metrics across all periods."""
        overall = {}
        
        # Aggregate prediction power
        prediction_powers = []
        for period, metrics in period_metrics.items():
            for key, value in metrics.items():
                if key.startswith('prediction_power_'):
                    prediction_powers.append(value)
                    
        if prediction_powers:
            overall['avg_prediction_power'] = sum(prediction_powers) / len(prediction_powers)
            
        # Aggregate noise ratios
        noise_ratios = [metrics.get('noise_ratio', np.nan) for metrics in period_metrics.values()]
        noise_ratios = [nr for nr in noise_ratios if not np.isnan(nr)]
        if noise_ratios:
            overall['avg_noise_ratio'] = sum(noise_ratios) / len(noise_ratios)
            
        # Aggregate signal accuracy
        accuracies = []
        for period, metrics in period_metrics.items():
            for key, value in metrics.items():
                if key.startswith('signal_accuracy_'):
                    accuracies.append(value)
                    
        if accuracies:
            overall['avg_signal_accuracy'] = sum(accuracies) / len(accuracies)
            
        # Aggregate timeliness
        lead_periods = [metrics.get('avg_lead_periods', np.nan) for metrics in period_metrics.values()]
        lead_periods = [lp for lp in lead_periods if not np.isnan(lp)]
        if lead_periods:
            overall['avg_lead_periods'] = sum(lead_periods) / len(lead_periods)
            
        return overall


class MarketRegimeClassifier:
    """
    Classifies market conditions into different regimes.
    
    This classifier uses various market characteristics to determine the
    current market regime (trending, ranging, etc.) which helps in
    indicator selection.
    """
    
    def __init__(
        self, 
        trend_period: int = 50,
        volatility_period: int = 20,
        volume_period: int = 20,
        **kwargs
    ):
        """
        Initialize Market Regime Classifier.
        
        Args:
            trend_period: Period for trend determination
            volatility_period: Period for volatility calculation
            volume_period: Period for volume analysis
            **kwargs: Additional parameters
        """
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.volume_period = volume_period
        
    def classify(self, data: pd.DataFrame) -> MarketRegime:
        """
        Classify the current market regime based on price data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            MarketRegime enum value
        """
        if len(data) < max(self.trend_period, self.volatility_period, self.volume_period):
            return MarketRegime.UNKNOWN
            
        # Extract relevant data
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume')  # Volume may not always be available
        
        # Calculate trend metrics
        sma_long = close.rolling(window=self.trend_period).mean()
        sma_short = close.rolling(window=self.trend_period // 3).mean()
        
        # Calculate linear regression slope over the trend period
        x = np.arange(self.trend_period)
        y = close.iloc[-self.trend_period:].values
        slope, _ = np.polyfit(x, y, 1)
        
        # Calculate volatility
        atr = self._calculate_atr(data, self.volatility_period)
        avg_atr = atr.rolling(window=self.volatility_period).mean().iloc[-1]
        
        # Calculate volume metrics (if available)
        if volume is not None:
            vol_ratio = volume.iloc[-1] / volume.rolling(window=self.volume_period).mean().iloc[-1]
        else:
            vol_ratio = 1.0
            
        # Determine trend direction and strength
        price_above_sma = close.iloc[-1] > sma_long.iloc[-1]
        short_above_long = sma_short.iloc[-1] > sma_long.iloc[-1]
        
        # Calculate price range as percentage of price
        price_range = (high.iloc[-self.volatility_period:].max() - 
                    low.iloc[-self.volatility_period:].min()) / close.iloc[-1]
                    
        # Calculate consecutive moves in same direction
        direction = np.sign(close.diff()).iloc[-5:]
        consecutive_moves = (direction == direction.iloc[-1]).sum()
        
        # Calculate RSI for overbought/oversold conditions
        rsi = self._calculate_rsi(close, 14)
        
        # Basic regime classification logic
        if short_above_long and price_above_sma and slope > 0:
            # Strong bullish trend
            if vol_ratio > 1.2 and rsi > 70:
                return MarketRegime.BREAKOUT  # Potential breakout
            return MarketRegime.TRENDING_BULLISH
        elif short_above_long == False and price_above_sma == False and slope < 0:
            # Strong bearish trend
            if vol_ratio > 1.2 and rsi < 30:
                return MarketRegime.REVERSAL  # Potential reversal
            return MarketRegime.TRENDING_BEARISH
        elif abs(slope) < 0.0002 and avg_atr / close.iloc[-1] < 0.005:
            # Low volatility range
            return MarketRegime.RANGING_TIGHT
        elif abs(slope) < 0.0005 and avg_atr / close.iloc[-1] >= 0.005:
            # High volatility range
            return MarketRegime.RANGING_VOLATILE
        elif consecutive_moves >= 4 and vol_ratio > 1.5:
            if rsi > 70 or rsi < 30:
                return MarketRegime.REVERSAL  # Potential reversal
            return MarketRegime.BREAKOUT  # Potential breakout
        else:
            return MarketRegime.UNKNOWN
            
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]


class IndicatorSelectionEngine:
    """
    Intelligent Indicator Selection Engine
    
    This engine dynamically selects the most appropriate technical indicators
    based on current market conditions, historical performance, and predefined rules.
    """
    
    def __init__(
        self, 
        indicators_registry: Dict[str, BaseIndicator],
        performance_history_path: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Indicator Selection Engine.
        
        Args:
            indicators_registry: Dictionary of available indicators (name -> indicator)
            performance_history_path: Path to save/load performance history
            config_path: Path to selection engine configuration
            **kwargs: Additional parameters
        """
        self.indicators_registry = indicators_registry
        self.performance_history_path = performance_history_path
        self.config_path = config_path
        
        # Initialize components
        self.regime_classifier = MarketRegimeClassifier()
        self.performance_tracker = IndicatorPerformanceMetrics()
        
        # Load or initialize performance history and configurations
        self.performance_history = self._load_performance_history()
        self.selection_config = self._load_selection_config()
        
        # Map each indicator to a category
        self.indicator_categories = self._categorize_indicators()
        
        # Initialize indicator combinations for different regimes
        self.regime_indicators = self._initialize_regime_indicators()
        
    def select_indicators(
        self, 
        data: pd.DataFrame, 
        max_indicators: int = 10
    ) -> Dict[str, BaseIndicator]:
        """
        Select the most appropriate indicators for the current market conditions.
        
        Args:
            data: DataFrame with OHLCV data
            max_indicators: Maximum number of indicators to select
            
        Returns:
            Dictionary of selected indicators (name -> indicator)
        """
        # Classify the current market regime
        current_regime = self.regime_classifier.classify(data)
        logger.info(f"Current market regime: {current_regime.value}")
        
        # Get indicator preferences for the current regime
        preferred_indicators = self.regime_indicators.get(current_regime, set())
        
        # Add some indicators from each important category
        selected_categories = self._get_important_categories(current_regime)
        
        # Sort indicators by performance
        sorted_indicators = self._sort_indicators_by_performance(data, current_regime)
        
        # Select the top indicators while ensuring category coverage
        selected = {}
        category_counts = defaultdict(int)
        
        # First, add preferred indicators for this regime (if they're performing well)
        for name, _ in sorted_indicators:
            if len(selected) >= max_indicators:
                break
                
            if name in preferred_indicators:
                indicator = self.indicators_registry[name]
                selected[name] = indicator
                category = self.indicator_categories.get(name, IndicatorCategory.CUSTOM)
                category_counts[category] += 1
                
        # Then, ensure coverage of important categories
        for category in selected_categories:
            if category_counts[category] < 2 and len(selected) < max_indicators:
                # Add top performing indicators from this category
                for name, _ in sorted_indicators:
                    if len(selected) >= max_indicators:
                        break
                        
                    if name not in selected and self.indicator_categories.get(name) == category:
                        indicator = self.indicators_registry[name]
                        selected[name] = indicator
                        category_counts[category] += 1
                        
                        if category_counts[category] >= 2:
                            break
                            
        # Finally, add any top performers not yet included
        for name, _ in sorted_indicators:
            if len(selected) >= max_indicators:
                break
                
            if name not in selected:
                indicator = self.indicators_registry[name]
                selected[name] = indicator
                
        logger.info(f"Selected {len(selected)} indicators for {current_regime.value} regime")
        return selected
        
    def update_performance(
        self, 
        indicator_name: str,
        indicator_values: pd.Series,
        price_data: pd.DataFrame,
        signals: Optional[pd.Series] = None,
        regime: Optional[MarketRegime] = None
    ) -> None:
        """
        Update the performance metrics for an indicator.
        
        Args:
            indicator_name: Name of the indicator
            indicator_values: Series with indicator values
            price_data: DataFrame with OHLCV data
            signals: Optional series with binary signals derived from indicator
            regime: Optional market regime for this performance data
        """
        # Calculate performance metrics
        metrics = self.performance_tracker.calculate_metrics(
            indicator_values, price_data, signals
        )
        
        # Determine the regime if not provided
        if regime is None:
            regime = self.regime_classifier.classify(price_data)
            
        # Update performance history
        timestamp = datetime.datetime.now().isoformat()
        
        if indicator_name not in self.performance_history:
            self.performance_history[indicator_name] = {
                'overall': [],
                'by_regime': {regime.value: [] for regime in MarketRegime}
            }
            
        # Add overall metrics
        self.performance_history[indicator_name]['overall'].append({
            'timestamp': timestamp,
            'metrics': metrics['overall']
        })
        
        # Add regime-specific metrics
        self.performance_history[indicator_name]['by_regime'][regime.value].append({
            'timestamp': timestamp,
            'metrics': metrics['overall']
        })
        
        # Limit history size to prevent memory issues
        max_history = 100
        if len(self.performance_history[indicator_name]['overall']) > max_history:
            self.performance_history[indicator_name]['overall'] = (
                self.performance_history[indicator_name]['overall'][-max_history:]
            )
            
        for regime_val in self.performance_history[indicator_name]['by_regime']:
            if len(self.performance_history[indicator_name]['by_regime'][regime_val]) > max_history:
                self.performance_history[indicator_name]['by_regime'][regime_val] = (
                    self.performance_history[indicator_name]['by_regime'][regime_val][-max_history:]
                )
                
        # Save updated performance history
        self._save_performance_history()
        
    def optimize_selections(self) -> None:
        """
        Optimize indicator selections based on historical performance.
        
        This method is typically called periodically to update the selection rules.
        """
        # For each market regime, update the preferred indicators
        for regime in MarketRegime:
            # Skip UNKNOWN regime
            if regime == MarketRegime.UNKNOWN:
                continue
                
            # Get the top performers for this regime
            top_performers = self._get_top_performers_for_regime(regime)
            
            # Update the regime indicators
            if top_performers:
                # Keep some existing indicators for stability
                existing = set(self.regime_indicators.get(regime, []))
                to_keep = set(list(existing)[:len(existing)//2]) if existing else set()
                
                # Add new top performers
                new_selection = to_keep.union(set(top_performers))
                self.regime_indicators[regime] = new_selection
                
        # Update the selection config
        self._update_selection_config()
        
    def _load_performance_history(self) -> Dict[str, Any]:
        """Load performance history from file or initialize if not exists."""
        if self.performance_history_path and Path(self.performance_history_path).exists():
            try:
                with open(self.performance_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load performance history: {e}")
                
        return {}
        
    def _save_performance_history(self) -> None:
        """Save performance history to file."""
        if self.performance_history_path:
            try:
                with open(self.performance_history_path, 'w') as f:
                    json.dump(self.performance_history, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save performance history: {e}")
                
    def _load_selection_config(self) -> Dict[str, Any]:
        """Load selection configuration from file or initialize defaults."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load selection config: {e}")
                
        # Default configuration
        return {
            'regime_category_weights': {
                MarketRegime.TRENDING_BULLISH.value: {
                    IndicatorCategory.TREND.value: 0.6,
                    IndicatorCategory.MOMENTUM.value: 0.3,
                    IndicatorCategory.VOLATILITY.value: 0.1,
                    IndicatorCategory.VOLUME.value: 0.2,
                    IndicatorCategory.SUPPORT_RESISTANCE.value: 0.1,
                    IndicatorCategory.PATTERN.value: 0.1,
                    IndicatorCategory.OSCILLATOR.value: 0.2,
                    IndicatorCategory.CUSTOM.value: 0.1
                },
                MarketRegime.TRENDING_BEARISH.value: {
                    IndicatorCategory.TREND.value: 0.6,
                    IndicatorCategory.MOMENTUM.value: 0.3,
                    IndicatorCategory.VOLATILITY.value: 0.1,
                    IndicatorCategory.VOLUME.value: 0.2,
                    IndicatorCategory.SUPPORT_RESISTANCE.value: 0.1,
                    IndicatorCategory.PATTERN.value: 0.1,
                    IndicatorCategory.OSCILLATOR.value: 0.2,
                    IndicatorCategory.CUSTOM.value: 0.1
                },
                MarketRegime.RANGING_VOLATILE.value: {
                    IndicatorCategory.TREND.value: 0.2,
                    IndicatorCategory.MOMENTUM.value: 0.2,
                    IndicatorCategory.VOLATILITY.value: 0.4,
                    IndicatorCategory.VOLUME.value: 0.3,
                    IndicatorCategory.SUPPORT_RESISTANCE.value: 0.4,
                    IndicatorCategory.PATTERN.value: 0.3,
                    IndicatorCategory.OSCILLATOR.value: 0.4,
                    IndicatorCategory.CUSTOM.value: 0.1
                },
                MarketRegime.RANGING_TIGHT.value: {
                    IndicatorCategory.TREND.value: 0.1,
                    IndicatorCategory.MOMENTUM.value: 0.2,
                    IndicatorCategory.VOLATILITY.value: 0.3,
                    IndicatorCategory.VOLUME.value: 0.4,
                    IndicatorCategory.SUPPORT_RESISTANCE.value: 0.5,
                    IndicatorCategory.PATTERN.value: 0.3,
                    IndicatorCategory.OSCILLATOR.value: 0.4,
                    IndicatorCategory.CUSTOM.value: 0.1
                },
                MarketRegime.BREAKOUT.value: {
                    IndicatorCategory.TREND.value: 0.4,
                    IndicatorCategory.MOMENTUM.value: 0.5,
                    IndicatorCategory.VOLATILITY.value: 0.4,
                    IndicatorCategory.VOLUME.value: 0.5,
                    IndicatorCategory.SUPPORT_RESISTANCE.value: 0.3,
                    IndicatorCategory.PATTERN.value: 0.3,
                    IndicatorCategory.OSCILLATOR.value: 0.2,
                    IndicatorCategory.CUSTOM.value: 0.1
                },
                MarketRegime.REVERSAL.value: {
                    IndicatorCategory.TREND.value: 0.3,
                    IndicatorCategory.MOMENTUM.value: 0.4,
                    IndicatorCategory.VOLATILITY.value: 0.3,
                    IndicatorCategory.VOLUME.value: 0.4,
                    IndicatorCategory.SUPPORT_RESISTANCE.value: 0.3,
                    IndicatorCategory.PATTERN.value: 0.4,
                    IndicatorCategory.OSCILLATOR.value: 0.5,
                    IndicatorCategory.CUSTOM.value: 0.1
                },
                MarketRegime.UNKNOWN.value: {
                    IndicatorCategory.TREND.value: 0.3,
                    IndicatorCategory.MOMENTUM.value: 0.3,
                    IndicatorCategory.VOLATILITY.value: 0.3,
                    IndicatorCategory.VOLUME.value: 0.3,
                    IndicatorCategory.SUPPORT_RESISTANCE.value: 0.3,
                    IndicatorCategory.PATTERN.value: 0.3,
                    IndicatorCategory.OSCILLATOR.value: 0.3,
                    IndicatorCategory.CUSTOM.value: 0.1
                }
            },
            'performance_weights': {
                'avg_prediction_power': 0.4,
                'avg_signal_accuracy': 0.3,
                'avg_lead_periods': 0.2,
                'avg_noise_ratio': -0.1  # Negative because lower noise is better
            }
        }
        
    def _update_selection_config(self) -> None:
        """Update and save the selection configuration."""
        if self.config_path:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self.selection_config, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save selection config: {e}")
                
    def _categorize_indicators(self) -> Dict[str, IndicatorCategory]:
        """Categorize all registered indicators."""
        categories = {}
        
        for name, indicator in self.indicators_registry.items():
            # Try to get category from indicator attributes
            if hasattr(indicator, 'category'):
                cat_str = getattr(indicator, 'category')
                # Map string category to enum
                try:
                    categories[name] = IndicatorCategory(cat_str)
                except ValueError:
                    # Default to CUSTOM if category doesn't match enum values
                    categories[name] = IndicatorCategory.CUSTOM
            else:
                # Attempt to guess category from name
                if any(term in name.lower() for term in ['ma', 'ema', 'sma', 'wma', 'average']):
                    categories[name] = IndicatorCategory.TREND
                elif any(term in name.lower() for term in ['rsi', 'cci', 'macd', 'momentum']):
                    categories[name] = IndicatorCategory.MOMENTUM
                elif any(term in name.lower() for term in ['atr', 'bollinger', 'std', 'volatility']):
                    categories[name] = IndicatorCategory.VOLATILITY
                elif any(term in name.lower() for term in ['volume', 'obv', 'money', 'flow']):
                    categories[name] = IndicatorCategory.VOLUME
                elif any(term in name.lower() for term in ['support', 'resistance', 'pivot']):
                    categories[name] = IndicatorCategory.SUPPORT_RESISTANCE
                elif any(term in name.lower() for term in ['pattern', 'formation', 'candle']):
                    categories[name] = IndicatorCategory.PATTERN
                elif any(term in name.lower() for term in ['oscillator', 'index', 'stoch']):
                    categories[name] = IndicatorCategory.OSCILLATOR
                else:
                    categories[name] = IndicatorCategory.CUSTOM
                    
        return categories
        
    def _initialize_regime_indicators(self) -> Dict[MarketRegime, Set[str]]:
        """Initialize preferred indicators for each market regime."""
        regime_indicators = {regime: set() for regime in MarketRegime}
        
        # If we have performance history, use it to select initial indicators
        if self.performance_history:
            for regime in MarketRegime:
                top_performers = self._get_top_performers_for_regime(regime)
                regime_indicators[regime] = set(top_performers)
        else:
            # Initialize with basic selections for each regime
            # For trending bullish regime, prefer trend-following indicators
            regime_indicators[MarketRegime.TRENDING_BULLISH] = {
                name for name, cat in self.indicator_categories.items()
                if cat in [IndicatorCategory.TREND, IndicatorCategory.MOMENTUM]
            }
            
            # For trending bearish regime, prefer trend and momentum indicators
            regime_indicators[MarketRegime.TRENDING_BEARISH] = {
                name for name, cat in self.indicator_categories.items()
                if cat in [IndicatorCategory.TREND, IndicatorCategory.MOMENTUM]
            }
            
            # For ranging volatile regime, prefer oscillators and support/resistance
            regime_indicators[MarketRegime.RANGING_VOLATILE] = {
                name for name, cat in self.indicator_categories.items()
                if cat in [IndicatorCategory.OSCILLATOR, IndicatorCategory.SUPPORT_RESISTANCE]
            }
            
            # For ranging tight regime, prefer breakout indicators and volume
            regime_indicators[MarketRegime.RANGING_TIGHT] = {
                name for name, cat in self.indicator_categories.items()
                if cat in [IndicatorCategory.VOLATILITY, IndicatorCategory.VOLUME]
            }
            
            # For breakout regime, prefer momentum and volume indicators
            regime_indicators[MarketRegime.BREAKOUT] = {
                name for name, cat in self.indicator_categories.items()
                if cat in [IndicatorCategory.MOMENTUM, IndicatorCategory.VOLUME]
            }
            
            # For reversal regime, prefer pattern and oscillator indicators
            regime_indicators[MarketRegime.REVERSAL] = {
                name for name, cat in self.indicator_categories.items()
                if cat in [IndicatorCategory.PATTERN, IndicatorCategory.OSCILLATOR]
            }
            
        return regime_indicators
        
    def _get_important_categories(self, regime: MarketRegime) -> List[IndicatorCategory]:
        """Get the most important indicator categories for a market regime."""
        # Get weights for each category in this regime
        weights = self.selection_config['regime_category_weights'][regime.value]
        
        # Sort categories by weight
        sorted_categories = sorted(
            [(IndicatorCategory(cat), weight) for cat, weight in weights.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return the top categories
        return [cat for cat, _ in sorted_categories[:4]]
        
    def _sort_indicators_by_performance(
        self, data: pd.DataFrame, regime: MarketRegime
    ) -> List[Tuple[str, float]]:
        """
        Sort indicators by performance score for the current market regime.
        
        Returns a list of (indicator_name, score) tuples, sorted by score.
        """
        scores = []
        
        for name in self.indicators_registry:
            # Calculate a performance score for this indicator
            score = self._calculate_indicator_score(name, regime)
            scores.append((name, score))
            
        # Sort by score in descending order
        return sorted(scores, key=lambda x: x[1], reverse=True)
        
    def _calculate_indicator_score(self, indicator_name: str, regime: MarketRegime) -> float:
        """Calculate a performance score for an indicator in the given regime."""
        # Default score if no performance history
        default_score = 0.5
        
        # If no performance history for this indicator, return default score
        if indicator_name not in self.performance_history:
            # Slightly boost score based on indicator category appropriateness for regime
            category = self.indicator_categories.get(indicator_name)
            if category:
                category_weight = self.selection_config['regime_category_weights'][regime.value].get(
                    category.value, 0.1
                )
                return default_score + (category_weight * 0.1)
            return default_score
            
        # Get the performance metrics for this regime
        regime_metrics = self.performance_history[indicator_name]['by_regime'].get(regime.value)
        
        # If no regime-specific metrics, use overall metrics
        if not regime_metrics:
            overall_metrics = self.performance_history[indicator_name]['overall']
            if not overall_metrics:
                return default_score
            metrics = overall_metrics[-1]['metrics']
        else:
            # Use the most recent metrics for this regime
            metrics = regime_metrics[-1]['metrics']
            
        # Calculate weighted score using the weights from config
        score = 0.0
        weight_sum = 0.0
        
        for metric, weight in self.selection_config['performance_weights'].items():
            if metric in metrics:
                score += metrics[metric] * weight
                weight_sum += abs(weight)
                
        # Normalize score
        if weight_sum > 0:
            score = score / weight_sum
            
        return score
        
    def _get_top_performers_for_regime(self, regime: MarketRegime) -> List[str]:
        """Get the top performing indicators for a specific market regime."""
        # Initialize scores for all indicators
        scores = []
        
        for name in self.indicators_registry:
            # Calculate score for this indicator in this regime
            score = self._calculate_indicator_score(name, regime)
            scores.append((name, score))
            
        # Sort by score and return top 10
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_scores[:10]]
