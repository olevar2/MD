"""
Regime Transition Predictor Module

This module provides functionality for detecting early signs of market regime transitions
before they fully manifest, allowing for proactive strategy adjustments.

Part of Phase 4 implementation to enhance market regime transition detection.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
from analysis_engine.services.market_regime_detector import MarketRegime, MarketRegimeAnalyzer
from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class TransitionProbability(Enum):
    """Probability levels for regime transitions"""
    VERY_LOW = 'very_low'
    LOW = 'low'
    MODERATE = 'moderate'
    HIGH = 'high'
    VERY_HIGH = 'very_high'


class RegimeTransitionPredictor:
    """
    Predicts transitions between market regimes before they fully manifest.

    This service:
    - Calculates transition probabilities between different market regimes
    - Identifies early warning indicators for regime shifts
    - Tracks historical regime transitions
    - Provides predictions for upcoming regime changes
    """

    def __init__(self, regime_detector: Optional[MarketRegimeAnalyzer]=None,
        transition_history_size: int=100, early_warning_threshold: float=
        0.7, lookback_periods: int=50, correlation_service: Optional[Any]=None
        ):
        """
        Initialize the regime transition predictor.

        Args:
            regime_detector: Optional market regime detector
            transition_history_size: Maximum number of transitions to keep in history
            early_warning_threshold: Threshold for early warning signals (0.0-1.0)
            lookback_periods: Number of periods to look back for analysis
            correlation_service: Optional correlation tracking service for inter-market analysis
        """
        self.regime_detector = regime_detector or MarketRegimeAnalyzer()
        self.transition_history_size = transition_history_size
        self.early_warning_threshold = early_warning_threshold
        self.lookback_periods = lookback_periods
        self.correlation_service = correlation_service
        self.logger = logging.getLogger(__name__)
        self.transition_history = {}
        self.transition_probabilities = self._initialize_transition_matrix()
        self.early_warning_indicators = {}
        self.correlated_markets = {}
        self.logger.info(
            f'RegimeTransitionPredictor initialized with {len(MarketRegime)} regimes'
            )

    def _initialize_transition_matrix(self) ->Dict[str, Dict[str, float]]:
        """
        Initialize the transition probability matrix.

        Returns:
            Dictionary mapping from_regime -> to_regime -> probability
        """
        transition_matrix = {}
        for from_regime in MarketRegime:
            transition_matrix[from_regime.value] = {}
            for to_regime in MarketRegime:
                if from_regime != to_regime:
                    transition_matrix[from_regime.value][to_regime.value
                        ] = 1.0 / (len(MarketRegime) - 1)
        return transition_matrix

    async def predict_regime_transition(self, symbol: str, price_data: pd.
        DataFrame, current_regime: Optional[MarketRegime]=None, timeframe:
        str='1h', use_inter_market_correlations: bool=True,
        correlated_markets_data: Optional[Dict[str, pd.DataFrame]]=None
        ) ->Dict[str, Any]:
        """
        Predict potential regime transitions based on current market conditions.

        Args:
            symbol: The trading symbol
            price_data: Price DataFrame
            current_regime: Optional current market regime (if not provided, will be detected)
            timeframe: Timeframe of the price data
            use_inter_market_correlations: Whether to use inter-market correlations for prediction
            correlated_markets_data: Optional price data for correlated markets

        Returns:
            Dictionary with transition predictions
        """
        if price_data.empty:
            return {'error': 'Empty price data provided'}
        if current_regime is None:
            regime_result = self.regime_detector.detect_regime(price_data)
            current_regime = regime_result.get('regime', MarketRegime.UNKNOWN)
        early_warnings = self._calculate_early_warning_indicators(price_data,
            current_regime)
        if current_regime.value in self.transition_probabilities:
            transition_probs = self.transition_probabilities[current_regime
                .value]
        else:
            transition_probs = {r.value: (1.0 / (len(MarketRegime) - 1)) for
                r in MarketRegime if r != current_regime}
        inter_market_signals = {}
        if use_inter_market_correlations and self.correlation_service:
            if (symbol not in self.correlated_markets or not self.
                correlated_markets[symbol]):
                await self.update_correlated_markets(symbol)
            if correlated_markets_data and self.correlated_markets.get(symbol):
                inter_market_signals = await self._analyze_correlated_markets(
                    symbol, current_regime, correlated_markets_data)
                if inter_market_signals:
                    early_warnings.update(inter_market_signals)
        adjusted_probs = self._adjust_probabilities(transition_probs,
            early_warnings)
        next_regime = max(adjusted_probs.items(), key=lambda x: x[1])
        transition_probability = next_regime[1]
        probability_level = self._map_probability_to_level(
            transition_probability)
        prediction = {'symbol': symbol, 'timeframe': timeframe,
            'current_regime': current_regime.value,
            'most_likely_next_regime': next_regime[0],
            'transition_probability': transition_probability,
            'probability_level': probability_level.value,
            'transition_probabilities': adjusted_probs,
            'early_warning_indicators': early_warnings,
            'inter_market_signals': inter_market_signals,
            'used_inter_market_correlations': bool(inter_market_signals),
            'correlated_markets_count': len(self.correlated_markets.get(
            symbol, {})), 'timestamp': datetime.now().isoformat()}
        self._update_transition_history(symbol, current_regime.value,
            prediction)
        return prediction

    def _calculate_early_warning_indicators(self, price_data: pd.DataFrame,
        current_regime: MarketRegime) ->Dict[str, float]:
        """
        Calculate early warning indicators for regime transitions.

        Args:
            price_data: Price DataFrame
            current_regime: Current market regime

        Returns:
            Dictionary mapping indicator names to values (0.0-1.0)
        """
        indicators = {}
        if len(price_data) < self.lookback_periods:
            return indicators
        close_col = next((col for col in price_data.columns if col.lower() in
            ['close', 'price', 'adj close']), None)
        high_col = next((col for col in price_data.columns if col.lower() ==
            'high'), None)
        low_col = next((col for col in price_data.columns if col.lower() ==
            'low'), None)
        if not close_col:
            return indicators
        recent_data = price_data.iloc[-self.lookback_periods:]
        close_prices = recent_data[close_col]
        if high_col and low_col:
            high_prices = recent_data[high_col]
            low_prices = recent_data[low_col]
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - close_prices.shift(1))
            tr3 = abs(low_prices - close_prices.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            recent_atr = atr.iloc[-5:].mean()
            historical_atr = atr.iloc[-20:-5].mean()
            if historical_atr > 0:
                atr_change = recent_atr / historical_atr - 1
                volatility_change = min(1.0, max(0.0, (atr_change + 0.5) / 1.5)
                    )
                indicators['volatility_change'] = volatility_change
        if high_col and low_col:
            plus_dm = high_prices.diff()
            minus_dm = low_prices.diff().multiply(-1)
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            period = 14
            plus_di = 100 * plus_dm.rolling(window=period).mean(
                ) / true_range.rolling(window=period).mean()
            minus_di = 100 * minus_dm.rolling(window=period).mean(
                ) / true_range.rolling(window=period).mean()
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            recent_adx = adx.iloc[-5:].mean()
            historical_adx = adx.iloc[-20:-5].mean()
            if not np.isnan(recent_adx) and not np.isnan(historical_adx
                ) and historical_adx > 0:
                adx_change = recent_adx / historical_adx - 1
                trend_strength_change = min(1.0, max(0.0, (adx_change + 0.5
                    ) / 1.5))
                indicators['trend_strength_change'] = trend_strength_change
        ma_short = close_prices.rolling(window=10).mean()
        ma_medium = close_prices.rolling(window=20).mean()
        ma_long = close_prices.rolling(window=50).mean()
        ma_crossover = 0.0
        if not np.isnan(ma_short.iloc[-1]) and not np.isnan(ma_medium.iloc[-1]
            ):
            if ma_short.iloc[-2] < ma_medium.iloc[-2] and ma_short.iloc[-1
                ] >= ma_medium.iloc[-1] or ma_short.iloc[-2] > ma_medium.iloc[
                -2] and ma_short.iloc[-1] <= ma_medium.iloc[-1]:
                ma_crossover = 0.7
        if not np.isnan(ma_medium.iloc[-1]) and not np.isnan(ma_long.iloc[-1]):
            if ma_medium.iloc[-2] < ma_long.iloc[-2] and ma_medium.iloc[-1
                ] >= ma_long.iloc[-1] or ma_medium.iloc[-2] > ma_long.iloc[-2
                ] and ma_medium.iloc[-1] <= ma_long.iloc[-1]:
                ma_crossover = 0.9
        indicators['ma_crossover'] = ma_crossover
        volume_col = next((col for col in price_data.columns if col.lower() ==
            'volume'), None)
        if volume_col:
            volume = recent_data[volume_col]
            recent_volume = volume.iloc[-5:].mean()
            historical_volume = volume.iloc[-20:-5].mean()
            if historical_volume > 0:
                volume_change = recent_volume / historical_volume - 1
                volume_change_indicator = min(1.0, max(0.0, (volume_change +
                    0.5) / 1.5))
                indicators['volume_change'] = volume_change_indicator
        if current_regime == MarketRegime.TRENDING:
            delta = close_prices.diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
            price_higher_high = close_prices.iloc[-1] > close_prices.iloc[-
                10:-2].max()
            rsi_lower_high = rsi.iloc[-1] < rsi.iloc[-10:-2].max()
            if price_higher_high and rsi_lower_high:
                indicators['momentum_divergence'] = 0.8
            else:
                indicators['momentum_divergence'] = 0.2
        elif current_regime == MarketRegime.RANGING:
            recent_range = (high_prices.iloc[-5:].max() - low_prices.iloc[-
                5:].min()) / close_prices.iloc[-5:].mean()
            historical_range = (high_prices.iloc[-20:-5].max() - low_prices
                .iloc[-20:-5].min()) / close_prices.iloc[-20:-5].mean()
            if historical_range > 0:
                range_expansion = recent_range / historical_range - 1
                range_expansion_indicator = min(1.0, max(0.0, (
                    range_expansion + 0.5) / 1.5))
                indicators['range_expansion'] = range_expansion_indicator
        if indicators:
            indicators['overall_warning'] = sum(indicators.values()) / len(
                indicators)
        return indicators

    def _adjust_probabilities(self, base_probabilities: Dict[str, float],
        early_warnings: Dict[str, float]) ->Dict[str, float]:
        """
        Adjust transition probabilities based on early warning indicators.

        Args:
            base_probabilities: Base transition probabilities
            early_warnings: Early warning indicators

        Returns:
            Adjusted transition probabilities
        """
        adjusted_probs = base_probabilities.copy()
        overall_warning = early_warnings.get('overall_warning', 0.0)
        if overall_warning >= self.early_warning_threshold:
            if 'volatility_change' in early_warnings and early_warnings[
                'volatility_change'] > 0.7:
                if MarketRegime.VOLATILE.value in adjusted_probs:
                    adjusted_probs[MarketRegime.VOLATILE.value] *= 1.5
            if 'trend_strength_change' in early_warnings and early_warnings[
                'trend_strength_change'] > 0.7:
                if MarketRegime.TRENDING.value in adjusted_probs:
                    adjusted_probs[MarketRegime.TRENDING.value] *= 1.5
            if 'ma_crossover' in early_warnings and early_warnings[
                'ma_crossover'] > 0.7:
                if MarketRegime.TRENDING.value in adjusted_probs:
                    adjusted_probs[MarketRegime.TRENDING.value] *= 1.3
            if 'range_expansion' in early_warnings and early_warnings[
                'range_expansion'] > 0.7:
                if MarketRegime.BREAKOUT.value in adjusted_probs:
                    adjusted_probs[MarketRegime.BREAKOUT.value] *= 1.5
            if 'correlated_markets_dominant_regime' in early_warnings:
                dominant_regime = early_warnings.get(
                    'correlated_markets_dominant_regime_type')
                confidence = early_warnings.get(
                    'correlated_markets_dominant_regime')
                if dominant_regime and dominant_regime in adjusted_probs:
                    adjusted_probs[dominant_regime] *= 1.0 + confidence * 0.5
            if 'inter_market_transition_signal' in early_warnings:
                signal_strength = early_warnings[
                    'inter_market_transition_signal']
                for regime in adjusted_probs:
                    if regime != current_regime.value:
                        adjusted_probs[regime] *= 1.0 + signal_strength * 0.3
            if 'leading_markets_signal' in early_warnings:
                leading_signal = early_warnings['leading_markets_signal']
                for regime in adjusted_probs:
                    if regime != current_regime.value:
                        adjusted_probs[regime] *= 1.0 + leading_signal * 0.5
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            adjusted_probs = {k: (v / total_prob) for k, v in
                adjusted_probs.items()}
        return adjusted_probs

    def _map_probability_to_level(self, probability: float
        ) ->TransitionProbability:
        """
        Map a numerical probability to a probability level.

        Args:
            probability: Numerical probability (0.0-1.0)

        Returns:
            TransitionProbability level
        """
        if probability < 0.2:
            return TransitionProbability.VERY_LOW
        elif probability < 0.4:
            return TransitionProbability.LOW
        elif probability < 0.6:
            return TransitionProbability.MODERATE
        elif probability < 0.8:
            return TransitionProbability.HIGH
        else:
            return TransitionProbability.VERY_HIGH

    def _update_transition_history(self, symbol: str, current_regime: str,
        prediction: Dict[str, Any]) ->None:
        """
        Update the history of regime transitions.

        Args:
            symbol: Trading symbol
            current_regime: Current market regime
            prediction: Prediction result
        """
        if symbol not in self.transition_history:
            self.transition_history[symbol] = []
        self.transition_history[symbol].append({'timestamp': datetime.now(),
            'current_regime': current_regime, 'prediction': prediction})
        if len(self.transition_history[symbol]) > self.transition_history_size:
            self.transition_history[symbol] = self.transition_history[symbol][
                -self.transition_history_size:]

    @with_resilience('update_transition_probabilities')
    def update_transition_probabilities(self, from_regime: str, to_regime:
        str, occurred: bool) ->None:
        """
        Update transition probabilities based on whether a predicted transition occurred.

        Args:
            from_regime: Starting regime
            to_regime: Target regime
            occurred: Whether the transition occurred
        """
        if from_regime not in self.transition_probabilities:
            self.transition_probabilities[from_regime] = {r.value: (1.0 / (
                len(MarketRegime) - 1)) for r in MarketRegime if r.value !=
                from_regime}
        alpha = 0.1
        current_prob = self.transition_probabilities[from_regime].get(to_regime
            , 0.0)
        if occurred:
            new_prob = current_prob + alpha * (1.0 - current_prob)
        else:
            new_prob = current_prob * (1.0 - alpha)
        self.transition_probabilities[from_regime][to_regime] = new_prob
        total_prob = sum(self.transition_probabilities[from_regime].values())
        if total_prob > 0:
            self.transition_probabilities[from_regime] = {k: (v /
                total_prob) for k, v in self.transition_probabilities[
                from_regime].items()}

    @with_resilience('get_transition_history')
    def get_transition_history(self, symbol: str, lookback_hours: Optional[
        int]=None) ->List[Dict[str, Any]]:
        """
        Get the history of regime transitions for a symbol.

        Args:
            symbol: Trading symbol
            lookback_hours: Optional number of hours to look back

        Returns:
            List of transition history entries
        """
        if symbol not in self.transition_history:
            return []
        history = self.transition_history[symbol]
        if lookback_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            history = [entry for entry in history if entry['timestamp'] >=
                cutoff_time]
        return history

    @with_resilience('get_transition_matrix')
    def get_transition_matrix(self) ->Dict[str, Dict[str, float]]:
        """
        Get the current transition probability matrix.

        Returns:
            Dictionary mapping from_regime -> to_regime -> probability
        """
        return self.transition_probabilities

    @with_resilience('get_most_likely_transitions')
    def get_most_likely_transitions(self, min_probability: float=0.3) ->List[
        Dict[str, Any]]:
        """
        Get the most likely regime transitions.

        Args:
            min_probability: Minimum probability threshold

        Returns:
            List of likely transitions
        """
        likely_transitions = []
        for from_regime, transitions in self.transition_probabilities.items():
            for to_regime, probability in transitions.items():
                if probability >= min_probability:
                    likely_transitions.append({'from_regime': from_regime,
                        'to_regime': to_regime, 'probability': probability,
                        'probability_level': self._map_probability_to_level
                        (probability).value})
        return sorted(likely_transitions, key=lambda x: x['probability'],
            reverse=True)

    @async_with_exception_handling
    async def _analyze_correlated_markets(self, symbol: str, current_regime:
        MarketRegime, correlated_markets_data: Dict[str, pd.DataFrame]) ->Dict[
        str, float]:
        """
        Analyze correlated markets for early warning signals of regime transitions.

        Args:
            symbol: The primary trading symbol
            current_regime: Current market regime
            correlated_markets_data: Price data for correlated markets

        Returns:
            Dictionary with inter-market signals
        """
        if not self.correlated_markets.get(symbol):
            return {}
        signals = {}
        leading_markets_count = 0
        regime_transitions_detected = 0
        regime_counts = defaultdict(int)
        for correlated_symbol, correlation in self.correlated_markets[symbol
            ].items():
            if (correlated_symbol not in correlated_markets_data or
                correlated_markets_data[correlated_symbol].empty):
                continue
            try:
                corr_regime_result = self.regime_detector.detect_regime(
                    correlated_markets_data[correlated_symbol])
                corr_regime = corr_regime_result.get('regime', MarketRegime
                    .UNKNOWN)
                regime_counts[corr_regime.value] += 1
                if corr_regime != current_regime:
                    regime_transitions_detected += 1
                    if abs(correlation) > 0.8:
                        leading_markets_count += 1
            except Exception as e:
                self.logger.warning(
                    f'Error detecting regime for {correlated_symbol}: {e}')
        total_markets = sum(regime_counts.values())
        if total_markets > 0:
            regime_percentages = {regime: (count / total_markets) for 
                regime, count in regime_counts.items()}
            dominant_regime = max(regime_percentages.items(), key=lambda x:
                x[1])
            if dominant_regime[0] != current_regime.value and dominant_regime[1
                ] >= 0.5:
                signals['correlated_markets_dominant_regime'
                    ] = dominant_regime[1]
                signals['correlated_markets_dominant_regime_type'
                    ] = dominant_regime[0]
            if regime_transitions_detected > 0:
                transition_signal = regime_transitions_detected / total_markets
                signals['inter_market_transition_signal'] = transition_signal
            if leading_markets_count > 0:
                leading_signal = leading_markets_count / total_markets
                signals['leading_markets_signal'] = leading_signal
        return signals

    @with_resilience('update_correlated_markets')
    @async_with_exception_handling
    async def update_correlated_markets(self, symbol: str, min_correlation:
        float=0.7) ->Dict[str, float]:
        """
        Update the list of correlated markets for a symbol.

        Args:
            symbol: The trading symbol
            min_correlation: Minimum correlation threshold

        Returns:
            Dictionary mapping correlated symbols to correlation values
        """
        if not self.correlation_service:
            self.logger.warning(
                'No correlation service available for inter-market analysis')
            return {}
        try:
            correlated_markets = (await self.correlation_service.
                get_highest_correlations(symbol=symbol, min_threshold=
                min_correlation))
            self.correlated_markets[symbol] = {market['symbol']: market[
                'correlation'] for market in correlated_markets}
            self.logger.info(
                f'Updated {len(self.correlated_markets[symbol])} correlated markets for {symbol}'
                )
            return self.correlated_markets[symbol]
        except Exception as e:
            self.logger.error(f'Error updating correlated markets: {e}')
            return {}
