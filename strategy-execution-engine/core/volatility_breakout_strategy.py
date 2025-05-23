"""
Volatility Breakout Strategy

This module implements a trading strategy based on volatility breakouts, which identifies
breakout opportunities when price exceeds volatility-based thresholds after periods of consolidation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from core.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.analysis.advanced_ta.volatility import VolatilityAnalyzer
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame
from core.execution_profiler import profile_execution
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceDetector
from analysis_engine.analysis.sequence_pattern_recognizer import SequencePatternRecognizer
from analysis_engine.services.regime_transition_predictor import RegimeTransitionPredictor


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class VolatilityBreakoutStrategy(AdvancedTAStrategy):
    """
    A strategy that detects breakouts from periods of contracting volatility, using
    multiple volatility measures and confirmation from volume and momentum indicators.
    """

    def __init__(self, name: str, timeframes: List[str], primary_timeframe:
        str, symbols: List[str], risk_per_trade_pct: float=1.0,
        currency_strength_analyzer: Optional[CurrencyStrengthAnalyzer]=None,
        related_pairs_detector: Optional[RelatedPairsConfluenceDetector]=
        None, pattern_recognizer: Optional[SequencePatternRecognizer]=None,
        regime_transition_predictor: Optional[RegimeTransitionPredictor]=
        None, **kwargs):
    """
      init  .
    
    Args:
        name: Description of name
        timeframes: Description of timeframes
        primary_timeframe: Description of primary_timeframe
        symbols: Description of symbols
        risk_per_trade_pct: Description of risk_per_trade_pct
        currency_strength_analyzer: Description of currency_strength_analyzer
        related_pairs_detector: Description of related_pairs_detector
        pattern_recognizer: Description of pattern_recognizer
        regime_transition_predictor: Description of regime_transition_predictor
        kwargs: Description of kwargs
    
    """

        super().__init__(name=name, timeframes=timeframes,
            primary_timeframe=primary_timeframe, symbols=symbols,
            risk_per_trade_pct=risk_per_trade_pct, **kwargs)
        self.currency_strength_analyzer = (currency_strength_analyzer or
            CurrencyStrengthAnalyzer())
        self.related_pairs_detector = (related_pairs_detector or
            RelatedPairsConfluenceDetector())
        self.pattern_recognizer = (pattern_recognizer or
            SequencePatternRecognizer())
        self.regime_transition_predictor = (regime_transition_predictor or
            RegimeTransitionPredictor())
        self._init_strategy_config()
        self.logger.info(
            f"VolatilityBreakoutStrategy '{name}' initialized with enhanced features"
            )

    def _init_strategy_config(self) ->None:
    """
     init strategy config.
    
    """

        self.volatility_analyzer = VolatilityAnalyzer(methods=['bollinger',
            'atr', 'keltner', 'donchian', 'historical'], lookback_period=20)
        self.adaptive_params = {'volatility_threshold': 1.5,
            'breakout_strength': 2.0, 'consolidation_periods': 10,
            'volume_confirmation': True, 'trailing_sl_factor': 2.0,
            'min_consolidation_bars': 5, 'max_consolidation_bars': 30,
            'volatility_percentile': 20, 'use_multi_timeframe': True}
        self.regime_parameters = {MarketRegime.TRENDING.value: {
            'volatility_threshold': 1.2, 'breakout_strength': 1.8,
            'min_consolidation_bars': 4, 'volatility_percentile': 25},
            MarketRegime.RANGING.value: {'volatility_threshold': 1.7,
            'breakout_strength': 2.2, 'min_consolidation_bars': 7,
            'volatility_percentile': 15}, MarketRegime.VOLATILE.value: {
            'volatility_threshold': 1.0, 'breakout_strength': 2.5,
            'min_consolidation_bars': 3, 'volatility_percentile': 30},
            MarketRegime.BREAKOUT.value: {'volatility_threshold': 1.4,
            'breakout_strength': 1.5, 'min_consolidation_bars': 5,
            'volatility_percentile': 20}}
        self.config.update({'preferred_direction': 'both', 'min_confidence':
            0.6, 'enable_batch_processing': True, 'max_lookback_periods': 
            100, 'use_currency_strength': True,
            'use_related_pairs_confluence': True, 'use_sequence_patterns': 
            True, 'use_regime_transition_prediction': True,
            'min_related_pairs_confluence': 0.6, 'min_pattern_confidence': 
            0.7, 'regime_transition_threshold': 0.7})
        self.historical_volatility = {}

    def _adapt_parameters_to_regime(self, regime: MarketRegime) ->None:
        """
        Adjust adaptive parameters based on market regime for volatility breakout strategy.

        Args:
            regime: Current market regime
        """
        params = self.regime_parameters.get(regime.value)
        if not params:
            return
        adaptive_updates = {k: v for k, v in params.items() if k in self.
            adaptive_params}
        for key, value in adaptive_updates.items():
            self.adaptive_params[key] = value
        self.logger.info(
            f'Adapted volatility breakout parameters to regime {regime}: {adaptive_updates}'
            )

    def _is_in_consolidation(self, df: pd.DataFrame) ->bool:
        """
        Check if the market is in a consolidation phase based on volatility metrics.

        Args:
            df: Price data with volatility indicators

        Returns:
            True if market is in consolidation, False otherwise
        """
        if df.empty or len(df) < self.adaptive_params['min_consolidation_bars'
            ]:
            return False
        min_periods = self.adaptive_params['min_consolidation_bars']
        max_periods = min(self.adaptive_params['max_consolidation_bars'], 
            len(df) - 1)
        if 'volatility_percentile' in df.columns:
            current_percentile = df['volatility_percentile'].iloc[-1]
            if current_percentile > self.adaptive_params[
                'volatility_percentile']:
                return False
        if 'bollinger_bandwidth' in df.columns:
            bandwidths = df['bollinger_bandwidth'].iloc[-max_periods:-1].values
            if len(bandwidths) < min_periods:
                return False
            bandwidth_diff = np.diff(bandwidths)
            negative_diffs = bandwidth_diff < 0
            contraction_ratio = np.sum(negative_diffs) / len(bandwidth_diff)
            if contraction_ratio < 0.6:
                return False
        elif 'ATR' in df.columns:
            recent_atrs = df['ATR'].iloc[-max_periods:].values
            if len(recent_atrs) < min_periods:
                return False
            atr_diff = np.diff(recent_atrs)
            negative_diffs = atr_diff < 0
            contraction_ratio = np.sum(negative_diffs) / len(atr_diff)
            if contraction_ratio < 0.6:
                return False
        else:
            return False
        return True

    def _detect_breakout(self, df: pd.DataFrame) ->Tuple[bool, str, float]:
        """
        Detect if a breakout is occurring and determine its direction.

        Args:
            df: Price data with volatility indicators

        Returns:
            Tuple of (is_breakout, direction, strength)
        """
        if df.empty or len(df) < 3:
            return False, 'neutral', 0.0
        latest_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        breakout_direction = 'neutral'
        breakout_strength = 0.0
        is_breakout = False
        if all(col in df.columns for col in ['bollinger_upper',
            'bollinger_lower', 'bollinger_middle']):
            upper_band = df['bollinger_upper'].iloc[-1]
            lower_band = df['bollinger_lower'].iloc[-1]
            middle_band = df['bollinger_middle'].iloc[-1]
            upper_distance = (latest_close - upper_band) / (upper_band -
                middle_band) if upper_band > middle_band else 0
            lower_distance = (lower_band - latest_close) / (middle_band -
                lower_band) if middle_band > lower_band else 0
            if latest_close > upper_band and prev_close <= upper_band:
                breakout_direction = 'bullish'
                breakout_strength = max(breakout_strength, upper_distance + 1.0
                    )
                is_breakout = True
            elif latest_close < lower_band and prev_close >= lower_band:
                breakout_direction = 'bearish'
                breakout_strength = max(breakout_strength, lower_distance + 1.0
                    )
                is_breakout = True
        if all(col in df.columns for col in ['keltner_upper', 'keltner_lower']
            ):
            keltner_upper = df['keltner_upper'].iloc[-1]
            keltner_lower = df['keltner_lower'].iloc[-1]
            if latest_close > keltner_upper and prev_close <= keltner_upper:
                if (breakout_direction == 'neutral' or breakout_direction ==
                    'bullish'):
                    breakout_direction = 'bullish'
                    breakout_strength += 1.0
                    is_breakout = True
            elif latest_close < keltner_lower and prev_close >= keltner_lower:
                if (breakout_direction == 'neutral' or breakout_direction ==
                    'bearish'):
                    breakout_direction = 'bearish'
                    breakout_strength += 1.0
                    is_breakout = True
        if all(col in df.columns for col in ['donchian_upper',
            'donchian_lower']):
            donchian_upper = df['donchian_upper'].iloc[-1]
            donchian_lower = df['donchian_lower'].iloc[-1]
            periods = self.adaptive_params['consolidation_periods']
            if latest_close > donchian_upper and all(df['close'].iloc[-
                periods - 1:-1] <= donchian_upper):
                breakout_direction = 'bullish'
                breakout_strength += 2.0
                is_breakout = True
            elif latest_close < donchian_lower and all(df['close'].iloc[-
                periods - 1:-1] >= donchian_lower):
                breakout_direction = 'bearish'
                breakout_strength += 2.0
                is_breakout = True
        if is_breakout and self.adaptive_params['volume_confirmation'
            ] and 'volume' in df.columns:
            avg_volume = df['volume'].iloc[-6:-1].mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = (current_volume / avg_volume if avg_volume > 0 else
                1.0)
            if volume_ratio >= 1.5:
                breakout_strength *= 1.2
            elif volume_ratio <= 0.8:
                breakout_strength *= 0.8
        return is_breakout, breakout_direction, breakout_strength

    @profile_execution('VolatilityBreakoutStrategy', 'analyze_timeframe')
    @with_exception_handling
    def _analyze_timeframe(self, symbol: str, tf: str, df: pd.DataFrame
        ) ->Dict[str, Any]:
        """
        Analyze a single timeframe for volatility breakout signals.

        Args:
            symbol: Trading symbol
            tf: Timeframe
            df: Price data

        Returns:
            Dictionary with timeframe analysis results
        """
        results = {'timeframe': tf, 'is_consolidation': False,
            'is_breakout': False, 'direction': 'neutral', 'strength': 0.0}
        if df.empty or len(df) < self.adaptive_params['min_consolidation_bars'
            ] + 2:
            return results
        try:
            df = self.volatility_analyzer.calculate(df)
            results['is_consolidation'] = self._is_in_consolidation(df)
            current_regime = self.get_current_regime(symbol, tf)
            if results['is_consolidation'
                ] or current_regime == MarketRegime.BREAKOUT:
                is_breakout, direction, strength = self._detect_breakout(df)
                results['is_breakout'] = is_breakout
                results['direction'] = direction
                results['strength'] = strength
                if 'ATR' in df.columns:
                    results['atr'] = df['ATR'].iloc[-1]
                for key in ['bollinger_bandwidth', 'atr_percent',
                    'volatility_percentile']:
                    if key in df.columns:
                        results[key] = df[key].iloc[-1]
        except Exception as e:
            self.logger.error(f'Error analyzing {symbol} on {tf}: {e}')
        return results

    @profile_execution('VolatilityBreakoutStrategy',
        'perform_strategy_analysis')
    def _perform_strategy_analysis(self, symbol: str, price_data: Dict[str,
        pd.DataFrame], confluence_results: Dict[str, Any], additional_data:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Perform volatility breakout analysis across timeframes.

        Args:
            symbol: Trading symbol
            price_data: Dictionary of price data by timeframe
            confluence_results: Results from other indicators
            additional_data: Any additional data for analysis

        Returns:
            Analysis results
        """
        results = {'timeframes': {}, 'consolidations': 0, 'breakouts': 0,
            'direction_votes': {'bullish': 0, 'bearish': 0, 'neutral': 0},
            'overall_strength': 0.0, 'signal_strength': 0}
        current_price = None
        for tf in sorted(price_data.keys()):
            df = price_data[tf]
            if not df.empty:
                current_price = df['close'].iloc[-1]
                break
        if current_price is None:
            self.logger.warning(f'No price data available for {symbol}')
            return results
        use_batch = self.config_manager.get('enable_batch_processing', False)
        if use_batch and len(price_data) >= 3:
            pass
        for tf, df in price_data.items():
            if df.empty:
                continue
            tf_results = self._analyze_timeframe(symbol, tf, df)
            results['timeframes'][tf] = tf_results
            if tf_results['is_consolidation']:
                results['consolidations'] += 1
            if tf_results['is_breakout']:
                results['breakouts'] += 1
                results['direction_votes'][tf_results['direction']] += 1
                results['overall_strength'] += tf_results['strength']
        direction_votes = results['direction_votes']
        if direction_votes['bullish'] > direction_votes['bearish']:
            results['direction'] = 'bullish'
        elif direction_votes['bearish'] > direction_votes['bullish']:
            results['direction'] = 'bearish'
        else:
            results['direction'] = 'neutral'
        threshold = self.adaptive_params['breakout_strength']
        if results['breakouts'] > 0:
            signal_confidence = min(results['overall_strength'] / threshold,
                3.0) / 3.0
            results['signal_strength'] = int(signal_confidence * 10)
        else:
            results['signal_strength'] = 0
        results['price'] = current_price
        if self.config_manager.get('use_currency_strength', False):
            currency_strength = self._analyze_currency_strength(symbol,
                price_data)
            if currency_strength:
                results['currency_strength'] = currency_strength
        if self.config_manager.get('use_related_pairs_confluence', False):
            related_pairs_confluence = self._analyze_related_pairs_confluence(
                symbol, price_data, results['direction'])
            if related_pairs_confluence:
                results['related_pairs_confluence'] = related_pairs_confluence
        if self.config_manager.get('use_sequence_patterns', False):
            sequence_patterns = self._detect_sequence_patterns(price_data)
            if sequence_patterns:
                results['sequence_patterns'] = sequence_patterns
        if self.config.get('use_regime_transition_prediction', False
            ) and self.market_regime:
            regime_transition = self._predict_regime_transition(symbol,
                price_data.get(self.primary_timeframe))
            if regime_transition:
                results['regime_transition'] = regime_transition
        return results

    @profile_execution('VolatilityBreakoutStrategy', 'generate_signals')
    def _generate_signals(self, symbol: str, strategy_analysis: Dict[str,
        Any], confluence_results: Dict[str, Any]) ->List[Dict[str, Any]]:
        """
        Generate trading signals based on volatility breakout analysis.

        Args:
            symbol: Trading symbol
            strategy_analysis: Results from volatility breakout analysis
            confluence_results: Results from other indicators

        Returns:
            List of trading signals
        """
        signals = []
        strength = strategy_analysis.get('signal_strength', 0)
        if strength < 3:
            return signals
        direction = strategy_analysis.get('direction', 'neutral')
        if direction == 'neutral':
            self.logger.info(
                f'No clear breakout direction for {symbol}, skipping signal')
            return signals
        preferred_direction = self.config['preferred_direction']
        if preferred_direction != 'both' and preferred_direction != direction:
            self.logger.info(
                f"Breakout direction {direction} doesn't match preferred {preferred_direction}"
                )
            return signals
        current_price = strategy_analysis.get('price')
        if not current_price:
            self.logger.warning(f'Missing current price for {symbol}')
            return signals
        stop_loss = None
        take_profit = None
        atr = None
        for tf_data in strategy_analysis.get('timeframes', {}).values():
            if 'atr' in tf_data:
                atr = tf_data['atr']
                break
        if atr:
            if direction == 'bullish':
                stop_loss = current_price - atr * self.adaptive_params[
                    'trailing_sl_factor']
                take_profit = current_price + atr * self.adaptive_params[
                    'trailing_sl_factor'] * 2
            else:
                stop_loss = current_price + atr * self.adaptive_params[
                    'trailing_sl_factor']
                take_profit = current_price - atr * self.adaptive_params[
                    'trailing_sl_factor'] * 2
        timeframe_info = []
        for tf, tf_data in strategy_analysis.get('timeframes', {}).items():
            if tf_data.get('is_breakout'):
                strength_desc = f"{tf_data['strength']:.1f}"
                timeframe_info.append(f'{tf}: {strength_desc}')
        timeframes_str = ', '.join(timeframe_info
            ) if timeframe_info else 'multiple timeframes'
        volatility_info = ''
        for tf_data in strategy_analysis.get('timeframes', {}).values():
            if 'volatility_percentile' in tf_data:
                volatility_info = (
                    f", Volatility: {int(tf_data['volatility_percentile'])}th percentile"
                    )
                break
        confidence = min(0.5 + strength * 0.05, 0.95)
        signal = {'symbol': symbol, 'timestamp': datetime.now().isoformat(),
            'signal_type': 'volatility_breakout', 'direction': direction,
            'strength': strength, 'confidence': confidence, 'entry_price':
            current_price, 'stop_loss': stop_loss, 'take_profit':
            take_profit, 'reason':
            f"Volatility breakout ({direction}) confirmed on {timeframes_str}{volatility_info} with strength {strategy_analysis['overall_strength']:.1f}"
            }
        self._enhance_signal_with_additional_analysis(signal,
            strategy_analysis, symbol)
        signals.append(signal)
        return signals

    @with_exception_handling
    def _analyze_currency_strength(self, symbol: str, price_data: Dict[str,
        pd.DataFrame]) ->Dict[str, Any]:
        """
        Analyze currency strength for the symbol's currencies.

        Args:
            symbol: Trading symbol
            price_data: Price data dictionary by timeframe

        Returns:
            Dictionary with currency strength analysis results
        """
        try:
            if '/' in symbol:
                base_currency, quote_currency = symbol.split('/')
            elif len(symbol) == 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
            else:
                self.logger.warning(
                    f'Unable to parse currencies from symbol: {symbol}')
                return {}
            strength_values = (self.currency_strength_analyzer.
                calculate_currency_strength(price_data))
            if not strength_values:
                return {}
            base_strength = strength_values.get(base_currency, 0.0)
            quote_strength = strength_values.get(quote_currency, 0.0)
            strength_diff = base_strength - quote_strength
            strongest_currencies = (self.currency_strength_analyzer.
                get_strongest_currencies(count=3))
            weakest_currencies = (self.currency_strength_analyzer.
                get_weakest_currencies(count=3))
            opportunities = (self.currency_strength_analyzer.
                find_pair_opportunities(price_data, min_strength_difference
                =0.3))
            related_opportunities = [opp for opp in opportunities if opp.
                get('base_currency') == base_currency or opp.get(
                'quote_currency') == quote_currency]
            return {'base_currency': base_currency, 'quote_currency':
                quote_currency, 'base_strength': base_strength,
                'quote_strength': quote_strength, 'strength_difference':
                strength_diff, 'strongest_currencies': strongest_currencies,
                'weakest_currencies': weakest_currencies,
                'related_opportunities': related_opportunities[:3]}
        except Exception as e:
            self.logger.error(f'Error in currency strength analysis: {e}')
            return {}

    @with_exception_handling
    def _analyze_related_pairs_confluence(self, symbol: str, price_data:
        Dict[str, pd.DataFrame], direction: str) ->Dict[str, Any]:
        """
        Analyze confluence signals across related currency pairs.

        Args:
            symbol: Trading symbol
            price_data: Price data dictionary by timeframe
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            Dictionary with related pairs confluence analysis results
        """
        try:
            related_pairs = {}
            if '/' in symbol:
                base_currency, quote_currency = symbol.split('/')
            elif len(symbol) == 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
            else:
                self.logger.warning(
                    f'Unable to parse currencies from symbol: {symbol}')
                return {}
            for pair in price_data.keys():
                if pair == symbol:
                    continue
                if '/' in pair:
                    pair_base, pair_quote = pair.split('/')
                elif len(pair) == 6:
                    pair_base = pair[:3]
                    pair_quote = pair[3:]
                else:
                    continue
                if (pair_base == base_currency or pair_base ==
                    quote_currency or pair_quote == base_currency or 
                    pair_quote == quote_currency):
                    related_pairs[pair] = (0.7 if pair_base ==
                        base_currency or pair_quote == quote_currency else -0.7
                        )
            confluence_result = self.related_pairs_detector.detect_confluence(
                symbol=symbol, price_data=price_data, signal_type=
                'breakout', signal_direction=direction, related_pairs=
                related_pairs)
            if confluence_result.get('confluence_score', 0) < self.config.get(
                'min_related_pairs_confluence', 0.6):
                return {}
            return confluence_result
        except Exception as e:
            self.logger.error(
                f'Error in related pairs confluence analysis: {e}')
            return {}

    @with_exception_handling
    def _detect_sequence_patterns(self, price_data: Dict[str, pd.DataFrame]
        ) ->Dict[str, Any]:
        """
        Detect sequence patterns across multiple timeframes.

        Args:
            price_data: Price data dictionary by timeframe

        Returns:
            Dictionary with sequence pattern detection results
        """
        try:
            pattern_results = self.pattern_recognizer.detect_patterns(
                price_data)
            min_confidence = self.config_manager.get('min_pattern_confidence', 0.7)
            if 'sequence_patterns' in pattern_results:
                high_confidence_patterns = [p for p in pattern_results[
                    'sequence_patterns'] if p.get('confidence', 0) >=
                    min_confidence]
                if high_confidence_patterns:
                    pattern_results['sequence_patterns'
                        ] = high_confidence_patterns
                    pattern_results['high_confidence_count'] = len(
                        high_confidence_patterns)
                    return pattern_results
            return {}
        except Exception as e:
            self.logger.error(f'Error in sequence pattern detection: {e}')
            return {}

    @with_exception_handling
    def _predict_regime_transition(self, symbol: str, price_data: pd.DataFrame
        ) ->Dict[str, Any]:
        """
        Predict potential market regime transitions.

        Args:
            symbol: Trading symbol
            price_data: Price DataFrame for primary timeframe

        Returns:
            Dictionary with regime transition prediction results
        """
        try:
            if (price_data is None or price_data.empty or self.
                market_regime is None):
                return {}
            prediction = (self.regime_transition_predictor.
                predict_regime_transition(symbol=symbol, price_data=
                price_data, current_regime=self.market_regime, timeframe=
                self.primary_timeframe))
            if prediction.get('transition_probability', 0) < self.config.get(
                'regime_transition_threshold', 0.7):
                return {}
            return prediction
        except Exception as e:
            self.logger.error(f'Error in regime transition prediction: {e}')
            return {}

    def _enhance_signal_with_additional_analysis(self, signal: Dict[str,
        Any], strategy_analysis: Dict[str, Any], symbol: str) ->None:
        """
        Enhance the trading signal with additional analysis results.

        Args:
            signal: Trading signal to enhance
            strategy_analysis: Strategy analysis results
            symbol: Trading symbol
        """
        currency_strength = strategy_analysis.get('currency_strength', {})
        if currency_strength:
            strength_diff = currency_strength.get('strength_difference', 0)
            signal['currency_strength_diff'] = strength_diff
            direction = signal.get('direction')
            strength_confirms = (direction == 'bullish' and strength_diff >
                0 or direction == 'bearish' and strength_diff < 0)
            signal['currency_strength_confirms'] = strength_confirms
            if strength_confirms:
                signal['confidence'] = min(signal.get('confidence', 0.5) + 
                    0.1, 0.95)
            else:
                signal['confidence'] = max(signal.get('confidence', 0.5) - 
                    0.1, 0.1)
        related_pairs = strategy_analysis.get('related_pairs_confluence', {})
        if related_pairs:
            signal['related_pairs_confluence'] = related_pairs.get(
                'confluence_score', 0)
            signal['related_pairs_confirmations'] = related_pairs.get(
                'confirmation_count', 0)
            confidence_boost = min(related_pairs.get('confluence_score', 0) *
                0.2, 0.15)
            signal['confidence'] = min(signal.get('confidence', 0.5) +
                confidence_boost, 0.95)
        patterns = strategy_analysis.get('sequence_patterns', {})
        if patterns and 'sequence_patterns' in patterns:
            top_patterns = patterns['sequence_patterns'][:1] if patterns[
                'sequence_patterns'] else []
            if top_patterns:
                top_pattern = top_patterns[0]
                signal['pattern_type'] = top_pattern.get('type')
                signal['pattern_confidence'] = top_pattern.get('confidence', 0)
                signal['pattern_timeframes'] = top_pattern.get('timeframes', []
                    )
                confidence_boost = min(top_pattern.get('confidence', 0) * 
                    0.15, 0.1)
                signal['confidence'] = min(signal.get('confidence', 0.5) +
                    confidence_boost, 0.95)
        regime_transition = strategy_analysis.get('regime_transition', {})
        if regime_transition:
            signal['predicted_regime_transition'] = {'current_regime':
                regime_transition.get('current_regime'), 'next_regime':
                regime_transition.get('most_likely_next_regime'),
                'probability': regime_transition.get(
                'transition_probability', 0)}
            if regime_transition.get('transition_probability', 0) > 0.8:
                signal['high_regime_transition_warning'] = True
                current_regime = regime_transition.get('current_regime')
                next_regime = regime_transition.get('most_likely_next_regime')
                direction = signal.get('direction')
                compatible = False
                if direction == 'bullish':
                    compatible = next_regime in ['TRENDING_UP', 'BREAKOUT']
                elif direction == 'bearish':
                    compatible = next_regime in ['TRENDING_DOWN', 'BREAKOUT']
                if not compatible:
                    signal['confidence'] = max(signal.get('confidence', 0.5
                        ) - 0.2, 0.1)
                    signal['regime_transition_warning'] = (
                        f'Signal may not perform well in predicted {next_regime} regime'
                        )
