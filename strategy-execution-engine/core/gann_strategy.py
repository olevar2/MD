"""GannTradingStrategy

This module implements a trading strategy based on Gann tools (angles, fans, grids, Square of 9)
and identifies confluence levels to generate trade signals.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import functools
from core.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.analysis.advanced_ta.gann_tools import GannToolsAnalyzer
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame
from optimization.caching import calculation_cache
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

class GannTradingStrategy(AdvancedTAStrategy):
    """
    Trading strategy based on Gann analysis with adaptive parameters and multi-timeframe confluence.
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
        self._init_caching()
        self.logger.info(
            f"GannTradingStrategy '{name}' initialized with enhanced features")

    def _init_caching(self) ->None:
        """Initialize caching for expensive Gann calculations"""
        self.calculation_cache = {}
        self.cache_ttl = 3600
        self.cache_hits = 0
        self.cache_misses = 0

    def _init_strategy_config(self) ->None:
    """
     init strategy config.
    
    """

        self.gann_analyzer = GannToolsAnalyzer(components=['angles', 'fan',
            'grid', 'square_of_9', 'seasonal_cycles'],
            pivot_detection_method='swing', lookback_period=100,
            price_scale='arithmetic', grid_divisions=8)
        self.adaptive_params = {'min_confluence_count': 2,
            'atr_multiple_sl': 1.5, 'use_multi_tf': True,
            'direction_filter': True, 'use_seasonal_cycles': True,
            'time_projection_weight': 0.3, 'use_square_of_9_timing': True}
        self.regime_parameters = {MarketRegime.TRENDING.value: {
            'min_confluence_count': 3, 'direction_filter': True,
            'time_projection_weight': 0.4}, MarketRegime.RANGING.value: {
            'min_confluence_count': 1, 'direction_filter': False,
            'time_projection_weight': 0.2}, MarketRegime.VOLATILE.value: {
            'min_confluence_count': 4, 'direction_filter': True,
            'time_projection_weight': 0.5}, MarketRegime.BREAKOUT.value: {
            'min_confluence_count': 2, 'direction_filter': True,
            'time_projection_weight': 0.3}}
        self.config.update({'preferred_direction': 'both', 'min_confidence':
            0.5, 'use_batch_processing': True,
            'enable_square_of_9_time_projections': True,
            'use_currency_strength': True, 'use_related_pairs_confluence': 
            True, 'use_sequence_patterns': True,
            'use_regime_transition_prediction': True,
            'min_related_pairs_confluence': 0.6, 'min_pattern_confidence': 
            0.7, 'regime_transition_threshold': 0.7})

    def _adapt_parameters_to_regime(self, regime: MarketRegime) ->None:
        """
        Adjust adaptive parameters based on market regime for Gann strategy.
        """
        params = self.regime_parameters.get(regime.value)
        if not params:
            return
        adaptive_updates = {k: v for k, v in params.items() if k in self.
            adaptive_params}
        self.apply_adaptive_parameters(adaptive_updates)
        self.logger.info(
            f'Adapted Gann strategy parameters to regime {regime}: {adaptive_updates}'
            )

    def apply_adaptive_parameters(self, parameters: Dict[str, Any]) ->None:
    """
    Apply adaptive parameters.
    
    Args:
        parameters: Description of parameters
        Any]: Description of Any]
    
    """

        if not parameters:
            return
        for key, value in parameters.items():
            if key in self.adaptive_params:
                self.adaptive_params[key] = value
                self.logger.debug(
                    f'Applied adaptive parameter: {key} = {value}')

    @functools.lru_cache(maxsize=128)
    def _cached_gann_calculation(self, symbol: str, timeframe: str,
        data_hash: str) ->pd.DataFrame:
        """
        Cache wrapper for Gann calculations to avoid recomputing expensive operations.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe being analyzed
            data_hash: A hash of the DataFrame to ensure cache validity

        Returns:
            DataFrame with Gann calculations applied
        """
        cache_key = f'{symbol}_{timeframe}_{data_hash}'
        if cache_key in self.calculation_cache:
            cached_result, timestamp = self.calculation_cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                self.cache_hits += 1
                self.logger.debug(
                    f'Cache hit for {cache_key}, total hits: {self.cache_hits}'
                    )
                return cached_result
        self.cache_misses += 1
        self.logger.debug(
            f'Cache miss for {cache_key}, total misses: {self.cache_misses}')
        return None

    def _detect_direction(self, df: pd.DataFrame) ->str:
        """
        Detect trade direction (bullish/bearish) based on Gann analysis.

        Args:
            df: DataFrame with Gann calculations applied

        Returns:
            Direction as "bullish", "bearish", or "neutral"
        """
        if 'gann_angle' not in df.columns:
            return 'neutral'
        latest_angle = df['gann_angle'].iloc[-1]
        square9_bullish = df.get('square9_support_test', 0).iloc[-1] > 0
        square9_bearish = df.get('square9_resistance_test', 0).iloc[-1] > 0
        seasonal_bullish = df.get('seasonal_cycle_bullish', 0).iloc[-1] > 0.5
        seasonal_bearish = df.get('seasonal_cycle_bearish', 0).iloc[-1] > 0.5
        bullish_signals = sum([latest_angle > 45, square9_bullish,
            seasonal_bullish])
        bearish_signals = sum([latest_angle < -45, square9_bearish,
            seasonal_bearish])
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'

    @with_exception_handling
    def _perform_strategy_analysis(self, symbol: str, price_data: Dict[str,
        pd.DataFrame], confluence_results: Dict[str, Any], additional_data:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
    """
     perform strategy analysis.
    
    Args:
        symbol: Description of symbol
        price_data: Description of price_data
        pd.DataFrame]: Description of pd.DataFrame]
        confluence_results: Description of confluence_results
        Any]: Description of Any]
        additional_data: Description of additional_data
        Any]]: Description of Any]]
    
    Returns:
        Dict[str, Any]: Description of return value
    
    """

        results = {}
        timeframe_scores = {}
        direction_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        use_batch = self.config_manager.get('use_batch_processing', False)
        if use_batch:
            batch_data = []
            timeframes = []
            for tf, df in price_data.items():
                if not df.empty:
                    batch_data.append(df)
                    timeframes.append(tf)
            if batch_data:
                try:
                    batch_results = self._process_batch_data(symbol,
                        batch_data, timeframes)
                    timeframe_scores = batch_results.get('scores', {})
                    direction_votes = batch_results.get('directions',
                        direction_votes)
                except Exception as e:
                    self.logger.error(
                        f'Batch processing error for {symbol}: {e}')
                    use_batch = False
        if not use_batch:
            for tf, df in price_data.items():
                if df.empty:
                    continue
                data_hash = str(hash(tuple(df.iloc[-50:].values.tobytes())))
                try:
                    cached_result = self._cached_gann_calculation(symbol,
                        tf, data_hash)
                    if cached_result is not None:
                        df = cached_result
                    else:
                        df = self.gann_analyzer.calculate(df)
                        self.calculation_cache[f'{symbol}_{tf}_{data_hash}'
                            ] = df, datetime.now().timestamp()
                    score = df['gann_confluence_score'].iloc[-1]
                    direction = self._detect_direction(df)
                    timeframe_scores[tf] = score
                    direction_votes[direction] += 1
                    if self.config.get('enable_square_of_9_time_projections',
                        False):
                        self._apply_time_projections(df, results)
                except Exception as e:
                    self.logger.error(f'Gann analysis error for {tf}: {e}')
                    timeframe_scores[tf] = 0
        if direction_votes['bullish'] > direction_votes['bearish']:
            results['direction'] = 'bullish'
        elif direction_votes['bearish'] > direction_votes['bullish']:
            results['direction'] = 'bearish'
        else:
            results['direction'] = 'neutral'
        if self.config_manager.get('use_timeframe_optimization', True):
            results['original_timeframe_scores'] = timeframe_scores.copy()
            weighted_avg, weighted_scores = (self.timeframe_optimizer.
                apply_weighted_score(timeframe_scores))
            timeframe_scores = weighted_scores
            results['weighted_average_score'] = weighted_avg
            self.logger.debug(
                f"Applied timeframe weights for {symbol}: original={results['original_timeframe_scores']}, weighted={timeframe_scores}"
                )
        if self.adaptive_params['use_multi_tf']:
            min_confluence = self.adaptive_params['min_confluence_count']
            valid = [s for s in timeframe_scores.values() if s >=
                min_confluence]
            results['signal_strength'] = len(valid)
            if self.adaptive_params.get('direction_filter', False) and results[
                'direction'] == 'neutral':
                results['signal_strength'] = 0
        else:
            results['signal_strength'] = timeframe_scores.get(self.
                primary_timeframe, 0)
        results['timeframe_scores'] = timeframe_scores
        if self.adaptive_params.get('use_seasonal_cycles', False):
            results['seasonal_analysis'] = self._analyze_seasonal_cycles(
                price_data, symbol)
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

    def _process_batch_data(self, symbol: str, batch_data: List[pd.
        DataFrame], timeframes: List[str]) ->Dict[str, Any]:
        """
        Process multiple timeframes in batch for performance optimization.

        Args:
            symbol: Trading symbol
            batch_data: List of DataFrames for each timeframe
            timeframes: List of timeframe strings

        Returns:
            Dictionary with processed results
        """
        batch_results = {'scores': {}, 'directions': {'bullish': 0,
            'bearish': 0, 'neutral': 0}}
        for i, df in enumerate(batch_data):
            tf = timeframes[i]
            processed_df = self.gann_analyzer.calculate(df)
            score = processed_df['gann_confluence_score'].iloc[-1]
            direction = self._detect_direction(processed_df)
            batch_results['scores'][tf] = score
            batch_results['directions'][direction] += 1
        return batch_results

    def _apply_time_projections(self, df: pd.DataFrame, results: Dict[str, Any]
        ) ->None:
        """
        Apply Square of 9 time projections to the analysis results.

        Args:
            df: DataFrame with Gann calculations
            results: Results dictionary to update
        """
        if 'square9_time_projections' in df.columns:
            projections = df['square9_time_projections'].iloc[-1]
            if isinstance(projections, list) and projections:
                weight = self.adaptive_params.get('time_projection_weight', 0.3
                    )
                results['time_projections'] = {'dates': projections,
                    'weight': weight}

    def _analyze_seasonal_cycles(self, price_data: Dict[str, pd.DataFrame],
        symbol: str) ->Dict[str, Any]:
        """
        Analyze seasonal cycles using Gann's methods.'

        Args:
            price_data: Price data dictionary by timeframe
            symbol: Trading symbol

        Returns:
            Dictionary with seasonal analysis results
        """
        seasonal_results = {}
        df = price_data.get(self.primary_timeframe)
        if df is None or df.empty:
            return seasonal_results
        if 'seasonal_cycle_phase' in df.columns:
            seasonal_results['current_phase'] = df['seasonal_cycle_phase'
                ].iloc[-1]
            seasonal_results['cycle_position'] = df['seasonal_cycle_position'
                ].iloc[-1]
            seasonal_results['historical_bias'] = df['seasonal_historical_bias'
                ].iloc[-1]
        return seasonal_results

    def _generate_signals(self, symbol: str, strategy_analysis: Dict[str,
        Any], confluence_results: Dict[str, Any]) ->List[Dict[str, Any]]:
    """
     generate signals.
    
    Args:
        symbol: Description of symbol
        strategy_analysis: Description of strategy_analysis
        Any]: Description of Any]
        confluence_results: Description of confluence_results
        Any]: Description of Any]
    
    Returns:
        List[Dict[str, Any]]: Description of return value
    
    """

        signals = []
        strength = strategy_analysis.get('signal_strength', 0)
        if strength < self.adaptive_params['min_confluence_count']:
            return signals
        direction = strategy_analysis.get('direction', 'neutral')
        if direction == 'neutral' and self.adaptive_params.get(
            'direction_filter', True):
            self.logger.info(
                f'No clear direction detected for {symbol}, skipping signal generation'
                )
            return signals
        preferred_direction = self.config['preferred_direction']
        if preferred_direction != 'both' and preferred_direction != direction:
            self.logger.info(
                f"Signal direction {direction} doesn't match preferred direction {preferred_direction}"
                )
            return signals
        current_price = confluence_results.get('price', None)
        if not current_price:
            self.logger.warning(
                f'Missing current price for {symbol}, cannot generate signal')
            return signals
        stop_loss = None
        take_profit = None
        df = confluence_results.get('df') if confluence_results else None
        if df is not None:
            if 'ATR' in df.columns:
                atr = df['ATR'].iloc[-1]
                if direction == 'bullish':
                    stop_loss = current_price - atr * self.adaptive_params[
                        'atr_multiple_sl']
                else:
                    stop_loss = current_price + atr * self.adaptive_params[
                        'atr_multiple_sl']
            if 'gann_take_profit_level' in df.columns:
                take_profit = df['gann_take_profit_level'].iloc[-1]
        seasonal_info = ''
        seasonal_analysis = strategy_analysis.get('seasonal_analysis', {})
        if seasonal_analysis:
            phase = seasonal_analysis.get('current_phase', 'unknown')
            bias = seasonal_analysis.get('historical_bias', 0)
            seasonal_info = f', Seasonal Phase: {phase} (Bias: {bias:.2f})'
        time_projection_info = ''
        time_projections = strategy_analysis.get('time_projections', {})
        if time_projections:
            next_dates = []
            for tf_proj in time_projections.values():
                if tf_proj.get('dates'):
                    next_dates.extend(tf_proj['dates'][:2])
            if next_dates:
                time_projection_info = (
                    f", Key dates: {', '.join(str(d) for d in next_dates[:3])}"
                    )
        signal = {'symbol': symbol, 'timestamp': datetime.now().isoformat(),
            'signal_type': 'gann', 'direction': direction, 'strength':
            strength, 'confidence': min(0.5 + strength * 0.1, 0.95),
            'entry_price': current_price, 'stop_loss': stop_loss,
            'take_profit': take_profit, 'reason':
            f'{strength} Gann levels confluence, Direction: {direction}{seasonal_info}{time_projection_info}'
            }
        timeframe_scores = strategy_analysis.get('timeframe_scores', {})
        if timeframe_scores:
            signal['timeframe_data'] = timeframe_scores
            original_scores = strategy_analysis.get('original_timeframe_scores'
                )
            if original_scores:
                signal['original_timeframe_data'] = original_scores
            weighted_avg = strategy_analysis.get('weighted_average_score')
            if weighted_avg is not None:
                signal['weighted_average_score'] = weighted_avg
        if self.config_manager.get('use_timeframe_optimization', True):
            recommended_tfs = (self.timeframe_optimizer.
                get_recommended_timeframes(max_count=3))
            signal['recommended_timeframes'] = recommended_tfs
        self._enhance_signal_with_additional_analysis(signal,
            strategy_analysis, symbol)
        signals.append(signal)
        return signals

    @with_exception_handling
    def _analyze_currency_strength(self, symbol: str, price_data: Dict[str,
        pd.DataFrame]) ->Dict[str, Any]:
        """
        Analyze currency strength for the symbol's currencies.'

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
                symbol=symbol, price_data=price_data, signal_type='trend',
                signal_direction=direction, related_pairs=related_pairs)
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
