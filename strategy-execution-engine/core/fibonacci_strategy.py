"""
FibonacciTradingStrategy

This module implements a trading strategy based on Fibonacci retracement and extension levels
to identify potential reversal zones and profit targets in trending markets.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from core.advanced_ta_strategy import AdvancedTAStrategy
from common_lib.effectiveness.interfaces import MarketRegimeEnum as MarketRegime, TimeFrameEnum as TimeFrame
from analysis_engine.analysis.fibonacci_tools import FibonacciTools
from analysis_engine.analysis.trend_analyzer import TrendAnalyzer


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FibonacciTradingStrategy(AdvancedTAStrategy):
    """
    Trading strategy based on Fibonacci retracement and extension levels with
    adaptive parameters based on market regime and multi-timeframe confluence.
    """

    def __init__(self, name: str, timeframes: List[str], primary_timeframe:
        str, symbols: List[str], risk_per_trade_pct: float=1.0, **kwargs):
        """
        Initialize the Fibonacci Trading Strategy

        Args:
            name: Strategy name
            timeframes: List of timeframes to analyze
            primary_timeframe: Primary timeframe for decision making
            symbols: List of symbols to trade
            risk_per_trade_pct: Risk per trade as percentage of account
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, timeframes=timeframes,
            primary_timeframe=primary_timeframe, symbols=symbols,
            risk_per_trade_pct=risk_per_trade_pct, **kwargs)
        self._init_strategy_config()
        self.logger.info(f"FibonacciTradingStrategy '{name}' initialized")

    def _init_strategy_config(self) ->None:
        """Initialize strategy-specific configuration parameters."""
        self.fibonacci_tools = FibonacciTools()
        self.trend_analyzer = TrendAnalyzer()
        self.adaptive_params = {'retracement_levels': [0.236, 0.382, 0.5, 
            0.618, 0.786], 'extension_levels': [1.272, 1.414, 1.618, 2.0, 
            2.618], 'zone_threshold_pips': 10, 'swing_lookback': 20,
            'min_swing_pips': 50, 'confirm_with_candle_pattern': True,
            'min_volume_increase': 1.5, 'wait_for_pullback': True,
            'multi_tf_confirmation': True, 'atr_multiple_sl': 1.5,
            'target_fib_level': 1.618, 'use_adaptive_targets': True}
        self.config.update({'trend_strength_threshold': 0.6,
            'use_historical_success_rates': True,
            'preferred_trade_direction': 'trend_following',
            'fibonacci_counting_method': 'swing_high_low'})
        self.level_success_rates = {'retracement': {(0.236): 0.65, (0.382):
            0.72, (0.5): 0.68, (0.618): 0.78, (0.786): 0.62}, 'extension':
            {(1.272): 0.67, (1.414): 0.65, (1.618): 0.75, (2.0): 0.63, (
            2.618): 0.58}}
        self.regime_parameters = {MarketRegime.TRENDING.value: {
            'preferred_levels': {'retracement': [0.382, 0.5, 0.618],
            'extension': [1.618, 2.0]}, 'min_swing_pips': 60,
            'target_fib_level': 1.618}, MarketRegime.RANGING.value: {
            'preferred_levels': {'retracement': [0.618, 0.786], 'extension':
            [1.272, 1.414]}, 'min_swing_pips': 30, 'target_fib_level': 
            1.272}, MarketRegime.VOLATILE.value: {'preferred_levels': {
            'retracement': [0.5, 0.618], 'extension': [1.414, 1.618]},
            'min_swing_pips': 80, 'target_fib_level': 1.414,
            'zone_threshold_pips': 15}, MarketRegime.BREAKOUT.value: {
            'preferred_levels': {'retracement': [0.236, 0.382], 'extension':
            [1.618, 2.0, 2.618]}, 'min_swing_pips': 40, 'target_fib_level':
            2.0}}
        self.level_metrics = {level: {'hits': 0, 'misses': 0} for level in
            self.adaptive_params['retracement_levels']}
        self.level_metrics.update({level: {'hits': 0, 'misses': 0} for
            level in self.adaptive_params['extension_levels']})

    def apply_adaptive_parameters(self, parameters: Dict[str, Any]) ->None:
        """Apply adaptive parameters to the strategy"""
        if not parameters:
            return
        for key, value in parameters.items():
            if key in self.adaptive_params:
                self.adaptive_params[key] = value
                self.logger.debug(
                    f'Applied adaptive parameter: {key} = {value}')

    def _adapt_parameters_to_regime(self, regime: MarketRegime) ->None:
        """
        Adjust strategy parameters based on the current market regime.
        """
        params = self.regime_parameters.get(regime.value)
        if not params:
            return
        adaptive_updates = {k: v for k, v in params.items() if k in self.
            adaptive_params}
        self.apply_adaptive_parameters(adaptive_updates)
        self.logger.info(
            f'Adapted Fibonacci parameters to regime {regime}: {adaptive_updates}'
            )

    @with_exception_handling
    def _perform_strategy_analysis(self, symbol: str, price_data: Dict[str,
        pd.DataFrame], confluence_results: Dict[str, Any], additional_data:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Perform Fibonacci analysis on price data across timeframes

        Args:
            symbol: Symbol being analyzed
            price_data: Dictionary of price data by timeframe
            confluence_results: Results from confluence analysis
            additional_data: Additional data for analysis

        Returns:
            Analysis results including Fibonacci levels and trade signals
        """
        try:
            analysis_result = {}
            market_regime = additional_data.get('market_regime',
                MarketRegime.UNKNOWN)
            regime_params = self.regime_parameters.get(market_regime.value,
                self.regime_parameters.get(MarketRegime.TRENDING.value))
            if self.adaptive_params['use_adaptive_targets']:
                preferred_levels = regime_params.get('preferred_levels', {})
                zone_threshold_pips = regime_params.get('zone_threshold_pips',
                    self.adaptive_params['zone_threshold_pips'])
                target_fib_level = regime_params.get('target_fib_level',
                    self.adaptive_params['target_fib_level'])
            else:
                preferred_levels = {'retracement': self.adaptive_params[
                    'retracement_levels'], 'extension': self.
                    adaptive_params['extension_levels']}
                zone_threshold_pips = self.adaptive_params[
                    'zone_threshold_pips']
                target_fib_level = self.adaptive_params['target_fib_level']
            timeframe_analysis = {}
            for timeframe, df in price_data.items():
                if len(df) < self.adaptive_params['swing_lookback'] + 10:
                    continue
                swings = self._identify_swing_points(df, lookback=self.
                    adaptive_params['swing_lookback'], min_swing_pips=
                    regime_params.get('min_swing_pips', self.
                    adaptive_params['min_swing_pips']))
                fib_levels = self._calculate_fibonacci_levels(df, swings,
                    preferred_levels)
                fib_zones = self._identify_fibonacci_zones(df, fib_levels,
                    zone_threshold_pips=zone_threshold_pips)
                zone_interactions = self._check_zone_interactions(df, fib_zones
                    )
                confirmations = {}
                if self.adaptive_params['confirm_with_candle_pattern']:
                    confirmations = self._check_confirmations(df, fib_zones,
                        zone_interactions)
                timeframe_analysis[timeframe] = {'swings': swings,
                    'fibonacci_levels': fib_levels, 'fibonacci_zones':
                    fib_zones, 'zone_interactions': zone_interactions,
                    'confirmations': confirmations}
            analysis_result['timeframe_analysis'] = timeframe_analysis
            analysis_result['fibonacci_confluence'
                ] = self._calculate_fibonacci_confluence(timeframe_analysis)
            potential_trades = self._generate_potential_trades(symbol,
                price_data[self.primary_timeframe], analysis_result[
                'fibonacci_confluence'], target_fib_level=target_fib_level,
                additional_data=additional_data)
            analysis_result['potential_trades'] = potential_trades
            combined_score = self._calculate_combined_score(analysis_result)
            analysis_result['combined_score'] = combined_score
            return analysis_result
        except Exception as e:
            self.logger.error(f'Error in Fibonacci strategy analysis: {str(e)}'
                , exc_info=True)
            return {'error': str(e)}

    def _identify_swing_points(self, df: pd.DataFrame, lookback: int=20,
        min_swing_pips: int=50) ->Dict[str, List[Dict[str, Any]]]:
        """
        Identify swing highs and lows in the price data

        Args:
            df: Price data
            lookback: Number of bars to look back
            min_swing_pips: Minimum movement in pips to consider as swing

        Returns:
            Dictionary with swing highs and lows
        """
        highs = []
        lows = []
        pip_multiplier = 10000 if 'JPY' not in df['symbol'].iloc[0] else 100
        for i in range(lookback, len(df) - lookback):
            if df['high'].iloc[i] == max(df['high'].iloc[i - lookback:i +
                lookback + 1]):
                left_min = min(df['low'].iloc[i - lookback:i])
                right_min = min(df['low'].iloc[i:i + lookback + 1])
                swing_size_pips = (df['high'].iloc[i] - min(left_min,
                    right_min)) * pip_multiplier
                if swing_size_pips >= min_swing_pips:
                    highs.append({'index': df.index[i], 'price': df['high']
                        .iloc[i], 'bar_index': i, 'magnitude': swing_size_pips}
                        )
        for i in range(lookback, len(df) - lookback):
            if df['low'].iloc[i] == min(df['low'].iloc[i - lookback:i +
                lookback + 1]):
                left_max = max(df['high'].iloc[i - lookback:i])
                right_max = max(df['high'].iloc[i:i + lookback + 1])
                swing_size_pips = (max(left_max, right_max) - df['low'].iloc[i]
                    ) * pip_multiplier
                if swing_size_pips >= min_swing_pips:
                    lows.append({'index': df.index[i], 'price': df['low'].
                        iloc[i], 'bar_index': i, 'magnitude': swing_size_pips})
        highs.sort(key=lambda x: x['bar_index'])
        lows.sort(key=lambda x: x['bar_index'])
        return {'highs': highs, 'lows': lows}

    def _calculate_fibonacci_levels(self, df: pd.DataFrame, swings: Dict[
        str, List[Dict[str, Any]]], preferred_levels: Dict[str, List[float]]
        ) ->Dict[str, List[Dict[str, Any]]]:
        """
        Calculate Fibonacci retracement and extension levels based on swing points

        Args:
            df: Price data
            swings: Dictionary with swing highs and lows
            preferred_levels: Dictionary with preferred levels

        Returns:
            Dictionary with calculated Fibonacci levels
        """
        retrace_levels = []
        extend_levels = []
        trend = self.trend_analyzer.detect_trend(df.tail(50))
        recent_swings = self._get_recent_swing_pair(swings, trend)
        if not recent_swings:
            return {'retracement': [], 'extension': []}
        start_point, end_point = recent_swings
        price_range = abs(end_point['price'] - start_point['price'])
        for level in preferred_levels.get('retracement', self.
            adaptive_params['retracement_levels']):
            if end_point['price'] > start_point['price']:
                price_level = end_point['price'] - price_range * level
            else:
                price_level = end_point['price'] + price_range * level
            retrace_levels.append({'level': level, 'price': price_level,
                'type': 'retracement', 'direction': 'uptrend' if end_point[
                'price'] > start_point['price'] else 'downtrend',
                'success_rate': self.level_success_rates['retracement'].get
                (level, 0.5)})
        for level in preferred_levels.get('extension', self.adaptive_params
            ['extension_levels']):
            if end_point['price'] > start_point['price']:
                price_level = end_point['price'] + price_range * level
            else:
                price_level = end_point['price'] - price_range * level
            extend_levels.append({'level': level, 'price': price_level,
                'type': 'extension', 'direction': 'uptrend' if end_point[
                'price'] > start_point['price'] else 'downtrend',
                'success_rate': self.level_success_rates['extension'].get(
                level, 0.5)})
        return {'retracement': retrace_levels, 'extension': extend_levels}

    def _get_recent_swing_pair(self, swings: Dict[str, List[Dict[str, Any]]
        ], trend: str) ->Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Get the most recent pair of swing points defining a complete move

        Args:
            swings: Dictionary with swing highs and lows
            trend: Current market trend

        Returns:
            Tuple of start point and end point or None if not found
        """
        highs = swings['highs']
        lows = swings['lows']
        if not highs or not lows:
            return None
        if trend == 'uptrend':
            if not lows:
                return None
            latest_low = max(lows, key=lambda x: x['bar_index'])
            highs_after_low = [h for h in highs if h['bar_index'] >
                latest_low['bar_index']]
            if not highs_after_low:
                return None
            latest_high = max(highs_after_low, key=lambda x: x['bar_index'])
            return latest_low, latest_high
        elif trend == 'downtrend':
            if not highs:
                return None
            latest_high = max(highs, key=lambda x: x['bar_index'])
            lows_after_high = [l for l in lows if l['bar_index'] >
                latest_high['bar_index']]
            if not lows_after_high:
                return None
            latest_low = max(lows_after_high, key=lambda x: x['bar_index'])
            return latest_high, latest_low
        else:
            all_points = highs + lows
            all_points.sort(key=lambda x: x['bar_index'], reverse=True)
            if len(all_points) < 2:
                return None
            return all_points[1], all_points[0]

    def _identify_fibonacci_zones(self, df: pd.DataFrame, fib_levels: Dict[
        str, List[Dict[str, Any]]], zone_threshold_pips: int=10) ->Dict[str,
        List[Dict[str, Any]]]:
        """
        Create zones around Fibonacci levels for entry/exit consideration

        Args:
            df: Price data
            fib_levels: Dictionary with Fibonacci levels
            zone_threshold_pips: Size of zones around levels in pips

        Returns:
            Dictionary with Fibonacci zones
        """
        pip_multiplier = 10000 if 'JPY' not in df['symbol'].iloc[0] else 100
        pip_size = 1 / pip_multiplier
        zone_threshold = zone_threshold_pips * pip_size
        retracement_zones = []
        extension_zones = []
        for level in fib_levels['retracement']:
            retracement_zones.append({'level': level['level'], 'price':
                level['price'], 'upper': level['price'] + zone_threshold,
                'lower': level['price'] - zone_threshold, 'type':
                'retracement', 'direction': level['direction'],
                'success_rate': level['success_rate']})
        for level in fib_levels['extension']:
            extension_zones.append({'level': level['level'], 'price': level
                ['price'], 'upper': level['price'] + zone_threshold,
                'lower': level['price'] - zone_threshold, 'type':
                'extension', 'direction': level['direction'],
                'success_rate': level['success_rate']})
        return {'retracement': retracement_zones, 'extension': extension_zones}

    def _check_zone_interactions(self, df: pd.DataFrame, fib_zones: Dict[
        str, List[Dict[str, Any]]]) ->Dict[str, List[Dict[str, Any]]]:
        """
        Check if price has interacted with Fibonacci zones

        Args:
            df: Price data
            fib_zones: Dictionary with Fibonacci zones

        Returns:
            Dictionary with zone interactions
        """
        retracement_interactions = []
        extension_interactions = []
        recent_data = df.tail(10)
        for zone in fib_zones['retracement']:
            for i, row in recent_data.iterrows():
                if row['low'] <= zone['upper'] and row['high'] >= zone['lower'
                    ]:
                    retracement_interactions.append({'level': zone['level'],
                        'price': zone['price'], 'candle_index': i,
                        'interaction_type': 'bounce' if zone['direction'] ==
                        'uptrend' and row['close'] > row['open'] or zone[
                        'direction'] == 'downtrend' and row['close'] < row[
                        'open'] else 'rejection', 'strength': abs(row[
                        'close'] - row['open']) / abs(row['high'] - row[
                        'low']), 'zone': zone})
                    break
        for zone in fib_zones['extension']:
            for i, row in recent_data.iterrows():
                if row['low'] <= zone['upper'] and row['high'] >= zone['lower'
                    ]:
                    extension_interactions.append({'level': zone['level'],
                        'price': zone['price'], 'candle_index': i,
                        'interaction_type': 'target_reached', 'strength': 
                        abs(row['close'] - row['open']) / abs(row['high'] -
                        row['low']), 'zone': zone})
                    break
        return {'retracement': retracement_interactions, 'extension':
            extension_interactions}

    def _check_confirmations(self, df: pd.DataFrame, fib_zones: Dict[str,
        List[Dict[str, Any]]], zone_interactions: Dict[str, List[Dict[str,
        Any]]]) ->Dict[str, List[Dict[str, Any]]]:
        """
        Check for confirming patterns at Fibonacci zones

        Args:
            df: Price data
            fib_zones: Dictionary with Fibonacci zones
            zone_interactions: Dictionary with zone interactions

        Returns:
            Dictionary with confirmations
        """
        retracement_confirmations = []
        for interaction in zone_interactions['retracement']:
            zone = interaction['zone']
            idx = interaction['candle_index']
            start_idx = df.index.get_loc(idx) - 2 if idx in df.index else 0
            end_idx = min(df.index.get_loc(idx) + 3 if idx in df.index else
                5, len(df) - 1)
            interaction_df = df.iloc[start_idx:end_idx + 1]
            volume_increase = False
            if 'volume' in df.columns and len(interaction_df) > 1:
                avg_volume = interaction_df['volume'].iloc[:-1].mean()
                current_volume = interaction_df['volume'].iloc[-1]
                volume_increase = (current_volume >= avg_volume * self.
                    adaptive_params['min_volume_increase'])
            pattern_confirmation = False
            if zone['direction'] == 'uptrend' and interaction[
                'interaction_type'] == 'bounce':
                candle = interaction_df.iloc[-1]
                prev_candle = interaction_df.iloc[-2] if len(interaction_df
                    ) > 1 else None
                if prev_candle is not None:
                    bullish_engulfing = candle['open'] < prev_candle['close'
                        ] and candle['close'] > prev_candle['open'] and candle[
                        'close'] > candle['open']
                    hammer = candle['close'] > candle['open'] and candle['high'
                        ] - candle['close'] < 0.3 * (candle['close'] -
                        candle['low']) and candle['close'] - candle['open'
                        ] > 0.6 * (candle['high'] - candle['low'])
                    pattern_confirmation = bullish_engulfing or hammer
            elif zone['direction'] == 'downtrend' and interaction[
                'interaction_type'] == 'bounce':
                candle = interaction_df.iloc[-1]
                prev_candle = interaction_df.iloc[-2] if len(interaction_df
                    ) > 1 else None
                if prev_candle is not None:
                    bearish_engulfing = candle['open'] > prev_candle['close'
                        ] and candle['close'] < prev_candle['open'] and candle[
                        'close'] < candle['open']
                    shooting_star = candle['close'] < candle['open'
                        ] and candle['close'] - candle['low'] < 0.3 * (candle
                        ['high'] - candle['close']) and candle['open'
                        ] - candle['close'] > 0.6 * (candle['high'] -
                        candle['low'])
                    pattern_confirmation = bearish_engulfing or shooting_star
            if volume_increase or pattern_confirmation:
                retracement_confirmations.append({'level': zone['level'],
                    'price': zone['price'], 'volume_confirmation':
                    volume_increase, 'pattern_confirmation':
                    pattern_confirmation, 'strength': (0.7 if
                    volume_increase else 0) + (0.8 if pattern_confirmation else
                    0), 'zone': zone, 'interaction': interaction})
        return {'retracement': retracement_confirmations, 'extension': []}

    def _calculate_fibonacci_confluence(self, timeframe_analysis: Dict[str,
        Dict[str, Any]]) ->Dict[str, Any]:
        """
        Calculate confluence of Fibonacci levels across timeframes

        Args:
            timeframe_analysis: Dictionary with analysis for each timeframe

        Returns:
            Dictionary with confluence analysis
        """
        confluence = {'retracement_confluence': [], 'extension_confluence': []}
        if not timeframe_analysis:
            return confluence
        all_retrace_levels = []
        all_extend_levels = []
        for tf, analysis in timeframe_analysis.items():
            for zone in analysis.get('fibonacci_zones', {}).get('retracement',
                []):
                all_retrace_levels.append({'timeframe': tf, 'price': zone[
                    'price'], 'upper': zone['upper'], 'lower': zone['lower'
                    ], 'level': zone['level'], 'success_rate': zone[
                    'success_rate'], 'direction': zone['direction']})
            for zone in analysis.get('fibonacci_zones', {}).get('extension', []
                ):
                all_extend_levels.append({'timeframe': tf, 'price': zone[
                    'price'], 'upper': zone['upper'], 'lower': zone['lower'
                    ], 'level': zone['level'], 'success_rate': zone[
                    'success_rate'], 'direction': zone['direction']})
        confluence['retracement_confluence'] = self._find_level_confluence(
            all_retrace_levels)
        confluence['extension_confluence'] = self._find_level_confluence(
            all_extend_levels)
        return confluence

    def _find_level_confluence(self, levels: List[Dict[str, Any]]) ->List[Dict
        [str, Any]]:
        """
        Find levels that are in confluence across timeframes

        Args:
            levels: List of level dictionaries

        Returns:
            List of confluence groups
        """
        if not levels:
            return []
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        confluence_groups = []
        current_group = [sorted_levels[0]]
        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            prev_level = current_group[-1]
            if current_level['lower'] <= prev_level['upper'] and current_level[
                'upper'] >= prev_level['lower']:
                current_group.append(current_level)
            else:
                if len(current_group) > 1:
                    avg_price = sum(l['price'] for l in current_group) / len(
                        current_group)
                    avg_success = sum(l['success_rate'] for l in current_group
                        ) / len(current_group)
                    timeframes = list(set(l['timeframe'] for l in
                        current_group))
                    directions = list(set(l['direction'] for l in
                        current_group))
                    confluence_groups.append({'price': avg_price, 'levels':
                        current_group, 'timeframes': timeframes,
                        'num_timeframes': len(timeframes), 'strength': len(
                        current_group) * avg_success, 'direction': 
                        directions[0] if len(directions) == 1 else 'mixed'})
                current_group = [current_level]
        if len(current_group) > 1:
            avg_price = sum(l['price'] for l in current_group) / len(
                current_group)
            avg_success = sum(l['success_rate'] for l in current_group) / len(
                current_group)
            timeframes = list(set(l['timeframe'] for l in current_group))
            directions = list(set(l['direction'] for l in current_group))
            confluence_groups.append({'price': avg_price, 'levels':
                current_group, 'timeframes': timeframes, 'num_timeframes':
                len(timeframes), 'strength': len(current_group) *
                avg_success, 'direction': directions[0] if len(directions) ==
                1 else 'mixed'})
        confluence_groups.sort(key=lambda x: x['strength'], reverse=True)
        return confluence_groups

    def _generate_potential_trades(self, symbol: str, df: pd.DataFrame,
        fibonacci_confluence: Dict[str, List[Dict[str, Any]]],
        target_fib_level: float=1.618, additional_data: Optional[Dict[str,
        Any]]=None) ->List[Dict[str, Any]]:
        """
        Generate potential trades based on Fibonacci analysis

        Args:
            symbol: Symbol being analyzed
            df: Price data
            fibonacci_confluence: Fibonacci confluence analysis
            target_fib_level: Target Fibonacci level
            additional_data: Additional data for analysis

        Returns:
            List of potential trades
        """
        potential_trades = []
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else df['close'].iloc[
            -1] * 0.001
        for conf in fibonacci_confluence.get('retracement_confluence', []):
            if conf['num_timeframes'] < 2 and self.adaptive_params[
                'multi_tf_confirmation']:
                continue
            price_near = abs(current_price - conf['price']) < atr * 2
            if conf['direction'] == 'uptrend' and price_near:
                stop_loss = conf['price'] - atr * self.adaptive_params[
                    'atr_multiple_sl']
                targets = []
                for ext_conf in fibonacci_confluence.get('extension_confluence'
                    , []):
                    if ext_conf['direction'] == 'uptrend' and ext_conf['price'
                        ] > current_price:
                        targets.append(ext_conf['price'])
                if not targets and fibonacci_confluence.get(
                    'extension_confluence'):
                    for ext in fibonacci_confluence.get('extension_confluence',
                        []):
                        if ext['direction'] == 'uptrend' and ext['price'
                            ] > current_price:
                            targets.append(ext['price'])
                            break
                if not targets:
                    price_range = conf['price'] - stop_loss
                    targets = [current_price + price_range * 1.5]
                potential_trades.append({'symbol': symbol, 'direction':
                    'buy', 'entry_price': current_price, 'stop_loss':
                    stop_loss, 'targets': targets, 'risk_reward': (targets[
                    0] - current_price) / (current_price - stop_loss) if
                    targets else 0, 'confidence': conf['strength'],
                    'timeframes': conf['timeframes'], 'reason':
                    f"Bullish setup at {conf['levels'][0]['level']} retracement with {conf['num_timeframes']} timeframe confluence"
                    })
            elif conf['direction'] == 'downtrend' and price_near:
                stop_loss = conf['price'] + atr * self.adaptive_params[
                    'atr_multiple_sl']
                targets = []
                for ext_conf in fibonacci_confluence.get('extension_confluence'
                    , []):
                    if ext_conf['direction'] == 'downtrend' and ext_conf[
                        'price'] < current_price:
                        targets.append(ext_conf['price'])
                if not targets and fibonacci_confluence.get(
                    'extension_confluence'):
                    for ext in fibonacci_confluence.get('extension_confluence',
                        []):
                        if ext['direction'] == 'downtrend' and ext['price'
                            ] < current_price:
                            targets.append(ext['price'])
                            break
                if not targets:
                    price_range = stop_loss - conf['price']
                    targets = [current_price - price_range * 1.5]
                potential_trades.append({'symbol': symbol, 'direction':
                    'sell', 'entry_price': current_price, 'stop_loss':
                    stop_loss, 'targets': targets, 'risk_reward': (
                    current_price - targets[0]) / (stop_loss -
                    current_price) if targets else 0, 'confidence': conf[
                    'strength'], 'timeframes': conf['timeframes'], 'reason':
                    f"Bearish setup at {conf['levels'][0]['level']} retracement with {conf['num_timeframes']} timeframe confluence"
                    })
        potential_trades.sort(key=lambda x: x['confidence'], reverse=True)
        return potential_trades

    def _calculate_combined_score(self, analysis_result: Dict[str, Any]
        ) ->float:
        """
        Calculate a combined confidence score for the trade signal

        Args:
            analysis_result: Analysis result dictionary

        Returns:
            Confidence score between 0 and 1
        """
        if not analysis_result.get('potential_trades'):
            return 0.0
        best_trade = analysis_result['potential_trades'][0]
        confidence = best_trade['confidence']
        rr = best_trade.get('risk_reward', 0)
        rr_factor = min(rr / 2, 1.0) if rr > 0 else 0.0
        tf_count = len(best_trade.get('timeframes', []))
        tf_factor = min(tf_count / 3, 1.0)
        final_score = 0.5 * confidence + 0.3 * rr_factor + 0.2 * tf_factor
        return min(max(final_score, 0.0), 1.0)

    def _generate_signals(self, symbol: str, strategy_analysis: Dict[str,
        Any], confluence_results: Dict[str, Any]) ->List[Dict[str, Any]]:
        """
        Generate trading signals based on Fibonacci analysis

        Args:
            symbol: Symbol being analyzed
            strategy_analysis: Strategy analysis results
            confluence_results: Confluence analysis results

        Returns:
            List of trading signals
        """
        signals = []
        if 'error' in strategy_analysis:
            return []
        potential_trades = strategy_analysis.get('potential_trades', [])
        if not potential_trades:
            return []
        best_trade = potential_trades[0]
        min_confidence = 0.6
        if best_trade['confidence'] < min_confidence:
            return []
        signal = {'symbol': symbol, 'timestamp': datetime.now().isoformat(),
            'signal_type': 'fibonacci', 'direction': best_trade['direction'
            ], 'confidence': best_trade['confidence'], 'entry_price':
            best_trade['entry_price'], 'stop_loss': best_trade['stop_loss'],
            'targets': best_trade['targets'], 'risk_reward': best_trade[
            'risk_reward'], 'timeframes': best_trade['timeframes'],
            'reason': best_trade['reason']}
        signals.append(signal)
        return signals
