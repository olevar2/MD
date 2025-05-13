"""
HarmonicPatternStrategy

This module implements a trading strategy based on harmonic price patterns including:
- Gartley patterns
- Butterfly patterns
- Bat patterns
- Crab patterns
- Shark patterns
- Cypher patterns

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from core.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.analysis.harmonic_pattern_detector import HarmonicPatternDetector


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class HarmonicPatternStrategy(AdvancedTAStrategy):
    """
    Trading strategy based on harmonic price patterns with confluence-based
    confirmation and adaptation to market regimes.
    """

    def _init_strategy_config(self) ->None:
        """Initialize strategy-specific configuration parameters."""
        self.pattern_detector = HarmonicPatternDetector()
        self.adaptive_params = {'min_pattern_confidence': 0.7,
            'preferred_patterns': ['butterfly', 'gartley', 'bat', 'crab',
            'shark', 'cypher'], 'completion_threshold': 0.95,
            'confirmation_bars': 1, 'fib_extension_entry': 1.0,
            'stop_loss_buffer_pct': 0.1, 'profit_target_multiplier': {
            'conservative': 1.5, 'moderate': 2.0, 'aggressive': 3.0}}
        self.config.update({'risk_profile': 'moderate',
            'require_indicator_confluence': True, 'require_sr_confluence': 
            False, 'max_active_patterns_per_symbol': 2, 'pattern_weights':
            {'gartley': 1.0, 'butterfly': 1.1, 'bat': 0.9, 'crab': 1.2,
            'shark': 0.8, 'cypher': 0.7}, 'use_additional_filters': True})
        self.logger.info(
            f'Initialized {self.name} with harmonic pattern parameters')

    def _adapt_parameters_to_regime(self, regime: MarketRegime) ->None:
        """Adjust strategy parameters based on the current market regime."""
        self.logger.info(f'Adapting parameters to {regime} regime')
        if regime == MarketRegime.TRENDING:
            self.adaptive_params['min_pattern_confidence'] = 0.75
            self.adaptive_params['preferred_patterns'] = ['butterfly',
                'bat', 'crab']
            self.config['risk_profile'] = 'aggressive'
            self.config['require_indicator_confluence'] = True
        elif regime == MarketRegime.RANGING:
            self.adaptive_params['min_pattern_confidence'] = 0.8
            self.adaptive_params['preferred_patterns'] = ['gartley', 'bat',
                'cypher']
            self.config['risk_profile'] = 'moderate'
            self.config['require_sr_confluence'] = True
        elif regime == MarketRegime.VOLATILE:
            self.adaptive_params['min_pattern_confidence'] = 0.85
            self.adaptive_params['preferred_patterns'] = ['crab', 'butterfly']
            self.config['risk_profile'] = 'conservative'
            self.config['require_indicator_confluence'] = True
            self.config['require_sr_confluence'] = True
        elif regime == MarketRegime.BREAKOUT:
            self.adaptive_params['stop_loss_buffer_pct'] = 0.15
            self.adaptive_params['min_pattern_confidence'] = 0.8
            self.config['risk_profile'] = 'moderate'

    @with_exception_handling
    def _perform_strategy_analysis(self, symbol: str, price_data: Dict[str,
        pd.DataFrame], confluence_results: Dict[str, Any], additional_data:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Perform analysis for the harmonic pattern strategy.
        
        Args:
            symbol: The trading symbol
            price_data: Dict of price data frames indexed by timeframe
            confluence_results: Results from confluence analysis
            additional_data: Optional additional data
            
        Returns:
            Dictionary with strategy-specific analysis results
        """
        self.logger.info(f'Performing harmonic pattern analysis for {symbol}')
        analysis_result = {'detected_patterns': {}, 'filtered_patterns': [],
            'confluence_matches': [], 'combined_score': 0.0}
        try:
            if (confluence_results and 'harmonic_patterns' in
                confluence_results):
                harmonic_patterns = confluence_results['harmonic_patterns']
                if harmonic_patterns:
                    self.logger.info(
                        f'Found {len(harmonic_patterns)} harmonic patterns from confluence analyzer'
                        )
                    filtered_patterns = [pattern for pattern in
                        harmonic_patterns if pattern.get('confidence_score',
                        0) >= self.adaptive_params['min_pattern_confidence'
                        ] and pattern.get('completion_percentage', 0) >=
                        self.adaptive_params['completion_threshold']]
                    analysis_result['filtered_patterns'] = filtered_patterns
            detected_patterns = {}
            for timeframe, df in price_data.items():
                if df is None or df.empty:
                    continue
                timeframe_patterns = (self.pattern_detector.
                    detect_harmonic_patterns(df))
                timeframe_patterns = [pattern for pattern in
                    timeframe_patterns if pattern.get('confidence_score', 0
                    ) >= self.adaptive_params['min_pattern_confidence'] and
                    pattern.get('completion_percentage', 0) >= self.
                    adaptive_params['completion_threshold']]
                for pattern in timeframe_patterns:
                    pattern['timeframe'] = timeframe
                detected_patterns[timeframe] = timeframe_patterns
            analysis_result['detected_patterns'] = detected_patterns
            all_patterns = analysis_result['filtered_patterns'].copy()
            for timeframe, patterns in detected_patterns.items():
                all_patterns.extend(patterns)
            all_patterns.sort(key=lambda p: p.get('confidence_score', 0) *
                self.config['pattern_weights'].get(p.get('type', ''), 0.5),
                reverse=True)
            confluence_matches = []
            sr_zones = confluence_results.get('support_resistance_confluence',
                [])
            for pattern in all_patterns:
                d_price = pattern.get('points', {}).get('D', {}).get('price', 0
                    )
                if not d_price:
                    continue
                sr_match = self._check_sr_confluence(d_price, sr_zones)
                indicator_match = self._check_indicator_confluence(pattern,
                    confluence_results.get('indicator_confluences', []))
                if sr_match or indicator_match:
                    confluence_matches.append({'pattern': pattern,
                        'sr_match': sr_match, 'indicator_match':
                        indicator_match, 'confluence_score': sr_match.get(
                        'score', 0) + indicator_match.get('score', 0)})
            analysis_result['confluence_matches'] = confluence_matches
            if all_patterns:
                top_pattern_score = all_patterns[0].get('confidence_score', 0)
                confluence_factor = 1.0 + 0.2 * len(confluence_matches)
                analysis_result['combined_score'] = min(top_pattern_score *
                    confluence_factor, 1.0)
            return analysis_result
        except Exception as e:
            self.logger.error(f'Error in harmonic pattern analysis: {str(e)}',
                exc_info=True)
            return analysis_result

    def _check_sr_confluence(self, price: float, sr_zones: List[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """Check if a price is confluent with any S/R zone."""
        if not sr_zones:
            return {}
        closest_zone = None
        min_distance_pct = float('inf')
        for zone in sr_zones:
            zone_price = zone.get('price', 0)
            distance_pct = abs(price - zone_price) / zone_price
            if distance_pct < min_distance_pct:
                min_distance_pct = distance_pct
                closest_zone = zone
        if min_distance_pct <= 0.002:
            return {'zone': closest_zone, 'distance_pct': min_distance_pct,
                'score': max(0, 0.5 - min_distance_pct * 100)}
        return {}

    def _check_indicator_confluence(self, pattern: Dict[str, Any],
        indicator_confluences: List[Dict[str, Any]]) ->Dict[str, Any]:
        """Check if a pattern aligns with indicator signals."""
        if not indicator_confluences:
            return {}
        pattern_direction = pattern.get('direction', '')
        for confluence in indicator_confluences:
            bias = confluence.get('bias', '')
            intensity = confluence.get('intensity', 0)
            if bias == 'neutral' or intensity < 0.6:
                continue
            if (pattern_direction == 'bullish' and bias == 'bullish' or 
                pattern_direction == 'bearish' and bias == 'bearish'):
                return {'bias': bias, 'intensity': intensity, 'score': 
                    intensity * 0.5}
        return {}

    @with_exception_handling
    def _generate_signals(self, symbol: str, strategy_analysis: Dict[str,
        Any], confluence_results: Dict[str, Any]) ->List[Dict[str, Any]]:
        """Generate trading signals based on strategy analysis."""
        signals = []
        try:
            all_patterns = []
            for match in strategy_analysis.get('confluence_matches', []):
                if match.get('pattern'):
                    all_patterns.append({**match['pattern'],
                        'has_confluence': True, 'confluence_details': {
                        'sr_match': match.get('sr_match', {}),
                        'indicator_match': match.get('indicator_match', {})}})
            filtered_patterns = strategy_analysis.get('filtered_patterns', [])
            detected_patterns = strategy_analysis.get('detected_patterns', {})
            for timeframe, patterns in detected_patterns.items():
                for pattern in patterns:
                    if not any(p.get('id') == pattern.get('id') for p in
                        all_patterns if 'id' in p and 'id' in pattern):
                        pattern['has_confluence'] = False
                        all_patterns.append(pattern)
            all_patterns.sort(key=lambda p: p.get('confidence_score', 0) *
                self.config['pattern_weights'].get(p.get('type', ''), 0.5) *
                (1.2 if p.get('has_confluence', False) else 1.0), reverse=True)
            all_patterns = all_patterns[:self.config[
                'max_active_patterns_per_symbol']]
            for pattern in all_patterns:
                if pattern.get('confidence_score', 0) < self.adaptive_params[
                    'min_pattern_confidence']:
                    continue
                if self.config['require_sr_confluence'] and not pattern.get(
                    'has_confluence', False):
                    continue
                pattern_type = pattern.get('type', '')
                direction = pattern.get('direction', '')
                if pattern_type not in self.adaptive_params[
                    'preferred_patterns']:
                    continue
                points = pattern.get('points', {})
                if not points or 'D' not in points:
                    continue
                d_price = pattern['points']['D']['price']
                c_price = pattern['points'].get('C', {}).get('price', 0)
                x_price = pattern['points'].get('X', {}).get('price', 0)
                if direction == 'bullish':
                    entry_price = d_price
                    stop_loss = d_price * (1 - self.adaptive_params[
                        'stop_loss_buffer_pct'])
                    risk = entry_price - stop_loss
                    profit_target = entry_price + risk * self.adaptive_params[
                        'profit_target_multiplier'][self.config['risk_profile']
                        ]
                    if x_price and x_price > entry_price:
                        alt_target = x_price
                        profit_target = max(profit_target, alt_target)
                    signal_direction = 'buy'
                elif direction == 'bearish':
                    entry_price = d_price
                    stop_loss = d_price * (1 + self.adaptive_params[
                        'stop_loss_buffer_pct'])
                    risk = stop_loss - entry_price
                    profit_target = entry_price - risk * self.adaptive_params[
                        'profit_target_multiplier'][self.config['risk_profile']
                        ]
                    if x_price and x_price < entry_price:
                        alt_target = x_price
                        profit_target = min(profit_target, alt_target)
                    signal_direction = 'sell'
                else:
                    continue
                signal = {'symbol': symbol, 'strategy': self.name,
                    'direction': signal_direction, 'type':
                    f'harmonic_{pattern_type}', 'entry_price': entry_price,
                    'stop_loss': stop_loss, 'take_profit': profit_target,
                    'timeframe': pattern.get('timeframe', ''), 'confidence':
                    pattern.get('confidence_score', 0.5), 'timestamp':
                    datetime.now().isoformat(), 'metadata': {'pattern': {
                    'type': pattern_type, 'direction': direction,
                    'completion': pattern.get('completion_percentage', 1.0)
                    }, 'points': {'X': x_price, 'A': pattern['points'].get(
                    'A', {}).get('price', 0), 'B': pattern['points'].get(
                    'B', {}).get('price', 0), 'C': c_price, 'D': d_price},
                    'confluence': pattern.get('has_confluence', False),
                    'confluence_details': pattern.get('confluence_details',
                    {}), 'visualization': pattern.get('visualization_data',
                    {})}}
                if signal['direction'] == 'buy':
                    risk = entry_price - stop_loss
                    reward = profit_target - entry_price
                else:
                    risk = stop_loss - entry_price
                    reward = entry_price - profit_target
                signal['reward_risk_ratio'] = reward / risk if risk else 0
                signals.append(signal)
            return signals[:self.config['max_active_patterns_per_symbol']]
        except Exception as e:
            self.logger.error(f'Error generating signals: {str(e)}',
                exc_info=True)
        return signals
