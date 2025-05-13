"""
Pivot Point Confluence Strategy

This module implements a trading strategy based on pivot point confluence across multiple timeframes,
identifying strong support and resistance levels for high-probability trades.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from strategy_execution_engine.strategies.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.analysis.advanced_ta.pivot_points import PivotPointAnalyzer
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame
from optimization.caching.calculation_cache import memoize_with_ttl


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class PivotConfluenceStrategy(AdvancedTAStrategy):
    """
    Trading strategy that identifies high-probability trading opportunities at pivot point confluence zones
    across multiple timeframes, with adaptive parameters based on market regimes.
    """

    def __init__(self, name: str, timeframes: List[str], primary_timeframe:
        str, symbols: List[str], risk_per_trade_pct: float=1.0, **kwargs):
    """
      init  .
    
    Args:
        name: Description of name
        timeframes: Description of timeframes
        primary_timeframe: Description of primary_timeframe
        symbols: Description of symbols
        risk_per_trade_pct: Description of risk_per_trade_pct
        kwargs: Description of kwargs
    
    """

        super().__init__(name=name, timeframes=timeframes,
            primary_timeframe=primary_timeframe, symbols=symbols,
            risk_per_trade_pct=risk_per_trade_pct, **kwargs)
        self._init_strategy_config()
        self.logger.info(f"PivotConfluenceStrategy '{name}' initialized")

    def _init_strategy_config(self) ->None:
    """
     init strategy config.
    
    """

        self.pivot_analyzer = PivotPointAnalyzer(methods=['standard',
            'fibonacci', 'woodie', 'camarilla'], lookback_period=10,
            include_midpoints=True)
        self.adaptive_params = {'confluence_radius_pips': 10,
            'min_confluence_count': 3, 'atr_multiple_sl': 1.0,
            'use_adr_adjustment': True, 'batch_processing': True,
            'max_pivot_age': 5, 'score_threshold': 7}
        self.regime_parameters = {MarketRegime.TRENDING.value: {
            'confluence_radius_pips': 15, 'min_confluence_count': 4,
            'max_pivot_age': 3}, MarketRegime.RANGING.value: {
            'confluence_radius_pips': 8, 'min_confluence_count': 3,
            'max_pivot_age': 7}, MarketRegime.VOLATILE.value: {
            'confluence_radius_pips': 20, 'min_confluence_count': 5,
            'max_pivot_age': 2}, MarketRegime.BREAKOUT.value: {
            'confluence_radius_pips': 12, 'min_confluence_count': 3,
            'max_pivot_age': 4}}
        self.config.update({'preferred_direction': 'both', 'min_confidence':
            0.6, 'pivot_types': ['PP', 'S1', 'S2', 'R1', 'R2', 'S3', 'R3']})

    def _adapt_parameters_to_regime(self, regime: MarketRegime) ->None:
        """
        Adjust adaptive parameters based on market regime for pivot confluence strategy.
        
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
            f'Adapted pivot confluence parameters to regime {regime}: {adaptive_updates}'
            )
        if regime == MarketRegime.TRENDING:
            self.pivot_analyzer.include_midpoints = False
        elif regime == MarketRegime.RANGING:
            self.pivot_analyzer.include_midpoints = True

    @memoize_with_ttl(ttl=300)
    def _calculate_pivot_points(self, symbol: str, df: pd.DataFrame, method:
        str) ->pd.DataFrame:
        """
        Calculate pivot points with caching for performance optimization.
        
        Args:
            symbol: The trading symbol
            df: Price data DataFrame
            method: Pivot calculation method
            
        Returns:
            DataFrame with pivot calculations
        """
        return self.pivot_analyzer.calculate(df, method=method)

    @with_exception_handling
    def _find_confluence_zones(self, price_data: Dict[str, pd.DataFrame],
        symbol: str, radius_pips: float) ->List[Dict[str, Any]]:
        """
        Find zones where multiple pivot points from different timeframes are in close proximity.
        
        Args:
            price_data: Dictionary of price data by timeframe
            symbol: Trading symbol
            radius_pips: Radius in pips to consider pivots in confluence
            
        Returns:
            List of confluence zones with scores and pivot details
        """
        all_pivots = []
        current_price = None
        for tf, df in price_data.items():
            if df.empty:
                continue
            if current_price is None and len(df) > 0:
                current_price = df['close'].iloc[-1]
            for method in ['standard', 'fibonacci', 'woodie', 'camarilla']:
                try:
                    pivot_df = self._calculate_pivot_points(symbol, df, method)
                    for pivot_type in self.config['pivot_types']:
                        if f'{method}_{pivot_type}' in pivot_df.columns:
                            pivot_value = pivot_df[f'{method}_{pivot_type}'
                                ].iloc[-1]
                            if pd.notna(pivot_value):
                                all_pivots.append({'timeframe': tf,
                                    'method': method, 'type': pivot_type,
                                    'value': pivot_value, 'age': 0})
                    max_age = self.adaptive_params['max_pivot_age']
                    for i in range(1, min(max_age + 1, len(pivot_df))):
                        for pivot_type in self.config['pivot_types']:
                            col_name = f'{method}_{pivot_type}'
                            if col_name in pivot_df.columns:
                                pivot_value = pivot_df[col_name].iloc[-(i + 1)]
                                if pd.notna(pivot_value):
                                    all_pivots.append({'timeframe': tf,
                                        'method': method, 'type':
                                        pivot_type, 'value': pivot_value,
                                        'age': i})
                except Exception as e:
                    self.logger.error(
                        f'Error calculating {method} pivots for {tf}: {e}')
        if not all_pivots:
            return []
        pips_factor = 0.0001
        radius = radius_pips * pips_factor
        all_pivots.sort(key=lambda x: x['value'])
        zones = []
        i = 0
        while i < len(all_pivots):
            zone_pivots = [all_pivots[i]]
            center = all_pivots[i]['value']
            j = i + 1
            while j < len(all_pivots) and abs(all_pivots[j]['value'] - center
                ) <= radius:
                zone_pivots.append(all_pivots[j])
                j += 1
            if len(zone_pivots) >= self.adaptive_params['min_confluence_count'
                ]:
                score = self._calculate_zone_score(zone_pivots)
                zone = {'center': sum(p['value'] for p in zone_pivots) /
                    len(zone_pivots), 'pivots': zone_pivots, 'count': len(
                    zone_pivots), 'score': score, 'unique_timeframes': len(
                    set(p['timeframe'] for p in zone_pivots)),
                    'unique_methods': len(set(p['method'] for p in
                    zone_pivots))}
                if current_price is not None:
                    zone['type'] = 'support' if zone['center'
                        ] < current_price else 'resistance'
                    zone['distance'] = abs(zone['center'] - current_price)
                zones.append(zone)
            i = j
        return zones

    def _calculate_zone_score(self, pivots: List[Dict[str, Any]]) ->float:
        """
        Calculate a score for a confluence zone based on pivot properties.
        
        Args:
            pivots: List of pivots in the zone
            
        Returns:
            Zone score
        """
        base_score = len(pivots)
        unique_timeframes = set(p['timeframe'] for p in pivots)
        unique_methods = set(p['method'] for p in pivots)
        unique_types = set(p['type'] for p in pivots)
        age_penalty = sum(p['age'] for p in pivots) / len(pivots)
        type_weights = {'PP': 1.5, 'S1': 1.2, 'R1': 1.2, 'S2': 1.1, 'R2': 
            1.1, 'S3': 1.0, 'R3': 1.0}
        type_bonus = sum(type_weights.get(p['type'], 1.0) for p in pivots)
        score = base_score * 1.5 + len(unique_timeframes) * 2 + len(
            unique_methods) * 1.5 + len(unique_types) * 0.5
        score += type_bonus - age_penalty
        return score

    def _perform_strategy_analysis(self, symbol: str, price_data: Dict[str,
        pd.DataFrame], confluence_results: Dict[str, Any], additional_data:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Perform pivot confluence analysis across timeframes.
        
        Args:
            symbol: Trading symbol
            price_data: Dictionary of price data by timeframe
            confluence_results: Results from other indicators
            additional_data: Any additional data for analysis
            
        Returns:
            Analysis results
        """
        results = {}
        current_price = None
        for tf in sorted(price_data.keys()):
            df = price_data[tf]
            if not df.empty:
                current_price = df['close'].iloc[-1]
                break
        if current_price is None:
            self.logger.warning(f'No price data available for {symbol}')
            return results
        radius_pips = self.adaptive_params['confluence_radius_pips']
        zones = self._find_confluence_zones(price_data, symbol, radius_pips)
        if not zones:
            self.logger.info(f'No confluence zones found for {symbol}')
            return results
        zones.sort(key=lambda z: z['score'], reverse=True)
        supports = [z for z in zones if z['type'] == 'support']
        resistances = [z for z in zones if z['type'] == 'resistance']
        nearest_support = min(supports, key=lambda z: z['distance']
            ) if supports else None
        nearest_resistance = min(resistances, key=lambda z: z['distance']
            ) if resistances else None
        results['zones'] = zones
        results['best_zone'] = zones[0] if zones else None
        results['nearest_support'] = nearest_support
        results['nearest_resistance'] = nearest_resistance
        results['price'] = current_price
        if zones:
            best_score = zones[0]['score']
            score_threshold = self.adaptive_params['score_threshold']
            min_count = self.adaptive_params['min_confluence_count']
            if best_score >= score_threshold and zones[0]['count'
                ] >= min_count:
                results['signal_strength'] = min(10, int(best_score /
                    score_threshold * 5))
            else:
                results['signal_strength'] = 0
        else:
            results['signal_strength'] = 0
        return results

    def _generate_signals(self, symbol: str, strategy_analysis: Dict[str,
        Any], confluence_results: Dict[str, Any]) ->List[Dict[str, Any]]:
        """
        Generate trading signals based on pivot confluence analysis.
        
        Args:
            symbol: Trading symbol
            strategy_analysis: Results from pivot confluence analysis
            confluence_results: Results from other indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        strength = strategy_analysis.get('signal_strength', 0)
        if strength < 1:
            return signals
        current_price = strategy_analysis.get('price')
        best_zone = strategy_analysis.get('best_zone')
        nearest_support = strategy_analysis.get('nearest_support')
        nearest_resistance = strategy_analysis.get('nearest_resistance')
        if not current_price or not best_zone:
            return signals
        if best_zone['type'] == 'support':
            direction = 'bullish'
            entry_price = nearest_support['center'] + 0.0001 * 5
            stop_loss = nearest_support['center'] - 0.0001 * 10
            if nearest_resistance:
                take_profit = nearest_resistance['center'] - 0.0001 * 5
            else:
                take_profit = entry_price + (entry_price - stop_loss) * 2
        else:
            direction = 'bearish'
            entry_price = nearest_resistance['center'] - 0.0001 * 5
            stop_loss = nearest_resistance['center'] + 0.0001 * 10
            if nearest_support:
                take_profit = nearest_support['center'] + 0.0001 * 5
            else:
                take_profit = entry_price - (stop_loss - entry_price) * 2
        zone_details = f"{best_zone['count']} pivots"
        if best_zone.get('unique_timeframes'):
            zone_details += (
                f" across {best_zone['unique_timeframes']} timeframes")
        if best_zone.get('unique_methods'):
            zone_details += (
                f" using {best_zone['unique_methods']} pivot methods")
        signal = {'symbol': symbol, 'timestamp': datetime.now().isoformat(),
            'signal_type': 'pivot_confluence', 'direction': direction,
            'strength': strength, 'confidence': min(0.5 + strength * 0.05, 
            0.95), 'entry_price': entry_price, 'stop_loss': stop_loss,
            'take_profit': take_profit, 'reason':
            f"Pivot confluence zone ({zone_details}) detected as {best_zone['type']} with score {best_zone['score']:.1f}"
            }
        signals.append(signal)
        return signals


""""""
