"""
AdaptiveMAStrategy

This module implements a trading strategy based on dynamically adjusted moving averages
that adapt to changing market conditions and volatility levels.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from core.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.analysis.technical_indicators import TechnicalIndicators
from analysis_engine.analysis.volatility_analysis import VolatilityAnalyzer
from analysis_engine.learning_from_mistakes.ma_optimization import MAOptimizationEngine
from adapters.feature_store_client import FeatureStoreClient


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdaptiveMAStrategy(AdvancedTAStrategy):
    """
    Trading strategy using dynamically adjusted moving averages that adapt to
    market volatility and regime changes.
    """

    def _init_strategy_config(self) ->None:
        """Initialize strategy-specific configuration parameters."""
        self.volatility_analyzer = VolatilityAnalyzer()
        self.ma_optimizer = MAOptimizationEngine()
        self.feature_store_client = FeatureStoreClient(use_cache=True,
            cache_ttl=300)
        self.adaptive_params = {'base_fast_ma_period': 12,
            'base_slow_ma_period': 26, 'base_signal_ma_period': 9,
            'volatility_factor': 0.5, 'ma_type': 'ema', 'confirmation_bars':
            2, 'atr_multiple_sl': 1.5, 'atr_multiple_tp': 2.5,
            'min_trend_strength': 0.3, 'profit_protection_level': 0.5,
            'ma_channel_multiplier': 1.0}
        self.config.update({'use_ma_crossover': True,
            'use_ma_zone_reversals': True, 'use_ma_channels': True,
            'use_additional_filters': True, 'filter_types': ['momentum',
            'volatility', 'divergence', 'support_resistance'],
            'adaptivity_level': 'medium', 'enable_ma_optimization': True,
            'enable_volatility_based_sizing': True, 'trailing_stop_enabled':
            True, 'use_adaptive_stop_loss': True, 'use_smart_exit_rules': 
            True, 'max_positions_per_pair': 2, 'auto_correlation_filter': 
            True, 'use_regime_specific_params': True, 'use_feature_store': 
            True, 'feature_store_indicators': ['sma', 'ema', 'atr', 'rsi',
            'macd', 'bollinger_bands']})
        self.adaptivity_coefficients = {'low': 0.3, 'medium': 0.5, 'high': 0.7}
        self.regime_ma_periods = {MarketRegime.TRENDING: {'fast': 12,
            'slow': 26, 'signal': 9}, MarketRegime.RANGING: {'fast': 8,
            'slow': 18, 'signal': 5}, MarketRegime.VOLATILE: {'fast': 18,
            'slow': 36, 'signal': 12}, MarketRegime.BREAKOUT: {'fast': 6,
            'slow': 18, 'signal': 4}}
        self.optimization_history = {'last_optimization': None,
            'optimization_count': 0, 'performance_improvements': []}
        self.logger.info(f'Initialized {self.name} with adaptive MA parameters'
            )

    @async_with_exception_handling
    async def optimize_parameters(self, symbol: str, timeframe: str,
        historic_data: pd.DataFrame) ->Dict[str, Any]:
        """
        Optimize the MA parameters based on historical performance.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            historic_data: Historical OHLC data

        Returns:
            Dictionary with optimization results
        """
        if not self.config['enable_ma_optimization']:
            return {'optimized': False, 'message': 'Optimization disabled'}
        try:
            self.logger.info(
                f'Running MA parameter optimization for {symbol} on {timeframe}'
                )
            regime_result = self.volatility_analyzer.detect_market_regime(
                historic_data)
            current_regime = regime_result.get('regime', MarketRegime.TRENDING)
            param_ranges = {'fast_period': range(5, 21), 'slow_period':
                range(15, 41, 2), 'signal_period': range(3, 15), 'ma_type':
                ['sma', 'ema', 'wma']}
            results = self.ma_optimizer.optimize_parameters(price_data=
                historic_data, param_ranges=param_ranges, market_regime=
                current_regime, symbol=symbol, timeframe=timeframe)
            if not results or not results.get('optimal_params'):
                return {'optimized': False, 'message':
                    'Optimization failed to find better parameters'}
            optimal_params = results['optimal_params']
            performance_gain = results['performance_improvement']
            if performance_gain > 0.1:
                self.adaptive_params['base_fast_ma_period'] = optimal_params[
                    'fast_period']
                self.adaptive_params['base_slow_ma_period'] = optimal_params[
                    'slow_period']
                self.adaptive_params['base_signal_ma_period'] = optimal_params[
                    'signal_period']
                self.adaptive_params['ma_type'] = optimal_params['ma_type']
                self.regime_ma_periods[current_regime] = {'fast':
                    optimal_params['fast_period'], 'slow': optimal_params[
                    'slow_period'], 'signal': optimal_params['signal_period']}
                self.optimization_history['last_optimization'] = datetime.now(
                    ).isoformat()
                self.optimization_history['optimization_count'] += 1
                self.optimization_history['performance_improvements'].append({
                    'timestamp': datetime.now().isoformat(), 'symbol':
                    symbol, 'timeframe': timeframe, 'regime': str(
                    current_regime), 'performance_gain': performance_gain,
                    'parameters': optimal_params})
                await self._adapt_parameters_to_regime(current_regime)
                return {'optimized': True, 'message':
                    f'Parameters optimized with {performance_gain:.2%} improvement'
                    , 'new_parameters': optimal_params}
            else:
                return {'optimized': False, 'message':
                    f'No significant improvement found ({performance_gain:.2%})'
                    }
        except Exception as e:
            self.logger.error(f'Error during parameter optimization: {str(e)}',
                exc_info=True)
            return {'optimized': False, 'error': str(e)}

    async def _adapt_parameters_to_regime(self, regime: MarketRegime) ->None:
        """Adjust strategy parameters based on the current market regime."""
        self.logger.info(f'Adapting parameters to {regime} regime')
        adapt_coef = self.adaptivity_coefficients.get(self.config[
            'adaptivity_level'], 0.5)
        regime_periods = self.regime_ma_periods.get(regime, self.
            regime_ma_periods[MarketRegime.TRENDING])
        new_fast_period = int(self.adaptive_params['base_fast_ma_period'] *
            (1 - adapt_coef) + regime_periods['fast'] * adapt_coef)
        new_slow_period = int(self.adaptive_params['base_slow_ma_period'] *
            (1 - adapt_coef) + regime_periods['slow'] * adapt_coef)
        new_signal_period = int(self.adaptive_params[
            'base_signal_ma_period'] * (1 - adapt_coef) + regime_periods[
            'signal'] * adapt_coef)
        new_fast_period = max(3, new_fast_period)
        new_slow_period = max(new_fast_period + 4, new_slow_period)
        new_signal_period = max(2, new_signal_period)
        self.adaptive_params['fast_ma_period'] = new_fast_period
        self.adaptive_params['slow_ma_period'] = new_slow_period
        self.adaptive_params['signal_ma_period'] = new_signal_period
        if self.config['auto_correlation_filter']:
            await self._apply_autocorrelation_tuning()
        if regime == MarketRegime.TRENDING:
            self.adaptive_params['atr_multiple_tp'] = 3.0
            self.adaptive_params['ma_channel_multiplier'] = 1.2
            self.adaptive_params['confirmation_bars'] = 2
            self.config['use_ma_crossover'] = True
        elif regime == MarketRegime.RANGING:
            self.adaptive_params['atr_multiple_tp'] = 1.5
            self.adaptive_params['ma_channel_multiplier'] = 0.8
            self.adaptive_params['confirmation_bars'] = 3
            self.config['use_ma_zone_reversals'] = True
        elif regime == MarketRegime.VOLATILE:
            self.adaptive_params['atr_multiple_sl'] = 2.0
            self.adaptive_params['ma_channel_multiplier'] = 1.5
            self.adaptive_params['confirmation_bars'] = 3
            self.config['use_additional_filters'] = True
        elif regime == MarketRegime.BREAKOUT:
            self.adaptive_params['atr_multiple_tp'] = 2.8
            self.adaptive_params['confirmation_bars'] = 1
            self.config['use_ma_channels'] = True
        self.logger.info(
            f'Adapted MA periods - Fast: {new_fast_period}, Slow: {new_slow_period}, Signal: {new_signal_period}'
            )

    @async_with_exception_handling
    async def _apply_autocorrelation_tuning(self) ->None:
        """Apply auto-correlation based tuning to parameters.

        This method analyzes recent price action to detect autocorrelation
        and adjusts MA periods accordingly for better responsiveness.
        """
        try:
            primary_data = self.get_recent_price_data(self.primary_timeframe)
            if primary_data is None or len(primary_data) < 50:
                self.logger.warning(
                    'Insufficient data for autocorrelation analysis')
                return
            if self.config.get('use_feature_store', False
                ) and self.feature_store_client:
                try:
                    symbol = primary_data.get('symbol', 'unknown')
                    if isinstance(symbol, pd.Series):
                        symbol = symbol.iloc[0]
                    start_date = primary_data.index[0
                        ] if not primary_data.empty else datetime.now(
                        ) - timedelta(days=30)
                    end_date = primary_data.index[-1
                        ] if not primary_data.empty else datetime.now()
                    indicators = (await self.feature_store_client.
                        get_indicators(symbol=symbol, start_date=start_date,
                        end_date=end_date, timeframe=self.primary_timeframe,
                        indicators=[
                        f"sma_{self.adaptive_params['base_fast_ma_period']}",
                        f"sma_{self.adaptive_params['base_slow_ma_period']}"]))
                    if not indicators.empty:
                        self.logger.info(
                            f'Retrieved {len(indicators)} indicator records from feature store'
                            )
                        indicators.set_index('timestamp', inplace=True)
                        for col in indicators.columns:
                            primary_data[col] = indicators[col]
                except Exception as e:
                    self.logger.warning(
                        f'Error getting indicators from feature store: {str(e)}, proceeding with direct calculation'
                        )
            returns = primary_data['close'].pct_change().dropna()
            autocorr = {}
            for lag in range(1, 11):
                joined_df = pd.DataFrame({'returns': returns,
                    f'returns_lag_{lag}': returns.shift(lag)}).dropna()
                if len(joined_df) > 30:
                    correlation = joined_df['returns'].corr(joined_df[
                        f'returns_lag_{lag}'])
                    autocorr[lag] = correlation
            if autocorr:
                max_autocorr_lag = max(autocorr.items(), key=lambda x: abs(
                    x[1]))
                significant_lag = max_autocorr_lag[0]
                correlation_strength = abs(max_autocorr_lag[1])
                if correlation_strength > 0.2:
                    self.logger.info(
                        f'Significant autocorrelation detected at lag {significant_lag} with strength {correlation_strength:.3f}'
                        )
                    adjustment_factor = min(1.5, 0.8 + correlation_strength)
                    if significant_lag <= 3:
                        self.adaptive_params['base_fast_ma_period'] = max(3,
                            int(self.adaptive_params['base_fast_ma_period'] /
                            adjustment_factor))
                        self.adaptive_params['base_slow_ma_period'] = max(8,
                            int(self.adaptive_params['base_slow_ma_period'] /
                            adjustment_factor))
                        self.adaptive_params['base_signal_ma_period'] = max(
                            2, int(self.adaptive_params[
                            'base_signal_ma_period'] / adjustment_factor))
                        self.logger.info(
                            f"Reduced MA periods based on short-term autocorrelation: fast={self.adaptive_params['base_fast_ma_period']}, slow={self.adaptive_params['base_slow_ma_period']}"
                            )
                    elif significant_lag >= 8:
                        self.adaptive_params['base_fast_ma_period'] = min(
                            24, int(self.adaptive_params[
                            'base_fast_ma_period'] * adjustment_factor))
                        self.adaptive_params['base_slow_ma_period'] = min(
                            50, int(self.adaptive_params[
                            'base_slow_ma_period'] * adjustment_factor))
                        self.adaptive_params['base_signal_ma_period'] = min(
                            18, int(self.adaptive_params[
                            'base_signal_ma_period'] * adjustment_factor))
                        self.logger.info(
                            f"Increased MA periods based on long-term autocorrelation: fast={self.adaptive_params['base_fast_ma_period']}, slow={self.adaptive_params['base_slow_ma_period']}"
                            )
                    self.optimization_history['autocorr_adjustments'
                        ] = self.optimization_history.get(
                        'autocorr_adjustments', []) + [{'timestamp':
                        datetime.now().isoformat(), 'lag': significant_lag,
                        'correlation': correlation_strength,
                        'adjustment_factor': adjustment_factor,
                        'fast_ma_period': self.adaptive_params[
                        'base_fast_ma_period'], 'slow_ma_period': self.
                        adaptive_params['base_slow_ma_period'],
                        'signal_ma_period': self.adaptive_params[
                        'base_signal_ma_period']}]
        except Exception as e:
            self.logger.error(f'Error in autocorrelation tuning: {str(e)}',
                exc_info=True)

    @with_exception_handling
    def analyze_performance(self, trade_results: List[Dict[str, Any]]) ->Dict[
        str, Any]:
        """
        Analyze strategy performance based on recent trades.

        Args:
            trade_results: List of recent trade results

        Returns:
            Dictionary with performance metrics and potential improvements
        """
        if not trade_results:
            return {'message': 'No trade data to analyze'}
        try:
            win_trades = [t for t in trade_results if t.get('result') == 'win']
            loss_trades = [t for t in trade_results if t.get('result') ==
                'loss']
            win_count = len(win_trades)
            loss_count = len(loss_trades)
            total_trades = win_count + loss_count
            if total_trades == 0:
                return {'message': 'No completed trades to analyze'}
            win_rate = win_count / total_trades
            avg_profit = sum(t.get('profit_pips', 0) for t in win_trades
                ) / max(1, win_count)
            avg_loss = sum(t.get('loss_pips', 0) for t in loss_trades) / max(
                1, loss_count)
            if avg_loss == 0:
                profit_factor = float('inf')
            else:
                profit_factor = avg_profit * win_count / (avg_loss * loss_count
                    ) if loss_count > 0 else float('inf')
            win_durations = [t.get('duration_minutes', 0) for t in win_trades]
            loss_durations = [t.get('duration_minutes', 0) for t in loss_trades
                ]
            avg_win_duration = sum(win_durations) / max(1, len(win_durations))
            avg_loss_duration = sum(loss_durations) / max(1, len(
                loss_durations))
            win_regimes = {}
            loss_regimes = {}
            for trade in win_trades:
                regime = trade.get('market_regime', 'unknown')
                win_regimes[regime] = win_regimes.get(regime, 0) + 1
            for trade in loss_trades:
                regime = trade.get('market_regime', 'unknown')
                loss_regimes[regime] = loss_regimes.get(regime, 0) + 1
            regime_effectiveness = {}
            for regime in set(list(win_regimes.keys()) + list(loss_regimes.
                keys())):
                wins = win_regimes.get(regime, 0)
                losses = loss_regimes.get(regime, 0)
                total = wins + losses
                if total > 0:
                    regime_effectiveness[regime] = {'win_rate': wins /
                        total, 'trade_count': total, 'wins': wins, 'losses':
                        losses}
            improvements = []
            for regime, stats in regime_effectiveness.items():
                if stats['win_rate'] < 0.4 and stats['trade_count'] >= 5:
                    improvements.append({'type': 'regime_performance',
                        'regime': regime, 'message':
                        f"Poor performance in {regime} regime (win rate: {stats['win_rate']:.2%})"
                        , 'suggestion':
                        'Consider modifying parameters for this regime or filtering trades'
                        })
            if avg_loss_duration < avg_win_duration * 0.5 and loss_count >= 3:
                improvements.append({'type': 'trade_duration', 'message':
                    'Loss trades are closing too quickly compared to winners',
                    'suggestion':
                    'Consider increasing stop loss distance or using time-based filters'
                    })
            if win_rate < 0.45 and total_trades >= 10:
                improvements.append({'type': 'win_rate', 'message':
                    f'Low win rate ({win_rate:.2%}) detected', 'suggestion':
                    'Consider optimizing parameters or adding stronger entry filters'
                    })
            return {'metrics': {'win_rate': win_rate, 'profit_factor':
                profit_factor, 'avg_profit': avg_profit, 'avg_loss':
                avg_loss, 'avg_win_duration': avg_win_duration,
                'avg_loss_duration': avg_loss_duration, 'total_trades':
                total_trades, 'regime_effectiveness': regime_effectiveness},
                'improvements': improvements}
        except Exception as e:
            self.logger.error(f'Error analyzing performance: {str(e)}',
                exc_info=True)
            return {'error': str(e)}

    @async_with_exception_handling
    async def cleanup(self) ->None:
        """Clean up resources used by the strategy."""
        try:
            if hasattr(self, 'feature_store_client'
                ) and self.feature_store_client:
                await self.feature_store_client.close()
                self.logger.info('Closed feature store client connection')
        except Exception as e:
            self.logger.error(f'Error during cleanup: {str(e)}', exc_info=True)
