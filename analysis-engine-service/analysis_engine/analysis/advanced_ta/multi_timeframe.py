"""
Multi-Timeframe Analyzer Module

This module analyzes trading indicators across multiple timeframes to identify
trend alignment, confirm signals, and generate more reliable trading decisions.
It helps reduce false signals by ensuring that trading decisions are supported
across different time horizons.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from analysis_engine.core.monitoring.async_performance_monitor import track_async_function, track_async_operation
from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.market_data import MarketData
from analysis_engine.models.analysis_result import AnalysisResult
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeframeAlignment(Enum):
    """Enumeration of timeframe alignment states"""
    STRONGLY_ALIGNED_BULLISH = 'strongly_aligned_bullish'
    WEAKLY_ALIGNED_BULLISH = 'weakly_aligned_bullish'
    NEUTRAL = 'neutral'
    WEAKLY_ALIGNED_BEARISH = 'weakly_aligned_bearish'
    STRONGLY_ALIGNED_BEARISH = 'strongly_aligned_bearish'
    CONFLICTING = 'conflicting'


class MultiTimeframeAnalyzer(BaseAnalyzer):
    """
    Analyzer for multi-timeframe technical analysis.

    Processes pre-calculated technical indicators across multiple timeframes
    to identify trend alignment, confirm signals, and generate more reliable
    trading decisions. Assumes input MarketData contains necessary indicator columns.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the Multi-Timeframe Analyzer

        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {'timeframes': ['1m', '5m', '15m', '1h', '4h',
            '1d'], 'primary_timeframe': '1h', 'price_column': 'close',
            'ma_periods': [20, 50, 200], 'use_ema': True, 'use_macd': True,
            'use_rsi': True, 'use_stochastic': False, 'use_atr': True,
            'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'stoch_period': 14, 'stoch_smooth_k': 3, 'stoch_smooth_d': 3,
            'atr_period': 14, 'alignment_threshold': 0.7,
            'weak_alignment_threshold': 0.5, 'min_data_points_per_tf': 50}
        super().__init__(name='multi_timeframe', parameters={**
            default_params, **parameters or {}})
        self.timeframe_cache = {}
        self.last_full_calculation = None

    @track_async_function
    @async_with_exception_handling
    async def analyze(self, data: Dict[str, MarketData]) ->AnalysisResult:
        """
        Analyze technical indicators across multiple timeframes using pre-calculated data.

        Args:
            data: Dictionary mapping timeframe strings to MarketData objects.
                  Each MarketData object's `data` DataFrame must contain OHLCV
                  and the required pre-calculated indicator columns.

        Returns:
            Analysis results including trend alignment and signal confirmation.
        """
        if not data or not isinstance(data, dict):
            logger.warning('Invalid data format for multi-timeframe analysis')
            return AnalysisResult(analyzer_name=self.name, result_data={
                'error': 'Invalid data format for multi-timeframe analysis'
                }, is_valid=False)
        required_timeframes = set(self.parameters['timeframes'])
        available_timeframes = set(data.keys())
        if not required_timeframes.issubset(available_timeframes):
            missing = required_timeframes - available_timeframes
            logger.warning(f'Missing data for timeframes: {missing}')
            if self.parameters['primary_timeframe'] in missing:
                return AnalysisResult(analyzer_name=self.name, result_data=
                    {'error':
                    f"Missing data for primary timeframe: {self.parameters['primary_timeframe']}"
                    }, is_valid=False)
        try:
            timeframe_analysis = {}
            min_required_rows = self.parameters['min_data_points_per_tf']
            for tf in self.parameters['timeframes']:
                if tf in data:
                    market_data_obj = data[tf]
                    if not isinstance(market_data_obj, MarketData
                        ) or market_data_obj.data is None:
                        logger.warning(
                            f'Invalid MarketData object for timeframe {tf}, skipping.'
                            )
                        continue
                    tf_data = market_data_obj.data
                    if len(tf_data) < min_required_rows:
                        logger.warning(
                            f'Insufficient data rows for timeframe {tf} (needed {min_required_rows}, got {len(tf_data)}), skipping.'
                            )
                        continue
                    async with track_async_operation(
                        f'multi_timeframe.analyze_timeframe.{tf}'):
                        analysis = self._analyze_timeframe(tf_data, tf)
                        if analysis:
                            timeframe_analysis[tf] = analysis
                        else:
                            logger.warning(
                                f'Analysis failed for timeframe {tf}, likely due to missing indicator columns.'
                                )
            if not timeframe_analysis:
                return AnalysisResult(analyzer_name=self.name, result_data=
                    {'error':
                    'Could not analyze any timeframes due to insufficient data or missing indicators'
                    }, is_valid=False)
            alignment_results = self._calculate_alignment(timeframe_analysis)
            signal_confirmations = self._generate_signal_confirmations(
                timeframe_analysis)
            self.timeframe_cache = {'timeframe_analysis':
                timeframe_analysis, 'last_updated': datetime.now()}
            self.last_full_calculation = datetime.now()
            result_data = {'alignment': alignment_results,
                'signal_confirmations': signal_confirmations,
                'timeframe_analysis': {tf: {k: (float(v) if isinstance(v, (
                np.float32, np.float64)) else v) for k, v in analysis.items
                () if k != 'raw_data'} for tf, analysis in
                timeframe_analysis.items()}, 'primary_timeframe': self.
                parameters['primary_timeframe'], 'timestamp': datetime.now(
                ).isoformat()}
            return AnalysisResult(analyzer_name=self.name, result_data=
                result_data, is_valid=True)
        except Exception as e:
            logger.error(f'Error in multi-timeframe analysis: {e}',
                exc_info=True)
            return AnalysisResult(analyzer_name=self.name, result_data={
                'error': f'Error in multi-timeframe analysis: {str(e)}'},
                is_valid=False)

    @track_async_function
    async def update_incremental(self, data: Dict[str, MarketData],
        previous_result: AnalysisResult) ->AnalysisResult:
        """
        Update multi-timeframe analysis incrementally with new data.
        Currently falls back to full recalculation.

        Args:
            data: Dictionary mapping timeframe strings to market data
            previous_result: Results from previous analysis

        Returns:
            Updated analysis results
        """
        logger.debug(
            'Incremental update requested for MultiTimeframeAnalyzer, performing full recalculation.'
            )
        return await self.analyze(data)

    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) ->Optional[
        Dict[str, Any]]:
        """
        Analyze technical indicators for a single timeframe using pre-calculated values
        present in the input DataFrame.

        Args:
            df: DataFrame with OHLCV data and pre-calculated indicators.
            timeframe: Timeframe string (e.g., "1h", "4h").

        Returns:
            Dictionary with analysis results for this timeframe, or None if essential
            indicators are missing.
        """
        if self.parameters['price_column'] not in df.columns:
            logger.error(
                f"Price column '{self.parameters['price_column']}' not found in DataFrame for timeframe {timeframe}"
                )
            return None
        price_series = df[self.parameters['price_column']]
        if price_series.empty or price_series.isna().all():
            logger.warning(
                f"Price column '{self.parameters['price_column']}' is empty or all NaN for timeframe {timeframe}"
                )
            return None
        current_price = price_series.iloc[-1]
        result = {'timeframe': timeframe}
        has_essential_data = True
        ma_type_prefix = 'ema' if self.parameters['use_ema'] else 'sma'
        for period in self.parameters['ma_periods']:
            ma_name = f'{ma_type_prefix}_{period}'
            if ma_name in df.columns and not df[ma_name].isna().all():
                ma_series = df[ma_name]
                result[ma_name] = ma_series.iloc[-1]
                valid_ma = ma_series.dropna()
                if len(valid_ma) >= 6:
                    if pd.notna(ma_series.iloc[-1]) and pd.notna(ma_series.
                        iloc[-6]) and ma_series.iloc[-6] != 0:
                        ma_slope = (ma_series.iloc[-1] - ma_series.iloc[-6]
                            ) / ma_series.iloc[-6]
                        result[f'{ma_name}_slope'] = ma_slope
                    else:
                        result[f'{ma_name}_slope'] = 0.0
                else:
                    result[f'{ma_name}_slope'] = 0.0
            else:
                logger.debug(
                    f"MA column '{ma_name}' not found or all NaN in DataFrame for timeframe {timeframe}"
                    )
                result[ma_name] = None
                result[f'{ma_name}_slope'] = None
        if self.parameters['use_rsi']:
            rsi_period = self.parameters['rsi_period']
            rsi_col = f'rsi_{rsi_period}'
            if rsi_col in df.columns and not df[rsi_col].isna().all():
                rsi = df[rsi_col]
                current_rsi = rsi.iloc[-1]
                result['rsi'] = current_rsi
                result['is_overbought'] = current_rsi > self.parameters[
                    'rsi_overbought']
                result['is_oversold'] = current_rsi < self.parameters[
                    'rsi_oversold']
                valid_rsi = rsi.dropna()
                if len(valid_rsi) >= 4:
                    if pd.notna(rsi.iloc[-1]) and pd.notna(rsi.iloc[-4]):
                        rsi_slope = rsi.iloc[-1] - rsi.iloc[-4]
                        result['rsi_slope'] = rsi_slope
                    else:
                        result['rsi_slope'] = 0.0
                else:
                    result['rsi_slope'] = 0.0
            else:
                logger.debug(
                    f"RSI column '{rsi_col}' not found or all NaN in DataFrame for timeframe {timeframe}"
                    )
                result['rsi'] = None
                result['is_overbought'] = None
                result['is_oversold'] = None
                result['rsi_slope'] = None
                has_essential_data = False
        if self.parameters['use_macd']:
            macd_line_col = 'macd_line'
            macd_signal_col = 'macd_signal'
            macd_hist_col = 'macd_histogram'
            required_macd_cols = [macd_line_col, macd_signal_col, macd_hist_col
                ]
            if all(col in df.columns and not df[col].isna().all() for col in
                required_macd_cols):
                macd_line = df[macd_line_col]
                macd_signal = df[macd_signal_col]
                macd_hist = df[macd_hist_col]
                result['macd'] = macd_line.iloc[-1]
                result['macd_signal'] = macd_signal.iloc[-1]
                result['macd_histogram'] = macd_hist.iloc[-1]
                valid_hist = macd_hist.dropna()
                if len(valid_hist) >= 2:
                    if pd.notna(macd_hist.iloc[-1]) and pd.notna(macd_hist.
                        iloc[-2]):
                        result['macd_bullish_cross'] = macd_hist.iloc[-2
                            ] <= 0 and macd_hist.iloc[-1] > 0
                        result['macd_bearish_cross'] = macd_hist.iloc[-2
                            ] >= 0 and macd_hist.iloc[-1] < 0
                        result['macd_histogram_direction'
                            ] = 1 if macd_hist.iloc[-1] > macd_hist.iloc[-2
                            ] else -1 if macd_hist.iloc[-1] < macd_hist.iloc[-2
                            ] else 0
                    else:
                        result['macd_bullish_cross'] = False
                        result['macd_bearish_cross'] = False
                        result['macd_histogram_direction'] = 0
                else:
                    result['macd_bullish_cross'] = False
                    result['macd_bearish_cross'] = False
                    result['macd_histogram_direction'] = 0
            else:
                missing_cols = [col for col in required_macd_cols if col not in
                    df.columns or df[col].isna().all()]
                logger.debug(
                    f'MACD columns {missing_cols} not found or all NaN in DataFrame for timeframe {timeframe}'
                    )
                result['macd'] = None
                result['macd_signal'] = None
                result['macd_histogram'] = None
                result['macd_bullish_cross'] = None
                result['macd_bearish_cross'] = None
                result['macd_histogram_direction'] = None
                has_essential_data = False
        if self.parameters['use_stochastic']:
            k_period = self.parameters['stoch_period']
            smooth_k = self.parameters['stoch_smooth_k']
            smooth_d = self.parameters['stoch_smooth_d']
            stoch_k_col = f'stoch_k_{k_period}_{smooth_k}_{smooth_d}'
            stoch_d_col = f'stoch_d_{k_period}_{smooth_k}_{smooth_d}'
            required_stoch_cols = [stoch_k_col, stoch_d_col]
            if all(col in df.columns and not df[col].isna().all() for col in
                required_stoch_cols):
                stoch_k = df[stoch_k_col]
                stoch_d = df[stoch_d_col]
                result['stoch_k'] = stoch_k.iloc[-1]
                result['stoch_d'] = stoch_d.iloc[-1]
                result['stoch_overbought'] = stoch_k.iloc[-1] > 80
                result['stoch_oversold'] = stoch_k.iloc[-1] < 20
                valid_k = stoch_k.dropna()
                valid_d = stoch_d.dropna()
                if len(valid_k) >= 2 and len(valid_d) >= 2:
                    if pd.notna(stoch_k.iloc[-1]) and pd.notna(stoch_k.iloc[-2]
                        ) and pd.notna(stoch_d.iloc[-1]) and pd.notna(stoch_d
                        .iloc[-2]):
                        result['stoch_bullish_cross'] = stoch_k.iloc[-2
                            ] <= stoch_d.iloc[-2] and stoch_k.iloc[-1
                            ] > stoch_d.iloc[-1]
                        result['stoch_bearish_cross'] = stoch_k.iloc[-2
                            ] >= stoch_d.iloc[-2] and stoch_k.iloc[-1
                            ] < stoch_d.iloc[-1]
                    else:
                        result['stoch_bullish_cross'] = False
                        result['stoch_bearish_cross'] = False
                else:
                    result['stoch_bullish_cross'] = False
                    result['stoch_bearish_cross'] = False
            else:
                missing_cols = [col for col in required_stoch_cols if col
                     not in df.columns or df[col].isna().all()]
                logger.debug(
                    f'Stochastic columns {missing_cols} not found or all NaN in DataFrame for timeframe {timeframe}'
                    )
                result['stoch_k'] = None
                result['stoch_d'] = None
                result['stoch_overbought'] = None
                result['stoch_oversold'] = None
                result['stoch_bullish_cross'] = None
                result['stoch_bearish_cross'] = None
        if self.parameters['use_atr']:
            atr_period = self.parameters['atr_period']
            atr_col = f'atr_{atr_period}'
            if atr_col in df.columns and not df[atr_col].isna().all():
                atr = df[atr_col]
                current_atr = atr.iloc[-1]
                result['atr'] = current_atr
                if (current_price is not None and current_price != 0 and pd
                    .notna(current_atr)):
                    result['atr_percent'] = current_atr / current_price * 100
                else:
                    result['atr_percent'] = 0.0
            else:
                logger.debug(
                    f"ATR column '{atr_col}' not found or all NaN in DataFrame for timeframe {timeframe}"
                    )
                result['atr'] = None
                result['atr_percent'] = None
        if has_essential_data:
            trend_results = self._determine_trend(df, result, current_price)
            if trend_results:
                result.update(trend_results)
            else:
                logger.warning(
                    f'Could not determine trend for timeframe {timeframe} due to missing indicators.'
                    )
                return None
        else:
            logger.warning(
                f'Skipping trend determination for timeframe {timeframe} due to missing essential indicators (RSI/MACD).'
                )
            return None
        return result

    def _determine_trend(self, df: pd.DataFrame, indicators: Dict[str, Any],
        current_price: float) ->Optional[Dict[str, Any]]:
        """
        Determine trend based on pre-read indicators.

        Args:
            df: DataFrame (used for MA cross checks).
            indicators: Dictionary of pre-read indicators for this timeframe.
            current_price: The current price for comparison.

        Returns:
            Dictionary with trend determination, or None if critical indicators are missing.
        """
        result = {}
        ma_trends = []
        ma_type_prefix = 'ema' if self.parameters['use_ema'] else 'sma'
        sorted_periods = sorted(self.parameters['ma_periods'])
        has_ma_data = False
        for period in sorted_periods:
            ma_key = f'{ma_type_prefix}_{period}'
            ma_value = indicators.get(ma_key)
            ma_slope = indicators.get(f'{ma_key}_slope')
            if ma_value is not None and ma_slope is not None and pd.notna(
                ma_value) and pd.notna(ma_slope):
                has_ma_data = True
                if current_price > ma_value and ma_slope > 0:
                    ma_trends.append(1)
                elif current_price > ma_value and ma_slope <= 0:
                    ma_trends.append(0.5)
                elif current_price < ma_value and ma_slope < 0:
                    ma_trends.append(-1)
                elif current_price < ma_value and ma_slope >= 0:
                    ma_trends.append(-0.5)
                else:
                    ma_trends.append(0)
            else:
                ma_trends.append(0)
        if has_ma_data and ma_trends:
            ma_trend_score = sum(ma_trends) / len([t for t in ma_trends if 
                t != 0 or any(indicators.get(f'{ma_type_prefix}_{p}') is not
                None for p in sorted_periods)])
            result['ma_trend_score'] = ma_trend_score
            if ma_trend_score > 0.7:
                result['ma_trend'] = 'strong_bullish'
            elif ma_trend_score > 0.3:
                result['ma_trend'] = 'bullish'
            elif ma_trend_score < -0.7:
                result['ma_trend'] = 'strong_bearish'
            elif ma_trend_score < -0.3:
                result['ma_trend'] = 'bearish'
            else:
                result['ma_trend'] = 'neutral'
        else:
            logger.warning(
                f"Cannot determine MA trend for timeframe {indicators.get('timeframe')} - no valid MA data found."
                )
            return None
        ma_crosses = []
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            short_ma_col = f'{ma_type_prefix}_{short_period}'
            long_ma_col = f'{ma_type_prefix}_{long_period}'
            if short_ma_col in df.columns and long_ma_col in df.columns:
                short_ma_series = df[short_ma_col]
                long_ma_series = df[long_ma_col]
                valid_short = short_ma_series.dropna()
                valid_long = long_ma_series.dropna()
                if len(valid_short) >= 2 and len(valid_long) >= 2:
                    if pd.notna(short_ma_series.iloc[-1]) and pd.notna(
                        long_ma_series.iloc[-1]) and pd.notna(short_ma_series
                        .iloc[-2]) and pd.notna(long_ma_series.iloc[-2]):
                        prev_diff = short_ma_series.iloc[-2
                            ] - long_ma_series.iloc[-2]
                        curr_diff = short_ma_series.iloc[-1
                            ] - long_ma_series.iloc[-1]
                        if prev_diff <= 0 and curr_diff > 0:
                            ma_crosses.append({'type': 'bullish', 'short':
                                short_period, 'long': long_period})
                        elif prev_diff >= 0 and curr_diff < 0:
                            ma_crosses.append({'type': 'bearish', 'short':
                                short_period, 'long': long_period})
        result['ma_crosses'] = ma_crosses
        trend_signals = []
        trend_signals.append(result['ma_trend_score'])
        if self.parameters['use_rsi']:
            rsi = indicators.get('rsi')
            rsi_slope = indicators.get('rsi_slope')
            if rsi is not None and pd.notna(rsi):
                if rsi > 60:
                    trend_signals.append(0.5)
                elif rsi < 40:
                    trend_signals.append(-0.5)
                if rsi_slope is not None and pd.notna(rsi_slope):
                    if rsi_slope > 0:
                        trend_signals.append(0.3)
                    elif rsi_slope < 0:
                        trend_signals.append(-0.3)
            else:
                pass
        if self.parameters['use_macd']:
            macd_hist = indicators.get('macd_histogram')
            macd_hist_dir = indicators.get('macd_histogram_direction')
            if macd_hist is not None and pd.notna(macd_hist):
                if macd_hist > 0:
                    trend_signals.append(0.5)
                elif macd_hist < 0:
                    trend_signals.append(-0.5)
                if macd_hist_dir is not None and pd.notna(macd_hist_dir):
                    trend_signals.append(macd_hist_dir * 0.3)
            else:
                pass
        if trend_signals:
            valid_signals = [s for s in trend_signals if s is not None and
                pd.notna(s)]
            if not valid_signals:
                logger.warning(
                    f"No valid trend signals found for timeframe {indicators.get('timeframe')}"
                    )
                return None
            overall_trend_score = sum(valid_signals) / len(valid_signals)
            result['overall_trend_score'] = overall_trend_score
            if overall_trend_score > 0.6:
                result['overall_trend'] = 'strong_bullish'
            elif overall_trend_score > 0.2:
                result['overall_trend'] = 'bullish'
            elif overall_trend_score < -0.6:
                result['overall_trend'] = 'strong_bearish'
            elif overall_trend_score < -0.2:
                result['overall_trend'] = 'bearish'
            else:
                result['overall_trend'] = 'neutral'
        else:
            logger.warning(
                f"No trend signals generated for timeframe {indicators.get('timeframe')}"
                )
            return None
        return result

    def _calculate_alignment(self, timeframe_analysis: Dict[str, Dict[str,
        Any]]) ->Dict[str, Any]:
        """
        Calculate trend alignment across timeframes based on their determined trends.

        Args:
            timeframe_analysis: Dictionary mapping timeframes to their analysis results.

        Returns:
            Dictionary with alignment analysis.
        """
        if not timeframe_analysis:
            return {'alignment': 'insufficient_data', 'error':
                'No valid timeframe analysis provided.'}
        trend_counts = {'strong_bullish': 0, 'bullish': 0, 'neutral': 0,
            'bearish': 0, 'strong_bearish': 0}
        valid_tf_count = 0
        for tf, analysis in timeframe_analysis.items():
            if analysis and 'overall_trend' in analysis and analysis[
                'overall_trend'] is not None:
                trend = analysis['overall_trend']
                if trend in trend_counts:
                    trend_counts[trend] += 1
                    valid_tf_count += 1
                else:
                    trend_counts['neutral'] += 1
                    valid_tf_count += 1
        if valid_tf_count == 0:
            return {'alignment': 'insufficient_data', 'error':
                'No timeframes had a valid overall trend.'}
        bullish_percent = (trend_counts['strong_bullish'] + trend_counts[
            'bullish']) / valid_tf_count
        bearish_percent = (trend_counts['strong_bearish'] + trend_counts[
            'bearish']) / valid_tf_count
        neutral_percent = trend_counts['neutral'] / valid_tf_count
        primary_tf = self.parameters['primary_timeframe']
        primary_trend = 'neutral'
        primary_analysis = timeframe_analysis.get(primary_tf)
        if (primary_analysis and 'overall_trend' in primary_analysis and 
            primary_analysis['overall_trend'] is not None):
            primary_trend = primary_analysis['overall_trend']
        else:
            logger.warning(
                f"Primary timeframe '{primary_tf}' analysis or trend is missing for alignment calculation."
                )
        alignment = TimeframeAlignment.NEUTRAL.value
        strong_thresh = self.parameters['alignment_threshold']
        weak_thresh = self.parameters['weak_alignment_threshold']
        if bullish_percent >= weak_thresh and bearish_percent >= weak_thresh:
            alignment = TimeframeAlignment.CONFLICTING.value
        elif bullish_percent >= strong_thresh:
            alignment = TimeframeAlignment.STRONGLY_ALIGNED_BULLISH.value
        elif bearish_percent >= strong_thresh:
            alignment = TimeframeAlignment.STRONGLY_ALIGNED_BEARISH.value
        elif bullish_percent >= weak_thresh:
            alignment = TimeframeAlignment.WEAKLY_ALIGNED_BULLISH.value
        elif bearish_percent >= weak_thresh:
            alignment = TimeframeAlignment.WEAKLY_ALIGNED_BEARISH.value
        return {'alignment': alignment, 'bullish_percent': round(
            bullish_percent, 4), 'bearish_percent': round(bearish_percent, 
            4), 'neutral_percent': round(neutral_percent, 4),
            'trend_counts': trend_counts, 'primary_trend': primary_trend,
            'alignment_strength': round(max(bullish_percent,
            bearish_percent), 4), 'valid_timeframes_count': valid_tf_count}

    def _generate_signal_confirmations(self, timeframe_analysis: Dict[str,
        Dict[str, Any]]) ->Dict[str, Any]:
        """
        Generate signal confirmations based on primary timeframe signals and higher timeframe trends.

        Args:
            timeframe_analysis: Dictionary mapping timeframes to their analysis results.

        Returns:
            Dictionary with signal confirmations.
        """
        primary_tf = self.parameters['primary_timeframe']
        if primary_tf not in timeframe_analysis or timeframe_analysis[
            primary_tf] is None:
            return {'error':
                f"Primary timeframe '{primary_tf}' analysis not available for signal confirmation."
                }
        primary_analysis = timeframe_analysis[primary_tf]
        signals = {'buy_signals': [], 'sell_signals': [], 'exit_signals': [
            ], 'confirmed_signals': []}
        ma_crosses = primary_analysis.get('ma_crosses', [])
        if ma_crosses:
            for cross in ma_crosses:
                signal_base = {'type': 'ma_cross', 'timeframe': primary_tf,
                    'short_period': cross.get('short'), 'long_period':
                    cross.get('long')}
                if cross.get('type') == 'bullish':
                    signals['buy_signals'].append({**signal_base,
                        'description':
                        f"{cross.get('short')}-{cross.get('long')} MA bullish cross"
                        })
                elif cross.get('type') == 'bearish':
                    signals['sell_signals'].append({**signal_base,
                        'description':
                        f"{cross.get('short')}-{cross.get('long')} MA bearish cross"
                        })
        if self.parameters['use_macd']:
            if primary_analysis.get('macd_bullish_cross'):
                signals['buy_signals'].append({'type': 'macd_cross',
                    'timeframe': primary_tf, 'description':
                    'MACD bullish cross'})
            if primary_analysis.get('macd_bearish_cross'):
                signals['sell_signals'].append({'type': 'macd_cross',
                    'timeframe': primary_tf, 'description':
                    'MACD bearish cross'})
        if self.parameters['use_rsi']:
            if primary_analysis.get('is_oversold'):
                signals['buy_signals'].append({'type': 'rsi_oversold',
                    'timeframe': primary_tf, 'description':
                    f"RSI oversold ({primary_analysis.get('rsi', 'N/A'):.1f})"}
                    )
            if primary_analysis.get('is_overbought'):
                signals['sell_signals'].append({'type': 'rsi_overbought',
                    'timeframe': primary_tf, 'description':
                    f"RSI overbought ({primary_analysis.get('rsi', 'N/A'):.1f})"
                    })
        if self.parameters['use_stochastic']:
            if primary_analysis.get('stoch_bullish_cross'):
                signals['buy_signals'].append({'type': 'stoch_cross',
                    'timeframe': primary_tf, 'description':
                    'Stochastic bullish cross'})
            if primary_analysis.get('stoch_bearish_cross'):
                signals['sell_signals'].append({'type': 'stoch_cross',
                    'timeframe': primary_tf, 'description':
                    'Stochastic bearish cross'})
        confirmed_signals = []
        analyzed_timeframes = list(timeframe_analysis.keys())
        for buy_signal in signals['buy_signals']:
            confirmation_count = 0
            confirming_timeframes = []
            for tf in analyzed_timeframes:
                if tf == primary_tf or self._is_lower_timeframe(tf, primary_tf
                    ):
                    continue
                tf_analysis = timeframe_analysis.get(tf)
                if tf_analysis and 'overall_trend' in tf_analysis:
                    if tf_analysis['overall_trend'] in ['bullish',
                        'strong_bullish']:
                        confirmation_count += 1
                        confirming_timeframes.append(tf)
            if confirmation_count > 0:
                confirmed_signals.append({**buy_signal, 'direction': 'buy',
                    'confirmed_by': confirming_timeframes,
                    'confirmation_count': confirmation_count})
        for sell_signal in signals['sell_signals']:
            confirmation_count = 0
            confirming_timeframes = []
            for tf in analyzed_timeframes:
                if tf == primary_tf or self._is_lower_timeframe(tf, primary_tf
                    ):
                    continue
                tf_analysis = timeframe_analysis.get(tf)
                if tf_analysis and 'overall_trend' in tf_analysis:
                    if tf_analysis['overall_trend'] in ['bearish',
                        'strong_bearish']:
                        confirmation_count += 1
                        confirming_timeframes.append(tf)
            if confirmation_count > 0:
                confirmed_signals.append({**sell_signal, 'direction':
                    'sell', 'confirmed_by': confirming_timeframes,
                    'confirmation_count': confirmation_count})
        signals['confirmed_signals'] = confirmed_signals
        return signals

    def _is_lower_timeframe(self, tf1: str, tf2: str) ->bool:
        """
        Check if timeframe 1 is lower than timeframe 2 based on approximate duration.

        Args:
            tf1: First timeframe string (e.g., "1m", "1h", "1d").
            tf2: Second timeframe string.

        Returns:
            True if tf1 represents a shorter duration than tf2, False otherwise.
        """
        timeframe_order = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
            '4h': 240, '1d': 1440, '1w': 10080, '1mo': 43200}
        tf1_value = timeframe_order.get(tf1.lower(), 0)
        tf2_value = timeframe_order.get(tf2.lower(), 0)
        return tf1_value > 0 and tf2_value > 0 and tf1_value < tf2_value
