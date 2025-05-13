"""
Currency Correlation Analyzer Module

This module provides functionality for analyzing correlations between currency pairs,
which is essential for risk management and understanding cross-market relationships.
It includes tools for calculating correlation coefficients, visualizing correlation
matrices, and detecting correlation regime changes.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.market_data import MarketData
from analysis_engine.models.analysis_result import AnalysisResult
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int
    ) ->pd.Series:
    """
    Placeholder: Calculates rolling correlation between two series.
    """
    if series1.isnull().all() or series2.isnull().all():
        return pd.Series(np.nan, index=series1.index)
    aligned_s1, aligned_s2 = series1.align(series2, join='inner')
    if aligned_s1.empty or len(aligned_s1) < window:
        return pd.Series(np.nan, index=series1.index)
    rolling_corr = aligned_s1.rolling(window=window).corr(aligned_s2)
    return rolling_corr.reindex(series1.index)


class CurrencyCorrelationAnalyzer(BaseAnalyzer):
    """
    Analyzer for currency pair correlations.

    Calculates and tracks correlations between multiple currency pairs over time,
    using pre-calculated OHLCV data provided for each instrument.
    Helps identify diversification opportunities and risks from correlated positions.
    """
    DEFAULT_PARAMS = {'base_window': 20, 'long_window': 60,
        'correlation_threshold': 0.7, 'regime_change_threshold': 0.3,
        'update_frequency_hours': 4, 'min_data_points': 60, 'price_column':
        'close', 'instrument_list': [], 'calculate_rolling': True,
        'calculate_stability': True}

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the Currency Correlation Analyzer.

        Args:
            parameters: Configuration parameters, overriding defaults.
        """
        resolved_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            resolved_params.update(parameters)
        resolved_params['min_data_points'] = max(resolved_params[
            'min_data_points'], resolved_params['long_window'])
        super().__init__(name='currency_correlation', parameters=
            resolved_params)
        self.correlation_cache = {}
        self.last_full_calculation_time = None
        logger.info(
            f'Initialized CurrencyCorrelationAnalyzer with params: {self.parameters}'
            )

    @with_exception_handling
    def analyze(self, data: Dict[str, MarketData]) ->AnalysisResult:
        """Perform correlation analysis on provided currency pair data.

        Args:
            data: Dictionary mapping instrument symbols to their MarketData objects.
                  Each MarketData object must contain a DataFrame with OHLCV data.

        Returns:
            AnalysisResult containing correlation matrices, rolling correlations (optional),
            regime changes, strong correlations, and stability metrics (optional).
        """
        if not data or not isinstance(data, dict) or len(data) < 2:
            msg = (
                'Insufficient data for correlation analysis (need at least 2 instruments)'
                )
            logger.warning(msg)
            return AnalysisResult(analyzer_name=self.name, result={'error':
                msg}, is_valid=False)
        instruments = self.parameters['instrument_list'] or list(data.keys())
        price_col = self.parameters['price_column']
        min_points = self.parameters['min_data_points']
        price_series_dict = {}
        valid_instruments = []
        for instrument in instruments:
            if instrument not in data or not isinstance(data[instrument],
                MarketData) or data[instrument].data is None:
                logger.warning(
                    f'Instrument {instrument} missing or has invalid MarketData, skipping.'
                    )
                continue
            df = data[instrument].data
            if price_col not in df.columns:
                logger.warning(
                    f"Price column '{price_col}' not found for instrument {instrument}, skipping."
                    )
                continue
            if len(df) < min_points:
                logger.warning(
                    f'Insufficient data points for {instrument} (need {min_points}, got {len(df)}), skipping.'
                    )
                continue
            price_series = df[price_col].copy()
            if price_series.isna().all():
                logger.warning(
                    f'Price data for {instrument} is all NaN, skipping.')
                continue
            price_series_dict[instrument] = price_series
            valid_instruments.append(instrument)
        if len(valid_instruments) < 2:
            msg = (
                f'Insufficient valid instruments for correlation analysis after filtering (need at least 2, got {len(valid_instruments)})'
                )
            logger.warning(msg)
            return AnalysisResult(analyzer_name=self.name, result={'error':
                msg}, is_valid=False)
        try:
            prices_df = pd.DataFrame(price_series_dict).sort_index()
            prices_df = prices_df.ffill()
            prices_df = prices_df.bfill()
            all_nan_cols = prices_df.columns[prices_df.isna().all()].tolist()
            if all_nan_cols:
                logger.warning(
                    f'Instruments {all_nan_cols} are still all NaN after fill, removing from analysis.'
                    )
                prices_df = prices_df.drop(columns=all_nan_cols)
                valid_instruments = [inst for inst in valid_instruments if 
                    inst not in all_nan_cols]
                if len(valid_instruments) < 2:
                    msg = 'Insufficient valid instruments after NaN handling.'
                    logger.warning(msg)
                    return AnalysisResult(analyzer_name=self.name, result={
                        'error': msg}, is_valid=False)
            returns_df = prices_df.pct_change().dropna()
            if returns_df.empty or len(returns_df) < max(self.parameters[
                'base_window'], self.parameters['long_window']):
                msg = (
                    f"Insufficient returns data after calculation (need at least {max(self.parameters['base_window'], self.parameters['long_window'])} rows, got {len(returns_df)})"
                    )
                logger.warning(msg)
                return AnalysisResult(analyzer_name=self.name, result={
                    'error': msg}, is_valid=False)
        except Exception as e:
            logger.error(f'Error preparing price/returns DataFrame: {e}',
                exc_info=True)
            return AnalysisResult(analyzer_name=self.name, result={'error':
                f'Data preparation failed: {str(e)}'}, is_valid=False)
        try:
            base_window = self.parameters['base_window']
            long_window = self.parameters['long_window']
            base_corr_matrix = self._calculate_correlation_matrix(returns_df,
                base_window)
            long_corr_matrix = self._calculate_correlation_matrix(returns_df,
                long_window)
            rolling_corrs = {}
            if self.parameters['calculate_rolling']:
                rolling_corrs = self._calculate_rolling_correlations(returns_df
                    , base_window)
            regime_changes = self._detect_correlation_regime_changes(
                base_corr_matrix, long_corr_matrix)
            strong_correlations = self._identify_strong_correlations(
                base_corr_matrix)
            correlation_stability = {}
            if self.parameters['calculate_stability']:
                correlation_stability = self._calculate_correlation_stability(
                    returns_df)
        except Exception as e:
            logger.error(f'Error during correlation calculations: {e}',
                exc_info=True)
            return AnalysisResult(analyzer_name=self.name, result={'error':
                f'Calculation failed: {str(e)}'}, is_valid=False)
        current_time = datetime.now()
        self.correlation_cache = {'base_corr_matrix': base_corr_matrix,
            'long_corr_matrix': long_corr_matrix, 'rolling_correlations':
            rolling_corrs, 'last_returns_df': returns_df, 'last_updated':
            current_time}
        self.last_full_calculation_time = current_time
        result_data = {'base_correlation_matrix': base_corr_matrix.to_dict(
            ) if base_corr_matrix is not None else None,
            'long_correlation_matrix': long_corr_matrix.to_dict() if 
            long_corr_matrix is not None else None, 'rolling_correlations':
            {pair: (series.to_dict() if series is not None else None) for 
            pair, series in rolling_corrs.items()}, 'regime_changes':
            regime_changes, 'strong_correlations': strong_correlations,
            'correlation_stability': correlation_stability,
            'analyzed_instruments': valid_instruments, 'timestamp':
            current_time.isoformat()}
        return AnalysisResult(analyzer_name=self.name, result=result_data,
            is_valid=True)

    @with_resilience('update_incremental')
    def update_incremental(self, data: Dict[str, MarketData],
        previous_result: AnalysisResult) ->AnalysisResult:
        """Update correlation analysis incrementally with new data.
        Currently falls back to full recalculation.

        Args:
            data: Dictionary mapping instrument symbols to their market data.
            previous_result: Results from previous analysis (not currently used effectively).

        Returns:
            Updated analysis results (via full recalculation).
        """
        update_freq_hours = self.parameters['update_frequency_hours']
        if self.last_full_calculation_time is None or datetime.now(
            ) - self.last_full_calculation_time > timedelta(hours=
            update_freq_hours):
            logger.info(
                f'Performing full correlation recalculation (time since last > {update_freq_hours}h)'
                )
            return self.analyze(data)
        if set(data.keys()) != set(previous_result.result.get(
            'analyzed_instruments', [])):
            logger.info(
                'Instrument set changed, performing full correlation recalculation.'
                )
            return self.analyze(data)
        logger.debug(
            'Incremental update requested for CurrencyCorrelation, performing full recalculation for robustness.'
            )
        return self.analyze(data)

    def _calculate_correlation_matrix(self, returns_df: pd.DataFrame,
        window: int) ->Optional[pd.DataFrame]:
        """Calculate correlation matrix for the most recent 'window' periods.

        Args:
            returns_df: DataFrame containing returns for multiple instruments.
            window: Window size for correlation calculation.

        Returns:
            Correlation matrix as DataFrame, or None if insufficient data.
        """
        if len(returns_df) < window:
            logger.warning(
                f'Insufficient data for correlation window {window} (need {window}, got {len(returns_df)})'
                )
            return None
        return returns_df.tail(window).corr()

    @with_exception_handling
    def _calculate_rolling_correlations(self, returns_df: pd.DataFrame,
        window: int) ->Dict[str, Optional[pd.Series]]:
        """Calculate rolling correlations for all pairs of instruments.

        Args:
            returns_df: DataFrame containing returns for multiple instruments.
            window: Window size for rolling correlation.

        Returns:
            Dictionary mapping pair names (e.g., "EURUSD_vs_GBPUSD") to rolling correlation Series.
            Returns None for a pair if calculation fails.
        """
        instruments = returns_df.columns
        result = {}
        min_periods_rolling = window // 2
        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                inst1 = instruments[i]
                inst2 = instruments[j]
                pair_name = f'{inst1}_vs_{inst2}'
                try:
                    rolling_corr = returns_df[inst1].rolling(window=window,
                        min_periods=min_periods_rolling).corr(returns_df[inst2]
                        )
                    result[pair_name] = rolling_corr
                except Exception as e:
                    logger.error(
                        f'Error calculating rolling correlation for {pair_name}: {e}'
                        )
                    result[pair_name] = None
        return result

    @with_exception_handling
    def _detect_correlation_regime_changes(self, current_corr: Optional[pd.
        DataFrame], long_term_corr: Optional[pd.DataFrame]) ->List[Dict[str,
        Any]]:
        """Detect significant changes between short-term and long-term correlation.

        Args:
            current_corr: Short-term correlation matrix (e.g., base_window).
            long_term_corr: Long-term correlation matrix (e.g., long_window).

        Returns:
            List of detected correlation regime changes.
        """
        changes = []
        if current_corr is None or long_term_corr is None:
            logger.warning(
                'Cannot detect regime changes, missing correlation matrix.')
            return changes
        threshold = self.parameters['regime_change_threshold']
        instruments = current_corr.columns
        common_instruments = instruments.intersection(long_term_corr.columns)
        for i in common_instruments:
            for j in common_instruments:
                if i >= j:
                    continue
                try:
                    current_val = current_corr.loc[i, j]
                    long_term_val = long_term_corr.loc[i, j]
                    if pd.isna(current_val) or pd.isna(long_term_val):
                        continue
                    diff = current_val - long_term_val
                    if abs(diff) > threshold:
                        changes.append({'pair': (i, j),
                            'current_correlation': round(current_val, 4),
                            'long_term_correlation': round(long_term_val, 4
                            ), 'change': round(diff, 4), 'is_significant': 
                            True})
                except KeyError:
                    logger.debug(
                        f'KeyError comparing correlation for pair ({i}, {j}). Skipping.'
                        )
                    continue
                except Exception as e:
                    logger.error(
                        f'Error comparing correlation for pair ({i}, {j}): {e}'
                        )
                    continue
        return changes

    @with_exception_handling
    def _identify_strong_correlations(self, corr_matrix: Optional[pd.DataFrame]
        ) ->Dict[str, List[Tuple[str, str, float]]]:
        """Identify strongly positively and negatively correlated pairs from a matrix.

        Args:
            corr_matrix: Correlation matrix.

        Returns:
            Dictionary with lists of strongly correlated ('positive') and
            inversely correlated ('negative') pairs.
        """
        results: Dict[str, List[Tuple[str, str, float]]] = {'positive': [],
            'negative': []}
        if corr_matrix is None:
            logger.warning(
                'Cannot identify strong correlations, missing correlation matrix.'
                )
            return results
        threshold = self.parameters['correlation_threshold']
        instruments = corr_matrix.columns
        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                inst1 = instruments[i]
                inst2 = instruments[j]
                try:
                    correlation = corr_matrix.loc[inst1, inst2]
                    if pd.isna(correlation):
                        continue
                    if correlation > threshold:
                        results['positive'].append((inst1, inst2, round(
                            correlation, 4)))
                    elif correlation < -threshold:
                        results['negative'].append((inst1, inst2, round(
                            correlation, 4)))
                except KeyError:
                    logger.debug(
                        f'KeyError identifying strong correlation for pair ({inst1}, {inst2}). Skipping.'
                        )
                    continue
                except Exception as e:
                    logger.error(
                        f'Error identifying strong correlation for pair ({inst1}, {inst2}): {e}'
                        )
                    continue
        results['positive'].sort(key=lambda x: x[2], reverse=True)
        results['negative'].sort(key=lambda x: x[2], reverse=False)
        return results

    def _calculate_correlation_stability(self, returns_df: pd.DataFrame
        ) ->Dict[str, Any]:
        """Calculate metrics for correlation stability by comparing two halves of the data.

        Args:
            returns_df: DataFrame containing returns.

        Returns:
            Dictionary of correlation stability metrics, or error message.
        """
        min_points_per_half = max(10, self.parameters['long_window'] // 2)
        if len(returns_df) < min_points_per_half * 2:
            return {'error':
                f'Insufficient data for stability calculation (need at least {min_points_per_half * 2} rows)'
                }
        half_point = len(returns_df) // 2
        first_half_returns = returns_df.iloc[:half_point]
        second_half_returns = returns_df.iloc[half_point:]
        first_half_corr = first_half_returns.corr()
        second_half_corr = second_half_returns.corr()
        first_half_corr, second_half_corr = first_half_corr.align(
            second_half_corr, join='inner')
        if first_half_corr.empty:
            return {'error':
                'Correlation matrix for stability calculation is empty after alignment.'
                }
        difference = (second_half_corr - first_half_corr).abs()
        upper_triangle_indices = np.triu_indices_from(difference.values, k=1)
        if not upper_triangle_indices or len(upper_triangle_indices[0]) == 0:
            return {'message':
                'No pairs to compare for stability (only one instrument?).'}
        flattened_diff = difference.values[upper_triangle_indices]
        valid_diffs = flattened_diff[~np.isnan(flattened_diff)]
        if len(valid_diffs) == 0:
            return {'error':
                'Could not calculate stability metrics (all differences were NaN).'
                }
        avg_change = valid_diffs.mean()
        max_change = valid_diffs.max()
        std_dev_change = valid_diffs.std()
        indices = np.argsort(valid_diffs)
        row_indices, col_indices = upper_triangle_indices
        instruments = difference.columns
        num_pairs_to_show = min(3, len(valid_diffs))
        most_stable_pairs = [(instruments[row_indices[idx]], instruments[
            col_indices[idx]], round(valid_diffs[idx], 4)) for idx in
            indices[:num_pairs_to_show]]
        most_volatile_pairs = [(instruments[row_indices[idx]], instruments[
            col_indices[idx]], round(valid_diffs[idx], 4)) for idx in
            indices[-num_pairs_to_show:][::-1]]
        return {'average_correlation_change': round(avg_change, 4),
            'maximum_correlation_change': round(max_change, 4),
            'stdev_correlation_change': round(std_dev_change, 4),
            'most_stable_pairs': most_stable_pairs, 'most_volatile_pairs':
            most_volatile_pairs, 'comparison_period_length': half_point}
