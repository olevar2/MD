"""
Time Cycle Analyzer Module

This module analyzes time-based cycles and seasonality patterns in price data
to identify recurring patterns that can be used for trading decisions.
It includes methods for detecting cycles of various frequencies and projecting
future cycle turning points.
"""
import logging
import numpy as np
import pandas as pd
import scipy.signal as signal
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy.fftpack import fft, ifft, fftfreq
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

class TimeCycleAnalyzer(BaseAnalyzer):
    """
    Analyzer for time cycles and seasonality in market data

    This analyzer detects cycles of various frequencies in price data
    and projects future turning points based on identified cycles.
    It combines spectral analysis with pattern recognition to identify
    reliable cycles and seasonality patterns.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the Time Cycle Analyzer

        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {'price_column': 'close', 'min_cycle_length': 5,
            'max_cycle_length': 252, 'detrend_method': 'linear',
            'polynomial_degree': 2, 'spectral_method': 'fft',
            'min_cycle_strength': 0.1, 'projection_bars': 50,
            'seasonality_periods': [5, 20, 60, 252], 'significance_level': 0.05
            }
        resolved_params = default_params.copy()
        if parameters:
            resolved_params.update(parameters)
        super().__init__('Time Cycle Analyzer', resolved_params)
        logger.info(
            f'Initialized TimeCycleAnalyzer with params: {resolved_params}')

    @with_exception_handling
    def analyze(self, market_data: MarketData) ->AnalysisResult:
        """
        Perform time cycle analysis on the provided market data.

        Args:
            market_data: MarketData object containing OHLCV data.

        Returns:
            AnalysisResult containing identified cycles and projected turning points.
        """
        df = market_data.data
        params = self.parameters
        price_col = params['price_column']
        if len(df) < params['max_cycle_length'] * 2:
            logger.warning('Not enough data for Time Cycle Analysis')
            return AnalysisResult(analyzer_name=self.name, result={
                'dominant_cycles': [], 'projections': {}})
        detrended_price = self._detrend_data(df[price_col], params[
            'detrend_method'], params['polynomial_degree'])
        if detrended_price is None or detrended_price.isna().all():
            logger.warning('Detrending failed or resulted in all NaNs')
            return AnalysisResult(analyzer_name=self.name, result={
                'dominant_cycles': [], 'projections': {}})
        try:
            frequencies, spectrum = self._perform_fft(detrended_price.dropna())
        except Exception as e:
            logger.error(f'FFT calculation failed: {e}')
            return AnalysisResult(analyzer_name=self.name, result={
                'dominant_cycles': [], 'projections': {}})
        dominant_cycles = self._identify_dominant_cycles(frequencies,
            spectrum, len(detrended_price.dropna()), params[
            'min_cycle_length'], params['max_cycle_length'], params[
            'min_cycle_strength'])
        projections = self._project_turning_points(df.index[-1],
            dominant_cycles, params['projection_bars'])
        analysis_data = {'dominant_cycles': dominant_cycles, 'projections':
            projections}
        return AnalysisResult(analyzer_name=self.name, result=analysis_data)

    def _detrend_data(self, series: pd.Series, method: str, degree: int=2
        ) ->Optional[pd.Series]:
        """Detrend the price series using the specified method."""
        series = series.dropna()
        if series.empty:
            return None
        if method == 'linear':
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series.values, 1)
            trend = intercept + slope * x
            return series - trend
        elif method == 'polynomial':
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, degree)
            trend = np.polyval(coeffs, x)
            return series - trend
        elif method == 'diff':
            return series.diff().dropna()
        elif method == 'none':
            return series
        else:
            logger.warning(f'Unknown detrend method: {method}. Using linear.')
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series.values, 1)
            trend = intercept + slope * x
            return series - trend

    def _perform_fft(self, series: pd.Series) ->Tuple[np.ndarray, np.ndarray]:
        """Perform Fast Fourier Transform on the detrended series."""
        n = len(series)
        if n == 0:
            return np.array([]), np.array([])
        fft_values = fft(series.values)
        frequencies = fftfreq(n, d=1)
        spectrum = np.abs(fft_values[1:n // 2]) ** 2
        positive_frequencies = frequencies[1:n // 2]
        return positive_frequencies, spectrum

    def _identify_dominant_cycles(self, frequencies: np.ndarray, spectrum:
        np.ndarray, n_points: int, min_len: int, max_len: int, min_strength:
        float) ->List[Dict]:
        """Identify dominant cycles from the spectrum."""
        if len(frequencies) == 0 or len(spectrum) == 0:
            return []
        dominant_cycles = []
        total_power = np.sum(spectrum)
        if total_power < 1e-09:
            return []
        peaks, properties = signal.find_peaks(spectrum, height=0)
        if len(peaks) == 0:
            return []
        sorted_peak_indices = np.argsort(properties['peak_heights'])[::-1]
        for peak_idx in sorted_peak_indices:
            peak_freq = frequencies[peaks[peak_idx]]
            peak_strength = properties['peak_heights'][peak_idx]
            if abs(peak_freq) < 1e-09:
                continue
            cycle_length = 1.0 / abs(peak_freq)
            relative_strength = peak_strength / total_power
            if (min_len <= cycle_length <= max_len and relative_strength >=
                min_strength):
                phase = 0
                dominant_cycles.append({'length': round(cycle_length, 2),
                    'strength': round(relative_strength, 4), 'phase': round
                    (phase, 4)})
                if len(dominant_cycles) >= 5:
                    break
        return dominant_cycles

    def _project_turning_points(self, last_date: datetime, cycles: List[
        Dict], projection_bars: int) ->Dict[float, List[datetime]]:
        """
        Project future cycle turning points based on dominant cycles.
        Assumes cycles continue with the same length and phase.
        This is a simplified projection.
        """
        projections = {}
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.Timestamp(last_date)
        for cycle in cycles:
            length = cycle['length']
            projected_dates = []
            for i in range(1, int(projection_bars / length) + 2):
                future_date_full = last_date + timedelta(days=int(i * length))
                future_date_half = last_date + timedelta(days=int((i - 0.5) *
                    length))
                if (future_date_half - last_date).days <= projection_bars:
                    projected_dates.append(future_date_half)
                if (future_date_full - last_date).days <= projection_bars:
                    projected_dates.append(future_date_full)
            if projected_dates:
                projections[length] = sorted(list(set(projected_dates)))[:
                    projection_bars]
        return projections

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate cycle information and add columns to the DataFrame.
        This is more complex as cycles are properties of the whole series.
        We can add the *current* dominant cycle length and strength.

        Args:
            df: DataFrame containing OHLCV data.

        Returns:
            DataFrame with added cycle information columns.
        """
        params = self.parameters
        price_col = params['price_column']
        result_df = df.copy()
        result_df['dominant_cycle_len'] = np.nan
        result_df['dominant_cycle_str'] = np.nan
        min_data_needed = params['max_cycle_length'] * 2
        if len(result_df) < min_data_needed:
            return result_df
        market_data_obj = MarketData(symbol='TEMP', timeframe='TEMP', data=
            result_df)
        analysis_result = self.analyze(market_data_obj)
        dominant_cycles = analysis_result.result.get('dominant_cycles', [])
        if dominant_cycles:
            strongest_cycle = max(dominant_cycles, key=lambda x: x['strength'])
            result_df.iloc[-1, result_df.columns.get_loc('dominant_cycle_len')
                ] = strongest_cycle['length']
            result_df.iloc[-1, result_df.columns.get_loc('dominant_cycle_str')
                ] = strongest_cycle['strength']
        return result_df


if __name__ == '__main__':
    periods = 400
    dates = pd.date_range(start='2022-01-01', periods=periods, freq='D')
    cycle1 = 5 * np.sin(2 * np.pi * np.arange(periods) / 50)
    cycle2 = 3 * np.cos(2 * np.pi * np.arange(periods) / 20)
    trend = 0.05 * np.arange(periods)
    noise = np.random.randn(periods) * 1.5
    price = 100 + cycle1 + cycle2 + trend + noise
    sample_df = pd.DataFrame({'close': price}, index=dates)
    sample_df['high'] = sample_df['close'] + np.random.rand(periods) * 1
    sample_df['low'] = sample_df['close'] - np.random.rand(periods) * 1
    sample_df['open'] = sample_df['low'] + (sample_df['high'] - sample_df[
        'low']) * np.random.rand(periods)
    sample_df['volume'] = 1000
    print('Sample Data Head:')
    print(sample_df.head())
    analyzer = TimeCycleAnalyzer(parameters={'min_cycle_length': 10,
        'max_cycle_length': 100, 'min_cycle_strength': 0.05,
        'projection_bars': 30})
    market_data_obj = MarketData(symbol='CYCLE_TEST', timeframe='D1', data=
        sample_df)
    analysis_result = analyzer.analyze(market_data_obj)
    print(f'\nTime Cycle Analysis Result:')
    print('Dominant Cycles:')
    for cycle in analysis_result.result.get('dominant_cycles', []):
        print(
            f"  - Length: {cycle['length']}, Strength: {cycle['strength']:.4f}"
            )
    print('\nProjected Turning Points (by cycle length):')
    for length, dates in analysis_result.result.get('projections', {}).items():
        print(
            f"  Cycle {length:.1f} days: {[d.strftime('%Y-%m-%d') for d in dates]}"
            )
    result_with_indicator = analyzer.calculate(sample_df)
    print('\nDataFrame with Time Cycle Columns (tail):')
    print(result_with_indicator.tail()[['close', 'dominant_cycle_len',
        'dominant_cycle_str']])
