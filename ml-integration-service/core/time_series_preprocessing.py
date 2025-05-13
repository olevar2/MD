"""
ML Integration: Time Series Preprocessing Module

This module provides specialized preprocessing tools for time series data 
in the context of financial machine learning models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Union, Optional, Any, Callable
from enum import Enum
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
import statsmodels.api as sm
from scipy import signal, stats
from tqdm import tqdm
logger = logging.getLogger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeSeriesResamplingMethod(Enum):
    """Methods for resampling time series data"""
    FORWARD_FILL = 'ffill'
    BACKWARD_FILL = 'bfill'
    LINEAR = 'linear'
    CUBIC = 'cubic'
    TIME = 'time'
    NEAREST = 'nearest'
    MEAN = 'mean'
    MEDIAN = 'median'
    OHLC = 'ohlc'


class TimeSeriesDecompositionMethod(Enum):
    """Methods for decomposing time series"""
    NONE = 'none'
    ADDITIVE = 'additive'
    MULTIPLICATIVE = 'multiplicative'
    STL = 'stl'
    WAVELET = 'wavelet'


class TimeSeriesSplitter:
    """
    Split time series data into train/validation/test sets with time ordering.
    """

    def __init__(self, train_ratio: float=0.7, validation_ratio: float=0.15,
        test_ratio: float=0.15, purge_gap: int=0, timestamp_col: str=None):
        """
        Initialize time series splitter
        
        Args:
            train_ratio: Portion of data for training
            validation_ratio: Portion of data for validation
            test_ratio: Portion of data for testing
            purge_gap: Number of samples to skip between splits (to avoid leakage)
            timestamp_col: Column to use for timestamps (if None, use index)
        """
        total = train_ratio + validation_ratio + test_ratio
        if not np.isclose(total, 1.0):
            warnings.warn(f'Split ratios sum to {total}, not 1.0')
            train_ratio /= total
            validation_ratio /= total
            test_ratio /= total
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.purge_gap = max(0, purge_gap)
        self.timestamp_col = timestamp_col

    def split(self, data: pd.DataFrame) ->Dict[str, pd.DataFrame]:
        """
        Split the data into train/validation/test sets
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        n_samples = len(data)
        if n_samples < 3:
            raise ValueError('Not enough samples to split')
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))
        val_start = train_end + self.purge_gap
        test_start = val_end + self.purge_gap
        if test_start >= n_samples:
            warnings.warn('Purge gap too large, reducing to fit data')
            max_gap = max(0, (n_samples - train_end - 1) // 2)
            val_start = train_end + max_gap
            test_start = val_start + max_gap
        if val_start >= val_end or test_start >= n_samples:
            warnings.warn('Not enough data for validation/test with purge gap')
            val_start = train_end
            test_start = val_end
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[val_start:val_end].copy()
        test_data = data.iloc[test_start:].copy()
        return {'train': train_data, 'validation': val_data, 'test': test_data}

    def walk_forward_split(self, data: pd.DataFrame, window_size: int,
        step_size: int=None, max_windows: int=None) ->List[Dict[str, pd.
        DataFrame]]:
        """
        Create a series of expanding/rolling windows for walk-forward validation
        
        Args:
            data: DataFrame with time series data
            window_size: Initial window size for training
            step_size: Size of each step forward (default: 10% of window)
            max_windows: Maximum number of windows to create
            
        Returns:
            List of dictionaries containing train/validation/test for each window
        """
        n_samples = len(data)
        if window_size >= n_samples:
            raise ValueError('Window size must be smaller than data length')
        if step_size is None:
            step_size = max(1, int(window_size * 0.1))
        val_size = max(1, int(window_size * self.validation_ratio / self.
            train_ratio))
        test_size = max(1, int(window_size * self.test_ratio / self.
            train_ratio))
        total_window = (window_size + self.purge_gap + val_size + self.
            purge_gap + test_size)
        if total_window > n_samples:
            warnings.warn(
                'Not enough samples for complete windows, reducing sizes')
            scale_factor = (n_samples - 2 * self.purge_gap) / (window_size +
                val_size + test_size)
            window_size = max(1, int(window_size * scale_factor))
            val_size = max(1, int(val_size * scale_factor))
            test_size = max(1, int(test_size * scale_factor))
            total_window = (window_size + self.purge_gap + val_size + self.
                purge_gap + test_size)
        available_windows = (n_samples - total_window) // step_size + 1
        if max_windows is not None:
            available_windows = min(available_windows, max_windows)
        if available_windows <= 0:
            raise ValueError('Not enough data for walk-forward validation')
        windows = []
        for i in range(available_windows):
            start_idx = i * step_size
            train_end = start_idx + window_size
            val_start = train_end + self.purge_gap
            val_end = val_start + val_size
            test_start = val_end + self.purge_gap
            test_end = test_start + test_size
            if test_end > n_samples:
                break
            window_split = {'train': data.iloc[start_idx:train_end].copy(),
                'validation': data.iloc[val_start:val_end].copy(), 'test':
                data.iloc[test_start:test_end].copy()}
            windows.append(window_split)
        return windows


class SeriesResampler:
    """
    Utility for resampling time series data to different frequencies.
    """

    @with_exception_handling
    def __init__(self, method: Union[str, TimeSeriesResamplingMethod]=
        TimeSeriesResamplingMethod.FORWARD_FILL):
        """
        Initialize resampler
        
        Args:
            method: Resampling method to use
        """
        if isinstance(method, str):
            try:
                method = TimeSeriesResamplingMethod(method)
            except ValueError:
                warnings.warn(
                    f'Unknown resampling method {method}, using forward fill')
                method = TimeSeriesResamplingMethod.FORWARD_FILL
        self.method = method

    @with_exception_handling
    def resample(self, data: pd.DataFrame, new_freq: str, timestamp_col:
        str=None) ->pd.DataFrame:
        """
        Resample data to new frequency
        
        Args:
            data: DataFrame with time series data
            new_freq: New frequency (pandas frequency string)
            timestamp_col: Column containing timestamps (None to use index)
            
        Returns:
            Resampled DataFrame
        """
        df = data.copy()
        if timestamp_col is not None:
            if timestamp_col not in df.columns:
                raise ValueError(f'Timestamp column {timestamp_col} not found')
            df = df.set_index(timestamp_col)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError(
                    'Index must be convertible to datetime for resampling')
        if self.method == TimeSeriesResamplingMethod.OHLC:
            return self._resample_ohlc(df, new_freq)
        else:
            return self._resample_general(df, new_freq)

    def _resample_ohlc(self, df: pd.DataFrame, new_freq: str) ->pd.DataFrame:
        """OHLC resampling for price data"""
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col.lower() not in
            map(str.lower, df.columns)]
        if missing_cols:
            if 'close' in df.columns and len(missing_cols) < 4:
                df = self._create_ohlc_from_close(df)
            else:
                warnings.warn(
                    f'Missing columns for OHLC: {missing_cols}, using close only'
                    )
                if len(df.columns) == 1:
                    df = df.rename(columns={df.columns[0]: 'close'})
                    df = self._create_ohlc_from_close(df)
                else:
                    raise ValueError(
                        'Cannot perform OHLC resampling without price data')
        col_map = {col: col.lower() for col in df.columns if col.lower() in
            ['open', 'high', 'low', 'close', 'volume']}
        df = df.rename(columns=col_map)
        resampler = df.resample(new_freq)
        result = pd.DataFrame(index=resampler.indices.keys())
        if 'open' in df.columns:
            result['open'] = resampler['open'].first()
        if 'high' in df.columns:
            result['high'] = resampler['high'].max()
        if 'low' in df.columns:
            result['low'] = resampler['low'].min()
        if 'close' in df.columns:
            result['close'] = resampler['close'].last()
        if 'volume' in df.columns:
            result['volume'] = resampler['volume'].sum()
        other_cols = [col for col in df.columns if col.lower() not in [
            'open', 'high', 'low', 'close', 'volume']]
        for col in other_cols:
            result[col] = resampler[col].mean()
        return result

    def _create_ohlc_from_close(self, df: pd.DataFrame) ->pd.DataFrame:
        """Create OHLC data from close prices"""
        result = df.copy()
        close_col = next((col for col in df.columns if col.lower() ==
            'close'), df.columns[0])
        if 'open' not in df.columns:
            result['open'] = df[close_col].shift(1)
        if 'high' not in df.columns:
            result['high'] = df[close_col]
        if 'low' not in df.columns:
            result['low'] = df[close_col]
        if 'open' in result.columns:
            result['open'] = result['open'].fillna(result[close_col])
        return result

    def _resample_general(self, df: pd.DataFrame, new_freq: str
        ) ->pd.DataFrame:
        """General resampling for any time series data"""
        resampler = df.resample(new_freq)
        if self.method == TimeSeriesResamplingMethod.MEAN:
            result = resampler.mean()
        elif self.method == TimeSeriesResamplingMethod.MEDIAN:
            result = resampler.median()
        elif self.method == TimeSeriesResamplingMethod.FORWARD_FILL:
            result = resampler.first().ffill()
        elif self.method == TimeSeriesResamplingMethod.BACKWARD_FILL:
            result = resampler.last().bfill()
        elif self.method in [TimeSeriesResamplingMethod.LINEAR,
            TimeSeriesResamplingMethod.CUBIC]:
            result = resampler.asfreq()
            method = self.method.value
            result = result.interpolate(method=method)
        elif self.method == TimeSeriesResamplingMethod.NEAREST:
            result = resampler.nearest()
        elif self.method == TimeSeriesResamplingMethod.TIME:
            result = resampler.mean()
        else:
            result = resampler.first().ffill()
        return result


class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessor for time series data before ML model training.
    """

    @with_exception_handling
    def __init__(self, fill_missing: bool=True, remove_outliers: bool=True,
        normalize: bool=True, decompose: bool=False, lag_features: List[int
        ]=None, rolling_features: List[int]=None, outlier_std_threshold:
        float=3.0, normalization_method: str='minmax', decomposition_method:
        TimeSeriesDecompositionMethod=TimeSeriesDecompositionMethod.NONE):
        """
        Initialize preprocessor
        
        Args:
            fill_missing: Whether to fill missing values
            remove_outliers: Whether to remove outliers
            normalize: Whether to normalize features
            decompose: Whether to decompose time series
            lag_features: List of lag periods to create features for
            rolling_features: List of rolling window sizes
            outlier_std_threshold: Threshold for outlier detection
            normalization_method: Method for normalization
            decomposition_method: Method for time series decomposition
        """
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.normalize = normalize
        self.decompose = decompose
        self.lag_features = lag_features or []
        self.rolling_features = rolling_features or []
        self.outlier_std_threshold = outlier_std_threshold
        if normalization_method == 'standard':
            self.normalizer = StandardScaler()
        elif normalization_method == 'minmax':
            self.normalizer = MinMaxScaler()
        elif normalization_method == 'robust':
            self.normalizer = RobustScaler()
        else:
            warnings.warn(
                f'Unknown normalization method {normalization_method}, using MinMaxScaler'
                )
            self.normalizer = MinMaxScaler()
        if isinstance(decomposition_method, str):
            try:
                self.decomposition_method = TimeSeriesDecompositionMethod(
                    decomposition_method)
            except ValueError:
                warnings.warn(
                    f'Unknown decomposition method {decomposition_method}, using None'
                    )
                self.decomposition_method = TimeSeriesDecompositionMethod.NONE
        else:
            self.decomposition_method = decomposition_method
        self.fitted = False
        self.feature_means = {}
        self.feature_stds = {}
        self.decomposition_models = {}
        self.logger = logging.getLogger(__name__)

    def fit(self, X: pd.DataFrame, y: pd.Series=None
        ) ->'TimeSeriesPreprocessor':
        """
        Fit the preprocessor to training data
        
        Args:
            X: DataFrame with features
            y: Optional target variable
            
        Returns:
            Self for method chaining
        """
        if X.empty:
            raise ValueError('Cannot fit to empty DataFrame')
        self.feature_means = X.mean().to_dict()
        self.feature_stds = X.std().to_dict()
        if self.normalize:
            X_filled = X.fillna(X.mean())
            self.normalizer.fit(X_filled)
        if (self.decompose and self.decomposition_method !=
            TimeSeriesDecompositionMethod.NONE):
            self._fit_decomposition_models(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) ->pd.DataFrame:
        """
        Transform the data using fitted parameters
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            warnings.warn('Preprocessor not fitted, fitting with provided data'
                )
            self.fit(X)
        result = X.copy()
        if self.fill_missing:
            result = self._fill_missing_values(result)
        if self.remove_outliers:
            result = self._remove_outliers(result)
        if (self.decompose and self.decomposition_method !=
            TimeSeriesDecompositionMethod.NONE):
            result = self._apply_decomposition(result)
        if self.lag_features:
            result = self._create_lag_features(result)
        if self.rolling_features:
            result = self._create_rolling_features(result)
        if self.normalize:
            result = self._normalize_features(result)
        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series=None) ->pd.DataFrame:
        """
        Fit to data, then transform it
        
        Args:
            X: DataFrame to fit and transform
            y: Optional target variable
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)

    def _fill_missing_values(self, X: pd.DataFrame) ->pd.DataFrame:
        """Fill missing values in the data"""
        result = X.ffill()
        for col in result.columns:
            if result[col].isna().any():
                if col in self.feature_means:
                    result[col] = result[col].fillna(self.feature_means[col])
                else:
                    result[col] = result[col].fillna(result[col].mean())
        return result

    def _remove_outliers(self, X: pd.DataFrame) ->pd.DataFrame:
        """Remove outliers from the data"""
        result = X.copy()
        for col in result.columns:
            if col in self.feature_means and col in self.feature_stds:
                mean = self.feature_means[col]
                std = self.feature_stds[col]
            else:
                mean = result[col].mean()
                std = result[col].std()
            if std < 1e-10:
                continue
            lower_bound = mean - self.outlier_std_threshold * std
            upper_bound = mean + self.outlier_std_threshold * std
            outlier_mask = (result[col] < lower_bound) | (result[col] >
                upper_bound)
            result.loc[outlier_mask, col] = mean
        return result

    def _normalize_features(self, X: pd.DataFrame) ->pd.DataFrame:
        """Normalize features using fitted scaler"""
        if X.empty:
            return X
        X_filled = X.fillna(X.mean())
        normalized_data = self.normalizer.transform(X_filled)
        result = pd.DataFrame(normalized_data, index=X.index, columns=X.columns
            )
        return result

    @with_exception_handling
    def _fit_decomposition_models(self, X: pd.DataFrame) ->None:
        """Fit decomposition models for each column"""
        if self.decomposition_method in [TimeSeriesDecompositionMethod.STL,
            TimeSeriesDecompositionMethod.ADDITIVE,
            TimeSeriesDecompositionMethod.MULTIPLICATIVE]:
            for col in X.columns:
                if X[col].isna().sum() > len(X) * 0.5:
                    continue
                series = X[col].fillna(method='ffill').fillna(method='bfill')
                try:
                    if (self.decomposition_method ==
                        TimeSeriesDecompositionMethod.STL):
                        freq = self._detect_seasonality(series)
                        if freq > 1:
                            if len(series) >= 2 * freq:
                                model = sm.tsa.STL(series, seasonal=freq,
                                    robust=True)
                                self.decomposition_models[col] = {'type':
                                    'stl', 'model': model, 'period': freq}
                    elif self.decomposition_method in [
                        TimeSeriesDecompositionMethod.ADDITIVE,
                        TimeSeriesDecompositionMethod.MULTIPLICATIVE]:
                        freq = self._detect_seasonality(series)
                        if freq > 1:
                            self.decomposition_models[col] = {'type': self.
                                decomposition_method.value, 'period': freq}
                except Exception as e:
                    self.logger.warning(
                        f'Failed to fit decomposition model for {col}: {str(e)}'
                        )

    @with_exception_handling
    def _apply_decomposition(self, X: pd.DataFrame) ->pd.DataFrame:
        """Apply decomposition to the data"""
        result = X.copy()
        if not self.decomposition_models:
            return result
        for col, model_info in self.decomposition_models.items():
            if col not in result.columns:
                continue
            series = result[col].fillna(method='ffill').fillna(method='bfill')
            try:
                if model_info['type'] == 'stl':
                    stl_result = model_info['model'].fit()
                    result[f'{col}_trend'] = stl_result.trend
                    result[f'{col}_seasonal'] = stl_result.seasonal
                    result[f'{col}_residual'] = stl_result.resid
                elif model_info['type'] in ['additive', 'multiplicative']:
                    period = model_info['period']
                    if len(series) >= 2 * period:
                        decomposition = sm.tsa.seasonal_decompose(series,
                            model=model_info['type'], period=period)
                        result[f'{col}_trend'] = decomposition.trend
                        result[f'{col}_seasonal'] = decomposition.seasonal
                        result[f'{col}_residual'] = decomposition.resid
            except Exception as e:
                self.logger.warning(
                    f'Failed to apply decomposition for {col}: {str(e)}')
        return result

    def _create_lag_features(self, X: pd.DataFrame) ->pd.DataFrame:
        """Create lag features"""
        result = X.copy()
        for lag in self.lag_features:
            if lag <= 0:
                continue
            for col in X.columns:
                result[f'{col}_lag_{lag}'] = X[col].shift(lag)
        return result

    def _create_rolling_features(self, X: pd.DataFrame) ->pd.DataFrame:
        """Create rolling window features"""
        result = X.copy()
        for window in self.rolling_features:
            if window <= 1:
                continue
            for col in X.columns:
                result[f'{col}_roll_mean_{window}'] = X[col].rolling(window
                    =window).mean()
                result[f'{col}_roll_std_{window}'] = X[col].rolling(window=
                    window).std()
                if window >= 5:
                    result[f'{col}_roll_min_{window}'] = X[col].rolling(window
                        =window).min()
                    result[f'{col}_roll_max_{window}'] = X[col].rolling(window
                        =window).max()
        return result

    @with_exception_handling
    def _detect_seasonality(self, series: pd.Series) ->int:
        """
        Detect seasonality period in a time series
        
        Args:
            series: Time series to analyze
            
        Returns:
            Detected period (0 if none found)
        """
        if len(series) < 10:
            return 0
        candidates = [5, 20, 60, 252]
        try:
            detrended = series - series.rolling(window=min(len(series) // 3,
                20)).mean()
            detrended = detrended.fillna(0)
            acf = sm.tsa.acf(detrended, nlags=min(len(detrended) - 1, max(
                candidates) * 2), fft=True)
            peaks, _ = signal.find_peaks(acf, height=0.1)
            if len(peaks) > 0:
                for candidate in candidates:
                    nearest_peak = min(peaks, key=lambda x: abs(x - candidate))
                    if abs(nearest_peak - candidate) <= candidate * 0.25:
                        return candidate
                return peaks[0] if len(peaks) > 0 else 0
            else:
                return 0
        except Exception:
            return 0
