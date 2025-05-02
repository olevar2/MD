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

# Configure logging
logger = logging.getLogger(__name__)


class TimeSeriesResamplingMethod(Enum):
    """Methods for resampling time series data"""
    FORWARD_FILL = "ffill"  # Forward fill missing values
    BACKWARD_FILL = "bfill"  # Backward fill missing values
    LINEAR = "linear"  # Linear interpolation
    CUBIC = "cubic"  # Cubic interpolation
    TIME = "time"  # Time-weighted interpolation
    NEAREST = "nearest"  # Nearest value
    MEAN = "mean"  # Mean of values in period
    MEDIAN = "median"  # Median of values in period
    OHLC = "ohlc"  # Convert to OHLC format


class TimeSeriesDecompositionMethod(Enum):
    """Methods for decomposing time series"""
    NONE = "none"  # No decomposition
    ADDITIVE = "additive"  # Additive decomposition (trend + seasonality + residual)
    MULTIPLICATIVE = "multiplicative"  # Multiplicative decomposition (trend * seasonality * residual)
    STL = "stl"  # Seasonal-Trend decomposition using LOESS
    WAVELET = "wavelet"  # Wavelet decomposition


class TimeSeriesSplitter:
    """
    Split time series data into train/validation/test sets with time ordering.
    """
    
    def __init__(
        self, 
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_gap: int = 0,
        timestamp_col: str = None  # If None, assumes index is timestamp
    ):
        """
        Initialize time series splitter
        
        Args:
            train_ratio: Portion of data for training
            validation_ratio: Portion of data for validation
            test_ratio: Portion of data for testing
            purge_gap: Number of samples to skip between splits (to avoid leakage)
            timestamp_col: Column to use for timestamps (if None, use index)
        """
        # Validate ratios
        total = train_ratio + validation_ratio + test_ratio
        if not np.isclose(total, 1.0):
            warnings.warn(f"Split ratios sum to {total}, not 1.0")
            # Normalize
            train_ratio /= total
            validation_ratio /= total
            test_ratio /= total
            
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.purge_gap = max(0, purge_gap)  # Ensure non-negative
        self.timestamp_col = timestamp_col
        
    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the data into train/validation/test sets
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        n_samples = len(data)
        
        if n_samples < 3:
            raise ValueError("Not enough samples to split")
            
        # Calculate split points
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))
        
        # Apply purge gap
        val_start = train_end + self.purge_gap
        test_start = val_end + self.purge_gap
        
        # Handle case where purge gap pushes beyond data bounds
        if test_start >= n_samples:
            warnings.warn("Purge gap too large, reducing to fit data")
            # Recalculate with smaller gaps
            max_gap = max(0, (n_samples - train_end - 1) // 2)
            val_start = train_end + max_gap
            test_start = val_start + max_gap
            
        if val_start >= val_end or test_start >= n_samples:
            warnings.warn("Not enough data for validation/test with purge gap")
            # Fall back to no gap
            val_start = train_end
            test_start = val_end
        
        # Split the data
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[val_start:val_end].copy()
        test_data = data.iloc[test_start:].copy()
        
        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
    
    def walk_forward_split(
        self,
        data: pd.DataFrame,
        window_size: int,
        step_size: int = None,
        max_windows: int = None
    ) -> List[Dict[str, pd.DataFrame]]:
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
            raise ValueError("Window size must be smaller than data length")
            
        # Default step size is 10% of window size
        if step_size is None:
            step_size = max(1, int(window_size * 0.1))
        
        # Calculate validation and test sizes based on ratios
        val_size = max(1, int(window_size * self.validation_ratio / self.train_ratio))
        test_size = max(1, int(window_size * self.test_ratio / self.train_ratio))
        
        # Calculate total size needed for one complete window
        total_window = window_size + self.purge_gap + val_size + self.purge_gap + test_size
        
        if total_window > n_samples:
            warnings.warn("Not enough samples for complete windows, reducing sizes")
            # Adjust sizes to fit
            scale_factor = (n_samples - 2 * self.purge_gap) / (window_size + val_size + test_size)
            window_size = max(1, int(window_size * scale_factor))
            val_size = max(1, int(val_size * scale_factor))
            test_size = max(1, int(test_size * scale_factor))
            total_window = window_size + self.purge_gap + val_size + self.purge_gap + test_size
        
        # Calculate how many windows we can create
        available_windows = (n_samples - total_window) // step_size + 1
        
        if max_windows is not None:
            available_windows = min(available_windows, max_windows)
            
        if available_windows <= 0:
            raise ValueError("Not enough data for walk-forward validation")
        
        windows = []
        
        for i in range(available_windows):
            # Calculate boundaries for this window
            start_idx = i * step_size
            train_end = start_idx + window_size
            val_start = train_end + self.purge_gap
            val_end = val_start + val_size
            test_start = val_end + self.purge_gap
            test_end = test_start + test_size
            
            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                break
                
            # Create the split
            window_split = {
                "train": data.iloc[start_idx:train_end].copy(),
                "validation": data.iloc[val_start:val_end].copy(),
                "test": data.iloc[test_start:test_end].copy()
            }
            
            windows.append(window_split)
            
        return windows


class SeriesResampler:
    """
    Utility for resampling time series data to different frequencies.
    """
    
    def __init__(
        self,
        method: Union[str, TimeSeriesResamplingMethod] = TimeSeriesResamplingMethod.FORWARD_FILL
    ):
        """
        Initialize resampler
        
        Args:
            method: Resampling method to use
        """
        if isinstance(method, str):
            try:
                method = TimeSeriesResamplingMethod(method)
            except ValueError:
                warnings.warn(f"Unknown resampling method {method}, using forward fill")
                method = TimeSeriesResamplingMethod.FORWARD_FILL
                
        self.method = method
        
    def resample(
        self,
        data: pd.DataFrame,
        new_freq: str,
        timestamp_col: str = None
    ) -> pd.DataFrame:
        """
        Resample data to new frequency
        
        Args:
            data: DataFrame with time series data
            new_freq: New frequency (pandas frequency string)
            timestamp_col: Column containing timestamps (None to use index)
            
        Returns:
            Resampled DataFrame
        """
        # Ensure we have a datetime index
        df = data.copy()
        
        if timestamp_col is not None:
            if timestamp_col not in df.columns:
                raise ValueError(f"Timestamp column {timestamp_col} not found")
                
            # Set timestamp as index
            df = df.set_index(timestamp_col)
        
        # Check if index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                # Try to convert to datetime
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError("Index must be convertible to datetime for resampling")
        
        # Apply different resampling methods
        if self.method == TimeSeriesResamplingMethod.OHLC:
            # OHLC resampling for price data
            return self._resample_ohlc(df, new_freq)
        else:
            # General resampling
            return self._resample_general(df, new_freq)
            
    def _resample_ohlc(self, df: pd.DataFrame, new_freq: str) -> pd.DataFrame:
        """OHLC resampling for price data"""
        # Check if we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col.lower() not in map(str.lower, df.columns)]
        
        if missing_cols:
            # If we have close but not others, try to create OHLC
            if 'close' in df.columns and len(missing_cols) < 4:
                df = self._create_ohlc_from_close(df)
            else:
                warnings.warn(f"Missing columns for OHLC: {missing_cols}, using close only")
                # If we only have a single column, treat it as close price
                if len(df.columns) == 1:
                    df = df.rename(columns={df.columns[0]: 'close'})
                    df = self._create_ohlc_from_close(df)
                else:
                    raise ValueError("Cannot perform OHLC resampling without price data")
        
        # Map column names to lowercase
        col_map = {col: col.lower() for col in df.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']}
        df = df.rename(columns=col_map)
        
        # Define resampling functions
        resampler = df.resample(new_freq)
        result = pd.DataFrame(index=resampler.indices.keys())
        
        # OHLC aggregation
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
        
        # For other columns, use mean
        other_cols = [col for col in df.columns if col.lower() not in ['open', 'high', 'low', 'close', 'volume']]
        for col in other_cols:
            result[col] = resampler[col].mean()
            
        return result
        
    def _create_ohlc_from_close(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create OHLC data from close prices"""
        result = df.copy()
        
        # Find close column
        close_col = next((col for col in df.columns if col.lower() == 'close'), df.columns[0])
        
        # Create other columns if they don't exist
        if 'open' not in df.columns:
            result['open'] = df[close_col].shift(1)
        if 'high' not in df.columns:
            result['high'] = df[close_col]
        if 'low' not in df.columns:
            result['low'] = df[close_col]
            
        # Fill NAs in open with close
        if 'open' in result.columns:
            result['open'] = result['open'].fillna(result[close_col])
            
        return result
            
    def _resample_general(self, df: pd.DataFrame, new_freq: str) -> pd.DataFrame:
        """General resampling for any time series data"""
        # Create a resampler object
        resampler = df.resample(new_freq)
        
        # Apply the appropriate method
        if self.method == TimeSeriesResamplingMethod.MEAN:
            result = resampler.mean()
        elif self.method == TimeSeriesResamplingMethod.MEDIAN:
            result = resampler.median()
        elif self.method == TimeSeriesResamplingMethod.FORWARD_FILL:
            result = resampler.first().ffill()
        elif self.method == TimeSeriesResamplingMethod.BACKWARD_FILL:
            result = resampler.last().bfill()
        elif self.method in [TimeSeriesResamplingMethod.LINEAR, TimeSeriesResamplingMethod.CUBIC]:
            # For interpolation methods, first resample with NaN values
            result = resampler.asfreq()
            # Then interpolate
            method = self.method.value
            result = result.interpolate(method=method)
        elif self.method == TimeSeriesResamplingMethod.NEAREST:
            result = resampler.nearest()
        elif self.method == TimeSeriesResamplingMethod.TIME:
            # Time-weighted resampling (useful for prices)
            result = resampler.mean()
        else:
            # Default to forward fill
            result = resampler.first().ffill()
            
        return result


class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessor for time series data before ML model training.
    """
    
    def __init__(
        self,
        fill_missing: bool = True,
        remove_outliers: bool = True,
        normalize: bool = True,
        decompose: bool = False,
        lag_features: List[int] = None,
        rolling_features: List[int] = None,
        outlier_std_threshold: float = 3.0,
        normalization_method: str = 'minmax',
        decomposition_method: TimeSeriesDecompositionMethod = TimeSeriesDecompositionMethod.NONE
    ):
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
        
        # Set normalization method
        if normalization_method == 'standard':
            self.normalizer = StandardScaler()
        elif normalization_method == 'minmax':
            self.normalizer = MinMaxScaler()
        elif normalization_method == 'robust':
            self.normalizer = RobustScaler()
        else:
            warnings.warn(f"Unknown normalization method {normalization_method}, using MinMaxScaler")
            self.normalizer = MinMaxScaler()
            
        # Set decomposition method
        if isinstance(decomposition_method, str):
            try:
                self.decomposition_method = TimeSeriesDecompositionMethod(decomposition_method)
            except ValueError:
                warnings.warn(f"Unknown decomposition method {decomposition_method}, using None")
                self.decomposition_method = TimeSeriesDecompositionMethod.NONE
        else:
            self.decomposition_method = decomposition_method
            
        self.fitted = False
        self.feature_means = {}
        self.feature_stds = {}
        self.decomposition_models = {}
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'TimeSeriesPreprocessor':
        """
        Fit the preprocessor to training data
        
        Args:
            X: DataFrame with features
            y: Optional target variable
            
        Returns:
            Self for method chaining
        """
        if X.empty:
            raise ValueError("Cannot fit to empty DataFrame")
            
        # Store means and stds for outlier detection
        self.feature_means = X.mean().to_dict()
        self.feature_stds = X.std().to_dict()
        
        # Fit normalizer if needed
        if self.normalize:
            # Fill NaNs temporarily for fitting
            X_filled = X.fillna(X.mean())
            self.normalizer.fit(X_filled)
            
        # Fit decomposition models if needed
        if self.decompose and self.decomposition_method != TimeSeriesDecompositionMethod.NONE:
            self._fit_decomposition_models(X)
            
        self.fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted parameters
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            warnings.warn("Preprocessor not fitted, fitting with provided data")
            self.fit(X)
            
        result = X.copy()
        
        # Fill missing values
        if self.fill_missing:
            result = self._fill_missing_values(result)
            
        # Remove outliers
        if self.remove_outliers:
            result = self._remove_outliers(result)
            
        # Apply decomposition if needed
        if self.decompose and self.decomposition_method != TimeSeriesDecompositionMethod.NONE:
            result = self._apply_decomposition(result)
            
        # Create lag features
        if self.lag_features:
            result = self._create_lag_features(result)
            
        # Create rolling features
        if self.rolling_features:
            result = self._create_rolling_features(result)
            
        # Normalize features
        if self.normalize:
            result = self._normalize_features(result)
            
        return result
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit to data, then transform it
        
        Args:
            X: DataFrame to fit and transform
            y: Optional target variable
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
        
    def _fill_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the data"""
        # Forward fill first (common for time series)
        result = X.ffill()
        
        # Any remaining NaN (e.g., at the start) get filled with column means
        for col in result.columns:
            if result[col].isna().any():
                if col in self.feature_means:
                    # Use stored mean from training
                    result[col] = result[col].fillna(self.feature_means[col])
                else:
                    # Use current data mean
                    result[col] = result[col].fillna(result[col].mean())
                    
        return result
        
    def _remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data"""
        result = X.copy()
        
        # For each column, replace outliers with column mean
        for col in result.columns:
            if col in self.feature_means and col in self.feature_stds:
                # Use training statistics
                mean = self.feature_means[col]
                std = self.feature_stds[col]
            else:
                # Use current data statistics
                mean = result[col].mean()
                std = result[col].std()
                
            # Skip if std is too small or zero
            if std < 1e-10:
                continue
                
            # Define outlier range
            lower_bound = mean - self.outlier_std_threshold * std
            upper_bound = mean + self.outlier_std_threshold * std
            
            # Replace outliers
            outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
            result.loc[outlier_mask, col] = mean
            
        return result
        
    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using fitted scaler"""
        # Handle empty DataFrame
        if X.empty:
            return X
            
        # Fill NaNs temporarily for normalization
        X_filled = X.fillna(X.mean())
        
        # Transform with scaler
        normalized_data = self.normalizer.transform(X_filled)
        
        # Create new DataFrame with normalized values
        result = pd.DataFrame(
            normalized_data,
            index=X.index,
            columns=X.columns
        )
        
        return result
        
    def _fit_decomposition_models(self, X: pd.DataFrame) -> None:
        """Fit decomposition models for each column"""
        # Only STL and additive/multiplicative methods need fitting
        if self.decomposition_method in [
            TimeSeriesDecompositionMethod.STL, 
            TimeSeriesDecompositionMethod.ADDITIVE,
            TimeSeriesDecompositionMethod.MULTIPLICATIVE
        ]:
            for col in X.columns:
                # Skip columns with too many missing values
                if X[col].isna().sum() > len(X) * 0.5:
                    continue
                    
                # Fill missing values for fitting
                series = X[col].fillna(method='ffill').fillna(method='bfill')
                
                try:
                    if self.decomposition_method == TimeSeriesDecompositionMethod.STL:
                        # For STL, try to detect seasonality
                        freq = self._detect_seasonality(series)
                        if freq > 1:
                            # STL requires 2 full cycles of data
                            if len(series) >= 2 * freq:
                                # Fit STL model
                                model = sm.tsa.STL(
                                    series, 
                                    seasonal=freq,
                                    robust=True
                                )
                                self.decomposition_models[col] = {
                                    'type': 'stl',
                                    'model': model,
                                    'period': freq
                                }
                    elif self.decomposition_method in [
                        TimeSeriesDecompositionMethod.ADDITIVE,
                        TimeSeriesDecompositionMethod.MULTIPLICATIVE
                    ]:
                        # For classical decomposition, also detect seasonality
                        freq = self._detect_seasonality(series)
                        if freq > 1:
                            # Store parameters
                            self.decomposition_models[col] = {
                                'type': self.decomposition_method.value,
                                'period': freq
                            }
                except Exception as e:
                    self.logger.warning(f"Failed to fit decomposition model for {col}: {str(e)}")
        
    def _apply_decomposition(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply decomposition to the data"""
        result = X.copy()
        
        if not self.decomposition_models:
            return result
            
        for col, model_info in self.decomposition_models.items():
            if col not in result.columns:
                continue
                
            # Fill missing values for decomposition
            series = result[col].fillna(method='ffill').fillna(method='bfill')
            
            try:
                if model_info['type'] == 'stl':
                    # Apply STL decomposition
                    stl_result = model_info['model'].fit()
                    
                    # Add decomposition components as new columns
                    result[f"{col}_trend"] = stl_result.trend
                    result[f"{col}_seasonal"] = stl_result.seasonal
                    result[f"{col}_residual"] = stl_result.resid
                    
                elif model_info['type'] in ['additive', 'multiplicative']:
                    # Apply classical decomposition
                    period = model_info['period']
                    if len(series) >= 2 * period:
                        decomposition = sm.tsa.seasonal_decompose(
                            series, 
                            model=model_info['type'],
                            period=period
                        )
                        
                        # Add decomposition components
                        result[f"{col}_trend"] = decomposition.trend
                        result[f"{col}_seasonal"] = decomposition.seasonal
                        result[f"{col}_residual"] = decomposition.resid
            except Exception as e:
                self.logger.warning(f"Failed to apply decomposition for {col}: {str(e)}")
                
        return result
        
    def _create_lag_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        result = X.copy()
        
        for lag in self.lag_features:
            if lag <= 0:
                continue
                
            for col in X.columns:
                result[f"{col}_lag_{lag}"] = X[col].shift(lag)
                
        return result
        
    def _create_rolling_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        result = X.copy()
        
        for window in self.rolling_features:
            if window <= 1:
                continue
                
            for col in X.columns:
                # Add mean and std features
                result[f"{col}_roll_mean_{window}"] = X[col].rolling(window=window).mean()
                result[f"{col}_roll_std_{window}"] = X[col].rolling(window=window).std()
                
                # Add min/max features if window is large enough
                if window >= 5:
                    result[f"{col}_roll_min_{window}"] = X[col].rolling(window=window).min()
                    result[f"{col}_roll_max_{window}"] = X[col].rolling(window=window).max()
                    
        return result
        
    def _detect_seasonality(self, series: pd.Series) -> int:
        """
        Detect seasonality period in a time series
        
        Args:
            series: Time series to analyze
            
        Returns:
            Detected period (0 if none found)
        """
        # Need enough data for reliable detection
        if len(series) < 10:
            return 0
            
        # Check for common seasonalities in financial data
        candidates = [5, 20, 60, 252]  # day, week, month, year in trading days
        
        try:
            # Remove trend for better periodicity detection
            detrended = series - series.rolling(window=min(len(series)//3, 20)).mean()
            detrended = detrended.fillna(0)
            
            # Calculate autocorrelation
            acf = sm.tsa.acf(detrended, nlags=min(len(detrended)-1, max(candidates)*2), fft=True)
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(acf, height=0.1)
            
            if len(peaks) > 0:
                # Find the peak closest to common seasonality periods
                for candidate in candidates:
                    # Check if there's a peak near this candidate
                    nearest_peak = min(peaks, key=lambda x: abs(x - candidate))
                    if abs(nearest_peak - candidate) <= candidate * 0.25:  # Within 25% of candidate
                        return candidate
                        
                # If no common period found, return the first peak
                return peaks[0] if len(peaks) > 0 else 0
            else:
                return 0
        except Exception:
            return 0  # Return 0 on error
