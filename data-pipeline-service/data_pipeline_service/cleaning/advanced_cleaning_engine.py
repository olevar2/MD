"""
Advanced Data Cleaning Engine

This module provides sophisticated data cleaning capabilities for financial market data,
including advanced imputation techniques, anomaly correction, and data transformation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CleaningMethod(Enum):
    """Cleaning methods for different types of issues"""
    DROP = 'drop'
    IMPUTE_MEAN = 'impute_mean'
    IMPUTE_MEDIAN = 'impute_median'
    IMPUTE_MODE = 'impute_mode'
    IMPUTE_KNN = 'impute_knn'
    IMPUTE_FFILL = 'impute_ffill'
    IMPUTE_BFILL = 'impute_bfill'
    IMPUTE_INTERPOLATE = 'impute_interpolate'
    IMPUTE_TIMESERIES = 'impute_timeseries'
    OUTLIER_WINSORIZE = 'outlier_winsorize'
    OUTLIER_CLIP = 'outlier_clip'
    TRANSFORM_LOG = 'transform_log'
    TRANSFORM_SQRT = 'transform_sqrt'
    CORRECT_OHLC = 'correct_ohlc'


class CleaningAction:
    """Represents a cleaning action to be performed"""

    def __init__(self, method: CleaningMethod, target_columns: Optional[
        List[str]]=None, params: Optional[Dict[str, Any]]=None):
        """
        Initialize cleaning action
        
        Args:
            method: Cleaning method to apply
            target_columns: Columns to apply the method to, or None for all appropriate cols
            params: Additional parameters for the cleaning method
        """
        self.method = method
        self.target_columns = target_columns
        self.params = params or {}

    def __str__(self) ->str:
        """String representation"""
        cols_str = ', '.join(self.target_columns
            ) if self.target_columns else 'all appropriate'
        return f'CleaningAction({self.method.value} on {cols_str})'


class CleanedDataResult:
    """Results of a data cleaning operation"""

    def __init__(self, data: pd.DataFrame, original_data: pd.DataFrame,
        cleaning_actions: List[CleaningAction], modifications: Dict[str, Any]):
        """
        Initialize cleaning result
        
        Args:
            data: Cleaned DataFrame
            original_data: Original DataFrame before cleaning
            cleaning_actions: List of actions that were applied
            modifications: Details about modifications made
        """
        self.data = data
        self.original_data = original_data
        self.cleaning_actions = cleaning_actions
        self.modifications = modifications
        self.timestamp = datetime.utcnow()

    def get_summary(self) ->Dict[str, Any]:
        """Get summary of cleaning operations"""
        return {'rows_before': len(self.original_data), 'rows_after': len(
            self.data), 'rows_dropped': len(self.original_data) - len(self.
            data), 'actions_applied': len(self.cleaning_actions),
            'columns_modified': list(self.modifications.keys()),
            'modification_counts': {col: details.get('count', 0) for col,
            details in self.modifications.items()}, 'timestamp': self.
            timestamp.isoformat()}

    def get_diff_stats(self) ->Dict[str, Dict[str, float]]:
        """Get statistics on differences between original and cleaned data"""
        diff_stats = {}
        numeric_cols = [col for col in self.data.columns if col in self.
            original_data.columns and pd.api.types.is_numeric_dtype(self.
            data[col]) and pd.api.types.is_numeric_dtype(self.original_data
            [col])]
        for col in numeric_cols:
            if len(self.data) != len(self.original_data):
                common_idx = self.data.index.intersection(self.
                    original_data.index)
                orig_values = self.original_data.loc[common_idx, col]
                cleaned_values = self.data.loc[common_idx, col]
            else:
                orig_values = self.original_data[col]
                cleaned_values = self.data[col]
            valid_mask = ~orig_values.isna() & ~cleaned_values.isna()
            if valid_mask.sum() > 0:
                diffs = cleaned_values[valid_mask] - orig_values[valid_mask]
                diff_stats[col] = {'mean_diff': float(diffs.mean()) if len(
                    diffs) > 0 else 0.0, 'max_diff': float(diffs.abs().max(
                    )) if len(diffs) > 0 else 0.0, 'pct_changed': float((
                    diffs != 0).sum() / len(diffs)) if len(diffs) > 0 else 0.0}
        return diff_stats


class DataCleaningEngine:
    """
    Advanced cleaning engine for financial data with multiple
    specialized cleaning strategies and detailed tracking of changes
    """

    def __init__(self):
        """Initialize data cleaning engine"""
        self.logger = logging.getLogger(__name__)

    @with_exception_handling
    def clean_data(self, data: pd.DataFrame, actions: List[CleaningAction],
        copy: bool=True) ->CleanedDataResult:
        """
        Clean data according to specified actions
        
        Args:
            data: DataFrame to clean
            actions: List of cleaning actions to apply
            copy: Whether to work on a copy of the data
            
        Returns:
            Cleaned data and information about changes
        """
        original_data = data.copy()
        if copy:
            data = data.copy()
        modifications = {}
        for action in actions:
            try:
                data, mod_details = self._apply_cleaning_method(data,
                    action.method, action.target_columns, action.params)
                for col, details in mod_details.items():
                    if col in modifications:
                        modifications[col]['count'] += details.get('count', 0)
                    else:
                        modifications[col] = details
            except Exception as e:
                self.logger.error(
                    f'Error applying cleaning action {action.method.value}: {str(e)}'
                    )
        return CleanedDataResult(data=data, original_data=original_data,
            cleaning_actions=actions, modifications=modifications)

    def _apply_cleaning_method(self, data: pd.DataFrame, method:
        CleaningMethod, target_columns: Optional[List[str]]=None, params:
        Optional[Dict[str, Any]]=None) ->Tuple[pd.DataFrame, Dict[str, Dict
        [str, Any]]]:
        """
        Apply a specific cleaning method to the data
        
        Args:
            data: DataFrame to clean
            method: Cleaning method to apply
            target_columns: Columns to apply to, or None for all appropriate
            params: Additional parameters for the method
            
        Returns:
            Tuple of (cleaned_data, modifications_details)
        """
        params = params or {}
        modifications = {}
        if method == CleaningMethod.DROP:
            return self._clean_drop(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_MEAN:
            return self._clean_impute_mean(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_MEDIAN:
            return self._clean_impute_median(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_MODE:
            return self._clean_impute_mode(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_KNN:
            return self._clean_impute_knn(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_FFILL:
            return self._clean_impute_ffill(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_BFILL:
            return self._clean_impute_bfill(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_INTERPOLATE:
            return self._clean_impute_interpolate(data, target_columns, params)
        elif method == CleaningMethod.IMPUTE_TIMESERIES:
            return self._clean_impute_timeseries(data, target_columns, params)
        elif method == CleaningMethod.OUTLIER_WINSORIZE:
            return self._clean_outlier_winsorize(data, target_columns, params)
        elif method == CleaningMethod.OUTLIER_CLIP:
            return self._clean_outlier_clip(data, target_columns, params)
        elif method == CleaningMethod.TRANSFORM_LOG:
            return self._clean_transform_log(data, target_columns, params)
        elif method == CleaningMethod.TRANSFORM_SQRT:
            return self._clean_transform_sqrt(data, target_columns, params)
        elif method == CleaningMethod.CORRECT_OHLC:
            return self._clean_correct_ohlc(data, target_columns, params)
        else:
            self.logger.warning(f'Unknown cleaning method: {method}')
            return data, {}

    def _clean_drop(self, data: pd.DataFrame, target_columns: Optional[List
        [str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame, Dict[str,
        Dict[str, Any]]]:
        """Drop rows with issues"""
        cols = target_columns if target_columns else data.columns.tolist()
        cols = [col for col in cols if col in data.columns]
        how = params.get('how', 'any')
        threshold = params.get('threshold', None)
        rows_before = len(data)
        cleaned_data = data.dropna(subset=cols, how=how, thresh=threshold)
        rows_dropped = rows_before - len(cleaned_data)
        modifications = {'dropped_rows': {'count': rows_dropped, 'method':
            'drop', 'params': {'how': how, 'threshold': threshold}}}
        return cleaned_data, modifications

    def _clean_impute_mean(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Impute missing values with mean"""
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        imputer = SimpleImputer(strategy='mean')
        modifications = {}
        for col in cols:
            nulls_before = data[col].isna().sum()
            if nulls_before > 0:
                values = data[col].values.reshape(-1, 1)
                imputed_values = imputer.fit_transform(values).flatten()
                cleaned_data[col] = imputed_values
                mean_value = data[col].mean()
                modifications[col] = {'count': nulls_before, 'method':
                    'impute_mean', 'params': {'value': mean_value}}
        return cleaned_data, modifications

    def _clean_impute_median(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Impute missing values with median"""
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        imputer = SimpleImputer(strategy='median')
        modifications = {}
        for col in cols:
            nulls_before = data[col].isna().sum()
            if nulls_before > 0:
                values = data[col].values.reshape(-1, 1)
                imputed_values = imputer.fit_transform(values).flatten()
                cleaned_data[col] = imputed_values
                median_value = data[col].median()
                modifications[col] = {'count': nulls_before, 'method':
                    'impute_median', 'params': {'value': median_value}}
        return cleaned_data, modifications

    def _clean_impute_mode(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Impute missing values with mode (most frequent)"""
        cols = target_columns if target_columns else data.columns.tolist()
        cols = [col for col in cols if col in data.columns]
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        modifications = {}
        for col in cols:
            nulls_before = data[col].isna().sum()
            if nulls_before > 0:
                mode_value = data[col].mode().iloc[0] if not data[col].dropna(
                    ).empty else None
                if mode_value is not None:
                    cleaned_data[col] = data[col].fillna(mode_value)
                    modifications[col] = {'count': nulls_before, 'method':
                        'impute_mode', 'params': {'value': mode_value}}
        return cleaned_data, modifications

    @with_exception_handling
    def _clean_impute_knn(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Impute missing values using K-Nearest Neighbors"""
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if target_columns:
            cols = [col for col in target_columns if col in numeric_cols]
        else:
            cols = numeric_cols
        if not cols:
            return data, {}
        has_nulls = any(data[col].isna().any() for col in cols)
        if not has_nulls:
            return data, {}
        n_neighbors = params.get('n_neighbors', 5)
        weights = params.get('weights', 'uniform')
        cleaned_data = data.copy()
        modifications = {}
        numeric_data = data[numeric_cols].copy()
        nulls_before = {col: data[col].isna().sum() for col in cols}
        try:
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imputed_values = imputer.fit_transform(numeric_data)
            imputed_df = pd.DataFrame(imputed_values, columns=numeric_cols,
                index=data.index)
            for col in cols:
                if nulls_before[col] > 0:
                    cleaned_data[col] = imputed_df[col]
                    modifications[col] = {'count': nulls_before[col],
                        'method': 'impute_knn', 'params': {'n_neighbors':
                        n_neighbors, 'weights': weights}}
        except Exception as e:
            self.logger.error(f'Error in KNN imputation: {str(e)}')
        return cleaned_data, modifications

    def _clean_impute_ffill(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Fill missing values by carrying forward last valid value"""
        cols = target_columns if target_columns else data.columns.tolist()
        cols = [col for col in cols if col in data.columns]
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        limit = params.get('limit', None)
        modifications = {}
        for col in cols:
            nulls_before = data[col].isna().sum()
            if nulls_before > 0:
                cleaned_data[col] = data[col].ffill(limit=limit)
                nulls_after = cleaned_data[col].isna().sum()
                modifications[col] = {'count': nulls_before - nulls_after,
                    'method': 'impute_ffill', 'params': {'limit': limit}}
        return cleaned_data, modifications

    def _clean_impute_bfill(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Fill missing values by carrying backward from next valid value"""
        cols = target_columns if target_columns else data.columns.tolist()
        cols = [col for col in cols if col in data.columns]
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        limit = params.get('limit', None)
        modifications = {}
        for col in cols:
            nulls_before = data[col].isna().sum()
            if nulls_before > 0:
                cleaned_data[col] = data[col].bfill(limit=limit)
                nulls_after = cleaned_data[col].isna().sum()
                modifications[col] = {'count': nulls_before - nulls_after,
                    'method': 'impute_bfill', 'params': {'limit': limit}}
        return cleaned_data, modifications

    def _clean_impute_interpolate(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Fill missing values using interpolation"""
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        method = params.get('method', 'linear')
        limit = params.get('limit', None)
        limit_direction = params.get('limit_direction', 'forward')
        modifications = {}
        for col in cols:
            nulls_before = data[col].isna().sum()
            if nulls_before > 0:
                cleaned_data[col] = data[col].interpolate(method=method,
                    limit=limit, limit_direction=limit_direction)
                nulls_after = cleaned_data[col].isna().sum()
                modifications[col] = {'count': nulls_before - nulls_after,
                    'method': 'impute_interpolate', 'params': {'method':
                    method, 'limit': limit, 'limit_direction': limit_direction}
                    }
        return cleaned_data, modifications

    @with_exception_handling
    def _clean_impute_timeseries(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """
        Time-series aware imputation that considers patterns such as
        seasonality and trend when filling missing values
        """
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        timestamp_col = params.get('timestamp_col', None)
        cleaned_data = data.copy()
        modifications = {}
        if timestamp_col and timestamp_col in data.columns:
            if not pd.api.types.is_datetime64_dtype(data[timestamp_col]):
                try:
                    timestamp_values = pd.to_datetime(data[timestamp_col])
                    has_timestamp = True
                except:
                    self.logger.warning(
                        f'Failed to convert {timestamp_col} to datetime')
                    has_timestamp = False
            else:
                timestamp_values = data[timestamp_col]
                has_timestamp = True
            if has_timestamp:
                data_with_time_idx = cleaned_data.copy()
                data_with_time_idx.index = timestamp_values
                freq = params.get('freq', None)
                method = params.get('method', 'time')
                limit = params.get('limit', None)
                for col in cols:
                    nulls_before = data[col].isna().sum()
                    if nulls_before > 0:
                        filled_values = data_with_time_idx[col].interpolate(
                            method=method, limit=limit, freq=freq)
                        cleaned_data[col] = filled_values.values
                        nulls_after = cleaned_data[col].isna().sum()
                        modifications[col] = {'count': nulls_before -
                            nulls_after, 'method': 'impute_timeseries',
                            'params': {'method': method, 'limit': limit,
                            'freq': freq}}
        else:
            self.logger.warning(
                'No valid timestamp column for time-series imputation, falling back to regular interpolation'
                )
            return self._clean_impute_interpolate(data, target_columns, params)
        return cleaned_data, modifications

    def _clean_outlier_winsorize(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Cap outliers at specified percentiles (winsorizing)"""
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        lower_percentile = params.get('lower', 0.05)
        upper_percentile = params.get('upper', 0.95)
        modifications = {}
        for col in cols:
            values = data[col].dropna()
            lower_bound = values.quantile(lower_percentile)
            upper_bound = values.quantile(upper_percentile)
            lower_mask = cleaned_data[col] < lower_bound
            upper_mask = cleaned_data[col] > upper_bound
            outlier_count = lower_mask.sum() + upper_mask.sum()
            if outlier_count > 0:
                cleaned_data.loc[lower_mask, col] = lower_bound
                cleaned_data.loc[upper_mask, col] = upper_bound
                modifications[col] = {'count': outlier_count, 'method':
                    'outlier_winsorize', 'params': {'lower_percentile':
                    lower_percentile, 'upper_percentile': upper_percentile,
                    'lower_bound': lower_bound, 'upper_bound': upper_bound}}
        return cleaned_data, modifications

    def _clean_outlier_clip(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Clip values outside specified range"""
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        modifications = {}
        for col in cols:
            lower = params.get(f'lower_{col}', params.get('lower', None))
            upper = params.get(f'upper_{col}', params.get('upper', None))
            if lower is None and upper is None:
                n_std = params.get('n_std', 3)
                mean = data[col].mean()
                std = data[col].std()
                if not np.isnan(mean) and not np.isnan(std):
                    lower = mean - n_std * std
                    upper = mean + n_std * std
                else:
                    continue
            outside_count = 0
            if lower is not None:
                outside_count += (cleaned_data[col] < lower).sum()
            if upper is not None:
                outside_count += (cleaned_data[col] > upper).sum()
            if outside_count > 0:
                cleaned_data[col] = cleaned_data[col].clip(lower=lower,
                    upper=upper)
                modifications[col] = {'count': outside_count, 'method':
                    'outlier_clip', 'params': {'lower': lower, 'upper': upper}}
        return cleaned_data, modifications

    def _clean_transform_log(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Apply log transformation to reduce skewness"""
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        modifications = {}
        offset = params.get('offset', 1.0)
        for col in cols:
            if data[col].isna().all():
                continue
            min_val = data[col].min()
            if min_val is not None and min_val <= 0:
                col_offset = abs(min_val) + offset
            else:
                col_offset = 0
            cleaned_data[col] = np.log(data[col] + col_offset)
            modifications[col] = {'count': len(data[col].dropna()),
                'method': 'transform_log', 'params': {'offset': col_offset}}
        return cleaned_data, modifications

    def _clean_transform_sqrt(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Apply square root transformation"""
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return data, {}
        cleaned_data = data.copy()
        modifications = {}
        offset = params.get('offset', 0.0)
        for col in cols:
            if data[col].isna().all():
                continue
            min_val = data[col].min()
            if min_val is not None and min_val < 0:
                col_offset = abs(min_val) + offset
            else:
                col_offset = 0
            cleaned_data[col] = np.sqrt(data[col] + col_offset)
            modifications[col] = {'count': len(data[col].dropna()),
                'method': 'transform_sqrt', 'params': {'offset': col_offset}}
        return cleaned_data, modifications

    def _clean_correct_ohlc(self, data: pd.DataFrame, target_columns:
        Optional[List[str]], params: Dict[str, Any]) ->Tuple[pd.DataFrame,
        Dict[str, Dict[str, Any]]]:
        """Correct OHLC price data integrity issues"""
        open_col = params.get('open', 'open')
        high_col = params.get('high', 'high')
        low_col = params.get('low', 'low')
        close_col = params.get('close', 'close')
        required_cols = [open_col, high_col, low_col, close_col]
        missing_cols = [col for col in required_cols if col not in data.columns
            ]
        if missing_cols:
            self.logger.warning(
                f'Missing required OHLC columns: {missing_cols}')
            return data, {}
        cleaned_data = data.copy()
        modifications = {}
        correction_count = 0
        high_fix_mask = (cleaned_data[high_col] < cleaned_data[open_col]) | (
            cleaned_data[high_col] < cleaned_data[close_col])
        high_fix_count = high_fix_mask.sum()
        if high_fix_count > 0:
            cleaned_data.loc[high_fix_mask, high_col] = cleaned_data.loc[
                high_fix_mask, [open_col, close_col, high_col]].max(axis=1)
            correction_count += high_fix_count
        low_fix_mask = (cleaned_data[low_col] > cleaned_data[open_col]) | (
            cleaned_data[low_col] > cleaned_data[close_col])
        low_fix_count = low_fix_mask.sum()
        if low_fix_count > 0:
            cleaned_data.loc[low_fix_mask, low_col] = cleaned_data.loc[
                low_fix_mask, [open_col, close_col, low_col]].min(axis=1)
            correction_count += low_fix_count
        if correction_count > 0:
            modifications['ohlc_corrections'] = {'count': correction_count,
                'method': 'correct_ohlc', 'params': {'high_corrections':
                int(high_fix_count), 'low_corrections': int(low_fix_count)}}
        return cleaned_data, modifications

    def clean_forex_data(self, data: pd.DataFrame, data_type: str='ohlc'
        ) ->CleanedDataResult:
        """
        Apply default cleaning strategy for Forex data
        
        Args:
            data: DataFrame to clean
            data_type: 'ohlc' or 'tick'
            
        Returns:
            Cleaned data and information about changes
        """
        actions = []
        if data_type.lower() == 'ohlc':
            actions = [CleaningAction(method=CleaningMethod.CORRECT_OHLC,
                params={'open': 'open', 'high': 'high', 'low': 'low',
                'close': 'close'}), CleaningAction(method=CleaningMethod.
                IMPUTE_FFILL, target_columns=['open', 'high', 'low',
                'close'], params={'limit': 5}), CleaningAction(method=
                CleaningMethod.IMPUTE_INTERPOLATE, target_columns=['open',
                'high', 'low', 'close'], params={'method': 'linear',
                'limit': 10}), CleaningAction(method=CleaningMethod.
                OUTLIER_CLIP, target_columns=['volume'] if 'volume' in data
                .columns else [], params={'lower': 0}), CleaningAction(
                method=CleaningMethod.OUTLIER_CLIP, target_columns=['open',
                'high', 'low', 'close'], params={'n_std': 5})]
        elif data_type.lower() == 'tick':
            actions = [CleaningAction(method=CleaningMethod.IMPUTE_FFILL,
                params={'limit': 5}), CleaningAction(method=CleaningMethod.
                OUTLIER_CLIP, target_columns=['spread'] if 'spread' in data
                .columns else [], params={'lower': 0}), CleaningAction(
                method=CleaningMethod.OUTLIER_CLIP, target_columns=['bid',
                'ask'] if all(col in data.columns for col in ['bid', 'ask']
                ) else [], params={'n_std': 5})]
        else:
            self.logger.warning(f'Unknown data type: {data_type}')
            return CleanedDataResult(data=data.copy(), original_data=data.
                copy(), cleaning_actions=[], modifications={})
        return self.clean_data(data, actions)

    @with_exception_handling
    def clean_with_isolation_forest(self, data: pd.DataFrame,
        target_columns: Optional[List[str]]=None, contamination: float=0.05
        ) ->CleanedDataResult:
        """
        Clean data using Isolation Forest for anomaly detection
        
        Args:
            data: DataFrame to clean
            target_columns: Columns to analyze, or None for all numeric
            contamination: Expected proportion of anomalies
            
        Returns:
            Cleaned data with anomalies removed
        """
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and
                pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
        if not cols:
            return CleanedDataResult(data=data.copy(), original_data=data.
                copy(), cleaning_actions=[], modifications={})
        original_data = data.copy()
        cleaned_data = data.copy()
        try:
            X = data[cols].copy()
            X = X.fillna(X.mean())
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            model = IsolationForest(contamination=contamination,
                random_state=42)
            anomaly_labels = model.fit_predict(X_scaled)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            normal_indices = np.where(anomaly_labels == 1)[0]
            cleaned_data = data.iloc[normal_indices].copy()
            action = CleaningAction(method=CleaningMethod.DROP,
                target_columns=cols, params={'anomaly_detection':
                'isolation_forest', 'contamination': contamination})
            modifications = {'anomaly_removal': {'count': len(
                anomaly_indices), 'method': 'isolation_forest', 'params': {
                'contamination': contamination, 'features_used': cols}}}
            return CleanedDataResult(data=cleaned_data, original_data=
                original_data, cleaning_actions=[action], modifications=
                modifications)
        except Exception as e:
            self.logger.error(f'Error in Isolation Forest cleaning: {str(e)}')
            return CleanedDataResult(data=data.copy(), original_data=
                original_data, cleaning_actions=[], modifications={'error':
                {'message': str(e), 'type': 'isolation_forest_error'}})
