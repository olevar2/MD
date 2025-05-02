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


class CleaningMethod(Enum):
    """Cleaning methods for different types of issues"""
    DROP = "drop"                     # Remove problematic rows
    IMPUTE_MEAN = "impute_mean"       # Replace with mean value
    IMPUTE_MEDIAN = "impute_median"   # Replace with median value
    IMPUTE_MODE = "impute_mode"       # Replace with most common value
    IMPUTE_KNN = "impute_knn"         # Replace using K-nearest neighbors
    IMPUTE_FFILL = "impute_ffill"     # Forward fill (carry last value forward)
    IMPUTE_BFILL = "impute_bfill"     # Backward fill (carry next value backward)
    IMPUTE_INTERPOLATE = "impute_interpolate"  # Linear interpolation
    IMPUTE_TIMESERIES = "impute_timeseries"    # Time-series aware imputation
    OUTLIER_WINSORIZE = "outlier_winsorize"    # Cap outliers at percentile
    OUTLIER_CLIP = "outlier_clip"              # Clip values outside range
    TRANSFORM_LOG = "transform_log"            # Apply log transformation
    TRANSFORM_SQRT = "transform_sqrt"          # Apply square root transformation
    CORRECT_OHLC = "correct_ohlc"              # Fix OHLC price integrity issues


class CleaningAction:
    """Represents a cleaning action to be performed"""
    
    def __init__(
        self,
        method: CleaningMethod,
        target_columns: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ):
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
        
    def __str__(self) -> str:
        """String representation"""
        cols_str = ", ".join(self.target_columns) if self.target_columns else "all appropriate"
        return f"CleaningAction({self.method.value} on {cols_str})"


class CleanedDataResult:
    """Results of a data cleaning operation"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        original_data: pd.DataFrame,
        cleaning_actions: List[CleaningAction],
        modifications: Dict[str, Any]
    ):
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
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operations"""
        return {
            'rows_before': len(self.original_data),
            'rows_after': len(self.data),
            'rows_dropped': len(self.original_data) - len(self.data),
            'actions_applied': len(self.cleaning_actions),
            'columns_modified': list(self.modifications.keys()),
            'modification_counts': {
                col: details.get('count', 0)
                for col, details in self.modifications.items()
            },
            'timestamp': self.timestamp.isoformat()
        }
        
    def get_diff_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics on differences between original and cleaned data"""
        diff_stats = {}
        
        # Only include numeric columns that exist in both frames
        numeric_cols = [
            col for col in self.data.columns 
            if col in self.original_data.columns and
            pd.api.types.is_numeric_dtype(self.data[col]) and 
            pd.api.types.is_numeric_dtype(self.original_data[col])
        ]
        
        for col in numeric_cols:
            # Create a mask for rows that exist in both frames
            if len(self.data) != len(self.original_data):
                # Find common indices if row counts differ
                common_idx = self.data.index.intersection(self.original_data.index)
                orig_values = self.original_data.loc[common_idx, col]
                cleaned_values = self.data.loc[common_idx, col]
            else:
                orig_values = self.original_data[col]
                cleaned_values = self.data[col]
                
            # Calculate differences where both values are not null
            valid_mask = (~orig_values.isna()) & (~cleaned_values.isna())
            if valid_mask.sum() > 0:
                diffs = cleaned_values[valid_mask] - orig_values[valid_mask]
                
                diff_stats[col] = {
                    'mean_diff': float(diffs.mean()) if len(diffs) > 0 else 0.0,
                    'max_diff': float(diffs.abs().max()) if len(diffs) > 0 else 0.0,
                    'pct_changed': float((diffs != 0).sum() / len(diffs)) if len(diffs) > 0 else 0.0
                }
        
        return diff_stats


class DataCleaningEngine:
    """
    Advanced cleaning engine for financial data with multiple
    specialized cleaning strategies and detailed tracking of changes
    """
    
    def __init__(self):
        """Initialize data cleaning engine"""
        self.logger = logging.getLogger(__name__)
        
    def clean_data(
        self,
        data: pd.DataFrame,
        actions: List[CleaningAction],
        copy: bool = True
    ) -> CleanedDataResult:
        """
        Clean data according to specified actions
        
        Args:
            data: DataFrame to clean
            actions: List of cleaning actions to apply
            copy: Whether to work on a copy of the data
            
        Returns:
            Cleaned data and information about changes
        """
        # Keep original for reference
        original_data = data.copy()
        
        # Work on a copy if requested
        if copy:
            data = data.copy()
            
        # Track modifications
        modifications = {}
        
        # Apply each cleaning action in sequence
        for action in actions:
            try:
                # Apply the cleaning method
                data, mod_details = self._apply_cleaning_method(
                    data, action.method, action.target_columns, action.params
                )
                
                # Update modifications tracking
                for col, details in mod_details.items():
                    if col in modifications:
                        modifications[col]['count'] += details.get('count', 0)
                    else:
                        modifications[col] = details
                        
            except Exception as e:
                self.logger.error(f"Error applying cleaning action {action.method.value}: {str(e)}")
                
        # Return cleaned data and details
        return CleanedDataResult(
            data=data,
            original_data=original_data,
            cleaning_actions=actions,
            modifications=modifications
        )
        
    def _apply_cleaning_method(
        self,
        data: pd.DataFrame,
        method: CleaningMethod,
        target_columns: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
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
        
        # Handle different cleaning methods
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
            self.logger.warning(f"Unknown cleaning method: {method}")
            return data, {}
            
    def _clean_drop(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Drop rows with issues"""
        # Default to all columns if not specified
        cols = target_columns if target_columns else data.columns.tolist()
        
        # Only use columns that exist
        cols = [col for col in cols if col in data.columns]
        
        # Check parameters
        how = params.get('how', 'any')  # 'any' or 'all'
        threshold = params.get('threshold', None)  # Min non-NA values
        
        # Count rows before
        rows_before = len(data)
        
        # Drop NA rows
        cleaned_data = data.dropna(subset=cols, how=how, thresh=threshold)
        
        # Count modifications
        rows_dropped = rows_before - len(cleaned_data)
        
        # Create modifications record
        modifications = {
            'dropped_rows': {
                'count': rows_dropped,
                'method': 'drop',
                'params': {
                    'how': how,
                    'threshold': threshold
                }
            }
        }
        
        return cleaned_data, modifications
        
    def _clean_impute_mean(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Impute missing values with mean"""
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Set up imputer
        imputer = SimpleImputer(strategy='mean')
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Count nulls before
            nulls_before = data[col].isna().sum()
            
            if nulls_before > 0:
                # Reshape for imputer (2D array)
                values = data[col].values.reshape(-1, 1)
                
                # Impute values
                imputed_values = imputer.fit_transform(values).flatten()
                
                # Update data
                cleaned_data[col] = imputed_values
                
                # Record modifications
                mean_value = data[col].mean()
                modifications[col] = {
                    'count': nulls_before,
                    'method': 'impute_mean',
                    'params': {
                        'value': mean_value
                    }
                }
                
        return cleaned_data, modifications
        
    def _clean_impute_median(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Impute missing values with median"""
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Set up imputer
        imputer = SimpleImputer(strategy='median')
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Count nulls before
            nulls_before = data[col].isna().sum()
            
            if nulls_before > 0:
                # Reshape for imputer (2D array)
                values = data[col].values.reshape(-1, 1)
                
                # Impute values
                imputed_values = imputer.fit_transform(values).flatten()
                
                # Update data
                cleaned_data[col] = imputed_values
                
                # Record modifications
                median_value = data[col].median()
                modifications[col] = {
                    'count': nulls_before,
                    'method': 'impute_median',
                    'params': {
                        'value': median_value
                    }
                }
                
        return cleaned_data, modifications
        
    def _clean_impute_mode(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Impute missing values with mode (most frequent)"""
        # Default to all columns if not specified
        cols = target_columns if target_columns else data.columns.tolist()
        
        # Only use columns that exist
        cols = [col for col in cols if col in data.columns]
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Count nulls before
            nulls_before = data[col].isna().sum()
            
            if nulls_before > 0:
                # Calculate mode
                mode_value = data[col].mode().iloc[0] if not data[col].dropna().empty else None
                
                if mode_value is not None:
                    # Fill nulls with mode
                    cleaned_data[col] = data[col].fillna(mode_value)
                    
                    # Record modifications
                    modifications[col] = {
                        'count': nulls_before,
                        'method': 'impute_mode',
                        'params': {
                            'value': mode_value
                        }
                    }
                
        return cleaned_data, modifications
        
    def _clean_impute_knn(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Impute missing values using K-Nearest Neighbors"""
        # This only works on numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if target_columns:
            cols = [col for col in target_columns if col in numeric_cols]
        else:
            cols = numeric_cols
            
        if not cols:
            return data, {}
            
        # Check if we have any missing values to impute
        has_nulls = any(data[col].isna().any() for col in cols)
        if not has_nulls:
            return data, {}
            
        # Parameters for KNN
        n_neighbors = params.get('n_neighbors', 5)
        weights = params.get('weights', 'uniform')
        
        # Create copy of data
        cleaned_data = data.copy()
        
        # Track modifications per column
        modifications = {}
        
        # Only use numeric columns for imputation
        numeric_data = data[numeric_cols].copy()
        
        # Count nulls before
        nulls_before = {col: data[col].isna().sum() for col in cols}
        
        try:
            # Set up and fit KNN imputer
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imputed_values = imputer.fit_transform(numeric_data)
            
            # Update data
            imputed_df = pd.DataFrame(
                imputed_values, 
                columns=numeric_cols, 
                index=data.index
            )
            
            # Only update target columns
            for col in cols:
                if nulls_before[col] > 0:
                    cleaned_data[col] = imputed_df[col]
                    
                    # Record modifications
                    modifications[col] = {
                        'count': nulls_before[col],
                        'method': 'impute_knn',
                        'params': {
                            'n_neighbors': n_neighbors,
                            'weights': weights
                        }
                    }
                    
        except Exception as e:
            self.logger.error(f"Error in KNN imputation: {str(e)}")
                
        return cleaned_data, modifications
        
    def _clean_impute_ffill(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Fill missing values by carrying forward last valid value"""
        # Default to all columns if not specified
        cols = target_columns if target_columns else data.columns.tolist()
        
        # Only use columns that exist
        cols = [col for col in cols if col in data.columns]
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Get optional limit parameter (max consecutive fills)
        limit = params.get('limit', None)
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Count nulls before
            nulls_before = data[col].isna().sum()
            
            if nulls_before > 0:
                # Forward fill
                cleaned_data[col] = data[col].ffill(limit=limit)
                
                # Count nulls after
                nulls_after = cleaned_data[col].isna().sum()
                
                # Record modifications
                modifications[col] = {
                    'count': nulls_before - nulls_after,
                    'method': 'impute_ffill',
                    'params': {
                        'limit': limit
                    }
                }
                
        return cleaned_data, modifications
        
    def _clean_impute_bfill(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Fill missing values by carrying backward from next valid value"""
        # Default to all columns if not specified
        cols = target_columns if target_columns else data.columns.tolist()
        
        # Only use columns that exist
        cols = [col for col in cols if col in data.columns]
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Get optional limit parameter (max consecutive fills)
        limit = params.get('limit', None)
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Count nulls before
            nulls_before = data[col].isna().sum()
            
            if nulls_before > 0:
                # Backward fill
                cleaned_data[col] = data[col].bfill(limit=limit)
                
                # Count nulls after
                nulls_after = cleaned_data[col].isna().sum()
                
                # Record modifications
                modifications[col] = {
                    'count': nulls_before - nulls_after,
                    'method': 'impute_bfill',
                    'params': {
                        'limit': limit
                    }
                }
                
        return cleaned_data, modifications
        
    def _clean_impute_interpolate(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Fill missing values using interpolation"""
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Get parameters
        method = params.get('method', 'linear')
        limit = params.get('limit', None)
        limit_direction = params.get('limit_direction', 'forward')
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Count nulls before
            nulls_before = data[col].isna().sum()
            
            if nulls_before > 0:
                # Interpolate
                cleaned_data[col] = data[col].interpolate(
                    method=method,
                    limit=limit,
                    limit_direction=limit_direction
                )
                
                # Count nulls after
                nulls_after = cleaned_data[col].isna().sum()
                
                # Record modifications
                modifications[col] = {
                    'count': nulls_before - nulls_after,
                    'method': 'impute_interpolate',
                    'params': {
                        'method': method,
                        'limit': limit,
                        'limit_direction': limit_direction
                    }
                }
                
        return cleaned_data, modifications
        
    def _clean_impute_timeseries(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Time-series aware imputation that considers patterns such as
        seasonality and trend when filling missing values
        """
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # We need a datetime index for proper time-series imputation
        timestamp_col = params.get('timestamp_col', None)
        
        # Create copy of data
        cleaned_data = data.copy()
        
        # Track modifications
        modifications = {}
        
        # If we have a timestamp column, set it as index temporarily
        if timestamp_col and timestamp_col in data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(data[timestamp_col]):
                try:
                    timestamp_values = pd.to_datetime(data[timestamp_col])
                    has_timestamp = True
                except:
                    self.logger.warning(f"Failed to convert {timestamp_col} to datetime")
                    has_timestamp = False
            else:
                timestamp_values = data[timestamp_col]
                has_timestamp = True
                
            # If we have valid timestamps, create a time-indexed frame
            if has_timestamp:
                data_with_time_idx = cleaned_data.copy()
                data_with_time_idx.index = timestamp_values
                
                # Get frequency and method parameters
                freq = params.get('freq', None)  # Infer or specify ('D', 'H', etc.)
                method = params.get('method', 'time')
                limit = params.get('limit', None)
                
                for col in cols:
                    # Count nulls before
                    nulls_before = data[col].isna().sum()
                    
                    if nulls_before > 0:
                        # Interpolate with time-series awareness
                        filled_values = data_with_time_idx[col].interpolate(
                            method=method,
                            limit=limit,
                            freq=freq
                        )
                        
                        # Apply back to original data
                        cleaned_data[col] = filled_values.values
                        
                        # Count nulls after
                        nulls_after = cleaned_data[col].isna().sum()
                        
                        # Record modifications
                        modifications[col] = {
                            'count': nulls_before - nulls_after,
                            'method': 'impute_timeseries',
                            'params': {
                                'method': method,
                                'limit': limit,
                                'freq': freq
                            }
                        }
        else:
            # Fall back to regular interpolation
            self.logger.warning("No valid timestamp column for time-series imputation, falling back to regular interpolation")
            return self._clean_impute_interpolate(data, target_columns, params)
                
        return cleaned_data, modifications
        
    def _clean_outlier_winsorize(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Cap outliers at specified percentiles (winsorizing)"""
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Get parameters
        lower_percentile = params.get('lower', 0.05)  # Default 5th percentile
        upper_percentile = params.get('upper', 0.95)  # Default 95th percentile
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Get values and percentile thresholds
            values = data[col].dropna()
            lower_bound = values.quantile(lower_percentile)
            upper_bound = values.quantile(upper_percentile)
            
            # Count values outside bounds
            lower_mask = cleaned_data[col] < lower_bound
            upper_mask = cleaned_data[col] > upper_bound
            outlier_count = lower_mask.sum() + upper_mask.sum()
            
            if outlier_count > 0:
                # Apply winsorizing
                cleaned_data.loc[lower_mask, col] = lower_bound
                cleaned_data.loc[upper_mask, col] = upper_bound
                
                # Record modifications
                modifications[col] = {
                    'count': outlier_count,
                    'method': 'outlier_winsorize',
                    'params': {
                        'lower_percentile': lower_percentile,
                        'upper_percentile': upper_percentile,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                }
                
        return cleaned_data, modifications
        
    def _clean_outlier_clip(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Clip values outside specified range"""
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Track modifications
        modifications = {}
        
        for col in cols:
            # Get clip thresholds
            lower = params.get(f'lower_{col}', params.get('lower', None))
            upper = params.get(f'upper_{col}', params.get('upper', None))
            
            if lower is None and upper is None:
                # Try to auto-calculate based on std deviations
                n_std = params.get('n_std', 3)
                mean = data[col].mean()
                std = data[col].std()
                
                if not np.isnan(mean) and not np.isnan(std):
                    lower = mean - n_std * std
                    upper = mean + n_std * std
                else:
                    continue  # Skip this column
            
            # Count values outside bounds
            outside_count = 0
            if lower is not None:
                outside_count += (cleaned_data[col] < lower).sum()
            if upper is not None:
                outside_count += (cleaned_data[col] > upper).sum()
            
            if outside_count > 0:
                # Apply clipping
                cleaned_data[col] = cleaned_data[col].clip(lower=lower, upper=upper)
                
                # Record modifications
                modifications[col] = {
                    'count': outside_count,
                    'method': 'outlier_clip',
                    'params': {
                        'lower': lower,
                        'upper': upper
                    }
                }
                
        return cleaned_data, modifications
        
    def _clean_transform_log(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Apply log transformation to reduce skewness"""
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Track modifications
        modifications = {}
        
        # Get offset param (to handle zeros/negatives)
        offset = params.get('offset', 1.0)
        
        for col in cols:
            # Skip if all values are NaN
            if data[col].isna().all():
                continue
                
            # Check if we have negative or zero values
            min_val = data[col].min()
            
            # Calculate offset if needed
            if min_val is not None and min_val <= 0:
                col_offset = abs(min_val) + offset
            else:
                col_offset = 0
                
            # Apply transformation
            cleaned_data[col] = np.log(data[col] + col_offset)
            
            # Record modifications
            modifications[col] = {
                'count': len(data[col].dropna()),
                'method': 'transform_log',
                'params': {
                    'offset': col_offset
                }
            }
                
        return cleaned_data, modifications
        
    def _clean_transform_sqrt(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Apply square root transformation"""
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Track modifications
        modifications = {}
        
        # Get offset param (to handle negatives)
        offset = params.get('offset', 0.0)
        
        for col in cols:
            # Skip if all values are NaN
            if data[col].isna().all():
                continue
                
            # Check if we have negative values
            min_val = data[col].min()
            
            # Calculate offset if needed
            if min_val is not None and min_val < 0:
                col_offset = abs(min_val) + offset
            else:
                col_offset = 0
                
            # Apply transformation
            cleaned_data[col] = np.sqrt(data[col] + col_offset)
            
            # Record modifications
            modifications[col] = {
                'count': len(data[col].dropna()),
                'method': 'transform_sqrt',
                'params': {
                    'offset': col_offset
                }
            }
                
        return cleaned_data, modifications
        
    def _clean_correct_ohlc(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]],
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Correct OHLC price data integrity issues"""
        # Get column mappings
        open_col = params.get('open', 'open')
        high_col = params.get('high', 'high')
        low_col = params.get('low', 'low')
        close_col = params.get('close', 'close')
        
        # Check if all required columns exist
        required_cols = [open_col, high_col, low_col, close_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing required OHLC columns: {missing_cols}")
            return data, {}
            
        # Create copy of data
        cleaned_data = data.copy()
        
        # Track modifications and counts
        modifications = {}
        correction_count = 0
        
        # Fix high values (should be >= open, close)
        high_fix_mask = ((cleaned_data[high_col] < cleaned_data[open_col]) | 
                         (cleaned_data[high_col] < cleaned_data[close_col]))
        high_fix_count = high_fix_mask.sum()
        
        if high_fix_count > 0:
            # Set high to max of open, close, current high
            cleaned_data.loc[high_fix_mask, high_col] = cleaned_data.loc[
                high_fix_mask, [open_col, close_col, high_col]
            ].max(axis=1)
            correction_count += high_fix_count
        
        # Fix low values (should be <= open, close)
        low_fix_mask = ((cleaned_data[low_col] > cleaned_data[open_col]) | 
                        (cleaned_data[low_col] > cleaned_data[close_col]))
        low_fix_count = low_fix_mask.sum()
        
        if low_fix_count > 0:
            # Set low to min of open, close, current low
            cleaned_data.loc[low_fix_mask, low_col] = cleaned_data.loc[
                low_fix_mask, [open_col, close_col, low_col]
            ].min(axis=1)
            correction_count += low_fix_count
            
        # Record modifications if any made
        if correction_count > 0:
            modifications['ohlc_corrections'] = {
                'count': correction_count,
                'method': 'correct_ohlc',
                'params': {
                    'high_corrections': int(high_fix_count),
                    'low_corrections': int(low_fix_count)
                }
            }
                
        return cleaned_data, modifications
        
    def clean_forex_data(
        self,
        data: pd.DataFrame,
        data_type: str = 'ohlc'
    ) -> CleanedDataResult:
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
            # Standard OHLC cleaning
            actions = [
                # Fix integrity issues
                CleaningAction(
                    method=CleaningMethod.CORRECT_OHLC,
                    params={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}
                ),
                
                # Handle missing values
                CleaningAction(
                    method=CleaningMethod.IMPUTE_FFILL,
                    target_columns=['open', 'high', 'low', 'close'],
                    params={'limit': 5}
                ),
                
                # Fall back to interpolation for remaining nulls
                CleaningAction(
                    method=CleaningMethod.IMPUTE_INTERPOLATE,
                    target_columns=['open', 'high', 'low', 'close'],
                    params={'method': 'linear', 'limit': 10}
                ),
                
                # Clean volume if present (can't be negative)
                CleaningAction(
                    method=CleaningMethod.OUTLIER_CLIP,
                    target_columns=['volume'] if 'volume' in data.columns else [],
                    params={'lower': 0}
                ),
                
                # Clip extreme outliers
                CleaningAction(
                    method=CleaningMethod.OUTLIER_CLIP,
                    target_columns=['open', 'high', 'low', 'close'],
                    params={'n_std': 5}
                )
            ]
        
        elif data_type.lower() == 'tick':
            # Tick data cleaning
            actions = [
                # Handle missing values
                CleaningAction(
                    method=CleaningMethod.IMPUTE_FFILL,
                    params={'limit': 5}
                ),
                
                # Fix bid-ask relationship if needed
                CleaningAction(
                    method=CleaningMethod.OUTLIER_CLIP,
                    target_columns=['spread'] if 'spread' in data.columns else [],
                    params={'lower': 0}
                ),
                
                # Clip extreme outliers
                CleaningAction(
                    method=CleaningMethod.OUTLIER_CLIP,
                    target_columns=['bid', 'ask'] if all(col in data.columns for col in ['bid', 'ask']) else [],
                    params={'n_std': 5}
                )
            ]
        
        else:
            self.logger.warning(f"Unknown data type: {data_type}")
            return CleanedDataResult(
                data=data.copy(),
                original_data=data.copy(),
                cleaning_actions=[],
                modifications={}
            )
            
        # Apply the cleaning actions
        return self.clean_data(data, actions)

    def clean_with_isolation_forest(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        contamination: float = 0.05
    ) -> CleanedDataResult:
        """
        Clean data using Isolation Forest for anomaly detection
        
        Args:
            data: DataFrame to clean
            target_columns: Columns to analyze, or None for all numeric
            contamination: Expected proportion of anomalies
            
        Returns:
            Cleaned data with anomalies removed
        """
        # Default to numeric columns if not specified
        if target_columns:
            cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        else:
            cols = data.select_dtypes(include=['number']).columns.tolist()
            
        if not cols:
            # No valid columns to process
            return CleanedDataResult(
                data=data.copy(),
                original_data=data.copy(),
                cleaning_actions=[],
                modifications={}
            )
            
        # Keep original for reference
        original_data = data.copy()
        cleaned_data = data.copy()
        
        try:
            # Prepare data for Isolation Forest
            X = data[cols].copy()
            
            # Handle missing values for model
            X = X.fillna(X.mean())
            
            # Standardize data
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            
            # Fit and predict
            anomaly_labels = model.fit_predict(X_scaled)
            
            # -1 indicates anomalies, 1 indicates normal data
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            normal_indices = np.where(anomaly_labels == 1)[0]
            
            # Create a new DataFrame without anomalies
            cleaned_data = data.iloc[normal_indices].copy()
            
            # Create cleaning action (for record)
            action = CleaningAction(
                method=CleaningMethod.DROP,
                target_columns=cols,
                params={
                    'anomaly_detection': 'isolation_forest',
                    'contamination': contamination
                }
            )
            
            # Create modifications record
            modifications = {
                'anomaly_removal': {
                    'count': len(anomaly_indices),
                    'method': 'isolation_forest',
                    'params': {
                        'contamination': contamination,
                        'features_used': cols
                    }
                }
            }
            
            return CleanedDataResult(
                data=cleaned_data,
                original_data=original_data,
                cleaning_actions=[action],
                modifications=modifications
            )
            
        except Exception as e:
            self.logger.error(f"Error in Isolation Forest cleaning: {str(e)}")
            return CleanedDataResult(
                data=data.copy(),
                original_data=original_data,
                cleaning_actions=[],
                modifications={
                    'error': {
                        'message': str(e),
                        'type': 'isolation_forest_error'
                    }
                }
            )
