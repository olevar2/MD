"""
DataCleaningEngine module.

Provides data cleaning and imputation strategies for handling missing values,
outliers, and other data quality issues.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from core_foundations.utils.logger import get_logger
from data_pipeline_service.models.schemas import OHLCVData, TickData

# Initialize logger
logger = get_logger("data-cleaning-engine")


class DataType(Enum):
    """Enum representing different data types for cleaning strategies."""
    
    OHLCV = "ohlcv"
    TICK = "tick"
    GENERIC = "generic"


class ImputationStrategy(ABC):
    """Base class for imputation strategies."""
    
    @abstractmethod
    def impute(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Impute missing values in the specified column.
        
        Args:
            data: Data to impute (pandas DataFrame or list of dicts)
            column: Column name to impute
            
        Returns:
            Data with imputed values
        """
        pass


class MeanImputation(ImputationStrategy):
    """Impute missing values with mean of the column."""
    
    def impute(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Impute missing values with mean of the column.
        
        Args:
            data: Data to impute
            column: Column name to impute
            
        Returns:
            Data with imputed values
        """
        if isinstance(data, list):
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            df[column] = df[column].fillna(df[column].mean())
            return df.to_dict('records')
        else:
            # Handle pandas DataFrame
            data[column] = data[column].fillna(data[column].mean())
            return data


class MedianImputation(ImputationStrategy):
    """Impute missing values with median of the column."""
    
    def impute(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Impute missing values with median of the column.
        
        Args:
            data: Data to impute
            column: Column name to impute
            
        Returns:
            Data with imputed values
        """
        if isinstance(data, list):
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            df[column] = df[column].fillna(df[column].median())
            return df.to_dict('records')
        else:
            # Handle pandas DataFrame
            data[column] = data[column].fillna(data[column].median())
            return data


class ForwardFillImputation(ImputationStrategy):
    """Impute missing values with last known value (forward fill)."""

    def impute(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Impute missing values using forward fill.

        Args:
            data: Data to impute
            column: Column name to impute

        Returns:
            Data with imputed values
        """
        if isinstance(data, list):
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            # Ensure data is sorted by time if applicable before ffill
            if 'timestamp' in df.columns:
                 try:
                     if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                         df['timestamp'] = pd.to_datetime(df['timestamp'])
                     df = df.sort_values('timestamp')
                 except Exception as e:
                     logger.warning(f"Could not sort by timestamp for ffill: {e}")

            df[column] = df[column].ffill()
            return df.to_dict('records')
        else:
            # Handle pandas DataFrame
            # Ensure data is sorted by time if applicable before ffill
            if 'timestamp' in data.columns:
                 try:
                     if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                         data['timestamp'] = pd.to_datetime(data['timestamp'])
                     data = data.sort_values('timestamp')
                 except Exception as e:
                     logger.warning(f"Could not sort by timestamp for ffill: {e}")

            data[column] = data[column].ffill()
            return data


class BackwardFillImputation(ImputationStrategy):
    """Impute missing values with next known value (backward fill)."""
    
    def impute(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Impute missing values with next known value.
        
        Args:
            data: Data to impute
            column: Column name to impute
            
        Returns:
            Data with imputed values
        """
        if isinstance(data, list):
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            df[column] = df[column].fillna(method='bfill')
            return df.to_dict('records')
        else:
            # Handle pandas DataFrame
            data[column] = data[column].fillna(method='bfill')
            return data


class LinearInterpolationImputation(ImputationStrategy):
    """Impute missing values with linear interpolation."""
    
    def impute(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Impute missing values with linear interpolation.
        
        Args:
            data: Data to impute
            column: Column name to impute
            
        Returns:
            Data with imputed values
        """
        if isinstance(data, list):
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            df[column] = df[column].interpolate(method='linear')
            return df.to_dict('records')
        else:
            # Handle pandas DataFrame
            data[column] = data[column].interpolate(method='linear')
            return data


class OutlierDetectionStrategy(ABC):
    """Base class for outlier detection strategies."""
    
    @abstractmethod
    def detect(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> List[int]:
        """
        Detect outliers in the specified column.
        
        Args:
            data: Data to analyze
            column: Column name to analyze
            
        Returns:
            List of indices of outliers
        """
        pass


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    """Detect outliers using Z-score (standard deviations from mean)."""
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize with threshold.
        
        Args:
            threshold: Number of standard deviations to consider as outlier
        """
        self.threshold = threshold
    
    def detect(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> List[int]:
        """
        Detect outliers using Z-score.
        
        Args:
            data: Data to analyze
            column: Column name to analyze
            
        Returns:
            List of indices of outliers
        """
        if isinstance(data, list):
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            series = df[column]
        else:
            series = data[column]
        
        # Calculate Z-scores
        mean = series.mean()
        std = series.std()
        z_scores = abs((series - mean) / std)
        
        # Get indices where Z-score exceeds threshold
        outlier_indices = z_scores[z_scores > self.threshold].index.tolist()
        
        return outlier_indices


class IQROutlierDetection(OutlierDetectionStrategy):
    """Detect outliers using IQR (Interquartile Range)."""
    
    def __init__(self, multiplier: float = 1.5):
        """
        Initialize with IQR multiplier.
        
        Args:
            multiplier: IQR multiplier for outlier threshold
        """
        self.multiplier = multiplier
    
    def detect(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], column: str
    ) -> List[int]:
        """
        Detect outliers using IQR.
        
        Args:
            data: Data to analyze
            column: Column name to analyze
            
        Returns:
            List of indices of outliers
        """
        if isinstance(data, list):
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            series = df[column]
        else:
            series = data[column]
        
        # Calculate Q1, Q3, and IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        # Define lower and upper bounds
        lower_bound = q1 - (self.multiplier * iqr)
        upper_bound = q3 + (self.multiplier * iqr)
        
        # Find outliers
        outlier_indices = series[
            (series < lower_bound) | (series > upper_bound)
        ].index.tolist()
        
        return outlier_indices


class DataCleaningEngine:
    """
    Engine for cleaning data by handling missing values, outliers, and other issues.
    
    This engine provides methods for cleaning different types of data using
    various imputation and outlier handling strategies.
    """
    
    def __init__(self):
        """Initialize the data cleaning engine with default strategies."""
        # Default imputation strategies by data type and column
        self._default_imputation_strategies = {
            DataType.OHLCV: {
                "open": LinearInterpolationImputation(),
                "high": LinearInterpolationImputation(),
                "low": LinearInterpolationImputation(),
                "close": LinearInterpolationImputation(),
                "volume": ForwardFillImputation(),
                "default": LinearInterpolationImputation(),
            },
            DataType.TICK: {
                "bid": ForwardFillImputation(),
                "ask": ForwardFillImputation(),
                "bid_volume": MedianImputation(),
                "ask_volume": MedianImputation(),
                "default": ForwardFillImputation(),
            },
            DataType.GENERIC: {
                "default": MeanImputation(),
            }
        }
        
        # Default outlier detection strategies
        self._default_outlier_strategies = {
            DataType.OHLCV: ZScoreOutlierDetection(threshold=4.0),
            DataType.TICK: ZScoreOutlierDetection(threshold=4.0),
            DataType.GENERIC: IQROutlierDetection(),
        }
        
        # Custom strategies (can be set by users)
        self._custom_imputation_strategies = {}
        self._custom_outlier_strategies = {}
    
    def set_imputation_strategy(
        self, 
        data_type: DataType,
        column: str,
        strategy: ImputationStrategy
    ) -> None:
        """
        Set a custom imputation strategy for a specific column and data type.
        
        Args:
            data_type: Type of data
            column: Column name
            strategy: Imputation strategy to use
        """
        if data_type not in self._custom_imputation_strategies:
            self._custom_imputation_strategies[data_type] = {}
        
        self._custom_imputation_strategies[data_type][column] = strategy
        logger.info(f"Set custom imputation strategy for {data_type.value}.{column}")
    
    def set_outlier_strategy(
        self,
        data_type: DataType,
        strategy: OutlierDetectionStrategy
    ) -> None:
        """
        Set a custom outlier detection strategy for a data type.
        
        Args:
            data_type: Type of data
            strategy: Outlier detection strategy to use
        """
        self._custom_outlier_strategies[data_type] = strategy
        logger.info(f"Set custom outlier detection strategy for {data_type.value}")
    
    def get_imputation_strategy(
        self, data_type: DataType, column: str
    ) -> ImputationStrategy:
        """
        Get the appropriate imputation strategy for a column and data type.
        
        Args:
            data_type: Type of data
            column: Column name
            
        Returns:
            Imputation strategy to use
        """
        # Check for custom strategy
        if (
            data_type in self._custom_imputation_strategies and
            column in self._custom_imputation_strategies[data_type]
        ):
            return self._custom_imputation_strategies[data_type][column]
        
        # Check for default strategy for this column
        if (
            data_type in self._default_imputation_strategies and
            column in self._default_imputation_strategies[data_type]
        ):
            return self._default_imputation_strategies[data_type][column]
        
        # Use default strategy for this data type
        if data_type in self._default_imputation_strategies:
            return self._default_imputation_strategies[data_type]["default"]
        
        # Fall back to generic default
        return self._default_imputation_strategies[DataType.GENERIC]["default"]
    
    def get_outlier_strategy(self, data_type: DataType) -> OutlierDetectionStrategy:
        """
        Get the appropriate outlier detection strategy for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Outlier detection strategy to use
        """
        # Check for custom strategy
        if data_type in self._custom_outlier_strategies:
            return self._custom_outlier_strategies[data_type]
        
        # Use default strategy for this data type
        if data_type in self._default_outlier_strategies:
            return self._default_outlier_strategies[data_type]
        
        # Fall back to generic default
        return self._default_outlier_strategies[DataType.GENERIC]
    
    def clean_ohlcv_data(
        self, data: Union[List[OHLCVData], pd.DataFrame]
    ) -> Union[List[OHLCVData], pd.DataFrame]:
        """
        Clean OHLCV data by imputing missing values and handling outliers.
        
        Args:
            data: OHLCV data to clean (list of OHLCVData objects or DataFrame)
            
        Returns:
            Cleaned OHLCV data
        """
        # Convert OHLCVData objects to dict if needed
        original_type = type(data)
        if isinstance(data, list) and all(isinstance(item, OHLCVData) for item in data):
            data_dicts = [item.model_dump() for item in data]
        elif isinstance(data, pd.DataFrame):
            data_dicts = data
        else:
            data_dicts = data
        
        # Process each column that might have missing values
        for column in ["open", "high", "low", "close", "volume"]:
            # Get the appropriate imputation strategy
            strategy = self.get_imputation_strategy(DataType.OHLCV, column)
            
            # Apply the imputation
            data_dicts = strategy.impute(data_dicts, column)
        
        # Handle outlier detection and treatment for price columns
        for column in ["open", "high", "low", "close"]:
            # Detect outliers
            outlier_strategy = self.get_outlier_strategy(DataType.OHLCV)
            outlier_indices = outlier_strategy.detect(data_dicts, column)
            
            if outlier_indices:
                logger.warning(f"Detected {len(outlier_indices)} outliers in {column}")
                
                # If working with DataFrame
                if isinstance(data_dicts, pd.DataFrame):
                    # Create a mask for the outliers
                    mask = data_dicts.index.isin(outlier_indices)
                    
                    # Get the appropriate imputation strategy
                    strategy = self.get_imputation_strategy(DataType.OHLCV, column)
                    
                    # Store original values
                    original_values = data_dicts.loc[mask, column].copy()
                    
                    # Replace outliers with NaN
                    data_dicts.loc[mask, column] = np.nan
                    
                    # Impute the NaN values
                    data_dicts = strategy.impute(data_dicts, column)
                    
                    # Log the changes
                    logger.info(
                        f"Replaced {len(outlier_indices)} outliers in {column} "
                        f"(before: {original_values.tolist()}, "
                        f"after: {data_dicts.loc[mask, column].tolist()})"
                    )
        
        # Convert back to original type if necessary
        if original_type == list and all(isinstance(item, OHLCVData) for item in data):
            if isinstance(data_dicts, pd.DataFrame):
                data_dicts = data_dicts.to_dict('records')
            return [OHLCVData(**item) for item in data_dicts]
        else:
            return data_dicts
    
    def clean_tick_data(
        self, data: Union[List[TickData], pd.DataFrame]
    ) -> Union[List[TickData], pd.DataFrame]:
        """
        Clean tick data by imputing missing values and handling outliers.
        
        Args:
            data: Tick data to clean (list of TickData objects or DataFrame)
            
        Returns:
            Cleaned tick data
        """
        # Convert TickData objects to dict if needed
        original_type = type(data)
        if isinstance(data, list) and all(isinstance(item, TickData) for item in data):
            data_dicts = [item.model_dump() for item in data]
        elif isinstance(data, pd.DataFrame):
            data_dicts = data
        else:
            data_dicts = data
        
        # Process each column that might have missing values
        for column in ["bid", "ask", "bid_volume", "ask_volume"]:
            # Skip if column doesn't exist (older data might not have volumes)
            if isinstance(data_dicts, pd.DataFrame) and column not in data_dicts.columns:
                continue
            
            # Get the appropriate imputation strategy
            strategy = self.get_imputation_strategy(DataType.TICK, column)
            
            # Apply the imputation
            data_dicts = strategy.impute(data_dicts, column)
        
        # Handle outlier detection and treatment for bid/ask
        for column in ["bid", "ask"]:
            # Detect outliers
            outlier_strategy = self.get_outlier_strategy(DataType.TICK)
            outlier_indices = outlier_strategy.detect(data_dicts, column)
            
            if outlier_indices:
                logger.warning(f"Detected {len(outlier_indices)} outliers in {column}")
                
                # If working with DataFrame
                if isinstance(data_dicts, pd.DataFrame):
                    # Create a mask for the outliers
                    mask = data_dicts.index.isin(outlier_indices)
                    
                    # Get the appropriate imputation strategy
                    strategy = self.get_imputation_strategy(DataType.TICK, column)
                    
                    # Store original values
                    original_values = data_dicts.loc[mask, column].copy()
                    
                    # Replace outliers with NaN
                    data_dicts.loc[mask, column] = np.nan
                    
                    # Impute the NaN values
                    data_dicts = strategy.impute(data_dicts, column)
                    
                    # Log the changes
                    logger.info(
                        f"Replaced {len(outlier_indices)} outliers in {column} "
                        f"(before: {original_values.tolist()}, "
                        f"after: {data_dicts.loc[mask, column].tolist()})"
                    )
        
        # Convert back to original type if necessary
        if original_type == list and all(isinstance(item, TickData) for item in data):
            if isinstance(data_dicts, pd.DataFrame):
                data_dicts = data_dicts.to_dict('records')
            return [TickData(**item) for item in data_dicts]
        else:
            return data_dicts
    
    def clean_data(
        self,
        data: Union[List[Dict[str, Any]], pd.DataFrame],
        data_type: DataType,
        columns: Optional[List[str]] = None,
    ) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Generic method to clean any type of data.
        
        Args:
            data: Data to clean
            data_type: Type of data
            columns: List of columns to clean (if None, all columns are considered)
            
        Returns:
            Cleaned data
        """
        # Convert list of dicts to DataFrame for easier processing
        original_type = type(data)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Process each column
        for column in columns:
            if column not in df.columns:
                continue
                
            # Get the appropriate imputation strategy
            strategy = self.get_imputation_strategy(data_type, column)
            
            # Apply the imputation
            df = strategy.impute(df, column)
        
        # Handle outliers for each column
        for column in columns:
            if column not in df.columns:
                continue
                
            # Skip if non-numeric
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
                
            # Detect outliers
            outlier_strategy = self.get_outlier_strategy(data_type)
            outlier_indices = outlier_strategy.detect(df, column)
            
            if outlier_indices:
                logger.warning(f"Detected {len(outlier_indices)} outliers in {column}")
                
                # Create a mask for the outliers
                mask = df.index.isin(outlier_indices)
                
                # Get the appropriate imputation strategy
                strategy = self.get_imputation_strategy(data_type, column)
                
                # Store original values
                original_values = df.loc[mask, column].copy()
                
                # Replace outliers with NaN
                df.loc[mask, column] = np.nan
                
                # Impute the NaN values
                df = strategy.impute(df, column)
                
                # Log the changes
                logger.info(
                    f"Replaced {len(outlier_indices)} outliers in {column} "
                    f"(before: {original_values.tolist()}, "
                    f"after: {df.loc[mask, column].tolist()})"
                )
        
        # Convert back to original type if necessary
        if original_type == list:
            return df.to_dict('records')
        else:
            return df