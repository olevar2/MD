"""
Enhanced Memory-Optimized DataFrame Module

This module provides a memory-efficient wrapper for pandas DataFrame with lazy evaluation
and optimized data types for performance-critical operations.

Features:
- Automatic data type optimization
- Lazy evaluation of computed columns
- Memory-efficient views instead of copies
- Reduced memory footprint
- Resilience patterns integration
- Exception handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
import logging
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class MemoryOptimizedDataFrame:
    """
    Memory-efficient wrapper for pandas DataFrame with lazy evaluation.
    
    This class provides a memory-efficient wrapper for pandas DataFrame with
    automatic data type optimization and lazy evaluation of computed columns.
    
    Features:
    - Automatic data type optimization
    - Lazy evaluation of computed columns
    - Memory-efficient views instead of copies
    - Reduced memory footprint
    - Resilience patterns integration
    - Exception handling
    """
    
    def __init__(
        self,
        data: Optional[Union[pd.DataFrame, Dict[str, List[Any]], List[Dict[str, Any]]]] = None,
        optimize_dtypes: bool = True,
        enable_lazy_evaluation: bool = True,
        enable_view_caching: bool = True,
        copy: bool = False
    ):
        """
        Initialize a memory-optimized DataFrame.
        
        Args:
            data: Input data (DataFrame, dictionary, or list)
            optimize_dtypes: Whether to optimize data types
            enable_lazy_evaluation: Whether to enable lazy evaluation
            enable_view_caching: Whether to enable view caching
            copy: Whether to copy the input data
        """
        self._df = pd.DataFrame() if data is None else pd.DataFrame(data, copy=copy)
        self._computed_columns: Dict[str, Callable] = {}
        self._computed_values: Dict[str, Any] = {}
        self._views: Dict[Tuple, pd.DataFrame] = {}
        self._optimize_dtypes = optimize_dtypes
        self._enable_lazy_evaluation = enable_lazy_evaluation
        self._enable_view_caching = enable_view_caching
        self._max_views_cache = 10
        
        # Optimize data types if requested
        if optimize_dtypes and not self._df.empty:
            self.optimize_dtypes()
        
        logger.debug(f"MemoryOptimizedDataFrame initialized with shape {self._df.shape}")
    
    def optimize_dtypes(self) -> 'MemoryOptimizedDataFrame':
        """
        Optimize data types to reduce memory usage.
        
        Returns:
            Self for method chaining
        """
        start_mem = self._df.memory_usage(deep=True).sum()
        
        for col in self._df.columns:
            self._optimize_column_dtype(col)
        
        end_mem = self._df.memory_usage(deep=True).sum()
        reduction = (start_mem - end_mem) / start_mem
        
        logger.debug(
            f"Memory usage reduced from {start_mem:,} to {end_mem:,} bytes ({reduction:.2%} reduction)"
        )
        
        return self
    
    def _optimize_column_dtype(self, column: str) -> None:
        """
        Optimize data type for a specific column.
        
        Args:
            column: Column name
        """
        if column not in self._df.columns:
            return
        
        # Get column data
        col_data = self._df[column]
        
        # Skip optimization for certain types
        if col_data.dtype == 'object' and col_data.map(type).nunique() > 1:
            return
        
        # Optimize numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            # Check if column contains integers
            if pd.api.types.is_integer_dtype(col_data) or (
                pd.api.types.is_float_dtype(col_data) and col_data.dropna().apply(lambda x: x.is_integer()).all()
            ):
                # Find the smallest integer type that can hold the data
                min_val = col_data.min()
                max_val = col_data.max()
                
                # Check for NaN values
                has_nan = col_data.isna().any()
                
                if has_nan:
                    # Use nullable integer types if available (pandas >= 1.0.0)
                    if hasattr(pd, 'Int8Dtype') and min_val >= -128 and max_val <= 127:
                        self._df[column] = self._df[column].astype('Int8')
                    elif hasattr(pd, 'Int16Dtype') and min_val >= -32768 and max_val <= 32767:
                        self._df[column] = self._df[column].astype('Int16')
                    elif hasattr(pd, 'Int32Dtype') and min_val >= -2147483648 and max_val <= 2147483647:
                        self._df[column] = self._df[column].astype('Int32')
                    elif hasattr(pd, 'Int64Dtype'):
                        self._df[column] = self._df[column].astype('Int64')
                else:
                    # Use standard integer types
                    if min_val >= 0:
                        if max_val <= 255:
                            self._df[column] = self._df[column].astype(np.uint8)
                        elif max_val <= 65535:
                            self._df[column] = self._df[column].astype(np.uint16)
                        elif max_val <= 4294967295:
                            self._df[column] = self._df[column].astype(np.uint32)
                        else:
                            self._df[column] = self._df[column].astype(np.uint64)
                    else:
                        if min_val >= -128 and max_val <= 127:
                            self._df[column] = self._df[column].astype(np.int8)
                        elif min_val >= -32768 and max_val <= 32767:
                            self._df[column] = self._df[column].astype(np.int16)
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            self._df[column] = self._df[column].astype(np.int32)
                        else:
                            self._df[column] = self._df[column].astype(np.int64)
            elif pd.api.types.is_float_dtype(col_data):
                # Check if float32 is sufficient
                if col_data.min() > np.finfo(np.float32).min and col_data.max() < np.finfo(np.float32).max:
                    self._df[column] = self._df[column].astype(np.float32)
        
        # Optimize categorical columns
        elif pd.api.types.is_object_dtype(col_data):
            # Check if column contains strings
            if col_data.dropna().apply(lambda x: isinstance(x, str)).all():
                # Check if column has few unique values
                if col_data.nunique() < len(col_data) * 0.5:
                    self._df[column] = self._df[column].astype('category')
        
        # Optimize boolean columns
        elif pd.api.types.is_bool_dtype(col_data):
            self._df[column] = self._df[column].astype(bool)
        
        # Optimize datetime columns
        elif pd.api.types.is_datetime64_dtype(col_data):
            self._df[column] = pd.to_datetime(self._df[column])
    
    def add_computed_column(
        self, 
        name: str, 
        compute_func: Callable[[pd.DataFrame], pd.Series],
        *args,
        **kwargs
    ) -> 'MemoryOptimizedDataFrame':
        """
        Add a computed column that will be evaluated lazily.
        
        Args:
            name: Column name
            compute_func: Function to compute the column values
            *args: Additional arguments for the function
            **kwargs: Additional keyword arguments for the function
            
        Returns:
            Self for method chaining
        """
        if not self._enable_lazy_evaluation:
            # Compute immediately if lazy evaluation is disabled
            self._df[name] = compute_func(self._df, *args, **kwargs)
            if self._optimize_dtypes:
                self._optimize_column_dtype(name)
        else:
            # Store the compute function for lazy evaluation
            wrapped_func = lambda df: compute_func(df, *args, **kwargs)
            self._computed_columns[name] = wrapped_func
            # Clear any cached values
            if name in self._computed_values:
                del self._computed_values[name]
        
        logger.debug(f"Added computed column '{name}'")
        return self
    
    def get_view(
        self, 
        columns: Optional[List[str]] = None, 
        rows: Optional[slice] = None
    ) -> pd.DataFrame:
        """
        Get a view of the data without copying.
        
        Args:
            columns: Optional list of columns to include
            rows: Optional slice of rows to include
            
        Returns:
            DataFrame view
        """
        key = (tuple(columns) if columns else None, rows)
        
        if self._enable_view_caching and key in self._views:
            return self._views[key]
        
        if columns is None and rows is None:
            view = self._df
        elif columns is None:
            view = self._df.iloc[rows]
        elif rows is None:
            view = self._df[columns]
        else:
            view = self._df.loc[rows, columns]
        
        if self._enable_view_caching:
            if len(self._views) >= self._max_views_cache:
                try:
                    oldest_key = next(iter(self._views))
                    del self._views[oldest_key]
                except (StopIteration, KeyError):
                    pass
            self._views[key] = view
        
        return view
    
    def __getitem__(self, key: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """
        Get a column or subset of columns.
        
        Args:
            key: Column name or list of column names
            
        Returns:
            Series or DataFrame with the requested columns
        """
        # Handle single column
        if isinstance(key, str):
            # Check if it's a computed column
            if key in self._computed_columns and key not in self._df.columns:
                # Check if we have a cached value
                if key not in self._computed_values:
                    # Compute and cache the value
                    self._computed_values[key] = self._computed_columns[key](self._df)
                return self._computed_values[key]
            return self._df[key]
        
        # Handle list of columns
        if isinstance(key, list):
            # Check if any computed columns are requested
            computed_cols = [col for col in key if col in self._computed_columns and col not in self._df.columns]
            if computed_cols:
                # Create a copy of the DataFrame
                result = self._df.copy()
                # Add computed columns
                for col in computed_cols:
                    if col not in self._computed_values:
                        self._computed_values[col] = self._computed_columns[col](self._df)
                    result[col] = self._computed_values[col]
                return result[key]
            return self._df[key]
        
        # Handle other indexing
        return self._df[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a column value.
        
        Args:
            key: Column name
            value: Column value
        """
        self._df[key] = value
        if self._optimize_dtypes:
            self._optimize_column_dtype(key)
        
        # Remove from computed columns if it exists
        if key in self._computed_columns:
            del self._computed_columns[key]
        if key in self._computed_values:
            del self._computed_values[key]
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.
        
        Returns:
            Pandas DataFrame
        """
        # Create a copy of the DataFrame
        result = self._df.copy()
        
        # Add computed columns
        for col, func in self._computed_columns.items():
            if col not in self._computed_values:
                self._computed_values[col] = func(self._df)
            result[col] = self._computed_values[col]
        
        return result
    
    # Alias for backward compatibility
    to_dataframe = to_pandas
    
    def memory_usage(self, deep: bool = True) -> pd.Series:
        """
        Get memory usage information.
        
        Args:
            deep: Whether to perform a deep introspection
            
        Returns:
            Series with memory usage information
        """
        return self._df.memory_usage(deep=deep)
    
    def total_memory_usage(self, deep: bool = True) -> int:
        """
        Get total memory usage in bytes.
        
        Args:
            deep: Whether to perform a deep introspection
            
        Returns:
            Total memory usage in bytes
        """
        return self._df.memory_usage(deep=deep).sum()
    
    def __len__(self) -> int:
        """Get the number of rows."""
        return len(self._df)
    
    @property
    def columns(self) -> pd.Index:
        """Get the column names."""
        # Combine actual columns and computed columns
        all_columns = list(self._df.columns) + list(self._computed_columns.keys())
        return pd.Index(all_columns)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape (rows, columns)."""
        return (len(self._df), len(self.columns))
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Get the first n rows.
        
        Args:
            n: Number of rows to return
            
        Returns:
            DataFrame with the first n rows
        """
        # Create a copy of the first n rows
        result = self._df.head(n).copy()
        
        # Add computed columns
        for col, func in self._computed_columns.items():
            result[col] = func(self._df).head(n)
        
        return result
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """
        Get the last n rows.
        
        Args:
            n: Number of rows to return
            
        Returns:
            DataFrame with the last n rows
        """
        # Create a copy of the last n rows
        result = self._df.tail(n).copy()
        
        # Add computed columns
        for col, func in self._computed_columns.items():
            result[col] = func(self._df).tail(n)
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the view cache to free memory."""
        self._views.clear()
        self._computed_values.clear()
    
    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        self.clear_cache()
        if hasattr(self, '_df'):
            self._df = None
        if hasattr(self, '_views'):
            self._views = None
        if hasattr(self, '_computed_columns'):
            self._computed_columns = None
        if hasattr(self, '_computed_values'):
            self._computed_values = None
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MemoryOptimizedDataFrame(shape={self.shape}, columns={list(self._df.columns)})"
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying DataFrame."""
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._df, name)
