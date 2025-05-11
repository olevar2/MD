"""
Memory-Optimized DataFrame

This module provides a memory-efficient wrapper for pandas DataFrame with lazy evaluation
and optimized data types for performance-critical operations like confluence and divergence detection.

Features:
- Automatic data type optimization
- Lazy evaluation of computed columns
- Memory-efficient views instead of copies
- Reduced memory footprint
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizedDataFrame:
    """
    Memory-optimized wrapper for pandas DataFrame with lazy evaluation.

    Features:
    - Automatic data type optimization
    - Lazy evaluation of computed columns
    - Memory-efficient views instead of copies
    - Reduced memory footprint
    """

    def __init__(self, data: Union[pd.DataFrame, Dict, List], copy: bool = False):
        """
        Initialize the optimized DataFrame.

        Args:
            data: Input data (DataFrame, dictionary, or list)
            copy: Whether to copy the data if it's already a DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self._data = data.copy() if copy else data
        else:
            self._data = pd.DataFrame(data)

        self._views = {}
        self._computed_columns = set()
        self._max_views_cache = 10  # Limit the number of cached views to prevent memory leaks

        logger.debug(f"MemoryOptimizedDataFrame initialized with shape {self._data.shape}")

    def optimize_dtypes(self) -> 'MemoryOptimizedDataFrame':
        """
        Optimize data types to reduce memory usage.

        Returns:
            Self for method chaining
        """
        start_mem = self._data.memory_usage(deep=True).sum()

        for col in self._data.columns:
            col_data = self._data[col]

            # Optimize integers
            if pd.api.types.is_integer_dtype(col_data):
                c_min, c_max = col_data.min(), col_data.max()

                if c_min >= 0:
                    if c_max < 256:
                        self._data[col] = col_data.astype(np.uint8)
                    elif c_max < 65536:
                        self._data[col] = col_data.astype(np.uint16)
                    elif c_max < 4294967296:
                        self._data[col] = col_data.astype(np.uint32)
                else:
                    if c_min > -128 and c_max < 128:
                        self._data[col] = col_data.astype(np.int8)
                    elif c_min > -32768 and c_max < 32768:
                        self._data[col] = col_data.astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483648:
                        self._data[col] = col_data.astype(np.int32)

            # Optimize floats
            elif pd.api.types.is_float_dtype(col_data):
                # Check if we can use float32 instead of float64
                if col_data.min() > np.finfo(np.float32).min and col_data.max() < np.finfo(np.float32).max:
                    self._data[col] = col_data.astype(np.float32)

        # Log memory savings
        end_mem = self._data.memory_usage(deep=True).sum()
        reduction = (start_mem - end_mem) / start_mem

        logger.debug(f"Memory usage reduced from {start_mem} to {end_mem} bytes ({reduction:.2%} reduction)")

        return self

    def get_view(self, columns: Optional[List[str]] = None, rows: Optional[slice] = None) -> pd.DataFrame:
        """
        Get a view of the data without copying.

        Args:
            columns: Optional list of columns to include
            rows: Optional slice of rows to include

        Returns:
            DataFrame view
        """
        key = (tuple(columns) if columns else None, rows)

        if key in self._views:
            return self._views[key]

        if columns is None and rows is None:
            view = self._data
        elif columns is None:
            view = self._data.iloc[rows]
        elif rows is None:
            view = self._data[columns]
        else:
            view = self._data.loc[rows, columns]

        # Limit the number of cached views to prevent memory leaks
        if len(self._views) >= self._max_views_cache:
            # Remove the oldest view (first item in the dictionary)
            try:
                oldest_key = next(iter(self._views))
                del self._views[oldest_key]
            except (StopIteration, KeyError):
                pass  # Dictionary might be empty or key might be gone

        self._views[key] = view
        return view

    def add_computed_column(self, name: str, func: Callable, *args, **kwargs) -> 'MemoryOptimizedDataFrame':
        """
        Add a computed column with lazy evaluation.

        Args:
            name: Column name
            func: Function to compute the column
            *args: Additional arguments for the function
            **kwargs: Additional keyword arguments for the function

        Returns:
            Self for method chaining
        """
        if name in self._computed_columns:
            return self

        # Add as property with lazy evaluation
        setattr(self.__class__, name, property(lambda self: func(self._data, *args, **kwargs)))
        self._computed_columns.add(name)

        logger.debug(f"Added computed column '{name}'")

        return self

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the DataFrame."""
        return self._data.shape

    def __getitem__(self, key):
        """Get item from the DataFrame."""
        return self._data[key]

    def __getattr__(self, name):
        """Forward attribute access to the underlying DataFrame."""
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._data, name)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a standard pandas DataFrame.

        Returns:
            pandas DataFrame
        """
        return self._data.copy()

    def __repr__(self):
        """String representation."""
        return f"MemoryOptimizedDataFrame(shape={self.shape}, columns={list(self._data.columns)})"

    def __len__(self):
        """Get the length of the DataFrame."""
        return len(self._data)
        
    def clear_cache(self):
        """Clear the view cache to free memory."""
        self._views.clear()
        
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.clear_cache()
        # Remove references to large objects
        if hasattr(self, '_data'):
            self._data = None
        if hasattr(self, '_views'):
            self._views = None
        if hasattr(self, '_computed_columns'):
            self._computed_columns = None
