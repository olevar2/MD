"""
Memory Optimization Module

This module implements memory optimization techniques for efficient storage and
processing of historical data in the forex trading platform.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
import logging
import weakref
import sys
import psutil
import gc
from enum import Enum
import threading
from datetime import datetime, timedelta
import os
import pickle
import hashlib
import mmap
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CompressionLevel(Enum):
    """Data compression level options."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class DataPrecision(Enum):
    """Data precision options for numeric values."""
    FULL = 0
    STANDARD = 1
    LOW = 2
    ADAPTIVE = 3


class HistoricalDataManager:
    """
    Memory-efficient manager for historical financial data.
    
    This class optimizes memory usage for historical data storage through
    techniques like data compression, precision reduction, and smart caching.
    """

    def __init__(self, base_path: Optional[str]=None, max_memory_percent:
        float=70.0, enable_disk_offload: bool=True, enable_compression:
        bool=True, default_precision: DataPrecision=DataPrecision.STANDARD):
        """
        Initialize the historical data manager.
        
        Args:
            base_path: Base directory for disk offloading
            max_memory_percent: Maximum memory usage as percentage of system RAM
            enable_disk_offload: Whether to offload data to disk when memory is low
            enable_compression: Whether to use data compression
            default_precision: Default numeric precision for data
        """
        self.base_path = base_path
        if base_path and not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        self.max_memory_percent = max_memory_percent
        self.enable_disk_offload = enable_disk_offload
        self.enable_compression = enable_compression
        self.default_precision = default_precision
        self.dataframes = {}
        self.metadata = {}
        self.disk_offloaded = {}
        self.lru_tracking = {}
        self.access_counts = {}
        self._stop_monitor = threading.Event()
        self._monitor_thread = None
        self._memory_check_interval = 60
        self._start_memory_monitor()
        logger.info(
            f'Historical data manager initialized with max memory: {max_memory_percent}%, disk offload: {enable_disk_offload}, compression: {enable_compression}'
            )

    def _start_memory_monitor(self):
        """Start the memory monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(target=self.
                _monitor_memory, daemon=True)
            self._monitor_thread.start()
            logger.debug('Memory monitoring started')

    @with_exception_handling
    def _monitor_memory(self):
        """Periodically check memory usage and optimize if needed."""
        while not self._stop_monitor.is_set():
            try:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.max_memory_percent:
                    logger.warning(
                        f'Memory usage ({memory_percent}%) exceeds threshold ({self.max_memory_percent}%), optimizing...'
                        )
                    self.optimize_memory()
                self._stop_monitor.wait(timeout=self._memory_check_interval)
            except Exception as e:
                logger.error(f'Error in memory monitor: {e}')
                time.sleep(5)

    def _stop_memory_monitor(self):
        """Stop the memory monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_monitor.set()
            self._monitor_thread.join(timeout=2)
            logger.debug('Memory monitoring stopped')

    def store(self, key: str, data: pd.DataFrame, metadata: dict=None,
        precision: DataPrecision=None, compression: CompressionLevel=None,
        persist: bool=False) ->str:
        """
        Store a dataframe with optimized memory usage.
        
        Args:
            key: Unique identifier for the dataframe
            data: Pandas DataFrame to store
            metadata: Additional metadata for the dataframe
            precision: Data precision level (overrides default)
            compression: Compression level (overrides default)
            persist: Whether to persist the data on disk
            
        Returns:
            Storage key
        """
        precision = precision or self.default_precision
        compression = compression or (CompressionLevel.MEDIUM if self.
            enable_compression else CompressionLevel.NONE)
        optimized_data = self._optimize_dataframe(data, precision, compression)
        self.dataframes[key] = optimized_data
        self.metadata[key] = metadata or {}
        self.metadata[key]['stored_at'] = datetime.now()
        self.metadata[key]['precision'] = precision
        self.metadata[key]['compression'] = compression
        self.metadata[key]['original_size'] = sys.getsizeof(data)
        self.metadata[key]['optimized_size'] = sys.getsizeof(optimized_data)
        self.metadata[key]['original_shape'] = data.shape
        self.lru_tracking[key] = datetime.now()
        self.access_counts[key] = 0
        if persist and self.base_path:
            self._persist_to_disk(key, optimized_data)
        logger.debug(
            f"Stored dataframe '{key}' with shape {data.shape}, compression: {compression.name}, precision: {precision.name}"
            )
        return key

    def get(self, key: str) ->pd.DataFrame:
        """
        Retrieve a stored dataframe.
        
        Args:
            key: Storage key for the dataframe
            
        Returns:
            Original dataframe
            
        Raises:
            KeyError: If key not found
        """
        if key in self.dataframes:
            self.lru_tracking[key] = datetime.now()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self._restore_dataframe(self.dataframes[key], self.
                metadata[key]['precision'], self.metadata[key]['compression'])
        elif key in self.disk_offloaded and self.base_path:
            filepath = os.path.join(self.base_path, f'{key}.pkl')
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    optimized_data = pickle.load(f)
                self.dataframes[key] = optimized_data
                self.disk_offloaded.pop(key)
                self.lru_tracking[key] = datetime.now()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return self._restore_dataframe(optimized_data, self.
                    metadata[key]['precision'], self.metadata[key][
                    'compression'])
        raise KeyError(f"Data key '{key}' not found")

    def has_key(self, key: str) ->bool:
        """
        Check if a key exists.
        
        Args:
            key: Storage key
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.dataframes or key in self.disk_offloaded

    def remove(self, key: str) ->bool:
        """
        Remove a stored dataframe.
        
        Args:
            key: Storage key
            
        Returns:
            True if removed, False if not found
        """
        removed = False
        if key in self.dataframes:
            del self.dataframes[key]
            removed = True
        if key in self.disk_offloaded and self.base_path:
            filepath = os.path.join(self.base_path, f'{key}.pkl')
            if os.path.exists(filepath):
                os.unlink(filepath)
            self.disk_offloaded.pop(key)
            removed = True
        if key in self.metadata:
            del self.metadata[key]
        if key in self.lru_tracking:
            del self.lru_tracking[key]
        if key in self.access_counts:
            del self.access_counts[key]
        return removed

    def get_metadata(self, key: str) ->Optional[dict]:
        """
        Get metadata for a stored dataframe.
        
        Args:
            key: Storage key
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata.get(key)

    def optimize_memory(self, target_percent: float=None) ->int:
        """
        Optimize memory usage by offloading or compressing data.
        
        Args:
            target_percent: Target memory usage percentage (default: max_memory_percent - 10)
            
        Returns:
            Number of items optimized
        """
        if target_percent is None:
            target_percent = max(50, self.max_memory_percent - 10)
        current_memory = psutil.virtual_memory().percent
        if current_memory <= target_percent:
            logger.debug(
                f'Memory usage ({current_memory}%) already below target ({target_percent}%)'
                )
            return 0
        items_to_optimize = len(self.dataframes) // 4
        if items_to_optimize < 1 and self.dataframes:
            items_to_optimize = 1
        if not self.dataframes:
            return 0
        candidates = list(self.dataframes.keys())
        candidates.sort(key=lambda k: self.lru_tracking.get(k, datetime.min))
        optimized_count = 0
        for i in range(min(items_to_optimize, len(candidates))):
            key = candidates[i]
            if self.enable_disk_offload and self.base_path:
                self._offload_to_disk(key)
                optimized_count += 1
            elif key in self.dataframes:
                current_compression = self.metadata[key]['compression']
                if current_compression != CompressionLevel.HIGH:
                    new_compression = CompressionLevel(min(
                        current_compression.value + 1, CompressionLevel.
                        HIGH.value))
                    optimized_data = self._compress_dataframe(self.
                        dataframes[key], new_compression)
                    self.dataframes[key] = optimized_data
                    self.metadata[key]['compression'] = new_compression
                    optimized_count += 1
        gc.collect()
        new_memory = psutil.virtual_memory().percent
        logger.info(
            f'Memory optimization complete: {optimized_count} items optimized, memory usage {current_memory}% -> {new_memory}%'
            )
        return optimized_count

    def _optimize_dataframe(self, df: pd.DataFrame, precision:
        DataPrecision, compression: CompressionLevel) ->pd.DataFrame:
        """
        Optimize a dataframe for memory efficiency.
        
        Args:
            df: Pandas DataFrame to optimize
            precision: Desired precision level
            compression: Desired compression level
            
        Returns:
            Optimized dataframe
        """
        df = self._reduce_precision(df, precision)
        if compression != CompressionLevel.NONE:
            df = self._compress_dataframe(df, compression)
        return df

    def _reduce_precision(self, df: pd.DataFrame, precision: DataPrecision
        ) ->pd.DataFrame:
        """
        Reduce numeric precision of a dataframe.
        
        Args:
            df: Pandas DataFrame
            precision: Desired precision level
            
        Returns:
            Dataframe with reduced precision
        """
        if precision == DataPrecision.FULL:
            return df
        result = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            if precision == DataPrecision.STANDARD:
                result[col] = df[col].astype(np.float32)
            elif precision == DataPrecision.LOW:
                result[col] = df[col].astype(np.float16)
            elif precision == DataPrecision.ADAPTIVE:
                col_min, col_max = df[col].min(), df[col].max()
                col_range = col_max - col_min
                if abs(col_min) < 1000 and abs(col_max
                    ) < 1000 and col_range < 100:
                    result[col] = df[col].astype(np.float16)
                else:
                    result[col] = df[col].astype(np.float32)
        return result

    def _compress_dataframe(self, df: pd.DataFrame, level: CompressionLevel
        ) ->pd.DataFrame:
        """
        Apply compression to a dataframe based on the specified level.
        
        Args:
            df: Pandas DataFrame
            level: Compression level
            
        Returns:
            Compressed dataframe
        """
        if level == CompressionLevel.NONE:
            return df
        if level == CompressionLevel.LOW:
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < len(df) * 0.5:
                    df[col] = df[col].astype('category')
        elif level == CompressionLevel.MEDIUM:
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < len(df) * 0.7:
                    df[col] = df[col].astype('category')
            for col in df.select_dtypes(include=[np.number]).columns:
                zero_pct = (df[col] == 0).mean()
                if zero_pct > 0.7:
                    df[col] = df[col].astype(pd.SparseDtype(df[col].dtype))
        elif level == CompressionLevel.HIGH:
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype('category')
            for col in df.select_dtypes(include=[np.number]).columns:
                zero_pct = (df[col] == 0).mean()
                if zero_pct > 0.5:
                    df[col] = df[col].astype(pd.SparseDtype(df[col].dtype))
        return df

    def _restore_dataframe(self, df: pd.DataFrame, precision: DataPrecision,
        compression: CompressionLevel) ->pd.DataFrame:
        """
        Restore a dataframe to its original form after optimization.
        
        Args:
            df: Optimized dataframe
            precision: Applied precision level
            compression: Applied compression level
            
        Returns:
            Restored dataframe
        """
        return df

    @with_exception_handling
    def _offload_to_disk(self, key: str) ->bool:
        """
        Offload a dataframe from memory to disk.
        
        Args:
            key: Storage key
            
        Returns:
            True if offloaded, False otherwise
        """
        if not self.base_path or key not in self.dataframes:
            return False
        try:
            os.makedirs(self.base_path, exist_ok=True)
            filepath = os.path.join(self.base_path, f'{key}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(self.dataframes[key], f)
            self.disk_offloaded[key] = filepath
            del self.dataframes[key]
            logger.debug(f"Offloaded dataframe '{key}' to disk")
            return True
        except Exception as e:
            logger.error(f"Error offloading dataframe '{key}' to disk: {e}")
            return False

    @with_exception_handling
    def _persist_to_disk(self, key: str, data: pd.DataFrame) ->bool:
        """
        Persist a dataframe to disk without removing from memory.
        
        Args:
            key: Storage key
            data: DataFrame to persist
            
        Returns:
            True if persisted, False otherwise
        """
        if not self.base_path:
            return False
        try:
            os.makedirs(self.base_path, exist_ok=True)
            filepath = os.path.join(self.base_path, f'{key}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            self.metadata[key]['persisted'] = True
            self.metadata[key]['persist_path'] = filepath
            logger.debug(f"Persisted dataframe '{key}' to disk")
            return True
        except Exception as e:
            logger.error(f"Error persisting dataframe '{key}' to disk: {e}")
            return False

    def get_storage_stats(self) ->dict:
        """
        Get statistics about storage usage.
        
        Returns:
            Dictionary with storage statistics
        """
        total_memory_size = sum(sys.getsizeof(df) for df in self.dataframes
            .values())
        total_orig_size = sum(meta.get('original_size', 0) for meta in self
            .metadata.values())
        compression_ratio = total_orig_size / max(1, total_memory_size)
        return {'items_in_memory': len(self.dataframes), 'items_on_disk':
            len(self.disk_offloaded), 'total_items': len(self.metadata),
            'memory_size_bytes': total_memory_size, 'original_size_bytes':
            total_orig_size, 'compression_ratio': compression_ratio,
            'system_memory_used_percent': psutil.virtual_memory().percent}

    def __del__(self):
        """Clean up resources when object is deleted."""
        self._stop_memory_monitor()


class MemoryMappedArray:
    """
    Memory-mapped array implementation for large datasets that can't fit in memory.
    
    This class provides a numpy-like interface to arrays stored on disk, allowing
    efficient access to data larger than available RAM.
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]], dtype=np.float32,
        filename: Optional[str]=None, mode: str='r+'):
        """
        Initialize a memory-mapped array.
        
        Args:
            shape: Shape of the array
            dtype: Data type
            filename: File path for the memory-mapped file (None for temporary)
            mode: File access mode ('r' for read-only, 'r+' for read-write)
        """
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = np.dtype(dtype)
        self.filename = filename
        self.mode = mode
        self.temp_file = None
        self.mmap = None
        self._initialize()

    def _initialize(self):
        """Initialize the memory-mapped array."""
        if self.filename is None:
            import tempfile
            self.temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.filename = self.temp_file.name
        size_in_bytes = int(np.prod(self.shape) * self.dtype.itemsize)
        if not os.path.exists(self.filename) or os.path.getsize(self.filename
            ) < size_in_bytes:
            with open(self.filename, 'wb') as f:
                f.seek(size_in_bytes - 1)
                f.write(b'\x00')
        self.mmap = np.memmap(self.filename, dtype=self.dtype, mode=self.
            mode, shape=self.shape)

    def __getitem__(self, index):
        """Get item at the specified index."""
        return self.mmap[index]

    def __setitem__(self, index, value):
        """Set item at the specified index."""
        if self.mode in ('r+', 'w', 'w+'):
            self.mmap[index] = value
        else:
            raise ValueError('Array is read-only')

    def flush(self):
        """Flush changes to disk."""
        if self.mmap is not None:
            self.mmap.flush()

    def close(self):
        """Close the memory-mapped file."""
        if self.mmap is not None:
            self.mmap = None
        if self.temp_file:
            os.unlink(self.filename)
            self.temp_file = None

    def __del__(self):
        """Clean up resources when object is deleted."""
        self.close()


class MemoryOptimizer:
    """
    Memory optimization utilities for pandas DataFrames and NumPy arrays.
    
    This class provides static methods to optimize memory usage of data structures
    without requiring a full manager instance.
    """

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, downcast: bool=True,
        categorize: bool=True, sparse: bool=True) ->pd.DataFrame:
        """
        Optimize memory usage of a pandas DataFrame.
        
        Args:
            df: DataFrame to optimize
            downcast: Whether to downcast numeric columns
            categorize: Whether to convert string columns to categorical
            sparse: Whether to convert to sparse when appropriate
            
        Returns:
            Optimized DataFrame
        """
        result = df.copy()
        if downcast:
            for col in result.select_dtypes(include=['integer']).columns:
                result[col] = pd.to_numeric(result[col], downcast='integer')
            for col in result.select_dtypes(include=['float']).columns:
                result[col] = pd.to_numeric(result[col], downcast='float')
        if categorize:
            for col in result.select_dtypes(include=['object']).columns:
                num_unique = result[col].nunique()
                if num_unique < len(result) * 0.5:
                    result[col] = result[col].astype('category')
        if sparse:
            for col in result.columns:
                if result[col].dtype != 'category':
                    most_common_val = result[col].mode().iloc[0]
                    fill_ratio = (result[col] == most_common_val).mean()
                    if fill_ratio > 0.7:
                        sparse_dtype = pd.SparseDtype(result[col].dtype,
                            most_common_val)
                        result[col] = result[col].astype(sparse_dtype)
        return result

    @staticmethod
    def estimate_dataframe_size(df: pd.DataFrame) ->Dict[str, int]:
        """
        Estimate memory usage of a DataFrame with breakdown by column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with memory usage information
        """
        total_size = df.memory_usage(deep=True).sum()
        column_sizes = {col: df[col].memory_usage(deep=True) for col in df.
            columns}
        index_size = df.index.memory_usage(deep=True)
        return {'total_bytes': total_size, 'total_mb': total_size / (1024 *
            1024), 'column_bytes': column_sizes, 'index_bytes': index_size}

    @staticmethod
    def batch_process(df: pd.DataFrame, func: Callable, batch_size: int=100000
        ) ->pd.DataFrame:
        """
        Process a large DataFrame in batches to reduce memory usage.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each batch
            batch_size: Number of rows per batch
            
        Returns:
            Processed DataFrame
        """
        if len(df) <= batch_size:
            return func(df)
        result_parts = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            processed_batch = func(batch)
            result_parts.append(processed_batch)
            batch = None
            processed_batch = None
            gc.collect()
        return pd.concat(result_parts)


historical_data_manager = None


def get_historical_data_manager(base_path: str=None, **kwargs
    ) ->HistoricalDataManager:
    """
    Get or create the global historical data manager.
    
    Args:
        base_path: Base directory for disk offloading
        **kwargs: Additional arguments for HistoricalDataManager
        
    Returns:
        HistoricalDataManager instance
    """
    global historical_data_manager
    if historical_data_manager is None:
        if base_path is None:
            base_path = os.path.join(os.path.expanduser('~'),
                '.forex_platform', 'data_cache')
        historical_data_manager = HistoricalDataManager(base_path=base_path,
            **kwargs)
    return historical_data_manager
