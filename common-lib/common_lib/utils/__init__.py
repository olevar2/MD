"""
Utilities package for the forex trading platform.

This package provides common utilities for the platform.
"""

from common_lib.utils.platform_compatibility import PlatformInfo, PlatformCompatibility
from common_lib.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
from common_lib.utils.export_service import (
    convert_to_csv,
    convert_to_json,
    convert_to_parquet,
    convert_to_excel,
    format_ohlcv_for_json,
    format_timestamp,
    format_numeric
)
from common_lib.utils.parallel_processor import (
    ParallelProcessor,
    Task,
    TaskPriority,
    ParallelizationMethod
)

__all__ = [
    # Platform compatibility
    'PlatformInfo',
    'PlatformCompatibility',

    # Memory optimized DataFrame
    'MemoryOptimizedDataFrame',

    # Export service
    'convert_to_csv',
    'convert_to_json',
    'convert_to_parquet',
    'convert_to_excel',
    'format_ohlcv_for_json',
    'format_timestamp',
    'format_numeric',

    # Parallel processor
    'ParallelProcessor',
    'Task',
    'TaskPriority',
    'ParallelizationMethod'
]