"""
Parallel Processing Package for Forex Trading Platform.

This package provides comprehensive parallel processing capabilities for the forex trading platform,
optimizing data retrieval, processing, and analysis across multiple instruments, timeframes, and feature types.

Features:
- Dynamic selection between thread, process, and async-based parallelism
- Resource-aware worker allocation
- Priority-based task scheduling
- Dependency-aware task execution
- Comprehensive error handling and reporting
- Performance monitoring and metrics collection
- Specialized processors for different use cases
"""

from common_lib.parallel.parallel_processor import (
    ParallelProcessor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
    TaskScheduler,
    get_parallel_processor
)

from common_lib.parallel.specialized_processors import (
    TimeframeHierarchy,
    FeatureSpec,
    MultiInstrumentProcessor,
    get_multi_instrument_processor
)

__all__ = [
    # Core framework
    'ParallelProcessor',
    'ParallelizationMethod',
    'ResourceManager',
    'TaskDefinition',
    'TaskPriority',
    'TaskResult',
    'TaskScheduler',
    'get_parallel_processor',
    
    # Specialized processors
    'TimeframeHierarchy',
    'FeatureSpec',
    'MultiInstrumentProcessor',
    'get_multi_instrument_processor'
]
