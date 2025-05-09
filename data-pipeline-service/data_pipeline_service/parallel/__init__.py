"""
Parallel Processing Package for Data Pipeline Service.

This package provides comprehensive parallel processing capabilities for
the data pipeline service, optimizing data retrieval, processing, and
analysis across multiple instruments, timeframes, and feature types.

Components:
- parallel_processing_framework: Core framework for parallel processing
- multi_instrument_processor: Specialized processor for multiple instruments
- multi_timeframe_processor: Specialized processor for multiple timeframes
- batch_feature_processor: Specialized processor for batch feature engineering
- error_handling: Utilities for error handling in parallel processing
"""

from data_pipeline_service.parallel.parallel_processing_framework import (
    ParallelExecutor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
)

from data_pipeline_service.parallel.multi_instrument_processor import (
    MultiInstrumentProcessor,
)

from data_pipeline_service.parallel.multi_timeframe_processor import (
    MultiTimeframeProcessor,
    TimeframeHierarchy,
)

from data_pipeline_service.parallel.batch_feature_processor import (
    BatchFeatureProcessor,
    FeatureSpec,
)

from data_pipeline_service.parallel.error_handling import (
    ErrorAggregator,
    ErrorCategory,
    ErrorRecoveryStrategy,
    ErrorSeverity,
    ParallelProcessingError,
)

__all__ = [
    # Framework
    'ParallelExecutor',
    'ParallelizationMethod',
    'ResourceManager',
    'TaskDefinition',
    'TaskPriority',
    'TaskResult',
    
    # Processors
    'MultiInstrumentProcessor',
    'MultiTimeframeProcessor',
    'BatchFeatureProcessor',
    
    # Utilities
    'TimeframeHierarchy',
    'FeatureSpec',
    
    # Error handling
    'ErrorAggregator',
    'ErrorCategory',
    'ErrorRecoveryStrategy',
    'ErrorSeverity',
    'ParallelProcessingError',
]
