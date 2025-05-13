"""
This file has been replaced by the standardized parallel processing implementation.

The original implementation has been backed up to:
analysis-engine-service\analysis_engine\utils/optimized_parallel_processor.py.bak.20250512230806

Please use the standardized parallel processing from common-lib instead:

from common_lib.parallel import (
    ParallelProcessor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
    get_parallel_processor,
    TimeframeHierarchy,
    FeatureSpec,
    MultiInstrumentProcessor,
    get_multi_instrument_processor
)
"""

from common_lib.parallel import (
    ParallelProcessor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
    get_parallel_processor,
    TimeframeHierarchy,
    FeatureSpec,
    MultiInstrumentProcessor,
    get_multi_instrument_processor
)

# For backward compatibility
OptimizedParallelProcessor = ParallelProcessor
parallel_processor = get_parallel_processor()
multi_instrument_processor = get_multi_instrument_processor()
