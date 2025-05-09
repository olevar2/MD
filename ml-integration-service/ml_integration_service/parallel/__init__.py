"""
Parallel Processing Package for ML Integration Service.

This package provides specialized parallel processing capabilities for
ML model inference, optimizing prediction across multiple models,
instruments, and timeframes.

Components:
- parallel_inference: Specialized processor for parallel ML model inference
"""

from ml_integration_service.parallel.parallel_inference import (
    ModelInferenceSpec,
    ParallelInferenceProcessor,
)

__all__ = [
    'ModelInferenceSpec',
    'ParallelInferenceProcessor',
]
