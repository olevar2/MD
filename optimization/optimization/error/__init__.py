"""
Error handling package for the Optimization module.

This package provides error handling utilities and custom exceptions
for the Optimization module.
"""

from .exceptions import (
    OptimizationError,
    ParameterValidationError,
    OptimizationConvergenceError,
    ResourceAllocationError,
    CachingError,
    MLOptimizationError
)

from .error_handler import (
    handle_error,
    with_error_handling
)
