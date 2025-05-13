"""
Custom exceptions for the Optimization module.

This module defines custom exceptions that align with the common-lib exceptions
used throughout the Forex Trading Platform.
"""

from typing import Any, Dict, Optional

# Import common-lib exceptions if available
try:
    from common_lib.exceptions import ForexTradingPlatformError
except ImportError:
    # Define base exception if common-lib is not available
    class ForexTradingPlatformError(Exception):
        """Base exception for all Forex Trading Platform errors."""
        
        def __init__(
            self,
            message: str,
            error_code: str = "FOREX_PLATFORM_ERROR",
            details: Optional[Dict[str, Any]] = None
        ):
            """
            Initialize the exception.
            
            Args:
                message: Human-readable error message
                error_code: Error code for categorization
                details: Additional error details
            """
            self.message = message
            self.error_code = error_code
            self.details = details or {}
            super().__init__(self.message)
        
        def to_dict(self) -> Dict[str, Any]:
            """
            Convert the exception to a dictionary.
            
            Returns:
                Dictionary representation of the exception
            """
            return {
                "error_type": self.__class__.__name__,
                "message": self.message,
                "error_code": self.error_code,
                "details": self.details
            }


# Optimization module specific exceptions
class OptimizationError(ForexTradingPlatformError):
    """Base exception for all optimization errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "OPTIMIZATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details
        )


class ParameterValidationError(OptimizationError):
    """Error related to parameter validation."""
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        parameter_name: Description of parameter_name
        parameter_value: Description of parameter_value
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if parameter_name:
            details["parameter_name"] = parameter_name
        if parameter_value is not None:
            details["parameter_value"] = str(parameter_value)
        super().__init__(
            message=message,
            error_code="PARAMETER_VALIDATION_ERROR",
            details=details
        )
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value


class OptimizationConvergenceError(OptimizationError):
    """Error related to optimization convergence."""
    
    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        iterations: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        algorithm: Description of algorithm
        iterations: Description of iterations
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if algorithm:
            details["algorithm"] = algorithm
        if iterations is not None:
            details["iterations"] = iterations
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_CONVERGENCE_ERROR",
            details=details
        )
        self.algorithm = algorithm
        self.iterations = iterations


class ResourceAllocationError(OptimizationError):
    """Error related to resource allocation."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        resource_type: Description of resource_type
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        super().__init__(
            message=message,
            error_code="RESOURCE_ALLOCATION_ERROR",
            details=details
        )
        self.resource_type = resource_type


class CachingError(OptimizationError):
    """Error related to caching."""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        cache_key: Description of cache_key
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if cache_key:
            details["cache_key"] = cache_key
        super().__init__(
            message=message,
            error_code="CACHING_ERROR",
            details=details
        )
        self.cache_key = cache_key


class MLOptimizationError(OptimizationError):
    """Error related to ML optimization."""
    
    def __init__(
        self,
        message: str,
        model_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        model_type: Description of model_type
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if model_type:
            details["model_type"] = model_type
        super().__init__(
            message=message,
            error_code="ML_OPTIMIZATION_ERROR",
            details=details
        )
        self.model_type = model_type
