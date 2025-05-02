"""
Exceptions Bridge Module

This module serves as a bridge between common-lib exceptions and service-specific exceptions.
It imports and re-exports common-lib exceptions, making it easier to manage exception imports
and future changes to the exception hierarchy.

Usage:
    from analysis_engine.core.exceptions_bridge import (
        ForexTradingPlatformError,
        DataValidationError,
        DataFetchError,
        ServiceUnavailableError,
        # etc.
    )
"""

from typing import Dict, Any, Optional, Type

# Import common-lib exceptions
from common_lib.exceptions import (
    # Base exception
    ForexTradingPlatformError,
    
    # Configuration exceptions
    ConfigurationError,
    ConfigNotFoundError,
    ConfigValidationError,
    
    # Data exceptions
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    
    # Service exceptions
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    
    # Authentication/Authorization exceptions
    AuthenticationError,
    AuthorizationError,
    
    # Model exceptions
    ModelError,
    ModelTrainingError,
    ModelPredictionError,
    
    # Utility function
    get_all_exception_classes
)

# Re-export all imported exceptions
__all__ = [
    # Base exception
    "ForexTradingPlatformError",
    
    # Configuration exceptions
    "ConfigurationError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    
    # Data exceptions
    "DataError",
    "DataValidationError",
    "DataFetchError",
    "DataStorageError",
    "DataTransformationError",
    
    # Service exceptions
    "ServiceError",
    "ServiceUnavailableError",
    "ServiceTimeoutError",
    
    # Authentication/Authorization exceptions
    "AuthenticationError",
    "AuthorizationError",
    
    # Model exceptions
    "ModelError",
    "ModelTrainingError",
    "ModelPredictionError",
    
    # Utility function
    "get_all_exception_classes"
]

# Dictionary mapping common exception names to their classes
# This can be used for dynamic exception handling or mapping
EXCEPTION_MAP: Dict[str, Type[ForexTradingPlatformError]] = {
    name: cls for name, cls in get_all_exception_classes().items()
}
