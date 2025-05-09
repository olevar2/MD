"""
Error Handling Utilities for Parallel Processing.

This module provides specialized error handling utilities for parallel
processing operations, including error aggregation, categorization,
recovery strategies, and reporting.

Features:
- Error aggregation from parallel tasks
- Error categorization and prioritization
- Recovery strategies for different error types
- Comprehensive error reporting and logging
"""

import asyncio
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from common_lib.exceptions import (
    DataProcessingError,
    DataValidationError,
    ServiceError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors that can occur during parallel processing."""
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    PROCESSING = "processing"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class ErrorSeverity(int, Enum):
    """Severity levels for errors."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class ParallelProcessingError:
    """
    Represents an error that occurred during parallel processing.
    
    This class provides a standardized way to represent, categorize,
    and report errors that occur during parallel processing operations.
    """
    
    def __init__(self,
                 task_id: str,
                 error: Exception,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 timestamp: Optional[datetime] = None,
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize a parallel processing error.
        
        Args:
            task_id: ID of the task that failed
            error: The exception that occurred
            category: Category of the error
            severity: Severity of the error
            timestamp: When the error occurred
            context: Additional context for the error
        """
        self.task_id = task_id
        self.error = error
        self.category = category
        self.severity = severity
        self.timestamp = timestamp or datetime.now()
        self.context = context or {}
        self.traceback = traceback.format_exception(
            type(error), error, error.__traceback__
        )
    
    @property
    def message(self) -> str:
        """Get the error message."""
        return str(self.error)
    
    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.category.value.upper()} error in task {self.task_id}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "traceback": "".join(self.traceback)
        }
    
    @classmethod
    def categorize_error(cls, error: Exception) -> ErrorCategory:
        """
        Categorize an error based on its type.
        
        Args:
            error: The exception to categorize
            
        Returns:
            Error category
        """
        if isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT
        elif isinstance(error, DataValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, DataProcessingError):
            return ErrorCategory.PROCESSING
        elif isinstance(error, (MemoryError, ResourceWarning)):
            return ErrorCategory.RESOURCE
        elif isinstance(error, (ImportError, ModuleNotFoundError, AttributeError)):
            return ErrorCategory.DEPENDENCY
        else:
            return ErrorCategory.UNKNOWN
    
    @classmethod
    def determine_severity(cls, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """
        Determine the severity of an error.
        
        Args:
            error: The exception
            category: Error category
            
        Returns:
            Error severity
        """
        # Timeouts are usually high severity
        if category == ErrorCategory.TIMEOUT:
            return ErrorSeverity.HIGH
        
        # Resource errors are critical
        elif category == ErrorCategory.RESOURCE:
            return ErrorSeverity.CRITICAL
        
        # Validation errors are medium severity
        elif category == ErrorCategory.VALIDATION:
            return ErrorSeverity.MEDIUM
        
        # Processing errors depend on the specific error
        elif category == ErrorCategory.PROCESSING:
            if isinstance(error, DataProcessingError) and hasattr(error, "severity"):
                # Map from DataProcessingError severity to ErrorSeverity
                return ErrorSeverity(min(error.severity, 3))
            return ErrorSeverity.MEDIUM
        
        # Dependency errors are high severity
        elif category == ErrorCategory.DEPENDENCY:
            return ErrorSeverity.HIGH
        
        # Unknown errors are medium severity by default
        else:
            return ErrorSeverity.MEDIUM


class ErrorAggregator:
    """
    Aggregates and analyzes errors from parallel processing operations.
    
    This class provides utilities for collecting, categorizing, and analyzing
    errors that occur during parallel processing operations.
    """
    
    def __init__(self):
        """Initialize the error aggregator."""
        self.errors: List[ParallelProcessingError] = []
    
    def add_error(self, error: ParallelProcessingError):
        """
        Add an error to the aggregator.
        
        Args:
            error: The error to add
        """
        self.errors.append(error)
    
    def add_exception(self,
                     task_id: str,
                     error: Exception,
                     context: Optional[Dict[str, Any]] = None):
        """
        Add an exception to the aggregator.
        
        Args:
            task_id: ID of the task that failed
            error: The exception that occurred
            context: Additional context for the error
        """
        category = ParallelProcessingError.categorize_error(error)
        severity = ParallelProcessingError.determine_severity(error, category)
        
        self.add_error(ParallelProcessingError(
            task_id=task_id,
            error=error,
            category=category,
            severity=severity,
            context=context
        ))
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ParallelProcessingError]:
        """
        Get errors by category.
        
        Args:
            category: Error category
            
        Returns:
            List of errors in the specified category
        """
        return [e for e in self.errors if e.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ParallelProcessingError]:
        """
        Get errors by severity.
        
        Args:
            severity: Error severity
            
        Returns:
            List of errors with the specified severity
        """
        return [e for e in self.errors if e.severity == severity]
    
    def get_most_severe_error(self) -> Optional[ParallelProcessingError]:
        """
        Get the most severe error.
        
        Returns:
            Most severe error, or None if no errors
        """
        if not self.errors:
            return None
        
        return min(self.errors, key=lambda e: e.severity.value)
    
    def has_critical_errors(self) -> bool:
        """
        Check if there are any critical errors.
        
        Returns:
            True if there are critical errors
        """
        return any(e.severity == ErrorSeverity.CRITICAL for e in self.errors)
    
    def get_error_summary(self) -> Dict[str, int]:
        """
        Get a summary of errors by category.
        
        Returns:
            Dictionary mapping error categories to counts
        """
        summary = {category.value: 0 for category in ErrorCategory}
        
        for error in self.errors:
            summary[error.category.value] += 1
        
        return summary
    
    def clear(self):
        """Clear all errors."""
        self.errors = []


class ErrorRecoveryStrategy:
    """
    Defines strategies for recovering from errors during parallel processing.
    
    This class provides utilities for defining and applying recovery strategies
    for different types of errors that occur during parallel processing.
    """
    
    @staticmethod
    async def retry_with_backoff(func: Callable,
                          args: Tuple = (),
                          kwargs: Dict[str, Any] = None,
                          max_retries: int = 3,
                          initial_delay: float = 0.1,
                          backoff_factor: float = 2.0,
                          exceptions: Tuple[Exception, ...] = (Exception,)) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries (seconds)
            backoff_factor: Factor to increase delay by after each retry
            exceptions: Exceptions to catch and retry on
            
        Returns:
            Result of the function
            
        Raises:
            The last exception if all retries fail
        """
        kwargs = kwargs or {}
        delay = initial_delay
        last_exception = None
        
        for retry in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if retry < max_retries:
                    logger.warning(f"Retry {retry + 1}/{max_retries} after error: {str(e)}")
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(f"All {max_retries} retries failed with error: {str(e)}")
                    raise
        
        # This should never happen, but just in case
        raise last_exception if last_exception else RuntimeError("Retry failed for unknown reason")
    
    @staticmethod
    def get_recovery_strategy(error: ParallelProcessingError) -> Optional[Callable]:
        """
        Get a recovery strategy for an error.
        
        Args:
            error: The error to recover from
            
        Returns:
            Recovery strategy function, or None if no strategy available
        """
        # Define recovery strategies based on error category
        if error.category == ErrorCategory.TIMEOUT:
            return ErrorRecoveryStrategy.handle_timeout
        elif error.category == ErrorCategory.VALIDATION:
            return ErrorRecoveryStrategy.handle_validation_error
        elif error.category == ErrorCategory.PROCESSING:
            return ErrorRecoveryStrategy.handle_processing_error
        elif error.category == ErrorCategory.RESOURCE:
            return ErrorRecoveryStrategy.handle_resource_error
        elif error.category == ErrorCategory.DEPENDENCY:
            return ErrorRecoveryStrategy.handle_dependency_error
        else:
            return None
    
    @staticmethod
    def handle_timeout(error: ParallelProcessingError) -> Dict[str, Any]:
        """
        Handle a timeout error.
        
        Args:
            error: The timeout error
            
        Returns:
            Recovery result
        """
        logger.warning(f"Timeout in task {error.task_id}: {error.message}")
        
        # For timeouts, we might want to retry with a longer timeout
        return {
            "action": "retry",
            "timeout_multiplier": 1.5,
            "max_retries": 2
        }
    
    @staticmethod
    def handle_validation_error(error: ParallelProcessingError) -> Dict[str, Any]:
        """
        Handle a validation error.
        
        Args:
            error: The validation error
            
        Returns:
            Recovery result
        """
        logger.warning(f"Validation error in task {error.task_id}: {error.message}")
        
        # For validation errors, we might want to use default values
        return {
            "action": "use_defaults",
            "log_warning": True
        }
    
    @staticmethod
    def handle_processing_error(error: ParallelProcessingError) -> Dict[str, Any]:
        """
        Handle a processing error.
        
        Args:
            error: The processing error
            
        Returns:
            Recovery result
        """
        logger.error(f"Processing error in task {error.task_id}: {error.message}")
        
        # For processing errors, we might want to retry with different parameters
        return {
            "action": "retry",
            "with_fallback_method": True,
            "max_retries": 1
        }
    
    @staticmethod
    def handle_resource_error(error: ParallelProcessingError) -> Dict[str, Any]:
        """
        Handle a resource error.
        
        Args:
            error: The resource error
            
        Returns:
            Recovery result
        """
        logger.error(f"Resource error in task {error.task_id}: {error.message}")
        
        # For resource errors, we might want to reduce parallelism
        return {
            "action": "reduce_parallelism",
            "reduction_factor": 0.5,
            "min_workers": 1
        }
    
    @staticmethod
    def handle_dependency_error(error: ParallelProcessingError) -> Dict[str, Any]:
        """
        Handle a dependency error.
        
        Args:
            error: The dependency error
            
        Returns:
            Recovery result
        """
        logger.error(f"Dependency error in task {error.task_id}: {error.message}")
        
        # For dependency errors, we might want to use a fallback
        return {
            "action": "use_fallback",
            "log_error": True
        }
