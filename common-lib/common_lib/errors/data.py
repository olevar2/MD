"""
Data Error Types.

This module defines error types related to data operations in the forex trading platform.
"""

from typing import Any, Dict, Optional, List

from common_lib.errors.base import BaseError, ErrorCode, ErrorSeverity


class DataError(BaseError):
    """
    Base class for all data-related errors.
    
    This class is used for errors that occur during data operations.
    """
    
    def __init__(
        self,
        message: str,
        data_source: str,
        code: ErrorCode = ErrorCode.DATA_PROCESSING_FAILED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a data error.
        
        Args:
            message: Error message
            data_source: Name of the data source where the error occurred
            code: Error code
            severity: Error severity level
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        details = details or {}
        details["data_source"] = data_source
        
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )
        
        self.data_source = data_source


class DataValidationError(DataError):
    """
    Error raised when data validation fails.
    
    This error is raised when data does not meet validation requirements.
    """
    
    def __init__(
        self,
        data_source: str,
        validation_errors: Dict[str, str],
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a data validation error.
        
        Args:
            data_source: Name of the data source where validation failed
            validation_errors: Dictionary mapping field names to error messages
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Data validation failed for source '{data_source}'"
        
        details = details or {}
        details["validation_errors"] = validation_errors
        
        super().__init__(
            message=message,
            data_source=data_source,
            code=ErrorCode.DATA_VALIDATION_FAILED,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class DataNotFoundError(DataError):
    """
    Error raised when data is not found.
    
    This error is raised when requested data cannot be found.
    """
    
    def __init__(
        self,
        data_source: str,
        data_type: str,
        query: Dict[str, Any],
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a data not found error.
        
        Args:
            data_source: Name of the data source where the data was not found
            data_type: Type of the data that was not found
            query: Query parameters used to search for the data
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Data of type '{data_type}' not found in source '{data_source}'"
        
        details = details or {}
        details["data_type"] = data_type
        details["query"] = query
        
        super().__init__(
            message=message,
            data_source=data_source,
            code=ErrorCode.DATA_NOT_FOUND,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class DataDuplicateError(DataError):
    """
    Error raised when duplicate data is detected.
    
    This error is raised when an operation would result in duplicate data.
    """
    
    def __init__(
        self,
        data_source: str,
        data_type: str,
        duplicate_keys: List[str],
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a data duplicate error.
        
        Args:
            data_source: Name of the data source where the duplicate was detected
            data_type: Type of the data that caused the duplicate
            duplicate_keys: List of keys that caused the duplicate
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Duplicate data of type '{data_type}' detected in source '{data_source}'"
        
        details = details or {}
        details["data_type"] = data_type
        details["duplicate_keys"] = duplicate_keys
        
        super().__init__(
            message=message,
            data_source=data_source,
            code=ErrorCode.DATA_DUPLICATE,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class DataCorruptionError(DataError):
    """
    Error raised when data corruption is detected.
    
    This error is raised when data is found to be corrupted or inconsistent.
    """
    
    def __init__(
        self,
        data_source: str,
        data_type: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a data corruption error.
        
        Args:
            data_source: Name of the data source where corruption was detected
            data_type: Type of the corrupted data
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Corrupted data of type '{data_type}' detected in source '{data_source}'"
        
        details = details or {}
        details["data_type"] = data_type
        
        super().__init__(
            message=message,
            data_source=data_source,
            code=ErrorCode.DATA_CORRUPTION,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class DataProcessingError(DataError):
    """
    Error raised when data processing fails.
    
    This error is raised when an error occurs during data processing.
    """
    
    def __init__(
        self,
        data_source: str,
        operation: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a data processing error.
        
        Args:
            data_source: Name of the data source where processing failed
            operation: Name of the operation that failed
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Data processing operation '{operation}' failed for source '{data_source}'"
        
        details = details or {}
        details["operation"] = operation
        
        super().__init__(
            message=message,
            data_source=data_source,
            code=ErrorCode.DATA_PROCESSING_FAILED,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )