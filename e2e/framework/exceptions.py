"""
Custom exceptions for the E2E testing framework.

This module defines custom exceptions that align with the common-lib exceptions
used throughout the Forex Trading Platform.
"""

from typing import Any, Dict, Optional


class E2ETestError(Exception):
    """Base exception for all E2E testing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E2E_ERROR",
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


class TestEnvironmentError(E2ETestError):
    """Error related to test environment setup or configuration."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="TEST_ENVIRONMENT_ERROR",
            details=details
        )


class ServiceVirtualizationError(E2ETestError):
    """Error related to service virtualization."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="SERVICE_VIRTUALIZATION_ERROR",
            details=details
        )


class TestDataError(E2ETestError):
    """Error related to test data generation or management."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="TEST_DATA_ERROR",
            details=details
        )


class TestExecutionError(E2ETestError):
    """Error during test execution."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="TEST_EXECUTION_ERROR",
            details=details
        )


class TestAssertionError(E2ETestError):
    """Error during test assertion."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="TEST_ASSERTION_ERROR",
            details=details
        )


class TestCleanupError(E2ETestError):
    """Error during test cleanup."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="TEST_CLEANUP_ERROR",
            details=details
        )
