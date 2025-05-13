"""
Custom exceptions for data reconciliation.

This module provides custom exceptions for the data reconciliation framework.
"""

from typing import Any, Dict, Optional


class ReconciliationError(Exception):
    """Base exception for all reconciliation errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reconciliation error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.details = details or {}
        
    def __str__(self) -> str:
        if self.details:
            return f"{super().__str__()} - Details: {self.details}"
        return super().__str__()


class SourceDataError(ReconciliationError):
    """Error when fetching or processing source data."""
    
    def __init__(
        self,
        message: str,
        source_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize source data error.
        
        Args:
            message: Error message
            source_id: ID of the source that caused the error
            details: Additional error details
        """
        super().__init__(message, details)
        self.source_id = source_id
        
    def __str__(self) -> str:
        return f"Source {self.source_id}: {super().__str__()}"


class ResolutionStrategyError(ReconciliationError):
    """Error when applying a resolution strategy."""
    
    def __init__(
        self,
        message: str,
        strategy: str,
        discrepancy_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize resolution strategy error.
        
        Args:
            message: Error message
            strategy: Name of the strategy that caused the error
            discrepancy_id: ID of the discrepancy being resolved
            details: Additional error details
        """
        super().__init__(message, details)
        self.strategy = strategy
        self.discrepancy_id = discrepancy_id
        
    def __str__(self) -> str:
    """
      str  .
    
    Returns:
        str: Description of return value
    
    """

        if self.discrepancy_id:
            return f"Strategy {self.strategy} for discrepancy {self.discrepancy_id}: {super().__str__()}"
        return f"Strategy {self.strategy}: {super().__str__()}"


class ReconciliationTimeoutError(ReconciliationError):
    """Error when reconciliation times out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        reconciliation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reconciliation timeout error.
        
        Args:
            message: Error message
            timeout_seconds: Timeout in seconds
            reconciliation_id: ID of the reconciliation process
            details: Additional error details
        """
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds
        self.reconciliation_id = reconciliation_id
        
    def __str__(self) -> str:
    """
      str  .
    
    Returns:
        str: Description of return value
    
    """

        if self.reconciliation_id:
            return f"Reconciliation {self.reconciliation_id} timed out after {self.timeout_seconds} seconds: {super().__str__()}"
        return f"Reconciliation timed out after {self.timeout_seconds} seconds: {super().__str__()}"


class InconsistentDataError(ReconciliationError):
    """Error when data is inconsistent and cannot be reconciled."""
    
    def __init__(
        self,
        message: str,
        field: str,
        sources: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize inconsistent data error.
        
        Args:
            message: Error message
            field: Field with inconsistent data
            sources: Dictionary mapping source IDs to their values
            details: Additional error details
        """
        super().__init__(message, details)
        self.field = field
        self.sources = sources
        
    def __str__(self) -> str:
    """
      str  .
    
    Returns:
        str: Description of return value
    
    """

        return f"Inconsistent data for field {self.field}: {super().__str__()}"
