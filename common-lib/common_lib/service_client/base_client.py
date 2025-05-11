"""
Base service client classes for the forex trading platform.

This module provides base classes for service clients used across the platform.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

from common_lib.errors import (
    ForexTradingError,
    ServiceUnavailableError,
    ThirdPartyServiceError,
    TimeoutError,
)

# Type variables for generic service client
T = TypeVar('T')
R = TypeVar('R')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    """Maximum number of retry attempts."""
    
    initial_backoff_ms: int = 100
    """Initial backoff time in milliseconds."""
    
    max_backoff_ms: int = 10000
    """Maximum backoff time in milliseconds."""
    
    backoff_multiplier: float = 2.0
    """Multiplier for backoff time between retries."""
    
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=list)
    """List of exception types that should trigger a retry."""
    
    retry_on_status_codes: List[int] = field(default_factory=list)
    """List of HTTP status codes that should trigger a retry."""


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    failure_threshold: int = 5
    """Number of failures before opening the circuit."""
    
    recovery_timeout_ms: int = 30000
    """Time in milliseconds before attempting to close the circuit."""
    
    half_open_success_threshold: int = 3
    """Number of successful requests needed to close the circuit."""


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""
    
    connect_timeout_ms: int = 5000
    """Connection timeout in milliseconds."""
    
    read_timeout_ms: int = 30000
    """Read timeout in milliseconds."""
    
    total_timeout_ms: int = 60000
    """Total request timeout in milliseconds."""


@dataclass
class ServiceClientConfig:
    """Configuration for service clients."""
    
    service_name: str
    """Name of the service."""
    
    base_url: str
    """Base URL for the service."""
    
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    """Configuration for retry behavior."""
    
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    """Configuration for circuit breaker behavior."""
    
    timeout_config: TimeoutConfig = field(default_factory=TimeoutConfig)
    """Configuration for timeout behavior."""
    
    headers: Dict[str, str] = field(default_factory=dict)
    """Default headers to include in all requests."""


class BaseServiceClient(Generic[T, R], ABC):
    """
    Base class for synchronous service clients.
    
    This abstract class defines the interface that all service clients
    must implement. It provides common functionality and enforces a consistent
    API across all service client implementations.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the service client.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(f"{self.__class__.__name__}.{config.service_name}")
        
        # Circuit breaker state
        self._circuit_open = False
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_success_count = 0
    
    @abstractmethod
    def send_request(self, request: T) -> R:
        """
        Send a request to the service.
        
        Args:
            request: Request to send
            
        Returns:
            Response from the service
            
        Raises:
            ForexTradingError: If the request fails
        """
        raise NotImplementedError("Subclasses must implement send_request() method")
    
    def _should_retry(
        self,
        exception: Exception,
        status_code: Optional[int] = None,
        retry_count: int = 0
    ) -> bool:
        """
        Determine if a request should be retried.
        
        Args:
            exception: Exception that was raised
            status_code: HTTP status code (if applicable)
            retry_count: Current retry count
            
        Returns:
            True if the request should be retried, False otherwise
        """
        # Check if we've reached the maximum number of retries
        if retry_count >= self.config.retry_config.max_retries:
            return False
        
        # Check if the exception type should trigger a retry
        if any(isinstance(exception, exc_type) for exc_type in self.config.retry_config.retry_on_exceptions):
            return True
        
        # Check if the status code should trigger a retry
        if status_code is not None and status_code in self.config.retry_config.retry_on_status_codes:
            return True
        
        return False
    
    def _get_backoff_time(self, retry_count: int) -> float:
        """
        Calculate the backoff time for a retry.
        
        Args:
            retry_count: Current retry count
            
        Returns:
            Backoff time in seconds
        """
        backoff_ms = min(
            self.config.retry_config.initial_backoff_ms * (self.config.retry_config.backoff_multiplier ** retry_count),
            self.config.retry_config.max_backoff_ms
        )
        return backoff_ms / 1000.0
    
    def _update_circuit_breaker(self, success: bool) -> None:
        """
        Update the circuit breaker state.
        
        Args:
            success: Whether the request was successful
        """
        if success:
            if self._circuit_open:
                # In half-open state, count successful requests
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.config.circuit_breaker_config.half_open_success_threshold:
                    # Close the circuit
                    self._circuit_open = False
                    self._failure_count = 0
                    self._half_open_success_count = 0
                    self.logger.info(f"Circuit closed for {self.config.service_name}")
            else:
                # Reset failure count on success
                self._failure_count = 0
        else:
            # Increment failure count
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            # Check if we should open the circuit
            if not self._circuit_open and self._failure_count >= self.config.circuit_breaker_config.failure_threshold:
                self._circuit_open = True
                self.logger.warning(f"Circuit opened for {self.config.service_name}")
    
    def _check_circuit_breaker(self) -> None:
        """
        Check if the circuit breaker is open.
        
        Raises:
            ServiceUnavailableError: If the circuit is open
        """
        if self._circuit_open:
            # Check if we should try to close the circuit
            elapsed_ms = (time.time() - self._last_failure_time) * 1000
            if elapsed_ms >= self.config.circuit_breaker_config.recovery_timeout_ms:
                # Enter half-open state
                self._half_open_success_count = 0
                self.logger.info(f"Circuit half-open for {self.config.service_name}")
            else:
                # Circuit is still open
                raise ServiceUnavailableError(
                    message=f"Circuit breaker open for {self.config.service_name}",
                    service_name=self.config.service_name
                )


class AsyncBaseServiceClient(Generic[T, R], ABC):
    """
    Base class for asynchronous service clients.
    
    This abstract class defines the interface that all asynchronous service clients
    must implement. It provides common functionality and enforces a consistent
    API across all service client implementations.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the service client.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(f"{self.__class__.__name__}.{config.service_name}")
        
        # Circuit breaker state
        self._circuit_open = False
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_success_count = 0
    
    @abstractmethod
    async def send_request(self, request: T) -> R:
        """
        Send a request to the service.
        
        Args:
            request: Request to send
            
        Returns:
            Response from the service
            
        Raises:
            ForexTradingError: If the request fails
        """
        raise NotImplementedError("Subclasses must implement send_request() method")
    
    def _should_retry(
        self,
        exception: Exception,
        status_code: Optional[int] = None,
        retry_count: int = 0
    ) -> bool:
        """
        Determine if a request should be retried.
        
        Args:
            exception: Exception that was raised
            status_code: HTTP status code (if applicable)
            retry_count: Current retry count
            
        Returns:
            True if the request should be retried, False otherwise
        """
        # Check if we've reached the maximum number of retries
        if retry_count >= self.config.retry_config.max_retries:
            return False
        
        # Check if the exception type should trigger a retry
        if any(isinstance(exception, exc_type) for exc_type in self.config.retry_config.retry_on_exceptions):
            return True
        
        # Check if the status code should trigger a retry
        if status_code is not None and status_code in self.config.retry_config.retry_on_status_codes:
            return True
        
        return False
    
    def _get_backoff_time(self, retry_count: int) -> float:
        """
        Calculate the backoff time for a retry.
        
        Args:
            retry_count: Current retry count
            
        Returns:
            Backoff time in seconds
        """
        backoff_ms = min(
            self.config.retry_config.initial_backoff_ms * (self.config.retry_config.backoff_multiplier ** retry_count),
            self.config.retry_config.max_backoff_ms
        )
        return backoff_ms / 1000.0
    
    def _update_circuit_breaker(self, success: bool) -> None:
        """
        Update the circuit breaker state.
        
        Args:
            success: Whether the request was successful
        """
        if success:
            if self._circuit_open:
                # In half-open state, count successful requests
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.config.circuit_breaker_config.half_open_success_threshold:
                    # Close the circuit
                    self._circuit_open = False
                    self._failure_count = 0
                    self._half_open_success_count = 0
                    self.logger.info(f"Circuit closed for {self.config.service_name}")
            else:
                # Reset failure count on success
                self._failure_count = 0
        else:
            # Increment failure count
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            # Check if we should open the circuit
            if not self._circuit_open and self._failure_count >= self.config.circuit_breaker_config.failure_threshold:
                self._circuit_open = True
                self.logger.warning(f"Circuit opened for {self.config.service_name}")
    
    def _check_circuit_breaker(self) -> None:
        """
        Check if the circuit breaker is open.
        
        Raises:
            ServiceUnavailableError: If the circuit is open
        """
        if self._circuit_open:
            # Check if we should try to close the circuit
            elapsed_ms = (time.time() - self._last_failure_time) * 1000
            if elapsed_ms >= self.config.circuit_breaker_config.recovery_timeout_ms:
                # Enter half-open state
                self._half_open_success_count = 0
                self.logger.info(f"Circuit half-open for {self.config.service_name}")
            else:
                # Circuit is still open
                raise ServiceUnavailableError(
                    message=f"Circuit breaker open for {self.config.service_name}",
                    service_name=self.config.service_name
                )