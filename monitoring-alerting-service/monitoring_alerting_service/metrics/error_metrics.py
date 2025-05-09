"""
Error Metrics Module

This module provides functionality for collecting and exporting error metrics.
"""
import logging
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Gauge, Histogram

from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    MetricsExporterError
)

logger = logging.getLogger(__name__)

# Define Prometheus metrics
error_counter = Counter(
    'forex_platform_errors_total',
    'Total number of errors in the Forex Trading Platform',
    ['service', 'error_code', 'error_type', 'component']
)

http_error_counter = Counter(
    'forex_platform_http_errors_total',
    'Total number of HTTP errors in the Forex Trading Platform',
    ['service', 'endpoint', 'method', 'status_code']
)

circuit_breaker_state = Gauge(
    'forex_platform_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service', 'circuit']
)

error_duration = Histogram(
    'forex_platform_error_duration_seconds',
    'Duration of error handling in seconds',
    ['service', 'error_code'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)


class ErrorMetricsExporter:
    """
    Error metrics exporter for the Forex Trading Platform.
    
    This class provides methods for recording and exporting error metrics
    to Prometheus.
    """
    
    def __init__(self, service_name: str):
        """
        Initialize the error metrics exporter.
        
        Args:
            service_name: The name of the service
        """
        self.service_name = service_name
        logger.info(f"Initialized ErrorMetricsExporter for {service_name}")
    
    @with_exception_handling
    def record_error(
        self,
        error_code: str,
        error_type: str,
        component: str = "general",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an error metric.
        
        Args:
            error_code: The error code
            error_type: The type of error
            component: The component where the error occurred
            details: Additional error details
        """
        try:
            error_counter.labels(
                service=self.service_name,
                error_code=error_code,
                error_type=error_type,
                component=component
            ).inc()
            
            logger.debug(
                f"Recorded error metric: {error_code} in {component}",
                extra={
                    "service": self.service_name,
                    "error_code": error_code,
                    "error_type": error_type,
                    "component": component,
                    "details": details
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to record error metric: {str(e)}",
                exc_info=True
            )
            raise MetricsExporterError(
                message=f"Failed to record error metric: {str(e)}",
                exporter="ErrorMetricsExporter",
                details={
                    "service": self.service_name,
                    "error_code": error_code,
                    "error_type": error_type,
                    "component": component
                }
            )
    
    @with_exception_handling
    def record_http_error(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an HTTP error metric.
        
        Args:
            endpoint: The API endpoint
            method: The HTTP method
            status_code: The HTTP status code
            details: Additional error details
        """
        try:
            http_error_counter.labels(
                service=self.service_name,
                endpoint=endpoint,
                method=method,
                status_code=str(status_code)
            ).inc()
            
            logger.debug(
                f"Recorded HTTP error metric: {status_code} for {method} {endpoint}",
                extra={
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "details": details
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to record HTTP error metric: {str(e)}",
                exc_info=True
            )
            raise MetricsExporterError(
                message=f"Failed to record HTTP error metric: {str(e)}",
                exporter="ErrorMetricsExporter",
                details={
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code
                }
            )
    
    @with_exception_handling
    def update_circuit_breaker_state(
        self,
        circuit: str,
        state: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update circuit breaker state metric.
        
        Args:
            circuit: The circuit name
            state: The circuit state (0=closed, 1=open, 2=half-open)
            details: Additional details
        """
        try:
            circuit_breaker_state.labels(
                service=self.service_name,
                circuit=circuit
            ).set(state)
            
            state_name = {0: "closed", 1: "open", 2: "half-open"}.get(state, "unknown")
            
            logger.debug(
                f"Updated circuit breaker state: {circuit} is {state_name}",
                extra={
                    "service": self.service_name,
                    "circuit": circuit,
                    "state": state,
                    "state_name": state_name,
                    "details": details
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to update circuit breaker state: {str(e)}",
                exc_info=True
            )
            raise MetricsExporterError(
                message=f"Failed to update circuit breaker state: {str(e)}",
                exporter="ErrorMetricsExporter",
                details={
                    "service": self.service_name,
                    "circuit": circuit,
                    "state": state
                }
            )


# Global instance for the monitoring-alerting-service
_error_metrics_exporter = None


def initialize_error_metrics(service_name: str) -> ErrorMetricsExporter:
    """
    Initialize the error metrics exporter.
    
    Args:
        service_name: The name of the service
        
    Returns:
        The error metrics exporter instance
    """
    global _error_metrics_exporter
    if _error_metrics_exporter is None:
        _error_metrics_exporter = ErrorMetricsExporter(service_name)
    return _error_metrics_exporter


def record_error(
    error_code: str,
    error_type: str,
    component: str = "general",
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Record an error metric.
    
    Args:
        error_code: The error code
        error_type: The type of error
        component: The component where the error occurred
        details: Additional error details
    """
    global _error_metrics_exporter
    if _error_metrics_exporter is None:
        _error_metrics_exporter = initialize_error_metrics("monitoring-alerting-service")
    
    _error_metrics_exporter.record_error(
        error_code=error_code,
        error_type=error_type,
        component=component,
        details=details
    )


def record_http_error(
    endpoint: str,
    method: str,
    status_code: int,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Record an HTTP error metric.
    
    Args:
        endpoint: The API endpoint
        method: The HTTP method
        status_code: The HTTP status code
        details: Additional error details
    """
    global _error_metrics_exporter
    if _error_metrics_exporter is None:
        _error_metrics_exporter = initialize_error_metrics("monitoring-alerting-service")
    
    _error_metrics_exporter.record_http_error(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        details=details
    )
