"""
Monitoring package for the forex trading platform.

This package provides monitoring functionality for the platform.
It includes both the legacy monitoring system and the new standardized monitoring system.
"""

# Legacy monitoring system
from common_lib.monitoring.metrics import MetricsManager, MetricType, track_time
from common_lib.monitoring.logging import LoggingManager, JsonFormatter, log_execution
from common_lib.monitoring.tracing import TracingManager, trace_function
from common_lib.monitoring.health import HealthManager, HealthCheck, HealthStatus
from common_lib.monitoring.alerting import AlertManager, Alert, AlertSeverity, AlertChannel

# Standardized monitoring system
from common_lib.monitoring.logging_config import (
    configure_logging,
    get_logger,
    log_with_context,
    CorrelationIdFilter,
    StructuredLogFormatter
)

__all__ = [
    # Legacy monitoring system
    'MetricsManager',
    'MetricType',
    'track_time',
    'LoggingManager',
    'JsonFormatter',
    'log_execution',
    'TracingManager',
    'trace_function',
    'HealthManager',
    'HealthCheck',
    'HealthStatus',
    'AlertManager',
    'Alert',
    'AlertSeverity',
    'AlertChannel',

    # Standardized monitoring system
    'configure_logging',
    'get_logger',
    'log_with_context',
    'CorrelationIdFilter',
    'StructuredLogFormatter'
]