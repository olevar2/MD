"""
Monitoring package for the forex trading platform.

This package provides monitoring functionality for the platform.
"""

from common_lib.monitoring.metrics import MetricsManager, MetricType, track_time
from common_lib.monitoring.logging import LoggingManager, JsonFormatter, log_execution
from common_lib.monitoring.tracing import TracingManager, trace_function
from common_lib.monitoring.health import HealthManager, HealthCheck, HealthStatus
from common_lib.monitoring.alerting import AlertManager, Alert, AlertSeverity, AlertChannel

__all__ = [
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
    'AlertChannel'
]