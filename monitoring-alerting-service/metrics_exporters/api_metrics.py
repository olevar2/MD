"""
API Performance Monitoring System.
Collects and exports API performance metrics to Prometheus.
"""
from datetime import datetime
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram

# API Request Metrics
API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["service", "endpoint", "method", "version"]
)

API_REQUEST_COUNT = Counter(
    "api_request_total",
    "Total number of API requests",
    ["service", "endpoint", "method", "version", "status"]
)

API_ERROR_COUNT = Counter(
    "api_error_total",
    "Total number of API errors",
    ["service", "endpoint", "method", "version", "error_type"]
)

# Rate Limiting Metrics
RATE_LIMIT_HIT = Counter(
    "api_rate_limit_hits_total",
    "Total number of rate limit hits",
    ["service", "endpoint", "client_id"]
)

# Version Usage Metrics
VERSION_USAGE = Counter(
    "api_version_usage_total",
    "API version usage count",
    ["service", "version"]
)

class ApiMetricsCollector:
    """
    ApiMetricsCollector class.
    
    Attributes:
        Add attributes here
    """

    def __init__(self, service_name: str):
    """
      init  .
    
    Args:
        service_name: Description of service_name
    
    """

        self.service_name = service_name
        self._active_requests: Dict[str, datetime] = {}

    def record_request_start(self, endpoint: str, method: str, version: str, request_id: str) -> None:
        """Record the start of an API request."""
        self._active_requests[request_id] = datetime.now()
        VERSION_USAGE.labels(service=self.service_name, version=version).inc()

    def record_request_end(
        self,
        endpoint: str,
        method: str,
        version: str,
        status: int,
        request_id: str,
        error_type: Optional[str] = None
    ) -> None:
        """Record the end of an API request."""
        start_time = self._active_requests.pop(request_id, None)
        if start_time:
            duration = (datetime.now() - start_time).total_seconds()
            API_REQUEST_LATENCY.labels(
                service=self.service_name,
                endpoint=endpoint,
                method=method,
                version=version
            ).observe(duration)

        API_REQUEST_COUNT.labels(
            service=self.service_name,
            endpoint=endpoint,
            method=method,
            version=version,
            status=status
        ).inc()

        if error_type:
            API_ERROR_COUNT.labels(
                service=self.service_name,
                endpoint=endpoint,
                method=method,
                version=version,
                error_type=error_type
            ).inc()

    def record_rate_limit_hit(self, endpoint: str, client_id: str) -> None:
        """Record a rate limit hit."""
        RATE_LIMIT_HIT.labels(
            service=self.service_name,
            endpoint=endpoint,
            client_id=client_id
        ).inc()
