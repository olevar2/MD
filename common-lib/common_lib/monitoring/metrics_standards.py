"""
Standardized Metrics for Forex Trading Platform.

This module defines standardized metrics for all services in the forex trading platform,
ensuring consistent naming, labeling, and collection of metrics across the platform.
"""

import logging
from typing import Dict, List, Optional, Any, Callable

# Import metrics functions
from common_lib.monitoring.metrics import (
    get_counter, get_gauge, get_histogram, get_summary,
    track_execution_time, track_memory_usage
)

# Configure logging
logger = logging.getLogger(__name__)

# Standard metric prefixes for different domains
PREFIX_API = "api"
PREFIX_DB = "db"
PREFIX_CACHE = "cache"
PREFIX_BROKER = "broker"
PREFIX_MODEL = "model"
PREFIX_STRATEGY = "strategy"
PREFIX_FEATURE = "feature"
PREFIX_MARKET = "market"
PREFIX_SYSTEM = "system"

# Standard label names
LABEL_SERVICE = "service"
LABEL_ENDPOINT = "endpoint"
LABEL_METHOD = "method"
LABEL_STATUS_CODE = "status_code"
LABEL_ERROR_TYPE = "error_type"
LABEL_OPERATION = "operation"
LABEL_DATABASE = "database"
LABEL_TABLE = "table"
LABEL_CACHE_TYPE = "cache_type"
LABEL_BROKER = "broker"
LABEL_INSTRUMENT = "instrument"
LABEL_TIMEFRAME = "timeframe"
LABEL_STRATEGY = "strategy"
LABEL_MODEL = "model"
LABEL_FEATURE = "feature"
LABEL_COMPONENT = "component"
LABEL_INSTANCE = "instance"

# Standard buckets for latency histograms (in seconds)
LATENCY_BUCKETS_FAST = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
LATENCY_BUCKETS_MEDIUM = [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
LATENCY_BUCKETS_SLOW = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]

# Standard buckets for size histograms (in bytes)
SIZE_BUCKETS = [
    1024,                  # 1 KB
    10 * 1024,             # 10 KB
    100 * 1024,            # 100 KB
    1024 * 1024,           # 1 MB
    10 * 1024 * 1024,      # 10 MB
    100 * 1024 * 1024,     # 100 MB
    1024 * 1024 * 1024     # 1 GB
]

# Standard buckets for count histograms
COUNT_BUCKETS = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

class StandardMetrics:
    """
    Standardized metrics for forex trading platform services.
    
    This class provides standardized metrics for all services in the forex trading platform,
    ensuring consistent naming, labeling, and collection of metrics across the platform.
    """
    
    def __init__(self, service_name: str):
        """
        Initialize the standard metrics.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        
        # Initialize standard metrics
        self._init_api_metrics()
        self._init_database_metrics()
        self._init_cache_metrics()
        self._init_broker_metrics()
        self._init_system_metrics()
        
        logger.info(f"Initialized standard metrics for {service_name}")
    
    def _init_api_metrics(self) -> None:
        """Initialize API metrics."""
        # API request counter
        self.api_requests_total = get_counter(
            name=f"{PREFIX_API}_requests_total",
            description="Total number of API requests",
            labels=[LABEL_SERVICE, LABEL_ENDPOINT, LABEL_METHOD, LABEL_STATUS_CODE]
        )
        
        # API request duration
        self.api_request_duration_seconds = get_histogram(
            name=f"{PREFIX_API}_request_duration_seconds",
            description="API request duration in seconds",
            labels=[LABEL_SERVICE, LABEL_ENDPOINT, LABEL_METHOD],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # API request size
        self.api_request_size_bytes = get_histogram(
            name=f"{PREFIX_API}_request_size_bytes",
            description="API request size in bytes",
            labels=[LABEL_SERVICE, LABEL_ENDPOINT, LABEL_METHOD],
            buckets=SIZE_BUCKETS
        )
        
        # API response size
        self.api_response_size_bytes = get_histogram(
            name=f"{PREFIX_API}_response_size_bytes",
            description="API response size in bytes",
            labels=[LABEL_SERVICE, LABEL_ENDPOINT, LABEL_METHOD],
            buckets=SIZE_BUCKETS
        )
        
        # API errors
        self.api_errors_total = get_counter(
            name=f"{PREFIX_API}_errors_total",
            description="Total number of API errors",
            labels=[LABEL_SERVICE, LABEL_ENDPOINT, LABEL_METHOD, LABEL_ERROR_TYPE]
        )
    
    def _init_database_metrics(self) -> None:
        """Initialize database metrics."""
        # Database operations counter
        self.db_operations_total = get_counter(
            name=f"{PREFIX_DB}_operations_total",
            description="Total number of database operations",
            labels=[LABEL_SERVICE, LABEL_DATABASE, LABEL_OPERATION, LABEL_TABLE]
        )
        
        # Database operation duration
        self.db_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_DB}_operation_duration_seconds",
            description="Database operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_DATABASE, LABEL_OPERATION, LABEL_TABLE],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Database errors
        self.db_errors_total = get_counter(
            name=f"{PREFIX_DB}_errors_total",
            description="Total number of database errors",
            labels=[LABEL_SERVICE, LABEL_DATABASE, LABEL_OPERATION, LABEL_ERROR_TYPE]
        )
        
        # Database connections
        self.db_connections = get_gauge(
            name=f"{PREFIX_DB}_connections",
            description="Number of database connections",
            labels=[LABEL_SERVICE, LABEL_DATABASE]
        )
        
        # Database connection pool metrics
        self.db_connection_pool_size = get_gauge(
            name=f"{PREFIX_DB}_connection_pool_size",
            description="Size of the database connection pool",
            labels=[LABEL_SERVICE, LABEL_DATABASE]
        )
        
        self.db_connection_pool_used = get_gauge(
            name=f"{PREFIX_DB}_connection_pool_used",
            description="Number of used connections in the database connection pool",
            labels=[LABEL_SERVICE, LABEL_DATABASE]
        )
    
    def _init_cache_metrics(self) -> None:
        """Initialize cache metrics."""
        # Cache operations counter
        self.cache_operations_total = get_counter(
            name=f"{PREFIX_CACHE}_operations_total",
            description="Total number of cache operations",
            labels=[LABEL_SERVICE, LABEL_CACHE_TYPE, LABEL_OPERATION]
        )
        
        # Cache hits
        self.cache_hits_total = get_counter(
            name=f"{PREFIX_CACHE}_hits_total",
            description="Total number of cache hits",
            labels=[LABEL_SERVICE, LABEL_CACHE_TYPE]
        )
        
        # Cache misses
        self.cache_misses_total = get_counter(
            name=f"{PREFIX_CACHE}_misses_total",
            description="Total number of cache misses",
            labels=[LABEL_SERVICE, LABEL_CACHE_TYPE]
        )
        
        # Cache hit ratio
        self.cache_hit_ratio = get_gauge(
            name=f"{PREFIX_CACHE}_hit_ratio",
            description="Cache hit ratio",
            labels=[LABEL_SERVICE, LABEL_CACHE_TYPE]
        )
        
        # Cache size
        self.cache_size = get_gauge(
            name=f"{PREFIX_CACHE}_size",
            description="Number of items in the cache",
            labels=[LABEL_SERVICE, LABEL_CACHE_TYPE]
        )
        
        # Cache memory usage
        self.cache_memory_usage_bytes = get_gauge(
            name=f"{PREFIX_CACHE}_memory_usage_bytes",
            description="Memory usage of the cache in bytes",
            labels=[LABEL_SERVICE, LABEL_CACHE_TYPE]
        )
        
        # Cache operation duration
        self.cache_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_CACHE}_operation_duration_seconds",
            description="Cache operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_CACHE_TYPE, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_FAST
        )
    
    def _init_broker_metrics(self) -> None:
        """Initialize broker metrics."""
        # Broker operations counter
        self.broker_operations_total = get_counter(
            name=f"{PREFIX_BROKER}_operations_total",
            description="Total number of broker operations",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION]
        )
        
        # Broker operation duration
        self.broker_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_BROKER}_operation_duration_seconds",
            description="Broker operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Broker errors
        self.broker_errors_total = get_counter(
            name=f"{PREFIX_BROKER}_errors_total",
            description="Total number of broker errors",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION, LABEL_ERROR_TYPE]
        )
        
        # Broker connection status
        self.broker_connection_status = get_gauge(
            name=f"{PREFIX_BROKER}_connection_status",
            description="Broker connection status (1 = connected, 0 = disconnected)",
            labels=[LABEL_SERVICE, LABEL_BROKER]
        )
    
    def _init_system_metrics(self) -> None:
        """Initialize system metrics."""
        # System health
        self.system_health = get_gauge(
            name=f"{PREFIX_SYSTEM}_health",
            description="System health status (1 = healthy, 0 = unhealthy)",
            labels=[LABEL_SERVICE, LABEL_COMPONENT]
        )
        
        # System uptime
        self.system_uptime_seconds = get_gauge(
            name=f"{PREFIX_SYSTEM}_uptime_seconds",
            description="System uptime in seconds",
            labels=[LABEL_SERVICE]
        )
        
        # System resource usage
        self.system_cpu_usage_percent = get_gauge(
            name=f"{PREFIX_SYSTEM}_cpu_usage_percent",
            description="CPU usage percentage",
            labels=[LABEL_SERVICE, LABEL_INSTANCE]
        )
        
        self.system_memory_usage_bytes = get_gauge(
            name=f"{PREFIX_SYSTEM}_memory_usage_bytes",
            description="Memory usage in bytes",
            labels=[LABEL_SERVICE, LABEL_INSTANCE]
        )
        
        self.system_disk_usage_bytes = get_gauge(
            name=f"{PREFIX_SYSTEM}_disk_usage_bytes",
            description="Disk usage in bytes",
            labels=[LABEL_SERVICE, LABEL_INSTANCE]
        )
        
        # System events
        self.system_events_total = get_counter(
            name=f"{PREFIX_SYSTEM}_events_total",
            description="Total number of system events",
            labels=[LABEL_SERVICE, LABEL_COMPONENT, "event_type"]
        )
