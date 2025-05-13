"""
Comprehensive metrics collection for the Analysis Engine Service.

This module defines Prometheus metrics for tracking all key operations,
performance indicators, and resource usage in the Analysis Engine Service.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Dict, Any, Optional

# Analysis Operation Metrics
ANALYSIS_REQUESTS_TOTAL = Counter(
    'analysis_engine_requests_total',
    'Total number of analysis requests',
    ['operation', 'symbol', 'timeframe']
)

ANALYSIS_ERRORS_TOTAL = Counter(
    'analysis_engine_errors_total',
    'Total number of analysis errors',
    ['operation', 'symbol', 'timeframe', 'error_type']
)

ANALYSIS_DURATION_SECONDS = Histogram(
    'analysis_engine_duration_seconds',
    'Duration of analysis operations in seconds',
    ['operation', 'symbol', 'timeframe'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Resource Usage Metrics
RESOURCE_USAGE = Gauge(
    'analysis_engine_resource_usage',
    'Resource usage percentage',
    ['resource_type']  # cpu, memory, disk, etc.
)

# Cache Metrics
CACHE_OPERATIONS_TOTAL = Counter(
    'analysis_engine_cache_operations_total',
    'Total number of cache operations',
    ['operation', 'cache_type']  # get, set, delete, etc.
)

CACHE_HITS_TOTAL = Counter(
    'analysis_engine_cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISSES_TOTAL = Counter(
    'analysis_engine_cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

# Dependency Health Metrics
DEPENDENCY_HEALTH = Gauge(
    'analysis_engine_dependency_health',
    'Health status of dependencies (1=healthy, 0=unhealthy)',
    ['dependency_name']  # database, redis, feature-store, etc.
)

DEPENDENCY_LATENCY_SECONDS = Histogram(
    'analysis_engine_dependency_latency_seconds',
    'Latency of dependency operations in seconds',
    ['dependency_name', 'operation'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

# Queue Metrics
QUEUE_SIZE = Gauge(
    'analysis_engine_queue_size',
    'Current size of processing queues',
    ['queue_name']
)

QUEUE_PROCESSING_TIME_SECONDS = Histogram(
    'analysis_engine_queue_processing_time_seconds',
    'Time taken to process items from queues',
    ['queue_name'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
)

# API Metrics
API_REQUEST_DURATION_SECONDS = Histogram(
    'analysis_engine_api_duration_seconds',
    'Duration of API requests in seconds',
    ['endpoint', 'method', 'status_code'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

API_REQUESTS_TOTAL = Counter(
    'analysis_engine_api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code']
)

# Model Performance Metrics
MODEL_PREDICTION_DURATION_SECONDS = Histogram(
    'analysis_engine_model_prediction_seconds',
    'Duration of model predictions in seconds',
    ['model_name', 'model_version'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

MODEL_PREDICTION_ERRORS_TOTAL = Counter(
    'analysis_engine_model_errors_total',
    'Total number of model prediction errors',
    ['model_name', 'model_version', 'error_type']
)

# Business Metrics
ANALYSIS_QUALITY_SCORE = Gauge(
    'analysis_engine_quality_score',
    'Quality score of analysis results',
    ['analysis_type', 'symbol', 'timeframe']
)

SIGNAL_STRENGTH = Gauge(
    'analysis_engine_signal_strength',
    'Strength of trading signals',
    ['signal_type', 'symbol', 'timeframe']
)

# Utility functions for metrics
class MetricsRecorder:
    """Utility class for recording metrics with consistent labels."""
    
    @staticmethod
    def record_analysis_request(operation: str, symbol: str, timeframe: str) -> None:
        """Record an analysis request."""
        ANALYSIS_REQUESTS_TOTAL.labels(
            operation=operation,
            symbol=symbol,
            timeframe=timeframe
        ).inc()
    
    @staticmethod
    def record_analysis_error(operation: str, symbol: str, timeframe: str, error_type: str) -> None:
        """Record an analysis error."""
        ANALYSIS_ERRORS_TOTAL.labels(
            operation=operation,
            symbol=symbol,
            timeframe=timeframe,
            error_type=error_type
        ).inc()
    
    @staticmethod
    def time_analysis_operation(operation: str, symbol: str, timeframe: str):
        """Return a context manager for timing an analysis operation."""
        return ANALYSIS_DURATION_SECONDS.labels(
            operation=operation,
            symbol=symbol,
            timeframe=timeframe
        ).time()
    
    @staticmethod
    def record_resource_usage(resource_type: str, usage: float) -> None:
        """Record resource usage."""
        RESOURCE_USAGE.labels(resource_type=resource_type).set(usage)
    
    @staticmethod
    def record_cache_operation(operation: str, cache_type: str) -> None:
        """Record a cache operation."""
        CACHE_OPERATIONS_TOTAL.labels(
            operation=operation,
            cache_type=cache_type
        ).inc()
    
    @staticmethod
    def record_cache_hit(cache_type: str) -> None:
        """Record a cache hit."""
        CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def record_cache_miss(cache_type: str) -> None:
        """Record a cache miss."""
        CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def record_dependency_health(dependency_name: str, healthy: bool) -> None:
        """Record dependency health status."""
        DEPENDENCY_HEALTH.labels(dependency_name=dependency_name).set(1 if healthy else 0)
    
    @staticmethod
    def time_dependency_operation(dependency_name: str, operation: str):
        """Return a context manager for timing a dependency operation."""
        return DEPENDENCY_LATENCY_SECONDS.labels(
            dependency_name=dependency_name,
            operation=operation
        ).time()
    
    @staticmethod
    def record_queue_size(queue_name: str, size: int) -> None:
        """Record queue size."""
        QUEUE_SIZE.labels(queue_name=queue_name).set(size)
    
    @staticmethod
    def time_queue_processing(queue_name: str):
        """Return a context manager for timing queue processing."""
        return QUEUE_PROCESSING_TIME_SECONDS.labels(queue_name=queue_name).time()
    
    @staticmethod
    def time_api_request(endpoint: str, method: str):
        """Return a context manager for timing an API request."""
        return API_REQUEST_DURATION_SECONDS.labels(
            endpoint=endpoint,
            method=method,
            status_code="unknown"  # Will be updated after the request completes
        ).time()
    
    @staticmethod
    def record_api_request(endpoint: str, method: str, status_code: int) -> None:
        """Record an API request."""
        API_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
    
    @staticmethod
    def time_model_prediction(model_name: str, model_version: str):
        """Return a context manager for timing a model prediction."""
        return MODEL_PREDICTION_DURATION_SECONDS.labels(
            model_name=model_name,
            model_version=model_version
        ).time()
    
    @staticmethod
    def record_model_error(model_name: str, model_version: str, error_type: str) -> None:
        """Record a model prediction error."""
        MODEL_PREDICTION_ERRORS_TOTAL.labels(
            model_name=model_name,
            model_version=model_version,
            error_type=error_type
        ).inc()
    
    @staticmethod
    def record_analysis_quality(analysis_type: str, symbol: str, timeframe: str, score: float) -> None:
        """Record analysis quality score."""
        ANALYSIS_QUALITY_SCORE.labels(
            analysis_type=analysis_type,
            symbol=symbol,
            timeframe=timeframe
        ).set(score)
    
    @staticmethod
    def record_signal_strength(signal_type: str, symbol: str, timeframe: str, strength: float) -> None:
        """Record signal strength."""
        SIGNAL_STRENGTH.labels(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe
        ).set(strength)
