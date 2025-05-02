"""
Feature Store Metrics Module

This module provides monitoring for the feature store client.
"""
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import threading
import json

from core_foundations.utils.logger import get_logger

# Try to import Prometheus client for metrics
try:
    import prometheus_client as prom
    has_prometheus = True
except ImportError:
    has_prometheus = False


class FeatureStoreMetrics:
    """
    Metrics collector for feature store client.
    
    This class collects metrics about feature store client usage and performance,
    including API calls, cache hits/misses, and errors.
    
    Attributes:
        logger: Logger instance
        metrics: Dictionary of metrics
        prometheus_metrics: Dictionary of Prometheus metrics (if available)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one metrics collector exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FeatureStoreMetrics, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the metrics collector."""
        if self._initialized:
            return
            
        self.logger = get_logger("feature_store_metrics")
        
        # Initialize metrics
        self.metrics = {
            "api_calls": {
                "total": 0,
                "get_indicators": 0,
                "get_ohlcv_data": 0,
                "compute_feature": 0,
                "other": 0
            },
            "cache": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0
            },
            "errors": {
                "total": 0,
                "connection": 0,
                "timeout": 0,
                "auth": 0,
                "not_found": 0,
                "other": 0
            },
            "performance": {
                "avg_response_time_ms": 0,
                "total_response_time_ms": 0,
                "request_count": 0,
                "max_response_time_ms": 0
            },
            "fallbacks": {
                "total": 0,
                "get_indicators": 0,
                "get_ohlcv_data": 0,
                "compute_feature": 0
            }
        }
        
        # Initialize Prometheus metrics if available
        self.prometheus_metrics = {}
        if has_prometheus:
            self._init_prometheus_metrics()
        
        self._initialized = True
        self.logger.info("Feature store metrics initialized")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # API call counters
        self.prometheus_metrics["api_calls_total"] = prom.Counter(
            "feature_store_api_calls_total",
            "Total number of API calls to the feature store",
            ["method"]
        )
        
        # Cache metrics
        self.prometheus_metrics["cache_hits"] = prom.Counter(
            "feature_store_cache_hits",
            "Number of cache hits"
        )
        self.prometheus_metrics["cache_misses"] = prom.Counter(
            "feature_store_cache_misses",
            "Number of cache misses"
        )
        
        # Error counters
        self.prometheus_metrics["errors_total"] = prom.Counter(
            "feature_store_errors_total",
            "Total number of errors",
            ["error_type"]
        )
        
        # Performance metrics
        self.prometheus_metrics["response_time"] = prom.Histogram(
            "feature_store_response_time_seconds",
            "Response time for feature store API calls",
            ["method"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0)
        )
        
        # Fallback counters
        self.prometheus_metrics["fallbacks_total"] = prom.Counter(
            "feature_store_fallbacks_total",
            "Total number of fallbacks to direct calculation",
            ["method"]
        )
    
    def track_api_call(self, method: str, start_time: float = None):
        """
        Track an API call to the feature store.
        
        Args:
            method: API method name
            start_time: Start time of the API call (for performance tracking)
            
        Returns:
            Decorator function
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Record start time if not provided
                nonlocal start_time
                if start_time is None:
                    start_time = time.time()
                
                # Update API call metrics
                self.metrics["api_calls"]["total"] += 1
                if method in self.metrics["api_calls"]:
                    self.metrics["api_calls"][method] += 1
                else:
                    self.metrics["api_calls"]["other"] += 1
                
                # Update Prometheus metrics if available
                if has_prometheus and "api_calls_total" in self.prometheus_metrics:
                    self.prometheus_metrics["api_calls_total"].labels(method=method).inc()
                
                try:
                    # Call the original function
                    result = await func(*args, **kwargs)
                    
                    # Record performance metrics
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000
                    
                    self.metrics["performance"]["request_count"] += 1
                    self.metrics["performance"]["total_response_time_ms"] += response_time_ms
                    self.metrics["performance"]["avg_response_time_ms"] = (
                        self.metrics["performance"]["total_response_time_ms"] / 
                        self.metrics["performance"]["request_count"]
                    )
                    
                    if response_time_ms > self.metrics["performance"]["max_response_time_ms"]:
                        self.metrics["performance"]["max_response_time_ms"] = response_time_ms
                    
                    # Update Prometheus metrics if available
                    if has_prometheus and "response_time" in self.prometheus_metrics:
                        self.prometheus_metrics["response_time"].labels(method=method).observe(
                            response_time_ms / 1000  # Convert to seconds
                        )
                    
                    return result
                    
                except Exception as e:
                    # Track error
                    self.track_error(str(type(e).__name__))
                    raise
                    
            return wrapper
        return decorator
    
    def track_cache_hit(self):
        """Track a cache hit."""
        self.metrics["cache"]["hits"] += 1
        self._update_cache_hit_rate()
        
        # Update Prometheus metrics if available
        if has_prometheus and "cache_hits" in self.prometheus_metrics:
            self.prometheus_metrics["cache_hits"].inc()
    
    def track_cache_miss(self):
        """Track a cache miss."""
        self.metrics["cache"]["misses"] += 1
        self._update_cache_hit_rate()
        
        # Update Prometheus metrics if available
        if has_prometheus and "cache_misses" in self.prometheus_metrics:
            self.prometheus_metrics["cache_misses"].inc()
    
    def _update_cache_hit_rate(self):
        """Update cache hit rate."""
        total = self.metrics["cache"]["hits"] + self.metrics["cache"]["misses"]
        if total > 0:
            self.metrics["cache"]["hit_rate"] = self.metrics["cache"]["hits"] / total
    
    def track_error(self, error_type: str):
        """
        Track an error.
        
        Args:
            error_type: Type of error
        """
        self.metrics["errors"]["total"] += 1
        
        if error_type == "FeatureStoreConnectionError":
            self.metrics["errors"]["connection"] += 1
            prometheus_error_type = "connection"
        elif error_type == "FeatureStoreTimeoutError":
            self.metrics["errors"]["timeout"] += 1
            prometheus_error_type = "timeout"
        elif error_type == "FeatureStoreAuthError":
            self.metrics["errors"]["auth"] += 1
            prometheus_error_type = "auth"
        elif error_type == "FeatureNotFoundError":
            self.metrics["errors"]["not_found"] += 1
            prometheus_error_type = "not_found"
        else:
            self.metrics["errors"]["other"] += 1
            prometheus_error_type = "other"
        
        # Update Prometheus metrics if available
        if has_prometheus and "errors_total" in self.prometheus_metrics:
            self.prometheus_metrics["errors_total"].labels(error_type=prometheus_error_type).inc()
    
    def track_fallback(self, method: str):
        """
        Track a fallback to direct calculation.
        
        Args:
            method: API method name
        """
        self.metrics["fallbacks"]["total"] += 1
        
        if method in self.metrics["fallbacks"]:
            self.metrics["fallbacks"][method] += 1
        
        # Update Prometheus metrics if available
        if has_prometheus and "fallbacks_total" in self.prometheus_metrics:
            self.prometheus_metrics["fallbacks_total"].labels(method=method).inc()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary with metrics
        """
        return self.metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.__init__()
    
    def export_metrics_json(self) -> str:
        """
        Export metrics as JSON string.
        
        Returns:
            JSON string with metrics
        """
        return json.dumps(self.metrics, indent=2)
    
    def log_metrics_summary(self):
        """Log a summary of current metrics."""
        self.logger.info(f"Feature store API calls: {self.metrics['api_calls']['total']}")
        self.logger.info(f"Cache hit rate: {self.metrics['cache']['hit_rate']:.2f}")
        self.logger.info(f"Errors: {self.metrics['errors']['total']}")
        self.logger.info(f"Avg response time: {self.metrics['performance']['avg_response_time_ms']:.2f} ms")
        self.logger.info(f"Fallbacks: {self.metrics['fallbacks']['total']}")


# Singleton instance
feature_store_metrics = FeatureStoreMetrics()
