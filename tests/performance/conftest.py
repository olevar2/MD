"""
Fixtures for performance tests.
"""

import os
import pytest
import asyncio
import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import MagicMock, AsyncMock

from common_lib.config import (
    ConfigManager,
    DatabaseConfig,
    LoggingConfig,
    ServiceClientConfig,
    RetryConfig,
    CircuitBreakerConfig
)
from common_lib.service_client import ResilientServiceClient


@pytest.fixture
def performance_metrics():
    """Create a performance metrics collector."""
    class PerformanceMetrics:
        def __init__(self):
            self.metrics = {}
        
        def record(self, name: str, duration: float):
            """Record a performance metric."""
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
        
        def get_stats(self, name: str) -> Dict[str, float]:
            """Get statistics for a performance metric."""
            if name not in self.metrics or not self.metrics[name]:
                return {
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "median": 0,
                    "p95": 0,
                    "p99": 0,
                    "count": 0
                }
            
            values = self.metrics[name]
            sorted_values = sorted(values)
            p95_index = int(len(sorted_values) * 0.95)
            p99_index = int(len(sorted_values) * 0.99)
            
            return {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": sorted_values[p95_index],
                "p99": sorted_values[p99_index],
                "count": len(values)
            }
        
        def get_all_stats(self) -> Dict[str, Dict[str, float]]:
            """Get statistics for all performance metrics."""
            return {name: self.get_stats(name) for name in self.metrics}
        
        def clear(self):
            """Clear all performance metrics."""
            self.metrics = {}
    
    return PerformanceMetrics()


@pytest.fixture
def time_it(performance_metrics):
    """Create a timer decorator."""
    def decorator(name: str):
    """
    Decorator.
    
    Args:
        name: Description of name
    
    """

        def inner_decorator(func):
    """
    Inner decorator.
    
    Args:
        func: Description of func
    
    """

            async def async_wrapper(*args, **kwargs):
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                performance_metrics.record(name, duration)
                return result
            
            def sync_wrapper(*args, **kwargs):
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_metrics.record(name, duration)
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return inner_decorator
    
    return decorator


@pytest.fixture
def mock_database():
    """Create a mock database with performance metrics."""
    class PerformanceMockDatabase:
        def __init__(self, performance_metrics):
            self.performance_metrics = performance_metrics
            self.pool = None
        
        async def connect(self):
            """Connect to the database."""
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate connection time
            duration = time.time() - start_time
            self.performance_metrics.record("database_connect", duration)
            self.pool = MagicMock()
        
        async def close(self):
            """Close the database connection."""
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate disconnection time
            duration = time.time() - start_time
            self.performance_metrics.record("database_close", duration)
            self.pool = None
        
        async def execute(self, query, *args, **kwargs):
            """Execute a query."""
            start_time = time.time()
            await asyncio.sleep(0.02)  # Simulate query execution time
            duration = time.time() - start_time
            self.performance_metrics.record("database_execute", duration)
            return "OK"
        
        async def fetch(self, query, *args, **kwargs):
            """Fetch rows from the database."""
            start_time = time.time()
            await asyncio.sleep(0.03)  # Simulate query execution time
            duration = time.time() - start_time
            self.performance_metrics.record("database_fetch", duration)
            return [{"id": 1, "name": "test"}]
        
        async def fetchrow(self, query, *args, **kwargs):
            """Fetch a single row from the database."""
            start_time = time.time()
            await asyncio.sleep(0.02)  # Simulate query execution time
            duration = time.time() - start_time
            self.performance_metrics.record("database_fetchrow", duration)
            return {"id": 1, "name": "test"}
        
        async def fetchval(self, query, *args, **kwargs):
            """Fetch a single value from the database."""
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate query execution time
            duration = time.time() - start_time
            self.performance_metrics.record("database_fetchval", duration)
            return 1
        
        async def transaction(self):
            """Start a transaction."""
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate transaction start time
            duration = time.time() - start_time
            self.performance_metrics.record("database_transaction_start", duration)
            
            class Transaction:
    """
    Transaction class.
    
    Attributes:
        Add attributes here
    """

                async def __aenter__(self):
    """
      aenter  .
    
    """

                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
    """
      aexit  .
    
    Args:
        exc_type: Description of exc_type
        exc_val: Description of exc_val
        exc_tb: Description of exc_tb
    
    """

                    start_time = time.time()
                    await asyncio.sleep(0.01)  # Simulate transaction end time
                    duration = time.time() - start_time
                    self.performance_metrics.record("database_transaction_end", duration)
            
            return Transaction()
    
    return PerformanceMockDatabase(performance_metrics)


@pytest.fixture
def mock_service_client(performance_metrics):
    """Create a mock service client with performance metrics."""
    class PerformanceMockServiceClient:
        def __init__(self, performance_metrics):
            self.performance_metrics = performance_metrics
        
        async def get(self, path, **kwargs):
            """Make a GET request."""
            start_time = time.time()
            await asyncio.sleep(0.05)  # Simulate request time
            duration = time.time() - start_time
            self.performance_metrics.record("service_client_get", duration)
            return {"status": "success"}
        
        async def post(self, path, **kwargs):
            """Make a POST request."""
            start_time = time.time()
            await asyncio.sleep(0.07)  # Simulate request time
            duration = time.time() - start_time
            self.performance_metrics.record("service_client_post", duration)
            return {"status": "success", "id": "123"}
        
        async def put(self, path, **kwargs):
            """Make a PUT request."""
            start_time = time.time()
            await asyncio.sleep(0.06)  # Simulate request time
            duration = time.time() - start_time
            self.performance_metrics.record("service_client_put", duration)
            return {"status": "success"}
        
        async def delete(self, path, **kwargs):
            """Make a DELETE request."""
            start_time = time.time()
            await asyncio.sleep(0.04)  # Simulate request time
            duration = time.time() - start_time
            self.performance_metrics.record("service_client_delete", duration)
            return {"status": "success"}
        
        async def connect(self):
            """Connect to the service."""
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate connection time
            duration = time.time() - start_time
            self.performance_metrics.record("service_client_connect", duration)
        
        async def close(self):
            """Close the connection to the service."""
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate disconnection time
            duration = time.time() - start_time
            self.performance_metrics.record("service_client_close", duration)
    
    return PerformanceMockServiceClient(performance_metrics)
