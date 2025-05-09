"""
Query Performance Monitoring Module.

This module provides utilities for monitoring database query performance.
"""

import logging
import time
import functools
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import statistics

from prometheus_client import Histogram, Counter, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type', 'table', 'status']
)

QUERY_COUNT = Counter(
    'db_query_count_total',
    'Total number of database queries',
    ['query_type', 'table', 'status']
)

SLOW_QUERY_COUNT = Counter(
    'db_slow_query_count_total',
    'Total number of slow database queries',
    ['query_type', 'table']
)

CACHE_HIT_RATIO = Gauge(
    'db_query_cache_hit_ratio',
    'Database query cache hit ratio',
    ['query_type', 'table']
)

# In-memory metrics for detailed analysis
query_history = []
slow_query_threshold = 0.5  # seconds
max_history_size = 1000


def track_query_performance(query_type: str, table: str):
    """
    Decorator to track query performance.
    
    Args:
        query_type: Type of query (e.g., 'select', 'insert', 'update')
        table: Table being queried
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                # Execute the query
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                # Record query duration
                duration = time.time() - start_time
                
                # Update Prometheus metrics
                QUERY_DURATION.labels(query_type=query_type, table=table, status=status).observe(duration)
                QUERY_COUNT.labels(query_type=query_type, table=table, status=status).inc()
                
                # Check if this is a slow query
                if duration > slow_query_threshold:
                    SLOW_QUERY_COUNT.labels(query_type=query_type, table=table).inc()
                    
                    # Log slow query
                    logger.warning(
                        f"Slow query detected: {query_type} on {table} took {duration:.3f}s"
                    )
                
                # Add to query history
                query_info = {
                    "query_type": query_type,
                    "table": table,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "status": status
                }
                
                # Add query parameters if available
                if hasattr(args[0], "__dict__"):
                    # If first arg is self, get relevant attributes
                    query_info["params"] = {
                        k: v for k, v in args[0].__dict__.items() 
                        if k not in ["session", "pool", "db", "logger"]
                    }
                
                # Add to history with size limit
                query_history.append(query_info)
                if len(query_history) > max_history_size:
                    query_history.pop(0)
                
        return wrapper
    return decorator


def get_query_performance_stats() -> Dict[str, Any]:
    """
    Get query performance statistics.
    
    Returns:
        Dictionary with query performance statistics
    """
    if not query_history:
        return {
            "total_queries": 0,
            "avg_duration": 0,
            "slow_queries": 0,
            "error_rate": 0,
            "queries_per_table": {},
            "queries_per_type": {}
        }
    
    # Calculate statistics
    durations = [q["duration"] for q in query_history]
    avg_duration = statistics.mean(durations)
    slow_queries = sum(1 for d in durations if d > slow_query_threshold)
    error_count = sum(1 for q in query_history if q["status"] == "error")
    
    # Group by table
    queries_per_table = {}
    for q in query_history:
        table = q["table"]
        if table not in queries_per_table:
            queries_per_table[table] = {
                "count": 0,
                "avg_duration": 0,
                "slow_queries": 0,
                "error_count": 0
            }
        
        queries_per_table[table]["count"] += 1
        queries_per_table[table]["avg_duration"] = (
            (queries_per_table[table]["avg_duration"] * (queries_per_table[table]["count"] - 1) + q["duration"]) / 
            queries_per_table[table]["count"]
        )
        if q["duration"] > slow_query_threshold:
            queries_per_table[table]["slow_queries"] += 1
        if q["status"] == "error":
            queries_per_table[table]["error_count"] += 1
    
    # Group by query type
    queries_per_type = {}
    for q in query_history:
        query_type = q["query_type"]
        if query_type not in queries_per_type:
            queries_per_type[query_type] = {
                "count": 0,
                "avg_duration": 0,
                "slow_queries": 0,
                "error_count": 0
            }
        
        queries_per_type[query_type]["count"] += 1
        queries_per_type[query_type]["avg_duration"] = (
            (queries_per_type[query_type]["avg_duration"] * (queries_per_type[query_type]["count"] - 1) + q["duration"]) / 
            queries_per_type[query_type]["count"]
        )
        if q["duration"] > slow_query_threshold:
            queries_per_type[query_type]["slow_queries"] += 1
        if q["status"] == "error":
            queries_per_type[query_type]["error_count"] += 1
    
    return {
        "total_queries": len(query_history),
        "avg_duration": avg_duration,
        "slow_queries": slow_queries,
        "error_rate": error_count / len(query_history) if query_history else 0,
        "queries_per_table": queries_per_table,
        "queries_per_type": queries_per_type
    }


def get_slow_queries(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get the slowest queries.
    
    Args:
        limit: Maximum number of queries to return
        
    Returns:
        List of slow queries
    """
    # Sort by duration (descending)
    sorted_queries = sorted(query_history, key=lambda q: q["duration"], reverse=True)
    
    # Return top N
    return sorted_queries[:limit]


def update_slow_query_threshold(threshold: float) -> None:
    """
    Update the slow query threshold.
    
    Args:
        threshold: New threshold in seconds
    """
    global slow_query_threshold
    slow_query_threshold = threshold
    logger.info(f"Slow query threshold updated to {threshold}s")
