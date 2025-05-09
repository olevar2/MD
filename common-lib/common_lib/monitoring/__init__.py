"""
Monitoring package for common library.

This package provides monitoring utilities for the forex trading platform,
including distributed tracing, metrics collection, and performance monitoring.
"""

from common_lib.monitoring.tracing import (
    setup_tracing,
    get_tracer,
    trace_function,
    trace_async_function,
    inject_trace_context,
    extract_trace_context,
    instrument_fastapi,
    instrument_aiohttp_client,
    instrument_asyncpg,
    instrument_redis
)

from common_lib.monitoring.metrics import (
    setup_metrics,
    get_counter,
    get_gauge,
    get_histogram,
    get_summary,
    track_execution_time,
    track_memory_usage
)

__all__ = [
    # Tracing
    "setup_tracing",
    "get_tracer",
    "trace_function",
    "trace_async_function",
    "inject_trace_context",
    "extract_trace_context",
    "instrument_fastapi",
    "instrument_aiohttp_client",
    "instrument_asyncpg",
    "instrument_redis",
    
    # Metrics
    "setup_metrics",
    "get_counter",
    "get_gauge",
    "get_histogram",
    "get_summary",
    "track_execution_time",
    "track_memory_usage"
]
