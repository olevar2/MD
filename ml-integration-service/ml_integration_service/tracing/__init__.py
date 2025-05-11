"""
Distributed tracing package for the ML Integration Service.
"""

from ml_integration_service.tracing.tracing import (
    trace_method,
    trace_function,
    get_current_span,
    set_current_span,
    tracer,
)

__all__ = [
    "trace_method",
    "trace_function",
    "get_current_span",
    "set_current_span",
    "tracer",
]
