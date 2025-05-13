"""
Distributed tracing package for the ML Integration Service.
"""

from core.tracing_1 import (
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
