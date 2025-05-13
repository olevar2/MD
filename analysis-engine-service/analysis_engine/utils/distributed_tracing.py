"""
Distributed Tracing Module

This module provides distributed tracing capabilities for the forex trading platform,
enabling performance monitoring and troubleshooting across microservices.

Features:
- OpenTelemetry integration
- Automatic instrumentation of key components
- Context propagation across service boundaries
- Span creation and management
- Trace sampling and filtering
"""
import logging
import time
import uuid
import threading
import functools
import inspect
from typing import Dict, List, Any, Optional, Callable, Union, Set
from contextlib import contextmanager
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class DistributedTracer:
    """
    Distributed tracing implementation for the forex trading platform.
    
    Features:
    - OpenTelemetry integration when available
    - Fallback to lightweight local tracing when OpenTelemetry is not available
    - Automatic context propagation
    - Performance metrics collection
    """

    def __init__(self, service_name: str, enable_tracing: bool=True,
        sampling_rate: float=0.1, otlp_endpoint: Optional[str]=None):
        """
        Initialize the distributed tracer.
        
        Args:
            service_name: Name of the service
            enable_tracing: Whether to enable tracing
            sampling_rate: Sampling rate (0.0 to 1.0)
            otlp_endpoint: Optional OpenTelemetry collector endpoint
        """
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.sampling_rate = sampling_rate
        self.otlp_endpoint = otlp_endpoint
        self.context = threading.local()
        self.tracer = None
        if enable_tracing:
            if OPENTELEMETRY_AVAILABLE and otlp_endpoint:
                self._init_opentelemetry()
            else:
                self._init_local_tracer()
        logger.info(
            f"DistributedTracer initialized for service '{service_name}' with tracing {'enabled' if enable_tracing else 'disabled'}"
            )

    @with_exception_handling
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracer."""
        try:
            resource = Resource.create({ResourceAttributes.SERVICE_NAME:
                self.service_name})
            provider = TracerProvider(resource=resource)
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(self.service_name)
            logger.info(
                f"OpenTelemetry tracer initialized for service '{self.service_name}'"
                )
        except Exception as e:
            logger.error(f'Failed to initialize OpenTelemetry tracer: {e}',
                exc_info=True)
            self._init_local_tracer()

    def _init_local_tracer(self):
        """Initialize local tracer as fallback."""
        self.tracer = LocalTracer(self.service_name)
        logger.info(
            f"Local tracer initialized for service '{self.service_name}'")

    @contextmanager
    @with_exception_handling
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]]=None):
        """
        Start a new span.
        
        Args:
            name: Span name
            attributes: Optional span attributes
            
        Yields:
            Span object
        """
        if not self.enable_tracing:
            yield None
            return
        if self.sampling_rate < 1.0 and self.sampling_rate <= 0.0:
            if not hasattr(self.context, 'trace_id'
                ) or not self.context.trace_id:
                if time.time() % 100 / 100 > self.sampling_rate:
                    yield None
                    return
        parent_span = getattr(self.context, 'current_span', None)
        if OPENTELEMETRY_AVAILABLE and isinstance(self.tracer, trace.Tracer):
            with self.tracer.start_as_current_span(name, attributes=attributes
                ) as span:
                self.context.current_span = span
                yield span
        else:
            span = self.tracer.start_span(name, parent_span, attributes)
            prev_span = getattr(self.context, 'current_span', None)
            self.context.current_span = span
            try:
                yield span
            finally:
                self.context.current_span = prev_span
                span.end()

    @with_exception_handling
    def trace(self, name: Optional[str]=None, attributes: Optional[Dict[str,
        Any]]=None):
        """
        Decorator for tracing functions.
        
        Args:
            name: Optional span name (defaults to function name)
            attributes: Optional span attributes
            
        Returns:
            Decorated function
        """

        @with_exception_handling
        def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """


            @functools.wraps(func)
            @with_exception_handling
            def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

                if not self.enable_tracing:
                    return func(*args, **kwargs)
                span_name = name or f'{func.__module__}.{func.__name__}'
                span_attrs = attributes.copy() if attributes else {}
                try:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for arg_name, arg_value in bound_args.arguments.items():
                        if arg_name in ('self', 'cls'):
                            continue
                        if isinstance(arg_value, (str, int, float, bool)):
                            span_attrs[f'arg.{arg_name}'] = str(arg_value)
                except Exception:
                    pass
                with self.start_span(span_name, span_attrs) as span:
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        if span:
                            span.set_attribute('execution_time_ms', 
                                execution_time * 1000)
                        return result
                    except Exception as e:
                        if span:
                            span.set_attribute('error', True)
                            span.set_attribute('error.type', e.__class__.
                                __name__)
                            span.set_attribute('error.message', str(e))
                        raise
            return wrapper
        if callable(name):
            func = name
            name = None
            return decorator(func)
        return decorator

    @with_resilience('get_current_trace_id')
    def get_current_trace_id(self) ->Optional[str]:
        """Get the current trace ID."""
        if not self.enable_tracing:
            return None
        if hasattr(self.context, 'current_span') and self.context.current_span:
            if OPENTELEMETRY_AVAILABLE and hasattr(self.context.
                current_span, 'get_span_context'):
                return format(self.context.current_span.get_span_context().
                    trace_id, '032x')
            elif hasattr(self.context.current_span, 'trace_id'):
                return self.context.current_span.trace_id
        return None

    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]
        ]=None):
        """
        Add an event to the current span.
        
        Args:
            name: Event name
            attributes: Optional event attributes
        """
        if not self.enable_tracing or not hasattr(self.context, 'current_span'
            ) or not self.context.current_span:
            return
        span = self.context.current_span
        if OPENTELEMETRY_AVAILABLE and hasattr(span, 'add_event'):
            span.add_event(name, attributes)
        elif hasattr(span, 'add_event'):
            span.add_event(name, attributes)


class LocalTracer:
    """
    Simple local tracer implementation for when OpenTelemetry is not available.
    """

    def __init__(self, service_name: str):
        """
        Initialize the local tracer.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name

    def start_span(self, name: str, parent_span=None, attributes=None
        ) ->'LocalSpan':
        """
        Start a new span.
        
        Args:
            name: Span name
            parent_span: Optional parent span
            attributes: Optional span attributes
            
        Returns:
            LocalSpan object
        """
        return LocalSpan(name, parent_span, attributes)


class LocalSpan:
    """
    Simple span implementation for local tracing.
    """

    def __init__(self, name: str, parent_span=None, attributes=None):
        """
        Initialize a local span.
        
        Args:
            name: Span name
            parent_span: Optional parent span
            attributes: Optional span attributes
        """
        self.name = name
        self.parent_span = parent_span
        self.attributes = attributes or {}
        self.events = []
        self.start_time = time.time()
        self.end_time = None
        if parent_span:
            self.trace_id = parent_span.trace_id
            self.parent_id = parent_span.span_id
        else:
            self.trace_id = str(uuid.uuid4())
            self.parent_id = None
        self.span_id = str(uuid.uuid4())

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes=None):
        """Add an event to the span."""
        self.events.append({'name': name, 'attributes': attributes or {},
            'timestamp': time.time()})

    def end(self):
        """End the span."""
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        logger.debug(
            f"Span '{self.name}' completed in {duration_ms:.2f}ms [trace_id={self.trace_id}, span_id={self.span_id}]"
            )
