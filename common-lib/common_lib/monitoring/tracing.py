"""
Tracing Module

This module provides distributed tracing functionality for the platform.
"""

import logging
import functools
import inspect
import asyncio
from typing import Dict, Any, Optional, List, Callable, ClassVar, Union, TypeVar, cast

import opentelemetry.trace as trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.context import Context, get_current

from common_lib.config.config_manager import ConfigManager


T = TypeVar('T')


class TracingManager:
    """
    Tracing manager for the platform.
    
    This class provides a singleton manager for distributed tracing.
    """
    
    _instance: ClassVar[Optional["TracingManager"]] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the tracing manager.
        
        Returns:
            Singleton instance of the tracing manager
        """
        if cls._instance is None:
            cls._instance = super(TracingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        service_name: str,
        exporter_type: str = "jaeger",
        exporter_endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the tracing manager.
        
        Args:
            service_name: Name of the service
            exporter_type: Type of the exporter (jaeger or otlp)
            exporter_endpoint: Endpoint of the exporter
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.service_name = service_name
        self.exporter_type = exporter_type
        self.exporter_endpoint = exporter_endpoint
        
        # Initialize tracing
        self._initialize_tracing()
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
        # Create propagator
        self.propagator = TraceContextTextMapPropagator()
        
        self._initialized = True
    
    def _initialize_tracing(self):
        """
        Initialize tracing.
        """
        # Create resource
        resource = Resource(attributes={
            SERVICE_NAME: self.service_name
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Create exporter
        if self.exporter_type.lower() == "jaeger":
            # Create Jaeger exporter
            exporter = JaegerExporter(
                agent_host_name=self.exporter_endpoint.split(":")[0] if self.exporter_endpoint else "localhost",
                agent_port=int(self.exporter_endpoint.split(":")[1]) if self.exporter_endpoint and ":" in self.exporter_endpoint else 6831
            )
        elif self.exporter_type.lower() == "otlp":
            # Create OTLP exporter
            exporter = OTLPSpanExporter(
                endpoint=self.exporter_endpoint or "localhost:4317"
            )
        else:
            raise ValueError(f"Invalid exporter type: {self.exporter_type}")
        
        # Create span processor
        processor = BatchSpanProcessor(exporter)
        
        # Add span processor to tracer provider
        provider.add_span_processor(processor)
        
        # Set tracer provider
        trace.set_tracer_provider(provider)
    
    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: Optional[trace.SpanKind] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> trace.Span:
        """
        Start a span.
        
        Args:
            name: Name of the span
            context: Context for the span
            kind: Kind of the span
            attributes: Attributes for the span
            
        Returns:
            Span
        """
        return self.tracer.start_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes
        )
    
    def inject_context(
        self,
        context: Optional[Context] = None,
        carrier: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Inject context into a carrier.
        
        Args:
            context: Context to inject
            carrier: Carrier to inject into
            
        Returns:
            Carrier with injected context
        """
        carrier = carrier or {}
        self.propagator.inject(carrier, context=context)
        return carrier
    
    def extract_context(
        self,
        carrier: Dict[str, str],
        context: Optional[Context] = None
    ) -> Context:
        """
        Extract context from a carrier.
        
        Args:
            carrier: Carrier to extract from
            context: Context to extract into
            
        Returns:
            Extracted context
        """
        return self.propagator.extract(carrier, context=context)
    
    def get_current_span(self) -> trace.Span:
        """
        Get the current span.
        
        Returns:
            Current span
        """
        return trace.get_current_span()
    
    def get_current_context(self) -> Context:
        """
        Get the current context.
        
        Returns:
            Current context
        """
        return get_current()


def trace_function(
    name: Optional[str] = None,
    kind: Optional[trace.SpanKind] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator for tracing function execution.
    
    Args:
        name: Name of the span (if None, uses the function name)
        kind: Kind of the span
        attributes: Attributes for the span
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Get function name
        func_name = name or func.__name__
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get tracing manager
                tracing_manager = TracingManager()
                
                # Start span
                with tracing_manager.start_span(
                    func_name,
                    kind=kind,
                    attributes=attributes
                ) as span:
                    # Add function arguments to span
                    if args:
                        span.set_attribute("args", str(args))
                    if kwargs:
                        span.set_attribute("kwargs", str(kwargs))
                    
                    try:
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Add result to span
                        span.set_attribute("result", str(result))
                        
                        return result
                    except Exception as e:
                        # Record exception
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        
                        # Re-raise exception
                        raise
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get tracing manager
                tracing_manager = TracingManager()
                
                # Start span
                with tracing_manager.start_span(
                    func_name,
                    kind=kind,
                    attributes=attributes
                ) as span:
                    # Add function arguments to span
                    if args:
                        span.set_attribute("args", str(args))
                    if kwargs:
                        span.set_attribute("kwargs", str(kwargs))
                    
                    try:
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Add result to span
                        span.set_attribute("result", str(result))
                        
                        return result
                    except Exception as e:
                        # Record exception
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        
                        # Re-raise exception
                        raise
            
            return sync_wrapper
    
    return decorator