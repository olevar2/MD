"""
Distributed Tracing Module for Forex Trading Platform.

This module provides distributed tracing capabilities using OpenTelemetry,
allowing for detailed request tracing across services.
"""

import os
import logging
import functools
from typing import Dict, Any, Optional, Callable, List, Union, TypeVar, cast
import asyncio
import inspect
from contextlib import contextmanager

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Default configuration
DEFAULT_OTLP_ENDPOINT = "http://jaeger:4317"
DEFAULT_SERVICE_NAME = "forex-trading-service"
DEFAULT_SERVICE_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "development"

# Global variables
_tracer_provider = None
_tracer = None

def setup_tracing(
    service_name: str = None,
    service_version: str = None,
    otlp_endpoint: str = None,
    environment: str = None,
    sampling_rate: float = 1.0
) -> None:
    """
    Set up OpenTelemetry tracing.
    
    This function initializes the OpenTelemetry tracer provider and configures
    exporters and instrumentors.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: Endpoint for the OpenTelemetry collector
        environment: Deployment environment
        sampling_rate: Sampling rate for traces (0.0-1.0)
    """
    global _tracer_provider, _tracer
    
    # Get configuration from parameters or environment variables
    service_name = service_name or os.environ.get("SERVICE_NAME", DEFAULT_SERVICE_NAME)
    service_version = service_version or os.environ.get("SERVICE_VERSION", DEFAULT_SERVICE_VERSION)
    otlp_endpoint = otlp_endpoint or os.environ.get("OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    environment = environment or os.environ.get("ENVIRONMENT", DEFAULT_ENVIRONMENT)
    
    # Create a resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": environment
    })
    
    # Create a tracer provider with the resource
    _tracer_provider = TracerProvider(resource=resource)
    
    # Configure the OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    
    # Add the exporter to the tracer provider
    _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set the tracer provider
    trace.set_tracer_provider(_tracer_provider)
    
    # Get the tracer
    _tracer = trace.get_tracer(service_name, service_version)
    
    logger.info(
        f"OpenTelemetry tracing initialized for {service_name} ({service_version}) "
        f"with exporter at {otlp_endpoint}"
    )

def get_tracer():
    """
    Get the OpenTelemetry tracer.
    
    Returns:
        OpenTelemetry tracer
    """
    global _tracer
    if _tracer is None:
        setup_tracing()
    return _tracer

def instrument_fastapi(app):
    """
    Instrument a FastAPI application for distributed tracing.
    
    Args:
        app: FastAPI application
    """
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    logger.info("FastAPI instrumented for distributed tracing")

def instrument_aiohttp_client():
    """Instrument aiohttp client for distributed tracing."""
    AioHttpClientInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    logger.info("aiohttp client instrumented for distributed tracing")

def instrument_asyncpg():
    """Instrument asyncpg for distributed tracing."""
    AsyncPGInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    logger.info("asyncpg instrumented for distributed tracing")

def instrument_redis():
    """Instrument Redis for distributed tracing."""
    RedisInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    logger.info("Redis instrumented for distributed tracing")

def trace_function(name: str = None) -> Callable[[F], F]:
    """
    Decorator for tracing synchronous functions.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the tracer
            tracer = get_tracer()
            
            # Get the span name
            span_name = name or func.__name__
            
            # Start a new span
            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as span attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record the exception
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return cast(F, wrapper)
    
    return decorator

def trace_async_function(name: str = None) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator for tracing asynchronous functions.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the tracer
            tracer = get_tracer()
            
            # Get the span name
            span_name = name or func.__name__
            
            # Start a new span
            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as span attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    # Call the function
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record the exception
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return cast(AsyncF, wrapper)
    
    return decorator

def inject_trace_context(carrier: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into a carrier for propagation.
    
    Args:
        carrier: Dictionary to inject context into
        
    Returns:
        Carrier with injected context
    """
    propagator = TraceContextTextMapPropagator()
    propagator.inject(carrier)
    return carrier

def extract_trace_context(carrier: Dict[str, str]) -> None:
    """
    Extract trace context from a carrier.
    
    Args:
        carrier: Dictionary containing trace context
    """
    propagator = TraceContextTextMapPropagator()
    context = propagator.extract(carrier)
    return context

@contextmanager
def trace_span(name: str, attributes: Dict[str, Any] = None):
    """
    Context manager for creating a trace span.
    
    Args:
        name: Name of the span
        attributes: Span attributes
        
    Yields:
        The created span
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
