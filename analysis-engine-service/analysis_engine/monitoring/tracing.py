"""
Distributed Tracing Module for Analysis Engine Service.

This module provides distributed tracing capabilities using OpenTelemetry,
allowing for detailed request tracing across services.
"""

import os
import logging
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import asyncio

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

# Local imports
from analysis_engine.monitoring.structured_logging import (
    get_structured_logger,
    get_correlation_id
)

logger = get_structured_logger(__name__)

# Get configuration from environment variables
OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "http://jaeger:4317")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "analysis-engine-service")
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "1.0.0")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# Initialize tracer provider
def setup_tracing():
    """
    Set up OpenTelemetry tracing.
    
    This function initializes the OpenTelemetry tracer provider and configures
    exporters and instrumentors.
    """
    # Create a resource with service information
    resource = Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "deployment.environment": ENVIRONMENT
    })
    
    # Create a tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Configure the OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT)
    
    # Add the exporter to the tracer provider
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set the tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Get the tracer
    tracer = trace.get_tracer(__name__, SERVICE_VERSION)
    
    logger.info(
        f"OpenTelemetry tracing initialized with exporter at {OTLP_ENDPOINT}",
        {
            "service_name": SERVICE_NAME,
            "service_version": SERVICE_VERSION,
            "environment": ENVIRONMENT
        }
    )
    
    return tracer

# Get the tracer
def get_tracer():
    """
    Get the OpenTelemetry tracer.
    
    Returns:
        OpenTelemetry tracer
    """
    return trace.get_tracer(__name__, SERVICE_VERSION)

# Instrument FastAPI
def instrument_fastapi(app):
    """
    Instrument a FastAPI application for distributed tracing.
    
    Args:
        app: FastAPI application
    """
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    logger.info("FastAPI instrumented for distributed tracing")

# Instrument aiohttp client
def instrument_aiohttp_client():
    """Instrument aiohttp client for distributed tracing."""
    AioHttpClientInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    logger.info("aiohttp client instrumented for distributed tracing")

# Instrument asyncpg
def instrument_asyncpg():
    """Instrument asyncpg for distributed tracing."""
    AsyncPGInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    logger.info("asyncpg instrumented for distributed tracing")

# Instrument Redis
def instrument_redis():
    """Instrument Redis for distributed tracing."""
    RedisInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    logger.info("Redis instrumented for distributed tracing")

# Decorator for tracing functions
def trace_function(name=None):
    """
    Decorator for tracing functions.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the tracer
            tracer = get_tracer()
            
            # Get the span name
            span_name = name or func.__name__
            
            # Start a span
            with tracer.start_as_current_span(span_name) as span:
                # Add correlation ID as an attribute
                correlation_id = get_correlation_id()
                if correlation_id:
                    span.set_attribute("correlation_id", correlation_id)
                
                # Add function arguments as attributes
                for i, arg in enumerate(args):
                    if isinstance(arg, (str, int, float, bool)):
                        span.set_attribute(f"arg_{i}", str(arg))
                
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"kwarg_{key}", str(value))
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record the exception
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    
    return decorator

# Decorator for tracing async functions
def trace_async_function(name=None):
    """
    Decorator for tracing async functions.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated async function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the tracer
            tracer = get_tracer()
            
            # Get the span name
            span_name = name or func.__name__
            
            # Start a span
            with tracer.start_as_current_span(span_name) as span:
                # Add correlation ID as an attribute
                correlation_id = get_correlation_id()
                if correlation_id:
                    span.set_attribute("correlation_id", correlation_id)
                
                # Add function arguments as attributes
                for i, arg in enumerate(args):
                    if isinstance(arg, (str, int, float, bool)):
                        span.set_attribute(f"arg_{i}", str(arg))
                
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"kwarg_{key}", str(value))
                
                try:
                    # Call the async function
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record the exception
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    
    return decorator

# Context manager for tracing code blocks
class TraceContext:
    """Context manager for tracing code blocks."""
    
    def __init__(self, name, attributes=None):
        """
        Initialize the trace context.
        
        Args:
            name: Name of the span
            attributes: Attributes to add to the span
        """
        self.name = name
        self.attributes = attributes or {}
        self.tracer = get_tracer()
        self.span = None
    
    def __enter__(self):
        """Enter the context manager."""
        # Start a span
        self.span = self.tracer.start_span(self.name)
        
        # Add attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        # Add correlation ID as an attribute
        correlation_id = get_correlation_id()
        if correlation_id:
            self.span.set_attribute("correlation_id", correlation_id)
        
        # Make the span the current span
        self.span.__enter__()
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if exc_type is not None:
            # Record the exception
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        
        # End the span
        self.span.__exit__(exc_type, exc_val, exc_tb)

# Async context manager for tracing code blocks
class AsyncTraceContext:
    """Async context manager for tracing code blocks."""
    
    def __init__(self, name, attributes=None):
        """
        Initialize the async trace context.
        
        Args:
            name: Name of the span
            attributes: Attributes to add to the span
        """
        self.name = name
        self.attributes = attributes or {}
        self.tracer = get_tracer()
        self.span = None
    
    async def __aenter__(self):
        """Enter the async context manager."""
        # Start a span
        self.span = self.tracer.start_span(self.name)
        
        # Add attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        # Add correlation ID as an attribute
        correlation_id = get_correlation_id()
        if correlation_id:
            self.span.set_attribute("correlation_id", correlation_id)
        
        # Make the span the current span
        self.span.__enter__()
        
        return self.span
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        if exc_type is not None:
            # Record the exception
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        
        # End the span
        self.span.__exit__(exc_type, exc_val, exc_tb)

# Propagate trace context to external services
def inject_trace_context(headers):
    """
    Inject trace context into HTTP headers.
    
    Args:
        headers: HTTP headers dictionary
        
    Returns:
        Updated HTTP headers dictionary
    """
    # Get the propagator
    propagator = TraceContextTextMapPropagator()
    
    # Inject the trace context
    propagator.inject(headers)
    
    return headers

# Extract trace context from external services
def extract_trace_context(headers):
    """
    Extract trace context from HTTP headers.
    
    Args:
        headers: HTTP headers dictionary
        
    Returns:
        Trace context
    """
    # Get the propagator
    propagator = TraceContextTextMapPropagator()
    
    # Extract the trace context
    context = propagator.extract(headers)
    
    return context
