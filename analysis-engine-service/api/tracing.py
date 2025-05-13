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
from analysis_engine.monitoring.structured_logging import get_structured_logger, get_correlation_id
logger = get_structured_logger(__name__)
OTLP_ENDPOINT = os.environ.get('OTLP_ENDPOINT', 'http://jaeger:4317')
SERVICE_NAME = os.environ.get('SERVICE_NAME', 'analysis-engine-service')
SERVICE_VERSION = os.environ.get('SERVICE_VERSION', '1.0.0')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def setup_tracing():
    """
    Set up OpenTelemetry tracing.
    
    This function initializes the OpenTelemetry tracer provider and configures
    exporters and instrumentors.
    """
    resource = Resource.create({'service.name': SERVICE_NAME,
        'service.version': SERVICE_VERSION, 'deployment.environment':
        ENVIRONMENT})
    tracer_provider = TracerProvider(resource=resource)
    otlp_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__, SERVICE_VERSION)
    logger.info(
        f'OpenTelemetry tracing initialized with exporter at {OTLP_ENDPOINT}',
        {'service_name': SERVICE_NAME, 'service_version': SERVICE_VERSION,
        'environment': ENVIRONMENT})
    return tracer


def get_tracer():
    """
    Get the OpenTelemetry tracer.
    
    Returns:
        OpenTelemetry tracer
    """
    return trace.get_tracer(__name__, SERVICE_VERSION)


def instrument_fastapi(app):
    """
    Instrument a FastAPI application for distributed tracing.
    
    Args:
        app: FastAPI application
    """
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.
        get_tracer_provider())
    logger.info('FastAPI instrumented for distributed tracing')


def instrument_aiohttp_client():
    """Instrument aiohttp client for distributed tracing."""
    AioHttpClientInstrumentor().instrument(tracer_provider=trace.
        get_tracer_provider())
    logger.info('aiohttp client instrumented for distributed tracing')


def instrument_asyncpg():
    """Instrument asyncpg for distributed tracing."""
    AsyncPGInstrumentor().instrument(tracer_provider=trace.
        get_tracer_provider())
    logger.info('asyncpg instrumented for distributed tracing')


def instrument_redis():
    """Instrument Redis for distributed tracing."""
    RedisInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    logger.info('Redis instrumented for distributed tracing')


@with_exception_handling
def trace_function(name=None):
    """
    Decorator for tracing functions.
    
    Args:
        name: Name of the span (defaults to function name)
        
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


        @wraps(func)
        @with_exception_handling
        def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            tracer = get_tracer()
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                correlation_id = get_correlation_id()
                if correlation_id:
                    span.set_attribute('correlation_id', correlation_id)
                for i, arg in enumerate(args):
                    if isinstance(arg, (str, int, float, bool)):
                        span.set_attribute(f'arg_{i}', str(arg))
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f'kwarg_{key}', str(value))
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator


@with_exception_handling
def trace_async_function(name=None):
    """
    Decorator for tracing async functions.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated async function
    """

    @with_exception_handling
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """


        @wraps(func)
        @async_with_exception_handling
        async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            tracer = get_tracer()
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                correlation_id = get_correlation_id()
                if correlation_id:
                    span.set_attribute('correlation_id', correlation_id)
                for i, arg in enumerate(args):
                    if isinstance(arg, (str, int, float, bool)):
                        span.set_attribute(f'arg_{i}', str(arg))
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f'kwarg_{key}', str(value))
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator


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
        self.span = self.tracer.start_span(self.name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        correlation_id = get_correlation_id()
        if correlation_id:
            self.span.set_attribute('correlation_id', correlation_id)
        self.span.__enter__()
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if exc_type is not None:
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        self.span.__exit__(exc_type, exc_val, exc_tb)


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
        self.span = self.tracer.start_span(self.name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        correlation_id = get_correlation_id()
        if correlation_id:
            self.span.set_attribute('correlation_id', correlation_id)
        self.span.__enter__()
        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        if exc_type is not None:
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        self.span.__exit__(exc_type, exc_val, exc_tb)


def inject_trace_context(headers):
    """
    Inject trace context into HTTP headers.
    
    Args:
        headers: HTTP headers dictionary
        
    Returns:
        Updated HTTP headers dictionary
    """
    propagator = TraceContextTextMapPropagator()
    propagator.inject(headers)
    return headers


def extract_trace_context(headers):
    """
    Extract trace context from HTTP headers.
    
    Args:
        headers: HTTP headers dictionary
        
    Returns:
        Trace context
    """
    propagator = TraceContextTextMapPropagator()
    context = propagator.extract(headers)
    return context
