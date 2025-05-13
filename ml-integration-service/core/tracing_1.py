"""
Distributed tracing for the ML Integration Service.

This module provides functionality for distributed tracing using OpenTelemetry.
"""
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Union, List
import logging
import functools
import inspect
import asyncio
from contextvars import ContextVar
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.context.context import Context
from config.enhanced_settings import enhanced_settings
logger = logging.getLogger(__name__)
tracer_provider = TracerProvider(resource=Resource.create({SERVICE_NAME:
    'ml-integration-service'}))
jaeger_exporter = JaegerExporter(agent_host_name=enhanced_settings.
    JAEGER_AGENT_HOST, agent_port=enhanced_settings.JAEGER_AGENT_PORT)
tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
current_span_var = ContextVar('current_span', default=None)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def get_current_span():
    """
    Get the current span from the context.
    
    Returns:
        Current span, or None if no span is active
    """
    return current_span_var.get()


def set_current_span(span):
    """
    Set the current span in the context.
    
    Args:
        span: Span to set as current
    """
    current_span_var.set(span)


@with_exception_handling
def trace_method(name: Optional[str]=None, attributes: Optional[Dict[str,
    Any]]=None):
    """
    Decorator for tracing methods.
    
    Args:
        name: Name of the span
        attributes: Attributes to add to the span
        
    Returns:
        Decorated method
    """

    @with_exception_handling
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(self, *args, **kwargs):
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            method_name = name or func.__name__
            class_name = self.__class__.__name__
            span_name = f'{class_name}.{method_name}'
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            span_attributes = {}
            for param_name, param_value in bound_args.arguments.items():
                if param_name != 'self':
                    span_attributes[f'method.{param_name}'] = str(param_value)
            if attributes:
                span_attributes.update(attributes)
            with tracer.start_as_current_span(span_name, attributes=
                span_attributes) as span:
                set_current_span(span)
                try:
                    result = await func(self, *args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(self, *args, **kwargs):
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            method_name = name or func.__name__
            class_name = self.__class__.__name__
            span_name = f'{class_name}.{method_name}'
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            span_attributes = {}
            for param_name, param_value in bound_args.arguments.items():
                if param_name != 'self':
                    span_attributes[f'method.{param_name}'] = str(param_value)
            if attributes:
                span_attributes.update(attributes)
            with tracer.start_as_current_span(span_name, attributes=
                span_attributes) as span:
                set_current_span(span)
                try:
                    result = func(self, *args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


@with_exception_handling
def trace_function(name: Optional[str]=None, attributes: Optional[Dict[str,
    Any]]=None):
    """
    Decorator for tracing functions.
    
    Args:
        name: Name of the span
        attributes: Attributes to add to the span
        
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
        @async_with_exception_handling
        async def async_wrapper(*args, **kwargs):
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            function_name = name or func.__name__
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            span_attributes = {}
            for param_name, param_value in bound_args.arguments.items():
                span_attributes[f'function.{param_name}'] = str(param_value)
            if attributes:
                span_attributes.update(attributes)
            with tracer.start_as_current_span(function_name, attributes=
                span_attributes) as span:
                set_current_span(span)
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args, **kwargs):
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            function_name = name or func.__name__
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            span_attributes = {}
            for param_name, param_value in bound_args.arguments.items():
                span_attributes[f'function.{param_name}'] = str(param_value)
            if attributes:
                span_attributes.update(attributes)
            with tracer.start_as_current_span(function_name, attributes=
                span_attributes) as span:
                set_current_span(span)
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
