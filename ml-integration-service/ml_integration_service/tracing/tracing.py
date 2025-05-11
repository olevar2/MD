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

from ml_integration_service.config.enhanced_settings import enhanced_settings

# Setup logger
logger = logging.getLogger(__name__)

# Create a tracer provider
tracer_provider = TracerProvider(
    resource=Resource.create({SERVICE_NAME: "ml-integration-service"})
)

# Create a Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name=enhanced_settings.JAEGER_AGENT_HOST,
    agent_port=enhanced_settings.JAEGER_AGENT_PORT,
)

# Add the exporter to the tracer provider
tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

# Set the tracer provider
trace.set_tracer_provider(tracer_provider)

# Create a tracer
tracer = trace.get_tracer(__name__)

# Create a context variable for the current span
current_span_var = ContextVar("current_span", default=None)


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


def trace_method(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator for tracing methods.
    
    Args:
        name: Name of the span
        attributes: Attributes to add to the span
        
    Returns:
        Decorated method
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Get the method name
            method_name = name or func.__name__
            
            # Get the class name
            class_name = self.__class__.__name__
            
            # Create a span name
            span_name = f"{class_name}.{method_name}"
            
            # Get the method parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            
            # Create span attributes
            span_attributes = {}
            
            # Add method parameters to span attributes
            for param_name, param_value in bound_args.arguments.items():
                if param_name != "self":
                    # Convert param_value to string to avoid serialization issues
                    span_attributes[f"method.{param_name}"] = str(param_value)
            
            # Add custom attributes
            if attributes:
                span_attributes.update(attributes)
            
            # Start a span
            with tracer.start_as_current_span(span_name, attributes=span_attributes) as span:
                # Set the current span
                set_current_span(span)
                
                try:
                    # Call the method
                    result = await func(self, *args, **kwargs)
                    
                    # Set the span status
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                except Exception as e:
                    # Set the span status
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record the exception
                    span.record_exception(e)
                    
                    # Re-raise the exception
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Get the method name
            method_name = name or func.__name__
            
            # Get the class name
            class_name = self.__class__.__name__
            
            # Create a span name
            span_name = f"{class_name}.{method_name}"
            
            # Get the method parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            
            # Create span attributes
            span_attributes = {}
            
            # Add method parameters to span attributes
            for param_name, param_value in bound_args.arguments.items():
                if param_name != "self":
                    # Convert param_value to string to avoid serialization issues
                    span_attributes[f"method.{param_name}"] = str(param_value)
            
            # Add custom attributes
            if attributes:
                span_attributes.update(attributes)
            
            # Start a span
            with tracer.start_as_current_span(span_name, attributes=span_attributes) as span:
                # Set the current span
                set_current_span(span)
                
                try:
                    # Call the method
                    result = func(self, *args, **kwargs)
                    
                    # Set the span status
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                except Exception as e:
                    # Set the span status
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record the exception
                    span.record_exception(e)
                    
                    # Re-raise the exception
                    raise
        
        # Check if the method is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator for tracing functions.
    
    Args:
        name: Name of the span
        attributes: Attributes to add to the span
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the function name
            function_name = name or func.__name__
            
            # Get the function parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Create span attributes
            span_attributes = {}
            
            # Add function parameters to span attributes
            for param_name, param_value in bound_args.arguments.items():
                # Convert param_value to string to avoid serialization issues
                span_attributes[f"function.{param_name}"] = str(param_value)
            
            # Add custom attributes
            if attributes:
                span_attributes.update(attributes)
            
            # Start a span
            with tracer.start_as_current_span(function_name, attributes=span_attributes) as span:
                # Set the current span
                set_current_span(span)
                
                try:
                    # Call the function
                    result = await func(*args, **kwargs)
                    
                    # Set the span status
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                except Exception as e:
                    # Set the span status
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record the exception
                    span.record_exception(e)
                    
                    # Re-raise the exception
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get the function name
            function_name = name or func.__name__
            
            # Get the function parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Create span attributes
            span_attributes = {}
            
            # Add function parameters to span attributes
            for param_name, param_value in bound_args.arguments.items():
                # Convert param_value to string to avoid serialization issues
                span_attributes[f"function.{param_name}"] = str(param_value)
            
            # Add custom attributes
            if attributes:
                span_attributes.update(attributes)
            
            # Start a span
            with tracer.start_as_current_span(function_name, attributes=span_attributes) as span:
                # Set the current span
                set_current_span(span)
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Set the span status
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                except Exception as e:
                    # Set the span status
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record the exception
                    span.record_exception(e)
                    
                    # Re-raise the exception
                    raise
        
        # Check if the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
