#!/usr/bin/env python3
"""
Distributed tracing setup for the service.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# Constants
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown-service")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "localhost:4317")

def setup_tracing(
    service_name: str = SERVICE_NAME,
    environment: str = ENVIRONMENT,
    otlp_endpoint: str = OTLP_ENDPOINT
) -> None:
    """
    Set up distributed tracing.
    
    Args:
        service_name: Name of the service
        environment: Deployment environment
        otlp_endpoint: OpenTelemetry collector endpoint
    """
    # Create resource
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Create exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    
    # Create processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    
    # Add processor to provider
    tracer_provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer.
    
    Args:
        name: Tracer name
        
    Returns:
        Tracer
    """
    return trace.get_tracer(name)
