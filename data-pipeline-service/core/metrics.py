#!/usr/bin/env python3
"""
Metrics collection and export for the service.
"""

import logging
import time
import os
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response, Request

logger = logging.getLogger(__name__)

# Metrics router
metrics_router = APIRouter(tags=["Metrics"])

# Create a registry
registry = CollectorRegistry()

# Define metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    registry=registry
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"],
    registry=registry
)

# Business metrics
business_operation_total = Counter(
    "business_operation_total",
    "Total number of business operations",
    ["operation", "status"],
    registry=registry
)

business_operation_duration_seconds = Histogram(
    "business_operation_duration_seconds",
    "Business operation duration in seconds",
    ["operation"],
    registry=registry
)

@metrics_router.get("/metrics")
async def get_metrics() -> Response:
    """
    Endpoint to expose Prometheus metrics.
    
    Returns:
        Prometheus metrics in text format
    """
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )
