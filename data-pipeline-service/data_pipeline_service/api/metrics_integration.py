"""
Metrics Integration for Data Pipeline Service.

This module integrates the standardized metrics middleware with the FastAPI application.
"""

import logging
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from common_lib.monitoring.middleware import StandardMetricsMiddleware
from common_lib.monitoring.metrics_standards import StandardMetrics
from data_pipeline_service.monitoring.service_metrics import DataPipelineMetrics

# Configure logging
logger = logging.getLogger(__name__)

def setup_metrics(app: FastAPI, service_name: str = "data-pipeline-service") -> None:
    """
    Set up metrics for the FastAPI application.
    
    Args:
        app: FastAPI application
        service_name: Name of the service
    """
    # Initialize metrics
    metrics = DataPipelineMetrics()
    
    # Add metrics middleware
    app.add_middleware(
        StandardMetricsMiddleware,
        service_name=service_name,
        exclude_paths=["/metrics", "/health", "/docs", "/redoc", "/openapi.json"],
        metrics_instance=metrics
    )
    
    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    logger.info(f"Metrics setup complete for {service_name}")
    
    return metrics
