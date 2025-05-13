"""
Monitoring Configuration for Strategy Execution Engine

This module provides monitoring configuration for the Strategy Execution Engine.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, Info

from config.config_1 import get_settings

logger = logging.getLogger(__name__)

# Prometheus metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

strategy_executions_total = Counter(
    "strategy_executions_total",
    "Total number of strategy executions",
    ["strategy_id", "status"]
)

strategy_execution_duration_seconds = Histogram(
    "strategy_execution_duration_seconds",
    "Strategy execution duration in seconds",
    ["strategy_id"]
)

backtest_executions_total = Counter(
    "backtest_executions_total",
    "Total number of backtest executions",
    ["strategy_id", "status"]
)

backtest_execution_duration_seconds = Histogram(
    "backtest_execution_duration_seconds",
    "Backtest execution duration in seconds",
    ["strategy_id"]
)

active_strategies = Gauge(
    "active_strategies",
    "Number of active strategies"
)

service_info = Info(
    "service_info",
    "Service information"
)

async def setup_monitoring(app: FastAPI) -> None:
    """
    Set up monitoring for the application.
    
    Args:
        app: FastAPI application instance
    """
    settings = get_settings()
    
    if not settings.enable_prometheus:
        logger.info("Prometheus metrics disabled")
        return
    
    # Set service info
    service_info.info({
        "name": "strategy_execution_engine",
        "version": settings.app_version,
        "environment": "production" if not settings.debug_mode else "development"
    })
    
    logger.info("Prometheus metrics initialized")

def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """
    Record request metrics.
    
    Args:
        method: HTTP method
        endpoint: Endpoint path
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    settings = get_settings()
    
    if not settings.enable_prometheus:
        return
    
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

def record_strategy_execution_metrics(strategy_id: str, status: str, duration: float) -> None:
    """
    Record strategy execution metrics.
    
    Args:
        strategy_id: Strategy ID
        status: Execution status
        duration: Execution duration in seconds
    """
    settings = get_settings()
    
    if not settings.enable_prometheus:
        return
    
    strategy_executions_total.labels(strategy_id=strategy_id, status=status).inc()
    strategy_execution_duration_seconds.labels(strategy_id=strategy_id).observe(duration)

def record_backtest_execution_metrics(strategy_id: str, status: str, duration: float) -> None:
    """
    Record backtest execution metrics.
    
    Args:
        strategy_id: Strategy ID
        status: Execution status
        duration: Execution duration in seconds
    """
    settings = get_settings()
    
    if not settings.enable_prometheus:
        return
    
    backtest_executions_total.labels(strategy_id=strategy_id, status=status).inc()
    backtest_execution_duration_seconds.labels(strategy_id=strategy_id).observe(duration)

def update_active_strategies_count(count: int) -> None:
    """
    Update active strategies count.
    
    Args:
        count: Number of active strategies
    """
    settings = get_settings()
    
    if not settings.enable_prometheus:
        return
    
    active_strategies.set(count)
