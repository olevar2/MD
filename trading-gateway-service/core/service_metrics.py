"""
Service-specific metrics for the Trading Gateway Service.

This module defines service-specific metrics for the Trading Gateway Service,
extending the standardized metrics defined in common_lib.monitoring.metrics_standards.
"""

import logging
from typing import Dict, List, Optional, Any, Callable

# Import standard metrics
from common_lib.monitoring.metrics import (
    get_counter, get_gauge, get_histogram, get_summary,
    track_execution_time, track_memory_usage
)
from common_lib.monitoring.metrics_standards import (
    StandardMetrics, LATENCY_BUCKETS_FAST, LATENCY_BUCKETS_MEDIUM,
    PREFIX_BROKER, PREFIX_MARKET, PREFIX_SYSTEM,
    LABEL_SERVICE, LABEL_BROKER, LABEL_INSTRUMENT, LABEL_OPERATION
)

# Configure logging
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "trading-gateway-service"

class TradingGatewayMetrics(StandardMetrics):
    """
    Service-specific metrics for the Trading Gateway Service.
    
    This class extends the StandardMetrics class to provide service-specific metrics
    for the Trading Gateway Service.
    """
    
    def __init__(self):
        """Initialize the Trading Gateway metrics."""
        super().__init__(SERVICE_NAME)
        
        # Initialize service-specific metrics
        self._init_order_metrics()
        self._init_execution_metrics()
        self._init_market_data_metrics()
        self._init_broker_metrics()
        
        logger.info(f"Initialized Trading Gateway metrics")
    
    def _init_order_metrics(self) -> None:
        """Initialize order-related metrics."""
        # Order operations counter
        self.order_operations_total = get_counter(
            name="order_operations_total",
            description="Total number of order operations",
            labels=[LABEL_SERVICE, "order_type", "operation"]
        )
        
        # Order operation duration
        self.order_operation_duration_seconds = get_histogram(
            name="order_operation_duration_seconds",
            description="Order operation duration in seconds",
            labels=[LABEL_SERVICE, "order_type", "operation"],
            buckets=LATENCY_BUCKETS_FAST
        )
        
        # Order errors
        self.order_errors_total = get_counter(
            name="order_errors_total",
            description="Total number of order errors",
            labels=[LABEL_SERVICE, "order_type", "operation", "error_type"]
        )
        
        # Order count
        self.order_count = get_counter(
            name="order_count",
            description="Number of orders",
            labels=[LABEL_SERVICE, "order_type", "order_status", LABEL_INSTRUMENT]
        )
        
        # Order fill rate
        self.order_fill_rate = get_gauge(
            name="order_fill_rate",
            description="Order fill rate (0-1)",
            labels=[LABEL_SERVICE, "order_type", LABEL_INSTRUMENT]
        )
        
        # Order rejection rate
        self.order_rejection_rate = get_gauge(
            name="order_rejection_rate",
            description="Order rejection rate (0-1)",
            labels=[LABEL_SERVICE, "order_type", LABEL_INSTRUMENT]
        )
    
    def _init_execution_metrics(self) -> None:
        """Initialize execution-related metrics."""
        # Execution operations counter
        self.execution_operations_total = get_counter(
            name="execution_operations_total",
            description="Total number of execution operations",
            labels=[LABEL_SERVICE, "execution_type", "operation"]
        )
        
        # Execution operation duration
        self.execution_operation_duration_seconds = get_histogram(
            name="execution_operation_duration_seconds",
            description="Execution operation duration in seconds",
            labels=[LABEL_SERVICE, "execution_type", "operation"],
            buckets=LATENCY_BUCKETS_FAST
        )
        
        # Execution errors
        self.execution_errors_total = get_counter(
            name="execution_errors_total",
            description="Total number of execution errors",
            labels=[LABEL_SERVICE, "execution_type", "operation", "error_type"]
        )
        
        # Execution count
        self.execution_count = get_counter(
            name="execution_count",
            description="Number of executions",
            labels=[LABEL_SERVICE, "execution_type", LABEL_INSTRUMENT]
        )
        
        # Execution slippage
        self.execution_slippage_bps = get_histogram(
            name="execution_slippage_bps",
            description="Execution slippage in basis points",
            labels=[LABEL_SERVICE, "execution_type", LABEL_INSTRUMENT],
            buckets=[-50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50]
        )
        
        # Execution latency
        self.execution_latency_seconds = get_histogram(
            name="execution_latency_seconds",
            description="Execution latency in seconds",
            labels=[LABEL_SERVICE, "execution_type", LABEL_INSTRUMENT],
            buckets=LATENCY_BUCKETS_FAST
        )
    
    def _init_market_data_metrics(self) -> None:
        """Initialize market data metrics."""
        # Market data operations counter
        self.market_data_operations_total = get_counter(
            name="market_data_operations_total",
            description="Total number of market data operations",
            labels=[LABEL_SERVICE, "data_type", "operation"]
        )
        
        # Market data operation duration
        self.market_data_operation_duration_seconds = get_histogram(
            name="market_data_operation_duration_seconds",
            description="Market data operation duration in seconds",
            labels=[LABEL_SERVICE, "data_type", "operation"],
            buckets=LATENCY_BUCKETS_FAST
        )
        
        # Market data errors
        self.market_data_errors_total = get_counter(
            name="market_data_errors_total",
            description="Total number of market data errors",
            labels=[LABEL_SERVICE, "data_type", "operation", "error_type"]
        )
        
        # Market data update rate
        self.market_data_update_rate = get_gauge(
            name="market_data_update_rate",
            description="Market data update rate (updates per second)",
            labels=[LABEL_SERVICE, "data_type", LABEL_INSTRUMENT]
        )
        
        # Market data latency
        self.market_data_latency_seconds = get_histogram(
            name="market_data_latency_seconds",
            description="Market data latency in seconds",
            labels=[LABEL_SERVICE, "data_type", LABEL_INSTRUMENT],
            buckets=LATENCY_BUCKETS_FAST
        )
        
        # Market data quality
        self.market_data_quality = get_gauge(
            name="market_data_quality",
            description="Market data quality score (0-1)",
            labels=[LABEL_SERVICE, "data_type", LABEL_INSTRUMENT]
        )
    
    def _init_broker_metrics(self) -> None:
        """Initialize broker-specific metrics."""
        # Broker operations counter
        self.broker_operations_total = get_counter(
            name=f"{PREFIX_BROKER}_operations_total",
            description="Total number of broker operations",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION]
        )
        
        # Broker operation duration
        self.broker_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_BROKER}_operation_duration_seconds",
            description="Broker operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Broker errors
        self.broker_errors_total = get_counter(
            name=f"{PREFIX_BROKER}_errors_total",
            description="Total number of broker errors",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION, "error_type"]
        )
        
        # Broker connection status
        self.broker_connection_status = get_gauge(
            name=f"{PREFIX_BROKER}_connection_status",
            description="Broker connection status (1 = connected, 0 = disconnected)",
            labels=[LABEL_SERVICE, LABEL_BROKER]
        )
        
        # Broker request rate
        self.broker_request_rate = get_gauge(
            name=f"{PREFIX_BROKER}_request_rate",
            description="Broker request rate (requests per second)",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION]
        )
        
        # Broker rate limits
        self.broker_rate_limit_remaining = get_gauge(
            name=f"{PREFIX_BROKER}_rate_limit_remaining",
            description="Broker rate limit remaining",
            labels=[LABEL_SERVICE, LABEL_BROKER, LABEL_OPERATION]
        )


# Singleton instance
trading_gateway_metrics = TradingGatewayMetrics()
