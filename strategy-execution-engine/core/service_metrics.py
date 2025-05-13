"""
Service-specific metrics for the Strategy Execution Engine.

This module defines service-specific metrics for the Strategy Execution Engine,
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
    StandardMetrics, LATENCY_BUCKETS_FAST, LATENCY_BUCKETS_MEDIUM, LATENCY_BUCKETS_SLOW,
    PREFIX_STRATEGY, PREFIX_SYSTEM, PREFIX_EXECUTION,
    LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION, LABEL_INSTRUMENT
)

# Configure logging
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "strategy-execution-engine"

class StrategyExecutionMetrics(StandardMetrics):
    """
    Service-specific metrics for the Strategy Execution Engine.
    
    This class extends the StandardMetrics class to provide service-specific metrics
    for the Strategy Execution Engine.
    """
    
    def __init__(self):
        """Initialize the Strategy Execution metrics."""
        super().__init__(SERVICE_NAME)
        
        # Initialize service-specific metrics
        self._init_strategy_metrics()
        self._init_execution_metrics()
        self._init_backtest_metrics()
        self._init_performance_metrics()
        
        logger.info(f"Initialized Strategy Execution metrics")
    
    def _init_strategy_metrics(self) -> None:
        """Initialize strategy-related metrics."""
        # Strategy operations counter
        self.strategy_operations_total = get_counter(
            name=f"{PREFIX_STRATEGY}_operations_total",
            description="Total number of strategy operations",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION]
        )
        
        # Strategy operation duration
        self.strategy_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_STRATEGY}_operation_duration_seconds",
            description="Strategy operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Strategy errors
        self.strategy_errors_total = get_counter(
            name=f"{PREFIX_STRATEGY}_errors_total",
            description="Total number of strategy errors",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION, "error_type"]
        )
        
        # Strategy count
        self.strategy_count = get_gauge(
            name=f"{PREFIX_STRATEGY}_count",
            description="Number of strategies",
            labels=[LABEL_SERVICE, "strategy_type", "status"]
        )
        
        # Strategy signals
        self.strategy_signals_total = get_counter(
            name=f"{PREFIX_STRATEGY}_signals_total",
            description="Total number of strategy signals",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "signal_type"]
        )
        
        # Strategy signal quality
        self.strategy_signal_quality = get_gauge(
            name=f"{PREFIX_STRATEGY}_signal_quality",
            description="Strategy signal quality (0-1)",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "signal_type"]
        )
    
    def _init_execution_metrics(self) -> None:
        """Initialize execution-related metrics."""
        # Execution operations counter
        self.execution_operations_total = get_counter(
            name=f"{PREFIX_EXECUTION}_operations_total",
            description="Total number of execution operations",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION]
        )
        
        # Execution operation duration
        self.execution_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_EXECUTION}_operation_duration_seconds",
            description="Execution operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_FAST
        )
        
        # Execution errors
        self.execution_errors_total = get_counter(
            name=f"{PREFIX_EXECUTION}_errors_total",
            description="Total number of execution errors",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION, "error_type"]
        )
        
        # Execution trades
        self.execution_trades_total = get_counter(
            name=f"{PREFIX_EXECUTION}_trades_total",
            description="Total number of execution trades",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "trade_type"]
        )
        
        # Execution trade size
        self.execution_trade_size = get_histogram(
            name=f"{PREFIX_EXECUTION}_trade_size",
            description="Execution trade size",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "trade_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        )
        
        # Execution trade latency
        self.execution_trade_latency_seconds = get_histogram(
            name=f"{PREFIX_EXECUTION}_trade_latency_seconds",
            description="Execution trade latency in seconds",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "trade_type"],
            buckets=LATENCY_BUCKETS_FAST
        )
    
    def _init_backtest_metrics(self) -> None:
        """Initialize backtest-related metrics."""
        # Backtest operations counter
        self.backtest_operations_total = get_counter(
            name="backtest_operations_total",
            description="Total number of backtest operations",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION]
        )
        
        # Backtest operation duration
        self.backtest_operation_duration_seconds = get_histogram(
            name="backtest_operation_duration_seconds",
            description="Backtest operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_SLOW
        )
        
        # Backtest errors
        self.backtest_errors_total = get_counter(
            name="backtest_errors_total",
            description="Total number of backtest errors",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_OPERATION, "error_type"]
        )
        
        # Backtest data points
        self.backtest_data_points = get_gauge(
            name="backtest_data_points",
            description="Number of data points in backtest",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
        
        # Backtest trades
        self.backtest_trades_total = get_counter(
            name="backtest_trades_total",
            description="Total number of backtest trades",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "trade_type"]
        )
        
        # Backtest processing speed
        self.backtest_processing_speed = get_gauge(
            name="backtest_processing_speed",
            description="Backtest processing speed (data points per second)",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
    
    def _init_performance_metrics(self) -> None:
        """Initialize performance-related metrics."""
        # Performance metrics
        self.performance_profit_loss = get_gauge(
            name="performance_profit_loss",
            description="Performance profit/loss",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
        
        self.performance_win_rate = get_gauge(
            name="performance_win_rate",
            description="Performance win rate (0-1)",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
        
        self.performance_sharpe_ratio = get_gauge(
            name="performance_sharpe_ratio",
            description="Performance Sharpe ratio",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
        
        self.performance_max_drawdown = get_gauge(
            name="performance_max_drawdown",
            description="Performance maximum drawdown",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
        
        self.performance_profit_factor = get_gauge(
            name="performance_profit_factor",
            description="Performance profit factor",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
        
        self.performance_expectancy = get_gauge(
            name="performance_expectancy",
            description="Performance expectancy",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )
        
        self.performance_average_trade = get_gauge(
            name="performance_average_trade",
            description="Performance average trade",
            labels=[LABEL_SERVICE, LABEL_STRATEGY, LABEL_INSTRUMENT, "timeframe"]
        )


# Singleton instance
strategy_execution_metrics = StrategyExecutionMetrics()
