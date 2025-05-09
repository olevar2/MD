"""
Service-specific metrics for the Analysis Engine Service.

This module defines service-specific metrics for the Analysis Engine Service,
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
    StandardMetrics, LATENCY_BUCKETS_MEDIUM, LATENCY_BUCKETS_SLOW,
    PREFIX_MODEL, PREFIX_STRATEGY, PREFIX_FEATURE, PREFIX_MARKET,
    LABEL_SERVICE, LABEL_MODEL, LABEL_STRATEGY, LABEL_FEATURE, LABEL_INSTRUMENT, LABEL_TIMEFRAME
)

# Configure logging
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "analysis-engine-service"

class AnalysisEngineMetrics(StandardMetrics):
    """
    Service-specific metrics for the Analysis Engine Service.
    
    This class extends the StandardMetrics class to provide service-specific metrics
    for the Analysis Engine Service.
    """
    
    def __init__(self):
        """Initialize the Analysis Engine metrics."""
        super().__init__(SERVICE_NAME)
        
        # Initialize service-specific metrics
        self._init_analysis_metrics()
        self._init_pattern_metrics()
        self._init_signal_metrics()
        self._init_regime_metrics()
        
        logger.info(f"Initialized Analysis Engine metrics")
    
    def _init_analysis_metrics(self) -> None:
        """Initialize analysis metrics."""
        # Analysis operations counter
        self.analysis_operations_total = get_counter(
            name="analysis_operations_total",
            description="Total number of analysis operations",
            labels=[LABEL_SERVICE, "analysis_type"]
        )
        
        # Analysis operation duration
        self.analysis_operation_duration_seconds = get_histogram(
            name="analysis_operation_duration_seconds",
            description="Analysis operation duration in seconds",
            labels=[LABEL_SERVICE, "analysis_type"],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Analysis errors
        self.analysis_errors_total = get_counter(
            name="analysis_errors_total",
            description="Total number of analysis errors",
            labels=[LABEL_SERVICE, "analysis_type", "error_type"]
        )
        
        # Analysis quality score
        self.analysis_quality_score = get_gauge(
            name="analysis_quality_score",
            description="Analysis quality score (0-1)",
            labels=[LABEL_SERVICE, "analysis_type"]
        )
    
    def _init_pattern_metrics(self) -> None:
        """Initialize pattern recognition metrics."""
        # Pattern recognition operations counter
        self.pattern_recognition_operations_total = get_counter(
            name="pattern_recognition_operations_total",
            description="Total number of pattern recognition operations",
            labels=[LABEL_SERVICE, "pattern_type", LABEL_INSTRUMENT, LABEL_TIMEFRAME]
        )
        
        # Pattern recognition operation duration
        self.pattern_recognition_duration_seconds = get_histogram(
            name="pattern_recognition_duration_seconds",
            description="Pattern recognition operation duration in seconds",
            labels=[LABEL_SERVICE, "pattern_type", LABEL_INSTRUMENT, LABEL_TIMEFRAME],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Pattern recognition accuracy
        self.pattern_recognition_accuracy = get_gauge(
            name="pattern_recognition_accuracy",
            description="Pattern recognition accuracy (0-1)",
            labels=[LABEL_SERVICE, "pattern_type", LABEL_INSTRUMENT, LABEL_TIMEFRAME]
        )
        
        # Pattern detection count
        self.pattern_detection_count = get_counter(
            name="pattern_detection_count",
            description="Number of patterns detected",
            labels=[LABEL_SERVICE, "pattern_type", LABEL_INSTRUMENT, LABEL_TIMEFRAME]
        )
    
    def _init_signal_metrics(self) -> None:
        """Initialize signal generation metrics."""
        # Signal generation operations counter
        self.signal_generation_operations_total = get_counter(
            name="signal_generation_operations_total",
            description="Total number of signal generation operations",
            labels=[LABEL_SERVICE, "signal_type", LABEL_INSTRUMENT, LABEL_TIMEFRAME]
        )
        
        # Signal generation operation duration
        self.signal_generation_duration_seconds = get_histogram(
            name="signal_generation_duration_seconds",
            description="Signal generation operation duration in seconds",
            labels=[LABEL_SERVICE, "signal_type", LABEL_INSTRUMENT, LABEL_TIMEFRAME],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Signal quality score
        self.signal_quality_score = get_gauge(
            name="signal_quality_score",
            description="Signal quality score (0-1)",
            labels=[LABEL_SERVICE, "signal_type", LABEL_INSTRUMENT, LABEL_TIMEFRAME]
        )
        
        # Signal count
        self.signal_count = get_counter(
            name="signal_count",
            description="Number of signals generated",
            labels=[LABEL_SERVICE, "signal_type", "signal_direction", LABEL_INSTRUMENT, LABEL_TIMEFRAME]
        )
    
    def _init_regime_metrics(self) -> None:
        """Initialize market regime metrics."""
        # Market regime detection operations counter
        self.market_regime_detection_operations_total = get_counter(
            name="market_regime_detection_operations_total",
            description="Total number of market regime detection operations",
            labels=[LABEL_SERVICE, LABEL_INSTRUMENT, LABEL_TIMEFRAME]
        )
        
        # Market regime detection operation duration
        self.market_regime_detection_duration_seconds = get_histogram(
            name="market_regime_detection_duration_seconds",
            description="Market regime detection operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_INSTRUMENT, LABEL_TIMEFRAME],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Market regime
        self.market_regime = get_gauge(
            name="market_regime",
            description="Current market regime (0 = ranging, 1 = trending, 2 = volatile)",
            labels=[LABEL_SERVICE, LABEL_INSTRUMENT, LABEL_TIMEFRAME, "regime_type"]
        )
        
        # Market regime confidence
        self.market_regime_confidence = get_gauge(
            name="market_regime_confidence",
            description="Market regime confidence (0-1)",
            labels=[LABEL_SERVICE, LABEL_INSTRUMENT, LABEL_TIMEFRAME, "regime_type"]
        )


# Singleton instance
analysis_engine_metrics = AnalysisEngineMetrics()
