"""
Enhanced Effectiveness Framework Metrics Exporter

This module exports metrics related to the Enhanced Effectiveness Framework to the monitoring system.
It tracks analysis requests, tool performance metrics across different timeframes and market regimes,
and statistical significance of results.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
import requests

from prometheus_client import Counter, Gauge, Histogram, Summary, push_to_gateway
from prometheus_client.exposition import basic_auth_handler

logger = logging.getLogger(__name__)

# Metrics configuration
PROMETHEUS_GATEWAY = os.environ.get("PROMETHEUS_GATEWAY", "localhost:9091")
PUSH_INTERVAL_SECONDS = int(os.environ.get("METRICS_PUSH_INTERVAL", "30"))
AUTH_USERNAME = os.environ.get("METRICS_AUTH_USERNAME", "")
AUTH_PASSWORD = os.environ.get("METRICS_AUTH_PASSWORD", "")

# Counter metrics
ANALYSIS_REQUESTS = Counter(
    'forex_effectiveness_analysis_requests',
    'Number of effectiveness analysis requests',
    ['analysis_type', 'tool_id']
)

REGIME_ANALYSIS_REQUESTS = Counter(
    'forex_regime_analysis_requests',
    'Number of regime-specific analysis requests',
    ['regime_type', 'tool_id']
)

CROSS_TIMEFRAME_REQUESTS = Counter(
    'forex_cross_timeframe_requests',
    'Number of cross-timeframe consistency analysis requests',
    ['tool_id']
)

# Gauge metrics
TOOL_EFFECTIVENESS_SCORE = Gauge(
    'forex_tool_effectiveness_score',
    'Current effectiveness score of trading tools (0.0-1.0)',
    ['tool_id', 'timeframe', 'regime']
)

PERFORMANCE_DECAY = Gauge(
    'forex_tool_performance_decay',
    'Performance decay rate of a tool over time (-1.0 to 1.0)',
    ['tool_id', 'timeframe']
)

STATISTICAL_SIGNIFICANCE = Gauge(
    'forex_effectiveness_stat_significance',
    'Statistical significance of effectiveness results (p-value)',
    ['tool_id', 'metric']
)

# Histogram metrics
ANALYSIS_DURATION = Histogram(
    'forex_effectiveness_analysis_duration_seconds',
    'Time taken to complete effectiveness analysis',
    ['analysis_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
)

EFFECTIVENESS_DISTRIBUTION = Histogram(
    'forex_effectiveness_score_distribution',
    'Distribution of effectiveness scores',
    ['tool_id'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
)

# Summary metrics
ANALYSIS_MEMORY_USAGE = Summary(
    'forex_effectiveness_analysis_memory_mb',
    'Memory usage during effectiveness analysis in MB',
    ['analysis_type']
)


class EnhancedEffectivenessMetricsExporter:
    """
    Exports metrics related to the Enhanced Effectiveness Framework for monitoring.
    """
    
    def __init__(self):
        """Initialize the metrics exporter."""
        self.last_push_time = datetime.now()
        self.tool_performance_cache = {}
    
    def record_analysis_request(self, analysis_type: str, tool_id: str, duration: float) -> None:
        """
        Record an effectiveness analysis request.
        
        Args:
            analysis_type: Type of analysis performed
            tool_id: ID of the tool being analyzed
            duration: Time taken to complete the analysis (seconds)
        """
        ANALYSIS_REQUESTS.labels(analysis_type=analysis_type, tool_id=tool_id).inc()
        ANALYSIS_DURATION.labels(analysis_type=analysis_type).observe(duration)
    
    def record_regime_analysis(self, regime_type: str, tool_id: str) -> None:
        """
        Record a regime-specific analysis request.
        
        Args:
            regime_type: Type of market regime (e.g., 'trending', 'ranging', 'volatile')
            tool_id: ID of the tool being analyzed
        """
        REGIME_ANALYSIS_REQUESTS.labels(regime_type=regime_type, tool_id=tool_id).inc()
    
    def record_cross_timeframe_analysis(self, tool_id: str) -> None:
        """
        Record a cross-timeframe consistency analysis request.
        
        Args:
            tool_id: ID of the tool being analyzed
        """
        CROSS_TIMEFRAME_REQUESTS.labels(tool_id=tool_id).inc()
    
    def update_tool_effectiveness(self, 
                                 tool_id: str, 
                                 timeframe: str, 
                                 regime: str, 
                                 score: float) -> None:
        """
        Update the effectiveness score for a tool.
        
        Args:
            tool_id: ID of the tool
            timeframe: Analysis timeframe
            regime: Market regime type
            score: Effectiveness score (0.0-1.0)
        """
        TOOL_EFFECTIVENESS_SCORE.labels(
            tool_id=tool_id,
            timeframe=timeframe,
            regime=regime
        ).set(score)
        
        # Record to effectiveness score distribution
        EFFECTIVENESS_DISTRIBUTION.labels(tool_id=tool_id).observe(score)
        
        # Update cache
        key = f"{tool_id}_{timeframe}_{regime}"
        self.tool_performance_cache[key] = score
    
    def update_performance_decay(self, tool_id: str, timeframe: str, decay_rate: float) -> None:
        """
        Update the performance decay rate for a tool.
        
        Args:
            tool_id: ID of the tool
            timeframe: Analysis timeframe
            decay_rate: Rate of performance decay (-1.0 to 1.0)
        """
        PERFORMANCE_DECAY.labels(tool_id=tool_id, timeframe=timeframe).set(decay_rate)
    
    def update_statistical_significance(self, tool_id: str, metric: str, p_value: float) -> None:
        """
        Update statistical significance data for effectiveness metrics.
        
        Args:
            tool_id: ID of the tool
            metric: Name of the effectiveness metric
            p_value: Statistical significance (p-value)
        """
        STATISTICAL_SIGNIFICANCE.labels(tool_id=tool_id, metric=metric).set(p_value)
    
    def record_memory_usage(self, analysis_type: str, memory_mb: float) -> None:
        """
        Record memory usage during analysis.
        
        Args:
            analysis_type: Type of analysis performed
            memory_mb: Memory usage in megabytes
        """
        ANALYSIS_MEMORY_USAGE.labels(analysis_type=analysis_type).observe(memory_mb)
    
    def push_metrics(self) -> bool:
        """
        Push metrics to Prometheus gateway if the push interval has elapsed.
        
        Returns:
            bool: True if metrics were pushed, False otherwise
        """
        now = datetime.now()
        if (now - self.last_push_time).total_seconds() < PUSH_INTERVAL_SECONDS:
            return False
        
        try:
            # Setup auth handler if credentials are provided
            auth_handler = None
            if AUTH_USERNAME and AUTH_PASSWORD:
                def auth_handler(url, method, timeout, headers, data):
                    return basic_auth_handler(url, method, timeout, headers, data, 
                                             AUTH_USERNAME, AUTH_PASSWORD)
            
            # Push metrics
            push_to_gateway(
                PROMETHEUS_GATEWAY, 
                job='enhanced_effectiveness_framework',
                handler=auth_handler
            )
            
            self.last_push_time = now
            logger.info(f"Successfully pushed enhanced effectiveness metrics to gateway")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")
            return False
    
    def periodic_metrics_update(self) -> None:
        """Run periodic updates for metrics that need refreshing."""
        # Push metrics
        self.push_metrics()


# Singleton instance
metrics_exporter = EnhancedEffectivenessMetricsExporter()
