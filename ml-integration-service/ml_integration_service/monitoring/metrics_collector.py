"""
Monitoring and Metrics Collection for Enhanced ML Integration Components

This module provides monitoring and metrics collection capabilities for the
enhanced visualization, optimization, and stress testing components.
"""
from typing import Dict, Any, Optional
import time
from datetime import datetime
import logging
from prometheus_client import Counter, Gauge, Histogram, Summary

# Initialize metrics
VISUALIZATION_REQUESTS = Counter(
    'ml_integration_visualization_requests_total',
    'Total number of visualization requests',
    ['visualization_type']
)

VISUALIZATION_LATENCY = Histogram(
    'ml_integration_visualization_latency_seconds',
    'Latency of visualization generation',
    ['visualization_type']
)

OPTIMIZATION_RUNS = Counter(
    'ml_integration_optimization_runs_total',
    'Total number of optimization runs',
    ['optimization_type']
)

OPTIMIZATION_DURATION = Histogram(
    'ml_integration_optimization_duration_seconds',
    'Duration of optimization runs',
    ['optimization_type']
)

OPTIMIZATION_IMPROVEMENT = Gauge(
    'ml_integration_optimization_improvement_ratio',
    'Improvement ratio achieved by optimization',
    ['optimization_type']
)

STRESS_TEST_RUNS = Counter(
    'ml_integration_stress_test_runs_total',
    'Total number of stress test runs',
    ['test_type']
)

STRESS_TEST_FAILURES = Counter(
    'ml_integration_stress_test_failures_total',
    'Number of stress test failures',
    ['test_type', 'failure_reason']
)

MODEL_ROBUSTNESS_SCORE = Gauge(
    'ml_integration_model_robustness_score',
    'Model robustness score from stress testing',
    ['model_id']
)

PERFORMANCE_METRICS = Gauge(
    'ml_integration_performance_metrics',
    'Various performance metrics',
    ['metric_name', 'model_id']
)

API_REQUEST_DURATION = Histogram(
    'ml_integration_api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method']
)

class MetricsCollector:
    """Collects and records metrics for ML Integration Service components."""

    @staticmethod
    def record_visualization_request(visualization_type: str) -> None:
        """Record a visualization request."""
        VISUALIZATION_REQUESTS.labels(visualization_type=visualization_type).inc()

    @staticmethod
    def time_visualization_generation(visualization_type: str) -> None:
        """Return a context manager for timing visualization generation."""
        return VISUALIZATION_LATENCY.labels(visualization_type=visualization_type).time()

    @staticmethod
    def record_optimization_run(
        optimization_type: str,
        duration: float,
        improvement: float
    ) -> None:
        """Record metrics for an optimization run."""
        OPTIMIZATION_RUNS.labels(optimization_type=optimization_type).inc()
        OPTIMIZATION_DURATION.labels(optimization_type=optimization_type).observe(duration)
        OPTIMIZATION_IMPROVEMENT.labels(optimization_type=optimization_type).set(improvement)

    @staticmethod
    def record_stress_test(
        test_type: str,
        success: bool,
        failure_reason: Optional[str] = None
    ) -> None:
        """Record stress test execution metrics."""
        STRESS_TEST_RUNS.labels(test_type=test_type).inc()
        if not success:
            STRESS_TEST_FAILURES.labels(
                test_type=test_type,
                failure_reason=failure_reason or 'unknown'
            ).inc()

    @staticmethod
    def update_robustness_score(model_id: str, score: float) -> None:
        """Update the robustness score for a model."""
        MODEL_ROBUSTNESS_SCORE.labels(model_id=model_id).set(score)

    @staticmethod
    def record_performance_metric(
        metric_name: str,
        value: float,
        model_id: str
    ) -> None:
        """Record a performance metric."""
        PERFORMANCE_METRICS.labels(
            metric_name=metric_name,
            model_id=model_id
        ).set(value)

    @staticmethod
    def time_api_request(endpoint: str, method: str) -> None:
        """Return a context manager for timing API requests."""
        return API_REQUEST_DURATION.labels(
            endpoint=endpoint,
            method=method
        ).time()
