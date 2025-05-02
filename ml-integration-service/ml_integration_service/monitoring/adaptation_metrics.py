"""
Enhanced Model Adaptation Metrics Collection System.
"""
from datetime import datetime
from typing import Dict, Optional
import prometheus_client as prom
from prometheus_client import Counter, Gauge, Histogram

# Adaptation success/failure metrics
ADAPTATION_ATTEMPTS = Counter(
    'model_adaptation_attempts_total',
    'Total number of model adaptation attempts',
    ['model_id', 'model_type']
)

ADAPTATION_SUCCESS = Counter(
    'model_adaptation_success_total',
    'Total number of successful model adaptations',
    ['model_id', 'model_type']
)

ADAPTATION_FAILURES = Counter(
    'model_adaptation_failures_total',
    'Total number of failed model adaptations',
    ['model_id', 'model_type', 'failure_reason']
)

# Performance impact metrics
PERFORMANCE_SCORE = Gauge(
    'model_performance_score',
    'Current model performance score',
    ['model_id', 'metric_type']
)

PERFORMANCE_DELTA = Gauge(
    'model_performance_delta',
    'Change in model performance after adaptation',
    ['model_id', 'metric_type']
)

# Timing metrics
ADAPTATION_DURATION = Histogram(
    'model_adaptation_duration_seconds',
    'Time taken for model adaptation',
    ['model_id', 'adaptation_type'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# Stability metrics
ROLLBACK_RATE = Gauge(
    'model_adaptation_rollback_rate',
    'Rate of adaptation rollbacks',
    ['model_id', 'model_type']
)

# Trigger metrics
ADAPTATION_TRIGGERS = Counter(
    'model_adaptation_triggers_total',
    'Count of different adaptation triggers',
    ['model_id', 'trigger_type']
)

class AdaptationMetricsCollector:
    """Collects and exports model adaptation metrics."""

    def record_adaptation_attempt(
        self,
        model_id: str,
        model_type: str,
        adaptation_type: str,
        start_time: datetime
    ) -> None:
        """Record a model adaptation attempt."""
        ADAPTATION_ATTEMPTS.labels(
            model_id=model_id,
            model_type=model_type
        ).inc()

    def record_adaptation_result(
        self,
        model_id: str,
        model_type: str,
        success: bool,
        duration: float,
        failure_reason: Optional[str] = None
    ) -> None:
        """Record the result of a model adaptation."""
        if success:
            ADAPTATION_SUCCESS.labels(
                model_id=model_id,
                model_type=model_type
            ).inc()
        else:
            ADAPTATION_FAILURES.labels(
                model_id=model_id,
                model_type=model_type,
                failure_reason=failure_reason or 'unknown'
            ).inc()

        ADAPTATION_DURATION.labels(
            model_id=model_id,
            adaptation_type='standard'
        ).observe(duration)

    def update_performance_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float],
        is_post_adaptation: bool = False
    ) -> None:
        """Update model performance metrics."""
        for metric_type, value in metrics.items():
            PERFORMANCE_SCORE.labels(
                model_id=model_id,
                metric_type=metric_type
            ).set(value)

        if is_post_adaptation:
            for metric_type, value in metrics.items():
                current = PERFORMANCE_SCORE.labels(
                    model_id=model_id,
                    metric_type=metric_type
                )._value.get()
                if current is not None:
                    delta = value - current
                    PERFORMANCE_DELTA.labels(
                        model_id=model_id,
                        metric_type=metric_type
                    ).set(delta)

    def record_adaptation_trigger(
        self,
        model_id: str,
        trigger_type: str
    ) -> None:
        """Record what triggered an adaptation."""
        ADAPTATION_TRIGGERS.labels(
            model_id=model_id,
            trigger_type=trigger_type
        ).inc()

    def record_rollback(
        self,
        model_id: str,
        model_type: str
    ) -> None:
        """Record a model adaptation rollback."""
        current_rate = ROLLBACK_RATE.labels(
            model_id=model_id,
            model_type=model_type
        )._value.get()
        
        # Increment rollback rate with decay
        new_rate = (current_rate or 0) * 0.9 + 0.1
        ROLLBACK_RATE.labels(
            model_id=model_id,
            model_type=model_type
        ).set(new_rate)
