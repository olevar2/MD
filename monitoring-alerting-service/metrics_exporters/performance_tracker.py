"""
Performance Tracking System.
Implements comprehensive performance monitoring and metrics tracking.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel
import numpy as np
from prometheus_client import Gauge, Histogram

# Performance Metrics
MODEL_PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["model_id", "version"]
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model accuracy metric",
    ["model_id", "version", "metric_type"]
)

MODEL_DRIFT = Gauge(
    "model_drift",
    "Model drift metric",
    ["model_id", "version", "feature"]
)


class MetricTimeWindow(BaseModel):
    start_time: datetime
    end_time: datetime
    sample_count: int
    mean: float
    std_dev: float
    min_value: float
    max_value: float


class PerformanceTracker:
    def __init__(self, window_size: timedelta = timedelta(hours=1)):
        self.window_size = window_size
        self._metrics: Dict[str, List[float]] = {}
        self._timestamps: Dict[str, List[datetime]] = {}

    def record_metric(
        self,
        metric_name: str,
        value: float,
        model_id: str,
        version: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a new metric value."""
        key = f"{model_id}:{version}:{metric_name}"
        if key not in self._metrics:
            self._metrics[key] = []
            self._timestamps[key] = []

        current_time = timestamp or datetime.utcnow()
        self._metrics[key].append(value)
        self._timestamps[key].append(current_time)

        # Update Prometheus metrics
        if "latency" in metric_name.lower():
            MODEL_PREDICTION_LATENCY.labels(
                model_id=model_id,
                version=version
            ).observe(value)
        elif "accuracy" in metric_name.lower():
            MODEL_ACCURACY.labels(
                model_id=model_id,
                version=version,
                metric_type=metric_name
            ).set(value)
        elif "drift" in metric_name.lower():
            MODEL_DRIFT.labels(
                model_id=model_id,
                version=version,
                feature=metric_name
            ).set(value)

        # Clean up old data
        self._cleanup_old_data(key, current_time)

    def get_metric_window(
        self,
        metric_name: str,
        model_id: str,
        version: str,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None
    ) -> Optional[MetricTimeWindow]:
        """Get metric statistics for a specific time window."""
        key = f"{model_id}:{version}:{metric_name}"
        if key not in self._metrics:
            return None

        end_time = window_end or datetime.utcnow()
        start_time = window_start or (end_time - self.window_size)

        # Filter data points within the window
        indices = [
            i for i, ts in enumerate(self._timestamps[key])
            if start_time <= ts <= end_time
        ]

        if not indices:
            return None

        values = [self._metrics[key][i] for i in indices]
        return MetricTimeWindow(
            start_time=start_time,
            end_time=end_time,
            sample_count=len(values),
            mean=float(np.mean(values)),
            std_dev=float(np.std(values)),
            min_value=float(np.min(values)),
            max_value=float(np.max(values))
        )

    def _cleanup_old_data(self, key: str, current_time: datetime) -> None:
        """Remove data points outside the window."""
        cutoff_time = current_time - self.window_size
        while (self._timestamps[key] and 
               self._timestamps[key][0] < cutoff_time):
            self._timestamps[key].pop(0)
            self._metrics[key].pop(0)
