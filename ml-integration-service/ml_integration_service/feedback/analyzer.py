"""
Feedback Analysis System.
Implements continuous learning and adaptation mechanisms for the trading platform.
"""
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel


class PerformanceMetric(BaseModel):
    metric_name: str
    value: float
    timestamp: datetime
    confidence: float


class ModelPerformance(BaseModel):
    model_id: str
    version: str
    metrics: List[PerformanceMetric]
    sample_size: int
    evaluation_period: str


class FeedbackAnalyzer:
    def __init__(self, min_sample_size: int = 1000):
        self.min_sample_size = min_sample_size
        self._performance_history: Dict[str, List[ModelPerformance]] = {}

    def add_performance_data(self, performance: ModelPerformance) -> None:
        """Add new performance data for analysis."""
        if performance.model_id not in self._performance_history:
            self._performance_history[performance.model_id] = []
        self._performance_history[performance.model_id].append(performance)

    def detect_performance_drift(
        self, 
        model_id: str,
        metric_name: str,
        threshold: float = 0.1
    ) -> Optional[float]:
        """
        Detect if model performance has drifted significantly.
        Returns the drift magnitude if detected, None otherwise.
        """
        if model_id not in self._performance_history:
            return None

        history = self._performance_history[model_id]
        if len(history) < 2:
            return None

        recent = history[-1]
        previous = history[-2]

        recent_metric = next(
            (m for m in recent.metrics if m.metric_name == metric_name),
            None
        )
        previous_metric = next(
            (m for m in previous.metrics if m.metric_name == metric_name),
            None
        )

        if not recent_metric or not previous_metric:
            return None

        drift = abs(recent_metric.value - previous_metric.value)
        return drift if drift > threshold else None

    def analyze_adaptation_needs(
        self,
        model_id: str,
        min_confidence: float = 0.95
    ) -> bool:
        """
        Analyze if a model needs adaptation based on performance history.
        Returns True if adaptation is recommended.
        """
        if model_id not in self._performance_history:
            return False

        history = self._performance_history[model_id]
        if not history:
            return False

        recent = history[-1]
        
        # Check if sample size is sufficient
        if recent.sample_size < self.min_sample_size:
            return False

        # Check if any metric has low confidence
        low_confidence = any(
            metric.confidence < min_confidence
            for metric in recent.metrics
        )
        
        return low_confidence

    def get_performance_trend(
        self,
        model_id: str,
        metric_name: str,
        window_size: int = 5
    ) -> Optional[float]:
        """
        Calculate the trend of a specific performance metric.
        Returns the slope of the trend line if enough data points exist.
        """
        if model_id not in self._performance_history:
            return None

        history = self._performance_history[model_id]
        if len(history) < window_size:
            return None

        recent_history = history[-window_size:]
        
        values = []
        for perf in recent_history:
            metric = next(
                (m for m in perf.metrics if m.metric_name == metric_name),
                None
            )
            if metric:
                values.append(metric.value)

        if len(values) < window_size:
            return None

        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return slope
