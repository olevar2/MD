"""
Defines the abstract base class for metrics collectors.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseMetricsCollector(ABC):
    """
    Abstract base class for metrics collection.

    Concrete implementations should provide mechanisms to record metrics,
    e.g., logging, sending to Prometheus, etc.
    """

    @abstractmethod
    def increment(self, metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increments a counter metric.

        Args:
            metric_name: The name of the metric to increment.
            value: The amount to increment by (default is 1).
            labels: Optional dictionary of labels (key-value pairs) to associate with the metric.
        """
        pass

    # Add other metric types as needed (e.g., gauge, histogram)
    # @abstractmethod
    # def gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """
    Gauge.
    
    Args:
        metric_name: Description of metric_name
        value: Description of value
        labels: Description of labels
        str]]: Description of str]]
    
    """

    #     pass
    #
    # @abstractmethod
    # def histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """
    Histogram.
    
    Args:
        metric_name: Description of metric_name
        value: Description of value
        labels: Description of labels
        str]]: Description of str]]
    
    """

    #     pass
