"""
Simple implementation of BaseMetricsCollector that logs metrics to the console.
"""
import logging
from typing import Dict, Optional

from .base_collector import BaseMetricsCollector

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - METRICS - %(message)s')

class LoggingMetricsCollector(BaseMetricsCollector):
    """
    A metrics collector that logs metric increments to the standard logger.
    """

    def increment(self, metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """
        Logs the increment of a counter metric.

        Args:
            metric_name: The name of the metric.
            value: The increment value.
            labels: Associated labels.
        """
        label_str = "" if not labels else "{" + ", ".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
        logging.info(f"COUNTER {metric_name}{label_str} +{value}")

    # Implement other methods if added to BaseMetricsCollector
    # def gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    #     label_str = "" if not labels else "{" + ", ".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    #     logging.info(f"GAUGE {metric_name}{label_str} = {value}")
    #
    # def histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    #     label_str = "" if not labels else "{" + ", ".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    #     logging.info(f"HISTOGRAM {metric_name}{label_str} observed {value}")
