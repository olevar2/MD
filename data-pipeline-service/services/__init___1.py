"""
Mock monitoring module for testing.
"""

from typing import Dict, Any, Optional, List, Union, Callable


class MetricsCollector:
    """Mock metrics collector for testing."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {}
    
    def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record a counter metric."""
        self.metrics[name] = {"value": value, "tags": tags or {}, "type": "counter"}
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric."""
        self.metrics[name] = {"value": value, "tags": tags or {}, "type": "gauge"}
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        self.metrics[name] = {"value": value, "tags": tags or {}, "type": "histogram"}
    
    def start_timer(self, name: str, tags: Dict[str, str] = None):
        """Start a timer."""
        self.metrics[name] = {"start": 0, "tags": tags or {}, "type": "timer"}
        return lambda: None  # Return a no-op stop function
    
    def record_timer(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a timer metric."""
        self.metrics[name] = {"value": value, "tags": tags or {}, "type": "timer"}


class AlertManager:
    """Mock alert manager for testing."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.alerts = []
    
    def send_alert(self, alert_name: str, severity: str, message: str, details: Dict[str, Any] = None):
        """Send an alert."""
        self.alerts.append({
            "name": alert_name,
            "severity": severity,
            "message": message,
            "details": details or {}
        })
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
