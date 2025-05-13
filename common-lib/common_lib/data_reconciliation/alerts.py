#!/usr/bin/env python3
"""
Alerts for data reconciliation.
"""

import logging
import datetime
from typing import Dict, List, Any, Optional, Union

from .reconciliation_engine import ReconciliationResult

logger = logging.getLogger(__name__)

class ReconciliationAlert:
    """Alert for reconciliation issues."""
    
    def __init__(
        self,
        job_name: str,
        result: ReconciliationResult,
        alert_type: str,
        message: str,
        severity: str = "warning",
        timestamp: Optional[datetime.datetime] = None
    ):
        """Initialize a reconciliation alert.
        
        Args:
            job_name: Name of the job
            result: Reconciliation result
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            timestamp: Alert timestamp
        """
        self.job_name = job_name
        self.result = result
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.timestamp = timestamp or datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary."""
        return {
            "job_name": self.job_name,
            "entity_type": self.result.entity_type,
            "source_system": self.result.source_system,
            "target_system": self.result.target_system,
            "alert_type": self.alert_type,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "result_summary": {
                "matched_count": self.result.matched_count,
                "missing_in_target_count": self.result.missing_in_target_count,
                "missing_in_source_count": self.result.missing_in_source_count,
                "mismatched_count": self.result.mismatched_count,
                "success": self.result.success
            }
        }


class AlertManager:
    """Manager for reconciliation alerts."""
    
    def __init__(self, alert_handlers: Optional[List[callable]] = None):
        """Initialize the alert manager.
        
        Args:
            alert_handlers: List of functions to handle alerts
        """
        self.alert_handlers = alert_handlers or []
    
    def add_handler(self, handler: callable) -> None:
        """Add an alert handler.
        
        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)
    
    def send_alert(self, alert: ReconciliationAlert) -> None:
        """Send an alert.
        
        Args:
            alert: Reconciliation alert
        """
        # Log the alert
        log_level = self._severity_to_log_level(alert.severity)
        logger.log(log_level, f"Reconciliation alert: {alert.message}")
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
    
    def create_and_send_alert(
        self,
        job_name: str,
        result: ReconciliationResult,
        alert_type: str,
        message: str,
        severity: str = "warning"
    ) -> ReconciliationAlert:
        """Create and send an alert.
        
        Args:
            job_name: Name of the job
            result: Reconciliation result
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            
        Returns:
            Created alert
        """
        alert = ReconciliationAlert(
            job_name=job_name,
            result=result,
            alert_type=alert_type,
            message=message,
            severity=severity
        )
        
        self.send_alert(alert)
        
        return alert
    
    @staticmethod
    def _severity_to_log_level(severity: str) -> int:
        """Convert severity to log level.
        
        Args:
            severity: Alert severity
            
        Returns:
            Log level
        """
        severity_map = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        
        return severity_map.get(severity.lower(), logging.WARNING)


# Alert handlers

def log_alert_handler(alert: ReconciliationAlert) -> None:
    """Log an alert.
    
    Args:
        alert: Reconciliation alert
    """
    log_level = AlertManager._severity_to_log_level(alert.severity)
    logger.log(log_level, f"Reconciliation alert: {alert.message}")

def prometheus_alert_handler(alert: ReconciliationAlert) -> None:
    """Send an alert to Prometheus.
    
    Args:
        alert: Reconciliation alert
    """
    # TODO: Implement Prometheus alert handler
    pass

def email_alert_handler(alert: ReconciliationAlert) -> None:
    """Send an alert via email.
    
    Args:
        alert: Reconciliation alert
    """
    # TODO: Implement email alert handler
    pass

def slack_alert_handler(alert: ReconciliationAlert) -> None:
    """Send an alert to Slack.
    
    Args:
        alert: Reconciliation alert
    """
    # TODO: Implement Slack alert handler
    pass
