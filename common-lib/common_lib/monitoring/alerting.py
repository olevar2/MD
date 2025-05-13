"""
Alerting Module

This module provides alerting functionality for the platform.
"""

import logging
import json
import time
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, ClassVar, Awaitable, Union

import aiohttp


class AlertSeverity(Enum):
    """
    Alert severity.
    """
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """
    Alert channel.
    """
    
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


class Alert:
    """
    Alert.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        severity: AlertSeverity,
        service: str,
        timestamp: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize the alert.
        
        Args:
            name: Name of the alert
            description: Description of the alert
            severity: Severity of the alert
            service: Service that triggered the alert
            timestamp: Timestamp of the alert (if None, uses current time)
            details: Additional details for the alert
            tags: Tags for the alert
        """
        self.name = name
        self.description = description
        self.severity = severity
        self.service = service
        self.timestamp = timestamp or time.time()
        self.details = details or {}
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the alert to a dictionary.
        
        Returns:
            Dictionary representation of the alert
        """
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "service": self.service,
            "timestamp": self.timestamp,
            "details": self.details,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """
        Create an alert from a dictionary.
        
        Args:
            data: Dictionary representation of the alert
            
        Returns:
            Alert
        """
        return cls(
            name=data["name"],
            description=data["description"],
            severity=AlertSeverity(data["severity"]),
            service=data["service"],
            timestamp=data.get("timestamp"),
            details=data.get("details"),
            tags=data.get("tags")
        )


class AlertManager:
    """
    Alert manager for the platform.
    
    This class provides a singleton manager for alerts.
    """
    
    _instance: ClassVar[Optional["AlertManager"]] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the alert manager.
        
        Returns:
            Singleton instance of the alert manager
        """
        if cls._instance is None:
            cls._instance = super(AlertManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        service_name: str,
        channels: Optional[Dict[AlertChannel, Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the alert manager.
        
        Args:
            service_name: Name of the service
            channels: Alert channels configuration
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.service_name = service_name
        self.channels = channels or {}
        
        self._initialized = True
    
    async def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[AlertChannel]] = None
    ) -> None:
        """
        Send an alert.
        
        Args:
            alert: Alert to send
            channels: Channels to send the alert to (if None, sends to all configured channels)
        """
        # Set channels to send to
        if channels is None:
            channels = list(self.channels.keys())
        
        # Send alert to each channel
        for channel in channels:
            if channel not in self.channels:
                self.logger.warning(f"Alert channel not configured: {channel.value}")
                continue
            
            try:
                # Send alert to channel
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
                elif channel == AlertChannel.SMS:
                    await self._send_sms_alert(alert)
                elif channel == AlertChannel.PAGERDUTY:
                    await self._send_pagerduty_alert(alert)
                else:
                    self.logger.warning(f"Unsupported alert channel: {channel.value}")
            except Exception as e:
                self.logger.error(f"Error sending alert to {channel.value}: {str(e)}")
    
    async def _send_email_alert(self, alert: Alert) -> None:
        """
        Send an alert via email.
        
        Args:
            alert: Alert to send
        """
        # Get email configuration
        config = self.channels.get(AlertChannel.EMAIL)
        if not config:
            self.logger.warning("Email alert channel not configured")
            return
        
        # TODO: Implement email alert
        self.logger.info(f"Sending email alert: {alert.name}")
    
    async def _send_slack_alert(self, alert: Alert) -> None:
        """
        Send an alert via Slack.
        
        Args:
            alert: Alert to send
        """
        # Get Slack configuration
        config = self.channels.get(AlertChannel.SLACK)
        if not config:
            self.logger.warning("Slack alert channel not configured")
            return
        
        # Get webhook URL
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            self.logger.warning("Slack webhook URL not configured")
            return
        
        # Create message
        message = {
            "text": f"*{alert.severity.value.upper()}*: {alert.name}",
            "attachments": [
                {
                    "color": self._get_color_for_severity(alert.severity),
                    "title": alert.name,
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Service",
                            "value": alert.service,
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value,
                            "short": True
                        }
                    ],
                    "footer": f"Alert triggered at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}"
                }
            ]
        }
        
        # Add details
        if alert.details:
            message["attachments"][0]["fields"].append({
                "title": "Details",
                "value": json.dumps(alert.details, indent=2),
                "short": False
            })
        
        # Add tags
        if alert.tags:
            message["attachments"][0]["fields"].append({
                "title": "Tags",
                "value": ", ".join(alert.tags),
                "short": False
            })
        
        # Send message
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status != 200:
                    self.logger.error(f"Error sending Slack alert: {await response.text()}")
                else:
                    self.logger.info(f"Sent Slack alert: {alert.name}")
    
    async def _send_webhook_alert(self, alert: Alert) -> None:
        """
        Send an alert via webhook.
        
        Args:
            alert: Alert to send
        """
        # Get webhook configuration
        config = self.channels.get(AlertChannel.WEBHOOK)
        if not config:
            self.logger.warning("Webhook alert channel not configured")
            return
        
        # Get webhook URL
        webhook_url = config.get("url")
        if not webhook_url:
            self.logger.warning("Webhook URL not configured")
            return
        
        # Create payload
        payload = alert.to_dict()
        
        # Send payload
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status not in [200, 201, 202, 204]:
                    self.logger.error(f"Error sending webhook alert: {await response.text()}")
                else:
                    self.logger.info(f"Sent webhook alert: {alert.name}")
    
    async def _send_sms_alert(self, alert: Alert) -> None:
        """
        Send an alert via SMS.
        
        Args:
            alert: Alert to send
        """
        # Get SMS configuration
        config = self.channels.get(AlertChannel.SMS)
        if not config:
            self.logger.warning("SMS alert channel not configured")
            return
        
        # TODO: Implement SMS alert
        self.logger.info(f"Sending SMS alert: {alert.name}")
    
    async def _send_pagerduty_alert(self, alert: Alert) -> None:
        """
        Send an alert via PagerDuty.
        
        Args:
            alert: Alert to send
        """
        # Get PagerDuty configuration
        config = self.channels.get(AlertChannel.PAGERDUTY)
        if not config:
            self.logger.warning("PagerDuty alert channel not configured")
            return
        
        # Get integration key
        integration_key = config.get("integration_key")
        if not integration_key:
            self.logger.warning("PagerDuty integration key not configured")
            return
        
        # Create payload
        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.name,
                "source": alert.service,
                "severity": self._map_severity_to_pagerduty(alert.severity),
                "custom_details": {
                    "description": alert.description,
                    "details": alert.details,
                    "tags": alert.tags
                }
            }
        }
        
        # Send payload
        async with aiohttp.ClientSession() as session:
            async with session.post("https://events.pagerduty.com/v2/enqueue", json=payload) as response:
                if response.status != 202:
                    self.logger.error(f"Error sending PagerDuty alert: {await response.text()}")
                else:
                    self.logger.info(f"Sent PagerDuty alert: {alert.name}")
    
    def _get_color_for_severity(self, severity: AlertSeverity) -> str:
        """
        Get color for severity.
        
        Args:
            severity: Severity
            
        Returns:
            Color
        """
        if severity == AlertSeverity.INFO:
            return "#2196F3"  # Blue
        elif severity == AlertSeverity.WARNING:
            return "#FFC107"  # Yellow
        elif severity == AlertSeverity.ERROR:
            return "#F44336"  # Red
        elif severity == AlertSeverity.CRITICAL:
            return "#9C27B0"  # Purple
        else:
            return "#9E9E9E"  # Grey
    
    def _map_severity_to_pagerduty(self, severity: AlertSeverity) -> str:
        """
        Map severity to PagerDuty severity.
        
        Args:
            severity: Severity
            
        Returns:
            PagerDuty severity
        """
        if severity == AlertSeverity.INFO:
            return "info"
        elif severity == AlertSeverity.WARNING:
            return "warning"
        elif severity == AlertSeverity.ERROR:
            return "error"
        elif severity == AlertSeverity.CRITICAL:
            return "critical"
        else:
            return "info"