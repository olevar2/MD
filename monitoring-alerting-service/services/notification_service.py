"""
Notification Service Module

This module provides functionality for sending notifications for alerts.
"""
import logging
import json
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List, Union
import requests
from datetime import datetime

from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    NotificationError
)

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Notification service for sending alerts.
    
    This class provides methods for sending notifications through various
    channels such as email, Slack, and Microsoft Teams.
    """
    
    def __init__(self, service_name: str):
        """
        Initialize the notification service.
        
        Args:
            service_name: The name of the service
        """
        self.service_name = service_name
        
        # Load configuration from environment variables
        self.email_enabled = os.environ.get("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true"
        self.email_host = os.environ.get("EMAIL_HOST", "smtp.example.com")
        self.email_port = int(os.environ.get("EMAIL_PORT", "587"))
        self.email_username = os.environ.get("EMAIL_USERNAME", "")
        self.email_password = os.environ.get("EMAIL_PASSWORD", "")
        self.email_from = os.environ.get("EMAIL_FROM", "alerts@forex-platform.local")
        self.email_to = os.environ.get("EMAIL_TO", "").split(",")
        
        self.slack_enabled = os.environ.get("SLACK_NOTIFICATIONS_ENABLED", "false").lower() == "true"
        self.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        
        self.teams_enabled = os.environ.get("TEAMS_NOTIFICATIONS_ENABLED", "false").lower() == "true"
        self.teams_webhook_url = os.environ.get("TEAMS_WEBHOOK_URL", "")
        
        logger.info(f"Initialized NotificationService for {service_name}")
    
    @with_exception_handling
    def send_email_notification(
        self,
        subject: str,
        message: str,
        recipients: Optional[List[str]] = None,
        html_message: Optional[str] = None
    ) -> bool:
        """
        Send an email notification.
        
        Args:
            subject: The email subject
            message: The email message (plain text)
            recipients: List of email recipients (defaults to configured recipients)
            html_message: HTML version of the message (optional)
            
        Returns:
            True if the email was sent successfully, False otherwise
            
        Raises:
            NotificationError: If the email could not be sent
        """
        if not self.email_enabled:
            logger.warning("Email notifications are disabled")
            return False
        
        if not recipients:
            recipients = self.email_to
        
        if not recipients:
            logger.warning("No email recipients configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{self.service_name}] {subject}"
            msg["From"] = self.email_from
            msg["To"] = ", ".join(recipients)
            
            # Add plain text version
            msg.attach(MIMEText(message, "plain"))
            
            # Add HTML version if provided
            if html_message:
                msg.attach(MIMEText(html_message, "html"))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.email_host, self.email_port) as server:
                server.starttls()
                if self.email_username and self.email_password:
                    server.login(self.email_username, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Sent email notification: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}", exc_info=True)
            raise NotificationError(
                message=f"Failed to send email notification: {str(e)}",
                channel="email",
                details={
                    "subject": subject,
                    "recipients": recipients,
                    "error": str(e)
                }
            )
    
    @with_exception_handling
    def send_slack_notification(
        self,
        title: str,
        message: str,
        color: str = "#ff0000",
        fields: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """
        Send a Slack notification.
        
        Args:
            title: The notification title
            message: The notification message
            color: The color of the Slack attachment
            fields: Additional fields for the Slack attachment
            
        Returns:
            True if the notification was sent successfully, False otherwise
            
        Raises:
            NotificationError: If the notification could not be sent
        """
        if not self.slack_enabled:
            logger.warning("Slack notifications are disabled")
            return False
        
        if not self.slack_webhook_url:
            logger.warning("No Slack webhook URL configured")
            return False
        
        try:
            # Create Slack message payload
            payload = {
                "attachments": [
                    {
                        "fallback": f"{title}: {message}",
                        "color": color,
                        "title": title,
                        "text": message,
                        "fields": fields or [],
                        "footer": f"{self.service_name} | {datetime.now().isoformat()}"
                    }
                ]
            }
            
            # Send to Slack webhook
            response = requests.post(
                self.slack_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise NotificationError(
                    message=f"Failed to send Slack notification: {response.text}",
                    channel="slack",
                    details={
                        "status_code": response.status_code,
                        "response": response.text
                    }
                )
            
            logger.info(f"Sent Slack notification: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}", exc_info=True)
            raise NotificationError(
                message=f"Failed to send Slack notification: {str(e)}",
                channel="slack",
                details={
                    "title": title,
                    "error": str(e)
                }
            )
    
    @with_exception_handling
    def send_teams_notification(
        self,
        title: str,
        message: str,
        color: str = "#ff0000",
        facts: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """
        Send a Microsoft Teams notification.
        
        Args:
            title: The notification title
            message: The notification message
            color: The color of the Teams card
            facts: Additional facts for the Teams card
            
        Returns:
            True if the notification was sent successfully, False otherwise
            
        Raises:
            NotificationError: If the notification could not be sent
        """
        if not self.teams_enabled:
            logger.warning("Microsoft Teams notifications are disabled")
            return False
        
        if not self.teams_webhook_url:
            logger.warning("No Microsoft Teams webhook URL configured")
            return False
        
        try:
            # Create Teams message payload
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color.lstrip("#"),
                "summary": title,
                "sections": [
                    {
                        "activityTitle": title,
                        "activitySubtitle": f"{self.service_name} | {datetime.now().isoformat()}",
                        "text": message,
                        "facts": facts or []
                    }
                ]
            }
            
            # Send to Teams webhook
            response = requests.post(
                self.teams_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise NotificationError(
                    message=f"Failed to send Microsoft Teams notification: {response.text}",
                    channel="teams",
                    details={
                        "status_code": response.status_code,
                        "response": response.text
                    }
                )
            
            logger.info(f"Sent Microsoft Teams notification: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Microsoft Teams notification: {str(e)}", exc_info=True)
            raise NotificationError(
                message=f"Failed to send Microsoft Teams notification: {str(e)}",
                channel="teams",
                details={
                    "title": title,
                    "error": str(e)
                }
            )
    
    @with_exception_handling
    def send_notification(
        self,
        title: str,
        message: str,
        severity: str = "critical",
        details: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Send a notification through all enabled channels.
        
        Args:
            title: The notification title
            message: The notification message
            severity: The severity of the notification (info, warning, critical)
            details: Additional details for the notification
            channels: List of channels to use (defaults to all enabled channels)
            
        Returns:
            Dictionary of channel names and success status
            
        Raises:
            NotificationError: If the notification could not be sent
        """
        results = {}
        details = details or {}
        
        # Determine color based on severity
        color_map = {
            "info": "#2196f3",  # Blue
            "warning": "#ff9800",  # Orange
            "critical": "#f44336"  # Red
        }
        color = color_map.get(severity.lower(), "#f44336")
        
        # Determine channels to use
        if not channels:
            channels = []
            if self.email_enabled:
                channels.append("email")
            if self.slack_enabled:
                channels.append("slack")
            if self.teams_enabled:
                channels.append("teams")
        
        # Send notifications through each channel
        for channel in channels:
            try:
                if channel == "email":
                    # Create HTML message with details
                    html_message = f"<h2>{title}</h2><p>{message}</p>"
                    if details:
                        html_message += "<h3>Details</h3><ul>"
                        for key, value in details.items():
                            html_message += f"<li><strong>{key}:</strong> {value}</li>"
                        html_message += "</ul>"
                    
                    results["email"] = self.send_email_notification(
                        subject=title,
                        message=f"{message}\n\nDetails:\n" + "\n".join([f"{k}: {v}" for k, v in details.items()]),
                        html_message=html_message
                    )
                
                elif channel == "slack":
                    # Create Slack fields from details
                    fields = []
                    for key, value in details.items():
                        fields.append({
                            "title": key,
                            "value": str(value),
                            "short": len(str(value)) < 20
                        })
                    
                    results["slack"] = self.send_slack_notification(
                        title=title,
                        message=message,
                        color=color,
                        fields=fields
                    )
                
                elif channel == "teams":
                    # Create Teams facts from details
                    facts = []
                    for key, value in details.items():
                        facts.append({
                            "name": key,
                            "value": str(value)
                        })
                    
                    results["teams"] = self.send_teams_notification(
                        title=title,
                        message=message,
                        color=color,
                        facts=facts
                    )
                
                else:
                    logger.warning(f"Unknown notification channel: {channel}")
                    results[channel] = False
            
            except Exception as e:
                logger.error(f"Failed to send notification through {channel}: {str(e)}", exc_info=True)
                results[channel] = False
        
        return results


# Global instance for the monitoring-alerting-service
_notification_service = None


def initialize_notification_service(service_name: str) -> NotificationService:
    """
    Initialize the notification service.
    
    Args:
        service_name: The name of the service
        
    Returns:
        The notification service instance
    """
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService(service_name)
    return _notification_service


def send_notification(
    title: str,
    message: str,
    severity: str = "critical",
    details: Optional[Dict[str, Any]] = None,
    channels: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Send a notification through all enabled channels.
    
    Args:
        title: The notification title
        message: The notification message
        severity: The severity of the notification (info, warning, critical)
        details: Additional details for the notification
        channels: List of channels to use (defaults to all enabled channels)
        
    Returns:
        Dictionary of channel names and success status
    """
    global _notification_service
    if _notification_service is None:
        _notification_service = initialize_notification_service("monitoring-alerting-service")
    
    return _notification_service.send_notification(
        title=title,
        message=message,
        severity=severity,
        details=details,
        channels=channels
    )
