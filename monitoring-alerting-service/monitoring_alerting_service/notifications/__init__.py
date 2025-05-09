"""
Notifications Package

This package provides notification functionality for alerts.
"""

from monitoring_alerting_service.notifications.notification_service import (
    NotificationService,
    initialize_notification_service,
    send_notification
)

__all__ = [
    "NotificationService",
    "initialize_notification_service",
    "send_notification"
]
