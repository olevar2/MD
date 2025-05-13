"""
Notifications Package

This package provides notification functionality for alerts.
"""

from services.notification_service import (
    NotificationService,
    initialize_notification_service,
    send_notification
)

__all__ = [
    "NotificationService",
    "initialize_notification_service",
    "send_notification"
]
