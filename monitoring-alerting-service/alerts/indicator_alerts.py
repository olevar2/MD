"""
Monitoring & Alerting Service: Indicator Alert System Logic

This module provides an alert system based on indicator signals, including
prioritization, filtering, and dispatching to notification channels.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import uuid
import time
import threading
import queue
import os

from monitoring_alerting_service.error import (
    with_exception_handling,
    async_with_exception_handling,
    MonitoringAlertingError,
    AlertNotFoundError,
    NotificationError,
    AlertStorageError,
    AlertRuleError,
    ThresholdValidationError
)

# Configure logging
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Enum for alert severity levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5

    def __str__(self):
        """String representation of the alert level"""
        return self.name


class AlertType(Enum):
    """Enum for different types of alerts"""
    PRICE_LEVEL = auto()        # Price reached a specified level
    INDICATOR_SIGNAL = auto()   # Indicator generated a signal
    INDICATOR_EXTREME = auto()  # Indicator reached extreme value
    PATTERN_DETECTED = auto()   # Chart pattern detected
    DIVERGENCE = auto()         # Divergence between price and indicator
    VOLATILITY = auto()         # Volatility threshold breached
    CORRELATION = auto()        # Correlation threshold reached
    CONCORDANCE = auto()        # Indicator concordance threshold reached
    CUSTOM = auto()             # Custom alert type

    def __str__(self):
        """String representation of the alert type"""
        return self.name


class NotificationChannel(Enum):
    """Enum for notification channels"""
    EMAIL = auto()
    SMS = auto()
    PUSH = auto()
    SLACK = auto()
    WEBHOOK = auto()
    UI = auto()

    def __str__(self):
        """String representation of the notification channel"""
        return self.name


@dataclass
class Alert:
    """Class representing an alert"""
    id: str
    timestamp: datetime
    level: AlertLevel
    type: AlertType
    title: str
    message: str
    instrument: str
    timeframe: str
    source: str  # Indicator, pattern, or system generating the alert
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels: List[NotificationChannel] = field(default_factory=list)
    expiration: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    sent_notifications: Dict[str, datetime] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize optional fields after creation"""
        if not self.id:
            self.id = str(uuid.uuid4())

        if not self.channels:
            # Default to UI channel
            self.channels = [NotificationChannel.UI]

    @property
    def age(self) -> timedelta:
        """Get the age of the alert"""
        return datetime.now() - self.timestamp

    @property
    def is_expired(self) -> bool:
        """Check if the alert has expired"""
        if self.expiration is None:
            return False
        return datetime.now() > self.expiration

    @property
    def priority_score(self) -> float:
        """
        Calculate priority score for the alert.
        Lower score means higher priority.
        """
        # Base score from level
        base_score = self.level.value * 10

        # Age factor (older alerts get lower priority)
        age_hours = self.age.total_seconds() / 3600
        age_factor = min(10, age_hours / 2)  # Max 10 points for age

        # Acknowledgment factor (acknowledged alerts get lower priority)
        ack_factor = 50 if self.acknowledged else 0

        return base_score + age_factor + ack_factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        result = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": str(self.level),
            "type": str(self.type),
            "title": self.title,
            "message": self.message,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "source": self.source,
            "metadata": self.metadata,
            "channels": [str(c) for c in self.channels],
            "acknowledged": self.acknowledged,
            "priority_score": self.priority_score
        }

        if self.expiration:
            result["expiration"] = self.expiration.isoformat()

        if self.acknowledged_at:
            result["acknowledged_at"] = self.acknowledged_at.isoformat()

        if self.acknowledged_by:
            result["acknowledged_by"] = self.acknowledged_by

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary"""
        # Handle serialized enums
        level = AlertLevel[data["level"]] if isinstance(data["level"], str) else data["level"]
        alert_type = AlertType[data["type"]] if isinstance(data["type"], str) else data["type"]

        # Handle serialized channels
        channels = []
        for channel in data.get("channels", []):
            if isinstance(channel, str):
                channels.append(NotificationChannel[channel])
            else:
                channels.append(channel)

        # Handle serialized timestamps
        timestamp = (datetime.fromisoformat(data["timestamp"])
                   if isinstance(data["timestamp"], str) else data["timestamp"])

        expiration = None
        if "expiration" in data:
            expiration = (datetime.fromisoformat(data["expiration"])
                       if isinstance(data["expiration"], str) else data["expiration"])

        acknowledged_at = None
        if "acknowledged_at" in data:
            acknowledged_at = (datetime.fromisoformat(data["acknowledged_at"])
                             if isinstance(data["acknowledged_at"], str) else data["acknowledged_at"])

        # Create alert
        return cls(
            id=data["id"],
            timestamp=timestamp,
            level=level,
            type=alert_type,
            title=data["title"],
            message=data["message"],
            instrument=data["instrument"],
            timeframe=data["timeframe"],
            source=data["source"],
            metadata=data.get("metadata", {}),
            channels=channels,
            expiration=expiration,
            acknowledged=data.get("acknowledged", False),
            acknowledged_at=acknowledged_at,
            acknowledged_by=data.get("acknowledged_by"),
            sent_notifications=data.get("sent_notifications", {})
        )


class AlertFilter:
    """Filter for alerts"""

    def __init__(self):
        """Initialize the alert filter"""
        self.level_filters: Dict[AlertLevel, bool] = {level: True for level in AlertLevel}
        self.type_filters: Dict[AlertType, bool] = {atype: True for atype in AlertType}
        self.instrument_filters: Set[str] = set()
        self.timeframe_filters: Set[str] = set()
        self.source_filters: Set[str] = set()
        self.min_timestamp: Optional[datetime] = None
        self.max_timestamp: Optional[datetime] = None
        self.acknowledged_filter: Optional[bool] = None

    def filter_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """
        Apply filters to a list of alerts

        Args:
            alerts: List of alerts to filter

        Returns:
            Filtered list of alerts
        """
        result = []

        for alert in alerts:
            # Skip if filtered by level
            if not self.level_filters.get(alert.level, True):
                continue

            # Skip if filtered by type
            if not self.type_filters.get(alert.type, True):
                continue

            # Skip if filtered by instrument
            if self.instrument_filters and alert.instrument not in self.instrument_filters:
                continue

            # Skip if filtered by timeframe
            if self.timeframe_filters and alert.timeframe not in self.timeframe_filters:
                continue

            # Skip if filtered by source
            if self.source_filters and alert.source not in self.source_filters:
                continue

            # Skip if filtered by timestamp
            if self.min_timestamp and alert.timestamp < self.min_timestamp:
                continue

            if self.max_timestamp and alert.timestamp > self.max_timestamp:
                continue

            # Skip if filtered by acknowledgment status
            if self.acknowledged_filter is not None and alert.acknowledged != self.acknowledged_filter:
                continue

            # Alert passed all filters
            result.append(alert)

        return result


class AlertManager:
    """Manager for alerts"""

    def __init__(self, data_dir: str = "./data/alerts"):
        """
        Initialize the alert manager

        Args:
            data_dir: Directory to store alert data
        """
        self.data_dir = data_dir

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Store alerts
        self.alerts: Dict[str, Alert] = {}

        # Load existing alerts
        self._load_alerts()

        # Clean up expired alerts
        self._cleanup_expired_alerts()

    @with_exception_handling
    def add_alert(self, alert: Alert) -> str:
        """
        Add an alert

        Args:
            alert: The alert to add

        Returns:
            Alert ID

        Raises:
            AlertStorageError: If there's an error storing the alert
        """
        # Generate ID if not provided
        if not alert.id:
            alert.id = str(uuid.uuid4())

        # Add alert
        self.alerts[alert.id] = alert

        try:
            # Save alert to disk
            self._save_alert(alert)
        except Exception as e:
            raise AlertStorageError(
                message=f"Failed to save alert: {str(e)}",
                operation="add_alert",
                details={"alert_id": alert.id}
            )

        logger.debug(f"Added alert {alert.id}: {alert.title}")

        return alert.id

    @with_exception_handling
    def get_alert(self, alert_id: str) -> Alert:
        """
        Get an alert by ID

        Args:
            alert_id: ID of the alert

        Returns:
            The alert

        Raises:
            AlertNotFoundError: If the alert is not found
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            raise AlertNotFoundError(
                message=f"Alert with ID {alert_id} not found",
                alert_id=alert_id
            )
        return alert

    @with_exception_handling
    def get_alerts(self, filter: Optional[AlertFilter] = None,
                 sort_by_priority: bool = True) -> List[Alert]:
        """
        Get alerts with optional filtering and sorting

        Args:
            filter: Filter to apply
            sort_by_priority: Whether to sort by priority

        Returns:
            List of alerts

        Raises:
            MonitoringAlertingError: If there's an error retrieving or filtering alerts
        """
        try:
            # Get alerts
            all_alerts = list(self.alerts.values())

            # Apply filter if provided
            if filter:
                all_alerts = filter.filter_alerts(all_alerts)

            # Sort by priority if requested
            if sort_by_priority:
                all_alerts.sort(key=lambda a: a.priority_score)

            return all_alerts
        except Exception as e:
            raise MonitoringAlertingError(
                message=f"Failed to retrieve alerts: {str(e)}",
                error_code="ALERT_RETRIEVAL_ERROR",
                details={"error": str(e)}
            )

    @with_exception_handling
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: ID of the alert
            user: User acknowledging the alert

        Returns:
            True if the alert was acknowledged

        Raises:
            AlertNotFoundError: If the alert is not found
            AlertStorageError: If there's an error saving the alert
        """
        try:
            alert = self.get_alert(alert_id)

            # Acknowledge alert
            alert.acknowledged = True
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = user

            # Save alert
            self._save_alert(alert)

            return True
        except AlertNotFoundError:
            # Re-raise the exception
            raise
        except Exception as e:
            raise AlertStorageError(
                message=f"Failed to acknowledge alert: {str(e)}",
                operation="acknowledge_alert",
                details={"alert_id": alert_id, "user": user}
            )

    @with_exception_handling
    def delete_alert(self, alert_id: str) -> bool:
        """
        Delete an alert

        Args:
            alert_id: ID of the alert

        Returns:
            True if the alert was deleted

        Raises:
            AlertNotFoundError: If the alert is not found
            AlertStorageError: If there's an error deleting the alert
        """
        if alert_id not in self.alerts:
            raise AlertNotFoundError(
                message=f"Alert with ID {alert_id} not found",
                alert_id=alert_id
            )

        try:
            # Delete from memory
            del self.alerts[alert_id]

            # Delete from disk
            path = self._alert_path(alert_id)
            if os.path.exists(path):
                os.remove(path)

            return True
        except Exception as e:
            raise AlertStorageError(
                message=f"Failed to delete alert: {str(e)}",
                operation="delete_alert",
                details={"alert_id": alert_id}
            )

    def _alert_path(self, alert_id: str) -> str:
        """Get the path to an alert file"""
        return os.path.join(self.data_dir, f"alert_{alert_id}.json")

    @with_exception_handling
    def _save_alert(self, alert: Alert) -> None:
        """
        Save an alert to disk

        Args:
            alert: The alert to save

        Raises:
            AlertStorageError: If there's an error saving the alert
        """
        path = self._alert_path(alert.id)

        try:
            with open(path, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2)
        except Exception as e:
            raise AlertStorageError(
                message=f"Failed to save alert to disk: {str(e)}",
                operation="save_alert",
                details={"alert_id": alert.id, "path": path}
            )

    @with_exception_handling
    def _load_alerts(self) -> None:
        """
        Load alerts from disk

        Raises:
            AlertStorageError: If there's an error loading alerts
        """
        if not os.path.exists(self.data_dir):
            return

        try:
            # Find all alert files
            files = [f for f in os.listdir(self.data_dir)
                   if f.startswith("alert_") and f.endswith(".json")]

            load_errors = []
            for file in files:
                path = os.path.join(self.data_dir, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)

                    alert = Alert.from_dict(data)
                    self.alerts[alert.id] = alert

                except Exception as e:
                    error_msg = f"Error loading alert from {path}: {str(e)}"
                    logger.error(error_msg)
                    load_errors.append({"file": file, "error": str(e)})

            logger.info(f"Loaded {len(self.alerts)} alerts")

            # If we had errors loading some alerts, log them but don't fail
            if load_errors:
                logger.warning(f"Failed to load {len(load_errors)} alerts")
        except Exception as e:
            raise AlertStorageError(
                message=f"Failed to load alerts from disk: {str(e)}",
                operation="load_alerts",
                details={"data_dir": self.data_dir}
            )

    @with_exception_handling
    def _cleanup_expired_alerts(self) -> None:
        """
        Clean up expired alerts

        Raises:
            AlertStorageError: If there's an error deleting expired alerts
        """
        try:
            expired = [a.id for a in self.alerts.values() if a.is_expired]

            for alert_id in expired:
                try:
                    self.delete_alert(alert_id)
                except Exception as e:
                    logger.warning(f"Failed to delete expired alert {alert_id}: {str(e)}")

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired alerts")
        except Exception as e:
            raise AlertStorageError(
                message=f"Failed to clean up expired alerts: {str(e)}",
                operation="cleanup_expired_alerts",
                details={"error": str(e)}
            )


class NotificationDispatcher:
    """Dispatcher for sending notifications"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the notification dispatcher

        Args:
            config: Configuration for notification channels
        """
        self.config = config or {}

        # Initialize notification handlers
        self.handlers: Dict[NotificationChannel, Callable] = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.SMS: self._send_sms,
            NotificationChannel.PUSH: self._send_push,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.UI: self._send_ui
        }

        # Initialize notification queue
        self.queue: queue.Queue = queue.Queue()

        # Start dispatcher thread
        self.stop_flag = threading.Event()
        self.dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            daemon=True
        )
        self.dispatcher_thread.start()

    @with_exception_handling
    def dispatch(self, alert: Alert) -> None:
        """
        Dispatch notifications for an alert

        Args:
            alert: The alert to dispatch

        Raises:
            NotificationError: If there's an error dispatching the notification
        """
        try:
            # Add to queue
            self.queue.put(alert)
        except Exception as e:
            raise NotificationError(
                message=f"Failed to dispatch notification: {str(e)}",
                alert_id=alert.id,
                details={"error": str(e)}
            )

    @with_exception_handling
    def shutdown(self) -> None:
        """
        Shut down the dispatcher

        Raises:
            NotificationError: If there's an error shutting down the dispatcher
        """
        try:
            self.stop_flag.set()
            self.dispatcher_thread.join(timeout=5.0)

            # Check if thread is still alive
            if self.dispatcher_thread.is_alive():
                logger.warning("Dispatcher thread did not shut down gracefully within timeout")
        except Exception as e:
            raise NotificationError(
                message=f"Failed to shut down notification dispatcher: {str(e)}",
                details={"error": str(e)}
            )

    def _dispatcher_loop(self) -> None:
        """Background thread for dispatching notifications"""
        while not self.stop_flag.is_set():
            try:
                # Get alert from queue (with timeout to check stop flag)
                try:
                    alert = self.queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Dispatch to each channel
                for channel in alert.channels:
                    try:
                        # Get handler for channel
                        handler = self.handlers.get(channel)

                        if handler:
                            # Send notification
                            success = handler(alert)

                            if success:
                                # Record successful notification
                                channel_name = str(channel)
                                alert.sent_notifications[channel_name] = datetime.now()

                                logger.debug(f"Sent {channel_name} notification for alert {alert.id}")
                            else:
                                logger.warning(f"Failed to send {channel} notification for alert {alert.id}")

                    except Exception as e:
                        logger.error(f"Error sending {channel} notification for alert {alert.id}: {str(e)}")

                # Mark task as done
                self.queue.task_done()

            except Exception as e:
                logger.error(f"Error in notification dispatcher: {str(e)}")

    @with_exception_handling
    def _send_email(self, alert: Alert) -> bool:
        """
        Send email notification

        Args:
            alert: The alert to send

        Returns:
            True if the notification was sent successfully

        Raises:
            NotificationError: If there's an error sending the notification
        """
        try:
            # This would integrate with an email service
            # For now, just log it
            logger.info(f"Would send email for alert {alert.id}: {alert.title}")
            return True
        except Exception as e:
            raise NotificationError(
                message=f"Failed to send email notification: {str(e)}",
                channel="EMAIL",
                alert_id=alert.id,
                details={"error": str(e)}
            )

    @with_exception_handling
    def _send_sms(self, alert: Alert) -> bool:
        """
        Send SMS notification

        Args:
            alert: The alert to send

        Returns:
            True if the notification was sent successfully

        Raises:
            NotificationError: If there's an error sending the notification
        """
        try:
            # This would integrate with an SMS service
            # For now, just log it
            logger.info(f"Would send SMS for alert {alert.id}: {alert.title}")
            return True
        except Exception as e:
            raise NotificationError(
                message=f"Failed to send SMS notification: {str(e)}",
                channel="SMS",
                alert_id=alert.id,
                details={"error": str(e)}
            )

    @with_exception_handling
    def _send_push(self, alert: Alert) -> bool:
        """
        Send push notification

        Args:
            alert: The alert to send

        Returns:
            True if the notification was sent successfully

        Raises:
            NotificationError: If there's an error sending the notification
        """
        try:
            # This would integrate with a push notification service
            # For now, just log it
            logger.info(f"Would send push notification for alert {alert.id}: {alert.title}")
            return True
        except Exception as e:
            raise NotificationError(
                message=f"Failed to send push notification: {str(e)}",
                channel="PUSH",
                alert_id=alert.id,
                details={"error": str(e)}
            )

    @with_exception_handling
    def _send_slack(self, alert: Alert) -> bool:
        """
        Send Slack notification

        Args:
            alert: The alert to send

        Returns:
            True if the notification was sent successfully

        Raises:
            NotificationError: If there's an error sending the notification
        """
        try:
            # This would integrate with Slack
            # For now, just log it
            logger.info(f"Would send Slack message for alert {alert.id}: {alert.title}")
            return True
        except Exception as e:
            raise NotificationError(
                message=f"Failed to send Slack notification: {str(e)}",
                channel="SLACK",
                alert_id=alert.id,
                details={"error": str(e)}
            )

    @with_exception_handling
    def _send_webhook(self, alert: Alert) -> bool:
        """
        Send webhook notification

        Args:
            alert: The alert to send

        Returns:
            True if the notification was sent successfully

        Raises:
            NotificationError: If there's an error sending the notification
        """
        try:
            # This would call a webhook
            # For now, just log it
            logger.info(f"Would send webhook for alert {alert.id}: {alert.title}")
            return True
        except Exception as e:
            raise NotificationError(
                message=f"Failed to send webhook notification: {str(e)}",
                channel="WEBHOOK",
                alert_id=alert.id,
                details={"error": str(e)}
            )

    @with_exception_handling
    def _send_ui(self, alert: Alert) -> bool:
        """
        Send UI notification

        Args:
            alert: The alert to send

        Returns:
            True if the notification was sent successfully

        Raises:
            NotificationError: If there's an error sending the notification
        """
        try:
            # This would send to the UI via websocket or similar
            # For now, just log it
            logger.info(f"Would send UI notification for alert {alert.id}: {alert.title}")
            return True
        except Exception as e:
            raise NotificationError(
                message=f"Failed to send UI notification: {str(e)}",
                channel="UI",
                alert_id=alert.id,
                details={"error": str(e)}
            )


class AlertGenerator:
    """Generator for indicator-based alerts"""

    def __init__(self, manager: AlertManager, dispatcher: NotificationDispatcher):
        """
        Initialize the alert generator

        Args:
            manager: Alert manager
            dispatcher: Notification dispatcher
        """
        self.manager = manager
        self.dispatcher = dispatcher

        # Store indicator thresholds
        self.indicator_thresholds: Dict[str, Dict[str, float]] = {}

    @with_exception_handling
    def set_indicator_threshold(self, indicator_name: str, threshold_type: str, value: float) -> None:
        """
        Set a threshold for indicator alerts

        Args:
            indicator_name: Name of the indicator
            threshold_type: Type of threshold (e.g., "overbought", "oversold")
            value: Threshold value

        Raises:
            ThresholdValidationError: If the threshold value is invalid
        """
        # Validate inputs
        if not indicator_name:
            raise ThresholdValidationError(
                message="Indicator name cannot be empty",
                threshold_type=threshold_type,
                value=value
            )

        if not threshold_type:
            raise ThresholdValidationError(
                message="Threshold type cannot be empty",
                threshold_type=threshold_type,
                value=value
            )

        # Validate threshold values based on type
        if threshold_type == "overbought" and value <= 0:
            raise ThresholdValidationError(
                message="Overbought threshold must be greater than 0",
                threshold_type=threshold_type,
                value=value
            )

        if threshold_type == "oversold" and value <= 0:
            raise ThresholdValidationError(
                message="Oversold threshold must be greater than 0",
                threshold_type=threshold_type,
                value=value
            )

        # Store threshold
        if indicator_name not in self.indicator_thresholds:
            self.indicator_thresholds[indicator_name] = {}

        self.indicator_thresholds[indicator_name][threshold_type] = value

    @with_exception_handling
    def check_indicator_value(self, indicator_name: str, value: float,
                            instrument: str, timeframe: str) -> Optional[Alert]:
        """
        Check if an indicator value triggers an alert

        Args:
            indicator_name: Name of the indicator
            value: Current value of the indicator
            instrument: Trading instrument
            timeframe: Chart timeframe

        Returns:
            Alert if triggered, None otherwise

        Raises:
            AlertRuleError: If there's an error checking the indicator value
        """
        try:
            if not indicator_name:
                raise AlertRuleError(
                    message="Indicator name cannot be empty",
                    rule="check_indicator_value"
                )

            if indicator_name not in self.indicator_thresholds:
                return None

            thresholds = self.indicator_thresholds[indicator_name]

            # Check for overbought condition
            if "overbought" in thresholds and value >= thresholds["overbought"]:
                return self._create_indicator_alert(
                    indicator_name=indicator_name,
                    instrument=instrument,
                    timeframe=timeframe,
                    condition="overbought",
                    value=value,
                    threshold=thresholds["overbought"]
                )

            # Check for oversold condition
            if "oversold" in thresholds and value <= thresholds["oversold"]:
                return self._create_indicator_alert(
                    indicator_name=indicator_name,
                    instrument=instrument,
                    timeframe=timeframe,
                    condition="oversold",
                    value=value,
                    threshold=thresholds["oversold"]
                )

            # No alert triggered
            return None
        except Exception as e:
            if isinstance(e, AlertRuleError):
                raise
            raise AlertRuleError(
                message=f"Error checking indicator value: {str(e)}",
                rule="check_indicator_value",
                details={
                    "indicator_name": indicator_name,
                    "value": value,
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "error": str(e)
                }
            )

    def check_indicator_cross(self, indicator_name: str, value: float, previous_value: float,
                            reference_value: float, instrument: str, timeframe: str) -> Optional[Alert]:
        """
        Check if an indicator crosses a reference value

        Args:
            indicator_name: Name of the indicator
            value: Current value of the indicator
            previous_value: Previous value of the indicator
            reference_value: Reference value to check crossing against
            instrument: Trading instrument
            timeframe: Chart timeframe

        Returns:
            Alert if triggered, None otherwise
        """
        # Check for crossing above
        if previous_value <= reference_value and value > reference_value:
            return self._create_indicator_alert(
                indicator_name=indicator_name,
                instrument=instrument,
                timeframe=timeframe,
                condition="cross_above",
                value=value,
                threshold=reference_value
            )

        # Check for crossing below
        if previous_value >= reference_value and value < reference_value:
            return self._create_indicator_alert(
                indicator_name=indicator_name,
                instrument=instrument,
                timeframe=timeframe,
                condition="cross_below",
                value=value,
                threshold=reference_value
            )

        # No alert triggered
        return None

    def check_indicators_cross(self, indicator1_name: str, indicator1_value: float,
                             indicator2_name: str, indicator2_value: float,
                             previous1_value: float, previous2_value: float,
                             instrument: str, timeframe: str) -> Optional[Alert]:
        """
        Check if two indicators cross each other

        Args:
            indicator1_name: Name of the first indicator
            indicator1_value: Current value of the first indicator
            indicator2_name: Name of the second indicator
            indicator2_value: Current value of the second indicator
            previous1_value: Previous value of the first indicator
            previous2_value: Previous value of the second indicator
            instrument: Trading instrument
            timeframe: Chart timeframe

        Returns:
            Alert if triggered, None otherwise
        """
        # Check for crossing above
        if previous1_value <= previous2_value and indicator1_value > indicator2_value:
            return Alert(
                id="",
                timestamp=datetime.now(),
                level=AlertLevel.MEDIUM,
                type=AlertType.INDICATOR_SIGNAL,
                title=f"{indicator1_name} crossed above {indicator2_name}",
                message=f"{indicator1_name} ({indicator1_value:.4f}) crossed above {indicator2_name} ({indicator2_value:.4f}) on {instrument} {timeframe}",
                instrument=instrument,
                timeframe=timeframe,
                source=f"{indicator1_name}/{indicator2_name}",
                metadata={
                    "indicator1": indicator1_name,
                    "indicator1_value": indicator1_value,
                    "indicator2": indicator2_name,
                    "indicator2_value": indicator2_value,
                    "condition": "cross_above"
                }
            )

        # Check for crossing below
        if previous1_value >= previous2_value and indicator1_value < indicator2_value:
            return Alert(
                id="",
                timestamp=datetime.now(),
                level=AlertLevel.MEDIUM,
                type=AlertType.INDICATOR_SIGNAL,
                title=f"{indicator1_name} crossed below {indicator2_name}",
                message=f"{indicator1_name} ({indicator1_value:.4f}) crossed below {indicator2_name} ({indicator2_value:.4f}) on {instrument} {timeframe}",
                instrument=instrument,
                timeframe=timeframe,
                source=f"{indicator1_name}/{indicator2_name}",
                metadata={
                    "indicator1": indicator1_name,
                    "indicator1_value": indicator1_value,
                    "indicator2": indicator2_name,
                    "indicator2_value": indicator2_value,
                    "condition": "cross_below"
                }
            )

        # No alert triggered
        return None

    def check_price_level(self, price: float, level: float, instrument: str,
                        direction: str = "any", tolerance: float = 0.0001) -> Optional[Alert]:
        """
        Check if price reaches a specific level

        Args:
            price: Current price
            level: Price level to check
            instrument: Trading instrument
            direction: Direction to check ("above", "below", or "any")
            tolerance: Tolerance for level comparison

        Returns:
            Alert if triggered, None otherwise
        """
        # Check if price is within tolerance of level
        if abs(price - level) <= tolerance:
            condition = "reached"
        # Check if price crossed above level
        elif direction in ("above", "any") and price > level:
            condition = "above"
        # Check if price crossed below level
        elif direction in ("below", "any") and price < level:
            condition = "below"
        else:
            # No alert triggered
            return None

        return Alert(
            id="",
            timestamp=datetime.now(),
            level=AlertLevel.HIGH,
            type=AlertType.PRICE_LEVEL,
            title=f"Price {condition} {level}",
            message=f"Price ({price:.4f}) {condition} {level:.4f} on {instrument}",
            instrument=instrument,
            timeframe="",
            source="price",
            metadata={
                "price": price,
                "level": level,
                "condition": condition
            }
        )

    def check_concordance(self, concordance_level: str, concordance_score: float,
                        instrument: str, indicators: List[str]) -> Optional[Alert]:
        """
        Check if indicators reach a concordance threshold

        Args:
            concordance_level: Concordance level (e.g., "STRONG_AGREEMENT")
            concordance_score: Concordance score (-1 to 1)
            instrument: Trading instrument
            indicators: List of indicators included in concordance

        Returns:
            Alert if triggered, None otherwise
        """
        # Only alert on strong agreement or disagreement
        if concordance_level not in ("STRONG_AGREEMENT", "STRONG_DISAGREEMENT"):
            return None

        direction = "bullish" if concordance_score > 0 else "bearish"

        return Alert(
            id="",
            timestamp=datetime.now(),
            level=AlertLevel.HIGH,
            type=AlertType.CONCORDANCE,
            title=f"Strong {direction} indicator concordance",
            message=f"Indicators show strong {direction} concordance (score: {concordance_score:.2f}) on {instrument}",
            instrument=instrument,
            timeframe="",
            source="concordance",
            metadata={
                "concordance_level": concordance_level,
                "concordance_score": concordance_score,
                "direction": direction,
                "indicators": indicators
            }
        )

    @with_exception_handling
    def submit_alert(self, alert: Alert) -> str:
        """
        Submit an alert for processing

        Args:
            alert: The alert to submit

        Returns:
            Alert ID

        Raises:
            MonitoringAlertingError: If there's an error submitting the alert
            AlertStorageError: If there's an error storing the alert
            NotificationError: If there's an error dispatching notifications
        """
        try:
            # Validate alert
            if not alert:
                raise MonitoringAlertingError(
                    message="Alert cannot be None",
                    error_code="INVALID_ALERT",
                    details={"alert": str(alert)}
                )

            # Add alert to manager
            alert_id = self.manager.add_alert(alert)

            # Dispatch notifications
            self.dispatcher.dispatch(alert)

            return alert_id
        except (AlertStorageError, NotificationError):
            # Re-raise specific exceptions
            raise
        except Exception as e:
            raise MonitoringAlertingError(
                message=f"Failed to submit alert: {str(e)}",
                error_code="ALERT_SUBMISSION_ERROR",
                details={"error": str(e)}
            )

    @with_exception_handling
    def _create_indicator_alert(self, indicator_name: str, instrument: str, timeframe: str,
                              condition: str, value: float, threshold: float) -> Alert:
        """
        Create an alert for an indicator condition

        Args:
            indicator_name: Name of the indicator
            instrument: Trading instrument
            timeframe: Chart timeframe
            condition: Condition that triggered the alert
            value: Current indicator value
            threshold: Threshold that was crossed

        Returns:
            Alert object

        Raises:
            AlertRuleError: If there's an error creating the alert
        """
        try:
            # Validate inputs
            if not indicator_name:
                raise AlertRuleError(
                    message="Indicator name cannot be empty",
                    rule="create_indicator_alert"
                )

            if not condition:
                raise AlertRuleError(
                    message="Condition cannot be empty",
                    rule="create_indicator_alert"
                )

            # Determine alert level based on condition
            if condition in ("overbought", "oversold"):
                level = AlertLevel.MEDIUM
                alert_type = AlertType.INDICATOR_EXTREME
                title = f"{indicator_name} {condition}"
                message = f"{indicator_name} is {condition} ({value:.4f}) on {instrument} {timeframe}"
            elif condition.startswith("cross_"):
                level = AlertLevel.MEDIUM
                alert_type = AlertType.INDICATOR_SIGNAL
                title = f"{indicator_name} crossed {condition.split('_')[1]} {threshold}"
                message = f"{indicator_name} ({value:.4f}) crossed {condition.split('_')[1]} {threshold} on {instrument} {timeframe}"
            else:
                raise AlertRuleError(
                    message=f"Unknown condition: {condition}",
                    rule="create_indicator_alert",
                    details={"condition": condition}
                )

            return Alert(
                id="",
                timestamp=datetime.now(),
                level=level,
                type=alert_type,
                title=title,
                message=message,
                instrument=instrument,
                timeframe=timeframe,
                source=indicator_name,
                metadata={
                    "indicator": indicator_name,
                    "value": value,
                    "threshold": threshold,
                    "condition": condition
                }
            )
        except Exception as e:
            if isinstance(e, AlertRuleError):
                raise
            raise AlertRuleError(
                message=f"Error creating indicator alert: {str(e)}",
                rule="create_indicator_alert",
                details={
                    "indicator_name": indicator_name,
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "condition": condition,
                    "value": value,
                    "threshold": threshold,
                    "error": str(e)
                }
            )


@with_exception_handling
def implement_alert_system():
    """
    Implements the alert system based on indicator signals.
    - Develops mechanism for prioritizing alerts.
    - Creates integrated notification system (integration point).

    Returns:
        Tuple of (alert manager, notification dispatcher, alert generator)

    Raises:
        MonitoringAlertingError: If there's an error initializing the alert system
    """
    try:
        # Initialize alert manager
        manager = AlertManager(data_dir="./data/alerts")

        # Initialize notification dispatcher
        dispatcher = NotificationDispatcher()

        # Initialize alert generator
        generator = AlertGenerator(manager, dispatcher)

        # Set up some example indicator thresholds
        generator.set_indicator_threshold("RSI", "overbought", 70.0)
        generator.set_indicator_threshold("RSI", "oversold", 30.0)
        generator.set_indicator_threshold("Stochastic", "overbought", 80.0)
        generator.set_indicator_threshold("Stochastic", "oversold", 20.0)

        # Create a sample alert for demonstration
        sample_alert = Alert(
            id="",
            timestamp=datetime.now(),
            level=AlertLevel.MEDIUM,
            type=AlertType.INDICATOR_SIGNAL,
            title="MACD Bullish Crossover",
            message="MACD crossed above signal line on EUR/USD 1H",
            instrument="EUR/USD",
            timeframe="1H",
            source="MACD",
            metadata={
                "indicator": "MACD",
                "condition": "cross_above",
                "value": 0.0023,
                "threshold": 0.0
            }
        )

        # Add sample alert
        generator.submit_alert(sample_alert)

        logger.info("Alert system initialized")

        return (manager, dispatcher, generator)
    except Exception as e:
        if isinstance(e, (AlertStorageError, NotificationError, AlertRuleError)):
            # Re-raise specific exceptions
            raise
        raise MonitoringAlertingError(
            message=f"Failed to initialize alert system: {str(e)}",
            error_code="ALERT_SYSTEM_INITIALIZATION_ERROR",
            details={"error": str(e)}
        )
