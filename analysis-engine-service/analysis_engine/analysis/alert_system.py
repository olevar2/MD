"""
Alert System Module for Indicator Signals

This module provides a comprehensive system for alerting on indicator signals with:
- Alert definition and management
- Multi-channel notification dispatching
- Alert filtering and prioritization
- Alert history and persistence
"""

import logging
import uuid
import json
import threading
import queue
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# Local imports
from analysis_engine.analysis.signal_system import SignalType, Signal, AggregatedSignal

# Configure logging
logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Priority levels for alerts"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    INFO = 4


class AlertStatus(Enum):
    """Status values for alerts"""
    NEW = "new"
    NOTIFYING = "notifying"  # Currently sending notifications
    NOTIFIED = "notified"    # Notifications sent
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    IGNORED = "ignored"


class NotificationType(Enum):
    """Types of notification channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    UI = "ui"
    MOBILE_PUSH = "mobile_push"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel"""
    type: NotificationType
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    # For testing/debugging
    simulate_success: bool = False
    simulate_failure: bool = False
    simulate_delay: float = 0.0


@dataclass
class AlertRule:
    """Rule defining when to trigger an alert"""
    id: str
    name: str
    description: str
    priority: AlertPriority
    instruments: List[str]  # List of instruments (e.g., 'EURUSD', 'GBPJPY')
    timeframes: List[str]   # List of timeframes (e.g., '1m', '5m', '1h', '1d')
    
    # Signal criteria
    signal_type: Optional[SignalType] = None
    min_strength: float = 0.0
    min_confidence: float = 0.0
    
    # Indicator criteria
    indicators: List[str] = field(default_factory=list)  # Which indicators to monitor
    
    # Additional conditions (custom function)
    condition_func: Optional[Callable[[AggregatedSignal], bool]] = None
    
    # Notification settings
    notification_channels: List[NotificationType] = field(default_factory=list)
    throttle_period: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # Message templates
    title_template: str = "{signal_type} signal for {instrument} on {timeframe}"
    message_template: str = "A {priority} {signal_type} signal was generated for {instrument} on {timeframe} with strength {strength:.2f} and confidence {confidence:.2f}"
    
    # Rule status
    enabled: bool = True
    
    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        if not self.notification_channels:
            # Default to UI notification
            self.notification_channels = [NotificationType.UI]
    
    def matches(self, signal: AggregatedSignal, instrument: str, timeframe: str) -> bool:
        """
        Check if a signal matches this alert rule
        
        Args:
            signal: The signal to check
            instrument: The instrument the signal is for
            timeframe: The timeframe the signal is for
            
        Returns:
            True if the signal matches the rule, False otherwise
        """
        # Check instrument and timeframe
        if instrument not in self.instruments:
            return False
        
        if timeframe not in self.timeframes:
            return False
        
        # Check signal type if specified
        if self.signal_type is not None and signal.signal_type != self.signal_type:
            return False
        
        # Check strength and confidence
        if signal.strength < self.min_strength:
            return False
        
        if signal.confidence < self.min_confidence:
            return False
        
        # Check indicators if specified
        if self.indicators:
            # Check if any of the contributing signals are from the specified indicators
            contributing_indicators = {s.indicator_name for s in signal.contributing_signals}
            if not any(ind in contributing_indicators for ind in self.indicators):
                return False
        
        # Check custom condition if specified
        if self.condition_func and not self.condition_func(signal):
            return False
        
        # All conditions met
        return True
    
    def format_title(self, signal: AggregatedSignal, instrument: str, timeframe: str) -> str:
        """Format the alert title using the template"""
        return self.title_template.format(
            signal_type=signal.signal_type.name,
            instrument=instrument,
            timeframe=timeframe,
            priority=self.priority.name,
            strength=signal.strength,
            confidence=signal.confidence
        )
    
    def format_message(self, signal: AggregatedSignal, instrument: str, timeframe: str) -> str:
        """Format the alert message using the template"""
        contributing = ", ".join(s.indicator_name for s in signal.contributing_signals)
        return self.message_template.format(
            signal_type=signal.signal_type.name,
            instrument=instrument,
            timeframe=timeframe,
            priority=self.priority.name,
            strength=signal.strength,
            confidence=signal.confidence,
            contributing_indicators=contributing,
            num_indicators=len(signal.contributing_signals)
        )


@dataclass
class Alert:
    """Alert instance with notification status"""
    id: str
    rule_id: str
    instrument: str
    timeframe: str
    signal: AggregatedSignal
    title: str
    message: str
    priority: AlertPriority
    created_at: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.NEW
    notifications: Dict[NotificationType, Dict[str, Any]] = field(default_factory=dict)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def acknowledge(self, user_id: str) -> None:
        """
        Acknowledge the alert
        
        Args:
            user_id: ID of the user acknowledging the alert
        """
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = user_id
    
    def resolve(self) -> None:
        """Mark the alert as resolved"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
    
    def ignore(self) -> None:
        """Mark the alert as ignored"""
        self.status = AlertStatus.IGNORED
    
    def update_notification_status(self, 
                                 channel: NotificationType, 
                                 success: bool, 
                                 details: Dict[str, Any] = None) -> None:
        """
        Update the notification status for a channel
        
        Args:
            channel: The notification channel
            success: Whether the notification was successful
            details: Optional details about the notification
        """
        if not details:
            details = {}
            
        if channel not in self.notifications:
            self.notifications[channel] = {
                "attempts": 0,
                "success": False,
                "last_attempt": None,
                "first_attempt": None,
                "details": {}
            }
        
        notification = self.notifications[channel]
        notification["attempts"] += 1
        notification["last_attempt"] = datetime.now()
        
        if "first_attempt" not in notification or notification["first_attempt"] is None:
            notification["first_attempt"] = datetime.now()
        
        notification["success"] = success
        notification["details"].update(details)
    
    def should_expire(self) -> bool:
        """Check if the alert should be marked as expired"""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False


class NotificationDispatcher:
    """Handles dispatching notifications to various channels"""
    
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        """
        Initialize the notification dispatcher
        
        Args:
            config: Configuration for notification channels
        """
        self.config = config
        self._notification_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """Start the notification dispatcher worker thread"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Notification dispatcher already running")
            return
            
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._notification_worker,
            daemon=True
        )
        self._worker_thread.start()
        logger.info("Notification dispatcher worker thread started")
    
    def stop(self) -> None:
        """Stop the notification dispatcher worker thread"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            logger.warning("Notification dispatcher not running")
            return
            
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)
        logger.info("Notification dispatcher worker thread stopped")
    
    def dispatch(self, 
               alert: Alert, 
               channels: List[NotificationType] = None) -> None:
        """
        Dispatch an alert to notification channels
        
        Args:
            alert: The alert to dispatch
            channels: The channels to use (defaults to all configured channels)
        """
        if channels is None:
            # Use alert's rule notification channels if available
            if hasattr(alert, 'rule') and hasattr(alert.rule, 'notification_channels'):
                channels = alert.rule.notification_channels
            else:
                # Default to all configured channels
                channels = [NotificationType(c) for c in self.config.keys() if self.config[c].get("enabled", False)]
        
        # Queue for each channel
        for channel in channels:
            if str(channel.value) in self.config and self.config[str(channel.value)].get("enabled", False):
                self._notification_queue.put((alert, channel))
    
    def _notification_worker(self) -> None:
        """Worker thread to process the notification queue"""
        while not self._stop_event.is_set():
            try:
                # Get an item from the queue with timeout
                try:
                    alert, channel = self._notification_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the notification
                try:
                    success, details = self._send_notification(alert, channel)
                    alert.update_notification_status(channel, success, details)
                except Exception as e:
                    logger.error(f"Error sending {channel} notification: {str(e)}")
                    alert.update_notification_status(channel, False, {"error": str(e)})
                
                # Mark task as done
                self._notification_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in notification worker: {str(e)}")
    
    def _send_notification(self, 
                          alert: Alert, 
                          channel: NotificationType) -> Tuple[bool, Dict[str, Any]]:
        """
        Send a notification through a specific channel
        
        Args:
            alert: The alert to notify about
            channel: The notification channel to use
            
        Returns:
            Tuple of (success, details)
        """
        channel_config = self.config.get(str(channel.value), {})
        
        # Check for simulation settings
        if channel_config.get("simulate_delay", 0) > 0:
            time.sleep(channel_config["simulate_delay"])
        
        if channel_config.get("simulate_failure", False):
            return False, {"error": "Simulated failure"}
        
        if channel_config.get("simulate_success", False):
            return True, {"message": "Simulated success"}
        
        # Real notification logic
        if channel == NotificationType.EMAIL:
            return self._send_email_notification(alert, channel_config)
            
        elif channel == NotificationType.SMS:
            return self._send_sms_notification(alert, channel_config)
            
        elif channel == NotificationType.SLACK:
            return self._send_slack_notification(alert, channel_config)
            
        elif channel == NotificationType.WEBHOOK:
            return self._send_webhook_notification(alert, channel_config)
            
        elif channel == NotificationType.UI:
            return self._send_ui_notification(alert, channel_config)
            
        elif channel == NotificationType.MOBILE_PUSH:
            return self._send_mobile_notification(alert, channel_config)
            
        else:
            return False, {"error": f"Unsupported notification channel: {channel}"}
    
    def _send_email_notification(self, 
                               alert: Alert, 
                               config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Send an email notification"""
        try:
            smtp_host = config.get("smtp_host", "localhost")
            smtp_port = config.get("smtp_port", 25)
            use_ssl = config.get("use_ssl", False)
            use_tls = config.get("use_tls", False)
            username = config.get("username")
            password = config.get("password")
            from_email = config.get("from_email", "alerts@forex-platform.com")
            to_emails = config.get("to_emails", [])
            
            if not to_emails:
                return False, {"error": "No recipient emails configured"}
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = from_email
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = f"Forex Alert: {alert.title}"
            
            # Create HTML and plain text versions
            text = f"{alert.message}\n\nAlert ID: {alert.id}\nCreated: {alert.created_at}\nPriority: {alert.priority.name}"
            html = f"""
            <html>
              <head></head>
              <body>
                <h1>{alert.title}</h1>
                <p>{alert.message}</p>
                <p>
                  <strong>Alert ID:</strong> {alert.id}<br>
                  <strong>Created:</strong> {alert.created_at}<br>
                  <strong>Priority:</strong> {alert.priority.name}<br>
                  <strong>Instrument:</strong> {alert.instrument}<br>
                  <strong>Timeframe:</strong> {alert.timeframe}<br>
                </p>
              </body>
            </html>
            """
            
            plain_part = MIMEText(text, "plain")
            html_part = MIMEText(html, "html")
            
            msg.attach(plain_part)
            msg.attach(html_part)
            
            # Connect to SMTP server and send
            if use_ssl:
                server = smtplib.SMTP_SSL(smtp_host, smtp_port)
            else:
                server = smtplib.SMTP(smtp_host, smtp_port)
            
            if use_tls:
                server.starttls()
                
            if username and password:
                server.login(username, password)
                
            server.send_message(msg)
            server.quit()
            
            return True, {"recipients": len(to_emails)}
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _send_sms_notification(self, 
                             alert: Alert, 
                             config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Send an SMS notification"""
        try:
            # Implementation depends on SMS provider
            provider = config.get("provider", "twilio")
            
            if provider == "twilio":
                # Example Twilio implementation
                api_key = config.get("api_key")
                api_secret = config.get("api_secret")
                from_number = config.get("from_number")
                to_numbers = config.get("to_numbers", [])
                
                if not (api_key and api_secret and from_number and to_numbers):
                    return False, {"error": "Incomplete Twilio configuration"}
                
                # In a real implementation, would import and use Twilio client
                # For now, simulate success
                return True, {
                    "provider": "twilio",
                    "recipients": len(to_numbers)
                }
            else:
                return False, {"error": f"Unsupported SMS provider: {provider}"}
        
        except Exception as e:
            return False, {"error": str(e)}
    
    def _send_slack_notification(self, 
                               alert: Alert, 
                               config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Send a Slack notification"""
        try:
            webhook_url = config.get("webhook_url")
            
            if not webhook_url:
                return False, {"error": "No Slack webhook URL configured"}
            
            # Build the Slack message
            message = {
                "text": f"*{alert.title}*",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": alert.title
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": alert.message
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Alert ID:* {alert.id}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Priority:* {alert.priority.name}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Created:* {alert.created_at}"
                            }
                        ]
                    }
                ]
            }
            
            # Send the message
            response = requests.post(
                webhook_url,
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            success = response.status_code == 200
            
            return success, {
                "status_code": response.status_code,
                "response": response.text
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _send_webhook_notification(self, 
                                 alert: Alert, 
                                 config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Send a webhook notification"""
        try:
            webhook_url = config.get("webhook_url")
            headers = config.get("headers", {})
            method = config.get("method", "POST")
            
            if not webhook_url:
                return False, {"error": "No webhook URL configured"}
            
            # Build the payload
            payload = {
                "alert_id": alert.id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "priority": alert.priority.name,
                "instrument": alert.instrument,
                "timeframe": alert.timeframe,
                "created_at": alert.created_at.isoformat(),
                "signal_type": alert.signal.signal_type.name,
                "signal_strength": alert.signal.strength,
                "signal_confidence": alert.signal.confidence,
            }
            
            # Send the webhook
            response = requests.request(
                method,
                webhook_url,
                json=payload,
                headers=headers
            )
            
            success = response.status_code >= 200 and response.status_code < 300
            
            return success, {
                "status_code": response.status_code,
                "response": response.text
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _send_ui_notification(self, 
                            alert: Alert, 
                            config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Send a UI notification"""
        try:
            # In a real implementation, would push to a message queue or store
            # for the UI to retrieve. For now, simulate success
            return True, {"message": "UI notification queued"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def _send_mobile_notification(self, 
                                alert: Alert, 
                                config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Send a mobile push notification"""
        try:
            provider = config.get("provider", "firebase")
            
            if provider == "firebase":
                # Example Firebase implementation
                api_key = config.get("api_key")
                device_tokens = config.get("device_tokens", [])
                
                if not (api_key and device_tokens):
                    return False, {"error": "Incomplete Firebase configuration"}
                
                # In a real implementation, would import and use Firebase client
                # For now, simulate success
                return True, {
                    "provider": "firebase",
                    "recipients": len(device_tokens)
                }
            else:
                return False, {"error": f"Unsupported push notification provider: {provider}"}
        except Exception as e:
            return False, {"error": str(e)}


class AlertManager:
    """Manages alerts, rules, and notification dispatch"""
    
    def __init__(self, notification_config: Dict[str, Dict[str, Any]] = None):
        """
        Initialize the alert manager
        
        Args:
            notification_config: Configuration for notification channels
        """
        # Initialize default configuration if none provided
        if notification_config is None:
            notification_config = {
                "ui": {"enabled": True},
                "email": {"enabled": False},
                "slack": {"enabled": False},
                "webhook": {"enabled": False},
                "sms": {"enabled": False},
                "mobile_push": {"enabled": False}
            }
            
        # Alert storage
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._max_history = 1000
        
        # Alert processing
        self._last_signal_timestamps: Dict[Tuple[str, str], datetime] = {}
        
        # Initialize notification dispatcher
        self._dispatcher = NotificationDispatcher(notification_config)
        self._dispatcher.start()
    
    def add_rule(self, rule: AlertRule) -> str:
        """
        Add a new alert rule
        
        Args:
            rule: The alert rule to add
            
        Returns:
            The ID of the added rule
        """
        self._rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.id})")
        return rule.id
    
    def update_rule(self, rule: AlertRule) -> bool:
        """
        Update an existing alert rule
        
        Args:
            rule: The alert rule to update
            
        Returns:
            True if the rule was updated, False if not found
        """
        if rule.id in self._rules:
            self._rules[rule.id] = rule
            logger.info(f"Updated alert rule: {rule.name} ({rule.id})")
            return True
        return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule
        
        Args:
            rule_id: The ID of the rule to remove
            
        Returns:
            True if the rule was removed, False if not found
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """
        Get an alert rule by ID
        
        Args:
            rule_id: The ID of the rule to get
            
        Returns:
            The alert rule, or None if not found
        """
        return self._rules.get(rule_id)
    
    def list_rules(self) -> List[AlertRule]:
        """
        List all alert rules
        
        Returns:
            List of alert rules
        """
        return list(self._rules.values())
    
    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable an alert rule
        
        Args:
            rule_id: The ID of the rule to enable
            
        Returns:
            True if the rule was enabled, False if not found
        """
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable an alert rule
        
        Args:
            rule_id: The ID of the rule to disable
            
        Returns:
            True if the rule was disabled, False if not found
        """
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            return True
        return False
    
    def process_signal(self, 
                      signal: AggregatedSignal,
                      instrument: str,
                      timeframe: str) -> List[Alert]:
        """
        Process a signal and generate alerts if it matches any rules
        
        Args:
            signal: The signal to process
            instrument: The instrument the signal is for
            timeframe: The timeframe the signal is for
            
        Returns:
            List of generated alerts
        """
        generated_alerts = []
        
        # Check each rule
        for rule in self._rules.values():
            # Skip disabled rules
            if not rule.enabled:
                continue
                
            # Check if rule matches signal
            if rule.matches(signal, instrument, timeframe):
                # Check throttling
                signal_key = (instrument, timeframe, rule.id)
                current_time = datetime.now()
                
                if signal_key in self._last_signal_timestamps:
                    last_time = self._last_signal_timestamps[signal_key]
                    if current_time - last_time < rule.throttle_period:
                        # Skip alert due to throttling
                        logger.debug(f"Throttled alert for rule {rule.id} and instrument {instrument}")
                        continue
                
                # Generate alert
                alert = Alert(
                    id=str(uuid.uuid4()),
                    rule_id=rule.id,
                    instrument=instrument,
                    timeframe=timeframe,
                    signal=signal,
                    title=rule.format_title(signal, instrument, timeframe),
                    message=rule.format_message(signal, instrument, timeframe),
                    priority=rule.priority,
                    created_at=current_time
                )
                
                # Store alert
                self._active_alerts[alert.id] = alert
                
                # Update throttling timestamp
                self._last_signal_timestamps[signal_key] = current_time
                
                # Dispatch notifications
                self._dispatcher.dispatch(alert, rule.notification_channels)
                
                generated_alerts.append(alert)
                logger.info(f"Generated alert: {alert.title} ({alert.id}) for rule {rule.name}")
        
        return generated_alerts
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: The ID of the alert to acknowledge
            user_id: The ID of the user acknowledging the alert
            
        Returns:
            True if the alert was acknowledged, False if not found
        """
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.acknowledge(user_id)
            
            # Move to history if resolved
            if alert.status in [AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED, AlertStatus.IGNORED]:
                self._move_to_history(alert_id)
                
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert
        
        Args:
            alert_id: The ID of the alert to resolve
            
        Returns:
            True if the alert was resolved, False if not found
        """
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolve()
            self._move_to_history(alert_id)
            return True
        return False
    
    def ignore_alert(self, alert_id: str) -> bool:
        """
        Ignore an alert
        
        Args:
            alert_id: The ID of the alert to ignore
            
        Returns:
            True if the alert was ignored, False if not found
        """
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.ignore()
            self._move_to_history(alert_id)
            return True
        return False
    
    def _move_to_history(self, alert_id: str) -> None:
        """
        Move an alert from active to history
        
        Args:
            alert_id: The ID of the alert to move
        """
        if alert_id in self._active_alerts:
            alert = self._active_alerts.pop(alert_id)
            self._alert_history.append(alert)
            
            # Trim history if needed
            if len(self._alert_history) > self._max_history:
                self._alert_history = self._alert_history[-self._max_history:]
    
    def get_active_alerts(self, 
                        instrument: str = None,
                        timeframe: str = None,
                        priority: AlertPriority = None,
                        rule_id: str = None) -> List[Alert]:
        """
        Get active alerts, optionally filtered
        
        Args:
            instrument: Filter by instrument
            timeframe: Filter by timeframe
            priority: Filter by priority
            rule_id: Filter by rule ID
            
        Returns:
            List of matching active alerts
        """
        alerts = list(self._active_alerts.values())
        
        # Apply filters
        if instrument:
            alerts = [a for a in alerts if a.instrument == instrument]
        
        if timeframe:
            alerts = [a for a in alerts if a.timeframe == timeframe]
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        if rule_id:
            alerts = [a for a in alerts if a.rule_id == rule_id]
        
        return sorted(alerts, key=lambda a: (a.priority.value, a.created_at))
    
    def get_alert_history(self, 
                        instrument: str = None,
                        timeframe: str = None,
                        priority: AlertPriority = None,
                        rule_id: str = None,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        status: AlertStatus = None) -> List[Alert]:
        """
        Get alert history, optionally filtered
        
        Args:
            instrument: Filter by instrument
            timeframe: Filter by timeframe
            priority: Filter by priority
            rule_id: Filter by rule ID
            start_date: Filter by start date
            end_date: Filter by end date
            status: Filter by status
            
        Returns:
            List of matching historical alerts
        """
        alerts = self._alert_history.copy()
        
        # Apply filters
        if instrument:
            alerts = [a for a in alerts if a.instrument == instrument]
        
        if timeframe:
            alerts = [a for a in alerts if a.timeframe == timeframe]
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        if rule_id:
            alerts = [a for a in alerts if a.rule_id == rule_id]
        
        if start_date:
            alerts = [a for a in alerts if a.created_at >= start_date]
        
        if end_date:
            alerts = [a for a in alerts if a.created_at <= end_date]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def cleanup_expired_alerts(self) -> int:
        """
        Check for and clean up expired alerts
        
        Returns:
            Number of alerts cleaned up
        """
        expired_count = 0
        expired_ids = []
        
        for alert_id, alert in self._active_alerts.items():
            if alert.should_expire():
                alert.status = AlertStatus.EXPIRED
                expired_ids.append(alert_id)
                expired_count += 1
                
        for alert_id in expired_ids:
            self._move_to_history(alert_id)
            
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired alerts")
            
        return expired_count
    
    def save_rules_to_file(self, filepath: str) -> bool:
        """
        Save alert rules to a JSON file
        
        Args:
            filepath: The file to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert rules to dictionaries
            rules_dict = {}
            for rule_id, rule in self._rules.items():
                rule_dict = {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "priority": rule.priority.name,
                    "instruments": rule.instruments,
                    "timeframes": rule.timeframes,
                    "signal_type": rule.signal_type.name if rule.signal_type else None,
                    "min_strength": rule.min_strength,
                    "min_confidence": rule.min_confidence,
                    "indicators": rule.indicators,
                    "notification_channels": [ch.name for ch in rule.notification_channels],
                    "throttle_period_seconds": rule.throttle_period.total_seconds(),
                    "title_template": rule.title_template,
                    "message_template": rule.message_template,
                    "enabled": rule.enabled
                }
                rules_dict[rule_id] = rule_dict
            
            # Write to file
            with open(filepath, "w") as f:
                json.dump(rules_dict, f, indent=2)
                
            logger.info(f"Saved {len(rules_dict)} alert rules to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving alert rules to file: {str(e)}")
            return False
    
    def load_rules_from_file(self, filepath: str) -> int:
        """
        Load alert rules from a JSON file
        
        Args:
            filepath: The file to load from
            
        Returns:
            Number of rules loaded
        """
        try:
            with open(filepath, "r") as f:
                rules_dict = json.load(f)
            
            rules_count = 0
            for rule_id, rule_data in rules_dict.items():
                # Create AlertRule
                rule = AlertRule(
                    id=rule_data.get("id", str(uuid.uuid4())),
                    name=rule_data["name"],
                    description=rule_data.get("description", ""),
                    priority=AlertPriority[rule_data["priority"]],
                    instruments=rule_data["instruments"],
                    timeframes=rule_data["timeframes"],
                    signal_type=SignalType[rule_data["signal_type"]] if rule_data.get("signal_type") else None,
                    min_strength=rule_data.get("min_strength", 0.0),
                    min_confidence=rule_data.get("min_confidence", 0.0),
                    indicators=rule_data.get("indicators", []),
                    notification_channels=[NotificationType[ch] for ch in rule_data.get("notification_channels", ["UI"])],
                    throttle_period=timedelta(seconds=rule_data.get("throttle_period_seconds", 3600)),
                    title_template=rule_data.get("title_template", "{signal_type} signal for {instrument} on {timeframe}"),
                    message_template=rule_data.get("message_template", "A {priority} {signal_type} signal was generated for {instrument} on {timeframe}"),
                    enabled=rule_data.get("enabled", True)
                )
                
                # Add rule
                self._rules[rule.id] = rule
                rules_count += 1
            
            logger.info(f"Loaded {rules_count} alert rules from {filepath}")
            return rules_count
            
        except Exception as e:
            logger.error(f"Error loading alert rules from file: {str(e)}")
            return 0
    
    def shutdown(self) -> None:
        """Shut down the alert manager"""
        self._dispatcher.stop()
        logger.info("Alert manager shut down")
    
    def __del__(self) -> None:
        """Ensure dispatcher is stopped when object is destroyed"""
        try:
            self._dispatcher.stop()
        except:
            pass


class AlertGenerator:
    """Utility class to generate alerts from indicator signals"""
    
    def __init__(self, alert_manager: AlertManager):
        """
        Initialize the alert generator
        
        Args:
            alert_manager: The alert manager to use
        """
        self.alert_manager = alert_manager
    
    def create_default_rules(self) -> List[str]:
        """
        Create a set of default alert rules
        
        Returns:
            List of created rule IDs
        """
        rule_ids = []
        
        # Strong buy signal
        strong_buy = AlertRule(
            id=str(uuid.uuid4()),
            name="Strong Buy Signal",
            description="Alert when a strong buy signal is generated",
            priority=AlertPriority.HIGH,
            instruments=["*"],  # All instruments
            timeframes=["*"],   # All timeframes
            signal_type=SignalType.BUY,
            min_strength=0.8,
            min_confidence=0.7,
            notification_channels=[NotificationType.UI, NotificationType.EMAIL]
        )
        rule_ids.append(self.alert_manager.add_rule(strong_buy))
        
        # Strong sell signal
        strong_sell = AlertRule(
            id=str(uuid.uuid4()),
            name="Strong Sell Signal",
            description="Alert when a strong sell signal is generated",
            priority=AlertPriority.HIGH,
            instruments=["*"],  # All instruments
            timeframes=["*"],   # All timeframes
            signal_type=SignalType.SELL,
            min_strength=0.8,
            min_confidence=0.7,
            notification_channels=[NotificationType.UI, NotificationType.EMAIL]
        )
        rule_ids.append(self.alert_manager.add_rule(strong_sell))
        
        # High volatility alert
        high_volatility = AlertRule(
            id=str(uuid.uuid4()),
            name="High Volatility",
            description="Alert when volatility increases significantly",
            priority=AlertPriority.MEDIUM,
            instruments=["*"],  # All instruments
            timeframes=["*"],   # All timeframes
            min_strength=0.7,
            indicators=["ATR", "Bollinger Bands"],
            notification_channels=[NotificationType.UI]
        )
        rule_ids.append(self.alert_manager.add_rule(high_volatility))
        
        # Reversal pattern alert
        reversal = AlertRule(
            id=str(uuid.uuid4()),
            name="Potential Reversal",
            description="Alert when indicators suggest a potential price reversal",
            priority=AlertPriority.MEDIUM,
            instruments=["*"],  # All instruments
            timeframes=["*"],   # All timeframes
            min_confidence=0.6,
            indicators=["RSI", "MACD", "Stochastic"],
            notification_channels=[NotificationType.UI, NotificationType.SLACK]
        )
        rule_ids.append(self.alert_manager.add_rule(reversal))
        
        return rule_ids
    
    def generate_alerts_from_signals(self, 
                                   signals: List[AggregatedSignal],
                                   instrument: str,
                                   timeframe: str) -> List[Alert]:
        """
        Process multiple signals and generate alerts
        
        Args:
            signals: List of signals to process
            instrument: The instrument the signals are for
            timeframe: The timeframe the signals are for
            
        Returns:
            List of generated alerts
        """
        all_alerts = []
        
        for signal in signals:
            alerts = self.alert_manager.process_signal(signal, instrument, timeframe)
            all_alerts.extend(alerts)
        
        return all_alerts


# Create global instances for easy import
alert_manager = AlertManager()
alert_generator = AlertGenerator(alert_manager)
