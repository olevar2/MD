"""
Alerts and Notifications System for Forex Trading Platform

This module provides a comprehensive alerts and notification system
for defining alert rules based on indicator signals and sending notifications
through multiple channels.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
import json
from datetime import datetime
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertRule:
    """
    Represents a rule for generating alerts based on indicator signals
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 indicator_id: str, 
                 condition: str, 
                 threshold_value: float = None,
                 timeframe: str = "1h",
                 enabled: bool = True,
                 severity: str = "info"):
        """
        Initialize an alert rule
        
        Args:
            name: Name of the alert rule
            description: Description of what the alert detects
            indicator_id: ID of the indicator to monitor
            condition: Condition to check (e.g. "above", "below", "crosses_above", "crosses_below")
            threshold_value: Value to compare against (optional)
            timeframe: Timeframe the rule applies to
            enabled: Whether the rule is active
            severity: Alert severity (info, warning, critical)
        """
        self.name = name
        self.description = description
        self.indicator_id = indicator_id
        self.condition = condition
        self.threshold_value = threshold_value
        self.timeframe = timeframe
        self.enabled = enabled
        self.severity = severity
        self.last_triggered = None
        self.cooldown_minutes = 60  # Default 1 hour cooldown
        self.id = f"{indicator_id}_{condition}_{str(threshold_value).replace('.', '_')}_{timeframe}"
    
    def check_condition(self, indicator_value: float, previous_value: float = None) -> bool:
        """
        Check if the condition is met for the given indicator value
        
        Args:
            indicator_value: Current value of the indicator
            previous_value: Previous value of the indicator (for cross conditions)
            
        Returns:
            True if condition is met, False otherwise
        """
        if not self.enabled:
            return False
            
        # Apply cooldown if recently triggered
        if (self.last_triggered and 
            (datetime.now() - self.last_triggered).total_seconds() < self.cooldown_minutes * 60):
            return False
            
        if self.condition == "above" and self.threshold_value is not None:
            return indicator_value > self.threshold_value
        elif self.condition == "below" and self.threshold_value is not None:
            return indicator_value < self.threshold_value
        elif self.condition == "equal" and self.threshold_value is not None:
            return abs(indicator_value - self.threshold_value) < 0.00001
        elif self.condition == "crosses_above" and previous_value is not None and self.threshold_value is not None:
            return indicator_value > self.threshold_value and previous_value <= self.threshold_value
        elif self.condition == "crosses_below" and previous_value is not None and self.threshold_value is not None:
            return indicator_value < self.threshold_value and previous_value >= self.threshold_value
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert rule to a dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "indicator_id": self.indicator_id,
            "condition": self.condition,
            "threshold_value": self.threshold_value,
            "timeframe": self.timeframe,
            "enabled": self.enabled,
            "severity": self.severity,
            "cooldown_minutes": self.cooldown_minutes,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """Create an alert rule from a dictionary"""
        rule = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            indicator_id=data.get("indicator_id", ""),
            condition=data.get("condition", ""),
            threshold_value=data.get("threshold_value"),
            timeframe=data.get("timeframe", "1h"),
            enabled=data.get("enabled", True),
            severity=data.get("severity", "info")
        )
        rule.cooldown_minutes = data.get("cooldown_minutes", 60)
        if data.get("last_triggered"):
            rule.last_triggered = datetime.fromisoformat(data["last_triggered"])
        return rule


class AlertNotification:
    """Represents a notification generated from an alert"""
    
    def __init__(self, 
                 rule: AlertRule, 
                 indicator_value: float,
                 timestamp: datetime = None,
                 symbol: str = None,
                 message: str = None):
        """
        Initialize an alert notification
        
        Args:
            rule: The rule that triggered the alert
            indicator_value: Value of the indicator that triggered the alert
            timestamp: When the alert was triggered
            symbol: The trading symbol related to the alert
            message: Custom message for the alert
        """
        self.rule = rule
        self.indicator_value = indicator_value
        self.timestamp = timestamp or datetime.now()
        self.symbol = symbol
        self.rule.last_triggered = self.timestamp
        
        # Generate default message if none provided
        if not message:
            self.message = (f"Alert: {rule.name} - {rule.description} - "
                          f"Indicator {rule.indicator_id} {rule.condition} "
                          f"{rule.threshold_value if rule.threshold_value is not None else ''} "
                          f"Current value: {indicator_value}")
        else:
            self.message = message
            
        self.id = f"{rule.id}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the notification to a dictionary for serialization"""
        return {
            "id": self.id,
            "rule": self.rule.to_dict() if self.rule else None,
            "indicator_value": self.indicator_value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "message": self.message,
            "severity": self.rule.severity if self.rule else "info"
        }


class NotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize the notification channel
        
        Args:
            name: Name of the channel
            enabled: Whether the channel is enabled
        """
        self.name = name
        self.enabled = enabled
    
    async def send_notification(self, notification: AlertNotification) -> bool:
        """
        Send a notification through this channel
        
        Args:
            notification: The notification to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        # Base class doesn't implement sending
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the channel to a dictionary for serialization"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "enabled": self.enabled
        }


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 username: str, password: str, 
                 from_address: str, to_addresses: List[str],
                 name: str = "Email", enabled: bool = True):
        """
        Initialize the email notification channel
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_address: Email sender address
            to_addresses: List of recipient email addresses
            name: Name of the channel
            enabled: Whether the channel is enabled
        """
        super().__init__(name, enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses
    
    async def send_notification(self, notification: AlertNotification) -> bool:
        """Send an email notification"""
        if not self.enabled:
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ", ".join(self.to_addresses)
            msg['Subject'] = f"Trading Alert: {notification.rule.name} [{notification.rule.severity.upper()}]"
            
            body = f"""
            <html>
                <body>
                    <h2>Trading Alert: {notification.rule.name}</h2>
                    <p><strong>Time:</strong> {notification.timestamp}</p>
                    <p><strong>Severity:</strong> {notification.rule.severity}</p>
                    <p><strong>Symbol:</strong> {notification.symbol or 'N/A'}</p>
                    <p><strong>Message:</strong> {notification.message}</p>
                    <p><strong>Indicator Value:</strong> {notification.indicator_value}</p>
                    <p><strong>Condition:</strong> {notification.rule.condition} {notification.rule.threshold_value}</p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # This would run in a separate thread/task in production code
            # to avoid blocking the async loop
            def send_mail():
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                server.quit()
            
            # For now, just use a simple thread
            import threading
            thread = threading.Thread(target=send_mail)
            thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the channel to a dictionary for serialization"""
        data = super().to_dict()
        data.update({
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "username": self.username,
            # Password is masked for security
            "from_address": self.from_address,
            "to_addresses": self.to_addresses
        })
        return data


class InAppNotificationChannel(NotificationChannel):
    """In-app notification channel"""
    
    def __init__(self, name: str = "In-App", enabled: bool = True):
        """Initialize the in-app notification channel"""
        super().__init__(name, enabled)
        self.notifications_queue = []
        self.callbacks = []
    
    def register_callback(self, callback: Callable[[AlertNotification], None]):
        """Register a callback to be called when a notification is sent"""
        self.callbacks.append(callback)
    
    async def send_notification(self, notification: AlertNotification) -> bool:
        """Send an in-app notification"""
        if not self.enabled:
            return False
            
        # Add to queue
        self.notifications_queue.append(notification)
        
        # Call all registered callbacks
        for callback in self.callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Error in in-app notification callback: {str(e)}")
        
        return True
    
    def get_pending_notifications(self) -> List[Dict[str, Any]]:
        """Get pending notifications and clear the queue"""
        notifications = [n.to_dict() for n in self.notifications_queue]
        self.notifications_queue = []
        return notifications


class PushNotificationChannel(NotificationChannel):
    """Push notification channel (mobile)"""
    
    def __init__(self, api_key: str, name: str = "Push", enabled: bool = True):
        """
        Initialize the push notification channel
        
        Args:
            api_key: API key for push notification service
            name: Name of the channel
            enabled: Whether the channel is enabled
        """
        super().__init__(name, enabled)
        self.api_key = api_key
    
    async def send_notification(self, notification: AlertNotification) -> bool:
        """Send a push notification"""
        if not self.enabled:
            return False
            
        # In a real implementation, this would make an API call to a push
        # notification service like Firebase Cloud Messaging
        logger.info(f"Would send push notification: {notification.message}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the channel to a dictionary for serialization"""
        data = super().to_dict()
        # API key is masked for security
        data["api_key_configured"] = bool(self.api_key)
        return data


class AlertSystem:
    """
    The main alert system that manages rules and notifications
    """
    
    def __init__(self):
        """Initialize the alert system"""
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.notification_history: List[AlertNotification] = []
        self.previous_values: Dict[str, Dict[str, float]] = {}  # {indicator_id: {timeframe: value}}
    
    def add_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.rules[rule.id] = rule
        logger.info(f"Rule added: {rule.name} ({rule.id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Rule removed: {rule_id}")
            return True
        return False
    
    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel"""
        self.channels[channel.name] = channel
        logger.info(f"Channel added: {channel.name}")
    
    def remove_channel(self, channel_name: str) -> bool:
        """Remove a notification channel"""
        if channel_name in self.channels:
            del self.channels[channel_name]
            logger.info(f"Channel removed: {channel_name}")
            return True
        return False
    
    async def process_indicator_update(self, indicator_id: str, timeframe: str, 
                                      value: float, symbol: str = None) -> List[AlertNotification]:
        """
        Process an indicator update and check if any alerts should be triggered
        
        Args:
            indicator_id: ID of the updated indicator
            timeframe: Timeframe of the update
            value: New value of the indicator
            symbol: Symbol the indicator applies to
            
        Returns:
            List of triggered notifications
        """
        # Get previous value
        previous_value = None
        if indicator_id in self.previous_values and timeframe in self.previous_values[indicator_id]:
            previous_value = self.previous_values[indicator_id][timeframe]
        
        # Update stored value
        if indicator_id not in self.previous_values:
            self.previous_values[indicator_id] = {}
        self.previous_values[indicator_id][timeframe] = value
        
        # Check rules
        triggered_notifications = []
        for rule_id, rule in self.rules.items():
            if rule.indicator_id == indicator_id and rule.timeframe == timeframe:
                if rule.check_condition(value, previous_value):
                    # Create notification
                    notification = AlertNotification(rule, value, symbol=symbol)
                    self.notification_history.append(notification)
                    triggered_notifications.append(notification)
                    logger.info(f"Alert triggered: {rule.name}")
        
        # Send notifications
        await self._send_notifications(triggered_notifications)
        
        return triggered_notifications
    
    async def _send_notifications(self, notifications: List[AlertNotification]):
        """Send notifications through all enabled channels"""
        for notification in notifications:
            for channel_name, channel in self.channels.items():
                if channel.enabled:
                    logger.debug(f"Sending notification via {channel_name}")
                    await channel.send_notification(notification)
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Get all active alert rules"""
        return [rule.to_dict() for rule_id, rule in self.rules.items() if rule.enabled]
    
    def get_notification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get notification history"""
        return [n.to_dict() for n in self.notification_history[-limit:]]
    
    def save_rules_to_file(self, filepath: str):
        """Save rules to a JSON file"""
        rules_data = [rule.to_dict() for rule in self.rules.values()]
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def load_rules_from_file(self, filepath: str):
        """Load rules from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                rules_data = json.load(f)
                
            for rule_data in rules_data:
                rule = AlertRule.from_dict(rule_data)
                self.add_rule(rule)
                
            logger.info(f"Loaded {len(rules_data)} rules")
            return True
        except Exception as e:
            logger.error(f"Failed to load rules: {str(e)}")
            return False


class SmartAlertFactory:
    """
    Factory for creating smart alerts based on indicator concordance
    """
    
    @staticmethod
    def create_concordance_alert(name: str, indicators: List[str], 
                                condition: str, timeframe: str = "1h") -> AlertRule:
        """
        Create a smart alert based on multiple indicators
        
        Args:
            name: Name for the alert
            indicators: List of indicator IDs
            condition: The condition type ("all_above", "all_below", "majority_above", etc.)
            timeframe: Timeframe for the alert
            
        Returns:
            AlertRule configured for the concordance condition
        """
        # In a real implementation, this would create a special type of rule
        # that checks multiple indicators. For simplicity, we'll return a 
        # placeholder rule that would represent this functionality
        rule = AlertRule(
            name=name,
            description=f"Smart Alert based on concordance of {len(indicators)} indicators",
            indicator_id=f"concordance_{'-'.join(indicators)}",
            condition=condition,
            timeframe=timeframe
        )
        return rule
"""
