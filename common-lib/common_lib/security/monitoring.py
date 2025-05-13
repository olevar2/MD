"""
Security Monitoring Module for Forex Trading Platform

This module provides comprehensive security monitoring capabilities for the platform,
including security event logging, threat detection, and security analytics.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import ipaddress
import re
import threading
from collections import defaultdict, deque

from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class SecurityEventSeverity:
    """Security event severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecurityEventCategory:
    """Security event categories"""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_ACCESS = "DATA_ACCESS"
    CONFIGURATION = "CONFIGURATION"
    SYSTEM = "SYSTEM"
    API = "API"
    NETWORK = "NETWORK"
    USER_MANAGEMENT = "USER_MANAGEMENT"
    THREAT = "THREAT"


class EnhancedSecurityEvent(BaseModel):
    """Enhanced model for security-related events for auditing and monitoring"""
    timestamp: datetime
    event_id: str
    event_type: str
    category: str
    severity: str
    user_id: str
    resource: str
    action: str
    status: str
    service: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


class SecurityThreshold(BaseModel):
    """Security threshold configuration"""
    name: str
    description: str
    threshold: int
    time_window_seconds: int
    severity: str = SecurityEventSeverity.WARNING
    actions: List[str] = Field(default_factory=list)


class SecurityMonitor:
    """
    Security monitoring service for the forex trading platform.

    This class provides security event logging, threat detection, and security analytics.
    """

    def __init__(
        self,
        service_name: str,
        elk_endpoint: Optional[str] = None,
        thresholds: Optional[List[SecurityThreshold]] = None
    ):
        """
        Initialize the security monitor.

        Args:
            service_name: Name of the service
            elk_endpoint: Endpoint for ELK stack (Elasticsearch, Logstash, Kibana)
            thresholds: List of security thresholds
        """
        self.service_name = service_name
        self.elk_endpoint = elk_endpoint
        self.thresholds = thresholds or []

        # Event counters for threshold monitoring
        self.event_counters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # IP blacklist
        self.ip_blacklist: Set[str] = set()

        # User blacklist
        self.user_blacklist: Set[str] = set()

        # Suspicious activity tracking
        self.suspicious_activity: Dict[str, Dict[str, Any]] = {}

        # Lock for thread safety
        self.lock = threading.RLock()

        # Alert callbacks
        self.alert_callbacks: List[Callable[[EnhancedSecurityEvent], None]] = []

        logger.info(f"Security monitor initialized for service: {service_name}")

    def log_security_event(
        self,
        event_type: str,
        category: str,
        severity: str,
        user_id: str,
        resource: str,
        action: str,
        status: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        event_id: Optional[str] = None
    ) -> EnhancedSecurityEvent:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            category: Category of security event
            severity: Severity of security event
            user_id: ID of the user involved
            resource: Resource being accessed
            action: Action being performed
            status: Status of the event
            ip_address: Optional client IP address
            user_agent: Optional client user agent
            session_id: Optional session ID
            correlation_id: Optional correlation ID
            details: Optional additional details
            tags: Optional tags for the event
            event_id: Optional event ID (generated if not provided)

        Returns:
            EnhancedSecurityEvent object
        """
        # Create security event
        event = EnhancedSecurityEvent(
            timestamp=datetime.now(timezone.utc),
            event_id=event_id or f"{self.service_name}-{time.time()}-{hash(event_type)}",
            event_type=event_type,
            category=category,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=action,
            status=status,
            service=self.service_name,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            correlation_id=correlation_id,
            details=details,
            tags=tags or []
        )

        # Log event
        self._log_event(event)

        # Process event for threat detection
        self._process_event(event)

        return event

    def _log_event(self, event: EnhancedSecurityEvent) -> None:
        """
        Log a security event.

        Args:
            event: Security event to log
        """
        try:
            # Log to local logger
            log_level = self._get_log_level(event.severity)
            logger.log(
                log_level,
                f"Security event: {event.event_type} - {event.status}",
                extra={
                    "security_event": event.dict(),
                    "event_type": event.event_type,
                    "category": event.category,
                    "severity": event.severity,
                    "user_id": event.user_id,
                    "resource": event.resource,
                    "action": event.action,
                    "status": event.status,
                    "service": event.service
                }
            )

            # Send to ELK if configured
            if self.elk_endpoint:
                self._send_to_elk(event)
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")

    def _get_log_level(self, severity: str) -> int:
        """
        Get log level for severity.

        Args:
            severity: Severity level

        Returns:
            Log level
        """
        severity_map = {
            SecurityEventSeverity.INFO: logging.INFO,
            SecurityEventSeverity.WARNING: logging.WARNING,
            SecurityEventSeverity.ERROR: logging.ERROR,
            SecurityEventSeverity.CRITICAL: logging.CRITICAL
        }

        return severity_map.get(severity, logging.INFO)

    def _send_to_elk(self, event: EnhancedSecurityEvent) -> None:
        """
        Send event to ELK stack.

        Args:
            event: Security event to send
        """
        # This is a placeholder for sending events to ELK
        # In a real implementation, this would use the ELK API
        pass

    def _process_event(self, event: EnhancedSecurityEvent) -> None:
        """
        Process event for threat detection.

        Args:
            event: Security event to process
        """
        with self.lock:
            # Update event counters
            self._update_event_counters(event)

            # Check thresholds
            self._check_thresholds(event)

            # Check for suspicious activity
            self._check_suspicious_activity(event)

    def _update_event_counters(self, event: EnhancedSecurityEvent) -> None:
        """
        Update event counters.

        Args:
            event: Security event
        """
        # Add event timestamp to counters
        now = time.time()

        # Update counters for different dimensions
        self.event_counters[f"type:{event.event_type}"].append(now)
        self.event_counters[f"category:{event.category}"].append(now)
        self.event_counters[f"severity:{event.severity}"].append(now)
        self.event_counters[f"user:{event.user_id}"].append(now)
        self.event_counters[f"resource:{event.resource}"].append(now)
        self.event_counters[f"action:{event.action}"].append(now)
        self.event_counters[f"status:{event.status}"].append(now)

        if event.ip_address:
            self.event_counters[f"ip:{event.ip_address}"].append(now)

        # Combined dimensions for more specific tracking
        self.event_counters[f"user:{event.user_id}:type:{event.event_type}"].append(now)
        self.event_counters[f"user:{event.user_id}:status:{event.status}"].append(now)

        if event.ip_address:
            self.event_counters[f"ip:{event.ip_address}:type:{event.event_type}"].append(now)
            self.event_counters[f"ip:{event.ip_address}:status:{event.status}"].append(now)

    def _check_thresholds(self, event: EnhancedSecurityEvent) -> None:
        """
        Check security thresholds.

        Args:
            event: Security event
        """
        now = time.time()

        for threshold in self.thresholds:
            # Get relevant counter
            counter = self.event_counters.get(threshold.name, deque())

            # Filter events within time window
            time_window = now - threshold.time_window_seconds
            events_in_window = [t for t in counter if t >= time_window]

            # Check if threshold is exceeded
            if len(events_in_window) >= threshold.threshold:
                # Create alert event
                alert_event = EnhancedSecurityEvent(
                    timestamp=datetime.now(timezone.utc),
                    event_id=f"threshold-{threshold.name}-{now}",
                    event_type="threshold_exceeded",
                    category=SecurityEventCategory.THREAT,
                    severity=threshold.severity,
                    user_id=event.user_id if "user:" in threshold.name else "system",
                    resource=threshold.name,
                    action="monitor",
                    status="alert",
                    service=self.service_name,
                    ip_address=event.ip_address if "ip:" in threshold.name else None,
                    details={
                        "threshold_name": threshold.name,
                        "threshold_value": threshold.threshold,
                        "current_value": len(events_in_window),
                        "time_window_seconds": threshold.time_window_seconds,
                        "description": threshold.description
                    },
                    tags=["threshold", "security", "alert"]
                )

                # Log alert
                self._log_event(alert_event)

                # Execute actions
                self._execute_threshold_actions(threshold, alert_event)

    def _execute_threshold_actions(self, threshold: SecurityThreshold, event: EnhancedSecurityEvent) -> None:
        """
        Execute threshold actions.

        Args:
            threshold: Security threshold
            event: Alert event
        """
        for action in threshold.actions:
            if action == "blacklist_ip" and event.ip_address:
                self.blacklist_ip(event.ip_address)
            elif action == "blacklist_user" and event.user_id != "system":
                self.blacklist_user(event.user_id)
            elif action == "alert":
                self._trigger_alert(event)

    def _check_suspicious_activity(self, event: EnhancedSecurityEvent) -> None:
        """
        Check for suspicious activity patterns.

        Args:
            event: Security event
        """
        # Check for failed login attempts followed by success
        if event.event_type == "login" and event.status == "success":
            # Check for recent failed login attempts
            user_failed_logins = self.event_counters.get(f"user:{event.user_id}:type:login:status:failed", deque())
            now = time.time()
            recent_failed_logins = [t for t in user_failed_logins if t >= now - 3600]  # Last hour

            if len(recent_failed_logins) >= 3:
                # Suspicious login after multiple failures
                alert_event = EnhancedSecurityEvent(
                    timestamp=datetime.now(timezone.utc),
                    event_id=f"suspicious-login-{event.user_id}-{now}",
                    event_type="suspicious_login",
                    category=SecurityEventCategory.THREAT,
                    severity=SecurityEventSeverity.WARNING,
                    user_id=event.user_id,
                    resource="authentication",
                    action="login",
                    status="suspicious",
                    service=self.service_name,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    session_id=event.session_id,
                    correlation_id=event.correlation_id,
                    details={
                        "failed_attempts": len(recent_failed_logins),
                        "time_window": "1 hour"
                    },
                    tags=["suspicious", "authentication", "brute_force"]
                )

                # Log alert
                self._log_event(alert_event)

                # Trigger alert
                self._trigger_alert(alert_event)

    def _trigger_alert(self, event: EnhancedSecurityEvent) -> None:
        """
        Trigger security alert.

        Args:
            event: Security event
        """
        # Call all registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def register_alert_callback(self, callback: Callable[[EnhancedSecurityEvent], None]) -> None:
        """
        Register alert callback.

        Args:
            callback: Callback function
        """
        self.alert_callbacks.append(callback)

    def blacklist_ip(self, ip_address: str) -> None:
        """
        Add IP address to blacklist.

        Args:
            ip_address: IP address to blacklist
        """
        with self.lock:
            self.ip_blacklist.add(ip_address)
            logger.warning(f"IP address blacklisted: {ip_address}")

    def blacklist_user(self, user_id: str) -> None:
        """
        Add user to blacklist.

        Args:
            user_id: User ID to blacklist
        """
        with self.lock:
            self.user_blacklist.add(user_id)
            logger.warning(f"User blacklisted: {user_id}")

    def is_ip_blacklisted(self, ip_address: str) -> bool:
        """
        Check if IP address is blacklisted.

        Args:
            ip_address: IP address to check

        Returns:
            True if blacklisted, False otherwise
        """
        return ip_address in self.ip_blacklist

    def is_user_blacklisted(self, user_id: str) -> bool:
        """
        Check if user is blacklisted.

        Args:
            user_id: User ID to check

        Returns:
            True if blacklisted, False otherwise
        """
        return user_id in self.user_blacklist
