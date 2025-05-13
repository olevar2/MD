"""
Tests for error handling in Monitoring & Alerting Service.
"""
import pytest
from fastapi import HTTPException

from monitoring_alerting_service.error import (
    MonitoringAlertingError,
    AlertNotFoundError,
    NotificationError,
    AlertStorageError,
    MetricsExporterError,
    DashboardError,
    AlertRuleError,
    ThresholdValidationError,
    convert_to_http_exception
)


def test_monitoring_alerting_error():
    """Test MonitoringAlertingError."""
    error = MonitoringAlertingError(
        message="Test error",
        error_code="TEST_ERROR",
        details={"test": "value"}
    )
    
    assert error.message == "Test error"
    assert error.error_code == "TEST_ERROR"
    assert error.details == {"test": "value"}


def test_alert_not_found_error():
    """Test AlertNotFoundError."""
    error = AlertNotFoundError(
        message="Alert not found",
        alert_id="test-alert"
    )
    
    assert error.message == "Alert not found"
    assert error.error_code == "ALERT_NOT_FOUND_ERROR"
    assert error.details == {"alert_id": "test-alert"}


def test_notification_error():
    """Test NotificationError."""
    error = NotificationError(
        message="Failed to send notification",
        channel="EMAIL",
        alert_id="test-alert"
    )
    
    assert error.message == "Failed to send notification"
    assert error.error_code == "NOTIFICATION_ERROR"
    assert error.details == {"channel": "EMAIL", "alert_id": "test-alert"}


def test_alert_storage_error():
    """Test AlertStorageError."""
    error = AlertStorageError(
        message="Failed to store alert",
        operation="save_alert"
    )
    
    assert error.message == "Failed to store alert"
    assert error.error_code == "ALERT_STORAGE_ERROR"
    assert error.details == {"operation": "save_alert"}


def test_metrics_exporter_error():
    """Test MetricsExporterError."""
    error = MetricsExporterError(
        message="Failed to export metrics",
        exporter="prometheus"
    )
    
    assert error.message == "Failed to export metrics"
    assert error.error_code == "METRICS_EXPORTER_ERROR"
    assert error.details == {"exporter": "prometheus"}


def test_dashboard_error():
    """Test DashboardError."""
    error = DashboardError(
        message="Failed to create dashboard",
        dashboard="performance"
    )
    
    assert error.message == "Failed to create dashboard"
    assert error.error_code == "DASHBOARD_ERROR"
    assert error.details == {"dashboard": "performance"}


def test_alert_rule_error():
    """Test AlertRuleError."""
    error = AlertRuleError(
        message="Invalid alert rule",
        rule="check_indicator_value"
    )
    
    assert error.message == "Invalid alert rule"
    assert error.error_code == "ALERT_RULE_ERROR"
    assert error.details == {"rule": "check_indicator_value"}


def test_threshold_validation_error():
    """Test ThresholdValidationError."""
    error = ThresholdValidationError(
        message="Invalid threshold",
        threshold_type="overbought",
        value=-10.0
    )
    
    assert error.message == "Invalid threshold"
    assert error.error_code == "THRESHOLD_VALIDATION_ERROR"
    assert error.details == {"threshold_type": "overbought", "value": -10.0}


def test_convert_to_http_exception():
    """Test convert_to_http_exception function."""
    # Test AlertNotFoundError (404)
    error = AlertNotFoundError(
        message="Alert not found",
        alert_id="test-alert"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 404
    assert http_exc.detail["error_code"] == "ALERT_NOT_FOUND_ERROR"
    assert http_exc.detail["message"] == "Alert not found"
    
    # Test NotificationError (503)
    error = NotificationError(
        message="Failed to send notification",
        channel="EMAIL",
        alert_id="test-alert"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 503
    assert http_exc.detail["error_code"] == "NOTIFICATION_ERROR"
    assert http_exc.detail["message"] == "Failed to send notification"
    
    # Test AlertRuleError (400)
    error = AlertRuleError(
        message="Invalid alert rule",
        rule="check_indicator_value"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 400
    assert http_exc.detail["error_code"] == "ALERT_RULE_ERROR"
    assert http_exc.detail["message"] == "Invalid alert rule"
    
    # Test generic MonitoringAlertingError (500)
    error = MonitoringAlertingError(
        message="Generic error",
        error_code="GENERIC_ERROR"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 500
    assert http_exc.detail["error_code"] == "GENERIC_ERROR"
    assert http_exc.detail["message"] == "Generic error"
