"""
Unit tests for error handling in the risk management service.
"""
import pytest
import json
from fastapi.testclient import TestClient

from risk_management_service.error import (
    DataValidationError,
    DataFetchError,
    ModelError,
    ServiceUnavailableError,
    RiskCalculationError,
    RiskParameterError,
    RiskLimitBreachError
)


def test_data_validation_error(test_client):
    """Test that data validation errors return 400 status code."""
    # Make a request with invalid data
    response = test_client.post(
        "/api/risk/dynamic/strategy/weaknesses",
        json={
            # Missing required fields
        },
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check response
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_risk_limit_breach_error(test_client, monkeypatch):
    """Test that risk limit breach errors return 403 status code."""
    # Mock the risk adjuster to raise a RiskLimitBreachError
    def mock_monitor_risk_thresholds(*args, **kwargs):
    """
    Mock monitor risk thresholds.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        raise RiskLimitBreachError(
            message="Risk limit breached",
            limit_type="drawdown",
            current_value=0.25,
            limit_value=0.2
        )
    
    # Apply the mock
    from risk_management_service.services.dynamic_risk_adjuster import DynamicRiskAdjuster
    monkeypatch.setattr(
        DynamicRiskAdjuster,
        "monitor_risk_thresholds",
        mock_monitor_risk_thresholds
    )
    
    # Make a request
    response = test_client.post(
        "/api/risk/dynamic/monitor/thresholds",
        json={
            "account_id": "test-account",
            "current_risk_metrics": {
                "drawdown": 0.25,
                "sharpe_ratio": 1.5
            },
            "thresholds": {
                "drawdown": 0.2,
                "sharpe_ratio": 1.0
            }
        },
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check response
    assert response.status_code == 403
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "RISK_LIMIT_BREACH_ERROR"
    assert "drawdown" in data["error"]["details"]["limit_type"]


def test_service_unavailable_error(test_client, monkeypatch):
    """Test that service unavailable errors return 503 status code."""
    # Mock the risk adjuster to raise a ServiceUnavailableError
    def mock_process_ml_risk_feedback(*args, **kwargs):
    """
    Mock process ml risk feedback.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        raise ServiceUnavailableError("ML service is currently unavailable")
    
    # Apply the mock
    from risk_management_service.services.dynamic_risk_adjuster import DynamicRiskAdjuster
    monkeypatch.setattr(
        DynamicRiskAdjuster,
        "process_ml_risk_feedback",
        mock_process_ml_risk_feedback
    )
    
    # Make a request
    response = test_client.post(
        "/api/risk/dynamic/ml/feedback",
        json={
            "ml_predictions": [{"prediction": 0.5}],
            "actual_outcomes": [{"outcome": 0.6}]
        },
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check response
    assert response.status_code == 503
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "SERVICE_UNAVAILABLE_ERROR"


def test_model_error(test_client, monkeypatch):
    """Test that model errors return 500 status code."""
    # Mock the risk adjuster to raise a ModelError
    def mock_analyze_strategy_weaknesses(*args, **kwargs):
    """
    Mock analyze strategy weaknesses.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        raise ModelError("Failed to load risk model")
    
    # Apply the mock
    from risk_management_service.services.dynamic_risk_adjuster import DynamicRiskAdjuster
    monkeypatch.setattr(
        DynamicRiskAdjuster,
        "analyze_strategy_weaknesses",
        mock_analyze_strategy_weaknesses
    )
    
    # Make a request
    response = test_client.post(
        "/api/risk/dynamic/strategy/weaknesses",
        json={
            "strategy_id": "test-strategy",
            "historical_performance": [{"pnl": 100}],
            "market_regimes_history": [{"regime": "trending"}]
        },
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check response
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "MODEL_ERROR"


def test_correlation_id_propagation(test_client):
    """Test that correlation IDs are properly propagated."""
    # Make a request with a correlation ID
    correlation_id = "test-correlation-id-12345"
    response = test_client.post(
        "/api/risk/dynamic/strategy/weaknesses",
        json={
            "strategy_id": "test-strategy",
            "historical_performance": [{"pnl": 100}],
            "market_regimes_history": [{"regime": "trending"}]
        },
        headers={
            "X-API-Key": "test-api-key",
            "X-Correlation-ID": correlation_id
        }
    )
    
    # Check that the correlation ID is in the response headers
    assert "X-Correlation-ID" in response.headers
    assert response.headers["X-Correlation-ID"] == correlation_id
    
    # Check that the correlation ID is in the response body
    data = response.json()
    if "error" in data:
        assert data["error"]["correlation_id"] == correlation_id


def test_auto_generated_correlation_id(test_client):
    """Test that correlation IDs are auto-generated if not provided."""
    # Make a request without a correlation ID
    response = test_client.post(
        "/api/risk/dynamic/strategy/weaknesses",
        json={
            "strategy_id": "test-strategy",
            "historical_performance": [{"pnl": 100}],
            "market_regimes_history": [{"regime": "trending"}]
        },
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check that a correlation ID was generated
    assert "X-Correlation-ID" in response.headers
    assert response.headers["X-Correlation-ID"] is not None
    assert len(response.headers["X-Correlation-ID"]) > 0
