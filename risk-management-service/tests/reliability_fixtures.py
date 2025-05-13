"""
Reliability test fixtures for risk management service.

These fixtures provide mock objects and data for testing the reliability
of the risk management service.
"""
import pytest
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
import uuid

from fastapi.testclient import TestClient
from core.main_1 import app
from risk_management_service.error import (
    DataValidationError,
    DataFetchError,
    ModelError,
    ServiceUnavailableError,
    RiskCalculationError,
    RiskParameterError,
    RiskLimitBreachError
)


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_account_id():
    """Generate a mock account ID."""
    return f"test-account-{uuid.uuid4()}"


@pytest.fixture
def mock_strategy_id():
    """Generate a mock strategy ID."""
    return f"test-strategy-{uuid.uuid4()}"


@pytest.fixture
def mock_risk_metrics():
    """Generate mock risk metrics data."""
    return {
        "drawdown": random.uniform(0.01, 0.15),
        "sharpe_ratio": random.uniform(0.5, 2.5),
        "volatility": random.uniform(0.01, 0.1),
        "var_95": random.uniform(0.02, 0.1),
        "max_leverage": random.uniform(1.0, 10.0),
        "exposure_ratio": random.uniform(0.1, 0.8),
        "win_rate": random.uniform(0.4, 0.7),
        "profit_factor": random.uniform(1.1, 2.0),
        "recovery_factor": random.uniform(1.5, 5.0),
        "expected_shortfall": random.uniform(0.03, 0.12),
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def mock_risk_thresholds():
    """Generate mock risk threshold data."""
    return {
        "drawdown": 0.2,
        "sharpe_ratio": 0.5,
        "volatility": 0.15,
        "var_95": 0.12,
        "max_leverage": 15.0,
        "exposure_ratio": 0.9,
        "win_rate": 0.3,
        "profit_factor": 1.0,
        "recovery_factor": 1.0,
        "expected_shortfall": 0.15
    }


@pytest.fixture
def mock_historical_performance():
    """Generate mock historical performance data."""
    now = datetime.now()
    return [
        {
            "timestamp": (now - timedelta(days=i)).isoformat(),
            "pnl": random.uniform(-100, 200),
            "drawdown": random.uniform(0, 0.15),
            "win_rate": random.uniform(0.4, 0.7),
            "trade_count": random.randint(5, 50)
        }
        for i in range(30)
    ]


@pytest.fixture
def mock_market_regimes_history():
    """Generate mock market regimes history data."""
    regimes = ["trending_bullish", "trending_bearish", "ranging_narrow", "ranging_wide", "volatile"]
    now = datetime.now()
    return [
        {
            "timestamp": (now - timedelta(days=i)).isoformat(),
            "symbol": "EUR/USD",
            "regime": random.choice(regimes),
            "confidence": random.uniform(0.6, 0.95)
        }
        for i in range(30)
    ]


@pytest.fixture
def mock_ml_predictions():
    """Generate mock ML prediction data."""
    return [
        {
            "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
            "symbol": "EUR/USD",
            "predicted_risk_level": random.uniform(0.1, 0.9),
            "confidence": random.uniform(0.6, 0.95),
            "features": {
                "volatility": random.uniform(0.01, 0.1),
                "trend_strength": random.uniform(0.1, 0.9),
                "market_regime": random.choice(["trending", "ranging", "volatile"])
            }
        }
        for i in range(10)
    ]


@pytest.fixture
def mock_actual_outcomes():
    """Generate mock actual outcome data."""
    return [
        {
            "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
            "symbol": "EUR/USD",
            "actual_risk_level": random.uniform(0.1, 0.9),
            "realized_loss": random.uniform(0, 500) if random.random() > 0.7 else 0,
            "max_drawdown": random.uniform(0.01, 0.15)
        }
        for i in range(10)
    ]


@pytest.fixture
def mock_alert_data():
    """Generate mock alert data."""
    alert_types = ["drawdown_breach", "leverage_breach", "volatility_spike", "correlation_shift"]
    severities = ["low", "medium", "high", "critical"]
    
    return [
        {
            "alert_type": random.choice(alert_types),
            "severity": random.choice(severities),
            "timestamp": datetime.now().isoformat(),
            "metric": random.choice(["drawdown", "leverage", "volatility", "correlation"]),
            "current_value": random.uniform(0.1, 0.3),
            "threshold_value": random.uniform(0.05, 0.2),
            "symbol": "EUR/USD"
        }
        for _ in range(random.randint(1, 3))
    ]


@pytest.fixture
def mock_risk_service():
    """Create a mock risk service that raises different exceptions."""
    class MockRiskService:
        def __init__(self):
            self.error_mode = None
            self.call_count = 0
        
        def set_error_mode(self, mode: str):
            """Set the error mode for the service."""
            self.error_mode = mode
        
        async def check_risk(self, *args, **kwargs):
            """Mock risk check method that can raise exceptions."""
            self.call_count += 1
            
            if self.error_mode == "validation":
                raise DataValidationError("Invalid risk parameters")
            elif self.error_mode == "calculation":
                raise RiskCalculationError("Failed to calculate risk metrics")
            elif self.error_mode == "limit_breach":
                raise RiskLimitBreachError("Risk limit breached", "drawdown", 0.25, 0.2)
            elif self.error_mode == "service_unavailable":
                raise ServiceUnavailableError("Dependency service unavailable")
            elif self.error_mode == "data_fetch":
                raise DataFetchError("Failed to fetch required data")
            elif self.error_mode == "model":
                raise ModelError("Risk model error")
            
            # Default success response
            return {
                "approved": True,
                "risk_score": random.uniform(0.1, 0.9),
                "max_position_size": random.uniform(1000, 10000),
                "timestamp": datetime.now().isoformat()
            }
    
    return MockRiskService()


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker for testing."""
    class MockCircuitBreaker:
    """
    MockCircuitBreaker class.
    
    Attributes:
        Add attributes here
    """

        def __init__(self):
    """
      init  .
    
    """

            self.state = "closed"  # closed, open, half-open
            self.failure_count = 0
            self.last_failure_time = None
            self.reset_timeout = 60  # seconds
        
        def record_failure(self):
            """Record a failure and potentially open the circuit."""
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= 3:
                self.state = "open"
        
        def record_success(self):
            """Record a success and potentially close the circuit."""
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
        
        def allow_request(self):
            """Check if a request should be allowed through."""
            if self.state == "closed":
                return True
            
            if self.state == "open":
                # Check if timeout has elapsed
                if self.last_failure_time and \
                   (datetime.now() - self.last_failure_time).total_seconds() > self.reset_timeout:
                    self.state = "half-open"
                    return True
                return False
            
            # Half-open state allows one test request
            return True
    
    return MockCircuitBreaker()
