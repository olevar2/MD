"""
Tests for the Analysis API endpoints.

This module provides comprehensive tests for the Analysis API endpoints using FastAPI's TestClient.
"""

import pytest
import json
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import patch, MagicMock

from analysis_engine.main import app
from analysis_engine.core.errors import AnalysisError
from analysis_engine.services.analysis_service import AnalysisService

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create 100 data points
    dates = [(datetime.now() - timedelta(hours=i)).isoformat() for i in range(100, 0, -1)]
    
    # Create a price series
    close_prices = [1.1000]
    for i in range(1, 100):
        # Add trend
        trend = 0.0001 * i
        # Add some noise
        noise = np.random.normal(0, 0.0002)
        close_prices.append(close_prices[-1] + trend + noise)
    
    # Create high and low prices
    high_prices = [price + np.random.uniform(0.0005, 0.0015) for price in close_prices]
    low_prices = [price - np.random.uniform(0.0005, 0.0015) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.0003) for prev_close in close_prices[:-1]]
    open_prices.insert(0, close_prices[0] - 0.0005)
    
    # Create volume data
    volumes = [int(1000 * (1 + np.random.uniform(-0.3, 0.3))) for _ in range(100)]
    
    # Convert to dictionary format expected by the API
    market_data = {
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }
    
    return market_data

@pytest.fixture
def sample_analysis_request(sample_market_data):
    """Create a sample analysis request for testing."""
    return {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": sample_market_data,
        "analysis_types": ["confluence", "market_regime", "multi_timeframe"]
    }

def test_analyze_endpoint_success(client, sample_analysis_request):
    """Test the /analyze endpoint with valid data."""
    # Mock the analysis service to return a successful result
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_result = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "timestamp": datetime.now().isoformat(),
            "analysis_results": {
                "confluence": {
                    "confluence_zones": [
                        {
                            "price_level": 1.1050,
                            "strength": 0.8,
                            "direction": "resistance"
                        }
                    ],
                    "market_regime": "TRENDING"
                },
                "market_regime": {
                    "regime": "TRENDING",
                    "direction": "BULLISH",
                    "strength": 0.75
                }
            },
            "metadata": {
                "analysis_duration_ms": 150,
                "version": "1.0.0"
            }
        }
        mock_service.analyze.return_value = mock_result
        
        response = client.post("/api/v1/analyze", json=sample_analysis_request)
        
        assert response.status_code == 200
        assert response.json() == mock_result
        mock_service.analyze.assert_called_once()

def test_analyze_endpoint_validation_error(client):
    """Test the /analyze endpoint with invalid data."""
    # Missing required fields
    invalid_request = {
        "symbol": "EURUSD",
        # Missing timeframe
        "market_data": {}
    }
    
    response = client.post("/api/v1/analyze", json=invalid_request)
    
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()
    assert "timeframe" in str(response.json())

def test_analyze_endpoint_analysis_error(client, sample_analysis_request):
    """Test the /analyze endpoint when an analysis error occurs."""
    # Mock the analysis service to raise an AnalysisError
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_service.analyze.side_effect = AnalysisError(
            message="Analysis failed",
            details={"reason": "Insufficient data"}
        )
        
        response = client.post("/api/v1/analyze", json=sample_analysis_request)
        
        assert response.status_code == 400  # Bad Request
        assert "error_type" in response.json()
        assert response.json()["error_type"] == "AnalysisError"
        assert "message" in response.json()
        assert "details" in response.json()
        assert response.json()["details"]["reason"] == "Insufficient data"

def test_analyze_endpoint_internal_error(client, sample_analysis_request):
    """Test the /analyze endpoint when an unexpected error occurs."""
    # Mock the analysis service to raise an unexpected exception
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_service.analyze.side_effect = Exception("Unexpected error")
        
        response = client.post("/api/v1/analyze", json=sample_analysis_request)
        
        assert response.status_code == 500  # Internal Server Error
        assert "error_type" in response.json()
        assert "message" in response.json()

def test_get_available_analyzers_endpoint(client):
    """Test the /analyzers endpoint."""
    # Mock the analysis service to return a list of available analyzers
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_service.get_available_analyzers.return_value = [
            {
                "name": "confluence_analyzer",
                "description": "Identifies confluence zones where multiple technical factors align",
                "parameters": {
                    "min_tools_for_confluence": 2,
                    "effectiveness_threshold": 0.5
                }
            },
            {
                "name": "market_regime_analyzer",
                "description": "Identifies market regimes (trending, ranging, volatile)",
                "parameters": {
                    "atr_period": 14,
                    "adx_period": 14
                }
            }
        ]
        
        response = client.get("/api/v1/analyzers")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 2
        assert response.json()[0]["name"] == "confluence_analyzer"
        assert response.json()[1]["name"] == "market_regime_analyzer"

def test_get_analyzer_details_endpoint(client):
    """Test the /analyzers/{analyzer_name} endpoint."""
    # Mock the analysis service to return analyzer details
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_service.get_analyzer_details.return_value = {
            "name": "confluence_analyzer",
            "description": "Identifies confluence zones where multiple technical factors align",
            "parameters": {
                "min_tools_for_confluence": 2,
                "effectiveness_threshold": 0.5,
                "sr_proximity_threshold": 0.0015,
                "zone_width_pips": 20
            },
            "input_format": {
                "symbol": "string",
                "timeframe": "string",
                "market_data": "object"
            },
            "output_format": {
                "confluence_zones": "array",
                "market_regime": "string"
            }
        }
        
        response = client.get("/api/v1/analyzers/confluence_analyzer")
        
        assert response.status_code == 200
        assert response.json()["name"] == "confluence_analyzer"
        assert "description" in response.json()
        assert "parameters" in response.json()
        assert "input_format" in response.json()
        assert "output_format" in response.json()

def test_get_analyzer_details_not_found(client):
    """Test the /analyzers/{analyzer_name} endpoint with a non-existent analyzer."""
    # Mock the analysis service to return None for a non-existent analyzer
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_service.get_analyzer_details.return_value = None
        
        response = client.get("/api/v1/analyzers/non_existent_analyzer")
        
        assert response.status_code == 404  # Not Found
        assert "error_type" in response.json()
        assert "message" in response.json()
        assert "not found" in response.json()["message"].lower()

def test_health_check_endpoint(client):
    """Test the /health endpoint."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"
    assert "version" in response.json()
    assert "uptime" in response.json()

def test_analyze_with_custom_parameters(client, sample_analysis_request):
    """Test the /analyze endpoint with custom analyzer parameters."""
    # Add custom parameters to the request
    sample_analysis_request["analyzer_parameters"] = {
        "confluence_analyzer": {
            "min_tools_for_confluence": 3,
            "effectiveness_threshold": 0.7
        },
        "market_regime_analyzer": {
            "atr_period": 10,
            "adx_period": 10
        }
    }
    
    # Mock the analysis service to return a successful result
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_result = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "timestamp": datetime.now().isoformat(),
            "analysis_results": {
                "confluence": {
                    "confluence_zones": [
                        {
                            "price_level": 1.1050,
                            "strength": 0.8,
                            "direction": "resistance"
                        }
                    ],
                    "market_regime": "TRENDING"
                },
                "market_regime": {
                    "regime": "TRENDING",
                    "direction": "BULLISH",
                    "strength": 0.75
                }
            },
            "metadata": {
                "analysis_duration_ms": 150,
                "version": "1.0.0",
                "custom_parameters": True
            }
        }
        mock_service.analyze.return_value = mock_result
        
        response = client.post("/api/v1/analyze", json=sample_analysis_request)
        
        assert response.status_code == 200
        assert response.json() == mock_result
        # Verify that the custom parameters were passed to the analysis service
        call_args = mock_service.analyze.call_args[0][0]
        assert "analyzer_parameters" in call_args
        assert call_args["analyzer_parameters"]["confluence_analyzer"]["min_tools_for_confluence"] == 3

def test_analyze_with_invalid_parameters(client, sample_analysis_request):
    """Test the /analyze endpoint with invalid analyzer parameters."""
    # Add invalid parameters to the request
    sample_analysis_request["analyzer_parameters"] = {
        "confluence_analyzer": {
            "min_tools_for_confluence": "invalid"  # Should be an integer
        }
    }
    
    # Mock the analysis service to raise a validation error
    with patch("analysis_engine.api.routes.analysis_service") as mock_service:
        mock_service.analyze.side_effect = AnalysisError(
            message="Invalid analyzer parameters",
            details={"parameter": "min_tools_for_confluence", "reason": "Must be an integer"}
        )
        
        response = client.post("/api/v1/analyze", json=sample_analysis_request)
        
        assert response.status_code == 400  # Bad Request
        assert "error_type" in response.json()
        assert "message" in response.json()
        assert "details" in response.json()
        assert "parameter" in response.json()["details"]
