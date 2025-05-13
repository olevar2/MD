"""
Tests for the reconciliation API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from core.main_1 import app
from services.reconciliation_service import ReconciliationService
from common_lib.data_reconciliation import (
    ReconciliationResult,
    ReconciliationStatus,
    ReconciliationConfig,
    ReconciliationStrategy,
    ReconciliationSeverity,
)

client = TestClient(app)


@pytest.fixture
def mock_reconciliation_result():
    """Create a mock reconciliation result."""
    result = MagicMock(spec=ReconciliationResult)
    result.reconciliation_id = "test-reconciliation-id"
    result.status = ReconciliationStatus.COMPLETED
    result.discrepancy_count = 5
    result.resolution_count = 4
    result.resolution_rate = 80.0
    result.duration_seconds = 2.5
    result.start_time = datetime.utcnow() - timedelta(seconds=5)
    result.end_time = datetime.utcnow()
    return result


@patch.object(ReconciliationService, "reconcile_ohlcv_data")
async def test_reconcile_ohlcv_data(mock_reconcile, mock_reconciliation_result):
    """Test the reconcile_ohlcv_data endpoint."""
    # Configure the mock
    mock_reconcile.return_value = mock_reconciliation_result
    
    # Make the request
    response = client.post(
        "/api/v1/reconciliation/ohlcv",
        json={
            "symbol": "EURUSD",
            "start_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "timeframe": "1h",
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        },
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["reconciliation_id"] == "test-reconciliation-id"
    assert data["status"] == "COMPLETED"
    assert data["discrepancy_count"] == 5
    assert data["resolution_count"] == 4
    assert data["resolution_rate"] == 80.0
    assert data["duration_seconds"] == 2.5
    assert "start_time" in data
    assert "end_time" in data
    
    # Check that the mock was called with the correct arguments
    mock_reconcile.assert_called_once()
    call_args = mock_reconcile.call_args[1]
    assert call_args["symbol"] == "EURUSD"
    assert isinstance(call_args["start_date"], datetime)
    assert isinstance(call_args["end_date"], datetime)
    assert call_args["timeframe"] == "1h"
    assert call_args["strategy"] == ReconciliationStrategy.SOURCE_PRIORITY
    assert call_args["tolerance"] == 0.001
    assert call_args["auto_resolve"] is True
    assert call_args["notification_threshold"] == ReconciliationSeverity.HIGH


@patch.object(ReconciliationService, "reconcile_tick_data")
async def test_reconcile_tick_data(mock_reconcile, mock_reconciliation_result):
    """Test the reconcile_tick_data endpoint."""
    # Configure the mock
    mock_reconcile.return_value = mock_reconciliation_result
    
    # Make the request
    response = client.post(
        "/api/v1/reconciliation/tick-data",
        json={
            "symbol": "EURUSD",
            "start_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "strategy": "MOST_RECENT",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        },
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["reconciliation_id"] == "test-reconciliation-id"
    assert data["status"] == "COMPLETED"
    assert data["discrepancy_count"] == 5
    assert data["resolution_count"] == 4
    assert data["resolution_rate"] == 80.0
    assert data["duration_seconds"] == 2.5
    assert "start_time" in data
    assert "end_time" in data
    
    # Check that the mock was called with the correct arguments
    mock_reconcile.assert_called_once()
    call_args = mock_reconcile.call_args[1]
    assert call_args["symbol"] == "EURUSD"
    assert isinstance(call_args["start_date"], datetime)
    assert isinstance(call_args["end_date"], datetime)
    assert call_args["strategy"] == ReconciliationStrategy.MOST_RECENT
    assert call_args["tolerance"] == 0.001
    assert call_args["auto_resolve"] is True
    assert call_args["notification_threshold"] == ReconciliationSeverity.HIGH


@patch.object(ReconciliationService, "get_reconciliation_status")
async def test_get_reconciliation_status(mock_get_status, mock_reconciliation_result):
    """Test the get_reconciliation_status endpoint."""
    # Configure the mock
    mock_get_status.return_value = mock_reconciliation_result
    
    # Make the request
    response = client.get(
        "/api/v1/reconciliation/test-reconciliation-id",
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["reconciliation_id"] == "test-reconciliation-id"
    assert data["status"] == "COMPLETED"
    assert data["discrepancy_count"] == 5
    assert data["resolution_count"] == 4
    assert data["resolution_rate"] == 80.0
    assert data["duration_seconds"] == 2.5
    assert "start_time" in data
    assert "end_time" in data
    
    # Check that the mock was called with the correct arguments
    mock_get_status.assert_called_once_with("test-reconciliation-id")


@patch.object(ReconciliationService, "get_reconciliation_status")
async def test_get_reconciliation_status_not_found(mock_get_status):
    """Test the get_reconciliation_status endpoint when the reconciliation is not found."""
    # Configure the mock
    mock_get_status.return_value = None
    
    # Make the request
    response = client.get(
        "/api/v1/reconciliation/nonexistent-id",
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Check the response
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"]
    
    # Check that the mock was called with the correct arguments
    mock_get_status.assert_called_once_with("nonexistent-id")
