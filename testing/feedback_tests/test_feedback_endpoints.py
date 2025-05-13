# filepath: d:\MD\forex_trading_platform\testing\feedback_tests\test_feedback_endpoints.py
"""
Tests for the feedback monitoring API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid

# Assume the FastAPI app is created in main.py and includes the feedback router
# Adjust the import path based on your actual application structure
from analysis_engine.main import app # Assuming app is defined in analysis_engine/main.py
from analysis_engine.api import feedback_endpoints # Import to patch internal functions

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the FastAPI app."""
    return TestClient(app)

@pytest.fixture(autouse=True)
def reset_placeholder_store():
    """Clear the placeholder store before each test."""
    feedback_endpoints._RECENT_FEEDBACK_STORE = []
    yield # Run the test
    feedback_endpoints._RECENT_FEEDBACK_STORE = [] # Clear after test

# --- Helper to populate placeholder store --- 
def populate_store(count=5):
    """
    Populate store.
    
    Args:
        count: Description of count
    
    """

    now = datetime.utcnow()
    for i in range(count):
        event = MagicMock()
        event.feedback_id = uuid.uuid4()
        event.feedback_type = f"type_{i % 2}" # Alternate types
        event.source = f"source_{i}"
        event.timestamp = now
        feedback_endpoints.record_feedback_processed(event)

# --- Test Cases --- 

def test_get_feedback_status_empty(client):
    """Test GET /feedback/status when no events have been processed."""
    response = client.get("/feedback/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Operational"
    assert data["consumer_status"] == "Connected" # Default from model
    assert data["last_event_processed_at"] is None

def test_get_feedback_status_with_data(client):
    """Test GET /feedback/status after processing events."""
    populate_store(1)
    response = client.get("/feedback/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Operational"
    assert data["last_event_processed_at"] is not None
    # Check if the timestamp format is correct (ISO format)
    try:
        datetime.fromisoformat(data["last_event_processed_at"].replace('Z', '+00:00'))
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO format")

def test_get_recent_feedback_empty(client):
    """Test GET /feedback/recent when no events exist."""
    response = client.get("/feedback/recent")
    assert response.status_code == 200
    assert response.json() == []

def test_get_recent_feedback_with_data(client):
    """Test GET /feedback/recent returns stored events."""
    populate_store(3)
    response = client.get("/feedback/recent")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # Events are inserted at index 0, so the first item is the last processed
    assert data[0]["source"] == "source_2"
    assert data[2]["source"] == "source_0"
    assert "feedback_id" in data[0]
    assert "feedback_type" in data[0]
    assert "timestamp" in data[0]

def test_get_recent_feedback_limit(client):
    """Test the limit parameter for GET /feedback/recent."""
    populate_store(10)
    response = client.get("/feedback/recent?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert data[0]["source"] == "source_9"

def test_get_feedback_stats_empty(client):
    """Test GET /feedback/stats when no events exist."""
    response = client.get("/feedback/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 0
    assert data["processed_by_type"] == {}

def test_get_feedback_stats_with_data(client):
    """Test GET /feedback/stats calculates correctly."""
    populate_store(7) # type_0: 4 times, type_1: 3 times
    response = client.get("/feedback/stats")
    assert response.status_code == 200
    data = response.json()
    # Note: Placeholder stats are based on the limited _RECENT_FEEDBACK_STORE size
    assert data["total_processed"] == 7 
    assert data["processed_by_type"] == {
        "type_1": 3, # i = 1, 3, 5
        "type_0": 4  # i = 0, 2, 4, 6
    }

# Add tests for error handling if the underlying service (when implemented)
# could raise specific exceptions that should result in 500 errors.
