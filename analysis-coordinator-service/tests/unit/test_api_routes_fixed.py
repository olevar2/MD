import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta, UTC
import uuid
import json

from fastapi.testclient import TestClient
from fastapi import FastAPI
from analysis_coordinator_service.api.v1.coordinator import router as coordinator_router
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisServiceType,
    IntegratedAnalysisResponse,
    AnalysisTaskStatus
)

# Create a test app
app = FastAPI()
app.include_router(coordinator_router)

# Create a test client
client = TestClient(app)

@pytest.fixture
def mock_coordinator_service():
    """
    Create a mock for CoordinatorService.
    """
    service = AsyncMock()
    
    # Generate a task ID to use consistently
    task_id = str(uuid.uuid4())
    
    # Mock run_integrated_analysis method
    service.run_integrated_analysis.return_value = IntegratedAnalysisResponse(
        task_id=task_id,
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC)
    )
    
    # Mock create_analysis_task method
    service.create_analysis_task.return_value = AnalysisTaskResponse(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC)
    )
    
    # Mock get_task_result method
    service.get_task_result.return_value = AnalysisTaskResult(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )
    
    # Mock get_task_status method
    service.get_task_status.return_value = AnalysisTaskStatus(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.RUNNING,
        created_at=datetime.now(UTC),
        progress=0.5,
        message="Processing data"
    )
    
    # Mock delete_task method
    service.delete_task.return_value = True
    
    # Mock cancel_task method
    service.cancel_task.return_value = True
    
    # Mock list_tasks method
    service.list_tasks.return_value = [
        AnalysisTaskResult(
            task_id=task_id,
            service_type=AnalysisServiceType.MARKET_ANALYSIS,
            status=AnalysisTaskStatusEnum.COMPLETED,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
            error=None
        )
    ]
    
    # Mock get_available_services method
    service.get_available_services.return_value = {
        "market_analysis": [
            "pattern_recognition",
            "support_resistance",
            "market_regime",
            "correlation_analysis"
        ],
        "causal_analysis": [
            "causal_graph",
            "intervention_effect",
            "counterfactual_scenario"
        ],
        "backtesting": [
            "strategy_backtest",
            "performance_analysis",
            "optimization"
        ]
    }
    
    return service

# Mock the dependency injection functions
@pytest.fixture(autouse=True)
def mock_dependencies(mock_coordinator_service):
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        yield

def test_run_integrated_analysis(mock_coordinator_service):
    # Arrange
    # Act
    response = client.post(
        "/coordinator/integrated-analysis",
        json={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "start_date": datetime.now(UTC).isoformat(),
            "end_date": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "services": ["market_analysis", "causal_analysis"],
            "parameters": {
                "market_analysis": {"patterns": ["head_and_shoulders"]},
                "causal_analysis": {"variables": ["price", "volume"]}
            }
        }
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == AnalysisTaskStatusEnum.PENDING

def test_create_analysis_task(mock_coordinator_service):
    # Arrange
    # Act
    response = client.post(
        "/coordinator/tasks",
        json={
            "service_type": "market_analysis",
            "symbol": "EURUSD",
            "timeframe": "1h",
            "start_date": datetime.now(UTC).isoformat(),
            "end_date": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "parameters": {"patterns": ["head_and_shoulders"]}
        }
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["service_type"] == AnalysisServiceType.MARKET_ANALYSIS
    assert data["status"] == AnalysisTaskStatusEnum.PENDING

def test_get_task_result(mock_coordinator_service):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_result.return_value.task_id
    
    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.get(f"/coordinator/tasks/{task_id}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == AnalysisTaskStatusEnum.COMPLETED

def test_get_task_result_not_found(mock_coordinator_service):
    # Arrange
    task_id = str(uuid.uuid4())
    # Override the mock for this specific test
    mock_coordinator_service.get_task_result.return_value = None

    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.get(f"/coordinator/tasks/{task_id}")

    # Assert
    assert response.status_code == 404

def test_get_task_status(mock_coordinator_service):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_status.return_value.task_id
    
    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.get(f"/coordinator/tasks/{task_id}/status")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == AnalysisTaskStatusEnum.RUNNING

def test_get_task_status_not_found(mock_coordinator_service):
    # Arrange
    task_id = str(uuid.uuid4())
    # Override the mock for this specific test
    mock_coordinator_service.get_task_status.return_value = None

    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.get(f"/coordinator/tasks/{task_id}/status")

    # Assert
    assert response.status_code == 404

def test_delete_task(mock_coordinator_service):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_result.return_value.task_id
    
    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.delete(f"/coordinator/tasks/{task_id}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_delete_task_not_found(mock_coordinator_service):
    # Arrange
    task_id = str(uuid.uuid4())
    # Override the mock for this specific test
    mock_coordinator_service.delete_task.return_value = False

    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.delete(f"/coordinator/tasks/{task_id}")

    # Assert
    assert response.status_code == 404

def test_list_tasks(mock_coordinator_service):
    # Arrange
    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.get("/coordinator/tasks")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_cancel_task(mock_coordinator_service):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_status.return_value.task_id
    
    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.post(f"/coordinator/tasks/{task_id}/cancel")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_cancel_task_not_found(mock_coordinator_service):
    # Arrange
    task_id = str(uuid.uuid4())
    # Override the mock for this specific test
    mock_coordinator_service.cancel_task.return_value = False

    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.post(f"/coordinator/tasks/{task_id}/cancel")

    # Assert
    assert response.status_code == 404

def test_get_available_services(mock_coordinator_service):
    # Arrange
    # Act
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        response = client.get("/coordinator/available-services")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "market_analysis" in data
    assert "causal_analysis" in data
    assert "backtesting" in data