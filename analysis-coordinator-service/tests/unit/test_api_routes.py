import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta, UTC
import uuid
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from fastapi.testclient import TestClient
from analysis_coordinator_service.main import app
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisTaskStatus,
    AnalysisServiceType,
    IntegratedAnalysisResponse
)
from analysis_coordinator_service.utils.dependency_injection import (
    get_coordinator_service,
    get_command_bus,
    get_query_bus
)
from tests.unit.test_repositories import TestTaskRepository

# Create a test client
client = TestClient(app)

# Mock the dependency injection functions
@pytest.fixture(autouse=True)
def mock_dependencies(mock_coordinator_service, mock_task_repository):
    # Create a task in the repository for testing
    task_id = mock_coordinator_service.get_task_result.return_value.task_id
    
    # Create a task in the repository
    mock_task_repository.tasks[task_id] = mock_coordinator_service.get_task_result.return_value
    mock_task_repository.task_statuses[task_id] = mock_coordinator_service.get_task_status.return_value
    
    # Patch the dependencies
    with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
        with patch('analysis_coordinator_service.services.coordinator_service.TaskRepository', return_value=mock_task_repository):
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

def test_get_task_result(mock_coordinator_service, mock_task_repository):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_result.return_value.task_id
    
    # Act
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
    original_return_value = mock_coordinator_service.get_task_result.return_value
    mock_coordinator_service.get_task_result.return_value = None

    # Act
    response = client.get(f"/coordinator/tasks/{task_id}")

    # Assert
    assert response.status_code == 404
    
    # Restore the original mock
    mock_coordinator_service.get_task_result.return_value = original_return_value

def test_get_task_status(mock_coordinator_service, mock_task_repository):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_status.return_value.task_id
    
    # Act
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
    original_return_value = mock_coordinator_service.get_task_status.return_value
    mock_coordinator_service.get_task_status.return_value = None

    # Act
    response = client.get(f"/coordinator/tasks/{task_id}/status")

    # Assert
    assert response.status_code == 404
    
    # Restore the original mock
    mock_coordinator_service.get_task_status.return_value = original_return_value

def test_delete_task(mock_coordinator_service, mock_task_repository):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_result.return_value.task_id
    
    # Act
    response = client.delete(f"/coordinator/tasks/{task_id}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_delete_task_not_found(mock_coordinator_service):
    # Arrange
    task_id = str(uuid.uuid4())
    # Override the mock for this specific test
    original_return_value = mock_coordinator_service.delete_task.return_value
    mock_coordinator_service.delete_task.return_value = False

    # Act
    response = client.delete(f"/coordinator/tasks/{task_id}")

    # Assert
    assert response.status_code == 404
    
    # Restore the original mock
    mock_coordinator_service.delete_task.return_value = original_return_value

def test_list_tasks(mock_coordinator_service, mock_task_repository):
    # Arrange
    # Create a task in the repository
    task_id = mock_coordinator_service.get_task_result.return_value.task_id
    mock_task_repository.tasks[task_id] = mock_coordinator_service.get_task_result.return_value
    
    # Act
    response = client.get("/coordinator/tasks")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_cancel_task(mock_coordinator_service, mock_task_repository):
    # Arrange
    # Use the same task_id that's set in the mock
    task_id = mock_coordinator_service.get_task_status.return_value.task_id
    
    # Act
    response = client.post(f"/coordinator/tasks/{task_id}/cancel")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_cancel_task_not_found(mock_coordinator_service):
    # Arrange
    task_id = str(uuid.uuid4())
    # Override the mock for this specific test
    original_return_value = mock_coordinator_service.cancel_task.return_value
    mock_coordinator_service.cancel_task.return_value = False

    # Act
    response = client.post(f"/coordinator/tasks/{task_id}/cancel")

    # Assert
    assert response.status_code == 404
    
    # Restore the original mock
    mock_coordinator_service.cancel_task.return_value = original_return_value

def test_get_available_services(mock_coordinator_service):
    # Arrange
    # Act
    response = client.get("/coordinator/available-services")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "market_analysis" in data
    assert "causal_analysis" in data
    assert "backtesting" in data