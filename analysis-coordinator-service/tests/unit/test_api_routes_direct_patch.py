import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta, UTC
import uuid

from analysis_coordinator_service.api.v1.coordinator import router
from analysis_coordinator_service.models.coordinator_models import (
    IntegratedAnalysisRequest,
    IntegratedAnalysisResponse,
    AnalysisTaskRequest,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisTaskStatus,
    AnalysisServiceType,
    AnalysisTaskStatusEnum
)
from analysis_coordinator_service.services.coordinator_service import CoordinatorService
from analysis_coordinator_service.repositories.task_repository import TaskRepository
from analysis_coordinator_service.config.settings import get_settings

# Create a test client
from fastapi import FastAPI
app = FastAPI()
settings = get_settings()
app.include_router(router, prefix=settings.api_prefix)
client = TestClient(app)

# Mock the task repository
mock_task_repository = AsyncMock(spec=TaskRepository)
mock_task_repository.update_task_status = AsyncMock(return_value=True)
mock_task_repository.create_task = AsyncMock(return_value="test-task-id")
mock_task_repository.get_task = AsyncMock(return_value={"task_id": "test-task-id", "status": "completed"})

# Mock the coordinator service
def get_mock_coordinator_service():
    mock_service = AsyncMock(spec=CoordinatorService)
    return mock_service

@pytest.mark.asyncio
async def test_run_integrated_analysis():
    # Create a mock response
    mock_response = IntegratedAnalysisResponse(
        task_id=str(uuid.uuid4()),
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC) + timedelta(minutes=5)
    )
    
    # Create a mock service
    mock_service = get_mock_coordinator_service()
    mock_service.run_integrated_analysis.return_value = mock_response
    
    # Patch the dependencies
    with patch('analysis_coordinator_service.core.service_dependencies.get_coordinator_service', return_value=mock_service), \
         patch('analysis_coordinator_service.core.service_dependencies.get_task_repository', return_value=mock_task_repository):
        # Act
        response = client.post(
            f"{settings.api_prefix}/coordinator/integrated-analysis",
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
        assert data["status"] == "pending"
        mock_service.run_integrated_analysis.assert_called_once()

@pytest.mark.asyncio
async def test_create_analysis_task():
    # Create a mock response
    mock_response = AnalysisTaskResponse(
        task_id=str(uuid.uuid4()),
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC) + timedelta(minutes=2)
    )
    
    # Create a mock service
    mock_service = get_mock_coordinator_service()
    mock_service.create_analysis_task.return_value = mock_response
    
    # Patch the dependencies
    with patch('analysis_coordinator_service.core.service_dependencies.get_coordinator_service', return_value=mock_service), \
         patch('analysis_coordinator_service.core.service_dependencies.get_task_repository', return_value=mock_task_repository):
        # Act
        response = client.post(
            f"{settings.api_prefix}/coordinator/tasks",
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
        assert data["service_type"] == "market_analysis"
        assert data["status"] == "pending"
        mock_service.create_analysis_task.assert_called_once()

@pytest.mark.asyncio
async def test_get_task_result():
    # Create a mock response
    task_id = str(uuid.uuid4())
    mock_response = AnalysisTaskResult(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC) + timedelta(minutes=2),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )
    
    # Create a mock service
    mock_service = get_mock_coordinator_service()
    mock_service.get_task_result.return_value = mock_response
    
    # Patch the dependencies
    with patch('analysis_coordinator_service.core.service_dependencies.get_coordinator_service', return_value=mock_service), \
         patch('analysis_coordinator_service.core.service_dependencies.get_task_repository', return_value=mock_task_repository):
        # Act
        response = client.get(f"{settings.api_prefix}/coordinator/tasks/{task_id}")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "completed"
        assert "result" in data
        mock_service.get_task_result.assert_called_once_with(task_id)
