import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from datetime import datetime, timedelta, UTC
import uuid

from analysis_coordinator_service.api.v1.coordinator import router
from analysis_coordinator_service.models.coordinator_models import (
    IntegratedAnalysisResponse,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisTaskStatus,
    AnalysisServiceType,
    AnalysisTaskStatusEnum
)

# Create a test client
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.mark.asyncio
async def test_run_integrated_analysis(mock_coordinator_service):
    # Configure mock responses
    mock_coordinator_service.run_integrated_analysis.return_value = IntegratedAnalysisResponse(
        task_id=str(uuid.uuid4()),
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC) + timedelta(minutes=5)
    )
    
    # Arrange
    with patch('analysis_coordinator_service.core.service_dependencies.get_coordinator_service', return_value=mock_coordinator_service):
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
        assert data["status"] == "pending"
        mock_coordinator_service.run_integrated_analysis.assert_called_once()

@pytest.mark.asyncio
async def test_create_analysis_task(mock_coordinator_service):
    # Configure mock responses
    mock_coordinator_service.create_analysis_task.return_value = AnalysisTaskResponse(
        task_id=str(uuid.uuid4()),
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC) + timedelta(minutes=2)
    )
    
    # Arrange
    with patch('analysis_coordinator_service.core.service_dependencies.get_coordinator_service', return_value=mock_coordinator_service):
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
        assert data["service_type"] == "market_analysis"
        assert data["status"] == "pending"
        mock_coordinator_service.create_analysis_task.assert_called_once()

@pytest.mark.asyncio
async def test_get_task_result(mock_coordinator_service):
    # Configure mock responses
    mock_coordinator_service.get_task_result.return_value = AnalysisTaskResult(
        task_id=str(uuid.uuid4()),
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC) + timedelta(minutes=2),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )
    
    # Arrange
    task_id = str(uuid.uuid4())
    with patch('analysis_coordinator_service.core.service_dependencies.get_coordinator_service', return_value=mock_coordinator_service):
        # Act
        response = client.get(f"/coordinator/tasks/{task_id}")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "completed"
        assert "result" in data
        mock_coordinator_service.get_task_result.assert_called_once_with(task_id)
