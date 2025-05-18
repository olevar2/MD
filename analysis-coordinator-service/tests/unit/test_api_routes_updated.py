import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta, UTC
import uuid
import json # Add this import

from analysis_coordinator_service.api.v1.coordinator import router
from analysis_coordinator_service.core.service_dependencies import get_coordinator_service
from analysis_coordinator_service.models.coordinator_models import (
    IntegratedAnalysisResponse,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisTaskStatus,
    AnalysisServiceType,
    AnalysisTaskStatusEnum,
    IntegratedAnalysisRequest,
    AnalysisTaskRequest
)
from analysis_coordinator_service.services.coordinator_service import CoordinatorService

# Create a test client
from fastapi import FastAPI

# Create a test client
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)
client = TestClient(app)

# Remove the fixture as patching will be done per test

@pytest.mark.asyncio
async def test_run_integrated_analysis():
    # Patch the get_settings dependency
    with patch('analysis_coordinator_service.core.service_dependencies.get_settings') as MockGetSettings:
        # Configure the mock settings object returned by get_settings
        mock_settings_instance = AsyncMock() # Use AsyncMock as settings might be async
        mock_settings_instance.market_analysis_service_url = "http://mock-market-analysis"
        mock_settings_instance.causal_analysis_service_url = "http://mock-causal-analysis"
        mock_settings_instance.backtesting_service_url = "http://mock-backtesting"
        mock_settings_instance.feature_store_service_url = "http://mock-feature-store"
        mock_settings_instance.ml_integration_service_url = "http://mock-ml-integration"
        MockGetSettings.return_value = mock_settings_instance

        # Patch the CoordinatorService class
        with patch('analysis_coordinator_service.api.v1.coordinator.CoordinatorService') as MockCoordinatorService:
            # Configure the mock instance returned by the patch
            mock_service_instance = AsyncMock()
        MockCoordinatorService.return_value = mock_service_instance

        # Configure mock responses for this specific test
        mock_service_instance.run_integrated_analysis.return_value = {
            "task_id": str(uuid.uuid4()),
            "status": AnalysisTaskStatusEnum.PENDING.value,
            "created_at": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "estimated_completion_time": (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace('+00:00', 'Z')
        }

        # Create the request body using the Pydantic model
        # Create the request body using the Pydantic model
        start_date_str = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        end_date_str = (datetime.now(UTC) + timedelta(days=1)).isoformat().replace('+00:00', 'Z')
        request_body = {
            "symbol": "EURUSD",
            "timeframe": "1h",
            "start_date": start_date_str,
            "end_date": end_date_str,
            "services": [AnalysisServiceType.MARKET_ANALYSIS.value, AnalysisServiceType.CAUSAL_ANALYSIS.value],
            "parameters": {
                "market_analysis": {"patterns": ["head_and_shoulders"]},
                "causal_analysis": {"variables": ["price", "volume"]}
            }
        }

        # Act
        response = client.post(
            "/coordinator/integrated-analysis",
            json=request_body
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        mock_service_instance.run_integrated_analysis.assert_called_once()

@pytest.mark.asyncio
async def test_create_analysis_task():
    # Patch the get_settings dependency
    with patch('analysis_coordinator_service.core.service_dependencies.get_settings') as MockGetSettings:
        # Configure the mock settings object returned by get_settings
        mock_settings_instance = AsyncMock() # Use AsyncMock as settings might be async
        mock_settings_instance.market_analysis_service_url = "http://mock-market-analysis"
        mock_settings_instance.causal_analysis_service_url = "http://mock-causal-analysis"
        mock_settings_instance.backtesting_service_url = "http://mock-backtesting"
        mock_settings_instance.feature_store_service_url = "http://mock-feature-store"
        mock_settings_instance.ml_integration_service_url = "http://mock-ml-integration"
        MockGetSettings.return_value = mock_settings_instance

        # Patch the CoordinatorService class
        with patch('analysis_coordinator_service.api.v1.coordinator.CoordinatorService') as MockCoordinatorService:
            # Configure the mock instance returned by the patch
            mock_service_instance = AsyncMock()
        MockCoordinatorService.return_value = mock_service_instance

        # Configure mock responses for this specific test
        mock_service_instance.create_analysis_task.return_value = {
            "task_id": str(uuid.uuid4()),
            "service_type": AnalysisServiceType.MARKET_ANALYSIS.value,
            "status": AnalysisTaskStatusEnum.PENDING.value,
            "created_at": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "estimated_completion_time": (datetime.now(UTC) + timedelta(minutes=2)).isoformat().replace('+00:00', 'Z')
        }

        # Create the request body using the Pydantic model
        # Create the request body using the Pydantic model
        start_date_str = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        end_date_str = (datetime.now(UTC) + timedelta(days=1)).isoformat().replace('+00:00', 'Z')
        request_body = {
            "service_type": AnalysisServiceType.MARKET_ANALYSIS.value,
            "symbol": "EURUSD",
            "timeframe": "1h",
            "start_date": start_date_str,
            "end_date": end_date_str,
            "parameters": {"patterns": ["head_and_shoulders"]}
        }

        # Act
        response = client.post(
            "/coordinator/tasks",
            json=request_body
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["service_type"] == "market_analysis"
        assert data["status"] == "pending"
        mock_service_instance.create_analysis_task.assert_called_once()

@pytest.mark.asyncio
async def test_get_task_result():
    # Patch the get_settings dependency
    with patch('analysis_coordinator_service.core.service_dependencies.get_settings') as MockGetSettings:
        # Configure the mock settings object returned by get_settings
        mock_settings_instance = AsyncMock() # Use AsyncMock as settings might be async
        mock_settings_instance.market_analysis_service_url = "http://mock-market-analysis"
        mock_settings_instance.causal_analysis_service_url = "http://mock-causal-analysis"
        mock_settings_instance.backtesting_service_url = "http://mock-backtesting"
        mock_settings_instance.feature_store_service_url = "http://mock-feature-store"
        mock_settings_instance.ml_integration_service_url = "http://mock-ml-integration"
        MockGetSettings.return_value = mock_settings_instance

        # Patch the CoordinatorService class
        with patch('analysis_coordinator_service.api.v1.coordinator.CoordinatorService') as MockCoordinatorService:
            # Configure the mock instance returned by the patch
            mock_service_instance = AsyncMock()
        MockCoordinatorService.return_value = mock_service_instance

        # Configure mock responses for this specific test
        mock_service_instance.get_task_result.return_value = {
            "task_id": str(uuid.uuid4()),
            "service_type": AnalysisServiceType.MARKET_ANALYSIS.value,
            "status": AnalysisTaskStatusEnum.COMPLETED.value,
            "created_at": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "completed_at": (datetime.now(UTC) + timedelta(minutes=2)).isoformat().replace('+00:00', 'Z'),
            "result": {"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
            "error": None
        }

        # Arrange
        task_id = str(uuid.uuid4())

        # Act
        response = client.get(f"/coordinator/tasks/{task_id}")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "completed"
        assert "result" in data
        mock_service_instance.get_task_result.assert_called_once_with(task_id)
