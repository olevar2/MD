import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from fastapi.testclient import TestClient
from datetime import datetime, UTC
import uuid

from analysis_coordinator_service.main import app
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisTaskStatus,
    AnalysisServiceType,
    IntegratedAnalysisResponse
)
from tests.unit.test_repositories import TestTaskRepository

# Create a test client
client = TestClient(app)

class AsyncContextManagerMock(MagicMock):
    """
    Mock for async context managers.
    This class extends MagicMock to properly handle async context manager protocol.
    """
    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.fixture
def mock_session():
    """
    Create a mock for aiohttp.ClientSession that properly handles async context manager.
    """
    # Create the main session mock
    session = MagicMock()
    
    # Create response mocks
    post_response = MagicMock()
    post_response.status = 200
    post_response.json = AsyncMock(return_value={"result": "success"})
    post_response.text = AsyncMock(return_value="success")
    
    get_response = MagicMock()
    get_response.status = 200
    get_response.json = AsyncMock(return_value={"result": "success"})
    get_response.text = AsyncMock(return_value="success")
    
    # Set up the post method
    post_context_manager = AsyncContextManagerMock()
    post_context_manager.return_value = post_response
    session.post = MagicMock(return_value=post_context_manager)
    
    # Set up the get method
    get_context_manager = AsyncContextManagerMock()
    get_context_manager.return_value = get_response
    session.get = MagicMock(return_value=get_context_manager)
    
    return session

@pytest.fixture
def mock_task_repository():
    """
    Create a mock for TaskRepository.
    """
    return TestTaskRepository()

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