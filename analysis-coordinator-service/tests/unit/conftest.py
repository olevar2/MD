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