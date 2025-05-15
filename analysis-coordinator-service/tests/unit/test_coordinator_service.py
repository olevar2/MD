import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime, timedelta, UTC

from analysis_coordinator_service.services.coordinator_service import CoordinatorService
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisTaskStatus,
    AnalysisServiceType,
    IntegratedAnalysisResponse
)
from tests.unit.test_repositories import TestTaskRepository

@pytest.fixture
def mock_market_analysis_adapter():
    """
    Create a mock for MarketAnalysisAdapter.
    """
    adapter = AsyncMock()
    adapter.analyze_market.return_value = {
        "patterns": [{"type": "head_and_shoulders", "confidence": 0.85}],
        "support_resistance": [{"type": "support", "level": 1.1000}, {"type": "resistance", "level": 1.2000}],
        "market_regime": {"type": "trending", "strength": 0.75}
    }
    return adapter

@pytest.fixture
def mock_causal_analysis_adapter():
    """
    Create a mock for CausalAnalysisAdapter.
    """
    adapter = AsyncMock()
    adapter.generate_causal_graph.return_value = {
        "relationships": [
            {"cause": "price", "effect": "volume", "strength": 0.65},
            {"cause": "news", "effect": "price", "strength": 0.75}
        ]
    }
    return adapter

@pytest.fixture
def mock_backtesting_adapter():
    """
    Create a mock for BacktestingAdapter.
    """
    adapter = AsyncMock()
    adapter.run_backtest.return_value = {
        "performance": {
            "profit_factor": 1.75,
            "sharpe_ratio": 1.25,
            "max_drawdown": 0.15
        }
    }
    return adapter

@pytest.fixture
def task_repository():
    """
    Create a TestTaskRepository.
    """
    return TestTaskRepository()

@pytest.fixture
def coordinator_service(
    mock_market_analysis_adapter,
    mock_causal_analysis_adapter,
    mock_backtesting_adapter,
    task_repository
):
    """
    Create a CoordinatorService with mocked dependencies.
    """
    return CoordinatorService(
        market_analysis_adapter=mock_market_analysis_adapter,
        causal_analysis_adapter=mock_causal_analysis_adapter,
        backtesting_adapter=mock_backtesting_adapter,
        task_repository=task_repository
    )

@pytest.mark.asyncio
async def test_run_integrated_analysis(coordinator_service):
    """
    Test run_integrated_analysis method.
    """
    # Arrange
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    end_date = datetime.now(UTC) + timedelta(days=1)
    services = ["market_analysis", "causal_analysis"]
    parameters = {
        "market_analysis": {"patterns": ["head_and_shoulders"]},
        "causal_analysis": {"variables": ["price", "volume"]}
    }
    
    # Act
    result = await coordinator_service.run_integrated_analysis(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        services=services,
        parameters=parameters
    )
    
    # Assert
    assert result is not None
    assert isinstance(result, IntegratedAnalysisResponse)
    assert result.status == AnalysisTaskStatusEnum.PENDING
    assert result.task_id is not None

@pytest.mark.asyncio
async def test_create_analysis_task(coordinator_service):
    """
    Test create_analysis_task method.
    """
    # Arrange
    service_type = AnalysisServiceType.MARKET_ANALYSIS
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    end_date = datetime.now(UTC) + timedelta(days=1)
    parameters = {"patterns": ["head_and_shoulders"]}
    
    # Act
    result = await coordinator_service.create_analysis_task(
        service_type=service_type,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters
    )
    
    # Assert
    assert result is not None
    assert isinstance(result, AnalysisTaskResponse)
    assert result.status == AnalysisTaskStatusEnum.PENDING
    assert result.task_id is not None
    assert result.service_type == service_type

@pytest.mark.asyncio
async def test_get_task_result(coordinator_service, task_repository):
    """
    Test get_task_result method.
    """
    # Arrange
    task_id = str(uuid.uuid4())
    task_repository.tasks[task_id] = AnalysisTaskResult(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )
    
    # Act
    result = await coordinator_service.get_task_result(task_id)
    
    # Assert
    assert result is not None
    assert isinstance(result, AnalysisTaskResult)
    assert result.task_id == task_id
    assert result.status == AnalysisTaskStatusEnum.COMPLETED
    assert result.result is not None
    assert "patterns" in result.result

@pytest.mark.asyncio
async def test_get_task_status(coordinator_service, task_repository):
    """
    Test get_task_status method.
    """
    # Arrange
    task_id = str(uuid.uuid4())
    task_repository.task_statuses[task_id] = AnalysisTaskStatus(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.RUNNING,
        created_at=datetime.now(UTC),
        progress=0.5,
        message="Processing data"
    )
    
    # Act
    result = await coordinator_service.get_task_status(task_id)
    
    # Assert
    assert result is not None
    assert isinstance(result, AnalysisTaskStatus)
    assert result.task_id == task_id
    assert result.status == AnalysisTaskStatusEnum.RUNNING
    assert result.progress == 0.5
    assert result.message == "Processing data"

@pytest.mark.asyncio
async def test_delete_task(coordinator_service, task_repository):
    """
    Test delete_task method.
    """
    # Arrange
    task_id = str(uuid.uuid4())
    task_repository.tasks[task_id] = AnalysisTaskResult(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )
    
    # Act
    result = await coordinator_service.delete_task(task_id)
    
    # Assert
    assert result is True
    assert task_id not in task_repository.tasks

@pytest.mark.asyncio
async def test_list_tasks(coordinator_service, task_repository):
    """
    Test list_tasks method.
    """
    # Arrange
    task_id1 = str(uuid.uuid4())
    task_id2 = str(uuid.uuid4())
    
    task_repository.tasks[task_id1] = AnalysisTaskResult(
        task_id=task_id1,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )
    
    task_repository.tasks[task_id2] = AnalysisTaskResult(
        task_id=task_id2,
        service_type=AnalysisServiceType.CAUSAL_ANALYSIS,
        status=AnalysisTaskStatusEnum.RUNNING,
        created_at=datetime.now(UTC),
        completed_at=None,
        result=None,
        error=None
    )
    
    # Act
    result = await coordinator_service.list_tasks()
    
    # Assert
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2

@pytest.mark.asyncio
async def test_cancel_task(coordinator_service, task_repository):
    """
    Test cancel_task method.
    """
    # Arrange
    task_id = str(uuid.uuid4())
    task_repository.tasks[task_id] = AnalysisTaskResult(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.RUNNING,
        created_at=datetime.now(UTC),
        completed_at=None,
        result=None,
        error=None
    )
    
    task_repository.task_statuses[task_id] = AnalysisTaskStatus(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.RUNNING,
        created_at=datetime.now(UTC),
        progress=0.5,
        message="Processing data"
    )
    
    # Act
    result = await coordinator_service.cancel_task(task_id)
    
    # Assert
    assert result is True
    assert task_repository.task_statuses[task_id].status == AnalysisTaskStatusEnum.CANCELLED

@pytest.mark.asyncio
async def test_get_available_services(coordinator_service):
    """
    Test get_available_services method.
    """
    # Act
    result = await coordinator_service.get_available_services()
    
    # Assert
    assert result is not None
    assert isinstance(result, dict)
    assert "market_analysis" in result
    assert "causal_analysis" in result
    assert "backtesting" in result