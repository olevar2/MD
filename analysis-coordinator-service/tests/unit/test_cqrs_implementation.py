import pytest
from unittest.mock import AsyncMock, MagicMock
import uuid
from datetime import datetime, UTC

from analysis_coordinator_service.cqrs.commands import (
    RunIntegratedAnalysisCommand,
    CreateAnalysisTaskCommand,
    CancelAnalysisTaskCommand,
    DeleteAnalysisTaskCommand
)
from analysis_coordinator_service.cqrs.queries import (
    GetAnalysisTaskQuery,
    ListAnalysisTasksQuery,
    GetIntegratedAnalysisTaskQuery,
    ListIntegratedAnalysisTasksQuery,
    GetAnalysisTaskStatusQuery,
    GetAvailableServicesQuery
)
from analysis_coordinator_service.cqrs.handlers.command_handlers import (
    RunIntegratedAnalysisCommandHandler,
    CreateAnalysisTaskCommandHandler,
    CancelAnalysisTaskCommandHandler,
    DeleteAnalysisTaskCommandHandler
)
from analysis_coordinator_service.cqrs.handlers.query_handlers import (
    GetAnalysisTaskQueryHandler,
    ListAnalysisTasksQueryHandler,
    GetIntegratedAnalysisTaskQueryHandler,
    ListIntegratedAnalysisTasksQueryHandler,
    GetAnalysisTaskStatusQueryHandler,
    GetAvailableServicesQueryHandler
)
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisServiceType
)
from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus

@pytest.fixture
def mock_coordinator_service():
    """
    Create a mock for CoordinatorService.
    """
    service = AsyncMock()
    
    # Mock _execute_integrated_analysis method
    service._execute_integrated_analysis = AsyncMock()
    
    # Mock _execute_analysis_task method
    service._execute_analysis_task = AsyncMock()
    
    # Mock get_available_services method
    service.get_available_services = AsyncMock(return_value={
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
    })
    
    return service

@pytest.fixture
def mock_task_repository():
    """
    Create a mock for TaskRepository.
    """
    repository = AsyncMock()
    
    # Mock create_integrated_task method
    repository.create_integrated_task = AsyncMock()
    
    # Mock create_task method
    repository.create_task = AsyncMock()
    
    # Mock get_by_id method
    task_id = str(uuid.uuid4())
    repository.get_by_id = AsyncMock(return_value={
        "task_id": task_id,
        "service_type": AnalysisServiceType.MARKET_ANALYSIS,
        "status": AnalysisTaskStatusEnum.COMPLETED,
        "created_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "result": {"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        "error": None
    })
    
    # Mock get_task_status method
    repository.get_task_status = AsyncMock(return_value={
        "status": AnalysisTaskStatusEnum.RUNNING,
        "progress": 0.5,
        "message": "Processing data"
    })
    
    # Mock cancel_task method
    repository.cancel_task = AsyncMock(return_value=True)
    
    # Mock delete method
    repository.delete = AsyncMock()
    
    return repository

@pytest.mark.asyncio
async def test_run_integrated_analysis_command_handler(mock_coordinator_service, mock_task_repository):
    """
    Test the RunIntegratedAnalysisCommandHandler.
    """
    # Arrange
    handler = RunIntegratedAnalysisCommandHandler(
        coordinator_service=mock_coordinator_service,
        repository=mock_task_repository
    )
    
    command = RunIntegratedAnalysisCommand(
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC),
        services=[AnalysisServiceType.MARKET_ANALYSIS, AnalysisServiceType.CAUSAL_ANALYSIS],
        parameters={
            "market_analysis": {"patterns": ["head_and_shoulders"]},
            "causal_analysis": {"variables": ["price", "volume"]}
        }
    )
    
    # Act
    result = await handler.handle(command)
    
    # Assert
    assert mock_task_repository.create_integrated_task.called
    assert mock_coordinator_service._execute_integrated_analysis.called
    assert isinstance(result, str)
    assert len(result) > 0  # Should be a UUID string

@pytest.mark.asyncio
async def test_create_analysis_task_command_handler(mock_coordinator_service, mock_task_repository):
    """
    Test the CreateAnalysisTaskCommandHandler.
    """
    # Arrange
    handler = CreateAnalysisTaskCommandHandler(
        coordinator_service=mock_coordinator_service,
        repository=mock_task_repository
    )
    
    command = CreateAnalysisTaskCommand(
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC),
        parameters={"patterns": ["head_and_shoulders"]}
    )
    
    # Act
    result = await handler.handle(command)
    
    # Assert
    assert mock_task_repository.create_task.called
    assert mock_coordinator_service._execute_analysis_task.called
    assert isinstance(result, str)
    assert len(result) > 0  # Should be a UUID string

@pytest.mark.asyncio
async def test_get_analysis_task_query_handler(mock_task_repository):
    """
    Test the GetAnalysisTaskQueryHandler.
    """
    # Arrange
    handler = GetAnalysisTaskQueryHandler(repository=mock_task_repository)
    
    task_id = str(uuid.uuid4())
    query = GetAnalysisTaskQuery(task_id=task_id)
    
    # Act
    result = await handler.handle(query)
    
    # Assert
    assert mock_task_repository.get_by_id.called
    assert result is not None
    assert result["service_type"] == AnalysisServiceType.MARKET_ANALYSIS
    assert result["status"] == AnalysisTaskStatusEnum.COMPLETED

@pytest.mark.asyncio
async def test_command_bus_integration():
    """
    Test the CommandBus integration with command handlers.
    """
    # Arrange
    command_bus = CommandBus()
    
    # Create mocks
    mock_coordinator_service = AsyncMock()
    mock_coordinator_service._execute_integrated_analysis = AsyncMock()
    
    mock_task_repository = AsyncMock()
    mock_task_repository.create_integrated_task = AsyncMock()
    
    # Create command handlers
    run_integrated_analysis_handler = RunIntegratedAnalysisCommandHandler(
        coordinator_service=mock_coordinator_service,
        repository=mock_task_repository
    )
    
    # Register command handlers
    command_bus.register_handler(RunIntegratedAnalysisCommand, run_integrated_analysis_handler)
    
    # Create command
    command = RunIntegratedAnalysisCommand(
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC),
        services=[AnalysisServiceType.MARKET_ANALYSIS, AnalysisServiceType.CAUSAL_ANALYSIS],
        parameters={
            "market_analysis": {"patterns": ["head_and_shoulders"]},
            "causal_analysis": {"variables": ["price", "volume"]}
        }
    )
    
    # Act
    result = await command_bus.dispatch(command)
    
    # Assert
    assert mock_task_repository.create_integrated_task.called
    assert mock_coordinator_service._execute_integrated_analysis.called
    assert isinstance(result, str)
    assert len(result) > 0  # Should be a UUID string

@pytest.mark.asyncio
async def test_query_bus_integration():
    """
    Test the QueryBus integration with query handlers.
    """
    # Arrange
    query_bus = QueryBus()
    
    # Create mocks
    mock_task_repository = AsyncMock()
    
    # Mock get_by_id method
    task_id = str(uuid.uuid4())
    mock_task_repository.get_by_id = AsyncMock(return_value={
        "task_id": task_id,
        "service_type": AnalysisServiceType.MARKET_ANALYSIS,
        "status": AnalysisTaskStatusEnum.COMPLETED,
        "created_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "result": {"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        "error": None
    })
    
    # Create query handlers
    get_analysis_task_handler = GetAnalysisTaskQueryHandler(repository=mock_task_repository)
    
    # Register query handlers
    query_bus.register_handler(GetAnalysisTaskQuery, get_analysis_task_handler)
    
    # Create query
    query = GetAnalysisTaskQuery(task_id=task_id)
    
    # Act
    result = await query_bus.dispatch(query)
    
    # Assert
    assert mock_task_repository.get_by_id.called
    assert result is not None
    assert result["task_id"] == task_id
    assert result["service_type"] == AnalysisServiceType.MARKET_ANALYSIS
    assert result["status"] == AnalysisTaskStatusEnum.COMPLETED