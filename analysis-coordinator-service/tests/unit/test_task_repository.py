import pytest
from datetime import datetime, timedelta, UTC
import uuid

from analysis_coordinator_service.repositories.task_repository import TaskRepository
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisServiceType,
    AnalysisTaskStatusEnum
)

@pytest.fixture
def task_repository():
    return TaskRepository(connection_string="mock_connection_string")

@pytest.mark.asyncio
async def test_create_task(task_repository):
    # Arrange
    task_id = str(uuid.uuid4())
    service_type = AnalysisServiceType.MARKET_ANALYSIS
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    end_date = start_date + timedelta(days=1)
    parameters = {"patterns": ["head_and_shoulders"]}

    # Act
    await task_repository.create_task(
        task_id=task_id,
        service_type=service_type,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters
    )

    # Assert
    assert task_id in task_repository.tasks
    assert task_repository.tasks[task_id]["service_type"] == service_type
    assert task_repository.tasks[task_id]["symbol"] == symbol
    assert task_repository.tasks[task_id]["timeframe"] == timeframe
    assert task_repository.tasks[task_id]["start_date"] == start_date
    assert task_repository.tasks[task_id]["end_date"] == end_date
    assert task_repository.tasks[task_id]["parameters"] == parameters
    assert task_repository.tasks[task_id]["status"] == AnalysisTaskStatusEnum.PENDING

@pytest.mark.asyncio
async def test_create_integrated_task(task_repository):
    # Arrange
    task_id = str(uuid.uuid4())
    services = [AnalysisServiceType.MARKET_ANALYSIS, AnalysisServiceType.CAUSAL_ANALYSIS]
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    end_date = start_date + timedelta(days=1)
    parameters = {
        "market_analysis": {"patterns": ["head_and_shoulders"]},
        "causal_analysis": {"variables": ["price", "volume"]}
    }

    # Act
    await task_repository.create_integrated_task(
        task_id=task_id,
        services=services,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters
    )

    # Assert
    assert task_id in task_repository.integrated_tasks
    assert task_repository.integrated_tasks[task_id]["services"] == services
    assert task_repository.integrated_tasks[task_id]["symbol"] == symbol
    assert task_repository.integrated_tasks[task_id]["timeframe"] == timeframe
    assert task_repository.integrated_tasks[task_id]["start_date"] == start_date
    assert task_repository.integrated_tasks[task_id]["end_date"] == end_date
    assert task_repository.integrated_tasks[task_id]["parameters"] == parameters
    assert task_repository.integrated_tasks[task_id]["status"] == AnalysisTaskStatusEnum.PENDING
    assert len(task_repository.integrated_tasks[task_id]["subtasks"]) == len(services)

@pytest.mark.asyncio
async def test_get_task(task_repository):
    # Arrange
    task_id = str(uuid.uuid4())
    service_type = AnalysisServiceType.MARKET_ANALYSIS
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)

    await task_repository.create_task(
        task_id=task_id,
        service_type=service_type,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date
    )

    # Act
    task = await task_repository.get_task(task_id)

    # Assert
    assert task is not None
    assert task.task_id == task_id
    assert task.service_type == service_type
    assert task.status == AnalysisTaskStatusEnum.PENDING

@pytest.mark.asyncio
async def test_get_task_status(task_repository):
    # Arrange
    task_id = str(uuid.uuid4())
    service_type = AnalysisServiceType.MARKET_ANALYSIS
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)

    await task_repository.create_task(
        task_id=task_id,
        service_type=service_type,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date
    )

    # Act
    status = await task_repository.get_task_status(task_id)

    # Assert
    assert status is not None
    assert status.task_id == task_id
    assert status.service_type == service_type
    assert status.status == AnalysisTaskStatusEnum.PENDING
    assert status.progress == 0.0

@pytest.mark.asyncio
async def test_update_task_status(task_repository):
    # Arrange
    task_id = str(uuid.uuid4())
    service_type = AnalysisServiceType.MARKET_ANALYSIS
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)

    await task_repository.create_task(
        task_id=task_id,
        service_type=service_type,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date
    )

    # Act
    await task_repository.update_task_status(
        task_id=task_id,
        status=AnalysisTaskStatusEnum.RUNNING,
        progress=0.5,
        message="Task is running"
    )

    # Assert
    status = await task_repository.get_task_status(task_id)
    assert status is not None
    assert status.status == AnalysisTaskStatusEnum.RUNNING
    assert status.progress == 0.5
    assert status.message == "Task is running"

@pytest.mark.asyncio
async def test_delete_task(task_repository):
    # Arrange
    task_id = str(uuid.uuid4())
    service_type = AnalysisServiceType.MARKET_ANALYSIS
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)

    await task_repository.create_task(
        task_id=task_id,
        service_type=service_type,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date
    )

    # Act
    result = await task_repository.delete_task(task_id)

    # Assert
    assert result is True
    assert task_id not in task_repository.tasks

@pytest.mark.asyncio
async def test_list_tasks(task_repository):
    # Arrange
    # Create a few tasks
    for i in range(5):
        task_id = str(uuid.uuid4())
        service_type = AnalysisServiceType.MARKET_ANALYSIS
        symbol = f"EURUSD{i}"
        timeframe = "1h"
        start_date = datetime.now(UTC)

        await task_repository.create_task(
            task_id=task_id,
            service_type=service_type,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date
        )

    # Act
    tasks = await task_repository.list_tasks(limit=10, offset=0)

    # Assert
    assert len(tasks) == 5

    # Test pagination
    tasks_paginated = await task_repository.list_tasks(limit=2, offset=0)
    assert len(tasks_paginated) == 2

    # Test filtering by status
    await task_repository.update_task_status(
        task_id=list(task_repository.tasks.keys())[0],
        status=AnalysisTaskStatusEnum.RUNNING
    )

    tasks_filtered = await task_repository.list_tasks(status=AnalysisTaskStatusEnum.RUNNING)
    assert len(tasks_filtered) == 1
    assert tasks_filtered[0]["status"] == AnalysisTaskStatusEnum.RUNNING