import pytest
from datetime import datetime, UTC
from pydantic import ValidationError

from analysis_coordinator_service.models.coordinator_models import (
    IntegratedAnalysisRequest,
    IntegratedAnalysisResponse,
    AnalysisTaskRequest,
    AnalysisTaskResponse,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType,
    AnalysisTaskStatusEnum
)

def test_analysis_service_type_enum():
    # Test valid values
    assert AnalysisServiceType.MARKET_ANALYSIS == "market_analysis"
    assert AnalysisServiceType.CAUSAL_ANALYSIS == "causal_analysis"
    assert AnalysisServiceType.BACKTESTING == "backtesting"

    # Test conversion from string
    assert AnalysisServiceType("market_analysis") == AnalysisServiceType.MARKET_ANALYSIS
    assert AnalysisServiceType("causal_analysis") == AnalysisServiceType.CAUSAL_ANALYSIS
    assert AnalysisServiceType("backtesting") == AnalysisServiceType.BACKTESTING

    # Test invalid value
    with pytest.raises(ValueError):
        AnalysisServiceType("invalid_service")

def test_analysis_task_status_enum():
    # Test valid values
    assert AnalysisTaskStatusEnum.PENDING == "pending"
    assert AnalysisTaskStatusEnum.RUNNING == "running"
    assert AnalysisTaskStatusEnum.COMPLETED == "completed"
    assert AnalysisTaskStatusEnum.FAILED == "failed"
    assert AnalysisTaskStatusEnum.CANCELLED == "cancelled"

    # Test conversion from string
    assert AnalysisTaskStatusEnum("pending") == AnalysisTaskStatusEnum.PENDING
    assert AnalysisTaskStatusEnum("running") == AnalysisTaskStatusEnum.RUNNING
    assert AnalysisTaskStatusEnum("completed") == AnalysisTaskStatusEnum.COMPLETED
    assert AnalysisTaskStatusEnum("failed") == AnalysisTaskStatusEnum.FAILED
    assert AnalysisTaskStatusEnum("cancelled") == AnalysisTaskStatusEnum.CANCELLED

    # Test invalid value
    with pytest.raises(ValueError):
        AnalysisTaskStatusEnum("invalid_status")

def test_integrated_analysis_request():
    # Test valid request
    request = IntegratedAnalysisRequest(
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

    assert request.symbol == "EURUSD"
    assert request.timeframe == "1h"
    assert isinstance(request.start_date, datetime)
    assert isinstance(request.end_date, datetime)
    assert len(request.services) == 2
    assert request.services[0] == AnalysisServiceType.MARKET_ANALYSIS
    assert request.services[1] == AnalysisServiceType.CAUSAL_ANALYSIS
    assert "market_analysis" in request.parameters
    assert "causal_analysis" in request.parameters

    # Test request with missing required fields
    with pytest.raises(ValidationError):
        IntegratedAnalysisRequest(
            timeframe="1h",
            start_date=datetime.now(UTC),
            services=[AnalysisServiceType.MARKET_ANALYSIS]
        )

    # Test request with invalid service type
    with pytest.raises(ValidationError):
        IntegratedAnalysisRequest(
            symbol="EURUSD",
            timeframe="1h",
            start_date=datetime.now(UTC),
            services=["invalid_service"]
        )

def test_integrated_analysis_response():
    # Test valid response
    response = IntegratedAnalysisResponse(
        task_id="test-task-id",
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC)
    )

    assert response.task_id == "test-task-id"
    assert response.status == AnalysisTaskStatusEnum.PENDING
    assert isinstance(response.created_at, datetime)
    assert isinstance(response.estimated_completion_time, datetime)

    # Test response with default values
    response = IntegratedAnalysisResponse()

    assert response.task_id != ""  # Should generate a UUID
    assert response.status == AnalysisTaskStatusEnum.PENDING
    assert isinstance(response.created_at, datetime)
    assert response.estimated_completion_time is None

def test_analysis_task_request():
    # Test valid request
    request = AnalysisTaskRequest(
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC),
        parameters={"patterns": ["head_and_shoulders"]}
    )

    assert request.service_type == AnalysisServiceType.MARKET_ANALYSIS
    assert request.symbol == "EURUSD"
    assert request.timeframe == "1h"
    assert isinstance(request.start_date, datetime)
    assert isinstance(request.end_date, datetime)
    assert "patterns" in request.parameters

    # Test request with missing required fields
    with pytest.raises(ValidationError):
        AnalysisTaskRequest(
            symbol="EURUSD",
            timeframe="1h",
            start_date=datetime.now(UTC)
        )

    # Test request with invalid service type
    with pytest.raises(ValidationError):
        AnalysisTaskRequest(
            service_type="invalid_service",
            symbol="EURUSD",
            timeframe="1h",
            start_date=datetime.now(UTC)
        )

def test_analysis_task_response():
    # Test valid response
    response = AnalysisTaskResponse(
        task_id="test-task-id",
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC)
    )

    assert response.task_id == "test-task-id"
    assert response.service_type == AnalysisServiceType.MARKET_ANALYSIS
    assert response.status == AnalysisTaskStatusEnum.PENDING
    assert isinstance(response.created_at, datetime)
    assert isinstance(response.estimated_completion_time, datetime)

    # Test response with default values
    response = AnalysisTaskResponse(
        service_type=AnalysisServiceType.MARKET_ANALYSIS
    )

    assert response.task_id != ""  # Should generate a UUID
    assert response.service_type == AnalysisServiceType.MARKET_ANALYSIS
    assert response.status == AnalysisTaskStatusEnum.PENDING
    assert isinstance(response.created_at, datetime)
    assert response.estimated_completion_time is None

def test_analysis_task_status():
    # Test valid status
    status = AnalysisTaskStatus(
        task_id="test-task-id",
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.RUNNING,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        progress=0.5,
        message="Task is running"
    )

    assert status.task_id == "test-task-id"
    assert status.service_type == AnalysisServiceType.MARKET_ANALYSIS
    assert status.status == AnalysisTaskStatusEnum.RUNNING
    assert isinstance(status.created_at, datetime)
    assert isinstance(status.updated_at, datetime)
    assert status.progress == 0.5
    assert status.message == "Task is running"

    # Test status with missing required fields
    with pytest.raises(ValidationError):
        AnalysisTaskStatus(
            service_type=AnalysisServiceType.MARKET_ANALYSIS,
            status=AnalysisTaskStatusEnum.RUNNING,
            created_at=datetime.now(UTC)
        )

def test_analysis_task_result():
    # Test valid result
    result = AnalysisTaskResult(
        task_id="test-task-id",
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )

    assert result.task_id == "test-task-id"
    assert result.service_type == AnalysisServiceType.MARKET_ANALYSIS
    assert result.status == AnalysisTaskStatusEnum.COMPLETED
    assert isinstance(result.created_at, datetime)
    assert isinstance(result.completed_at, datetime)
    assert "patterns" in result.result
    assert result.error is None

    # Test result with missing required fields
    with pytest.raises(ValidationError):
        AnalysisTaskResult(
            service_type=AnalysisServiceType.MARKET_ANALYSIS,
            status=AnalysisTaskStatusEnum.COMPLETED,
            created_at=datetime.now(UTC)
        )

    # Test result with error
    result = AnalysisTaskResult(
        task_id="test-task-id",
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.FAILED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        result=None,
        error="Task failed with an error"
    )

    assert result.status == AnalysisTaskStatusEnum.FAILED
    assert result.result is None
    assert result.error == "Task failed with an error"