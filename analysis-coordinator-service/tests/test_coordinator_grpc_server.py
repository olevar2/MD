import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from common_lib.errors.decorators import (
    with_exception_handling,
    async_with_exception_handling
)

from analysis_coordinator_service.services.coordinator_service import CoordinatorService
from analysis_coordinator_service.models.coordinator_models import (
    CreateTaskRequest,
    CreateTaskResponse,
    GetTaskStatusRequest,
    GetTaskStatusResponse,
    ListTasksRequest,
    ListTasksResponse,
    CancelTaskRequest,
    CancelTaskResponse,
    GetTaskResultRequest,
    GetTaskResultResponse,
    TaskResult as CoordinatorTaskResult,
    Task as CoordinatorTask,
    TaskStatus as CoordinatorTaskStatus,
    TaskType as CoordinatorTaskType
)
from analysis_coordinator import analysis_coordinator_service_pb2
from analysis_coordinator import analysis_coordinator_service_pb2_grpc

class AnalysisCoordinatorGrpcServicer(analysis_coordinator_service_pb2_grpc.AnalysisCoordinatorServiceServicer):
# Mock the gRPC context
    class MockContext:
        def set_code(self, code):
            self._code = code

        def set_details(self, details):
            self._details = details

        def code(self):
            return self._code if hasattr(self, '_code') else None

        def details(self):
            return self._details if hasattr(self, '_details') else None

@pytest.fixture
def mock_coordinator_service():
    """Fixture to provide a mock CoordinatorService."""
    return AsyncMock(spec=CoordinatorService)

@pytest.fixture
def grpc_servicer(mock_coordinator_service):
    """Fixture to provide an instance of the gRPC servicer."""
    return AnalysisCoordinatorGrpcServicer(mock_coordinator_service)

@pytest.fixture
def mock_context():
    """Fixture to provide a mock gRPC context."""
    return MockContext()

@pytest.mark.asyncio
async def test_create_task_success(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful CreateTask gRPC call."""
    grpc_request = analysis_coordinator_service_pb2.CreateTaskRequest(
        task_type="TEST_TASK",
        parameters={"param1": "value1"},
        callback_url="http://example.com/callback"
    )

    mock_pydantic_response = CreateTaskResponse(
        task_id="test-task-id",
        status=CoordinatorTaskStatus.PENDING,
        message="Task created successfully"
    )
    mock_coordinator_service.create_task.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.CreateTask(grpc_request, mock_context)

    mock_coordinator_service.create_task.assert_called_once_with(
        CreateTaskRequest(
            task_type="TEST_TASK",
            parameters={"param1": "value1"},
            callback_url="http://example.com/callback"
        )
    )
    assert grpc_response.task_id == "test-task-id"
    assert grpc_response.status == analysis_coordinator_service_pb2.TaskStatus.PENDING
    assert grpc_response.message == "Task created successfully"
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_create_task_no_callback(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful CreateTask gRPC call without callback URL."""
    grpc_request = analysis_coordinator_service_pb2.CreateTaskRequest(
        task_type="ANOTHER_TASK",
        parameters={"param2": "value2"}
    )

    mock_pydantic_response = CreateTaskResponse(
        task_id="another-task-id",
        status=CoordinatorTaskStatus.RUNNING,
        message="Task started"
    )
    mock_coordinator_service.create_task.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.CreateTask(grpc_request, mock_context)

    mock_coordinator_service.create_task.assert_called_once_with(
        CreateTaskRequest(
            task_type="ANOTHER_TASK",
            parameters={"param2": "value2"},
            callback_url=None
        )
    )
    assert grpc_response.task_id == "another-task-id"
    assert grpc_response.status == analysis_coordinator_service_pb2.TaskStatus.RUNNING
    assert grpc_response.message == "Task started"
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_create_task_exception(grpc_servicer, mock_coordinator_service, mock_context):
    """Test CreateTask gRPC call with an exception from the service."""
    grpc_request = analysis_coordinator_service_pb2.CreateTaskRequest(
        task_type="FAILING_TASK",
        parameters={}
    )

    mock_coordinator_service.create_task.side_effect = Exception("Service error")

    grpc_response = await grpc_servicer.CreateTask(grpc_request, mock_context)

    mock_coordinator_service.create_task.assert_called_once_with(
        CreateTaskRequest(
            task_type="FAILING_TASK",
            parameters={},
            callback_url=None
        )
    )
    assert grpc_response == analysis_coordinator_service_pb2.CreateTaskResponse() # Should return default empty response on error
    assert mock_context.code() == grpc.StatusCode.INTERNAL
    assert "Internal server error: Service error" in mock_context.details()

@pytest.mark.asyncio
async def test_get_task_status_success(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful GetTaskStatus gRPC call."""
    grpc_request = analysis_coordinator_service_pb2.GetTaskStatusRequest(
        task_id="status-task-id"
    )

    mock_pydantic_response = GetTaskStatusResponse(
        task_id="status-task-id",
        status=CoordinatorTaskStatus.COMPLETED,
        message="Task finished",
        result={"output": "analysis result"},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    mock_coordinator_service.get_task_status.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.GetTaskStatus(grpc_request, mock_context)

    mock_coordinator_service.get_task_status.assert_called_once_with(
        GetTaskStatusRequest(task_id="status-task-id")
    )
    assert grpc_response.task_id == "status-task-id"
    assert grpc_response.status == analysis_coordinator_service_pb2.TaskStatus.COMPLETED
    assert grpc_response.message == "Task finished"
    assert grpc_response.result == '{"output": "analysis result"}' # Pydantic dict becomes JSON string in protobuf
    assert grpc_response.created_at == mock_pydantic_response.created_at.isoformat()
    assert grpc_response.updated_at == mock_pydantic_response.updated_at.isoformat()
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_get_task_status_exception(grpc_servicer, mock_coordinator_service, mock_context):
    """Test GetTaskStatus gRPC call with an exception from the service.""" # ... existing code ...
    grpc_request = analysis_coordinator_service_pb2.GetTaskStatusRequest(
        task_id="failing-status-id"
    )

    mock_coordinator_service.get_task_status.side_effect = Exception("Status error")

    grpc_response = await grpc_servicer.GetTaskStatus(grpc_request, mock_context)

    mock_coordinator_service.get_task_status.assert_called_once_with(
        GetTaskStatusRequest(task_id="failing-status-id")
    )
    assert grpc_response == analysis_coordinator_service_pb2.GetTaskStatusResponse()
    assert mock_context.code() == grpc.StatusCode.INTERNAL
    assert "Internal server error: Status error" in mock_context.details()

@pytest.mark.asyncio
async def test_list_tasks_success(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful ListTasks gRPC call."""
    grpc_request = analysis_coordinator_service_pb2.ListTasksRequest(
        # Add pagination/filtering fields if needed
    )

    mock_pydantic_response = ListTasksResponse(
        tasks=[
            CoordinatorTask(
                task_id="task-1",
                task_type="TYPE_A",
                status=CoordinatorTaskStatus.COMPLETED,
                message="Done",
                result={"data": 1},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            CoordinatorTask(
                task_id="task-2",
                task_type="TYPE_B",
                status=CoordinatorTaskStatus.PENDING,
                message="Waiting",
                result=None,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ],
        next_page_token="next-token"
    )
    mock_coordinator_service.list_tasks.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.ListTasks(grpc_request, mock_context)

    mock_coordinator_service.list_tasks.assert_called_once_with(
        ListTasksRequest(
            # Add pagination/filtering fields if needed
        )
    )
    assert len(grpc_response.tasks) == 2
    assert grpc_response.tasks[0].task_id == "task-1"
    assert grpc_response.tasks[0].status == analysis_coordinator_service_pb2.TaskStatus.COMPLETED
    assert grpc_response.tasks[1].task_id == "task-2"
    assert grpc_response.tasks[1].status == analysis_coordinator_service_pb2.TaskStatus.PENDING
    assert grpc_response.next_page_token == "next-token"
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_list_tasks_exception(grpc_servicer, mock_coordinator_service, mock_context):
    """Test ListTasks gRPC call with an exception from the service."""
    grpc_request = analysis_coordinator_service_pb2.ListTasksRequest()

    mock_coordinator_service.list_tasks.side_effect = Exception("List error")

    grpc_response = await grpc_servicer.ListTasks(grpc_request, mock_context)

    mock_coordinator_service.list_tasks.assert_called_once_with(
        ListTasksRequest()
    )
    assert grpc_response == analysis_coordinator_service_pb2.ListTasksResponse()
    assert mock_context.code() == grpc.StatusCode.INTERNAL
    assert "Internal server error: List error" in mock_context.details()

@pytest.mark.asyncio
async def test_cancel_task_success(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful CancelTask gRPC call."""
    grpc_request = analysis_coordinator_service_pb2.CancelTaskRequest(
        task_id="cancel-task-id"
    )

    mock_pydantic_response = CancelTaskResponse(
        task_id="cancel-task-id",
        success=True,
        message="Task cancelled"
    )
    mock_coordinator_service.cancel_task.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.CancelTask(grpc_request, mock_context)

    mock_coordinator_service.cancel_task.assert_called_once_with(
        CancelTaskRequest(task_id="cancel-task-id")
    )
    assert grpc_response.task_id == "cancel-task-id"
    assert grpc_response.success is True
    assert grpc_response.message == "Task cancelled"
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_cancel_task_exception(grpc_servicer, mock_coordinator_service, mock_context):
    """Test CancelTask gRPC call with an exception from the service."""
    grpc_request = analysis_coordinator_service_pb2.CancelTaskRequest(
        task_id="failing-cancel-id"
    )

    mock_coordinator_service.cancel_task.side_effect = Exception("Cancel error")

    grpc_response = await grpc_servicer.CancelTask(grpc_request, mock_context)

    mock_coordinator_service.cancel_task.assert_called_once_with(
        CancelTaskRequest(task_id="failing-cancel-id")
    )
    assert grpc_response == analysis_coordinator_service_pb2.CancelTaskResponse()
    assert mock_context.code() == grpc.StatusCode.INTERNAL
    assert "Internal server error: Cancel error" in mock_context.details()

@pytest.mark.asyncio
async def test_get_task_result_success(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful GetTaskResult gRPC call."""
    grpc_request = analysis_coordinator_service_pb2.GetTaskResultRequest(
        task_id="result-task-id"
    )

    mock_pydantic_response = GetTaskResultResponse(
        result=CoordinatorTaskResult(
            task=CoordinatorTask(
                task_id="result-task-id",
                task_type=CoordinatorTaskType.INTEGRATED_ANALYSIS,
                status=CoordinatorTaskStatus.COMPLETED,
                message="Result available",
                result={"final": "output"},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                start_time=datetime.now(),
                end_time=datetime.now(),
                progress=100,
                error_message=None,
                parameters={"input": "data"},
                metadata={"user": "abc"}
            ),
            result_type="JSON",
            result_data=b'{"final": "output"}',
            summary="Analysis completed successfully."
        ),
        error=None
    )
    mock_coordinator_service.get_task_result.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.GetTaskResult(grpc_request, mock_context)

    mock_coordinator_service.get_task_result.assert_called_once_with(
        GetTaskResultRequest(task_id="result-task-id")
    )
    assert grpc_response.result.task.id == "result-task-id"
    assert grpc_response.result.task.type == analysis_coordinator_service_pb2.TaskType.INTEGRATED_ANALYSIS
    assert grpc_response.result.task.status == analysis_coordinator_service_pb2.TaskStatus.COMPLETED
    assert grpc_response.result.result_type == "JSON"
    assert grpc_response.result.result_data == b'{"final": "output"}'
    assert grpc_response.result.summary == "Analysis completed successfully."
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_get_task_result_exception(grpc_servicer, mock_coordinator_service, mock_context):
    """Test GetTaskResult gRPC call with an exception from the service."""
    grpc_request = analysis_coordinator_service_pb2.GetTaskResultRequest(
        task_id="failing-result-id"
    )

    mock_coordinator_service.get_task_result.side_effect = Exception("Result error")

    grpc_response = await grpc_servicer.GetTaskResult(grpc_request, mock_context)

    mock_coordinator_service.get_task_result.assert_called_once_with(
        GetTaskResultRequest(task_id="failing-result-id")
    )
    assert grpc_response == analysis_coordinator_service_pb2.GetTaskResultResponse()
    assert mock_context.code() == grpc.StatusCode.INTERNAL
    assert "Internal server error: Result error" in mock_context.details()

@pytest.mark.asyncio
async def test_perform_integrated_analysis_success(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful PerformIntegratedAnalysis gRPC call."""
    start_date_str = "2023-01-01T00:00:00Z"
    end_date_str = "2023-01-31T23:59:59Z"
    grpc_request = analysis_coordinator_service_pb2.PerformIntegratedAnalysisRequest(
        symbol="EURUSD",
        timeframe="H1",
        start_date=start_date_str,
        end_date=end_date_str,
        services=["SERVICE_A", "SERVICE_B"],
        parameters={"strategy": "momentum"}
    )

    mock_pydantic_response = CreateTaskResponse(
        task_id="integrated-task-id",
        status=CoordinatorTaskStatus.PENDING,
        message="Integrated analysis task created",
        created_at=datetime.now(),
        estimated_completion_time=datetime.now()
    )
    mock_coordinator_service.run_integrated_analysis.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.PerformIntegratedAnalysis(grpc_request, mock_context)

    expected_start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
    expected_end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))

    mock_coordinator_service.run_integrated_analysis.assert_called_once_with(
        symbol="EURUSD",
        timeframe="H1",
        start_date=expected_start_date,
        end_date=expected_end_date,
        services=["SERVICE_A", "SERVICE_B"],
        parameters={"strategy": "momentum"}
    )
    assert grpc_response.task_id == "integrated-task-id"
    assert grpc_response.status == analysis_coordinator_service_pb2.TaskStatus.PENDING
    assert grpc_response.created_at == mock_pydantic_response.created_at.isoformat()
    assert grpc_response.estimated_completion_time == mock_pydantic_response.estimated_completion_time.isoformat()
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_perform_integrated_analysis_no_end_date(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful PerformIntegratedAnalysis gRPC call without end date."""
    start_date_str = "2023-01-01T00:00:00Z"
    grpc_request = analysis_coordinator_service_pb2.PerformIntegratedAnalysisRequest(
        symbol="GBPUSD",
        timeframe="D1",
        start_date=start_date_str,
        services=["SERVICE_C"],
        parameters={}
    )

    mock_pydantic_response = CreateTaskResponse(
        task_id="integrated-task-id-no-end",
        status=CoordinatorTaskStatus.RUNNING,
        message="Integrated analysis task created",
        created_at=datetime.now(),
        estimated_completion_time=datetime.now()
    )
    mock_coordinator_service.run_integrated_analysis.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.PerformIntegratedAnalysis(grpc_request, mock_context)

    expected_start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))

    mock_coordinator_service.run_integrated_analysis.assert_called_once_with(
        symbol="GBPUSD",
        timeframe="D1",
        start_date=expected_start_date,
        end_date=None,
        services=["SERVICE_C"],
        parameters={}
    )
    assert grpc_response.task_id == "integrated-task-id-no-end"
    assert grpc_response.status == analysis_coordinator_service_pb2.TaskStatus.RUNNING
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_perform_integrated_analysis_exception(grpc_servicer, mock_coordinator_service, mock_context):
    """Test PerformIntegratedAnalysis gRPC call with an exception from the service."""
    grpc_request = analysis_coordinator_service_pb2.PerformIntegratedAnalysisRequest(
        symbol="USDJPY",
        timeframe="M15",
        start_date="2023-01-01T00:00:00Z",
        services=[],
        parameters={}
    )

    mock_coordinator_service.run_integrated_analysis.side_effect = Exception("Integrated analysis error")

    grpc_response = await grpc_servicer.PerformIntegratedAnalysis(grpc_request, mock_context)

    assert grpc_response == analysis_coordinator_service_pb2.PerformIntegratedAnalysisResponse()
    assert mock_context.code() == grpc.StatusCode.INTERNAL
    assert "Internal server error: Integrated analysis error" in mock_context.details()

@pytest.mark.asyncio
async def test_perform_multi_timeframe_analysis_success(grpc_servicer, mock_coordinator_service, mock_context):
    """Test successful PerformMultiTimeframeAnalysis gRPC call."""
    start_date_str = "2023-02-01T00:00:00Z"
    end_date_str = "2023-02-28T23:59:59Z"
    grpc_request = analysis_coordinator_service_pb2.PerformMultiTimeframeAnalysisRequest(
        symbol="AUDCAD",
        timeframe="H4", # Base timeframe
        start_date=start_date_str,
        end_date=end_date_str,
        timeframes=["H1", "D1"],
        parameters={"indicator": "RSI"}
    )

    mock_pydantic_response = CreateTaskResponse(
        task_id="multi-timeframe-task-id",
        status=CoordinatorTaskStatus.PENDING,
        message="Multi-timeframe analysis task created",
        created_at=datetime.now(),
        estimated_completion_time=datetime.now()
    )
    mock_coordinator_service.create_analysis_task.return_value = mock_pydantic_response

    grpc_response = await grpc_servicer.PerformMultiTimeframeAnalysis(grpc_request, mock_context)

    expected_start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
    expected_end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))

    mock_coordinator_service.create_analysis_task.assert_called_once_with(
        service_type=CoordinatorTaskType.MULTI_TIMEFRAME_ANALYSIS.name,
        symbol="AUDCAD",
        timeframe="H4",
        start_date=expected_start_date,
        end_date=expected_end_date,
        parameters={
            "timeframes": ["H1", "D1"],
            "indicator": "RSI"
        }
    )
    assert grpc_response.task_id == "multi-timeframe-task-id"
    assert grpc_response.status == analysis_coordinator_service_pb2.TaskStatus.PENDING
    assert grpc_response.message == "Multi-timeframe analysis task created"
    assert grpc_response.created_at == mock_pydantic_response.created_at.isoformat()
    assert grpc_response.estimated_completion_time == mock_pydantic_response.estimated_completion_time.isoformat()
    assert mock_context.code() is None
    assert mock_context.details() is None

@pytest.mark.asyncio
async def test_perform_multi_timeframe_analysis_exception(grpc_servicer, mock_coordinator_service, mock_context):
    """Test PerformMultiTimeframeAnalysis gRPC call with an exception from the service."""
    grpc_request = analysis_coordinator_service_pb2.PerformMultiTimeframeAnalysisRequest(
        symbol="NZDCAD",
        timeframe="H1",
        start_date="2023-02-01T00:00:00Z",
        timeframes=[],
        parameters={}
    )

    mock_coordinator_service.create_analysis_task.side_effect = Exception("Multi-timeframe error")

    grpc_response = await grpc_servicer.PerformMultiTimeframeAnalysis(grpc_request, mock_context)

    assert grpc_response == analysis_coordinator_service_pb2.PerformMultiTimeframeAnalysisResponse()
    assert mock_context.code() == grpc.StatusCode.INTERNAL
    assert "Internal server error: Multi-timeframe error" in mock_context.details()

# Add tests for other methods (ListTasks, CancelTask, GetTaskResult) following the same pattern.