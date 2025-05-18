import grpc
import logging
from datetime import datetime
import time

# Import generated protobuf code
from analysis_coordinator import analysis_coordinator_service_pb2
from analysis_coordinator import analysis_coordinator_service_pb2_grpc

# Import existing service and models
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
    TaskStatus as CoordinatorTaskStatus
)

# Import monitoring and tracing components
from common_lib.monitoring import MetricsManager, TracingManager, track_time, trace_function

logger = logging.getLogger(__name__)

# Initialize metrics
metrics_manager = MetricsManager()
grpc_requests_total = metrics_manager.create_counter(
    'grpc_requests_total',
    'Total number of gRPC requests received, by method and status',
    ['method', 'status']
)
grpc_request_duration_seconds = metrics_manager.create_histogram(
    'grpc_request_duration_seconds',
    'gRPC request duration in seconds, by method',
    ['method']
)

class AnalysisCoordinatorGrpcServicer(analysis_coordinator_service_pb2_grpc.AnalysisCoordinatorServiceServicer):
    """
    Implements the gRPC service definition for the Analysis Coordinator Service.
    """

    def __init__(self, coordinator_service: CoordinatorService):
        """
        Initializes the gRPC servicer with an instance of the CoordinatorService.
        """
        self.coordinator_service = coordinator_service
        self.logger = logger

    @trace_function
    async def CreateTask(self, request, context):
        """
        gRPC method to create a new analysis task.
        """
        method_name = "CreateTask"
        self.logger.info(f"{method_name} gRPC call received for task type: {request.task_type}")
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert gRPC request to Pydantic model
            pydantic_request = CreateTaskRequest(
                task_type=request.task_type,
                parameters=request.parameters,
                callback_url=request.callback_url if request.HasField("callback_url") else None
            )

            pydantic_response = await self.coordinator_service.create_task(pydantic_request)

            # Convert Pydantic response to gRPC response
            grpc_response = analysis_coordinator_service_pb2.CreateTaskResponse(
                task_id=pydantic_response.task_id,
                status=pydantic_response.status.value, # Assuming Pydantic Enum has .value
                message=pydantic_response.message
            )
            status = "OK"
            return grpc_response
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in {method_name} gRPC call: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            # Return an empty response or an error response if defined in proto
            return analysis_coordinator_service_pb2.CreateTaskResponse()
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def GetTaskStatus(self, request, context):
        """
        gRPC method to get the status of an analysis task.
        """
        method_name = "GetTaskStatus"
        self.logger.info(f"{method_name} gRPC call received for task ID: {request.task_id}")
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            pydantic_request = GetTaskStatusRequest(task_id=request.task_id)
            pydantic_response = await self.coordinator_service.get_task_status(pydantic_request)

            grpc_response = analysis_coordinator_service_pb2.GetTaskStatusResponse(
                task_id=pydantic_response.task_id,
                status=pydantic_response.status.value, # Assuming Pydantic Enum has .value
                message=pydantic_response.message
            )
            status = "OK"
            return grpc_response
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in {method_name} gRPC call: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return analysis_coordinator_service_pb2.GetTaskStatusResponse()
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def ListTasks(self, request, context):
        """
        gRPC method to list analysis tasks.
        """
        method_name = "ListTasks"
        self.logger.info(f"{method_name} gRPC call received")
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            pydantic_request = ListTasksRequest(
                status=request.status if request.HasField("status") else None
            )
            pydantic_response = await self.coordinator_service.list_tasks(pydantic_request)

            grpc_response = analysis_coordinator_service_pb2.ListTasksResponse()
            for task in pydantic_response.tasks:
                grpc_task = grpc_response.tasks.add()
                grpc_task.task_id = task.task_id
                grpc_task.task_type = task.task_type
                grpc_task.status = task.status.value
                grpc_task.message = task.message
                if task.created_at:
                    grpc_task.created_at.seconds = int(task.created_at.timestamp())
                if task.updated_at:
                    grpc_task.updated_at.seconds = int(task.updated_at.timestamp())

            status = "OK"
            return grpc_response
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in {method_name} gRPC call: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return analysis_coordinator_service_pb2.ListTasksResponse()
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def CancelTask(self, request, context):
        """
        gRPC method to cancel an analysis task.
        """
        method_name = "CancelTask"
        self.logger.info(f"{method_name} gRPC call received for task ID: {request.task_id}")
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            pydantic_request = CancelTaskRequest(task_id=request.task_id)
            pydantic_response = await self.coordinator_service.cancel_task(pydantic_request)

            grpc_response = analysis_coordinator_service_pb2.CancelTaskResponse(
                task_id=pydantic_response.task_id,
                success=pydantic_response.success,
                message=pydantic_response.message
            )
            status = "OK"
            return grpc_response
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in {method_name} gRPC call: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return analysis_coordinator_service_pb2.CancelTaskResponse()
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def GetTaskResult(self, request, context):
        """
        gRPC method to get the result of an analysis task.
        """
        method_name = "GetTaskResult"
        self.logger.info(f"{method_name} gRPC call received for task ID: {request.task_id}")
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Assuming coordinator_service.get_task_result returns a dict or similar structure
            result_data = await self.coordinator_service.get_task_result(request.task_id)

            grpc_response = analysis_coordinator_service_pb2.GetTaskResultResponse(
                task_id=request.task_id,
                status=result_data.get("status", "UNKNOWN"), # Assuming status is in result_data
                message=result_data.get("message", "") # Assuming message is in result_data
            )

            # Assuming result_data contains a 'result' field with the actual analysis result
            # You might need a conversion method similar to BacktestingService if the result structure is complex
            if "result" in result_data:
                 # Example: If result is a simple string or JSON string
                 grpc_response.result_json = result_data["result"]
                 # If result is a complex protobuf message, you'd need a conversion method:
                 # analysis_result_msg = self._convert_to_analysis_result(result_data["result"])
                 # grpc_response.analysis_result.CopyFrom(analysis_result_msg)

            status = "OK"
            return grpc_response
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in {method_name} gRPC call: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return analysis_coordinator_service_pb2.GetTaskResultResponse()
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def PerformIntegratedAnalysis(self, request, context):
        """
        gRPC method to perform integrated analysis.
        """
        method_name = "PerformIntegratedAnalysis"
        self.logger.info(f"{method_name} gRPC call received")
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert gRPC request to domain model/parameters for the service
            analysis_params = {
                "strategy_id": request.strategy_id,
                "symbol": request.symbol.name,
                "timeframe": request.timeframe.name,
                "start_date": request.start_date.seconds,
                "end_date": request.end_date.seconds if request.HasField("end_date") else None,
                "parameters": [
                    {"name": param.name, "value": param.value} for param in request.parameters
                ],
                "options": dict(request.options)
            }

            # Call the service method
            analysis_result = await self.coordinator_service.perform_integrated_analysis(analysis_params)

            # Convert service result to gRPC response
            grpc_response = analysis_coordinator_service_pb2.PerformIntegratedAnalysisResponse()

            # Assuming analysis_result is a dictionary containing the result data
            # You might need a conversion method here if the result structure is complex
            if "result" in analysis_result:
                 # Example: If result is a simple string or JSON string
                 grpc_response.result_json = analysis_result["result"]
                 # If result is a complex protobuf message, you'd need a conversion method:
                 # analysis_result_msg = self._convert_to_integrated_analysis_result(analysis_result["result"])
                 # grpc_response.analysis_result.CopyFrom(analysis_result_msg)

            status = "OK"
            return grpc_response
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in {method_name} gRPC call: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return analysis_coordinator_service_pb2.PerformIntegratedAnalysisResponse()
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def PerformMultiTimeframeAnalysis(self, request, context):
        """
        gRPC method to perform multi-timeframe analysis.
        """
        method_name = "PerformMultiTimeframeAnalysis"
        self.logger.info(f"{method_name} gRPC call received")
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert gRPC request to domain model/parameters for the service
            analysis_params = {
                "strategy_id": request.strategy_id,
                "symbol": request.symbol.name,
                "timeframes": [tf.name for tf in request.timeframes],
                "start_date": request.start_date.seconds,
                "end_date": request.end_date.seconds if request.HasField("end_date") else None,
                "parameters": [
                    {"name": param.name, "value": param.value} for param in request.parameters
                ],
                "options": dict(request.options)
            }

            # Call the service method
            analysis_result = await self.coordinator_service.perform_multi_timeframe_analysis(analysis_params)

            # Convert service result to gRPC response
            grpc_response = analysis_coordinator_service_pb2.PerformMultiTimeframeAnalysisResponse()

            # Assuming analysis_result is a dictionary containing the result data
            # You might need a conversion method here if the result structure is complex
            if "result" in analysis_result:
                 # Example: If result is a simple string or JSON string
                 grpc_response.result_json = analysis_result["result"]
                 # If result is a complex protobuf message, you'd need a conversion method:
                 # analysis_result_msg = self._convert_to_multi_timeframe_analysis_result(analysis_result["result"])
                 # grpc_response.analysis_result.CopyFrom(analysis_result_msg)

            status = "OK"
            return grpc_response
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in {method_name} gRPC call: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return analysis_coordinator_service_pb2.PerformMultiTimeframeAnalysisResponse()
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

# async def serve():
#     server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
#     coordinator_service_instance = CoordinatorService(...)
#     analysis_coordinator_service_pb2_grpc.add_AnalysisCoordinatorServiceServicer_to_server(
#         AnalysisCoordinatorGrpcServicer(coordinator_service_instance), server
#     )
#     listen_addr = '[::]:50052' # Example port
#     server.add_insecure_port(listen_addr)
#     logging.info("Starting server on %s", listen_addr)
#     await server.start()
#     await server.wait_for_termination()