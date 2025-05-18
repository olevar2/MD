"""
gRPC server implementation for the Backtesting Service.
"""

import asyncio
import logging
import grpc
from concurrent import futures
from typing import Optional, Dict, Any

from common_lib.grpc.backtesting import backtesting_service_pb2 as pb2
from common_lib.grpc.backtesting import backtesting_service_pb2_grpc as pb2_grpc
from common_lib.grpc.common import common_types_pb2, error_types_pb2

from backtesting_service.services.backtest_service import BacktestService
from backtesting_service.services.optimization_service import OptimizationService
from backtesting_service.services.walk_forward_service import WalkForwardService
from backtesting_service.services.strategy_service import StrategyService

from backtesting_service.grpc_server.interceptors import AuthInterceptor, LoggingInterceptor, ErrorHandlingInterceptor

# Import monitoring and tracing components
from common_lib.monitoring import MetricsManager, TracingManager, track_time, trace_function
import time

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


class BacktestingServicer(pb2_grpc.BacktestingServiceServicer):
    """
    gRPC servicer for the Backtesting Service.
    Implements the methods defined in the backtesting_service.proto file.
    """
    
    def __init__(
        self,
        backtest_service: BacktestService,
        optimization_service: OptimizationService,
        walk_forward_service: WalkForwardService,
        strategy_service: StrategyService
    ):
        """
        Initialize the BacktestingServicer.
        
        Args:
            backtest_service: Service for backtest operations
            optimization_service: Service for optimization operations
            walk_forward_service: Service for walk-forward testing operations
            strategy_service: Service for strategy operations
        """
        self.backtest_service = backtest_service
        self.optimization_service = optimization_service
        self.walk_forward_service = walk_forward_service
        self.strategy_service = strategy_service
        self.logger = logger
    
    @trace_function
    async def RunBacktest(
        self,
        request: pb2.RunBacktestRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.RunBacktestResponse:
        """
        Run a backtest for a trading strategy.
        
        Args:
            request: The request containing backtest parameters
            context: The gRPC context
            
        Returns:
            A response containing the backtest result or task ID
        """
        method_name = "RunBacktest"
        self.logger.info(f"{method_name} request received for strategy: {request.strategy_name}")
        response = pb2.RunBacktestResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert request parameters to domain model
            backtest_request = {
            
            # Convert request parameters to domain model
            backtest_request = {
                "strategy_name": request.strategy_name,
                "symbol": request.symbol.name,
                "timeframe": request.timeframe.name,
                "start_date": request.start_date.seconds,
                "end_date": request.end_date.seconds,
                "initial_capital": request.initial_capital,
                "parameters": [
                    {"name": param.name, "value": param.value} for param in request.parameters
                ],
                "options": dict(request.options)
            }
            
            # Call the service
            result = await self.backtest_service.run_backtest(backtest_request)
            
            # Create response
            response = pb2.RunBacktestResponse()
            
            # If the result contains a task ID, it's an asynchronous operation
            if "task_id" in result:
                response.task_id = result["task_id"]
            else:
                # Convert domain model to response
                backtest_result = self._convert_to_backtest_result(result)
                response.result.CopyFrom(backtest_result)
            
            self.logger.info(f"RunBacktest response sent for strategy: {request.strategy_name}")
            status = "OK"
            self.logger.info(f"{method_name} response sent for strategy: {request.strategy_name}")
            return response
            
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in RunBacktest: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error running backtest: {str(e)}")
            
            # Create error response
            response = pb2.RunBacktestResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error running backtest: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
            
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    @trace_function
    async def GetBacktestResult(
        self,
        request: pb2.GetBacktestResultRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.GetBacktestResultResponse:
        """
        Get the result of a backtest.
        
        Args:
            request: The request containing the backtest ID
            context: The gRPC context
            
        Returns:
            A response containing the backtest result
        """
        method_name = "GetBacktestResult"
        self.logger.info(f"{method_name} request received for ID: {request.id}")
        response = pb2.GetBacktestResultResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Call the service
            
            # Call the service
            result = await self.backtest_service.get_backtest_result(request.id)
            
            # Create response
            response = pb2.GetBacktestResultResponse()
            
            # If the result contains a status, it's an asynchronous operation
            if "status" in result:
                response.status = result["status"]
            
            # Convert domain model to response
            if "result" in result:
                backtest_result = self._convert_to_backtest_result(result["result"])
                response.result.CopyFrom(backtest_result)
            
            self.logger.info(f"GetBacktestResult response sent for ID: {request.id}")
            status = "OK"
            self.logger.info(f"{method_name} response sent for ID: {request.id}")
            return response
            
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in GetBacktestResult: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting backtest result: {str(e)}")
            
            # Create error response
            response = pb2.GetBacktestResultResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error getting backtest result: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
            
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def RunOptimization(
        self,
        request: pb2.RunOptimizationRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.RunOptimizationResponse:
        """
        Run an optimization for a trading strategy.
    
        Args:
            request: The request containing optimization parameters
            context: The gRPC context
    
        Returns:
            A response containing the optimization result or task ID
        """
        method_name = "RunOptimization"
        self.logger.info(f"{method_name} request received for strategy: {request.strategy_name}")
        response = pb2.RunOptimizationResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert request parameters to domain model
            optimization_request = {
                "strategy_name": request.strategy_name,
                "symbol": request.symbol.name,
                "timeframe": request.timeframe.name,
                "start_date": request.start_date.seconds,
                "end_date": request.end_date.seconds,
                "initial_capital": request.initial_capital,
                "parameters": [
                    {"name": param.name, "value": param.value} for param in request.parameters
                ],
                "options": dict(request.options)
            }

            # Call the service
            result = await self.optimization_service.run_optimization(optimization_request)

            # Create response
            response = pb2.RunOptimizationResponse()

            # If the result contains a task ID, it's an asynchronous operation
            if "task_id" in result:
                response.task_id = result["task_id"]
            else:
                # Convert domain model to response
                optimization_result = self._convert_to_optimization_result(result)
                response.result.CopyFrom(optimization_result)

            self.logger.info(f"RunOptimization response sent for strategy: {request.strategy_name}")
            status = "OK"
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in RunOptimization: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error running optimization: {str(e)}")

            # Create error response
            response = pb2.RunOptimizationResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error running optimization: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)

            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def GetOptimizationResult(
        self,
        request: pb2.GetOptimizationResultRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.GetOptimizationResultResponse:
        """
        Get the result of an optimization.
    
        Args:
            request: The request containing the optimization ID
            context: The gRPC context
    
        Returns:
            A response containing the optimization result
        """
        method_name = "GetOptimizationResult"
        self.logger.info(f"{method_name} request received for ID: {request.id}")
        response = pb2.GetOptimizationResultResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Call the service
            result = await self.optimization_service.get_optimization_result(request.id)

            # Create response
            response = pb2.GetOptimizationResultResponse()

            # If the result contains a status, it's an asynchronous operation
            if "status" in result:
                response.status = result["status"]

            # Convert domain model to response
            if "result" in result:
                optimization_result = self._convert_to_optimization_result(result["result"])
                response.result.CopyFrom(optimization_result)

            self.logger.info(f"GetOptimizationResult response sent for ID: {request.id}")
            status = "OK"
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in GetOptimizationResult: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting optimization result: {str(e)}")

            # Create error response
            response = pb2.GetOptimizationResultResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error getting optimization result: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)

            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def RunWalkForward(
        self,
        request: pb2.RunWalkForwardRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.RunWalkForwardResponse:
        """
        Run a walk-forward test for a trading strategy.
    
        Args:
            request: The request containing walk-forward parameters
            context: The gRPC context
    
        Returns:
            A response containing the walk-forward result or task ID
        """
        method_name = "RunWalkForward"
        self.logger.info(f"{method_name} request received for strategy: {request.strategy_name}")
        response = pb2.RunWalkForwardResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert request parameters to domain model
            walk_forward_request = {
                "strategy_name": request.strategy_name,
                "symbol": request.symbol.name,
                "timeframe": request.timeframe.name,
                "start_date": request.start_date.seconds,
                "end_date": request.end_date.seconds,
                "initial_capital": request.initial_capital,
                "parameters": [
                    {"name": param.name, "value": param.value} for param in request.parameters
                ],
                "options": dict(request.options)
            }

            # Call the service
            result = await self.walk_forward_service.run_walk_forward(walk_forward_request)

            # Create response
            response = pb2.RunWalkForwardResponse()

            # If the result contains a task ID, it's an asynchronous operation
            if "task_id" in result:
                response.task_id = result["task_id"]
            else:
                # Convert domain model to response
                walk_forward_result = self._convert_to_walk_forward_result(result)
                response.result.CopyFrom(walk_forward_result)

            self.logger.info(f"RunWalkForward response sent for strategy: {request.strategy_name}")
            status = "OK"
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in RunWalkForward: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error running walk-forward test: {str(e)}")

            # Create error response
            response = pb2.RunWalkForwardResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error running walk-forward test: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)

            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def GetWalkForwardResult(
        self,
        request: pb2.GetWalkForwardResultRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.GetWalkForwardResultResponse:
        """
        Get the result of a walk-forward test.
    
        Args:
            request: The request containing the walk-forward ID
            context: The gRPC context
    
        Returns:
            A response containing the walk-forward result
        """
        method_name = "GetWalkForwardResult"
        self.logger.info(f"{method_name} request received for ID: {request.id}")
        response = pb2.GetWalkForwardResultResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Call the service
            result = await self.walk_forward_service.get_walk_forward_result(request.id)

            # Create response
            response = pb2.GetWalkForwardResultResponse()

            # If the result contains a status, it's an asynchronous operation
            if "status" in result:
                response.status = result["status"]

            # Convert domain model to response
            if "result" in result:
                walk_forward_result = self._convert_to_walk_forward_result(result["result"])
                response.result.CopyFrom(walk_forward_result)

            self.logger.info(f"GetWalkForwardResult response sent for ID: {request.id}")
            status = "OK"
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in GetWalkForwardResult: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting walk-forward result: {str(e)}")

            # Create error response
            response = pb2.GetWalkForwardResultResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error getting walk-forward result: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)

            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def CreateStrategy(
        self,
        request: pb2.CreateStrategyRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.CreateStrategyResponse:
        """
        Create a new trading strategy.
    
        Args:
            request: The request containing strategy details
            context: The gRPC context
    
        Returns:
            A response containing the created strategy ID
        """
        method_name = "CreateStrategy"
        self.logger.info(f"{method_name} request received for strategy: {request.name}")
        response = pb2.CreateStrategyResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert request parameters to domain model
            strategy_data = {
                "name": request.name,
                "description": request.description,
                "code": request.code,
                "parameters": [
                    {"name": param.name, "type": param.type, "default_value": param.default_value} for param in request.parameters
                ]
            }

            # Call the service
            strategy_id = await self.strategy_service.create_strategy(strategy_data)

            # Create response
            response.id = strategy_id

            self.logger.info(f"CreateStrategy response sent for strategy: {request.name}")
            status = "OK"
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in CreateStrategy: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error creating strategy: {str(e)}")

            # Create error response
            response = pb2.CreateStrategyResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error creating strategy: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)

            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def GetStrategy(
        self,
        request: pb2.GetStrategyRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.GetStrategyResponse:
        """
        Get a trading strategy by ID.
    
        Args:
            request: The request containing the strategy ID
            context: The gRPC context
    
        Returns:
            A response containing the strategy details
        """
        method_name = "GetStrategy"
        self.logger.info(f"{method_name} request received for ID: {request.id}")
        response = pb2.GetStrategyResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Call the service
            strategy_data = await self.strategy_service.get_strategy(request.id)

            # Create response
            response = pb2.GetStrategyResponse()

            # Convert domain model to response
            if strategy_data:
                strategy_msg = response.strategy
                strategy_msg.id = strategy_data.get("id", "")
                strategy_msg.name = strategy_data.get("name", "")
                strategy_msg.description = strategy_data.get("description", "")
                strategy_msg.code = strategy_data.get("code", "")
                for param in strategy_data.get("parameters", []):
                    param_msg = strategy_msg.parameters.add()
                    param_msg.name = param.get("name", "")
                    param_msg.type = param.get("type", "")
                    param_msg.default_value = param.get("default_value", "")

            self.logger.info(f"GetStrategy response sent for ID: {request.id}")
            status = "OK"
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in GetStrategy: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting strategy: {str(e)}")

            # Create error response
            response = pb2.GetStrategyResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error getting strategy: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)

            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def UpdateStrategy(
        self,
        request: pb2.UpdateStrategyRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.UpdateStrategyResponse:
        """
        Update an existing trading strategy.
    
        Args:
            request: The request containing updated strategy details
            context: The gRPC context
    
        Returns:
            A response indicating success or failure
        """
        method_name = "UpdateStrategy"
        self.logger.info(f"{method_name} request received for ID: {request.id}")
        response = pb2.UpdateStrategyResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert request parameters to domain model
            strategy_data = {
                "id": request.id,
                "name": request.name,
                "description": request.description,
                "code": request.code,
                "parameters": [
                    {"name": param.name, "type": param.type, "default_value": param.default_value} for param in request.parameters
                ]
            }

            # Call the service
            success = await self.strategy_service.update_strategy(strategy_data)

            # Create response
            response.success = success

            self.logger.info(f"UpdateStrategy response sent for ID: {request.id}")
            status = "OK"
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in UpdateStrategy: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error updating strategy: {str(e)}")

            # Create error response
            response = pb2.UpdateStrategyResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error updating strategy: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)

            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()
    
    @trace_function
    async def DeleteStrategy(
        self,
        request: pb2.DeleteStrategyRequest,
        context: grpc.aio.ServicerContext
    ) -> pb2.DeleteStrategyResponse:
        """
        Delete a trading strategy by ID.
    
        Args:
            request: The request containing the strategy ID
            context: The gRPC context
    
        Returns:
            A response indicating success or failure
        """
        method_name = "DeleteStrategy"
        self.logger.info(f"{method_name} request received for ID: {request.id}")
        response = pb2.DeleteStrategyResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()
    
            # Call the service
            success = await self.strategy_service.delete_strategy(request.id)
    
            # Create response
            response.success = success
    
            self.logger.info(f"DeleteStrategy response sent for ID: {request.id}")
            status = "OK"
            return response
    
        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in DeleteStrategy: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error deleting strategy: {str(e)}")
    
            # Create error response
            response = pb2.DeleteStrategyResponse()
            error = error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f"Error deleting strategy: {str(e)}",
                request_id=context.invocation_metadata().get("x-request-id", "unknown")
            )
            response.error.CopyFrom(error)
    
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    def _convert_to_backtest_result(self, result: Dict[str, Any]) -> pb2.BacktestResult:
        """
        Convert a domain model backtest result to a protobuf message.

        Args:
            result: The domain model backtest result

        Returns:
            The protobuf backtest result message
        """
        backtest_result = pb2.BacktestResult()
        
        # Set basic fields
        backtest_result.id = result.get("id", "")
        backtest_result.strategy_name = result.get("strategy_name", "")
        
        # Set symbol
        symbol = common_types_pb2.Symbol()
        symbol.name = result.get("symbol", "")
        backtest_result.symbol.CopyFrom(symbol)
        
        # Set timeframe
        timeframe = common_types_pb2.Timeframe()
        timeframe.name = result.get("timeframe", "")
        timeframe.minutes = result.get("timeframe_minutes", 0)
        backtest_result.timeframe.CopyFrom(timeframe)
        
        # Set dates
        if "start_date" in result:
            start_date = common_types_pb2.Timestamp()
            start_date.seconds = result["start_date"]
            backtest_result.start_date.CopyFrom(start_date)
        
        if "end_date" in result:
            end_date = common_types_pb2.Timestamp()
            end_date.seconds = result["end_date"]
            backtest_result.end_date.CopyFrom(end_date)
        
        # Set performance metrics
        backtest_result.total_trades = result.get("total_trades", 0)
        backtest_result.winning_trades = result.get("winning_trades", 0)
        backtest_result.losing_trades = result.get("losing_trades", 0)
        backtest_result.profit_factor = result.get("profit_factor", 0.0)
        backtest_result.sharpe_ratio = result.get("sharpe_ratio", 0.0)
        backtest_result.max_drawdown = result.get("max_drawdown", 0.0)
        backtest_result.total_return = result.get("total_return", 0.0)
        backtest_result.annualized_return = result.get("annualized_return", 0.0)
        
        # Set trades
        for trade in result.get("trades", []):
            trade_msg = backtest_result.trades.add()
            trade_msg.id = trade.get("id", "")
            trade_msg.symbol = trade.get("symbol", "")
            trade_msg.direction = trade.get("direction", "")
            trade_msg.entry_price = trade.get("entry_price", 0.0)
            trade_msg.exit_price = trade.get("exit_price", 0.0)
            trade_msg.profit_loss = trade.get("profit_loss", 0.0)
            
            if "entry_time" in trade:
                entry_time = common_types_pb2.Timestamp()
                entry_time.seconds = trade["entry_time"]
                trade_msg.entry_time.CopyFrom(entry_time)
            
            if "exit_time" in trade:
                exit_time = common_types_pb2.Timestamp()
                exit_time.seconds = trade["exit_time"]
                trade_msg.exit_time.CopyFrom(exit_time)
        
        # Set parameters
        for param in result.get("parameters", []):
            param_msg = backtest_result.parameters.add()
            param_msg.name = param.get("name", "")
            param_msg.value = param.get("value", "")
        
        # Set equity curve
        if "equity_curve" in result:
            for timestamp, value in result["equity_curve"].items():
                backtest_result.equity_curve[int(timestamp)] = value
        
        return backtest_result


class GrpcServer:
    """
    gRPC server for the Backtesting Service.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50052,
        max_workers: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the gRPC server.
        
        Args:
            host: The host to bind the server to
            port: The port to bind the server to
            max_workers: The maximum number of workers for the server
            config: Optional configuration dictionary
        """
        self.host = host
        self.port = port
        self.max_workers = max_workers or (asyncio.get_event_loop().get_default_executor()._max_workers)
        self.config = config or {}
        self.server = None
        self.logger = logger
    
    async def start(
        self,
        backtest_service: BacktestService,
        optimization_service: OptimizationService,
        walk_forward_service: WalkForwardService,
        strategy_service: StrategyService
    ):
        """
        Start the gRPC server.
        
        Args:
            backtest_service: Service for backtest operations
            optimization_service: Service for optimization operations
            walk_forward_service: Service for walk-forward testing operations
            strategy_service: Service for strategy operations
            
        Returns:
            The running gRPC server
        """
        # Create interceptors
        auth_interceptor = AuthInterceptor()
        logging_interceptor = LoggingInterceptor()
        error_handling_interceptor = ErrorHandlingInterceptor()
        
        # Create server with interceptors
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ],
            interceptors=[auth_interceptor, logging_interceptor, error_handling_interceptor]
        )
        
        # Create servicer
        servicer = BacktestingServicer(
            backtest_service=backtest_service,
            optimization_service=optimization_service,
            walk_forward_service=walk_forward_service,
            strategy_service=strategy_service
        )
        
        # Add servicer to server
        pb2_grpc.add_BacktestingServiceServicer_to_server(servicer, self.server)
        
        # Add secure port if TLS is enabled
        if self.config.get("tls_enabled", False):
            server_credentials = grpc.ssl_server_credentials(
                [(self.config["private_key"], self.config["certificate"])]
            )
            address = f"{self.host}:{self.port}"
            self.server.add_secure_port(address, server_credentials)
            self.logger.info(f"gRPC server starting on {address} with TLS")
        else:
            address = f"{self.host}:{self.port}"
            self.server.add_insecure_port(address)
            self.logger.info(f"gRPC server starting on {address} without TLS")
        
        # Start the server
        await self.server.start()
        
        self.logger.info(f"gRPC server started on {address}")
        
        return self.server
    
    async def stop(self, grace: float = 5.0):
        """
        Stop the gRPC server.
        
        Args:
            grace: Grace period in seconds for stopping the server
        """
        if self.server:
            self.logger.info(f"Stopping gRPC server with {grace}s grace period")
            await self.server.stop(grace)
            self.logger.info("gRPC server stopped")

    def _convert_to_optimization_result(self, result: Dict[str, Any]) -> pb2.OptimizationResult:
        """
        Convert a domain model optimization result to a protobuf message.

        Args:
            result: The domain model optimization result

        Returns:
            The protobuf optimization result message
        """
        optimization_result = pb2.OptimizationResult()

        # Set basic fields
        optimization_result.id = result.get("id", "")
        optimization_result.strategy_name = result.get("strategy_name", "")

        # Set symbol
        symbol = common_types_pb2.Symbol()
        symbol.name = result.get("symbol", "")
        optimization_result.symbol.CopyFrom(symbol)

        # Set timeframe
        timeframe = common_types_pb2.Timeframe()
        timeframe.name = result.get("timeframe", "")
        timeframe.minutes = result.get("timeframe_minutes", 0)
        optimization_result.timeframe.CopyFrom(timeframe)

        # Set dates
        if "start_date" in result:
            start_date = common_types_pb2.Timestamp()
            start_date.seconds = result["start_date"]
            optimization_result.start_date.CopyFrom(start_date)

        if "end_date" in result:
            end_date = common_types_pb2.Timestamp()
            end_date.seconds = result["end_date"]
            optimization_result.end_date.CopyFrom(end_date)

        # Set objective value
        optimization_result.objective_value = result.get("objective_value", 0.0)

        # Set best parameters
        for param in result.get("best_parameters", []):
            param_msg = optimization_result.best_parameters.add()
            param_msg.name = param.get("name", "")
            param_msg.value = param.get("value", "")

        # Set results per parameter set
        for param_set_result in result.get("results_per_parameter_set", []):
            param_set_result_msg = optimization_result.results_per_parameter_set.add()
            param_set_result_msg.objective_value = param_set_result.get("objective_value", 0.0)
            param_set_result_msg.total_return = param_set_result.get("total_return", 0.0)
            param_set_result_msg.max_drawdown = param_set_result.get("max_drawdown", 0.0)
            param_set_result_msg.sharpe_ratio = param_set_result.get("sharpe_ratio", 0.0)
            for param in param_set_result.get("parameters", []):
                param_msg = param_set_result_msg.parameters.add()
                param_msg.name = param.get("name", "")
                param_msg.value = param.get("value", "")

        return optimization_result


    def _convert_to_walk_forward_result(self, result: Dict[str, Any]) -> pb2.WalkForwardResult:
        """
        Convert a domain model walk-forward result to a protobuf message.

        Args:
            result: The domain model walk-forward result

        Returns:
            The protobuf walk-forward result message
        """
        walk_forward_result = pb2.WalkForwardResult()

        # Set basic fields
        walk_forward_result.id = result.get("id", "")
        walk_forward_result.strategy_name = result.get("strategy_name", "")

        # Set symbol
        symbol = common_types_pb2.Symbol()
        symbol.name = result.get("symbol", "")
        walk_forward_result.symbol.CopyFrom(symbol)

        # Set timeframe
        timeframe = common_types_pb2.Timeframe()
        timeframe.name = result.get("timeframe", "")
        timeframe.minutes = result.get("timeframe_minutes", 0)
        walk_forward_result.timeframe.CopyFrom(timeframe)

        # Set dates
        if "start_date" in result:
            start_date = common_types_pb2.Timestamp()
            start_date.seconds = result["start_date"]
            walk_forward_result.start_date.CopyFrom(start_date)

        if "end_date" in result:
            end_date = common_types_pb2.Timestamp()
            end_date.seconds = result["end_date"]
            walk_forward_result.end_date.CopyFrom(end_date)

        # Set overall performance metrics
        overall_metrics = result.get("overall_metrics", {})
        walk_forward_result.overall_metrics.total_return = overall_metrics.get("total_return", 0.0)
        walk_forward_result.overall_metrics.annualized_return = overall_metrics.get("annualized_return", 0.0)
        walk_forward_result.overall_metrics.max_drawdown = overall_metrics.get("max_drawdown", 0.0)
        walk_forward_result.overall_metrics.sharpe_ratio = overall_metrics.get("sharpe_ratio", 0.0)
        walk_forward_result.overall_metrics.profit_factor = overall_metrics.get("profit_factor", 0.0)

        # Set periods
        for period in result.get("periods", []):
            period_msg = walk_forward_result.periods.add()
            period_msg.start_date.seconds = period.get("start_date", 0)
            period_msg.end_date.seconds = period.get("end_date", 0)
            period_msg.in_sample_start_date.seconds = period.get("in_sample_start_date", 0)
            period_msg.in_sample_end_date.seconds = period.get("in_sample_end_date", 0)
            period_msg.out_of_sample_start_date.seconds = period.get("out_of_sample_start_date", 0)
            period_msg.out_of_sample_end_date.seconds = period.get("out_of_sample_end_date", 0)

            # Set period metrics
            period_metrics = period.get("metrics", {})
            period_msg.metrics.total_return = period_metrics.get("total_return", 0.0)
            period_msg.metrics.annualized_return = period_metrics.get("annualized_return", 0.0)
            period_msg.metrics.max_drawdown = period_metrics.get("max_drawdown", 0.0)
            period_msg.metrics.sharpe_ratio = period_metrics.get("sharpe_ratio", 0.0)
            period_msg.metrics.profit_factor = period_metrics.get("profit_factor", 0.0)

            # Set best parameters for the period
            for param in period.get("best_parameters", []):
                param_msg = period_msg.best_parameters.add()
                param_msg.name = param.get("name", "")
                param_msg.value = param.get("value", "")

        return walk_forward_result


async def serve():
    # Initialize services
    backtest_service = BacktestService()
    optimization_service = OptimizationService()
    walk_forward_service = WalkForwardService()
    strategy_service = StrategyService()

    # Create and start gRPC server
    grpc_server = GrpcServer()
    server = await grpc_server.start(
        backtest_service=backtest_service,
        optimization_service=optimization_service,
        walk_forward_service=walk_forward_service,
        strategy_service=strategy_service
    )

    # Keep server running
    await server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())