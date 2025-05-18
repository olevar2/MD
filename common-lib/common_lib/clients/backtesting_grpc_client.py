# common_lib/clients/backtesting_grpc_client.py

import grpc
from typing import Dict, Any, List, Optional

from common_lib.interfaces.backtesting_service_interface import IBacktestingService
from common_lib.clients.grpc_client_factory import GrpcClientFactory

# Assuming proto-generated stubs will be available in common_lib.proto
from common_lib.proto.backtesting import backtesting_service_pb2
from common_lib.proto.backtesting import backtesting_service_pb2_grpc
from common_lib.proto.common import common_types_pb2

class BacktestingGrpcClient(IBacktestingService):
    """gRPC client for the Backtesting Service."""

    def __init__(self, client_factory: GrpcClientFactory):
        """
        Initializes the client with a gRPC client factory.

        Args:
            client_factory: The factory to get gRPC stubs.
        """
        self._client_factory = client_factory
        self._stub = self._client_factory.get_stub('backtesting', backtesting_service_pb2_grpc.BacktestingServiceStub)
        # TODO: Ensure proto compilation is set up and stubs are available.

    async def run_backtest(self,
                           strategy_config: Dict[str, Any],
                           data_config: Dict[str, Any],
                           time_range: Dict[str, Any],
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a backtest using gRPC.
        """
        if not self._stub:
            raise NotImplementedError("gRPC stub not initialized. Ensure proto stubs are available and factory is configured.")

        # Convert input dictionaries to protobuf messages
        # Assuming strategy_config, data_config, time_range, and config are already in the correct format or can be directly used
        # If conversion is needed, implement it here or in a helper function
        request = backtesting_service_pb2.RunBacktestRequest(
            strategy_config=common_types_pb2.JsonConfig(json_data=str(strategy_config)),
            data_config=common_types_pb2.JsonConfig(json_data=str(data_config)),
            time_range=common_types_pb2.TimeRange(start_time=time_range.get('start_time', ''), end_time=time_range.get('end_time', '')),
            config=common_types_pb2.JsonConfig(json_data=str(config)) if config else None
        )

        try:
            response = await self._stub.RunBacktest(request)
            # Assuming the response contains the necessary fields
            return {"backtest_id": response.backtest_id, "status": response.status}
        except grpc.RpcError as e:
            print(f"gRPC error calling run_backtest: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred calling run_backtest: {e}")
            raise

    async def get_backtest_results(self,
                                   backtest_id: str,
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get backtest results using gRPC.
        """
        if not self._stub:
            raise NotImplementedError("gRPC stub not initialized.")
        # Convert input to protobuf message
        request = backtesting_service_pb2.GetBacktestResultsRequest(
            backtest_id=backtest_id,
            config=common_types_pb2.JsonConfig(json_data=str(config)) if config else None
        )

        try:
            response = await self._stub.GetBacktestResults(request)
            # Convert response protobuf message back to Dict[str, Any]
            # Assuming a helper function or logic exists to perform this conversion
            # For now, returning a placeholder based on the response structure.
            return {"results": response.results} # Assuming response has a 'results' field
        except grpc.RpcError as e:
            print(f"gRPC error calling get_backtest_results: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred calling get_backtest_results: {e}")
            raise

    async def list_backtests(self,
                             user_id: Optional[str] = None,
                             status: Optional[str] = None,
                             config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List backtests using gRPC.
        """
        if not self._stub:
            raise NotImplementedError("gRPC stub not initialized.")
        # Convert input to protobuf message
        request = backtesting_service_pb2.ListBacktestsRequest(
            user_id=user_id if user_id else '',
            status=status if status else '',
            config=common_types_pb2.JsonConfig(json_data=str(config)) if config else None
        )

        try:
            response = await self._stub.ListBacktests(request)
            # Convert response protobuf message back to List[Dict[str, Any]]
            # Assuming response has a 'backtests' field which is a list of messages
            # Need to convert each message in the list to a dictionary.
            # Placeholder conversion:
            return [{"id": bt.backtest_id, "status": bt.status} for bt in response.backtests]
        except grpc.RpcError as e:
            print(f"gRPC error calling list_backtests: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred calling list_backtests: {e}")
            raise

    async def cancel_backtest(self,
                              backtest_id: str,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Cancel a backtest using gRPC.
        """
        if not self._stub:
            raise NotImplementedError("gRPC stub not initialized.")
        # Convert input to protobuf message
        request = backtesting_service_pb2.CancelBacktestRequest(
            backtest_id=backtest_id,
            config=common_types_pb2.JsonConfig(json_data=str(config)) if config else None
        )

        try:
            response = await self._stub.CancelBacktest(request)
            # Convert response protobuf message back to Dict[str, Any]
            # Assuming response has a 'success' field
            return {"success": response.success}
        except grpc.RpcError as e:
            print(f"gRPC error calling cancel_backtest: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred calling cancel_backtest: {e}")
            raise

    async def delete_backtest(self,
                              backtest_id: str,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Delete a backtest using gRPC.
        """
        if not self._stub:
            raise NotImplementedError("gRPC stub not initialized.")
        # Convert input to protobuf message
        request = backtesting_service_pb2.DeleteBacktestRequest(
            backtest_id=backtest_id,
            config=common_types_pb2.JsonConfig(json_data=str(config)) if config else None
        )

        try:
            response = await self._stub.DeleteBacktest(request)
            # Convert response protobuf message back to Dict[str, Any]
            # Assuming response has a 'success' field
            return {"success": response.success}
        except grpc.RpcError as e:
            print(f"gRPC error calling delete_backtest: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred calling delete_backtest: {e}")
            raise