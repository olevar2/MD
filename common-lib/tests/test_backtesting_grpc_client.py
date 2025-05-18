import unittest
from unittest.mock import MagicMock
import grpc

# Assuming proto-generated stubs and client are available
from common_lib.grpc.backtesting import backtesting_service_pb2
from common_lib.grpc.backtesting import backtesting_service_pb2_grpc
from common_lib.grpc.common import common_types_pb2
from common_lib.clients.backtesting_grpc_client import BacktestingGrpcClient

class TestBacktestingGrpcClient(unittest.TestCase):

    def setUp(self):
        # Setup mock client factory and stub
        self.mock_client_factory = MagicMock()
        self.mock_stub = MagicMock()
        self.mock_client_factory.get_stub.return_value = self.mock_stub
        self.client = BacktestingGrpcClient(self.mock_client_factory)

    def tearDown(self):
        # Clean up resources if necessary
        pass

    async def test_run_backtest_success(self):
        # Setup mock response
        mock_response = backtesting_service_pb2.RunBacktestResponse(
            backtest_id='test_id',
            status='COMPLETED'
        )
        self.mock_stub.RunBacktest.future.return_value.result.return_value = mock_response

        # Call the method
        strategy_config = {"strategy": "config"}
        data_config = {"data": "config"}
        time_range = {"start_time": "", "end_time": ""}
        result = await self.client.run_backtest(strategy_config, data_config, time_range)

        # Assertions
        self.assertEqual(result, {"backtest_id": "test_id", "status": "COMPLETED"})
        self.mock_stub.RunBacktest.assert_called_once()
        # Add more assertions for request content if needed

if __name__ == '__main__':
    unittest.main()