# common_lib/clients/grpc_client_factory.py

import grpc
from typing import Dict, Type

# Assuming proto-generated stubs will be available in common_lib.proto
# from common_lib.proto.market_analysis import market_analysis_service_pb2_grpc
# from common_lib.proto.causal_analysis import causal_analysis_service_pb2_grpc
# from common_lib.proto.backtesting import backtesting_service_pb2_grpc
# from common_lib.proto.analysis_coordinator import analysis_coordinator_service_pb2_grpc

class GrpcClientFactory:
    """Factory for creating gRPC client stubs."""

    def __init__(self, service_addresses: Dict[str, str]):
        """
        Initializes the factory with service addresses.

        Args:
            service_addresses: A dictionary mapping service names to their gRPC addresses (e.g., {'market_analysis': 'localhost:50051'}).
        """
        self._channels: Dict[str, grpc.Channel] = {}
        self._stubs: Dict[str, object] = {}
        self._service_addresses = service_addresses

    def _get_channel(self, service_name: str) -> grpc.Channel:
        """
        Gets or creates a gRPC channel for a given service.
        """
        if service_name not in self._channels:
            address = self._service_addresses.get(service_name)
            if not address:
                raise ValueError(f"Address for service '{service_name}' not found.")
            # TODO: Add resilience patterns (retries, timeouts, circuit breakers)
            channel = grpc.insecure_channel(address) # Use secure_channel for production
            self._channels[service_name] = channel
        return self._channels[service_name]

    def get_stub(self, service_name: str, stub_type: Type) -> object:
        """
        Gets or creates a gRPC stub for a given service and stub type.

        Args:
            service_name: The name of the service (e.g., 'market_analysis').
            stub_type: The generated gRPC stub class (e.g., market_analysis_service_pb2_grpc.MarketAnalysisServiceStub).

        Returns:
            An instance of the gRPC stub.
        """
        if service_name not in self._stubs:
            channel = self._get_channel(service_name)
            stub = stub_type(channel)
            self._stubs[service_name] = stub
        return self._stubs[service_name]

    def close_channels(self):
        """
        Closes all open gRPC channels.
        """\
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()
        self._stubs.clear()

# Example Usage (for demonstration)
# if __name__ == '__main__':
#     service_addresses = {
#         'market_analysis': 'localhost:50051',
#         'causal_analysis': 'localhost:50052',
#         # Add other service addresses
#     }
#     factory = GrpcClientFactory(service_addresses)

#     # Example of getting a Market Analysis Service stub
#     # try:
#     #     market_analysis_stub = factory.get_stub('market_analysis', market_analysis_service_pb2_grpc.MarketAnalysisServiceStub)
#     #     # Use the stub to make calls
#     #     # response = market_analysis_stub.SomeMethod(some_request_pb2.SomeRequest())
#     #     # print(response)
#     # except Exception as e:
#     #     print(f"Error getting Market Analysis stub: {e}")
#     # finally:
#     #     factory.close_channels()