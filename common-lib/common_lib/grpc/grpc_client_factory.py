import grpc

# Import generated gRPC stubs
from causal_analysis import causal_analysis_service_pb2_grpc
from backtesting import backtesting_service_pb2_grpc
from market_analysis import market_analysis_service_pb2_grpc
from analysis_coordinator import analysis_coordinator_service_pb2_grpc

class GrpcClientFactory:
    """
    Factory for creating configured gRPC client stubs.
    """

    def __init__(
        self,
        causal_analysis_service_address: str,
        backtesting_service_address: str,
        market_analysis_service_address: str,
        analysis_coordinator_service_address: str,
        timeout_seconds: int = 10,
        max_retries: int = 3,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initializes the GrpcClientFactory.

        Args:
            causal_analysis_service_address: Address of the Causal Analysis Service gRPC server.
            backtesting_service_address: Address of the Backtesting Service gRPC server.
            market_analysis_service_address: Address of the Market Analysis Service gRPC server.
            analysis_coordinator_service_address: Address of the Analysis Coordinator Service gRPC server.
            timeout_seconds: Default timeout for gRPC calls.
            max_retries: Maximum number of retries for gRPC calls.
            enable_circuit_breaker: Whether to enable circuit breaking.
        """
        self._channels = {}
        self._service_addresses = {
            'causal_analysis': causal_analysis_service_address,
            'backtesting': backtesting_service_address,
            'market_analysis': market_analysis_service_address,
            'analysis_coordinator': analysis_coordinator_service_address,
        }
        # Store resilience configurations
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._enable_circuit_breaker = enable_circuit_breaker

    def _get_channel(self, service_name: str):
        """
        Gets or creates a gRPC channel for a given service.
        Implements channel management and connection pooling.
        """
        if service_name not in self._channels:
            address = self._service_addresses.get(service_name)
            if not address:
                raise ValueError(f"Unknown service: {service_name}")

            # Configure channel options, including a basic timeout
            options = [
                ('grpc.max_send_message_length', -1), # -1 means unlimited
                ('grpc.max_receive_message_length', -1) # -1 means unlimited
            ]
            if self._timeout_seconds is not None:
                 options.append(('grpc.default_deadline', self._timeout_seconds))

            # For simplicity, creating a new insecure channel each time for now.
            # In a real implementation, connection pooling and secure channels would be used.
            # Resilience features like retries and circuit breakers would typically be
            # implemented using client-side interceptors on the channel.
            self._channels[service_name] = grpc.insecure_channel(address, options=options)

        return self._channels[service_name]

    def get_causal_analysis_client(self) -> causal_analysis_service_pb2_grpc.CausalAnalysisServiceStub:
        """Gets a gRPC client stub for the Causal Analysis Service."""
        channel = self._get_channel('causal_analysis')
        # Apply client-side resilience (e.g., interceptors for retries/circuit breaking)
        # This would involve creating and composing interceptors before creating the stub.
        return causal_analysis_service_pb2_grpc.CausalAnalysisServiceStub(channel)

    def get_backtesting_client(self) -> backtesting_service_pb2_grpc.BacktestingServiceStub:
        """Gets a gRPC client stub for the Backtesting Service."""
        channel = self._get_channel('backtesting')
        # Apply client-side resilience
        return backtesting_service_pb2_grpc.BacktestingServiceStub(channel)

    def get_market_analysis_client(self) -> market_analysis_service_pb2_grpc.MarketAnalysisServiceStub:
        """Gets a gRPC client stub for the Market Analysis Service."""
        channel = self._get_channel('market_analysis')
        # Apply client-side resilience
        return market_analysis_service_pb2_grpc.MarketAnalysisServiceStub(channel)

    def get_analysis_coordinator_client(self) -> analysis_coordinator_service_pb2_grpc.AnalysisCoordinatorServiceStub:
        """Gets a gRPC client stub for the Analysis Coordinator Service."""
        channel = self._get_channel('analysis_coordinator')
        # Apply client-side resilience
        return analysis_coordinator_service_pb2_grpc.AnalysisCoordinatorServiceStub(channel)

    def close_channels(self):
        """Closes all open gRPC channels."""
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()

# Example Usage (would typically be in a service's dependency injection setup):
# from common_lib.grpc.grpc_client_factory import GrpcClientFactory
# 
# # Assuming service addresses are loaded from configuration
# client_factory = GrpcClientFactory(
#     causal_analysis_service_address="localhost:50051",
#     backtesting_service_address="localhost:50052",
#     market_analysis_service_address="localhost:50053",
#     analysis_coordinator_service_address="localhost:50054",
#     timeout_seconds=5,
#     max_retries=2
# )
# 
# # Get a client for a specific service
# causal_analysis_client = client_factory.get_causal_analysis_client()
# 
# # Use the client
# # response = causal_analysis_client.SomeMethod(...)
# 
# # Close channels when done (e.g., during application shutdown)
# # client_factory.close_channels()
