import grpc
import logging
from typing import Optional

# Attempt to import generated protobufs
try:
    from generated_protos import analysis_engine_pb2
    from generated_protos import analysis_engine_pb2_grpc
except ImportError:
    logging.error("Could not import generated_protos. Ensure it's in PYTHONPATH and generated correctly.")
    # Define dummy classes if import fails, to allow module loading for now
    # This helps in environments where stubs might not be immediately available during initial setup phases.
    class analysis_engine_pb2:
        class GetMarketOverviewRequest: pass
        class MarketOverviewResponse: pass
        class GetTradingOpportunityRequest: pass
        class TradingOpportunityResponse: pass
        class HealthCheckRequest: pass
        class HealthCheckResponse: pass
    class analysis_engine_pb2_grpc:
        class AnalysisEngineStub: pass

logger = logging.getLogger(__name__)

class AnalysisEngineGrpcClient:
    """
    A gRPC client for interacting with the Analysis Engine Service from the API Gateway.
    """
    def __init__(self, grpc_server_address: str):
        if not grpc_server_address:
            logger.error("gRPC server address for Analysis Engine is not configured for API Gateway.")
            raise ValueError("API Gateway: Analysis Engine gRPC server address cannot be empty.")
        
        self.grpc_server_address = grpc_server_address
        self._channel = grpc.aio.insecure_channel(self.grpc_server_address)
        logger.info(f"API Gateway: gRPC client for Analysis Engine initialized with address: {grpc_server_address}")

    async def _get_stub(self) -> analysis_engine_pb2_grpc.AnalysisEngineStub:
        """Returns a new stub instance."""
        return analysis_engine_pb2_grpc.AnalysisEngineStub(self._channel)

    async def close(self):
        """Closes the gRPC channel."""
        if self._channel:
            await self._channel.close()
            logger.info("API Gateway: gRPC channel to Analysis Engine closed.")

    async def get_market_overview(
        self, symbol: str, timeframe: str, lookback_days: int
    ) -> Optional[analysis_engine_pb2.MarketOverviewResponse]:
        logger.info(f"API Gateway: Requesting market overview for {symbol} via gRPC.")
        try:
            stub = await self._get_stub()
            request = analysis_engine_pb2.GetMarketOverviewRequest(
                symbol=symbol,
                timeframe=timeframe,
                lookback_days=lookback_days
            )
            response = await stub.GetMarketOverview(request, timeout=10) 
            logger.info(f"API Gateway: Received market overview for {symbol} via gRPC.")
            return response
        except grpc.aio.AioRpcError as e:
            logger.error(f"API Gateway: gRPC error getting market overview for {symbol}: {e.code()} - {e.details()}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"API Gateway: Unexpected error in get_market_overview for {symbol}: {e}", exc_info=True)
            return None

    async def get_trading_opportunity(
        self, symbol: str, timeframe: str, account_id: str, risk_percentage: float
    ) -> Optional[analysis_engine_pb2.TradingOpportunityResponse]:
        logger.info(f"API Gateway: Requesting trading opportunity for {symbol} via gRPC.")
        try:
            stub = await self._get_stub()
            request = analysis_engine_pb2.GetTradingOpportunityRequest(
                symbol=symbol,
                timeframe=timeframe,
                account_id=account_id,
                risk_percentage=risk_percentage
            )
            response = await stub.GetTradingOpportunity(request, timeout=10)
            logger.info(f"API Gateway: Received trading opportunity for {symbol} via gRPC.")
            return response
        except grpc.aio.AioRpcError as e:
            logger.error(f"API Gateway: gRPC error getting trading opportunity for {symbol}: {e.code()} - {e.details()}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"API Gateway: Unexpected error in get_trading_opportunity for {symbol}: {e}", exc_info=True)
            return None

    async def health_check_analysis_engine(self) -> Optional[analysis_engine_pb2.HealthCheckResponse]:
        logger.info("API Gateway: Requesting health check from Analysis Engine Service via gRPC.")
        try:
            stub = await self._get_stub()
            request = analysis_engine_pb2.HealthCheckRequest()
            response = await stub.HealthCheck(request, timeout=5)
            logger.info(f"API Gateway: Received health status from Analysis Engine: {response.status}")
            return response
        except grpc.aio.AioRpcError as e:
            logger.error(f"API Gateway: gRPC error during Analysis Engine health check: {e.code()} - {e.details()}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"API Gateway: Unexpected error in Analysis Engine health_check: {e}", exc_info=True)
            return None

# Example usage for testing (if run directly)
async def _test_client():
    import asyncio
    logging.basicConfig(level=logging.INFO)
    # This address should match the Analysis Engine's gRPC server
    # It should be configurable in a real application
    client = AnalysisEngineGrpcClient(grpc_server_address="localhost:50051") 
    
    try:
        health_status = await client.health_check_analysis_engine()
        if health_status:
            logging.info(f"Test: Analysis Engine Health Status: {health_status.status}")

        overview = await client.get_market_overview("EURUSD", "H1", 30)
        if overview:
            logging.info(f"Test: Market Overview for EURUSD (gRPC): Open={overview.market_data.open}")
        
        opportunity = await client.get_trading_opportunity("EURUSD", "M15", "test_account", 1.0)
        if opportunity:
            logging.info(f"Test: Trading Opportunity for EURUSD (gRPC): Entry={opportunity.entry_price}")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(_test_client())
