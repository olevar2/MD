import grpc
import logging
from typing import Optional

# Assuming generated_protos is in the PYTHONPATH or directly accessible
# For a structured project, you might need to adjust import paths
# e.g., from analysis_coordinator_service.generated_protos import analysis_engine_pb2, analysis_engine_pb2_grpc
try:
    from generated_protos import analysis_engine_pb2
    from generated_protos import analysis_engine_pb2_grpc
except ImportError:
    # This fallback is for environments where the path might not be set up yet,
    # though in a real execution, this should be resolved by PYTHONPATH configuration.
    logging.error("Could not import generated_protos. Ensure it's in PYTHONPATH.")
    # Define dummy classes if import fails, to allow module loading for now
    class analysis_engine_pb2:
        class GetMarketOverviewRequest: pass
        class GetTradingOpportunityRequest: pass
    class analysis_engine_pb2_grpc:
        class AnalysisEngineStub: pass


logger = logging.getLogger(__name__)

class AnalysisEngineGrpcClient:
    """
    A gRPC client for interacting with the Analysis Engine Service.
    """
    def __init__(self, grpc_server_address: str):
        if not grpc_server_address:
            logger.error("gRPC server address for Analysis Engine is not configured.")
            raise ValueError("Analysis Engine gRPC server address cannot be empty.")
        
        self.grpc_server_address = grpc_server_address
        # Create the channel once per client instance
        self._channel = grpc.aio.insecure_channel(self.grpc_server_address)
        logger.info(f"gRPC client for Analysis Engine initialized with address: {grpc_server_address}")

    async def _get_stub(self) -> analysis_engine_pb2_grpc.AnalysisEngineStub:
        """Creates and returns a new stub. Useful if channel needs to be frequently checked or recreated."""
        # For now, we use the channel created in __init__.
        # If channel management becomes more complex (e.g., needing to recreate on certain errors),
        # this method could be expanded.
        return analysis_engine_pb2_grpc.AnalysisEngineStub(self._channel)

    async def close(self):
        """Closes the gRPC channel."""
        if self._channel:
            await self._channel.close()
            logger.info("gRPC channel to Analysis Engine closed.")

    async def get_market_overview(
        self, symbol: str, timeframe: str, lookback_days: int
    ) -> Optional[analysis_engine_pb2.MarketOverviewResponse]:
        """
        Calls the GetMarketOverview RPC on the Analysis Engine Service.
        """
        logger.info(f"Requesting market overview for {symbol}, timeframe {timeframe}, lookback {lookback_days} days.")
        try:
            stub = await self._get_stub()
            request = analysis_engine_pb2.GetMarketOverviewRequest(
                symbol=symbol,
                timeframe=timeframe,
                lookback_days=lookback_days
            )
            response = await stub.GetMarketOverview(request, timeout=10) # 10-second timeout
            logger.info(f"Received market overview for {symbol}.")
            return response
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC error while getting market overview for {symbol}: {e.code()} - {e.details()}", exc_info=True)
            # Depending on requirements, you might raise a custom error, return None, or a default object
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_market_overview for {symbol}: {e}", exc_info=True)
            return None

    async def get_trading_opportunity(
        self, symbol: str, timeframe: str, account_id: str, risk_percentage: float
    ) -> Optional[analysis_engine_pb2.TradingOpportunityResponse]:
        """
        Calls the GetTradingOpportunity RPC on the Analysis Engine Service.
        """
        logger.info(f"Requesting trading opportunity for {symbol}, account {account_id}, risk {risk_percentage}%.")
        try:
            stub = await self._get_stub()
            request = analysis_engine_pb2.GetTradingOpportunityRequest(
                symbol=symbol,
                timeframe=timeframe,
                account_id=account_id,
                risk_percentage=risk_percentage
            )
            response = await stub.GetTradingOpportunity(request, timeout=10) # 10-second timeout
            logger.info(f"Received trading opportunity for {symbol}.")
            return response
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC error while getting trading opportunity for {symbol}: {e.code()} - {e.details()}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_trading_opportunity for {symbol}: {e}", exc_info=True)
            return None

    async def health_check(self) -> Optional[analysis_engine_pb2.HealthCheckResponse]:
        """
        Calls the HealthCheck RPC on the Analysis Engine Service.
        """
        logger.info("Requesting health check from Analysis Engine Service.")
        try:
            stub = await self._get_stub()
            request = analysis_engine_pb2.HealthCheckRequest()
            response = await stub.HealthCheck(request, timeout=5) # 5-second timeout
            logger.info(f"Received health status: {response.status}")
            return response
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC error during health check: {e.code()} - {e.details()}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in health_check: {e}", exc_info=True)
            return None

# Example Usage (for testing purposes, typically not run directly like this in a service)
async def main():
    import asyncio
    logging.basicConfig(level=logging.INFO)
    # This address should match the Analysis Engine's gRPC server
    # It should be configurable in a real application
    client = AnalysisEngineGrpcClient(grpc_server_address="localhost:50051") 
    
    try:
        # Health Check
        health_response = await client.health_check()
        if health_response:
            logging.info(f"Analysis Engine Health: {health_response.status}")
        else:
            logging.warning("Analysis Engine Health Check failed.")

        # Market Overview
        overview_response = await client.get_market_overview(symbol="EURUSD", timeframe="H1", lookback_days=30)
        if overview_response:
            logging.info(f"Market Overview for EURUSD: Open={overview_response.market_data.open}, Close={overview_response.market_data.close}")
        else:
            logging.warning("Failed to get market overview for EURUSD.")

        # Trading Opportunity
        opportunity_response = await client.get_trading_opportunity(symbol="GBPUSD", timeframe="M15", account_id="acc123", risk_percentage=1.5)
        if opportunity_response:
            logging.info(f"Trading Opportunity for GBPUSD: Direction={opportunity_response.trade_direction}, Entry={opportunity_response.entry_price}")
        else:
            logging.warning("Failed to get trading opportunity for GBPUSD.")
            
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
