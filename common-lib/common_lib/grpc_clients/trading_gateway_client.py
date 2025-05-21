import grpc
import logging
import time # For timestamp in example
from typing import Optional, List, Tuple

# Assuming 'generated_protos' is in PYTHONPATH.
# These paths match the output structure of the protoc command used earlier.
from common_pb2 import UUID as CommonUUID, Timestamp as CommonTimestamp
from trading_gateway_service import trading_gateway_pb2
from trading_gateway_service import trading_gateway_pb2_grpc

logger = logging.getLogger(__name__)

class TradingGatewayGRPCClient:
    """
    A gRPC client for interacting with the TradingGatewayService.
    """

    def __init__(self, server_address: str, jwt_token: Optional[str] = None, secure: bool = False, ca_cert: Optional[bytes] = None, client_cert: Optional[bytes] = None, client_key: Optional[bytes] = None):
        """
        Initializes the TradingGatewayGRPCClient.

        Args:
            server_address: The address (e.g., "localhost:50051") of the gRPC server.
            jwt_token: Optional JWT token for authentication.
            secure: If True, uses a secure channel. 'ca_cert' is required.
                    'client_cert' and 'client_key' are for mutual TLS.
            ca_cert: Bytes of the CA certificate for secure channel.
            client_cert: Bytes of the client certificate for mTLS.
            client_key: Bytes of the client private key for mTLS.
        """
        self.server_address = server_address
        self._jwt_token = jwt_token
        self._channel = None
        self.stub = None

        if secure:
            if not ca_cert:
                raise ValueError("CA certificate (ca_cert) is required for a secure channel.")
            credentials_list = [grpc.ssl_channel_credentials(ca_cert)]
            if client_cert and client_key:
                credentials_list.append(grpc.metadata_call_credentials(lambda context, callback: callback([('client_cert', client_cert), ('client_key', client_key)], None))) # This is not how mTLS is typically added for channel creds
                # Correct mTLS channel credentials setup:
                channel_credentials = grpc.ssl_channel_credentials(
                    root_certificates=ca_cert,
                    private_key=client_key,
                    certificate_chain=client_cert
                )
            else: # Server-side TLS only
                channel_credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)

            self._channel = grpc.aio.secure_channel(server_address, channel_credentials)
            logger.info(f"Created secure gRPC channel to {server_address}")
        else:
            self._channel = grpc.aio.insecure_channel(server_address)
            logger.info(f"Created insecure gRPC channel to {server_address}")
        
        self.stub = trading_gateway_pb2_grpc.TradingGatewayServiceStub(self._channel)
        logger.info("TradingGatewayServiceStub initialized.")

    def _get_call_metadata(self) -> Optional[List[Tuple[str, str]]]:
        """Prepares gRPC call metadata, including the JWT token if available."""
        if self._jwt_token:
            return [('authorization', f'Bearer {self._jwt_token}')]
        return None

    async def execute_order_async(
        self,
        order_id_value: str,
        instrument_symbol: str,
        order_type: trading_gateway_pb2.OrderType, # Use the enum directly
        quantity: float,
        price: Optional[float] = None # Optional for market orders
    ) -> Optional[trading_gateway_pb2.ExecutionReport]:
        """
        Asynchronously executes a trading order.

        Args:
            order_id_value: String value for the order's UUID.
            instrument_symbol: The symbol of the instrument to trade (e.g., "EURUSD").
            order_type: The type of order (MARKET, LIMIT) from trading_gateway_pb2.OrderType enum.
            quantity: The quantity to trade.
            price: The price for limit orders.

        Returns:
            An ExecutionReport if successful, None otherwise.
        """
        current_time = time.time()
        order_request = trading_gateway_pb2.OrderRequest(
            order_id=CommonUUID(value=order_id_value),
            instrument_symbol=instrument_symbol,
            order_type=order_type,
            quantity=quantity,
            timestamp=CommonTimestamp(seconds=int(current_time), nanos=int((current_time % 1) * 1e9))
        )
        if price is not None:
            order_request.price = price

        call_metadata = self._get_call_metadata()
        
        logger.info(f"Executing order: {order_request} with metadata: {call_metadata}")
        try:
            response = await self.stub.ExecuteOrder(order_request, metadata=call_metadata)
            logger.info(f"Order execution report received: {response}")
            return response
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC call to ExecuteOrder failed: {e.code()} - {e.details()}", exc_info=True)
            # Example: Re-raise as a custom exception or handle specific codes
            # if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            #     raise YourAuthError("Authentication failed") from e
            # elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            #     raise YourInvalidInputError(f"Invalid arguments: {e.details()}") from e
            return None # Or raise a custom exception
        except Exception as e:
            logger.error(f"An unexpected error occurred during ExecuteOrder: {e}", exc_info=True)
            return None # Or raise

    async def close(self):
        """Closes the gRPC channel."""
        if self._channel:
            await self._channel.close()
            logger.info(f"gRPC channel to {self.server_address} closed.")

# --- Conceptual Usage Example ---
# This part would typically be in another service or an example script.
# It's included here for completeness of the outline.

async def example_usage():
    """
    Conceptual example of how to use the TradingGatewayGRPCClient.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Assuming TradingGatewayService gRPC server is running on localhost:50051
    # And assuming you have a valid JWT token.
    # In a real scenario, the token would be obtained from an auth service/system.
    server_addr = "localhost:50051" # Replace with actual server address from config
    jwt = "valid-super-secret-token-for-dev" # Replace with a real token

    client = TradingGatewayGRPCClient(server_address=server_addr, jwt_token=jwt)

    try:
        # Example 1: Execute a market order
        logger.info("\n--- Example 1: Executing Market Order ---")
        market_order_report = await client.execute_order_async(
            order_id_value="market-order-001",
            instrument_symbol="EUR/USD",
            order_type=trading_gateway_pb2.OrderType.MARKET,
            quantity=1000.0
        )
        if market_order_report:
            logger.info(f"Market Order Execution Report: ID {market_order_report.execution_id.value}, Status {trading_gateway_pb2.OrderStatus.Name(market_order_report.status)}")

        # Example 2: Execute a limit order
        logger.info("\n--- Example 2: Executing Limit Order ---")
        limit_order_report = await client.execute_order_async(
            order_id_value="limit-order-002",
            instrument_symbol="BTC/USD",
            order_type=trading_gateway_pb2.OrderType.LIMIT,
            quantity=0.5,
            price=50000.00
        )
        if limit_order_report:
            logger.info(f"Limit Order Execution Report: ID {limit_order_report.execution_id.value}, Status {trading_gateway_pb2.OrderStatus.Name(limit_order_report.status)}")

        # Example 3: Order that might be rejected by server (e.g., invalid token if interceptor is strict)
        logger.info("\n--- Example 3: Order with potentially invalid parameters (or token) ---")
        # To test invalid token, you might re-initialize the client with a bad token:
        # bad_token_client = TradingGatewayGRPCClient(server_address=server_addr, jwt_token="invalid-token")
        # report = await bad_token_client.execute_order_async(...)
        # await bad_token_client.close()
        
        # For now, let's simulate an invalid argument by the client (e.g., negative quantity)
        # The server-side validation (not implemented in detail yet) or gRPC itself might catch this.
        # The current ExecuteOrder servicer placeholder doesn't do much validation.
        error_report = await client.execute_order_async(
            order_id_value="error-order-003",
            instrument_symbol="XYZ/INVALID", # Potentially invalid symbol
            order_type=trading_gateway_pb2.OrderType.MARKET,
            quantity=-100 # Invalid quantity
        )
        if error_report:
            logger.info(f"Error Order Report: ID {error_report.execution_id.value}, Status {trading_gateway_pb2.OrderStatus.Name(error_report.status)}, Message: {error_report.message}")
        else:
            logger.warning("Error order call did not return a report (likely failed due to gRPC error).")

    except Exception as e:
        logger.error(f"An error occurred in the example usage: {e}", exc_info=True)
    finally:
        await client.close()

# if __name__ == "__main__":
#     import asyncio
#     # This is needed to run the async example_usage function
#     # Ensure PYTHONPATH is set up correctly if running this file directly for testing:
#     # export PYTHONPATH=$PYTHONPATH:/path/to/project_root:/path/to/project_root/generated_protos
#     # (assuming common-lib is under project_root and generated_protos is also there)
#     # For common-lib, if it's installed as a package, this might not be needed.
#     # If common-lib is a directory at the same level as generated_protos,
#     # and project_root is the parent of both, then adding project_root to PYTHONPATH helps.
#
#     # Example sys.path adjustment for direct execution:
#     # import os, sys
#     # current_dir = os.path.dirname(os.path.abspath(__file__)) # .../common-lib/common_lib/grpc_clients
#     # common_lib_dir = os.path.dirname(os.path.dirname(current_dir)) # .../common-lib/common_lib
#     # project_root_common_lib = os.path.dirname(common_lib_dir) # .../common-lib (if this is project root for common-lib itself)
#     # project_root_main = os.path.dirname(project_root_common_lib) # Main project root that contains common-lib and generated_protos
#
#     # if project_root_main not in sys.path:
#     #    sys.path.insert(0, project_root_main)
#     # if os.path.join(project_root_main, "generated_protos") not in sys.path:
#     #    sys.path.insert(0, os.path.join(project_root_main, "generated_protos"))
#
#     asyncio.run(example_usage())

logger.info("common-lib/common_lib/grpc_clients/trading_gateway_client.py loaded")

# Ensure __init__.py exists in common-lib/common_lib/grpc_clients/
# To make `from common_lib.grpc_clients.trading_gateway_client import TradingGatewayGRPCClient` work,
# `common-lib/common_lib/__init__.py` and `common-lib/common_lib/grpc_clients/__init__.py` are needed.
# Also, `common-lib`'s parent directory should be in PYTHONPATH if `common-lib` itself isn't directly in PYTHONPATH.
# Or, if `common-lib` is intended to be an installable package.
# Given the structure, if `/app` is project root, and `common-lib` is `/app/common-lib`,
# then `/app` should be in PYTHONPATH.
#
# The `sys.path` modifications in the actual service `main.py` files (like trading-gateway-service/core/main.py)
# add the project root (`/app`) and `/app/generated_protos`.
# This makes `from common_lib.security...` and `from common_pb2...` work.
# So, `from common_lib.grpc_clients...` should also work.
# I will create the __init__.py for grpc_clients.
