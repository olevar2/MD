"""
Authentication interceptor for gRPC server.
"""

import logging
import grpc
from typing import Callable, Any, Awaitable

logger = logging.getLogger(__name__)


class AuthInterceptor(grpc.aio.ServerInterceptor):
    """
    Interceptor for authenticating gRPC requests.
    """
    
    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """
        Intercept incoming requests to authenticate them.
        
        Args:
            continuation: Function to continue the request processing
            handler_call_details: Details about the request
            
        Returns:
            The RPC method handler
        """
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata)
        method_name = handler_call_details.method
        
        # Check for authentication token
        auth_token = metadata.get('authorization')
        
        # For now, we'll just log the authentication attempt
        # In a real implementation, we would validate the token
        if auth_token:
            logger.info(f"Authenticated request to {method_name}")
        else:
            logger.warning(f"Unauthenticated request to {method_name}")
            # For now, we'll allow unauthenticated requests
            # In a real implementation, we might reject them
            # return self._unauthenticated_response()
        
        # Continue with the request
        return await continuation(handler_call_details)
    
    def _unauthenticated_response(self) -> grpc.RpcMethodHandler:
        """
        Create a response for unauthenticated requests.
        
        Returns:
            An RPC method handler that rejects the request
        """
        async def _unary_unary(request: Any, context: grpc.aio.ServicerContext) -> Any:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Authentication required")
            return None
        
        return grpc.unary_unary_rpc_method_handler(_unary_unary)