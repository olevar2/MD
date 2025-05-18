"""
Error handling interceptor for gRPC server.
"""

import logging
import traceback
import grpc
from typing import Callable, Any, Awaitable

from common_lib.grpc.common import error_types_pb2

logger = logging.getLogger(__name__)


class ErrorHandlingInterceptor(grpc.aio.ServerInterceptor):
    """
    Interceptor for handling errors in gRPC requests.
    """
    
    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """
        Intercept incoming requests to handle errors.
        
        Args:
            continuation: Function to continue the request processing
            handler_call_details: Details about the request
            
        Returns:
            The RPC method handler
        """
        # Get the RPC method handler
        handler = await continuation(handler_call_details)
        
        if handler and handler.request_streaming and handler.response_streaming:
            return await self._wrap_stream_stream_handler(handler)
        elif handler and handler.request_streaming:
            return await self._wrap_stream_unary_handler(handler)
        elif handler and handler.response_streaming:
            return await self._wrap_unary_stream_handler(handler)
        elif handler:
            return await self._wrap_unary_unary_handler(handler)
        else:
            return handler
    
    async def _wrap_unary_unary_handler(self, handler: grpc.RpcMethodHandler) -> grpc.RpcMethodHandler:
        """
        Wrap a unary-unary RPC method handler with error handling.
        
        Args:
            handler: The original RPC method handler
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.unary_unary
        
        async def _wrapped_unary_unary(request: Any, context: grpc.aio.ServicerContext) -> Any:
            try:
                # Call the original handler
                return await original_func(request, context)
                
            except Exception as e:
                # Handle the error
                await self._handle_error(e, context)
                return None
        
        return grpc.unary_unary_rpc_method_handler(
            _wrapped_unary_unary,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _wrap_unary_stream_handler(self, handler: grpc.RpcMethodHandler) -> grpc.RpcMethodHandler:
        """
        Wrap a unary-stream RPC method handler with error handling.
        
        Args:
            handler: The original RPC method handler
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.unary_stream
        
        async def _wrapped_unary_stream(request: Any, context: grpc.aio.ServicerContext) -> Any:
            try:
                # Call the original handler
                async for response in original_func(request, context):
                    yield response
                    
            except Exception as e:
                # Handle the error
                await self._handle_error(e, context)
        
        return grpc.unary_stream_rpc_method_handler(
            _wrapped_unary_stream,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _wrap_stream_unary_handler(self, handler: grpc.RpcMethodHandler) -> grpc.RpcMethodHandler:
        """
        Wrap a stream-unary RPC method handler with error handling.
        
        Args:
            handler: The original RPC method handler
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.stream_unary
        
        async def _wrapped_stream_unary(request_iterator: Any, context: grpc.aio.ServicerContext) -> Any:
            try:
                # Call the original handler
                return await original_func(request_iterator, context)
                
            except Exception as e:
                # Handle the error
                await self._handle_error(e, context)
                return None
        
        return grpc.stream_unary_rpc_method_handler(
            _wrapped_stream_unary,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _wrap_stream_stream_handler(self, handler: grpc.RpcMethodHandler) -> grpc.RpcMethodHandler:
        """
        Wrap a stream-stream RPC method handler with error handling.
        
        Args:
            handler: The original RPC method handler
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.stream_stream
        
        async def _wrapped_stream_stream(request_iterator: Any, context: grpc.aio.ServicerContext) -> Any:
            try:
                # Call the original handler
                async for response in original_func(request_iterator, context):
                    yield response
                    
            except Exception as e:
                # Handle the error
                await self._handle_error(e, context)
        
        return grpc.stream_stream_rpc_method_handler(
            _wrapped_stream_stream,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _handle_error(self, error: Exception, context: grpc.aio.ServicerContext) -> None:
        """
        Handle an error by setting the appropriate gRPC status code and details.
        
        Args:
            error: The exception that was raised
            context: The gRPC service context
        """
        # Log the error
        logger.error(f"gRPC error: {str(error)}", exc_info=True)
        
        # Get the error details
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        # Map the error to a gRPC status code
        status_code, error_code = self._map_error_to_status_code(error)
        
        # Set the status code and details
        context.set_code(status_code)
        context.set_details(error_message)
        
        # Add error details to the trailing metadata
        metadata = [
            ('error-type', error.__class__.__name__),
            ('error-code', str(error_code.value)),
            ('error-traceback', error_traceback)
        ]
        
        # Add the metadata
        for key, value in metadata:
            await context.add_trailing_metadata((key, value))
    
    def _map_error_to_status_code(self, error: Exception) -> tuple[grpc.StatusCode, error_types_pb2.ErrorCode]:
        """
        Map an exception to a gRPC status code and error code.
        
        Args:
            error: The exception to map
            
        Returns:
            A tuple of (gRPC status code, error code)
        """
        # Map common exceptions to gRPC status codes
        if isinstance(error, ValueError):
            return grpc.StatusCode.INVALID_ARGUMENT, error_types_pb2.ErrorCode.INVALID_INPUT
        elif isinstance(error, KeyError):
            return grpc.StatusCode.NOT_FOUND, error_types_pb2.ErrorCode.NOT_FOUND
        elif isinstance(error, PermissionError):
            return grpc.StatusCode.PERMISSION_DENIED, error_types_pb2.ErrorCode.AUTHORIZATION_FAILED
        elif isinstance(error, TimeoutError):
            return grpc.StatusCode.DEADLINE_EXCEEDED, error_types_pb2.ErrorCode.TIMEOUT
        elif isinstance(error, NotImplementedError):
            return grpc.StatusCode.UNIMPLEMENTED, error_types_pb2.ErrorCode.INTERNAL_ERROR
        else:
            # Default to internal error
            return grpc.StatusCode.INTERNAL, error_types_pb2.ErrorCode.INTERNAL_ERROR