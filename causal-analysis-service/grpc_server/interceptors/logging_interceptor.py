"""
Logging interceptor for gRPC server.
"""

import logging
import time
import uuid
import grpc
from typing import Callable, Any, Awaitable

logger = logging.getLogger(__name__)


class LoggingInterceptor(grpc.aio.ServerInterceptor):
    """
    Interceptor for logging gRPC requests and responses.
    """
    
    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """
        Intercept incoming requests to log them.
        
        Args:
            continuation: Function to continue the request processing
            handler_call_details: Details about the request
            
        Returns:
            The RPC method handler
        """
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata)
        method_name = handler_call_details.method
        
        # Generate request ID if not present
        request_id = metadata.get('x-request-id', str(uuid.uuid4()))
        
        # Log the request
        logger.info(f"gRPC request received: method={method_name}, request_id={request_id}")
        
        # Get the RPC method handler
        handler = await continuation(handler_call_details)
        
        if handler and handler.request_streaming and handler.response_streaming:
            return await self._wrap_stream_stream_handler(handler, method_name, request_id)
        elif handler and handler.request_streaming:
            return await self._wrap_stream_unary_handler(handler, method_name, request_id)
        elif handler and handler.response_streaming:
            return await self._wrap_unary_stream_handler(handler, method_name, request_id)
        elif handler:
            return await self._wrap_unary_unary_handler(handler, method_name, request_id)
        else:
            return handler
    
    async def _wrap_unary_unary_handler(
        self,
        handler: grpc.RpcMethodHandler,
        method_name: str,
        request_id: str
    ) -> grpc.RpcMethodHandler:
        """
        Wrap a unary-unary RPC method handler with logging.
        
        Args:
            handler: The original RPC method handler
            method_name: The name of the RPC method
            request_id: The request ID
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.unary_unary
        
        async def _wrapped_unary_unary(request: Any, context: grpc.aio.ServicerContext) -> Any:
            start_time = time.time()
            
            try:
                # Add request ID to context
                context = await self._add_request_id_to_context(context, request_id)
                
                # Call the original handler
                response = await original_func(request, context)
                
                # Log the response
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"gRPC response sent: method={method_name}, request_id={request_id}, status=OK, elapsed_time={elapsed_time:.2f}ms")
                
                return response
                
            except Exception as e:
                # Log the error
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"gRPC error: method={method_name}, request_id={request_id}, error={str(e)}, elapsed_time={elapsed_time:.2f}ms", exc_info=True)
                raise
        
        return grpc.unary_unary_rpc_method_handler(
            _wrapped_unary_unary,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _wrap_unary_stream_handler(
        self,
        handler: grpc.RpcMethodHandler,
        method_name: str,
        request_id: str
    ) -> grpc.RpcMethodHandler:
        """
        Wrap a unary-stream RPC method handler with logging.
        
        Args:
            handler: The original RPC method handler
            method_name: The name of the RPC method
            request_id: The request ID
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.unary_stream
        
        async def _wrapped_unary_stream(request: Any, context: grpc.aio.ServicerContext) -> Any:
            start_time = time.time()
            
            try:
                # Add request ID to context
                context = await self._add_request_id_to_context(context, request_id)
                
                # Log the request
                logger.info(f"gRPC stream started: method={method_name}, request_id={request_id}")
                
                # Call the original handler
                async for response in original_func(request, context):
                    yield response
                
                # Log the response
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"gRPC stream completed: method={method_name}, request_id={request_id}, status=OK, elapsed_time={elapsed_time:.2f}ms")
                
            except Exception as e:
                # Log the error
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"gRPC stream error: method={method_name}, request_id={request_id}, error={str(e)}, elapsed_time={elapsed_time:.2f}ms", exc_info=True)
                raise
        
        return grpc.unary_stream_rpc_method_handler(
            _wrapped_unary_stream,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _wrap_stream_unary_handler(
        self,
        handler: grpc.RpcMethodHandler,
        method_name: str,
        request_id: str
    ) -> grpc.RpcMethodHandler:
        """
        Wrap a stream-unary RPC method handler with logging.
        
        Args:
            handler: The original RPC method handler
            method_name: The name of the RPC method
            request_id: The request ID
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.stream_unary
        
        async def _wrapped_stream_unary(request_iterator: Any, context: grpc.aio.ServicerContext) -> Any:
            start_time = time.time()
            
            try:
                # Add request ID to context
                context = await self._add_request_id_to_context(context, request_id)
                
                # Log the request
                logger.info(f"gRPC stream request started: method={method_name}, request_id={request_id}")
                
                # Call the original handler
                response = await original_func(request_iterator, context)
                
                # Log the response
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"gRPC response sent: method={method_name}, request_id={request_id}, status=OK, elapsed_time={elapsed_time:.2f}ms")
                
                return response
                
            except Exception as e:
                # Log the error
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"gRPC error: method={method_name}, request_id={request_id}, error={str(e)}, elapsed_time={elapsed_time:.2f}ms", exc_info=True)
                raise
        
        return grpc.stream_unary_rpc_method_handler(
            _wrapped_stream_unary,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _wrap_stream_stream_handler(
        self,
        handler: grpc.RpcMethodHandler,
        method_name: str,
        request_id: str
    ) -> grpc.RpcMethodHandler:
        """
        Wrap a stream-stream RPC method handler with logging.
        
        Args:
            handler: The original RPC method handler
            method_name: The name of the RPC method
            request_id: The request ID
            
        Returns:
            The wrapped RPC method handler
        """
        original_func = handler.stream_stream
        
        async def _wrapped_stream_stream(request_iterator: Any, context: grpc.aio.ServicerContext) -> Any:
            start_time = time.time()
            
            try:
                # Add request ID to context
                context = await self._add_request_id_to_context(context, request_id)
                
                # Log the request
                logger.info(f"gRPC bidirectional stream started: method={method_name}, request_id={request_id}")
                
                # Call the original handler
                async for response in original_func(request_iterator, context):
                    yield response
                
                # Log the response
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"gRPC bidirectional stream completed: method={method_name}, request_id={request_id}, status=OK, elapsed_time={elapsed_time:.2f}ms")
                
            except Exception as e:
                # Log the error
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"gRPC bidirectional stream error: method={method_name}, request_id={request_id}, error={str(e)}, elapsed_time={elapsed_time:.2f}ms", exc_info=True)
                raise
        
        return grpc.stream_stream_rpc_method_handler(
            _wrapped_stream_stream,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    async def _add_request_id_to_context(
        self,
        context: grpc.aio.ServicerContext,
        request_id: str
    ) -> grpc.aio.ServicerContext:
        """
        Add the request ID to the context metadata.
        
        Args:
            context: The gRPC service context
            request_id: The request ID
            
        Returns:
            The updated context
        """
        # Add request ID to outgoing metadata
        await context.send_initial_metadata((('x-request-id', request_id),))
        
        return context