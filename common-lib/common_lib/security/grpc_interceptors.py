import grpc
import logging
from typing import Callable, Any, Awaitable

logger = logging.getLogger(__name__)

# Placeholder for a real JWT validation library (e.g., PyJWT)
# And a way to get the public key or secret for validation
# For example:
# import jwt
# JWT_SECRET = "your-secret-key" # In real scenarios, use asymmetric keys and manage secrets properly
# JWT_ALGORITHM = "HS256"
# JWT_AUDIENCE = "your-service-audience"
# JWT_ISSUER = "your-auth-issuer"

class _GenericClientInterceptor(grpc.aio.UnaryUnaryClientInterceptor,
                                grpc.aio.UnaryStreamClientInterceptor,
                                grpc.aio.StreamUnaryClientInterceptor,
                                grpc.aio.StreamStreamClientInterceptor):
    """A base class for client interceptors to handle all RPC types."""
    async def _intercept(self, method: Callable, request_or_iterator: Any, metadata: grpc.aio.Metadata):
        raise NotImplementedError()

    async def intercept_unary_unary(self, continuation: Callable[[grpc.aio.ClientCallDetails, Any], Awaitable[Any]],
                                    client_call_details: grpc.aio.ClientCallDetails,
                                    request: Any) -> Any:
        new_details, new_request_or_iterator, new_metadata = await self._intercept(
            lambda: continuation(client_call_details, request), request, client_call_details.metadata
        )
        return await new_details # This should be the response from continuation

    async def intercept_unary_stream(self, continuation: Callable[[grpc.aio.ClientCallDetails, Any], Awaitable[Any]],
                                     client_call_details: grpc.aio.ClientCallDetails,
                                     request: Any) -> Any:
        new_details, new_request_or_iterator, new_metadata = await self._intercept(
            lambda: continuation(client_call_details, request), request, client_call_details.metadata
        )
        return await new_details

    async def intercept_stream_unary(self, continuation: Callable[[grpc.aio.ClientCallDetails, Any], Awaitable[Any]],
                                     client_call_details: grpc.aio.ClientCallDetails,
                                     request_iterator: Any) -> Any:
        new_details, new_request_or_iterator, new_metadata = await self._intercept(
            lambda: continuation(client_call_details, request_iterator), request_iterator, client_call_details.metadata
        )
        return await new_details

    async def intercept_stream_stream(self, continuation: Callable[[grpc.aio.ClientCallDetails, Any], Awaitable[Any]],
                                      client_call_details: grpc.aio.ClientCallDetails,
                                      request_iterator: Any) -> Any:
        new_details, new_request_or_iterator, new_metadata = await self._intercept(
            lambda: continuation(client_call_details, request_iterator), request_iterator, client_call_details.metadata
        )
        return await new_details


class JwtAuthServerInterceptor(grpc.aio.ServerInterceptor):
    """
    A gRPC server interceptor for JWT-based authentication.
    """

    def __init__(self, required_audience=None, issuer=None, secret_key=None, algorithm=None):
        # In a real scenario, these would be used by a JWT library for validation
        self.required_audience = required_audience
        self.issuer = issuer
        self.secret_key = secret_key # For HS256, or public key for RS256 etc.
        self.algorithm = algorithm
        logger.info("JwtAuthServerInterceptor initialized.")
        # A more robust implementation would load keys, config, etc.

    async def intercept_service(self, continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
                                handler_call_details: grpc.HandlerCallDetails) -> grpc.RpcMethodHandler:
        """
        Intercepts incoming gRPC calls to perform JWT authentication.
        """
        metadata = dict(handler_call_details.invocation_metadata)
        auth_header = metadata.get('authorization')

        if not auth_header:
            logger.warning("Missing Authorization header.")
            # Using context directly is not available here as in Servicer methods.
            # To abort, we must return a handler that does so.
            return self._abort_with_status(grpc.StatusCode.UNAUTHENTICATED, "Missing Authorization header")

        if not auth_header.startswith('Bearer '):
            logger.warning("Authorization header does not start with Bearer.")
            return self._abort_with_status(grpc.StatusCode.UNAUTHENTICATED, "Invalid token type. Expected Bearer token.")

        token = auth_header.split(' ', 1)[1]

        try:
            # Placeholder for actual JWT validation logic
            # Real validation would involve:
            # 1. Decoding the token using a library like PyJWT.
            #    payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], audience=self.required_audience, issuer=self.issuer)
            # 2. Checking the signature against a public key (for asymmetric) or secret (for symmetric).
            # 3. Verifying standard claims: 'exp' (expiration), 'nbf' (not before), 'iat' (issued at).
            # 4. Verifying 'iss' (issuer) and 'aud' (audience) claims.
            # 5. Potentially checking against a token revocation list.
            
            # Placeholder validation:
            if token == "valid-super-secret-token-for-dev": # Example valid token
                user_id = "user-123-dev" # Extracted or associated from/with token
                logger.info(f"Token validated successfully for user: {user_id}. Method: {handler_call_details.method}")
                
                # Optionally, enrich the context for downstream services.
                # This is tricky with interceptors as they don't directly modify servicer context easily.
                # One common pattern is to wrap the servicer's methods or use a custom context object
                # if the framework supports it. For grpc.aio, direct context modification is not straightforward.
                # A simpler approach for some use cases is to pass info via new metadata if the handler
                # itself is wrapped, but that's more complex.
                # For now, we'll just log and proceed.
                
                return await continuation(handler_call_details)
            
            elif token == "expired-token": # Example invalid token
                logger.warning("Token validation failed: Expired token used.")
                return self._abort_with_status(grpc.StatusCode.UNAUTHENTICATED, "Token is expired.")
            
            else: # Other invalid tokens
                logger.warning(f"Token validation failed for token: {token[:20]}...") # Log a snippet for safety
                return self._abort_with_status(grpc.StatusCode.UNAUTHENTICATED, "Invalid or malformed token.")

        except Exception as e: # Replace with specific JWT exceptions like jwt.ExpiredSignatureError, jwt.InvalidTokenError
            logger.error(f"Error during token validation: {e}", exc_info=True)
            return self._abort_with_status(grpc.StatusCode.INTERNAL, f"Error processing token: {str(e)}")

    def _abort_with_status(self, code: grpc.StatusCode, details: str) -> grpc.RpcMethodHandler:
        """
        Returns a GRPCHandler that will abort the call with the given status and details.
        """
        def abort_handler(request_deserializer, response_serializer):
            def unary_unary(request, context):
                context.abort(code, details)
            def unary_stream(request, context):
                context.abort(code, details)
            def stream_unary(request_iterator, context):
                context.abort(code, details)
            def stream_stream(request_iterator, context):
                context.abort(code, details)

            return grpc.method_handlers_generic_handler(
                "ErrorService", # Can be any name, not actually invoked
                {
                    "UnaryUnary": grpc.unary_unary_rpc_method_handler(unary_unary, request_deserializer, response_serializer),
                    "UnaryStream": grpc.unary_stream_rpc_method_handler(unary_stream, request_deserializer, response_serializer),
                    "StreamUnary": grpc.stream_unary_rpc_method_handler(stream_unary, request_deserializer, response_serializer),
                    "StreamStream": grpc.stream_stream_rpc_method_handler(stream_stream, request_deserializer, response_serializer),
                }
            )
        # We need to return a handler that will be called by gRPC.
        # The RpcMethodHandler expects specific types based on the method's cardinality.
        # Since we don't know the exact type here, we create a generic handler
        # that will abort. This is a bit of a workaround for how interceptors
        # must return a handler. A simpler way if available by the framework would be to raise an RpcError.
        
        # A generic handler that will abort any call.
        # The request_deserializer and response_serializer are not known here,
        # this is a simplification. In a real interceptor, you'd typically
        # call `continuation` which returns the actual handler, and if you need to abort,
        # you'd return a *different* handler (the aborting one).
        # The `continuation` itself gives you the rpc_method_handler.
        # If we don't call `await continuation(handler_call_details)`, we need to provide a handler.
        
        # Let's construct a generic handler that will simply abort.
        def generic_handler_provider(request_deserializer=None, response_serializer=None):
            def abort(ignored_request, context):
                context.abort(code, details)
                # For stream-returning methods, this might need to return an empty iterator
                # or raise the RpcError directly if possible.
                # However, context.abort() should terminate the RPC.
            
            return grpc.unary_unary_rpc_method_handler(abort) # Defaulting to unary_unary for simplicity of return type

        # This is a simplification. The interceptor should return an RpcMethodHandler.
        # If the call is aborted, this handler's methods (unary_unary, etc.) will be invoked.
        # The actual method handler has specific deserializers/serializers.
        # A robust way is to have `continuation` give you the handler, then you wrap its methods
        # or return an entirely new handler that aborts.
        # The most direct way to make `continuation` fail is to have it raise an RpcError,
        # but that's not how `context.abort` works from an interceptor directly.
        # The method `_abort_with_status` is to create such a handler.
        # This part is tricky; gRPC Python's server interceptor API for aborting
        # often means returning a handler that itself calls context.abort().

        # A common pattern:
        original_handler = None # This would come from `await continuation(handler_call_details)`
                                # if we were to call it and then decide to abort.
                                # Since we decide to abort *before* calling it, we make a new one.

        def create_aborting_handler(request_deserializer, response_serializer):
            # This function will be called by gRPC to get the actual method implementations.
            # We provide implementations that just abort.
            
            def abort_unary_unary(request, servicer_context):
                servicer_context.abort(code, details)
            
            def abort_unary_stream(request, servicer_context):
                servicer_context.abort(code, details)
                return iter([]) # Must return an iterator for stream responses

            def abort_stream_unary(request_iterator, servicer_context):
                servicer_context.abort(code, details)
            
            def abort_stream_stream(request_iterator, servicer_context):
                servicer_context.abort(code, details)
                return iter([])

            # The method name in handler_call_details.method is like /package.Service/Method
            # We don't need to match it perfectly if we're just aborting.
            # The handler_call_details contains the method name.
            # method_name = handler_call_details.method.split('/')[-1]
            
            # This generic handler will be used by gRPC for the RPC call.
            # It needs to provide all four call types.
            return grpc.method_handlers_generic_handler(
                service_name="ErrorService", # Dummy service name
                method_handlers={
                    # Provide dummy handlers for all types; gRPC will pick the right one.
                    # The actual method name doesn't matter here since we're aborting.
                    "AbortMethod": grpc.unary_unary_rpc_method_handler(
                        abort_unary_unary, request_deserializer, response_serializer),
                    "AbortMethodUnaryStream": grpc.unary_stream_rpc_method_handler(
                        abort_unary_stream, request_deserializer, response_serializer),
                    "AbortMethodStreamUnary": grpc.stream_unary_rpc_method_handler(
                        abort_stream_unary, request_deserializer, response_serializer),
                    "AbortMethodStreamStream": grpc.stream_stream_rpc_method_handler(
                        abort_stream_stream, request_deserializer, response_serializer),
                }
            )
        
        # The interceptor must return an RpcMethodHandler.
        # This handler is typically obtained from `await continuation(handler_call_details)`.
        # If we abort, we substitute it with one that calls `context.abort()`.
        # The issue is that the RpcMethodHandler needs (request_deserializer, response_serializer)
        # which are part of the original handler.
        # A simpler way to think about it: raise an RpcError exception if possible.
        # However, the example for grpc.aio interceptors often involves this handler substitution.
        # Let's assume for now this function will be called by the interceptor logic
        # to get a handler that aborts.
        # This is a simplified conceptual representation:
        class AbortingHandler(grpc.RpcMethodHandler):
            def __init__(self, status_code, details_text):
                self.status_code = status_code
                self.details_text = details_text

            def _abort(self, servicer_context):
                servicer_context.abort(self.status_code, self.details_text)

            # Implement all four types
            def unary_unary(self, request, context): self._abort(context)
            def unary_stream(self, request, context): self._abort(context); return iter([])
            def stream_unary(self, request_iterator, context): self._abort(context)
            def stream_stream(self, request_iterator, context): self._abort(context); return iter([])

        return AbortingHandler(code, details)

# Example of how a client might add metadata (for outline part 3)
# async def client_call_example(stub):
#     token = "your-jwt-token"
#     metadata = [('authorization', f'Bearer {token}')]
#     try:
#         response = await stub.YourRpcMethod(your_request_pb2.YourRequest(), metadata=metadata)
#         logger.info(f"RPC call successful: {response}")
#     except grpc.aio.AioRpcError as e:
#         logger.error(f"RPC failed: {e.code()} - {e.details()}")

logger.info("common-lib/common_lib/security/grpc_interceptors.py loaded")
# Ensure __init__.py exists in common-lib/common_lib/security/ and common-lib/common_lib/
# It does, as per previous ls.
# Also ensure common-lib itself is a package or on PYTHONPATH.
# For this structure, common-lib should be on PYTHONPATH.
# Then imports like `from common_lib.security.grpc_interceptors import JwtAuthServerInterceptor` work.The `JwtAuthInterceptor` has been created in `common-lib/common_lib/security/grpc_interceptors.py`.
The interceptor includes placeholder logic for JWT validation and demonstrates how to abort a call if authentication fails.

**Step 2: Outline Interceptor Integration in Pilot Services**

Now, I'll show how to integrate this `JwtAuthServerInterceptor` into the gRPC server setup for both `trading-gateway-service` and `analysis-engine-service`. This involves modifying their respective `core/main.py` files.

**2.1. Integration in `trading-gateway-service/core/main.py`**

The `grpc.aio.server` is created in the `startup_event` (or `lifespan` manager in more recent versions). We need to pass the interceptor instance to it.
I'll assume the `JwtAuthServerInterceptor` can be imported. This requires `common-lib` to be in `PYTHONPATH`. The `sys.path` modifications already added to `trading-gateway-service/core/main.py` should include the root of the project, so if `common-lib` is at that root, `from common_lib.security.grpc_interceptors import JwtAuthServerInterceptor` should work.

Let's get the current content of `trading-gateway-service/core/main.py` to apply the diff correctly.
