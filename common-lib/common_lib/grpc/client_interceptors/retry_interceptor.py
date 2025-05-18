"""
Client-side gRPC interceptor for retries.
"""

import grpc
import asyncio
import logging
import time
from typing import Callable, Any, Awaitable

logger = logging.getLogger(__name__)

class RetryInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    """
    A client-side interceptor that adds retry logic to unary-unary gRPC calls.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        retryable_statuses: tuple = (
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.INTERNAL, # Consider carefully if INTERNAL should be retried
        )
    ):
        """
        Initializes the RetryInterceptor.

        Args:
            max_retries: The maximum number of times to retry a failed call.
            initial_backoff_seconds: The initial delay before the first retry.
            backoff_multiplier: The multiplier for the backoff delay for subsequent retries.
            retryable_statuses: A tuple of gRPC status codes that should trigger a retry.
        """
        self._max_retries = max_retries
        self._initial_backoff_seconds = initial_backoff_seconds
        self._backoff_multiplier = backoff_multiplier
        self._retryable_statuses = retryable_statuses
        self.logger = logger

    async def intercept_unary_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Awaitable[Any]],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        """
        Intercepts a unary-unary call to add retry logic.
        """
        method = client_call_details.method
        last_exception = None
        backoff_delay = self._initial_backoff_seconds

        for attempt in range(self._max_retries + 1):
            try:
                # Call the original method
                response = await continuation(client_call_details, request)
                # If successful, return the response immediately
                return response
            except grpc.RpcError as e:
                last_exception = e
                status_code = e.code()

                if status_code in self._retryable_statuses and attempt < self._max_retries:
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self._max_retries + 1} failed for method {method} with status {status_code}. "
                        f"Retrying in {backoff_delay:.2f} seconds..."
                    )
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= self._backoff_multiplier
                else:
                    # If not a retryable status or max retries reached, re-raise the exception
                    self.logger.error(
                        f"Method {method} failed after {attempt + 1} attempts with status {status_code}. "
                        f"Giving up."
                    )
                    raise
            except Exception as e:
                # Catch other unexpected exceptions and re-raise immediately
                last_exception = e
                self.logger.error(
                    f"Method {method} failed with unexpected error on attempt {attempt + 1}/{self._max_retries + 1}. "
                    f"Giving up."
                )
                raise

        # This part should theoretically not be reached if an exception is always raised
        # but included for completeness. Re-raise the last known exception.
        if last_exception:
             raise last_exception
        else:
             # Should not happen, but as a fallback
             raise Exception(f"Method {method} failed after {self._max_retries + 1} attempts without a specific exception.")

# Example Usage (in client factory or service client creation):
# from common_lib.grpc.client_interceptors.retry_interceptor import RetryInterceptor
# 
# retry_interceptor = RetryInterceptor(max_retries=3)
# channel = grpc.intercept_channel(original_channel, retry_interceptor)
# stub = YourServiceStub(channel)