"""
Tests for the resilience decorators.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from common_lib.errors import ServiceError, TimeoutError
from common_lib.resilience import (
    circuit_breaker,
    retry_with_backoff,
    timeout,
    bulkhead,
    with_resilience
)


class TestResilienceDecorators(unittest.TestCase):
    """Test case for resilience decorators."""

    def setUp(self):
        """Set up test case."""
        # Create mock logger
        self.mock_logger = MagicMock()

        # Create patchers
        self.circuit_breaker_patcher = patch('common_lib.resilience.decorators.CircuitBreaker')
        self.retry_policy_patcher = patch('common_lib.resilience.decorators.RetryPolicy')
        self.timeout_patcher = patch('common_lib.resilience.decorators.Timeout')
        self.bulkhead_patcher = patch('common_lib.resilience.decorators.Bulkhead')
        self.circuit_breaker_config_patcher = patch('common_lib.resilience.decorators.CircuitBreakerConfig')

        # Start patchers
        self.mock_circuit_breaker = self.circuit_breaker_patcher.start()
        self.mock_retry_policy = self.retry_policy_patcher.start()
        self.mock_timeout = self.timeout_patcher.start()
        self.mock_bulkhead = self.bulkhead_patcher.start()
        self.mock_circuit_breaker_config = self.circuit_breaker_config_patcher.start()

        # Configure mocks
        self.mock_circuit_breaker_instance = MagicMock()
        self.mock_retry_policy_instance = MagicMock()
        self.mock_timeout_instance = MagicMock()
        self.mock_bulkhead_instance = MagicMock()
        self.mock_circuit_breaker_config_instance = MagicMock()

        self.mock_circuit_breaker.return_value = self.mock_circuit_breaker_instance
        self.mock_retry_policy.return_value = self.mock_retry_policy_instance
        self.mock_timeout.return_value = self.mock_timeout_instance
        self.mock_bulkhead.return_value = self.mock_bulkhead_instance
        self.mock_circuit_breaker_config.return_value = self.mock_circuit_breaker_config_instance

        # Configure execute methods with AsyncMock
        async def async_execute_side_effect(f):
    """
    Async execute side effect.
    
    Args:
        f: Description of f
    
    """

            result = f()
            if asyncio.iscoroutine(result):
                return await result
            return result

        self.mock_circuit_breaker_instance.execute = AsyncMock(side_effect=async_execute_side_effect)
        self.mock_retry_policy_instance.execute = AsyncMock(side_effect=async_execute_side_effect)
        self.mock_timeout_instance.execute = AsyncMock(side_effect=async_execute_side_effect)
        self.mock_bulkhead_instance.execute = AsyncMock(side_effect=async_execute_side_effect)

    def tearDown(self):
        """Tear down test case."""
        # Stop patchers
        self.circuit_breaker_patcher.stop()
        self.retry_policy_patcher.stop()
        self.timeout_patcher.stop()
        self.bulkhead_patcher.stop()
        self.circuit_breaker_config_patcher.stop()

    def test_circuit_breaker_decorator(self):
        """Test circuit_breaker decorator."""
        # Define test function
        @circuit_breaker()
        def test_func():
            return "success"

        # Call function
        result = test_func()

        # Check result
        self.assertEqual(result, "success")

        # Check if circuit breaker config was created with correct parameters
        self.mock_circuit_breaker_config.assert_called_once()

        # Check if circuit breaker was created
        self.mock_circuit_breaker.assert_called_once()

        # Check if execute was called
        self.mock_circuit_breaker_instance.execute.assert_called()

    def test_retry_with_backoff_decorator(self):
        """Test retry_with_backoff decorator."""
        # Define test function
        @retry_with_backoff()
        def test_func():
            return "success"

        # Call function
        result = test_func()

        # Check result
        self.assertEqual(result, "success")

        # Check if retry policy was created with correct parameters
        self.mock_retry_policy.assert_called_once_with(
            retries=3,
            delay=1.0,
            max_delay=60.0,
            backoff=2.0,
            exceptions=None
        )

        # Check if execute was called
        self.mock_retry_policy_instance.execute.assert_called()

    def test_timeout_decorator(self):
        """Test timeout decorator."""
        # Define test function
        @timeout()
        def test_func():
            return "success"

        # Call function
        result = test_func()

        # Check result
        self.assertEqual(result, "success")

        # Check if timeout was created with correct parameters
        self.mock_timeout.assert_called_once_with(30.0, operation='test_func')

        # Check if execute was called
        self.mock_timeout_instance.execute.assert_called()

    def test_bulkhead_decorator(self):
        """Test bulkhead decorator."""
        # Define test function
        @bulkhead()
        def test_func():
            return "success"

        # Call function
        result = test_func()

        # Check result
        self.assertEqual(result, "success")

        # Check if bulkhead was created with correct parameters
        self.mock_bulkhead.assert_called_once_with(
            name='tests.resilience.test_decorators.test_func',
            max_concurrent_calls=10,
            max_queue_size=10
        )

        # Check if execute was called
        self.mock_bulkhead_instance.execute.assert_called()

    def test_with_resilience_decorator(self):
        """Test with_resilience decorator."""
        # Configure mocks to return the value directly
        async def async_success_side_effect(f):
    """
    Async success side effect.
    
    Args:
        f: Description of f
    
    """

            return "success"

        self.mock_circuit_breaker_instance.execute = AsyncMock(side_effect=async_success_side_effect)
        self.mock_retry_policy_instance.execute = AsyncMock(side_effect=async_success_side_effect)
        self.mock_timeout_instance.execute = AsyncMock(side_effect=async_success_side_effect)
        self.mock_bulkhead_instance.execute = AsyncMock(side_effect=async_success_side_effect)

        # Define test function
        @with_resilience()
        def test_func():
            return "success"

        # Call function
        result = test_func()

        # Check result
        self.assertEqual(result, "success")

        # Check if all resilience components were created
        self.mock_timeout.assert_called()
        self.mock_retry_policy.assert_called()
        self.mock_circuit_breaker.assert_called()
        self.mock_bulkhead.assert_called()

    def test_with_resilience_decorator_with_disabled_components(self):
        """Test with_resilience decorator with disabled components."""
        # Define test function
        @with_resilience(
            enable_circuit_breaker=False,
            enable_retry=False,
            enable_bulkhead=False,
            enable_timeout=False
        )
        def test_func():
            return "success"

        # Call function
        result = test_func()

        # Check result
        self.assertEqual(result, "success")

        # Check if no resilience components were created
        self.mock_timeout.assert_not_called()
        self.mock_retry_policy.assert_not_called()
        self.mock_circuit_breaker.assert_not_called()
        self.mock_bulkhead.assert_not_called()

    def test_with_resilience_decorator_with_error(self):
        """Test with_resilience decorator with error."""
        # Configure mock to raise an error
        self.mock_circuit_breaker_instance.execute.side_effect = ServiceError(
            message="Test error",
            service_name="test-service",
            operation="test-operation"
        )

        # Define test function
        @with_resilience(
            enable_bulkhead=False,  # Disable bulkhead to avoid nested event loop
            enable_retry=False,     # Disable retry to avoid multiple calls
            enable_timeout=False    # Disable timeout to simplify the test
        )
        def test_func():
            return "success"

        # Call function and check exception
        with self.assertRaises(ServiceError):
            test_func()

    def test_async_function_with_resilience(self):
        """Test with_resilience decorator with async function."""
        # Configure mocks to return the value directly
        async def async_success_side_effect(f):
    """
    Async success side effect.
    
    Args:
        f: Description of f
    
    """

            return "success"

        self.mock_circuit_breaker_instance.execute = AsyncMock(side_effect=async_success_side_effect)
        self.mock_retry_policy_instance.execute = AsyncMock(side_effect=async_success_side_effect)
        self.mock_timeout_instance.execute = AsyncMock(side_effect=async_success_side_effect)
        self.mock_bulkhead_instance.execute = AsyncMock(side_effect=async_success_side_effect)

        # Define test function
        @with_resilience()
        async def test_func():
            return "success"

        # Create a new event loop for this test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Call function
            result = loop.run_until_complete(test_func())

            # Check result
            self.assertEqual(result, "success")
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check if all resilience components were created
        self.mock_timeout.assert_called()
        self.mock_retry_policy.assert_called()
        self.mock_circuit_breaker.assert_called()
        self.mock_bulkhead.assert_called()


if __name__ == '__main__':
    unittest.main()
