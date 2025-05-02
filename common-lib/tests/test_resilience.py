"""
Unit tests for the resilience module.
"""

print("Importing test_resilience.py...")

import asyncio
import unittest
import time
import os
import sys
from unittest.mock import AsyncMock, patch

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_lib.resilience import (
    # Circuit breaker
    CircuitBreaker, CircuitBreakerConfig, CircuitState, create_circuit_breaker, CircuitBreakerOpen,
    
    # Retry policy
    retry_with_policy, RetryExhaustedException, register_common_retryable_exceptions,
    
    # Timeout handler
    timeout_handler, async_timeout, TimeoutError,
    
    # Bulkhead
    bulkhead, BulkheadFullException
)


class TestCircuitBreaker(unittest.TestCase):
    """Tests for the circuit breaker implementation."""
    
    def setUp(self):
        self.config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.1
        )
        self.cb = create_circuit_breaker(
            service_name="test-service",
            resource_name="test-resource",
            config=self.config
        )
        
    def test_circuit_breaker_opens_after_failures(self):
        """Test that a circuit breaks open after specified number of failures."""
        async def run_test():
            # Mock function that will fail
            mock_func = AsyncMock(side_effect=ValueError("Service unavailable"))
            
            # First attempt - should fail but circuit still closed
            with self.assertRaises(ValueError):
                await self.cb.execute(mock_func)
            self.assertEqual(self.cb.current_state, CircuitState.CLOSED)
            
            # Second attempt - should fail and circuit should open
            with self.assertRaises(ValueError):
                await self.cb.execute(mock_func)
            self.assertEqual(self.cb.current_state, CircuitState.OPEN)
            
            # Third attempt - circuit is open, should raise CircuitBreakerOpen
            with self.assertRaises(CircuitBreakerOpen):
                await self.cb.execute(mock_func)
        
        # Run the async test
        asyncio.run(run_test())
            
    def test_circuit_breaker_resets_after_timeout(self):
        """Test that a circuit resets to half-open after timeout."""
        async def run_test():
            # Mock function that will fail
            mock_func = AsyncMock(side_effect=ValueError("Service unavailable"))
            
            # First two attempts - circuit should open
            with self.assertRaises(ValueError):
                await self.cb.execute(mock_func)
            with self.assertRaises(ValueError):
                await self.cb.execute(mock_func)
            self.assertEqual(self.cb.current_state, CircuitState.OPEN)
            
            # Wait for reset timeout
            await asyncio.sleep(0.2)
            
            # Mock function that will succeed
            mock_func.side_effect = None
            mock_func.return_value = "success"
            
            # Next attempt should be allowed (half-open state)
            result = await self.cb.execute(mock_func)
            self.assertEqual(result, "success")
            self.assertEqual(self.cb.current_state, CircuitState.CLOSED)
        
        # Run the async test
        asyncio.run(run_test())


class TestRetryPolicy(unittest.TestCase):
    """Tests for the retry policy implementation."""
    
    def test_retry_with_policy_succeeds_eventually(self):
        """Test that retry_with_policy retries until success."""
        async def run_test():
            attempts = 0
            
            @retry_with_policy(
                max_attempts=3,
                base_delay=0.01,
                exceptions=[ValueError]
            )
            async def flaky_function():
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise ValueError("Temporary failure")
                return "success"
            
            result = await flaky_function()
            self.assertEqual(result, "success")
            self.assertEqual(attempts, 3)
        
        # Run the async test
        asyncio.run(run_test())
        
    def test_retry_exhausted(self):
        """Test that retry_with_policy raises RetryExhaustedException after max attempts."""
        async def run_test():
            @retry_with_policy(
                max_attempts=2,
                base_delay=0.01,
                exceptions=[ValueError]
            )
            async def failing_function():
                raise ValueError("Always fails")
            
            with self.assertRaises(RetryExhaustedException):
                await failing_function()
        
        # Run the async test
        asyncio.run(run_test())


class TestTimeout(unittest.TestCase):
    """Tests for the timeout handler implementation."""
    
    def test_timeout_handler_allows_fast_operations(self):
        """Test that timeout_handler allows operations that complete within timeout."""
        async def run_test():
            @timeout_handler(timeout_seconds=0.5)
            async def fast_operation():
                await asyncio.sleep(0.1)
                return "completed"
            
            result = await fast_operation()
            self.assertEqual(result, "completed")
        
        # Run the async test
        asyncio.run(run_test())
        
    def test_timeout_handler_stops_slow_operations(self):
        """Test that timeout_handler stops operations that exceed timeout."""
        async def run_test():
            @timeout_handler(timeout_seconds=0.1)
            async def slow_operation():
                await asyncio.sleep(0.5)
                return "completed"
            
            with self.assertRaises(TimeoutError):
                await slow_operation()
        
        # Run the async test
        asyncio.run(run_test())


class TestBulkhead(unittest.TestCase):
    """Tests for the bulkhead pattern implementation."""
    
    def test_bulkhead_limits_concurrent_executions(self):
        """Test that bulkhead limits the number of concurrent executions."""
        async def run_test():
            execution_count = 0
            max_concurrent_observed = 0
            
            @bulkhead(name="test-bulkhead", max_concurrent=2)
            async def guarded_operation(duration):
                nonlocal execution_count, max_concurrent_observed
                execution_count += 1
                max_concurrent_observed = max(max_concurrent_observed, execution_count)
                await asyncio.sleep(duration)
                execution_count -= 1
                return f"completed in {duration}s"
            
            # Start 5 operations
            tasks = [
                guarded_operation(0.1),  # Will complete quickly
                guarded_operation(0.2),  # Will complete quickly
                guarded_operation(0.3),  # Has to wait for a slot
                guarded_operation(0.1),  # Has to wait for a slot
                guarded_operation(0.2),  # Has to wait for a slot
            ]
            
            # Run all operations
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Check that all operations completed
            self.assertEqual(len(results), 5)
            self.assertEqual(execution_count, 0)
            
            # Check that max concurrent was limited to 2
            self.assertEqual(max_concurrent_observed, 2)
        
        # Run the async test
        asyncio.run(run_test())
        
    def test_bulkhead_rejects_when_full(self):
        """Test that bulkhead rejects executions when full and max_waiting is exceeded."""
        async def run_test():
            # Create a bulkhead that allows 1 concurrent and 1 waiting
            @bulkhead(name="test-bulkhead-full", max_concurrent=1, max_waiting=1)
            async def guarded_operation():
                await asyncio.sleep(0.2)
                return "completed"
            
            # Start first operation - should execute immediately
            task1 = asyncio.create_task(guarded_operation())
            
            # Give it time to start
            await asyncio.sleep(0.05)
            
            # Start second operation - should wait
            task2 = asyncio.create_task(guarded_operation())
            
            # Start third operation - should be rejected
            with self.assertRaises(BulkheadFullException):
                await guarded_operation()
            
            # Wait for tasks to complete
            await task1
            await task2
        
        # Run the async test
        asyncio.run(run_test())


# Add proper test discovery hook
if __name__ == "__main__":
    unittest.main()
