"""
End-to-End Test Suite for Forex Trading Platform

This module provides a comprehensive end-to-end testing framework for the
Forex trading platform, validating the integration of all components from
market data ingestion through signal generation, order execution, and
portfolio management. It ensures that the entire system works together
as expected in various scenarios.

Key capabilities:
1. Testing the full trading lifecycle across multiple services
2. Validating signal generation through to order execution
3. Testing feedback loops and adaptation mechanisms
4. Simulating various market conditions and edge cases
5. Validating degraded mode operations and resilience features
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Configure logger
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Possible outcomes of a test."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"      # Test encountered an error
    SKIPPED = "SKIPPED"  # Test was skipped
    TIMEOUT = "TIMEOUT"  # Test timed out


@dataclass
class TestCase:
    """Base class for all test cases."""
    name: str
    description: str
    timeout_seconds: float = 60.0  # Default timeout
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    async def setup(self) -> None:
        """Set up the test environment."""
        pass
        
    async def execute(self) -> Tuple[TestResult, str]:
        """
        Execute the test.
        
        Returns:
            Tuple of (result, message)
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    async def teardown(self) -> None:
        """Clean up after the test."""
        pass


@dataclass
class TestSuite:
    """A collection of related test cases."""
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_hooks: List[Callable[[], None]] = field(default_factory=list)
    teardown_hooks: List[Callable[[], None]] = field(default_factory=list)
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)


@dataclass
class TestReport:
    """Report containing test results."""
    suite_name: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    results: Dict[str, Tuple[TestResult, str]] = field(default_factory=dict)
    environment_info: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate the duration of the test run in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def summary(self) -> Dict[str, int]:
        """Generate a summary of test results."""
        result_counts = {result.value: 0 for result in TestResult}
        for result, _ in self.results.values():
            result_counts[result.value] += 1
        return result_counts
    
    def to_json(self) -> str:
        """Convert the report to JSON format."""
        report_dict = {
            "suite_name": self.suite_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "results": {
                name: {
                    "result": result.value,
                    "message": message
                }
                for name, (result, message) in self.results.items()
            },
            "environment_info": self.environment_info,
            "summary": self.summary
        }
        return json.dumps(report_dict, indent=2)


class TestRunner:
    """Runner for executing test suites."""
    
    def __init__(self, parallelism: int = 1):
        """
        Initialize the test runner.
        
        Args:
            parallelism: Number of tests to run in parallel
        """
        self.parallelism = parallelism
        
    async def run_test_case(
        self, 
        test_case: TestCase
    ) -> Tuple[TestResult, str]:
        """
        Run a single test case with timeout.
        
        Args:
            test_case: The test case to run
            
        Returns:
            Tuple of (result, message)
        """
        logger.info(f"Starting test: {test_case.name}")
        
        try:
            # Setup
            await test_case.setup()
            
            # Execute with timeout
            try:
                result, message = await asyncio.wait_for(
                    test_case.execute(),
                    timeout=test_case.timeout_seconds
                )
            except asyncio.TimeoutError:
                result = TestResult.TIMEOUT
                message = f"Test timed out after {test_case.timeout_seconds} seconds"
            
            logger.info(f"Test {test_case.name}: {result.value} - {message}")
            return result, message
            
        except Exception as e:
            logger.exception(f"Error running test {test_case.name}")
            return TestResult.ERROR, f"Error: {str(e)}"
        finally:
            # Teardown
            try:
                await test_case.teardown()
            except Exception as e:
                logger.error(f"Error in teardown for {test_case.name}: {e}")
    
    async def run_suite(self, test_suite: TestSuite) -> TestReport:
        """
        Run all test cases in a suite.
        
        Args:
            test_suite: The test suite to run
            
        Returns:
            TestReport with results
        """
        # Create report
        report = TestReport(
            suite_name=test_suite.name,
            start_time=datetime.datetime.utcnow(),
            environment_info=self._collect_environment_info()
        )
        
        # Run setup hooks
        for hook in test_suite.setup_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Error in setup hook: {e}")
        
        # Run test cases
        if self.parallelism > 1:
            # Run tests in parallel
            tasks = [
                self.run_test_case(test_case) 
                for test_case in test_suite.test_cases
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for test_case, result in zip(test_suite.test_cases, results):
                if isinstance(result, Exception):
                    report.results[test_case.name] = (
                        TestResult.ERROR, 
                        f"Error: {str(result)}"
                    )
                else:
                    report.results[test_case.name] = result
        else:
            # Run tests sequentially
            for test_case in test_suite.test_cases:
                result = await self.run_test_case(test_case)
                report.results[test_case.name] = result
        
        # Run teardown hooks
        for hook in test_suite.teardown_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Error in teardown hook: {e}")
        
        # Complete report
        report.end_time = datetime.datetime.utcnow()
        return report
    
    def _collect_environment_info(self) -> Dict[str, str]:
        """Collect information about the test environment."""
        info = {}
        
        # Python version
        import platform
        info["python_version"] = platform.python_version()
        info["platform"] = platform.platform()
        
        # Environment variables (filtering sensitive ones)
        for key, value in os.environ.items():
            if not any(x in key.lower() for x in ["secret", "password", "token", "key"]):
                if key.startswith("FOREX_") or key.startswith("TEST_"):
                    info[f"env_{key}"] = value
        
        # Test runner settings
        info["parallelism"] = str(self.parallelism)
        info["timestamp"] = datetime.datetime.utcnow().isoformat()
        
        return info


# Specific test cases for the Forex Trading Platform

class MarketDataFlowTestCase(TestCase):
    """Test the flow of market data through the system."""
    
    def __init__(
        self,
        data_service_url: str,
        market_data_symbols: List[str],
        expected_latency_ms: float = 100.0
    ):
        """
        Initialize the test case.
        
        Args:
            data_service_url: URL of the data pipeline service
            market_data_symbols: List of symbols to test
            expected_latency_ms: Expected maximum latency in milliseconds
        """
        super().__init__(
            name="Market Data Flow Test",
            description="Validates that market data flows correctly from source to consumers",
            tags=["market-data", "integration"]
        )
        self.data_service_url = data_service_url
        self.market_data_symbols = market_data_symbols
        self.expected_latency_ms = expected_latency_ms
        
    async def setup(self) -> None:
        """Set up the test environment."""
        # Import here to avoid dependency issues
        import aiohttp
        self.session = aiohttp.ClientSession()
        
    async def execute(self) -> Tuple[TestResult, str]:
        """Execute the test."""
        import aiohttp
        
        try:
            # Send request to trigger market data flow
            request_time = time.time()
            async with self.session.post(
                f"{self.data_service_url}/api/v1/market-data/request",
                json={
                    "symbols": self.market_data_symbols,
                    "test_id": str(uuid.uuid4())
                }
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"API returned status {response.status}"
                
                response_data = await response.json()
                
            # Check response format
            if not isinstance(response_data, dict) or "request_id" not in response_data:
                return TestResult.FAILED, "Invalid response format"
                
            request_id = response_data["request_id"]
            
            # Poll for results
            max_polls = 10
            polls = 0
            while polls < max_polls:
                async with self.session.get(
                    f"{self.data_service_url}/api/v1/market-data/status/{request_id}"
                ) as status_response:
                    if status_response.status != 200:
                        return TestResult.FAILED, f"Status API returned {status_response.status}"
                    
                    status_data = await status_response.json()
                    if status_data.get("status") == "complete":
                        break
                        
                await asyncio.sleep(0.5)
                polls += 1
                
            if polls >= max_polls:
                return TestResult.TIMEOUT, "Timed out waiting for market data flow completion"
                
            # Check results
            async with self.session.get(
                f"{self.data_service_url}/api/v1/market-data/results/{request_id}"
            ) as results_response:
                if results_response.status != 200:
                    return TestResult.FAILED, f"Results API returned {results_response.status}"
                
                results_data = await results_response.json()
                
            # Validate results
            if "latency_ms" not in results_data:
                return TestResult.FAILED, "Results missing latency information"
                
            latency_ms = results_data["latency_ms"]
            if latency_ms > self.expected_latency_ms:
                return TestResult.FAILED, (
                    f"Market data latency ({latency_ms}ms) exceeds "
                    f"expected maximum ({self.expected_latency_ms}ms)"
                )
                
            # Check symbol coverage
            received_symbols = set(results_data.get("symbols", []))
            requested_symbols = set(self.market_data_symbols)
            
            if not received_symbols.issuperset(requested_symbols):
                missing_symbols = requested_symbols - received_symbols
                return TestResult.FAILED, f"Missing data for symbols: {missing_symbols}"
                
            return TestResult.PASSED, (
                f"Market data flow completed successfully with {latency_ms}ms latency"
            )
                
        except aiohttp.ClientError as e:
            return TestResult.ERROR, f"HTTP client error: {e}"
        except Exception as e:
            return TestResult.ERROR, f"Unexpected error: {e}"
            
    async def teardown(self) -> None:
        """Clean up after the test."""
        await self.session.close()


class OrderExecutionTestCase(TestCase):
    """Test the complete order execution flow."""
    
    def __init__(
        self,
        trading_api_url: str,
        symbol: str = "EUR/USD",
        order_type: str = "MARKET",
        side: str = "BUY",
        quantity: float = 10000.0,
        max_execution_time_ms: float = 500.0
    ):
        """
        Initialize the test case.
        
        Args:
            trading_api_url: URL of the trading API
            symbol: Symbol to trade
            order_type: Type of order to place
            side: Order side (BUY/SELL)
            quantity: Order quantity
            max_execution_time_ms: Maximum acceptable execution time
        """
        super().__init__(
            name="Order Execution Flow Test",
            description="Tests the complete order execution flow from submission to fill",
            tags=["order-execution", "integration"]
        )
        self.trading_api_url = trading_api_url
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.max_execution_time_ms = max_execution_time_ms
        
    async def setup(self) -> None:
        """Set up the test environment."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.order_id = None
        
    async def execute(self) -> Tuple[TestResult, str]:
        """Execute the test."""
        import aiohttp
        
        try:
            # Create order
            order_request = {
                "symbol": self.symbol,
                "type": self.order_type,
                "side": self.side,
                "quantity": self.quantity,
                "test_mode": True,  # Use test mode to avoid real orders
                "client_order_id": f"test-{uuid.uuid4()}"
            }
            
            start_time = time.time()
            
            async with self.session.post(
                f"{self.trading_api_url}/api/v1/orders",
                json=order_request
            ) as response:
                if response.status != 201:
                    return TestResult.FAILED, f"Order creation failed with status {response.status}"
                
                order_data = await response.json()
                
            if "order_id" not in order_data:
                return TestResult.FAILED, "Order creation response missing order ID"
                
            self.order_id = order_data["order_id"]
            
            # Poll for order status until filled or timeout
            max_polls = 20
            polls = 0
            
            while polls < max_polls:
                async with self.session.get(
                    f"{self.trading_api_url}/api/v1/orders/{self.order_id}"
                ) as status_response:
                    if status_response.status != 200:
                        return TestResult.FAILED, f"Order status check failed with status {status_response.status}"
                    
                    status_data = await status_response.json()
                    order_status = status_data.get("status")
                    
                    if order_status == "FILLED":
                        break
                    elif order_status in ["REJECTED", "CANCELLED", "EXPIRED"]:
                        return TestResult.FAILED, f"Order failed with status {order_status}: {status_data.get('reason', 'No reason provided')}"
                        
                await asyncio.sleep(0.25)
                polls += 1
                
            if polls >= max_polls:
                return TestResult.TIMEOUT, "Timed out waiting for order to fill"
                
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            if execution_time_ms > self.max_execution_time_ms:
                return TestResult.FAILED, (
                    f"Order execution time ({execution_time_ms:.2f}ms) exceeds "
                    f"maximum expected time ({self.max_execution_time_ms}ms)"
                )
                
            # Verify fill details
            if "fill_price" not in status_data:
                return TestResult.FAILED, "Fill details missing from order status"
                
            return TestResult.PASSED, (
                f"Order executed successfully in {execution_time_ms:.2f}ms "
                f"at price {status_data['fill_price']}"
            )
                
        except aiohttp.ClientError as e:
            return TestResult.ERROR, f"HTTP client error: {e}"
        except Exception as e:
            return TestResult.ERROR, f"Unexpected error: {e}"
            
    async def teardown(self) -> None:
        """Clean up after the test."""
        # Cancel order if still active
        if self.order_id:
            try:
                await self.session.delete(
                    f"{self.trading_api_url}/api/v1/orders/{self.order_id}"
                )
            except:
                pass  # Ignore errors in cleanup
        await self.session.close()


class FeedbackLoopTestCase(TestCase):
    """Test the feedback loop system for strategy adaptation."""
    
    def __init__(
        self,
        strategy_api_url: str,
        ml_api_url: str,
        strategy_id: str,
        parameter_name: str,
        feedback_type: str = "performance",
        feedback_value: float = 0.8
    ):
        """
        Initialize the test case.
        
        Args:
            strategy_api_url: URL of the strategy service API
            ml_api_url: URL of the ML service API
            strategy_id: ID of the strategy to test
            parameter_name: Name of the parameter to monitor for adaptation
            feedback_type: Type of feedback to provide
            feedback_value: Value of the feedback (typically 0.0-1.0)
        """
        super().__init__(
            name="Feedback Loop Integration Test",
            description="Tests the complete feedback loop from trading outcome to model adaptation",
            tags=["feedback_loop", "adaptation", "ml"],
            timeout_seconds=120.0  # Longer timeout for adaptation process
        )
        self.strategy_api_url = strategy_api_url
        self.ml_api_url = ml_api_url
        self.strategy_id = strategy_id
        self.parameter_name = parameter_name
        self.feedback_type = feedback_type
        self.feedback_value = feedback_value
        
    async def setup(self) -> None:
        """Set up the test environment."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        
        # Get initial parameter value
        async with self.session.get(
            f"{self.strategy_api_url}/api/v1/strategies/{self.strategy_id}/parameters"
        ) as response:
            params_data = await response.json()
            self.initial_param_value = params_data.get("parameters", {}).get(self.parameter_name)
            
        if self.initial_param_value is None:
            raise ValueError(f"Parameter {self.parameter_name} not found in strategy {self.strategy_id}")
        
    async def execute(self) -> Tuple[TestResult, str]:
        """Execute the test."""
        import aiohttp
        
        try:
            # Submit feedback
            feedback_data = {
                "strategy_id": self.strategy_id,
                "type": self.feedback_type,
                "value": self.feedback_value,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "source": "e2e_test",
                "metadata": {
                    "test_id": str(uuid.uuid4())
                }
            }
            
            async with self.session.post(
                f"{self.ml_api_url}/api/v1/feedback",
                json=feedback_data
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"Feedback submission failed with status {response.status}"
                
                submission_data = await response.json()
                
            feedback_id = submission_data.get("feedback_id")
            if not feedback_id:
                return TestResult.FAILED, "Feedback submission response missing ID"
                
            # Monitor for adaptation
            adaptation_detected = False
            max_polls = 20
            polls = 0
            
            while polls < max_polls:
                # Check if parameter was updated
                async with self.session.get(
                    f"{self.strategy_api_url}/api/v1/strategies/{self.strategy_id}/parameters"
                ) as response:
                    params_data = await response.json()
                    current_param_value = params_data.get("parameters", {}).get(self.parameter_name)
                    
                if current_param_value != self.initial_param_value:
                    adaptation_detected = True
                    break
                    
                # Check adaptation status
                async with self.session.get(
                    f"{self.ml_api_url}/api/v1/feedback/{feedback_id}/status"
                ) as response:
                    if response.status != 200:
                        return TestResult.FAILED, f"Feedback status check failed with status {response.status}"
                    
                    status_data = await response.json()
                    
                if status_data.get("status") == "processed":
                    # One more check for parameter update
                    async with self.session.get(
                        f"{self.strategy_api_url}/api/v1/strategies/{self.strategy_id}/parameters"
                    ) as response:
                        params_data = await response.json()
                        current_param_value = params_data.get("parameters", {}).get(self.parameter_name)
                        
                    if current_param_value != self.initial_param_value:
                        adaptation_detected = True
                        break
                    else:
                        return TestResult.FAILED, "Feedback was processed but parameter was not updated"
                        
                await asyncio.sleep(1)
                polls += 1
                
            if not adaptation_detected:
                return TestResult.TIMEOUT, "Timed out waiting for parameter adaptation"
                
            # Get adaptation details
            async with self.session.get(
                f"{self.ml_api_url}/api/v1/feedback/{feedback_id}/details"
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"Feedback details check failed with status {response.status}"
                
                details_data = await response.json()
                
            return TestResult.PASSED, (
                f"Parameter adapted from {self.initial_param_value} to {current_param_value} "
                f"based on feedback (reason: {details_data.get('adaptation_reason', 'unknown')})"
            )
                
        except aiohttp.ClientError as e:
            return TestResult.ERROR, f"HTTP client error: {e}"
        except Exception as e:
            return TestResult.ERROR, f"Unexpected error: {e}"
            
    async def teardown(self) -> None:
        """Clean up after the test."""
        # Reset parameter to original value if changed
        try:
            await self.session.put(
                f"{self.strategy_api_url}/api/v1/strategies/{self.strategy_id}/parameters",
                json={"parameters": {self.parameter_name: self.initial_param_value}}
            )
        except:
            pass  # Ignore errors in cleanup
            
        await self.session.close()


class DegradedModeTestCase(TestCase):
    """Test the system's behavior in degraded mode."""
    
    def __init__(
        self,
        trading_gateway_url: str,
        health_api_url: str,
        degradation_level: str = "MODERATE",
        degradation_reason: str = "BROKER_CONNECTIVITY"
    ):
        """
        Initialize the test case.
        
        Args:
            trading_gateway_url: URL of the trading gateway service
            health_api_url: URL of the health monitoring API
            degradation_level: Level of degradation to test
            degradation_reason: Reason for degradation
        """
        super().__init__(
            name="Degraded Mode Test",
            description="Tests the system's behavior under degraded operational conditions",
            tags=["resilience", "degraded-mode"],
            timeout_seconds=90.0
        )
        self.trading_gateway_url = trading_gateway_url
        self.health_api_url = health_api_url
        self.degradation_level = degradation_level
        self.degradation_reason = degradation_reason
        
    async def setup(self) -> None:
        """Set up the test environment."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        
    async def execute(self) -> Tuple[TestResult, str]:
        """Execute the test."""
        import aiohttp
        
        try:
            # Trigger degraded mode
            degrade_request = {
                "level": self.degradation_level,
                "reason": self.degradation_reason,
                "message": "Triggered by end-to-end test",
                "test_mode": True
            }
            
            async with self.session.post(
                f"{self.trading_gateway_url}/api/v1/admin/degrade",
                json=degrade_request
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"Failed to trigger degraded mode, status {response.status}"
                
            # Wait for degradation to take effect
            await asyncio.sleep(2)
            
            # Verify degraded status is reported
            async with self.session.get(
                f"{self.health_api_url}/api/v1/services/trading-gateway-service"
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"Health check failed with status {response.status}"
                
                health_data = await response.json()
                
            service_status = health_data.get("status")
            if service_status != "degraded":
                return TestResult.FAILED, f"Service status is {service_status}, expected 'degraded'"
                
            # Check active fallbacks
            async with self.session.get(
                f"{self.trading_gateway_url}/api/v1/admin/degraded-status"
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"Degraded status check failed with status {response.status}"
                
                status_data = await response.json()
                
            if "active_fallbacks" not in status_data:
                return TestResult.FAILED, "Degraded status missing active fallbacks"
                
            active_fallbacks = status_data["active_fallbacks"]
            if not active_fallbacks:
                return TestResult.FAILED, "No active fallbacks in degraded mode"
                
            # Test operation prioritization
            operations = {
                "critical": {
                    "endpoint": f"{self.trading_gateway_url}/api/v1/orders/emergency-close",
                    "payload": {"position_id": "test-position", "test_mode": True},
                    "expected_status": 200
                },
                "low": {
                    "endpoint": f"{self.trading_gateway_url}/api/v1/reports/generate",
                    "payload": {"report_type": "performance", "test_mode": True},
                    "expected_status": 503  # Should be rejected in degraded mode
                }
            }
            
            results = {}
            
            for priority, operation in operations.items():
                async with self.session.post(
                    operation["endpoint"],
                    json=operation["payload"]
                ) as response:
                    results[priority] = (
                        response.status == operation["expected_status"],
                        response.status
                    )
                    
            if not results["critical"][0]:
                return TestResult.FAILED, f"Critical operation failed with status {results['critical'][1]}"
                
            if not results["low"][0]:
                return TestResult.FAILED, f"Low priority operation should have been rejected, got status {results['low'][1]}"
                
            # Reset degraded mode
            async with self.session.post(
                f"{self.trading_gateway_url}/api/v1/admin/reset-degraded"
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, "Failed to reset degraded mode"
                
            # Wait for reset to take effect
            await asyncio.sleep(2)
            
            # Verify service is back to normal
            async with self.session.get(
                f"{self.health_api_url}/api/v1/services/trading-gateway-service"
            ) as response:
                health_data = await response.json()
                
            if health_data.get("status") != "healthy":
                return TestResult.FAILED, f"Service not back to healthy status after reset"
                
            return TestResult.PASSED, (
                f"Service correctly handled degraded mode with level {self.degradation_level}, "
                f"reason {self.degradation_reason}, and active fallbacks: {', '.join(active_fallbacks)}"
            )
                
        except aiohttp.ClientError as e:
            return TestResult.ERROR, f"HTTP client error: {e}"
        except Exception as e:
            return TestResult.ERROR, f"Unexpected error: {e}"
            
    async def teardown(self) -> None:
        """Clean up after the test."""
        # Ensure degraded mode is reset
        try:
            await self.session.post(
                f"{self.trading_gateway_url}/api/v1/admin/reset-degraded"
            )
        except:
            pass  # Ignore errors in cleanup
            
        await self.session.close()


class EventReplayTestCase(TestCase):
    """Test the event replay capability."""
    
    def __init__(
        self,
        events_api_url: str,
        event_types: List[str],
        replay_window_minutes: int = 10
    ):
        """
        Initialize the test case.
        
        Args:
            events_api_url: URL of the events API
            event_types: List of event types to replay
            replay_window_minutes: Time window for event replay in minutes
        """
        super().__init__(
            name="Event Replay Test",
            description="Tests the ability to replay events from persistence store",
            tags=["event-replay", "persistence"],
            timeout_seconds=180.0  # Longer timeout for event replay
        )
        self.events_api_url = events_api_url
        self.event_types = event_types
        self.replay_window_minutes = replay_window_minutes
        
    async def setup(self) -> None:
        """Set up the test environment."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.test_id = str(uuid.uuid4())
        
        # Create a test event to ensure something to replay
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(minutes=5)
        
        test_event = {
            "event_type": self.event_types[0],
            "source_service": "e2e_test",
            "data": {
                "test_id": self.test_id,
                "message": "Test event for replay"
            },
            "metadata": {
                "test_run": True
            }
        }
        
        await self.session.post(
            f"{self.events_api_url}/api/v1/events",
            json=test_event
        )
        
    async def execute(self) -> Tuple[TestResult, str]:
        """Execute the test."""
        import aiohttp
        
        try:
            # Get current time for replay window
            end_time = datetime.datetime.utcnow()
            start_time = end_time - datetime.timedelta(minutes=self.replay_window_minutes)
            
            # Request event replay
            replay_request = {
                "event_types": self.event_types,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "correlation_id": self.test_id,
                "batch_size": 20,
                "test_mode": True
            }
            
            async with self.session.post(
                f"{self.events_api_url}/api/v1/replay",
                json=replay_request
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"Replay request failed with status {response.status}"
                
                replay_data = await response.json()
                
            if "replay_id" not in replay_data:
                return TestResult.FAILED, "Replay response missing replay ID"
                
            replay_id = replay_data["replay_id"]
            
            # Monitor replay status
            max_polls = 30
            polls = 0
            
            while polls < max_polls:
                async with self.session.get(
                    f"{self.events_api_url}/api/v1/replay/{replay_id}/status"
                ) as response:
                    if response.status != 200:
                        return TestResult.FAILED, f"Replay status check failed with status {response.status}"
                    
                    status_data = await response.json()
                    replay_status = status_data.get("status")
                    
                    if replay_status == "completed":
                        break
                    elif replay_status == "failed":
                        return TestResult.FAILED, f"Replay failed: {status_data.get('error', 'Unknown error')}"
                        
                await asyncio.sleep(1)
                polls += 1
                
            if polls >= max_polls:
                return TestResult.TIMEOUT, "Timed out waiting for replay to complete"
                
            # Get replay results
            async with self.session.get(
                f"{self.events_api_url}/api/v1/replay/{replay_id}/results"
            ) as response:
                if response.status != 200:
                    return TestResult.FAILED, f"Replay results check failed with status {response.status}"
                
                results_data = await response.json()
                
            events_found = results_data.get("events_found", 0)
            events_replayed = results_data.get("events_replayed", 0)
            
            if events_found == 0:
                return TestResult.FAILED, "No events found for replay"
                
            if events_replayed == 0:
                return TestResult.FAILED, "No events were successfully replayed"
                
            # Verify our test event was among them
            replay_included_test_event = False
            
            # Check in receiver service logs or a specific API endpoint
            # For this example, we'll assume there's an endpoint to check
            async with self.session.get(
                f"{self.events_api_url}/api/v1/events/search?correlation_id={self.test_id}&include_replayed=true"
            ) as response:
                if response.status == 200:
                    search_data = await response.json()
                    events = search_data.get("events", [])
                    
                    for event in events:
                        if event.get("metadata", {}).get("replayed"):
                            replay_included_test_event = True
                            break
            
            if not replay_included_test_event:
                return TestResult.FAILED, "Test event was not included in replay"
                
            return TestResult.PASSED, (
                f"Successfully replayed {events_replayed}/{events_found} events "
                f"including test event with ID {self.test_id}"
            )
                
        except aiohttp.ClientError as e:
            return TestResult.ERROR, f"HTTP client error: {e}"
        except Exception as e:
            return TestResult.ERROR, f"Unexpected error: {e}"
            
    async def teardown(self) -> None:
        """Clean up after the test."""
        await self.session.close()


# Function to create the main end-to-end test suite
def create_main_test_suite() -> TestSuite:
    """
    Create the main end-to-end test suite for the Forex trading platform.
    
    Returns:
        TestSuite instance with all test cases
    """
    suite = TestSuite(
        name="Forex Trading Platform E2E Tests",
        description="End-to-end tests for the complete Forex trading platform"
    )
    
    # Add test cases
    # Replace placeholders with actual service URLs for your environment
    base_url = "http://localhost:8000"  # Replace with actual base URL
    
    suite.add_test_case(MarketDataFlowTestCase(
        data_service_url=f"{base_url}/data-pipeline",
        market_data_symbols=["EUR/USD", "GBP/USD", "USD/JPY"]
    ))
    
    suite.add_test_case(OrderExecutionTestCase(
        trading_api_url=f"{base_url}/trading-gateway"
    ))
    
    suite.add_test_case(FeedbackLoopTestCase(
        strategy_api_url=f"{base_url}/strategy-execution",
        ml_api_url=f"{base_url}/ml-integration",
        strategy_id="trend_following_v2",
        parameter_name="momentum_threshold"
    ))
    
    suite.add_test_case(DegradedModeTestCase(
        trading_gateway_url=f"{base_url}/trading-gateway",
        health_api_url=f"{base_url}/monitoring"
    ))
    
    suite.add_test_case(EventReplayTestCase(
        events_api_url=f"{base_url}/events",
        event_types=["order.created", "position.opened", "position.closed"]
    ))
    
    return suite


async def run_tests():
    """Run the end-to-end test suite."""
    # Create and run test suite
    suite = create_main_test_suite()
    runner = TestRunner(parallelism=1)  # Sequential execution for predictability
    
    print(f"Starting test suite: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"Number of test cases: {len(suite.test_cases)}")
    
    report = await runner.run_suite(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Suite: {report.suite_name}")
    print(f"Duration: {report.duration_seconds:.2f} seconds")
    print("\nSummary:")
    for result, count in report.summary.items():
        print(f"  {result}: {count}")
    
    print("\nDetailed Results:")
    for name, (result, message) in report.results.items():
        print(f"  {name}: {result.value}")
        if result != TestResult.PASSED:
            print(f"    Message: {message}")
    
    # Save report to file
    with open("e2e_test_report.json", "w") as f:
        f.write(report.to_json())
    
    print("\nTest report saved to e2e_test_report.json")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    asyncio.run(run_tests())
