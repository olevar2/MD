"""
Test Environment for E2E Testing

This module provides utilities for setting up and tearing down the test environment
for end-to-end tests of the Forex Trading Platform.

It handles:
- Starting required services in Docker containers
- Setting up test databases
- Establishing network connections
- Creating test users and accounts
- Cleaning up resources after tests
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import docker
import pytest
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from core_foundations.events.event_schema import Event, EventType, ServiceName
from core_foundations.exceptions.service_exceptions import ServiceError

# Configure logger
logger = logging.getLogger(__name__)


class EnvironmentMode(Enum):
    """Available environment modes for testing."""
    SIMULATED = "simulated"  # Fully simulated environment with mock services
    HYBRID = "hybrid"        # Mix of real and mock services
    PRODUCTION = "production"  # Using actual production services (for staging only)


@dataclass
class TestEnvironmentConfig:
    """Configuration for test environment setup."""
    mode: EnvironmentMode = EnvironmentMode.SIMULATED
    use_live_market_data: bool = False
    use_persistent_storage: bool = False
    enable_service_logs: bool = True
    browser_headless: bool = True


class ServiceManager:
    """
    Manages the lifecycle of services required for end-to-end testing.
    
    Handles starting, stopping, and monitoring services using Docker containers.
    """
    
    def __init__(self, config: TestEnvironmentConfig):
        """Initialize the service manager with the given configuration."""
        self.config = config
        self.client = docker.from_env()
        self.containers = {}
        self.service_urls = {}
        
    def start_service(self, service_name: str, image: str, 
                      environment: Optional[Dict[str, str]] = None, 
                      ports: Optional[Dict[int, int]] = None) -> str:
        """
        Start a service in a Docker container.
        
        Args:
            service_name: Name of the service
            image: Docker image name
            environment: Environment variables for the container
            ports: Port mappings for the container
            
        Returns:
            URL for accessing the service
        """
        logger.info(f"Starting service {service_name} using image {image}")
        
        try:
            container = self.client.containers.run(
                image,
                detach=True,
                environment=environment or {},
                ports=ports or {},
                name=f"e2e-test-{service_name}-{int(time.time())}"
            )
            
            self.containers[service_name] = container
            
            # Wait for service to be ready
            self._wait_for_service_ready(service_name, container)
            
            # Determine service URL
            host = "localhost"
            port = list(ports.values())[0] if ports else None
            url = f"http://{host}:{port}" if port else None
            
            self.service_urls[service_name] = url
            return url
            
        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {str(e)}")
            raise
    
    def _wait_for_service_ready(self, service_name: str, container, timeout: int = 30) -> None:
        """Wait for a service to be ready by checking its health status."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            container.reload()
            status = container.status
            
            if status == "running":
                # Additional health check could be added here
                logger.info(f"Service {service_name} is ready")
                return
            
            if status in ("exited", "dead"):
                logs = container.logs().decode("utf-8")
                logger.error(f"Service {service_name} failed to start. Logs: {logs}")
                raise RuntimeError(f"Service {service_name} failed to start")
                
            time.sleep(1)
            
        raise TimeoutError(f"Timed out waiting for {service_name} to become ready")
    
    def stop_all_services(self) -> None:
        """Stop all running services."""
        for service_name, container in self.containers.items():
            logger.info(f"Stopping service {service_name}")
            try:
                container.stop(timeout=10)
                container.remove()
            except Exception as e:
                logger.error(f"Error stopping service {service_name}: {str(e)}")
        
        self.containers = {}
        self.service_urls = {}


class TestDatabase:
    """
    Manages test databases for end-to-end testing.
    
    Handles creating test databases, loading test data, and cleanup.
    """
    
    def __init__(self, config: TestEnvironmentConfig):
        """Initialize the test database manager with the given configuration."""
        self.config = config
        self.databases = set()
        
    def create_test_database(self, db_name: str) -> None:
        """
        Create a test database with the given name.
        
        Args:
            db_name: Name of the database to create
        """
        logger.info(f"Creating test database: {db_name}")
        # Implementation depends on your database technology
        # This is a placeholder
        self.databases.add(db_name)
        
    def load_test_data(self, db_name: str, data_file: str) -> None:
        """
        Load test data into a database from a file.
        
        Args:
            db_name: Name of the database to load data into
            data_file: Path to the file containing test data
        """
        logger.info(f"Loading test data into {db_name} from {data_file}")
        # Implementation depends on your database technology
        
    def cleanup_all_databases(self) -> None:
        """Clean up all test databases."""
        for db_name in self.databases:
            logger.info(f"Cleaning up test database: {db_name}")
            # Implementation depends on your database technology
            
        self.databases = set()


class BrowserManager:
    """
    Manages browser instances for UI testing with Playwright.
    """
    
    def __init__(self, config: TestEnvironmentConfig):
        """Initialize the browser manager with the given configuration."""
        self.config = config
        self.playwright = None
        self.browser = None
        
    async def start_browser(self) -> Browser:
        """
        Start a browser instance.
        
        Returns:
            Browser: The browser instance
        """
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.config.browser_headless
        )
        return self.browser
    
    async def new_context(self, **kwargs) -> BrowserContext:
        """
        Create a new browser context.
        
        Args:
            **kwargs: Additional arguments for the browser context
            
        Returns:
            BrowserContext: The browser context
        """
        if not self.browser:
            await self.start_browser()
            
        return await self.browser.new_context(**kwargs)
    
    async def new_page(self, **kwargs) -> Page:
        """
        Create a new page in a new browser context.
        
        Args:
            **kwargs: Additional arguments for the browser context
            
        Returns:
            Page: The page instance
        """
        context = await self.new_context(**kwargs)
        return await context.new_page()
    
    async def close(self) -> None:
        """Close the browser and playwright instance."""
        if self.browser:
            await self.browser.close()
            self.browser = None
            
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None


class TestEnvironment:
    """
    Main class for managing the end-to-end test environment.
    
    Coordinates service management, database setup, and browser instances.
    """
    
    def __init__(self, config: Optional[TestEnvironmentConfig] = None):
        """Initialize the test environment with the given configuration."""
        self.config = config or TestEnvironmentConfig()
        self.service_manager = ServiceManager(self.config)
        self.database = TestDatabase(self.config)
        self.browser_manager = BrowserManager(self.config)
        self._setup_complete = False
        
    async def setup(self) -> None:
        """Set up the test environment."""
        logger.info(f"Setting up test environment in {self.config.mode.value} mode")
        
        # Start required services
        if self.config.mode != EnvironmentMode.PRODUCTION:
            # Start mock or hybrid services as needed
            self._start_required_services()
            
        # Setup test databases if needed
        if not self.config.use_persistent_storage:
            self._setup_test_databases()
            
        self._setup_complete = True
          def _start_required_services(self) -> None:
        """Start the required services based on the environment mode."""
        # Start services based on the environment mode
        if self.config.mode == EnvironmentMode.SIMULATED:
            # Start all services as mocks
            self.service_manager.start_service(
                "api-gateway", 
                "forex-trading-platform/mock-api-gateway:latest",
                ports={8000: 8000}
            )
            
            self.service_manager.start_service(
                "data-service", 
                "forex-trading-platform/mock-data-service:latest",
                ports={8001: 8001}
            )
            
            # Start mock trading services
            self.service_manager.start_service(
                "strategy-execution-engine",
                "forex-trading-platform/mock-strategy-execution:latest",
                ports={8002: 8002},
                environment={
                    "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
                    "DATABASE_URL": "sqlite:///tmp/forex_test.db",
                    "LOG_LEVEL": "INFO"
                }
            )
            
            self.service_manager.start_service(
                "trading-gateway",
                "forex-trading-platform/mock-trading-gateway:latest",
                ports={8003: 8003},
                environment={
                    "EXECUTION_MODE": "SIMULATED",
                    "BROKER_API_KEY": "test-api-key",
                    "BROKER_SECRET": "test-secret"
                }
            )
            
            self.service_manager.start_service(
                "portfolio-management",
                "forex-trading-platform/mock-portfolio-management:latest",
                ports={8004: 8004}
            )
            
            self.service_manager.start_service(
                "risk-management",
                "forex-trading-platform/mock-risk-management:latest",
                ports={8005: 8005}
            )
            
        elif self.config.mode == EnvironmentMode.HYBRID:
            # Start core services as real instances and supporting services as mocks
            
            # Start real API gateway
            self.service_manager.start_service(
                "api-gateway",
                "forex-trading-platform/api-gateway:latest",
                ports={8000: 8000},
                environment={
                    "SERVICE_DISCOVERY_URL": "http://localhost:8500",
                    "AUTH_SERVICE_URL": "http://localhost:8006",
                    "LOG_LEVEL": "INFO"
                }
            )
            
            # Start real trading gateway service
            self.service_manager.start_service(
                "trading-gateway",
                "forex-trading-platform/trading-gateway-service:latest",
                ports={8003: 8003},
                environment={
                    "EXECUTION_MODE": "SIMULATED",  # Still use simulated execution in hybrid mode
                    "BROKER_API_KEY": os.environ.get("TEST_BROKER_API_KEY", "test-api-key"),
                    "BROKER_SECRET": os.environ.get("TEST_BROKER_SECRET", "test-secret"),
                    "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
                    "DATABASE_URL": "sqlite:///tmp/forex_test.db"
                }
            )
            
            # Start real strategy execution engine
            self.service_manager.start_service(
                "strategy-execution-engine",
                "forex-trading-platform/strategy-execution-engine:latest",
                ports={8002: 8002},
                environment={
                    "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
                    "DATABASE_URL": "sqlite:///tmp/forex_test.db",
                    "FEATURE_STORE_URL": "http://localhost:8007",
                    "LOG_LEVEL": "INFO"
                }
            )
            
            # Start mock portfolio management
            self.service_manager.start_service(
                "portfolio-management",
                "forex-trading-platform/mock-portfolio-management:latest",
                ports={8004: 8004}
            )
            
        elif self.config.mode == EnvironmentMode.PRODUCTION:
            # In production test mode, we don't start services - we use existing ones
            logger.info("Using production services - no services started locally")
            
            # Set service URLs for external services
            self.service_manager.service_urls = {
                "api-gateway": os.environ.get("API_GATEWAY_URL", "https://api.forex-platform.example.com"),
                "trading-gateway": os.environ.get("TRADING_GATEWAY_URL", "https://trading.forex-platform.example.com"),
                "portfolio-management": os.environ.get("PORTFOLIO_URL", "https://portfolio.forex-platform.example.com"),
                "risk-management": os.environ.get("RISK_URL", "https://risk.forex-platform.example.com"),
            }
    
    def _setup_test_databases(self) -> None:
        """Set up test databases."""
        # Create and populate test databases
        self.database.create_test_database("forex_trading_test")
        self.database.load_test_data(
            "forex_trading_test", 
            "e2e/fixtures/test_data.sql"
        )
        
    async def teardown(self) -> None:
        """Tear down the test environment and clean up resources."""
        logger.info("Tearing down test environment")
        
        # Close browser instances
        await self.browser_manager.close()
        
        # Stop services
        self.service_manager.stop_all_services()
        
        # Clean up databases
        if not self.config.use_persistent_storage:
            self.database.cleanup_all_databases()
            
    @asynccontextmanager
    async def get_page(self, **kwargs) -> Generator[Page, None, None]:
        """
        Get a browser page for UI testing.
        
        Yields:
            Page: A playwright page instance
        """
        page = await self.browser_manager.new_page(**kwargs)
        try:
            yield page
        finally:
            await page.close()
            
    @contextmanager
    def service_context(self) -> Generator[Dict[str, str], None, None]:
        """
        Get a context manager for accessing service URLs.
        
        Yields:
            Dict[str, str]: A dictionary of service names to URLs
        """
        try:
            yield self.service_manager.service_urls
        finally:
            pass  # No cleanup needed here


@pytest.fixture(scope="session")
async def test_environment() -> Generator[TestEnvironment, None, None]:
    """
    Pytest fixture that provides a test environment for the test session.
    
    Yields:
        TestEnvironment: The test environment
    """
    env = TestEnvironment()
    await env.setup()
    try:
        yield env
    finally:
        await env.teardown()


@pytest.fixture(scope="function")
async def page(test_environment: TestEnvironment) -> Generator[Page, None, None]:
    """
    Pytest fixture that provides a browser page for UI testing.
    
    Args:
        test_environment: The test environment
        
    Yields:
        Page: A playwright page instance
    """
    async with test_environment.get_page() as page:
        yield page
