"""
Integration Test Framework

This module provides a framework for testing interactions between
services in the Forex Trading Platform.
"""

import json
import logging
import os
import random
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, Union

import docker
import pytest
import requests
from pydantic import BaseModel
from testcontainers.core.container import DockerContainer
from testcontainers.kafka import KafkaContainer

from core_foundations.events.event_schema import Event, EventType, ServiceName
from core_foundations.exceptions.service_exceptions import ServiceError

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Represents a service container for integration testing.
    
    This class manages the lifecycle of a service container and provides
    methods for interacting with the service API.
    """
    
    def __init__(
        self,
        service_name: str,
        image_name: str,
        environment: Optional[Dict[str, str]] = None,
        ports: Optional[Dict[int, int]] = None,
        volumes: Optional[Dict[str, str]] = None,
        dependencies: Optional[List['ServiceContainer']] = None,
        ready_timeout: int = 30,
        health_check_endpoint: str = "/health/ready",
    ):
        """
        Initialize service container.
        
        Args:
            service_name: Name of the service
            image_name: Docker image name
            environment: Environment variables
            ports: Port mapping (host:container)
            volumes: Volume mapping (host:container)
            dependencies: List of service containers this service depends on
            ready_timeout: Timeout in seconds for service to become ready
            health_check_endpoint: Endpoint to check for service readiness
        """
        self.service_name = service_name
        self.image_name = image_name
        self.environment = environment or {}
        self.ports = ports or {}
        self.volumes = volumes or {}
        self.dependencies = dependencies or []
        self.ready_timeout = ready_timeout
        self.health_check_endpoint = health_check_endpoint
        
        self.container = None
        self.container_id = None
        self.host = None
        self.ports_mapping = {}
    
    def start(self) -> None:
        """
        Start the service container.
        
        Raises:
            ServiceError: If the container fails to start
        """
        try:
            # Create container
            container = DockerContainer(self.image_name)
            
            # Add environment variables
            for key, value in self.environment.items():
                container.with_env(key, value)
            
            # Add port mappings
            for host_port, container_port in self.ports.items():
                container.with_bind_ports(container_port, host_port)
            
            # Add volume mappings
            for host_path, container_path in self.volumes.items():
                container.with_volume_mapping(host_path, container_path)
            
            # Start the container
            self.container = container.start()
            self.container_id = self.container.get_container_id()
            
            # Get container info
            docker_client = docker.from_env()
            container_info = docker_client.containers.get(self.container_id)
            
            # Extract port mappings
            self.ports_mapping = {}
            for container_port, host_ports in container_info.ports.items():
                if host_ports:
                    port_num = int(container_port.split('/')[0])
                    host_port = int(host_ports[0]['HostPort'])
                    self.ports_mapping[port_num] = host_port
            
            # Set host
            self.host = f"http://localhost:{self.ports_mapping[list(self.ports_mapping.keys())[0]]}"
            
            # Wait for service to be ready
            self._wait_for_service_ready()
            
            logger.info(f"Started service container {self.service_name} at {self.host}")
            
        except Exception as e:
            logger.error(f"Failed to start service container {self.service_name}: {e}")
            raise ServiceError(f"Failed to start service container: {e}")
    
    def _wait_for_service_ready(self) -> None:
        """
        Wait for the service to become ready.
        
        Raises:
            ServiceError: If the service is not ready within the timeout
        """
        start_time = time.time()
        ready = False
        
        while time.time() - start_time < self.ready_timeout:
            try:
                # Check health endpoint
                response = requests.get(f"{self.host}{self.health_check_endpoint}", timeout=2)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'UP':
                        ready = True
                        break
            except Exception:
                pass
            
            # Wait before retrying
            time.sleep(1)
        
        if not ready:
            self.stop()
            raise ServiceError(f"Service {self.service_name} failed to become ready within {self.ready_timeout} seconds")
    
    def stop(self) -> None:
        """
        Stop the service container.
        
        This method stops and removes the container.
        """
        if self.container:
            try:
                self.container.stop()
                logger.info(f"Stopped service container {self.service_name}")
            except Exception as e:
                logger.warning(f"Error stopping container {self.service_name}: {e}")
    
    def call_api(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Call a service API endpoint.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            ServiceError: If the API call fails
        """
        try:
            url = f"{self.host}{endpoint}"
            response = requests.request(method, url, **kwargs)
            return response
        except Exception as e:
            logger.error(f"API call to {self.service_name} {endpoint} failed: {e}")
            raise ServiceError(f"API call failed: {e}")


class TestEnvironment:
    """
    Test environment for integration testing.
    
    This class manages a set of service containers and provides methods
    for interacting with them.
    """
    
    def __init__(self):
        """Initialize the test environment"""
        self.services = {}
        self.kafka_container = None
        self.zookeeper_container = None
        self.network = None
        self.minio_container = None
    
    def setup_kafka(self, exposed_port: int = 9092) -> None:
        """
        Set up Kafka for testing.
        
        Args:
            exposed_port: Port to expose Kafka on
            
        Raises:
            ServiceError: If Kafka setup fails
        """
        try:
            # Start Kafka container
            self.kafka_container = KafkaContainer().with_exposed_ports(exposed_port)
            self.kafka_container.start()
            
            bootstrap_servers = self.kafka_container.get_bootstrap_server()
            logger.info(f"Started Kafka at {bootstrap_servers}")
            
        except Exception as e:
            logger.error(f"Failed to set up Kafka: {e}")
            raise ServiceError(f"Failed to set up Kafka: {e}")
    
    def setup_minio(
        self,
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        exposed_port: int = 9000,
        console_port: int = 9001,
    ) -> None:
        """
        Set up MinIO for testing.
        
        Args:
            access_key: MinIO access key
            secret_key: MinIO secret key
            exposed_port: API port
            console_port: Web console port
            
        Raises:
            ServiceError: If MinIO setup fails
        """
        try:
            # Start MinIO container
            self.minio_container = DockerContainer("minio/minio:latest")
            self.minio_container.with_env("MINIO_ACCESS_KEY", access_key)
            self.minio_container.with_env("MINIO_SECRET_KEY", secret_key)
            self.minio_container.with_command("server /data --console-address :9001")
            self.minio_container.with_bind_ports(9000, exposed_port)
            self.minio_container.with_bind_ports(9001, console_port)
            
            # Start container
            self.minio_container = self.minio_container.start()
            
            # Get endpoint URL
            endpoint_url = f"http://localhost:{exposed_port}"
            logger.info(f"Started MinIO at {endpoint_url}")
            
        except Exception as e:
            logger.error(f"Failed to set up MinIO: {e}")
            raise ServiceError(f"Failed to set up MinIO: {e}")
    
    def add_service(
        self,
        service_name: str,
        image_name: str,
        environment: Optional[Dict[str, str]] = None,
        ports: Optional[Dict[int, int]] = None,
        volumes: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Add a service to the test environment.
        
        Args:
            service_name: Name of the service
            image_name: Docker image name
            environment: Environment variables
            ports: Port mapping (host:container)
            volumes: Volume mapping (host:container)
            dependencies: List of service names this service depends on
        """
        # Resolve dependencies to service containers
        resolved_deps = []
        if dependencies:
            for dep_name in dependencies:
                if dep_name not in self.services:
                    logger.warning(f"Dependency {dep_name} not found for {service_name}")
                else:
                    resolved_deps.append(self.services[dep_name])
        
        # Create service container
        service = ServiceContainer(
            service_name=service_name,
            image_name=image_name,
            environment=environment,
            ports=ports,
            volumes=volumes,
            dependencies=resolved_deps
        )
        
        self.services[service_name] = service
    
    def start_all(self) -> None:
        """
        Start all services in dependency order.
        
        Raises:
            ServiceError: If service startup fails
        """
        try:
            # Build dependency graph
            graph = {name: set(s.dependencies) for name, s in self.services.items()}
            
            # Topological sort for dependency order
            visited = set()
            start_order = []
            
            def visit(node):
                if node in visited:
                    return
                visited.add(node)
                for dep in graph.get(node, set()):
                    visit(dep.service_name)
                start_order.append(node)
            
            for node in graph:
                if node not in visited:
                    visit(node)
            
            # Start services in order
            for service_name in start_order:
                logger.info(f"Starting service: {service_name}")
                self.services[service_name].start()
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            raise ServiceError(f"Failed to start services: {e}")
    
    def stop_all(self) -> None:
        """Stop all services"""
        # Stop in reverse dependency order
        for service in reversed(list(self.services.values())):
            service.stop()
        
        # Stop infrastructure
        if self.minio_container:
            self.minio_container.stop()
        
        if self.kafka_container:
            self.kafka_container.stop()
        
        logger.info("All services stopped")
    
    def get_service(self, service_name: str) -> Optional[ServiceContainer]:
        """Get a service by name"""
        return self.services.get(service_name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()


class IntegrationTest:
    """Base class for integration tests"""
    
    @staticmethod
    @contextmanager
    def create_environment() -> Generator[TestEnvironment, None, None]:
        """
        Create a test environment.
        
        Yields:
            Test environment
            
        Example:
            with IntegrationTest.create_environment() as env:
                env.setup_kafka()
                env.add_service("service1", "image1")
                env.start_all()
                # Run tests
        """
        env = TestEnvironment()
        try:
            yield env
        finally:
            env.stop_all()
    
    @staticmethod
    def forex_platform_environment(
        services: List[str] = None, 
        with_kafka: bool = True,
        with_minio: bool = True
    ) -> TestEnvironment:
        """
        Create a test environment with common Forex platform services.
        
        Args:
            services: List of service names to include
            with_kafka: Whether to include Kafka
            with_minio: Whether to include MinIO
            
        Returns:
            Test environment
            
        Example:
            env = IntegrationTest.forex_platform_environment(
                services=["data-pipeline", "feature-store"]
            )
            env.start_all()
        """
        env = TestEnvironment()
        
        # Set up infrastructure if needed
        if with_kafka:
            env.setup_kafka()
        
        if with_minio:
            env.setup_minio()
        
        # Add requested services
        if not services:
            return env
        
        # Service definitions
        service_configs = {
            "data-pipeline": {
                "image": "forex-platform/data-pipeline-service:latest",
                "ports": {8001: 8001},
                "dependencies": []
            },
            "feature-store": {
                "image": "forex-platform/feature-store-service:latest",
                "ports": {8002: 8002},
                "dependencies": ["data-pipeline"]
            },
            "analysis-engine": {
                "image": "forex-platform/analysis-engine-service:latest",
                "ports": {8003: 8003},
                "dependencies": ["feature-store"]
            },
            "ml-integration": {
                "image": "forex-platform/ml-integration-service:latest",
                "ports": {8004: 8004},
                "dependencies": ["feature-store", "analysis-engine"]
            },
            "strategy-execution": {
                "image": "forex-platform/strategy-execution-engine:latest",
                "ports": {8005: 8005},
                "dependencies": ["analysis-engine", "ml-integration"]
            },
            "portfolio-management": {
                "image": "forex-platform/portfolio-management-service:latest",
                "ports": {8006: 8006},
                "dependencies": ["strategy-execution"]
            },
            "risk-management": {
                "image": "forex-platform/risk-management-service:latest",
                "ports": {8007: 8007},
                "dependencies": ["portfolio-management"]
            },
            "trading-gateway": {
                "image": "forex-platform/trading-gateway-service:latest",
                "ports": {8008: 8008},
                "dependencies": ["portfolio-management", "risk-management"]
            }
        }
        
        # Add each requested service
        for service_name in services:
            if service_name not in service_configs:
                logger.warning(f"Unknown service: {service_name}")
                continue
            
            config = service_configs[service_name]
            env.add_service(
                service_name=service_name,
                image_name=config["image"],
                ports=config["ports"],
                dependencies=config["dependencies"]
            )
        
        return env


# Example test scenario classes

class TradingScenario:
    """
    Represents a trading scenario for testing.
    
    This class provides methods for setting up and executing
    common trading scenarios for integration testing.
    """
    
    def __init__(self, env: TestEnvironment):
        """
        Initialize the trading scenario.
        
        Args:
            env: Test environment
        """
        self.env = env
        self.orders = []
        self.trades = []
    
    def setup_market_data(self, symbol: str, timeframe: str) -> None:
        """
        Set up market data for testing.
        
        Args:
            symbol: Currency pair symbol (e.g., "EURUSD")
            timeframe: Candlestick timeframe (e.g., "1h")
            
        Raises:
            ServiceError: If market data setup fails
        """
        try:
            # Get data pipeline service
            data_pipeline = self.env.get_service("data-pipeline")
            if not data_pipeline:
                raise ServiceError("Data pipeline service not found")
            
            # Create test data
            test_data = self._generate_test_market_data(symbol, timeframe)
            
            # Upload test data to data pipeline
            response = data_pipeline.call_api(
                method="POST",
                endpoint="/api/v1/data/import",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": test_data
                }
            )
            
            if response.status_code != 200:
                raise ServiceError(f"Failed to upload market data: {response.text}")
            
            logger.info(f"Uploaded test market data for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Failed to set up market data: {e}")
            raise ServiceError(f"Failed to set up market data: {e}")
    
    def _generate_test_market_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Generate test market data"""
        # Get timeframe in minutes
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }.get(timeframe, 60)
        
        # Generate 100 candles
        candles = []
        base_price = 1.10000 if symbol.startswith("EUR") else 1.30000
        current_time = datetime.utcnow() - timedelta(minutes=timeframe_minutes * 100)
        
        price = base_price
        for i in range(100):
            # Random price movement
            price_change = random.uniform(-0.001, 0.001)
            price += price_change
            
            # Generate candle
            open_price = price
            high_price = price * (1 + random.uniform(0, 0.001))
            low_price = price * (1 - random.uniform(0, 0.001))
            close_price = price * (1 + random.uniform(-0.0005, 0.0005))
            volume = random.uniform(50, 200)
            
            candle_time = current_time + timedelta(minutes=timeframe_minutes * i)
            
            candles.append({
                "timestamp": candle_time.isoformat(),
                "open": round(open_price, 5),
                "high": round(high_price, 5),
                "low": round(low_price, 5),
                "close": round(close_price, 5),
                "volume": round(volume, 2)
            })
        
        return candles
    
    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        order_type: str = "MARKET"
    ) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            symbol: Currency pair symbol
            direction: "BUY" or "SELL"
            volume: Order volume in lots
            order_type: Order type (MARKET, LIMIT, etc.)
            
        Returns:
            Order details
            
        Raises:
            ServiceError: If order placement fails
        """
        try:
            # Get trading gateway service
            trading_gateway = self.env.get_service("trading-gateway")
            if not trading_gateway:
                raise ServiceError("Trading gateway service not found")
            
            # Place order
            response = trading_gateway.call_api(
                method="POST",
                endpoint="/api/v1/orders",
                json={
                    "symbol": symbol,
                    "direction": direction,
                    "volume": volume,
                    "type": order_type,
                    "account_id": "test_account"
                }
            )
            
            if response.status_code != 201:
                raise ServiceError(f"Failed to place order: {response.text}")
            
            order = response.json()
            self.orders.append(order)
            
            logger.info(f"Placed {direction} {order_type} order for {volume} lots of {symbol}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise ServiceError(f"Failed to place order: {e}")
    
    def verify_position(self, symbol: str, direction: str, volume: float) -> bool:
        """
        Verify that a position exists.
        
        Args:
            symbol: Currency pair symbol
            direction: "BUY" or "SELL"
            volume: Position volume in lots
            
        Returns:
            True if position exists, False otherwise
        """
        try:
            # Get portfolio management service
            portfolio = self.env.get_service("portfolio-management")
            if not portfolio:
                raise ServiceError("Portfolio management service not found")
            
            # Get positions
            response = portfolio.call_api(
                method="GET",
                endpoint="/api/v1/positions",
                params={"account_id": "test_account"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to get positions: {response.text}")
                return False
            
            positions = response.json()
            
            # Check if position exists
            for position in positions:
                if (position['symbol'] == symbol and
                        position['direction'] == direction and
                        abs(position['volume'] - volume) < 0.001):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to verify position: {e}")
            return False
