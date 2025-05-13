"""
Manages the setup and teardown of the E2E test environment.
Provides comprehensive environment management for isolated testing of the Forex trading platform.
"""
import enum
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import docker
import psutil
import pytest
import requests
import yaml
from docker.errors import DockerException
from playwright.sync_api import sync_playwright

# Import configurations and utilities
from ..fixtures.market_conditions import MarketCondition
from ..reporting.test_reporter import TestReporter
from ..utils.data_seeder import DataSeeder
from ..utils.service_health import ServiceHealthChecker
from ..utils.config import load_configuration

logger = logging.getLogger(__name__)

class TestMode(enum.Enum):
    """Defines the available testing modes for the E2E test environment."""
    SIMULATED = "simulated"  # Fully simulated environment with mock services
    HYBRID = "hybrid"        # Mix of real and simulated services
    PRODUCTION = "production"  # Production-like environment with real services

@dataclass
class TestEnvironmentConfig:
    """Configuration for the test environment."""
    mode: TestMode
    services_to_virtualize: List[str] = None
    data_seed_profile: str = "default"
    test_data_path: str = None
    docker_compose_file: str = None
    container_startup_timeout: int = 180  # seconds
    service_health_check_timeout: int = 120  # seconds
    cleanup_on_failure: bool = True
    persistent_data: bool = False
    resource_limits: Dict[str, Any] = None
    environment_variables: Dict[str, str] = None

    def __post_init__(self):
    """
      post init  .
    
    """

        # Set defaults based on mode if not explicitly provided
        if self.services_to_virtualize is None:
            if self.mode == TestMode.SIMULATED:
                self.services_to_virtualize = ["all"]
            elif self.mode == TestMode.HYBRID:
                # By default virtualize external dependencies in hybrid mode
                self.services_to_virtualize = [
                    "market_data_provider",
                    "exchange_connector",
                    "news_service",
                    "external_risk_provider"
                ]
            else:
                self.services_to_virtualize = []

        # Set default docker compose file if not specified
        if self.docker_compose_file is None:
            base_dir = Path(__file__).parent.parent.parent / "infrastructure" / "docker"
            self.docker_compose_file = str(base_dir / f"docker-compose.e2e-{self.mode.value}.yml")

        # Set default test data path if not provided
        if self.test_data_path is None:
            self.test_data_path = str(Path(__file__).parent.parent / "fixtures" / "test_data")

        # Set default resource limits if not specified
        if self.resource_limits is None:
            self.resource_limits = {
                "cpu_limit": "4",
                "memory_limit": "8g",
                "storage_limit": "10g"
            }

        # Set default environment variables if not specified
        if self.environment_variables is None:
            self.environment_variables = {
                "ENVIRONMENT": "testing",
                "LOG_LEVEL": "INFO",
                "DISABLE_SECURITY_FEATURES": "true" if self.mode != TestMode.PRODUCTION else "false",
                "USE_SIMULATED_TIME": "true" if self.mode == TestMode.SIMULATED else "false"
            }

class TestEnvironment:
    """
    Handles the lifecycle of the test environment for E2E tests.
    """

    def __init__(self):
    """
      init  .
    
    """

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        # TODO: Add state for managed services (e.g., Docker containers)
        logger.info("Initializing TestEnvironment...")

    @with_error_handling(error_class=TestEnvironmentError)
    def setup(self):
        """
        Sets up the test environment:
        - Starts necessary services (e.g., using Docker Compose).
        - Seeds database with test data.
        - Initializes Playwright browser.
        """
        logger.info("Setting up E2E test environment...")
        try:
            # TODO: Start dependent services (mock or real in containers)
            # Example: subprocess.run(["docker-compose", "-f", "path/to/docker-compose.yml", "up", "-d"])

            # TODO: Wait for services to be healthy
            time.sleep(10) # Placeholder

            # TODO: Seed data
            # Example: seed_test_data()

            # TODO: Initialize Playwright
            self.playwright = sync_playwright().start()
            # TODO: Configure browser options (headless, browser type)
            self.browser = self.playwright.chromium.launch(headless=True)
            self.context = self.browser.new_context(
                # TODO: Configure context options (viewport, base URL)
                # viewport={"width": 1920, "height": 1080},
                # base_url="http://localhost:3000" # Example UI service URL
            )
            self.page = self.context.new_page()
            logger.info("Playwright initialized.")

            logger.info("E2E test environment setup complete.")
            return self.page # Return the page object for tests

        except Exception as e:
            # Convert to TestEnvironmentError for consistent handling
            error_details = {
                "traceback": traceback.format_exc(),
                "component": "test_environment_setup"
            }
            raise TestEnvironmentError(
                message=f"Failed to set up test environment: {str(e)}",
                details=error_details
            ) from e

    @with_error_handling(error_class=TestCleanupError, reraise=False)
    def teardown(self):
        """
        Tears down the test environment:
        - Closes Playwright browser.
        - Stops services.
        - Cleans up resources.
        """
        logger.info("Tearing down E2E test environment...")
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            logger.info("Playwright closed.")

            # TODO: Stop dependent services
            # Example: subprocess.run(["docker-compose", "-f", "path/to/docker-compose.yml", "down"], check=True)

            # TODO: Add any other cleanup tasks

            logger.info("E2E test environment teardown complete.")
        except Exception as e:
            # Convert to TestCleanupError for consistent handling
            error_details = {
                "traceback": traceback.format_exc(),
                "component": "test_environment_teardown"
            }
            # Log but don't re-raise to ensure cleanup continues
            logger.error(f"Error during teardown: {str(e)}", extra=error_details)
            # We don't re-raise here because we want to continue with cleanup
            # The @with_error_handling decorator will handle logging

class ServiceVirtualizer(ABC):
    """
    Abstract base class for service virtualization implementations.
    Enables replacing real services with simulated ones for testing.
    """
    @abstractmethod
    def start(self, service_name: str, config: Dict[str, Any]) -> str:
        """Start a virtualized service and return its endpoint."""
        pass

    @abstractmethod
    def stop(self, service_name: str) -> None:
        """Stop a virtualized service."""
        pass

    @abstractmethod
    def reset(self, service_name: str) -> None:
        """Reset a virtualized service to its initial state."""
        pass

    @abstractmethod
    def configure_behavior(self, service_name: str, behavior_config: Dict[str, Any]) -> None:
        """Configure the behavior of a virtualized service."""
        pass


class WireMockServiceVirtualizer(ServiceVirtualizer):
    """
    Service virtualization implementation using WireMock.
    Used for virtualizing HTTP-based services.
    """
    def __init__(self, base_port: int = 8080):
    """
      init  .
    
    Args:
        base_port: Description of base_port
    
    """

        self.base_port = base_port
        self.running_services: Dict[str, Tuple[subprocess.Popen, int]] = {}
        self.stub_dir = Path(__file__).parent.parent / "fixtures" / "wiremock_stubs"

    @with_error_handling(error_class=ServiceVirtualizationError)
    def start(self, service_name: str, config: Dict[str, Any]) -> str:
        """Start a WireMock instance for the specified service."""
        if service_name in self.running_services:
            logger.info(f"Service {service_name} already virtualized, reusing")
            return f"http://localhost:{self.running_services[service_name][1]}"

        try:
            # Find an available port
            port = self._find_available_port()

            # Determine stub directory for this service
            service_stub_dir = self.stub_dir / service_name
            if not service_stub_dir.exists():
                logger.warning(f"No stubs found for {service_name}, creating default")
                service_stub_dir.mkdir(parents=True, exist_ok=True)
                self._create_default_stubs(service_name, service_stub_dir)

            # Start WireMock process
            cmd = [
                "java", "-jar", "wiremock.jar",
                "--port", str(port),
                "--root-dir", str(service_stub_dir),
                "--verbose"
            ]

            # Add any extra config options
            if config.get("global_response_templating", False):
                cmd.append("--global-response-templating")

            if config.get("no_request_journal", False):
                cmd.append("--no-request-journal")

            # Start the process
            logger.info(f"Starting WireMock for {service_name} on port {port}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for WireMock to start
            self._wait_for_wiremock(port)

            # Store the running service
            self.running_services[service_name] = (process, port)

            # Return the endpoint
            return f"http://localhost:{port}"
        except Exception as e:
            # Convert to ServiceVirtualizationError for consistent handling
            error_details = {
                "service_name": service_name,
                "config": config,
                "traceback": traceback.format_exc()
            }
            raise ServiceVirtualizationError(
                message=f"Failed to start WireMock for {service_name}: {str(e)}",
                details=error_details
            ) from e

    def stop(self, service_name: str) -> None:
        """Stop the WireMock instance for the specified service."""
        if service_name not in self.running_services:
            logger.warning(f"Service {service_name} not running, nothing to stop")
            return

        process, _ = self.running_services[service_name]
        logger.info(f"Stopping WireMock for {service_name}")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"WireMock for {service_name} didn't terminate gracefully, killing")
            process.kill()

        del self.running_services[service_name]

    def reset(self, service_name: str) -> None:
        """Reset the WireMock instance for the specified service."""
        if service_name not in self.running_services:
            logger.warning(f"Service {service_name} not running, cannot reset")
            return

        _, port = self.running_services[service_name]
        logger.info(f"Resetting WireMock for {service_name}")

        # Call WireMock API to reset mappings and requests
        try:
            requests.post(f"http://localhost:{port}/__admin/reset", timeout=5)
        except requests.RequestException as e:
            logger.error(f"Failed to reset WireMock for {service_name}: {e}")

    def configure_behavior(self, service_name: str, behavior_config: Dict[str, Any]) -> None:
        """Configure the behavior of the WireMock instance for the specified service."""
        if service_name not in self.running_services:
            logger.warning(f"Service {service_name} not running, cannot configure")
            return

        _, port = self.running_services[service_name]
        logger.info(f"Configuring WireMock behavior for {service_name}")

        # Call WireMock API to configure behavior
        try:
            requests.post(
                f"http://localhost:{port}/__admin/mappings",
                json=behavior_config,
                timeout=5
            )
        except requests.RequestException as e:
            logger.error(f"Failed to configure WireMock for {service_name}: {e}")

    def _find_available_port(self) -> int:
        """Find an available port to run WireMock on."""
        # Start from base port and find first available
        port = self.base_port
        used_ports = set(p[1] for p in self.running_services.values())

        while port in used_ports or self._is_port_in_use(port):
            port += 1

        return port

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if a port is already in use."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    @staticmethod
    def _wait_for_wiremock(port: int, timeout: int = 30) -> None:
        """Wait for WireMock to start and be responsive."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/__admin/mappings", timeout=1)
                if response.status_code == 200:
                    logger.info(f"WireMock is up on port {port}")
                    return
            except requests.RequestException:
                pass

            time.sleep(0.5)

        raise TimeoutError(f"WireMock failed to start on port {port} within {timeout} seconds")

    @staticmethod
    def _create_default_stubs(service_name: str, stub_dir: Path) -> None:
        """Create default stubs for a service."""
        # Create simple health check stub
        health_check = {
            "request": {
                "method": "GET",
                "url": "/health"
            },
            "response": {
                "status": 200,
                "body": json.dumps({"status": "UP", "name": service_name}),
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        }

        with open(stub_dir / "mappings" / "health.json", "w") as f:
            stub_dir.mkdir(parents=True, exist_ok=True)
            (stub_dir / "mappings").mkdir(exist_ok=True)
            json.dump(health_check, f, indent=2)


class KafkaServiceVirtualizer(ServiceVirtualizer):
    """
    Service virtualization for Kafka messaging.
    Provides in-memory Kafka broker for testing event-driven interactions.
    """
    def __init__(self):
    """
      init  .
    
    """

        # Implementation would use an embedded Kafka broker or a mock
        pass

    def start(self, service_name: str, config: Dict[str, Any]) -> str:
    """
    Start.
    
    Args:
        service_name: Description of service_name
        config: Description of config
        Any]: Description of Any]
    
    Returns:
        str: Description of return value
    
    """

        # Implementation for starting virtualized Kafka broker
        # This would likely use test-containers or embedded-kafka
        logger.info(f"Starting virtualized Kafka for {service_name}")
        return "localhost:9092"  # Return the broker connection string

    def stop(self, service_name: str) -> None:
    """
    Stop.
    
    Args:
        service_name: Description of service_name
    
    """

        # Implementation to stop the virtualized Kafka broker
        logger.info(f"Stopping virtualized Kafka for {service_name}")

    def reset(self, service_name: str) -> None:
    """
    Reset.
    
    Args:
        service_name: Description of service_name
    
    """

        # Implementation to clear all messages and reset state
        logger.info(f"Resetting virtualized Kafka for {service_name}")

    def configure_behavior(self, service_name: str, behavior_config: Dict[str, Any]) -> None:
        # Configure broker behavior (latency, error scenarios, etc.)
        logger.info(f"Configuring virtualized Kafka behavior for {service_name}")


class ServiceVirtualizationManager:
    """
    Manages all service virtualization for the test environment.
    Provides a unified interface for controlling virtualized services.
    """
    def __init__(self):
    """
      init  .
    
    """

        self.virtualizers = {
            "http": WireMockServiceVirtualizer(),
            "kafka": KafkaServiceVirtualizer(),
            # Add other virtualizer types as needed
        }
        self.active_virtualizations = {}  # service_name -> (virtualizer_type, endpoint)

    @with_error_handling(error_class=ServiceVirtualizationError)
    def start_virtualization(self, service_name: str, service_type: str, config: Dict[str, Any]) -> str:
        """
        Start virtualization for a service.

        Args:
            service_name: Name of the service to virtualize
            service_type: Type of virtualizer to use (http, kafka, etc.)
            config: Configuration for the virtualizer

        Returns:
            The endpoint URL for connecting to the virtualized service
        """
        logger.info(f"Starting virtualization for {service_name} using {service_type} virtualizer")

        try:
            if service_type not in self.virtualizers:
                raise ValueError(f"Unknown virtualizer type: {service_type}")

            virtualizer = self.virtualizers[service_type]
            endpoint = virtualizer.start(service_name, config)

            self.active_virtualizations[service_name] = (service_type, endpoint)
            return endpoint
        except Exception as e:
            # If it's already a ServiceVirtualizationError, let it propagate
            if isinstance(e, ServiceVirtualizationError):
                raise

            # Otherwise, convert to ServiceVirtualizationError
            error_details = {
                "service_name": service_name,
                "service_type": service_type,
                "config": config,
                "traceback": traceback.format_exc()
            }
            raise ServiceVirtualizationError(
                message=f"Failed to start virtualization for {service_name}: {str(e)}",
                details=error_details
            ) from e

    def stop_virtualization(self, service_name: str) -> None:
        """Stop virtualization for a service."""
        if service_name not in self.active_virtualizations:
            logger.warning(f"Service {service_name} not virtualized, nothing to stop")
            return

        service_type, _ = self.active_virtualizations[service_name]
        virtualizer = self.virtualizers[service_type]

        logger.info(f"Stopping virtualization for {service_name}")
        virtualizer.stop(service_name)

        del self.active_virtualizations[service_name]

    def reset_virtualization(self, service_name: str) -> None:
        """Reset virtualization for a service to initial state."""
        if service_name not in self.active_virtualizations:
            logger.warning(f"Service {service_name} not virtualized, cannot reset")
            return

        service_type, _ = self.active_virtualizations[service_name]
        virtualizer = self.virtualizers[service_type]

        logger.info(f"Resetting virtualization for {service_name}")
        virtualizer.reset(service_name)

    def configure_virtualization(
            self, service_name: str, behavior_config: Dict[str, Any]) -> None:
        """Configure behavior of a virtualized service."""
        if service_name not in self.active_virtualizations:
            logger.warning(f"Service {service_name} not virtualized, cannot configure")
            return

        service_type, _ = self.active_virtualizations[service_name]
        virtualizer = self.virtualizers[service_type]

        logger.info(f"Configuring virtualization for {service_name}")
        virtualizer.configure_behavior(service_name, behavior_config)

    def get_endpoint(self, service_name: str) -> Optional[str]:
        """Get the endpoint for a virtualized service."""
        if service_name not in self.active_virtualizations:
            return None

        _, endpoint = self.active_virtualizations[service_name]
        return endpoint

    def stop_all(self) -> None:
        """Stop all virtualized services."""
        for service_name in list(self.active_virtualizations.keys()):
            self.stop_virtualization(service_name)

class DockerManager:
    """
    Manages Docker containers for the test environment.
    Handles starting, stopping, and monitoring containers.
    """
    def __init__(self):
    """
      init  .
    
    """

        try:
            self.client = docker.from_env()
            # Validate Docker is running by listing containers
            self.client.containers.list()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker initialization failed: {e}")

        self.active_containers = {}  # container_name -> container_id
        self.networks = set()  # Set of created networks
        self.volumes = set()  # Set of created volumes

    def start_containers(
        self,
        compose_file: str,
        env_vars: Dict[str, str] = None,
        project_name: str = None
    ) -> Dict[str, str]:
        """
        Start containers using docker-compose.

        Args:
            compose_file: Path to docker-compose.yml file
            env_vars: Environment variables to pass to containers
            project_name: Project name for docker-compose

        Returns:
            Dictionary mapping service names to container IDs
        """
        if not os.path.exists(compose_file):
            raise FileNotFoundError(f"Docker compose file not found: {compose_file}")

        # Generate a unique project name if not provided
        if project_name is None:
            project_name = f"e2e-test-{int(time.time())}"

        # Prepare environment variables
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Start containers using docker-compose
        cmd = [
            "docker-compose",
            "-f", compose_file,
            "--project-name", project_name,
            "up",
            "-d"
        ]

        logger.info(f"Starting containers with docker-compose: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start containers: {e.stderr}")
            raise RuntimeError(f"Container startup failed: {e.stderr}")

        # Get container IDs for all services in the compose file
        self._refresh_container_list(project_name)

        logger.info(f"Started containers: {list(self.active_containers.keys())}")
        return self.active_containers

    def stop_containers(self, project_name: str) -> None:
        """
        Stop all containers for a project.

        Args:
            project_name: Project name used with docker-compose
        """
        logger.info(f"Stopping containers for project {project_name}")
        try:
            subprocess.run(
                ["docker-compose", "--project-name", project_name, "down", "--volumes", "--remove-orphans"],
                check=True,
                capture_output=True,
                text=True
            )

            # Clean up internal state
            self.active_containers = {
                k: v for k, v in self.active_containers.items()
                if not k.startswith(f"{project_name}_")
            }

            logger.info(f"Containers for project {project_name} stopped and removed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop containers: {e.stderr}")

    def get_container_logs(self, container_name: str) -> str:
        """Get logs from a container."""
        if container_name not in self.active_containers:
            logger.warning(f"Container {container_name} not found")
            return ""

        container_id = self.active_containers[container_name]
        try:
            container = self.client.containers.get(container_id)
            return container.logs().decode('utf-8')
        except docker.errors.NotFound:
            logger.warning(f"Container {container_name} ({container_id}) not found")
            return ""
        except Exception as e:
            logger.error(f"Failed to get logs for container {container_name}: {e}")
            return f"Error getting logs: {str(e)}"

    def wait_for_container_health(
        self,
        container_names: List[str],
        timeout: int = 120
    ) -> Dict[str, bool]:
        """
        Wait for containers to become healthy.

        Args:
            container_names: List of container names to check
            timeout: Timeout in seconds

        Returns:
            Dictionary mapping container names to health status (True if healthy)
        """
        start_time = time.time()
        pending_containers = set(container_names)
        health_status = {name: False for name in container_names}

        while pending_containers and (time.time() - start_time < timeout):
            for container_name in list(pending_containers):
                if container_name not in self.active_containers:
                    logger.warning(f"Container {container_name} not found")
                    pending_containers.remove(container_name)
                    continue

                container_id = self.active_containers[container_name]
                try:
                    container = self.client.containers.get(container_id)

                    # Check if the container is running
                    if container.status != "running":
                        logger.warning(f"Container {container_name} not running: {container.status}")
                        continue

                    # If the container has health check, wait for it to become healthy
                    if hasattr(container, 'attrs') and 'Health' in container.attrs.get('State', {}):
                        health_status = container.attrs['State']['Health']['Status']
                        if health_status == 'healthy':
                            logger.info(f"Container {container_name} is healthy")
                            pending_containers.remove(container_name)
                            health_status[container_name] = True
                        elif health_status == 'unhealthy':
                            logger.error(f"Container {container_name} is unhealthy")
                            pending_containers.remove(container_name)
                        else:
                            logger.debug(f"Container {container_name} health status: {health_status}")
                    else:
                        # Container doesn't have a health check, consider it ready if running
                        logger.info(f"Container {container_name} is running (no health check)")
                        pending_containers.remove(container_name)
                        health_status[container_name] = True
                except docker.errors.NotFound:
                    logger.warning(f"Container {container_name} not found")
                    pending_containers.remove(container_name)
                except Exception as e:
                    logger.error(f"Error checking health for {container_name}: {e}")

            if pending_containers:
                time.sleep(2)

        # Log which containers are still not healthy after timeout
        if pending_containers:
            logger.warning(
                f"Containers not healthy after {timeout} seconds: {pending_containers}"
            )

        return health_status

    def get_container_ports(self, container_name: str) -> Dict[str, int]:
        """Get mapped ports for a container."""
        if container_name not in self.active_containers:
            logger.warning(f"Container {container_name} not found")
            return {}

        container_id = self.active_containers[container_name]
        try:
            container = self.client.containers.get(container_id)
            port_mapping = {}

            if hasattr(container, 'attrs') and 'NetworkSettings' in container.attrs:
                ports = container.attrs['NetworkSettings'].get('Ports', {})
                for container_port, host_bindings in ports.items():
                    if host_bindings:
                        # Get port number without protocol (e.g., '80/tcp' -> '80')
                        container_port_num = container_port.split('/')[0]
                        # Get host port from first binding
                        host_port = host_bindings[0]['HostPort']
                        port_mapping[container_port_num] = int(host_port)

            return port_mapping
        except docker.errors.NotFound:
            logger.warning(f"Container {container_name} ({container_id}) not found")
            return {}
        except Exception as e:
            logger.error(f"Failed to get ports for container {container_name}: {e}")
            return {}

    def _refresh_container_list(self, project_name: str) -> None:
        """Refresh the internal list of active containers."""
        try:
            # Get all containers with the project label
            containers = self.client.containers.list(
                filters={"label": f"com.docker.compose.project={project_name}"}
            )

            # Update the active containers dictionary
            self.active_containers = {}
            for container in containers:
                service = container.labels.get('com.docker.compose.service', 'unknown')
                container_name = f"{project_name}_{service}"
                self.active_containers[container_name] = container.id

            logger.debug(f"Active containers: {self.active_containers}")
        except Exception as e:
            logger.error(f"Failed to refresh container list: {e}")

    def cleanup(self) -> None:
        """Clean up all resources created for testing."""
        logger.info("Cleaning up Docker resources")

        # Stop any remaining containers
        try:
            for name, container_id in list(self.active_containers.items()):
                try:
                    logger.info(f"Stopping container {name}")
                    container = self.client.containers.get(container_id)
                    container.stop(timeout=10)
                    container.remove(force=True)
                except docker.errors.NotFound:
                    pass
                except Exception as e:
                    logger.error(f"Error stopping container {name}: {e}")
        except Exception as e:
            logger.error(f"Error during container cleanup: {e}")

        # Clean up networks
        for network_name in self.networks:
            try:
                logger.info(f"Removing network {network_name}")
                network = self.client.networks.get(network_name)
                network.remove()
            except docker.errors.NotFound:
                pass
            except Exception as e:
                logger.error(f"Error removing network {network_name}: {e}")

        # Clean up volumes
        for volume_name in self.volumes:
            try:
                logger.info(f"Removing volume {volume_name}")
                volume = self.client.volumes.get(volume_name)
                volume.remove()
            except docker.errors.NotFound:
                pass
            except Exception as e:
                logger.error(f"Error removing volume {volume_name}: {e}")


# Example usage (often integrated with pytest fixtures)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     env = TestEnvironment()
#     try:
#         page = env.setup()
#         print("Environment ready. Page object available.")
#         # Run tests using 'page'
#         time.sleep(5) # Simulate test execution
#     finally:
#         env.teardown()

class EnhancedTestEnvironment(TestEnvironment):
    """
    Enhanced test environment for E2E tests of the Forex trading platform.
    Supports different testing modes, service virtualization, and Docker container management.
    Provides comprehensive environment setup, data seeding, and reporting.
    """

    def __init__(self, config: TestEnvironmentConfig = None):
        """
        Initialize the enhanced test environment.

        Args:
            config: Configuration for the test environment
        """
        # Initialize base class
        super().__init__()

        # Initialize with default configuration if not provided
        self.config = config or TestEnvironmentConfig(mode=TestMode.SIMULATED)

        # Initialize components
        self.docker_manager = DockerManager()
        self.service_virtualizer = ServiceVirtualizationManager()
        self.data_seeder = None  # Will be initialized when needed
        self.health_checker = None  # Will be initialized when needed

        # Test reporting
        self.test_reporter = None  # Will be initialized when needed

        # Environment state
        self.env_id = f"forex-e2e-{int(time.time())}"
        self.project_name = f"forex-e2e-{self.config.mode.value}-{int(time.time())}"
        self.service_endpoints = {}
        self.active_services = set()
        self.temp_dir = None

        logger.info(f"Initializing EnhancedTestEnvironment in {self.config.mode.value} mode...")

    def setup(self):
        """
        Sets up the test environment:
        - Creates isolated environment
        - Starts necessary services using Docker Compose
        - Virtualizes services as configured
        - Seeds database with test data
        - Initializes Playwright browser for UI testing

        Returns:
            Self for method chaining
        """
        logger.info(f"Setting up E2E test environment ({self.config.mode.value})...")

        # Create temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp(prefix=f"forex-e2e-{self.config.mode.value}-")
        logger.info(f"Created temporary directory: {self.temp_dir}")

        try:
            # Import lazily to avoid import errors if modules are missing
            if not self.health_checker:
                try:
                    from ..utils.service_health import ServiceHealthChecker
                    self.health_checker = ServiceHealthChecker()
                except ImportError:
                    logger.warning("ServiceHealthChecker not available, health checks will be limited")

            if not self.data_seeder:
                try:
                    from ..utils.data_seeder import DataSeeder
                    self.data_seeder = DataSeeder()
                except ImportError:
                    logger.warning("DataSeeder not available, data seeding will be limited")

            if not self.test_reporter:
                try:
                    from ..reporting.test_reporter import TestReporter
                    self.test_reporter = TestReporter()
                except ImportError:
                    logger.warning("TestReporter not available, test reporting will be limited")

            # Start containers if not fully simulated mode
            if self.config.mode != TestMode.SIMULATED or os.path.exists(self.config.docker_compose_file):
                self._start_docker_services()

            # Setup service virtualization
            if self.config.services_to_virtualize:
                self._setup_virtualized_services()

            # Wait for all services to be healthy
            self._wait_for_services()

            # Seed test data
            self._seed_test_data()

            # Initialize UI testing with Playwright
            super().setup()  # Use base class to set up Playwright

            logger.info("E2E test environment setup complete.")
            return self

        except Exception as e:
            logger.error(f"Failed to set up test environment: {e}", exc_info=True)
            # Attempt cleanup on setup failure if configured
            if self.config.cleanup_on_failure:
                self.teardown()
            raise

    def teardown(self):
        """
        Tears down the test environment:
        - Closes Playwright browser
        - Stops virtualized services
        - Stops Docker containers
        - Cleans up resources
        - Collects test artifacts

        Returns:
            Self for method chaining
        """
        logger.info("Tearing down E2E test environment...")

        # Close Playwright resources (use base class teardown)
        super().teardown()

        # Stop virtualized services
        if self.service_virtualizer:
            logger.info("Stopping virtualized services...")
            self.service_virtualizer.stop_all()

        # Stop Docker containers
        if hasattr(self, 'docker_manager') and self.docker_manager:
            logger.info(f"Stopping Docker containers for project {self.project_name}...")
            self.docker_manager.stop_containers(self.project_name)

        # Clean up temporary directory if not preserving data
        if self.temp_dir and not self.config.persistent_data:
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory: {e}")

        logger.info("E2E test environment teardown complete.")
        return self

    def reset(self):
        """
        Reset the test environment to a clean state without complete teardown.
        Preserves Docker containers but resets data and virtualized services.

        Returns:
            Self for method chaining
        """
        logger.info("Resetting test environment...")

        # Reset virtualized services
        for service_name in self.service_endpoints.keys():
            if self.service_virtualizer.get_endpoint(service_name):
                logger.info(f"Resetting virtualized service: {service_name}")
                self.service_virtualizer.reset_virtualization(service_name)

        # Reset databases and test data
        self._reset_test_data()

        # Reset UI context if needed
        if self.context:
            logger.info("Resetting browser context")
            self.context.close()
            self.context = self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                base_url=self._get_ui_service_url()
            )
            self.page = self.context.new_page()

        logger.info("Test environment reset complete.")
        return self

    def get_service_endpoint(self, service_name: str) -> str:
        """Get the endpoint URL for a service."""
        # Check if we have a virtualized endpoint first
        virtualized_endpoint = self.service_virtualizer.get_endpoint(service_name)
        if virtualized_endpoint:
            return virtualized_endpoint

        # Otherwise, check real service endpoints
        if service_name in self.service_endpoints:
            return self.service_endpoints[service_name]

        raise ValueError(f"Unknown service: {service_name}")

    def inject_market_conditions(self, market_condition: Any) -> 'EnhancedTestEnvironment':
        """
        Inject specific market conditions into the test environment.

        Args:
            market_condition: The market condition to inject

        Returns:
            Self for method chaining
        """
        logger.info(f"Injecting market condition: {getattr(market_condition, 'name', market_condition)}")

        # Get market data provider service
        market_data_provider = "market_data_provider"
        if self.service_virtualizer.get_endpoint(market_data_provider):
            # Configure the virtualized market data provider
            if hasattr(market_condition, 'to_wiremock_config'):
                config = market_condition.to_wiremock_config()
                self.service_virtualizer.configure_virtualization(market_data_provider, config)
            else:
                logger.warning("Market condition doesn't have to_wiremock_config method")
        else:
            # For real market data providers, we need a different approach
            logger.warning("Cannot inject market conditions into non-virtualized market data provider")

        return self

    def collect_logs(self) -> Dict[str, str]:
        """
        Collect logs from all services.

        Returns:
            Dictionary mapping service names to logs
        """
        logs = {}

        # Collect Docker container logs
        for service_name in self.active_services:
            container_name = f"{self.project_name}_{service_name}"
            logs[service_name] = self.docker_manager.get_container_logs(container_name)

        return logs

    def create_test_report(self, test_name: str, status: str, duration: float,
                          artifacts: Dict[str, str] = None) -> None:
        """
        Create a test report for the current test.

        Args:
            test_name: Name of the test
            status: Test status (PASS, FAIL, SKIP)
            duration: Test duration in seconds
            artifacts: Dictionary of test artifacts
        """
        if not self.test_reporter:
            logger.warning("Test reporter not available")
            return

        # Collect logs
        logs = self.collect_logs()

        # Create report
        self.test_reporter.add_test_result(
            test_name=test_name,
            status=status,
            duration=duration,
            environment=self.config.mode.value,
            logs=logs,
            artifacts=artifacts or {}
        )

    def _start_docker_services(self):
        """Start Docker services based on the environment configuration."""
        logger.info(f"Starting Docker services using {self.config.docker_compose_file}")

        # Start containers
        self.docker_manager.start_containers(
            compose_file=self.config.docker_compose_file,
            env_vars=self.config.environment_variables,
            project_name=self.project_name
        )

        # Load service list from docker-compose file
        try:
            with open(self.config.docker_compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
                services = compose_config.get('services', {}).keys()
                self.active_services.update(services)
        except Exception as e:
            logger.error(f"Failed to load docker-compose file: {e}")

        # Get port mappings for services
        self.service_endpoints = {}
        for service_name in self.active_services:
            container_name = f"{self.project_name}_{service_name}"
            ports = self.docker_manager.get_container_ports(container_name)

            # Use first port as service endpoint (assuming HTTP)
            if ports:
                port = next(iter(ports.values()))
                self.service_endpoints[service_name] = f"http://localhost:{port}"
                logger.info(f"Service {service_name} available at {self.service_endpoints[service_name]}")

    def _setup_virtualized_services(self):
        """Set up service virtualization based on configuration."""
        logger.info("Setting up virtualized services...")

        # Determine which services to virtualize
        services_to_virtualize = self.config.services_to_virtualize

        if "all" in services_to_virtualize and self.config.mode == TestMode.SIMULATED:
            # In fully simulated mode, virtualize all required services
            services_to_virtualize = [
                "market_data_provider",
                "exchange_connector",
                "news_service",
                "auth_service",
                "user_service",
                "order_service",
                "position_service",
                "portfolio_service",
                "analysis_service",
                "risk_service",
                "notification_service"
            ]

        # Start virtualization for each service
        for service_name in services_to_virtualize:
            if service_name in self.active_services:
                logger.info(f"Service {service_name} already running in Docker, skipping virtualization")
                continue

            # Determine service type (http or kafka)
            service_type = "http"
            if "kafka" in service_name or service_name in ["event_bus", "message_broker"]:
                service_type = "kafka"

            # Configure virtualization
            config = {"global_response_templating": True}

            # Start virtualization
            endpoint = self.service_virtualizer.start_virtualization(
                service_name=service_name,
                service_type=service_type,
                config=config
            )

            # Store endpoint
            self.service_endpoints[service_name] = endpoint
            logger.info(f"Virtualized {service_name} available at {endpoint}")

    def _wait_for_services(self):
        """Wait for all services to be healthy."""
        logger.info("Waiting for services to be healthy...")

        # Check Docker container health
        if self.active_services and self.config.mode != TestMode.SIMULATED:
            container_names = [f"{self.project_name}_{service}" for service in self.active_services]
            health_status = self.docker_manager.wait_for_container_health(
                container_names,
                timeout=self.config.container_startup_timeout
            )

            # Log health status
            for container, is_healthy in health_status.items():
                if not is_healthy:
                    logger.warning(f"Container {container} is not healthy")

        # Check service endpoints
        if self.health_checker:
            for service_name, endpoint in self.service_endpoints.items():
                try:
                    logger.info(f"Checking health of {service_name} at {endpoint}")

                    # Skip for non-HTTP endpoints
                    if not endpoint.startswith("http"):
                        continue

                    # Check health endpoint
                    health_url = f"{endpoint}/health"
                    self.health_checker.check_service_health(
                        service_name=service_name,
                        health_url=health_url,
                        timeout=self.config.service_health_check_timeout
                    )

                    logger.info(f"Service {service_name} is healthy")
                except Exception as e:
                    logger.warning(f"Service {service_name} health check failed: {e}")

    def _seed_test_data(self):
        """Seed test data into the environment."""
        logger.info(f"Seeding test data using profile: {self.config.data_seed_profile}")

        # Seed data for all services
        if self.data_seeder:
            self.data_seeder.seed_all_services(
                service_endpoints=self.service_endpoints,
                profile=self.config.data_seed_profile,
                test_data_path=self.config.test_data_path
            )
        else:
            logger.warning("Data seeder not available, skipping data seeding")

    def _reset_test_data(self):
        """Reset test data to initial state."""
        logger.info("Resetting test data...")

        # Clear and re-seed data
        if self.data_seeder:
            self.data_seeder.reset_all_services(
                service_endpoints=self.service_endpoints,
                profile=self.config.data_seed_profile,
                test_data_path=self.config.test_data_path
            )
        else:
            logger.warning("Data seeder not available, skipping data reset")

    def _get_ui_service_url(self) -> str:
        """Get the URL for the UI service."""
        if "ui_service" in self.service_endpoints:
            return self.service_endpoints["ui_service"]
        elif "ui-service" in self.service_endpoints:
            return self.service_endpoints["ui-service"]
        else:
            # Default UI service URL
            return "http://localhost:3000"
