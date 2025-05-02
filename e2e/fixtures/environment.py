"""
Pytest fixtures for test environment setup and management.
Provides test environments in different modes for E2E testing.
"""
import logging
import os
import pytest
from typing import Dict, Any

from ..framework.test_environment import EnhancedTestEnvironment, TestEnvironmentConfig, TestMode

logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    """Add custom command line options for E2E tests."""
    parser.addoption(
        "--test-mode",
        action="store",
        default="simulated",
        choices=["simulated", "hybrid", "production"],
        help="Test environment mode: simulated, hybrid, or production",
    )
    parser.addoption(
        "--test-data-profile",
        action="store",
        default="default",
        help="Test data profile to use for seeding",
    )
    parser.addoption(
        "--docker-compose-file",
        action="store",
        help="Custom Docker Compose file to use",
    )
    parser.addoption(
        "--persistent-data",
        action="store_true",
        default=False,
        help="Keep test data after tests run",
    )


@pytest.fixture(scope="session")
def test_config(request) -> Dict[str, Any]:
    """Get the test configuration from command line options."""
    return {
        "mode": request.config.getoption("--test-mode"),
        "data_seed_profile": request.config.getoption("--test-data-profile"),
        "docker_compose_file": request.config.getoption("--docker-compose-file"),
        "persistent_data": request.config.getoption("--persistent-data"),
    }


@pytest.fixture(scope="session")
def environment_config(test_config) -> TestEnvironmentConfig:
    """Create the test environment configuration."""
    # Convert string mode to enum
    mode = TestMode(test_config["mode"])
    
    config = TestEnvironmentConfig(
        mode=mode,
        data_seed_profile=test_config["data_seed_profile"],
        persistent_data=test_config["persistent_data"],
    )
    
    # Override Docker compose file if specified
    if test_config.get("docker_compose_file"):
        config.docker_compose_file = test_config["docker_compose_file"]
    
    # Set environment variables based on CI/CD environment
    if os.environ.get("CI") == "true":
        # Adjust configuration for CI environment
        config.environment_variables = {
            **config.environment_variables,
            "CI": "true",
            "LOG_LEVEL": "DEBUG",
        }
    
    return config


@pytest.fixture(scope="session")
def test_environment(environment_config) -> EnhancedTestEnvironment:
    """
    Create and set up the test environment.
    This is a session-scoped fixture that sets up the environment once for all tests.
    """
    logger.info(f"Setting up {environment_config.mode.value} test environment")
    
    env = EnhancedTestEnvironment(config=environment_config)
    env.setup()
    
    yield env
    
    logger.info(f"Tearing down {environment_config.mode.value} test environment")
    env.teardown()


@pytest.fixture
def test_case_environment(test_environment, request) -> EnhancedTestEnvironment:
    """
    Environment fixture for each test case.
    Resets the environment to a clean state for each test.
    """
    test_name = request.node.name
    logger.info(f"Preparing environment for test: {test_name}")
    
    # Reset the environment for each test case
    test_environment.reset()
    
    yield test_environment
    
    # Collect test results
    outcome = "PASS" if not request.node.failed else "FAIL"
    test_environment.create_test_report(
        test_name=test_name,
        status=outcome,
        duration=request.node.duration if hasattr(request.node, "duration") else 0,
    )
