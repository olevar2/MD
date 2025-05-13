"""
Unit tests for the service template configuration module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from common_lib.templates.service_template.config import (
    ServiceConfig,
    get_service_config,
    get_database_config,
    get_logging_config,
    get_service_client_config
)


class TestServiceConfig:
    """Tests for the ServiceConfig class."""

    def test_init(self):
        """Test initialization."""
        config = ServiceConfig()
        assert config.name == "service-template"
        assert config.version == "0.1.0"
        assert config.environment == "development"

    def test_validation(self):
        """Test validation."""
        # Test valid config
        config = ServiceConfig(
            name="test-service",
            version="1.0.0",
            environment="production",
            api_prefix="/api/v1",
            cors_origins=["*"],
            max_workers=4,
            cache_size=1000,
            max_requests_per_minute=60,
            max_retries=3,
            retry_delay_seconds=5,
            timeout_seconds=30
        )
        assert config.name == "test-service"
        assert config.version == "1.0.0"
        assert config.environment == "production"

        # Test invalid max_workers
        with pytest.raises(ValueError):
            ServiceConfig(max_workers=0)

        # Test invalid cache_size
        with pytest.raises(ValueError):
            ServiceConfig(cache_size=-1)

        # Test invalid max_requests_per_minute
        with pytest.raises(ValueError):
            ServiceConfig(max_requests_per_minute=0)

        # Test invalid max_retries
        with pytest.raises(ValueError):
            ServiceConfig(max_retries=-1)

        # Test invalid retry_delay_seconds
        with pytest.raises(ValueError):
            ServiceConfig(retry_delay_seconds=-1)

        # Test invalid timeout_seconds
        with pytest.raises(ValueError):
            ServiceConfig(timeout_seconds=-1)


def test_get_service_config():
    """Test get_service_config function."""
    # Call the function
    config = get_service_config()

    # Verify the result
    assert isinstance(config, ServiceConfig)
    assert config.name == "service-template"
    assert config.version == "0.1.0"
    assert config.environment == "development"


def test_get_database_config():
    """Test get_database_config function."""
    # Call the function
    config = get_database_config()

    # Verify the result
    assert config is not None


def test_get_logging_config():
    """Test get_logging_config function."""
    # Call the function
    config = get_logging_config()

    # Verify the result
    assert config is not None


def test_get_service_client_config():
    """Test get_service_client_config function."""
    # Call the function
    config = get_service_client_config("test-service")

    # Verify the result
    assert config is not None
