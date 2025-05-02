"""
Tests for configuration and logging functionality.

This module contains tests for configuration loading, validation,
and logging setup.
"""

import pytest
import os
import logging
from unittest.mock import patch, MagicMock
from analysis_engine.config import AnalysisEngineSettings as Settings, get_settings
from analysis_engine.core.logging import setup_logging

@pytest.fixture
def test_env_vars():
    """Set up test environment variables."""
    env_vars = {
        "API_HOST": "localhost",
        "API_PORT": "8000",
        "LOG_LEVEL": "DEBUG",
        "DB_URL": "postgresql://test:test@localhost:5432/testdb",
        "REDIS_URL": "redis://localhost:6379/0",
        "JWT_SECRET": "test_secret",
        "JWT_ALGORITHM": "HS256",
        "RATE_LIMIT_REQUESTS": "100",
        "RATE_LIMIT_PERIOD": "60"
    }

    # Save original environment
    original_env = {}
    for key in env_vars:
        if key in os.environ:
            original_env[key] = os.environ[key]

    # Set test environment
    for key, value in env_vars.items():
        os.environ[key] = value

    yield env_vars

    # Restore original environment
    for key in env_vars:
        if key in original_env:
            os.environ[key] = original_env[key]
        else:
            del os.environ[key]

def test_settings_initialization(test_env_vars):
    """Test settings initialization from environment variables."""
    settings = Settings()

    assert settings.api_host == "localhost"
    assert settings.api_port == 8000
    assert settings.log_level == "DEBUG"
    assert settings.db_url == "postgresql://test:test@localhost:5432/testdb"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.jwt_secret == "test_secret"
    assert settings.jwt_algorithm == "HS256"
    assert settings.rate_limit_requests == 100
    assert settings.rate_limit_period == 60

def test_settings_validation():
    """Test settings validation."""
    with pytest.raises(ValueError):
        Settings(api_port=-1)

    with pytest.raises(ValueError):
        Settings(rate_limit_requests=0)

    with pytest.raises(ValueError):
        Settings(rate_limit_period=0)

def test_get_settings_singleton():
    """Test that get_settings returns a singleton instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2

def test_logging_setup():
    """Test logging setup."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging()

        mock_get_logger.assert_called_once()
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

def test_logging_level_validation():
    """Test logging level validation."""
    with pytest.raises(ValueError):
        Settings(log_level="INVALID_LEVEL")

def test_database_url_validation():
    """Test database URL validation."""
    with pytest.raises(ValueError):
        Settings(db_url="invalid_url")

def test_redis_url_validation():
    """Test Redis URL validation."""
    with pytest.raises(ValueError):
        Settings(redis_url="invalid_url")

def test_jwt_settings_validation():
    """Test JWT settings validation."""
    with pytest.raises(ValueError):
        Settings(jwt_secret="")

    with pytest.raises(ValueError):
        Settings(jwt_algorithm="INVALID_ALGORITHM")

def test_rate_limit_settings_validation():
    """Test rate limit settings validation."""
    with pytest.raises(ValueError):
        Settings(rate_limit_requests=-1)

    with pytest.raises(ValueError):
        Settings(rate_limit_period=-1)

def test_optional_settings():
    """Test optional settings with default values."""
    settings = Settings(
        api_host="localhost",
        api_port=8000,
        db_url="postgresql://test:test@localhost:5432/testdb"
    )

    assert settings.log_level == "INFO"  # Default value
    assert settings.rate_limit_requests == 100  # Default value
    assert settings.rate_limit_period == 60  # Default value

def test_settings_repr():
    """Test settings string representation."""
    settings = Settings()
    settings_str = str(settings)

    assert "Settings" in settings_str
    assert "api_host" in settings_str
    assert "api_port" in settings_str
    assert "log_level" in settings_str

def test_settings_dict():
    """Test settings dictionary conversion."""
    settings = Settings()
    settings_dict = settings.dict()

    assert isinstance(settings_dict, dict)
    assert "api_host" in settings_dict
    assert "api_port" in settings_dict
    assert "log_level" in settings_dict

def test_settings_json():
    """Test settings JSON conversion."""
    settings = Settings()
    settings_json = settings.json()

    assert isinstance(settings_json, str)
    assert "api_host" in settings_json
    assert "api_port" in settings_json
    assert "log_level" in settings_json