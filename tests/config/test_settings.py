"""
Tests for the consolidated configuration settings.

This module contains tests for the AnalysisEngineSettings class and related functionality.
"""

import os
import sys
import pytest
from unittest.mock import patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from analysis_engine.config.settings import (
    AnalysisEngineSettings,
    get_settings,
    get_db_url,
    get_redis_url,
    get_market_data_service_url,
    get_notification_service_url,
    get_analysis_settings,
    get_rate_limits,
    get_db_settings,
    get_api_settings,
    get_security_settings,
    get_monitoring_settings,
    get_websocket_settings,
    ConfigurationManager
)


@pytest.fixture
def test_env_vars():
    """Set up test environment variables."""
    # Save original environment
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["HOST"] = "localhost"
    os.environ["PORT"] = "8000"
    os.environ["DEBUG_MODE"] = "True"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["JWT_SECRET"] = "test_secret"
    os.environ["JWT_ALGORITHM"] = "HS256"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/testdb"
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = "6379"
    os.environ["REDIS_DB"] = "0"
    os.environ["MARKET_DATA_SERVICE_URL"] = "http://test-market-data:8001"
    os.environ["NOTIFICATION_SERVICE_URL"] = "http://test-notification:8002"
    os.environ["ANALYSIS_TIMEOUT"] = "45"
    os.environ["MAX_CONCURRENT_ANALYSES"] = "5"
    os.environ["RATE_LIMIT_REQUESTS"] = "100"
    os.environ["RATE_LIMIT_PERIOD"] = "60"
    os.environ["RATE_LIMIT_PREMIUM"] = "500"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def test_settings_initialization(test_env_vars):
    """Test settings initialization from environment variables."""
    settings = AnalysisEngineSettings()

    assert settings.HOST == "localhost"
    assert settings.PORT == 8000
    assert settings.DEBUG_MODE is True
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.JWT_SECRET == "test_secret"
    assert settings.JWT_ALGORITHM == "HS256"
    assert settings.DATABASE_URL_OVERRIDE == "postgresql://test:test@localhost:5432/testdb"
    assert settings.DATABASE_URL == "postgresql://test:test@localhost:5432/testdb"
    assert settings.REDIS_HOST == "localhost"
    assert settings.REDIS_PORT == 6379
    assert settings.REDIS_DB == 0
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.MARKET_DATA_SERVICE_URL == "http://test-market-data:8001"
    assert settings.NOTIFICATION_SERVICE_URL == "http://test-notification:8002"
    assert settings.ANALYSIS_TIMEOUT == 45
    assert settings.MAX_CONCURRENT_ANALYSES == 5
    assert settings.RATE_LIMIT_REQUESTS == 100
    assert settings.RATE_LIMIT_PERIOD == 60
    assert settings.RATE_LIMIT_PREMIUM == 500


def test_get_settings_caching():
    """Test that get_settings caches the settings instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2  # Same instance


def test_helper_functions(test_env_vars):
    """Test helper functions return correct values."""
    # Force reload settings
    get_settings.cache_clear()

    assert get_db_url() == "postgresql://test:test@localhost:5432/testdb"
    assert get_redis_url() == "redis://localhost:6379/0"
    assert get_market_data_service_url() == "http://test-market-data:8001"
    assert get_notification_service_url() == "http://test-notification:8002"

    analysis_settings = get_analysis_settings()
    assert analysis_settings["analysis_timeout"] == 45
    assert analysis_settings["max_concurrent_analyses"] == 5

    rate_limits = get_rate_limits()
    assert rate_limits["rate_limit_requests"] == 100
    assert rate_limits["rate_limit_period"] == 60
    assert rate_limits["rate_limit_premium"] == 500

    db_settings = get_db_settings()
    assert db_settings["database_url"] == "postgresql://test:test@localhost:5432/testdb"
    assert db_settings["redis_url"] == "redis://localhost:6379/0"

    api_settings = get_api_settings()
    assert api_settings["host"] == "localhost"
    assert api_settings["port"] == 8000
    assert api_settings["debug_mode"] is True

    security_settings = get_security_settings()
    assert security_settings["jwt_secret"] == "test_secret"
    assert security_settings["jwt_algorithm"] == "HS256"

    monitoring_settings = get_monitoring_settings()
    assert monitoring_settings["enable_metrics"] is True

    websocket_settings = get_websocket_settings()
    assert websocket_settings["ws_heartbeat_interval"] == 30
    assert websocket_settings["ws_max_connections"] == 1000


def test_configuration_manager(test_env_vars):
    """Test ConfigurationManager functionality."""
    # Force reload settings
    get_settings.cache_clear()

    manager = ConfigurationManager()

    # Test get method
    assert manager.get("HOST") == "localhost"
    assert manager.get("PORT") == 8000
    assert manager.get("non_existent_key", "default") == "default"

    # Test set method
    manager.set("HOST", "new_host")
    assert manager.get("HOST") == "new_host"

    # Test reload method
    manager.reload()
    assert manager.get("HOST") == "localhost"  # Back to original value


def test_log_level_validation():
    """Test log level validation."""
    # Valid log level
    settings = AnalysisEngineSettings(LOG_LEVEL="DEBUG")
    assert settings.LOG_LEVEL == "DEBUG"

    # Invalid log level should raise ValueError
    with pytest.raises(ValueError):
        AnalysisEngineSettings(LOG_LEVEL="INVALID_LEVEL")


def test_database_url_override():
    """Test DATABASE_URL_OVERRIDE takes precedence."""
    # When DATABASE_URL_OVERRIDE is provided
    settings = AnalysisEngineSettings(DATABASE_URL_OVERRIDE="sqlite:///test.db")
    assert settings.DATABASE_URL == "sqlite:///test.db"

    # When both DATABASE_URL_OVERRIDE and DB_* fields are provided
    settings = AnalysisEngineSettings(
        DATABASE_URL_OVERRIDE="sqlite:///override.db",
        DB_USER="user",
        DB_PASSWORD="pass",
        DB_HOST="host",
        DB_PORT=5432,
        DB_NAME="name"
    )
    assert settings.DATABASE_URL == "sqlite:///override.db"  # Override takes precedence
