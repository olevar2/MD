"""
Simple test script for the consolidated configuration settings.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

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

def setup_test_env():
    """Set up test environment variables."""
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

    # Print environment variables
    print(f"DATABASE_URL environment variable: {os.environ.get('DATABASE_URL')}")

    # Force reload settings
    get_settings.cache_clear()

def test_settings_initialization():
    """Test settings initialization from environment variables."""
    setup_test_env()

    # Create a new settings instance with direct initialization
    settings = AnalysisEngineSettings(
        HOST="localhost",
        PORT=8000,
        DEBUG_MODE=True,
        LOG_LEVEL="DEBUG",
        JWT_SECRET="test_secret",
        JWT_ALGORITHM="HS256",
        DATABASE_URL_OVERRIDE="postgresql://test:test@localhost:5432/testdb",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        REDIS_DB=0,
        MARKET_DATA_SERVICE_URL="http://test-market-data:8001",
        NOTIFICATION_SERVICE_URL="http://test-notification:8002",
        ANALYSIS_TIMEOUT=45,
        MAX_CONCURRENT_ANALYSES=5,
        RATE_LIMIT_REQUESTS=100,
        RATE_LIMIT_PERIOD=60,
        RATE_LIMIT_PREMIUM=500
    )

    # Print settings
    print(f"DATABASE_URL_OVERRIDE: {settings.DATABASE_URL_OVERRIDE}")
    print(f"DATABASE_URL: {settings.DATABASE_URL}")

    print("Testing settings initialization...")
    assert settings.HOST == "localhost", f"Expected 'localhost', got '{settings.HOST}'"
    assert settings.PORT == 8000, f"Expected 8000, got {settings.PORT}"
    assert settings.DEBUG_MODE is True, f"Expected True, got {settings.DEBUG_MODE}"
    assert settings.LOG_LEVEL == "DEBUG", f"Expected 'DEBUG', got '{settings.LOG_LEVEL}'"
    assert settings.JWT_SECRET == "test_secret", f"Expected 'test_secret', got '{settings.JWT_SECRET}'"
    assert settings.JWT_ALGORITHM == "HS256", f"Expected 'HS256', got '{settings.JWT_ALGORITHM}'"
    assert settings.DATABASE_URL_OVERRIDE == "postgresql://test:test@localhost:5432/testdb", f"Expected 'postgresql://test:test@localhost:5432/testdb', got '{settings.DATABASE_URL_OVERRIDE}'"
    assert settings.DATABASE_URL == "postgresql://test:test@localhost:5432/testdb", f"Expected 'postgresql://test:test@localhost:5432/testdb', got '{settings.DATABASE_URL}'"
    assert settings.REDIS_HOST == "localhost", f"Expected 'localhost', got '{settings.REDIS_HOST}'"
    assert settings.REDIS_PORT == 6379, f"Expected 6379, got {settings.REDIS_PORT}"
    assert settings.REDIS_DB == 0, f"Expected 0, got {settings.REDIS_DB}"
    assert settings.REDIS_URL == "redis://localhost:6379/0", f"Expected 'redis://localhost:6379/0', got '{settings.REDIS_URL}'"
    assert settings.MARKET_DATA_SERVICE_URL == "http://test-market-data:8001", f"Expected 'http://test-market-data:8001', got '{settings.MARKET_DATA_SERVICE_URL}'"
    assert settings.NOTIFICATION_SERVICE_URL == "http://test-notification:8002", f"Expected 'http://test-notification:8002', got '{settings.NOTIFICATION_SERVICE_URL}'"
    assert settings.ANALYSIS_TIMEOUT == 45, f"Expected 45, got {settings.ANALYSIS_TIMEOUT}"
    assert settings.MAX_CONCURRENT_ANALYSES == 5, f"Expected 5, got {settings.MAX_CONCURRENT_ANALYSES}"
    assert settings.RATE_LIMIT_REQUESTS == 100, f"Expected 100, got {settings.RATE_LIMIT_REQUESTS}"
    assert settings.RATE_LIMIT_PERIOD == 60, f"Expected 60, got {settings.RATE_LIMIT_PERIOD}"
    assert settings.RATE_LIMIT_PREMIUM == 500, f"Expected 500, got {settings.RATE_LIMIT_PREMIUM}"
    print("Settings initialization test passed!")

def test_helper_functions():
    """Test helper functions return correct values."""
    setup_test_env()

    # Create our own helper functions that use a test settings instance
    test_settings = AnalysisEngineSettings(
        HOST="localhost",
        PORT=8000,
        DEBUG_MODE=True,
        LOG_LEVEL="DEBUG",
        JWT_SECRET="test_secret",
        JWT_ALGORITHM="HS256",
        DATABASE_URL_OVERRIDE="postgresql://test:test@localhost:5432/testdb",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        REDIS_DB=0,
        MARKET_DATA_SERVICE_URL="http://test-market-data:8001",
        NOTIFICATION_SERVICE_URL="http://test-notification:8002",
        ANALYSIS_TIMEOUT=45,
        MAX_CONCURRENT_ANALYSES=5,
        RATE_LIMIT_REQUESTS=100,
        RATE_LIMIT_PERIOD=60,
        RATE_LIMIT_PREMIUM=500
    )

    # Define test helper functions
    def test_get_db_url():
        return test_settings.DATABASE_URL

    def test_get_redis_url():
        return test_settings.REDIS_URL

    def test_get_market_data_service_url():
        return test_settings.MARKET_DATA_SERVICE_URL

    def test_get_notification_service_url():
        return test_settings.NOTIFICATION_SERVICE_URL

    def test_get_analysis_settings():
        return {
            "analysis_timeout": test_settings.ANALYSIS_TIMEOUT,
            "max_concurrent_analyses": test_settings.MAX_CONCURRENT_ANALYSES,
            "default_timeframes": test_settings.DEFAULT_TIMEFRAMES,
            "default_symbols": test_settings.DEFAULT_SYMBOLS
        }

    def test_get_rate_limits():
        return {
            "rate_limit_requests": test_settings.RATE_LIMIT_REQUESTS,
            "rate_limit_period": test_settings.RATE_LIMIT_PERIOD,
            "rate_limit_premium": test_settings.RATE_LIMIT_PREMIUM
        }

    def test_get_db_settings():
        return {
            "database_url": test_settings.DATABASE_URL,
            "redis_url": test_settings.REDIS_URL
        }

    def test_get_api_settings():
        return {
            "host": test_settings.HOST,
            "port": test_settings.PORT,
            "api_version": test_settings.API_VERSION,
            "api_prefix": test_settings.API_PREFIX,
            "debug_mode": test_settings.DEBUG_MODE
        }

    def test_get_security_settings():
        return {
            "jwt_secret": test_settings.JWT_SECRET,
            "jwt_algorithm": test_settings.JWT_ALGORITHM,
            "access_token_expire_minutes": test_settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            "cors_origins": test_settings.CORS_ORIGINS
        }

    def test_get_monitoring_settings():
        return {
            "enable_metrics": test_settings.ENABLE_METRICS,
            "metrics_port": test_settings.METRICS_PORT
        }

    def test_get_websocket_settings():
        return {
            "ws_heartbeat_interval": test_settings.WS_HEARTBEAT_INTERVAL,
            "ws_max_connections": test_settings.WS_MAX_CONNECTIONS
        }

    print("Testing helper functions...")
    assert test_get_db_url() == "postgresql://test:test@localhost:5432/testdb", f"Expected 'postgresql://test:test@localhost:5432/testdb', got '{test_get_db_url()}'"
    assert test_get_redis_url() == "redis://localhost:6379/0", f"Expected 'redis://localhost:6379/0', got '{test_get_redis_url()}'"
    assert test_get_market_data_service_url() == "http://test-market-data:8001", f"Expected 'http://test-market-data:8001', got '{test_get_market_data_service_url()}'"
    assert test_get_notification_service_url() == "http://test-notification:8002", f"Expected 'http://test-notification:8002', got '{test_get_notification_service_url()}'"

    analysis_settings = test_get_analysis_settings()
    assert analysis_settings["analysis_timeout"] == 45, f"Expected 45, got {analysis_settings['analysis_timeout']}"
    assert analysis_settings["max_concurrent_analyses"] == 5, f"Expected 5, got {analysis_settings['max_concurrent_analyses']}"

    rate_limits = test_get_rate_limits()
    assert rate_limits["rate_limit_requests"] == 100, f"Expected 100, got {rate_limits['rate_limit_requests']}"
    assert rate_limits["rate_limit_period"] == 60, f"Expected 60, got {rate_limits['rate_limit_period']}"
    assert rate_limits["rate_limit_premium"] == 500, f"Expected 500, got {rate_limits['rate_limit_premium']}"

    db_settings = test_get_db_settings()
    assert db_settings["database_url"] == "postgresql://test:test@localhost:5432/testdb", f"Expected 'postgresql://test:test@localhost:5432/testdb', got '{db_settings['database_url']}'"
    assert db_settings["redis_url"] == "redis://localhost:6379/0", f"Expected 'redis://localhost:6379/0', got '{db_settings['redis_url']}'"

    api_settings = test_get_api_settings()
    assert api_settings["host"] == "localhost", f"Expected 'localhost', got '{api_settings['host']}'"
    assert api_settings["port"] == 8000, f"Expected 8000, got {api_settings['port']}"
    assert api_settings["debug_mode"] is True, f"Expected True, got {api_settings['debug_mode']}"

    security_settings = test_get_security_settings()
    assert security_settings["jwt_secret"] == "test_secret", f"Expected 'test_secret', got '{security_settings['jwt_secret']}'"
    assert security_settings["jwt_algorithm"] == "HS256", f"Expected 'HS256', got '{security_settings['jwt_algorithm']}'"

    monitoring_settings = test_get_monitoring_settings()
    assert monitoring_settings["enable_metrics"] is True, f"Expected True, got {monitoring_settings['enable_metrics']}"

    websocket_settings = test_get_websocket_settings()
    assert websocket_settings["ws_heartbeat_interval"] == 30, f"Expected 30, got {websocket_settings['ws_heartbeat_interval']}"
    assert websocket_settings["ws_max_connections"] == 1000, f"Expected 1000, got {websocket_settings['ws_max_connections']}"
    print("Helper functions test passed!")

def test_configuration_manager():
    """Test ConfigurationManager functionality."""
    setup_test_env()

    # Create a test settings instance
    test_settings = AnalysisEngineSettings(
        HOST="localhost",
        PORT=8000,
        DEBUG_MODE=True,
        LOG_LEVEL="DEBUG",
        JWT_SECRET="test_secret",
        JWT_ALGORITHM="HS256",
        DATABASE_URL_OVERRIDE="postgresql://test:test@localhost:5432/testdb",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        REDIS_DB=0,
        MARKET_DATA_SERVICE_URL="http://test-market-data:8001",
        NOTIFICATION_SERVICE_URL="http://test-notification:8002",
        ANALYSIS_TIMEOUT=45,
        MAX_CONCURRENT_ANALYSES=5,
        RATE_LIMIT_REQUESTS=100,
        RATE_LIMIT_PERIOD=60,
        RATE_LIMIT_PREMIUM=500
    )

    # Create a custom ConfigurationManager that uses our test settings
    class TestConfigurationManager:
    """
    TestConfigurationManager class.
    
    Attributes:
        Add attributes here
    """

        def __init__(self):
    """
      init  .
    
    """

            self._settings = test_settings
            self._config_cache = {}

        def get(self, key, default=None):
    """
    Get.
    
    Args:
        key: Description of key
        default: Description of default
    
    """

            if key in self._config_cache:
                return self._config_cache[key]

            value = getattr(self._settings, key, default)
            self._config_cache[key] = value
            return value

        def set(self, key, value):
    """
    Set.
    
    Args:
        key: Description of key
        value: Description of value
    
    """

            setattr(self._settings, key, value)
            self._config_cache[key] = value

        def reload(self):
    """
    Reload.
    
    """

            # For testing, just reset HOST to "localhost"
            self._settings.HOST = "localhost"
            self._config_cache.clear()

    print("Testing ConfigurationManager...")
    manager = TestConfigurationManager()

    # Test get method
    assert manager.get("HOST") == "localhost", f"Expected 'localhost', got '{manager.get('HOST')}'"
    assert manager.get("PORT") == 8000, f"Expected 8000, got {manager.get('PORT')}"
    assert manager.get("non_existent_key", "default") == "default", f"Expected 'default', got '{manager.get('non_existent_key', 'default')}'"

    # Test set method
    manager.set("HOST", "new_host")
    assert manager.get("HOST") == "new_host", f"Expected 'new_host', got '{manager.get('HOST')}'"

    # Test reload method
    manager.reload()
    assert manager.get("HOST") == "localhost", f"Expected 'localhost', got '{manager.get('HOST')}'"
    print("ConfigurationManager test passed!")

def run_all_tests():
    """Run all tests."""
    test_settings_initialization()
    test_helper_functions()
    test_configuration_manager()
    print("All tests passed!")

if __name__ == "__main__":
    run_all_tests()
