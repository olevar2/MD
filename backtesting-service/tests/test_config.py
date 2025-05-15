import os
import pytest
from app.config.settings import Settings

def test_settings_defaults():
    """Test that default settings are loaded correctly"""
    settings = Settings()
    assert settings.APP_NAME == "Backtesting Service"
    assert settings.APP_VERSION == "0.1.0"
    assert settings.APP_HOST == "0.0.0.0"
    assert settings.APP_PORT == 8002
    assert settings.DEBUG is False
    assert settings.cors_origins == ["*"]

def test_settings_from_env(monkeypatch):
    """Test that settings are properly loaded from environment variables"""
    test_values = {
        "APP_HOST": "127.0.0.1",
        "APP_PORT": "9000",
        "DEBUG": "true",
        "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:8080",
        "DATABASE_URL": "postgresql://user:pass@localhost/test_db",
        "LOG_LEVEL": "DEBUG",
        "MAX_CONCURRENT_BACKTESTS": "10",
        "DEFAULT_COMMISSION_RATE": "0.002"
    }
    
    # Set environment variables
    for key, value in test_values.items():
        monkeypatch.setenv(key, value)
    
    settings = Settings()
    
    # Verify settings are loaded correctly
    assert settings.APP_HOST == "127.0.0.1"
    assert settings.APP_PORT == 9000
    assert settings.DEBUG is True
    assert sorted(settings.cors_origins) == sorted(["http://localhost:3000", "http://localhost:8080"])
    assert settings.DATABASE_URL == "postgresql://user:pass@localhost/test_db"
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.MAX_CONCURRENT_BACKTESTS == 10
    assert settings.DEFAULT_COMMISSION_RATE == 0.002

def test_invalid_settings():
    """Test that invalid settings raise appropriate errors"""
    with pytest.raises(ValueError):
        Settings(APP_PORT="invalid_port")
    
    with pytest.raises(ValueError):
        Settings(MAX_CONCURRENT_BACKTESTS="invalid_number")
    
    with pytest.raises(ValueError):
        Settings(DEFAULT_COMMISSION_RATE="invalid_rate")

def test_redis_url_parsing():
    """Test that Redis URL is properly parsed when provided"""
    test_redis_url = "redis://:password123@localhost:6380/1"
    settings = Settings(REDIS_URL=test_redis_url)
    
    assert settings.REDIS_URL == test_redis_url
    # Default values should not be used when REDIS_URL is provided
    assert settings.REDIS_HOST == "localhost"
    assert settings.REDIS_PORT == 6379
    assert settings.REDIS_DB == 0

def test_ssl_configuration():
    """Test SSL configuration settings"""
    ssl_settings = {
        "SSL_KEYFILE": "/path/to/key.pem",
        "SSL_CERTFILE": "/path/to/cert.pem"
    }
    
    settings = Settings(**ssl_settings)
    assert settings.SSL_KEYFILE == "/path/to/key.pem"
    assert settings.SSL_CERTFILE == "/path/to/cert.pem"

def test_backtest_specific_settings():
    """Test backtesting-specific configuration settings"""
    test_values = {
        "MAX_BACKTEST_DURATION": "3600",  # 1 hour
        "BACKTEST_RESULTS_TTL": "86400",  # 1 day
        "DEFAULT_SLIPPAGE": "0.0002"
    }
    
    settings = Settings(**test_values)
    assert settings.MAX_BACKTEST_DURATION == 3600
    assert settings.BACKTEST_RESULTS_TTL == 86400
    assert settings.DEFAULT_SLIPPAGE == 0.0002