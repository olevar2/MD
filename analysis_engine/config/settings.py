"""
Configuration settings for the Analysis Engine Service.

This module provides a centralized configuration management system
that consolidates settings from various sources and follows the
common-lib pattern for configuration management.
"""
from functools import lru_cache
from typing import Dict, Any, List, Optional

from pydantic import Field, SecretStr, computed_field
from common_lib.config import AppSettings, load_settings


class AnalysisEngineSettings(AppSettings):
    """Settings specific to the Analysis Engine Service."""

    # --- Service Specific Metadata ---
    SERVICE_NAME: str = "analysis-engine-service"
    # DEBUG_MODE and LOG_LEVEL are inherited from AppSettings

    # --- API Settings ---
    API_VERSION: str = Field(default="v1", env="API_VERSION", description="API version")
    API_PREFIX: str = Field(default="/api/v1", env="API_PREFIX", description="API endpoint prefix")
    HOST: str = Field(default="0.0.0.0", env="HOST", description="Host to bind the API server")
    PORT: int = Field(default=8000, env="PORT", description="Port to bind the API server")

    # --- Security Settings ---
    # JWT settings
    JWT_SECRET: str = Field(default="your-secret-key", env="JWT_SECRET", description="Secret key for JWT tokens")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM", description="Algorithm for JWT tokens")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES", description="JWT token expiration time in minutes")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS", description="Allowed CORS origins")

    # --- Database Settings ---
    # Override the DATABASE_URL from AppSettings to support direct URL configuration
    DATABASE_URL_OVERRIDE: Optional[str] = Field(default=None, env="DATABASE_URL", description="Direct database URL (overrides individual DB settings)")

    @computed_field(description="SQLAlchemy compatible database connection URL")
    @property
    def DATABASE_URL(self) -> str:
        """Get the database URL, either from direct override or constructed from components."""
        if self.DATABASE_URL_OVERRIDE:
            return self.DATABASE_URL_OVERRIDE
        # Fall back to the computed URL from AppSettings
        return super().DATABASE_URL

    # --- Redis Settings ---
    # REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_TIMEOUT
    # and the computed REDIS_URL are inherited from AppSettings

    # --- External Services ---
    MARKET_DATA_SERVICE_URL: str = Field(
        default="http://market-data-service:8000/api/v1",
        env="MARKET_DATA_SERVICE_URL",
        description="URL for the Market Data Service"
    )
    NOTIFICATION_SERVICE_URL: str = Field(
        default="http://notification-service:8000/api/v1",
        env="NOTIFICATION_SERVICE_URL",
        description="URL for the Notification Service"
    )

    # --- Analysis Settings ---
    ANALYSIS_TIMEOUT: int = Field(default=30, env="ANALYSIS_TIMEOUT", description="Timeout for analysis operations in seconds")
    MAX_CONCURRENT_ANALYSES: int = Field(default=10, env="MAX_CONCURRENT_ANALYSES", description="Maximum number of concurrent analyses")
    DEFAULT_TIMEFRAMES: List[str] = Field(
        default=["M15", "H1", "H4", "D1"],
        env="DEFAULT_TIMEFRAMES",
        description="Default timeframes for analysis"
    )
    DEFAULT_SYMBOLS: List[str] = Field(
        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        env="DEFAULT_SYMBOLS",
        description="Default symbols for analysis"
    )

    # --- Rate Limiting ---
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS", description="Standard rate limit requests per minute")
    RATE_LIMIT_PERIOD: int = Field(default=60, env="RATE_LIMIT_PERIOD", description="Rate limit period in seconds")
    RATE_LIMIT_PREMIUM: int = Field(default=500, env="RATE_LIMIT_PREMIUM", description="Premium rate limit requests per minute")

    # --- Logging ---
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Logging format string"
    )

    # --- Performance Monitoring ---
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS", description="Enable performance metrics collection")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT", description="Port for metrics server")

    # --- WebSocket Settings ---
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL", description="WebSocket heartbeat interval in seconds")
    WS_MAX_CONNECTIONS: int = Field(default=1000, env="WS_MAX_CONNECTIONS", description="Maximum number of WebSocket connections")


# Create a cached getter function for the settings
@lru_cache()
def get_settings() -> AnalysisEngineSettings:
    """
    Get cached settings instance for Analysis Engine Service.
    
    Returns:
        AnalysisEngineSettings: The settings instance
    """
    return load_settings(AnalysisEngineSettings)


# Create a global instance for direct import
settings = get_settings()


# Helper functions for backward compatibility and convenience

def get_db_url() -> str:
    """Get database URL with proper formatting."""
    return settings.DATABASE_URL


def get_redis_url() -> str:
    """Get Redis URL if configured."""
    return settings.REDIS_URL


def get_market_data_service_url() -> str:
    """Get market data service URL."""
    return settings.MARKET_DATA_SERVICE_URL


def get_notification_service_url() -> str:
    """Get notification service URL if configured."""
    return settings.NOTIFICATION_SERVICE_URL


def get_analysis_settings() -> Dict[str, Any]:
    """Get analysis-specific settings."""
    return {
        "analysis_timeout": settings.ANALYSIS_TIMEOUT,
        "max_concurrent_analyses": settings.MAX_CONCURRENT_ANALYSES,
        "default_timeframes": settings.DEFAULT_TIMEFRAMES,
        "default_symbols": settings.DEFAULT_SYMBOLS
    }


def get_rate_limits() -> Dict[str, Any]:
    """Get rate limiting settings."""
    return {
        "rate_limit_requests": settings.RATE_LIMIT_REQUESTS,
        "rate_limit_period": settings.RATE_LIMIT_PERIOD,
        "rate_limit_premium": settings.RATE_LIMIT_PREMIUM
    }


def get_db_settings() -> Dict[str, Any]:
    """Get database-specific settings."""
    return {
        "database_url": settings.DATABASE_URL,
        "redis_url": settings.REDIS_URL
    }


def get_api_settings() -> Dict[str, Any]:
    """Get API-specific settings."""
    return {
        "host": settings.HOST,
        "port": settings.PORT,
        "api_version": settings.API_VERSION,
        "api_prefix": settings.API_PREFIX,
        "debug_mode": settings.DEBUG_MODE
    }


def get_security_settings() -> Dict[str, Any]:
    """Get security-specific settings."""
    return {
        "jwt_secret": settings.JWT_SECRET,
        "jwt_algorithm": settings.JWT_ALGORITHM,
        "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "cors_origins": settings.CORS_ORIGINS
    }


def get_monitoring_settings() -> Dict[str, Any]:
    """Get monitoring-specific settings."""
    return {
        "enable_metrics": settings.ENABLE_METRICS,
        "metrics_port": settings.METRICS_PORT
    }


def get_websocket_settings() -> Dict[str, Any]:
    """Get WebSocket-specific settings."""
    return {
        "ws_heartbeat_interval": settings.WS_HEARTBEAT_INTERVAL,
        "ws_max_connections": settings.WS_MAX_CONNECTIONS
    }


# Configuration manager for backward compatibility
class ConfigurationManager:
    """Configuration manager for the application."""

    def __init__(self):
        """Initialize the configuration manager."""
        self._settings = get_settings()
        self._config_cache: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key
            default: The default value if key is not found

        Returns:
            The configuration value or default
        """
        if key in self._config_cache:
            return self._config_cache[key]

        value = getattr(self._settings, key, default)
        self._config_cache[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: The configuration key
            value: The configuration value
        """
        setattr(self._settings, key, value)
        self._config_cache[key] = value

    def reload(self) -> None:
        """Reload configuration from environment."""
        # Clear the lru_cache to force reloading settings
        get_settings.cache_clear()
        self._settings = get_settings()
        self._config_cache.clear()
