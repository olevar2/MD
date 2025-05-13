"""
Configuration settings for the Analysis Engine Service.

This module provides a centralized configuration management system
that consolidates settings from various sources and follows the
common-lib pattern for configuration management.
"""
from functools import lru_cache
from typing import Dict, Any, List, Optional, Literal
import re
from ipaddress import IPv4Address, IPv6Address
from pydantic import Field, SecretStr, computed_field, field_validator, model_validator, AnyUrl, validator
from common_lib.config import AppSettings, load_settings
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AnalysisEngineSettings(AppSettings):
    """Settings specific to the Analysis Engine Service."""
    SERVICE_NAME: str = Field(default='analysis-engine-service', env=
        'SERVICE_NAME', description='Name of the service')
    API_VERSION: str = Field(default='v1', env='API_VERSION', description=
        'API version')
    API_PREFIX: str = Field(default='/api/v1', env='API_PREFIX',
        description='API endpoint prefix')
    HOST: str = Field(default='0.0.0.0', env='HOST', description=
        'Host to bind the API server')
    PORT: int = Field(default=8000, ge=1024, le=65535, env='PORT',
        description='Port to bind the API server')

    @field_validator('API_VERSION')
    @classmethod
    def validate_api_version(cls, v: str) ->str:
        """Validate API version format."""
        if not re.match('^v\\d+(\\.\\d+)?$', v):
            raise ValueError(
                f"Invalid API version format: {v}. Must be in format 'v1' or 'v1.0'"
                )
        return v

    @field_validator('API_PREFIX')
    @classmethod
    def validate_api_prefix(cls, v: str) ->str:
        """Validate API prefix format."""
        if not v.startswith('/'):
            v = f'/{v}'
        return v

    @field_validator('HOST')
    @classmethod
    @with_exception_handling
    def validate_host(cls, v: str) ->str:
        """Validate host address."""
        if v != 'localhost' and v != '0.0.0.0':
            try:
                IPv4Address(v)
            except ValueError:
                try:
                    IPv6Address(v)
                except ValueError:
                    raise ValueError(f'Invalid host address: {v}')
        return v
    JWT_SECRET: str = Field(default='your-secret-key', env='JWT_SECRET',
        description='Secret key for JWT tokens')
    JWT_ALGORITHM: str = Field(default='HS256', env='JWT_ALGORITHM',
        description='Algorithm for JWT tokens')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=5, le=1440, env
        ='ACCESS_TOKEN_EXPIRE_MINUTES', description=
        'JWT token expiration time in minutes')
    CORS_ORIGINS: List[str] = Field(default=['*'], env='CORS_ORIGINS',
        description='Allowed CORS origins')

    @field_validator('JWT_SECRET')
    @classmethod
    def validate_jwt_secret(cls, v: str) ->str:
        """Validate JWT secret."""
        if v == 'your-secret-key':
            import warnings
            warnings.warn(
                'Using default JWT_SECRET is not secure. Please set a proper secret key.'
                , UserWarning, stacklevel=2)
        if len(v) < 16:
            raise ValueError('JWT_SECRET must be at least 16 characters long')
        return v

    @field_validator('JWT_ALGORITHM')
    @classmethod
    def validate_jwt_algorithm(cls, v: str) ->str:
        """Validate JWT algorithm."""
        allowed_algorithms = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384',
            'RS512', 'ES256', 'ES384', 'ES512']
        if v not in allowed_algorithms:
            raise ValueError(
                f'Invalid JWT algorithm: {v}. Must be one of {allowed_algorithms}'
                )
        return v

    @field_validator('CORS_ORIGINS')
    @classmethod
    def validate_cors_origins(cls, v: List[str]) ->List[str]:
        """Validate CORS origins."""
        if '*' in v and len(v) > 1:
            raise ValueError(
                "If wildcard '*' is used in CORS_ORIGINS, it must be the only item"
                )
        for origin in v:
            if origin != '*':
                if not origin.startswith(('http://', 'https://')):
                    raise ValueError(
                        f'Invalid CORS origin: {origin}. Must start with http:// or https://'
                        )
        return v
    DB_USER: Optional[str] = Field(default=None, env='DB_USER', description
        ='Database user')
    DB_PASSWORD: Optional[SecretStr] = Field(default=None, env=
        'DB_PASSWORD', description='Database password')
    DB_HOST: Optional[str] = Field(default=None, env='DB_HOST', description
        ='Database host')
    DB_PORT: int = Field(default=5432, ge=1, le=65535, env='DB_PORT',
        description='Database port')
    DB_NAME: Optional[str] = Field(default=None, env='DB_NAME', description
        ='Database name')
    DATABASE_URL_OVERRIDE: Optional[str] = Field(default=None, env=
        'DATABASE_URL', description=
        'Direct database URL (overrides individual DB settings)')
    DB_POOL_SIZE: int = Field(default=5, ge=1, le=100, env='DB_POOL_SIZE',
        description='Database connection pool size')
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0, le=100, env=
        'DB_MAX_OVERFLOW', description='Maximum overflow connections')
    DB_POOL_TIMEOUT: int = Field(default=30, ge=1, le=300, env=
        'DB_POOL_TIMEOUT', description='Connection pool timeout in seconds')
    DB_POOL_RECYCLE: int = Field(default=1800, ge=1, env='DB_POOL_RECYCLE',
        description='Connection recycle time in seconds')

    @computed_field(description='SQLAlchemy compatible database connection URL'
        )
    @property
    def database_url(self) ->str:
        """Get the database URL, either from direct override or constructed from components."""
        if self.DATABASE_URL_OVERRIDE:
            return self.DATABASE_URL_OVERRIDE
        if self.DB_USER and self.DB_PASSWORD and self.DB_HOST and self.DB_NAME:
            return super().DATABASE_URL
        return 'sqlite:///./analysis_engine.db'

    @field_validator('DATABASE_URL_OVERRIDE')
    @classmethod
    def validate_database_url(cls, v: Optional[str]) ->Optional[str]:
        """Validate database URL format."""
        if v is None:
            return v
        valid_prefixes = ['postgresql://', 'postgresql+psycopg2://',
            'postgresql+asyncpg://', 'mysql://', 'mysql+pymysql://',
            'mysql+aiomysql://', 'sqlite://', 'oracle://', 'mssql://',
            'cockroachdb://']
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f'Invalid database URL format: {v}. Must start with one of {valid_prefixes}'
                )
        return v

    @with_database_resilience('validate_db_settings')
    @model_validator(mode='after')
    def validate_db_settings(self) ->'AnalysisEngineSettings':
        """Validate database settings consistency."""
        if self.DATABASE_URL_OVERRIDE:
            return self
        db_components = [self.DB_USER, self.DB_HOST, self.DB_NAME]
        if any(db_components) and not all(db_components):
            missing = []
            if not self.DB_USER:
                missing.append('DB_USER')
            if not self.DB_HOST:
                missing.append('DB_HOST')
            if not self.DB_NAME:
                missing.append('DB_NAME')
            raise ValueError(
                f"Missing required database settings: {', '.join(missing)}")
        return self
    REDIS_HOST: str = Field(default='localhost', env='REDIS_HOST',
        description='Redis host')
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535, env='REDIS_PORT',
        description='Redis port')
    REDIS_DB: int = Field(default=0, ge=0, le=15, env='REDIS_DB',
        description='Redis database index')
    REDIS_PASSWORD: Optional[SecretStr] = Field(default=None, env=
        'REDIS_PASSWORD', description='Redis password')
    REDIS_TIMEOUT: int = Field(default=10, ge=1, le=60, env='REDIS_TIMEOUT',
        description='Redis connection timeout in seconds')
    REDIS_SSL: bool = Field(default=False, env='REDIS_SSL', description=
        'Use SSL for Redis connection')
    REDIS_POOL_SIZE: int = Field(default=10, ge=1, le=100, env=
        'REDIS_POOL_SIZE', description='Redis connection pool size')
    KAFKA_BOOTSTRAP_SERVERS: List[str] = Field(default=['localhost:9092'],
        env='KAFKA_BOOTSTRAP_SERVERS', description=
        'List of Kafka broker addresses (host:port)')
    KAFKA_ANALYSIS_TOPIC: str = Field(default='analysis_events', env=
        'KAFKA_ANALYSIS_TOPIC', description=
        'Kafka topic for analysis-related events')
    KAFKA_REGIME_TOPIC: str = Field(default='market_regime_events', env=
        'KAFKA_REGIME_TOPIC', description=
        'Kafka topic for market regime change events')
    KAFKA_SIGNAL_TOPIC: str = Field(default='trading_signals', env=
        'KAFKA_SIGNAL_TOPIC', description=
        'Kafka topic for generated trading signals')

    @field_validator('KAFKA_BOOTSTRAP_SERVERS')
    @classmethod
    def validate_kafka_servers(cls, v: List[str]) ->List[str]:
        """Validate Kafka bootstrap server format."""
        if not v:
            raise ValueError('KAFKA_BOOTSTRAP_SERVERS cannot be empty')
        for server in v:
            if ':' not in server or not server.split(':')[-1].isdigit():
                raise ValueError(
                    f'Invalid Kafka server format: {server}. Must be host:port'
                    )
        return v

    @computed_field(description='Redis connection URL')
    @computed_field(description='Redis connection URL')
    @property
    def redis_url(self) ->str:
        """Get the Redis URL with proper formatting."""
        protocol = 'rediss' if self.REDIS_SSL else 'redis'
        auth = (f':{self.REDIS_PASSWORD.get_secret_value()}@' if self.
            REDIS_PASSWORD else '')
        return (
            f'{protocol}://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}'
            )

    @field_validator('REDIS_HOST')
    @classmethod
    @with_exception_handling
    def validate_redis_host(cls, v: str) ->str:
        """Validate Redis host."""
        if v != 'localhost':
            try:
                IPv4Address(v)
            except ValueError:
                try:
                    IPv6Address(v)
                except ValueError:
                    if not re.match(
                        '^[a-zA-Z0-9]([a-zA-Z0-9\\-]{0,61}[a-zA-Z0-9])?(\\.[a-zA-Z0-9]([a-zA-Z0-9\\-]{0,61}[a-zA-Z0-9])?)*$'
                        , v):
                        raise ValueError(f'Invalid Redis host: {v}')
        return v
    MARKET_DATA_SERVICE_URL: str = Field(default=
        'http://market-data-service:8000/api/v1', env=
        'MARKET_DATA_SERVICE_URL', description=
        'URL for the Market Data Service')
    NOTIFICATION_SERVICE_URL: str = Field(default=
        'http://notification-service:8000/api/v1', env=
        'NOTIFICATION_SERVICE_URL', description=
        'URL for the Notification Service')
    ML_INTEGRATION_SERVICE_URL: str = Field(default=
        'http://ml-integration-service:8000/api/v1', env=
        'ML_INTEGRATION_SERVICE_URL', description=
        'URL for the ML Integration Service')
    FEATURE_STORE_SERVICE_URL: str = Field(default=
        'http://feature-store-service:8000/api/v1', env=
        'FEATURE_STORE_SERVICE_URL', description=
        'URL for the Feature Store Service')
    SERVICE_TIMEOUT_SECONDS: int = Field(default=30, ge=1, le=300, env=
        'SERVICE_TIMEOUT_SECONDS', description=
        'Default timeout for external service requests in seconds')
    SERVICE_MAX_RETRIES: int = Field(default=3, ge=0, le=10, env=
        'SERVICE_MAX_RETRIES', description=
        'Maximum number of retries for external service requests')
    SERVICE_RETRY_BACKOFF: float = Field(default=0.5, ge=0.1, le=60.0, env=
        'SERVICE_RETRY_BACKOFF', description=
        'Backoff factor for retries in seconds')

    @field_validator('MARKET_DATA_SERVICE_URL', 'NOTIFICATION_SERVICE_URL',
        'ML_INTEGRATION_SERVICE_URL', 'FEATURE_STORE_SERVICE_URL')
    @classmethod
    def validate_service_url(cls, v: str) ->str:
        """Validate service URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError(
                f'Invalid service URL: {v}. Must start with http:// or https://'
                )
        return v
    ANALYSIS_TIMEOUT: int = Field(default=30, ge=5, le=300, env=
        'ANALYSIS_TIMEOUT', description=
        'Timeout for analysis operations in seconds')
    MAX_CONCURRENT_ANALYSES: int = Field(default=10, ge=1, le=100, env=
        'MAX_CONCURRENT_ANALYSES', description=
        'Maximum number of concurrent analyses')
    DEFAULT_TIMEFRAMES: List[str] = Field(default=['M15', 'H1', 'H4', 'D1'],
        env='DEFAULT_TIMEFRAMES', description='Default timeframes for analysis'
        )
    DEFAULT_SYMBOLS: List[str] = Field(default=['EURUSD', 'GBPUSD',
        'USDJPY', 'AUDUSD'], env='DEFAULT_SYMBOLS', description=
        'Default symbols for analysis')
    ENABLE_ADVANCED_ANALYSIS: bool = Field(default=True, env=
        'ENABLE_ADVANCED_ANALYSIS', description=
        'Enable advanced analysis features')
    ENABLE_ML_INTEGRATION: bool = Field(default=True, env=
        'ENABLE_ML_INTEGRATION', description=
        'Enable machine learning integration')
    ANALYSIS_CACHE_TTL: int = Field(default=3600, ge=0, le=86400, env=
        'ANALYSIS_CACHE_TTL', description=
        'Time-to-live for analysis cache in seconds (0 to disable caching)')

    @field_validator('DEFAULT_TIMEFRAMES')
    @classmethod
    def validate_timeframes(cls, v: List[str]) ->List[str]:
        """Validate timeframe format."""
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1',
            'W1', 'MN']
        for timeframe in v:
            if timeframe not in valid_timeframes:
                raise ValueError(
                    f'Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}'
                    )
        return v

    @field_validator('DEFAULT_SYMBOLS')
    @classmethod
    def validate_symbols(cls, v: List[str]) ->List[str]:
        """Validate symbol format."""
        for symbol in v:
            if not re.match('^[A-Z]{6}$', symbol):
                raise ValueError(
                    f'Invalid forex symbol: {symbol}. Must be 6 uppercase letters (e.g., EURUSD)'
                    )
        return v
    RATE_LIMIT_REQUESTS: int = Field(default=100, ge=1, le=10000, env=
        'RATE_LIMIT_REQUESTS', description=
        'Standard rate limit requests per minute')
    RATE_LIMIT_PERIOD: int = Field(default=60, ge=1, le=3600, env=
        'RATE_LIMIT_PERIOD', description='Rate limit period in seconds')
    RATE_LIMIT_PREMIUM: int = Field(default=500, ge=1, le=50000, env=
        'RATE_LIMIT_PREMIUM', description=
        'Premium rate limit requests per minute')
    RATE_LIMIT_BURST: int = Field(default=20, ge=1, le=1000, env=
        'RATE_LIMIT_BURST', description='Burst capacity for rate limiting')
    LOG_FORMAT: str = Field(default=
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', env=
        'LOG_FORMAT', description='Logging format string')
    LOG_FILE: Optional[str] = Field(default=None, env='LOG_FILE',
        description='Log file path (None for stdout only)')
    LOG_ROTATION: bool = Field(default=True, env='LOG_ROTATION',
        description='Enable log rotation')
    LOG_MAX_SIZE: int = Field(default=10485760, ge=1048576, le=104857600,
        env='LOG_MAX_SIZE', description=
        'Maximum log file size in bytes before rotation')
    LOG_BACKUP_COUNT: int = Field(default=5, ge=0, le=100, env=
        'LOG_BACKUP_COUNT', description='Number of backup log files to keep')
    ENABLE_METRICS: bool = Field(default=True, env='ENABLE_METRICS',
        description='Enable performance metrics collection')
    METRICS_PORT: int = Field(default=9090, ge=1024, le=65535, env=
        'METRICS_PORT', description='Port for metrics server')
    ENABLE_TRACING: bool = Field(default=False, env='ENABLE_TRACING',
        description='Enable distributed tracing')
    TRACING_SAMPLE_RATE: float = Field(default=0.1, ge=0.0, le=1.0, env=
        'TRACING_SAMPLE_RATE', description=
        'Sampling rate for distributed tracing (0.0-1.0)')
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, ge=5, le=300, env=
        'WS_HEARTBEAT_INTERVAL', description=
        'WebSocket heartbeat interval in seconds')
    WS_MAX_CONNECTIONS: int = Field(default=1000, ge=10, le=10000, env=
        'WS_MAX_CONNECTIONS', description=
        'Maximum number of WebSocket connections')
    WS_CLOSE_TIMEOUT: int = Field(default=10, ge=1, le=60, env=
        'WS_CLOSE_TIMEOUT', description=
        'Timeout for WebSocket close handshake in seconds')
    FEATURE_MULTI_TIMEFRAME_ANALYSIS: bool = Field(default=True, env=
        'FEATURE_MULTI_TIMEFRAME_ANALYSIS', description=
        'Enable multi-timeframe analysis')
    FEATURE_MARKET_REGIME_DETECTION: bool = Field(default=True, env=
        'FEATURE_MARKET_REGIME_DETECTION', description=
        'Enable market regime detection')
    FEATURE_SENTIMENT_ANALYSIS: bool = Field(default=True, env=
        'FEATURE_SENTIMENT_ANALYSIS', description='Enable sentiment analysis')
    FEATURE_CORRELATION_ANALYSIS: bool = Field(default=True, env=
        'FEATURE_CORRELATION_ANALYSIS', description=
        'Enable correlation analysis')

    @field_validator('LOG_FORMAT')
    @classmethod
    def validate_log_format(cls, v: str) ->str:
        """Validate log format string."""
        required_fields = ['%(levelname)s', '%(message)s']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Log format must include {field}')
        return v


@lru_cache()
def get_settings() ->AnalysisEngineSettings:
    """
    Get cached settings instance for Analysis Engine Service.

    Returns:
        AnalysisEngineSettings: The settings instance
    """
    return load_settings(AnalysisEngineSettings)


settings = get_settings()


def get_db_url() ->str:
    """Get database URL with proper formatting."""
    return settings.DATABASE_URL


def get_redis_url() ->str:
    """Get Redis URL if configured."""
    return settings.REDIS_URL


def get_market_data_service_url() ->str:
    """Get market data service URL."""
    return settings.MARKET_DATA_SERVICE_URL


def get_notification_service_url() ->str:
    """Get notification service URL if configured."""
    return settings.NOTIFICATION_SERVICE_URL


def get_analysis_settings() ->Dict[str, Any]:
    """Get analysis-specific settings."""
    return {'analysis_timeout': settings.ANALYSIS_TIMEOUT,
        'max_concurrent_analyses': settings.MAX_CONCURRENT_ANALYSES,
        'default_timeframes': settings.DEFAULT_TIMEFRAMES,
        'default_symbols': settings.DEFAULT_SYMBOLS}


def get_rate_limits() ->Dict[str, Any]:
    """Get rate limiting settings."""
    return {'rate_limit_requests': settings.RATE_LIMIT_REQUESTS,
        'rate_limit_period': settings.RATE_LIMIT_PERIOD,
        'rate_limit_premium': settings.RATE_LIMIT_PREMIUM}


def get_db_settings() ->Dict[str, Any]:
    """Get database-specific settings."""
    return {'database_url': settings.DATABASE_URL, 'redis_url': settings.
        REDIS_URL}


def get_api_settings() ->Dict[str, Any]:
    """Get API-specific settings."""
    return {'host': settings.HOST, 'port': settings.PORT, 'api_version':
        settings.API_VERSION, 'api_prefix': settings.API_PREFIX,
        'debug_mode': settings.DEBUG_MODE}


def get_security_settings() ->Dict[str, Any]:
    """Get security-specific settings."""
    return {'jwt_secret': settings.JWT_SECRET, 'jwt_algorithm': settings.
        JWT_ALGORITHM, 'access_token_expire_minutes': settings.
        ACCESS_TOKEN_EXPIRE_MINUTES, 'cors_origins': settings.CORS_ORIGINS}


def get_monitoring_settings() ->Dict[str, Any]:
    """Get monitoring-specific settings."""
    return {'enable_metrics': settings.ENABLE_METRICS, 'metrics_port':
        settings.METRICS_PORT}


def get_websocket_settings() ->Dict[str, Any]:
    """Get WebSocket-specific settings."""
    return {'ws_heartbeat_interval': settings.WS_HEARTBEAT_INTERVAL,
        'ws_max_connections': settings.WS_MAX_CONNECTIONS}


class ConfigurationManager:
    """Configuration manager for the application."""

    def __init__(self):
        """Initialize the configuration manager."""
        self._settings = get_settings()
        self._config_cache: Dict[str, Any] = {}

    def get(self, key: str, default: Any=None) ->Any:
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

    def set(self, key: str, value: Any) ->None:
        """
        Set a configuration value.

        Args:
            key: The configuration key
            value: The configuration value
        """
        setattr(self._settings, key, value)
        self._config_cache[key] = value

    def reload(self) ->None:
        """Reload configuration from environment."""
        get_settings.cache_clear()
        self._settings = get_settings()
        self._config_cache.clear()
