"""
Configuration for Strategy Execution Engine

This module provides configuration settings for the Strategy Execution Engine.
"""

import os
from typing import List, Dict, Any, Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration settings for the Strategy Execution Engine.
    """
    # Service configuration
    app_name: str = "Strategy Execution Engine"
    app_version: str = "0.1.0"
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8003, env="PORT")

    # CORS configuration
    cors_origins: List[str] = Field(default=["*"])

    # API keys
    api_key: str = Field(default="", env="API_KEY")
    service_api_key: str = Field(default="", env="SERVICE_API_KEY")

    # Service URLs
    analysis_engine_url: str = Field(default="http://analysis-engine-service:8002", env="ANALYSIS_ENGINE_URL")
    feature_store_url: str = Field(default="http://feature-store-service:8001", env="FEATURE_STORE_URL")
    trading_gateway_url: str = Field(default="http://trading-gateway-service:8004", env="TRADING_GATEWAY_URL")
    risk_management_url: str = Field(default="http://risk-management-service:8000", env="RISK_MANAGEMENT_URL")
    portfolio_management_url: str = Field(default="http://portfolio-management-service:8000", env="PORTFOLIO_MANAGEMENT_URL")
    monitoring_service_url: str = Field(default="http://monitoring-alerting-service:8005", env="MONITORING_SERVICE_URL")

    # Service API keys
    trading_gateway_key: str = Field(default="", env="TRADING_GATEWAY_KEY")
    analysis_engine_key: str = Field(default="", env="ANALYSIS_ENGINE_KEY")
    feature_store_key: str = Field(default="", env="FEATURE_STORE_KEY")

    # Strategy configuration
    strategies_dir: str = Field(default="./strategies", env="STRATEGIES_DIR")

    # Backtesting configuration
    backtest_data_dir: str = Field(default="./backtest_data", env="BACKTEST_DATA_DIR")

    # Monitoring configuration
    enable_prometheus: bool = Field(default=True, env="ENABLE_PROMETHEUS")

    # Client configuration
    client_timeout_seconds: float = Field(default=5.0, env="CLIENT_TIMEOUT_SECONDS")
    client_max_retries: int = Field(default=3, env="CLIENT_MAX_RETRIES")
    client_retry_base_delay: float = Field(default=0.1, env="CLIENT_RETRY_BASE_DELAY")
    client_retry_backoff_factor: float = Field(default=2.0, env="CLIENT_RETRY_BACKOFF_FACTOR")
    client_circuit_breaker_threshold: int = Field(default=5, env="CLIENT_CIRCUIT_BREAKER_THRESHOLD")
    client_circuit_breaker_reset_timeout: int = Field(default=30, env="CLIENT_CIRCUIT_BREAKER_RESET_TIMEOUT")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings.

    Returns:
        Settings: Application settings
    """
    return Settings()

def get_service_url(service_name: str) -> str:
    """
    Get URL for a specific service.

    Args:
        service_name: Name of the service

    Returns:
        str: Service URL
    """
    settings = get_settings()

    service_urls = {
        "analysis_engine": settings.analysis_engine_url,
        "feature_store": settings.feature_store_url,
        "trading_gateway": settings.trading_gateway_url,
        "risk_management": settings.risk_management_url,
        "portfolio_management": settings.portfolio_management_url,
        "monitoring": settings.monitoring_service_url
    }

    return service_urls.get(service_name, "")
