from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings.
    """
    # Service URLs
    market_analysis_service_url: str = Field(default="http://market-analysis-service:8000")
    causal_analysis_service_url: str = Field(default="http://causal-analysis-service:8000")
    backtesting_service_url: str = Field(default="http://backtesting-service:8000")

    # Database settings
    database_connection_string: str = Field(default="postgresql://postgres:postgres@postgres:5432/analysis_coordinator")

    # Service settings
    service_name: str = Field(default="analysis-coordinator-service")
    log_level: str = Field(default="INFO")

    # API settings
    api_prefix: str = Field(default="/api/v1")

    # Resilience settings
    retry_count: int = Field(default=3)
    retry_backoff_factor: float = Field(default=0.5)
    circuit_breaker_failure_threshold: int = Field(default=5)
    circuit_breaker_recovery_timeout: int = Field(default=30)

    # Task settings
    task_cleanup_interval_hours: int = Field(default=24)
    task_max_age_days: int = Field(default=7)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings.
    """
    return Settings()
