from pydantic import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings.
    """
    # Service URLs
    market_analysis_service_url: str = "http://market-analysis-service:8000"
    causal_analysis_service_url: str = "http://causal-analysis-service:8000"
    backtesting_service_url: str = "http://backtesting-service:8000"
    
    # Database settings
    database_connection_string: str = "postgresql://postgres:postgres@postgres:5432/analysis_coordinator"
    
    # Service settings
    service_name: str = "analysis-coordinator-service"
    log_level: str = "INFO"
    
    # API settings
    api_prefix: str = "/api/v1"
    
    # Resilience settings
    retry_count: int = 3
    retry_backoff_factor: float = 0.5
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30
    
    # Task settings
    task_cleanup_interval_hours: int = 24
    task_max_age_days: int = 7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings.
    """
    return Settings()