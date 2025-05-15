"""
Settings configuration for the Chat Service
"""
from typing import List
from pydantic import BaseSettings, Field, validator
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Settings configuration using Pydantic."""
    
    # Application Settings
    APP_NAME: str = "chat-service"
    APP_VERSION: str = "1.0.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    
    # API Settings
    API_PREFIX: str = "/api/v1"
    API_DEBUG: bool = False
    
    # Security Settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    API_KEY_NAME: str = "X-API-Key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    
    @property
    def cors_origins(self) -> List[str]:
        """Get the CORS origins."""
        if not self.ALLOWED_ORIGINS:
            return ["*"]
        elif isinstance(self.ALLOWED_ORIGINS, str):
            return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
        return self.ALLOWED_ORIGINS
    
    # Database Settings
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis Settings for Caching
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    @validator("REDIS_URL", pre=True)
    def parse_redis_url(cls, v: str) -> str:
        """Parse Redis URL and extract host, port, and db."""
        if not v:
            return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
        return v
    
    # Event Bus Settings
    EVENT_BUS_TYPE: str = "kafka"  # or "in-memory" for development
    KAFKA_BOOTSTRAP_SERVERS: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    KAFKA_CONSUMER_GROUP: str = "chat-service-group"
    
    # Logging Settings
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Service Integration URLs
    ANALYSIS_SERVICE_URL: str = Field(..., env="ANALYSIS_SERVICE_URL")
    ML_SERVICE_URL: str = Field(..., env="ML_SERVICE_URL")
    
    # Chat Service Specific Settings
    MAX_MESSAGE_LENGTH: int = 1000
    MESSAGE_RATE_LIMIT: int = 100  # messages per minute
    HISTORY_DEFAULT_LIMIT: int = 50
    HISTORY_MAX_LIMIT: int = 100
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()