# d:\MD\forex_trading_platform\backtesting-service\app\config\settings.py
import os
from typing import List, Union, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()  # Load .env file

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Backtesting Service"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1/backtesting"
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8002"))  # Default port for backtesting-service
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    SSL_KEYFILE: Optional[str] = os.getenv("SSL_KEYFILE")
    SSL_CERTFILE: Optional[str] = os.getenv("SSL_CERTFILE")

    # CORS settings
    ALLOWED_ORIGINS: Union[str, List[str]] = os.getenv("ALLOWED_ORIGINS", "*")
    
    @property
    def cors_origins(self) -> List[str]:
        if isinstance(self.ALLOWED_ORIGINS, str):
            if self.ALLOWED_ORIGINS == "*":
                return ["*"]
            return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]
        return list(self.ALLOWED_ORIGINS)

    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./backtesting.db")
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "5"))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))

    # Redis settings for caching and task queue
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

    # Service integration URLs
    DATA_MANAGEMENT_SERVICE_URL: Optional[str] = os.getenv("DATA_MANAGEMENT_SERVICE_URL")
    MARKET_ANALYSIS_SERVICE_URL: Optional[str] = os.getenv("MARKET_ANALYSIS_SERVICE_URL")
    ANALYSIS_ENGINE_SERVICE_URL: Optional[str] = os.getenv("ANALYSIS_ENGINE_SERVICE_URL")

    # Backtesting specific settings
    MAX_BACKTEST_DURATION: int = int(os.getenv("MAX_BACKTEST_DURATION", "7200"))  # 2 hours in seconds
    MAX_CONCURRENT_BACKTESTS: int = int(os.getenv("MAX_CONCURRENT_BACKTESTS", "5"))
    BACKTEST_RESULTS_TTL: int = int(os.getenv("BACKTEST_RESULTS_TTL", "604800"))  # 7 days in seconds
    DEFAULT_COMMISSION_RATE: float = float(os.getenv("DEFAULT_COMMISSION_RATE", "0.001"))
    DEFAULT_SLIPPAGE: float = float(os.getenv("DEFAULT_SLIPPAGE", "0.0001"))

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()